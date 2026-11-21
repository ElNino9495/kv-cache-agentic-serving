"""Per-step fetch study, core: decode one token at a time over a quantized history, swapping a
per-step top-k set of history tokens to bf16 for that step only.

Each decode step runs every scheme as one batch row through a custom attention function
(registered with transformers' AttentionInterface, active only during decode steps). The history
KV is stored once per precision and shared by all rows; each row only owns the KV of the tokens
it decoded itself. Attention is computed by hand for the single query, so per-layer, per-head
attention probabilities come out of the same pass at no extra cost.

KV segments seen by a row at decode step j:
  history   P tokens at the row's base precision (bf16 / int2 / int3; block 0, the recent window and
            the trailing partial block are always bf16, as in run_gate.py)
  fresh     the tool output (minus its last two tokens), prefilled with sdpa over the history at
            the row's `fresh` precision, with no fetching
  generated the tokens this row decoded through this function (bf16)

Fetching (exact, cheap): with fetched set S of a KV head, history logits are q.K_base^T with
entries in S replaced by q.K16[S], and the output is softmax(.) @ V_base + sum_{i in S} p_i (V16_i - V_base_i).
tests/test_fetch_partial.py checks this against attention over a cache rebuilt with S in bf16.

Selection policies (top-k over the quantized middle tokens, per KV head unless per_layer):
  s1        previous step's attention over int2 keys (query heads of a KV head summed)
  s2        current step's attention over int2 keys (SpeCache's non-speculative variant)
  spec      speculative token anchored on the real tokens: at step j a shadow row decodes the guess of
            step j's input (the spec row's greedy prediction at step j-1, i.e. given the real previous
            target token) at the same position; its attention over int2 keys selects step j's set. A
            correct guess gives exactly s2's set; the row measures what wrong guesses cost.
  spec_chain SpeCache as published (arXiv 2503.16163, Eq. 1): the shadow decodes the previous speculative
            token's output at position +1 (each guess conditioned on the previous guess, not on the real
            token); its attention over int2 keys selects the next step's set
  sens_a    previous-step attention x query-independent quantization error:
            sum_g p_g,i (||dV_i|| + ||q_g|| ||dK_i|| / sqrt(d)), deployable (norms stored at offload)
  sens_b    previous-step attention x first-order output error of fixing token i:
            sum_g p_g,i ||dV_i + dl_g,i (V2_i - o_g)||, dl_g,i = q_g . dK_i / sqrt(d) (needs bf16 keys)
  s2sens_a  sens_a's score with the current step's int2 attention and query
  s2sens_b  sens_b's score with the current step's int2 attention, query and int2 output
  oracle    current step's true bf16 attention
  all       every middle token (ceiling for per-step fetching given the row's tool-output prefill;
            with a bf16 tool-output prefill it equals the bf16 row exactly: the noise_ctrl row)
d = bf16 - int2. per_layer: one set per layer, scored by summing over all heads, used by every head.

Base precision (--base-bits): the history base is int2 by default; with base_bits=1 it is int1 with
SpeCache's modified 1-bit quantizer (fakequant.py), every "int2" above reads "int1", and uniform int2 and
int3 run as references. Rows with k = -1 ("match") fetch as many tokens per KV head and layer as make
the GPU-resident history (base + one step's fetched set) equal to uniform int(base + 1).

Prefill fetch (prefill_k > 0): the tool output is prefilled in chunks over the base history, and for each
chunk the top-k history tokens per KV head and layer by the chunk's own queries (attention over base keys,
summed over the chunk's queries and the query heads of the KV head) are swapped to bf16 for that chunk
only. The fetch rows and fetch_all then start from this prefill ("pf" tool-output group); the uniform base
row and fetch_all_noprefill keep the plain base prefill, so the ceiling lift is measured in the same run.

Shadow rows: each spec / spec_chain row is directly followed by its shadow, which shares its fetched set
(same history view), tool output and generated tokens; the shadow's own KV is discarded and it is not a
scheme (not written out). The shadow scores with int2 keys (as s1/s2 do).
  spec shadow        same position as its spec row; sees the spec row's generated tokens before the current
                     one, then its own token (the guess of the current input). Selects for the current step.
  spec_chain shadow  position +1; sees the spec row's generated tokens including the current one, then its own
                     token (the guess of the next input). Its argmax is the next guess; its scores select for
                     the next step.
run_pause logs every guess and whether it matched the token it guessed (res["guesses"]).
"""

import math
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from transformers import AttentionInterface, DynamicCache

from experiments.gate.fakequant import roundtrip
from kvm.quant import block_bytes

ATTN_NAME = "gate_fetch"
PREV_POLICIES = ("s1", "sens_a", "sens_b", "spec_chain")   # select with scores stored at the previous step
CUR_INT2_POLICIES = ("s2", "s2sens_a", "s2sens_b")   # select with the current step's int2 attention
SPEC_POLICIES = ("spec", "spec_chain")               # rows that come with a shadow row
FETCH_POLICIES = PREV_POLICIES + CUR_INT2_POLICIES + ("spec", "oracle", "all")
_CTX = None


@dataclass(frozen=True)
class Row:
    name: str
    hist: str = "int2"
    fresh: str = "int2"
    policy: str = "none"
    k: int = 0
    per_layer: bool = False
    shadow: bool = False  # speculative-token row; directly follows its spec / spec_chain row (same policy, k)


MATCH = -1  # k resolved per pause so that base + fetched set uses the memory of uniform int(base + 1)


def default_rows(ks=(16, 32, 64, 128, MATCH), base_bits=2, prefill=False, uniform_refs=(2, 3)) -> list[Row]:
    """uniform_refs: uniform precisions run as references besides the base (each costs one history copy
    on the GPU; a reference left out can be joined from another run with summarize_fetch --join)."""
    B = f"int{base_bits}"
    fr = "pf" if prefill else B
    rows = [Row("bf16", "bf16", "bf16"), Row(B, B, B)]
    rows += [Row(f"int{b}", f"int{b}", f"int{b}") for b in uniform_refs if b != base_bits]
    rows += [Row("fetch_all", B, fr, policy="all"),
             # noise-floor control: equal to the bf16 row in exact arithmetic, computed through the fetch path,
             # so its dNLL is the bf16 rounding noise of the per-step path on that pause
             Row("noise_ctrl", B, "bf16", policy="all")]
    if prefill:
        rows.append(Row("fetch_all_noprefill", B, B, policy="all"))
    for k in ks:
        rows += [Row(p, B, fr, policy=p, k=k) for p in ("s1", "s2", "sens_a", "sens_b", "oracle", "s2sens_a", "s2sens_b")]
        rows.append(Row("s1_layer", B, fr, policy="s1", k=k, per_layer=True))
        rows += [Row("spec", B, fr, policy="spec", k=k), Row("spec_shadow", B, fr, policy="spec", k=k, shadow=True),
                 Row("spec_chain", B, fr, policy="spec_chain", k=k),
                 Row("spec_chain_shadow", B, fr, policy="spec_chain", k=k, shadow=True)]
    return rows


def diagnostic_rows(ks=(16, 64, 128), base_bits=2, prefill=False) -> list[Row]:
    B = f"int{base_bits}"
    fr = "pf" if prefill else B
    return [Row("bf16", "bf16", "bf16"), Row(B, B, B)] + \
           [Row(p, B, fr, policy=p, k=k) for p in ("s1", "s2", "oracle") for k in ks]


def first_order_norm(dV, dK, V2, q, o, scaling):
    """||dV_i + dl_g,i (V2_i - o_g)|| for every query head g and token i, without forming the
    (n, kvh, G, P, d) tensor. dV, dK, V2: (kvh, P, d); q, o: (n, kvh, G, d) -> (n, kvh, G, P)."""
    dl = torch.einsum("rkgd,kpd->rkgp", q, dK) * scaling
    a = (dV * dV).sum(-1)[None, :, None, :]
    b = (dV * V2).sum(-1)[None, :, None, :] - torch.einsum("kpd,rkgd->rkgp", dV, o)
    c = ((V2 * V2).sum(-1)[None, :, None, :] - 2 * torch.einsum("kpd,rkgd->rkgp", V2, o)
         + (o * o).sum(-1)[..., None])
    return (a + 2 * dl * b + dl * dl * c).clamp_min(0).sqrt()


class FetchContext:
    def __init__(self, rows, hist, fresh, lo, hi, n_layers, max_tokens, base="int2"):
        self.rows, self.hist, self.fresh, self.base = rows, hist, fresh, base
        self.lo, self.hi, self.n_layers, self.max_tokens = lo, hi, n_layers, max_tokens
        any_k = next(iter(hist.values()))[0][0]
        dev = any_k.device
        self.P, self.F = any_k.shape[1], next(iter(fresh.values()))[0][0].shape[1]
        self.t, self.fetch, self.diag = 0, False, False
        R = len(rows)

        def ix(cond):
            return torch.tensor([i for i, r in enumerate(rows) if cond(r)], device=dev, dtype=torch.long)

        # shadow i reads its spec row's generated KV and selection scores; it directly follows that row
        self.pair = {}
        for i, r in enumerate(rows):
            if r.shadow:
                s = rows[i - 1] if i else None
                assert s is not None and r.policy in SPEC_POLICIES and s.policy == r.policy and not s.shadow \
                    and s.k == r.k and s.per_layer == r.per_layer and s.hist == r.hist and s.fresh == r.fresh, \
                    "a shadow row must directly follow its spec / spec_chain row (same policy, k, per_layer)"
                self.pair[i] = i - 1
        self.shadow_of = {s: i for i, s in self.pair.items()}
        self.score_src = [self.pair.get(i, i) for i in range(R)]
        self.gen_src = torch.tensor(self.score_src, device=dev, dtype=torch.long)
        self.is_shadow = torch.tensor([r.shadow for r in rows], device=dev)
        # spec shadows sit at their spec row's position: they must not see the spec row's current token
        self.same_pos_shadow = torch.tensor([r.shadow and r.policy == "spec" for r in rows], device=dev)

        self.by_hist = {b: ix(lambda r, b=b: r.hist == b) for b in {r.hist for r in rows}}
        self.by_fresh = {g: ix(lambda r, g=g: r.fresh == g) for g in {r.fresh for r in rows}}
        # one selection group per (policy, per_layer, base): rows in a group are scored together
        self.groups = {}
        for i, r in enumerate(rows):
            if r.policy != "none":
                self.groups.setdefault((r.policy, r.per_layer, r.hist), []).append(i)
        self.fetch_rows = ix(lambda r: r.policy != "none")
        self.gen = [None] * n_layers
        self.prev = [dict() for _ in range(n_layers)]
        self.err = [None] * n_layers
        self.union = [None] * n_layers
        z = lambda: torch.zeros(R, device=dev, dtype=torch.float64)
        self.st_pairs, self.st_distinct, self.st_recall, self.st_obs = z(), z(), z(), z()
        self.last_idx = {}
        self.diag_records = []

    def errors(self, l):
        """Per-layer quantities that do not depend on the query, over the middle tokens."""
        if self.err[l] is None:
            lo, hi = self.lo, self.hi
            K16, V16 = (x[:, lo:hi].float() for x in self.hist["bf16"][l])
            K2, V2 = (x[:, lo:hi].float() for x in self.hist[self.base][l])
            self.err[l] = {"dk": (K16 - K2).norm(dim=-1), "dv": (V16 - V2).norm(dim=-1)}
        return self.err[l]


def fetch_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    ctx = _CTX
    l = module.layer_idx
    R, H, q_len, d = query.shape
    assert q_len == 1, "fetch attention is for single-token decode steps"
    kvh = key.shape[1]
    G = H // kvh
    P, F, t, lo, hi = ctx.P, ctx.F, ctx.t, ctx.lo, ctx.hi
    dev = query.device
    if ctx.gen[l] is None:
        ctx.gen[l] = (key.new_zeros(R, kvh, ctx.max_tokens, d), value.new_zeros(R, kvh, ctx.max_tokens, d))
    gk, gv = ctx.gen[l]
    gk[:, :, t], gv[:, :, t] = key[:, :, 0], value[:, :, 0]
    q = query[:, :, 0].float().reshape(R, kvh, G, d)
    k_own = key[:, :, 0].float()
    v_own = value[:, :, 0].float()
    hk = torch.arange(kvh, device=dev)[None, :, None]
    src = ctx.gen_src

    Kf = {b: ctx.hist[b][l][0].float() for b in ctx.hist}
    Vf = {b: ctx.hist[b][l][1].float() for b in ctx.hist}
    lh = q.new_empty(R, kvh, G, P)
    for b, rows in ctx.by_hist.items():
        lh[rows] = torch.einsum("rkgd,kpd->rkgp", q[rows], Kf[b]) * scaling
    # tail: tool output | generated tokens (a shadow reads its spec row's) | own token (shadows only)
    lt = q.new_empty(R, kvh, G, F + t + 2)
    for g, rows in ctx.by_fresh.items():
        lt[rows, :, :, :F] = torch.einsum("rkgd,kfd->rkgf", q[rows], ctx.fresh[g][l][0].float()) * scaling
    lt[..., F:F + t + 1] = torch.einsum("rkgd,rktd->rkgt", q, gk[src, :, :t + 1].float()) * scaling
    own = torch.einsum("rkgd,rkd->rkg", q, k_own) * scaling
    lt[..., -1] = torch.where(ctx.is_shadow[:, None, None], own, torch.full_like(own, float("-inf")))
    if ctx.same_pos_shadow.any():
        lt[ctx.same_pos_shadow, :, :, F + t] = float("-inf")
    lh_pre = lh.clone() if ctx.diag else None

    def mid_probs(lhist, rows):
        """Attention probabilities over the middle history tokens, (n, kvh, G, hi - lo)."""
        return torch.cat([lhist, lt[rows]], dim=-1).softmax(dim=-1)[..., lo:hi]

    def base_output(rows, fg):
        """Attention output of `rows` with the base history and no fetch, (n, kvh, G, d)."""
        p = torch.cat([lh[rows], lt[rows]], dim=-1).softmax(dim=-1)
        out = torch.einsum("rkgp,kpd->rkgd", p[..., :P], Vf[ctx.base])
        out += torch.einsum("rkgf,kfd->rkgd", p[..., P:P + F], ctx.fresh[fg][l][1].float())
        out += torch.einsum("rkgt,rktd->rkgd", p[..., P + F:P + F + t + 1], gv[src[rows], :, :t + 1].float())
        return out

    # int2 (pre-fetch) probabilities of rows that score with them, before any logit is replaced
    p2 = {}
    for (pol, per_layer, base), rr in ctx.groups.items():
        if pol in PREV_POLICIES or pol in CUR_INT2_POLICIES or pol == "spec":
            rows = torch.tensor(rr, device=dev)
            p2[(pol, per_layer, base)] = (rows, mid_probs(lh[rows], rows))
    l16 = None
    if ctx.fetch and len(ctx.fetch_rows):
        l16 = torch.einsum("rkgd,kpd->rkgp", q[ctx.fetch_rows], Kf["bf16"]) * scaling
        pos16 = {int(r): i for i, r in enumerate(ctx.fetch_rows.tolist())}

    selections = []
    if ctx.fetch:
        for (pol, per_layer, base), rr in ctx.groups.items():
            rows = torch.tensor(rr, device=dev)
            n = len(rr)
            if pol == "all":
                top = torch.arange(lo, hi, device=dev).expand(n, kvh, -1)
                valid = torch.ones(n, hi - lo, dtype=torch.bool, device=dev)
            else:
                if pol in PREV_POLICIES:
                    score = torch.stack([ctx.prev[l][ctx.score_src[i]] for i in rr])
                elif pol == "spec":  # every spec row and its shadow use the shadow's current int2 attention
                    pr = p2[(pol, per_layer, base)][1].sum(dim=2)
                    score = pr[[rr.index(ctx.shadow_of.get(i, i)) for i in rr]]
                elif pol in CUR_INT2_POLICIES:
                    pr = p2[(pol, per_layer, base)][1]
                    if pol == "s2":
                        score = pr.sum(dim=2)
                    elif pol == "s2sens_a":
                        e = ctx.errors(l)
                        qn = q[rows].norm(dim=-1)[..., None]
                        score = (pr * (e["dv"][None, :, None, :] + qn * e["dk"][None, :, None, :] * scaling)).sum(dim=2)
                    else:  # s2sens_b
                        K16m, V16m = (x[:, lo:hi].float() for x in ctx.hist["bf16"][l])
                        K2m, V2m = (x[:, lo:hi].float() for x in ctx.hist[ctx.base][l])
                        score = (pr * first_order_norm(V16m - V2m, K16m - K2m, V2m, q[rows],
                                                       base_output(rows, ctx.rows[rr[0]].fresh), scaling)).sum(dim=2)
                else:  # oracle
                    score = mid_probs(l16[[pos16[i] for i in rr]], rows).sum(dim=2)
                if per_layer:
                    score = score.sum(dim=1, keepdim=True)
                ks = torch.tensor([min(ctx.rows[i].k, hi - lo) for i in rr], device=dev)
                kmax = int(ks.max())
                top = score.topk(kmax, dim=-1).indices.expand(n, kvh, kmax) + lo
                valid = torch.arange(kmax, device=dev)[None, :] < ks[:, None]
            idxG = top[:, :, None, :].expand(-1, -1, G, -1)
            vG = valid[:, None, None, :]
            new = torch.einsum("rkgd,rkjd->rkgj", q[rows], Kf["bf16"][hk, top]) * scaling
            sub = lh[rows]
            sub.scatter_(-1, idxG, torch.where(vG, new, sub.gather(-1, idxG)))
            lh[rows] = sub
            selections.append((pol, per_layer, base, rr, rows, top, valid, idxG, vG))
            for j, i in enumerate(rr):
                ctx.last_idx[i] = top[j][:, valid[j]]

    p = torch.cat([lh, lt], dim=-1).softmax(dim=-1)
    ph, pt = p[..., :P], p[..., P:]
    o = q.new_zeros(R, kvh, G, d)
    for b, rows in ctx.by_hist.items():
        o[rows] = torch.einsum("rkgp,kpd->rkgd", ph[rows], Vf[b])
    for g, rows in ctx.by_fresh.items():
        o[rows] += torch.einsum("rkgf,kfd->rkgd", pt[rows][..., :F], ctx.fresh[g][l][1].float())
    o += torch.einsum("rkgt,rktd->rkgd", pt[..., F:F + t + 1], gv[src, :, :t + 1].float())
    o += pt[..., -1:] * v_own[:, :, None, :]
    for pol, per_layer, base, rr, rows, top, valid, idxG, vG in selections:
        dV = Vf["bf16"][hk, top] - Vf[base][hk, top]
        o[rows] += torch.einsum("rkgj,rkjd->rkgd", ph[rows].gather(-1, idxG) * vG, dV)

    # statistics of this step's fetch (shadow rows are not schemes and are skipped)
    for pol, per_layer, base, rr, rows, top, valid, idxG, vG in selections:
        n = len(rr)
        mask = torch.zeros(n, kvh, hi - lo, dtype=torch.bool, device=dev)
        mask.scatter_(-1, top - lo, valid[:, None, :].expand(-1, kvh, -1))
        if ctx.union[l] is None:
            ctx.union[l] = torch.zeros(R, kvh, hi - lo, dtype=torch.bool, device=dev)
        ctx.union[l][rows] |= mask
        ctx.st_pairs[rows] += mask.sum(dim=(1, 2)).double()
        ctx.st_distinct[rows] += mask.any(dim=1).sum(dim=-1).double()
        ctx.st_obs[rows] += 1
        if pol != "all":
            p16 = mid_probs(l16[[pos16[i] for i in rr]], rows)  # (n, kvh, G, mid), current bf16 attention
            s16 = p16.sum(dim=2)
            if per_layer:
                s16 = s16.sum(dim=1, keepdim=True)
            otop = s16.topk(top.shape[-1], dim=-1).indices.expand(n, kvh, -1)
            omask = torch.zeros_like(mask).scatter_(-1, otop, valid[:, None, :].expand(-1, kvh, -1))
            ctx.st_recall[rows] += ((mask & omask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1)).double()
            if ctx.diag:
                pm2 = torch.cat([lh_pre[rows], lt[rows]], dim=-1).softmax(dim=-1)[..., lo:hi]
                pma = ph[rows][..., lo:hi]
                mass = lambda pr, m: (pr * m[:, :, None, :]).sum(dim=-1).mean(dim=2)  # (n, kvh), mean over G
                for j, i in enumerate(rr):
                    if ctx.rows[i].shadow:
                        continue
                    ctx.diag_records.append({
                        "row": i, "t": t, "layer": l, "idx": top[j][:, valid[j]].to(torch.int32).cpu(),
                        "mass_actual": mass(pma[j:j + 1], mask[j:j + 1])[0].cpu(),
                        "mass_int2": mass(pm2[j:j + 1], mask[j:j + 1])[0].cpu(),
                        "mass_bf16": mass(p16[j:j + 1], mask[j:j + 1])[0].cpu(),
                        "oracle_mass_bf16": mass(p16[j:j + 1], omask[j:j + 1])[0].cpu(),
                        "mid_mass_bf16": p16[j].sum(dim=-1).mean(dim=1).cpu(),
                        "mid_mass_int2": pm2[j].sum(dim=-1).mean(dim=1).cpu(),
                        "overlap": (mask[j] & omask[j]).sum(dim=-1).cpu(),
                    })

    # scores for the next step's selection (previous-step policies; shadows score for their spec row)
    for (pol, per_layer, base), (rows, pr) in p2.items():
        if pol in CUR_INT2_POLICIES or pol == "spec":
            continue
        if pol in ("s1", "spec_chain"):
            score = pr.sum(dim=2)
        elif pol == "sens_a":
            e = ctx.errors(l)
            qn = q[rows].norm(dim=-1)[..., None]
            score = (pr * (e["dv"][None, :, None, :] + qn * e["dk"][None, :, None, :] * scaling)).sum(dim=2)
        else:  # sens_b
            K16m, V16m = (x[:, lo:hi].float() for x in ctx.hist["bf16"][l])
            K2m, V2m = (x[:, lo:hi].float() for x in ctx.hist[ctx.base][l])
            score = (pr * first_order_norm(V16m - V2m, K16m - K2m, V2m, q[rows], o[rows], scaling)).sum(dim=2)
        for j, i in enumerate(rows.tolist()):
            if pol == "spec_chain":
                if ctx.rows[i].shadow:
                    ctx.prev[l][ctx.pair[i]] = score[j]
            else:
                ctx.prev[l][i] = score[j]

    return o.reshape(R, 1, H, d).to(query.dtype), None


AttentionInterface.register(ATTN_NAME, fetch_attention)


@contextmanager
def fetch_mode(model, ctx):
    global _CTX
    prev_impl = model.config._attn_implementation
    _CTX = ctx
    model.set_attn_implementation(ATTN_NAME)
    try:
        yield ctx
    finally:
        model.set_attn_implementation(prev_impl)
        _CTX = None


_PCTX = None
PREFILL_ATTN_NAME = "gate_prefill_fetch"


class PrefillContext:
    """Chunked tool-output prefill over the base history with a per-chunk top-k bf16 fetch."""

    def __init__(self, hist_base, hist16, lo, hi, k, n_layers):
        self.hist_base, self.hist16, self.lo, self.hi, self.k = hist_base, hist16, lo, hi, k
        self.fk = [[] for _ in range(n_layers)]
        self.fv = [[] for _ in range(n_layers)]
        self.pairs = 0  # (layer, KV head, token) fetched over the whole prefill
        self.last_idx = {}


def prefill_fetch_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    ctx = _PCTX
    l = module.layer_idx
    _, H, c, d = query.shape
    kvh = key.shape[1]
    G = H // kvh
    lo, hi = ctx.lo, ctx.hi
    q = query[0].float().reshape(kvh, G, c, d)
    Kb, Vb = (x.float() for x in ctx.hist_base[l])
    K16, V16 = (x.float() for x in ctx.hist16[l])
    P = Kb.shape[1]
    kc, vc = key[0].float(), value[0].float()
    Kp = torch.cat(ctx.fk[l], dim=1).float() if ctx.fk[l] else kc[:, :0]
    Vp = torch.cat(ctx.fv[l], dim=1).float() if ctx.fv[l] else vc[:, :0]
    Fp = Kp.shape[1]
    lh = torch.einsum("kgcd,kpd->kgcp", q, Kb) * scaling
    lf = torch.einsum("kgcd,kfd->kgcf", q, Kp) * scaling
    causal = torch.ones(c, c, dtype=torch.bool, device=q.device).tril()
    lc = (torch.einsum("kgcd,ked->kgce", q, kc) * scaling).masked_fill(~causal, float("-inf"))
    top = None
    if ctx.k > 0:
        k = min(ctx.k, hi - lo)
        score = torch.cat([lh, lf, lc], dim=-1).softmax(dim=-1)[..., lo:hi].sum(dim=(1, 2))  # (kvh, mid)
        top = score.topk(k, dim=-1).indices + lo                                            # (kvh, k)
        hk = torch.arange(kvh, device=q.device)[:, None]
        idx = top[:, None, None, :].expand(-1, G, c, -1)
        lh.scatter_(-1, idx, torch.einsum("kgcd,kjd->kgcj", q, K16[hk, top]) * scaling)
        ctx.pairs += kvh * k
        ctx.last_idx[l] = top
    p = torch.cat([lh, lf, lc], dim=-1).softmax(dim=-1)
    out = (torch.einsum("kgcp,kpd->kgcd", p[..., :P], Vb) + torch.einsum("kgcf,kfd->kgcd", p[..., P:P + Fp], Vp)
           + torch.einsum("kgce,ked->kgcd", p[..., P + Fp:], vc))
    if top is not None:
        out += torch.einsum("kgcj,kjd->kgcd", p[..., :P].gather(-1, idx), V16[hk, top] - Vb[hk, top])
    ctx.fk[l].append(key[0])
    ctx.fv[l].append(value[0])
    return out.reshape(H, c, d).permute(1, 0, 2)[None].to(query.dtype), None


AttentionInterface.register(PREFILL_ATTN_NAME, prefill_fetch_attention)


@torch.no_grad()
def prefill_with_fetch(model, ids, hist_base, hist16, lo, hi, k, chunk, pos0):
    """Prefill `ids` (1, F) at positions pos0.. over the base history, fetching the top-k history tokens
    per KV head and layer to bf16 for each chunk. Returns ([(K, V)] per layer, (kvh, F, d)), fetched pairs."""
    global _PCTX
    ctx = PrefillContext(hist_base, hist16, lo, hi, k, model.config.num_hidden_layers)
    prev_impl = model.config._attn_implementation
    _PCTX = ctx
    model.set_attn_implementation(PREFILL_ATTN_NAME)
    try:
        for s in range(0, ids.shape[1], chunk):
            x = ids[:, s:s + chunk]
            pos = torch.arange(pos0 + s, pos0 + s + x.shape[1], device=ids.device)[None]
            model(input_ids=x, position_ids=pos, use_cache=False, logits_to_keep=1)
    finally:
        model.set_attn_implementation(prev_impl)
        _PCTX = None
    return [(torch.cat(a, dim=1), torch.cat(b, dim=1)) for a, b in zip(ctx.fk, ctx.fv)], ctx.pairs


def estimate_peak_bytes(cfg, weight_bytes, P, F, rows, base_bits=2, max_steps=64, prefill_chunk=2048):
    """Rough peak GPU memory of run_pause on one pause (history P tokens, tool output F tokens), in bytes.

    History copies (one per precision used: bf16, base, uniform references) dominate. The peak is the larger of
    the tool-output prefill phase (copies + one more copy while DynamicCache concatenates) and the decode phase
    (copies + tool-output KV + per-layer scratch: fp32 casts of the copies, ~6 attention-probability tensors of
    (rows, heads, P), previous-step scores, fetch bookkeeping). Adds 1.5 GB for the CUDA context / allocator.
    """
    L, kvh, H = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.num_attention_heads
    d = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
    tok = 2 * kvh * d * 2 * L  # bf16 K + V of one token, all layers
    base = f"int{base_bits}"
    bases = ({r.hist for r in rows} | {r.fresh for r in rows} | {"bf16", base}) - {"pf"}
    copies = len(bases) * P * tok
    R = len(rows)
    n_fresh = len({r.fresh for r in rows} | {base})
    prefill_act = prefill_chunk * (cfg.hidden_size * 6 + getattr(cfg, "intermediate_size", 4 * cfg.hidden_size) * 3) * 2 \
        + prefill_chunk * H * 64 * 4
    prefill_phase = copies + P * tok + prefill_act
    n_prev = sum(r.policy in PREV_POLICIES and not r.shadow for r in rows)
    scratch = (len(bases) * 2 * kvh * P * d * 4            # fp32 casts of the history copies (one layer)
               + 6 * R * H * (P + F + max_steps) * 4)      # logits / probabilities (one layer)
    keep = n_prev * kvh * P * 4 * L + R * kvh * P * L    # previous-step scores, union masks (all layers)
    gen = R * L * kvh * (max_steps + 1) * d * 2 * 2
    decode_phase = copies + n_fresh * F * tok + scratch + keep + gen
    return int(weight_bytes + max(prefill_phase, decode_phase) + 1.5e9)


def hist_bytes(bits, P, lo, hi, block, kvh, d, n_layers):
    """Reload bytes of the history with the middle blocks at `bits` and everything else bf16."""
    mid = (hi - lo) // block
    return (mid * block_bytes(bits, block, kvh, d) + (P - (hi - lo)) * kvh * d * 4) * n_layers


@torch.no_grad()
def run_pause(model, hist_ids, fresh_ids, target_ids, rows, block=16, recent=64, max_steps=64,
              check=False, diag=False, base_bits=2, prefill_k=0, prefill_chunk=64):
    """Run every row on one pause. Returns per-row log-probs/argmax of the first max_steps target
    tokens, fetch statistics and byte accounting; with check=True also compares the per-step
    bf16 and base rows with sdpa teacher forcing over the same tokens (run_gate.run_turn); with
    diag=True also returns per-step, per-layer fetch records (diag_records). Rows with k = MATCH get
    k resolved so that the base history + one step's fetched set uses uniform int(base + 1)'s memory."""
    import dataclasses

    from experiments.gate.run_gate import prefill, run_turn

    cfg = model.config
    n_layers, kvh = cfg.num_hidden_layers, cfg.num_key_value_heads
    d = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    dev = hist_ids.device
    P, F_all = hist_ids.shape[1], fresh_ids.shape[1]
    assert F_all >= 3, "tool output segment too short"
    T = min(max_steps, target_ids.shape[1])
    tgt = target_ids[:, :T]
    nb = P // block
    n_recent = min(nb - 1, math.ceil(recent / block))
    lo, hi = block, (nb - n_recent) * block
    assert hi > lo, "history has no quantized middle blocks"
    base = f"int{base_bits}"
    pair_bytes = 2 * d * 2  # K + V of one token, one KV head, one layer, bf16
    nbytes = {f"int{b}_bytes": hist_bytes(b, P, lo, hi, block, kvh, d, n_layers) for b in (1, 2, 3)}
    nbytes["bf16_bytes"] = P * kvh * d * 4 * n_layers
    k_match = max(0, min(hi - lo, (nbytes[f"int{base_bits + 1}_bytes"] - nbytes[f"{base}_bytes"])
                         // (kvh * n_layers * pair_bytes)))
    k_label = [("match" if r.k == MATCH else str(r.k)) for r in rows]
    rows = [dataclasses.replace(r, k=k_match) if r.k == MATCH else r for r in rows]

    cache, _ = prefill(model, hist_ids)
    K16 = [layer.keys[0] for layer in cache.layers]
    V16 = [layer.values[0] for layer in cache.layers]
    del cache

    def qcopy(src, bits, kind):
        out = []
        for x in src:
            y = roundtrip(x[None], bits, block, kind)[0]
            y[:, :lo], y[:, hi:] = x[:, :lo], x[:, hi:]
            out.append(y)
        return out

    bases = ({r.hist for r in rows} | {r.fresh for r in rows} | {"bf16", base}) - {"pf"}
    hist = {"bf16": list(zip(K16, V16))}
    for b in sorted(bases - {"bf16"}):
        bits = int(b[3:])
        hist[b] = list(zip(qcopy(K16, bits, "key"), qcopy(V16, bits, "value")))

    fresh, prefill_pairs = {}, 0
    for g in sorted({r.fresh for r in rows} | {base}):
        if g == "pf":
            fresh[g], prefill_pairs = prefill_with_fetch(model, fresh_ids[:, :-2], hist[base], hist["bf16"], lo, hi,
                                                         prefill_k, prefill_chunk, P)
            continue
        c = DynamicCache()
        for i, (k, v) in enumerate(hist[g]):
            c.update(k[None], v[None], i)
        c, _ = prefill(model, fresh_ids[:, :-2], c)
        fresh[g] = [(layer.keys[0, :, P:].clone(), layer.values[0, :, P:].clone()) for layer in c.layers]
        del c

    R = len(rows)
    ctx = FetchContext(rows, {b: hist[b] for b in hist}, fresh, lo, hi, n_layers, T + 1, base=base)
    ctx.diag = diag
    shadow = torch.tensor([r.shadow for r in rows], device=dev)
    chain = torch.tensor([r.shadow and r.policy == "spec_chain" for r in rows], device=dev)  # position +1
    same = ctx.same_pos_shadow                                                              # spec shadows
    src = torch.tensor(ctx.score_src, device=dev)  # a shadow's spec row (others: themselves)
    logp = torch.zeros(R, T, device=dev)
    pred = torch.zeros(R, T, dtype=torch.long, device=dev)
    guesses = torch.zeros(R, T, dtype=torch.long, device=dev)  # shadows: token decoded as the guess
    actual = torch.zeros(R, T, dtype=torch.long, device=dev)   # shadows: the token that guess stood for
    pos0 = P + F_all - 2
    with fetch_mode(model, ctx):
        # warm-up: second-to-last tool-output token, no fetching; spec_chain shadows decode the last one (known)
        ctx.t, ctx.fetch = 0, False
        x = torch.where(chain, fresh_ids[0, -1], fresh_ids[0, -2])[:, None]
        pos = torch.full((R, 1), pos0, device=dev) + chain[:, None].long()
        out = model(input_ids=x, position_ids=pos, use_cache=False)
        am = out.logits[:, -1].argmax(dim=-1)
        # spec_chain shadows: their own output (guess of target[0], the input of step 1);
        # spec shadows: their spec row's greedy prediction of step 0's input (the last tool-output token)
        guess = torch.where(same, am[src], am)
        ctx.fetch = True
        for j in range(T):
            ctx.t = j + 1
            real = fresh_ids[0, -1] if j == 0 else tgt[0, j - 1]
            guesses[:, j] = guess
            actual[:, j] = torch.where(chain, tgt[0, j], real)  # chain guesses the next input, spec this one
            x = torch.where(shadow, guess, real)[:, None]
            pos = torch.full((R, 1), pos0 + 1 + j, device=dev) + chain[:, None].long()
            out = model(input_ids=x, position_ids=pos, use_cache=False)
            lg = out.logits[:, -1].float()
            logp[:, j] = torch.log_softmax(lg, dim=-1)[:, tgt[0, j]]
            pred[:, j] = lg.argmax(dim=-1)
            # spec_chain: the shadow's output (conditioned on its own guess); spec: the spec row's greedy
            # prediction given the real token it just decoded
            guess = torch.where(same, pred[src, j], pred[:, j])
    hits = (guesses == actual) & shadow[:, None]
    guess_hits = hits.float().sum(dim=1)

    res = {"P": P, "F": F_all, "T": T, "lo": lo, "hi": hi, "base_bits": base_bits, "k_match": k_match,
           **nbytes, "base_bytes": nbytes[f"{base}_bytes"],
           "logp": logp.cpu(), "pred": pred.cpu(), "rows": []}
    for i, r in enumerate(rows):
        obs = max(float(ctx.st_obs[i]), 1.0)
        union = sum(int(u[i].sum()) for u in ctx.union if u is not None)
        k_res = (hi - lo) if r.policy == "all" else min(r.k, hi - lo)
        sh = next((s for s, p_ in ctx.pair.items() if p_ == i), None)
        pf = r.fresh == "pf"
        res["rows"].append({
            "k": r.k, "k_label": k_label[i],
            "fetched_pairs_per_step": float(ctx.st_pairs[i]) / T,
            "distinct_tokens_per_layer_step": float(ctx.st_distinct[i]) / obs,
            "recall_vs_oracle": (float(ctx.st_recall[i]) / obs) if r.policy not in ("none", "all") else float("nan"),
            "moved_bytes_turn": float(ctx.st_pairs[i]) * pair_bytes,
            "unique_bytes_turn": union * pair_bytes,
            "prefill_moved_bytes": prefill_pairs * pair_bytes if pf else 0,
            "resident_bytes": (nbytes[f"{r.hist}_bytes"] + (k_res * kvh * n_layers * pair_bytes
                                                            if r.policy != "none" else 0)),
            "spec_guess_acc": float(guess_hits[sh]) / T if sh is not None else float("nan"),
        })
    res["guesses"] = [{"scheme": rows[s_].name, "k_label": k_label[s_], "step": j,
                       "guess_id": int(g_), "actual_id": int(a_), "match": int(g_ == a_)}
                      for sh_, s_ in ctx.pair.items()
                      for j, (g_, a_) in enumerate(zip(guesses[sh_].tolist(), actual[sh_].tolist()))]
    res["last_idx"] = {i: v.cpu() for i, v in ctx.last_idx.items()}
    if diag:
        res["diag_records"] = ctx.diag_records

    if check:
        names = [r.name for r in rows]
        for tag, name in (("bf16", "bf16"), ("base", base)):
            if name not in names:
                continue
            i = names.index(name)
            lp, _ = run_turn(model, lambda li, b=name: (hist[b][li][0][None], hist[b][li][1][None]),
                             n_layers, fresh_ids, tgt)
            res[f"check_{tag}_max_abs_logp"] = float((lp.cpu() - logp[i].cpu()).abs().max())
            res[f"check_{tag}_nll_batched"] = float(-lp.mean())
            res[f"check_{tag}_nll_perstep"] = float(-logp[i].mean())
    return res
