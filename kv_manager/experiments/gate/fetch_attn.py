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

Selection policies (top-k over the quantized middle tokens, per KV head unless per_layer):
  s1      previous step's attention over int2 keys (query heads of a KV head summed)
  s2      current step's attention over int2 keys (SpeCache's non-speculative variant)
  sens_a  previous-step attention x query-independent quantization error:
          sum_g p_g,i (||dV_i|| + ||q_g|| ||dK_i|| / sqrt(d)), deployable (norms stored at offload)
  sens_b  previous-step attention x first-order output error of fixing token i:
          sum_g p_g,i ||dV_i + dl_g,i (V2_i - o_g)||, dl_g,i = q_g . dK_i / sqrt(d) (needs bf16 keys)
  oracle  current step's true bf16 attention
  all     every middle token (ceiling for per-step fetching given the row's tool-output prefill;
          with a bf16 tool-output prefill it equals the bf16 row exactly: the noise_ctrl row)
d = bf16 - int2. per_layer: one set per layer, scored by summing over all heads, used by every head.
"""

import math
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from transformers import AttentionInterface, DynamicCache

from experiments.gate.fakequant import roundtrip
from kvm.quant import block_bytes

ATTN_NAME = "gate_fetch"
PREV_POLICIES = ("s1", "sens_a", "sens_b")
FETCH_POLICIES = PREV_POLICIES + ("s2", "oracle", "all")
_CTX = None


@dataclass(frozen=True)
class Row:
    name: str
    hist: str = "int2"
    fresh: str = "int2"
    policy: str = "none"
    k: int = 0
    per_layer: bool = False


def default_rows(ks=(16, 32, 64, 128)) -> list[Row]:
    rows = [Row("bf16", "bf16", "bf16"), Row("int2"), Row("int3", "int3", "int3"), Row("fetch_all", policy="all"),
            # noise-floor control: equal to the bf16 row in exact arithmetic, computed through the fetch path,
            # so its dNLL is the bf16 rounding noise of the per-step path on that pause
            Row("noise_ctrl", hist="int2", fresh="bf16", policy="all")]
    for k in ks:
        rows += [Row(p, policy=p, k=k) for p in ("s1", "s2", "sens_a", "sens_b", "oracle")]
        rows.append(Row("s1_layer", policy="s1", k=k, per_layer=True))
    return rows


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
    def __init__(self, rows, hist, fresh, lo, hi, n_layers, max_tokens):
        self.rows, self.hist, self.fresh = rows, hist, fresh
        self.lo, self.hi, self.n_layers, self.max_tokens = lo, hi, n_layers, max_tokens
        any_k = next(iter(hist.values()))[0][0]
        dev = any_k.device
        self.P, self.F = any_k.shape[1], next(iter(fresh.values()))[0][0].shape[1]
        self.t, self.fetch = 0, False
        R = len(rows)

        def ix(cond):
            return torch.tensor([i for i, r in enumerate(rows) if cond(r)], device=dev, dtype=torch.long)

        self.by_hist = {b: ix(lambda r, b=b: r.hist == b) for b in {r.hist for r in rows}}
        self.by_fresh = {g: ix(lambda r, g=g: r.fresh == g) for g in {r.fresh for r in rows}}
        # one selection group per (policy, per_layer, base): rows in a group are scored together
        self.groups = {}
        for i, r in enumerate(rows):
            if r.policy != "none":
                self.groups.setdefault((r.policy, r.per_layer, r.hist), []).append(i)
        self.fetch_rows = ix(lambda r: r.policy != "none")
        self.prev_rows = ix(lambda r: r.policy in PREV_POLICIES)
        self.gen = [None] * n_layers
        self.prev = [dict() for _ in range(n_layers)]
        self.err = [None] * n_layers
        self.union = [None] * n_layers
        z = lambda: torch.zeros(R, device=dev, dtype=torch.float64)
        self.st_pairs, self.st_distinct, self.st_recall, self.st_obs = z(), z(), z(), z()
        self.last_idx = {}

    def errors(self, l):
        """Per-layer quantities that do not depend on the query, over the middle tokens."""
        if self.err[l] is None:
            lo, hi = self.lo, self.hi
            K16, V16 = (x[:, lo:hi].float() for x in self.hist["bf16"][l])
            K2, V2 = (x[:, lo:hi].float() for x in self.hist["int2"][l])
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
    if ctx.gen[l] is None:
        ctx.gen[l] = (key.new_zeros(R, kvh, ctx.max_tokens, d), value.new_zeros(R, kvh, ctx.max_tokens, d))
    gk, gv = ctx.gen[l]
    gk[:, :, t], gv[:, :, t] = key[:, :, 0], value[:, :, 0]
    q = query[:, :, 0].float().reshape(R, kvh, G, d)
    hk = torch.arange(kvh, device=q.device)[None, :, None]

    Kf = {b: ctx.hist[b][l][0].float() for b in ctx.hist}
    Vf = {b: ctx.hist[b][l][1].float() for b in ctx.hist}
    lh = q.new_empty(R, kvh, G, P)
    for b, rows in ctx.by_hist.items():
        lh[rows] = torch.einsum("rkgd,kpd->rkgp", q[rows], Kf[b]) * scaling
    lt = q.new_empty(R, kvh, G, F + t + 1)
    for g, rows in ctx.by_fresh.items():
        lt[rows, :, :, :F] = torch.einsum("rkgd,kfd->rkgf", q[rows], ctx.fresh[g][l][0].float()) * scaling
    lt[..., F:] = torch.einsum("rkgd,rktd->rkgt", q, gk[:, :, :t + 1].float()) * scaling

    def mid_probs(lhist, rows):
        """Attention probabilities over the middle history tokens, (n, kvh, G, hi - lo)."""
        return torch.cat([lhist, lt[rows]], dim=-1).softmax(dim=-1)[..., lo:hi]

    # int2 (pre-fetch) probabilities of rows that score with them, before any logit is replaced
    p2 = {}
    for (pol, per_layer, base), rr in ctx.groups.items():
        if pol in PREV_POLICIES or pol == "s2":
            rows = torch.tensor(rr, device=q.device)
            p2[(pol, per_layer, base)] = (rows, mid_probs(lh[rows], rows))
    l16 = None
    if ctx.fetch and len(ctx.fetch_rows):
        l16 = torch.einsum("rkgd,kpd->rkgp", q[ctx.fetch_rows], Kf["bf16"]) * scaling
        pos16 = {int(r): i for i, r in enumerate(ctx.fetch_rows.tolist())}

    selections = []
    if ctx.fetch:
        for (pol, per_layer, base), rr in ctx.groups.items():
            rows = torch.tensor(rr, device=q.device)
            n = len(rr)
            if pol == "all":
                top = torch.arange(lo, hi, device=q.device).expand(n, kvh, -1)
                valid = torch.ones(n, hi - lo, dtype=torch.bool, device=q.device)
            else:
                if pol in PREV_POLICIES:
                    score = torch.stack([ctx.prev[l][i] for i in rr])
                elif pol == "s2":
                    score = p2[(pol, per_layer, base)][1].sum(dim=2)
                else:  # oracle
                    score = mid_probs(l16[[pos16[i] for i in rr]], rows).sum(dim=2)
                if per_layer:
                    score = score.sum(dim=1, keepdim=True)
                ks = torch.tensor([min(ctx.rows[i].k, hi - lo) for i in rr], device=q.device)
                kmax = int(ks.max())
                top = score.topk(kmax, dim=-1).indices.expand(n, kvh, kmax) + lo
                valid = torch.arange(kmax, device=q.device)[None, :] < ks[:, None]
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
    o += torch.einsum("rkgt,rktd->rkgd", pt[..., F:], gv[:, :, :t + 1].float())
    for pol, per_layer, base, rr, rows, top, valid, idxG, vG in selections:
        dV = Vf["bf16"][hk, top] - Vf[base][hk, top]
        o[rows] += torch.einsum("rkgj,rkjd->rkgd", ph[rows].gather(-1, idxG) * vG, dV)

    # statistics of this step's fetch
    for pol, per_layer, base, rr, rows, top, valid, idxG, vG in selections:
        n = len(rr)
        mask = torch.zeros(n, kvh, hi - lo, dtype=torch.bool, device=q.device)
        mask.scatter_(-1, top - lo, valid[:, None, :].expand(-1, kvh, -1))
        if ctx.union[l] is None:
            ctx.union[l] = torch.zeros(R, kvh, hi - lo, dtype=torch.bool, device=q.device)
        ctx.union[l][rows] |= mask
        ctx.st_pairs[rows] += mask.sum(dim=(1, 2)).double()
        ctx.st_distinct[rows] += mask.any(dim=1).sum(dim=-1).double()
        ctx.st_obs[rows] += 1
        if pol != "all":
            s16 = mid_probs(l16[[pos16[i] for i in rr]], rows).sum(dim=2)
            if per_layer:
                s16 = s16.sum(dim=1, keepdim=True)
            otop = s16.topk(top.shape[-1], dim=-1).indices.expand(n, kvh, -1)
            omask = torch.zeros_like(mask).scatter_(-1, otop, valid[:, None, :].expand(-1, kvh, -1))
            ctx.st_recall[rows] += ((mask & omask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1)).double()

    # scores for the next step's selection (previous-step policies)
    for (pol, per_layer, base), (rows, pr) in p2.items():
        if pol == "s1":
            score = pr.sum(dim=2)
        elif pol == "sens_a":
            e = ctx.errors(l)
            qn = q[rows].norm(dim=-1)[..., None]
            score = (pr * (e["dv"][None, :, None, :] + qn * e["dk"][None, :, None, :] * scaling)).sum(dim=2)
        elif pol == "sens_b":
            K16, V16 = (x[:, lo:hi].float() for x in ctx.hist["bf16"][l])
            K2, V2 = (x[:, lo:hi].float() for x in ctx.hist["int2"][l])
            score = (pr * first_order_norm(V16 - V2, K16 - K2, V2, q[rows], o[rows], scaling)).sum(dim=2)
        else:
            continue
        for j, i in enumerate(rows.tolist()):
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


def hist_bytes(bits, P, lo, hi, block, kvh, d, n_layers):
    """Reload bytes of the history with the middle blocks at `bits` and everything else bf16."""
    mid = (hi - lo) // block
    return (mid * block_bytes(bits, block, kvh, d) + (P - (hi - lo)) * kvh * d * 4) * n_layers


@torch.no_grad()
def run_pause(model, hist_ids, fresh_ids, target_ids, rows, block=16, recent=64, max_steps=64,
              check=False):
    """Run every row on one pause. Returns per-row log-probs/argmax of the first max_steps target
    tokens, fetch statistics and byte accounting; with check=True also compares the per-step
    bf16 and int2 rows with sdpa teacher forcing over the same tokens (run_gate.run_turn)."""
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

    bases = {r.hist for r in rows} | {r.fresh for r in rows} | {"bf16", "int2"}
    hist = {"bf16": list(zip(K16, V16))}
    for b in sorted(bases - {"bf16"}):
        bits = int(b[3:])
        hist[b] = list(zip(qcopy(K16, bits, "key"), qcopy(V16, bits, "value")))

    fresh = {}
    for g in sorted({r.fresh for r in rows}):
        c = DynamicCache()
        for i, (k, v) in enumerate(hist[g]):
            c.update(k[None], v[None], i)
        c, _ = prefill(model, fresh_ids[:, :-2], c)
        fresh[g] = [(layer.keys[0, :, P:].clone(), layer.values[0, :, P:].clone()) for layer in c.layers]
        del c

    R = len(rows)
    ctx = FetchContext(rows, {b: hist[b] for b in hist}, fresh, lo, hi, n_layers, T + 1)
    logp = torch.zeros(R, T, device=dev)
    pred = torch.zeros(R, T, dtype=torch.long, device=dev)
    pos0 = P + F_all - 2
    with fetch_mode(model, ctx):
        ctx.t, ctx.fetch = 0, False  # warm-up: second-to-last tool-output token, no fetching
        model(input_ids=fresh_ids[:, -2:-1].expand(R, 1), position_ids=torch.full((R, 1), pos0, device=dev),
              use_cache=False)
        ctx.fetch = True
        for j in range(T):
            ctx.t = j + 1
            x = fresh_ids[:, -1:] if j == 0 else tgt[:, j - 1:j]
            out = model(input_ids=x.expand(R, 1), position_ids=torch.full((R, 1), pos0 + 1 + j, device=dev),
                        use_cache=False)
            lg = out.logits[:, -1].float()
            logp[:, j] = torch.log_softmax(lg, dim=-1)[:, tgt[0, j]]
            pred[:, j] = lg.argmax(dim=-1)

    pair_bytes = 2 * d * 2  # K + V of one token, one KV head, one layer, bf16
    res = {"P": P, "F": F_all, "T": T, "lo": lo, "hi": hi,
           "bf16_bytes": P * kvh * d * 4 * n_layers,
           "int2_bytes": hist_bytes(2, P, lo, hi, block, kvh, d, n_layers),
           "int3_bytes": hist_bytes(3, P, lo, hi, block, kvh, d, n_layers),
           "logp": logp.cpu(), "pred": pred.cpu(), "rows": []}
    for i, r in enumerate(rows):
        obs = max(float(ctx.st_obs[i]), 1.0)
        union = sum(int(u[i].sum()) for u in ctx.union if u is not None)
        k_res = (hi - lo) if r.policy == "all" else min(r.k, hi - lo)
        res["rows"].append({
            "fetched_pairs_per_step": float(ctx.st_pairs[i]) / T,
            "distinct_tokens_per_layer_step": float(ctx.st_distinct[i]) / obs,
            "recall_vs_oracle": (float(ctx.st_recall[i]) / obs) if r.policy not in ("none", "all") else float("nan"),
            "moved_bytes_turn": float(ctx.st_pairs[i]) * pair_bytes,
            "unique_bytes_turn": union * pair_bytes,
            "resident_bytes": (res[f"{r.hist}_bytes"] + (k_res * kvh * n_layers * pair_bytes
                                                         if r.policy != "none" else 0)),
        })
    res["last_idx"] = {i: v.cpu() for i, v in ctx.last_idx.items()}

    if check:
        names = [r.name for r in rows]
        for name, base in (("bf16", "bf16"), ("int2", "int2")):
            if name not in names:
                continue
            i = names.index(name)
            lp, _ = run_turn(model, lambda li, b=base: (hist[b][li][0][None], hist[b][li][1][None]),
                             n_layers, fresh_ids, tgt)
            res[f"check_{name}_max_abs_logp"] = float((lp.cpu() - logp[i].cpu()).abs().max())
            res[f"check_{name}_nll_batched"] = float(-lp.mean())
            res[f"check_{name}_nll_perstep"] = float(-logp[i].mean())
    return res
