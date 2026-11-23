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

Multi-guess speculation (row field m > 1; m = 1 reproduces spec / spec_chain exactly): a spec-family row is followed
by m shadows (cand 0..m-1). The candidates are the top-m tokens of the distribution the single guess is the argmax
of: for the anchored family (spec_union, spec_wtd) the spec row's own next-token distribution at the previous step
(given the real token), for the chained family (spec_chain_union, spec_chain_wtd) the cand-0 shadow's output at the
previous step (the chain continues from candidate 0 only). Each shadow decodes its candidate and scores the base
keys with its query (anchored: for the current step; chained: for the next step).
  *_union   the union of each candidate's top-k per KV head (up to m*k tokens; the fetched count is reported)
  *_wtd     top-k of sum_c p_c * score_c, p_c the candidate's probability (budget k, as s2)
spec_cand_cover = share of steps where the token a guess stands for is among the m candidates.

Fetch pool (row field pool = M > 0; policies s2_pool, spec_chain_pool): a bf16 pool of M tokens per (layer, KV head).
Each step the row scores as s2 / spec_chain, takes the top-k, fetches only the ones not already in the pool (misses),
evicts the least recently selected tokens (ties: the lower current score) to stay at M, and attends with the whole
pool. Moved bytes = misses only; pool_hit_rate = 1 - misses / selected. With k = M the pool is the top-k set at
every step (equal to s2 at that k). M may be MATCH (k_match) or HALF (k_match // 2).

s2_k2score: ranks with the current query over the int2 reference copy's keys (sharper than an int1 base's), then
attends over the row's own base with that set fetched (diagnostic: isolates the scoring keys).

Base k2v1 (run_pause base="k2v1"): keys int2, values int1 (SpeCache's 1-bit quantizer), same KIVI blocking; its
uniform reference for equal memory is int2. Rows may sit on different bases in one run (e.g. int1 comparators in a
k2v1 run, with their own tool-output group "pf:int1"); MATCH resolves per row's base so that base + fetched set =
uniform int2's memory (int(b+1) for an int base b).
"""

import dataclasses
import math
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from transformers import AttentionInterface, DynamicCache

from experiments.gate.fakequant import roundtrip
from kvm.quant import block_bytes

ATTN_NAME = "gate_fetch"
ANCHORED = ("spec", "spec_union", "spec_wtd")                                   # guess the current input
CHAIN = ("spec_chain", "spec_chain_union", "spec_chain_wtd", "spec_chain_pool")  # chained guess of the next input
UNION = ("spec_union", "spec_chain_union")
WTD = ("spec_wtd", "spec_chain_wtd")
POOL = ("s2_pool", "spec_chain_pool")
PREV_POLICIES = ("s1", "sens_a", "sens_b") + CHAIN   # select with scores stored at the previous step
CUR_INT2_POLICIES = ("s2", "s2sens_a", "s2sens_b", "s2_pool")   # select with the current step's base attention
SPEC_POLICIES = ANCHORED + CHAIN                     # rows that come with shadow rows
FETCH_POLICIES = PREV_POLICIES + CUR_INT2_POLICIES + ANCHORED + ("s2_k2score", "oracle", "all")
NEXT = {"int1": "int2", "int2": "int3", "k2v1": "int2"}  # uniform reference with the equal-memory budget
_CTX = None


@dataclass(frozen=True)
class Row:
    name: str
    hist: str = "int2"
    fresh: str = "int2"
    policy: str = "none"
    k: int = 0
    per_layer: bool = False
    shadow: bool = False  # speculative-token row; follows its spec-family row (same policy, k), after cand - 1 others
    m: int = 1            # spec family: number of candidates (= shadows)
    cand: int = 0         # shadows: candidate index (0 = the argmax, the single guess of spec / spec_chain)
    pool: int = 0         # pool policies: pool size M per (layer, KV head) (MATCH / HALF resolved per pause)


MATCH = -1  # k resolved per pause so that base + fetched set uses the memory of uniform int(base + 1)
HALF = -2   # pool size k_match // 2


def spec_family(name, pol, hist, fresh, k, m=1):
    """A spec-family row followed by its m shadows."""
    return [Row(name, hist, fresh, policy=pol, k=k, m=m)] + \
           [Row(f"{name}_shadow{c or ''}", hist, fresh, policy=pol, k=k, m=m, shadow=True, cand=c) for c in range(m)]


ROW_SETS = ("default", "a1", "a2", "c")


def study_rows(row_set, base="int1", prefill=True, uniform_refs=(2,), s2_extra_ks=()) -> list[Row]:
    """Fetch study run 5 row sets (each job's rules are decided within that job):
      a1  multi-guess speculation: s2, oracle, spec, spec_chain at 64/128/match (+ s2 at s2_extra_ks for the union
          rows' size comparison); spec_union{2,3} and spec_chain_union{2,3} at 128; spec_wtd{2,3} and
          spec_chain_wtd{2,3} at 128 and match
      a2  pools and scoring keys: s2, oracle, s2_k2score, spec_chain at 16/64/128/match; s2_pool and spec_chain_pool
          at 16/64/128 with M = k_match and k_match // 2
      c   base k2v1: s2, oracle, spec_chain at 16/64/128/match; spec_chain_wtd3 at 128/match; s2_pool at 16/64/128
          with M = k_match and at 16/64 with M = k_match // 2; int1-base comparators s2@16, oracle@16, s2@match (prefilled over int1 with the same prefill fetch)
    Every set has bf16, the base, the uniform references, fetch_all, noise_ctrl (and fetch_all_noprefill)."""
    fr = "pf" if prefill else base
    rows = [Row("bf16", "bf16", "bf16"), Row(base, base, base)]
    rows += [Row(f"int{b}", f"int{b}", f"int{b}") for b in uniform_refs if f"int{b}" != base]
    rows += [Row("fetch_all", base, fr, policy="all"), Row("noise_ctrl", base, "bf16", policy="all")]
    if prefill:
        rows.append(Row("fetch_all_noprefill", base, base, policy="all"))
    B = base
    if row_set == "a1":
        for k in (64, 128, MATCH):
            rows += [Row("s2", B, fr, policy="s2", k=k), Row("oracle", B, fr, policy="oracle", k=k)]
            rows += spec_family("spec", "spec", B, fr, k) + spec_family("spec_chain", "spec_chain", B, fr, k)
        rows += [Row("s2", B, fr, policy="s2", k=k) for k in s2_extra_ks]
        for m in (2, 3):
            rows += spec_family(f"spec_union{m}", "spec_union", B, fr, 128, m)
            rows += spec_family(f"spec_chain_union{m}", "spec_chain_union", B, fr, 128, m)
        for k in (128, MATCH):
            for m in (2, 3):
                rows += spec_family(f"spec_wtd{m}", "spec_wtd", B, fr, k, m)
                rows += spec_family(f"spec_chain_wtd{m}", "spec_chain_wtd", B, fr, k, m)
    elif row_set == "a2":
        for k in (16, 64, 128, MATCH):
            rows += [Row(p, B, fr, policy=p, k=k) for p in ("s2", "oracle", "s2_k2score")]
            rows += spec_family("spec_chain", "spec_chain", B, fr, k)
        for k in (16, 64, 128):
            for M, tag in ((MATCH, "Mmatch"), (HALF, "Mhalf")):
                rows.append(Row(f"s2_pool_{tag}", B, fr, policy="s2_pool", k=k, pool=M))
                rows += [dataclasses.replace(r, pool=M) for r in
                         spec_family(f"spec_chain_pool_{tag}", "spec_chain_pool", B, fr, k)]
    elif row_set == "c":
        for k in (16, 64, 128, MATCH):
            rows += [Row("s2", B, fr, policy="s2", k=k), Row("oracle", B, fr, policy="oracle", k=k)]
            rows += spec_family("spec_chain", "spec_chain", B, fr, k)
        for k in (128, MATCH):
            rows += spec_family("spec_chain_wtd3", "spec_chain_wtd", B, fr, k, 3)
        # k2v1's k_match is ~half of int1's (min 247 over the 31 pauses, so M = k_match // 2 >= 123), below 128 on
        # short pauses: s2_pool_Mhalf only at 16 / 64 (run_fetch checks k <= M on every pause before starting)
        rows += [Row("s2_pool_Mmatch", B, fr, policy="s2_pool", k=k, pool=MATCH) for k in (16, 64, 128)]
        rows += [Row("s2_pool_Mhalf", B, fr, policy="s2_pool", k=k, pool=HALF) for k in (16, 64)]
        f1 = "pf:int1" if prefill else "int1"
        rows += [Row("s2_int1base", "int1", f1, policy="s2", k=16), Row("oracle_int1base", "int1", f1, policy="oracle", k=16),
                 Row("s2_int1base", "int1", f1, policy="s2", k=MATCH)]
    else:
        raise ValueError(f"unknown row set {row_set!r}")
    return rows


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

        # shadow i reads its spec row's generated KV and selection scores; a spec-family row is directly followed by
        # its m shadows (cand 0..m-1)
        self.pair, self.shadows = {}, {}
        for i, r in enumerate(rows):
            if r.shadow:
                j = i - 1
                while j >= 0 and rows[j].shadow:
                    j -= 1
                s = rows[j] if j >= 0 else None
                assert s is not None and r.policy in SPEC_POLICIES and s.policy == r.policy and s.k == r.k \
                    and s.m == r.m and s.pool == r.pool and s.per_layer == r.per_layer and s.hist == r.hist \
                    and s.fresh == r.fresh and r.cand == i - j - 1 < r.m, \
                    "shadows must directly follow their spec-family row (same policy, k, m), cand 0..m-1 in order"
                self.pair[i] = j
                self.shadows.setdefault(j, []).append(i)
        for j, sh in self.shadows.items():
            assert len(sh) == rows[j].m, f"{rows[j].name}: {len(sh)} shadows for m = {rows[j].m}"
        for i, r in enumerate(rows):
            assert r.policy not in SPEC_POLICIES or r.shadow or i in self.shadows, f"{r.name} has no shadow row"
        self.shadow_of = {s: sh[0] for s, sh in self.shadows.items()}   # the cand-0 shadow
        self.score_src = [self.pair.get(i, i) for i in range(R)]
        self.gen_src = torch.tensor(self.score_src, device=dev, dtype=torch.long)
        self.is_shadow = torch.tensor([r.shadow for r in rows], device=dev)
        # anchored shadows sit at their spec row's position: they must not see the spec row's current token
        self.same_pos_shadow = torch.tensor([r.shadow and r.policy in ANCHORED for r in rows], device=dev)
        # probability of each shadow's candidate (set by run_pause before every step; 1 at the warm-up)
        self.cand_w = torch.ones(R, device=dev)
        if any(r.policy == "s2_k2score" for r in rows):
            assert "int2" in hist, "s2_k2score needs the int2 history copy (an int2 uniform reference row)"

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
        self.err = [dict() for _ in range(n_layers)]
        self.union = [None] * n_layers
        self.pool = [dict() for _ in range(n_layers)]   # pool rows: (in-pool mask, last-selected step), (kvh, mid)
        z = lambda: torch.zeros(R, device=dev, dtype=torch.float64)
        self.st_pairs, self.st_distinct, self.st_recall, self.st_obs = z(), z(), z(), z()
        self.st_moved, self.st_sel = z(), z()   # pairs moved (pool rows: misses only), pairs selected (pool rows)
        self.last_idx = {}
        self.diag_records = []

    def errors(self, l, base=None):
        """Per-layer quantities that do not depend on the query, over the middle tokens."""
        base = base or self.base
        if base not in self.err[l]:
            lo, hi = self.lo, self.hi
            K16, V16 = (x[:, lo:hi].float() for x in self.hist["bf16"][l])
            K2, V2 = (x[:, lo:hi].float() for x in self.hist[base][l])
            self.err[l][base] = {"dk": (K16 - K2).norm(dim=-1), "dv": (V16 - V2).norm(dim=-1)}
        return self.err[l][base]


def _top_from_score(score, ks, kvh):
    """Top-k per row (k may differ per row) -> indices (n, kvh, kmax) into the middle, valid (n, kvh, kmax)."""
    kmax = max(int(max(ks)), 1)
    n = score.shape[0]
    top = score.topk(kmax, dim=-1).indices.expand(n, kvh, kmax)
    kt = torch.as_tensor(ks, device=score.device)
    valid = (torch.arange(kmax, device=score.device)[None, None, :] < kt[:, None, None]).expand(n, kvh, kmax)
    return top, valid


def _top_from_mask(mask):
    """A set per (row, KV head) given as a mask (n, kvh, mid) -> indices (n, kvh, kmax), valid (prefix per head)."""
    kmax = max(int(mask.sum(dim=-1).max()), 1)
    v, top = mask.float().topk(kmax, dim=-1)
    return top, v > 0


def _topk_mask(score, k):
    """(..., mid) score -> mask of its top-k entries."""
    k = min(int(k), score.shape[-1])
    return torch.zeros_like(score, dtype=torch.bool).scatter_(-1, score.topk(k, dim=-1).indices, True)


def _pool_update(ctx, l, i, score, k, M):
    """Insert row i's top-k (by score, (kvh, mid)) into its pool, evict the least recently selected tokens (ties: the
    lower current score) down to M per KV head, count misses. Returns the pool mask (kvh, mid)."""
    if i not in ctx.pool[l]:
        ctx.pool[l][i] = (torch.zeros_like(score, dtype=torch.bool), torch.zeros_like(score, dtype=torch.int32))
    pool, stamp = ctx.pool[l][i]
    sel = _topk_mask(score, k)
    ctx.st_moved[i] += float((sel & ~pool).sum())
    ctx.st_sel[i] += float(sel.sum())
    stamp[sel] = ctx.t
    pool |= sel
    over = pool.sum(dim=-1) - M
    if int(over.max()) > 0:
        rel = score / score.amax(dim=-1, keepdim=True).clamp_min(1e-30)   # in [0, 1]: breaks ties within a step
        key = torch.where(pool & ~sel, stamp.double() + 0.5 * rel.double(), torch.full_like(rel, float("inf"),
                                                                                           dtype=torch.float64))
        n_ev = int(over.max())
        ev = key.topk(n_ev, dim=-1, largest=False).indices
        ok = torch.arange(n_ev, device=score.device)[None, :] < over[:, None]
        pool.scatter_(-1, ev, pool.gather(-1, ev) & ~ok)
    return pool


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

    def base_output(rows, fg, b):
        """Attention output of `rows` with their base history `b` and no fetch, (n, kvh, G, d)."""
        p = torch.cat([lh[rows], lt[rows]], dim=-1).softmax(dim=-1)
        out = torch.einsum("rkgp,kpd->rkgd", p[..., :P], Vf[b])
        out += torch.einsum("rkgf,kfd->rkgd", p[..., P:P + F], ctx.fresh[fg][l][1].float())
        out += torch.einsum("rkgt,rktd->rkgd", p[..., P + F:P + F + t + 1], gv[src[rows], :, :t + 1].float())
        return out

    mid = hi - lo

    def spec_combine(pol, s_, sc):
        """Selection of spec-family row s_ from its shadows' scores sc(shadow) (kvh, mid): the cand-0 score (spec,
        spec_chain, m = 1), the probability-weighted sum (wtd), or the union of the top-k sets (a mask)."""
        shs = ctx.shadows[s_]
        if pol in UNION:
            k_ = min(ctx.rows[s_].k, mid)
            u = _topk_mask(sc(shs[0]), k_)
            for sh in shs[1:]:
                u |= _topk_mask(sc(sh), k_)
            return u
        if pol in WTD and len(shs) > 1:
            return sum(ctx.cand_w[sh] * sc(sh) for sh in shs)
        return sc(shs[0])

    # base (pre-fetch) probabilities of rows that score with them, before any logit is replaced
    p2 = {}
    for (pol, per_layer, base), rr in ctx.groups.items():
        if pol in PREV_POLICIES or pol in CUR_INT2_POLICIES or pol in ANCHORED:
            rows = torch.tensor(rr, device=dev)
            p2[(pol, per_layer, base)] = (rows, mid_probs(lh[rows], rows))
        elif pol == "s2_k2score" and ctx.fetch:   # scored over the int2 copy's keys
            rows = torch.tensor(rr, device=dev)
            lk2 = torch.einsum("rkgd,kpd->rkgp", q[rows], Kf["int2"]) * scaling
            p2[(pol, per_layer, base)] = (rows, mid_probs(lk2, rows))
            del lk2
    l16 = None
    if ctx.fetch and len(ctx.fetch_rows):
        l16 = torch.einsum("rkgd,kpd->rkgp", q[ctx.fetch_rows], Kf["bf16"]) * scaling
        pos16 = {int(r): i for i, r in enumerate(ctx.fetch_rows.tolist())}

    selections = []
    if ctx.fetch:
        for (pol, per_layer, base), rr in ctx.groups.items():
            key_ = (pol, per_layer, base)
            rows = torch.tensor(rr, device=dev)
            n = len(rr)
            pos = {i: j for j, i in enumerate(rr)}
            if pol == "all":
                top = torch.arange(lo, hi, device=dev).expand(n, kvh, -1)
                valid = torch.ones(n, kvh, mid, dtype=torch.bool, device=dev)
            else:
                if pol in PREV_POLICIES:
                    score = torch.stack([ctx.prev[l][ctx.score_src[i]] for i in rr])
                elif pol in ANCHORED:  # a spec row and its shadows use the shadows' current base attention
                    pr = p2[key_][1].sum(dim=2)
                    comb = {}
                    for i in rr:
                        s_ = ctx.score_src[i]
                        if s_ not in comb:
                            comb[s_] = spec_combine(pol, s_, lambda sh: pr[pos[sh]])
                    score = torch.stack([comb[ctx.score_src[i]] for i in rr])
                elif pol in CUR_INT2_POLICIES or pol == "s2_k2score":
                    pr = p2[key_][1]
                    if pol in ("s2", "s2_pool", "s2_k2score"):
                        score = pr.sum(dim=2)
                    elif pol == "s2sens_a":
                        e = ctx.errors(l, base)
                        qn = q[rows].norm(dim=-1)[..., None]
                        score = (pr * (e["dv"][None, :, None, :] + qn * e["dk"][None, :, None, :] * scaling)).sum(dim=2)
                    else:  # s2sens_b
                        K16m, V16m = (x[:, lo:hi].float() for x in ctx.hist["bf16"][l])
                        K2m, V2m = (x[:, lo:hi].float() for x in ctx.hist[base][l])
                        score = (pr * first_order_norm(V16m - V2m, K16m - K2m, V2m, q[rows],
                                                       base_output(rows, ctx.rows[rr[0]].fresh, base), scaling)).sum(dim=2)
                else:  # oracle
                    score = mid_probs(l16[[pos16[i] for i in rr]], rows).sum(dim=2)
                if per_layer:
                    score = score.sum(dim=1, keepdim=True)
                if pol in POOL:   # one pool per spec / s2_pool row; its shadow attends with the same pool
                    masks = {}
                    for i in rr:
                        s_ = ctx.score_src[i]
                        if s_ not in masks:
                            r_ = ctx.rows[s_]
                            masks[s_] = _pool_update(ctx, l, s_, score[pos[i]], min(r_.k, mid), r_.pool)
                    top, valid = _top_from_mask(torch.stack([masks[ctx.score_src[i]] for i in rr]))
                elif pol in UNION:
                    top, valid = _top_from_mask(score if score.dtype == torch.bool else score > 0)
                else:
                    top, valid = _top_from_score(score, [min(ctx.rows[i].k, mid) for i in rr], kvh)
                top = top + lo
            idxG = top[:, :, None, :].expand(-1, -1, G, -1)
            vG = valid[:, :, None, :]
            new = torch.einsum("rkgd,rkjd->rkgj", q[rows], Kf["bf16"][hk, top]) * scaling
            sub = lh[rows]
            sub.scatter_(-1, idxG, torch.where(vG, new, sub.gather(-1, idxG)))
            lh[rows] = sub
            selections.append((pol, per_layer, base, rr, rows, top, valid, idxG, vG))
            for j, i in enumerate(rr):
                vj = valid[j]
                # (kvh, k) when every head has the same set size, else a list of per-head index tensors (unions, pools)
                ctx.last_idx[i] = (top[j][:, vj[0]] if bool((vj == vj[0]).all())
                                   else [top[j][h][vj[h]] for h in range(kvh)])

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
        mask.scatter_(-1, top - lo, valid)
        if ctx.union[l] is None:
            ctx.union[l] = torch.zeros(R, kvh, hi - lo, dtype=torch.bool, device=dev)
        ctx.union[l][rows] |= mask
        ctx.st_pairs[rows] += mask.sum(dim=(1, 2)).double()
        if pol not in POOL:   # pool rows count their misses in _pool_update
            ctx.st_moved[rows] += mask.sum(dim=(1, 2)).double()
        ctx.st_distinct[rows] += mask.any(dim=1).sum(dim=-1).double()
        ctx.st_obs[rows] += 1
        if pol != "all":
            p16 = mid_probs(l16[[pos16[i] for i in rr]], rows)  # (n, kvh, G, mid), current bf16 attention
            s16 = p16.sum(dim=2)
            if per_layer:
                s16 = s16.sum(dim=1, keepdim=True)
            otop = s16.topk(top.shape[-1], dim=-1).indices.expand(n, kvh, -1)
            omask = torch.zeros_like(mask).scatter_(-1, otop, valid)
            ctx.st_recall[rows] += ((mask & omask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1)).double()
            if ctx.diag:
                pm2 = torch.cat([lh_pre[rows], lt[rows]], dim=-1).softmax(dim=-1)[..., lo:hi]
                pma = ph[rows][..., lo:hi]
                mass = lambda pr, m: (pr * m[:, :, None, :]).sum(dim=-1).mean(dim=2)  # (n, kvh), mean over G
                for j, i in enumerate(rr):
                    if ctx.rows[i].shadow:
                        continue
                    ctx.diag_records.append({
                        "row": i, "t": t, "layer": l, "idx": top[j][:, valid[j].all(dim=0)].to(torch.int32).cpu(),
                        "mass_actual": mass(pma[j:j + 1], mask[j:j + 1])[0].cpu(),
                        "mass_int2": mass(pm2[j:j + 1], mask[j:j + 1])[0].cpu(),
                        "mass_bf16": mass(p16[j:j + 1], mask[j:j + 1])[0].cpu(),
                        "oracle_mass_bf16": mass(p16[j:j + 1], omask[j:j + 1])[0].cpu(),
                        "mid_mass_bf16": p16[j].sum(dim=-1).mean(dim=1).cpu(),
                        "mid_mass_int2": pm2[j].sum(dim=-1).mean(dim=1).cpu(),
                        "overlap": (mask[j] & omask[j]).sum(dim=-1).cpu(),
                    })

    # scores for the next step's selection (previous-step policies; chained shadows score for their spec row)
    for (pol, per_layer, base), (rows, pr) in p2.items():
        if pol in CUR_INT2_POLICIES or pol in ANCHORED or pol == "s2_k2score":
            continue
        if pol == "s1" or pol in CHAIN:
            score = pr.sum(dim=2)
        elif pol == "sens_a":
            e = ctx.errors(l, base)
            qn = q[rows].norm(dim=-1)[..., None]
            score = (pr * (e["dv"][None, :, None, :] + qn * e["dk"][None, :, None, :] * scaling)).sum(dim=2)
        else:  # sens_b
            K16m, V16m = (x[:, lo:hi].float() for x in ctx.hist["bf16"][l])
            K2m, V2m = (x[:, lo:hi].float() for x in ctx.hist[base][l])
            score = (pr * first_order_norm(V16m - V2m, K16m - K2m, V2m, q[rows], o[rows], scaling)).sum(dim=2)
        rl = rows.tolist()
        if pol in CHAIN:
            pos = {i: j for j, i in enumerate(rl)}
            for s_ in dict.fromkeys(ctx.pair[i] for i in rl if ctx.rows[i].shadow):
                v = spec_combine(pol, s_, lambda sh: score[pos[sh]])
                ctx.prev[l][s_] = v.float() if v.dtype == torch.bool else v
        else:
            for j, i in enumerate(rl):
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


def _hist_names(rows, base):
    """History copies a row set needs (tool-output groups "pf" / "pf:<base>" prefill over a history copy)."""
    out = {"bf16", base}
    for r in rows:
        out.add(r.hist)
        for g in (r.fresh,):
            out.add(base if g == "pf" else g[3:] if g.startswith("pf:") else g)
    return out


def estimate_peak_bytes(cfg, weight_bytes, P, F, rows, base_bits=2, max_steps=64, prefill_chunk=2048, base=None):
    """Rough peak GPU memory of run_pause on one pause (history P tokens, tool output F tokens), in bytes.

    History copies (one per precision used: bf16, base, uniform references) dominate. The peak is the larger of
    the tool-output prefill phase (copies + one more copy while DynamicCache concatenates) and the decode phase
    (copies + tool-output KV + per-layer scratch: fp32 casts of the copies, ~6 attention-probability tensors of
    (rows, heads, P), previous-step scores, fetch bookkeeping, pool state). Adds 1.5 GB for the CUDA context /
    allocator. (Run 3/4 measured ~1.6 GB more than this on the largest pause.)
    """
    L, kvh, H = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.num_attention_heads
    d = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
    tok = 2 * kvh * d * 2 * L  # bf16 K + V of one token, all layers
    base = base or f"int{base_bits}"
    bases = _hist_names(rows, base)
    q = bases - {"bf16"}
    # K and V halves are stored once per bit width (k2v1 shares the int2 keys and the int1 values)
    copies = (1 + (len({_kv_bits(b)[0] for b in q}) + len({_kv_bits(b)[1] for b in q})) / 2) * P * tok
    R = len(rows)
    n_fresh = len({r.fresh for r in rows} | {base})
    prefill_act = prefill_chunk * (cfg.hidden_size * 6 + getattr(cfg, "intermediate_size", 4 * cfg.hidden_size) * 3) * 2 \
        + prefill_chunk * H * 64 * 4
    prefill_phase = copies + P * tok + prefill_act
    n_prev = sum(r.policy in PREV_POLICIES and not r.shadow for r in rows)
    n_k2 = sum(r.policy == "s2_k2score" for r in rows)
    n_pool = sum(r.policy in POOL and not r.shadow for r in rows)
    scratch = (len(bases) * 2 * kvh * P * d * 4            # fp32 casts of the history copies (one layer)
               + (6 * R + 2 * n_k2) * H * (P + F + max_steps) * 4)   # logits / probabilities (one layer)
    keep = n_prev * kvh * P * 4 * L + R * kvh * P * L + n_pool * kvh * P * 5 * L  # scores, union masks, pools
    gen = R * L * kvh * (max_steps + 1) * d * 2 * 2
    decode_phase = copies + n_fresh * F * tok + scratch + keep + gen
    return int(weight_bytes + max(prefill_phase, decode_phase) + 1.5e9)


def _kv_bits(base):
    """(key bits, value bits) of a quantized base: "intN" -> (N, N), "k2v1" -> (2, 1)."""
    return (2, 1) if base == "k2v1" else (int(base[3:]), int(base[3:]))


def _block_bytes(base, block, kvh, d):
    """Bytes of one block of one layer at a base precision ("intN" or "k2v1": int2 keys + int1 values)."""
    if base == "k2v1":
        elems = kvh * block * d
        return elems * 2 // 8 + elems * 1 // 8 + kvh * d * 2 * 2 + kvh * block * 2 * 2
    return block_bytes(int(base[3:]), block, kvh, d)


def hist_bytes(bits, P, lo, hi, block, kvh, d, n_layers):
    """Reload bytes of the history with the middle blocks at `bits` (int or "k2v1") and everything else bf16."""
    mid = (hi - lo) // block
    b = bits if isinstance(bits, str) else f"int{bits}"
    return (mid * _block_bytes(b, block, kvh, d) + (P - (hi - lo)) * kvh * d * 4) * n_layers


@torch.no_grad()
def generate_self_target(model, hist_ids, fresh_ids, max_new, temperature=0.0, seed=0, eos_ids=()):
    """The model's own next message after the pause: bf16 prefill of the history and the tool output (which ends
    with the assistant header), then up to max_new tokens, greedy at temperature 0, otherwise sampled from
    softmax(logits / temperature) with a torch.Generator seeded with `seed` (no top-k/top-p). Stops after the
    first token in eos_ids (kept, like the end-of-turn token of the real targets). Returns a list of token ids."""
    from experiments.gate.run_gate import prefill

    dev = hist_ids.device
    cache, logits = prefill(model, torch.cat([hist_ids, fresh_ids], dim=1))
    logits = logits[0]
    gen = torch.Generator(device=dev).manual_seed(int(seed)) if temperature > 0 else None
    out, eos = [], set(int(e) for e in eos_ids)
    for i in range(max_new):
        if temperature > 0:
            t = int(torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1, generator=gen))
        else:
            t = int(logits.argmax())
        out.append(t)
        if t in eos or i == max_new - 1:
            break
        o = model(input_ids=torch.tensor([[t]], device=dev), past_key_values=cache, use_cache=True)
        cache, logits = o.past_key_values, o.logits[0, -1].float()
    del cache
    return out


def pause_geometry(P, block, recent, kvh, d, n_layers):
    """Middle (quantized) range [lo, hi), reload bytes per base, and k_match(base): the tokens per KV head and
    layer that fill the base up to its uniform reference NEXT[base] (MATCH)."""
    nb = P // block
    n_recent = min(nb - 1, math.ceil(recent / block))
    lo, hi = block, (nb - n_recent) * block
    assert hi > lo, "history has no quantized middle blocks"
    pair_bytes = 2 * d * 2
    nbytes = {f"{b}_bytes": hist_bytes(b, P, lo, hi, block, kvh, d, n_layers) for b in ("int1", "int2", "int3", "k2v1")}
    nbytes["bf16_bytes"] = P * kvh * d * 4 * n_layers

    def kmatch(h):
        return max(0, min(hi - lo, (nbytes[f"{NEXT[h]}_bytes"] - nbytes[f"{h}_bytes"]) // (kvh * n_layers * pair_bytes)))

    return lo, hi, nbytes, kmatch


def resolve_rows(rows, P, block, recent, kvh, d, n_layers):
    """Rows with k = MATCH and pool = MATCH / HALF resolved for a history of P tokens (per row's base). Returns
    (rows, problems): a pool row whose k exceeds its pool size M is a problem (never clamped)."""
    lo, hi, _, kmatch = pause_geometry(P, block, recent, kvh, d, n_layers)
    out, bad = [], []
    for r in rows:
        km = kmatch(r.hist) if (r.k == MATCH or r.pool in (MATCH, HALF)) else 0
        k = km if r.k == MATCH else r.k
        M = km if r.pool == MATCH else km // 2 if r.pool == HALF else r.pool
        if r.policy in POOL and not r.shadow and not 0 < min(k, hi - lo) <= M:
            bad.append(f"{r.name}@{'match' if r.k == MATCH else r.k}: pool size {M} below k = {k}")
        out.append(dataclasses.replace(r, k=k, pool=M))
    return out, bad


@torch.no_grad()
def run_pause(model, hist_ids, fresh_ids, target_ids, rows, block=16, recent=64, max_steps=64,
              check=False, diag=False, base_bits=2, prefill_k=0, prefill_chunk=64, base=None):
    """Run every row on one pause. Returns per-row log-probs/argmax of the first max_steps target
    tokens, fetch statistics and byte accounting; with check=True also compares the per-step
    bf16 and base rows with sdpa teacher forcing over the same tokens (run_gate.run_turn); with
    diag=True also returns per-step, per-layer fetch records (diag_records). Rows with k = MATCH get
    k resolved so that the base history + one step's fetched set uses uniform int(base + 1)'s memory (per row's
    base; NEXT); pool sizes MATCH / HALF resolve to that k_match / k_match // 2. base: "int1" / "int2" / "k2v1"
    (default int<base_bits>)."""
    from experiments.gate.run_gate import prefill, run_turn

    cfg = model.config
    n_layers, kvh = cfg.num_hidden_layers, cfg.num_key_value_heads
    d = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    dev = hist_ids.device
    P, F_all = hist_ids.shape[1], fresh_ids.shape[1]
    assert F_all >= 3, "tool output segment too short"
    T = min(max_steps, target_ids.shape[1])
    tgt = target_ids[:, :T]
    base = base or f"int{base_bits}"
    pair_bytes = 2 * d * 2  # K + V of one token, one KV head, one layer, bf16
    lo, hi, nbytes, kmatch = pause_geometry(P, block, recent, kvh, d, n_layers)
    k_match = kmatch(base)
    k_label = [("match" if r.k == MATCH else str(r.k)) for r in rows]
    pool_label = [("match" if r.pool == MATCH else "half" if r.pool == HALF else str(r.pool)) for r in rows]
    rows, bad = resolve_rows(rows, P, block, recent, kvh, d, n_layers)
    assert not bad, "pool rows with k > M (no clamping; run_fetch checks every pause before starting): " + "; ".join(bad)

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

    hist = {"bf16": list(zip(K16, V16))}
    # quantized keys / values per bit width, shared between bases: k2v1 = the int2 copy's keys + the int1 copy's
    # values (the same tensors, so a k2v1 base next to int1 and int2 references costs no extra copy)
    qk, qv = {}, {}
    for b in sorted(_hist_names(rows, base) - {"bf16"}):
        kb, vb = _kv_bits(b)
        if kb not in qk:
            qk[kb] = qcopy(K16, kb, "key")
        if vb not in qv:
            qv[vb] = qcopy(V16, vb, "value")
        hist[b] = list(zip(qk[kb], qv[vb]))
    del qk, qv

    fresh, prefill_pairs = {}, {}
    for g in sorted({r.fresh for r in rows} | {base}):
        if g == "pf" or g.startswith("pf:"):
            hb = base if g == "pf" else g[3:]
            fresh[g], prefill_pairs[g] = prefill_with_fetch(model, fresh_ids[:, :-2], hist[hb], hist["bf16"], lo,
                                                            hi, prefill_k, prefill_chunk, P)
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
    chain = torch.tensor([r.shadow and r.policy in CHAIN for r in rows], device=dev)  # position +1
    src = torch.tensor(ctx.score_src, device=dev)  # a shadow's spec row (others: themselves)
    # where each shadow's candidates come from: anchored -> its spec row's prediction (given the real token);
    # chained -> the cand-0 shadow's own output (the chain continues from candidate 0)
    gsrc = torch.tensor([(ctx.shadow_of[ctx.pair[i]] if r.policy in CHAIN else ctx.pair[i]) if r.shadow else i
                         for i, r in enumerate(rows)], device=dev)
    cand = torch.tensor([r.cand for r in rows], device=dev)
    mmax = max([r.m for r in rows if r.shadow] + [1])

    def next_guess(lg):
        """Each shadow's candidate for the next step (cand 0 = the argmax) and its probability."""
        g = lg.argmax(dim=-1)[gsrc]
        if mmax > 1:
            tk = lg[gsrc].topk(mmax, dim=-1).indices
            g = torch.where(cand == 0, g, tk.gather(1, cand[:, None].clamp_max(mmax - 1))[:, 0])
        w = torch.softmax(lg[gsrc].float(), dim=-1).gather(1, g[:, None])[:, 0]
        return g, w
    logp = torch.zeros(R, T, device=dev)
    pred = torch.zeros(R, T, dtype=torch.long, device=dev)
    # KL(bf16 row || row) of the next-token distribution, summed over steps (the bf16 row: bf16 history and
    # tool-output prefill, no fetch)
    ref = next((i for i, r in enumerate(rows) if r.hist == "bf16" and r.fresh == "bf16" and r.policy == "none"
                and not r.shadow), None)
    kl_sum = torch.zeros(R, device=dev)
    guesses = torch.zeros(R, T, dtype=torch.long, device=dev)  # shadows: token decoded as the guess
    actual = torch.zeros(R, T, dtype=torch.long, device=dev)   # shadows: the token that guess stood for
    pos0 = P + F_all - 2
    with fetch_mode(model, ctx):
        # warm-up: second-to-last tool-output token, no fetching; spec_chain shadows decode the last one (known)
        ctx.t, ctx.fetch = 0, False
        x = torch.where(chain, fresh_ids[0, -1], fresh_ids[0, -2])[:, None]
        pos = torch.full((R, 1), pos0, device=dev) + chain[:, None].long()
        out = model(input_ids=x, position_ids=pos, use_cache=False)
        # chained shadows: the cand-0 shadow's output (guesses of target[0], the input of step 1);
        # anchored shadows: their spec row's prediction of step 0's input (the last tool-output token)
        guess, w = next_guess(out.logits[:, -1])
        ctx.fetch = True
        for j in range(T):
            ctx.t = j + 1
            real = fresh_ids[0, -1] if j == 0 else tgt[0, j - 1]
            guesses[:, j] = guess
            actual[:, j] = torch.where(chain, tgt[0, j], real)  # chain guesses the next input, spec this one
            x = torch.where(shadow, guess, real)[:, None]
            ctx.cand_w = w
            pos = torch.full((R, 1), pos0 + 1 + j, device=dev) + chain[:, None].long()
            out = model(input_ids=x, position_ids=pos, use_cache=False)
            lg = out.logits[:, -1].float()
            lsm = torch.log_softmax(lg, dim=-1)
            logp[:, j] = lsm[:, tgt[0, j]]
            pred[:, j] = lg.argmax(dim=-1)
            if ref is not None:
                kl_sum += (lsm[ref].exp()[None] * (lsm[ref][None] - lsm)).sum(dim=-1)
            del lsm
            # chained: the cand-0 shadow's output (conditioned on its own guess); anchored: the spec row's
            # prediction given the real token it just decoded
            guess, w = next_guess(lg)
    hits = (guesses == actual) & shadow[:, None]
    guess_hits = hits.float().sum(dim=1)
    cover = {s_: hits[sh].any(dim=0).float().sum() for s_, sh in ctx.shadows.items()}   # any candidate right

    res = {"P": P, "F": F_all, "T": T, "lo": lo, "hi": hi, "base_bits": base_bits, "base": base, "k_match": k_match,
           **nbytes, "base_bytes": nbytes[f"{base}_bytes"],
           "logp": logp.cpu(), "pred": pred.cpu(), "rows": []}
    for i, r in enumerate(rows):
        obs = max(float(ctx.st_obs[i]), 1.0)
        union = sum(int(u[i].sum()) for u in ctx.union if u is not None)
        k_res = (hi - lo) if r.policy == "all" else min(r.k, hi - lo)
        sh = ctx.shadow_of.get(i)
        pool_rows = r.policy in POOL
        res["rows"].append({
            "k": r.k, "k_label": k_label[i], "m": r.m, "pool_M": r.pool, "pool_label": pool_label[i],
            "fetched_pairs_per_step": float(ctx.st_pairs[i]) / T,
            "distinct_tokens_per_layer_step": float(ctx.st_distinct[i]) / obs,
            "recall_vs_oracle": (float(ctx.st_recall[i]) / obs) if r.policy not in ("none", "all") else float("nan"),
            "moved_bytes_turn": float(ctx.st_moved[i]) * pair_bytes,
            "unique_bytes_turn": union * pair_bytes,
            "prefill_moved_bytes": prefill_pairs.get(r.fresh, 0) * pair_bytes,
            # resident: base + one step's fetched set (pools: M; unions: the mean fetched count per head and layer)
            "resident_bytes": (nbytes[f"{r.hist}_bytes"]
                               + (0 if r.policy == "none" else
                                  min(r.pool, hi - lo) * kvh * n_layers * pair_bytes if pool_rows else
                                  float(ctx.st_pairs[i]) / T * pair_bytes if r.policy in UNION else
                                  k_res * kvh * n_layers * pair_bytes)),
            "hist_bytes": nbytes[f"{r.hist}_bytes"],
            "next_bytes": nbytes[f"{NEXT[r.hist]}_bytes"] if r.hist in NEXT else float("nan"),
            "pool_hit_rate": (1 - float(ctx.st_moved[i]) / max(float(ctx.st_sel[i]), 1.0)) if pool_rows else float("nan"),
            "spec_guess_acc": float(guess_hits[sh]) / T if sh is not None else float("nan"),
            "spec_cand_cover": float(cover[i]) / T if i in cover else float("nan"),
            "kl_bf16": float(kl_sum[i]) / T if ref is not None else float("nan"),
        })
    res["guesses"] = [{"scheme": rows[s_].name, "k_label": k_label[s_], "cand": rows[sh_].cand, "step": j,
                       "guess_id": int(g_), "actual_id": int(a_), "match": int(g_ == a_)}
                      for sh_, s_ in ctx.pair.items()
                      for j, (g_, a_) in enumerate(zip(guesses[sh_].tolist(), actual[sh_].tolist()))]
    res["last_idx"] = {i: (v.cpu() if torch.is_tensor(v) else [x.cpu() for x in v]) for i, v in ctx.last_idx.items()}
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
