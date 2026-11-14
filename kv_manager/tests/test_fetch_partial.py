"""Partial fetching against a brute-force reference.

For a fetched set S (per KV head, or one set per layer), the fetch path (int2 logits with S replaced
by bf16 logits, plus a value correction) must equal plain attention over a cache rebuilt with the
tokens in S at bf16 and everything else at int2. The previous-step (s1) path is checked twice: with
injected scores (so S is known exactly) and over two consecutive steps (S must be the top-k of the
previous step's int2 attention).
"""

import pytest
import torch

import experiments.gate.fetch_attn as fa
from experiments.gate.fakequant import roundtrip
from experiments.gate.fetch_attn import FetchContext, Row, fetch_attention

KVH, G, D, P, F = 2, 3, 8, 112, 5
LO, HI = 16, 80  # quantized middle tokens
SCALE = D ** -0.5


class _Mod:
    layer_idx = 0


def _cache(seed):
    g = torch.Generator().manual_seed(seed)
    K16, V16 = torch.randn(KVH, P, D, generator=g), torch.randn(KVH, P, D, generator=g)
    K16[:, :, 3] *= 6  # an outlier key channel, so int2 errors are large

    def q2(x, kind):
        y = roundtrip(x[None], 2, 16, kind)[0]
        y[:, :LO], y[:, HI:] = x[:, :LO], x[:, HI:]
        return y

    hist = {"bf16": [(K16, V16)], "int2": [(q2(K16, "key"), q2(V16, "value"))]}
    fresh = {"int2": [(torch.randn(KVH, F, D, generator=g), torch.randn(KVH, F, D, generator=g))]}
    return hist, fresh, g


def _ctx(rows, hist, fresh, t, g):
    ctx = FetchContext(rows, hist, fresh, LO, HI, n_layers=1, max_tokens=t + 2)
    R = len(rows)
    ctx.gen[0] = (torch.randn(R, KVH, t + 2, D, generator=g), torch.randn(R, KVH, t + 2, D, generator=g))
    return ctx


def _call(ctx, t, fetch, g):
    R = len(ctx.rows)
    ctx.t, ctx.fetch = t, fetch
    q = torch.randn(R, KVH * G, 1, D, generator=g)
    k, v = torch.randn(R, KVH, 1, D, generator=g), torch.randn(R, KVH, 1, D, generator=g)
    fa._CTX = ctx
    try:
        out, _ = fetch_attention(_Mod(), q, k, v, None, SCALE)
    finally:
        fa._CTX = None
    _call.kv = (k, v)
    return q, out


def _reference(ctx, hist, fresh, i, q, S):
    """Plain attention for row i with the tokens in S[h] at bf16 (S: list per KV head)."""
    K16, V16 = hist["bf16"][0]
    K, V = (x.clone() for x in hist["int2"][0])
    for h in range(KVH):
        K[h, S[h]], V[h, S[h]] = K16[h, S[h]], V16[h, S[h]]
    gk, gv = ctx.gen[0]
    t = ctx.t
    Kf, Vf = fresh["int2"][0]
    K = torch.cat([K, Kf, gk[i, :, :t + 1]], dim=1)
    V = torch.cat([V, Vf, gv[i, :, :t + 1]], dim=1)
    out = torch.empty(KVH * G, D)
    for hq in range(KVH * G):
        h = hq // G
        p = torch.softmax(q[i, hq, 0] @ K[h].T * SCALE, dim=-1)
        out[hq] = p @ V[h]
    return out


def _inject(rows, ctx, g):
    """Scores with exactly k ones per head (per layer: the same positions on every head)."""
    want = {}
    for i, r in enumerate(rows):
        if r.policy != "s1":
            continue
        score = torch.rand(KVH, HI - LO, generator=g) * 1e-3
        if r.per_layer:
            pos = torch.randperm(HI - LO, generator=g)[:r.k]
            score[:, pos] += 1
            want[i] = [sorted((pos + LO).tolist())] * KVH
        else:
            want[i] = []
            for h in range(KVH):
                pos = torch.randperm(HI - LO, generator=g)[:r.k]
                score[h, pos] += 1
                want[i].append(sorted((pos + LO).tolist()))
        ctx.prev[0][i] = score
    return want


KS = (1, 5, 17, HI - LO)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_s1_partial_fetch_matches_bruteforce(seed):
    rows = [Row("int2")] + [Row("s1", policy="s1", k=k) for k in KS] + \
           [Row("s1_layer", policy="s1", k=k, per_layer=True) for k in KS]
    hist, fresh, g = _cache(seed)
    ctx = _ctx(rows, hist, fresh, t=3, g=g)
    want = _inject(rows, ctx, g)
    q, out = _call(ctx, t=3, fetch=True, g=g)
    for i, r in enumerate(rows):
        if r.policy == "none":
            ref = _reference(ctx, hist, fresh, i, q, [[] for _ in range(KVH)])
        else:
            got = [sorted(x.tolist()) for x in ctx.last_idx[i]]
            assert got == want[i], f"{r.name}@{r.k}: fetched set differs from the injected one"
            ref = _reference(ctx, hist, fresh, i, q, want[i])
        assert torch.allclose(out[i, 0], ref, atol=1e-5), f"{r.name}@{r.k} differs from brute force"


@pytest.mark.parametrize("seed", [3, 4])
def test_s1_uses_previous_steps_int2_attention(seed):
    rows = [Row("s1", policy="s1", k=k) for k in KS] + [Row("s1_layer", policy="s1", k=k, per_layer=True) for k in KS]
    hist, fresh, g = _cache(seed)
    ctx = _ctx(rows, hist, fresh, t=0, g=g)
    q1, _ = _call(ctx, t=0, fetch=False, g=g)  # step j-1: no fetch, stores scores
    gk = ctx.gen[0][0]
    K2 = hist["int2"][0][0]
    Kf = fresh["int2"][0][0]
    want = {}
    for i, r in enumerate(rows):
        K = torch.cat([K2, Kf, gk[i, :, :1]], dim=1)
        qi = q1[i, :, 0].reshape(KVH, G, D)
        p = torch.softmax(torch.einsum("kgd,kpd->kgp", qi, K) * SCALE, dim=-1)[..., LO:HI].sum(dim=1)
        if r.per_layer:
            pos = p.sum(dim=0).topk(r.k).indices
            want[i] = [sorted((pos + LO).tolist())] * KVH
        else:
            want[i] = [sorted((p[h].topk(r.k).indices + LO).tolist()) for h in range(KVH)]
    q2, out = _call(ctx, t=1, fetch=True, g=g)  # step j: fetches by step j-1's scores
    for i, r in enumerate(rows):
        got = [sorted(x.tolist()) for x in ctx.last_idx[i]]
        assert got == want[i], f"{r.name}@{r.k}: not the top-k of the previous step's int2 attention"
        assert torch.allclose(out[i, 0], _reference(ctx, hist, fresh, i, q2, want[i]), atol=1e-5)


@pytest.mark.parametrize("seed", [5, 6])
def test_current_step_policies_match_bruteforce(seed):
    rows = [Row(p, policy=p, k=k, per_layer=pl) for p in ("s2", "oracle") for k in KS for pl in (False, True)]
    hist, fresh, g = _cache(seed)
    ctx = _ctx(rows, hist, fresh, t=2, g=g)
    q, out = _call(ctx, t=2, fetch=True, g=g)
    gk = ctx.gen[0][0]
    for i, r in enumerate(rows):
        base = hist["int2" if r.policy == "s2" else "bf16"][0][0]
        K = torch.cat([base, fresh["int2"][0][0], gk[i, :, :3]], dim=1)
        p = torch.softmax(torch.einsum("kgd,kpd->kgp", q[i, :, 0].reshape(KVH, G, D), K) * SCALE,
                          dim=-1)[..., LO:HI].sum(dim=1)
        if r.per_layer:
            want = [sorted((p.sum(dim=0).topk(r.k).indices + LO).tolist())] * KVH
        else:
            want = [sorted((p[h].topk(r.k).indices + LO).tolist()) for h in range(KVH)]
        assert [sorted(x.tolist()) for x in ctx.last_idx[i]] == want, f"{r.name}@{r.k} per_layer={r.per_layer}"
        assert torch.allclose(out[i, 0], _reference(ctx, hist, fresh, i, q, want), atol=1e-5)


def _int2_attention(ctx, hist, fresh, i, q, src=None, own=None):
    """Probabilities over the middle tokens (KVH, G, mid) and output (KVH, G, D) of plain attention
    over the int2 history, tool output and generated tokens (of row `src`), plus an own token."""
    gk, gv = ctx.gen[0]
    t, src = ctx.t, i if src is None else src
    K = torch.cat([hist["int2"][0][0], fresh["int2"][0][0], gk[src, :, :t + 1]], dim=1)
    V = torch.cat([hist["int2"][0][1], fresh["int2"][0][1], gv[src, :, :t + 1]], dim=1)
    if own is not None:
        K, V = torch.cat([K, own[0][:, None]], dim=1), torch.cat([V, own[1][:, None]], dim=1)
    qi = q[i, :, 0].reshape(KVH, G, D)
    p = torch.softmax(torch.einsum("kgd,kpd->kgp", qi, K) * SCALE, dim=-1)
    return p[..., LO:HI], torch.einsum("kgp,kpd->kgd", p, V)


@pytest.mark.parametrize("seed", [7, 8])
def test_speculative_shadow_matches_bruteforce(seed):
    rows = [Row("spec", policy="spec", k=5), Row("spec_shadow", policy="spec", k=5, shadow=True)]
    hist, fresh, g = _cache(seed)
    ctx = _ctx(rows, hist, fresh, t=2, g=g)
    want = []
    score = torch.rand(KVH, HI - LO, generator=g) * 1e-3
    for h in range(KVH):
        pos = torch.randperm(HI - LO, generator=g)[:5]
        score[h, pos] += 1
        want.append(sorted((pos + LO).tolist()))
    ctx.prev[0][0] = score  # what the shadow stored at the previous step
    q, out = _call(ctx, t=2, fetch=True, g=g)
    k, v = _call.kv
    for i in (0, 1):
        assert [sorted(x.tolist()) for x in ctx.last_idx[i]] == want  # the shadow shares the spec row's set
    assert torch.allclose(out[0, 0], _reference(ctx, hist, fresh, 0, q, want), atol=1e-5)
    # shadow: same history view, the spec row's generated tokens, then its own token
    K16, V16 = hist["bf16"][0]
    K, V = (x.clone() for x in hist["int2"][0])
    for h in range(KVH):
        K[h, want[h]], V[h, want[h]] = K16[h, want[h]], V16[h, want[h]]
    gk, gv = ctx.gen[0]
    K = torch.cat([K, fresh["int2"][0][0], gk[0, :, :3], k[1, :, 0][:, None]], dim=1)
    V = torch.cat([V, fresh["int2"][0][1], gv[0, :, :3], v[1, :, 0][:, None]], dim=1)
    ref = torch.stack([torch.softmax(q[1, hq, 0] @ K[hq // G].T * SCALE, -1) @ V[hq // G] for hq in range(KVH * G)])
    assert torch.allclose(out[1, 0], ref, atol=1e-5)
    # the shadow's int2 attention is stored as the spec row's score for the next step
    p_mid, _ = _int2_attention(ctx, hist, fresh, 1, q, src=0, own=(k[1, :, 0], v[1, :, 0]))
    assert torch.allclose(ctx.prev[0][0], p_mid.sum(dim=1), atol=1e-6)


@pytest.mark.parametrize("seed", [9, 10])
def test_current_query_sensitivity_matches_bruteforce(seed):
    rows = [Row(p, policy=p, k=k) for p in ("s2sens_a", "s2sens_b") for k in KS]
    hist, fresh, g = _cache(seed)
    ctx = _ctx(rows, hist, fresh, t=2, g=g)
    q, out = _call(ctx, t=2, fetch=True, g=g)
    K16, V16 = (x[:, LO:HI] for x in hist["bf16"][0])
    K2, V2 = (x[:, LO:HI] for x in hist["int2"][0])
    dK, dV = K16 - K2, V16 - V2
    for i, r in enumerate(rows):
        p_mid, o2 = _int2_attention(ctx, hist, fresh, i, q)
        qi = q[i, :, 0].reshape(KVH, G, D)
        if r.policy == "s2sens_a":
            err = dV.norm(dim=-1)[:, None] + qi.norm(dim=-1)[..., None] * dK.norm(dim=-1)[:, None] * SCALE
        else:
            dl = torch.einsum("kgd,kpd->kgp", qi, dK) * SCALE
            err = (dV[:, None] + dl[..., None] * (V2[:, None] - o2[:, :, None])).norm(dim=-1)
        score = (p_mid * err).sum(dim=1)
        want = [sorted((score[h].topk(r.k).indices + LO).tolist()) for h in range(KVH)]
        assert [sorted(x.tolist()) for x in ctx.last_idx[i]] == want, f"{r.name}@{r.k}"
        assert torch.allclose(out[i, 0], _reference(ctx, hist, fresh, i, q, want), atol=1e-5)
