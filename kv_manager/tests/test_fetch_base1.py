"""1-bit base (SpeCache's modified quantizer), memory-matched k, and fetching during the tool-output prefill."""

import pytest
import torch

import experiments.gate.fetch_attn as fa
from experiments.gate.fakequant import fake_quant, roundtrip
from experiments.gate.fetch_attn import (MATCH, FetchContext, PrefillContext, Row, default_rows,
                                         prefill_fetch_attention, run_pause)
from kvm.quant import block_bytes

transformers = pytest.importorskip("transformers")


# ---- 1-bit quantizer (SpeCache Eq. 3) ----------------------------------------------------------

def test_one_bit_levels_are_specache_eq3():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4, 16, 32, generator=g)
    for dim in (-2, -1):
        y = fake_quant(x, 1, dim)
        mn, mx = x.amin(dim=dim, keepdim=True), x.amax(dim=dim, keepdim=True)
        r = mx - mn
        lo_level, hi_level = mn + r / 4, mn + 3 * r / 4
        tol = 2e-3 * r.abs() + 1e-3  # fp16 scale / zero-point
        on_lo = (y - lo_level).abs() <= tol
        on_hi = (y - hi_level).abs() <= tol
        assert bool((on_lo | on_hi).all()), "1-bit values must be min + range/4 or min + 3 range/4"
        mid = mn + r / 2  # values above the midpoint go to the upper level
        assert bool((on_hi == (x > mid + tol)).logical_or((x - mid).abs() <= tol).all())


def test_one_bit_beats_plain_kivi_one_bit_and_bytes():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(8, 16, 128, generator=g)
    mn, mx = x.amin(-2, keepdim=True), x.amax(-2, keepdim=True)
    kivi = torch.where(x - mn > (mx - mn) / 2, mx, mn)  # plain KIVI 1-bit: levels at min and max
    assert (fake_quant(x, 1, -2) - x).abs().mean() < (kivi - x).abs().mean()
    y = roundtrip(x[None].to(torch.bfloat16), 1, 16, "key")
    assert y.shape == (1, 8, 16, 128)
    assert block_bytes(1, 16, 8, 128) == 2 * 2048 + 8 * 128 * 4 + 8 * 16 * 4
    assert block_bytes(1, 16, 8, 128) < block_bytes(2, 16, 8, 128)


# ---- tiny model ---------------------------------------------------------------------------------

P, F, T, B = 200, 12, 10, 16
LO, HI = 16, 128


@pytest.fixture(scope="module")
def model():
    cfg = transformers.LlamaConfig(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                                   max_position_embeddings=1024)
    torch.manual_seed(0)
    m = transformers.LlamaForCausalLM(cfg).float().eval()
    m.set_attn_implementation("sdpa")
    return m


@pytest.fixture(scope="module")
def ids():
    g = torch.Generator().manual_seed(1)
    r = lambda n: torch.randint(0, 64, (1, n), generator=g)
    return r(P), r(F), r(T)


def _run(model, ids, rows, **kw):
    return run_pause(model, *ids, rows, block=B, recent=64, max_steps=T, **kw)


def test_base1_rows_match_memory_and_check(model, ids):
    rows = default_rows(ks=(4, MATCH), base_bits=1)
    names = [r.name for r in rows]
    assert {"int1", "int2", "int3"} <= set(names)
    res = _run(model, ids, rows, base_bits=1, check=True)
    assert res["check_base_max_abs_logp"] < 1e-4 and res["check_bf16_max_abs_logp"] < 1e-4
    km = res["k_match"]
    per_token = 2 * 2 * 2 * 16 * 2  # KV heads x layers x (K + V) x head_dim x bf16 bytes
    assert 0 < km <= HI - LO
    for i, r in enumerate(rows):
        st = res["rows"][i]
        if st["k_label"] == "match":
            assert st["k"] == km
            gap = res["int2_bytes"] - st["resident_bytes"]
            assert 0 <= gap < per_token, "int1 + matched fetch must use (just under) int2's memory"
    # the int1 base hurts more than int2 on random weights is not guaranteed; the rows must differ though
    assert not torch.allclose(res["logp"][names.index("int1")], res["logp"][names.index("int2")])


def test_full_prefill_fetch_lifts_the_ceiling_to_bf16(model, ids):
    rows = default_rows(ks=(), base_bits=2, prefill=True)
    names = [r.name for r in rows]
    res = _run(model, ids, rows, prefill_k=HI - LO, prefill_chunk=5)
    bf, fa_, fn = (res["logp"][names.index(n)] for n in ("bf16", "fetch_all", "fetch_all_noprefill"))
    assert torch.allclose(fa_, bf, atol=1e-4), "full prefill fetch + full per-step fetch must equal bf16"
    assert (fn - bf).abs().max() > 1e-4
    assert res["rows"][names.index("fetch_all")]["prefill_moved_bytes"] > 0
    assert res["rows"][names.index("fetch_all_noprefill")]["prefill_moved_bytes"] == 0


def test_zero_prefill_fetch_equals_plain_base_prefill(model, ids):
    rows = default_rows(ks=(), base_bits=2, prefill=True)
    names = [r.name for r in rows]
    res = _run(model, ids, rows, prefill_k=0, prefill_chunk=4)
    assert torch.allclose(res["logp"][names.index("fetch_all")], res["logp"][names.index("fetch_all_noprefill")],
                          atol=1e-4)


# ---- prefill fetch against brute force (attention level) -----------------------------------------

KVH, G, D, PP = 2, 2, 8, 96
PLO, PHI = 16, 64


class _Mod:
    layer_idx = 0


@pytest.mark.parametrize("k", [3, 11])
def test_partial_prefill_fetch_matches_bruteforce(k):
    g = torch.Generator().manual_seed(k)
    K16, V16 = torch.randn(KVH, PP, D, generator=g), torch.randn(KVH, PP, D, generator=g)
    K16[:, :, 2] *= 5
    def q1(x, kind):
        y = roundtrip(x[None], 1, 16, kind)[0]
        y[:, :PLO], y[:, PHI:] = x[:, :PLO], x[:, PHI:]
        return y
    Kb, Vb = q1(K16, "key"), q1(V16, "value")
    ctx = PrefillContext([(Kb, Vb)], [(K16, V16)], PLO, PHI, k, n_layers=1)
    scale = D ** -0.5
    fa._PCTX = ctx
    try:
        chunks = []
        for c in (4, 5):  # two chunks, so the second also attends to the first
            q = torch.randn(1, KVH * G, c, D, generator=g)
            kk, vv = torch.randn(1, KVH, c, D, generator=g), torch.randn(1, KVH, c, D, generator=g)
            out, _ = prefill_fetch_attention(_Mod(), q, kk, vv, None, scale)
            chunks.append((q, kk, vv, out))
    finally:
        fa._PCTX = None
    q, kk, vv, out = chunks[1]
    Kprev, Vprev = chunks[0][1][0], chunks[0][2][0]
    c = q.shape[2]
    qi = q[0].reshape(KVH, G, c, D)
    Kall = torch.cat([Kb, Kprev, kk[0]], dim=1)
    causal = torch.cat([torch.ones(c, PP + 4, dtype=torch.bool), torch.ones(c, c).tril().bool()], dim=1)
    logits = (torch.einsum("kgcd,kpd->kgcp", qi, Kall) * scale).masked_fill(~causal, float("-inf"))
    score = logits.softmax(-1)[..., PLO:PHI].sum(dim=(1, 2))
    want = score.topk(k, dim=-1).indices + PLO
    assert torch.equal(ctx.last_idx[0].sort(-1).values, want.sort(-1).values)
    Kh, Vh = Kb.clone(), Vb.clone()
    for h in range(KVH):
        Kh[h, want[h]], Vh[h, want[h]] = K16[h, want[h]], V16[h, want[h]]
    Kr, Vr = torch.cat([Kh, Kprev, kk[0]], dim=1), torch.cat([Vh, Vprev, vv[0]], dim=1)
    pr = (torch.einsum("kgcd,kpd->kgcp", qi, Kr) * scale).masked_fill(~causal, float("-inf")).softmax(-1)
    ref = torch.einsum("kgcp,kpd->kgcd", pr, Vr).reshape(KVH * G, c, D).permute(1, 0, 2)
    assert torch.allclose(out[0], ref, atol=1e-5)


# ---- per-step fetch on an int1 base against brute force -------------------------------------------

def test_s1_partial_fetch_on_int1_base_matches_bruteforce():
    g = torch.Generator().manual_seed(11)
    K16, V16 = torch.randn(KVH, PP, D, generator=g), torch.randn(KVH, PP, D, generator=g)
    def q1(x, kind):
        y = roundtrip(x[None], 1, 16, kind)[0]
        y[:, :PLO], y[:, PHI:] = x[:, :PLO], x[:, PHI:]
        return y
    hist = {"bf16": [(K16, V16)], "int1": [(q1(K16, "key"), q1(V16, "value"))]}
    fresh = {"int1": [(torch.randn(KVH, 3, D, generator=g), torch.randn(KVH, 3, D, generator=g))]}
    rows = [Row("s1", "int1", "int1", policy="s1", k=7)]
    ctx = FetchContext(rows, hist, fresh, PLO, PHI, n_layers=1, max_tokens=4, base="int1")
    ctx.gen[0] = (torch.randn(1, KVH, 4, D, generator=g), torch.randn(1, KVH, 4, D, generator=g))
    score = torch.rand(KVH, PHI - PLO, generator=g)
    ctx.prev[0][0] = score
    want = score.topk(7, dim=-1).indices + PLO
    ctx.t, ctx.fetch = 2, True
    q = torch.randn(1, KVH * G, 1, D, generator=g)
    k, v = torch.randn(1, KVH, 1, D, generator=g), torch.randn(1, KVH, 1, D, generator=g)
    fa._CTX = ctx
    try:
        out, _ = fa.fetch_attention(_Mod(), q, k, v, None, D ** -0.5)
    finally:
        fa._CTX = None
    Kh, Vh = (x.clone() for x in hist["int1"][0])
    for h in range(KVH):
        Kh[h, want[h]], Vh[h, want[h]] = K16[h, want[h]], V16[h, want[h]]
    gk, gv = ctx.gen[0]
    Kr = torch.cat([Kh, fresh["int1"][0][0], gk[0, :, :3]], dim=1)
    Vr = torch.cat([Vh, fresh["int1"][0][1], gv[0, :, :3]], dim=1)
    ref = torch.stack([torch.softmax(q[0, hq, 0] @ Kr[hq // G].T * D ** -0.5, -1) @ Vr[hq // G]
                       for hq in range(KVH * G)])
    assert torch.allclose(out[0, 0], ref, atol=1e-5)
