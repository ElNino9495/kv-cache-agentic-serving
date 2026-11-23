"""Fetch study run 5: multi-guess speculation, fetch pools, s2_k2score, the k2v1 base and the run-5 row sets."""

import dataclasses

import pytest
import torch

transformers = pytest.importorskip("transformers")

from experiments.gate.fetch_attn import (HALF, MATCH, POOL, Row, hist_bytes, run_pause,  # noqa: E402
                                         spec_family, study_rows)

P, F, T, B = 200, 10, 12, 16
LO, HI = 16, 128
KVH, L = 2, 2


@pytest.fixture(scope="module")
def model():
    cfg = transformers.LlamaConfig(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=L,
                                   num_attention_heads=4, num_key_value_heads=KVH, head_dim=16,
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


def _idx(rows, name, k=None):
    return next(i for i, r in enumerate(rows) if r.name == name and not r.shadow and (k is None or r.k == k))


def test_one_candidate_reproduces_spec_and_spec_chain(model, ids):
    rows = (spec_family("spec", "spec", "int2", "int2", 8) + spec_family("u1", "spec_union", "int2", "int2", 8, 1)
            + spec_family("w1", "spec_wtd", "int2", "int2", 8, 1)
            + spec_family("spec_chain", "spec_chain", "int2", "int2", 8)
            + spec_family("cu1", "spec_chain_union", "int2", "int2", 8, 1)
            + spec_family("cw1", "spec_chain_wtd", "int2", "int2", 8, 1))
    res = _run(model, ids, rows)
    lp = res["logp"]
    for a, b in (("u1", "spec"), ("w1", "spec"), ("cu1", "spec_chain"), ("cw1", "spec_chain")):
        assert torch.allclose(lp[_idx(rows, a)], lp[_idx(rows, b)], atol=1e-5), a
        assert res["rows"][_idx(rows, a)]["spec_guess_acc"] == res["rows"][_idx(rows, b)]["spec_guess_acc"]
        assert res["rows"][_idx(rows, a)]["fetched_pairs_per_step"] == 8 * KVH * L


def test_multi_guess_union_size_cover_and_log(model, ids):
    rows = [Row("int2")] + spec_family("u3", "spec_union", "int2", "int2", 8, 3) \
        + spec_family("cw2", "spec_chain_wtd", "int2", "int2", 8, 2)
    res = _run(model, ids, rows)
    u, cw = res["rows"][1], res["rows"][_idx(rows, "cw2")]
    assert 8 * KVH * L <= u["fetched_pairs_per_step"] <= 3 * 8 * KVH * L
    assert cw["fetched_pairs_per_step"] == 8 * KVH * L        # weighted: same budget as s2
    for r in (u, cw):
        assert r["spec_guess_acc"] <= r["spec_cand_cover"] <= 1
    g = [x for x in res["guesses"] if x["scheme"] == "u3"]
    assert sorted({x["cand"] for x in g}) == [0, 1, 2] and len(g) == 3 * T
    for j in range(T):   # the m candidates of a step are distinct tokens
        assert len({x["guess_id"] for x in g if x["step"] == j}) == 3


def test_full_k_multi_guess_and_pool_equal_fetch_all(model, ids):
    full = HI - LO
    rows = [Row("fetch_all", policy="all")] + spec_family("u2", "spec_union", "int2", "int2", full, 2) \
        + spec_family("cw3", "spec_chain_wtd", "int2", "int2", full, 3) \
        + [Row("pool", policy="s2_pool", k=full, pool=full)]
    res = _run(model, ids, rows)
    for n in ("u2", "cw3", "pool"):
        assert torch.allclose(res["logp"][_idx(rows, n)], res["logp"][0], atol=1e-4), n


def test_pool_with_k_equal_M_is_s2(model, ids):
    rows = [Row("s2", policy="s2", k=8), Row("pool", policy="s2_pool", k=8, pool=8),
            Row("s2", policy="s2", k=MATCH), Row("poolm", policy="s2_pool", k=MATCH, pool=MATCH)] \
        + spec_family("spec_chain", "spec_chain", "int2", "int2", 8) \
        + [dataclasses.replace(r, pool=8) for r in spec_family("cpool", "spec_chain_pool", "int2", "int2", 8)]
    res = _run(model, ids, rows)
    lp = res["logp"]
    assert torch.allclose(lp[1], lp[0], atol=1e-5)
    assert torch.allclose(lp[3], lp[2], atol=1e-5)
    assert torch.allclose(lp[_idx(rows, "cpool")], lp[_idx(rows, "spec_chain")], atol=1e-5)
    s2, pool = res["rows"][0], res["rows"][1]
    assert pool["fetched_pairs_per_step"] == s2["fetched_pairs_per_step"]
    assert pool["moved_bytes_turn"] <= s2["moved_bytes_turn"]   # hits are not moved again
    assert 0 <= pool["pool_hit_rate"] <= 1


def test_pool_keeps_M_tokens_and_moves_only_misses(model, ids):
    rows = [Row("s2", policy="s2", k=4), Row("pool", policy="s2_pool", k=4, pool=24)]
    res = _run(model, ids, rows)
    s2, pool = res["rows"]
    pair = 2 * 16 * 2
    assert pool["fetched_pairs_per_step"] <= 24 * KVH * L
    assert pool["fetched_pairs_per_step"] > s2["fetched_pairs_per_step"]   # the pool is larger than one top-k set
    sel = 4 * KVH * L * T
    assert pool["moved_bytes_turn"] == pytest.approx((1 - pool["pool_hit_rate"]) * sel * pair)
    assert pool["moved_bytes_turn"] <= s2["moved_bytes_turn"]
    assert pool["resident_bytes"] == res["int2_bytes"] + 24 * KVH * L * pair
    for h in res["last_idx"][1]:   # every head's pool holds at most M distinct middle tokens
        assert len(h) <= 24 and len(set(h.tolist())) == len(h) and int(h.min()) >= LO and int(h.max()) < HI


def test_k2score_equals_s2_on_an_int2_base(model, ids):
    rows = [Row("int2"), Row("s2", policy="s2", k=8), Row("k2", policy="s2_k2score", k=8)]
    res = _run(model, ids, rows)
    assert torch.allclose(res["logp"][2], res["logp"][1], atol=1e-6)


def test_k2score_on_int1_base_differs_from_s2(model, ids):
    rows = [Row("int1", "int1", "int1"), Row("int2"), Row("s2", "int1", "int1", policy="s2", k=8),
            Row("k2", "int1", "int1", policy="s2_k2score", k=8)]
    res = _run(model, ids, rows, base_bits=1)
    i1 = [sorted(x.tolist()) for x in res["last_idx"][2]]
    k2 = [sorted(x.tolist()) for x in res["last_idx"][3]]
    assert i1 != k2 or not torch.allclose(res["logp"][2], res["logp"][3])


def test_k2v1_base_bytes_match_and_bf16_row(model, ids):
    rows = [Row("bf16", "bf16", "bf16"), Row("k2v1", "k2v1", "k2v1"), Row("int2"), Row("int1", "int1", "int1"),
            Row("s2", "k2v1", "k2v1", policy="s2", k=MATCH), Row("s2_int1base", "int1", "int1", policy="s2", k=MATCH)]
    res = _run(model, ids, rows, base="k2v1")
    assert res["int1_bytes"] < res["k2v1_bytes"] < res["int2_bytes"]
    assert res["k2v1_bytes"] == hist_bytes("k2v1", P, LO, HI, B, KVH, 16, L)
    s2, s2i = res["rows"][4], res["rows"][5]
    # both fill up to uniform int2's memory from their own base
    for r in (s2, s2i):
        assert r["resident_bytes"] <= res["int2_bytes"] and r["next_bytes"] == res["int2_bytes"]
    assert s2["k"] < s2i["k"]
    lp = res["logp"]
    assert not torch.allclose(lp[1], lp[2]) and not torch.allclose(lp[1], lp[3])
    # the bf16 row does not depend on the base of the other rows
    ref = _run(model, ids, [Row("bf16", "bf16", "bf16"), Row("int1", "int1", "int1")], base_bits=1)
    assert torch.allclose(ref["logp"][0], lp[0], atol=1e-5)


def test_k2v1_copy_is_int2_keys_and_int1_values():
    from experiments.gate.fakequant import roundtrip
    g = torch.Generator().manual_seed(4)
    x = torch.randn(1, 2, 64, 16, generator=g)
    k2 = roundtrip(x, 2, 16, "key")
    v1 = roundtrip(x, 1, 16, "value")
    assert not torch.allclose(k2, roundtrip(x, 1, 16, "key"))
    assert not torch.allclose(v1, roundtrip(x, 2, 16, "value"))


@pytest.mark.parametrize("row_set,base", [("a1", "int1"), ("a2", "int1"), ("c", "k2v1")])
def test_study_row_sets_run(model, ids, row_set, base):
    rows = study_rows(row_set, base=base, prefill=True, uniform_refs=(2, 1) if base == "k2v1" else (2,),
                      s2_extra_ks=(16,) if row_set == "a1" else ())
    # the tiny model's k_match is ~7 tokens: shrink the pool rows' k below their M (real pauses: k_match >= 495)
    rows = [dataclasses.replace(r, k=1) if r.policy in POOL else r for r in rows]
    res = _run(model, ids, rows, base_bits=1, base=base, prefill_k=8, prefill_chunk=4)
    assert res["logp"].shape == (len(rows), T)
    assert torch.isfinite(res["logp"]).all()
    names = [r.name for r in rows]
    nc = names.index("noise_ctrl")
    # equal in exact arithmetic; fp32 rounding of the large int1 corrections is ~2e-4 on this tiny model (as before)
    assert torch.allclose(res["logp"][nc], res["logp"][0], atol=5e-4)
    if row_set == "c":
        i = names.index("s2_int1base")
        assert res["rows"][i]["prefill_moved_bytes"] > 0 and rows[i].fresh == "pf:int1"
    if row_set == "a2":
        hm = [i for i, r in enumerate(rows) if r.name == "s2_pool_Mhalf"]
        mm = [i for i, r in enumerate(rows) if r.name == "s2_pool_Mmatch"]
        assert all(res["rows"][a]["pool_M"] == res["rows"][b]["pool_M"] // 2 for a, b in zip(hm, mm))
        assert all(rows[i].pool == HALF for i in hm)


def test_uniform_k2v1_row_equals_independent_k2v1_cache(model, ids):
    # the k2v1 row (shared int2-key / int1-value tensors) must equal teacher forcing over a k2v1 cache built here
    from experiments.gate.fakequant import roundtrip
    from experiments.gate.run_gate import prefill, run_turn
    hist_ids, fresh_ids, tgt = ids
    rows = [Row("bf16", "bf16", "bf16"), Row("int1", "int1", "int1"), Row("int2"), Row("k2v1", "k2v1", "k2v1")]
    res = _run(model, ids, rows, base="k2v1")
    cache, _ = prefill(model, hist_ids)
    kv = []
    for layer in cache.layers:
        k, v = layer.keys, layer.values
        k2, v1 = roundtrip(k, 2, B, "key"), roundtrip(v, 1, B, "value")
        for q, x in ((k2, k), (v1, v)):
            q[:, :, :LO], q[:, :, HI:] = x[:, :, :LO], x[:, :, HI:]
        kv.append((k2, v1))
    lp, _ = run_turn(model, lambda li: kv[li], len(kv), fresh_ids, tgt[:, :T])
    assert torch.allclose(lp.cpu(), res["logp"][3], atol=1e-4)


def test_pool_k_above_M_is_reported_not_clamped(model, ids):
    from experiments.gate.fetch_attn import resolve_rows
    rows = [Row("ok", policy="s2_pool", k=4, pool=8), Row("bad", policy="s2_pool", k=16, pool=8)]
    res, bad = resolve_rows(rows, P, B, 64, KVH, 16, L)
    assert [r.k for r in res] == [4, 16] and len(bad) == 1 and bad[0].startswith("bad@16")
    with pytest.raises(AssertionError, match="k > M"):
        _run(model, ids, rows)
