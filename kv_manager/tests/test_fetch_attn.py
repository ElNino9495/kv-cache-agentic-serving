import pytest
import torch

transformers = pytest.importorskip("transformers")

from experiments.gate.fetch_attn import Row, default_rows, first_order_norm, run_pause  # noqa: E402

P, F, T, B = 200, 10, 12, 16  # middle tokens: [16, 128) with block 16 and a 64-token recent window
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


def test_fetch_all_with_bf16_tool_prefill_equals_bf16(model, ids):
    rows = [Row("bf16", "bf16", "bf16"), Row("fetch_all", hist="int2", fresh="bf16", policy="all")]
    res = _run(model, ids, rows)
    assert torch.allclose(res["logp"][1], res["logp"][0], atol=1e-4)
    assert torch.equal(res["pred"][1], res["pred"][0])


def test_fetch_all_with_int2_tool_prefill_is_not_bf16(model, ids):
    # the ceiling the study reports: per-step fetching cannot undo an int2 tool-output prefill
    rows = [Row("bf16", "bf16", "bf16"), Row("fetch_all", policy="all")]
    res = _run(model, ids, rows)
    assert (res["logp"][1] - res["logp"][0]).abs().max() > 1e-4


def test_k0_equals_uniform_int2_and_oracle_full_equals_fetch_all(model, ids):
    rows = [Row("int2"), Row("s1", policy="s1", k=0), Row("s2", policy="s2", k=0),
            Row("fetch_all", policy="all"), Row("oracle", policy="oracle", k=HI - LO)]
    res = _run(model, ids, rows)
    assert torch.allclose(res["logp"][1], res["logp"][0], atol=1e-5)
    assert torch.allclose(res["logp"][2], res["logp"][0], atol=1e-5)
    assert torch.allclose(res["logp"][4], res["logp"][3], atol=1e-4)


def test_per_step_matches_sdpa_teacher_forcing(model, ids):
    res = _run(model, ids, [Row("bf16", "bf16", "bf16"), Row("int2")], check=True)
    assert res["check_bf16_max_abs_logp"] < 1e-4
    assert res["check_int2_max_abs_logp"] < 1e-4


def test_selection_respects_pins_k_and_granularity(model, ids):
    rows = default_rows(ks=(8,))
    res = _run(model, ids, rows)
    names = [r.name for r in rows]
    for i, r in enumerate(rows):
        if r.policy in ("none", "all"):
            continue
        idx = res["last_idx"][i]  # (kv heads, k) of the last step, last layer
        assert idx.shape == (2, 8)
        assert int(idx.min()) >= LO and int(idx.max()) < HI
        if r.per_layer:
            assert torch.equal(idx[0], idx[1])
        st = res["rows"][i]
        assert st["fetched_pairs_per_step"] == 8 * 2 * 2  # k x KV heads x layers
        assert 0 <= st["recall_vs_oracle"] <= 1
    assert res["rows"][names.index("oracle")]["recall_vs_oracle"] == pytest.approx(1.0)
    assert {r.name for r in rows} >= {"noise_ctrl", "fetch_all", "bf16", "int2", "int3"}
    nc = names.index("noise_ctrl")
    assert torch.allclose(res["logp"][nc], res["logp"][names.index("bf16")], atol=1e-4)
    i = names.index("fetch_all")
    assert res["rows"][i]["fetched_pairs_per_step"] == (HI - LO) * 2 * 2
    assert res["int2_bytes"] < res["int3_bytes"] < res["bf16_bytes"]


def test_first_order_norm_matches_direct():
    g = torch.Generator().manual_seed(2)
    kvh, G, Pm, d, n = 2, 3, 7, 5, 2
    dV, dK, V2 = (torch.randn(kvh, Pm, d, generator=g, dtype=torch.float64) for _ in range(3))
    q, o = (torch.randn(n, kvh, G, d, generator=g, dtype=torch.float64) for _ in range(2))
    s = 0.3
    dl = torch.einsum("rkgd,kpd->rkgp", q, dK) * s
    direct = (dV[None, :, None] + dl[..., None] * (V2[None, :, None] - o[:, :, :, None])).norm(dim=-1)
    assert torch.allclose(first_order_norm(dV, dK, V2, q, o, s), direct, atol=1e-10)
