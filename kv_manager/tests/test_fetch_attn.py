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
    assert res["check_base_max_abs_logp"] < 1e-4


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


def test_full_k_spec_and_sensitivity_equal_fetch_all(model, ids):
    # with k = every middle token the selection no longer matters: the spec row (with its shadow decoded
    # alongside) and the current-query sensitivity rows must equal fetch_all
    full = HI - LO
    rows = [Row("fetch_all", policy="all"), Row("spec", policy="spec", k=full),
            Row("spec_shadow", policy="spec", k=full, shadow=True),
            Row("s2sens_a", policy="s2sens_a", k=full), Row("s2sens_b", policy="s2sens_b", k=full),
            Row("spec_chain", policy="spec_chain", k=full),
            Row("spec_chain_shadow", policy="spec_chain", k=full, shadow=True)]
    res = _run(model, ids, rows)
    for i in (1, 3, 4, 5):
        assert torch.allclose(res["logp"][i], res["logp"][0], atol=1e-4), rows[i].name
    assert 0 <= res["rows"][1]["spec_guess_acc"] <= 1


def test_speculative_guesses_logged_and_anchored(model, ids):
    hist_ids, fresh_ids, tgt = ids
    rows = [Row("int2"), Row("spec", policy="spec", k=8), Row("spec_shadow", policy="spec", k=8, shadow=True),
            Row("spec_chain", policy="spec_chain", k=8),
            Row("spec_chain_shadow", policy="spec_chain", k=8, shadow=True), Row("s2", policy="s2", k=8)]
    res = _run(model, ids, rows)
    g = res["guesses"]
    assert len(g) == 2 * T and {x["scheme"] for x in g} == {"spec", "spec_chain"}
    spec = sorted((x for x in g if x["scheme"] == "spec"), key=lambda x: x["step"])
    chain = sorted((x for x in g if x["scheme"] == "spec_chain"), key=lambda x: x["step"])
    real = [int(fresh_ids[0, -1])] + tgt[0, :-1].tolist()  # input of each step
    assert [x["actual_id"] for x in spec] == real[:T]         # spec guesses the step's own input
    assert [x["actual_id"] for x in chain] == tgt[0].tolist()  # spec_chain guesses the next step's input
    # spec's guess at step j is the spec row's greedy prediction at step j-1 (given the real token it decoded)
    assert [x["guess_id"] for x in spec[1:]] == res["pred"][1, :-1].tolist()
    for row_i, xs in ((1, spec), (3, chain)):
        assert all(x["match"] == int(x["guess_id"] == x["actual_id"]) for x in xs)
        assert res["rows"][row_i]["spec_guess_acc"] == pytest.approx(sum(x["match"] for x in xs) / T)
        assert all(x["k_label"] == "8" for x in xs)


def test_anchored_spec_equals_s2_when_every_guess_is_right(model, ids):
    # teacher-force the spec row's own greedy tokens (a fixed point, one more position per pass): then every
    # spec guess is right and the spec row must reproduce s2 exactly
    hist_ids, fresh_ids, tgt = ids
    rows = [Row("spec", policy="spec", k=8), Row("spec_shadow", policy="spec", k=8, shadow=True),
            Row("s2", policy="s2", k=8)]
    fresh, tgt = fresh_ids.clone(), tgt.clone()
    for _ in range(T + 2):
        res = _run(model, (hist_ids, fresh, tgt), rows)
        spec = sorted((x for x in res["guesses"] if x["scheme"] == "spec"), key=lambda x: x["step"])
        fresh[0, -1] = spec[0]["guess_id"]  # step 0's input = the warm-up prediction
        new = res["pred"][0:1].clone()
        if torch.equal(new, tgt) and all(x["match"] for x in spec):
            break
        tgt = new
    assert all(x["match"] for x in spec) and res["rows"][0]["spec_guess_acc"] == 1.0
    assert torch.allclose(res["logp"][0], res["logp"][2], atol=1e-5)
    assert torch.equal(res["pred"][0], res["pred"][2])
