"""Base-1 run without the int3 history copy: rows, memory estimate, and joining int3 from the base-2 run."""

import numpy as np
import pandas as pd
import pytest

from experiments.gate.fetch_attn import MATCH, default_rows, estimate_peak_bytes
from experiments.gate.summarize_fetch import summarize

transformers = pytest.importorskip("transformers")


def test_uniform_refs_drop_the_int3_copy():
    full = {r.name for r in default_rows(base_bits=1)}
    lean = default_rows(base_bits=1, uniform_refs=(2,))
    assert {"int1", "int2", "int3"} <= full
    assert "int3" not in {r.name for r in lean} and "int3" not in {r.hist for r in lean}
    assert {"int1", "int2"} <= {r.name for r in lean}
    assert {r.name for r in default_rows(base_bits=2)} >= {"int2", "int3"}


def test_memory_estimate_counts_history_copies():
    cfg = transformers.LlamaConfig(hidden_size=4096, intermediate_size=14336, num_hidden_layers=32,
                                   num_attention_heads=32, num_key_value_heads=8, head_dim=128)
    w, P, F = 16.06e9, 33844, 2901
    tok = 2 * 8 * 128 * 2 * 32
    full = estimate_peak_bytes(cfg, w, P, F, default_rows(base_bits=1), base_bits=1)
    lean = estimate_peak_bytes(cfg, w, P, F, default_rows(base_bits=1, uniform_refs=(2,)), base_bits=1)
    base2 = estimate_peak_bytes(cfg, w, P, F, default_rows(base_bits=2), base_bits=2)
    # one history copy (int3) less, plus its tool-output KV and fp32 scratch
    assert P * tok <= full - lean <= 1.25 * P * tok
    assert 30e9 < lean < 40e9 and 30e9 < base2 < 40e9    # fits an A100 40 GB (42.5e9 bytes)
    assert full > 0.95 * 42.5e9                          # with int3, base 1 would not


def _run_dir(path, schemes, base_bits, pauses, seed, ref_nll):
    rng = np.random.default_rng(seed)
    rows, samples = [], []
    for (inst, turn), ref in zip(pauses, ref_nll):
        for name, k in schemes:
            rows.append({"instance_id": inst, "turn": turn, "hist_len": 9000, "fresh_len": 50, "steps": 40,
                         "scheme": name, "policy": "s2" if k else ("all" if name == "noise_ctrl" else "none"),
                         "per_layer": 0, "base_bits": base_bits, "prefill_fetch_k": 0, "tool_prefill": "x",
                         "delta_nll": 0.0 if name == "bf16" else float(rng.normal(0.01, 0.01)),
                         "top1_agree": 0.97, "k": k if k != MATCH else 500, "k_label": "match" if k == MATCH else str(k),
                         "fetched_pairs_per_step": 1.0, "distinct_tokens_per_layer_step": 1.0,
                         "recall_vs_oracle": 0.5 if k else float("nan"), "moved_bytes_turn": 1e6,
                         "unique_bytes_turn": 5e5, "prefill_moved_bytes": 0, "resident_bytes": 2.0e8,
                         "spec_guess_acc": float("nan"), "bf16_bytes": 1e9, "base_bytes": 1.3e8 if base_bits == 1 else 2e8,
                         "int1_bytes": 1.3e8, "int2_bytes": 2e8, "int3_bytes": 2.6e8})
        samples.append({"instance_id": inst, "turn": turn, "hist_len": 9000, "fresh_len": 50, "steps": 40,
                        "ref_nll": ref, "seconds": 1.0, "model": "m", "base_bits": base_bits, "k_match": 500,
                        "check_bf16_max_abs_logp": 0.1, "check_bf16_nll_batched": ref, "check_bf16_nll_perstep": ref,
                        "check_base_max_abs_logp": 0.1, "check_base_nll_batched": ref + 0.03,
                        "check_base_nll_perstep": ref + 0.03})
    path.mkdir()
    pd.DataFrame(rows).to_csv(path / "metrics.csv", index=False)
    pd.DataFrame(samples).to_csv(path / "samples.csv", index=False)
    return pd.DataFrame(rows)


def test_summary_joins_int3_from_the_base2_run(tmp_path):
    pauses = [("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5)]
    b2 = _run_dir(tmp_path / "m-base2", [("bf16", 0), ("noise_ctrl", 0), ("int2", 0), ("int3", 0), ("s2", 16)],
                  2, pauses, 0, [0.5, 0.6, 0.7, 0.8, 0.9])
    _run_dir(tmp_path / "m-base1", [("bf16", 0), ("noise_ctrl", 0), ("int1", 0), ("int2", 0), ("s2", 16),
                                    ("s2", MATCH)], 1, pauses[:4], 1, [0.501, 0.6, 0.7, 0.8])
    text = summarize(tmp_path / "m-base1", tmp_path / "m-base2")
    assert "int3 joined from m-base2 (4/4 pauses" in text
    assert "max 0.0010" in text  # bf16 reference gap between the two runs
    # the joined int3 row is the base-2 run's int3 on the 4 shared pauses
    want = b2[(b2.scheme == "int3") & (b2.instance_id != "e")].delta_nll.mean()
    row = next(line for line in text.splitlines() if line.startswith("| int3 | "))
    assert f"{want:+.4f}" in row
    assert "dNLL - int3 [95% CI]" in text and "Equal GPU-resident memory: int1" in text
    # without --join there is no int3 column
    assert "dNLL - int3" not in summarize(tmp_path / "m-base1")
