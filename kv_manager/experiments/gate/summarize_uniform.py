"""Summarize uniform-only gate runs (run_gate.py --uniform-only) across target models.

Reads results/gate_uniform/<model>/{metrics,samples,dropped}.csv and writes
results/gate_uniform/summary.md with, per model and block size:
  - the uniform_bf16 control check (dNLL must be exactly 0)
  - bytes, dNLL, top-1 for uniform int4 / int3 / int2, and each one minus uniform_int4 (paired CI)
  - a cross-model table restricted to the pauses every model ran (Qwen3 drops pauses > 32,768 tokens)

For the model that the first gate ran on (--compare-model, default Llama-3.1-8B), it also checks the
rerun against results/gate/metrics.csv (same pauses, same history lengths, uniform_int4 reproduced) and
compares uniform int3 with the existing consensus and specprefill rows whose bytes are within
--bytes-tol of int3's, paired on the same (pause, block size).

    python -m experiments.gate.summarize_uniform
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.gate.plot_gate import paired_ci

ROOT = Path(__file__).resolve().parents[2]
KEY = ["instance_id", "turn"]
UNIFORM = ["uniform_int4", "uniform_int3", "uniform_int2"]


def _load(d: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    m = pd.read_csv(d / "metrics.csv")
    s = pd.read_csv(d / "samples.csv")
    dr = pd.read_csv(d / "dropped.csv") if (d / "dropped.csv").exists() else pd.DataFrame(columns=KEY)
    return m, s, dr


def _diff_ci(a: pd.Series, b: pd.Series) -> str:
    idx = a.index.intersection(b.index)
    diff = (a.loc[idx] - b.loc[idx]).to_numpy()
    lo, up = paired_ci(diff)
    return f"{diff.mean():+.4f} [{lo:+.4f}, {up:+.4f}]"


def model_section(name: str, m: pd.DataFrame, s: pd.DataFrame, dr: pd.DataFrame) -> list[str]:
    model = s["model"].iloc[0] if "model" in s else name
    lines = [f"## {model}", "",
             f"{len(s)} pauses evaluated, {len(dr)} skipped for context length"
             + (f" ({', '.join(f'{r.instance_id} t{r.turn}: {r.total}' for r in dr.itertuples())})"
                if len(dr) else "") + ".", ""]
    for B, g in m.groupby("block_size"):
        g = g.set_index(KEY)
        ctrl = g[g.method == "uniform_bf16"].delta_nll.abs().max()
        u4 = g[g.method == "uniform_int4"].delta_nll
        lines += [f"### Block size {B}", "",
                  f"Control uniform_bf16: max |dNLL| {ctrl:.2e} ({'OK' if ctrl == 0 else 'NOT ZERO'})", "",
                  "| scheme | bytes | dNLL | top-1 | dNLL - uniform_int4 [95% CI] |", "|---|---|---|---|---|"]
        for name_ in UNIFORM:
            r = g[g.method == name_]
            if not len(r):
                continue
            ci = "" if name_ == "uniform_int4" else _diff_ci(r.delta_nll, u4)
            lines.append(f"| {name_} | {r.bytes_frac.mean():.3f} | {r.delta_nll.mean():+.4f} | "
                         f"{r.top1_agree.mean():.3f} | {ci} |")
        lines.append("")
    return lines


def cross_model(runs: dict[str, pd.DataFrame]) -> list[str]:
    common = None
    for m in runs.values():
        keys = set(map(tuple, m[KEY].drop_duplicates().to_numpy().tolist()))
        common = keys if common is None else common & keys
    lines = ["## Across models, on the pauses every model ran", "",
             f"{len(common)} common pauses. Cells: mean dNLL (bytes).", ""]
    for B in sorted({b for m in runs.values() for b in m.block_size.unique()}):
        lines += [f"Block size {B}:", "", "| model | " + " | ".join(UNIFORM) + " |",
                  "|---|" + "---|" * len(UNIFORM)]
        for name, m in runs.items():
            g = m[(m.block_size == B)]
            g = g[[k in common for k in map(tuple, g[KEY].to_numpy().tolist())]]
            cells = []
            for u in UNIFORM:
                r = g[g.method == u]
                cells.append(f"{r.delta_nll.mean():+.4f} ({r.bytes_frac.mean():.3f})" if len(r) else "")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def compare_with_gate(m: pd.DataFrame, s: pd.DataFrame, gate_metrics: Path, tol: float) -> list[str]:
    old = pd.read_csv(gate_metrics)
    old["hi"] = old["hi"].fillna("")
    lines = [f"## Uniform int3 vs draft-guided rows from the first gate run ({gate_metrics.name})", ""]

    # Sanity: the rerun must be the same pauses with the same tokens and reproduce uniform_int4.
    if "gate_hist_len" in s:
        bad = s[s.hist_len != s.gate_hist_len]
        lines.append(f"History lengths match the first run on {len(s) - len(bad)}/{len(s)} pauses"
                     + (f"; mismatched: {', '.join(f'{r.instance_id} t{r.turn}' for r in bad.itertuples())}"
                        if len(bad) else "") + ".")
    for B in sorted(m.block_size.unique()):
        new4 = m[(m.block_size == B) & (m.method == "uniform_int4")].set_index(KEY)
        old4 = old[(old.block_size == B) & (old.method == "uniform_int4")].set_index(KEY)
        idx = new4.index.intersection(old4.index)
        d_ref = (new4.ref_nll.loc[idx] - old4.ref_nll.loc[idx]).abs().max()
        d_4 = (new4.delta_nll.loc[idx] - old4.delta_nll.loc[idx]).abs().max()
        lines.append(f"B={B}: {len(idx)} paired pauses; max |ref NLL new - old| {d_ref:.2e}, "
                     f"max |uniform_int4 dNLL new - old| {d_4:.2e}.")
    lines += ["", f"Draft rows with mean bytes within {tol} of uniform_int3's, paired on the same pauses. "
                  "Negative difference = uniform int3 better.", ""]

    for B in sorted(m.block_size.unique()):
        i3 = m[(m.block_size == B) & (m.method == "uniform_int3")].set_index(KEY)
        if not len(i3):
            continue
        b3 = i3.bytes_frac.mean()
        lines += [f"### Block size {B}: uniform_int3 at {b3:.3f} bytes, dNLL {i3.delta_nll.mean():+.4f}", "",
                  "| method | hi | keep | bytes | dNLL (method) | dNLL (int3, same pauses) | int3 - method [95% CI] |",
                  "|---|---|---|---|---|---|---|"]
        g = old[(old.block_size == B) & old.method.isin(["consensus", "specprefill"])]
        for (method, hi, keep), r in g.groupby(["method", "hi", "keep_frac"]):
            r = r.set_index(KEY)
            if abs(r.bytes_frac.mean() - b3) > tol:
                continue
            idx = i3.index.intersection(r.index)
            lines.append(f"| {method} | {hi} | {keep:g} | {r.bytes_frac.mean():.3f} | "
                         f"{r.delta_nll.loc[idx].mean():+.4f} | {i3.delta_nll.loc[idx].mean():+.4f} | "
                         f"{_diff_ci(i3.delta_nll, r.delta_nll)} |")
        lines.append("")
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(ROOT / "results" / "gate_uniform"))
    p.add_argument("--gate-metrics", default=str(ROOT / "results" / "gate" / "metrics.csv"))
    p.add_argument("--compare-model", default="llama-3.1-8b-instruct",
                   help="subdirectory of --root to compare with the first gate run")
    p.add_argument("--bytes-tol", type=float, default=0.03)
    args = p.parse_args()

    root = Path(args.root)
    order = ["llama-3.1-8b-instruct", "llama-3.2-3b-instruct", "qwen3-8b"]
    dirs = sorted((d for d in root.iterdir() if (d / "metrics.csv").exists()),
                  key=lambda d: (order.index(d.name) if d.name in order else len(order), d.name))
    if not dirs:
        raise SystemExit(f"no runs under {root}")

    lines = ["# Uniform reload across target models", "",
             "dNLL is in nats per target token (0 = identical to bf16 reload). bytes = reload bytes / bf16 "
             "reload bytes. int3 is quantized with kvm.quant's arithmetic without packing; its bytes are "
             "kvm.quant.block_bytes(3, ...).", ""]
    runs = {}
    for d in dirs:
        m, s, dr = _load(d)
        runs[d.name] = m
        lines += model_section(d.name, m, s, dr)
    if len(runs) > 1:
        lines += cross_model(runs)
    if args.compare_model in runs and Path(args.gate_metrics).exists():
        _, s, _ = _load(root / args.compare_model)
        lines += compare_with_gate(runs[args.compare_model], s, Path(args.gate_metrics), args.bytes_tol)

    out = root / "summary.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
