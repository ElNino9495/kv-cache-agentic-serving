"""Summarize the per-step fetch study (run_fetch.py) into results/gate_fetch/<model>/summary.md.

  - checks: bf16 control, per-step vs sdpa teacher forcing (NLL level; per-token max reported)
  - noise floor: dNLL of the noise_ctrl row (bf16 tool-output prefill + fetch-all, equal to the bf16 row in
    exact arithmetic), mean and spread over pauses, shown next to the scheme results
  - quality: dNLL and top-1 vs the bf16 row, paired 95% CIs against uniform int2, uniform int3 and s2 at
    the same k, recall vs the oracle set, tokens fetched per step, speculative-token guess accuracy
  - robustness per pause: mean, median, mean without the scheme's largest outlier, how many pauses each
    scheme beats int3 on (at all, and by more than the noise floor) and loses on by more than it
  - bytes: total bytes moved per turn by per-step fetching (fetch per step x steps, and the unique
    (layer, head, token) pairs over the turn) next to uniform int3's one-time extra reload bytes
    (int3 - int2 history); GPU-resident history (int2 + k fetched tokens) next to resident int3

    python -m experiments.gate.summarize_fetch [--dir results/gate_fetch/llama-3.1-8b-instruct]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.gate.plot_gate import paired_ci

ROOT = Path(__file__).resolve().parents[2]
KEY = ["instance_id", "turn"]
ORDER = ["bf16", "noise_ctrl", "int2", "int3", "fetch_all", "s1", "sens_a", "sens_b", "spec", "s2",
         "s2sens_a", "s2sens_b", "oracle", "s1_layer"]
CONTROLS = ("bf16", "noise_ctrl", "int2", "int3", "fetch_all")
MB = 2 ** 20


def _ci(a: pd.Series, b: pd.Series) -> str:
    idx = a.index.intersection(b.index)
    d = (a.loc[idx] - b.loc[idx]).to_numpy()
    lo, up = paired_ci(d)
    return f"{d.mean():+.4f} [{lo:+.4f}, {up:+.4f}]"


def _order(lab: str):
    base = lab.split("@")[0]
    return (ORDER.index(base) if base in ORDER else len(ORDER), int(lab.split("@")[1]) if "@" in lab else 0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=str(ROOT / "results" / "gate_fetch" / "llama-3.1-8b-instruct"))
    args = p.parse_args()
    d = Path(args.dir)
    m, s = pd.read_csv(d / "metrics.csv"), pd.read_csv(d / "samples.csv")
    m["label"] = m.scheme + np.where(m.k > 0, "@" + m.k.astype(str), "")
    m = m.set_index(KEY)
    by = {lab: g for lab, g in m.groupby("label")}
    labels = sorted(by, key=_order)
    int2, int3 = by["int2"].delta_nll, by["int3"].delta_nll

    ctrl = by["bf16"].delta_nll.abs().max()
    lines = [f"# Per-step fetch study: {s.model.iloc[0]}", "",
             f"{len(s)} pauses, {s.steps.mean():.1f} teacher-forced steps per pause on average (cap 64). "
             "dNLL in nats per target token against the per-step bf16 row. Base: int2 history, block 16; block 0, "
             "the recent 64 tokens and the trailing partial block are bf16. Tool output prefilled over the int2 "
             "history without fetching (int3 row: over int3; bf16 row: over bf16).", "",
             "## Checks", "",
             f"- bf16 row vs itself: max |dNLL| {ctrl:.1e}"]
    nll_b = (s.check_bf16_nll_perstep - s.check_bf16_nll_batched).abs()
    nll_2 = (s.check_int2_nll_perstep - s.check_int2_nll_batched).abs()
    dn_b = s.check_int2_nll_batched - s.check_bf16_nll_batched
    dn_p = s.check_int2_nll_perstep - s.check_bf16_nll_perstep
    lines += [f"- per-step vs sdpa teacher forcing, NLL difference per pause (the stop criterion): bf16 row max "
              f"{nll_b.max():.4f} (mean {nll_b.mean():.4f}), int2 row max {nll_2.max():.4f} (mean {nll_2.mean():.4f})",
              f"- uniform int2 dNLL, sdpa batched {dn_b.mean():+.4f} vs per-step {dn_p.mean():+.4f} "
              f"(max per-pause |difference| {(dn_b - dn_p).abs().max():.4f})",
              f"- largest single-token log-prob difference (reported only): bf16 {s.check_bf16_max_abs_logp.max():.3f}, "
              f"int2 {s.check_int2_max_abs_logp.max():.3f}"]
    if "seconds" in s and s.seconds.notna().any():
        lines.append(f"- {s.seconds.mean():.1f} s per pause on average, {s.seconds.sum() / 60:.1f} min total")
    spec = [lab for lab in labels if lab.startswith("spec@")]
    if spec and "spec_guess_acc" in m:
        lines.append("- speculative token (spec rows): greedy guess of the next input correct on "
                     + ", ".join(f"{by[lab].spec_guess_acc.mean():.1%} ({lab})" for lab in spec) + " of steps")
    lines.append("")

    noise, thr = None, None
    if "noise_ctrl" in by:
        nz = by["noise_ctrl"]
        nd = nz.delta_nll
        lo, up = paired_ci(nd.to_numpy())
        thr = 2 * nd.std(ddof=1)
        noise = (f"Noise floor (noise_ctrl: bf16 tool-output prefill + fetch-all, exactly the bf16 row in exact "
                 f"arithmetic, so any dNLL is bf16 rounding in the per-step path): mean {nd.mean():+.4f} "
                 f"[95% CI {lo:+.4f}, {up:+.4f}], s.d. over pauses {nd.std(ddof=1):.4f}, "
                 f"mean |dNLL| {nd.abs().mean():.4f}, max |dNLL| {nd.abs().max():.4f}; "
                 f"top-1 {nz.top1_agree.mean():.3f} (min {nz.top1_agree.min():.3f}). Per-pause scheme differences "
                 f"smaller than about {thr:.4f} (2 s.d.) are within rounding noise.")

    lines += ["## Quality", "",
              "Fetched = (layer, KV head, token) pairs fetched per step; distinct = distinct token positions per "
              "layer per step. Recall = share of each fetched set that is also in the oracle set of the same size, "
              "computed with the row's own current query. The last column compares each fetch scheme with s2 "
              "(current-step int2 attention) at the same k."]
    if noise:
        lines += ["", noise]
    lines += ["", "| scheme | dNLL | top-1 | recall | fetched / step | distinct / layer | dNLL - int2 [95% CI] | "
              "dNLL - int3 [95% CI] | dNLL - s2 same k [95% CI] |", "|---|---|---|---|---|---|---|---|---|"]
    for lab in labels:
        g = by[lab]
        fetch = g.policy.iloc[0] != "none"
        rec = "" if g.recall_vs_oracle.isna().all() else f"{g.recall_vs_oracle.mean():.2f}"
        s2 = f"s2@{lab.split('@')[1]}" if "@" in lab else None
        vs_s2 = _ci(g.delta_nll, by[s2].delta_nll) if s2 in by and s2 != lab else ""
        lines.append(f"| {lab} | {g.delta_nll.mean():+.4f} | {g.top1_agree.mean():.3f} | {rec} | "
                     f"{f'{g.fetched_pairs_per_step.mean():,.0f}' if fetch else ''} | "
                     f"{f'{g.distinct_tokens_per_layer_step.mean():,.0f}' if fetch else ''} | "
                     f"{'' if lab == 'int2' else _ci(g.delta_nll, int2)} | "
                     f"{'' if lab == 'int3' else _ci(g.delta_nll, int3)} | {vs_s2} |")
    lines.append("")

    n = len(int3)
    lines += ["## Robustness per pause", "",
              "Mean without outlier = mean over pauses after removing the scheme's single pause farthest from its "
              "median (named). Beats int3 = pauses where the scheme's dNLL is below int3's"
              + (f"; by > noise = by more than the noise floor ({thr:.4f}, 2 s.d. of noise_ctrl)" if thr else "")
              + f". {n} pauses.", "",
              "| scheme | mean | median | mean without outlier | outlier pause (dNLL) | beats int3 | "
              "beats int3 by > noise | loses to int3 by > noise |", "|---|---|---|---|---|---|---|---|"]
    for lab in labels:
        x = by[lab].delta_nll
        far = (x - x.median()).abs().idxmax()
        d3 = (x - int3.loc[x.index]).dropna()
        better = int((d3 < 0).sum())
        bt = f"{int((d3 < -thr).sum())}/{len(d3)}" if thr else ""
        wt = f"{int((d3 > thr).sum())}/{len(d3)}" if thr else ""
        lines.append(f"| {lab} | {x.mean():+.4f} | {x.median():+.4f} | {x.drop(far).mean():+.4f} | "
                     f"{far[0]} t{far[1]} ({x.loc[far]:+.4f}) | "
                     f"{'' if lab == 'int3' else f'{better}/{len(d3)}'} | {'' if lab == 'int3' else bt} | "
                     f"{'' if lab == 'int3' else wt} |")
    lines.append("")

    b = by["int2"]
    extra3 = (b.int3_bytes - b.int2_bytes)
    lines += ["## Bytes", "",
              f"Per pause on average: bf16 history {b.bf16_bytes.mean() / MB:,.0f} MB; resident int2 "
              f"{(b.int2_bytes / b.bf16_bytes).mean():.3f} and int3 {(b.int3_bytes / b.bf16_bytes).mean():.3f} of bf16; "
              f"uniform int3's one-time extra reload over int2: {extra3.mean() / MB:,.0f} MB "
              f"({(extra3 / b.bf16_bytes).mean():.3f} of bf16).", "",
              "Moved per turn = bytes fetched at every step, summed over the turn's steps (no reuse across steps). "
              "Unique per turn = distinct (layer, head, token) pairs fetched over the turn (what a fetch cache would "
              "move). Both are shown as MB and as a multiple of int3's one-time extra reload on the same pause. "
              "Resident = int2 history + one step's fetched set, as a fraction of bf16, next to resident int3. "
              "spec also decodes one speculative token per step (compute, not bytes).", "",
              "| scheme | moved / turn (MB) | x int3 extra | unique / turn (MB) | x int3 extra | resident | "
              "resident int3 |", "|---|---|---|---|---|---|---|"]
    for lab in labels:
        g = by[lab]
        if g.policy.iloc[0] == "none" or lab in CONTROLS:
            continue
        e = (g.int3_bytes - g.int2_bytes)
        lines.append(f"| {lab} | {g.moved_bytes_turn.mean() / MB:,.1f} | {(g.moved_bytes_turn / e).mean():.2f} | "
                     f"{g.unique_bytes_turn.mean() / MB:,.1f} | {(g.unique_bytes_turn / e).mean():.2f} | "
                     f"{(g.resident_bytes / g.bf16_bytes).mean():.4f} | {(g.int3_bytes / g.bf16_bytes).mean():.4f} |")
    lines.append("")

    out = d / "summary.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
