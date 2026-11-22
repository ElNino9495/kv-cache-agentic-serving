"""Summarize the per-step fetch study (run_fetch.py) into <dir>/summary.md.

  - checks: bf16 control, per-step vs sdpa teacher forcing (NLL level; per-token max reported)
  - noise floor: dNLL of the noise_ctrl row (bf16 tool-output prefill + fetch-all, equal to the bf16 row in
    exact arithmetic), mean and spread over pauses, shown next to the scheme results
  - quality: dNLL and top-1 vs the bf16 row, paired 95% CIs against the uniform base (int2, or int1 with
    --base-bits 1), the other uniform references (int2/int3) and s2 at the same k, recall vs the oracle set,
    tokens fetched per step, speculative-token guess accuracy
  - equal GPU-resident memory: rows with k = "match" (base + one step's fetched set = uniform int(base+1)'s
    memory) against that uniform row: int1 + fetch vs int2, or int2 + fetch vs int3
  - robustness per pause: mean, median, mean without the scheme's largest outlier, how many pauses each
    scheme beats the next uniform precision on (at all, and by more than the noise floor)
  - prefill fetch (if run with --prefill-fetch-k): fetch_all vs fetch_all_noprefill, prefill bytes
  - bytes: total bytes moved per turn by per-step fetching next to uniform int(base+1)'s one-time extra reload
    bytes over the base; GPU-resident history (base + k fetched tokens) next to resident int(base+1)

Runs with --target-source self (the bf16 model's own message as the target) get a KL column: KL(bf16 row || row)
of the next-token distribution averaged over steps, with paired CIs against the uniform references, since dNLL on
self-written text depends on which tokens were sampled.

Works on older runs too (base int2, no "match" rows, check_int2_* columns).

--join DIR takes uniform rows a run left out (default int3; e.g. the base-1 run skips the int3 history copy to
fit a 40 GB GPU) from another run on the same pauses (e.g. the base-2 run). Each joined row keeps its dNLL
against its own run's bf16 row; the summary reports how far the two runs' bf16 reference NLLs differ.

Run 5 row sets (run_fetch --rows a1 / a2 / c) add: a multi-guess table (candidate coverage, recall, fetched count,
dNLL / KL vs s2 at the same k), the union rows against the s2 row closest to their actual fetched count, a fetch-pool
table (hit rate, misses moved), the scoring-key gap (s2 / s2_k2score vs oracle), total bytes per turn including the
prefill fetch, and the pre-registered decision rules R1-R4 with the parts this run decides (PASS / FAIL).
--rules DIR [DIR ...] evaluates R1-R4 over all run-5 folders (roles found from each folder's row set, targets and
base) and writes rules_run5.md next to them (or --rules-out).

    python -m experiments.gate.summarize_fetch --dir results/gate_fetch/llama-3.1-8b-instruct-base2
    python -m experiments.gate.summarize_fetch --rules results/gate_fetch/*-a1 results/gate_fetch/*-a2 results/gate_fetch/*-c
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.gate.plot_gate import paired_ci

ROOT = Path(__file__).resolve().parents[2]
KEY = ["instance_id", "turn"]
ORDER = ["bf16", "noise_ctrl", "k2v1", "int1", "int2", "int3", "fetch_all", "fetch_all_noprefill", "s1", "sens_a",
         "sens_b", "spec_chain", "spec_chain_union2", "spec_chain_union3", "spec_chain_wtd2", "spec_chain_wtd3",
         "spec", "spec_union2", "spec_union3", "spec_wtd2", "spec_wtd3", "s2", "s2sens_a", "s2sens_b", "s2_k2score",
         "oracle", "s2_pool_Mmatch", "s2_pool_Mhalf", "spec_chain_pool_Mmatch", "spec_chain_pool_Mhalf", "s1_layer",
         "s2_int1base", "oracle_int1base"]
CONTROLS = ("bf16", "noise_ctrl", "k2v1", "int1", "int2", "int3", "fetch_all", "fetch_all_noprefill")
NEXT = {"int1": "int2", "int2": "int3", "k2v1": "int2"}

RULES_TEXT = [
    "R1: spec_chain_wtd3@128 improves on spec_chain@128 with 95% CI excluding 0, AND closes >= half the "
    "spec_chain-to-s2 gap at k=128, on both real and self-T0.7 targets.",
    "R2: on the k2v1 base, s2@16 closes >= half the s2-to-oracle gap seen on the int1 base, AND k2v1 s2@match beats "
    "int1 s2@match at equal resident memory (CI excludes 0).",
    "R3: s2_pool@64 (or @128) with M = k_match: the 95% CI upper bound of mean(s2_pool - s2@match) must be below "
    "2 x the noise control's per-pause s.d. (computed within that job), AND it moves <= 2x int2's one-time extra "
    "reload per turn (per-step bytes, excluding prefill).",
    "R4: s2_pool@64 with M = k_match/2 is no worse than int2 (95% CI upper bound < +0.01) on both real and self-T0.7 "
    "targets, while moving <= 2x int2's one-time extra reload per turn (per-step bytes, excluding prefill).",
    "Judge R1-R4 only on the rows named in them; do not substitute a better-looking selector (e.g. s2sens_b) after "
    "the fact.",
]
RULES_HOW = [
    "How each test is computed (fixed before the run): CIs are paired 95% bootstrap CIs over pauses (plot_gate."
    "paired_ci); means are over pauses; every comparison is within one job (one batch).",
    "R1 (jobs a1, real and self T=0.7): CI of spec_chain_wtd3@128 - spec_chain@128 entirely below 0, and "
    "mean(spec_chain@128) - mean(spec_chain_wtd3@128) >= 0.5 x (mean(spec_chain@128) - mean(s2@128)) with that "
    "gap > 0; both jobs must pass.",
    "R2 (job c): (s2@16 - oracle@16) on k2v1 <= 0.5 x (s2_int1base@16 - oracle_int1base@16) on int1 (means, each "
    "gap within its own base, int1 gap > 0), and the CI of s2@match (k2v1) - s2_int1base@match entirely below 0 "
    "(both fill up to uniform int2's resident memory).",
    "R3 (job a2, real targets; the self-target job is shown for information): for s2_pool_Mmatch@64 or @128, the "
    "CI upper bound of s2_pool_Mmatch@k - s2@match < 2 x s.d. over pauses of noise_ctrl's dNLL in that job, and the "
    "mean over pauses of moved bytes per turn / int2's one-time extra reload over int1 <= 2; passes if either k "
    "passes both parts.",
    "R4 (jobs a2, real and self T=0.7): CI upper bound of s2_pool_Mhalf@64 - int2 < +0.01, and the mean over pauses "
    "of moved bytes per turn / int2's one-time extra reload <= 2; both jobs must pass.",
]
MB = 2 ** 20


def _ci(a: pd.Series, b: pd.Series) -> str:
    idx = a.index.intersection(b.index)
    d = (a.loc[idx] - b.loc[idx]).dropna().to_numpy()
    if not len(d):
        return ""
    lo, up = paired_ci(d)
    return f"{d.mean():+.4f} [{lo:+.4f}, {up:+.4f}]"


def _order(lab: str):
    base, _, k = lab.partition("@")
    kk = 10 ** 9 if k == "match" else int(k) if k else 0
    return (ORDER.index(base) if base in ORDER else len(ORDER), kk)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=str(ROOT / "results" / "gate_fetch" / "llama-3.1-8b-instruct"))
    p.add_argument("--join", default=None, help="another run on the same pauses to take missing uniform rows from")
    p.add_argument("--join-rows", nargs="*", default=["int3"])
    p.add_argument("--rules", nargs="+", default=None, metavar="DIR",
                   help="evaluate R1-R4 over these run-5 folders and write rules_run5.md (see --rules-out)")
    p.add_argument("--rules-out", default=None, help="output file for --rules (default <first DIR's parent>/rules_run5.md)")
    args = p.parse_args()
    if args.rules:
        dirs = [Path(x) for x in args.rules]
        out = Path(args.rules_out) if args.rules_out else dirs[0].parent / "rules_run5.md"
        print(rules_report(dirs, out))
        return
    print(summarize(Path(args.dir), Path(args.join) if args.join else None, args.join_rows))


def summarize(d: Path, join: Path | None = None, join_rows=("int3",)) -> str:
    """Write <d>/summary.md and return its text."""
    m, s = pd.read_csv(d / "metrics.csv"), pd.read_csv(d / "samples.csv")
    joined = []
    if join is not None:
        jm, js = pd.read_csv(join / "metrics.csv"), pd.read_csv(join / "samples.csv")
        pauses = set(map(tuple, s[KEY].to_numpy().tolist()))
        for name in join_rows:
            if name in set(m.scheme):
                continue
            shared = np.array([tuple(x) in pauses for x in jm[KEY].to_numpy().tolist()])
            add = jm[(jm.scheme == name).to_numpy() & shared].copy()
            if not len(add):
                continue
            if "k_label" in m and "k_label" not in add:
                add["k_label"] = "0"
            if "base_bits" in m:
                add["base_bits"] = m.base_bits.iloc[0]
            m = pd.concat([m, add[[c for c in add.columns if c in m.columns]]], ignore_index=True)
            joined.append((name, len(add)))
        both = s.set_index(KEY).ref_nll.to_frame("a").join(js.set_index(KEY).ref_nll.to_frame("b"), how="inner")
        ref_gap = (both.a - both.b).abs()
    bits = int(m.base_bits.iloc[0]) if "base_bits" in m else 2
    base = m["base"].iloc[0] if "base" in m and isinstance(m["base"].iloc[0], str) else f"int{bits}"
    nxt = NEXT[base]
    if "k_label" not in m:
        m["k_label"] = m.k.astype(str)
    m["label"] = m.scheme + np.where(m.k_label.astype(str) != "0", "@" + m.k_label.astype(str), "")
    m = m.set_index(KEY)
    by = {lab: g for lab, g in m.groupby("label")}
    labels = sorted(by, key=_order)
    refs = [r for r in ("int1", "int2", "int3") if r != base and r in by and (r != "int1" or base == "k2v1")]
    ub, un = by[base].delta_nll, by[nxt].delta_nll
    ck = "base" if "check_base_nll_batched" in s else "int2"
    pf_k = int(m.prefill_fetch_k.iloc[0]) if "prefill_fetch_k" in m else 0
    self_tgt = "target_source" in s and (s.target_source == "self").any()
    temp = float(s.temperature.iloc[0]) if self_tgt else None
    kl = "kl_bf16" in m and m.kl_bf16.notna().any()

    lines = [f"# Per-step fetch study: {s.model.iloc[0]}, base {base}" + (f", prefill fetch k={pf_k}" if pf_k else "")
             + (f", self-written targets (T={temp:g})" if self_tgt else ""),
             "",
             f"{len(s)} pauses, {s.steps.mean():.1f} teacher-forced steps per pause on average (cap 64). "
             + (f"Targets: the bf16 model's own next message ({'greedy' if temp == 0 else f'sampled at temperature {temp:g}'}"
                f", seed {s.seed.iloc[0]} + per-pause offset; bf16 prefill of history and tool output; "
                f"{s.target_len.min()}-{s.target_len.max()} tokens, mean {s.target_len.mean():.1f}), saved in "
                "self_targets.jsonl; every row is teacher-forced on them. " if self_tgt else "")
             + f"dNLL in nats per target token against the per-step bf16 row. Base: {base} history, block 16"
             + (" (SpeCache's modified 1-bit quantizer)" if bits == 1 else "")
             + "; block 0, the recent 64 tokens and the trailing partial block are bf16. Tool output prefilled over "
             f"the {base} history " + (f"with a per-chunk top-{pf_k} fetch (fetch rows; the uniform {base} row and "
                                        "fetch_all_noprefill without)" if pf_k else "without fetching")
             + "; uniform rows over their own precision, the bf16 row over bf16."
             + (" k2v1 = int2 keys (the int2 copy's) + int1 values (the int1 copy's); its equal-memory reference is "
                "uniform int2. Rows named *_int1base sit on the int1 base (tool output prefilled over int1 with the same "
                "prefill fetch); their @match fills int1 up to int2's memory." if base == "k2v1" else ""), "",
             "## Checks", "",
             f"- bf16 row vs itself: max |dNLL| {by['bf16'].delta_nll.abs().max():.1e}"]
    nll_b = (s.check_bf16_nll_perstep - s.check_bf16_nll_batched).abs()
    nll_2 = (s[f"check_{ck}_nll_perstep"] - s[f"check_{ck}_nll_batched"]).abs()
    dn_b = s[f"check_{ck}_nll_batched"] - s.check_bf16_nll_batched
    dn_p = s[f"check_{ck}_nll_perstep"] - s.check_bf16_nll_perstep
    lines += [f"- per-step vs sdpa teacher forcing, NLL difference per pause (the stop criterion): bf16 row max "
              f"{nll_b.max():.4f} (mean {nll_b.mean():.4f}), {base} row max {nll_2.max():.4f} (mean {nll_2.mean():.4f})",
              f"- uniform {base} dNLL, sdpa batched {dn_b.mean():+.4f} vs per-step {dn_p.mean():+.4f} "
              f"(max per-pause |difference| {(dn_b - dn_p).abs().max():.4f})",
              f"- largest single-token log-prob difference (reported only): bf16 {s.check_bf16_max_abs_logp.max():.3f}, "
              f"{base} {s[f'check_{ck}_max_abs_logp'].max():.3f}"]
    if "seconds" in s and s.seconds.notna().any():
        lines.append(f"- {s.seconds.mean():.1f} s per pause on average, {s.seconds.sum() / 60:.1f} min total")
    for pre, what in (("spec@", "spec rows, guess of each step's input from the real previous token"),
                      ("spec_chain@", "spec_chain rows, SpeCache's chained guess of the next input")):
        spec = [lab for lab in labels if lab.startswith(pre)]
        if spec and "spec_guess_acc" in m:
            lines.append(f"- speculative token ({what}): correct on "
                         + ", ".join(f"{by[lab].spec_guess_acc.mean():.1%} ({lab})" for lab in spec) + " of steps")
    if "k_match" in s:
        lines.append(f"- k = match: {s.k_match.mean():,.0f} tokens per KV head and layer on average "
                     f"(min {s.k_match.min():,}, max {s.k_match.max():,}), i.e. {base} + fetched set = {nxt}'s memory")
    for name, n_join in joined:
        lines.append(f"- {name} joined from {join.name} ({n_join}/{len(s)} pauses; its dNLL is against that run's own "
                     f"bf16 row; the two runs' bf16 reference NLL differs by mean {ref_gap.mean():.4f}, "
                     f"max {ref_gap.max():.4f} per pause, i.e. bf16 rounding with a different batch)")
    if "peak_gpu_gb" in s and s.peak_gpu_gb.notna().any():
        lines.append(f"- GPU memory: measured peak {s.peak_gpu_gb.max():.1f} GB (allocated by PyTorch), estimated "
                     f"{s.est_peak_gb.max():.1f} GB")
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
                 f"top-1 {nz.top1_agree.mean():.3f} (min {nz.top1_agree.min():.3f})"
                 + (f"; KL vs bf16 mean {nz.kl_bf16.mean():.4f} (max {nz.kl_bf16.max():.4f})" if kl else "")
                 + f". Per-pause scheme differences smaller than about {thr:.4f} (2 s.d.) are within rounding noise.")

    cols = [base] + refs
    lines += ["## Quality", "",
              "Fetched = (layer, KV head, token) pairs fetched per step; distinct = distinct token positions per "
              "layer per step. Recall = share of each fetched set that is also in the oracle set of the same size, "
              "computed with the row's own current query. The last column compares each fetch scheme with s2 "
              "(current-step attention over base keys) at the same k."
              + (" KL = KL(bf16 row || row) of the next-token distribution, nats, averaged over steps and pauses."
                 if kl else "")]
    if noise:
        lines += ["", noise]
    lines += ["", "| scheme | dNLL | " + ("KL | " if kl else "") + "top-1 | recall | fetched / step | distinct / layer | "
              + " | ".join(f"dNLL - {c} [95% CI]" for c in cols) + " | dNLL - s2 same k [95% CI] |"
              + "".join(f" KL - {c} [95% CI] |" for c in (cols if kl else [])),
              "|---|---|" + ("---|" if kl else "") + "---|---|---|---|" + "---|" * (len(cols) + 1)
              + ("---|" * len(cols) if kl else "")]
    for lab in labels:
        g = by[lab]
        fetch = g.policy.iloc[0] != "none"
        rec = "" if g.recall_vs_oracle.isna().all() else f"{g.recall_vs_oracle.mean():.2f}"
        s2 = f"s2@{lab.split('@')[1]}" if "@" in lab else None
        vs_s2 = _ci(g.delta_nll, by[s2].delta_nll) if s2 in by and s2 != lab else ""
        lines.append(f"| {lab} | {g.delta_nll.mean():+.4f} | " + (f"{g.kl_bf16.mean():.4f} | " if kl else "")
                     + f"{g.top1_agree.mean():.3f} | {rec} | "
                     f"{f'{g.fetched_pairs_per_step.mean():,.0f}' if fetch else ''} | "
                     f"{f'{g.distinct_tokens_per_layer_step.mean():,.0f}' if fetch else ''} | "
                     + " | ".join("" if lab == c else _ci(g.delta_nll, by[c].delta_nll) for c in cols)
                     + f" | {vs_s2} |"
                     + "".join(f" {'' if lab == c else _ci(g.kl_bf16, by[c].kl_bf16)} |" for c in (cols if kl else [])))
    lines.append("")

    match = [lab for lab in labels if lab.endswith("@match")]
    if match:
        lines += [f"## Equal GPU-resident memory: {base} + fetched tokens vs uniform {nxt}", "",
                  f"Each @match row fetches, per KV head and layer and step, as many tokens as fill the memory "
                  f"difference between the {base} and {nxt} histories, so its resident history equals uniform {nxt}'s. "
                  "Negative = better than uniform " + nxt + "." + (f" By > noise = more than {thr:.4f}." if thr else ""),
                  "", f"| scheme | resident ({base} + fetch) | resident {nxt} | dNLL | dNLL {nxt} | "
                  f"dNLL - {nxt} [95% CI] | median difference | beats {nxt} | by > noise | loses by > noise |"
                  + (f" KL | KL {nxt} | KL - {nxt} [95% CI] | lower KL than {nxt} |" if kl else ""),
                  "|---|---|---|---|---|---|---|---|---|---|" + ("---|---|---|---|" if kl else "")]
        for lab in match:
            g = by[lab]
            dd = (g.delta_nll - un.loc[g.index]).dropna()
            lines.append(f"| {lab} | {(g.resident_bytes / g.bf16_bytes).mean():.4f} | "
                         f"{(g[f'{nxt}_bytes'] / g.bf16_bytes).mean():.4f} | {g.delta_nll.mean():+.4f} | "
                         f"{un.loc[g.index].mean():+.4f} | {_ci(g.delta_nll, un)} | {dd.median():+.4f} | "
                         f"{int((dd < 0).sum())}/{len(dd)} | "
                         f"{f'{int((dd < -thr).sum())}/{len(dd)}' if thr else ''} | "
                         f"{f'{int((dd > thr).sum())}/{len(dd)}' if thr else ''} |"
                         + (f" {g.kl_bf16.mean():.4f} | {by[nxt].kl_bf16.loc[g.index].mean():.4f} | "
                            f"{_ci(g.kl_bf16, by[nxt].kl_bf16)} | "
                            f"{int(((g.kl_bf16 - by[nxt].kl_bf16.loc[g.index]).dropna() < 0).sum())}/{len(dd)} |"
                            if kl and by[nxt].kl_bf16.notna().any() else (" | | | |" if kl else "")))
        lines.append("")

    n = len(un)
    lines += ["## Robustness per pause", "",
              f"Mean without outlier = mean over pauses after removing the scheme's single pause farthest from its "
              f"median (named). Beats {nxt} = pauses where the scheme's dNLL is below uniform {nxt}'s"
              + (f"; by > noise = by more than the noise floor ({thr:.4f}, 2 s.d. of noise_ctrl)" if thr else "")
              + f". {n} pauses.", "",
              f"| scheme | mean | median | mean without outlier | outlier pause (dNLL) | beats {nxt} | "
              f"beats {nxt} by > noise | loses to {nxt} by > noise |", "|---|---|---|---|---|---|---|---|"]
    for lab in labels:
        x = by[lab].delta_nll
        far = (x - x.median()).abs().idxmax()
        dd = (x - un.loc[x.index]).dropna()
        same = lab == nxt
        lines.append(f"| {lab} | {x.mean():+.4f} | {x.median():+.4f} | {x.drop(far).mean():+.4f} | "
                     f"{far[0]} t{far[1]} ({x.loc[far]:+.4f}) | "
                     f"{'' if same else f'{int((dd < 0).sum())}/{len(dd)}'} | "
                     f"{'' if same or not thr else f'{int((dd < -thr).sum())}/{len(dd)}'} | "
                     f"{'' if same or not thr else f'{int((dd > thr).sum())}/{len(dd)}'} |")
    lines.append("")

    if pf_k and "fetch_all_noprefill" in by:
        fa, fn = by["fetch_all"], by["fetch_all_noprefill"]
        pfb = fa.prefill_moved_bytes
        lines += ["## Prefill fetch", "",
                  f"Tool output prefilled with a top-{pf_k} fetch per chunk: fetch_all {fa.delta_nll.mean():+.4f} vs "
                  f"fetch_all_noprefill {fn.delta_nll.mean():+.4f}; difference {_ci(fa.delta_nll, fn.delta_nll)} "
                  f"(negative = the prefill fetch lifts the ceiling). Prefill fetch moves {pfb.mean() / MB:,.1f} MB per "
                  f"turn on average ({(pfb / fa.bf16_bytes).mean():.3f} of the bf16 history).", ""]

    b = by[base]
    extra = (b[f"{nxt}_bytes"] - b.base_bytes) if "base_bytes" in b else (b.int3_bytes - b.int2_bytes)
    lines += ["## Bytes", "",
              f"Per pause on average: bf16 history {b.bf16_bytes.mean() / MB:,.0f} MB; resident {base} "
              f"{(b[f'{base}_bytes'] / b.bf16_bytes).mean():.3f} and {nxt} {(b[f'{nxt}_bytes'] / b.bf16_bytes).mean():.3f} "
              f"of bf16; uniform {nxt}'s one-time extra reload over {base}: {extra.mean() / MB:,.0f} MB "
              f"({(extra / b.bf16_bytes).mean():.3f} of bf16).", "",
              "Moved per turn = bytes fetched at every step, summed over the turn's steps (no reuse across steps). "
              "Unique per turn = distinct (layer, head, token) pairs fetched over the turn (what a fetch cache would "
              f"move). Both are shown as MB and as a multiple of {nxt}'s one-time extra reload on the same pause. "
              f"Resident = {base} history + one step's fetched set, as a fraction of bf16, next to resident {nxt}. "
              "spec also decodes one speculative token per step (compute, not bytes)."
              + (" Pool rows move only their misses. Total = prefill fetch + per-step bytes (moved; unique for non-pool "
               "rows), as MB and as a multiple of the same one-time extra reload." if pf_k else ""), "",
              f"| scheme | moved / turn (MB) | x {nxt} extra | unique / turn (MB) | x {nxt} extra | resident | "
              f"resident {nxt} |" + (" total moved / turn (MB) | x extra | total unique / turn (MB) | x extra |"
                                      if pf_k else ""),
              "|---|---|---|---|---|---|---|" + ("---|---|---|---|" if pf_k else "")]
    for lab in labels:
        g = by[lab]
        if g.policy.iloc[0] == "none" or lab in CONTROLS:
            continue
        e = _extra(g, nxt)
        row = (f"| {lab} | {g.moved_bytes_turn.mean() / MB:,.1f} | {(g.moved_bytes_turn / e).mean():.2f} | "
               f"{g.unique_bytes_turn.mean() / MB:,.1f} | {(g.unique_bytes_turn / e).mean():.2f} | "
               f"{(g.resident_bytes / g.bf16_bytes).mean():.4f} | {(_next_bytes(g, nxt) / g.bf16_bytes).mean():.4f} |")
        if pf_k:
            tm = g.prefill_moved_bytes + g.moved_bytes_turn
            tu = g.prefill_moved_bytes + g.unique_bytes_turn
            pool = g.policy.iloc[0] in ("s2_pool", "spec_chain_pool")
            row += (f" {tm.mean() / MB:,.1f} | {(tm / e).mean():.2f} | "
                    + ("" if pool else f"{tu.mean() / MB:,.1f}") + " | " + ("" if pool else f"{(tu / e).mean():.2f}") + " |")
        lines.append(row)
    lines.append("")

    lines += _study_sections(by, labels, kl, thr, nxt)
    lines += _rules_section(by, s, thr, nxt)

    out = d / "summary.md"
    text = "\n".join(lines) + "\n"
    out.write_text(text)
    return text + f"wrote {out}"


def _next_bytes(g, nxt):
    """Bytes of the uniform reference a row's equal-memory budget refers to (per row's own base if recorded)."""
    return g.next_bytes if "next_bytes" in g and g.next_bytes.notna().all() else g[f"{nxt}_bytes"]


def _extra(g, nxt):
    """Uniform reference's one-time extra reload over the row's own base, per pause."""
    if "hist_bytes" in g and "next_bytes" in g and g.next_bytes.notna().all():
        return g.next_bytes - g.hist_bytes
    return (g[f"{nxt}_bytes"] - g.base_bytes) if "base_bytes" in g else (g.int3_bytes - g.int2_bytes)


def _paired(a, b):
    """(mean, lo, up) of the paired difference a - b over shared pauses, or None."""
    idx = a.index.intersection(b.index)
    d = (a.loc[idx] - b.loc[idx]).dropna().to_numpy()
    if not len(d):
        return None
    lo, up = paired_ci(d)
    return float(d.mean()), float(lo), float(up)


def _fmt(t):
    return "" if t is None else f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"


def _k(lab):
    return lab.split("@")[1] if "@" in lab else ""


def _per_head(g):
    """Mean fetched tokens per (layer, KV head) per step of a row (needs its k for the conversion: pairs / (kvh L))."""
    return g.fetched_pairs_per_step


def _study_sections(by, labels, kl, thr, nxt):
    """Run-5 tables: multi-guess, unions vs the closest s2, pools, scoring-key gap."""
    out = []
    kl_ci = lambda a, b: _ci(a.kl_bf16, b.kl_bf16) if kl else ""
    # pairs per (layer, KV head): from any s2 row with an integer k
    s2k = {int(_k(l)): by[l] for l in labels if l.startswith("s2@") and _k(l).isdigit()}
    per_hl = None
    for k_, g in s2k.items():
        if k_ > 0:
            per_hl = float(g.fetched_pairs_per_step.mean()) / k_
            break
    multi = [l for l in labels if l.split("@")[0].rstrip("0123456789").endswith(("_union", "_wtd"))]
    if multi:
        out += ["## Multi-guess speculation", "",
                "Candidates = the top-m tokens of the distribution the single guess is the argmax of (anchored spec_*: "
                "the row's own prediction given the real previous token; spec_chain_*: the chained shadow's output). "
                "Guess = candidate 0 right; cover = the token stood for is among the m candidates. Fetched per head = "
                "mean tokens fetched per (layer, KV head) per step (union rows: up to m x k). vs s2 = same k.", "",
                "| scheme | guess right | cover | recall | fetched / head | dNLL | KL | dNLL - s2 same k [95% CI] | "
                "KL - s2 same k [95% CI] |", "|---|---|---|---|---|---|---|---|---|"]
        for lab in multi + [l for l in labels if l.split("@")[0] in ("spec", "spec_chain") and _k(l) in
                            {_k(x) for x in multi}]:
            g = by[lab]
            s2 = f"s2@{_k(lab)}"
            ph = (g.fetched_pairs_per_step.mean() / per_hl) if per_hl else float("nan")
            cov = g.spec_cand_cover.mean() if "spec_cand_cover" in g and g.spec_cand_cover.notna().any() else float("nan")
            out.append(f"| {lab} | {g.spec_guess_acc.mean():.1%} | {cov:.1%} | {g.recall_vs_oracle.mean():.2f} | "
                       f"{ph:,.0f} | {g.delta_nll.mean():+.4f} | " + (f"{g.kl_bf16.mean():.4f}" if kl else "") + " | "
                       + (_ci(g.delta_nll, by[s2].delta_nll) if s2 in by else "") + " | "
                       + (kl_ci(g, by[s2]) if s2 in by else "") + " |")
        out.append("")
    unions = [l for l in multi if "_union" in l]
    if unions and per_hl and s2k:
        out += ["## Union rows (diagnostic) vs the s2 row closest to their fetched count", "",
                "Union rows fetch up to m x k tokens, so they are compared with the s2 row whose k is closest to the "
                "union's mean fetched tokens per (layer, KV head) per step (no rule uses these rows).", "",
                "| scheme | fetched / head | closest s2 | dNLL | dNLL s2 | dNLL - s2 [95% CI] | KL - s2 [95% CI] |",
                "|---|---|---|---|---|---|---|"]
        for lab in unions:
            g = by[lab]
            ph = float(g.fetched_pairs_per_step.mean()) / per_hl
            k_ = min(s2k, key=lambda x: abs(x - ph))
            ref = s2k[k_]
            out.append(f"| {lab} | {ph:,.0f} | s2@{k_} | {g.delta_nll.mean():+.4f} | {ref.delta_nll.mean():+.4f} | "
                       f"{_ci(g.delta_nll, ref.delta_nll)} | {kl_ci(g, ref)} |")
        out.append("")
    pools = [l for l in labels if "_pool_" in l]
    if pools:
        out += ["## Fetch pools", "",
                "A bf16 pool of M tokens per (layer, KV head); each step the top-k by the row's score is fetched if not "
                "already pooled (misses), the least recently selected tokens are evicted down to M, and attention uses "
                f"the whole pool. Moved = misses only. x extra = per-step moved bytes / {nxt}'s one-time extra reload "
                "(mean over pauses; prefill fetch excluded).", "",
                f"| scheme | M (mean) | hit rate | moved / turn (MB) | x extra | resident | dNLL | dNLL - s2@match "
                f"[95% CI] | dNLL - {nxt} [95% CI] | KL |", "|---|---|---|---|---|---|---|---|---|---|"]
        for lab in pools:
            g = by[lab]
            e = _extra(g, nxt)
            out.append(f"| {lab} | {g.pool_M.mean():,.0f} | {g.pool_hit_rate.mean():.3f} | "
                       f"{g.moved_bytes_turn.mean() / MB:,.1f} | {(g.moved_bytes_turn / e).mean():.2f} | "
                       f"{(g.resident_bytes / g.bf16_bytes).mean():.4f} | {g.delta_nll.mean():+.4f} | "
                       + (_ci(g.delta_nll, by["s2@match"].delta_nll) if "s2@match" in by else "") + " | "
                       + (_ci(g.delta_nll, by[nxt].delta_nll) if nxt in by else "") + " | "
                       + (f"{g.kl_bf16.mean():.4f}" if kl else "") + " |")
        out.append("")
    gaps = [(k_, l) for l in labels for k_ in [_k(l)] if l.split("@")[0] in ("s2", "s2_k2score", "s2_int1base")
            and (f"oracle@{k_}" in by or f"oracle_int1base@{k_}" in by)]
    if any(l.startswith(("s2_k2score", "s2_int1base")) for _, l in gaps):
        out += ["## Scoring keys: gap to the oracle at the same k", "",
                "dNLL(selector) - dNLL(oracle) at the same k and base (int1-base rows against oracle_int1base).", "",
                "| selector | oracle | gap (mean) | gap [95% CI] | recall |", "|---|---|---|---|---|"]
        for k_, lab in gaps:
            o = f"oracle_int1base@{k_}" if lab.startswith("s2_int1base") else f"oracle@{k_}"
            if o not in by:
                continue
            d = _paired(by[lab].delta_nll, by[o].delta_nll)
            out.append(f"| {lab} | {o} | {d[0]:+.4f} | {_fmt(d)} | {by[lab].recall_vs_oracle.mean():.2f} |")
        out.append("")
    return out


def _role(m, s):
    """Run-5 role of a folder: (row_set, 'real' | 'self-T<T>', base)."""
    rs = m["row_set"].iloc[0] if "row_set" in m else "default"
    src = "real" if "target_source" not in s or (s.target_source != "self").all() else f"self-T{float(s.temperature.iloc[0]):g}"
    if "base" in m and isinstance(m["base"].iloc[0], str):
        base = m["base"].iloc[0]
    else:
        base = f"int{int(m.base_bits.iloc[0])}" if "base_bits" in m else "int2"
    return rs, src, base


def _byte_ratio(g, nxt):
    return float((g.moved_bytes_turn / _extra(g, nxt)).mean())


def rule_parts(by, thr, nxt, role):
    """The rule parts one job decides: list of (rule, job tag, passed, detail lines)."""
    rs, src, base = role
    out = []
    need = lambda *ls: all(l in by for l in ls)
    if rs == "a1" and need("spec_chain_wtd3@128", "spec_chain@128", "s2@128"):
        w, c, s2 = (by[l].delta_nll for l in ("spec_chain_wtd3@128", "spec_chain@128", "s2@128"))
        d = _paired(w, c)
        gap, closed = c.mean() - s2.mean(), c.mean() - w.mean()
        a, b = d[2] < 0, gap > 0 and closed >= 0.5 * gap
        out.append(("R1", src, bool(a and b),
                    [f"spec_chain_wtd3@128 - spec_chain@128 = {_fmt(d)}: CI excludes 0 below -> {'yes' if a else 'no'}",
                     f"spec_chain-to-s2 gap at 128 = {gap:+.4f}, closed by wtd3 = {closed:+.4f} "
                     f"({closed / gap:.0%} of the gap) -> >= half: {'yes' if b else 'no'}" if gap > 0 else
                     f"spec_chain-to-s2 gap at 128 = {gap:+.4f} (not > 0) -> half-gap part fails"]))
    if rs == "a2" and need("s2@match") and thr is not None:
        parts, ok = [], False
        for k_ in ("64", "128"):
            lab = f"s2_pool_Mmatch@{k_}"
            if lab not in by:
                continue
            d = _paired(by[lab].delta_nll, by["s2@match"].delta_nll)
            br = _byte_ratio(by[lab], nxt)
            pa, pb = d[2] < thr, br <= 2
            ok |= pa and pb
            parts.append(f"{lab} - s2@match = {_fmt(d)}: CI upper {d[2]:+.4f} < 2 s.d. noise {thr:.4f} -> "
                         f"{'yes' if pa else 'no'}; moved {br:.2f}x int2's extra reload (<= 2: {'yes' if pb else 'no'})")
        if parts:
            out.append(("R3", src, bool(ok), parts))
    if rs == "a2" and need("s2_pool_Mhalf@64", nxt):
        g = by["s2_pool_Mhalf@64"]
        d = _paired(g.delta_nll, by[nxt].delta_nll)
        br = _byte_ratio(g, nxt)
        pa, pb = d[2] < 0.01, br <= 2
        out.append(("R4", src, bool(pa and pb),
                    [f"s2_pool_Mhalf@64 - {nxt} = {_fmt(d)}: CI upper < +0.01 -> {'yes' if pa else 'no'}; resident "
                     f"{(g.resident_bytes / g.bf16_bytes).mean():.4f} of bf16; moved {br:.2f}x {nxt}'s extra reload "
                     f"(<= 2: {'yes' if pb else 'no'})"]))
    if rs == "c" and need("s2@16", "oracle@16", "s2_int1base@16", "oracle_int1base@16", "s2@match", "s2_int1base@match"):
        gk = by["s2@16"].delta_nll.mean() - by["oracle@16"].delta_nll.mean()
        g1 = by["s2_int1base@16"].delta_nll.mean() - by["oracle_int1base@16"].delta_nll.mean()
        a = g1 > 0 and gk <= 0.5 * g1
        d = _paired(by["s2@match"].delta_nll, by["s2_int1base@match"].delta_nll)
        b = d[2] < 0
        out.append(("R2", src, bool(a and b),
                    [f"s2-to-oracle gap at 16: k2v1 {gk:+.4f}, int1 {g1:+.4f} -> k2v1 <= 0.5 x int1: {'yes' if a else 'no'}",
                     f"k2v1 s2@match - int1 s2@match = {_fmt(d)} (resident "
                     f"{(by['s2@match'].resident_bytes / by['s2@match'].bf16_bytes).mean():.4f} vs "
                     f"{(by['s2_int1base@match'].resident_bytes / by['s2_int1base@match'].bf16_bytes).mean():.4f} of bf16): "
                     f"CI below 0 -> {'yes' if b else 'no'}"]))
    return out


def _rules_section(by, s, thr, nxt):
    m0 = next(iter(by.values()))
    role = _role(m0.reset_index(), s)
    if role[0] not in ("a1", "a2", "c"):
        return []
    lines = ["## Pre-registered decision rules (run 5)", ""] + [f"- {t}" for t in RULES_TEXT] + [""] + \
            [f"- {t}" for t in RULES_HOW] + ["", f"### Parts decided by this job ({role[0]}, {role[1]} targets, base {role[2]})", ""]
    parts = rule_parts(by, thr, nxt, role)
    for rule, tag, ok, det in parts:
        lines.append(f"- **{rule} ({tag}): {'PASS' if ok else 'FAIL'}**")
        lines += [f"  - {x}" for x in det]
    if not parts:
        lines.append("- (none of R1-R4 is decided by this job's rows)")
    lines += ["", "The overall verdicts need every job: `python -m experiments.gate.summarize_fetch --rules <run-5 "
              "folders>` (rules_run5.md).", ""]
    return lines


def _load_by(d):
    m, s = pd.read_csv(d / "metrics.csv"), pd.read_csv(d / "samples.csv")
    role = _role(m, s)
    if "k_label" not in m:
        m["k_label"] = m.k.astype(str)
    m["label"] = m.scheme + np.where(m.k_label.astype(str) != "0", "@" + m.k_label.astype(str), "")
    m = m.set_index(KEY)
    by = {lab: g for lab, g in m.groupby("label")}
    thr = 2 * by["noise_ctrl"].delta_nll.std(ddof=1) if "noise_ctrl" in by else None
    return by, s, thr, role


def rules_report(dirs, out: Path) -> str:
    """Overall R1-R4 verdicts over run-5 folders (roles from each folder's row set, targets and base)."""
    parts, used = [], []
    for d in dirs:
        by, s, thr, role = _load_by(d)
        base = role[2]
        nxt = NEXT.get(base, "int2")
        ps = rule_parts(by, thr, nxt, role)
        parts += [(r, tag, ok, det, d.name) for r, tag, ok, det in ps]
        used.append(f"- {d.name}: row set {role[0]}, {role[1]} targets, base {base}, {len(s)} pauses"
                    + (f", run id {(d / 'run_id.txt').read_text().strip()}" if (d / "run_id.txt").exists() else ""))
    need = {"R1": ["real", "self-T0.7"], "R2": ["real"], "R3": ["real"], "R4": ["real", "self-T0.7"]}
    lines = ["# Fetch study run 5: pre-registered decision rules", "", "## Rules (verbatim)", ""] + \
            [f"- {t}" for t in RULES_TEXT] + ["", "## How they are tested", ""] + [f"- {t}" for t in RULES_HOW] + \
            ["", "## Jobs", ""] + used + ["", "## Verdicts", ""]
    for rule, tags in need.items():
        got = {tag: (ok, det, name) for r, tag, ok, det, name in parts if r == rule}
        missing = [t for t in tags if t not in got]
        if missing:
            verdict = f"INCOMPLETE (missing: {', '.join(missing)})"
        else:
            verdict = "PASS" if all(got[t][0] for t in tags) else "FAIL"
        lines.append(f"- **{rule}: {verdict}**")
        for tag, (ok, det, name) in got.items():
            info = "" if tag in tags else " (for information; not part of the verdict)"
            lines.append(f"  - {name} ({tag}): {'PASS' if ok else 'FAIL'}{info}")
            lines += [f"    - {x}" for x in det]
    text = "\n".join(lines) + "\n"
    out.write_text(text)
    return text + f"wrote {out}"


if __name__ == "__main__":
    main()
