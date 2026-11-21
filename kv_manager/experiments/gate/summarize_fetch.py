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

    python -m experiments.gate.summarize_fetch --dir results/gate_fetch/llama-3.1-8b-instruct-base2
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.gate.plot_gate import paired_ci

ROOT = Path(__file__).resolve().parents[2]
KEY = ["instance_id", "turn"]
ORDER = ["bf16", "noise_ctrl", "int1", "int2", "int3", "fetch_all", "fetch_all_noprefill", "s1", "sens_a",
         "sens_b", "spec_chain", "spec", "s2", "s2sens_a", "s2sens_b", "oracle", "s1_layer"]
CONTROLS = ("bf16", "noise_ctrl", "int1", "int2", "int3", "fetch_all", "fetch_all_noprefill")
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
    args = p.parse_args()
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
    base, nxt = f"int{bits}", f"int{bits + 1}"
    if "k_label" not in m:
        m["k_label"] = m.k.astype(str)
    m["label"] = m.scheme + np.where(m.k_label.astype(str) != "0", "@" + m.k_label.astype(str), "")
    m = m.set_index(KEY)
    by = {lab: g for lab, g in m.groupby("label")}
    labels = sorted(by, key=_order)
    refs = [r for r in ("int2", "int3") if r != base and r in by]
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
             + "; uniform rows over their own precision, the bf16 row over bf16.", "",
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
              + (" Prefill-fetch bytes are not included here (see Prefill fetch)." if pf_k else ""), "",
              f"| scheme | moved / turn (MB) | x {nxt} extra | unique / turn (MB) | x {nxt} extra | resident | "
              f"resident {nxt} |", "|---|---|---|---|---|---|---|"]
    for lab in labels:
        g = by[lab]
        if g.policy.iloc[0] == "none" or lab in CONTROLS:
            continue
        e = (g[f"{nxt}_bytes"] - g.base_bytes) if "base_bytes" in g else (g.int3_bytes - g.int2_bytes)
        lines.append(f"| {lab} | {g.moved_bytes_turn.mean() / MB:,.1f} | {(g.moved_bytes_turn / e).mean():.2f} | "
                     f"{g.unique_bytes_turn.mean() / MB:,.1f} | {(g.unique_bytes_turn / e).mean():.2f} | "
                     f"{(g.resident_bytes / g.bf16_bytes).mean():.4f} | {(g[f'{nxt}_bytes'] / g.bf16_bytes).mean():.4f} |")
    lines.append("")

    out = d / "summary.md"
    text = "\n".join(lines) + "\n"
    out.write_text(text)
    return text + f"wrote {out}"


if __name__ == "__main__":
    main()
