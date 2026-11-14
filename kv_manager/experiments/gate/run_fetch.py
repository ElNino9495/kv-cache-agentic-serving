"""Per-step fetch study at a 2-bit base (offline), on the first gate run's pauses.

For every pause: reload the history at int2 (block 16; block 0, the recent 64 tokens and the
trailing partial block in bf16), prefill the tool output over it without fetching, then teacher-force
the real next message one token at a time (first --max-steps tokens). At every step and layer, each
scheme swaps its own top-k history tokens (per KV head) to bf16 for that step only. See fetch_attn.py
for the schemes and how attention is computed.

Rows (one batch, all on the same pause):
  bf16       reference: bf16 history, bf16 tool-output prefill (dNLL and top-1 are measured against it)
  int2/int3  uniform, no fetch (tool output prefilled over the same precision)
  fetch_all  int2 tool-output prefill, then every middle token fetched at every step: the ceiling for
             any per-step fetch, given that the tool output was prefilled over int2
  noise_ctrl bf16 tool-output prefill + every middle token fetched: equal to the bf16 row in exact
             arithmetic, so its dNLL measures the bf16 rounding noise of the per-step path (noise floor)
  s1, s2, spec, sens_a, sens_b, s2sens_a, s2sens_b, oracle   per KV head, k in --ks
  s1_layer   per layer (ablation)
  spec_shadow  the speculative token decoded alongside each spec row (not a scheme; not written out,
               its guess accuracy is reported on the spec row)

Every pause also re-runs the bf16 and int2 rows with sdpa teacher forcing (run_gate.run_turn) over the
same tokens and records the NLL difference and the largest per-token log-prob difference (samples.csv,
check_* columns). --check-nll-tol stops the run if either row's NLL differs by more than that; the
per-token log-prob difference is only reported (bf16 rounding makes single tokens move by ~0.2).

--diagnose [INSTANCE:TURN] runs one pause (default pydata__xarray-4687:85) with s1, s2 and oracle at
k = 16, 64, 128 (plus bf16 and int2) and logs, for every step and layer, which tokens each scheme fetched
and the attention mass on them (diag_<instance>_t<turn>/: steps.csv, fetch.csv, fetch_idx.pt, report.md).

Outputs (in --out-dir, default results/gate_fetch/<model>/): metrics.csv (pause x row), samples.csv.

    cd kv_manager
    CUDA_VISIBLE_DEVICES=0 python -m experiments.gate.run_fetch --max-samples 1 --check-nll-tol 0.05  # smoke
    CUDA_VISIBLE_DEVICES=0 python -m experiments.gate.run_fetch
    CUDA_VISIBLE_DEVICES=0 python -m experiments.gate.run_fetch --diagnose
    python -m experiments.gate.summarize_fetch
"""

import argparse
import csv
from pathlib import Path

import torch

from experiments.gate import data
from experiments.gate.fetch_attn import default_rows, diagnostic_rows, run_pause
from experiments.gate.run_gate import ROOT, _append, load_model, model_slug, read_pauses

DIAG_DEFAULT = "pydata__xarray-4687:85"


def _label(r):
    return r.name + (f"@{r.k}" if r.k else "")


def write_diagnostics(out: Path, tok, rows, s, res) -> None:
    """steps.csv (per row and step: log-probs), fetch.csv (per row, step, layer, KV head: masses),
    fetch_idx.pt (the fetched indices), report.md (where each scheme loses against int2, and why)."""
    out.mkdir(parents=True, exist_ok=True)
    names = [_label(r) for r in rows]
    ref, i2 = names.index("bf16"), names.index("int2")
    T = res["T"]
    tgt = s["target_ids"][:T]
    lp = res["logp"]
    with open(out / "steps.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "step", "token_id", "token", "logp", "logp_bf16", "logp_int2",
                    "dlogp_vs_bf16", "dlogp_vs_int2"])
        for i, n in enumerate(names):
            for j in range(T):
                w.writerow([n, j, tgt[j], tok.decode([tgt[j]]), f"{lp[i, j]:.5f}", f"{lp[ref, j]:.5f}",
                            f"{lp[i2, j]:.5f}", f"{lp[i, j] - lp[ref, j]:+.5f}", f"{lp[i, j] - lp[i2, j]:+.5f}"])
    recs = res["diag_records"]
    fields = ["mass_actual", "mass_int2", "mass_bf16", "oracle_mass_bf16", "mid_mass_bf16", "mid_mass_int2"]
    # record t = step + 1 (t = 0 is the warm-up, which never fetches)
    with open(out / "fetch.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "step", "layer", "kv_head", "k"] + fields + ["overlap_with_oracle"])
        for r in recs:
            for h in range(r["idx"].shape[0]):
                w.writerow([names[r["row"]], r["t"] - 1, r["layer"], h, r["idx"].shape[1]]
                           + [f"{float(r[x][h]):.5f}" for x in fields] + [int(r["overlap"][h])])
    torch.save({(names[r["row"]], r["t"] - 1, r["layer"]): r["idx"] for r in recs}, out / "fetch_idx.pt")

    # report: per scheme, dNLL and the steps where it loses most against int2, with fetch coverage there
    agg = {}
    for r in recs:
        a = agg.setdefault((names[r["row"]], r["t"] - 1), {x: 0.0 for x in fields + ["overlap", "k", "n"]})
        for x in fields:
            a[x] += float(r[x].mean())
        a["overlap"] += float(r["overlap"].float().mean())
        a["k"] += r["idx"].shape[1]
        a["n"] += 1
    lines = [f"# Fetch diagnostics: {s['instance_id']} turn {s['turn']}", "",
             f"History {res['P']} tokens, {T} teacher-forced steps. Masses are attention probability over the "
             "history middle (the quantized tokens), averaged over query heads, KV heads and layers: "
             "`fetched (actual)` = mass on the fetched set in the row's own attention after fetching; "
             "`fetched (bf16)` = mass the current bf16 attention puts on the fetched set; `oracle (bf16)` = the "
             "same for the best set of that size; `middle int2 / bf16` = total mass on the middle.", "",
             "| scheme | dNLL vs bf16 | dNLL vs int2 | steps worse than int2 by > 0.5 nats |", "|---|---|---|---|"]
    for i, n in enumerate(names):
        d_ref = float((lp[ref] - lp[i]).mean())
        d_i2 = float((lp[i2] - lp[i]).mean())
        bad = int(((lp[i2] - lp[i]) > 0.5).sum())
        lines.append(f"| {n} | {d_ref:+.4f} | {d_i2:+.4f} | {bad} |")
    lines.append("")
    for i, n in enumerate(names):
        if rows[i].policy == "none":
            continue
        loss = (lp[i2] - lp[i])
        worst = loss.topk(min(5, T)).indices.tolist()
        lines += [f"## {n}: steps with the largest loss against int2", "",
                  "| step | token | log p (row / int2 / bf16) | fetched (actual) | fetched (bf16) | oracle (bf16) | "
                  "overlap / k | middle int2 / bf16 |", "|---|---|---|---|---|---|---|---|"]
        for j in worst:
            a = agg.get((n, j))
            if a is None:
                continue
            m = lambda x: a[x] / a["n"]
            lines.append(f"| {j} | {tok.decode([tgt[j]])!r} | {lp[i, j]:.2f} / {lp[i2, j]:.2f} / {lp[ref, j]:.2f} | "
                         f"{m('mass_actual'):.3f} | {m('mass_bf16'):.3f} | {m('oracle_mass_bf16'):.3f} | "
                         f"{a['overlap'] / a['n']:.1f} / {a['k'] / a['n']:.0f} | "
                         f"{m('mid_mass_int2'):.3f} / {m('mid_mass_bf16'):.3f} |")
        lines.append("")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:len(names) + 7]))
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", default="unsloth/Llama-3.1-8B-Instruct")
    p.add_argument("--samples-from", default=str(ROOT / "results" / "gate" / "samples.csv"))
    p.add_argument("--ks", nargs="*", type=int, default=[16, 32, 64, 128])
    p.add_argument("--max-steps", type=int, default=64, help="target tokens decoded per pause")
    p.add_argument("--block", type=int, default=16)
    p.add_argument("--recent", type=int, default=64)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--max-context", type=int, default=None,
                   help="skip pauses above this many tokens (default: the model's native context)")
    p.add_argument("--check-nll-tol", type=float, default=0.0,
                   help="stop if the bf16 or int2 row's NLL differs between per-step and sdpa teacher forcing "
                        "by more than this (0 = only record)")
    p.add_argument("--diagnose", nargs="?", const=DIAG_DEFAULT, default=None, metavar="INSTANCE:TURN",
                   help=f"diagnostic mode on one pause (default {DIAG_DEFAULT})")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--out-dir", default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir or ROOT / "results" / "gate_fetch" / model_slug(args.target))
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path, samples_path = out_dir / "metrics.csv", out_dir / "samples.csv"
    diag = None
    if args.diagnose:
        inst, turn = args.diagnose.rsplit(":", 1)
        diag = (inst, int(turn))
    done = set()
    if diag is None:
        if args.resume and samples_path.exists():
            with open(samples_path) as f:
                done = {(r["instance_id"], int(r["turn"])) for r in csv.DictReader(f)}
        else:
            metrics_path.unlink(missing_ok=True)
            samples_path.unlink(missing_ok=True)

    dev = args.device
    print(f"loading {args.target}", flush=True)
    tok, model = load_model(args.target, dev)
    if args.dtype == "float32":
        model = model.float()
    fmt = data.chat_format(model.config.model_type)
    max_ctx = fmt.native_context if args.max_context is None else args.max_context
    rows = diagnostic_rows() if diag else default_rows(args.ks)
    print(f"{len(rows)} rows: {', '.join(sorted({r.name for r in rows}))}"
          + ("" if diag else f"; ks {args.ks}"), flush=True)

    pauses = {diag[0]: [diag[1]]} if diag else read_pauses(Path(args.samples_from))
    n_done = 0
    for instance_id, turns in pauses.items():
        for s in data.make_samples(tok, data.load(instance_id), [], 10 ** 9, fmt=fmt, turns=turns):
            if args.max_samples and n_done >= args.max_samples:
                return
            if (instance_id, s["turn"]) in done:
                continue
            total = len(s["hist_ids"]) + len(s["fresh_ids"]) + min(args.max_steps, len(s["target_ids"]))
            if max_ctx and total > max_ctx:
                print(f"{instance_id} turn {s['turn']}: skipped, {total} tokens > {max_ctx}", flush=True)
                continue
            n_done += 1
            ids = lambda x: torch.tensor([x], device=dev)
            hist, fresh, tgt = ids(s["hist_ids"]), ids(s["fresh_ids"]), ids(s["target_ids"])
            t0 = torch.cuda.Event(enable_timing=True) if dev.startswith("cuda") else None
            t1 = torch.cuda.Event(enable_timing=True) if t0 else None
            if t0:
                t0.record()
            res = run_pause(model, hist, fresh, tgt, rows, block=args.block, recent=args.recent,
                            max_steps=args.max_steps, check=diag is None, diag=diag is not None)
            secs = None
            if t0:
                t1.record()
                torch.cuda.synchronize()
                secs = t0.elapsed_time(t1) / 1e3
            if diag:
                write_diagnostics(out_dir / f"diag_{instance_id}_t{s['turn']}", tok, rows, s, res)
                return

            names = [r.name for r in rows]
            ref = names.index("bf16")
            ref_nll = float(-res["logp"][ref].mean())
            out = []
            for i, r in enumerate(rows):
                if r.shadow:
                    continue
                out.append({"instance_id": instance_id, "turn": s["turn"], "hist_len": res["P"],
                            "fresh_len": res["F"], "steps": res["T"], "scheme": r.name, "policy": r.policy,
                            "k": r.k, "per_layer": int(r.per_layer),
                            "delta_nll": float(-res["logp"][i].mean()) - ref_nll,
                            "top1_agree": float((res["pred"][i] == res["pred"][ref]).float().mean()),
                            **res["rows"][i], "bf16_bytes": res["bf16_bytes"],
                            "int2_bytes": res["int2_bytes"], "int3_bytes": res["int3_bytes"]})
            _append(metrics_path, out)
            chk = {k: v for k, v in res.items() if k.startswith("check_")}
            nll_diff = max(abs(chk[f"check_{n}_nll_perstep"] - chk[f"check_{n}_nll_batched"])
                           for n in ("bf16", "int2"))
            chk["check_max_nll_diff"] = nll_diff
            _append(samples_path, [{"instance_id": instance_id, "turn": s["turn"], "hist_len": res["P"],
                                    "fresh_len": res["F"], "steps": res["T"], "ref_nll": ref_nll,
                                    "seconds": secs, "model": args.target, **chk}])
            by = {o["scheme"] + (f"@{o['k']}" if o["k"] else ""): o["delta_nll"] for o in out}
            print(f"{instance_id} turn {s['turn']}: hist {res['P']}, {res['T']} steps"
                  + (f", {secs:.1f} s" if secs else "")
                  + f" | int2 {by['int2']:+.4f} int3 {by['int3']:+.4f} fetch_all {by['fetch_all']:+.4f}"
                  + " | " + " ".join(f"{n}@64 {by[f'{n}@64']:+.4f}"
                                     for n in ("s1", "s2", "spec", "s2sens_a", "s2sens_b", "oracle")
                                     if f"{n}@64" in by)
                  + f" | check NLL diff {nll_diff:.2e}, max|dlogp| bf16 "
                  f"{chk.get('check_bf16_max_abs_logp', float('nan')):.2e} int2 "
                  f"{chk.get('check_int2_max_abs_logp', float('nan')):.2e}", flush=True)
            if args.check_nll_tol and nll_diff > args.check_nll_tol:
                raise SystemExit(f"per-step vs sdpa NLL check failed: {nll_diff:.3e} > {args.check_nll_tol}")
            del hist, fresh, tgt, res
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
