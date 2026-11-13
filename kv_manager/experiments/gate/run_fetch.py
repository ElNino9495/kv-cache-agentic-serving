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
  s1, s2, sens_a, sens_b, oracle  per KV head, k in --ks;  s1_layer  per layer (ablation)

Every pause also re-runs the bf16 and int2 rows with sdpa teacher forcing (run_gate.run_turn) over the
same tokens and records the largest per-token log-prob difference (samples.csv, check_* columns);
--check-tol makes the run stop if it is exceeded.

Outputs (in --out-dir, default results/gate_fetch/<model>/): metrics.csv (pause x row), samples.csv.

    cd kv_manager
    CUDA_VISIBLE_DEVICES=0 python -m experiments.gate.run_fetch --max-samples 1 --check-tol 0.05  # smoke
    CUDA_VISIBLE_DEVICES=0 python -m experiments.gate.run_fetch
    python -m experiments.gate.summarize_fetch
"""

import argparse
import csv
from pathlib import Path

import torch

from experiments.gate import data
from experiments.gate.fetch_attn import default_rows, run_pause
from experiments.gate.run_gate import ROOT, _append, load_model, model_slug, read_pauses


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
    p.add_argument("--check-tol", type=float, default=0.0,
                   help="stop if per-step and sdpa log-probs differ by more than this (0 = only record)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--out-dir", default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir or ROOT / "results" / "gate_fetch" / model_slug(args.target))
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path, samples_path = out_dir / "metrics.csv", out_dir / "samples.csv"
    done = set()
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
    rows = default_rows(args.ks)
    print(f"{len(rows)} rows: {', '.join(sorted({r.name for r in rows}))}; ks {args.ks}", flush=True)

    pauses = read_pauses(Path(args.samples_from))
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
                            max_steps=args.max_steps, check=True)
            secs = None
            if t0:
                t1.record()
                torch.cuda.synchronize()
                secs = t0.elapsed_time(t1) / 1e3

            names = [r.name for r in rows]
            ref = names.index("bf16")
            ref_nll = float(-res["logp"][ref].mean())
            out = []
            for i, r in enumerate(rows):
                out.append({"instance_id": instance_id, "turn": s["turn"], "hist_len": res["P"],
                            "fresh_len": res["F"], "steps": res["T"], "scheme": r.name, "policy": r.policy,
                            "k": r.k, "per_layer": int(r.per_layer),
                            "delta_nll": float(-res["logp"][i].mean()) - ref_nll,
                            "top1_agree": float((res["pred"][i] == res["pred"][ref]).float().mean()),
                            **res["rows"][i], "bf16_bytes": res["bf16_bytes"],
                            "int2_bytes": res["int2_bytes"], "int3_bytes": res["int3_bytes"]})
            _append(metrics_path, out)
            chk = {k: v for k, v in res.items() if k.startswith("check_")}
            _append(samples_path, [{"instance_id": instance_id, "turn": s["turn"], "hist_len": res["P"],
                                    "fresh_len": res["F"], "steps": res["T"], "ref_nll": ref_nll,
                                    "seconds": secs, "model": args.target, **chk}])
            by = {o["scheme"] + (f"@{o['k']}" if o["k"] else ""): o["delta_nll"] for o in out}
            print(f"{instance_id} turn {s['turn']}: hist {res['P']}, {res['T']} steps"
                  + (f", {secs:.1f} s" if secs else "")
                  + f" | int2 {by['int2']:+.4f} int3 {by['int3']:+.4f} fetch_all {by['fetch_all']:+.4f}"
                  + " | " + " ".join(f"{n}@64 {by[f'{n}@64']:+.4f}" for n in ("s1", "s2", "sens_a", "sens_b", "oracle")
                                     if f"{n}@64" in by)
                  + f" | check max|dlogp| bf16 {chk.get('check_bf16_max_abs_logp', float('nan')):.2e}"
                  f" int2 {chk.get('check_int2_max_abs_logp', float('nan')):.2e}", flush=True)
            worst = max(chk.get("check_bf16_max_abs_logp", 0), chk.get("check_int2_max_abs_logp", 0))
            if args.check_tol and worst > args.check_tol:
                raise SystemExit(f"per-step vs sdpa check failed: {worst:.3e} > {args.check_tol}")
            del hist, fresh, tgt, res
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
