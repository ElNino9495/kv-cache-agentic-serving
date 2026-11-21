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
  s1, s2, spec, spec_chain, sens_a, sens_b, s2sens_a, s2sens_b, oracle   per KV head, k in --ks
  s1_layer   per layer (ablation)
  spec_shadow, spec_chain_shadow  the speculative token decoded alongside each spec / spec_chain row (not a
               scheme; not written out; its guess accuracy is reported on the spec row, every guess in
               spec_guesses.csv). spec guesses step j's input from the real previous target token (a correct
               guess gives s2's set); spec_chain is SpeCache as published (each guess from the previous guess).

Every pause also re-runs the bf16 and int2 rows with sdpa teacher forcing (run_gate.run_turn) over the
same tokens and records the NLL difference and the largest per-token log-prob difference (samples.csv,
check_* columns). --check-nll-tol stops the run if either row's NLL differs by more than that; the
per-token log-prob difference is only reported (bf16 rounding makes single tokens move by ~0.2).

--diagnose [INSTANCE:TURN] runs one pause (default pydata__xarray-4687:85) with s1, s2 and oracle at
k = 16, 64, 128 (plus bf16 and int2) and logs, for every step and layer, which tokens each scheme fetched
and the attention mass on them (diag_<instance>_t<turn>/: steps.csv, fetch.csv, fetch_idx.pt, report.md).

--base-bits 1 runs the same rows on an int1 base (SpeCache's modified 1-bit quantizer, fakequant.py) with
uniform int1, int2 and int3 rows; every "int2" above then reads "int1". --ks may include "match": k chosen
per pause so that the base history + one step's fetched set uses the GPU memory of uniform int(base + 1)
(int1 + fetch vs int2, int2 + fetch vs int3).

--prefill-fetch-k K (> 0) prefills the tool output in --prefill-chunk token chunks with the top-K history
tokens per KV head and layer (by the chunk's own queries over base keys) swapped to bf16 for each chunk.
The fetch rows and fetch_all start from that prefill; the uniform base row and fetch_all_noprefill keep the
plain base prefill, so the lift of the prefill ceiling is measured in the same run.

--target-source self --temperature T replaces the real next message (written by the agent's own LLM) with the
target model's own: for each pause the bf16 model (bf16 prefill of the history and the tool output) writes up to
--max-steps tokens, greedy at T = 0, otherwise sampled at temperature T with a generator seeded per pause
(--seed + CRC32 of "instance:turn", so a pause's target does not depend on which other pauses run). Generation
stops after the end-of-turn token (kept, as in the real targets). The targets are saved to --self-targets
(default <out-dir>/self_targets.jsonl: ids, text, temperature, seed, model) before any row runs and are reused by
every later run or --resume into the same folder (a fresh run keeps that file; settings that differ from a saved
target stop the run). All rows are then teacher-forced on that target as usual; spec and spec_chain still guess
with the compressed row's greedy prediction, so their guess accuracy reflects the sampling.

Every row also records kl_bf16: KL(bf16 row || row) of the next-token distribution, averaged over the steps
(on self-written text dNLL depends on which tokens were sampled; the KL does not).

Outputs (in --out-dir, default results/gate_fetch/<model>-base<b>[-pf<K>][-self-T<T>]/): metrics.csv (pause x
row), samples.csv, spec_guesses.csv (pause x spec row x step: guessed token, the token it stood for, match),
self_targets.jsonl (self targets only).

    cd kv_manager
    CUDA_VISIBLE_DEVICES=0 python -m experiments.gate.run_fetch --max-samples 1 --check-nll-tol 0.05  # smoke
    CUDA_VISIBLE_DEVICES=0 python -m experiments.gate.run_fetch
    CUDA_VISIBLE_DEVICES=0 python -m experiments.gate.run_fetch --diagnose
    CUDA_VISIBLE_DEVICES=0 python -m experiments.gate.run_fetch --base-bits 1
    python -m experiments.gate.summarize_fetch
"""

import argparse
import csv
import json
import zlib
from pathlib import Path

import torch

from experiments.gate import data
from experiments.gate.fetch_attn import (MATCH, default_rows, diagnostic_rows, estimate_peak_bytes,
                                        generate_self_target, run_pause)
from experiments.gate.run_gate import ROOT, _append, load_model, model_slug, read_pauses

DIAG_DEFAULT = "pydata__xarray-4687:85"
GB = 1e9


def memory_report(cfg, weight_bytes, rows, samples, args, card_bytes):
    """Print the estimated peak GPU memory (largest pause) and warn if it exceeds the card. Returns it."""
    est = {(i, s["turn"]): estimate_peak_bytes(cfg, weight_bytes, len(s["hist_ids"]), len(s["fresh_ids"]), rows,
                                               base_bits=args.base_bits, max_steps=args.max_steps)
           for i, s in samples}
    if not est:
        return 0
    worst = max(est, key=est.get)
    peak = est[worst]
    hist_len = next(len(s["hist_ids"]) for i, s in samples if (i, s["turn"]) == worst)
    msg = (f"estimated peak GPU memory {peak / GB:.1f} GB ({len(rows)} rows, base int{args.base_bits}, "
           f"largest pause {worst[0]} t{worst[1]} with {hist_len} history tokens)")
    if card_bytes:
        msg += f"; card {card_bytes / GB:.1f} GB"
    print(msg, flush=True)
    if card_bytes and peak > 0.95 * card_bytes:
        print(f"WARNING: estimated peak {peak / GB:.1f} GB exceeds 95% of the card ({card_bytes / GB:.1f} GB). "
              "Drop a uniform reference (--uniform-refs), use fewer ks, or a larger GPU.", flush=True)
    return peak


def pause_seed(seed: int, instance_id: str, turn: int) -> int:
    return (int(seed) + zlib.crc32(f"{instance_id}:{turn}".encode())) % 2 ** 31


def load_self_targets(path: Path) -> dict:
    """{(instance_id, turn): record} from a self_targets.jsonl file (empty if it does not exist)."""
    out = {}
    if Path(path).exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out[(r["instance_id"], int(r["turn"]))] = r
    return out


def append_self_target(path: Path, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def self_target(model, tok, s, instance_id, store, path, *, target, temperature, seed, max_new, eos_ids, dev):
    """The pause's saved self-written target, or generate, save and return it (list of token ids)."""
    want = {"model": target, "temperature": float(temperature), "seed": pause_seed(seed, instance_id, s["turn"]),
            "max_new": int(max_new)}
    key = (instance_id, int(s["turn"]))
    if key in store:
        rec = store[key]
        diff = {k: (rec.get(k), v) for k, v in want.items() if rec.get(k) != v}
        if diff:
            raise SystemExit(f"{path}: saved target for {instance_id} t{s['turn']} was made with other settings "
                             f"(saved, requested): {diff}. Use another --out-dir or --self-targets.")
        return list(rec["target_ids"])
    ids = lambda x: torch.tensor([x], device=dev)
    out = generate_self_target(model, ids(s["hist_ids"]), ids(s["fresh_ids"]), max_new, temperature,
                               want["seed"], eos_ids)
    rec = {"instance_id": instance_id, "turn": int(s["turn"]), **want, "n_tokens": len(out),
           "ended": bool(out and out[-1] in set(eos_ids)), "target_ids": out, "text": tok.decode(out)}
    append_self_target(path, rec)
    store[key] = rec
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def eos_ids_for(model, tok, fmt) -> list[int]:
    ids = set(tok.encode(fmt.target_end, add_special_tokens=False))
    e = getattr(model.generation_config, "eos_token_id", None)
    ids |= set(e if isinstance(e, (list, tuple)) else [e] if e is not None else [])
    return sorted(ids)


def _label(r):
    return r.name + ("@match" if r.k == MATCH else f"@{r.k}" if r.k else "")


def _ks(values):
    return [MATCH if v == "match" else int(v) for v in values]


def write_diagnostics(out: Path, tok, rows, s, res, base: str) -> None:
    """steps.csv (per row and step: log-probs), fetch.csv (per row, step, layer, KV head: masses),
    fetch_idx.pt (the fetched indices), report.md (where each scheme loses against int2, and why)."""
    out.mkdir(parents=True, exist_ok=True)
    names = [_label(r) for r in rows]
    ref, i2 = names.index("bf16"), names.index(base)
    T = res["T"]
    tgt = s["target_ids"][:T]
    lp = res["logp"]
    with open(out / "steps.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "step", "token_id", "token", "logp", "logp_bf16", f"logp_{base}",
                    "dlogp_vs_bf16", f"dlogp_vs_{base}"])
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
             f"Base {base}. History {res['P']} tokens, {T} teacher-forced steps. `int2` below means the base "
             f"({base}). Masses are attention probability over the "
             "history middle (the quantized tokens), averaged over query heads, KV heads and layers: "
             "`fetched (actual)` = mass on the fetched set in the row's own attention after fetching; "
             "`fetched (bf16)` = mass the current bf16 attention puts on the fetched set; `oracle (bf16)` = the "
             "same for the best set of that size; `middle int2 / bf16` = total mass on the middle.", "",
             f"| scheme | dNLL vs bf16 | dNLL vs {base} | steps worse than {base} by > 0.5 nats |", "|---|---|---|---|"]
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
        lines += [f"## {n}: steps with the largest loss against {base}", "",
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
    p.add_argument("--ks", nargs="*", default=["16", "32", "64", "128", "match"],
                   help='fetched tokens per KV head and layer; "match" = equal memory to uniform int(base + 1)')
    p.add_argument("--base-bits", type=int, default=2, choices=[1, 2])
    p.add_argument("--prefill-fetch-k", type=int, default=0,
                   help="fetch this many history tokens per KV head and layer per chunk during the tool-output "
                        "prefill (0 = plain prefill over the base history)")
    p.add_argument("--prefill-chunk", type=int, default=64)
    p.add_argument("--uniform-refs", nargs="*", type=int, default=[2, 3],
                   help="uniform precisions run as references besides the base (each costs one history copy; "
                        "a missing one can be joined from another run with summarize_fetch --join)")
    p.add_argument("--estimate-memory", action="store_true",
                   help="only print the estimated peak GPU memory for these settings (no weights loaded) and exit")
    p.add_argument("--max-steps", type=int, default=64, help="target tokens decoded per pause")
    p.add_argument("--target-source", default="real", choices=["real", "self"],
                   help="real = the recorded next message; self = the bf16 target model's own (see above)")
    p.add_argument("--temperature", type=float, default=0.0, help="self targets: 0 = greedy")
    p.add_argument("--seed", type=int, default=0, help="self targets: base seed (per pause: + CRC32 of the pause)")
    p.add_argument("--self-targets", default=None,
                   help="self targets file (default <out-dir>/self_targets.jsonl; reused if it exists)")
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

    self_tgt = args.target_source == "self"
    if args.temperature < 0 or (args.temperature and not self_tgt):
        p.error("--temperature needs --target-source self and must be >= 0")
    run_name = (model_slug(args.target) + f"-base{args.base_bits}"
                + (f"-pf{args.prefill_fetch_k}" if args.prefill_fetch_k else "")
                + (f"-self-T{args.temperature:g}" if self_tgt else ""))
    out_dir = Path(args.out_dir or ROOT / "results" / "gate_fetch" / run_name)
    base = f"int{args.base_bits}"
    pf = args.prefill_fetch_k > 0
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path, samples_path = out_dir / "metrics.csv", out_dir / "samples.csv"
    guesses_path = out_dir / "spec_guesses.csv"
    targets_path = Path(args.self_targets) if args.self_targets else out_dir / "self_targets.jsonl"
    targets = load_self_targets(targets_path) if self_tgt else {}
    # saved targets made with other settings stop the run here, before any earlier results are deleted
    bad = [f"{i} t{t}" for (i, t), r in targets.items()
           if (r.get("model"), r.get("temperature"), r.get("max_new"), r.get("seed"))
           != (args.target, float(args.temperature), int(args.max_steps), pause_seed(args.seed, i, t))]
    if bad:
        raise SystemExit(f"{targets_path}: {len(bad)} saved target(s) were made with other settings (model, "
                         f"temperature, seed or --max-steps), e.g. {bad[0]}. Use another --out-dir or --self-targets.")
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
            guesses_path.unlink(missing_ok=True)

    dev = args.device
    pauses = {diag[0]: [diag[1]]} if diag else read_pauses(Path(args.samples_from))
    rows_for = lambda: (diagnostic_rows(base_bits=args.base_bits, prefill=pf) if diag
                        else default_rows(_ks(args.ks), base_bits=args.base_bits, prefill=pf,
                                          uniform_refs=args.uniform_refs))
    card = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0

    def load_samples(tok, fmt, max_ctx):
        out = []
        for instance_id, turns in pauses.items():
            for s in data.make_samples(tok, data.load(instance_id), [], 10 ** 9, fmt=fmt, turns=turns):
                total = len(s["hist_ids"]) + len(s["fresh_ids"]) + min(args.max_steps, len(s["target_ids"]))
                if max_ctx and total > max_ctx:
                    print(f"{instance_id} turn {s['turn']}: skipped, {total} tokens > {max_ctx}", flush=True)
                    continue
                out.append((instance_id, s))
        return out[:args.max_samples] if args.max_samples else out

    if args.estimate_memory:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        cfg = AutoConfig.from_pretrained(args.target)
        with torch.device("meta"):
            meta = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
        weight_bytes = sum(t.numel() * t.element_size() for t in meta.parameters())
        fmt = data.chat_format(cfg.model_type)
        max_ctx = fmt.native_context if args.max_context is None else args.max_context
        samples = load_samples(AutoTokenizer.from_pretrained(args.target), fmt, max_ctx)
        memory_report(cfg, weight_bytes, rows_for(), samples, args, card)
        return

    print(f"loading {args.target}", flush=True)
    tok, model = load_model(args.target, dev)
    if args.dtype == "float32":
        model = model.float()
    fmt = data.chat_format(model.config.model_type)
    max_ctx = fmt.native_context if args.max_context is None else args.max_context
    rows = rows_for()
    print(f"{len(rows)} rows on base {base}{f', prefill fetch k={args.prefill_fetch_k}' if pf else ''}: "
          f"{', '.join(sorted({r.name for r in rows}))}" + ("" if diag else f"; ks {args.ks}"), flush=True)
    samples = load_samples(tok, fmt, max_ctx)
    weight_bytes = sum(t.numel() * t.element_size() for t in model.parameters())
    memory_report(model.config, weight_bytes, rows, samples, args, card)
    if self_tgt:
        # every pause's target first (saved, so a rerun or --resume teacher-forces exactly the same tokens)
        eos = eos_ids_for(model, tok, fmt)
        n_new = sum((i, s["turn"]) not in targets for i, s in samples)
        print(f"self targets: bf16 model, temperature {args.temperature:g}, seed {args.seed}, up to "
              f"{args.max_steps} tokens, eos {eos}; {len(samples) - n_new} saved in {targets_path}, "
              f"{n_new} to generate", flush=True)
        for instance_id, s in samples:
            s["real_target_len"] = len(s["target_ids"])
            s["target_ids"] = self_target(model, tok, s, instance_id, targets, targets_path, target=args.target,
                                          temperature=args.temperature, seed=args.seed, max_new=args.max_steps,
                                          eos_ids=eos, dev=dev)
        lens = [len(s["target_ids"]) for _, s in samples]
        if lens:
            print(f"self targets: {min(lens)}-{max(lens)} tokens (mean {sum(lens) / len(lens):.1f})", flush=True)

    for instance_id, s in samples:
        if (instance_id, s["turn"]) in done:
            continue
        est = estimate_peak_bytes(model.config, weight_bytes, len(s["hist_ids"]), len(s["fresh_ids"]), rows,
                                  base_bits=args.base_bits, max_steps=args.max_steps)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        ids = lambda x: torch.tensor([x], device=dev)
        hist, fresh, tgt = ids(s["hist_ids"]), ids(s["fresh_ids"]), ids(s["target_ids"])
        t0 = torch.cuda.Event(enable_timing=True) if dev.startswith("cuda") else None
        t1 = torch.cuda.Event(enable_timing=True) if t0 else None
        if t0:
            t0.record()
        res = run_pause(model, hist, fresh, tgt, rows, block=args.block, recent=args.recent,
                        max_steps=args.max_steps, check=diag is None, diag=diag is not None,
                        base_bits=args.base_bits, prefill_k=args.prefill_fetch_k, prefill_chunk=args.prefill_chunk)
        secs = None
        if t0:
            t1.record()
            torch.cuda.synchronize()
            secs = t0.elapsed_time(t1) / 1e3
        if diag:
            write_diagnostics(out_dir / f"diag_{instance_id}_t{s['turn']}", tok, rows, s, res, base)
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
                        "per_layer": int(r.per_layer), "base_bits": args.base_bits,
                        "prefill_fetch_k": args.prefill_fetch_k, "tool_prefill": r.fresh,
                        "target_source": args.target_source, "temperature": args.temperature,
                        "delta_nll": float(-res["logp"][i].mean()) - ref_nll,
                        "top1_agree": float((res["pred"][i] == res["pred"][ref]).float().mean()),
                        **res["rows"][i], **{b: res[b] for b in ("bf16_bytes", "base_bytes", "int1_bytes",
                                                                   "int2_bytes", "int3_bytes")}})
        _append(metrics_path, out)
        tokstr = lambda i: tok.decode([i])
        _append(guesses_path, [{"instance_id": instance_id, "turn": s["turn"], **g,
                                "guess_token": tokstr(g["guess_id"]), "actual_token": tokstr(g["actual_id"])}
                               for g in res["guesses"]])
        chk = {k: v for k, v in res.items() if k.startswith("check_")}
        nll_diff = max(abs(chk[f"check_{n}_nll_perstep"] - chk[f"check_{n}_nll_batched"])
                       for n in ("bf16", "base"))
        chk["check_max_nll_diff"] = nll_diff
        _append(samples_path, [{"instance_id": instance_id, "turn": s["turn"], "hist_len": res["P"],
                                "fresh_len": res["F"], "steps": res["T"], "ref_nll": ref_nll,
                                "seconds": secs, "model": args.target, "base_bits": args.base_bits,
                                "prefill_fetch_k": args.prefill_fetch_k, "k_match": res["k_match"],
                                "target_source": args.target_source, "temperature": args.temperature,
                                "seed": args.seed if self_tgt else "", "target_len": len(s["target_ids"]),
                                "est_peak_gb": est / GB,
                                "peak_gpu_gb": (torch.cuda.max_memory_allocated() / GB
                                                if torch.cuda.is_available() else float("nan")),
                                "uniform_refs": " ".join(map(str, args.uniform_refs)), **chk}])
        by = {o["scheme"] + (f"@{o['k_label']}" if o["k_label"] != "0" else ""): o["delta_nll"] for o in out}
        print(f"{instance_id} turn {s['turn']}: hist {res['P']}, {res['T']} steps"
              + (f", {secs:.1f} s" if secs else "")
              + " | " + " ".join(f"{n} {by[n]:+.4f}" for n in ("int1", "int2", "int3", "fetch_all") if n in by)
              + " | " + " ".join(f"{n} {by[n]:+.4f}" for n in
                                 [f"{p_}@{k_}" for k_ in ("64", "match")
                                  for p_ in ("s1", "s2", "spec", "spec_chain", "s2sens_b", "oracle")] if n in by)
              + f" | k_match {res['k_match']} | check NLL diff {nll_diff:.2e}, max|dlogp| bf16 "
              f"{chk.get('check_bf16_max_abs_logp', float('nan')):.2e} {base} "
              f"{chk.get('check_base_max_abs_logp', float('nan')):.2e}", flush=True)
        if args.check_nll_tol and nll_diff > args.check_nll_tol:
            raise SystemExit(f"per-step vs sdpa NLL check failed: {nll_diff:.3e} > {args.check_nll_tol}")
        del hist, fresh, tgt, res
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
