# spec_kvagent: working notes for Claude

CSE 585 (University of Michigan) course project, "Speculative KV Cache Quantization for Agentic
Serving". Team: Arihan Iyer, Kushagra Mishra, Jatin Shihora, Rohit Puthucode. Final deadline: mid-December 2026.

Read `docs/README.md` for the background. `docs/02-idea.md` (the design), `docs/04-phases.md` (status) and
`docs/06-architecture.md` (the vLLM connector) are the most important. **The docs predate the GPU runs**
(04-phases still says the gate is "built, not run"); where they disagree with this file, this file is newer.

## The idea in one paragraph (original design, v2)

When an agent pauses for a tool call, park its KV cache in CPU memory and free the GPU. When the
tool returns, a small draft model (Llama-3.2-1B) writes 4 sampled guesses of 10 tokens and scores
each 16-token history block by consensus attention (max over steps within a guess, mean over
guesses). Every block is reloaded first at 2-bit (KIVI-style), then the top-scored blocks are
overwritten with their bf16 originals. Block 0 and the last 64 tokens are always bf16. No token is
ever dropped. Target model: Llama-3.1-8B-Instruct. Built as a plain vLLM 0.27.1 KV-connector
plugin: no fork, no custom kernels. **The gate rejected this design (NO-GO, below);** current work asks
whether per-step fetching of bf16 tokens on a low-bit base (SpeCache-style) can beat uniform low-bit reload.

## History (why the design looks like this)

- v1 (`proposal/proposal.tex`): GPU-resident mixed precision, triggered by memory pressure.
- Phase 1 (`Phase_1/`, done): consensus scoring finds ~52% of oracle tokens; protecting 10% of
  tokens keeps ΔNLL within ~0.05 nats of fp16; evicting instead of quantizing does 7-18x more
  damage. But uniform 2-bit was already cheap (ΔNLL ≈ 0.054), and GPU mixed precision needs custom
  kernels. So the idea moved to reload time (v2).
- Phase 1 limitations to keep in mind: same model used as draft and target; only 12 contexts
  (3 runs × 2k-5k tokens); fake quantization (no real memory or speed measured); SnapKV baseline
  is simplified (global layer/head average, pool kernel 7, window 32 vs. official per-head, kernel 5, window 64).
- Phase 2 gate (Nov 2026): NO-GO for draft-guided reload. Then a uniform-only follow-up and a per-step fetch
  study (below).

## Results so far (all measured; ΔNLL in nats per target token vs. bf16 reload)

### Phase 2 gate: NO-GO (`kv_manager/results/gate/`, REPORT.md + summary.md)

One A40 48 GB on Great Lakes, 31 pauses (8.0k-33.8k history), ~40 min. Pre-registered rule above.
- Uniform int4 is almost lossless: ΔNLL +0.0009 at 0.325 of bf16 bytes, top-1 0.991 (block 16).
  Uniform int2: +0.0332 at 0.200.
- No draft-guided configuration passes either part of the rule at either block size. Part (a) misses by ~10x;
  part (b) misses narrowly at block 16 (top 40% at int4, 0.250 bytes vs. the 0.244 limit, CI upper bounds
  +0.011-0.012 vs. +0.01). Block 64 is worse everywhere.
- The limit is the int2 floor, not the scorer: even the oracle never beats uniform int4 (all CIs above 0).
- Consensus and SpecPrefill are statistically tied (every paired CI includes 0); both clearly beat recency and
  random (recall vs. oracle at 10%: consensus 0.58, random 0.10, recency 0.22).
- Draft scoring latency (A40, HF eager, median): draft_window 81 ms, specprefill 246 ms, consensus 701 ms.
- Measurement fix in that run: the bf16 reference now uses sdpa like every scheme; uniform_bf16 control is 0.

### Uniform-only gate (`kv_manager/results/gate_uniform/<model>/`, summary.md)

`run_gate.py --uniform-only`: uniform int4/int3/int2 + bf16 control, no draft, same 31 pauses (re-tokenized
pause-for-pause; Llama lengths match the first run exactly). Block 16:
- Llama-3.1-8B: int4 +0.0009 (0.325), **int3 −0.0001 (0.262)**, int2 +0.0332 (0.200). At about the same bytes
  uniform int3 beats the gate's consensus and SpecPrefill rows (e.g. −0.013 vs. consensus top-5% bf16).
- Llama-3.2-3B: int4 +0.0006, int3 +0.0055, int2 +0.0347.
- Qwen3-8B (27 pauses; 4 dropped above its 32,768-token context, no YaRN): int4 −0.0061, int3 −0.0036,
  int2 −0.0134. Negative ΔNLL under added noise is suspicious; likely the Gemini-written targets are
  off-distribution for Qwen. Don't compare Qwen ΔNLLs with Llama's without checking this.

### Per-step fetch study, first run (`kv_manager/results/gate_fetch/llama-3.1-8b-instruct/`)

`run_fetch.py`: int2 history (block 16, same pins), tool output prefilled over int2 without fetching, first
≤64 target tokens teacher-forced one at a time; each step every scheme swaps its own top-k tokens per KV head
to bf16 for that step only. A40, 31 pauses, 29 rows, 22 min.
- Nothing beats uniform int3 (−0.0004): every fetch configuration has its CI vs. int3 entirely above 0.
  Best: s2@128 (current-step int2 attention) +0.0080; oracle@128 +0.0102.
- `fetch_all` (every middle token, every step) +0.0059: prefilling the tool output over int2 sets a ceiling
  per-step fetching cannot remove.
- Current-step selection (s2, oracle) recovers ~2/3 of int2's damage; previous-step selection (s1, sens_a,
  sens_b) barely helps and is worse than int2 by more than noise on 10-12 pauses
  (pydata__xarray-4687 t85: s1 +0.33 vs. int2 +0.21). Brute-force tests show this is not a fetch-math bug.
- Noise floor (`noise_ctrl`, equal to bf16 in exact arithmetic): s.d. 0.0045 per pause, so per-pause
  differences under ~0.009 are bf16 rounding; only averages over pauses mean anything.
- Per-step vs. batched sdpa: per-token log-prob differs by up to 0.19-0.22 (bf16), NLL per pause by ≤0.0115.

### Next run (code ready, not run yet)

Version 2 of the fetch study adds: SpeCache's speculative token (`spec`), current-query sensitivity
(`s2sens_a/b`), `--base-bits 1` (SpeCache's modified 1-bit quantizer, Eq. 3) with uniform int1/int2/int3,
`k = match` (base + fetched set = uniform int(base+1)'s GPU memory, for int1+fetch vs. int2 at equal memory),
`--prefill-fetch-k` (fetch during the tool-output prefill), `--diagnose` (per-step fetch logs for xarray t85),
medians / outlier-dropped means / win counts in the summary. Only CPU checks so far (tiny models, Llama-3.2-1B
fp32 / 3B bf16 on one pause).

## GPU situation

- Great Lakes (UMich, Slurm, partition spgpu): A40 48 GB. Ran the gate, the uniform-only gate and the first
  fetch run. All three `greatlakes*.sbatch` files take the Slurm account from `ACCOUNT=<your_account>` and
  submit themselves (`ACCOUNT=<acct> bash experiments/gate/greatlakes_fetch.sbatch`); no account is hard-coded.
- CloudLab d8545 (Nov 2026): 4x A100 40 GB (~42.5 GB usable). `cloudlab_fetch.sh` (plain bash + tmux, venv and
  HF_HOME on /mydata) runs base 2 on GPU 0 and base 1 on GPU 1 in parallel; base 1 skips the int3 history copy
  (`--uniform-refs 2`) to fit, and its summary joins int3 from base 2 (`summarize_fetch --join`). Estimated
  peaks: 36.2 GB each (base 1 with int3 would be 41.3 GB). `run_fetch --estimate-memory` checks without
  loading weights; samples.csv records estimate and measured peak per pause.
- Any Linux machine with A100s or RTX A6000s (40/48/80 GB, one or more): `local_fetch.sh` (no Slurm, no /mydata;
  parallel on two GPUs, skips int3 on base 1 below ~44 GiB). For shared servers it honours
  `CUDA_VISIBLE_DEVICES=1,2` (or `GPUS=1,2`, which wins), queries only those cards (`nvidia-smi --id`), sets
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`, and refuses to start on a card with > 1 GB in use (`MAX_USED_MIB`).
  Expected next: a shared 8× RTX A6000 server. Instructions for running it
  without project background: `docs/RUNBOOK_fetch_study.md`.
- Local Mac (16 GB RAM, no CUDA): CPU tests and small-model checks only; Llama-3.1-8B does not fit.

## Repo map

- `kv_manager/kvm/`: the connector. `quant.py` (pack/unpack blocks), `sched.py` (BlockIndex,
  plan_bits), `transfer.py` (pinned CPU pools, side CUDA stream), `connector.py` (AgentKVConnector).
  `draft.py` (M2) does not exist and won't (gate NO-GO).
- `kv_manager/experiments/gate/`:
  - `run_gate.py` (gate; `--uniform-only`), `data.py` (recordings, chat formats Llama-3 / Qwen3, pause
    selection by turn), `plot_gate.py`, `summarize_uniform.py`, `fakequant.py` (int3 and SpeCache 1-bit
    without packing, same arithmetic as `kvm/quant.py`).
  - `fetch_attn.py` (per-step fetch core: custom attention, schemes, prefill fetch, memory estimate),
    `run_fetch.py`, `summarize_fetch.py`.
  - Run scripts: `greatlakes.sbatch`, `greatlakes_uniform.sbatch`, `greatlakes_fetch.sbatch`,
    `cloudlab_fetch.sh`, `local_fetch.sh`, `run_on_gpu.sh`.
- `kv_manager/bench/`: online replay benchmark (`replay.py`, `serve.sh`, `run_systems.sh`).
- `kv_manager/tests/`: unit tests, CPU only.
- `Phase_1/`, `src/`, `results/` (repo root): earlier phases. Treat as finished.
- `TEMP.md`: a teammate's notes on TriForce and the offload idea.

## Results paths (all under `kv_manager/results/`)

| Path | Contents |
|---|---|
| `gate/` | Phase 2 gate: REPORT.md, summary.md, metrics.csv, samples.csv (the pause list every later run reuses), pareto_B16/B64.png, run.log |
| `gate_uniform/<model>/`, `gate_uniform/summary.md` | uniform-only gate per model (`llama-3.1-8b-instruct`, `llama-3.2-3b-instruct`, `qwen3-8b` with dropped.csv) |
| `gate_fetch/llama-3.1-8b-instruct/` | first fetch run (int2 base, 29 rows) + `gate_fetch/kv-gate-fetch-66088085.log` |
| `gate_fetch/<model>-base<b>[-pf<K>]/` | fetch v2 runs (base 2 / base 1, optional prefill fetch); `diag_<instance>_t<turn>/` inside for `--diagnose` |
| `gate_fetch_smoke/<name>/`, `gate_smoke/` | one-pause smoke checks |
| `gate_fetch/.status-*` | CloudLab window markers (git-ignored) |

The fetch results were first committed under the repo-root `results/gate_fetch/` and then moved here
(`22cd4ce`).

## Status (as of 17 Nov 2026)

| Piece | State |
|---|---|
| Phase 0, Phase 1 | Done |
| Phase 2 gate | Run on an A40: NO-GO. M2 skipped. |
| Uniform-only gate (int4/int3/int2, 3 models) | Done |
| Per-step fetch study v1 (int2 base) | Done: nothing beats uniform int3 |
| Per-step fetch study v2 (spec, base 1, equal memory, prefill fetch, diagnostics) | Built and CPU-tested; waiting for the CloudLab run |
| M1: connector with uniform reload | Built, unit tests pass on CPU, never run on a GPU |
| M2: draft scoring inside the connector | Skipped (gate NO-GO) |
| M3: full replay benchmark + quality via forced decoding | Planned |

Gate pass rule (fixed in code before running): draft-guided reload moves ≤ uniform-4-bit bytes and
halves ΔNLL, or moves ≤ 75% of uniform-4-bit bytes and is not worse. Bootstrap confidence intervals.

Branches: `fetch-study` holds all current work (it contains `arihan` and `uniform-int3`, which add nothing
beyond it); `main` is behind.

## Benchmarks and metrics

- Workload: 10 mini-SWE-agent runs on SWE-bench Verified (18k-56k tokens, 41-106 turns), from
  `john-b-yang/20260901_mini-v2.4.2_gemini-3-5-flash`. A browsing workload (likely WebArena) is a
  possible extension, not started.
- Gate (offline): 31 tool-call pauses at 8k-34k history. Scorers: consensus, draft_window,
  specprefill, recency, random, oracle. Schemes: uniform int4/int2, top 5/10/20/40% at bf16 or int4.
- Fetch study (offline): same pauses, ≤64 teacher-forced steps; schemes and rows in `run_fetch.py`'s docstring.
- Replay (online): N concurrent agents, simulated tool delay 0.5 s + Exp(mean 2 s), same GPU KV
  budget. Systems: vllm, vllm_offload, kvm_bf16, kvm_int4, kvm_int2, kvm_draft (M2, now skipped).
- Metrics: ΔNLL, top-1 agreement, recall vs. oracle, bytes fraction, bytes moved per turn, GPU-resident
  bytes, TTFT after tool return (p50/p90/p99, headline), throughput, load/save bytes, draft scoring latency.

## Commands

- Tests: `cd kv_manager && python -m pytest tests -q` (CPU, ~10 s; 70 tests, the connector import test
  needs vLLM installed)
- Environment: `bash kv_manager/envt/setup_env.sh` (Python 3.12, vLLM 0.27.1, pinned in
  `kv_manager/envt/requirements.lock.txt`). GPU runs of the gate/fetch study only need torch +
  transformers 5.14.1 + pandas (see `cloudlab_fetch.sh` / `run_on_gpu.sh`).
- Fetch study: `ACCOUNT=<acct> bash kv_manager/experiments/gate/greatlakes_fetch.sbatch` (Great Lakes) or
  `cd kv_manager && bash experiments/gate/cloudlab_fetch.sh` (CloudLab). Summaries:
  `python -m experiments.gate.summarize_fetch --dir results/gate_fetch/<run> [--join <other run>]`.
- Full run instructions: `kv_manager/README.md`

## Rules for changes

- Do not break the connector's correctness rules in `docs/06-architecture.md` (for example: a
  stored bf16 slot is never overwritten; a block reloaded at low precision is never saved again).
- Run the unit tests after any change under `kv_manager/kvm/` or `kv_manager/experiments/gate/`.
- No vLLM fork, no custom CUDA/Triton kernels, no model training. These are project non-goals.
- Do not report numbers that have not been measured. Unmeasured results are "next phase".
- Results are paired per pause: compare schemes with paired bootstrap CIs, and judge per-pause differences
  against the noise floor (`noise_ctrl`).

## Open ideas, not yet done

- Ablation: max across guesses instead of mean (`consensus_max`), reusing the same attention tensors.
- Note the SnapKV simplification in the final report.
- Update `docs/04-phases.md` (and the phase diagram) with the gate NO-GO and the fetch study.
- If fetching helps on a 1-bit base at equal memory, the connector would need a per-step fetch path (M1 only
  reloads whole blocks once).
