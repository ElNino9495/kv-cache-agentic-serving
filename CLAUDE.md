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
whether per-step fetching of bf16 tokens on a low-bit base (SpeCache-style) can beat uniform low-bit reload
(run 4: on the int1 base with a large prefill fetch, yes vs. uniform int2 at equal memory, CI below 0, but moving
~42x the bytes; int2 + fetch vs. int3 is only a tie).

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

### Per-step fetch study, run 2 (`kv_manager/results/gate_fetch/llama-3.1-8b-instruct-base{2,1}/`)

Run by a friend (Karthik) with `local_fetch.sh` on **2x A100 80 GB** (not the A6000 server), 21 Nov 2026, both bases
in parallel, base 1 with its own int3, ~30 min, **no prefill fetch**. Complete: 31/31 pauses per base, no resume,
smoke and per-step-vs-sdpa checks passed (max NLL diff 0.0117 base 2, 0.0184 base 1), xarray diagnostic for both.
Peaks 38.2 / 43.0 GB measured vs. 36.2 / 41.3 estimated (estimate ~2 GB low). Re-committed without the .tgz and
fetch_idx.pt files (`cfd8bea`).
- Base 2 (int2 + fetch vs. uniform int3 +0.0019, equal memory): nothing beats int3. Best @match s2sens_b +0.0072,
  s2sens_a +0.0073, s2 +0.0084, oracle +0.0091, spec +0.0105; CIs vs. int3 include 0 but every mean and median is
  worse, wins ~even. @match moves 5.9 GB/turn (42x int3's one-time extra reload). s2sens ≈ s2 (|Δ| ≤ 0.002);
  fetch_all +0.0073.
- Base 1 (int1 + fetch vs. uniform int2 +0.0302, equal memory): nothing beats int2. Uniform int1 +0.477. Best @match
  s2 +0.0415 (CI vs. int2 [−0.0024, +0.0257], median +0.0156); spec +0.0627, s1 +0.0684. fetch_all +0.0440 is
  already above int2: the int1 tool-output prefill alone costs more, so prefill fetch is required for int1 to win.
  On int1, s2sens_b beats s2 by 0.003-0.012 at k=32-128 and the oracle beats s2 by 0.05 at k=16.
- xarray t85: one token (step 14, ' x', int2 log p −8.6 vs. bf16 −1.2) carries the int2 damage; s1 makes it worse
  (previous-step set overlaps the oracle 5.7/16), s2 fixes it.
- Caveats: the old spec row (now `spec_chain`) guessed right only 31-38% (base 2) / 24-34% (base 1), bimodal per
  pause on base 1. It is chained as in the SpeCache paper (each guess from the previous guess), which under teacher
  forcing on Gemini-written targets keeps missing. Base 2's int2 row and base 1's int2 reference differ by up to
  0.015 per pause (batch composition changes fp32 summation order); A40 vs. A100 per-pause s.d. 0.018 (int2), so
  compare only within a run. Base-1 noise_ctrl mean +0.0038 (CI above 0), s.d. 0.0079 (2x base 2's).

### Per-step fetch study, run 3 (`kv_manager/results/gate_fetch/llama-3.1-8b-instruct-base{2,1}-pf128/`)

Great Lakes A40, run id `20261121-112651`, base 2 and base 1 as two jobs (~43 min each), both prefill fetch k=128,
31/31 pauses, peak 38.7 GB (est. 37.1). Base 1 without int3; int3 joined from this run's base 2 (bf16 ref NLL equal).
Logs in `gate_fetch/fetch_logs_run3/`. New rows: `spec` (guess of each step's input from the real previous target
token; a correct guess gives exactly s2's set) and `spec_chain` (SpeCache's chained guess); guesses in `spec_guesses.csv`.
- Base 2 (int2 + fetch vs. int3 −0.0015): ties int3. Best @match s2sens_a −0.0002 / s2sens_b −0.0001 (vs. int3
  +0.0013/+0.0014, CIs include 0); s2 +0.0009, oracle +0.0002, fetch_all −0.0015 (= int3). Prefill fetch lifted
  fetch_all from +0.0051 (noprefill) by −0.0066 [−0.0142, −0.0010].
- Base 1 (int1 + fetch vs. int2 +0.0302): s2@match +0.0179, −0.0123 vs. int2 [−0.0296, +0.0047], median −0.0016,
  17/31 wins (11 by > noise 0.0147, 9 losses); s2sens_b/oracle@match the same. Not significant. fetch_all +0.0165
  (noprefill +0.0405): the int1 prefill still costs +0.018 vs. int3 even with prefill fetch 128.
  Lead: s2sens_b@128 −0.0030 vs. int2 [−0.0196, +0.0146] at 0.147 resident vs. int2's 0.200 (74%), 136 MB unique
  fetched per turn (1.2x int2's extra reload); fails the gate's "not worse" bound (+0.01) on the CI.
- spec guesses right 82% (base 2) / 79-82% (base 1); spec_chain 36-40% / 21-35%. spec@match = s2 on base 2,
  +0.011 worse on base 1 (median 0); at small k much worse than s2. Misses hit the tokens that matter (xarray t85:
  input ' x' guessed "no"). Under teacher forcing the accuracy is Llama-greedy vs. Gemini-text agreement.
- Cost: @match moves 5.9 GB/turn (42x the 139 MB one-time extra reload), ~620 MB unique; prefill fetch 84 MB/turn.
- Noise: base-2 noise_ctrl s.d. 0.0039; base-1 noise_ctrl mean +0.0027 (CI above 0, as in run 2), s.d. 0.0073.

### Per-step fetch study, run 4 (`kv_manager/results/gate_fetch/llama-3.1-8b-instruct-base1-{pf512,pf128-self-T0,pf128-self-T0.7}/`)

Great Lakes A40, run ids `20261121-154503-b1-*`, three base-1 jobs (41.6-44.2 min each), 31/31 pauses, peak 38.7-38.8 GB
(est. 37.1), no int3 (no base-2 job). Logs in `gate_fetch/fetch_logs_run4/`. Each job hit one recovered CUDA allocator
OOM warning on sympy t72 (33.8k, 0.33 GB free; run 3 had the same): no headroom on an A40 for anything larger.
Self targets (`--target-source self --temperature T --seed 0`): bf16 writes ≤64 tokens per pause after a bf16 prefill,
saved in `self_targets.jsonl`; `kl_bf16` = KL(bf16 row || row), mean over steps.
- First significant fetch win: s2@match vs. uniform int2 at equal memory has its CI entirely below 0 in all three jobs.
  pf512 (real targets): +0.0081 vs. int2 +0.0302, −0.0221 [−0.0415, −0.0044], KL −0.0196 [−0.0297, −0.0123], lower KL
  on 29/31. Self T=0: −0.0174 [−0.0284, −0.0061]; T=0.7: −0.0210 [−0.0330, −0.0077]. s2sens_a/b, oracle and spec @match
  are the same; s1/sens/s1_layer no better than int2.
- pf512 vs. run 3's pf128 (int1/int2/noise_ctrl rows bit-identical, so exact pairing): every row better, e.g. fetch_all
  −0.0077 [−0.0129, −0.0028] (now +0.0089), s2@match −0.0098 [−0.0148, −0.0049], s2@16 −0.0242. Prefill fetch moves
  336 MB/turn (2.4x int2's 139 MB one-time extra reload) vs. 84 MB at k=128.
- At 74% of int2's memory (k=128, resident 0.147): s2sens_b@128 −0.0167 [−0.0381, +0.0038] (pf512), −0.0040
  [−0.0158, +0.0072] (self T0), −0.0096 [−0.0243, +0.0060] (self T0.7): upper bounds inside the gate's +0.01 "not worse"
  bound (an analogy: the gate rule counts bytes moved, these move 680 MB/turn + prefill). s2@128 pf512 misses (+0.0116).
- Self targets: bf16 NLL/token 0.24 (T0) / 0.36 (T0.7) vs. 0.74 on the Gemini targets, so those are off-distribution;
  but int2's damage is similar (+0.033 / +0.036 vs. +0.030) and the scheme ranking is unchanged. Per-pause rankings
  correlate weakly with real targets (Spearman 0.3-0.5); outliers move (flask t22, pytest t35, sympy t72, not xarray t85).
  Targets 20-64 tokens (mean 43.5 / 42.6), 6/31 and 4/31 hit the cap.
- spec guesses right 98% (T0) / 89% (T0.7) at match vs. 82% on real targets; spec@match = s2 then (+0.0005 / +0.0031),
  but spec@16 still +0.020 behind s2 at T0. spec_chain 24-56%.
- Noise: self-target noise_ctrl s.d. 0.0124 / 0.0122 (2x real), mean +0.0045 / +0.0051 (CI above 0); KL is cleaner.
- Cost unchanged: @match moves 5.7-6.1 GB/turn (42-43x int2's extra reload), ~600 MB unique (4.5x).

### Next run: fetch study run 5 (code ready, not run yet)

Five Great Lakes A40 jobs (`greatlakes_fetch.sbatch` default JOBS), each deciding its rules within one batch:
A1 `1:128:rows=a1` and B1 `1:128:self:0.7:rows=a1` (multi-guess spec: `spec[_chain]_union{2,3}@128`,
`spec[_chain]_wtd{2,3}@128/match`, baselines s2/oracle/spec/spec_chain @64/128/match, s2@256/384 for the union size
comparison); A2 `1:128:rows=a2` and B2 `1:128:self:0.7:rows=a2` (`s2_pool`/`spec_chain_pool` @16/64/128 with
M = k_match and k_match//2, `s2_k2score`, baselines @16/64/128/match); C `k2v1:128:rows=c` (base k2v1 = int2 keys +
int1 values, s2/oracle/spec_chain/spec_chain_wtd3/s2_pool, int1-base comparators `s2_int1base@16/match`,
`oracle_int1base@16`). B jobs reuse run 4's T=0.7 `self_targets.jsonl`. Pre-registered rules R1-R4 (verbatim, with
the exact tests, in every run-5 `summary.md`; `summarize_fetch --rules` writes `rules_run5_<id>.md`, run by the last
job): R1 wtd3 vs spec_chain @128 (A1+B1); R2 k2v1 vs int1 scoring gap and @match (C); R3 s2_pool_Mmatch@64/128 vs
s2@match, CI upper < 2 s.d. noise and ≤ 2x int2's extra reload moved (A2; B2 for information); R4
s2_pool_Mhalf@64 vs int2, CI upper < +0.01 and ≤ 2x moved (A2+B2). Union rows are diagnostic only. Estimated peaks
36.1-36.6 GB (k2v1 shares the int2 keys / int1 values tensors); ~40-65 min per job; 4 h limit. CPU checks only.

## GPU situation

- Great Lakes (UMich, Slurm, partition spgpu): A40 48 GB (~47.5 GB usable). Ran the gate, the uniform-only gate and
  the first fetch run and fetch runs 3 and 4; next: fetch run 5. Checkout: `~/kv-cache-agentic-serving/kv_manager`. All three `greatlakes*.sbatch` files take the Slurm
  account from `ACCOUNT=<your_account>` and submit themselves; no account is hard-coded. `greatlakes_fetch.sbatch` submits one
  job per `JOBS` entry `BASE:PREFILL_K[:self:T][:rows=SET]` (BASE 1 / 2 / k2v1; default = run 5's five jobs; run 4
  was "1:512 1:128:self:0 1:128:self:0.7", run 3 "2:128 1:128"), each with its own run id `<stamp>-<job>` (e.g. `-b1-pf128-self-T0.7`) and folder; 3 h limit,
  ~39 GB peak; run 3 took ~43 min per job. It prints each job's resume command (`RESUME=1 RUN_ID=<job's id>
  JOBS="<entry>"`). A base-1 job joins int3 only from a base-2 job with the same prefill k and targets in the same
  submission (run ids checked). Jobs write to the git-ignored `RESULTS_ROOT` (default `results/cluster/`: gate_fetch/,
  gate_fetch_smoke/, status markers, `fetch_study_results_<run id>.tgz` with the Slurm log in `gate_fetch/fetch_logs_<run id>/`),
  so `git pull` on the cluster never meets untracked result files; unpack locally with `tar xzf <tgz> -C results/`.
- CloudLab d8545 (Nov 2026): 4x A100 40 GB (~42.5 GB usable). `cloudlab_fetch.sh` (plain bash + tmux, venv and
  HF_HOME on /mydata) runs base 2 on GPU 0 and base 1 on GPU 1 in parallel; base 1 skips the int3 history copy
  (`--uniform-refs 2`) to fit, and its summary joins int3 from base 2 (`summarize_fetch --join`). Estimated
  peaks: 36.2 GB each (base 1 with int3 would be 41.3 GB). `run_fetch --estimate-memory` checks without
  loading weights; samples.csv records estimate and measured peak per pause.
- Any Linux machine with A100s or RTX A6000s (40/48/80 GB, one or more): `local_fetch.sh` (no Slurm, no /mydata;
  parallel on two GPUs, skips int3 on base 1 below ~44 GiB). For shared servers it honours
  `CUDA_VISIBLE_DEVICES=1,2` (or `GPUS=1,2`, which wins), queries only those cards (`nvidia-smi --id`), sets
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`, and refuses to start on a card with > 1 GB in use (`MAX_USED_MIB`).
  The A6000 options are untested on real hardware (run 2 used A100 80 GB). Instructions for running it
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
| `gate_fetch/<model>-base<b>[-pf<K>][-self-T<T>]/` | fetch runs 2+ (base 2 / base 1, optional prefill fetch, optional self-written targets): metrics.csv (kl_bf16 from run 4), samples.csv, summary.md, spec_guesses.csv (run 3+), self_targets.jsonl (self targets), run_id.txt (Great Lakes); `diag_<instance>_t<turn>/` inside for `--diagnose` (fetch_idx.pt is git-ignored) |
| `gate_fetch/llama-3.1-8b-instruct-base{2,1}-pf128/`, `fetch_logs_run3/` | fetch run 3 (A40, prefill fetch 128) and its Slurm logs |
| `gate_fetch/llama-3.1-8b-instruct-base1-{pf512,pf128-self-T0,pf128-self-T0.7}/`, `fetch_logs_run4/` | fetch run 4 (A40, base 1: prefill fetch 512; self targets T=0 / 0.7) and its Slurm logs |
| `gate_fetch/<model>-base{1,k2v1}-pf128[-self-T0.7]-{a1,a2,c}/`, `rules_run5_<id>.md`, `fetch_logs_<run id>/` | fetch run 5 (not run yet): row sets a1/a2/c, overall R1-R4 verdicts, logs (unpacked from `results/cluster/*.tgz`) |
| `results/cluster/` (git-ignored) | where cluster jobs write (`RESULTS_ROOT`); unpack a `.tgz` with `tar xzf <tgz> -C results/` |
| `gate_fetch/llama-3.1-8b-instruct-base{2,1}/`, `fetch_logs/` | fetch run 2 (2x A100 80 GB, no prefill fetch) and its logs + environment report |
| `gate_fetch_smoke/<name>/`, `gate_smoke/` | one-pause smoke checks |
| `gate_fetch/.status-*` | CloudLab / local / Great Lakes completion markers (git-ignored) |

The fetch results were first committed under the repo-root `results/gate_fetch/` and then moved here
(`22cd4ce`).

## Status (as of 22 Nov 2026)

| Piece | State |
|---|---|
| Phase 0, Phase 1 | Done |
| Phase 2 gate | Run on an A40: NO-GO. M2 skipped. |
| Uniform-only gate (int4/int3/int2, 3 models) | Done |
| Per-step fetch study v1 (int2 base) | Done: nothing beats uniform int3 |
| Per-step fetch study run 2 (spec, base 1, equal memory, diagnostics; no prefill fetch) | Done (2x A100 80 GB): nothing beats uniform int(b+1) at equal memory |
| Per-step fetch study run 3 (prefill fetch 128, anchored spec + spec_chain, guess log) | Done (A40): base 2 ties int3; base 1 beats int2 on the mean, not significantly |
| Per-step fetch study run 5 (multi-guess spec, fetch pools, s2_k2score, k2v1 base; rules R1-R4) | Built and CPU-tested; five Great Lakes A40 jobs |
| Per-step fetch study run 4 (base 1: prefill fetch 512; self-written targets T=0 / 0.7; KL) | Done (A40): s2@match beats int2 at equal memory, CI below 0, in all three jobs; ~42x the bytes moved |
| M1: connector with uniform reload | Built, unit tests pass on CPU, never run on a GPU |
| M2: draft scoring inside the connector | Skipped (gate NO-GO) |
| M3: full replay benchmark + quality via forced decoding | Planned |

Gate pass rule (fixed in code before running): draft-guided reload moves ≤ uniform-4-bit bytes and
halves ΔNLL, or moves ≤ 75% of uniform-4-bit bytes and is not worse. Bootstrap confidence intervals.

Branches: `fetch-study` holds all current work (it contains `arihan`; `uniform-int3` was deleted); `main` is
behind. Remotes: `origin` (kv-cache-agentic-serving) and `friend` (spec_kvagent, no shared history with origin's
main); push `fetch-study` to both.

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

- Tests: `cd kv_manager && python -m pytest tests -q` (CPU, ~15 s; 93 tests, the connector import test
  needs vLLM installed)
- Environment: `bash kv_manager/envt/setup_env.sh` (Python 3.12, vLLM 0.27.1, pinned in
  `kv_manager/envt/requirements.lock.txt`). GPU runs of the gate/fetch study only need torch +
  transformers 5.14.1 + pandas (see `cloudlab_fetch.sh` / `run_on_gpu.sh`).
- Fetch study: `cd kv_manager && ACCOUNT=<acct> bash experiments/gate/greatlakes_fetch.sbatch` (Great Lakes, one job per base) or
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
