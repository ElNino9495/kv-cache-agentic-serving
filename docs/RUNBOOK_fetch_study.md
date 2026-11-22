# Runbook: per-step fetch study (runs 2 to 4)

> **Status (22 Nov 2026).** Run 2 is done: it ran on 2 × A100 80 GB with this runbook, and the results are
> committed (`kv_manager/results/gate_fetch/llama-3.1-8b-instruct-base2/` and `-base1/`). Run 3 runs on the
> project's Slurm cluster (`experiments/gate/greatlakes_fetch.sbatch`), not with this runbook; it is done
> (`-base2-pf128/`, `-base1-pf128/`). Run 4 (base 1: prefill fetch 512, and self-written targets at temperature
> 0 and 0.7) also ran there and is done (`-base1-pf512/`, `-base1-pf128-self-T0/`, `-base1-pf128-self-T0.7/`). Part B still works for repeating a run on another Linux machine. It now also runs
> run 3's corrected speculative-token rows, but without prefill fetch or self-written targets.

For someone running the next experiment on their own Linux machine with NVIDIA A100s or RTX A6000s (a shared
server is fine), who knows the project up
to the gate's NO-GO. Part A explains what happened since; Part B is the step-by-step run.

---

## Part A: what happened since the NO-GO, and why this run matters

**Where we were.** The gate asked whether a small draft model can pick which parts of an agent's KV cache to
reload in full precision while the rest comes back at 2-bit, and beat plain 4-bit reload at the same number of
bytes. It could not. Uniform 4-bit turned out to be almost lossless for Llama-3.1-8B (ΔNLL +0.0009 nats per
token), and anything that puts most of the history at 2-bit costs more than that, however well the draft picks.
Even a perfect oracle lost. So the draft-in-vLLM milestone was dropped.

**Uniform-only follow-up.** We then measured plain uniform reload at 4, 3 and 2 bits on the same 31 pauses (real
coding-agent conversations, 8k–34k tokens of history), for three models. On Llama-3.1-8B, **uniform 3-bit is
effectively lossless** (ΔNLL −0.0001 at 26% of the bf16 bytes), and it beats the draft-guided schemes at the same
size. 2-bit costs about +0.03.

**A different idea: fetch per step, not per reload.** SpeCache (ICML 2025) keeps only a very low-bit copy of the
history on the GPU and, at every generated token, fetches the few full-precision tokens the model is about to
attend to. Selection is no longer a one-off guess at reload time; it adapts token by token. We built an offline
version: reload at 2-bit, then at every step swap a top-k set of history tokens (per layer and KV head) to bf16
for that step only, with several ways of choosing them.

**Run 1 (2-bit base, done).** Nothing beat uniform 3-bit. The best scheme (choose by the current token's attention
over the 2-bit keys, k=128) scored +0.008. Two reasons:

- The tool output is processed on top of the 2-bit history before any fetching starts, and that alone costs about
  +0.006. Fetching later cannot undo it, so even "fetch everything every step" stays above 3-bit.
- Choosing with the *previous* token's attention barely helps, and on some pauses is worse than not fetching at
  all. Tests against a brute-force reference rule out a bug in our code.

Run 1 also measured the noise from bf16 rounding: per-pause differences below about 0.009 nats are noise, so only
averages over the 31 pauses count.

**Run 2 (done, 21 Nov 2026) closed most of the remaining gaps:**

1. **SpeCache's actual selection rule**: a speculative copy of the model's next token, decoded alongside, chooses
   what to fetch. Run 1 only had cruder stand-ins.
2. **A 1-bit base at equal GPU memory.** A 1-bit history plus fetched tokens can use exactly as much GPU memory as a
   uniform 2-bit history. This is the regime SpeCache targets and where fetching has the most room to pay off.
3. **Fetching while the tool output is processed**, to lift the ceiling found in run 1.
4. **Diagnostics for the one pause where the previous-token rule fails badly**, so we can see why.

**What run 2 found.** At equal GPU memory, fetching did not beat plain uniform reload on either base.
- 2-bit + fetch vs. uniform 3-bit: best +0.0072 vs. +0.0019.
- 1-bit + fetch vs. uniform 2-bit: best +0.0415 vs. +0.0302.

On the 1-bit base, even "fetch everything every step" (+0.044) is worse than uniform 2-bit, because the tool output
was processed over the 1-bit history. Item 3 above (fetching during the tool output) was not run. The speculative
token also guessed right on only a third of the steps or less.

**Run 3 (next, on the Slurm cluster)** runs both bases with fetching during the tool output (128 tokens per chunk).
It also fixes the speculative token: `spec` now guesses each step's token from the real previous token, and
`spec_chain` keeps the paper's chained version for comparison. Every guess is logged in `spec_guesses.csv`.

**Run 3 found** (A40s): with prefill fetch, 2-bit + fetch ties uniform 3-bit (best +0.0013 vs. int3, CI includes
0), and 1-bit + fetch averages better than uniform 2-bit (s2 +0.0179 vs. +0.0302) but with a median near 0 and a
CI that includes 0. "Fetch everything" on the 1-bit base is still +0.0165, so the tool-output prefill still costs.

**Run 4 (Slurm cluster, base 1 only)** tests two explanations: a larger prefill fetch (512 tokens per chunk), and
targets written by the model itself (greedy and temperature 0.7) instead of the agent's recorded messages, with
the KL from the bf16 model reported next to ΔNLL.

**Run 4 found** (A40s, 31 pauses per job, about 43 min each):
- **1-bit + fetch now beats uniform 2-bit at equal memory**, with the 95% CI entirely below 0 in all three jobs.
  Current-step selection (s2) at equal memory: −0.0221 [−0.0415, −0.0044] vs. 2-bit with the 512-token prefill
  fetch; −0.0174 and −0.0210 on the model's own text at temperature 0 and 0.7. KL agrees, with tighter CIs.
- **The larger prefill fetch helps every row** (paired with run 3, whose controls are bit-identical): "fetch
  everything" improves by −0.0077 to +0.0089, s2 at equal memory by −0.0098. It moves 336 MB per tool output
  instead of 84 MB.
- **Self-written targets don't change the conclusions.** The model finds its own text far likelier (NLL 0.24 /
  0.36 per token vs. 0.74 on the recorded messages), but 2-bit's damage and the ranking of schemes stay the same.
  The speculative token guesses right 98% / 89% of the time there (82% on the recorded messages).
- **Cost:** fetching at equal memory moves about 6 GB per turn (about 600 MB of distinct tokens), against 2-bit's
  one-time extra reload of 139 MB. No uniform 3-bit row was run, so 1-bit + fetch vs. 3-bit is untested.

**Why it matters.** These runs decide how the project's final system reloads KV after a tool call: either plain
uniform low-bit reload (already built), or a per-step fetch path, which is worth building only if it beats uniform
reload at equal memory. The result feeds directly into the final evaluation.

---

## Part B: running it

### 1. Requirements

| Item | Needed |
|---|---|
| GPU | NVIDIA A100 (40 or 80 GB) or RTX A6000 (48 GB), or another Ampere-or-newer card with 40 GB or more. One idle card works; two run twice as fast. On a shared server, choose idle cards (step 3). |
| OS | Linux with the NVIDIA driver installed (`nvidia-smi` works). Driver 560 or newer is safest for the CUDA 12.6 PyTorch wheels. |
| Disk | About 100 GB free for the data directory (model 16 GB, Python environment ~8 GB, caches). Results are small (a few MB). |
| Python | 3.12, with `venv` (Ubuntu: `python3.12` and `python3.12-venv`; or `uv python install 3.12`). |
| Other | `git`, `tmux`, internet access during setup (Hugging Face and GitHub; no login or token needed). |

### 2. Get the code

Clone the repository you were given (either copy; they hold the same code), then switch to the `fetch-study`
branch. **Do not use `main`**: in at least one copy it is out of date and does not contain this experiment.

```bash
git clone <repository URL>
cd <repository>
git checkout fetch-study
git branch --show-current            # must print: fetch-study
ls kv_manager/experiments/gate/local_fetch.sh   # must exist; if not, you are on the wrong branch
cd kv_manager
```

All later commands run from this `kv_manager/` directory.

### 3. One command: setup, check, run

```bash
DATA_DIR=/path/with/100GB/free bash experiments/gate/local_fetch.sh
```

- `DATA_DIR` (default `~/kvfetch`) holds the Python environment, the pip cache and the model (`HF_HOME`
  defaults to `$DATA_DIR/hf`; set `HF_HOME` yourself to reuse an existing Hugging Face cache).
- If Python 3.12 is not on the path as `python3.12`, add `PYTHON=/path/to/python3.12`.

**On a shared server** (for example 8 × RTX A6000 where some cards are busy), first look at `nvidia-smi`, pick
one or two cards that show almost no memory in use, and name them with `CUDA_VISIBLE_DEVICES`:

```bash
nvidia-smi                                        # find idle cards, e.g. 1 and 2
CUDA_VISIBLE_DEVICES=1,2 DATA_DIR=/path/with/100GB/free bash experiments/gate/local_fetch.sh
```

- The script uses only the listed cards (the first two, if you list more) and queries only them. Numbers are
  the ones `nvidia-smi` shows. `GPUS=1,2` does the same and takes priority over `CUDA_VISIBLE_DEVICES`.
- It **refuses to start** if a chosen card already has more than 1 GB of memory in use, both at the beginning
  and again after setup, just before the run starts. Pick other cards, or wait. (`MAX_USED_MIB=<MiB>` changes the
  limit; leave it alone unless you are sure the memory in use is not another job.)
- Without `CUDA_VISIBLE_DEVICES` or `GPUS` it uses GPUs 0 and 1, which on a shared server are often busy.

What the script does:

1. **Environment** (pinned): a venv in `$DATA_DIR/venv` with `torch==2.14.1` from the CUDA 12.6 wheel index and
   `experiments/gate/requirements_fetch.txt` (transformers 5.14.1, huggingface-hub 1.22.0, tokenizers 0.22.2,
   safetensors 0.8.0, numpy 2.2.6, pandas 2.2.3, matplotlib 3.9.2). It stops if PyTorch cannot see a GPU.
2. **Downloads**: the model `unsloth/Llama-3.1-8B-Instruct` (~16 GB, public) and the 10 recorded agent
   conversations (small, from GitHub).
3. **Check**: prints your GPUs, the plan and an estimated peak memory per run, with a warning if it might not fit.
4. **Run**, inside a tmux session called `fetch`:
   - two GPUs: the 2-bit base on the first chosen card and the 1-bit base on the second, in parallel (windows
     `base2`, `base1`);
   - one GPU: 2-bit base, then 1-bit base.
   - On 40 GB cards, the 1-bit run leaves out its 3-bit reference copy to fit, and borrows the 3-bit numbers
     from the 2-bit run (same pauses). On 48 GB (RTX A6000) and 80 GB cards it keeps its own.
   - Each base: a one-pause smoke check, the full run over 31 pauses, a summary, and the one-pause diagnostic.
5. **Packs everything into one file** at the end (see step 7).

You can close the terminal after `started: ...` is printed; the run continues in tmux.

### 4. Expected time and memory

Measured: run 2 took 27–29 minutes per base on 2 × A100 80 GB, in parallel, about 31 minutes from start to the
packed file (55 schemes). The corrected speculative rows add about 20% (65 schemes). The figures below allow for
that; the A6000 figures are an estimate.

| Machine | Setup (first time) | Experiment |
|---|---|---|
| 1 × A100 (40 or 80 GB) | 15–30 min (downloads) | about 70–90 minutes (bases one after the other) |
| 2+ × A100 | 15–30 min | about 35–45 minutes (bases in parallel; run 2 measured 31 min with fewer schemes) |
| 2 × RTX A6000 | 15–30 min | about 50–70 minutes (bases in parallel; A40-class speed, ~1.5× slower than an A100) |

Estimated peak GPU memory per run (largest pause, 34k tokens of history):

| Run | 40 GB card | 48 GB or 80 GB card |
|---|---|---|
| 2-bit base | ~36 GB | ~36 GB |
| 1-bit base | ~36 GB (3-bit reference left out) | ~41 GB (with 3-bit reference) |

An A100 40 GB has about 42.5 GB usable, so 40 GB cards fit with a few GB to spare; an RTX A6000 has about 48–51 GB
(less with ECC on), so both runs fit with room left. This is also why a card someone else is using is refused. Each pause's measured peak is
written to the results.

### 5. Checking that it is working

Attach with `tmux attach -t fetch` (switch windows with `Ctrl-b n`, detach with `Ctrl-b d`). Logs are also in
`results/fetch_logs/`.

- **First ~5 minutes:** model loading, then the smoke check prints one line per base like
  `sphinx-doc__sphinx-8548 turn 23: hist 8523, 48 steps | int2 +0.0... | ... | check NLL diff 3.1e-03, ...`.
  The `check NLL diff` must be below 0.05 (expect around 0.01 or less); if it is not, the script stops on its own.
- **Then:** one such line per pause, 31 per base. The file `results/gate_fetch/llama-3.1-8b-instruct-base2/samples.csv`
  (and `-base1`) gains one row per finished pause.
- **`nvidia-smi`** in another terminal should show roughly 30–40 GB in use on each card you chose, and no change
  on the others.
- **Finished:** the last window prints `DONE. Send back this one file: ...`.
- **Problems:** a line starting with `!!`, a Python `Traceback`, or `CUDA out of memory` in the logs. On an
  out-of-memory error with 40 GB cards, rerun with `--resume` (step 6); if it repeats, send the log.

### 6. After an interruption (reboot, killed session, crash)

```bash
RESUME=1 DATA_DIR=/same/path bash experiments/gate/local_fetch.sh
```

On a shared server, add `CUDA_VISIBLE_DEVICES=...` again. Use the same cards if they are still free; other free
cards of the same kind work too.

Setup reruns quickly (everything is cached). Each base skips the pauses already in its `samples.csv` and continues;
the summaries and diagnostics are redone; the final file is packed again. If a tmux session called `fetch` still
exists, close it first (`tmux kill-session -t fetch`).

### 7. What to send back

Exactly one file, printed at the end of the run:

```
kv_manager/results/fetch_study_results_<date-time>.tgz
```

It contains both bases' results (`metrics.csv`, `samples.csv`, `summary.md`, `spec_guesses.csv`, the diagnostic
folder), the smoke checks, all logs and an environment report (`nvidia-smi`, installed package versions). If the run failed part-way,
send the newest `.tgz` anyway, or the `results/fetch_logs/` folder if no `.tgz` was made.
