#!/bin/bash
# Per-step fetch study (run_fetch.py) on the first gate run's 31 pauses, Llama-3.1-8B, on Great Lakes A40s
# (48 GB). Base int2 and base int1 run as TWO SEPARATE JOBS, one A40 each, both with prefill fetch (default
# PREFILL_FETCH_K=128; fetch_all_noprefill is the same-run comparison without it).
#
# Set ACCOUNT (first line after the #SBATCH block below, or in the environment) to your own Slurm account, then
# run this file with bash; it submits one job per base and prints the run id:
#   cd ~/kv_manager && ACCOUNT=<your_account> bash experiments/gate/greatlakes_fetch.sbatch
#   ACCOUNT=<your_account> MODE=diagnose bash experiments/gate/greatlakes_fetch.sbatch      # xarray diagnostic only
#   ACCOUNT=<your_account> RESUME=1 RUN_ID=<run id> BASES=1 bash experiments/gate/greatlakes_fetch.sbatch
#       continue one base after a timeout (same RUN_ID, so the int3 join below still finds this run's base 2)
# (Submitting with plain sbatch also works for one base: sbatch --account=<acct> --export=ALL,BASE=2 <this file>.)
#
# Every job runs, for its base: a one-pause smoke check (stops the job if per-step decoding and sdpa teacher
# forcing disagree by more than CHECK_NLL_TOL nats in either row's NLL), the full run, its summary, and the
# per-step diagnostic of one pause (MODE: full | diagnose | both, default both).
# Rows, all in the same run: bf16, the uniform base, uniform references (base 2: int3; base 1: int2), fetch_all,
# fetch_all_noprefill, noise_ctrl, and per k in 16/32/64/128/match: s1, s2, spec, spec_chain (each with its
# speculative shadow), sens_a, sens_b, s2sens_a, s2sens_b, oracle, s1_layer.
# Base 1 skips its int3 history copy (A40 headroom) and its summary takes int3 from THIS run's base 2 (same
# pauses): whichever job finishes second writes the joined base-1 summary, and only if the base-2 results carry
# this RUN_ID. By hand (login node): python -m experiments.gate.summarize_fetch \
#     --dir results/gate_fetch/llama-3.1-8b-instruct-base1-pf128 --join results/gate_fetch/llama-3.1-8b-instruct-base2-pf128
#
# Memory (estimate + the ~2 GB the A100 run measured above it): ~39 GB per job at the longest pause (34k history);
# an A40 has ~47.5 GB usable. Base 1 with int3 would be ~44 GB.
# Time: the A100 80 GB run took 27-29 min per base (55 rows, no prefill fetch, plus ~3 min smoke/diagnostic/
# loading). The A40 was ~1.5x slower per row in run 1 (45 s vs 30 s per row over 31 pauses); with 66 rows that is
# ~50 min, plus prefill fetch (tool outputs up to 2.9k tokens) and the smoke/diagnostic: ~60-75 min per job.
# Limit 3 h = 2.4x margin.
# One-time setup as for greatlakes.sbatch (venv at ~/gate-venv, model pre-downloaded, recordings in data/).
#SBATCH --job-name=kv-gate-fetch
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=03:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=%x-%j.log
ACCOUNT="${ACCOUNT:-}"   # <- your Slurm account (e.g. a class or lab account); not hard-coded on purpose
set -Eeuo pipefail

TARGET="${TARGET:-unsloth/Llama-3.1-8B-Instruct}"
MODE="${MODE:-both}"
DIAG_PAUSE="${DIAG_PAUSE:-pydata__xarray-4687:85}"
CHECK_NLL_TOL="${CHECK_NLL_TOL:-0.05}"
RESUME="${RESUME:-0}"
BASES="${BASES:-2 1}"
PREFILL_FETCH_K="${PREFILL_FETCH_K:-128}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"

if [ -z "${SLURM_JOB_ID:-}" ]; then
  # not inside a job: submit one job per base with the chosen account, passing the settings along
  if [ -z "$ACCOUNT" ]; then
    echo "Set ACCOUNT to your Slurm account (ACCOUNT=<name> bash $0, or edit the line after the #SBATCH block)." >&2
    exit 1
  fi
  export TARGET MODE DIAG_PAUSE CHECK_NLL_TOL RESUME PREFILL_FETCH_K RUN_ID
  for b in $BASES; do
    BASE="$b" sbatch --account="$ACCOUNT" --job-name="kv-fetch-b$b" --export=ALL "$0" "$@"
  done
  echo ">> run id $RUN_ID: one job per base ($BASES); logs kv-fetch-b<base>-<jobid>.log in $(pwd)"
  echo "   resume a base after a timeout: ACCOUNT=$ACCOUNT RESUME=1 RUN_ID=$RUN_ID BASES=<base> bash $0"
  exit 0
fi

[ -n "${BASE:-}" ] || { echo "BASE is not set: submit with 'bash $0' (one job per base) or --export=ALL,BASE=<1|2>" >&2; exit 1; }
cd "$SLURM_SUBMIT_DIR"
module load "${PY_MODULE:-python/3.12.1}"
source ~/gate-venv/bin/activate
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
slug=$(basename "$TARGET" | tr '[:upper:]' '[:lower:]')
pfx=""
[ "$PREFILL_FETCH_K" != 0 ] && pfx="-pf$PREFILL_FETCH_K"
dir_of() { echo "results/gate_fetch/$slug-base$1$pfx"; }
mark="results/gate_fetch/.status-$RUN_ID"
mkdir -p results/gate_fetch
trap 'touch "$mark-base$BASE.failed"' ERR

b="$BASE"
name="$slug-base$b$pfx"
common=(--target "$TARGET" --base-bits "$b")
[ "$PREFILL_FETCH_K" != 0 ] && common+=(--prefill-fetch-k "$PREFILL_FETCH_K")
[ "$b" = 1 ] && common+=(--uniform-refs 2)   # no int3 copy on base 1; int3 is joined from this run's base 2
echo ">> run $RUN_ID, base int$b on $(hostname): $(dir_of "$b") ($(date))"

if [ "$MODE" = full ] || [ "$MODE" = both ]; then
  if [ "$RESUME" = 1 ]; then
    python -m experiments.gate.run_fetch "${common[@]}" --resume "$@"
  else
    python -m experiments.gate.run_fetch "${common[@]}" --max-samples 1 --check-nll-tol "$CHECK_NLL_TOL" \
      --out-dir "results/gate_fetch_smoke/$name" "$@"
    python -m experiments.gate.run_fetch "${common[@]}" "$@"
  fi
  echo "$RUN_ID" > "$(dir_of "$b")/run_id.txt"
  python -m experiments.gate.summarize_fetch --dir "$(dir_of "$b")"
fi
if [ "$MODE" = diagnose ] || [ "$MODE" = both ]; then
  python -m experiments.gate.run_fetch "${common[@]}" --diagnose "$DIAG_PAUSE"
fi
touch "$mark-base$b.done"
trap - ERR

# the job that finishes second writes base 1's summary with int3 from this run's base 2, and packs both bases
if [ "$MODE" != diagnose ] && [ -e "$mark-base1.done" ] && [ -e "$mark-base2.done" ]; then
  d1=$(dir_of 1); d2=$(dir_of 2)
  if [ "$(cat "$d2/run_id.txt" 2>/dev/null)" = "$RUN_ID" ] && [ "$(cat "$d1/run_id.txt" 2>/dev/null)" = "$RUN_ID" ]; then
    python -m experiments.gate.summarize_fetch --dir "$d1" --join "$d2"
    shopt -s nullglob
    tar czf "results/fetch_study_results_$RUN_ID.tgz" "$d1" "$d2" results/gate_fetch_smoke/"$slug"-base*"$pfx" \
      kv-fetch-b*.log
    shopt -u nullglob
    echo ">> both bases done: $d1/summary.md (int3 joined from $d2), results/fetch_study_results_$RUN_ID.tgz"
  else
    echo "!! not joining: $d1 or $d2 does not carry run id $RUN_ID (run_id.txt); join by hand if they belong together" >&2
  fi
else
  echo ">> base int$b done; base 1's joined summary is written when the other base finishes"
fi
nvidia-smi --query-gpu=memory.used --format=csv
