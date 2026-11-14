#!/bin/bash
# Per-step fetch study (run_fetch.py) on the first gate run's 31 pauses, Llama-3.1-8B, one A40 48 GB on Great
# Lakes. Same code and flags as cloudlab_fetch.sh (CloudLab, plain bash + tmux).
#
# Set ACCOUNT (first line after the #SBATCH block below, or in the environment) to your own Slurm account, then:
#   cd ~/kv_manager && ACCOUNT=<your_account> bash experiments/gate/greatlakes_fetch.sbatch       # submits itself
#   ACCOUNT=<your_account> MODE=diagnose bash experiments/gate/greatlakes_fetch.sbatch           # xarray diagnostic only
#   ACCOUNT=<your_account> RESUME=1 bash experiments/gate/greatlakes_fetch.sbatch                 # continue after a timeout
# (`sbatch --account=<your_account> experiments/gate/greatlakes_fetch.sbatch` works too.)
#
# MODE: full (all pauses + summary) | diagnose (one pause, fetch logs) | both (default).
# Rows: bf16, int2, int3, fetch_all, noise_ctrl, and per k in 16/32/64/128: s1, s2, spec (+ its speculative
# shadow), sens_a, sens_b, s2sens_a, s2sens_b, oracle, s1_layer.
# One-time setup as for greatlakes.sbatch (venv at ~/gate-venv, model pre-downloaded, recordings in data/).
# Peak GPU memory ~35-40 GB at the longest pause. The smoke run on the first pause stops the job if per-step
# decoding and sdpa teacher forcing disagree by more than CHECK_NLL_TOL nats in either row's NLL (the last run's
# largest per-pause difference was 0.0115; a logic error moves NLL by much more).
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
set -euo pipefail

TARGET="${TARGET:-unsloth/Llama-3.1-8B-Instruct}"
MODE="${MODE:-both}"
DIAG_PAUSE="${DIAG_PAUSE:-pydata__xarray-4687:85}"
CHECK_NLL_TOL="${CHECK_NLL_TOL:-0.05}"
RESUME="${RESUME:-0}"

if [ -z "${SLURM_JOB_ID:-}" ]; then
  # not inside a job yet: submit this file with the chosen account, passing the settings along
  if [ -z "$ACCOUNT" ]; then
    echo "Set ACCOUNT to your Slurm account (ACCOUNT=<name> bash $0, or edit the line after the #SBATCH block)." >&2
    exit 1
  fi
  export TARGET MODE DIAG_PAUSE CHECK_NLL_TOL RESUME
  exec sbatch --account="$ACCOUNT" --export=ALL "$0" "$@"
fi

cd "$SLURM_SUBMIT_DIR"
module load "${PY_MODULE:-python/3.12.1}"
source ~/gate-venv/bin/activate
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
slug=$(basename "$TARGET" | tr '[:upper:]' '[:lower:]')

if [ "$MODE" = full ] || [ "$MODE" = both ]; then
  if [ "$RESUME" = 1 ]; then
    python -m experiments.gate.run_fetch --target "$TARGET" --resume "$@"
  else
    python -m experiments.gate.run_fetch --target "$TARGET" --max-samples 1 --check-nll-tol "$CHECK_NLL_TOL" \
      --out-dir results/gate_fetch_smoke "$@"
    python -m experiments.gate.run_fetch --target "$TARGET" "$@"
  fi
  python -m experiments.gate.summarize_fetch --dir "results/gate_fetch/$slug"
fi
if [ "$MODE" = diagnose ] || [ "$MODE" = both ]; then
  python -m experiments.gate.run_fetch --target "$TARGET" --diagnose "$DIAG_PAUSE"
fi
tar czf results/gate_fetch_results.tgz results/gate_fetch
nvidia-smi --query-gpu=memory.used --format=csv
echo ">> done: results/gate_fetch_results.tgz (summary: results/gate_fetch/$slug/summary.md)"
