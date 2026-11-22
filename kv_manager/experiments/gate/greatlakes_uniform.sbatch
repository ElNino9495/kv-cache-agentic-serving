#!/bin/bash
# Uniform-only gate (int4/int3/int2 + bf16 control, no draft) on the first gate's 31 pauses, for
# Llama-3.1-8B, then Llama-3.2-3B, then Qwen3-8B. A40 48 GB is plenty (largest: Qwen3-8B, ~22 GB).
# One-time setup on the login node (same venv as greatlakes.sbatch), models pre-downloaded:
#   huggingface-cli download unsloth/Llama-3.2-3B-Instruct
#   huggingface-cli download Qwen/Qwen3-8B
#   cd ~/kv-cache-agentic-serving/kv_manager && python -m experiments.gate.data          # recordings, if not already there
# Set ACCOUNT (first line after the #SBATCH block below, or in the environment) to your own Slurm account:
#   cd ~/kv-cache-agentic-serving/kv_manager && ACCOUNT=<your_account> bash experiments/gate/greatlakes_uniform.sbatch    # submits itself
# Needs results/gate/samples.csv from the first gate run (in the repo).
# Resume after a timeout: ACCOUNT=<your_account> bash experiments/gate/greatlakes_uniform.sbatch --resume
# (`sbatch --account=<your_account> experiments/gate/greatlakes_uniform.sbatch` works too.)
#SBATCH --job-name=kv-gate-uniform
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=%x-%j.log
ACCOUNT="${ACCOUNT:-}"   # <- your Slurm account (e.g. a class or lab account); not hard-coded on purpose
set -euo pipefail
if [ -z "${SLURM_JOB_ID:-}" ]; then
  # not inside a job yet: submit this file with the chosen account, passing the settings along
  if [ -z "$ACCOUNT" ]; then
    echo "Set ACCOUNT to your Slurm account (ACCOUNT=<name> bash $0, or edit the line after the #SBATCH block)." >&2
    exit 1
  fi
  export MODELS="${MODELS:-}"
  [ -n "$MODELS" ] || unset MODELS
  exec sbatch --account="$ACCOUNT" --export=ALL "$0" "$@"
fi
cd "$SLURM_SUBMIT_DIR"
module load "${PY_MODULE:-python/3.12.1}"
source ~/gate-venv/bin/activate
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

MODELS="${MODELS:-unsloth/Llama-3.1-8B-Instruct unsloth/Llama-3.2-3B-Instruct Qwen/Qwen3-8B}"
for m in $MODELS; do
  slug=$(basename "$m" | tr '[:upper:]' '[:lower:]')
  if [[ " $* " != *" --resume "* ]]; then
    python -m experiments.gate.run_gate --uniform-only --target "$m" --max-samples 1 \
      --out-dir "results/gate_uniform_smoke/$slug"
  fi
  python -m experiments.gate.run_gate --uniform-only --target "$m" "$@"
done
python -m experiments.gate.summarize_uniform
tar czf results/gate_uniform_results.tgz results/gate_uniform
nvidia-smi --query-gpu=memory.used --format=csv
echo ">> done: results/gate_uniform_results.tgz (summary: results/gate_uniform/summary.md)"
