#!/bin/bash
# Per-step fetch study (run_fetch.py) on the first gate run's 31 pauses, Llama-3.1-8B, one A40 48 GB.
# Peak GPU memory ~35-40 GB at the longest pause (weights + bf16/int2/int3 history copies + scratch).
# Same one-time setup as greatlakes.sbatch (venv at ~/gate-venv, model pre-downloaded, recordings in data/).
#   cd ~/kv_manager && sbatch experiments/gate/greatlakes_fetch.sbatch
#   resume after a timeout:  sbatch experiments/gate/greatlakes_fetch.sbatch --resume
# The smoke run on the first pause stops the job if per-step decoding and sdpa teacher forcing disagree
# by more than CHECK_TOL in any token's log-prob. Local checks on the first pause: Llama-3.2-1B fp32
# 2.1e-5 (logic exact); Llama-3.2-3B bf16 0.030 (bf16 row) / 0.098 (int2 row), i.e. bf16 rounding.
# A logic error (wrong segment, mask or position) gives O(1) differences; 0.25 leaves ~2.5x headroom.
#SBATCH --job-name=kv-gate-fetch
#SBATCH --account=cse595f26_class
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=%x-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load "${PY_MODULE:-python/3.12.1}"
source ~/gate-venv/bin/activate
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

TARGET="${TARGET:-unsloth/Llama-3.1-8B-Instruct}"
CHECK_TOL="${CHECK_TOL:-0.25}"
if [[ " $* " != *" --resume "* ]]; then
  python -m experiments.gate.run_fetch --target "$TARGET" --max-samples 1 --check-tol "$CHECK_TOL" \
    --out-dir results/gate_fetch_smoke
fi
python -m experiments.gate.run_fetch --target "$TARGET" "$@"
slug=$(basename "$TARGET" | tr '[:upper:]' '[:lower:]')
python -m experiments.gate.summarize_fetch --dir "results/gate_fetch/$slug"
tar czf results/gate_fetch_results.tgz results/gate_fetch
nvidia-smi --query-gpu=memory.used --format=csv
echo ">> done: results/gate_fetch_results.tgz (summary: results/gate_fetch/$slug/summary.md)"
