#!/bin/bash
# Uniform-only gate (int4/int3/int2 + bf16 control, no draft) on the first gate's 31 pauses, for
# Llama-3.1-8B, then Llama-3.2-3B, then Qwen3-8B. A40 48 GB is plenty (largest: Qwen3-8B, ~22 GB).
# One-time setup on the login node (same venv as greatlakes.sbatch), models pre-downloaded:
#   huggingface-cli download unsloth/Llama-3.2-3B-Instruct
#   huggingface-cli download Qwen/Qwen3-8B
#   cd ~/kv_manager && python -m experiments.gate.data          # recordings, if not already there
#   cd ~/kv_manager && sbatch experiments/gate/greatlakes_uniform.sbatch
# Needs results/gate/samples.csv from the first gate run (in the repo).
# Resume after a timeout: sbatch experiments/gate/greatlakes_uniform.sbatch --resume
#SBATCH --job-name=kv-gate-uniform
#SBATCH --account=cse595f26_class
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=%x-%j.log
set -euo pipefail
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
