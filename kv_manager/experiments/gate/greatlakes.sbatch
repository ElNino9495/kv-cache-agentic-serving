#!/bin/bash
# Gate experiment on UMich Great Lakes (A40 48 GB on spgpu; needs ~30 GB GPU memory).
# One-time setup on the login node: venv at ~/gate-venv + models pre-downloaded (HF_HUB_OFFLINE=1).
#   cd ~/kv_manager && sbatch experiments/gate/greatlakes.sbatch
#   resume after a timeout:  sbatch experiments/gate/greatlakes.sbatch --resume
#SBATCH --job-name=kv-gate
#SBATCH --account=cse595f26_class
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=08:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=%x-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load "${PY_MODULE:-python/3.12.1}"
source ~/gate-venv/bin/activate
export HF_HUB_OFFLINE=1          # models were downloaded on the login node
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

mkdir -p results
if [[ " $* " != *" --resume "* ]]; then
  python -m experiments.gate.run_gate --max-samples 1 --hist-lens 4000 --out-dir results/gate_smoke
fi
python -m experiments.gate.run_gate "$@"
python -m experiments.gate.plot_gate
# the job log is still being written, so it is not packed (copy it separately)
tar czf results/gate_results.tgz results/gate
nvidia-smi --query-gpu=memory.used --format=csv
echo ">> done: results/gate_results.tgz"
