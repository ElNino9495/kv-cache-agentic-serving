#!/usr/bin/env bash
# One-shot gate run on a CUDA machine (needs ~50 GB GPU memory: A100/H100 80GB).
# Lightweight env: torch + transformers only; the full vLLM env (envt/setup_env.sh) also works.
#
#   cd kv_manager && bash experiments/gate/run_on_gpu.sh            # smoke test, full run, summary
#   SKIP_SETUP=1 bash experiments/gate/run_on_gpu.sh --resume       # continue an interrupted run
set -euo pipefail
cd "$(dirname "$0")/../.."

if [ -z "${SKIP_SETUP:-}" ]; then
  python3 -m venv .gate-venv
  .gate-venv/bin/pip install -q torch "transformers==5.14.1" pandas matplotlib numpy
fi
PY=.gate-venv/bin/python
[ -x "$PY" ] || PY=python3
nvidia-smi --query-gpu=name,memory.total --format=csv

mkdir -p results
$PY -m experiments.gate.data
if [[ " $* " != *" --resume "* ]]; then
  $PY -m experiments.gate.run_gate --max-samples 1 --hist-lens 4000 --out-dir results/gate_smoke
fi
$PY -m experiments.gate.run_gate "$@" 2>&1 | tee -a results/gate_run.log
$PY -m experiments.gate.plot_gate
tar czf results/gate_results.tgz results/gate results/gate_run.log
echo ">> send back: $(pwd)/results/gate_results.tgz  (summary: results/gate/summary.md)"
