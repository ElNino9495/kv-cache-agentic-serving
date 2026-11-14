#!/usr/bin/env bash
# Per-step fetch study on a CloudLab GPU node (plain bash, no Slurm). Same code and flags as
# greatlakes_fetch.sbatch.
#
#   cd kv_manager && bash experiments/gate/cloudlab_fetch.sh            # setup, then start the run in tmux
#   bash experiments/gate/cloudlab_fetch.sh setup                       # only setup (venv, models, data)
#   MODE=diagnose bash experiments/gate/cloudlab_fetch.sh               # only the xarray diagnostic
#   RESUME=1 bash experiments/gate/cloudlab_fetch.sh                    # continue an interrupted run
#   tmux attach -t fetch                                                # watch it (detach: Ctrl-b d)
#
# Everything large (venv, pip cache, HF_HOME with the model, temp files) goes on DATA_ROOT, the node's
# large local disk. On CloudLab that is usually an extra partition mounted with mkextrafs at /mydata;
# the script mounts it if the mkextrafs tool is there and DATA_ROOT is not mounted yet (needs sudo).
# Needs ~40 GB free on DATA_ROOT (model 16 GB, venv ~6 GB) and one GPU with >= 40 GB (A40/A100/L40S/H100).
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/mydata}"
TARGET="${TARGET:-unsloth/Llama-3.1-8B-Instruct}"
MODE="${MODE:-both}"                         # full | diagnose | both
DIAG_PAUSE="${DIAG_PAUSE:-pydata__xarray-4687:85}"
CHECK_NLL_TOL="${CHECK_NLL_TOL:-0.05}"        # smoke run stops if per-step and sdpa NLL differ by more
SESSION="${SESSION:-fetch}"
RESUME="${RESUME:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"                 # passed to run_fetch.py, e.g. "--ks 16 64"

KV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$KV_DIR/experiments/gate/cloudlab_fetch.sh"
VENV="$DATA_ROOT/gate-venv"
export HF_HOME="$DATA_ROOT/hf"
export PIP_CACHE_DIR="$DATA_ROOT/pip-cache"
export TMPDIR="$DATA_ROOT/tmp"
SLUG="$(basename "$TARGET" | tr '[:upper:]' '[:lower:]')"
LOG="$KV_DIR/results/gate_fetch_cloudlab_${SLUG}.log"

setup() {
  if ! mountpoint -q "$DATA_ROOT" && [ -x /usr/local/etc/emulab/mkextrafs.pl ]; then
    echo ">> mounting the extra local disk at $DATA_ROOT"
    sudo mkdir -p "$DATA_ROOT"
    sudo /usr/local/etc/emulab/mkextrafs.pl "$DATA_ROOT"
  fi
  sudo mkdir -p "$DATA_ROOT" && sudo chown "$(id -u):$(id -g)" "$DATA_ROOT"
  mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR" "$KV_DIR/results"
  df -h "$DATA_ROOT"
  avail=$(df -BG --output=avail "$DATA_ROOT" | tail -1 | tr -dc '0-9')
  if [ "$avail" -lt 40 ]; then
    echo "!! only ${avail} GB free on $DATA_ROOT; need ~40 GB. Set DATA_ROOT to a larger disk." >&2
    exit 1
  fi

  missing=()
  command -v tmux >/dev/null || missing+=(tmux)
  python3 -c "import venv, ensurepip" 2>/dev/null || missing+=("python3-venv")
  if [ ${#missing[@]} -gt 0 ]; then
    sudo apt-get update -q && sudo apt-get install -y -q "${missing[@]}"
  fi
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

  if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV"
  fi
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q torch "transformers==5.14.1" pandas matplotlib numpy
  "$VENV/bin/python" -c "import torch; assert torch.cuda.is_available(), 'torch sees no GPU'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

  echo ">> downloading $TARGET to $HF_HOME"
  "$VENV/bin/python" -c "from huggingface_hub import snapshot_download as d; print(d('$TARGET', allow_patterns=['*.json', '*.safetensors']))"
  cd "$KV_DIR"
  "$VENV/bin/python" -m experiments.gate.data
}

run() {
  cd "$KV_DIR"
  export HF_HUB_OFFLINE=1
  PY="$VENV/bin/python"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  date
  if [ "$MODE" = full ] || [ "$MODE" = both ]; then
    if [ "$RESUME" = 1 ]; then
      # shellcheck disable=SC2086
      $PY -m experiments.gate.run_fetch --target "$TARGET" --resume $EXTRA_ARGS
    else
      # shellcheck disable=SC2086
      $PY -m experiments.gate.run_fetch --target "$TARGET" --max-samples 1 --check-nll-tol "$CHECK_NLL_TOL" \
        --out-dir results/gate_fetch_smoke $EXTRA_ARGS
      # shellcheck disable=SC2086
      $PY -m experiments.gate.run_fetch --target "$TARGET" $EXTRA_ARGS
    fi
    $PY -m experiments.gate.summarize_fetch --dir "results/gate_fetch/$SLUG"
  fi
  if [ "$MODE" = diagnose ] || [ "$MODE" = both ]; then
    $PY -m experiments.gate.run_fetch --target "$TARGET" --diagnose "$DIAG_PAUSE"
  fi
  tar czf results/gate_fetch_results.tgz results/gate_fetch
  nvidia-smi --query-gpu=memory.used --format=csv
  date
  echo ">> done: $KV_DIR/results/gate_fetch_results.tgz (summary: results/gate_fetch/$SLUG/summary.md)"
}

case "${1:-all}" in
  setup) setup ;;
  run) run ;;
  all)
    setup
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "!! tmux session '$SESSION' already exists; attach with: tmux attach -t $SESSION" >&2
      exit 1
    fi
    tmux new-session -d -s "$SESSION" \
      "DATA_ROOT='$DATA_ROOT' TARGET='$TARGET' MODE='$MODE' DIAG_PAUSE='$DIAG_PAUSE' CHECK_NLL_TOL='$CHECK_NLL_TOL' \
       RESUME='$RESUME' EXTRA_ARGS='$EXTRA_ARGS' bash '$SCRIPT' run 2>&1 | tee -a '$LOG'; \
       echo '>> finished; press Enter to close'; read"
    echo ">> started in tmux session '$SESSION' (log: $LOG)"
    echo "   attach: tmux attach -t $SESSION    detach: Ctrl-b d"
    ;;
  *) echo "usage: $0 [all|setup|run]" >&2; exit 2 ;;
esac
