#!/usr/bin/env bash
# Fetch study on any Linux machine with one or more NVIDIA A100s (40 or 80 GB). No Slurm, no special mounts.
# Step-by-step guide for someone new to the project: docs/RUNBOOK_fetch_study.md.
#
#   cd kv_manager && bash experiments/gate/local_fetch.sh          # setup, memory check, start in tmux
#   bash experiments/gate/local_fetch.sh setup                     # only setup (venv, model, recordings)
#   bash experiments/gate/local_fetch.sh check                     # only the GPU / memory check
#   RESUME=1 bash experiments/gate/local_fetch.sh                  # continue after an interruption
#   tmux attach -t fetch                                           # watch (Ctrl-b n: next window, Ctrl-b d: detach)
#
# At startup it reads GPU count and memory with nvidia-smi:
#   - 2+ GPUs: base int2 on GPU 0 and base int1 on GPU 1 in parallel, one tmux window each;
#   - 1 GPU:   base int2, then base int1, in one tmux window;
#   - cards under 48 GB (A100 40 GB): the base-1 run skips the int3 history copy and its summary takes int3 from
#     the base-2 run (same pauses); on 80 GB cards base 1 keeps int3.
# Each base: one-pause smoke check, the full run (31 pauses), its summary, the per-step diagnostic of one pause.
# At the end everything is packed into ONE file: results/fetch_study_results_<RUN_ID>.tgz (the file to send back).
#
# Settings (environment variables):
#   DATA_DIR  where the venv, pip cache and model go (default ~/kvfetch; needs ~100 GB free)
#   HF_HOME   Hugging Face cache (default $DATA_DIR/hf)
#   PYTHON    Python 3.12 interpreter used to create the venv (default python3.12)
#   GPUS      GPU indices to use (default: all visible, first two are used)
#   SESSION   tmux session name (default fetch)
set -Eeuo pipefail

DATA_DIR="${DATA_DIR:-$HOME/kvfetch}"
export HF_HOME="${HF_HOME:-$DATA_DIR/hf}"
PYTHON="${PYTHON:-python3.12}"
TARGET="${TARGET:-unsloth/Llama-3.1-8B-Instruct}"
SESSION="${SESSION:-fetch}"
RESUME="${RESUME:-0}"
DIAG_PAUSE="${DIAG_PAUSE:-pydata__xarray-4687:85}"
CHECK_NLL_TOL="${CHECK_NLL_TOL:-0.05}"
TORCH_VERSION="${TORCH_VERSION:-2.14.1}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"

KV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$KV_DIR/experiments/gate/local_fetch.sh"
VENV="$DATA_DIR/venv"
PY="$VENV/bin/python"
export PIP_CACHE_DIR="$DATA_DIR/pip-cache"
SLUG="$(basename "$TARGET" | tr '[:upper:]' '[:lower:]')"
LOGS="$KV_DIR/results/fetch_logs"
MARK="$KV_DIR/results/gate_fetch/.status-$RUN_ID"

gpu_info() {  # "index memory_MiB" per visible GPU
  nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' '
}

pick_gpus() {  # GPUs to use (first two of GPUS, or of all visible)
  if [ -n "${GPUS:-}" ]; then echo "$GPUS"; else gpu_info | awk '{print $1}' | tr '\n' ' '; fi
}

min_mem_mib() {
  local g m min=""
  for g in $(pick_gpus); do
    m=$(gpu_info | awk -v g="$g" '$1 == g {print $2}')
    [ -z "$min" ] || [ "$m" -lt "$min" ] && min=$m
  done
  echo "${min:-0}"
}

n_gpus() { wc -w <<< "$(pick_gpus)" | tr -d ' '; }

refs_base1() {  # uniform references on base 1: drop the int3 copy on cards under 48 GB
  if [ "$(min_mem_mib)" -lt 49152 ]; then echo "2"; else echo "2 3"; fi
}

args_of() {  # run_fetch.py arguments for base $1, one per line
  printf '%s\n' --target "$TARGET" --base-bits "$1"
  if [ "$1" = 1 ]; then
    printf '%s\n' --uniform-refs
    # shellcheck disable=SC2046
    printf '%s\n' $(refs_base1)
  fi
  # shellcheck disable=SC2086
  if [ -n "$EXTRA_ARGS" ]; then printf '%s\n' $EXTRA_ARGS; fi
}

setup() {
  mkdir -p "$DATA_DIR" "$HF_HOME" "$PIP_CACHE_DIR" "$LOGS"
  command -v nvidia-smi >/dev/null || { echo "!! nvidia-smi not found: this needs an NVIDIA GPU with drivers" >&2; exit 1; }
  command -v tmux >/dev/null || { echo "!! tmux not found (Ubuntu/Debian: sudo apt-get install tmux)" >&2; exit 1; }
  command -v "$PYTHON" >/dev/null || { echo "!! $PYTHON not found; install Python 3.12 or set PYTHON=..." >&2; exit 1; }
  avail=$(df -BG --output=avail "$DATA_DIR" | tail -1 | tr -dc '0-9')
  echo ">> $avail GB free on $DATA_DIR"
  [ "$avail" -ge 60 ] || { echo "!! need ~100 GB free on DATA_DIR (at least 60 GB); set DATA_DIR elsewhere" >&2; exit 1; }
  [ -x "$PY" ] || "$PYTHON" -m venv "$VENV"
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
  "$PY" -m pip install -q -r "$KV_DIR/experiments/gate/requirements_fetch.txt"
  "$PY" -c "import sys, torch, transformers; assert torch.cuda.is_available(), 'torch sees no GPU'; \
print('python', sys.version.split()[0], '| torch', torch.__version__, '| transformers', transformers.__version__, \
'|', torch.cuda.device_count(), 'x', torch.cuda.get_device_name(0))"
  echo ">> downloading $TARGET (~16 GB) to $HF_HOME"
  "$PY" -c "from huggingface_hub import snapshot_download as d; print(d('$TARGET', allow_patterns=['*.json', '*.safetensors']))"
  cd "$KV_DIR" && "$PY" -m experiments.gate.data >/dev/null && echo ">> agent recordings downloaded"
}

check() {  # GPUs, plan and estimated peak memory per run (warns, never stops)
  cd "$KV_DIR"
  local b g i=0 a
  echo ">> GPUs:"; nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
  read -r -a gl <<< "$(pick_gpus)"
  if [ "$(n_gpus)" -ge 2 ]; then
    echo ">> plan: base int2 on GPU ${gl[0]} and base int1 on GPU ${gl[1]}, in parallel"
  else
    echo ">> plan: base int2 then base int1 on GPU ${gl[0]:-0}"
  fi
  echo ">> base-1 uniform references: int$(refs_base1 | sed 's/ /, int/g') (smallest card $(min_mem_mib) MiB)"
  for b in 2 1; do
    g=${gl[0]:-0}; [ "$(n_gpus)" -ge 2 ] && [ "$b" = 1 ] && g=${gl[1]}
    a=(); while IFS= read -r x; do a+=("$x"); done < <(args_of "$b")
    echo ">> base int$b on GPU $g:"
    CUDA_VISIBLE_DEVICES="$g" HF_HUB_OFFLINE=1 "$PY" -m experiments.gate.run_fetch "${a[@]}" --estimate-memory 2>&1 \
      | grep -E "estimated peak|WARNING" || true
  done
}

summarize() {  # summary of base $1; base 1 joins int3 from base 2 once base 2 is done
  local name="$SLUG-base$1"
  [ -f "results/gate_fetch/$name/metrics.csv" ] || return 0
  if [ "$1" = 1 ] && [[ " $(refs_base1) " != *" 3 "* ]] && [ -e "$MARK-base2.done" ] \
     && [ -f "results/gate_fetch/$SLUG-base2/metrics.csv" ]; then
    "$PY" -m experiments.gate.summarize_fetch --dir "results/gate_fetch/$name" --join "results/gate_fetch/$SLUG-base2"
  else
    "$PY" -m experiments.gate.summarize_fetch --dir "results/gate_fetch/$name"
  fi
}

run_base() {  # one base on the GPU in CUDA_VISIBLE_DEVICES
  local b="$1" a
  cd "$KV_DIR"
  export HF_HUB_OFFLINE=1
  mkdir -p results/gate_fetch
  trap 'touch "$MARK-base$b.failed"' ERR
  a=(); while IFS= read -r x; do a+=("$x"); done < <(args_of "$b")
  echo ">> base int$b on GPU ${CUDA_VISIBLE_DEVICES:-?}: results/gate_fetch/$SLUG-base$b ($(date))"
  if [ "$RESUME" = 1 ]; then
    "$PY" -m experiments.gate.run_fetch "${a[@]}" --resume
  else
    "$PY" -m experiments.gate.run_fetch "${a[@]}" --max-samples 1 --check-nll-tol "$CHECK_NLL_TOL" \
      --out-dir "results/gate_fetch_smoke/$SLUG-base$b"
    "$PY" -m experiments.gate.run_fetch "${a[@]}"
  fi
  summarize "$b"
  "$PY" -m experiments.gate.run_fetch "${a[@]}" --diagnose "$DIAG_PAUSE"
  touch "$MARK-base$b.done"
  trap - ERR
  echo ">> base int$b finished ($(date))"
}

finish() {  # wait for both bases, redo the base-1 summary with int3 joined, pack everything into one file
  cd "$KV_DIR"
  local b
  for b in 2 1; do
    until [ -e "$MARK-base$b.done" ] || [ -e "$MARK-base$b.failed" ]; do
      echo ">> waiting for base int$b ($(date +%H:%M))"; sleep 60
    done
    [ -e "$MARK-base$b.failed" ] && echo "!! base int$b failed; its log is in results/fetch_logs/" >&2
  done
  summarize 1
  { echo "run $RUN_ID, $(date)"; nvidia-smi; "$PY" -m pip freeze; } > "$LOGS/environment_$RUN_ID.txt" 2>&1 || true
  local out="results/fetch_study_results_$RUN_ID.tgz" paths=()
  shopt -s nullglob
  paths=(results/gate_fetch/"$SLUG"-base* results/gate_fetch_smoke results/fetch_logs)
  shopt -u nullglob
  tar czf "$out" "${paths[@]}"
  echo ">> DONE. Send back this one file: $KV_DIR/$out ($(du -h "$out" | cut -f1))"
}

window_cmd() {  # tmux window running "$@" with this launch's settings, logged
  local log="$1"; shift
  printf "DATA_DIR='%s' HF_HOME='%s' PYTHON='%s' TARGET='%s' RESUME='%s' DIAG_PAUSE='%s' CHECK_NLL_TOL='%s' \
EXTRA_ARGS='%s' RUN_ID='%s' GPUS='%s' CUDA_VISIBLE_DEVICES='%s' bash '%s' %s 2>&1 | tee -a '%s'; \
echo '>> window finished; press Enter to close'; read" \
    "$DATA_DIR" "$HF_HOME" "$PYTHON" "$TARGET" "$RESUME" "$DIAG_PAUSE" "$CHECK_NLL_TOL" "$EXTRA_ARGS" "$RUN_ID" \
    "$(pick_gpus)" "$GPU_ENV" "$SCRIPT" "$*" "$log"
}

launch() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "!! tmux session '$SESSION' already exists; attach with: tmux attach -t $SESSION" >&2; exit 1
  fi
  mkdir -p "$LOGS"
  read -r -a gl <<< "$(pick_gpus)"
  if [ "$(n_gpus)" -ge 2 ]; then
    tmux new-session -d -s "$SESSION" -n base2 "$(GPU_ENV=${gl[0]} window_cmd "$LOGS/base2_$RUN_ID.log" run-base 2)"
    tmux new-window -t "$SESSION" -n base1 "$(GPU_ENV=${gl[1]} window_cmd "$LOGS/base1_$RUN_ID.log" last 1)"
    echo ">> started: base int2 on GPU ${gl[0]}, base int1 on GPU ${gl[1]} (tmux session '$SESSION', windows base2, base1)"
  else
    tmux new-session -d -s "$SESSION" -n run "$(GPU_ENV=${gl[0]:-0} window_cmd "$LOGS/run_$RUN_ID.log" run)"
    echo ">> started: base int2 then base int1 on GPU ${gl[0]:-0} (tmux session '$SESSION')"
  fi
  echo "   watch: tmux attach -t $SESSION   (Ctrl-b n: next window, Ctrl-b d: detach; the run keeps going)"
  echo "   logs:  $LOGS/"
}

case "${1:-all}" in
  setup) setup ;;
  check) check ;;
  run-base) run_base "$2" ;;                                                         # internal
  last) bash "$SCRIPT" run-base "$2" || true; finish ;;                             # internal: base, then pack
  run) bash "$SCRIPT" run-base 2 || true; bash "$SCRIPT" run-base 1 || true; finish ;; # internal: one GPU
  launch) launch ;;                                                       # internal
  all) setup; check; launch ;;
  *) echo "usage: $0 [all|setup|check]" >&2; exit 2 ;;
esac
