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
# GPU selection (shared servers): GPUS=1,2 or CUDA_VISIBLE_DEVICES=1,2 limits the run to those cards (GPUS wins
# if both are set; default: every GPU nvidia-smi sees). Only the first two listed are used, and only they are
# queried (nvidia-smi --id). Indices follow nvidia-smi's numbering (CUDA_DEVICE_ORDER=PCI_BUS_ID is set).
#   CUDA_VISIBLE_DEVICES=1,2 bash experiments/gate/local_fetch.sh
# It refuses to start if a chosen card already has more than MAX_USED_MIB (default 1024 MiB) in use, both before
# setup and again right before launching.
#   - 2 GPUs: base int2 on the first and base int1 on the second in parallel, one tmux window each;
#   - 1 GPU:  base int2, then base int1, in one tmux window;
#   - cards under ~44 GiB (A100 40 GB): the base-1 run skips the int3 history copy and its summary takes int3 from
#     the base-2 run (same pauses); 48 GB (RTX A6000, A40) and 80 GB cards keep int3.
# Each base: one-pause smoke check, the full run (31 pauses), its summary, the per-step diagnostic of one pause.
# At the end everything is packed into ONE file: results/fetch_study_results_<RUN_ID>.tgz (the file to send back).
#
# Settings (environment variables):
#   DATA_DIR  where the venv, pip cache and model go (default ~/kvfetch; needs ~100 GB free)
#   HF_HOME   Hugging Face cache (default $DATA_DIR/hf)
#   PYTHON    Python 3.12 interpreter used to create the venv (default python3.12)
#   GPUS      GPU indices or UUIDs to use, comma or space separated (default: CUDA_VISIBLE_DEVICES, else all;
#             the first two are used)
#   MAX_USED_MIB  refuse to start on a chosen card with more memory than this already in use (default 1024)
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
MAX_USED_MIB="${MAX_USED_MIB:-1024}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"   # CUDA numbers GPUs like nvidia-smi does
USE_GPUS=""

KV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$KV_DIR/experiments/gate/local_fetch.sh"
VENV="$DATA_DIR/venv"
PY="$VENV/bin/python"
export PIP_CACHE_DIR="$DATA_DIR/pip-cache"
SLUG="$(basename "$TARGET" | tr '[:upper:]' '[:lower:]')"
LOGS="$KV_DIR/results/fetch_logs"
MARK="$KV_DIR/results/gate_fetch/.status-$RUN_ID"

gpu_field() {  # gpu_field <id> <field>: one nvidia-smi query field of one card (index or UUID)
  nvidia-smi --id="$1" --query-gpu="$2" --format=csv,noheader,nounits | tr -d ' '
}

resolve_gpus() {  # sets USE_GPUS: the first two of GPUS, else of CUDA_VISIBLE_DEVICES, else of all cards
  command -v nvidia-smi >/dev/null || { echo "!! nvidia-smi not found: this needs an NVIDIA GPU with drivers" >&2; exit 1; }
  local l g ids=()
  if [ -n "${GPUS:-}" ]; then
    l="$GPUS"
  elif [ -n "${CUDA_VISIBLE_DEVICES+x}" ]; then
    [ -n "$CUDA_VISIBLE_DEVICES" ] || {
      echo "!! CUDA_VISIBLE_DEVICES is set but empty (no GPU); unset it or name cards, e.g. CUDA_VISIBLE_DEVICES=1,2" >&2
      exit 1; }
    l="$CUDA_VISIBLE_DEVICES"
  else
    l="$(nvidia-smi --query-gpu=index --format=csv,noheader)"
  fi
  read -r -a ids <<< "$(printf '%s' "$l" | tr ',\n' '  ')"
  [ "${#ids[@]}" -ge 1 ] || { echo "!! no GPU found" >&2; exit 1; }
  USE_GPUS="${ids[*]:0:2}"
  for g in $USE_GPUS; do
    gpu_field "$g" index >/dev/null 2>&1 || { echo "!! nvidia-smi does not know GPU '$g' (check GPUS / CUDA_VISIBLE_DEVICES)" >&2; exit 1; }
  done
}

gpu_csv() { echo "$USE_GPUS" | tr ' ' ','; }

require_idle() {  # stop (or with "warn", only report) if a chosen card already has > MAX_USED_MIB in use
  local g used busy=0
  for g in $USE_GPUS; do
    used=$(gpu_field "$g" memory.used)
    if [ "$used" -gt "$MAX_USED_MIB" ]; then
      echo "!! GPU $g already has $used MiB in use (limit $MAX_USED_MIB MiB): it looks busy." >&2; busy=1
    fi
  done
  if [ "$busy" = 0 ]; then echo ">> GPU(s) $USE_GPUS idle (at most $MAX_USED_MIB MiB in use each)"; return 0; fi
  [ "${1:-}" = warn ] && return 0
  echo "   Find idle cards with nvidia-smi and choose them, e.g.: CUDA_VISIBLE_DEVICES=1,2 bash $SCRIPT" >&2
  exit 1
}

pick_gpus() { echo "$USE_GPUS"; }

min_mem_mib() {
  local g m min=""
  for g in $USE_GPUS; do
    m=$(gpu_field "$g" memory.total)
    if [ -z "$min" ] || [ "$m" -lt "$min" ]; then min=$m; fi
  done
  echo "${min:-0}"
}

n_gpus() { wc -w <<< "$USE_GPUS" | tr -d ' '; }

refs_base1() {  # uniform references on base 1: drop the int3 copy below ~44 GiB (A100 40 GB = 40960 MiB)
  if [ "$(min_mem_mib)" -lt 45000 ]; then echo "2"; else echo "2 3"; fi
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
  echo ">> GPUs used:"; nvidia-smi --id="$(gpu_csv)" --query-gpu=index,name,memory.used,memory.total,driver_version --format=csv
  require_idle warn
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
  { echo "run $RUN_ID, $(date), GPUs $USE_GPUS"; nvidia-smi --id="$(gpu_csv)"; "$PY" -m pip freeze; } > "$LOGS/environment_$RUN_ID.txt" 2>&1 || true
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
  require_idle   # again: setup can take a while on a shared machine
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
  check) resolve_gpus; check ;;
  run-base) resolve_gpus; run_base "$2" ;;                                                         # internal
  last) resolve_gpus; bash "$SCRIPT" run-base "$2" || true; finish ;;                             # internal: base, then pack
  run) resolve_gpus; bash "$SCRIPT" run-base 2 || true; bash "$SCRIPT" run-base 1 || true; finish ;; # internal: one GPU
  launch) resolve_gpus; launch ;;                                                       # internal
  all) resolve_gpus; require_idle; setup; check; launch ;;
  *) echo "usage: $0 [all|setup|check]" >&2; exit 2 ;;
esac
