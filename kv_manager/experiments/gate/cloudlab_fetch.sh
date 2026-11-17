#!/usr/bin/env bash
# Per-step fetch study on a CloudLab GPU node (plain bash, no Slurm). Same code and flags as
# greatlakes_fetch.sbatch. Runs base int2 and base int1 (BASES="2 1") either in parallel, one GPU and one
# tmux window per base (GPUS="0 1", default when the node has enough GPUs, e.g. a d8545 with 4x A100 40 GB),
# or back to back in one window. Each base gets a smoke check, the full run, its summary and the xarray
# diagnostic, in results/gate_fetch/<model>-base<b>/.
#
# On base 1 the int3 history copy is skipped (REFS_BASE1="2") so the run fits a 40 GB card; when base 2 runs
# too, the base-1 summary joins int3 from the base-2 results on the same pauses once base 2 has finished.
# At startup the script estimates each run's peak GPU memory and warns if it exceeds the card.
#
#   cd kv_manager && bash experiments/gate/cloudlab_fetch.sh            # setup, memory check, start in tmux
#   bash experiments/gate/cloudlab_fetch.sh setup                       # only setup (venv, models, data)
#   bash experiments/gate/cloudlab_fetch.sh check-memory                # only the memory estimate
#   PARALLEL=0 bash experiments/gate/cloudlab_fetch.sh                  # bases back to back on GPU 0
#   MODE=diagnose bash experiments/gate/cloudlab_fetch.sh               # only the xarray diagnostic
#   BASES=1 bash experiments/gate/cloudlab_fetch.sh                     # only the 1-bit base
#   PREFILL_FETCH_K=128 bash experiments/gate/cloudlab_fetch.sh         # also fetch during the tool-output prefill
#   RESUME=1 bash experiments/gate/cloudlab_fetch.sh                    # continue an interrupted run
#   tmux attach -t fetch                                                # watch (Ctrl-b n: next window, Ctrl-b d: detach)
#
# Everything large (venv, pip cache, HF_HOME with the model, temp files) goes on DATA_ROOT, the node's
# large local disk. On CloudLab that is usually an extra partition mounted with mkextrafs at /mydata;
# the script mounts it if the mkextrafs tool is there and DATA_ROOT is not mounted yet (needs sudo).
# Needs ~40 GB free on DATA_ROOT (model 16 GB, venv ~6 GB) and GPUs with >= 40 GB.
set -Eeuo pipefail   # -E: the ERR trap that marks a failed base also fires inside functions

DATA_ROOT="${DATA_ROOT:-/mydata}"
TARGET="${TARGET:-unsloth/Llama-3.1-8B-Instruct}"
MODE="${MODE:-both}"                         # full | diagnose | both
DIAG_PAUSE="${DIAG_PAUSE:-pydata__xarray-4687:85}"
CHECK_NLL_TOL="${CHECK_NLL_TOL:-0.05}"        # smoke run stops if per-step and sdpa NLL differ by more
SESSION="${SESSION:-fetch}"
RESUME="${RESUME:-0}"
BASES="${BASES:-2 1}"                        # base precisions
GPUS="${GPUS:-0 1}"                          # one GPU per base, same order as BASES (parallel mode)
PARALLEL="${PARALLEL:-auto}"                 # auto: parallel if the node has a GPU for every base
REFS_BASE1="${REFS_BASE1:-2}"                # uniform references on base 1 ("2 3" keeps the int3 copy)
PREFILL_FETCH_K="${PREFILL_FETCH_K:-0}"      # > 0: top-k fetch per chunk during the tool-output prefill
EXTRA_ARGS="${EXTRA_ARGS:-}"                 # passed to run_fetch.py, e.g. "--ks 16 64"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"   # marks which windows belong to one launch

KV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$KV_DIR/experiments/gate/cloudlab_fetch.sh"
VENV="$DATA_ROOT/gate-venv"
PY="$VENV/bin/python"
export HF_HOME="$DATA_ROOT/hf"
export PIP_CACHE_DIR="$DATA_ROOT/pip-cache"
export TMPDIR="$DATA_ROOT/tmp"
SLUG="$(basename "$TARGET" | tr '[:upper:]' '[:lower:]')"

name_of() {  # results directory name of base $1
  local n="$SLUG-base$1"
  [ "$PREFILL_FETCH_K" != 0 ] && n="$n-pf$PREFILL_FETCH_K"
  echo "$n"
}

args_of() {  # run_fetch.py arguments of base $1 (one per line)
  printf '%s\n' --target "$TARGET" --base-bits "$1"
  [ "$PREFILL_FETCH_K" != 0 ] && printf '%s\n' --prefill-fetch-k "$PREFILL_FETCH_K"
  if [ "$1" = 1 ]; then
    printf '%s\n' --uniform-refs
    # shellcheck disable=SC2086
    printf '%s\n' $REFS_BASE1
  fi
  # shellcheck disable=SC2086
  [ -n "$EXTRA_ARGS" ] && printf '%s\n' $EXTRA_ARGS
  return 0
}

gpu_of() {  # GPU index for base $1 (position in BASES -> same position in GPUS)
  local i=0 b g
  read -r -a gl <<< "$GPUS"
  for b in $BASES; do
    if [ "$b" = "$1" ]; then g="${gl[$i]:-0}"; echo "$g"; return; fi
    i=$((i + 1))
  done
  echo 0
}

join_source() {  # base-2 results to join int3 from, if the base-1 run left int3 out and base 2 runs too
  [ "$1" = 1 ] || return 0
  [[ " $REFS_BASE1 " == *" 3 "* ]] && return 0
  [[ " $BASES " == *" 2 "* ]] && echo "results/gate_fetch/$(name_of 2)"
  return 0
}

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
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

  if [ ! -x "$PY" ]; then
    python3 -m venv "$VENV"
  fi
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q torch "transformers==5.14.1" pandas matplotlib numpy
  "$PY" -c "import torch; assert torch.cuda.is_available(), 'torch sees no GPU'; print('torch', torch.__version__, torch.cuda.device_count(), 'x', torch.cuda.get_device_name(0))"

  echo ">> downloading $TARGET to $HF_HOME"
  "$PY" -c "from huggingface_hub import snapshot_download as d; print(d('$TARGET', allow_patterns=['*.json', '*.safetensors']))"
  cd "$KV_DIR"
  "$PY" -m experiments.gate.data
}

check_memory() {  # estimated peak per base on its GPU; warns, does not stop
  cd "$KV_DIR"
  local b g mode_args=()
  [ "$MODE" = diagnose ] && mode_args=(--diagnose "$DIAG_PAUSE")
  for b in $BASES; do
    g=$(if [ "$(parallel_mode)" = 1 ]; then gpu_of "$b"; else echo 0; fi)
    a=(); while IFS= read -r x; do a+=("$x"); done < <(args_of "$b")
    echo ">> memory check, base int$b on GPU $g:"
    CUDA_VISIBLE_DEVICES="$g" HF_HUB_OFFLINE=1 "$PY" -m experiments.gate.run_fetch "${a[@]}" ${mode_args[@]+"${mode_args[@]}"} \
      --estimate-memory 2>&1 | grep -E "estimated peak|WARNING|skipped" || true
  done
}

parallel_mode() {
  if [ "$PARALLEL" = auto ]; then
    local n_gpu n_base
    n_gpu=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
    n_base=$(wc -w <<< "$BASES")
    [ "$n_gpu" -ge "$n_base" ] && [ "$(wc -w <<< "$GPUS")" -ge "$n_base" ] && echo 1 || echo 0
  else
    echo "$PARALLEL"
  fi
}

summarize() {  # summary of base $1, joining int3 from base 2 once that run has finished
  local name src
  name=$(name_of "$1")
  src=$(join_source "$1")
  [ -f "results/gate_fetch/$name/metrics.csv" ] || return 0
  if [ -n "$src" ] && [ -f "$src/metrics.csv" ] && [ -e "results/gate_fetch/.status-$RUN_ID-base2.done" ]; then
    "$PY" -m experiments.gate.summarize_fetch --dir "results/gate_fetch/$name" --join "$src"
  else
    "$PY" -m experiments.gate.summarize_fetch --dir "results/gate_fetch/$name"
  fi
}

run_base() {  # one base on the GPU in CUDA_VISIBLE_DEVICES; leaves a done/failed marker for the other window
  local b="$1" name marker
  cd "$KV_DIR"
  export HF_HUB_OFFLINE=1
  name=$(name_of "$b")
  marker="results/gate_fetch/.status-$RUN_ID-base$b"
  mkdir -p results/gate_fetch
  trap 'touch "$marker.failed"' ERR
  a=(); while IFS= read -r x; do a+=("$x"); done < <(args_of "$b")
  echo ">> base int$b on GPU ${CUDA_VISIBLE_DEVICES:-all}: results/gate_fetch/$name  ($(date))"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv
  if [ "$MODE" = full ] || [ "$MODE" = both ]; then
    if [ "$RESUME" = 1 ]; then
      "$PY" -m experiments.gate.run_fetch "${a[@]}" --resume
    else
      "$PY" -m experiments.gate.run_fetch "${a[@]}" --max-samples 1 --check-nll-tol "$CHECK_NLL_TOL" \
        --out-dir "results/gate_fetch_smoke/$name"
      "$PY" -m experiments.gate.run_fetch "${a[@]}"
    fi
    summarize "$b"
  fi
  if [ "$MODE" = diagnose ] || [ "$MODE" = both ]; then
    "$PY" -m experiments.gate.run_fetch "${a[@]}" --diagnose "$DIAG_PAUSE"
  fi
  touch "$marker.done"
  trap - ERR
  echo ">> base int$b finished ($(date))"
}

finish() {  # in the last window: wait for the other bases, redo summaries that join, pack the results
  cd "$KV_DIR"
  local b m
  for b in $BASES; do
    m="results/gate_fetch/.status-$RUN_ID-base$b"
    until [ -e "$m.done" ] || [ -e "$m.failed" ]; do
      echo ">> waiting for base int$b ($(date +%H:%M))"
      sleep 60
    done
    [ -e "$m.failed" ] && echo "!! base int$b failed; see its tmux window / log" >&2
  done
  if [ "$MODE" != diagnose ]; then
    for b in $BASES; do
      if [ -n "$(join_source "$b")" ]; then summarize "$b"; fi
    done
  fi
  tar czf results/gate_fetch_results.tgz results/gate_fetch
  nvidia-smi --query-gpu=index,memory.used --format=csv
  echo ">> done: $KV_DIR/results/gate_fetch_results.tgz (summaries: results/gate_fetch/$SLUG-base*/summary.md)"
}

window_cmd() {  # command line of a tmux window running "$@" with this launch's settings, logged
  local log="$1"; shift
  printf "DATA_ROOT='%s' TARGET='%s' MODE='%s' DIAG_PAUSE='%s' CHECK_NLL_TOL='%s' RESUME='%s' BASES='%s' GPUS='%s' \
PARALLEL='%s' REFS_BASE1='%s' PREFILL_FETCH_K='%s' EXTRA_ARGS='%s' RUN_ID='%s' %s bash '%s' %s 2>&1 | tee -a '%s'; \
echo '>> window finished; press Enter to close'; read" \
    "$DATA_ROOT" "$TARGET" "$MODE" "$DIAG_PAUSE" "$CHECK_NLL_TOL" "$RESUME" "$BASES" "$GPUS" "$PARALLEL" \
    "$REFS_BASE1" "$PREFILL_FETCH_K" "$EXTRA_ARGS" "$RUN_ID" "${GPU_ENV:-}" "$SCRIPT" "$*" "$log"
}

launch() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "!! tmux session '$SESSION' already exists; attach with: tmux attach -t $SESSION" >&2
    exit 1
  fi
  mkdir -p "$KV_DIR/results"
  local first=1 b last
  last=$(awk '{print $NF}' <<< "$BASES")
  if [ "$(parallel_mode)" = 1 ]; then
    for b in $BASES; do
      local g log cmd
      g=$(gpu_of "$b")
      log="$KV_DIR/results/gate_fetch_cloudlab_$(name_of "$b").log"
      local sub="run-base $b"
      [ "$b" = "$last" ] && sub="run-base $b finish"
      cmd=$(GPU_ENV="CUDA_VISIBLE_DEVICES=$g" window_cmd "$log" "$sub")
      if [ "$first" = 1 ]; then
        tmux new-session -d -s "$SESSION" -n "base$b" "$cmd"
        first=0
      else
        tmux new-window -t "$SESSION" -n "base$b" "$cmd"
      fi
      echo ">> base int$b on GPU $g in tmux window '$SESSION:base$b' (log: $log)"
    done
  else
    local log="$KV_DIR/results/gate_fetch_cloudlab_${SLUG}.log"
    tmux new-session -d -s "$SESSION" -n run "$(GPU_ENV="CUDA_VISIBLE_DEVICES=0" window_cmd "$log" "run")"
    echo ">> bases ($BASES) back to back on GPU 0 in tmux session '$SESSION' (log: $log)"
  fi
  echo "   attach: tmux attach -t $SESSION    next window: Ctrl-b n    detach: Ctrl-b d"
}

case "${1:-all}" in
  setup) setup; check_memory ;;
  check-memory) check_memory ;;
  run-base)  # internal: run-base <b> [finish]; a failed base still reaches finish (own process, own errexit)
    if [ "${3:-}" = finish ]; then
      bash "$SCRIPT" run-base "$2" || true
      finish
    else
      run_base "$2"
    fi
    ;;
  run)  # internal: all bases back to back; one failing does not stop the others
    for b in $BASES; do bash "$SCRIPT" run-base "$b" || true; done
    finish
    ;;
  all) setup; check_memory; launch ;;
  launch) launch ;;  # internal: start the tmux windows without setup
  *) echo "usage: $0 [all|setup|check-memory]" >&2; exit 2 ;;
esac
