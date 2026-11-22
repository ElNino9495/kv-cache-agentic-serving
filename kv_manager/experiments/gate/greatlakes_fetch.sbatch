#!/bin/bash
# Per-step fetch study (run_fetch.py) on the first gate run's 31 pauses, Llama-3.1-8B, on Great Lakes A40s
# (48 GB), one job per entry of JOBS, one A40 each (they queue under a one-GPU limit).
#
# JOBS is a space-separated list of BASE:PREFILL_FETCH_K[:self:TEMPERATURE][:rows=SET]; BASE is 1, 2 or k2v1 (int2
# keys + int1 values), SET a run_fetch --rows set (a1 | a2 | c; default = the run 2-4 rows). Default = fetch run 5:
#   1:128:rows=a1             multi-guess speculation, real targets           -> <model>-base1-pf128-a1
#   1:128:rows=a2             fetch pools + s2_k2score, real targets          -> <model>-base1-pf128-a2
#   1:128:self:0.7:rows=a1    as a1, self-written targets at T = 0.7          -> <model>-base1-pf128-self-T0.7-a1
#   1:128:self:0.7:rows=a2    as a2, self-written targets at T = 0.7          -> <model>-base1-pf128-self-T0.7-a2
#   k2v1:128:rows=c           k2v1 base with int1-base comparators            -> <model>-basek2v1-pf128-c
# (run 4 was JOBS="1:512 1:128:self:0 1:128:self:0.7"; run 3 "2:128 1:128", base 1's int3 joined from base 2.)
#
# Set ACCOUNT (first line after the #SBATCH block below, or in the environment) to your own Slurm account, then
# run this file with bash; it submits one job per entry and prints each job's run id and resume command:
#   cd ~/kv-cache-agentic-serving/kv_manager && ACCOUNT=<your_account> bash experiments/gate/greatlakes_fetch.sbatch
#   ACCOUNT=<your_account> JOBS="1:128:rows=a1" bash experiments/gate/greatlakes_fetch.sbatch   # one job
#   ACCOUNT=<your_account> MODE=diagnose bash experiments/gate/greatlakes_fetch.sbatch          # diagnostic only
#   ACCOUNT=<your_account> RESUME=1 RUN_ID=<job's run id> JOBS="<its entry>" bash experiments/gate/greatlakes_fetch.sbatch
#       continue one job after a timeout (printed at submission; self-written targets are reloaded from the
#       folder's self_targets.jsonl, so the resumed pauses use exactly the same targets)
#
# Every job runs: a one-pause smoke check (stops the job if per-step decoding and sdpa teacher forcing disagree by
# more than CHECK_NLL_TOL nats in either row's NLL), the full run, its summary, and the per-step diagnostic of one
# pause (MODE: full | diagnose | both, default both; run-5 row sets a1/a2 skip the diagnostic, which would repeat
# run 4's), then packs results/cluster/fetch_study_results_<run id>.tgz.
# Output goes to the git-ignored RESULTS_ROOT (default results/cluster/): <root>/gate_fetch/<name>,
# <root>/gate_fetch_smoke/<name>, status markers and the .tgz, so committed results under results/gate_fetch/
# never collide with a job's files and `git pull` on the cluster keeps working. The .tgz holds paths relative to
# the root (gate_fetch/<name>, gate_fetch_smoke/<name>, gate_fetch/fetch_logs_<run id>/ with the Slurm log);
# unpack it on your own machine with:  tar xzf fetch_study_results_<run id>.tgz -C results/
# (RESULTS_ROOT=results resumes a run started before this layout.)
# Rows: see run_fetch.py (--rows) and fetch_attn.study_rows; every row also records the KL from the bf16 row.
# Self-written targets (run_fetch --target-source self --temperature T, seed SEED): the bf16 model writes up to 64
# tokens per pause before any row runs; they are saved in <folder>/self_targets.jsonl. A job with a row set first
# copies the committed targets of the same base / prefill k / temperature without the row set (run 4's
# results/gate_fetch/<model>-base1-pf128-self-T0.7/self_targets.jsonl), so run 5's self-target jobs teacher-force
# exactly run 4's targets (run_fetch checks model, temperature, seed and length of every saved target).
# Base 1 skips its int3 history copy (A40 headroom). If the same submission also has a base-2 job with the same
# prefill k, targets and row set, base 1's summary takes int3 from it (whichever finishes second writes the joined
# summary, only if both folders carry the expected run ids); otherwise base 1's summary has no int3 column.
# Run-5 decision rules: every job's summary has the parts it decides; the last job of the submission to finish
# writes <root>/gate_fetch/rules_run5_<group>.md over all of them (summarize_fetch --rules, run ids checked) and
# packs it into its .tgz.
#
# Memory (largest pause, sympy t72, 33.8k tokens; run_fetch --estimate-memory; runs 3/4 measured ~1.6 GB above
# the estimate, 38.7 GB at an estimate of 37.1): a1 36.6, a2 36.1, c 36.5 GB estimated (c's k2v1 copy reuses the
# int2 keys and int1 values, no extra copy); an A40 has ~47.5 GB usable. PYTORCH_CUDA_ALLOC_CONF=expandable_segments
# reduces fragmentation (run 4 logged one allocator retry on that pause with 38.7 GB allocated).
# Time (run 4 on an A40, 66 rows, prefill k 128: ~43 min per job, 83 s per pause): a1 (68 rows) ~50-65 min,
# a2 (44 rows, pool bookkeeping) ~40-55 min, c (40 rows, two prefill fetches) ~40-55 min; self-target jobs the same
# (targets reused, no generation). Limit 4 h.
# One-time setup as for greatlakes.sbatch (venv at ~/gate-venv, model pre-downloaded, recordings in data/).
#SBATCH --job-name=kv-gate-fetch
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
set -Eeuo pipefail

TARGET="${TARGET:-unsloth/Llama-3.1-8B-Instruct}"
MODE="${MODE:-both}"
DIAG_PAUSE="${DIAG_PAUSE:-pydata__xarray-4687:85}"
CHECK_NLL_TOL="${CHECK_NLL_TOL:-0.05}"
RESUME="${RESUME:-0}"
JOBS="${JOBS:-1:128:rows=a1 1:128:rows=a2 1:128:self:0.7:rows=a1 1:128:self:0.7:rows=a2 k2v1:128:rows=c}"
SEED="${SEED:-0}"
RESULTS_ROOT="${RESULTS_ROOT:-results/cluster}"
slug=$(basename "$TARGET" | tr '[:upper:]' '[:lower:]')

# parse one JOBS entry into b, pf, src, temp, rows, suffix (job name part), name (results folder) and tname (the
# same folder without the row set: where committed self-written targets are taken from)
parse_job() {
  local spec="$1" f
  local -a parts
  IFS=: read -r -a parts <<< "$spec"
  [ "${#parts[@]}" -ge 2 ] || { echo "bad JOBS entry '$spec' (BASE:PREFILL_K[:self:T][:rows=SET])" >&2; exit 1; }
  b="${parts[0]}"; pf="${parts[1]}"; src=real; temp=0; rows=default
  local i=2
  while [ "$i" -lt "${#parts[@]}" ]; do
    f="${parts[$i]}"
    if [ "$f" = self ] && [ $((i + 1)) -lt "${#parts[@]}" ]; then
      src=self; temp="${parts[$((i + 1))]}"; i=$((i + 2))
    elif [ "${f#rows=}" != "$f" ]; then
      rows="${f#rows=}"; i=$((i + 1))
    else
      echo "bad field '$f' in JOBS entry '$spec' (BASE:PREFILL_K[:self:T][:rows=SET])" >&2; exit 1
    fi
  done
  case "$b" in 1|2|k2v1) ;; *) echo "bad base '$b' in '$spec' (1, 2 or k2v1)" >&2; exit 1 ;; esac
  case "$rows" in default|a1|a2|c) ;; *) echo "bad row set '$rows' in '$spec' (default, a1, a2, c)" >&2; exit 1 ;; esac
  [ "$b" != k2v1 ] || [ "$rows" != default ] || { echo "base k2v1 needs a row set (rows=c) in '$spec'" >&2; exit 1; }
  [[ "$pf" =~ ^[0-9]+$ ]] || { echo "bad prefill k '$pf' in '$spec'" >&2; exit 1; }
  [[ "$temp" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "bad temperature '$temp' in '$spec'" >&2; exit 1; }
  local tail=""
  [ "$pf" != 0 ] && tail="-pf$pf"
  [ "$src" = self ] && tail="$tail-self-T$temp"
  tname="$slug-base$b$tail"
  [ "$rows" != default ] && tail="$tail-$rows"
  suffix="b$b$tail"
  name="$slug-base$b$tail"
}

if [ -z "${SLURM_JOB_ID:-}" ]; then
  # not inside a job: submit one job per JOBS entry with the chosen account, passing the settings along
  if [ -z "$ACCOUNT" ]; then
    echo "Set ACCOUNT to your Slurm account (ACCOUNT=<name> bash $0, or edit the line after the #SBATCH block)." >&2
    exit 1
  fi
  read -r -a specs <<< "$JOBS"
  group="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
  rid=(); key=(); rset=(); rules_list=""
  for i in "${!specs[@]}"; do
    parse_job "${specs[$i]}"
    # one job + an explicit RUN_ID (a resume): use it as is; otherwise <group>-<job>
    if [ "${#specs[@]}" = 1 ] && [ -n "${RUN_ID:-}" ]; then rid[$i]="$RUN_ID"; else rid[$i]="$group-$suffix"; fi
    key[$i]="$pf|$src|$temp|$rows"
    rset[$i]="$rows"
    # run-5 jobs of this submission: the last one to finish writes the combined decision rules
    [ "$rows" != default ] && rules_list="$rules_list ${rid[$i]}=$name"
  done
  rules_list="${RULES_JOBS:-${rules_list# }}"
  [ "${#specs[@]}" -gt 1 ] || [ -n "${RULES_JOBS:-}" ] || rules_list=""   # a single job alone: no combined rules
  export TARGET MODE DIAG_PAUSE CHECK_NLL_TOL RESUME SEED RESULTS_ROOT
  echo ">> submitting ${#specs[@]} job(s), one A40 each:"
  for i in "${!specs[@]}"; do
    spec="${specs[$i]}"
    # int3 join partner: the other base with the same prefill k, targets and row set, in this submission (or PARTNER_*)
    partner_id="${PARTNER_RUN_ID:-}"; partner_spec="${PARTNER_JOB:-}"
    for o in "${!specs[@]}"; do
      [ "$o" = "$i" ] && continue
      ob="${specs[$o]%%:*}"; mb="${spec%%:*}"
      if [ "$ob" != "$mb" ] && [ "${key[$o]}" = "${key[$i]}" ] && [ "$ob" != k2v1 ] && [ "$mb" != k2v1 ]; then
        partner_id="${rid[$o]}"; partner_spec="${specs[$o]}"
      fi
    done
    parse_job "$spec"
    jid=$(JOB="$spec" JOB_RUN_ID="${rid[$i]}" PARTNER_RUN_ID="$partner_id" PARTNER_JOB="$partner_spec" \
          RULES_JOBS="$rules_list" RULES_GROUP="$group" \
          sbatch --parsable --account="$ACCOUNT" --job-name="kv-fetch-$suffix" --export=ALL "$0" "$@")
    echo "   $spec -> job $jid, run id ${rid[$i]}, $RESULTS_ROOT/gate_fetch/$name, log kv-fetch-$suffix-$jid.log"
    [ -n "$partner_id" ] && echo "      int3 join partner: $partner_spec (run id $partner_id)"
    echo "      resume: ACCOUNT=$ACCOUNT RESUME=1 RUN_ID=${rid[$i]} JOBS=\"$spec\"" \
         "${partner_id:+PARTNER_RUN_ID=$partner_id PARTNER_JOB=\"$partner_spec\" }" \
         "${rules_list:+RULES_JOBS=\"$rules_list\" RULES_GROUP=$group }bash $0"
  done
  [ -n "$rules_list" ] && echo ">> decision rules over all run-5 jobs: $RESULTS_ROOT/gate_fetch/rules_run5_$group.md," \
                               "written by whichever job finishes last"
  exit 0
fi

[ -n "${JOB:-}" ] && [ -n "${JOB_RUN_ID:-}" ] || {
  echo "JOB / JOB_RUN_ID not set: submit with 'bash $0' (one job per JOBS entry)" >&2; exit 1; }
cd "$SLURM_SUBMIT_DIR"
module load "${PY_MODULE:-python/3.12.1}"
source ~/gate-venv/bin/activate
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
parse_job "$JOB"
out="$RESULTS_ROOT/gate_fetch/$name"
smoke="$RESULTS_ROOT/gate_fetch_smoke/$name"
mark="$RESULTS_ROOT/gate_fetch/.status-$JOB_RUN_ID"
mkdir -p "$RESULTS_ROOT/gate_fetch"
trap 'touch "$mark.failed"' ERR

common=(--target "$TARGET")
if [ "$b" = k2v1 ]; then common+=(--base k2v1 --uniform-refs 2 1)   # k2v1 reuses the int2 keys and int1 values
else common+=(--base-bits "$b"); fi
[ "$pf" != 0 ] && common+=(--prefill-fetch-k "$pf")
[ "$src" = self ] && common+=(--target-source self --temperature "$temp" --seed "$SEED")
[ "$b" = 1 ] && common+=(--uniform-refs 2)   # no int3 copy on base 1; int3 is joined from a base-2 partner, if any
[ "$rows" != default ] && common+=(--rows "$rows")
[ "$rows" = a1 ] && common+=(--s2-extra-ks 256 384)   # s2 rows near the union rows' fetched counts
echo ">> run $JOB_RUN_ID ($JOB): base $b, prefill fetch k $pf, $src targets$([ "$src" = self ] && echo " T=$temp seed $SEED")," \
     "rows $rows, on $(hostname): $out ($(date))"

# self-written targets of a row-set job: start from the committed ones of the same settings (run 4), if any
if [ "$src" = self ] && [ "$rows" != default ] && [ -s "results/gate_fetch/$tname/self_targets.jsonl" ]; then
  for d in "$out" "$smoke"; do
    mkdir -p "$d"
    if [ ! -e "$d/self_targets.jsonl" ]; then
      cp "results/gate_fetch/$tname/self_targets.jsonl" "$d/self_targets.jsonl"
      echo ">> $d: self-written targets copied from results/gate_fetch/$tname/self_targets.jsonl"
    fi
  done
fi

pack() {   # pack <run id> <folder> <smoke folder> <job suffix> [extra file]; paths in the archive are relative to RESULTS_ROOT
  shopt -s nullglob
  local logs=(kv-fetch-"$4"-[0-9]*.log) logdir="gate_fetch/fetch_logs_$1" items
  mkdir -p "$RESULTS_ROOT/$logdir"
  [ "${#logs[@]}" -gt 0 ] && cp "${logs[@]}" "$RESULTS_ROOT/$logdir/"
  items=("${2#"$RESULTS_ROOT"/}" "$logdir")
  [ -d "$3" ] && items+=("${3#"$RESULTS_ROOT"/}")
  [ -n "${5:-}" ] && [ -e "$5" ] && items+=("${5#"$RESULTS_ROOT"/}")
  tar czf "$RESULTS_ROOT/fetch_study_results_$1.tgz" -C "$RESULTS_ROOT" "${items[@]}"
  shopt -u nullglob
  echo ">> packed $RESULTS_ROOT/fetch_study_results_$1.tgz (unpack locally with: tar xzf <file> -C results/)"
}

if [ "$MODE" = full ] || [ "$MODE" = both ]; then
  if [ "$RESUME" = 1 ]; then
    python -m experiments.gate.run_fetch "${common[@]}" --out-dir "$out" --resume "$@"
  else
    python -m experiments.gate.run_fetch "${common[@]}" --max-samples 1 --check-nll-tol "$CHECK_NLL_TOL" \
      --out-dir "$smoke" "$@"
    python -m experiments.gate.run_fetch "${common[@]}" --out-dir "$out" "$@"
  fi
  echo "$JOB_RUN_ID" > "$out/run_id.txt"
  python -m experiments.gate.summarize_fetch --dir "$out"
fi
if [ "$MODE" = diagnose ] || { [ "$MODE" = both ] && { [ "$rows" = default ] || [ "$rows" = c ]; }; }; then
  python -m experiments.gate.run_fetch "${common[@]}" --out-dir "$out" --diagnose "$DIAG_PAUSE"
fi
touch "$mark.done"
trap - ERR

# run-5 decision rules over the whole submission: written by the last job to finish (all done, run ids match)
rules_file=""
if [ "$MODE" != diagnose ] && [ -n "${RULES_JOBS:-}" ]; then
  dirs=(); all=1
  for e in $RULES_JOBS; do
    r="${e%%=*}"; n="${e#*=}"
    if [ ! -e "$RESULTS_ROOT/gate_fetch/.status-$r.done" ] \
       || [ "$(cat "$RESULTS_ROOT/gate_fetch/$n/run_id.txt" 2>/dev/null)" != "$r" ]; then all=0; fi
    dirs+=("$RESULTS_ROOT/gate_fetch/$n")
  done
  if [ "$all" = 1 ]; then
    rules_file="$RESULTS_ROOT/gate_fetch/rules_run5_${RULES_GROUP:-$JOB_RUN_ID}.md"
    python -m experiments.gate.summarize_fetch --rules "${dirs[@]}" --rules-out "$rules_file"
    echo ">> all run-5 jobs done: $rules_file"
  else
    echo ">> decision rules over all run-5 jobs: written when the last of them finishes"
  fi
fi
[ "$MODE" != diagnose ] && pack "$JOB_RUN_ID" "$out" "$smoke" "$suffix" "$rules_file"

# int3 join: only with a base-2 partner (same prefill k, targets and row set) from the same submission
if [ "$MODE" = diagnose ]; then
  :
elif [ -z "${PARTNER_RUN_ID:-}" ]; then
  [ "$b" = 1 ] && echo ">> no base-2 job in this run: base 1's summary has no int3 column (not joined)"
else
  me_b="$b"; me_out="$out"; me_name="$name"; me_suffix="$suffix"; me_smoke="$smoke"
  parse_job "$PARTNER_JOB"
  p_out="$RESULTS_ROOT/gate_fetch/$name"; p_smoke="$RESULTS_ROOT/gate_fetch_smoke/$name"; p_suffix="$suffix"
  if [ ! -e "$RESULTS_ROOT/gate_fetch/.status-$PARTNER_RUN_ID.done" ]; then
    echo ">> base int$me_b done; the int3 join is written when $PARTNER_JOB (run id $PARTNER_RUN_ID) finishes"
  elif [ "$(cat "$p_out/run_id.txt" 2>/dev/null)" != "$PARTNER_RUN_ID" ] \
       || [ "$(cat "$me_out/run_id.txt" 2>/dev/null)" != "$JOB_RUN_ID" ]; then
    echo "!! not joining: $me_out or $p_out does not carry the expected run id (run_id.txt); join by hand if they belong together" >&2
  else
    if [ "$me_b" = 1 ]; then d1="$me_out"; d2="$p_out"; r1="$JOB_RUN_ID"; s1="$me_smoke"; x1="$me_suffix"
    else d1="$p_out"; d2="$me_out"; r1="$PARTNER_RUN_ID"; s1="$p_smoke"; x1="$p_suffix"; fi
    python -m experiments.gate.summarize_fetch --dir "$d1" --join "$d2"
    pack "$r1" "$d1" "$s1" "$x1"
    echo ">> both bases done: $d1/summary.md (int3 joined from $d2)"
  fi
fi
nvidia-smi --query-gpu=memory.used --format=csv
