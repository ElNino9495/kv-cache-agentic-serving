#!/bin/bash
# Per-step fetch study (run_fetch.py) on the first gate run's 31 pauses, Llama-3.1-8B, on Great Lakes A40s
# (48 GB), one job per entry of JOBS, one A40 each (they queue under a one-GPU limit).
#
# JOBS is a space-separated list of BASE:PREFILL_FETCH_K[:self:TEMPERATURE]. Default = fetch run 4, three base-1 jobs:
#   1:512           larger prefill fetch (real targets)                    -> <model>-base1-pf512
#   1:128:self:0    self-written targets, greedy                           -> <model>-base1-pf128-self-T0
#   1:128:self:0.7  self-written targets, sampled at temperature 0.7       -> <model>-base1-pf128-self-T0.7
# (run 3 was JOBS="2:128 1:128": base 2 and base 1 with real targets, base 1's int3 joined from base 2.)
#
# Set ACCOUNT (first line after the #SBATCH block below, or in the environment) to your own Slurm account, then
# run this file with bash; it submits one job per entry and prints each job's run id and resume command:
#   cd ~/kv_manager && ACCOUNT=<your_account> bash experiments/gate/greatlakes_fetch.sbatch
#   ACCOUNT=<your_account> JOBS="1:512" bash experiments/gate/greatlakes_fetch.sbatch           # one job
#   ACCOUNT=<your_account> MODE=diagnose bash experiments/gate/greatlakes_fetch.sbatch          # diagnostic only
#   ACCOUNT=<your_account> RESUME=1 RUN_ID=<job's run id> JOBS="<its entry>" bash experiments/gate/greatlakes_fetch.sbatch
#       continue one job after a timeout (printed at submission; self-written targets are reloaded from the
#       folder's self_targets.jsonl, so the resumed pauses use exactly the same targets)
#
# Every job runs: a one-pause smoke check (stops the job if per-step decoding and sdpa teacher forcing disagree by
# more than CHECK_NLL_TOL nats in either row's NLL), the full run, its summary, and the per-step diagnostic of one
# pause (MODE: full | diagnose | both, default both), then packs results/cluster/fetch_study_results_<run id>.tgz.
# Output goes to the git-ignored RESULTS_ROOT (default results/cluster/): <root>/gate_fetch/<name>,
# <root>/gate_fetch_smoke/<name>, status markers and the .tgz, so committed results under results/gate_fetch/
# never collide with a job's files and `git pull` on the cluster keeps working. The .tgz holds paths relative to
# the root (gate_fetch/<name>, gate_fetch_smoke/<name>, gate_fetch/fetch_logs_<run id>/ with the Slurm log);
# unpack it on your own machine with:  tar xzf fetch_study_results_<run id>.tgz -C results/
# (RESULTS_ROOT=results resumes a run started before this layout.)
# Rows, all in the same run: bf16, the uniform base, uniform references (base 2: int3; base 1: int2), fetch_all,
# fetch_all_noprefill, noise_ctrl, and per k in 16/32/64/128/match: s1, s2, spec, spec_chain (each with its
# speculative shadow), sens_a, sens_b, s2sens_a, s2sens_b, oracle, s1_layer. Every row also records the KL from
# the bf16 row's next-token distribution.
# Self-written targets (run_fetch --target-source self --temperature T, seed SEED): the bf16 model writes up to 64
# tokens per pause before any row runs; they are saved in <folder>/self_targets.jsonl.
# Base 1 skips its int3 history copy (A40 headroom). If the same submission also has a base-2 job with the same
# prefill k and targets, base 1's summary takes int3 from it (whichever finishes second writes the joined summary,
# only if both folders carry the expected run ids); otherwise base 1's summary has no int3 column (no join).
#
# Memory: ~39 GB per job (run 3 measured 38.7 GB on base 1 and base 2 at prefill k 128; prefill k 512 and the
# self-target generation add little); an A40 has ~47.5 GB usable. Base 1 with int3 would be ~44 GB.
# Time (run 3 on an A40, base 1, prefill k 128: 41 min for 31 pauses, 43 min with smoke and diagnostic, 80 s per
# pause at 42.5 steps on average): pf512 ~45-55 min; self targets ~45-75 min (generation ~3 min; up to 64 steps
# per pause if the model writes long replies, 1.5x the real targets' average). Limit 3 h = 2.4x the worst case.
# One-time setup as for greatlakes.sbatch (venv at ~/gate-venv, model pre-downloaded, recordings in data/).
#SBATCH --job-name=kv-gate-fetch
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=03:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=%x-%j.log
ACCOUNT="${ACCOUNT:-}"   # <- your Slurm account (e.g. a class or lab account); not hard-coded on purpose
set -Eeuo pipefail

TARGET="${TARGET:-unsloth/Llama-3.1-8B-Instruct}"
MODE="${MODE:-both}"
DIAG_PAUSE="${DIAG_PAUSE:-pydata__xarray-4687:85}"
CHECK_NLL_TOL="${CHECK_NLL_TOL:-0.05}"
RESUME="${RESUME:-0}"
JOBS="${JOBS:-1:512 1:128:self:0 1:128:self:0.7}"
SEED="${SEED:-0}"
RESULTS_ROOT="${RESULTS_ROOT:-results/cluster}"
slug=$(basename "$TARGET" | tr '[:upper:]' '[:lower:]')

# parse one JOBS entry into b, pf, src, temp, suffix (job name part) and name (results folder)
parse_job() {
  local spec="$1" rest
  b="${spec%%:*}"; rest="${spec#*:}"
  [ "$rest" != "$spec" ] || { echo "bad JOBS entry '$spec' (BASE:PREFILL_K[:self:T])" >&2; exit 1; }
  pf="${rest%%:*}"; rest="${rest#"$pf"}"; rest="${rest#:}"
  src=real; temp=0
  if [ -n "$rest" ]; then
    src="${rest%%:*}"; temp="${rest#*:}"
    [ "$src" = self ] && [ "$temp" != "$rest" ] || { echo "bad JOBS entry '$spec' (BASE:PREFILL_K[:self:T])" >&2; exit 1; }
  fi
  case "$b" in 1|2) ;; *) echo "bad base '$b' in '$spec' (1 or 2)" >&2; exit 1 ;; esac
  [[ "$pf" =~ ^[0-9]+$ ]] || { echo "bad prefill k '$pf' in '$spec'" >&2; exit 1; }
  [[ "$temp" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "bad temperature '$temp' in '$spec'" >&2; exit 1; }
  local tail=""
  [ "$pf" != 0 ] && tail="-pf$pf"
  [ "$src" = self ] && tail="$tail-self-T$temp"
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
  rid=(); key=()
  for i in "${!specs[@]}"; do
    parse_job "${specs[$i]}"
    # one job + an explicit RUN_ID (a resume): use it as is; otherwise <group>-<job>
    if [ "${#specs[@]}" = 1 ] && [ -n "${RUN_ID:-}" ]; then rid[$i]="$RUN_ID"; else rid[$i]="$group-$suffix"; fi
    key[$i]="$pf|$src|$temp"
  done
  export TARGET MODE DIAG_PAUSE CHECK_NLL_TOL RESUME SEED RESULTS_ROOT
  echo ">> submitting ${#specs[@]} job(s), one A40 each:"
  for i in "${!specs[@]}"; do
    spec="${specs[$i]}"
    # int3 join partner: the other base with the same prefill k and targets, in this submission (or PARTNER_*)
    partner_id="${PARTNER_RUN_ID:-}"; partner_spec="${PARTNER_JOB:-}"
    for o in "${!specs[@]}"; do
      [ "$o" = "$i" ] && continue
      if [ "${specs[$o]%%:*}" != "${spec%%:*}" ] && [ "${key[$o]}" = "${key[$i]}" ]; then
        partner_id="${rid[$o]}"; partner_spec="${specs[$o]}"
      fi
    done
    parse_job "$spec"
    jid=$(JOB="$spec" JOB_RUN_ID="${rid[$i]}" PARTNER_RUN_ID="$partner_id" PARTNER_JOB="$partner_spec" \
          sbatch --parsable --account="$ACCOUNT" --job-name="kv-fetch-$suffix" --export=ALL "$0" "$@")
    echo "   $spec -> job $jid, run id ${rid[$i]}, $RESULTS_ROOT/gate_fetch/$name, log kv-fetch-$suffix-$jid.log"
    [ -n "$partner_id" ] && echo "      int3 join partner: $partner_spec (run id $partner_id)"
    echo "      resume: ACCOUNT=$ACCOUNT RESUME=1 RUN_ID=${rid[$i]} JOBS=\"$spec\"" \
         "${partner_id:+PARTNER_RUN_ID=$partner_id PARTNER_JOB=\"$partner_spec\" }bash $0"
  done
  exit 0
fi

[ -n "${JOB:-}" ] && [ -n "${JOB_RUN_ID:-}" ] || {
  echo "JOB / JOB_RUN_ID not set: submit with 'bash $0' (one job per JOBS entry)" >&2; exit 1; }
cd "$SLURM_SUBMIT_DIR"
module load "${PY_MODULE:-python/3.12.1}"
source ~/gate-venv/bin/activate
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
parse_job "$JOB"
out="$RESULTS_ROOT/gate_fetch/$name"
smoke="$RESULTS_ROOT/gate_fetch_smoke/$name"
mark="$RESULTS_ROOT/gate_fetch/.status-$JOB_RUN_ID"
mkdir -p "$RESULTS_ROOT/gate_fetch"
trap 'touch "$mark.failed"' ERR

common=(--target "$TARGET" --base-bits "$b")
[ "$pf" != 0 ] && common+=(--prefill-fetch-k "$pf")
[ "$src" = self ] && common+=(--target-source self --temperature "$temp" --seed "$SEED")
[ "$b" = 1 ] && common+=(--uniform-refs 2)   # no int3 copy on base 1; int3 is joined from a base-2 partner, if any
echo ">> run $JOB_RUN_ID ($JOB): base int$b, prefill fetch k $pf, $src targets$([ "$src" = self ] && echo " T=$temp seed $SEED")," \
     "on $(hostname): $out ($(date))"

pack() {   # pack <run id> <folder> <smoke folder> <job suffix>; paths in the archive are relative to RESULTS_ROOT
  shopt -s nullglob
  local logs=(kv-fetch-"$4"-[0-9]*.log) logdir="gate_fetch/fetch_logs_$1" items
  mkdir -p "$RESULTS_ROOT/$logdir"
  [ "${#logs[@]}" -gt 0 ] && cp "${logs[@]}" "$RESULTS_ROOT/$logdir/"
  items=("${2#"$RESULTS_ROOT"/}" "$logdir")
  [ -d "$3" ] && items+=("${3#"$RESULTS_ROOT"/}")
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
if [ "$MODE" = diagnose ] || [ "$MODE" = both ]; then
  python -m experiments.gate.run_fetch "${common[@]}" --out-dir "$out" --diagnose "$DIAG_PAUSE"
fi
touch "$mark.done"
trap - ERR
[ "$MODE" != diagnose ] && pack "$JOB_RUN_ID" "$out" "$smoke" "$suffix"

# int3 join: only with a base-2 partner (same prefill k and targets) from the same submission
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
