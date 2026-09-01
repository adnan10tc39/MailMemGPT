#!/bin/bash
# MailRecallAI re-run campaign driver.
# Sequences the full experimental plan on the free-tier Groq key:
# resumable at every step; on a daily-quota error it sleeps and retries.
# Usage: bash experiments/run_campaign.sh [step]
# Steps: gen enron p1 p2 p3 abl sweeps enron_runs analysis all
set -u
cd "$(dirname "$0")/.."   # repo root
LOG=experiments/campaign.log
STAMP() { date "+%F %T"; }
say() { echo "[$(STAMP)] $*" | tee -a "$LOG"; }

run_with_quota_retry() {
  # $1 = description; rest = command. Retries on GROQ_DAILY_LIMIT (sleep 4h),
  # up to 12 attempts; any other failure retries once then aborts the step.
  local desc="$1"; shift
  local attempt=1
  while [ $attempt -le 12 ]; do
    say "START $desc (attempt $attempt)"
    "$@" >> "$LOG" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then say "DONE $desc"; return 0; fi
    if tail -30 "$LOG" | grep -q "GROQ_DAILY_LIMIT"; then
      say "QUOTA hit during $desc - sleeping 4h then resuming"
      sleep 14400
    else
      say "FAIL $desc rc=$rc (see $LOG tail)"
      if [ $attempt -ge 2 ]; then return $rc; fi
      sleep 120
    fi
    attempt=$((attempt+1))
  done
  return 1
}

step_gen() {
  if [ ! -s experiments/data/synthetic_val_150.jsonl ]; then
    run_with_quota_retry "generate validation 150 (seed 7)" \
      python3 -m experiments.dataset.generate_synthetic --seed 7 --first 150 \
        --out experiments/data/synthetic_val_150.jsonl || return 1
  else say "gen: validation set exists, skip"; fi
  if [ ! -s experiments/data/synthetic_500.jsonl ]; then
    run_with_quota_retry "generate synthetic 500 (seed 13)" \
      python3 -m experiments.dataset.generate_synthetic --seed 13 || return 1
  else say "gen: main set exists, skip"; fi
}

step_calibrate() {
  # Hyperparameter calibration sweeps on the held-out validation split.
  run_with_quota_retry "sweep_tau (validation)" \
    python3 -m experiments.run.ablations --which sweep_tau \
      --dataset experiments/data/synthetic_val_150.jsonl \
      --results-root experiments/results/calibration && \
  run_with_quota_retry "sweep_delta (validation)" \
    python3 -m experiments.run.ablations --which sweep_delta \
      --dataset experiments/data/synthetic_val_150.jsonl \
      --results-root experiments/results/calibration
}

step_enron() {
  [ -s experiments/data/enron_200.jsonl ] && { say "enron: already exists, skip"; return 0; }
  run_with_quota_retry "enron sample+prelabel" \
    python3 -m experiments.dataset.prepare_enron
}

phase_run() { # $1 phase, $2 dataset, $3 namespace
  local resume=""
  [ -s "experiments/results/$3/log.jsonl" ] && resume="--resume"
  run_with_quota_retry "run $3" \
    python3 -m experiments.run.run_phase --phase "$1" --dataset "$2" --namespace "$3" --seed 13 $resume
}

step_p1() { phase_run p1 experiments/data/synthetic_500.jsonl p1_syn; }
step_p2() { phase_run p2 experiments/data/synthetic_500.jsonl p2_syn; }
step_p3() { phase_run p3 experiments/data/synthetic_500.jsonl p3_syn; }

step_abl() {
  run_with_quota_retry "ablations" \
    python3 -m experiments.run.ablations --which no_dedup && \
  run_with_quota_retry "ablations no_budget" \
    python3 -m experiments.run.ablations --which no_budget && \
  run_with_quota_retry "ablations vector_only" \
    python3 -m experiments.run.ablations --which vector_only && \
  run_with_quota_retry "ablations naive_rag" \
    python3 -m experiments.run.ablations --which naive_rag
}

step_sweeps() {
  for s in sweep_tau sweep_k sweep_delta sweep_budget; do
    run_with_quota_retry "$s" python3 -m experiments.run.ablations --which "$s" || return 1
  done
}

step_enron_runs() {
  phase_run p1 experiments/data/enron_200.jsonl p1_enron && \
  phase_run p2 experiments/data/enron_200.jsonl p2_enron && \
  phase_run p3 experiments/data/enron_200.jsonl p3_enron
}

step_analysis() {
  run_with_quota_retry "analysis synthetic" \
    python3 -m experiments.analysis.metrics --runs p1_syn p2_syn p3_syn --out experiments/results/analysis_syn
  run_with_quota_retry "analysis enron" \
    python3 -m experiments.analysis.metrics --runs p1_enron p2_enron p3_enron --out experiments/results/analysis_enron
}

case "${1:-stage1}" in
  gen) step_gen ;;
  enron) step_enron ;;
  calibrate) step_calibrate ;;
  p1) step_p1 ;;
  p2) step_p2 ;;
  p3) step_p3 ;;
  abl) step_abl ;;
  sweeps) step_sweeps ;;
  enron_runs) step_enron_runs ;;
  analysis) step_analysis ;;
  stage1)
    # Datasets + hyperparameter calibration; STOPS afterwards so the
    # calibrated tau/delta can be reviewed and frozen before the main runs.
    step_gen && step_enron && step_calibrate
    say "STAGE1 COMPLETE rc=$? - review experiments/results/calibration and freeze tau/delta before stage2"
    ;;
  stage2)
    # Main phase runs + ablations + test-set sensitivity + Enron + analysis.
    step_p1 && step_p2 && step_p3 && \
    step_abl && step_sweeps && step_enron_runs && step_analysis
    say "STAGE2 COMPLETE rc=$?"
    ;;
  *) echo "unknown step: $1"; exit 2 ;;
esac
