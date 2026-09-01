#!/bin/bash
# Follow-up chain after stage2: consistent-prompt synthetic re-runs + extra
# baselines + final analyses. Waits for the running campaign to exit first.
set -u
cd "$(dirname "$0")/.."
LOG=experiments/followup.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "waiting for running campaign to finish..."
while pgrep -f "run_campaign.sh" > /dev/null; do sleep 30; done
say "campaign finished; starting follow-up"

run() { say "START $*"; "$@" >> "$LOG" 2>&1 && say "DONE $1 $2 $3" || say "FAIL $* (rc=$?)"; }

# Re-run the three synthetic phases with the final response instructions.
run python3 -m experiments.run.run_phase --phase p1 --dataset experiments/data/synthetic_500.jsonl --namespace p1_syn --seed 13
run python3 -m experiments.run.run_phase --phase p2 --dataset experiments/data/synthetic_500.jsonl --namespace p2_syn --seed 13
run python3 -m experiments.run.run_phase --phase p3 --dataset experiments/data/synthetic_500.jsonl --namespace p3_syn --seed 13

# Extra baselines added after the audit.
run python3 -m experiments.run.ablations --which summary_only
run python3 -m experiments.run.ablations --which sweep_window

# Final analyses.
run python3 -m experiments.analysis.metrics --runs p1_syn p2_syn p3_syn --out experiments/results/analysis_syn
run python3 -m experiments.analysis.metrics --runs p1_enron p2_enron p3_enron --out experiments/results/analysis_enron
run python3 -m experiments.analysis.metrics --runs p1_syn abl_no_dedup abl_no_budget abl_vector_only abl_naive_rag abl_summary_only --out experiments/results/analysis_ablations
run python3 -m experiments.analysis.grounding --runs p1_syn p2_syn p3_syn --window 10 --out experiments/results/analysis_syn/grounding.json

say "FOLLOW-UP COMPLETE"
