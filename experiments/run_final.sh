#!/bin/bash
# Final campaign at the held-out-calibrated operating point (k=10, delta=0.40).
# p1/p2 on the synthetic corpus are NOT re-run: neither phase performs vector
# retrieval, so k and delta cannot affect them.
set -u
cd "$(dirname "$0")/.."
LOG=experiments/final.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
run() { say "START $*"; "$@" >> "$LOG" 2>&1 && say "DONE  $2 $3 $4" || say "FAIL  $* rc=$?"; }

SYN=experiments/data/synthetic_500.jsonl
ENR=experiments/data/enron_200.jsonl

run python3 -m experiments.run.run_phase --phase p3 --dataset $SYN --namespace p3_syn --seed 13
run python3 -m experiments.run.run_phase --phase p1 --dataset $ENR --namespace p1_enron --seed 13
run python3 -m experiments.run.run_phase --phase p2 --dataset $ENR --namespace p2_enron --seed 13
run python3 -m experiments.run.run_phase --phase p3 --dataset $ENR --namespace p3_enron --seed 13
for w in no_dedup no_budget vector_only naive_rag summary_only; do
  run python3 -m experiments.run.ablations --which $w
done
run python3 -m experiments.run.ablations --which sweep_tau
run python3 -m experiments.run.ablations --which sweep_delta
run python3 -m experiments.run.ablations --which sweep_k
run python3 -m experiments.run.ablations --which sweep_window
say "FINAL CAMPAIGN COMPLETE"
