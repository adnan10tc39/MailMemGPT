# MailRecallAI — evaluation harness and released artifacts

This directory contains everything needed to re-execute and audit the experiments
reported in *MailRecallAI: Hybrid Long-Term Memory for LLM-Based Email Agents*.

## What is here

| Path | Contents |
|---|---|
| `agent/` | `triage.py` (Eq. 1 routing + LLM fallback), `pipeline.py` (the five-stage memory-loading pipeline), `respond.py` (response generation with tool calling) |
| `common/` | `store.py` (PostgreSQL + `pgvector` hot/warm tiers and the gzip-JSONL cold archive), `llm.py` (provider-aware client, token accounting, cost cap), `exp_config.py` (every hyperparameter used) |
| `run/` | `run_phase.py` (chronological per-email runner), `ablations.py` (ablation baselines and sensitivity sweeps) |
| `analysis/` | `metrics.py` (accuracy, per-class P/R/F1, macro-F1, Wilson intervals, exact McNemar, seeded bootstrap), `grounding.py` (context-grounding metric) |
| `dataset/` | `generate_synthetic.py` (seeded corpus generator), `prepare_enron.py` (Enron sampler and reference labeler) |
| `data/` | The released corpora (see below) |
| `results/` | Per-email JSONL prediction logs and `run_meta.json` for every run reported in the article |

Every module has an offline self-test that runs without network or database
access: `python3 -m experiments.<module> --selftest`.

## Released corpora

- **`data/synthetic_500.jsonl`** — the 500-email synthetic corpus (200 IGNORE /
  150 NOTIFY / 150 RESPOND). 44 emails are marked `class_critical`: their correct
  label is decidable only from a parent message 15–117 positions earlier, beyond
  the 10-pair recent-history window. `context_note` names the parent.
- **`data/enron_200.jsonl`** — the 200-email subset of the public Enron corpus,
  with the complete labeling sheet in `data/enron_labels_for_review.csv`. Every
  record carries `label_source = llm_prelabel`: these are model-assigned
  reference labels (`gpt-4o`, temperature 0.0, ten emails per call) with **no
  human adjudication pass**, so Enron accuracies measure agreement with a
  documented rubric, not with human judgment.
- **`data/synthetic_val_150.jsonl`**, `synthetic_pilot_20.jsonl`,
  `synthetic_ccpilot_140.jsonl` — held-out validation and development splits.

The raw Enron maildir is not mirrored here; it is a large public download.

The **real-world 200-email corpus is not released.** It is confidential corporate
correspondence contributed by identifiable participants. Consent records and
per-participant labeling sheets are retained by the authors but cannot be
distributed. Results resting on it (the human-rated scores) are therefore
described in the article rather than independently verifiable; every
machine-measured number is reproducible from what is here.

## Reproducing a run

```bash
export OPENAI_API_KEY=...            # never committed; read from the environment
docker run -d --name mailrecall-pg -p 5433:5432 \
  -e POSTGRES_PASSWORD=mailrecall -e POSTGRES_DB=mailrecall pgvector/pgvector:pg16

python3 -m experiments.run.run_phase --phase p3 \
    --dataset experiments/data/synthetic_500.jsonl \
    --namespace p3_syn --seed 13

python3 -m experiments.analysis.metrics --runs p1_syn p2_syn p3_syn \
    --out experiments/results/analysis_syn
```

`--resume` continues an interrupted run from its JSONL log. Phases are `p1`
(SQL-only), `p2` (+ reactive function calling) and `p3` (+ proactive semantic
retrieval).

## Configuration of record

The values used for every reported run are in `common/exp_config.py` and are
echoed into each `results/*/run_meta.json` alongside the git hash: triage
threshold `τ = 0.55`, retrieval threshold `δ = 0.50`, retrieval depth `k = 5`,
prompt budget `B = 8000` tokens, history window `w = 10` pairs, seed 13,
`gpt-4o-mini` for triage/generation/summarization at temperature 0.0, and
`text-embedding-3-small` (1536-d) for embeddings.

`τ` and `δ` were selected on a held-out validation split (seed 7). `k` was
re-examined on a separate 500-email development corpus: raising it from 5 to 10
increases retrieval of the decisive parent message from 73% to 91%, but leaves
triage accuracy unchanged while doubling prompt size and latency, so `k = 5` is
retained. That run is preserved under `results/_k10_sweep/` so the trade-off can
be checked directly.
