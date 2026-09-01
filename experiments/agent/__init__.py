"""Agent logic: two-stage triage, five-stage context pipeline, response generation."""
from experiments.agent.pipeline import PipelineStats, StageStats, run_pipeline
from experiments.agent.respond import RespondResult, generate
from experiments.agent.triage import FEWSHOT_EXAMPLES, TriageResult, classify, seed_fewshot

__all__ = [
    "FEWSHOT_EXAMPLES",
    "PipelineStats",
    "RespondResult",
    "StageStats",
    "TriageResult",
    "classify",
    "generate",
    "run_pipeline",
    "seed_fewshot",
]
