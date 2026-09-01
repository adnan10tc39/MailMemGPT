"""Phase runner: chronological email loop with triage, pipeline, response
generation, memory writes and per-email JSONL logging.

CLI:
    python3 -m experiments.run.run_phase --phase p3 \
        --dataset experiments/data/synthetic_500.jsonl --namespace p3_syn \
        --seed 13 [--no-responses] [--limit N] [--resume]
Selftest (no OpenAI, no Postgres — injected fakes):
    python3 -m experiments.run.run_phase --selftest

Agent modules (experiments.agent.*, experiments.common.*) are imported at CALL
time inside main()/_build_real_deps so this module (and its selftest) works
standalone while those modules are being developed concurrently. All external
collaborators are passed through a RunDeps bundle (dependency injection).
"""
from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import os
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path("/media/adnan/DATA/Agentic-LongTerm-Memory")
RESULTS_ROOT = REPO_ROOT / "experiments" / "results"

LOG_LATENCY_KEYS: List[str] = [
    "triage_embed", "triage_query", "triage_llm", "sql_load", "retrieve",
    "dedup", "budget", "assemble", "respond_llm", "total",
]
SUMMARY_EVERY_N_RESPOND = 10


@dataclasses.dataclass
class RunDeps:
    """Injected collaborators for a run (real modules or selftest fakes)."""
    llm: Any
    store: Any
    triage_fn: Callable[..., Any]
    pipeline_fn: Optional[Callable[..., Any]] = None
    respond_fn: Optional[Callable[..., Any]] = None
    seed_fewshot_fn: Optional[Callable[..., Any]] = None
    seed_profile_fn: Optional[Callable[..., Any]] = None


# ------------------------------------------------------- flexible interop glue

_ALIASES: Dict[str, Tuple[str, ...]] = {
    "email": ("email", "mail", "message", "item", "email_dict"),
    "cfg": ("cfg", "config", "exp_config", "conf"),
    "llm": ("llm", "client", "llm_client"),
    "store": ("store", "db", "memory", "storage"),
    "system_prompt": ("system_prompt", "prompt", "sys_prompt", "system",
                      "assembled_prompt"),
    "pipeline_result": ("pipeline_result", "pipeline_stats", "pipeline",
                        "pstats", "stats"),
    "phase": ("phase",),
    "namespace": ("namespace", "ns", "run_namespace"),
    "pipeline_rebuild_fn": ("pipeline_rebuild_fn", "rebuild_fn",
                            "prompt_rebuild_fn", "rebuild"),
    "tool_results_text": ("tool_results_text", "tool_results",
                          "tool_results_block"),
}


def call_flexible(fn: Callable[..., Any], /, **provided: Any) -> Any:
    """Call fn binding only the provided values its signature accepts.

    Matches parameter names against provided keys and their aliases so the
    runner tolerates minor naming differences in concurrently-developed
    modules. Raises TypeError if a required parameter cannot be bound.
    """
    sig = inspect.signature(fn)
    kwargs: Dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL,
                          inspect.Parameter.VAR_KEYWORD):
            continue
        bound = False
        for key, val in provided.items():
            if name == key or name in _ALIASES.get(key, ()):
                kwargs[name] = val
                bound = True
                break
        if not bound and param.default is inspect.Parameter.empty and name != "self":
            raise TypeError(
                f"cannot bind required parameter {name!r} of {fn!r}; "
                f"provided keys: {sorted(provided)}")
    return fn(**kwargs)


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    """First attribute or dict key of obj matching one of names."""
    for n in names:
        if isinstance(obj, dict):
            if n in obj:
                return obj[n]
        elif hasattr(obj, n):
            return getattr(obj, n)
    return default


def _lat_of(container: Any, *keys: str) -> Optional[float]:
    """Look a latency value up in a latencies dict/object by candidate keys."""
    v = _get(container, *keys)
    return float(v) if v is not None else None


# --------------------------------------------------------------- result glue

def _split_pipeline_output(out: Any) -> Tuple[str, Any]:
    """Normalize pipeline_fn output to (system_prompt, stats_object)."""
    if isinstance(out, (tuple, list)) and len(out) == 2:
        return str(out[0]), out[1]
    prompt = _get(out, "system_prompt", "prompt", "final_prompt", "text")
    stats = _get(out, "stats", "pipeline_stats", default=out)
    return str(prompt or ""), stats


def _stage(stats: Any, name: str, *aliases: str) -> Any:
    """Fetch a StageStats-like entry from PipelineStats by stage name."""
    stages = _get(stats, "stages", "stage_stats", default=None)
    for key in (name,) + aliases:
        if stages is not None:
            s = _get(stages, key)
            if s is not None:
                return s
        s = _get(stats, key)
        if s is not None and not isinstance(s, (int, float, str)):
            return s
    return None


def _extract_pipeline_info(stats: Any, budget: int) -> Tuple[Dict[str, Optional[float]],
                                                             Dict[str, Any],
                                                             Optional[int]]:
    """Pull per-stage latencies, retrieval stats and final prompt tokens out of
    a PipelineStats-like object (tolerant to attribute naming)."""
    lats: Dict[str, Optional[float]] = {}
    stage_map = {"sql_load": ("sql_load", "sql", "load"),
                 "retrieve": ("vector_retrieve", "retrieve", "retrieval"),
                 "dedup": ("dedup", "deduplicate"),
                 "budget": ("budget", "budgeting"),
                 "assemble": ("assemble", "assembly")}
    # A PipelineStats may carry latencies as a flat dict (the real
    # agent.pipeline.PipelineStats.latency_ms) or as per-stage StageStats
    # objects (each with .latency_ms). Prefer the flat dict when present.
    lat_map = _get(stats, "latency_ms", "latencies", "stage_latency_ms")
    if not isinstance(lat_map, dict):
        lat_map = None
    stage_objs: Dict[str, Any] = {}
    for log_key, names in stage_map.items():
        s = _stage(stats, *names)
        stage_objs[log_key] = s
        v = _lat_of(lat_map, log_key, *names) if lat_map is not None else None
        if v is None and s is not None:
            v = _lat_of(s, "latency_ms", "latency")
        lats[log_key] = v

    ret = stage_objs["retrieve"]
    ded = stage_objs["dedup"]
    n_kept = _get(stats, "n_kept", default=_get(ret, "items") if ret is not None else None)
    removed = _get(ret, "removed_items", default=0) if ret is not None else 0
    n_raw = _get(stats, "n_raw",
                 default=(None if n_kept is None else int(n_kept) + int(removed or 0)))
    n_deduped = _get(stats, "n_deduped",
                     default=_get(ded, "removed_items") if ded is not None else None)
    tokens_saved = _get(stats, "tokens_saved",
                        default=_get(ded, "tokens_saved") if ded is not None else None)
    sims = _get(stats, "sims", "similarities",
                default=_get(ret, "sims", "similarities", default=[]) if ret is not None else [])
    retrieval = {"n_raw": n_raw, "n_kept": n_kept, "n_deduped": n_deduped,
                 "sims": list(sims or []), "tokens_saved": tokens_saved}

    fpt = _get(stats, "final_prompt_tokens", "prompt_tokens", "final_tokens")
    if fpt is None and stage_objs["assemble"] is not None:
        fpt = _get(stage_objs["assemble"], "tokens_in_block", "tokens")
    fpt = int(fpt) if fpt is not None else None
    return lats, retrieval, fpt


def _email_doc(email: dict, response_text: Optional[str]) -> str:
    """Canonical 'email + response' document stored in warm memory."""
    doc = (f"From: {email.get('sender', '')}\n"
           f"Subject: {email.get('subject', '')}\n"
           f"Date: {email.get('ts', '')}\n\n{email.get('body', '')}")
    if response_text:
        doc += f"\n\n--- Assistant response ---\n{response_text}"
    return doc


def email_meta(email: dict) -> Dict[str, Any]:
    """Metadata dict stored alongside a warm-memory document."""
    return {"email_id": email.get("email_id"), "thread_id": email.get("thread_id"),
            "sender": email.get("sender"), "subject": email.get("subject"),
            "day": email.get("day")}


# ------------------------------------------------------------------ run core

def _usage_snapshot(llm: Any) -> Dict[str, float]:
    u = getattr(llm, "usage_totals", {}) or {}
    return {"prompt_tokens": u.get("prompt_tokens", 0),
            "completion_tokens": u.get("completion_tokens", 0)}


def _update_summary(cfg: Any, deps: RunDeps, buffer: List[dict]) -> None:
    """Rolling summary of older interactions via the summary model."""
    prev = deps.store.latest_summary() or "(none yet)"
    items = "\n".join(
        f"- From {b['sender']} | {b['subject']}: {b['body'][:200]}"
        + (f" | reply: {b['response'][:200]}" if b.get("response") else "")
        for b in buffer)
    messages = [
        {"role": "system",
         "content": "You maintain a rolling summary of an email assistant's "
                    "older interactions. Be concise and factual."},
        {"role": "user",
         "content": ("update running summary of older interactions.\n\n"
                     f"Current summary:\n{prev}\n\nNew interactions:\n{items}\n\n"
                     "Return only the updated summary (max ~250 words).")},
    ]
    res = deps.llm.chat(messages, model=cfg.summary_model, max_tokens=500)
    content = getattr(res, "content", None) or (res.get("content") if isinstance(res, dict) else None)
    if content:
        deps.store.update_summary(content.strip())


def _process_email(cfg: Any, email: dict, deps: RunDeps,
                   run_state: Dict[str, Any]) -> dict:
    """Process one email end-to-end; returns the per-email log record."""
    t0 = time.perf_counter()
    usage0 = _usage_snapshot(deps.llm)
    latencies: Dict[str, Optional[float]] = {k: None for k in LOG_LATENCY_KEYS}

    tri = call_flexible(deps.triage_fn, email=email, cfg=cfg, llm=deps.llm,
                        store=deps.store, phase=cfg.phase)
    label_pred = _get(tri, "label", "classification", default="RESPOND")
    confidence = float(_get(tri, "confidence", default=0.0) or 0.0)
    method = _get(tri, "method")
    tool_calls: List[Any] = list(_get(tri, "tool_calls", default=[]) or [])
    tri_lats = _get(tri, "latencies", "latency_ms", default={}) or {}
    latencies["triage_embed"] = _lat_of(tri_lats, "triage_embed", "embed", "embedding")
    latencies["triage_query"] = _lat_of(tri_lats, "triage_query", "query", "fewshot_query")
    latencies["triage_llm"] = _lat_of(tri_lats, "triage_llm", "llm", "fallback", "llm_fallback")

    response_text: Optional[str] = None
    retrieval: Dict[str, Any] = {"n_raw": None, "n_kept": None, "n_deduped": None,
                                 "sims": [], "tokens_saved": None}
    final_prompt_tokens: Optional[int] = None

    if label_pred == "RESPOND" and deps.pipeline_fn is not None:
        # Prompt assembly always runs for RESPOND so token/dedup metrics are
        # real even when response generation is disabled (ablations).
        def _rebuild(tool_results_text: str = "") -> Any:
            """Re-run prompt assembly (respond loop injects tool results)."""
            return call_flexible(deps.pipeline_fn, email=email, cfg=cfg,
                                 llm=deps.llm, store=deps.store,
                                 phase=cfg.phase,
                                 tool_results_text=tool_results_text)

        pl_out = _rebuild()
        prompt, pstats = _split_pipeline_output(pl_out)
        pl_lats, retrieval, final_prompt_tokens = _extract_pipeline_info(
            pstats, cfg.prompt_budget)
        latencies.update(pl_lats)
        if cfg.generate_responses and deps.respond_fn is not None:
            r = call_flexible(deps.respond_fn, email=email, cfg=cfg,
                              llm=deps.llm, store=deps.store,
                              system_prompt=prompt, pipeline_result=pstats,
                              pipeline_rebuild_fn=_rebuild, phase=cfg.phase)
            response_text = _get(r, "text", "response", "content")
            tool_calls += list(_get(r, "tool_calls", default=[]) or [])
            r_lats = _get(r, "latencies", "latency_ms", default={}) or {}
            v = _lat_of(r_lats, "respond_llm", "llm", "total")
            latencies["respond_llm"] = v if v is not None else _lat_of(r, "latency_ms")

    deps.store.save_interaction(email, label_pred, confidence, response_text)
    deps.store.archive_email(email, response_text)

    if label_pred == "RESPOND":
        doc = _email_doc(email, response_text)
        vec = deps.llm.embed([doc]).vectors[0]
        deps.store.add_memory(doc, vec, email_meta(email))
        run_state["respond_count"] += 1
        run_state["summary_buffer"].append({
            "sender": email.get("sender", ""), "subject": email.get("subject", ""),
            "body": email.get("body", ""), "response": response_text or ""})
        if run_state["respond_count"] % SUMMARY_EVERY_N_RESPOND == 0:
            _update_summary(cfg, deps, run_state["summary_buffer"])
            run_state["summary_buffer"] = []

    usage1 = _usage_snapshot(deps.llm)
    latencies["total"] = (time.perf_counter() - t0) * 1000.0
    budget_used_pct = (100.0 * final_prompt_tokens / cfg.prompt_budget
                       if final_prompt_tokens is not None and cfg.prompt_budget else None)
    return {
        "email_id": email.get("email_id"),
        "day": email.get("day"),
        "label_true": email.get("label"),
        "label_pred": label_pred,
        "confidence": confidence,
        "method": method,
        "tool_calls": tool_calls,
        "latencies": latencies,
        "tokens": {
            "prompt": int(usage1["prompt_tokens"] - usage0["prompt_tokens"]),
            "completion": int(usage1["completion_tokens"] - usage0["completion_tokens"]),
            "final_prompt_tokens": final_prompt_tokens,
            "budget_used_pct": budget_used_pct,
        },
        "retrieval": retrieval,
        "response_len_chars": len(response_text) if response_text else 0,
        "error": None,
    }


def _error_record(email: dict, exc: Exception) -> dict:
    return {
        "email_id": email.get("email_id"), "day": email.get("day"),
        "label_true": email.get("label"), "label_pred": None,
        "confidence": None, "method": None, "tool_calls": [],
        "latencies": {k: None for k in LOG_LATENCY_KEYS},
        "tokens": {"prompt": None, "completion": None,
                   "final_prompt_tokens": None, "budget_used_pct": None},
        "retrieval": {"n_raw": None, "n_kept": None, "n_deduped": None,
                      "sims": [], "tokens_saved": None},
        "response_len_chars": 0,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
                              capture_output=True, text=True, timeout=10,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def _cfg_dump(cfg: Any) -> Dict[str, Any]:
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        return dataclasses.asdict(cfg)
    return {k: v for k, v in vars(cfg).items() if not k.startswith("_")}


def load_dataset(path: str | Path) -> List[dict]:
    """Load a dataset JSONL file, sorted chronologically by (day, ts)."""
    emails: List[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                emails.append(json.loads(line))
    emails.sort(key=lambda e: (e.get("day", 0), e.get("ts", "")))
    return emails


def seed_store(cfg: Any, deps: RunDeps) -> None:
    """Reset the store namespace and seed profile/rules + few-shot examples."""
    deps.store.reset()
    if deps.seed_profile_fn is not None:
        call_flexible(deps.seed_profile_fn, store=deps.store, cfg=cfg,
                      llm=deps.llm)
    if deps.seed_fewshot_fn is not None:
        call_flexible(deps.seed_fewshot_fn, store=deps.store, cfg=cfg,
                      llm=deps.llm)


def run(cfg: Any, namespace: str, deps: RunDeps, limit: Optional[int] = None,
        resume: bool = False, results_root: Optional[str | Path] = None,
        progress: bool = True) -> Dict[str, Any]:
    """Execute one experiment run: chronological loop, logging, run_meta.

    Returns a summary dict {n_processed, n_skipped, n_errors, log_path,
    meta_path}.
    """
    results_dir = Path(results_root or RESULTS_ROOT) / namespace
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "log.jsonl"
    meta_path = results_dir / "run_meta.json"

    emails = load_dataset(cfg.dataset_path)
    if limit is not None:
        emails = emails[:limit]

    done_ids: set = set()
    if resume and log_path.exists():
        for row in _read_jsonl_safe(log_path):
            if row.get("email_id"):
                done_ids.add(row["email_id"])
    elif log_path.exists():
        log_path.unlink()

    if not resume:
        seed_store(cfg, deps)

    random.seed(getattr(cfg, "seed", 13))
    started = datetime.now(timezone.utc).isoformat()
    meta = {"config": _cfg_dump(cfg), "namespace": namespace,
            "git_hash": _git_hash(), "started": started, "finished": None,
            "resumed": resume, "limit": limit, "usage_totals": None}
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    run_state: Dict[str, Any] = {"respond_count": 0, "summary_buffer": []}
    n_processed = n_errors = n_skipped = 0
    with open(log_path, "a", encoding="utf-8") as log_fh:
        for i, email in enumerate(emails):
            eid = email.get("email_id")
            if eid in done_ids:
                n_skipped += 1
                continue
            try:
                record = _process_email(cfg, email, deps, run_state)
            except Exception:
                try:  # one retry, then log the error and continue
                    record = _process_email(cfg, email, deps, run_state)
                except Exception as exc2:
                    record = _error_record(email, exc2)
                    n_errors += 1
            log_fh.write(json.dumps(record, default=str) + "\n")
            log_fh.flush()
            n_processed += 1
            if progress and (i + 1) % 10 == 0:
                print(f"[{namespace}] {i + 1}/{len(emails)} emails "
                      f"(errors={n_errors})")

    meta["finished"] = datetime.now(timezone.utc).isoformat()
    meta["usage_totals"] = dict(getattr(deps.llm, "usage_totals", {}) or {})
    meta["n_processed"] = n_processed
    meta["n_skipped"] = n_skipped
    meta["n_errors"] = n_errors
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    if progress:
        print(f"[{namespace}] done: {n_processed} processed, "
              f"{n_skipped} skipped, {n_errors} errors -> {log_path}")
    return {"n_processed": n_processed, "n_skipped": n_skipped,
            "n_errors": n_errors, "log_path": str(log_path),
            "meta_path": str(meta_path)}


def _read_jsonl_safe(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a torn final line from a crashed run
    return rows


# --------------------------------------------------- real-module wiring (CLI)

def _resolve(mod: Any, *candidates: str) -> Callable[..., Any]:
    for name in candidates:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    raise AttributeError(
        f"none of {candidates} found in {getattr(mod, '__name__', mod)}")


def _resolve_optional(mod: Any, *candidates: str) -> Optional[Callable[..., Any]]:
    try:
        return _resolve(mod, *candidates)
    except AttributeError:
        return None


def _build_real_deps(cfg: Any, namespace: str) -> RunDeps:
    """Import agent/common modules at call time and wire a RunDeps bundle."""
    from experiments.common.llm import LLM
    from experiments.common import store as store_mod
    from experiments.agent import triage as triage_mod
    from experiments.agent import pipeline as pipeline_mod
    from experiments.agent import respond as respond_mod

    llm = LLM(cfg)
    store = store_mod.Store(cfg, namespace)
    seed_profile_fn = _resolve_optional(
        store_mod, "seed_profile_and_rules", "seed_profile", "seed_defaults",
        "seed_namespace")
    if seed_profile_fn is None:
        seed_profile_fn = _resolve_optional(
            store, "seed_profile_and_rules", "seed_profile", "seed_defaults")
    return RunDeps(
        llm=llm,
        store=store,
        triage_fn=_resolve(triage_mod, "classify", "triage_email", "triage",
                           "classify_email", "run_triage"),
        pipeline_fn=_resolve(pipeline_mod, "run_pipeline", "build_prompt",
                             "assemble_prompt", "pipeline"),
        respond_fn=_resolve(respond_mod, "generate", "generate_response",
                            "respond"),
        seed_fewshot_fn=_resolve_optional(triage_mod, "seed_fewshot",
                                          "seed_fewshot_examples"),
        seed_profile_fn=seed_profile_fn,
    )


def build_config(phase: str, dataset: str, namespace: str, seed: int,
                 generate_responses: bool, **overrides: Any) -> Any:
    """Build an ExpConfig via the phase preset (imported at call time)."""
    from experiments.common.exp_config import preset
    return preset(phase, run_id=namespace, dataset_path=dataset, seed=seed,
                  generate_responses=generate_responses, **overrides)


# ----------------------------------------------------------- selftest fakes

class FakeEmbedResult(SimpleNamespace):
    pass


class FakeChatResult(SimpleNamespace):
    pass


class FakeLLM:
    """Deterministic in-memory LLM stand-in (no network)."""

    def __init__(self) -> None:
        self.usage_totals: Dict[str, float] = {
            "prompt_tokens": 0, "completion_tokens": 0,
            "embed_tokens": 0, "cost_usd": 0.0}
        self.chat_calls: List[dict] = []

    def chat(self, messages: Sequence[dict], functions: Any = None,
             function_call: Any = None, model: Optional[str] = None,
             temperature: Optional[float] = None,
             max_tokens: Optional[int] = None) -> FakeChatResult:
        self.chat_calls.append({"model": model, "messages": messages})
        self.usage_totals["prompt_tokens"] += 100
        self.usage_totals["completion_tokens"] += 20
        return FakeChatResult(content="FAKE SUMMARY", function_call=None,
                              prompt_tokens=100, completion_tokens=20,
                              latency_ms=1.0)

    def embed(self, texts: Sequence[str]) -> FakeEmbedResult:
        vecs = []
        for t in texts:
            h = abs(hash(t)) % 997
            vecs.append([(h % 13) / 13.0, ((h // 13) % 13) / 13.0, 0.5])
        self.usage_totals["embed_tokens"] += sum(len(t.split()) for t in texts)
        return FakeEmbedResult(vectors=vecs, latency_ms=0.5, tokens=len(texts))

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


class FakeStore:
    """In-memory Store stand-in implementing the pinned interface."""

    def __init__(self) -> None:
        self.reset_calls = 0
        self.interactions: List[dict] = []
        self.archived: List[dict] = []
        self.memories: List[dict] = []
        self.fewshot: Dict[str, List[Tuple[str, Any]]] = {}
        self.summaries: List[str] = []
        self.profile = "John Doe, software engineering manager"
        self.rules = "1) spam->IGNORE 2) FYI->NOTIFY 3) questions->RESPOND 4) urgent->RESPOND"

    def reset(self) -> None:
        self.reset_calls += 1
        self.interactions, self.archived, self.memories = [], [], []
        self.fewshot, self.summaries = {}, []

    def save_interaction(self, email: dict, classification: str,
                         confidence: float, response_text: Optional[str]) -> None:
        self.interactions.append({"email": email, "classification": classification,
                                  "confidence": confidence,
                                  "response_text": response_text})

    def recent_history(self, n_pairs: int) -> List[dict]:
        return self.interactions[-n_pairs:]

    def get_user_profile(self) -> str:
        return self.profile

    def get_rules(self) -> str:
        return self.rules

    def latest_summary(self) -> Optional[str]:
        return self.summaries[-1] if self.summaries else None

    def update_summary(self, text: str) -> None:
        self.summaries.append(text)

    def add_fewshot(self, label: str, doc: str, vec: Any) -> None:
        self.fewshot.setdefault(label, []).append((doc, vec))

    def query_fewshot(self, label: str, vec: Any, topn: int) -> List[Tuple[float, str]]:
        docs = self.fewshot.get(label, [])
        return [(0.5, d) for d, _ in docs[:topn]]

    def add_memory(self, doc: str, vec: Any, meta: dict) -> None:
        self.memories.append({"doc": doc, "vec": vec, "meta": meta})

    def query_memory(self, vec: Any, k: int) -> List[Tuple[float, str, dict]]:
        return [(0.8, m["doc"], m["meta"]) for m in self.memories[-k:]]

    def archive_email(self, email: dict, response_text: Optional[str]) -> None:
        self.archived.append({"email": email, "response_text": response_text})


def fake_config(**overrides: Any) -> SimpleNamespace:
    """ExpConfig stand-in with the pinned fields (selftests only)."""
    base: Dict[str, Any] = dict(
        phase="p3", run_id="selftest", dataset_path="", triage_tau=0.37,
        triage_topn=3, retrieval_k=5, retrieval_delta=0.70, prompt_budget=8000,
        history_window=10, chat_model="openai/gpt-oss-120b", summary_model="openai/gpt-oss-20b",
        embed_model="BAAI/bge-base-en-v1.5", provider="groq", embed_dim=768,
        embed_device="cuda", temperature=0.0,
        max_output_tokens=2000, seed=13, generate_responses=True,
        enable_dedup=True, enable_budget=True, enable_sql_history=True,
        enable_vector_retrieval=True, naive_rag=False)
    base.update(overrides)
    return SimpleNamespace(**base)


def fixture_emails(n_respond: int = 2, n_ignore: int = 2,
                   n_notify: int = 1) -> List[dict]:
    """Deterministic fixture emails for selftests."""
    emails: List[dict] = []
    labels = (["RESPOND"] * n_respond + ["IGNORE"] * n_ignore
              + ["NOTIFY"] * n_notify)
    for i, label in enumerate(labels):
        emails.append({
            "email_id": f"fx-{i + 1:04d}", "day": 1 + i // 2,
            "ts": f"2026-01-{5 + i:02d}T09:00:00", "thread_id": f"t-{i:03d}",
            "sender": f"Persona {i} <p{i}@example.com>",
            "subject": f"Fixture subject {i}",
            "body": f"Fixture body {i} asking about item {i}.",
            "label": label, "requires_context": False})
    return emails


def _fake_triage(email: dict, cfg: Any, llm: Any, store: Any) -> SimpleNamespace:
    return SimpleNamespace(
        label=email["label"], confidence=0.88, method="semantic",
        tool_calls=[],
        latencies={"triage_embed": 1.2, "triage_query": 0.4, "triage_llm": None})


def _fake_pipeline(email: dict, cfg: Any, llm: Any, store: Any) -> Tuple[str, SimpleNamespace]:
    stages = {
        "sql_load": SimpleNamespace(latency_ms=2.0, tokens_in_block=300,
                                    items=4, removed_items=0, tokens_saved=0),
        "vector_retrieve": SimpleNamespace(latency_ms=3.0, tokens_in_block=200,
                                           items=3, removed_items=2,
                                           tokens_saved=0,
                                           sims=[0.82, 0.78, 0.71]),
        "dedup": SimpleNamespace(latency_ms=0.3, tokens_in_block=150, items=2,
                                 removed_items=1, tokens_saved=90),
        "budget": SimpleNamespace(latency_ms=0.2, tokens_in_block=150, items=2,
                                  removed_items=0, tokens_saved=0),
        "assemble": SimpleNamespace(latency_ms=0.5, tokens_in_block=1600,
                                    items=8, removed_items=0, tokens_saved=0),
    }
    stats = SimpleNamespace(stages=stages, final_prompt_tokens=1600)
    return f"SYSTEM PROMPT for {email['email_id']}", stats


def _fake_respond(email: dict, cfg: Any, llm: Any, store: Any,
                  system_prompt: str) -> SimpleNamespace:
    llm.chat([{"role": "system", "content": system_prompt},
              {"role": "user", "content": email["body"]}], model=cfg.chat_model)
    return SimpleNamespace(text=f"Reply to {email['email_id']}",
                           tool_calls=[{"name": "check_calendar_availability"}],
                           latencies={"respond_llm": 12.5},
                           tokens={"prompt": 100, "completion": 20})


def _fake_seed_fewshot(store: Any, llm: Any) -> None:
    for label, doc in [("IGNORE", "spam offer"), ("NOTIFY", "build passed"),
                       ("RESPOND", "can we meet?")]:
        store.add_fewshot(label, doc, llm.embed([doc]).vectors[0])


def make_fake_deps() -> RunDeps:
    """Fully offline RunDeps bundle (shared with ablations selftest)."""
    llm = FakeLLM()
    return RunDeps(llm=llm, store=FakeStore(), triage_fn=_fake_triage,
                   pipeline_fn=_fake_pipeline, respond_fn=_fake_respond,
                   seed_fewshot_fn=_fake_seed_fewshot, seed_profile_fn=None)


def _selftest() -> None:
    """End-to-end run over 5 fixture emails with injected fakes; validates the
    log schema, resume behavior, error handling and summary cadence."""
    scratch = Path(os.environ.get(
        "MAILRECALL_SELFTEST_DIR",
        "/tmp/claude-1000/-media-adnan-DATA-Agentic-LongTerm-Memory/"
        "250f06c2-b063-439d-83ca-fb5215f31351/scratchpad")) / "run_phase_selftest"
    scratch.mkdir(parents=True, exist_ok=True)
    dataset_path = scratch / "fixture.jsonl"
    emails = fixture_emails()
    with open(dataset_path, "w", encoding="utf-8") as fh:
        for e in emails:
            fh.write(json.dumps(e) + "\n")

    cfg = fake_config(dataset_path=str(dataset_path))
    deps = make_fake_deps()
    out = run(cfg, "selftest_run", deps, results_root=scratch, progress=False)
    assert out["n_processed"] == 5 and out["n_errors"] == 0, out

    rows = _read_jsonl_safe(Path(out["log_path"]))
    assert len(rows) == 5
    expected_keys = {"email_id", "day", "label_true", "label_pred", "confidence",
                     "method", "tool_calls", "latencies", "tokens", "retrieval",
                     "response_len_chars", "error"}
    for r in rows:
        assert set(r) == expected_keys, set(r) ^ expected_keys
        assert set(r["latencies"]) == set(LOG_LATENCY_KEYS)
        assert set(r["tokens"]) == {"prompt", "completion",
                                    "final_prompt_tokens", "budget_used_pct"}
        assert set(r["retrieval"]) == {"n_raw", "n_kept", "n_deduped", "sims",
                                       "tokens_saved"}
        assert r["error"] is None
    respond_rows = [r for r in rows if r["label_pred"] == "RESPOND"]
    assert len(respond_rows) == 2
    for r in respond_rows:
        assert r["response_len_chars"] > 0
        assert r["retrieval"] == {"n_raw": 5, "n_kept": 3, "n_deduped": 1,
                                  "sims": [0.82, 0.78, 0.71], "tokens_saved": 90}
        assert r["tokens"]["final_prompt_tokens"] == 1600
        assert abs(r["tokens"]["budget_used_pct"] - 20.0) < 1e-9
        assert r["latencies"]["sql_load"] == 2.0
        assert r["latencies"]["respond_llm"] == 12.5
        assert r["tool_calls"] == [{"name": "check_calendar_availability"}]
    assert deps.store.reset_calls == 1
    assert len(deps.store.interactions) == 5
    assert len(deps.store.archived) == 5
    assert len(deps.store.memories) == 2
    assert deps.store.fewshot  # few-shot seeded
    meta = json.loads((Path(out["meta_path"])).read_text(encoding="utf-8"))
    assert meta["finished"] and meta["git_hash"] and meta["usage_totals"]
    assert meta["config"]["prompt_budget"] == 8000

    # resume: everything already logged -> nothing reprocessed, no reset
    out2 = run(cfg, "selftest_run", deps, resume=True, results_root=scratch,
               progress=False)
    assert out2["n_processed"] == 0 and out2["n_skipped"] == 5, out2
    assert deps.store.reset_calls == 1

    # error path: triage that always fails for one email -> retry then error line
    def flaky_triage(email: dict, cfg: Any, llm: Any, store: Any) -> SimpleNamespace:
        if email["email_id"] == "fx-0003":
            raise RuntimeError("boom")
        return _fake_triage(email, cfg, llm, store)

    deps_err = make_fake_deps()
    deps_err.triage_fn = flaky_triage
    out3 = run(cfg, "selftest_err", deps_err, results_root=scratch,
               progress=False)
    assert out3["n_processed"] == 5 and out3["n_errors"] == 1, out3
    err_rows = [r for r in _read_jsonl_safe(scratch / "selftest_err" / "log.jsonl")
                if r["error"]]
    assert len(err_rows) == 1 and err_rows[0]["email_id"] == "fx-0003"
    assert err_rows[0]["label_pred"] is None
    assert set(err_rows[0]) == expected_keys

    # summary cadence: 10 RESPOND emails -> exactly one summary update
    emails10 = fixture_emails(n_respond=10, n_ignore=0, n_notify=0)
    ds10 = scratch / "fixture10.jsonl"
    with open(ds10, "w", encoding="utf-8") as fh:
        for e in emails10:
            fh.write(json.dumps(e) + "\n")
    deps10 = make_fake_deps()
    run(fake_config(dataset_path=str(ds10)), "selftest_sum", deps10,
        results_root=scratch, progress=False)
    assert deps10.store.summaries == ["FAKE SUMMARY"], deps10.store.summaries
    summary_calls = [c for c in deps10.llm.chat_calls
                     if c["model"] == "openai/gpt-oss-20b"]
    assert len(summary_calls) == 1
    assert "update running summary of older interactions" in \
        summary_calls[0]["messages"][1]["content"]
    print("run_phase selftest OK:", scratch)


# ----------------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(description="Run one experiment phase.")
    ap.add_argument("--phase", choices=["p1", "p2", "p3"])
    ap.add_argument("--dataset", help="dataset JSONL path")
    ap.add_argument("--namespace", help="store/results namespace, e.g. p3_syn")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--no-responses", action="store_true",
                    help="skip response generation (triage + assembly metrics only)")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N emails")
    ap.add_argument("--resume", action="store_true",
                    help="skip email_ids already in the log; do not reset the store")
    ap.add_argument("--selftest", action="store_true",
                    help="offline selftest with injected fakes")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    if not (args.phase and args.dataset and args.namespace):
        ap.error("--phase, --dataset and --namespace are required (unless --selftest)")

    from dotenv import load_dotenv
    load_dotenv("/media/adnan/DATA/Agentic-LongTerm-Memory/.env")
    cfg = build_config(args.phase, args.dataset, args.namespace, args.seed,
                       generate_responses=not args.no_responses)
    deps = _build_real_deps(cfg, args.namespace)
    run(cfg, args.namespace, deps, limit=args.limit, resume=args.resume)


if __name__ == "__main__":
    main()
