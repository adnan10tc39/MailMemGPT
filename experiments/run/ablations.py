"""Ablation runs and sensitivity sweeps over the synthetic dataset.

Ablation runs reuse the run_phase machinery with an overridden ExpConfig
(always generate_responses=False; prompt assembly still runs so token/dedup
metrics are real), store namespaces abl_<name>:
    no_dedup, no_budget, vector_only, naive_rag
Sensitivity sweeps (triage-only / assembly-only, cheap):
    sweep_tau    tau in {0.27,0.32,0.37,0.42,0.47} — ONE embedding+query pass
                 records per-class max similarities per email, re-thresholds
                 offline; LLM fallback only for emails whose decision path
                 flips, cached across settings.
    sweep_k      k in {1,3,5,7,10}       — pipeline assembly reruns only
    sweep_delta  delta in {0.60,...,0.80} — pipeline assembly reruns only
    sweep_budget B in {4000,...,12000}    — pipeline assembly reruns only
Each sweep writes results/<sweep_name>/summary.json.

CLI:
    python3 -m experiments.run.ablations --which no_dedup|...|all
Selftest (no OpenAI, no Postgres):
    python3 -m experiments.run.ablations --selftest
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from experiments.analysis.metrics import CLASSES, _dist, confusion, prf
from experiments.run import run_phase
from experiments.run.run_phase import (RunDeps, _extract_pipeline_info, _get,
                                       _split_pipeline_output, call_flexible,
                                       email_meta, _email_doc)

REPO_ROOT = Path("/media/adnan/DATA/Agentic-LongTerm-Memory")
RESULTS_ROOT = REPO_ROOT / "experiments" / "results"
DEFAULT_DATASET = str(REPO_ROOT / "experiments" / "data" / "synthetic_500.jsonl")

ABLATION_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "no_dedup": {"enable_dedup": False},
    "no_budget": {"enable_budget": False},
    "vector_only": {"enable_sql_history": False},
    "naive_rag": {"naive_rag": True},
    # Summarization-only memory (R2 baseline): rolling summary retained, but no
    # raw recent-history pairs and no vector retrieval.
    "summary_only": {"enable_vector_retrieval": False, "history_window": 0},
}
# Wide grids covering the text-embedding-3-small cosine scale (typical
# email-to-exemplar max-sims 0.25-0.65) with headroom for higher-similarity
# scales. tau=0.37 is the original manuscript's operating point.
SWEEP_VALUES: Dict[str, List[Any]] = {
    "sweep_tau": [0.25, 0.30, 0.37, 0.45, 0.55, 0.65, 0.75],
    "sweep_k": [1, 3, 5, 7, 10],
    "sweep_delta": [0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
    "sweep_budget": [4000, 6000, 8000, 12000],
    "sweep_window": [2, 5, 10, 20],
}
SWEEP_PARAM_FIELD: Dict[str, str] = {
    "sweep_k": "retrieval_k",
    "sweep_delta": "retrieval_delta",
    "sweep_budget": "prompt_budget",
    "sweep_window": "history_window",
}
WHICH_CHOICES = list(ABLATION_OVERRIDES) + list(SWEEP_VALUES) + ["all"]


def _override(cfg: Any, **kw: Any) -> Any:
    """Copy a config (dataclass or namespace) with overridden fields."""
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        return dataclasses.replace(cfg, **kw)
    return SimpleNamespace(**{**vars(cfg), **kw})


def _score(y_true: Sequence[str], y_pred: Sequence[str]) -> Tuple[float, float]:
    """(accuracy, macro_f1) via the hand-built metrics implementation."""
    rep = prf(confusion(y_true, y_pred))
    return rep["accuracy"], rep["macro_f1"]


def _write_summary(out_dir: Path, summary: Dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return path


# ------------------------------------------------------------------ tau sweep

def default_embed_text(email: dict) -> str:
    """Paper's triage embedding text: subject + first 200 chars of body."""
    return f"{email.get('subject', '')}\n{email.get('body', '')[:200]}"


def sweep_tau(cfg: Any, emails: Sequence[dict], llm: Any, store: Any,
              fallback_fn: Callable[[dict], Tuple[str, float]],
              out_dir: Path, taus: Optional[Sequence[float]] = None,
              embed_text_fn: Optional[Callable[[dict], str]] = None,
              progress: bool = True) -> Dict[str, Any]:
    """Tau sensitivity sweep with a single embedding+query pass.

    Records per-class max cosine similarities once per email, then
    re-thresholds offline for every tau; fallback_fn (an LLM call) is invoked
    only for emails whose decision path flips to fallback at some tau, and its
    result is cached across settings.
    """
    taus = list(taus if taus is not None else SWEEP_VALUES["sweep_tau"])
    embed_text_fn = embed_text_fn or default_embed_text

    records: List[Dict[str, Any]] = []
    for i, email in enumerate(emails):
        vec = llm.embed([embed_text_fn(email)]).vectors[0]
        sims = {}
        for c in CLASSES:
            rows = store.query_fewshot(c, vec, cfg.triage_topn)
            sims[c] = max((float(s) for s, _ in rows), default=-1.0)
        records.append({"email_id": email.get("email_id"),
                        "label": email.get("label"), "sims": sims})
        if progress and (i + 1) % 50 == 0:
            print(f"[sweep_tau] embedded+queried {i + 1}/{len(emails)}")

    fallback_cache: Dict[str, Tuple[str, float]] = {}
    settings: List[Dict[str, Any]] = []
    for tau in taus:
        y_true: List[str] = []
        y_pred: List[str] = []
        n_semantic = 0
        for rec, email in zip(records, emails):
            sims = rec["sims"]
            smax = max(sims.values())
            if smax >= tau:
                pred = max(CLASSES, key=lambda c: sims[c])
                n_semantic += 1
            else:
                eid = rec["email_id"]
                if eid not in fallback_cache:
                    fallback_cache[eid] = fallback_fn(email)
                pred = fallback_cache[eid][0]
            y_true.append(rec["label"])
            y_pred.append(pred)
        acc, mf1 = _score(y_true, y_pred)
        n = len(records)
        settings.append({"tau": tau, "accuracy": acc, "macro_f1": mf1, "n": n,
                         "n_semantic": n_semantic, "n_fallback": n - n_semantic,
                         "semantic_pct": 100.0 * n_semantic / n if n else 0.0})
        if progress:
            print(f"[sweep_tau] tau={tau:.2f} acc={acc:.4f} "
                  f"semantic={n_semantic}/{n}")

    summary = {"sweep": "sweep_tau", "param": "triage_tau",
               "dataset": cfg.dataset_path, "seed": getattr(cfg, "seed", None),
               "created": datetime.now(timezone.utc).isoformat(),
               "n_fallback_llm_calls": len(fallback_cache),
               "per_email_sims": records, "settings": settings}
    _write_summary(out_dir, summary)
    return summary


# -------------------------------------------------------- k/delta/B sweeps

def sweep_pipeline(cfg: Any, emails: Sequence[dict], llm: Any, store: Any,
                   pipeline_fn: Callable[..., Any], param_field: str,
                   values: Sequence[Any], out_dir: Path, sweep_name: str,
                   progress: bool = True) -> Dict[str, Any]:
    """k/delta/budget sensitivity sweep: rerun pipeline assembly only.

    One chronological pass builds SQL/vector memory state (using TRUE labels,
    no triage LLM); for each true-RESPOND email the pipeline assembly runs
    once per setting to collect token/dedup metrics. No response generation;
    the only possible LLM use is the pipeline's own budget-summarization
    fallback. Triage accuracy is not re-measured here (semantic triage is
    independent of these parameters), so accuracy is reported as null.
    """
    cfgs = {v: _override(cfg, **{param_field: v}) for v in values}
    stats: Dict[Any, List[Dict[str, Any]]] = {v: [] for v in values}

    for i, email in enumerate(emails):
        if email.get("label") == "RESPOND":
            for v in values:
                cv = cfgs[v]
                out = call_flexible(pipeline_fn, email=email, cfg=cv, llm=llm,
                                    store=store, phase=cv.phase)
                _, pstats = _split_pipeline_output(out)
                _, retrieval, fpt = _extract_pipeline_info(pstats, cv.prompt_budget)
                stats[v].append({
                    "final_prompt_tokens": fpt,
                    "budget_used_pct": (100.0 * fpt / cv.prompt_budget
                                        if fpt is not None else None),
                    "n_raw": retrieval["n_raw"], "n_kept": retrieval["n_kept"],
                    "n_deduped": retrieval["n_deduped"],
                    "tokens_saved": retrieval["tokens_saved"],
                    "sims": retrieval["sims"]})
        # mutate memory state AFTER assembling, mirroring the main-run order
        store.save_interaction(email, email.get("label"), 1.0, None)
        if email.get("label") == "RESPOND":
            doc = _email_doc(email, None)
            store.add_memory(doc, llm.embed([doc]).vectors[0], email_meta(email))
        if progress and (i + 1) % 50 == 0:
            print(f"[{sweep_name}] processed {i + 1}/{len(emails)}")

    settings: List[Dict[str, Any]] = []
    for v in values:
        rows = stats[v]
        utils = [r["budget_used_pct"] for r in rows
                 if r["budget_used_pct"] is not None]
        saved = [r["tokens_saved"] for r in rows if r["tokens_saved"] is not None]
        sims = [s for r in rows for s in (r["sims"] or [])]
        setting = {
            param_field: v,
            "n_assemblies": len(rows),
            "accuracy": None,  # triage unaffected by this parameter; see docstring
            "final_prompt_tokens": _dist([r["final_prompt_tokens"] for r in rows]),
            "budget_used_pct": _dist(utils),
            "pct_over_budget": (100.0 * sum(1 for u in utils if u > 100.0)
                                / len(utils) if utils else None),
            "n_kept_mean": _mean([r["n_kept"] for r in rows]),
            "n_deduped_mean": _mean([r["n_deduped"] for r in rows]),
            "tokens_saved_mean": _mean(saved),
            "tokens_saved_total": float(sum(saved)) if saved else None,
            "retrieval_sim_mean": _mean(sims),
        }
        settings.append(setting)
        if progress:
            fpt = setting["final_prompt_tokens"]
            print(f"[{sweep_name}] {param_field}={v} "
                  f"final_tokens_mean={fpt['mean'] if fpt else None}")

    summary = {"sweep": sweep_name, "param": param_field,
               "dataset": cfg.dataset_path, "seed": getattr(cfg, "seed", None),
               "created": datetime.now(timezone.utc).isoformat(),
               "settings": settings}
    _write_summary(out_dir, summary)
    return summary


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


# --------------------------------------------------------------- real wiring

def _make_fallback_fn(cfg: Any, llm: Any, store: Any) -> Callable[[dict], Tuple[str, float]]:
    """LLM-fallback classifier for the tau sweep, from the triage module.

    Prefers a dedicated fallback helper; otherwise forces the full triage
    routine down the fallback path with an unreachable tau.
    """
    from experiments.agent import triage as triage_mod
    fb = run_phase._resolve_optional(
        triage_mod, "llm_fallback", "fallback_classify", "classify_fallback",
        "llm_classify", "fallback")
    if fb is None:
        triage_fn = run_phase._resolve(triage_mod, "classify", "triage_email",
                                       "triage", "classify_email", "run_triage")
        forced = _override(cfg, triage_tau=2.0)

        def fallback(email: dict) -> Tuple[str, float]:
            res = call_flexible(triage_fn, email=email, cfg=forced, llm=llm,
                                store=store, phase=forced.phase)
            return (_get(res, "label", "classification", default="RESPOND"),
                    float(_get(res, "confidence", default=0.5) or 0.5))
        return fallback

    def fallback(email: dict) -> Tuple[str, float]:
        res = call_flexible(fb, email=email, cfg=cfg, llm=llm, store=store,
                            phase=cfg.phase)
        return (_get(res, "label", "classification", default="RESPOND"),
                float(_get(res, "confidence", default=0.5) or 0.5))
    return fallback


def _base_config(name: str, dataset: str, seed: int,
                 overrides: Dict[str, Any]) -> Any:
    return run_phase.build_config("p3", dataset, f"abl_{name}", seed,
                                  generate_responses=False, **overrides)


def run_ablation(name: str, dataset: str, seed: int, limit: Optional[int],
                 results_root: Path) -> Dict[str, Any]:
    """One ablation run (run_phase machinery, namespace abl_<name>)."""
    cfg = _base_config(name, dataset, seed, ABLATION_OVERRIDES[name])
    namespace = f"abl_{name}"
    deps = run_phase._build_real_deps(cfg, namespace)
    return run_phase.run(cfg, namespace, deps, limit=limit,
                         results_root=results_root)


def run_sweep(name: str, dataset: str, seed: int, limit: Optional[int],
              results_root: Path) -> Dict[str, Any]:
    """One sensitivity sweep, writing results/<name>/summary.json."""
    cfg = _base_config(name, dataset, seed, {})
    namespace = f"abl_{name}"
    deps = run_phase._build_real_deps(cfg, namespace)
    run_phase.seed_store(cfg, deps)  # fresh namespace: profile/rules + few-shot
    emails = run_phase.load_dataset(dataset)
    if limit is not None:
        emails = emails[:limit]
    out_dir = results_root / name
    if name == "sweep_tau":
        from experiments.agent import triage as triage_mod
        embed_text_fn = run_phase._resolve_optional(
            triage_mod, "embed_text", "email_embed_text", "embedding_text",
            "triage_embed_text") or default_embed_text
        fallback_fn = _make_fallback_fn(cfg, deps.llm, deps.store)
        return sweep_tau(cfg, emails, deps.llm, deps.store, fallback_fn,
                         out_dir, embed_text_fn=embed_text_fn)
    return sweep_pipeline(cfg, emails, deps.llm, deps.store, deps.pipeline_fn,
                          SWEEP_PARAM_FIELD[name], SWEEP_VALUES[name],
                          out_dir, name)


# ------------------------------------------------------------------ selftest

def _selftest() -> None:
    """Offline checks of tau re-thresholding + fallback caching, pipeline
    sweep aggregation, and an ablation run via run_phase with fakes."""
    scratch = Path(os.environ.get(
        "MAILRECALL_SELFTEST_DIR",
        "/tmp/claude-1000/-media-adnan-DATA-Agentic-LongTerm-Memory/"
        "250f06c2-b063-439d-83ca-fb5215f31351/scratchpad")) / "ablations_selftest"
    scratch.mkdir(parents=True, exist_ok=True)

    emails = run_phase.fixture_emails()  # RESPOND,RESPOND,IGNORE,IGNORE,NOTIFY
    # --- sweep_tau: controlled sims per email via an index-encoding fake LLM
    sims_table = [
        {"IGNORE": 0.10, "NOTIFY": 0.20, "RESPOND": 0.45},
        {"IGNORE": 0.10, "NOTIFY": 0.20, "RESPOND": 0.30},
        {"IGNORE": 0.50, "NOTIFY": 0.10, "RESPOND": 0.10},
        {"IGNORE": 0.35, "NOTIFY": 0.30, "RESPOND": 0.10},
        {"IGNORE": 0.10, "NOTIFY": 0.40, "RESPOND": 0.20},
    ]
    id_to_idx = {e["email_id"]: i for i, e in enumerate(emails)}

    class IndexLLM(run_phase.FakeLLM):
        def embed(self, texts: Sequence[str]) -> Any:
            self.usage_totals["embed_tokens"] += len(texts)
            return SimpleNamespace(vectors=[[id_to_idx[t]] for t in texts],
                                   latency_ms=0.1, tokens=len(texts))

    class SimStore(run_phase.FakeStore):
        def query_fewshot(self, label: str, vec: Any, topn: int) -> List[Tuple[float, str]]:
            return [(sims_table[int(vec[0])][label], "doc")]

    fallback_answers = {0: "RESPOND", 1: "RESPOND", 3: "NOTIFY", 4: "NOTIFY"}
    fallback_calls: List[str] = []

    def fake_fallback(email: dict) -> Tuple[str, float]:
        fallback_calls.append(email["email_id"])
        return fallback_answers[id_to_idx[email["email_id"]]], 0.6

    cfg = run_phase.fake_config(dataset_path="fixture", generate_responses=False)
    summary = sweep_tau(cfg, emails, IndexLLM(), SimStore(), fake_fallback,
                        scratch / "sweep_tau",
                        taus=[0.27, 0.32, 0.37, 0.42, 0.47],
                        embed_text_fn=lambda e: e["email_id"], progress=False)
    accs = [s["accuracy"] for s in summary["settings"]]
    n_sem = [s["n_semantic"] for s in summary["settings"]]
    assert accs == [1.0, 1.0, 0.8, 0.8, 0.8], accs
    assert n_sem == [5, 4, 3, 2, 1], n_sem
    # fallback called once per flipped email (cached across taus): e0,e1,e3,e4
    assert len(fallback_calls) == 4 and len(set(fallback_calls)) == 4, fallback_calls
    assert summary["n_fallback_llm_calls"] == 4
    assert (scratch / "sweep_tau" / "summary.json").exists()

    # --- sweep_pipeline: metrics track the overridden parameter
    def fake_pipe(email: dict, cfg: Any, llm: Any, store: Any) -> Tuple[str, Any]:
        stats = SimpleNamespace(stages={}, final_prompt_tokens=cfg.retrieval_k * 100,
                                n_raw=10, n_kept=cfg.retrieval_k, n_deduped=1,
                                tokens_saved=50, sims=[0.9])
        return ("P", stats)

    store2 = run_phase.FakeStore()
    summary2 = sweep_pipeline(cfg, emails, run_phase.FakeLLM(), store2,
                              fake_pipe, "retrieval_k", [1, 3, 5],
                              scratch / "sweep_k", "sweep_k", progress=False)
    for setting, k in zip(summary2["settings"], [1, 3, 5]):
        assert setting["retrieval_k"] == k
        assert setting["n_assemblies"] == 2  # two true-RESPOND fixtures
        assert abs(setting["final_prompt_tokens"]["mean"] - k * 100) < 1e-9
        assert abs(setting["budget_used_pct"]["mean"] - k * 100 / 8000 * 100) < 1e-9
        assert abs(setting["n_kept_mean"] - k) < 1e-9
        assert abs(setting["tokens_saved_total"] - 100.0) < 1e-9
        assert setting["accuracy"] is None
    assert len(store2.interactions) == 5  # one save per email, not per setting
    assert len(store2.memories) == 2
    assert (scratch / "sweep_k" / "summary.json").exists()

    # --- ablation run through run_phase.run with fakes (no responses)
    ds = scratch / "fixture.jsonl"
    with open(ds, "w", encoding="utf-8") as fh:
        for e in emails:
            fh.write(json.dumps(e) + "\n")
    abl_cfg = run_phase.fake_config(dataset_path=str(ds),
                                    generate_responses=False,
                                    enable_dedup=False, run_id="abl_no_dedup")
    deps = run_phase.make_fake_deps()
    out = run_phase.run(abl_cfg, "abl_no_dedup", deps, results_root=scratch,
                        progress=False)
    assert out["n_processed"] == 5 and out["n_errors"] == 0, out
    meta = json.loads((scratch / "abl_no_dedup" / "run_meta.json").read_text())
    assert meta["config"]["enable_dedup"] is False
    assert meta["config"]["generate_responses"] is False
    rows = [json.loads(l) for l in
            (scratch / "abl_no_dedup" / "log.jsonl").read_text().splitlines()]
    respond_rows = [r for r in rows if r["label_pred"] == "RESPOND"]
    assert len(respond_rows) == 2
    for r in respond_rows:  # assembly ran, response generation skipped
        assert r["tokens"]["final_prompt_tokens"] == 1600
        assert r["response_len_chars"] == 0
        assert r["latencies"]["respond_llm"] is None
    print("ablations selftest OK:", scratch)


# ----------------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(description="Ablations and sensitivity sweeps.")
    ap.add_argument("--which", choices=WHICH_CHOICES,
                    help="ablation/sweep to run ('all' for everything; the "
                    "sql_only ablation is the p1 main run, reuse it)")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--results-root", default=str(RESULTS_ROOT))
    ap.add_argument("--selftest", action="store_true",
                    help="offline selftest with injected fakes")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    if not args.which:
        ap.error("--which is required (unless --selftest)")

    from dotenv import load_dotenv
    load_dotenv("/media/adnan/DATA/Agentic-LongTerm-Memory/.env")
    results_root = Path(args.results_root)
    names = (list(ABLATION_OVERRIDES) + list(SWEEP_VALUES)
             if args.which == "all" else [args.which])
    for name in names:
        print(f"=== {name} ===")
        if name in ABLATION_OVERRIDES:
            run_ablation(name, args.dataset, args.seed, args.limit, results_root)
        else:
            run_sweep(name, args.dataset, args.seed, args.limit, results_root)


if __name__ == "__main__":
    main()
