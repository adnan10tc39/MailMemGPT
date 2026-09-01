"""End-to-end offline dry run of the full experiment harness.

Wires a deterministic FakeLLM (canned chat outputs + cfg.embed_dim-d
bag-of-words hash embeddings, zero network calls) into the REAL Postgres Store (namespace
``dryrun``, reset at start) and the REAL triage/pipeline/respond/run_phase
code, over 8 hand-written fixture emails covering all three classes, a
duplicate-in-history retrieval case, and an over-budget prompt case.

Asserts:
- log.jsonl has 8 valid lines matching the pinned log schema;
- the confusion matrix is computable by analysis.metrics (and equals the
  expected matrix);
- archive day files were written for every processed email;
- budget actions were recorded by the real budget stage on the over-budget
  email (and on no other).

Run from the repo root:
    python3 -m experiments.run.dryrun
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from experiments.analysis import metrics as metrics_mod
from experiments.common.exp_config import ExpConfig
from experiments.run import run_phase

REPO_ROOT = Path("/media/adnan/DATA/Agentic-LongTerm-Memory")
RESULTS_ROOT = REPO_ROOT / "experiments" / "results"
ARCHIVE_ROOT = REPO_ROOT / "experiments" / "data" / "archive"
DATA_DIR = REPO_ROOT / "experiments" / "data"
NAMESPACE = "dryrun"
# Fake vectors must match the store's configured vector(embed_dim) columns.
EMBED_DIM = ExpConfig("p3", "dryrun", "").embed_dim
CLASSES = ("IGNORE", "NOTIFY", "RESPOND")

# ---------------------------------------------------------------------------
# Deterministic fake LLM
# ---------------------------------------------------------------------------


def _bow_vec(text: str) -> List[float]:
    """EMBED_DIM-d bag-of-words embedding via md5 word hashing, L2-normalized.

    Deterministic across processes (no salted hash()); texts sharing
    vocabulary get high cosine similarity, disjoint texts get ~0.
    """
    vec = [0.0] * EMBED_DIM
    for word in re.findall(r"[a-z0-9'/\-]+", text.lower()):
        h = int(hashlib.md5(word.encode("utf-8")).hexdigest()[:12], 16)
        vec[h % EMBED_DIM] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _extract_subject(user_content: str) -> str:
    """Pull the Subject: line out of a formatted incoming-email message."""
    m = re.search(r"^Subject: (.*)$", user_content, re.MULTILINE)
    return m.group(1).strip() if m else "your email"


class DryrunLLM:
    """Deterministic stand-in for common.llm.LLM (pinned interface subset).

    chat() routes on message content: rolling-summary updates, the budget
    stage's retrieved-memories compression, the triage JSON fallback, and the
    respond loop (one canned calendar tool call for the PROJ-TIMELINE-42
    email, canned replies otherwise). embed() is _bow_vec.
    """

    def __init__(self) -> None:
        self.usage_totals: Dict[str, float] = {
            "prompt_tokens": 0, "completion_tokens": 0,
            "embed_tokens": 0, "cost_usd": 0.0}
        self.chat_calls: List[dict] = []
        self.embed_calls: List[List[str]] = []

    # ---------------------------------------------------------------- chat
    def chat(self, messages: List[dict], functions: Any = None,
             function_call: Any = None, model: Optional[str] = None,
             temperature: Optional[float] = None,
             max_tokens: Optional[int] = None) -> SimpleNamespace:
        system = str(messages[0].get("content") or "")
        user = ""
        for m in messages[1:]:
            if m.get("role") == "user":
                user = str(m.get("content") or "")
        self.chat_calls.append({"model": model, "functions": functions,
                                "function_call": function_call,
                                "system": system, "user": user})
        fc: Optional[dict] = None
        if "update running summary of older interactions" in user:
            content: Optional[str] = ("Rolling summary: John handled API and "
                                      "meeting threads.")
        elif "Summarize the following retrieved email memories" in user:
            content = ("COMPRESSED MEMORY SUMMARY: Alice asked about missing "
                       "auth endpoints; John replied.")
        elif "email triage assistant" in system:
            # Triage LLM fallback: strict JSON classification.
            if "zorply" in user:
                content = '{"classification": "NOTIFY", "confidence": 0.8}'
            else:
                content = '{"classification": "RESPOND", "confidence": 0.55}'
        elif (functions is not None and "PROJ-TIMELINE-42" in user
              and "# Prior Tool Results" not in system
              and function_call != "none"):
            # Respond loop, meeting email, no tool results yet -> call a tool.
            content = None
            fc = {"name": "check_calendar_availability",
                  "arguments": {"date": "Tuesday"}}
        else:
            subject = _extract_subject(user)
            content = (f"Hi, regarding '{subject}': happy to help — I'll "
                       f"follow up with the details today. Best, John")
        pt = self.count_tokens(system) + self.count_tokens(user)
        ct = self.count_tokens(content or "") + (8 if fc else 0)
        self.usage_totals["prompt_tokens"] += pt
        self.usage_totals["completion_tokens"] += ct
        return SimpleNamespace(content=content, function_call=fc,
                               prompt_tokens=pt, completion_tokens=ct,
                               latency_ms=1.0)

    # --------------------------------------------------------------- embed
    def embed(self, texts: List[str]) -> SimpleNamespace:
        self.embed_calls.append(list(texts))
        tokens = sum(self.count_tokens(t) for t in texts)
        self.usage_totals["embed_tokens"] += tokens
        return SimpleNamespace(vectors=[_bow_vec(t) for t in texts],
                               latency_ms=0.5, tokens=tokens)

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Fixture emails
# ---------------------------------------------------------------------------

_MEETING_FILLER = (
    "Following up on the meeting to discuss the upcoming project timeline. "
    "Would a thirty minute call work this week? Let me know your "
    "availability. "
)


def fixture_emails() -> List[dict]:
    """8 fixture emails: 5 RESPOND / 2 IGNORE / 1 NOTIFY over 4 days.

    dr-0005 retrieves the out-of-window dr-0001 memory (kept); dr-0007
    retrieves the in-window dr-0004 memory (deduped: duplicate-in-history);
    dr-0008 assembles over budget because dr-0007's huge body sits in its
    history window, while its retrieval hits dr-0001 (kept) and dr-0005
    (deduped).
    """
    return [
        {"email_id": "dr-0001", "day": 1, "ts": "2026-02-02T09:00:00",
         "thread_id": "t-api",
         "sender": "Alice Smith <alice.smith@company.com>",
         "subject": "Quick question about API documentation",
         "body": ("Hi John,\n\nI was reviewing the API documentation for the "
                  "new authentication service and noticed two endpoints seem "
                  "to be missing from the specs. Could you help clarify if "
                  "this was intentional or if we should update the docs?\n\n"
                  "Specifically, I'm looking at:\n- /auth/refresh\n"
                  "- /auth/validate\n\nThanks!\nAlice"),
         "label": "RESPOND", "requires_context": False},
        {"email_id": "dr-0002", "day": 1, "ts": "2026-02-02T11:30:00",
         "thread_id": "t-promo",
         "sender": "Promotions <promo@dealsrus.example.com>",
         "subject": "Special Offer - 50% Off Everything",
         "body": ("Limited time offer! Get 50% off on all products.\n\n"
                  "Use code: SAVE50\n\nShop now before the offer ends!"),
         "label": "IGNORE", "requires_context": False},
        {"email_id": "dr-0003", "day": 2, "ts": "2026-02-03T08:15:00",
         "thread_id": "t-build",
         "sender": "Build System <build@company.com>",
         "subject": "Build Successful - Project v2.2",
         "body": ("Build completed successfully.\n\nProject: v2.2\n"
                  "Status: Passed\nDuration: 14 minutes\n\n"
                  "No action required."),
         "label": "NOTIFY", "requires_context": False},
        {"email_id": "dr-0004", "day": 2, "ts": "2026-02-03T10:00:00",
         "thread_id": "t-meet",
         "sender": "Bob Johnson <bob.johnson@company.com>",
         "subject": "Meeting Request - Project Discussion",
         "body": ("Hi John,\n\nI'd like to schedule a meeting to discuss the "
                  "upcoming project timeline (PROJ-TIMELINE-42). Would you be "
                  "available for a 30-minute call this week?\n\nLet me know "
                  "your availability.\n\nBest regards,\nBob"),
         "label": "RESPOND", "requires_context": False},
        {"email_id": "dr-0005", "day": 3, "ts": "2026-02-04T09:20:00",
         "thread_id": "t-api",
         "sender": "Alice Smith <alice.smith@company.com>",
         "subject": "Re: Quick question about API documentation",
         "body": ("Hi John,\n\nFollowing up on the API documentation question "
                  "about the new authentication service endpoints "
                  "/auth/refresh and /auth/validate — were the missing "
                  "endpoints intentional, or should we update the docs and "
                  "the specs?\n\nThanks!\nAlice"),
         "label": "RESPOND", "requires_context": True,
         "context_note": "follow-up referencing dr-0001"},
        {"email_id": "dr-0006", "day": 3, "ts": "2026-02-04T14:45:00",
         "thread_id": "t-raffle",
         "sender": "Zorply Prizes <win@zorply-amulets.example.biz>",
         "subject": "Snoof brimble cryptid amulet raffle",
         "body": ("Grabulon zorply amulet raffle! Brimble the snoof quambit "
                  "vex jolute. Clim the drazzle bortle wunkle vashtorn."),
         "label": "IGNORE", "requires_context": False},
        {"email_id": "dr-0007", "day": 4, "ts": "2026-02-05T09:05:00",
         "thread_id": "t-meet",
         "sender": "Bob Johnson <bob.johnson@company.com>",
         "subject": "Re: Meeting Request - Project Discussion",
         "body": ("Hi John,\n\nFollowing up on the meeting to discuss the "
                  "upcoming project timeline (TIMELINE-FOLLOWUP-77) — would a "
                  "30-minute call work this week? Let me know your "
                  "availability.\n\nBest regards,\nBob\n\n"
                  + _MEETING_FILLER * 60),
         "label": "RESPOND", "requires_context": True,
         "context_note": "follow-up referencing dr-0004"},
        {"email_id": "dr-0008", "day": 4, "ts": "2026-02-05T16:30:00",
         "thread_id": "t-api",
         "sender": "David Lee <david.lee@company.com>",
         "subject": "Re: Quick question about API documentation",
         "body": ("Hi John,\n\nI need your approval to update the API "
                  "documentation for the new authentication service endpoints "
                  "/auth/refresh and /auth/validate discussed earlier. Can "
                  "you review the missing endpoints in the specs and approve "
                  "the docs update?\n\nThanks,\nDavid"),
         "label": "RESPOND", "requires_context": True,
         "context_note": "follow-up referencing dr-0001/dr-0005"},
    ]


# ---------------------------------------------------------------------------
# Log-schema validation (pinned run_phase schema)
# ---------------------------------------------------------------------------

_TOP_KEYS = {"email_id", "day", "label_true", "label_pred", "confidence",
             "method", "tool_calls", "latencies", "tokens", "retrieval",
             "response_len_chars", "error"}
_TOKEN_KEYS = {"prompt", "completion", "final_prompt_tokens",
               "budget_used_pct"}
_RETRIEVAL_KEYS = {"n_raw", "n_kept", "n_deduped", "sims", "tokens_saved"}
_METHODS = {"semantic", "llm_p1", "llm_p2", "llm_p3"}


def _validate_log_row(row: dict) -> None:
    """Hard-assert one log line against the pinned per-email log schema."""
    assert set(row) == _TOP_KEYS, set(row) ^ _TOP_KEYS
    assert isinstance(row["email_id"], str) and row["email_id"]
    assert isinstance(row["day"], int)
    assert row["label_true"] in CLASSES, row["label_true"]
    assert row["label_pred"] in CLASSES, row["label_pred"]
    assert isinstance(row["confidence"], (int, float))
    assert 0.0 <= float(row["confidence"]) <= 1.0
    assert row["method"] in _METHODS, row["method"]
    assert isinstance(row["tool_calls"], list)
    assert set(row["latencies"]) == set(run_phase.LOG_LATENCY_KEYS)
    for key, val in row["latencies"].items():
        assert val is None or isinstance(val, (int, float)), (key, val)
    assert row["latencies"]["total"] is not None
    assert row["latencies"]["triage_embed"] is not None
    assert set(row["tokens"]) == _TOKEN_KEYS, set(row["tokens"])
    assert set(row["retrieval"]) == _RETRIEVAL_KEYS, set(row["retrieval"])
    assert isinstance(row["retrieval"]["sims"], list)
    assert isinstance(row["response_len_chars"], int)
    assert row["response_len_chars"] >= 0
    assert row["error"] is None, row["error"]
    if row["label_pred"] == "RESPOND":
        # Pipeline ran: real per-stage latencies and budget accounting.
        for stage in ("sql_load", "retrieve", "dedup", "budget", "assemble",
                      "respond_llm"):
            assert row["latencies"][stage] is not None, (row["email_id"], stage)
        assert isinstance(row["tokens"]["final_prompt_tokens"], int)
        assert isinstance(row["tokens"]["budget_used_pct"], (int, float))
        assert row["response_len_chars"] > 0
    else:
        assert row["tokens"]["final_prompt_tokens"] is None
        assert row["response_len_chars"] == 0


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the offline end-to-end dry run and hard-assert every requirement."""
    dataset_path = DATA_DIR / "dryrun_fixture_8.jsonl"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    emails = fixture_emails()
    with open(dataset_path, "w", encoding="utf-8") as fh:
        for e in emails:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")

    # Dryrun-scale config: window=3 so dr-0001 falls out of dr-0005's history
    # (kept retrieval) while dr-0004 stays inside dr-0007's (dedup); B=1500 so
    # dr-0007's huge body pushes dr-0008 over budget; delta=0.55 calibrates
    # the paper's 0.70 gate to the fake bag-of-words cosine scale, where
    # true follow-up matches score ~0.58-0.70 and unrelated docs ~0.15-0.31
    # (both keep and filter directions of the delta gate stay exercised).
    cfg = run_phase.build_config(
        "p3", str(dataset_path), NAMESPACE, seed=13, generate_responses=True,
        history_window=3, prompt_budget=1500, retrieval_delta=0.55)

    # Real wiring (real Store, real triage/pipeline/respond resolution) with
    # the network LLM swapped for the deterministic fake.
    deps = run_phase._build_real_deps(cfg, NAMESPACE)
    llm = DryrunLLM()
    deps.llm = llm

    # Wrap the real pipeline to record the budget stage's actions per email.
    real_pipeline_fn = deps.pipeline_fn
    budget_actions: Dict[str, List[str]] = {}

    def recording_pipeline(email: dict, cfg: Any, store: Any, llm: Any,
                           tool_results_text: str = "") -> Any:
        prompt, stats = real_pipeline_fn(
            email, cfg, store, llm, tool_results_text=tool_results_text)
        acts = [a["action"] for a in stats.budget_actions]
        budget_actions.setdefault(email["email_id"], []).extend(acts)
        return prompt, stats

    deps.pipeline_fn = recording_pipeline

    print(f"[dryrun] processing {len(emails)} fixture emails "
          f"(namespace={NAMESPACE}, B={cfg.prompt_budget}, "
          f"window={cfg.history_window})")
    out = run_phase.run(cfg, NAMESPACE, deps, results_root=RESULTS_ROOT,
                        progress=False)
    assert out["n_processed"] == 8 and out["n_errors"] == 0, out

    # ------------------------------------------------ 1) log schema (8 rows)
    log_path = Path(out["log_path"])
    rows = [json.loads(line) for line in
            log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 8, len(rows)
    for row in rows:
        _validate_log_row(row)
    by_id = {r["email_id"]: r for r in rows}
    assert list(by_id) == [e["email_id"] for e in emails]  # dataset order

    # Expected routing: dr-0006 falls back to the LLM (and is misclassified
    # IGNORE->NOTIFY); everything else classifies semantically.
    expected_pred = {"dr-0001": "RESPOND", "dr-0002": "IGNORE",
                     "dr-0003": "NOTIFY", "dr-0004": "RESPOND",
                     "dr-0005": "RESPOND", "dr-0006": "NOTIFY",
                     "dr-0007": "RESPOND", "dr-0008": "RESPOND"}
    for eid, pred in expected_pred.items():
        assert by_id[eid]["label_pred"] == pred, (eid, by_id[eid]["label_pred"])
    for eid, row in by_id.items():
        expect_method = "llm_p3" if eid == "dr-0006" else "semantic"
        assert row["method"] == expect_method, (eid, row["method"])
    assert by_id["dr-0006"]["confidence"] == 0.8

    # Tool-call loop ran (real respond + real pipeline rebuild) on dr-0004.
    tc = by_id["dr-0004"]["tool_calls"]
    assert len(tc) == 1 and tc[0]["name"] == "check_calendar_availability", tc
    rebuild_calls = [c for c in llm.chat_calls
                     if "# Prior Tool Results" in c["system"]]
    assert rebuild_calls, "prompt was not rebuilt with tool results"
    assert "9:00 AM" in rebuild_calls[0]["system"]
    for eid in ("dr-0001", "dr-0005", "dr-0007", "dr-0008"):
        assert by_id[eid]["tool_calls"] == [], eid

    # Retrieval: kept out-of-window memory on dr-0005; duplicate-in-history
    # dedup on dr-0007 (dr-0004 in the 3-pair window) and dr-0008 (dr-0005).
    r5 = by_id["dr-0005"]["retrieval"]
    assert r5["n_raw"] >= 1 and r5["n_kept"] >= 1, r5
    assert r5["n_deduped"] == 0 and len(r5["sims"]) == r5["n_raw"], r5
    r7 = by_id["dr-0007"]["retrieval"]
    assert r7["n_deduped"] >= 1 and r7["tokens_saved"] > 0, r7
    r8 = by_id["dr-0008"]["retrieval"]
    assert r8["n_kept"] >= 1 and r8["n_deduped"] >= 1, r8

    # ------------------------------------- 2) budget actions (dr-0008 only)
    acts8 = budget_actions.get("dr-0008", [])
    assert "summarize_retrieved" in acts8, acts8
    assert "drop_history_pairs" in acts8, acts8
    assert "over_budget_unresolved" not in acts8, acts8
    for eid in ("dr-0001", "dr-0004", "dr-0005", "dr-0007"):
        assert budget_actions.get(eid, []) == [], (eid, budget_actions[eid])
    assert by_id["dr-0008"]["tokens"]["final_prompt_tokens"] <= cfg.prompt_budget
    assert by_id["dr-0008"]["tokens"]["budget_used_pct"] <= 100.0
    summarize_calls = [
        c for c in llm.chat_calls
        if "Summarize the following retrieved email memories" in c["user"]]
    assert len(summarize_calls) == 1, len(summarize_calls)
    assert summarize_calls[0]["model"] == cfg.summary_model

    # -------------------------------- 3) confusion matrix via analysis.metrics
    analysis_dir = RESULTS_ROOT / "analysis_dryrun"
    metrics = metrics_mod.analyze({NAMESPACE: rows}, analysis_dir, seed=13,
                                  n_boot=200)
    run_m = metrics["runs"][NAMESPACE]
    assert run_m["n_valid"] == 8 and run_m["n_errors"] == 0, run_m
    expected_conf = [[1, 1, 0],  # true IGNORE: dr-0002 ok, dr-0006 -> NOTIFY
                     [0, 1, 0],  # true NOTIFY: dr-0003 ok
                     [0, 0, 5]]  # true RESPOND: all ok
    assert run_m["confusion"] == expected_conf, run_m["confusion"]
    assert abs(run_m["accuracy"] - 7.0 / 8.0) < 1e-12
    for fname in ("metrics.json", "report.txt", f"confusion_{NAMESPACE}.tex"):
        assert (analysis_dir / fname).exists(), fname

    # ---------------------------------------------------- 4) archive written
    archive_dir = ARCHIVE_ROOT / NAMESPACE
    day_files = sorted(p.name for p in archive_dir.glob("day_*.jsonl.gz"))
    assert day_files == ["day_001.jsonl.gz", "day_002.jsonl.gz",
                         "day_003.jsonl.gz", "day_004.jsonl.gz"], day_files
    archived: List[dict] = []
    for name in day_files:
        with gzip.open(archive_dir / name, "rt", encoding="utf-8") as fh:
            archived.extend(json.loads(line) for line in fh if line.strip())
    assert len(archived) == 8, len(archived)
    arch_by_id = {a["email_id"]: a for a in archived}
    assert set(arch_by_id) == set(expected_pred)
    for eid, pred in expected_pred.items():
        if pred == "RESPOND":
            assert arch_by_id[eid]["response_text"], eid
        else:
            assert arch_by_id[eid]["response_text"] is None, eid

    # ------------------------------------------- 5) store state + run meta
    with deps.store._conn.cursor() as cur:
        counts = {}
        for table in ("mr_interactions", "mr_memory", "mr_fewshot"):
            cur.execute(f"SELECT count(*) FROM {table} WHERE namespace = %s",
                        (NAMESPACE,))
            counts[table] = cur.fetchone()[0]
    assert counts["mr_interactions"] == 8, counts
    assert counts["mr_memory"] == 5, counts  # one per RESPOND-classified email
    assert counts["mr_fewshot"] == 17, counts
    assert deps.store.get_user_profile().startswith("John Doe")
    meta = json.loads(Path(out["meta_path"]).read_text(encoding="utf-8"))
    assert meta["finished"] and meta["usage_totals"]["embed_tokens"] > 0
    assert meta["config"]["prompt_budget"] == 1500
    assert meta["usage_totals"]["cost_usd"] == 0.0  # no API spend

    deps.store.close()
    print("[dryrun] OK: 8/8 schema-valid log lines, confusion "
          f"{run_m['confusion']}, accuracy {run_m['accuracy']:.3f}, "
          f"archive files {day_files}, budget actions on dr-0008: {acts8}")
    print(f"[dryrun] log: {log_path}")
    print(f"[dryrun] analysis: {analysis_dir}")


if __name__ == "__main__":
    main()
