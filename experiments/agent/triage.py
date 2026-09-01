"""Two-stage email triage (paper Eq. 1): semantic few-shot match, LLM fallback.

For each class c, s_c = max cosine similarity between the incoming email
embedding and that class's few-shot collection (topn neighbours). If
max_c(s_c) >= tau the classification is semantic; otherwise an LLM fallback
runs, conditioned on rules + phase-dependent context (p1: history only;
p2: reactive tools; p3: proactively retrieved context).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from experiments.agent.pipeline import (
    LABELS,
    format_email_doc,
    format_history_pair,
    retrieved_context_block,
)
from experiments.agent.respond import FUNCTION_SCHEMAS, execute_tool, parse_function_call

# The 17 few-shot examples, copied VERBATIM from
# src/initialize_few_shot_examples.py (5 IGNORE, 5 NOTIFY, 7 RESPOND).
FEWSHOT_EXAMPLES: list[dict] = [
    {
        "label": "IGNORE",
        "sender": "Newsletter <newsletter@company.com>",
        "subject": "Weekly Newsletter - Product Updates",
        "body": """Hi,

This is our weekly newsletter with product updates and promotions.

Check out our latest features and special offers!

Best regards,
Marketing Team"""
    },
    {
        "label": "IGNORE",
        "sender": "noreply@system.com",
        "subject": "System Notification - Maintenance Complete",
        "body": """System maintenance has been completed successfully.

No action required."""
    },
    {
        "label": "IGNORE",
        "sender": "Promotions <promo@company.com>",
        "subject": "Special Offer - 50% Off",
        "body": """Limited time offer! Get 50% off on all products.

Use code: SAVE50

Shop now!"""
    },
    {
        "label": "IGNORE",
        "sender": "Spam <spam@example.com>",
        "subject": "Buy Now! Limited Time",
        "body": """Click here to buy now! Limited time offer!"""
    },
    {
        "label": "IGNORE",
        "sender": "Auto-Reply <noreply@company.com>",
        "subject": "Out of Office - Automatic Reply",
        "body": """I am currently out of office. Will respond when I return."""
    },
    {
        "label": "NOTIFY",
        "sender": "System Admin <admin@company.com>",
        "subject": "Server Maintenance Scheduled",
        "body": """Hi Team,

We have scheduled server maintenance for this weekend (Saturday 2 AM - 4 AM).

Services will be temporarily unavailable during this time.

No action needed from your side.

Thanks!"""
    },
    {
        "label": "NOTIFY",
        "sender": "HR Team <hr@company.com>",
        "subject": "Holiday Calendar 2024",
        "body": """Dear Team,

Please find attached the holiday calendar for 2024.

Mark your calendars!

Best regards,
HR Team"""
    },
    {
        "label": "NOTIFY",
        "sender": "Build System <build@company.com>",
        "subject": "Build Successful - Project v2.1",
        "body": """Build completed successfully.

Project: v2.1
Status: Passed
Duration: 15 minutes

No action required."""
    },
    {
        "label": "NOTIFY",
        "sender": "IT Support <it@company.com>",
        "subject": "System Update Completed",
        "body": """Hello,

The system update has been completed successfully. All services are now running normally.

No action needed from you.

IT Team"""
    },
    {
        "label": "NOTIFY",
        "sender": "HR <hr@company.com>",
        "subject": "New Policy Announcement",
        "body": """Dear All,

We have updated our company policy. Please review the attached document.

Thank you.

HR Department"""
    },
    {
        "label": "RESPOND",
        "sender": "Alice Smith <alice.smith@company.com>",
        "subject": "Quick question about API documentation",
        "body": """Hi John,

I was reviewing the API documentation for the new authentication service and noticed a few endpoints seem to be missing from the specs. Could you help clarify if this was intentional or if we should update the docs?

Specifically, I'm looking at:
- /auth/refresh
- /auth/validate

Thanks!
Alice"""
    },
    {
        "label": "RESPOND",
        "sender": "Bob Johnson <bob.johnson@company.com>",
        "subject": "Meeting Request - Project Discussion",
        "body": """Hi John,

I'd like to schedule a meeting to discuss the upcoming project timeline. Would you be available for a 30-minute call this week?

Let me know your availability.

Best regards,
Bob"""
    },
    {
        "label": "RESPOND",
        "sender": "Sarah Chen <sarah.chen@company.com>",
        "subject": "Follow-up on UI mockups",
        "body": """Hi John,

Any update on the UI mockups for the dashboard that we discussed last week?

Thanks!
Sarah"""
    },
    {
        "label": "RESPOND",
        "sender": "Mike Davis <mike.davis@company.com>",
        "subject": "Urgent: Need help with deployment",
        "body": """Hi John,

We're having issues with the production deployment. Can you help us troubleshoot?

The error is: Connection timeout

Thanks!
Mike"""
    },
    {
        "label": "RESPOND",
        "sender": "Tom Wilson <tom.wilson@company.com>",
        "subject": "Can you help me?",
        "body": """Hi,

Can you help me with this problem? I need your assistance.

Thanks!"""
    },
    {
        "label": "RESPOND",
        "sender": "Lisa Brown <lisa.brown@company.com>",
        "subject": "Question about project",
        "body": """Hello,

I have a question about the project. When will it be completed?

Please let me know.

Lisa"""
    },
    {
        "label": "RESPOND",
        "sender": "David Lee <david.lee@company.com>",
        "subject": "Need your approval",
        "body": """Hi,

I need your approval for the budget proposal. Can you review it?

Thanks,
David"""
    },
]

_TRIAGE_TOOL_NAMES = ("search_similar_emails", "get_user_info")
_MAX_TRIAGE_TOOLS = 2

_JSON_INSTRUCTION = (
    "Classify the incoming email as exactly one of IGNORE, NOTIFY, RESPOND.\n"
    "Base the decision on ALL information above, not only the email's own wording: "
    "if the recent interaction history or the retrieved context records a prior "
    "decision or commitment about this topic, sender, or service (for example, that "
    "the user owns an open action, that the sender is handling the matter and the "
    "user's part is done, that a vendor was engaged or declined, or that a service "
    "is relied upon or was decommissioned), that recorded decision determines the "
    "correct class even when the email itself is neutral or generic. Only when no "
    "such prior context resolves the question should you classify from the email's "
    "wording and the triage rules alone.\n"
    'Reply with STRICT JSON only, no other text: '
    '{"classification": "IGNORE|NOTIFY|RESPOND", "confidence": <number between 0 and 1>}'
)

_RETRY_INSTRUCTION = (
    "Your previous reply was not valid JSON. Reply with ONLY the JSON object "
    '{"classification": "IGNORE|NOTIFY|RESPOND", "confidence": <number between 0 and 1>}.'
)


@dataclass
class TriageResult:
    """Outcome of triage for one email.

    class_sims carries the per-class max similarities (needed by the offline
    tau-sensitivity sweep) in addition to the pinned fields.
    """

    label: str
    confidence: float
    method: str  # "semantic" | "llm_p1" | "llm_p2" | "llm_p3"
    latencies: dict[str, float]  # triage_embed, triage_query, triage_llm (ms)
    tokens: dict[str, int]       # prompt, completion, embed
    tool_calls: list[dict]
    class_sims: dict[str, float] = field(default_factory=dict)


def triage_embed_text(email: dict) -> str:
    """The exact text classify() embeds for Eq. (1) similarity matching.

    Exposed so offline re-thresholding (the tau sensitivity sweep) records
    similarities from the same embedding text the live triage uses.
    """
    return format_email_doc(email)


def seed_fewshot(store: Any, llm: Any) -> int:
    """Embed and store all few-shot examples; returns the number seeded."""
    docs = [format_email_doc(ex) for ex in FEWSHOT_EXAMPLES]
    emb = llm.embed(docs)
    for ex, doc, vec in zip(FEWSHOT_EXAMPLES, docs, emb.vectors):
        store.add_fewshot(ex["label"], doc, vec)
    return len(docs)


def _parse_triage_json(text: str | None) -> tuple[str, float] | None:
    """Strictly parse {"classification", "confidence"}; None if invalid."""
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        obj = json.loads(s)
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    label = str(obj.get("classification", "")).strip().upper()
    if label not in LABELS:
        return None
    conf = obj.get("confidence")
    if isinstance(conf, bool) or not isinstance(conf, (int, float)):
        return None
    return label, min(1.0, max(0.0, float(conf)))


def _fallback_messages(email: dict, cfg: Any, store: Any, ctx_block: str) -> list[dict]:
    """Build the fallback system+user messages with phase-dependent context."""
    parts = [
        "You are an email triage assistant. Decide how the user's incoming email should be handled.",
        f"Triage rules:\n{store.get_rules()}",
    ]
    if cfg.enable_sql_history:
        history = store.recent_history(cfg.history_window)
        if history:
            block = "\n\n".join(format_history_pair(h) for h in history)
            parts.append(f"Recent interaction history:\n{block}")
    if ctx_block:
        parts.append(f"Relevant context retrieved from memory:\n{ctx_block}")
    parts.append(_JSON_INSTRUCTION)
    return [
        {"role": "system", "content": "\n\n".join(parts)},
        {"role": "user", "content": f"Incoming email:\n\n{format_email_doc(email)}"},
    ]


def classify(
    email: dict,
    cfg: Any,
    store: Any,
    llm: Any,
    pipeline_context_fn: Callable[[], str] | None = None,
) -> TriageResult:
    """Classify one email per Eq. (1); LLM fallback below tau.

    pipeline_context_fn: zero-arg callable returning the proactively retrieved
    context block for the p3 fallback; when None in p3, the pipeline's stage-2
    helper (retrieved_context_block) is used directly.
    """
    doc = format_email_doc(email)
    emb = llm.embed([doc])
    vec = emb.vectors[0]
    latencies = {"triage_embed": float(emb.latency_ms), "triage_query": 0.0, "triage_llm": 0.0}
    tokens = {"prompt": 0, "completion": 0, "embed": int(emb.tokens)}

    t0 = time.perf_counter()
    sims: dict[str, float] = {}
    for label in LABELS:
        rows = store.query_fewshot(label, vec, cfg.triage_topn)
        sims[label] = max((float(s) for s, _d in rows), default=0.0)
    latencies["triage_query"] = (time.perf_counter() - t0) * 1000.0

    best = max(LABELS, key=lambda l: sims[l])
    if sims[best] >= cfg.triage_tau:
        return TriageResult(best, sims[best], "semantic", latencies, tokens, [], sims)

    # --- LLM fallback ---
    t1 = time.perf_counter()
    ctx_block = ""
    if cfg.phase == "p3":
        ctx_block = pipeline_context_fn() if pipeline_context_fn is not None else \
            retrieved_context_block(email, cfg, store, llm)
    messages = _fallback_messages(email, cfg, store, ctx_block)
    tool_calls: list[dict] = []

    def _acc(res: Any) -> None:
        tokens["prompt"] += res.prompt_tokens or 0
        tokens["completion"] += res.completion_tokens or 0

    if cfg.phase == "p2":
        schemas = [s for s in FUNCTION_SCHEMAS if s["name"] in _TRIAGE_TOOL_NAMES]
        n_tools = 0
        content: str | None = None
        while True:
            mode = "auto" if n_tools < _MAX_TRIAGE_TOOLS else "none"
            res = llm.chat(messages, functions=schemas, function_call=mode)
            _acc(res)
            fc = parse_function_call(res.function_call)
            if fc is None or mode == "none":
                content = res.content
                break
            name, args = fc
            result = execute_tool(name, args, store, llm)
            tool_calls.append({"name": name, "arguments": args})
            messages.append({"role": "assistant", "content": None,
                             "function_call": {"name": name, "arguments": json.dumps(args)}})
            messages.append({"role": "function", "name": name, "content": result})
            n_tools += 1
    else:
        res = llm.chat(messages)
        _acc(res)
        content = res.content

    parsed = _parse_triage_json(content)
    if parsed is None:  # one strict retry
        retry_messages = messages + [
            {"role": "assistant", "content": content or ""},
            {"role": "user", "content": _RETRY_INSTRUCTION},
        ]
        res = llm.chat(retry_messages)
        _acc(res)
        parsed = _parse_triage_json(res.content)

    label, confidence = parsed if parsed is not None else ("RESPOND", 0.5)  # fail-safe
    latencies["triage_llm"] = (time.perf_counter() - t1) * 1000.0
    return TriageResult(label, confidence, f"llm_{cfg.phase}",
                        latencies, tokens, tool_calls, sims)


# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------

def _selftest() -> None:
    """Exercise seeding, tau routing, and all fallback paths. No OpenAI."""
    import math

    from experiments.agent.pipeline import FakeChatResult, FakeFunctionCall, FakeLLM, FakeStore

    # --- seed_fewshot ---
    store = FakeStore()
    llm = FakeLLM()
    n = seed_fewshot(store, llm)
    assert n == 17
    counts = {l: len(store.fewshot[l]) for l in LABELS}
    assert counts == {"IGNORE": 5, "NOTIFY": 5, "RESPOND": 7}, counts
    assert len(llm.embed_calls) == 1 and len(llm.embed_calls[0]) == 17
    assert llm.embed_calls[0][0].startswith("From: Newsletter <newsletter@company.com>")
    print("[triage-selftest] seed_fewshot: 17 examples (5/5/7) ok")

    # --- controlled store: one one-hot prototype per class (4-d) ---
    def proto_store() -> FakeStore:
        s = FakeStore()
        s.add_fewshot("IGNORE", "ig", [1.0, 0.0, 0.0, 0.0])
        s.add_fewshot("NOTIFY", "no", [0.0, 1.0, 0.0, 0.0])
        s.add_fewshot("RESPOND", "re", [0.0, 0.0, 1.0, 0.0])
        return s

    def email_with_sim(tag: str) -> dict:
        return {"email_id": f"e-{tag}", "sender": "X <x@y.com>", "subject": tag, "body": "b"}

    def vec_with_respond_sim(s: float) -> list[float]:
        return [0.0, 0.0, s, math.sqrt(1.0 - s * s)]  # 4th dim orthogonal to classes

    # semantic just above tau
    e_hi = email_with_sim("hi")
    llm_hi = FakeLLM(embed_table={format_email_doc(e_hi): vec_with_respond_sim(0.371)})
    r = classify(e_hi, __fake_cfg(phase="p1"), proto_store(), llm_hi)
    assert r.method == "semantic" and r.label == "RESPOND", r
    assert abs(r.confidence - 0.371) < 1e-9 and abs(r.class_sims["RESPOND"] - 0.371) < 1e-9
    assert r.tool_calls == [] and r.latencies["triage_llm"] == 0.0

    # fallback just below tau (p1)
    e_lo = email_with_sim("lo")
    llm_lo = FakeLLM(
        embed_table={format_email_doc(e_lo): vec_with_respond_sim(0.369)},
        chat_queue=[FakeChatResult(content='{"classification": "notify", "confidence": 0.9}')],
    )
    r = classify(e_lo, __fake_cfg(phase="p1"), proto_store(), llm_lo)
    assert r.method == "llm_p1" and r.label == "NOTIFY" and r.confidence == 0.9, r
    assert len(llm_lo.chat_calls) == 1 and llm_lo.chat_calls[0]["functions"] is None
    print("[triage-selftest] tau boundary routing (0.371 semantic / 0.369 fallback) ok")

    # --- p2 fallback: at most 2 tools, then forced JSON ---
    e_p2 = email_with_sim("p2")
    store2 = proto_store()
    store2.save_interaction({"email_id": "h-1", "sender": "A <a@b.c>",
                             "subject": "Old", "body": "old"}, "RESPOND", 0.9, "ok")
    store2.add_memory("Old deploy thread", [0.0, 0.0, 1.0, 0.0], {"email_id": "m-1"})
    llm_p2 = FakeLLM(
        embed_table={format_email_doc(e_p2): vec_with_respond_sim(0.1)},
        chat_queue=[
            FakeChatResult(function_call=FakeFunctionCall("search_similar_emails",
                                                          '{"query": "deploy"}')),
            FakeChatResult(function_call={"name": "get_user_info", "arguments": "{}"}),
            FakeChatResult(content='{"classification": "RESPOND", "confidence": 0.8}'),
        ],
    )
    r = classify(e_p2, __fake_cfg(phase="p2"), store2, llm_p2)
    assert r.method == "llm_p2" and r.label == "RESPOND" and r.confidence == 0.8, r
    assert [c["name"] for c in r.tool_calls] == ["search_similar_emails", "get_user_info"]
    modes = [c["function_call"] for c in llm_p2.chat_calls]
    assert modes == ["auto", "auto", "none"], modes
    fnames = [s["name"] for s in llm_p2.chat_calls[0]["functions"]]
    assert fnames == ["search_similar_emails", "get_user_info"], fnames
    # history block present in fallback system prompt
    assert "Recent interaction history:" in llm_p2.chat_calls[0]["messages"][0]["content"]
    print("[triage-selftest] p2 fallback with 2-tool cap ok")

    # --- invalid JSON: retry succeeds / retry fails -> RESPOND 0.5 ---
    e_bad = email_with_sim("bad")
    tbl = {format_email_doc(e_bad): vec_with_respond_sim(0.1)}
    llm_retry = FakeLLM(embed_table=tbl, chat_queue=[
        FakeChatResult(content="bogus"),
        FakeChatResult(content='```json\n{"classification": "ignore", "confidence": 0.7}\n```'),
    ])
    r = classify(e_bad, __fake_cfg(phase="p1"), proto_store(), llm_retry)
    assert r.label == "IGNORE" and r.confidence == 0.7 and len(llm_retry.chat_calls) == 2, r
    llm_fail = FakeLLM(embed_table=tbl, chat_queue=[
        FakeChatResult(content="bogus"),
        FakeChatResult(content='{"classification": "MAYBE", "confidence": 0.7}'),
    ])
    r = classify(e_bad, __fake_cfg(phase="p1"), proto_store(), llm_fail)
    assert r.label == "RESPOND" and r.confidence == 0.5 and len(llm_fail.chat_calls) == 2, r
    print("[triage-selftest] strict JSON parsing + one retry + fail-safe ok")

    # --- p3 fallback: injected context fn, and stage-2 helper when None ---
    e_p3 = email_with_sim("p3")
    tbl3 = {format_email_doc(e_p3): vec_with_respond_sim(0.1)}
    llm_p3 = FakeLLM(embed_table=tbl3, chat_queue=[
        FakeChatResult(content='{"classification": "RESPOND", "confidence": 0.95}')])
    r = classify(e_p3, __fake_cfg(phase="p3"), proto_store(), llm_p3,
                 pipeline_context_fn=lambda: "CTX-SENTINEL-123")
    assert r.method == "llm_p3", r
    assert "CTX-SENTINEL-123" in llm_p3.chat_calls[0]["messages"][0]["content"]

    from experiments.agent.pipeline import retrieval_query_text
    store3 = proto_store()
    store3.add_memory("PROACTIVE-DOC", [0.0, 1.0, 0.0, 0.0], {"email_id": "m-9"})
    llm_p3b = FakeLLM(
        embed_table={format_email_doc(e_p3): vec_with_respond_sim(0.1),
                     retrieval_query_text(e_p3): [0.0, 1.0, 0.0, 0.0]},
        chat_queue=[FakeChatResult(content='{"classification": "RESPOND", "confidence": 0.9}')],
    )
    r = classify(e_p3, __fake_cfg(phase="p3"), store3, llm_p3b)
    assert "PROACTIVE-DOC" in llm_p3b.chat_calls[0]["messages"][0]["content"]
    assert llm_p3b.chat_calls[0]["functions"] is None  # no tool decision needed in p3
    print("[triage-selftest] p3 proactive-context fallback ok")

    print("[triage-selftest] ALL PASSED")


def __fake_cfg(**overrides: Any) -> Any:
    """Selftest-only config shim (delegates to pipeline.fake_cfg)."""
    from experiments.agent.pipeline import fake_cfg
    return fake_cfg(**overrides)


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("usage: python3 -m experiments.agent.triage --selftest")
