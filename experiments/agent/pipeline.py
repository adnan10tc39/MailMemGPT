"""Five-stage context-assembly pipeline (paper Sec. 3 / Fig. 6).

Stages: sql_load -> vector_retrieve -> dedup -> budget -> assemble.
`run_pipeline` orchestrates them and returns (system_prompt, PipelineStats).

This module is the dependency leaf of the agent package: it imports nothing
from experiments.agent.* and also hosts the shared selftest fakes (FakeLLM,
FakeStore, fake_cfg) used by triage.py and respond.py selftests.
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

LABELS: tuple[str, str, str] = ("IGNORE", "NOTIFY", "RESPOND")

ROLE_PREAMBLE = "You are an intelligent email assistant acting on behalf of the user described below."

RESPONSE_INSTRUCTIONS = (
    "Handle the incoming email shown in the user message. If it needs a reply, draft it on the "
    "user's behalf: professional, concise, and grounded only in information present in this prompt "
    "(profile, rules, history, retrieved memories, tool results). Use prior commitments and thread "
    "context when relevant; when the reply concerns an ongoing thread, explicitly reference the "
    "specific identifiers, amounts, dates, names, or commitments from the prior context (e.g., "
    "invoice or ticket numbers, agreed deadlines) whenever they are available and relevant. "
    "Never invent facts, meetings, or agreements."
)

TOOL_DESCRIPTIONS_TEXT = (
    "- search_similar_emails(query): retrieve past emails semantically similar to the query.\n"
    "- get_user_info(): return the user's stored profile.\n"
    "- check_calendar_availability(date): list open meeting slots for a date.\n"
    "- schedule_meeting(attendees, subject, duration_minutes, preferred_day): book a meeting.\n"
    "- write_email_tool(recipient, subject, body): format an outgoing email draft."
)

_SECTION_ORDER: tuple[str, ...] = (
    "role_profile",
    "instructions_rules",
    "tool_descriptions",
    "prior_tool_results",
    "summary",
    "history",
    "retrieved",
    "response_instructions",
)

_HEADERS: dict[str, str] = {
    "role_profile": "# Role and User Profile",
    "instructions_rules": "# Agent Instructions and Triage Rules",
    "tool_descriptions": "# Available Tools",
    "prior_tool_results": "# Prior Tool Results",
    "summary": "# Summary of Older Interactions",
    "history": "# Recent Interaction History",
    "retrieved": "# Retrieved Semantic Memories",
    "response_instructions": "# Response Instructions",
}


@dataclass
class StageStats:
    """Per-stage measurement record."""

    latency_ms: float
    tokens_in_block: int
    items: int
    removed_items: int
    tokens_saved: int


@dataclass
class SqlContext:
    """Output of stage 1 (SQL hot memory load)."""

    profile: str
    rules: str
    summary: str
    history: list[dict]
    history_pairs: list[str]


@dataclass
class Sections:
    """Mutable prompt sections in the fixed assembly order (paper Fig. 6)."""

    role_profile: str
    instructions_rules: str
    tool_descriptions: str
    prior_tool_results: str
    summary: str
    history_pairs: list[str]
    retrieved: str
    response_instructions: str


@dataclass
class PipelineStats:
    """Aggregated pipeline measurements for one email."""

    latency_ms: dict[str, float]      # sql_load, retrieve, dedup, budget, assemble
    section_tokens: dict[str, int]    # tokens per assembled section (incl. header)
    sims: list[float]                 # raw retrieval similarities (before delta filter)
    n_raw: int                        # items returned by query_memory
    n_kept: int                       # items kept after delta filter + dedup
    n_deduped: int                    # items removed by dedup
    tokens_saved: int                 # tokens saved by dedup
    budget_actions: list[dict]        # actions taken by the budget stage
    final_prompt_tokens: int


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_email_doc(email: dict) -> str:
    """Canonical 'From/Subject/Body' full-text representation of an email."""
    return (
        f"From: {email.get('sender', '')}\n"
        f"Subject: {email.get('subject', '')}\n"
        f"Body: {email.get('body', '')}"
    )


def retrieval_query_text(email: dict) -> str:
    """Stage-2 embedding text: subject + first 200 chars of body (per paper)."""
    return f"{email.get('subject', '')}\n{str(email.get('body', ''))[:200]}"


def format_history_pair(h: dict) -> str:
    """Format one SQL history interaction as From/Subject/Body/Response."""
    resp = h.get("response_text") or h.get("response") or "(no response)"
    return (
        f"From: {h.get('sender', '')}\n"
        f"Subject: {h.get('subject', '')}\n"
        f"Body: {h.get('body', '')}\n"
        f"Response: {resp}"
    )


def format_retrieved_item(sim: float, doc: str, meta: dict | None) -> str:
    """Format one retrieved memory with a metadata line, then the document."""
    meta = meta or {}
    fields = [f"similarity={sim:.2f}"]
    for key in ("email_id", "thread_id", "sender", "subject", "day"):
        if meta.get(key) is not None:
            fields.append(f"{key}={meta[key]}")
    return "[" + " | ".join(fields) + "]\n" + doc


def normalize_address(sender: str) -> str:
    """Extract and lowercase the address part of 'Name <addr@dom>'."""
    m = re.search(r"<([^>]+)>", sender)
    return (m.group(1) if m else sender).strip().lower()


def strip_reply_prefixes(subject: str) -> str:
    """Strip repeated leading Re:/Fw:/Fwd: prefixes; lowercase the remainder."""
    s = subject.strip()
    pat = re.compile(r"^(re|fw|fwd)\s*:\s*", re.IGNORECASE)
    while True:
        m = pat.match(s)
        if not m:
            break
        s = s[m.end():]
    return s.strip().lower()


def render_sections(s: Sections) -> tuple[str, dict[str, str]]:
    """Render sections in the fixed order; returns (prompt, name -> full part text)."""
    content = {
        "role_profile": f"{ROLE_PREAMBLE}\n{s.role_profile}" if s.role_profile else "",
        "instructions_rules": s.instructions_rules,
        "tool_descriptions": s.tool_descriptions,
        "prior_tool_results": s.prior_tool_results,
        "summary": s.summary,
        "history": "\n\n".join(s.history_pairs),
        "retrieved": s.retrieved,
        "response_instructions": s.response_instructions,
    }
    parts: dict[str, str] = {}
    for name in _SECTION_ORDER:
        if content[name].strip():
            parts[name] = f"{_HEADERS[name]}\n{content[name]}"
    return "\n\n".join(parts.values()), parts


def _trim_to_tokens(text: str, max_tokens: int, llm: Any) -> str:
    """Trim text (by characters, binary search) to at most max_tokens tokens."""
    if max_tokens <= 0:
        return ""
    if llm.count_tokens(text) <= max_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if llm.count_tokens(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]


# ---------------------------------------------------------------------------
# Stage 1: SQL hot-memory load
# ---------------------------------------------------------------------------

def sql_load(cfg: Any, store: Any, llm: Any) -> tuple[SqlContext, StageStats]:
    """Load profile, rules, latest rolling summary, and recent history pairs."""
    t0 = time.perf_counter()
    profile = store.get_user_profile()
    rules = store.get_rules()
    use_hist = bool(cfg.enable_sql_history) and not cfg.naive_rag
    summary = (store.latest_summary() or "") if use_hist else ""
    history = list(store.recent_history(cfg.history_window)) if use_hist else []
    pairs = [format_history_pair(h) for h in history]
    block_tokens = llm.count_tokens("\n\n".join([profile, rules, summary, *pairs]))
    stats = StageStats((time.perf_counter() - t0) * 1000.0, block_tokens, len(history), 0, 0)
    return SqlContext(profile, rules, summary, history, pairs), stats


# ---------------------------------------------------------------------------
# Stage 2: vector retrieval
# ---------------------------------------------------------------------------

def vector_retrieve(
    email: dict, cfg: Any, store: Any, llm: Any, apply_delta: bool = True
) -> tuple[list[tuple[float, str, dict]], list[float], StageStats]:
    """Embed the email and query warm memory; returns (kept items, raw sims, stats)."""
    t0 = time.perf_counter()
    emb = llm.embed([retrieval_query_text(email)])
    raw = store.query_memory(emb.vectors[0], cfg.retrieval_k)
    raw_sims = [float(s) for s, _d, _m in raw]
    kept = [
        (float(s), d, m) for s, d, m in raw
        if (not apply_delta) or float(s) >= cfg.retrieval_delta
    ]
    stats = StageStats(
        (time.perf_counter() - t0) * 1000.0, 0, len(kept), len(raw) - len(kept), 0
    )
    return kept, raw_sims, stats


def retrieved_context_block(email: dict, cfg: Any, store: Any, llm: Any) -> str:
    """Stage-2 helper for p3 triage fallback: formatted top-k>=delta context, '' if none."""
    kept, _sims, _stats = vector_retrieve(email, cfg, store, llm)
    return "\n\n".join(format_retrieved_item(s, d, m) for s, d, m in kept)


# ---------------------------------------------------------------------------
# Stage 3: dedup
# ---------------------------------------------------------------------------

def dedup(
    retrieved: list[tuple[float, str, dict]], history: list[dict], llm: Any
) -> tuple[list[tuple[float, str, dict]], StageStats]:
    """Drop retrieved items already present in the recent history window.

    Primary rule: meta email_id matches a history email_id. Fallback (meta has
    no email_id): normalized sender address + Re/Fwd-stripped subject match.
    """
    t0 = time.perf_counter()
    ids = {h.get("email_id") for h in history if h.get("email_id")}
    fallback_keys = set()
    for h in history:
        addr = normalize_address(str(h.get("sender", "")))
        subj = strip_reply_prefixes(str(h.get("subject", "")))
        if addr and subj:
            fallback_keys.add((addr, subj))
    kept: list[tuple[float, str, dict]] = []
    removed: list[tuple[float, str, dict]] = []
    for sim, doc, meta in retrieved:
        meta = meta or {}
        mid = meta.get("email_id")
        if mid and mid in ids:
            removed.append((sim, doc, meta))
            continue
        if not mid:
            addr = normalize_address(str(meta.get("sender", "")))
            subj = strip_reply_prefixes(str(meta.get("subject", "")))
            if addr and subj and (addr, subj) in fallback_keys:
                removed.append((sim, doc, meta))
                continue
        kept.append((sim, doc, meta))
    tokens_saved = sum(
        llm.count_tokens(format_retrieved_item(s, d, m)) for s, d, m in removed
    )
    stats = StageStats(
        (time.perf_counter() - t0) * 1000.0, 0, len(kept), len(removed), tokens_saved
    )
    return kept, stats


# ---------------------------------------------------------------------------
# Stage 4: budget enforcement
# ---------------------------------------------------------------------------

def budget(sections: Sections, cfg: Any, llm: Any) -> tuple[Sections, list[dict], StageStats]:
    """Enforce the token budget B on the assembled prompt.

    If over budget: (1) summarize the retrieved block with cfg.summary_model;
    (2) if still over, truncate low-priority sections in order
    retrieved -> summary -> oldest history pairs. Profile, rules, and the
    response (current-email) instructions are never dropped.
    """
    t0 = time.perf_counter()
    actions: list[dict] = []
    b = int(cfg.prompt_budget)

    def total_tokens() -> int:
        prompt, _ = render_sections(sections)
        return llm.count_tokens(prompt)

    total0 = total_tokens()
    total = total0

    if total > b and sections.retrieved.strip():
        rtoks = llm.count_tokens(sections.retrieved)
        target = max(64, rtoks - (total - b) - 32)
        res = llm.chat(
            [
                {"role": "system", "content": "You compress retrieved email memories into a dense factual summary."},
                {"role": "user", "content": (
                    f"Summarize the following retrieved email memories in at most {target} tokens. "
                    f"Preserve senders, dates, commitments, and key facts.\n\n{sections.retrieved}"
                )},
            ],
            model=cfg.summary_model,
            max_tokens=max(64, min(target, cfg.max_output_tokens)),
        )
        summ = (res.content or "").strip()
        if summ:
            sections.retrieved = f"(summarized) {summ}"
        actions.append({
            "action": "summarize_retrieved",
            "tokens_before": rtoks,
            "tokens_after": llm.count_tokens(sections.retrieved),
        })
        total = total_tokens()

    for name in ("retrieved", "summary"):
        if total <= b:
            break
        text = getattr(sections, name)
        if not text.strip():
            continue
        need = total - b
        stoks = llm.count_tokens(text)
        if stoks <= need:
            setattr(sections, name, "")
            actions.append({"action": f"drop_{name}", "tokens_saved": stoks})
        else:
            setattr(sections, name, _trim_to_tokens(text, stoks - need, llm))
            actions.append({"action": f"truncate_{name}", "tokens_saved": need})
        total = total_tokens()

    pairs_dropped = 0
    while total > b and sections.history_pairs:
        sections.history_pairs.pop(0)  # oldest first
        pairs_dropped += 1
        total = total_tokens()
    if pairs_dropped:
        actions.append({"action": "drop_history_pairs", "count": pairs_dropped})
    if total > b:
        actions.append({"action": "over_budget_unresolved", "tokens": total})

    stats = StageStats(
        (time.perf_counter() - t0) * 1000.0, total, len(actions),
        pairs_dropped, max(0, total0 - total),
    )
    return sections, actions, stats


# ---------------------------------------------------------------------------
# Stage 5: assembly
# ---------------------------------------------------------------------------

def assemble(sections: Sections, llm: Any) -> tuple[str, dict[str, int], StageStats]:
    """Assemble the final system prompt in the fixed section order."""
    t0 = time.perf_counter()
    prompt, parts = render_sections(sections)
    section_tokens = {
        name: (llm.count_tokens(parts[name]) if name in parts else 0)
        for name in _SECTION_ORDER
    }
    final = llm.count_tokens(prompt)
    stats = StageStats((time.perf_counter() - t0) * 1000.0, final, len(parts), 0, 0)
    return prompt, section_tokens, stats


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_pipeline(
    email: dict, cfg: Any, store: Any, llm: Any, tool_results_text: str = ""
) -> tuple[str, PipelineStats]:
    """Run the five-stage pipeline; returns (system_prompt, PipelineStats).

    In naive_rag mode: no dedup, no budget, no structure — the prompt is
    profile + rules + raw concatenation of top-k retrieved docs + instructions.
    """
    lat = {"sql_load": 0.0, "retrieve": 0.0, "dedup": 0.0, "budget": 0.0, "assemble": 0.0}
    ctx, st1 = sql_load(cfg, store, llm)
    lat["sql_load"] = st1.latency_ms

    retrieved: list[tuple[float, str, dict]] = []
    raw_sims: list[float] = []
    do_retrieve = cfg.naive_rag or (cfg.phase == "p3" and cfg.enable_vector_retrieval)
    if do_retrieve:
        retrieved, raw_sims, st2 = vector_retrieve(
            email, cfg, store, llm, apply_delta=not cfg.naive_rag
        )
        lat["retrieve"] = st2.latency_ms

    if cfg.naive_rag:
        t0 = time.perf_counter()
        docs_block = "\n\n".join(doc for _s, doc, _m in retrieved)
        parts = [
            p for p in (
                ctx.profile, ctx.rules, docs_block,
                tool_results_text.strip(), RESPONSE_INSTRUCTIONS,
            ) if p
        ]
        prompt = "\n\n".join(parts)
        final = llm.count_tokens(prompt)
        lat["assemble"] = (time.perf_counter() - t0) * 1000.0
        stats = PipelineStats(
            latency_ms=lat,
            section_tokens={
                "role_profile": llm.count_tokens(ctx.profile),
                "instructions_rules": llm.count_tokens(ctx.rules),
                "tool_descriptions": 0,
                "prior_tool_results": llm.count_tokens(tool_results_text.strip()),
                "summary": 0,
                "history": 0,
                "retrieved": llm.count_tokens(docs_block),
                "response_instructions": llm.count_tokens(RESPONSE_INSTRUCTIONS),
            },
            sims=raw_sims, n_raw=len(raw_sims), n_kept=len(retrieved),
            n_deduped=0, tokens_saved=0, budget_actions=[], final_prompt_tokens=final,
        )
        return prompt, stats

    n_deduped = 0
    tokens_saved = 0
    if cfg.enable_dedup and retrieved:
        retrieved, st3 = dedup(retrieved, ctx.history, llm)
        lat["dedup"] = st3.latency_ms
        n_deduped = st3.removed_items
        tokens_saved = st3.tokens_saved

    sections = Sections(
        role_profile=ctx.profile,
        instructions_rules=ctx.rules,
        tool_descriptions=TOOL_DESCRIPTIONS_TEXT if cfg.phase in ("p2", "p3") else "",
        prior_tool_results=tool_results_text or "",
        summary=ctx.summary,
        history_pairs=list(ctx.history_pairs),
        retrieved="\n\n".join(format_retrieved_item(s, d, m) for s, d, m in retrieved),
        response_instructions=RESPONSE_INSTRUCTIONS,
    )

    budget_actions: list[dict] = []
    if cfg.enable_budget:
        sections, budget_actions, st4 = budget(sections, cfg, llm)
        lat["budget"] = st4.latency_ms

    prompt, section_tokens, st5 = assemble(sections, llm)
    lat["assemble"] = st5.latency_ms

    stats = PipelineStats(
        latency_ms=lat, section_tokens=section_tokens, sims=raw_sims,
        n_raw=len(raw_sims), n_kept=len(retrieved), n_deduped=n_deduped,
        tokens_saved=tokens_saved, budget_actions=budget_actions,
        final_prompt_tokens=st5.tokens_in_block,
    )
    return prompt, stats


# ---------------------------------------------------------------------------
# Selftest support (shared fakes — NO OpenAI, NO Postgres). Used by the
# --selftest paths of pipeline.py, triage.py, and respond.py only.
# ---------------------------------------------------------------------------

@dataclass
class FakeFunctionCall:
    """Stands in for ChatResult.function_call."""

    name: str
    arguments: Any  # dict or JSON string


@dataclass
class FakeChatResult:
    """Stands in for common.llm.ChatResult."""

    content: str | None = None
    function_call: Any = None
    prompt_tokens: int = 7
    completion_tokens: int = 3
    latency_ms: float = 0.5


@dataclass
class FakeEmbedResult:
    """Stands in for common.llm.EmbedResult."""

    vectors: list[list[float]]
    latency_ms: float
    tokens: int


def _hash_vec(text: str, dim: int) -> list[float]:
    """Deterministic pseudo-embedding from an md5 digest."""
    h = hashlib.md5(text.encode("utf-8")).digest()
    return [(h[i % len(h)] / 255.0) * 2.0 - 1.0 for i in range(dim)]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class FakeLLM:
    """Deterministic LLM stand-in implementing the pinned LLM interface subset.

    chat(): serves results from `chat_queue` (FIFO) or, if given, `chat_fn`
    (called as chat_fn(call_index, messages, kwargs)). Defaults to content 'OK'.
    embed(): exact / substring lookup in `embed_table`, else md5 hash vector.
    count_tokens(): max(1, len(text) // 4) — deterministic, tiktoken-free.
    """

    def __init__(
        self,
        chat_queue: list[FakeChatResult] | None = None,
        chat_fn: Any = None,
        embed_table: dict[str, list[float]] | None = None,
        dim: int = 4,
    ) -> None:
        self.chat_queue = list(chat_queue or [])
        self.chat_fn = chat_fn
        self.embed_table = dict(embed_table or {})
        self.dim = dim
        self.chat_calls: list[dict] = []
        self.embed_calls: list[list[str]] = []
        self.usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "embed_tokens": 0, "cost_usd": 0.0}

    def chat(self, messages: list[dict], functions: Any = None, function_call: Any = None,
             model: str | None = None, temperature: float | None = None,
             max_tokens: int | None = None) -> FakeChatResult:
        kwargs = {"functions": functions, "function_call": function_call,
                  "model": model, "temperature": temperature, "max_tokens": max_tokens}
        self.chat_calls.append({"messages": messages, **kwargs})
        if self.chat_fn is not None:
            res = self.chat_fn(len(self.chat_calls) - 1, messages, kwargs)
        elif self.chat_queue:
            res = self.chat_queue.pop(0)
        else:
            res = FakeChatResult(content="OK")
        self.usage_totals["prompt_tokens"] += res.prompt_tokens
        self.usage_totals["completion_tokens"] += res.completion_tokens
        return res

    def _vec(self, text: str) -> list[float]:
        if text in self.embed_table:
            return self.embed_table[text]
        for key, vec in self.embed_table.items():
            if key and key in text:
                return vec
        return _hash_vec(text, self.dim)

    def embed(self, texts: list[str]) -> FakeEmbedResult:
        self.embed_calls.append(list(texts))
        tokens = sum(self.count_tokens(t) for t in texts)
        self.usage_totals["embed_tokens"] += tokens
        return FakeEmbedResult([self._vec(t) for t in texts], 0.1, tokens)

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


class FakeStore:
    """In-memory stand-in for the pinned Store interface subset (cosine sims)."""

    def __init__(self) -> None:
        self.profile = "John Doe, software engineering manager at TechCorp."
        self.rules = (
            "1. Newsletters, promotions, spam, and auto-replies are IGNORE.\n"
            "2. FYI system notifications and broad announcements are NOTIFY.\n"
            "3. Direct questions, meeting requests, and approvals addressed to the user are RESPOND.\n"
            "4. When uncertain, prefer RESPOND."
        )
        self.summary: str | None = None
        self.history: list[dict] = []
        self.fewshot: dict[str, list[tuple[str, list[float]]]] = {l: [] for l in LABELS}
        self.memory: list[tuple[str, list[float], dict]] = []

    def reset(self) -> None:
        self.summary = None
        self.history = []
        self.fewshot = {l: [] for l in LABELS}
        self.memory = []

    def save_interaction(self, email: dict, classification: str, confidence: float,
                         response_text: str | None) -> None:
        self.history.append({
            "email_id": email.get("email_id"), "sender": email.get("sender", ""),
            "subject": email.get("subject", ""), "body": email.get("body", ""),
            "classification": classification, "confidence": confidence,
            "response_text": response_text,
        })

    def recent_history(self, n_pairs: int) -> list[dict]:
        return self.history[-n_pairs:]

    def get_user_profile(self) -> str:
        return self.profile

    def get_rules(self) -> str:
        return self.rules

    def latest_summary(self) -> str | None:
        return self.summary

    def update_summary(self, text: str) -> None:
        self.summary = text

    def add_fewshot(self, label: str, doc: str, vec: list[float]) -> None:
        self.fewshot[label].append((doc, vec))

    def query_fewshot(self, label: str, vec: list[float], topn: int) -> list[tuple[float, str]]:
        scored = [(_cosine(vec, v), d) for d, v in self.fewshot.get(label, [])]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:topn]

    def add_memory(self, doc: str, vec: list[float], meta: dict) -> None:
        self.memory.append((doc, vec, meta))

    def query_memory(self, vec: list[float], k: int) -> list[tuple[float, str, dict]]:
        scored = [(_cosine(vec, v), d, m) for d, v, m in self.memory]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]


def fake_cfg(**overrides: Any) -> SimpleNamespace:
    """ExpConfig stand-in with the paper-authoritative defaults (selftests only)."""
    base: dict[str, Any] = dict(
        phase="p3", run_id="selftest", dataset_path="",
        triage_tau=0.37, triage_topn=3, retrieval_k=5, retrieval_delta=0.70,
        prompt_budget=8000, history_window=10, chat_model="openai/gpt-oss-120b",
        summary_model="openai/gpt-oss-20b", embed_model="BAAI/bge-base-en-v1.5",
        provider="groq", embed_dim=768, embed_device="cuda",
        temperature=0.0, max_output_tokens=2000, seed=13, generate_responses=True,
        enable_dedup=True, enable_budget=True, enable_sql_history=True,
        enable_vector_retrieval=True, naive_rag=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------

def _make_store_with_context() -> FakeStore:
    """Store with profile/rules/summary, 2 history pairs, 3 memory items."""
    store = FakeStore()
    store.summary = "Earlier: John approved the Q1 platform migration plan."
    store.save_interaction(
        {"email_id": "h-1", "sender": "Alice <alice@corp.com>",
         "subject": "Budget question", "body": "Can you approve the budget?"},
        "RESPOND", 0.9, "Approved.")
    store.save_interaction(
        {"email_id": "h-2", "sender": "Bob <bob@corp.com>",
         "subject": "Re: Deploy", "body": "Deploy is done."},
        "NOTIFY", 0.8, None)
    store.add_memory("MEM-A migration kickoff notes",
                     [0.9, math.sqrt(1 - 0.81), 0.0, 0.0],
                     {"email_id": "m-1", "sender": "Carol <carol@corp.com>",
                      "subject": "Migration kickoff", "day": 3})
    store.add_memory("MEM-B duplicate of history budget mail",
                     [0.75, math.sqrt(1 - 0.5625), 0.0, 0.0],
                     {"email_id": "h-1", "sender": "Alice <alice@corp.com>",
                      "subject": "Budget question", "day": 5})
    store.add_memory("MEM-C low similarity note",
                     [0.5, math.sqrt(0.75), 0.0, 0.0],
                     {"email_id": "m-3", "sender": "Dan <dan@corp.com>",
                      "subject": "Lunch", "day": 6})
    return store


def _selftest() -> None:
    """Exercise dedup, budget, assembly order, and naive_rag without OpenAI."""
    email = {"email_id": "e-100", "sender": "Carol <carol@corp.com>",
             "subject": "Migration status?", "body": "What is the migration status?"}
    qkey = retrieval_query_text(email)

    # --- structured p3 run: delta filter + dedup + section order ---
    store = _make_store_with_context()
    llm = FakeLLM(embed_table={qkey: [1.0, 0.0, 0.0, 0.0]})
    cfg = fake_cfg()
    prompt, stats = run_pipeline(email, cfg, store, llm, tool_results_text="TOOLRES-1")
    assert stats.n_raw == 3 and len(stats.sims) == 3, stats
    assert stats.n_deduped == 1 and stats.tokens_saved > 0, stats
    assert stats.n_kept == 1, stats
    assert "MEM-A" in prompt and "MEM-B" not in prompt and "MEM-C" not in prompt
    idxs = [prompt.index(_HEADERS[n]) for n in _SECTION_ORDER]
    assert idxs == sorted(idxs), "section order violated"
    assert "TOOLRES-1" in prompt
    assert stats.final_prompt_tokens == llm.count_tokens(prompt)
    assert stats.budget_actions == []
    print("[pipeline-selftest] structured run: delta filter, dedup, section order ok")

    # --- fallback dedup (meta without email_id, Re:-prefixed subject) ---
    store2 = FakeStore()
    store2.save_interaction(
        {"email_id": "h-9", "sender": "Alice <alice@corp.com>",
         "subject": "Budget question", "body": "x"}, "RESPOND", 0.9, "ok")
    store2.add_memory("MEM-D fallback duplicate", [0.8, math.sqrt(1 - 0.64), 0.0, 0.0],
                      {"sender": "Alice Q <ALICE@corp.com>", "subject": "Re: Budget question"})
    llm2 = FakeLLM(embed_table={qkey: [1.0, 0.0, 0.0, 0.0]})
    prompt2, stats2 = run_pipeline(email, fake_cfg(), store2, llm2)
    assert stats2.n_deduped == 1 and "MEM-D" not in prompt2, stats2
    print("[pipeline-selftest] fallback dedup (sender+stripped-subject) ok")

    # --- budget: summarization then truncation cascade ---
    store3 = FakeStore()
    store3.summary = "OLD-SUMMARY " * 30
    for i in range(4):
        store3.save_interaction(
            {"email_id": f"h-b{i}", "sender": f"P{i} <p{i}@corp.com>",
             "subject": f"Topic {i}", "body": "word " * 40}, "RESPOND", 0.9, "reply " * 10)
    store3.add_memory("MEM-BIG " + "filler " * 400, [1.0, 0.0, 0.0, 0.0],
                      {"email_id": "big-1", "sender": "Z <z@corp.com>", "subject": "Big"})
    llm3 = FakeLLM(
        embed_table={qkey: [1.0, 0.0, 0.0, 0.0]},
        chat_queue=[FakeChatResult(content="SHORT SUMMARY OF MEMS.")],
    )
    # protected floor: prompt with everything optional absent
    floor_llm = FakeLLM()
    base_sections = Sections(store3.profile, store3.rules, TOOL_DESCRIPTIONS_TEXT,
                             "", "", [], "", RESPONSE_INSTRUCTIONS)
    floor_prompt, _ = render_sections(base_sections)
    b = floor_llm.count_tokens(floor_prompt) + 30
    cfg3 = fake_cfg(prompt_budget=b)
    prompt3, stats3 = run_pipeline(email, cfg3, store3, llm3)
    acts = [a["action"] for a in stats3.budget_actions]
    assert "summarize_retrieved" in acts, acts
    assert llm3.chat_calls[0]["model"] == cfg3.summary_model
    assert "drop_history_pairs" in acts, acts
    assert "over_budget_unresolved" not in acts, acts
    assert stats3.final_prompt_tokens <= b, (stats3.final_prompt_tokens, b)
    print(f"[pipeline-selftest] budget cascade ok (B={b}, actions={acts})")

    # --- naive_rag: no delta filter, no dedup, no budget, no structure ---
    store4 = _make_store_with_context()
    llm4 = FakeLLM(embed_table={qkey: [1.0, 0.0, 0.0, 0.0]})
    prompt4, stats4 = run_pipeline(email, fake_cfg(naive_rag=True, prompt_budget=10),
                                   store4, llm4)
    assert "MEM-A" in prompt4 and "MEM-B" in prompt4 and "MEM-C" in prompt4
    assert _HEADERS["retrieved"] not in prompt4
    assert stats4.budget_actions == [] and stats4.n_deduped == 0
    assert stats4.final_prompt_tokens > 10  # budget deliberately not enforced
    print("[pipeline-selftest] naive_rag mode ok")

    # --- phase/flag gating ---
    store5 = _make_store_with_context()
    llm5 = FakeLLM(embed_table={qkey: [1.0, 0.0, 0.0, 0.0]})
    p1_prompt, p1_stats = run_pipeline(email, fake_cfg(phase="p1"), store5, llm5)
    assert _HEADERS["tool_descriptions"] not in p1_prompt
    assert _HEADERS["retrieved"] not in p1_prompt and p1_stats.n_raw == 0
    nohist_prompt, _ = run_pipeline(email, fake_cfg(enable_sql_history=False), store5, llm5)
    assert _HEADERS["history"] not in nohist_prompt and _HEADERS["summary"] not in nohist_prompt
    print("[pipeline-selftest] phase/flag gating ok")

    print("[pipeline-selftest] ALL PASSED")


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("usage: python3 -m experiments.agent.pipeline --selftest")
