"""Response generation for RESPOND-classified emails (paper Sec. 3.4).

p1: single chat call, no functions. p2/p3: OpenAI function-calling loop with
signature-based loop detection, max 5 iterations; tool results accumulate into
the prior-tool-results section and the system prompt is rebuilt each iteration
via the injected pipeline_rebuild_fn(tool_results_text).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from experiments.agent.pipeline import format_email_doc

MAX_ITERATIONS = 5

FUNCTION_SCHEMAS: list[dict] = [
    {
        "name": "search_similar_emails",
        "description": "Search semantic memory for past emails similar to a query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Free-text search query."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_user_info",
        "description": "Return the stored profile of the user the assistant acts for.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "check_calendar_availability",
        "description": "Check the user's calendar availability for a given date.",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Date or day to check, e.g. 'Tuesday' or '2026-02-03'."},
            },
            "required": ["date"],
        },
    },
    {
        "name": "schedule_meeting",
        "description": "Schedule a calendar meeting with the given attendees.",
        "parameters": {
            "type": "object",
            "properties": {
                "attendees": {"type": "array", "items": {"type": "string"},
                              "description": "Attendee names or addresses."},
                "subject": {"type": "string", "description": "Meeting subject."},
                "duration_minutes": {"type": "integer", "description": "Meeting length in minutes."},
                "preferred_day": {"type": "string", "description": "Preferred day/date for the meeting."},
            },
            "required": ["attendees", "subject", "duration_minutes", "preferred_day"],
        },
    },
    {
        "name": "write_email_tool",
        "description": "Format an outgoing email draft to a recipient.",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {"type": "string", "description": "Recipient address or name."},
                "subject": {"type": "string", "description": "Email subject."},
                "body": {"type": "string", "description": "Email body text."},
            },
            "required": ["recipient", "subject", "body"],
        },
    },
]


@dataclass
class RespondResult:
    """Outcome of response generation for one email."""

    text: str
    tool_calls: list[dict]
    latencies: dict[str, float]  # respond_llm, respond_tools (ms)
    tokens: dict[str, int]       # prompt, completion


def parse_function_call(fc: Any) -> tuple[str, dict] | None:
    """Normalize a ChatResult.function_call into (name, arguments dict) or None.

    Accepts None, a (name, arguments) tuple, a dict, or an object with
    .name/.arguments; arguments given as a JSON string are parsed (invalid -> {}).
    """
    if fc is None:
        return None
    if isinstance(fc, (tuple, list)) and len(fc) == 2:
        name, args = fc
    elif isinstance(fc, dict):
        name, args = fc.get("name"), fc.get("arguments")
    else:
        name, args = getattr(fc, "name", None), getattr(fc, "arguments", None)
    if not name:
        return None
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return str(name), args


def execute_tool(name: str, arguments: dict | None, store: Any, llm: Any) -> str:
    """Execute one tool against the store; returns the tool result text."""
    args = arguments or {}
    if name == "search_similar_emails":
        query = str(args.get("query", ""))
        vec = llm.embed([query]).vectors[0]
        rows = store.query_memory(vec, 3)
        if not rows:
            return "No similar emails found."
        return "\n\n".join(f"[similarity={s:.2f}] {d}" for s, d, _m in rows)
    if name == "get_user_info":
        return store.get_user_profile()
    if name == "check_calendar_availability":
        date = str(args.get("date", "the requested day"))
        return f"Available slots on {date}: 9:00 AM, 2:00 PM, 4:00 PM"
    if name == "schedule_meeting":
        attendees = args.get("attendees") or []
        if isinstance(attendees, str):
            attendees = [attendees]
        subject = str(args.get("subject", "(no subject)"))
        duration = args.get("duration_minutes", 30)
        day = str(args.get("preferred_day", "TBD"))
        who = ", ".join(str(a) for a in attendees) or "no attendees"
        return f"Meeting '{subject}' scheduled on {day} for {duration} minutes with {who}."
    if name == "write_email_tool":
        return (
            "Email draft prepared:\n"
            f"To: {args.get('recipient', '')}\n"
            f"Subject: {args.get('subject', '')}\n\n"
            f"{args.get('body', '')}"
        )
    return f"Unknown tool: {name}"


def _rebuild(pipeline_rebuild_fn: Callable[[str], Any], tool_results_text: str) -> str:
    """Call the injected rebuild fn; accepts a bare prompt or (prompt, stats)."""
    out = pipeline_rebuild_fn(tool_results_text)
    return out[0] if isinstance(out, tuple) else out


def generate(
    email: dict,
    system_prompt: str,
    cfg: Any,
    store: Any,
    llm: Any,
    pipeline_rebuild_fn: Callable[[str], Any],
) -> RespondResult:
    """Generate the reply for a RESPOND-classified email.

    p1: one chat call without functions. p2/p3: function-calling loop
    (max MAX_ITERATIONS): each executed tool result is appended to
    tool_results_text, the system prompt is rebuilt via
    pipeline_rebuild_fn(tool_results_text), and the model is reinvoked.
    A repeated (name, sorted-args) signature forces a direct answer.
    """
    user_msg = {"role": "user", "content": f"Incoming email:\n\n{format_email_doc(email)}"}
    tokens = {"prompt": 0, "completion": 0}
    llm_ms = 0.0
    tool_ms = 0.0

    def _chat(prompt: str, **kw: Any) -> Any:
        nonlocal llm_ms
        res = llm.chat([{"role": "system", "content": prompt}, user_msg], **kw)
        tokens["prompt"] += res.prompt_tokens or 0
        tokens["completion"] += res.completion_tokens or 0
        llm_ms += res.latency_ms or 0.0
        return res

    if cfg.phase not in ("p2", "p3"):  # p1: no functions param
        res = _chat(system_prompt)
        return RespondResult(res.content or "", [],
                             {"respond_llm": llm_ms, "respond_tools": 0.0}, tokens)

    prompt = system_prompt
    tool_results_text = ""
    seen_sigs: set[tuple[str, str]] = set()
    tool_calls: list[dict] = []
    final: str | None = None

    for _ in range(MAX_ITERATIONS):
        res = _chat(prompt, functions=FUNCTION_SCHEMAS, function_call="auto")
        fc = parse_function_call(res.function_call)
        if fc is None:
            final = res.content or ""
            break
        name, args = fc
        sig = (name, json.dumps(args, sort_keys=True, default=str))
        if sig in seen_sigs:  # loop detected -> force direct answer
            res = _chat(prompt, functions=FUNCTION_SCHEMAS, function_call="none")
            final = res.content or ""
            break
        seen_sigs.add(sig)
        t0 = time.perf_counter()
        result = execute_tool(name, args, store, llm)
        tool_ms += (time.perf_counter() - t0) * 1000.0
        tool_calls.append({"name": name, "arguments": args, "result": result[:400]})
        tool_results_text += (
            f"Tool call: {name}({json.dumps(args, default=str)})\nResult: {result}\n\n"
        )
        prompt = _rebuild(pipeline_rebuild_fn, tool_results_text)

    if final is None:  # iteration budget exhausted -> force direct answer
        res = _chat(prompt, functions=FUNCTION_SCHEMAS, function_call="none")
        final = res.content or ""

    return RespondResult(final, tool_calls,
                         {"respond_llm": llm_ms, "respond_tools": tool_ms}, tokens)


# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------

def _selftest() -> None:
    """Exercise p1 path, tool loop, loop detection, and iteration cap. No OpenAI."""
    from experiments.agent.pipeline import (
        FakeChatResult, FakeFunctionCall, FakeLLM, FakeStore, fake_cfg,
    )

    email = {"email_id": "e-1", "sender": "Bob <bob@corp.com>",
             "subject": "Meeting?", "body": "Can we meet this week?"}
    store = FakeStore()
    store.add_memory("Past email about meetings", [1.0, 0.0, 0.0, 0.0], {"email_id": "m-1"})

    # --- tool executors ---
    out = execute_tool("check_calendar_availability", {"date": "Monday"}, store, FakeLLM())
    assert out == "Available slots on Monday: 9:00 AM, 2:00 PM, 4:00 PM"
    out = execute_tool("schedule_meeting", {"attendees": ["Bob"], "subject": "Sync",
                                            "duration_minutes": 30, "preferred_day": "Monday"},
                       store, FakeLLM())
    assert "'Sync' scheduled on Monday for 30 minutes with Bob" in out
    out = execute_tool("write_email_tool", {"recipient": "bob@corp.com", "subject": "Re: Meeting?",
                                            "body": "Yes."}, store, FakeLLM())
    assert out.startswith("Email draft prepared:\nTo: bob@corp.com\nSubject: Re: Meeting?")
    assert execute_tool("get_user_info", {}, store, FakeLLM()) == store.get_user_profile()
    out = execute_tool("search_similar_emails", {"query": "meeting"},
                       FakeStore(), FakeLLM())
    assert out == "No similar emails found."
    llm_s = FakeLLM(embed_table={"meeting": [1.0, 0.0, 0.0, 0.0]})
    out = execute_tool("search_similar_emails", {"query": "meeting"}, store, llm_s)
    assert "Past email about meetings" in out
    print("[respond-selftest] tool executors ok")

    # --- p1: no functions param ---
    llm1 = FakeLLM(chat_queue=[FakeChatResult(content="Plain reply.")])
    res1 = generate(email, "SYS-P1", fake_cfg(phase="p1"), store, llm1,
                    lambda t: (_ for _ in ()).throw(AssertionError("rebuild called in p1")))
    assert res1.text == "Plain reply." and res1.tool_calls == []
    assert llm1.chat_calls[0]["functions"] is None
    assert res1.tokens["prompt"] > 0 and res1.latencies["respond_llm"] > 0
    print("[respond-selftest] p1 no-functions path ok")

    # --- p2: tool call, rebuild, then answer ---
    rebuilds: list[str] = []

    def rebuild_fn(tool_results_text: str) -> tuple[str, None]:
        rebuilds.append(tool_results_text)
        return (f"SYS-REBUILT-{len(rebuilds)}", None)

    llm2 = FakeLLM(chat_queue=[
        FakeChatResult(function_call=FakeFunctionCall(
            "check_calendar_availability", '{"date": "Monday"}')),
        FakeChatResult(content="Final reply with slots."),
    ])
    res2 = generate(email, "SYS-0", fake_cfg(phase="p2"), store, llm2, rebuild_fn)
    assert res2.text == "Final reply with slots."
    assert [c["name"] for c in res2.tool_calls] == ["check_calendar_availability"]
    assert len(rebuilds) == 1 and "9:00 AM" in rebuilds[0]
    assert llm2.chat_calls[1]["messages"][0]["content"] == "SYS-REBUILT-1"
    print("[respond-selftest] tool call + prompt rebuild ok")

    # --- loop detection: identical signature twice -> forced direct answer ---
    llm3 = FakeLLM(chat_queue=[
        FakeChatResult(function_call={"name": "get_user_info", "arguments": "{}"}),
        FakeChatResult(function_call={"name": "get_user_info", "arguments": "{}"}),
        FakeChatResult(content="Forced final."),
    ])
    res3 = generate(email, "SYS-0", fake_cfg(phase="p3"), store, llm3, rebuild_fn)
    assert res3.text == "Forced final."
    assert len(res3.tool_calls) == 1  # second identical call not executed
    assert llm3.chat_calls[-1]["function_call"] == "none"
    print("[respond-selftest] loop detection terminates on repeated signature ok")

    # --- iteration cap: 5 distinct tool calls -> forced answer on call 6 ---
    def chat_fn(i: int, messages: list[dict], kw: dict) -> FakeChatResult:
        if kw["function_call"] == "none":
            return FakeChatResult(content="Capped final.")
        return FakeChatResult(function_call=FakeFunctionCall(
            "check_calendar_availability", json.dumps({"date": f"day-{i}"})))

    llm4 = FakeLLM(chat_fn=chat_fn)
    res4 = generate(email, "SYS-0", fake_cfg(phase="p2"), store, llm4, rebuild_fn)
    assert res4.text == "Capped final." and len(res4.tool_calls) == MAX_ITERATIONS
    assert len(llm4.chat_calls) == MAX_ITERATIONS + 1
    print("[respond-selftest] max-iteration cap ok")

    print("[respond-selftest] ALL PASSED")


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("usage: python3 -m experiments.agent.respond --selftest")
