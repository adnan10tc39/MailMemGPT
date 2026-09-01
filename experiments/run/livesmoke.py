"""Minimal LIVE smoke test for the Groq + local-embeddings adaptation.

Exercises, against the real Groq API and the real local embedder:
1. one strict-JSON chat completion on cfg.chat_model (gpt-oss-120b), parsed;
2. one tool-call round-trip (get_weather stub) through the wrapper's LEGACY
   functions/function_call interface (translated to tools at the boundary);
3. three local embeddings: shape == cfg.embed_dim (768) and
   cosine(similar pair) > cosine(dissimilar pairs);
4. prints token usage, cost, per-call latency, and throttle sleep time.

Total API usage is a few hundred tokens — far under the 15K budget.

Run from the repo root:
    python3 -m experiments.run.livesmoke
"""
from __future__ import annotations

import json
import re
import time

from experiments.common import llm as llm_mod
from experiments.common.exp_config import preset
from experiments.common.llm import LLM, _TokenThrottle


class _RecordingThrottle(_TokenThrottle):
    """Throttle that records total time slept in reserve()."""

    def __init__(self) -> None:
        super().__init__()
        self.slept_s = 0.0

    def reserve(self, estimate: float) -> list[float]:
        t0 = time.monotonic()
        event = super().reserve(estimate)
        self.slept_s += time.monotonic() - t0
        return event


def _strip_fences(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s


def main() -> None:
    cfg = preset("p3", run_id="livesmoke", dataset_path="")
    llm = LLM(cfg)
    throttle = _RecordingThrottle()
    llm._throttle = throttle

    # ---------------------------------------------- 1) strict JSON chat call
    messages = [
        {"role": "system",
         "content": ("You are an email triage assistant. Reply with STRICT "
                     "JSON only, no other text: {\"classification\": "
                     "\"IGNORE|NOTIFY|RESPOND\", \"confidence\": <0-1>}")},
        {"role": "user",
         "content": ("Incoming email:\nFrom: Bob <bob@corp.com>\n"
                     "Subject: Meeting request\nBody: Hi John, could we "
                     "schedule a 30-minute call this week to discuss the "
                     "project timeline? Thanks, Bob")},
    ]
    r1 = llm.chat(messages, max_tokens=200)  # floors to 256 live
    print(f"[livesmoke] 1) JSON chat: {r1.latency_ms:.0f} ms, "
          f"{r1.prompt_tokens}+{r1.completion_tokens} tok, "
          f"content={r1.content!r}")
    obj = json.loads(_strip_fences(r1.content or ""))
    assert obj["classification"] in ("IGNORE", "NOTIFY", "RESPOND"), obj
    conf = float(obj["confidence"])
    assert 0.0 <= conf <= 1.0, obj
    assert obj["classification"] == "RESPOND", obj  # a clear meeting request
    print(f"[livesmoke] 1) parsed: {obj}")

    # ------------------------------- 2) tool round-trip, legacy interface
    schema = {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string",
                                    "description": "City name"}},
            "required": ["city"],
        },
    }
    tool_messages = [
        {"role": "system",
         "content": "You are a helpful assistant with a get_weather tool."},
        {"role": "user", "content": "What is the weather in Lahore right now?"},
    ]
    r2 = llm.chat(tool_messages, functions=[schema],
                  function_call={"name": "get_weather"}, max_tokens=300)
    print(f"[livesmoke] 2) tool call: {r2.latency_ms:.0f} ms, "
          f"{r2.prompt_tokens}+{r2.completion_tokens} tok, "
          f"function_call={r2.function_call}")
    assert r2.function_call is not None, "model did not call the tool"
    assert r2.function_call["name"] == "get_weather", r2.function_call
    args = r2.function_call["arguments"]
    assert "lahore" in str(args.get("city", "")).lower(), args

    stub_result = json.dumps({"city": "Lahore", "temp_c": 31,
                              "conditions": "sunny"})
    tool_messages.append({"role": "assistant", "content": None,
                          "function_call": {"name": "get_weather",
                                            "arguments": json.dumps(args)}})
    tool_messages.append({"role": "function", "name": "get_weather",
                          "content": stub_result})
    r3 = llm.chat(tool_messages, functions=[schema], function_call="none",
                  max_tokens=300)
    print(f"[livesmoke] 2) final answer: {r3.latency_ms:.0f} ms, "
          f"{r3.prompt_tokens}+{r3.completion_tokens} tok, "
          f"content={r3.content!r}")
    assert r3.content, "no final content after tool result"
    low = r3.content.lower()
    assert "31" in low or "sunny" in low, r3.content
    print("[livesmoke] 2) legacy functions -> tools round-trip ok")

    # ------------------------------------------------ 3) local embeddings
    sents = [
        "The platform migration is scheduled for next quarter.",
        "We plan to migrate the platform early next quarter.",
        "My cat enjoys sleeping in the sun all afternoon.",
    ]
    e = llm.embed(sents)
    assert len(e.vectors) == 3
    for v in e.vectors:
        assert len(v) == cfg.embed_dim, len(v)
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))  # noqa: E731
    sim01 = dot(e.vectors[0], e.vectors[1])
    sim02 = dot(e.vectors[0], e.vectors[2])
    sim12 = dot(e.vectors[1], e.vectors[2])
    print(f"[livesmoke] 3) embeddings: dim={cfg.embed_dim}, "
          f"{e.latency_ms:.0f} ms, cos(similar)={sim01:.3f}, "
          f"cos(dissimilar)={sim02:.3f}/{sim12:.3f}")
    assert sim01 > sim02 and sim01 > sim12, (sim01, sim02, sim12)

    # ----------------------------------------------------------- 4) totals
    t = llm.usage_totals
    print(f"[livesmoke] 4) usage: {int(t['prompt_tokens'])} prompt + "
          f"{int(t['completion_tokens'])} completion tok over "
          f"{int(t['requests'])} requests, embed_tokens={int(t['embed_tokens'])}, "
          f"cost=${t['cost_usd']:.6f}, throttle slept {throttle.slept_s:.1f}s")
    assert t["cost_usd"] < 0.01
    print("[livesmoke] ALL PASSED")


if __name__ == "__main__":
    main()
