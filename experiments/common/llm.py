"""Groq chat wrapper (OpenAI-compatible endpoint) + local embeddings.

Provider adaptation (2026-08-31): chat runs on Groq's OpenAI-compatible API
(openai python SDK with ``base_url=https://api.groq.com/openai/v1``) using the
gpt-oss reasoning models with ``reasoning_effort="low"``; embeddings run
locally via sentence-transformers (BAAI/bge-base-en-v1.5, normalized, no
network). The public interface is unchanged: callers keep passing the legacy
``functions``/``function_call`` arguments and legacy function-role messages;
this wrapper translates them to the modern tools/tool_calls API at the
boundary.

Rate limiting: a sliding 60-second token window keeps estimated consumption
under 7000 tokens/minute (safety margin below Groq's 8000 TPM free tier), and
429 responses are honored via the retry-after header / "try again in Xs" body
text. Daily-quota 429s raise RuntimeError("GROQ_DAILY_LIMIT") immediately.

Selftest (``--selftest``) exercises token counting, cost math, the cap abort,
the legacy->modern translation shim, throttling, and 429 parsing via injected
fakes — no API calls, no model downloads.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional

import openai
import tiktoken

from experiments.common.exp_config import ExpConfig

ENV_PATH = "/media/adnan/DATA/Agentic-LongTerm-Memory/.env"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# USD per 1M tokens: model-prefix -> (input, output).
_CHAT_PRICES: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}
# Local sentence-transformers embeddings are free; OpenAI embeddings per 1M tokens.
_EMBED_PRICES: dict[str, float] = {
    "BAAI/bge-base-en-v1.5": 0.0,
    "text-embedding-3-small": 0.02,
}
_TRANSIENT_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)
_MAX_TRIES = 5
_DEFAULT_COST_CAP_USD = 25.0

# Reasoning models can burn the whole completion budget on reasoning tokens;
# never allow a cap so small that only reasoning fits.
_MIN_MAX_TOKENS = 256
# Sliding-window throttle: stay under 7000 tokens/min (8000 TPM free tier).
_TPM_BUDGET = 7000
_WINDOW_S = 60.0
_CHARS_PER_TOKEN = 3.5
_SLEEP_NOTICE_S = 5.0

_DURATION_UNITS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}

# Process-wide lazily loaded sentence-transformers models, keyed (name, device).
_ST_MODELS: dict[tuple[str, str], Any] = {}


def _get_st_model(name: str, device: str) -> Any:
    """Load (once per process) the sentence-transformers model ``name``."""
    key = (name, device)
    if key not in _ST_MODELS:
        from sentence_transformers import SentenceTransformer

        _ST_MODELS[key] = SentenceTransformer(name, device=device)
    return _ST_MODELS[key]


def _parse_duration_s(text: str) -> Optional[float]:
    """Parse a Go-style duration ("7.66s", "1m5s", "200ms") into seconds."""
    parts = re.findall(r"(\d+(?:\.\d+)?)(ms|s|m|h)", text)
    if not parts:
        return None
    return sum(float(num) * _DURATION_UNITS[unit] for num, unit in parts)


def _err_text(exc: Exception) -> str:
    """All available text of an API error (message + body) for matching."""
    return f"{exc} {getattr(exc, 'body', '') or ''}"


def _retry_after_s(exc: Exception) -> Optional[float]:
    """Seconds to wait per a 429: retry-after header, else 'try again in Xs'."""
    response = getattr(exc, "response", None)
    if response is not None:
        header = response.headers.get("retry-after")
        if header:
            try:
                return float(header)
            except ValueError:
                pass
    m = re.search(r"try again in ([0-9hms\.]+)", _err_text(exc), re.IGNORECASE)
    if m:
        return _parse_duration_s(m.group(1))
    return None


class _TokenThrottle:
    """Sliding-window token budget: sleep until a request's estimate fits."""

    def __init__(self, budget: int = _TPM_BUDGET, window_s: float = _WINDOW_S) -> None:
        self.budget = budget
        self.window_s = window_s
        self._events: list[list[float]] = []  # [monotonic_time, tokens]

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._events and self._events[0][0] <= cutoff:
            self._events.pop(0)

    def reserve(self, estimate: float) -> list[float]:
        """Block until ``estimate`` tokens fit in the window; record them.

        Returns the recorded event (mutable) so the caller can settle it with
        the actual token count once the response's usage is known.
        """
        while True:
            now = time.monotonic()
            self._prune(now)
            used = sum(tokens for _, tokens in self._events)
            if used + estimate <= self.budget or not self._events:
                event = [now, float(estimate)]
                self._events.append(event)
                return event
            sleep_s = max(0.05, self._events[0][0] + self.window_s - now)
            if sleep_s > _SLEEP_NOTICE_S:
                print(f"[llm] throttle: sleeping {sleep_s:.1f}s to stay under "
                      f"{self.budget} tokens/min", flush=True)
            time.sleep(sleep_s)

    @staticmethod
    def settle(event: list[float], actual_tokens: float) -> None:
        """Replace an event's estimate with the actual consumed tokens."""
        event[1] = max(0.0, float(actual_tokens))


@dataclass
class ChatResult:
    """One chat completion: text, parsed function call, usage, latency."""

    content: Optional[str]
    function_call: Optional[dict]  # {"name": str, "arguments": dict} or None
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


@dataclass
class EmbedResult:
    """One embeddings call: vectors (input order), latency, total tokens."""

    vectors: list[list[float]]
    latency_ms: float
    tokens: int


class LLM:
    """Groq chat client wrapper + local embedder, with usage totals and a cap.

    ``usage_totals`` accumulates ``{prompt_tokens, completion_tokens,
    embed_tokens, cost_usd, requests}`` across all calls; exceeding the cap
    (default $25, overridable via env ``MAILRECALL_COST_CAP``) raises
    RuntimeError.
    """

    def __init__(self, cfg: ExpConfig, client: Any = None, embedder: Any = None) -> None:
        """``client``/``embedder`` inject fakes for tests (no network)."""
        self.cfg = cfg
        self._client = client
        self._embedder = embedder
        self._enc = tiktoken.get_encoding("o200k_base")
        self._cost_cap = float(os.environ.get("MAILRECALL_COST_CAP", _DEFAULT_COST_CAP_USD))
        # OpenAI tier-1 limits are far higher than Groq's free tier.
        tpm = 150_000 if getattr(cfg, "provider", "groq") == "openai" else _TPM_BUDGET
        self._throttle = _TokenThrottle(budget=tpm)
        self.usage_totals: dict[str, float] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "embed_tokens": 0,
            "cost_usd": 0.0,
            "requests": 0,
        }

    # ------------------------------------------------------------------ client
    def _get_client(self) -> Any:
        """Lazily construct the chat client (keeps selftests offline).

        ``cfg.provider == "openai"`` -> OpenAI API with ``OPENAI_API_KEY``;
        otherwise Groq's OpenAI-compatible endpoint with ``GROQ_API_KEY``
        (the openai SDK's User-Agent is accepted by Groq; python-urllib is
        blocked).
        """
        if self._client is None:
            from dotenv import load_dotenv

            load_dotenv(ENV_PATH)
            if getattr(self.cfg, "provider", "groq") == "openai":
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise RuntimeError(f"OPENAI_API_KEY not found (looked in {ENV_PATH})")
                self._client = openai.OpenAI(api_key=api_key)
            else:
                api_key = os.environ.get("GROQ_API_KEY")
                if not api_key:
                    raise RuntimeError(f"GROQ_API_KEY not found (looked in {ENV_PATH})")
                self._client = openai.OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
        return self._client

    def _get_embedder(self) -> Any:
        """Lazily load the local sentence-transformers embedder."""
        if self._embedder is None:
            self._embedder = _get_st_model(self.cfg.embed_model, self.cfg.embed_device)
        return self._embedder

    def _with_retries(self, fn: Any) -> Any:
        """Call ``fn()`` with up to 5 tries; honor 429s, backoff transients."""
        for attempt in range(_MAX_TRIES):
            try:
                return fn()
            except openai.RateLimitError as exc:
                text = _err_text(exc).lower()
                if "insufficient_quota" in text or "credit" in text and "exhaust" in text \
                        or "no credits remaining" in text:
                    raise RuntimeError("OPENAI_CREDITS_EXHAUSTED") from exc
                if "per day" in text or "daily" in text:
                    raise RuntimeError("GROQ_DAILY_LIMIT") from exc
                if attempt == _MAX_TRIES - 1:
                    raise
                delay = _retry_after_s(exc)
                if delay is None:
                    delay = 2.0 ** attempt
                delay = min(delay + 0.25, 120.0)  # small safety pad, sane cap
                if delay > _SLEEP_NOTICE_S:
                    print(f"[llm] 429 rate-limited: sleeping {delay:.1f}s "
                          f"(attempt {attempt + 1}/{_MAX_TRIES})", flush=True)
                time.sleep(delay)
            except _TRANSIENT_ERRORS:
                if attempt == _MAX_TRIES - 1:
                    raise
                time.sleep(2.0 ** attempt)
        raise AssertionError("unreachable")

    # ---------------------------------------------- legacy -> modern translation
    @staticmethod
    def _modern_messages(messages: list[dict]) -> list[dict]:
        """Translate legacy function-call messages to tools/tool_calls form."""
        out: list[dict] = []
        last_ids: dict[str, str] = {}
        n = 0
        for m in messages:
            role = m.get("role")
            if role == "assistant" and m.get("function_call"):
                n += 1
                call_id = f"call_{n}"
                fc = m["function_call"]
                name = str(fc.get("name", ""))
                args = fc.get("arguments", "{}")
                if not isinstance(args, str):
                    args = json.dumps(args)
                last_ids[name] = call_id
                out.append({
                    "role": "assistant",
                    "content": m.get("content"),
                    "tool_calls": [{"id": call_id, "type": "function",
                                    "function": {"name": name, "arguments": args}}],
                })
            elif role == "function":
                name = str(m.get("name", ""))
                out.append({
                    "role": "tool",
                    "tool_call_id": last_ids.get(name, "call_0"),
                    "content": str(m.get("content") or ""),
                })
            else:
                out.append(m)
        return out

    @staticmethod
    def _modern_tool_choice(function_call: Any) -> Any:
        """Translate a legacy function_call directive to tool_choice."""
        if isinstance(function_call, dict):
            return {"type": "function", "function": {"name": function_call["name"]}}
        return function_call  # "auto" | "none" | "required" | None

    # -------------------------------------------------------------------- chat
    def chat(
        self,
        messages: list[dict],
        functions: Optional[list[dict]] = None,
        function_call: Any = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatResult:
        """Run a chat completion on Groq via the modern tools API.

        Keeps the legacy ``functions``/``function_call`` interface for callers;
        translates it (and any legacy function-role messages) to tools /
        tool_choice / tool messages at this boundary. Reasoning output is
        discarded: ``content`` is the final message content only.
        """
        requested_max = max_tokens or self.cfg.max_output_tokens
        effective_max = max(_MIN_MAX_TOKENS, int(requested_max))
        kwargs: dict[str, Any] = {
            "model": model or self.cfg.chat_model,
            "messages": self._modern_messages(messages),
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": effective_max,
        }
        if getattr(self.cfg, "provider", "groq") != "openai":
            # Groq's gpt-oss reasoning models; OpenAI models reject the field.
            kwargs["extra_body"] = {"reasoning_effort": "low"}
        if functions and function_call != "none":
            # Groq rejects requests where tool_choice="none" but the model
            # still emits a tool call; forcing a final answer is therefore
            # implemented by omitting the tools entirely.
            kwargs["tools"] = [{"type": "function", "function": f} for f in functions]
            choice = self._modern_tool_choice(function_call)
            if choice is not None:
                kwargs["tool_choice"] = choice

        # Throttle: estimate prompt chars/3.5 + completion cap, reserve window.
        est_chars = sum(len(str(m.get("content") or "")) for m in kwargs["messages"])
        if "tools" in kwargs:
            est_chars += len(json.dumps(kwargs["tools"]))
        estimate = est_chars / _CHARS_PER_TOKEN + effective_max
        event = self._throttle.reserve(estimate)

        def _create() -> Any:
            return self._get_client().chat.completions.create(**kwargs)

        t0 = time.perf_counter()
        resp = self._with_retries(_create)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        msg = resp.choices[0].message
        fc: Optional[dict] = None
        tool_calls = getattr(msg, "tool_calls", None)
        raw_fc = tool_calls[0].function if tool_calls else getattr(msg, "function_call", None)
        if raw_fc is not None:
            fc = {"name": raw_fc.name, "arguments": self._parse_arguments(raw_fc.arguments)}

        usage = getattr(resp, "usage", None)
        if usage is not None:
            pt, ct = usage.prompt_tokens, usage.completion_tokens
        else:  # usage missing: approximate with tiktoken
            pt = sum(self.count_tokens(str(m.get("content") or "")) for m in messages)
            ct = self.count_tokens(msg.content or "") + (
                self.count_tokens(json.dumps(fc)) if fc else 0
            )
        self._throttle.settle(event, pt + ct)
        self._accrue_chat(kwargs["model"], pt, ct)
        return ChatResult(
            content=msg.content,  # final content only; reasoning discarded
            function_call=fc,
            prompt_tokens=pt,
            completion_tokens=ct,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _parse_arguments(arguments: Any) -> dict:
        """Parse a function-call arguments payload into a dict."""
        if isinstance(arguments, dict):
            return arguments
        try:
            parsed = json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            return {"_raw": arguments}
        return parsed if isinstance(parsed, dict) else {"_raw": arguments}

    # ------------------------------------------------------------------- embed
    def embed(self, texts: list[str]) -> EmbedResult:
        """Embed ``texts``, normalized so cosine similarity equals dot product.

        ``text-embedding-*`` models are served by the OpenAI embeddings API
        (key ``OPENAI_API_KEY`` in the repo ``.env``); anything else is a local
        sentence-transformers model (no network, tokens=0, zero cost).
        Vectors keep input order.
        """
        if self.cfg.embed_model.startswith("text-embedding"):
            return self._embed_openai(list(texts))
        model = self._get_embedder()
        t0 = time.perf_counter()
        arr = model.encode(list(texts), normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        vectors = [[float(x) for x in row] for row in arr]
        self._accrue_embed(self.cfg.embed_model, 0)
        return EmbedResult(vectors=vectors, latency_ms=latency_ms, tokens=0)

    def _embed_openai(self, texts: list[str]) -> EmbedResult:
        """OpenAI embeddings API path (text-embedding-3-small, 1536-d)."""
        import openai as _openai
        if getattr(self, "_openai_embed_client", None) is None:
            from dotenv import load_dotenv
            load_dotenv(ENV_PATH, override=False)
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(f"OPENAI_API_KEY not found (looked in {ENV_PATH})")
            self._openai_embed_client = _openai.OpenAI(api_key=api_key)

        def _create() -> Any:
            return self._openai_embed_client.embeddings.create(
                model=self.cfg.embed_model, input=texts)

        t0 = time.perf_counter()
        resp = self._with_retries(_create)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        vectors = [list(d.embedding) for d in
                   sorted(resp.data, key=lambda d: d.index)]
        tokens = getattr(getattr(resp, "usage", None), "total_tokens", 0) or 0
        self._accrue_embed(self.cfg.embed_model, tokens)
        return EmbedResult(vectors=vectors, latency_ms=latency_ms, tokens=tokens)

    # ------------------------------------------------------------------ tokens
    def count_tokens(self, text: str) -> int:
        """Token count under tiktoken ``o200k_base``."""
        return len(self._enc.encode(text))

    # -------------------------------------------------------------------- cost
    @staticmethod
    def _price_for(model: str, table: dict[str, Any]) -> Any:
        """Longest-prefix price lookup; unknown models raise ValueError."""
        matches = [p for p in table if model.startswith(p)]
        if not matches:
            raise ValueError(f"no price known for model {model!r}")
        return table[max(matches, key=len)]

    def _accrue_chat(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        """Add chat usage to totals and enforce the cost cap."""
        in_price, out_price = self._price_for(model, _CHAT_PRICES)
        self.usage_totals["prompt_tokens"] += prompt_tokens
        self.usage_totals["completion_tokens"] += completion_tokens
        self.usage_totals["requests"] += 1
        self.usage_totals["cost_usd"] += (
            prompt_tokens * in_price + completion_tokens * out_price
        ) / 1e6
        self._check_cap()

    def _accrue_embed(self, model: str, tokens: int) -> None:
        """Add embedding usage to totals and enforce the cost cap."""
        price = self._price_for(model, _EMBED_PRICES)
        self.usage_totals["embed_tokens"] += tokens
        self.usage_totals["cost_usd"] += tokens * price / 1e6
        self._check_cap()

    def _check_cap(self) -> None:
        """Hard abort once cumulative cost exceeds the cap."""
        if self.usage_totals["cost_usd"] > self._cost_cap:
            raise RuntimeError(
                f"cost cap exceeded: ${self.usage_totals['cost_usd']:.4f} > "
                f"${self._cost_cap:.2f} (raise MAILRECALL_COST_CAP to override)"
            )


# --------------------------------------------------------------------- selftest
class _FakeClient:
    """Offline stand-in for the Groq client (modern tools API replies)."""

    def __init__(self) -> None:
        from types import SimpleNamespace

        self._ns = SimpleNamespace
        self.last_kwargs: dict[str, Any] = {}
        self.chat = self._ns(completions=self._ns(create=self._chat_create))

    def _chat_create(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs
        if "tools" in kwargs:
            fn = self._ns(name="get_user_info", arguments='{"field": "name"}')
            message = self._ns(content=None, function_call=None,
                               reasoning="thinking about tools...",
                               tool_calls=[self._ns(function=fn)])
        else:
            message = self._ns(content="hello back", function_call=None,
                               reasoning="pondering...", tool_calls=None)
        usage = self._ns(prompt_tokens=1000, completion_tokens=500)
        return self._ns(choices=[self._ns(message=message)], usage=usage)


class _FakeEmbedder:
    """Offline stand-in for SentenceTransformer (deterministic 4-d vectors)."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str], **kwargs: Any) -> Any:
        import numpy as np

        assert kwargs.get("normalize_embeddings") is True
        self.calls.append(list(texts))
        out = np.zeros((len(texts), 4), dtype=np.float32)
        for i in range(len(texts)):
            out[i, i % 4] = 1.0
        return out


def _fake_429(message: str, headers: Optional[dict] = None) -> openai.RateLimitError:
    """Build a real openai.RateLimitError around a synthetic httpx 429."""
    import httpx

    request = httpx.Request("POST", GROQ_BASE_URL + "/chat/completions")
    response = httpx.Response(429, headers=headers or {}, request=request)
    return openai.RateLimitError(message, response=response,
                                 body={"error": {"message": message}})


def _selftest() -> None:
    """Token counting, cost math, cap abort, translation shim, throttle, 429s.

    No API calls, no model downloads.
    """
    from experiments.common.exp_config import preset

    # A non-"text-embedding-*" model name keeps embed() on the local
    # sentence-transformers path so the injected fake embedder is used.
    # The Groq provider/models exercise the reasoning-model code path
    # (extra_body, max_tokens floor); the OpenAI path is checked below.
    cfg = preset("p3", embed_model="BAAI/bge-base-en-v1.5", embed_dim=768,
                 provider="groq", chat_model="openai/gpt-oss-120b",
                 summary_model="openai/gpt-oss-20b")
    llm = LLM(cfg, client=_FakeClient(), embedder=_FakeEmbedder())
    # The throttle is unit-tested directly below; unbound it here so the
    # selftest's fake calls never sleep.
    llm._throttle = _TokenThrottle(budget=10 ** 9)
    assert llm.count_tokens("") == 0
    assert llm.count_tokens("hello world") > 0
    assert llm.count_tokens("a" * 4000) > llm.count_tokens("a" * 40)

    # Plain chat + gpt-oss-120b cost: 1000*0.15/1M + 500*0.60/1M = 0.00045.
    r = llm.chat([{"role": "user", "content": "hi"}])
    assert r.content == "hello back" and r.function_call is None
    assert r.prompt_tokens == 1000 and r.completion_tokens == 500 and r.latency_ms >= 0
    assert abs(llm.usage_totals["cost_usd"] - 0.00045) < 1e-12
    assert llm.usage_totals["requests"] == 1
    kw = llm._client.last_kwargs
    assert kw["model"] == "openai/gpt-oss-120b"
    assert kw["extra_body"] == {"reasoning_effort": "low"}
    assert kw["max_tokens"] == cfg.max_output_tokens

    # max_tokens floor: reasoning models need room beyond reasoning tokens.
    llm.chat([{"role": "user", "content": "hi"}], max_tokens=50)
    assert llm._client.last_kwargs["max_tokens"] == _MIN_MAX_TOKENS
    llm.chat([{"role": "user", "content": "hi"}], max_tokens=500)
    assert llm._client.last_kwargs["max_tokens"] == 500

    # Legacy functions interface -> modern tools request, parsed tool_call.
    r = llm.chat([{"role": "user", "content": "hi"}],
                 functions=[{"name": "get_user_info", "parameters": {}}],
                 function_call={"name": "get_user_info"})
    kw = llm._client.last_kwargs
    assert "functions" not in kw and "function_call" not in kw
    assert kw["tools"] == [{"type": "function",
                            "function": {"name": "get_user_info", "parameters": {}}}]
    assert kw["tool_choice"] == {"type": "function",
                                 "function": {"name": "get_user_info"}}
    assert r.function_call == {"name": "get_user_info", "arguments": {"field": "name"}}
    assert r.content is None  # reasoning field is never surfaced

    # function_call="none" now omits the tools entirely (Groq rejects
    # tool_choice="none" when the model still emits a tool call).
    llm.chat([{"role": "user", "content": "hi"}],
             functions=[{"name": "f", "parameters": {}}], function_call="none")
    assert "tools" not in llm._client.last_kwargs
    assert "tool_choice" not in llm._client.last_kwargs

    # Legacy message roles translated: assistant function_call + function role.
    legacy = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None,
         "function_call": {"name": "get_user_info", "arguments": '{"a": 1}'}},
        {"role": "function", "name": "get_user_info", "content": "John Doe"},
    ]
    llm.chat(legacy)
    sent = llm._client.last_kwargs["messages"]
    assert sent[1]["tool_calls"][0]["function"]["name"] == "get_user_info"
    assert sent[1]["tool_calls"][0]["id"] == "call_1"
    assert sent[2] == {"role": "tool", "tool_call_id": "call_1", "content": "John Doe"}

    # gpt-oss-20b cost: 1000*0.075/1M + 500*0.30/1M = 0.000225.
    before = llm.usage_totals["cost_usd"]
    llm.chat([{"role": "user", "content": "hi"}], model="openai/gpt-oss-20b")
    assert abs(llm.usage_totals["cost_usd"] - before - 0.000225) < 1e-12

    # Local embeddings: shape, order, zero tokens, zero cost.
    before = llm.usage_totals["cost_usd"]
    e = llm.embed(["a", "b"])
    assert len(e.vectors) == 2 and len(e.vectors[0]) == 4
    assert e.vectors[0][0] == 1.0 and e.vectors[1][1] == 1.0 and e.tokens == 0
    assert llm.usage_totals["cost_usd"] == before
    assert llm.usage_totals["embed_tokens"] == 0

    # Invalid-JSON arguments are preserved, not silenced.
    assert LLM._parse_arguments("not json") == {"_raw": "not json"}

    # Throttle: second reservation must wait for the window to drain.
    th = _TokenThrottle(budget=100, window_s=0.3)
    ev = th.reserve(80)
    t0 = time.monotonic()
    th.reserve(80)
    assert time.monotonic() - t0 >= 0.2, "throttle did not sleep"
    _TokenThrottle.settle(ev, 10)
    assert ev[1] == 10.0
    # An oversized single request proceeds when the window is empty.
    big = _TokenThrottle(budget=100, window_s=0.3)
    t0 = time.monotonic()
    big.reserve(500)
    assert time.monotonic() - t0 < 0.1

    # 429 parsing: retry-after header, "try again in Xs" text, durations.
    assert _parse_duration_s("7.66s") == 7.66
    assert _parse_duration_s("1m5s") == 65.0
    assert _parse_duration_s("200ms") == 0.2
    assert _retry_after_s(_fake_429("slow down", headers={"retry-after": "3"})) == 3.0
    assert abs(_retry_after_s(_fake_429(
        "Rate limit reached. Please try again in 1.234s.")) - 1.234) < 1e-9

    # 429 honored then retried; daily exhaustion raises GROQ_DAILY_LIMIT.
    attempts = {"n": 0}

    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _fake_429("Please try again in 10ms.")
        return "ok"

    assert llm._with_retries(flaky) == "ok" and attempts["n"] == 2

    def daily() -> str:
        raise _fake_429("Rate limit reached ... limit 1000 requests per day.")

    try:
        llm._with_retries(daily)
    except RuntimeError as exc:
        assert str(exc) == "GROQ_DAILY_LIMIT", exc
    else:
        raise AssertionError("daily limit should raise GROQ_DAILY_LIMIT")

    # Cost cap abort, and env override.
    os.environ["MAILRECALL_COST_CAP"] = "0.0001"
    try:
        capped = LLM(cfg, client=_FakeClient())
        try:
            capped.chat([{"role": "user", "content": "hi"}])
        except RuntimeError:
            pass
        else:
            raise AssertionError("cost cap should have aborted")
        os.environ["MAILRECALL_COST_CAP"] = "100"
        assert LLM(cfg, client=_FakeClient())._cost_cap == 100.0
    finally:
        del os.environ["MAILRECALL_COST_CAP"]
    assert LLM(cfg, client=_FakeClient())._cost_cap == _DEFAULT_COST_CAP_USD

    # Unknown model pricing is an error, not a silent zero.
    try:
        llm._accrue_chat("mystery-model", 1, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown model should raise ValueError")
    print("llm selftest: OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("usage: python3 -m experiments.common.llm --selftest")
