"""Minimal chat-completions client for OpenAI-compatible endpoints.

Works against OpenAI, OpenRouter, vLLM and llama.cpp servers alike, since they
all speak the same `/chat/completions` shape. Every call records tokens,
latency and cost so the eval harness can report them per answer.
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypedDict

import httpx

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cost_usd: float | None = None
    """None when neither the provider nor the configured prices give a cost."""


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    usage: Usage


class LLMError(RuntimeError):
    pass


class BudgetExceeded(LLMError):
    pass


@dataclass
class UsageTracker:
    """Running totals across calls, with an optional spending cap.

    The cap is checked before each call, so a run can overshoot by at most one
    call's cost. Calls with unknown cost count as zero, which is why the
    clients prefer provider-reported cost when it's available.
    """

    budget_usd: float | None = None
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latencies_s: list[float] = field(default_factory=list)

    def check(self) -> None:
        if self.budget_usd is not None and self.cost_usd >= self.budget_usd:
            raise BudgetExceeded(f"spent ${self.cost_usd:.4f} of a ${self.budget_usd:.2f} budget")

    def record(self, usage: Usage) -> None:
        self.calls += 1
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.cost_usd += usage.cost_usd or 0.0
        self.latencies_s.append(usage.latency_s)


class LLMClient(Protocol):
    model: str

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> Completion: ...


_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        *,
        timeout_s: float = 60.0,
        max_retries: int = 4,
        backoff_s: float = 1.0,
        price_per_mtok: tuple[float, float] | None = None,
        tracker: UsageTracker | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.price_per_mtok = price_per_mtok
        self.tracker = tracker or UsageTracker()
        self._sleep = sleep
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout_s,
            transport=transport,
        )

    @classmethod
    def from_env(cls, **kwargs: Any) -> OpenAICompatibleClient:
        """Build a client from LLM_BASE_URL, LLM_MODEL and LLM_API_KEY."""
        base_url = os.environ.get("LLM_BASE_URL")
        model = os.environ.get("LLM_MODEL")
        if not base_url or not model:
            raise LLMError("LLM_BASE_URL and LLM_MODEL must be set")
        return cls(base_url, model, os.environ.get("LLM_API_KEY"), **kwargs)

    def close(self) -> None:
        self._http.close()

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> Completion:
        self.tracker.check()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        start = time.perf_counter()
        body = self._post_with_retries(payload)
        latency = time.perf_counter() - start

        try:
            text = body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response shape: {body!r:.300}") from exc
        raw_usage = body.get("usage") or {}
        prompt_tokens = int(raw_usage.get("prompt_tokens", 0))
        completion_tokens = int(raw_usage.get("completion_tokens", 0))
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=latency,
            cost_usd=self._cost(raw_usage, prompt_tokens, completion_tokens),
        )
        self.tracker.record(usage)
        return Completion(text=text, model=str(body.get("model", self.model)), usage=usage)

    def _cost(self, raw_usage: dict[str, Any], prompt: int, completion: int) -> float | None:
        # OpenRouter reports the charged cost directly; prefer it over our own
        # price table, which can drift from what the provider actually bills.
        if isinstance(raw_usage.get("cost"), int | float):
            return float(raw_usage["cost"])
        if self.price_per_mtok is not None:
            in_price, out_price = self.price_per_mtok
            return (prompt * in_price + completion * out_price) / 1_000_000
        return None

    def _post_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._http.post("/chat/completions", json=payload)
            except httpx.TransportError as exc:
                last_error = exc
                delay = self._backoff(attempt)
            else:
                if response.status_code < 400:
                    data = response.json()
                    if not isinstance(data, dict):
                        raise LLMError("response body is not a JSON object")
                    return data
                if response.status_code not in _RETRYABLE_STATUS:
                    raise LLMError(f"HTTP {response.status_code}: {response.text[:300]}")
                last_error = LLMError(f"HTTP {response.status_code}: {response.text[:300]}")
                delay = _retry_after(response) or self._backoff(attempt)
            if attempt < self.max_retries:
                self._sleep(delay)
        raise LLMError(f"giving up after {self.max_retries + 1} attempts") from last_error

    def _backoff(self, attempt: int) -> float:
        # Full jitter, so parallel eval workers don't retry in lockstep.
        return float(random.uniform(0, self.backoff_s * 2**attempt))


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


class FakeLLMClient:
    """Deterministic client for tests: replays canned replies, no network.

    `replies` is either a list consumed in order or a function of the
    messages. Token counts are whitespace word counts, latency is zero and
    cost is a fixed amount per call so budget logic can be exercised.
    """

    model = "fake"

    def __init__(
        self,
        replies: Sequence[str] | Callable[[Sequence[Message]], str],
        cost_per_call: float = 0.0,
        tracker: UsageTracker | None = None,
    ) -> None:
        self._replies = replies if callable(replies) else list(replies)
        self.cost_per_call = cost_per_call
        self.tracker = tracker or UsageTracker()
        self.calls: list[list[Message]] = []

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> Completion:
        self.tracker.check()
        self.calls.append(list(messages))
        if callable(self._replies):
            text = self._replies(messages)
        else:
            if not self._replies:
                raise LLMError("FakeLLMClient ran out of replies")
            text = self._replies.pop(0)
        usage = Usage(
            prompt_tokens=sum(len(m["content"].split()) for m in messages),
            completion_tokens=len(text.split()),
            latency_s=0.0,
            cost_usd=self.cost_per_call,
        )
        self.tracker.record(usage)
        return Completion(text=text, model=self.model, usage=usage)
