import json
from collections.abc import Callable

import httpx
import pytest

from groundguard.llm import (
    BudgetExceeded,
    FakeLLMClient,
    LLMError,
    Message,
    OpenAICompatibleClient,
    UsageTracker,
)

HELLO: list[Message] = [{"role": "user", "content": "Say hello"}]


def ok_body(text: str = "hello", usage: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "model": "test-model",
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": usage if usage is not None else {"prompt_tokens": 3, "completion_tokens": 1},
    }


def make_client(
    handler: Callable[[httpx.Request], httpx.Response], **kwargs: object
) -> tuple[OpenAICompatibleClient, list[float]]:
    sleeps: list[float] = []
    client = OpenAICompatibleClient(
        "https://llm.test/v1",
        "test-model",
        "secret",
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
        **kwargs,  # type: ignore[arg-type]
    )
    return client, sleeps


def test_sends_openai_style_request() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=ok_body())

    client, _ = make_client(handler)
    result = client.complete(HELLO, max_tokens=5, json_mode=True)

    assert result.text == "hello"
    request = seen[0]
    assert request.url == "https://llm.test/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer secret"
    payload = json.loads(request.content)
    assert payload["model"] == "test-model"
    assert payload["messages"] == HELLO
    assert payload["max_tokens"] == 5
    assert payload["response_format"] == {"type": "json_object"}


def test_records_tokens_latency_and_provider_cost() -> None:
    usage = {"prompt_tokens": 10, "completion_tokens": 4, "cost": 0.0021}
    client, _ = make_client(lambda r: httpx.Response(200, json=ok_body(usage=usage)))
    result = client.complete(HELLO)

    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 4
    assert result.usage.cost_usd == pytest.approx(0.0021)
    assert result.usage.latency_s >= 0
    assert client.tracker.calls == 1
    assert client.tracker.cost_usd == pytest.approx(0.0021)


def test_falls_back_to_configured_prices() -> None:
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 500_000}
    client, _ = make_client(
        lambda r: httpx.Response(200, json=ok_body(usage=usage)), price_per_mtok=(0.15, 0.60)
    )
    assert client.complete(HELLO).usage.cost_usd == pytest.approx(0.45)


def test_cost_is_unknown_without_prices() -> None:
    client, _ = make_client(lambda r: httpx.Response(200, json=ok_body()))
    assert client.complete(HELLO).usage.cost_usd is None


def test_retries_rate_limits_honouring_retry_after() -> None:
    responses = iter(
        [
            httpx.Response(429, headers={"Retry-After": "2"}, text="slow down"),
            httpx.Response(503, text="busy"),
            httpx.Response(200, json=ok_body("finally")),
        ]
    )
    client, sleeps = make_client(lambda r: next(responses), backoff_s=0.5)

    assert client.complete(HELLO).text == "finally"
    assert sleeps[0] == 2.0
    assert 0 <= sleeps[1] <= 0.5 * 2
    assert len(sleeps) == 2


def test_retries_transport_errors_then_gives_up() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client, sleeps = make_client(handler, max_retries=2)
    with pytest.raises(LLMError, match="3 attempts"):
        client.complete(HELLO)
    assert len(sleeps) == 2


def test_client_errors_are_not_retried() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(401, text="bad key")

    client, sleeps = make_client(handler)
    with pytest.raises(LLMError, match="401"):
        client.complete(HELLO)
    assert len(calls) == 1
    assert sleeps == []


def test_malformed_response_raises() -> None:
    client, _ = make_client(lambda r: httpx.Response(200, json={"choices": []}))
    with pytest.raises(LLMError, match="unexpected response"):
        client.complete(HELLO)


def test_budget_blocks_calls_once_spent() -> None:
    tracker = UsageTracker(budget_usd=0.01)
    usage = {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.006}
    client, _ = make_client(
        lambda r: httpx.Response(200, json=ok_body(usage=usage)), tracker=tracker
    )
    client.complete(HELLO)
    client.complete(HELLO)  # 0.006 < 0.01 at check time, so this goes through
    with pytest.raises(BudgetExceeded):
        client.complete(HELLO)
    assert tracker.calls == 2


def test_from_env_requires_url_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("LLM_MODEL", "m")
    with pytest.raises(LLMError):
        OpenAICompatibleClient.from_env()

    monkeypatch.setenv("LLM_BASE_URL", "https://llm.test/v1")
    client = OpenAICompatibleClient.from_env()
    assert client.model == "m"
    client.close()


def test_fake_client_replays_in_order_and_tracks_usage() -> None:
    fake = FakeLLMClient(["one two", "three"], cost_per_call=0.5)
    assert fake.complete(HELLO).text == "one two"
    assert fake.complete(HELLO).usage.completion_tokens == 1
    assert fake.tracker.cost_usd == 1.0
    assert fake.calls[0] == HELLO
    with pytest.raises(LLMError):
        fake.complete(HELLO)


def test_fake_client_accepts_reply_function() -> None:
    fake = FakeLLMClient(lambda msgs: msgs[-1]["content"].upper())
    assert fake.complete(HELLO).text == "SAY HELLO"
