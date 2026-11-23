from pathlib import Path

from groundguard.cache import CachedLLMClient, cache_key
from groundguard.llm import FakeLLMClient, Message

MSGS: list[Message] = [{"role": "user", "content": "hi"}]


def test_second_call_is_served_from_disk(tmp_path: Path) -> None:
    fake = FakeLLMClient(["first", "second"], cost_per_call=0.01)
    cached = CachedLLMClient(fake, tmp_path / "llm.sqlite")

    a = cached.complete(MSGS)
    b = cached.complete(MSGS)
    assert a == b
    assert len(fake.calls) == 1
    assert (cached.hits, cached.misses) == (1, 1)


def test_cache_survives_a_new_process(tmp_path: Path) -> None:
    path = tmp_path / "llm.sqlite"
    first = CachedLLMClient(FakeLLMClient(["stored"], cost_per_call=0.02), path)
    first.complete(MSGS, json_mode=True)
    first.close()

    fresh_fake = FakeLLMClient([])  # would raise if called
    resumed = CachedLLMClient(fresh_fake, path)
    result = resumed.complete(MSGS, json_mode=True)
    assert result.text == "stored"
    assert result.usage.cost_usd == 0.02
    assert fresh_fake.calls == []


def test_parameters_are_part_of_the_key(tmp_path: Path) -> None:
    fake = FakeLLMClient(["a", "b", "c"])
    cached = CachedLLMClient(fake, tmp_path / "llm.sqlite")
    cached.complete(MSGS)
    cached.complete(MSGS, json_mode=True)
    cached.complete(MSGS, temperature=0.7)
    assert len(fake.calls) == 3


def test_key_depends_on_model_and_message_content() -> None:
    base = cache_key("m", MSGS, 0.0, None, False)
    assert base == cache_key("m", list(MSGS), 0.0, None, False)
    assert base != cache_key("other", MSGS, 0.0, None, False)
    assert base != cache_key("m", [{"role": "user", "content": "hi!"}], 0.0, None, False)
