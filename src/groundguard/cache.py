"""On-disk cache for LLM calls, so interrupted eval runs resume for free.

Entries are keyed by a hash of everything that affects the reply (model,
messages, temperature, max_tokens, json_mode) and stored in SQLite, which
handles partial writes from a killed process better than an append-only
file would. A cache hit returns the original completion, including its
recorded cost and latency, so a resumed run reports the same totals as an
uninterrupted one; `hits` counts how many calls were served from disk.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from groundguard.llm import Completion, LLMClient, Message, Usage


def cache_key(
    model: str,
    messages: Sequence[Message],
    temperature: float,
    max_tokens: int | None,
    json_mode: bool,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "json_mode": json_mode,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class CachedLLMClient:
    def __init__(self, client: LLMClient, path: Path) -> None:
        self.client = client
        self.model = client.model
        self.hits = 0
        self.misses = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS completions (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> Completion:
        key = cache_key(self.model, messages, temperature, max_tokens, json_mode)
        row = self._db.execute("SELECT value FROM completions WHERE key = ?", (key,)).fetchone()
        if row is not None:
            self.hits += 1
            data = json.loads(row[0])
            return Completion(data["text"], data["model"], Usage(**data["usage"]))

        completion = self.client.complete(
            messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode
        )
        self.misses += 1
        self._db.execute(
            "INSERT OR REPLACE INTO completions (key, value) VALUES (?, ?)",
            (key, json.dumps(asdict(completion))),
        )
        self._db.commit()
        return completion
