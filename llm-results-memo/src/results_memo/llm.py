"""Provider-agnostic LLM clients.

OpenAICompatibleClient talks to any /chat/completions endpoint (stdlib only).
FakeLLM returns a fixed text and is what tests and CI use.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Protocol


class LLMClient(Protocol):
    name: str

    def complete(self, messages: list[dict[str, str]]) -> str: ...


class FakeLLM:
    """Returns a scripted draft. Not a language model."""

    def __init__(self, text: str, name: str = "fake-llm"):
        self.text = text
        self.name = name

    def complete(self, messages: list[dict[str, str]]) -> str:
        return self.text


class OpenAICompatibleClient:
    def __init__(self, api_key: str, base_url: str, model: str, timeout: float = 60.0):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.name = f"openai-compatible:{model}"

    @classmethod
    def from_env(cls) -> "OpenAICompatibleClient | None":
        key = os.environ.get("LLM_API_KEY")
        if not key:
            return None
        return cls(
            api_key=key,
            base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
            model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        )

    def complete(self, messages: list[dict[str, str]]) -> str:
        body = json.dumps({"model": self.model, "messages": messages, "temperature": 0}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]
