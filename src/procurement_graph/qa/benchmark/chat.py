"""OpenAI-compatible chat client for Azure AI Foundry deployments (Stage 2 LLM calls).

Uses the chat.completions API as requested for Stage 2, against deployments such as
``gpt-5.4-nano`` (question generation) and ``grok-4-1-fast-non-reasoning`` (Gate B). The
``openai`` import is lazy so package imports and dry-run debugging need no credentials.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from ...llm.azure_responses import extract_json_object

DEFAULT_BASE_URL = "https://uceeh01-5458-resource.services.ai.azure.com/openai/v1"


@dataclass(frozen=True)
class ChatResult:
    parsed: dict[str, Any]
    raw_text: str
    model: str
    usage: dict[str, Any]
    attempts: int


@dataclass
class ChatClient:
    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    timeout_seconds: float = 60.0
    max_retries: int = 3
    retry_wait_seconds: float = 2.0
    temperature: float = 0.0
    json_mode: bool = False
    _client: Any = field(default=None, repr=False)
    _lock: Any = field(default_factory=Lock, repr=False, compare=False)

    @classmethod
    def from_env(cls, **overrides: Any) -> "ChatClient":
        api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set AZURE_OPENAI_API_KEY (or OPENAI_API_KEY) for Stage 2 live calls."
            )
        base_url = os.getenv("AZURE_OPENAI_BASE_URL") or DEFAULT_BASE_URL
        return cls(base_url=base_url, api_key=api_key, **overrides)

    def _ensure(self) -> None:
        if self._client is not None:
            return
        with self._lock:
            if self._client is not None:  # double-checked locking for thread-safe lazy init
                return
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - exercised only with live deps
                raise RuntimeError("Missing optional dependency `openai`.") from exc
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout_seconds)

    def complete_json(self, *, model: str, system: str, user: str) -> ChatResult:
        """Call chat.completions and parse the first JSON object from the reply."""
        result = self.complete_text(model=model, system=system, user=user)
        return ChatResult(
            parsed=extract_json_object(result.raw_text),
            raw_text=result.raw_text,
            model=result.model,
            usage=result.usage,
            attempts=result.attempts,
        )

    def complete_schema(self, *, model: str, system: str, user: str,
                        schema: dict[str, Any]) -> ChatResult:
        """Call chat.completions with a strict json_schema response_format.

        The provider constrains generation to the schema, so the reply is guaranteed well-shaped
        (no nested-key echo, no placeholder slots, enum-only fields). `schema` is the full
        ``{"name", "strict", "schema"}`` object. Falls back to first-JSON-object extraction only if
        the (guaranteed) content is not directly parseable.
        """
        import json as _json

        self._ensure()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        kwargs: dict[str, Any] = {
            "model": model, "messages": messages, "temperature": self.temperature,
            "response_format": {"type": "json_schema", "json_schema": schema},
        }
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(**kwargs)
                text = response.choices[0].message.content or ""
                try:
                    parsed = _json.loads(text) if text.strip() else {}
                except (ValueError, TypeError):
                    parsed = extract_json_object(text)
                return ChatResult(parsed=parsed if isinstance(parsed, dict) else {}, raw_text=text,
                                  model=model, usage=_usage(response), attempts=attempt + 1)
            except Exception as exc:  # pragma: no cover - exercised only in live API calls
                last_exc = exc
                if attempt >= self.max_retries or _non_retryable(exc):
                    break
                time.sleep(self.retry_wait_seconds * (attempt + 1))
        raise RuntimeError(
            f"chat.completions (json_schema) failed for {model} after {self.max_retries + 1} attempts: {last_exc}"
        )

    def complete_text(self, *, model: str, system: str, user: str) -> ChatResult:
        """Call chat.completions and return raw text without requiring JSON."""
        self._ensure()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "temperature": self.temperature}
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(**kwargs)
                text = response.choices[0].message.content or ""
                return ChatResult(
                    parsed={},
                    raw_text=text,
                    model=model,
                    usage=_usage(response),
                    attempts=attempt + 1,
                )
            except Exception as exc:  # pragma: no cover - exercised only in live API calls
                last_exc = exc
                if attempt >= self.max_retries or _non_retryable(exc):
                    break
                time.sleep(self.retry_wait_seconds * (attempt + 1))
        raise RuntimeError(
            f"chat.completions failed for {model} after {self.max_retries + 1} attempts: {last_exc}"
        )


def _non_retryable(exc: Exception) -> bool:
    """4xx client errors (bad schema, invalid request) will fail identically on every retry —
    burning retry backoff on them turned one dead run into 15+ minutes of pure waiting.
    429 (rate limit) stays retryable."""
    text = str(exc)
    if "Error code: 429" in text:
        return False
    return "Error code: 4" in text or "invalid_request_error" in text


def _usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        try:
            return usage.model_dump()
        except Exception:
            pass
    return {
        key: getattr(usage, key)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if hasattr(usage, key)
    }


__all__ = ["ChatClient", "ChatResult", "DEFAULT_BASE_URL"]
