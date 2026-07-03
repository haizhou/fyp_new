"""Azure OpenAI Responses API helpers.

Imports for OpenAI/Azure SDKs are lazy so package imports and offline tests do
not require Azure credentials or optional dependencies.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

DEFAULT_SCOPE = "https://ai.azure.com/.default"


@dataclass(frozen=True)
class AzureResponsesConfig:
    endpoint: str
    deployment_name: str
    scope: str = DEFAULT_SCOPE
    max_retries: int = 2
    retry_wait_seconds: float = 2.0
    timeout_seconds: float | None = 120.0

    @classmethod
    def from_env(cls) -> "AzureResponsesConfig":
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_AI_OPENAI_ENDPOINT")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        scope = os.getenv("AZURE_OPENAI_SCOPE", DEFAULT_SCOPE)
        timeout_raw = os.getenv("AZURE_OPENAI_TIMEOUT_SECONDS", "120")
        if not endpoint:
            raise ValueError("Missing AZURE_OPENAI_ENDPOINT or AZURE_AI_OPENAI_ENDPOINT")
        if not deployment:
            raise ValueError("Missing AZURE_OPENAI_DEPLOYMENT or AZURE_OPENAI_DEPLOYMENT_NAME")
        timeout_seconds = float(timeout_raw) if timeout_raw else None
        return cls(endpoint=endpoint, deployment_name=deployment, scope=scope, timeout_seconds=timeout_seconds)


def create_client(config: AzureResponsesConfig):
    """Create an OpenAI client authenticated with Azure DefaultAzureCredential."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Missing optional dependency `openai`. Install requirements before live calls.") from exc

    api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if api_key:
        return OpenAI(base_url=config.endpoint, api_key=api_key, timeout=config.timeout_seconds)

    try:
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    except ImportError as exc:
        raise RuntimeError(
            "Missing optional dependency `azure-identity`. Install requirements before live Azure calls."
        ) from exc

    token_provider = get_bearer_token_provider(DefaultAzureCredential(), config.scope)
    return OpenAI(base_url=config.endpoint, api_key=token_provider, timeout=config.timeout_seconds)


def response_output_text(response: Any) -> str:
    """Extract text from an OpenAI Responses API result across SDK versions."""
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    output = getattr(response, "output", None)
    if output is None and isinstance(response, dict):
        output = response.get("output")
    if not output:
        return str(response)

    chunks: list[str] = []
    for item in output:
        content = getattr(item, "content", None)
        if content is None and isinstance(item, dict):
            content = item.get("content")
        if isinstance(content, str):
            chunks.append(content)
            continue
        for part in content or []:
            text = getattr(part, "text", None)
            if text is None and isinstance(part, dict):
                text = part.get("text")
            if text:
                chunks.append(str(text))
    return "\n".join(chunks).strip()


def response_to_dict(response: Any) -> dict[str, Any]:
    """Return a JSON-serialisable response audit record."""
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    if isinstance(response, dict):
        return response
    return {"repr": repr(response)}


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object in model text, including fenced JSON blocks."""
    stripped = (text or "").strip()
    if not stripped:
        raise ValueError("empty model output")

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return json.loads(fence.group(1))

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(stripped[start : end + 1])


def call_responses_api(
    client: Any,
    config: AzureResponsesConfig,
    input_payload: Any,
    text_format: dict[str, Any] | None = None,
) -> Any:
    """Call Responses API with simple retry on transient SDK errors."""
    response, _audit = call_responses_api_audited(client, config, input_payload, text_format=text_format)
    return response


def call_responses_api_audited(
    client: Any,
    config: AzureResponsesConfig,
    input_payload: Any,
    text_format: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Call Responses API and return retry telemetry for audit logs."""
    last_exc: Exception | None = None
    retry_errors: list[dict[str, str]] = []
    for attempt in range(config.max_retries + 1):
        try:
            kwargs: dict[str, Any] = {"model": config.deployment_name, "input": input_payload}
            if text_format:
                kwargs["text"] = {"format": text_format}
            return client.responses.create(**kwargs), {
                "attempts": attempt + 1,
                "retry_errors": retry_errors,
            }
        except Exception as exc:  # pragma: no cover - exercised only in live API calls
            last_exc = exc
            if attempt >= config.max_retries:
                break
            retry_errors.append({
                "attempt": str(attempt + 1),
                "error": type(exc).__name__,
                "message": str(exc),
            })
            time.sleep(config.retry_wait_seconds * (attempt + 1))
    raise RuntimeError(f"Azure OpenAI Responses call failed: {last_exc}") from last_exc


__all__ = [
    "AzureResponsesConfig",
    "call_responses_api",
    "call_responses_api_audited",
    "create_client",
    "extract_json_object",
    "response_output_text",
    "response_to_dict",
]
