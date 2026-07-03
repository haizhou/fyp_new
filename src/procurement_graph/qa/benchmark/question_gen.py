"""Stage 2a: natural-language question generation from an AnswerSpec.

`LLMQuestionGenerator` calls the generation model (e.g. gpt-5.4-nano). `DryRunQuestionGenerator`
builds the identical prompt but returns a deterministic templated question, so the full Stage 2
pipeline can be debugged offline with no API calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .chat import ChatClient
from .models import AnswerSpec
from .prompts import (
    GENERATION_OUTPUT_SCHEMA,
    GENERATION_SCHEMA_VERSION,
    build_factoid_generation_messages,
    build_generation_messages,
    describe_target,
    factoid_ask,
    generation_filters,
    generation_prompt_version,
    schema_hash,
)


@dataclass(frozen=True)
class GenerationOutcome:
    ok: bool
    question: str
    names_entity: bool = False
    disambiguators: tuple[str, ...] = ()
    error: str = ""
    raw: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    prompt_variant: str = "current"

    def provenance(self) -> dict[str, Any]:
        return {
            "prompt_version": generation_prompt_version(self.prompt_variant),
            "schema_version": GENERATION_SCHEMA_VERSION,
            "schema_hash": schema_hash(GENERATION_OUTPUT_SCHEMA),
            "prompt_variant": self.prompt_variant,
            "names_entity": self.names_entity,
            "disambiguators": list(self.disambiguators),
            "raw": self.raw,
            "usage": self.usage,
        }


@dataclass
class LLMQuestionGenerator:
    client: ChatClient
    model: str
    prompt_variant: str = "current"

    def generate(
        self,
        spec: AnswerSpec,
        *,
        cpv_description: str = "",
        anchor: dict[str, Any] | None = None,
        feedback: str = "",
    ) -> GenerationOutcome:
        if anchor is not None:
            system, user = build_factoid_generation_messages(
                spec.answer_field, anchor, variant=self.prompt_variant, feedback=feedback
            )
        else:
            system, user = build_generation_messages(
                spec, cpv_description=cpv_description, variant=self.prompt_variant, feedback=feedback
            )
        try:
            result = self.client.complete_json(model=self.model, system=system, user=user)
        except Exception as exc:  # pragma: no cover - live only
            return GenerationOutcome(ok=False, question="", error=f"generation_call_failed: {exc}")
        parsed = result.parsed if isinstance(result.parsed, dict) else {}
        question = str(parsed.get("question", "")).strip()
        if not question:
            return GenerationOutcome(
                ok=False,
                question="",
                error="empty_or_invalid_question",
                raw=result.raw_text,
                usage=result.usage,
                prompt_variant=self.prompt_variant,
            )
        return GenerationOutcome(
            ok=True,
            question=question,
            names_entity=bool(parsed.get("names_entity", False)),
            disambiguators=tuple(str(item) for item in parsed.get("disambiguators", []) or []),
            raw=result.raw_text,
            usage=result.usage,
            prompt_variant=self.prompt_variant,
        )


@dataclass
class DryRunQuestionGenerator:
    """Offline stand-in: assembles the real prompt, returns a deterministic question."""

    model: str = "dry-run-generator"
    prompt_variant: str = "current"

    def generate(
        self,
        spec: AnswerSpec,
        *,
        cpv_description: str = "",
        anchor: dict[str, Any] | None = None,
        feedback: str = "",
    ) -> GenerationOutcome:
        if anchor is not None:
            build_factoid_generation_messages(
                spec.answer_field, anchor, variant=self.prompt_variant, feedback=feedback
            )  # validate assembly
            clause = ", ".join(f"{key}={value}" for key, value in anchor.items() if key != "cpv_description")
            question = f"[dry-run] What is {factoid_ask(spec.answer_field)} for the contract with {clause}?"
            return GenerationOutcome(
                ok=True,
                question=question,
                raw="{\"dry_run\": true}",
                prompt_variant=self.prompt_variant,
            )
        build_generation_messages(
            spec, cpv_description=cpv_description, variant=self.prompt_variant
        )  # validate assembly
        filters = generation_filters(spec)
        clause = ", ".join(f"{item['field']}={item['value']}" for item in filters) or "the given contract"
        question = f"[dry-run] What is {describe_target(spec)} where {clause}?"
        return GenerationOutcome(
            ok=True,
            question=question,
            raw="{\"dry_run\": true}",
            prompt_variant=self.prompt_variant,
        )


__all__ = ["GenerationOutcome", "LLMQuestionGenerator", "DryRunQuestionGenerator"]
