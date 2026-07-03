"""Early prototype end-to-end QA benchmark orchestration.

The production build is split into ``stage1.build_stage1`` and ``stage2.build_stage2``.
This module remains for small mock tests and framework examples; it uses the legacy
backend-only ``run_gate_a`` rather than the independent production Gate A.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .executor import execute_answer_spec
from .gates import run_gate_a
from .kg_interface import QueryBackend
from .models import AnswerSpec, BenchmarkExample, GateReport, SemanticCheckResult
from .prompts import build_question_generation_prompt, build_semantic_verification_prompt


class QuestionGenerator(Protocol):
    def generate_question(
        self,
        spec: AnswerSpec,
        evidence_records: list[dict[str, Any]],
        golden_answer: Any,
    ) -> str:
        ...


class SemanticVerifier(Protocol):
    def verify(
        self,
        question: str,
        evidence_records: list[dict[str, Any]],
        golden_answer: Any,
    ) -> SemanticCheckResult:
        ...


@dataclass
class PromptOnlyQuestionGenerator:
    """Deterministic placeholder that exposes the prompt without calling an LLM."""

    def generate_question(
        self,
        spec: AnswerSpec,
        evidence_records: list[dict[str, Any]],
        golden_answer: Any,
    ) -> str:
        prompt = build_question_generation_prompt(spec, evidence_records, golden_answer)
        return f"What is the {spec.answer_field.replace('_', ' ')} for this contract?"


@dataclass
class ExactMatchSemanticVerifier:
    """Test verifier. Real runs should use an independent model implementation."""

    predicted_answer: Any | None = None

    def verify(
        self,
        question: str,
        evidence_records: list[dict[str, Any]],
        golden_answer: Any,
    ) -> SemanticCheckResult:
        prompt = build_semantic_verification_prompt(question, evidence_records)
        predicted = golden_answer if self.predicted_answer is None else self.predicted_answer
        passed = _normalise_answer(predicted) == _normalise_answer(golden_answer)
        return SemanticCheckResult(
            passed=passed,
            predicted_answer=predicted,
            reason="predicted answer matches golden answer" if passed else "predicted answer differs",
            raw_response={"prompt": prompt},
        )


@dataclass
class BenchmarkPipeline:
    backend: QueryBackend
    question_generator: QuestionGenerator
    semantic_verifier: SemanticVerifier

    def build_one(self, spec: AnswerSpec) -> BenchmarkExample | None:
        gate_reports = run_gate_a(self.backend, spec)
        if not all(report.passed for report in gate_reports):
            return None

        evidence_records = self.backend.query(spec.constraints)
        golden_answer = execute_answer_spec(self.backend, spec)
        question = self.question_generator.generate_question(spec, evidence_records, golden_answer)
        semantic_check = self.semantic_verifier.verify(question, evidence_records, golden_answer)
        gate_b = GateReport(
            gate="gate_b_semantic_verification",
            passed=semantic_check.passed,
            reason=semantic_check.reason,
            metrics={"predicted_answer": str(semantic_check.predicted_answer)},
        )
        all_reports = (*gate_reports, gate_b)
        if not semantic_check.passed:
            return None

        return BenchmarkExample(
            spec=spec,
            question=question,
            golden_answer=golden_answer,
            evidence_ids=tuple(sorted({self.backend.record_id(row) for row in evidence_records})),
            gate_reports=all_reports,
            semantic_check=semantic_check,
        )

    def build_many(self, specs: list[AnswerSpec]) -> list[BenchmarkExample]:
        examples: list[BenchmarkExample] = []
        for spec in specs:
            example = self.build_one(spec)
            if example is not None:
                examples.append(example)
        return examples


def _normalise_answer(value: Any) -> str:
    return str(value).strip().casefold()


__all__ = [
    "BenchmarkPipeline",
    "ExactMatchSemanticVerifier",
    "PromptOnlyQuestionGenerator",
    "QuestionGenerator",
    "SemanticVerifier",
]
