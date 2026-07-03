"""Legacy lightweight quality gates for generated QA answer specifications.

Production Stage 1 uses :mod:`procurement_graph.qa.benchmark.stage1`, which verifies
completeness through an independent :class:`ReferenceKGIndex`. This module is kept for
mock/unit tests and the early prototype ``BenchmarkPipeline`` only. Its completeness
check is backend-only, so a passing result must not be treated as the production Gate A
completeness signal.
"""

from __future__ import annotations

from .executor import AnswerExecutionError, execute_answer_spec
from .kg_interface import QueryBackend
from .models import AnswerSpec, GateReport


def run_gate_a(backend: QueryBackend, spec: AnswerSpec) -> tuple[GateReport, GateReport]:
    """Run the legacy backend-only Gate A checks.

    This intentionally stays small for mock backends. Use ``build_stage1`` for real KG
    outputs: it adds constraint-conflict checks, independent completeness
    re-derivation, and value-sanity gates.
    """

    full_rows = backend.query(spec.constraints)
    full_ids = {backend.record_id(row) for row in full_rows if backend.record_id(row)}
    sampled_ids = set(spec.sampled_evidence_ids)

    missing_from_sample = sorted(full_ids - sampled_ids)
    extra_in_sample = sorted(sampled_ids - full_ids)
    completeness_passed = bool(sampled_ids) and not missing_from_sample and not extra_in_sample
    completeness = GateReport(
        gate="gate_a_completeness",
        passed=completeness_passed,
        status="WARN" if completeness_passed else "FAIL",
        reason="sampled evidence exactly matches full-graph query"
        if completeness_passed
        else "sampled evidence does not match full-graph query",
        metrics={
            "full_count": len(full_ids),
            "sampled_count": len(sampled_ids),
            "missing_from_sample": missing_from_sample[:20],
            "extra_in_sample": extra_in_sample[:20],
            "production_note": "legacy backend-only check; use stage1.build_stage1 for independent completeness",
        },
    )

    try:
        answer = execute_answer_spec(backend, spec)
    except AnswerExecutionError as exc:
        uniqueness = GateReport(
            gate="gate_a_uniqueness",
            passed=False,
            status="FAIL",
            reason=str(exc),
            metrics={"answer_operation": spec.answer_operation, "answer_field": spec.answer_field},
        )
    else:
        uniqueness = GateReport(
            gate="gate_a_uniqueness",
            passed=True,
            status="PASS",
            reason="answer_spec produced one deterministic golden answer",
            metrics={"answer": str(answer), "answer_field": spec.answer_field},
        )

    return completeness, uniqueness


__all__ = ["run_gate_a"]
