"""Tests for the reasoning hardening layers borrowed from the prior pipeline:
grounding (generate-then-ground), answer-sanity, guarded verbalisation, reflector memory,
and their integration into the ReasoningPipeline."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from procurement_graph.reasoning import (
    CandidatePlan,
    LLMVerbalizer,
    QueryConstraint,
    ReasoningPipeline,
    ReflectorMemory,
    RuntimeQuerySpec,
    answer_preserves_atoms,
    check_sum_sanity,
    ground_spec,
    sanity_for_execution,
)
from procurement_graph.reasoning.models import AnswerCard, EvidenceBundle, ExecutionResult
from procurement_graph.reasoning.verifier import postflight_checks
from tests.test_reasoning_runtime import TabularRuntimeBackend, mock_records


# --- grounding -----------------------------------------------------------------------

class TestGrounding(unittest.TestCase):
    def _spec(self, **kw: Any) -> RuntimeQuerySpec:
        base = dict(spec_id="s", question="q", intent="aggregation_sum", constraints=(),
                    answer_operation="sum", answer_field="value_amount", answer_value_type="currency")
        base.update(kw)
        return RuntimeQuerySpec(**base)

    def test_sum_gets_additive_guard(self) -> None:
        grounded = ground_spec(self._spec(constraints=(QueryConstraint("release_year", "eq", 2024),)))
        self.assertTrue(grounded.ok)
        self.assertTrue(any(c.field == "value_is_additive" for c in grounded.spec.constraints))
        self.assertTrue(grounded.spec.requires_exhaustive_retrieval)

    def test_field_aliasing(self) -> None:
        spec = self._spec(answer_operation="count", answer_field="contract_node_id",
                          constraints=(QueryConstraint("year", "eq", 2024), QueryConstraint("cpv", "eq", "45000000")))
        grounded = ground_spec(spec)
        fields = {c.field for c in grounded.spec.constraints}
        self.assertIn("release_year", fields)
        self.assertIn("tender_cpv_id", fields)

    def test_unknown_field_rejected(self) -> None:
        grounded = ground_spec(self._spec(answer_operation="count",
                                          constraints=(QueryConstraint("carbon_saving", "eq", 1),)))
        self.assertFalse(grounded.ok)
        self.assertIn("carbon_saving", grounded.reason)

    def test_unsupported_operation_rejected(self) -> None:
        grounded = ground_spec(self._spec(answer_operation="average"))
        self.assertFalse(grounded.ok)
        self.assertIn("average", grounded.reason)

    def test_invalid_dedupe_key_reset_not_rejected(self) -> None:
        # A junk dedupe_key (the nano failure mode: it echoed the schema placeholder) must be
        # reset to contract_node_id, never leaked to preflight as a schema_error.
        grounded = ground_spec(self._spec(answer_operation="count", answer_field="contract_node_id",
                                          dedupe_key="optional KG field",
                                          constraints=(QueryConstraint("release_year", "eq", 2024),)))
        self.assertTrue(grounded.ok)
        self.assertEqual(grounded.spec.dedupe_key, "contract_node_id")
        self.assertTrue(any("dedupe_key" in c for c in grounded.changes))

    def test_invalid_sort_field_dropped(self) -> None:
        grounded = ground_spec(self._spec(answer_operation="count", answer_field="contract_node_id",
                                          sort_field="ranking",
                                          constraints=(QueryConstraint("release_year", "eq", 2024),)))
        self.assertTrue(grounded.ok)
        self.assertEqual(grounded.spec.sort_field, "")

    def test_signed_date_answer_field_aliased(self) -> None:
        grounded = ground_spec(self._spec(answer_operation="select_unique", answer_field="award_signed_date",
                                          constraints=(QueryConstraint("contract_node_id", "eq", "c1"),)))
        self.assertTrue(grounded.ok)
        self.assertEqual(grounded.spec.answer_field, "award_date_signed")


# --- answer sanity -------------------------------------------------------------------

class TestAnswerSanity(unittest.TestCase):
    def test_placeholder_sum_flagged(self) -> None:
        verdict = check_sum_sanity([1.0], 1.0)
        self.assertFalse(verdict.ok)
        self.assertIn("placeholder", verdict.caveat)

    def test_dominant_contributor_flagged(self) -> None:
        verdict = check_sum_sanity([5_000_000_000.0, 10_000.0, 20_000.0], 5_000_030_000.0)
        self.assertFalse(verdict.ok)
        self.assertGreaterEqual(verdict.dominant_share, 0.9)

    def test_healthy_sum_ok(self) -> None:
        self.assertTrue(check_sum_sanity([100.0, 120.0, 90.0, 110.0], 420.0).ok)

    def test_sanity_for_execution_reads_rows(self) -> None:
        spec = RuntimeQuerySpec(spec_id="s", question="q", intent="aggregation_sum", constraints=(),
                                answer_operation="sum", answer_field="value_amount", answer_value_type="currency")
        rows = ({"contract_node_id": "c1", "value_amount": "5000000000"},
                {"contract_node_id": "c2", "value_amount": "1000"})
        result = ExecutionResult(query_spec=spec, status="passed", answer="5000001000",
                                 evidence=EvidenceBundle(rows=rows, evidence_ids=("c1", "c2")))
        self.assertFalse(sanity_for_execution(result).ok)


# --- postflight (answer verification) ------------------------------------------------

class TestPostflight(unittest.TestCase):
    def _spec(self, op: str) -> RuntimeQuerySpec:
        return RuntimeQuerySpec(spec_id="s", question="q", intent="x", constraints=(),
                                answer_operation=op, answer_field="contract_node_id",
                                answer_value_type="integer")

    def test_population_coverage_flags_missing_supplier(self) -> None:
        rows = ({"contract_node_id": "c1", "supplier_count": 1, "buyer_count": 1},
                {"contract_node_id": "c2", "supplier_count": 0, "buyer_count": 1},
                {"contract_node_id": "c3", "supplier_count": 1, "buyer_count": 0})
        cov = next(c for c in postflight_checks(self._spec("count"), rows) if c["check"] == "population_coverage")
        self.assertFalse(cov["passed"])
        self.assertEqual(cov["without_supplier"], 1)
        self.assertEqual(cov["without_buyer"], 1)
        self.assertEqual(cov["matched"], 3)

    def test_full_coverage_passes(self) -> None:
        rows = ({"contract_node_id": "c1", "supplier_count": 2, "buyer_count": 1},)
        cov = next(c for c in postflight_checks(self._spec("count"), rows) if c["check"] == "population_coverage")
        self.assertTrue(cov["passed"])

    def test_select_unique_multiplicity(self) -> None:
        rows = ({"contract_node_id": "c1"}, {"contract_node_id": "c2"})
        uniq = next(c for c in postflight_checks(self._spec("select_unique"), rows) if c["check"] == "answer_uniqueness")
        self.assertFalse(uniq["passed"])
        self.assertEqual(uniq["matching_contracts"], 2)


# --- guarded verbalisation -----------------------------------------------------------

class _MockChat:
    def __init__(self, answer: str) -> None:
        self._answer = answer

    def complete_json(self, *, model: str, system: str, user: str) -> Any:
        class _R:
            parsed = {"answer": self._answer}
        return _R()


class TestVerbalize(unittest.TestCase):
    def _card(self, answer: Any) -> AnswerCard:
        spec = RuntimeQuerySpec(spec_id="s", question="How many?", intent="aggregation_count", constraints=(),
                                answer_operation="count", answer_field="contract_node_id", answer_value_type="integer")
        result = ExecutionResult(query_spec=spec, status="passed", answer=answer)
        return AnswerCard(question="How many?", answer=answer, answer_text=f"The answer is {answer}.",
                          query_spec=spec, execution=result, confidence_label="high")

    def test_atom_preserved_answer_applied(self) -> None:
        out = LLMVerbalizer(client=_MockChat("There are 42 matching contracts."), model="m").verbalize(self._card(42))
        self.assertEqual(out.source, "llm")
        self.assertIn("42", out.text)

    def test_dropped_atom_rejected_to_fallback(self) -> None:
        out = LLMVerbalizer(client=_MockChat("There are several contracts."), model="m").verbalize(self._card(42))
        self.assertEqual(out.source, "deterministic_fallback")
        self.assertEqual(out.status, "rejected_dropped_atoms")

    def test_answer_preserves_atoms_number_tolerance(self) -> None:
        ok, missing = answer_preserves_atoms("The total is GBP 1,912,915.", ["1912915"])
        self.assertTrue(ok)
        self.assertEqual(missing, [])


# --- reflector memory ----------------------------------------------------------------

class TestReflectorMemory(unittest.TestCase):
    def test_records_only_non_benign(self) -> None:
        tmp_root = Path(__file__).parent.parent / "tmp"
        tmp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
            mem = ReflectorMemory(path=Path(tmp) / "mem.jsonl")
            backend = TabularRuntimeBackend(mock_records())
            from procurement_graph.reasoning import RuleBasedDryRunPlanner

            pipeline = ReasoningPipeline(backend=backend, planner=RuleBasedDryRunPlanner())
            ok_trace = pipeline.run("How many works contracts were published in 2024?")
            unsupported_trace = pipeline.run("Which contract had the strongest carbon reduction clause?")
            self.assertFalse(mem.record(ok_trace))  # no_repair_needed -> not stored
            self.assertTrue(mem.record(unsupported_trace))  # mark_unsupported -> stored
            self.assertEqual(mem.diagnosis_counts().get("mark_unsupported"), 1)


# --- integration: hardened pipeline --------------------------------------------------

class _AliasedSumPlanner:
    """Emits an LLM-style sum with a natural field alias and NO additive guard."""

    def plan(self, question: str) -> tuple[CandidatePlan, ...]:
        spec = RuntimeQuerySpec(
            spec_id="aliased-sum", question=question, intent="aggregation_sum",
            constraints=(QueryConstraint("year", "eq", 2024), QueryConstraint("category", "eq", "works")),
            answer_operation="sum", answer_field="value", answer_value_type="currency",
            requires_exhaustive_retrieval=False,
        )
        return (CandidatePlan(plan_id="p0", query_spec=spec, status="planned", confidence=0.9),)


class _JunkBookkeepingPlanner:
    """Emits a valid count plan but with an LLM-invented dedupe_key + sort_field (the nano
    schema_error failure mode). Grounding must sanitize these instead of failing preflight."""

    def plan(self, question: str) -> tuple[CandidatePlan, ...]:
        spec = RuntimeQuerySpec(
            spec_id="junk-count", question=question, intent="aggregation_count",
            constraints=(QueryConstraint("release_year", "eq", 2024), QueryConstraint("tender_category", "eq", "works")),
            answer_operation="count", answer_field="contract_node_id", answer_value_type="integer",
            dedupe_key="optional KG field", sort_field="ranking", requires_exhaustive_retrieval=True,
        )
        return (CandidatePlan(plan_id="p0", query_spec=spec, status="planned", confidence=0.9),)


class TestHardenedPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = TabularRuntimeBackend(mock_records())

    def test_grounding_repairs_llm_sum_end_to_end(self) -> None:
        # year/category/value aliases + missing additive guard are all fixed by grounding,
        # then executed. mock_records: c1,c2 are 2024 works additive (100+250=350); c3 non-additive.
        trace = ReasoningPipeline(backend=self.backend, planner=_AliasedSumPlanner()).run(
            "What is the total value of 2024 works contracts?"
        )
        self.assertTrue(trace.execution.passed)
        self.assertEqual(str(trace.answer_card.answer), "350.00")
        self.assertIn("added value_is_additive guard for sum", trace.metadata["grounding_changes"])
        self.assertIn("execution", trace.answer_card.confidence_breakdown)

    def test_junk_bookkeeping_fields_do_not_schema_error(self) -> None:
        # Regression for the nano hard-20 run: a correct count plan carrying a junk dedupe_key /
        # sort_field used to die with execution_status='schema_error'. Grounding now sanitizes them.
        trace = ReasoningPipeline(backend=self.backend, planner=_JunkBookkeepingPlanner()).run(
            "How many works contracts were published in 2024?"
        )
        self.assertNotEqual(trace.execution.status, "schema_error")
        self.assertTrue(trace.execution.passed)
        self.assertEqual(trace.execution.query_spec.dedupe_key, "contract_node_id")
        self.assertEqual(trace.execution.query_spec.sort_field, "")

    def test_count_discloses_population_coverage_gap(self) -> None:
        # The conjunction over-count lesson: a count that includes notices with no recorded
        # supplier/buyer must SAY so on the answer card, not pass silently as high confidence.
        records = [
            {"contract_node_id": "c1", "release_year": 2024, "tender_category": "works",
             "supplier_count": 1, "buyer_count": 1},
            {"contract_node_id": "c2", "release_year": 2024, "tender_category": "works",
             "supplier_count": 0, "buyer_count": 1},
        ]
        trace = ReasoningPipeline(backend=TabularRuntimeBackend(records), planner=_JunkBookkeepingPlanner()).run(
            "How many works contracts were published in 2024?"
        )
        self.assertTrue(trace.execution.passed)
        self.assertEqual(trace.answer_card.answer, 2)
        self.assertTrue(any("population coverage" in lim for lim in trace.answer_card.limitations))

    def test_dominant_contributor_downgrades_confidence(self) -> None:
        records = [
            {"contract_node_id": "c1", "release_year": 2024, "tender_category": "works",
             "value_amount": "5000000000", "value_is_additive": True},
            {"contract_node_id": "c2", "release_year": 2024, "tender_category": "works",
             "value_amount": "1000", "value_is_additive": True},
        ]
        backend = TabularRuntimeBackend(records)
        trace = ReasoningPipeline(backend=backend, planner=_AliasedSumPlanner()).run(
            "What is the total value of 2024 works contracts?"
        )
        self.assertTrue(trace.execution.passed)
        self.assertEqual(trace.answer_card.confidence_label, "low")
        self.assertTrue(trace.answer_card.sanity_flags)
        self.assertTrue(any("answer sanity" in lim for lim in trace.answer_card.limitations))


if __name__ == "__main__":
    unittest.main()
