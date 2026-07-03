"""Tests for the runtime reasoning closed loop: linking, retrieval, reflector, evidence,
the end-to-end ReasoningPipeline, the KG adapter, and the LLM planner adapter."""

from __future__ import annotations

import sys
import unittest
from decimal import Decimal
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from procurement_graph.reasoning import (
    CandidatePlan,
    LexicalTopKCandidateRetriever,
    NullDocumentInspector,
    QueryConstraint,
    ReasoningPipeline,
    RuleBasedDryRunPlanner,
    RuntimeQuerySpec,
    TopScoreCandidateSelector,
    build_evidence_verdict,
    execute_query_spec,
    expansion_required,
    link_question,
    plan_retrieval,
    reflect,
    reflect_plan,
)
from procurement_graph.reasoning.evidence import DocumentInspection
from procurement_graph.reasoning.models import EntityLinkCandidate, EvidenceBundle, ExecutionResult
from tests.test_reasoning_runtime import TabularRuntimeBackend, mock_records


# --- linking -------------------------------------------------------------------------

class TestLinking(unittest.TestCase):
    def test_links_year_category_cpv_and_signed_date(self) -> None:
        link = link_question("How many services contracts published in 2024 under CPV 79341000 with a signed award date?")
        fields = {(c.field, c.value) for c in link.constraints}
        self.assertIn(("release_year", 2024), fields)
        self.assertIn(("tender_category", "services"), fields)
        self.assertIn(("tender_cpv_id", "79341000"), fields)
        self.assertIn(("has_award_signed_date", True), fields)
        self.assertTrue(link.asks_count)

    def test_flags_unsupported_concept(self) -> None:
        link = link_question("Which contract has the best carbon reduction clause?")
        self.assertIn("carbon", link.unsupported_reason)

    def test_org_resolver_adds_supplier_constraint(self) -> None:
        class _Resolver:
            def resolve(self, mention: str) -> list[EntityLinkCandidate]:
                return [EntityLinkCandidate(mention, "canon:1", "BuildCo Ltd", "organization", 1.0)]

        link = link_question("How many contracts awarded to BuildCo in 2024?", org_resolver=_Resolver())
        supplier = [c for c in link.constraints if c.field == "supplier_name"]
        self.assertEqual(len(supplier), 1)
        self.assertEqual(supplier[0].value, "BuildCo Ltd")
        self.assertEqual(supplier[0].metadata["canonical_id"], "canon:1")


# --- retrieval -----------------------------------------------------------------------

class TestRetrieval(unittest.TestCase):
    def _spec(self, **kw: Any) -> RuntimeQuerySpec:
        base = dict(spec_id="s", question="q", intent="aggregation_count", constraints=(),
                    answer_operation="count", answer_field="contract_node_id", answer_value_type="integer")
        base.update(kw)
        return RuntimeQuerySpec(**base)

    def test_exact_filters_for_supported_count(self) -> None:
        plan = plan_retrieval(self._spec())
        self.assertEqual(plan.mode, "exact_filters")

    def test_role_path_needs_expansion(self) -> None:
        self.assertIn("expansion", expansion_required(self._spec(intent="role_path")))
        self.assertEqual(plan_retrieval(self._spec(intent="role_path")).mode, "needs_expansion")

    def test_unsupported_operation_gated(self) -> None:
        # top_k/argmax/exists are now executable single-hop ops; compare/duration still need the
        # decomposition planner and are gated at the single-spec retrieval level.
        self.assertTrue(expansion_required(self._spec(answer_operation="duration")))
        self.assertEqual(plan_retrieval(self._spec(answer_operation="compare")).mode, "unsupported")


# --- reflector -----------------------------------------------------------------------

class TestReflector(unittest.TestCase):
    def _fail(self, status: str, spec: RuntimeQuerySpec) -> ExecutionResult:
        return ExecutionResult(query_spec=spec, status=status)

    def _spec(self, constraints, operation="count") -> RuntimeQuerySpec:
        return RuntimeQuerySpec(spec_id="s", question="q", intent="aggregation_count", constraints=tuple(constraints),
                                answer_operation=operation, answer_field="value_amount" if operation == "sum" else "contract_node_id",
                                answer_value_type="currency" if operation == "sum" else "integer")

    def test_reflect_plan_unsupported(self) -> None:
        planner = RuleBasedDryRunPlanner()
        [plan] = planner.plan("Which contract had the biggest carbon saving?")
        self.assertEqual(reflect_plan(plan).action, "mark_unsupported")

    def test_no_results_relaxes_most_specific(self) -> None:
        spec = self._spec((QueryConstraint("release_year", "eq", 2024), QueryConstraint("tender_cpv_id", "eq", "45000000")))
        plan = CandidatePlan(plan_id="p", query_spec=spec)
        action = reflect(self._fail("no_results", spec), plan, rounds_used=0, max_rounds=3)
        self.assertEqual(action.action, "relax_non_answer_constraints")
        self.assertEqual(action.metadata["dropped_field"], "tender_cpv_id")

    def test_non_additive_sum_replans_with_guard(self) -> None:
        spec = self._spec((QueryConstraint("contract_node_id", "exists"),), operation="sum")
        plan = CandidatePlan(plan_id="p", query_spec=spec)
        action = reflect(self._fail("incomplete_evidence", spec), plan)
        self.assertEqual(action.action, "replan_query")
        self.assertTrue(any(c.field == "value_is_additive" for c in action.suggested_constraints))

    def test_multiple_answers_asks_clarifying(self) -> None:
        spec = self._spec((QueryConstraint("buyer_name", "eq", "Alpha"),), operation="select_unique")
        action = reflect(self._fail("multiple_answers", spec), CandidatePlan(plan_id="p", query_spec=spec))
        self.assertEqual(action.action, "ask_clarifying_question")

    def test_schema_error_marks_unsupported(self) -> None:
        spec = self._spec((QueryConstraint("carbon_saving", "eq", 1),))
        action = reflect(self._fail("schema_error", spec), CandidatePlan(plan_id="p", query_spec=spec))
        self.assertEqual(action.action, "mark_unsupported")


# --- evidence ------------------------------------------------------------------------

class TestEvidence(unittest.TestCase):
    def _passed(self) -> ExecutionResult:
        spec = RuntimeQuerySpec(spec_id="s", question="q", intent="aggregation_count", constraints=(),
                                answer_operation="count", answer_field="contract_node_id", answer_value_type="integer")
        return ExecutionResult(query_spec=spec, status="passed", answer=3,
                               evidence=EvidenceBundle(evidence_ids=("c1", "c2", "c3")))

    def test_passed_is_kg_supported(self) -> None:
        verdict = build_evidence_verdict(self._passed())
        self.assertEqual(verdict.status, "kg_supported")
        self.assertEqual(verdict.document_status, "not_needed")
        self.assertGreater(verdict.confidence, 0.8)

    def test_failed_is_insufficient(self) -> None:
        spec = self._passed().query_spec
        verdict = build_evidence_verdict(ExecutionResult(query_spec=spec, status="no_results"))
        self.assertEqual(verdict.status, "insufficient_evidence")

    def test_document_contradiction_downgrades(self) -> None:
        class _Contradictor:
            def inspect(self, result: ExecutionResult) -> DocumentInspection:
                return DocumentInspection(status="checked_contradicted",
                                          contradictions=({"note": "document disagrees"},))

        verdict = build_evidence_verdict(self._passed(), inspector=_Contradictor(), need_documents=True)
        self.assertEqual(verdict.status, "contradicted")

    def test_null_inspector_keeps_kg_answer(self) -> None:
        verdict = build_evidence_verdict(self._passed(), inspector=NullDocumentInspector(), need_documents=True)
        self.assertEqual(verdict.status, "kg_supported")


# --- end-to-end pipeline -------------------------------------------------------------

class _UnguardedSumPlanner:
    """Planner that omits the additive-sum guard, so the reflector must repair it."""

    def plan(self, question: str) -> tuple[CandidatePlan, ...]:
        spec = RuntimeQuerySpec(
            spec_id="unguarded-sum", question=question, intent="aggregation_sum",
            constraints=(QueryConstraint("contract_node_id", "exists"),),
            answer_operation="sum", answer_field="value_amount", answer_value_type="currency",
            dedupe_key="contract_node_id", requires_exhaustive_retrieval=True,
        )
        return (CandidatePlan(plan_id="p0", query_spec=spec, status="planned", confidence=0.9),)


class _BadTitlePlanner:
    def plan(self, question: str) -> tuple[CandidatePlan, ...]:
        spec = RuntimeQuerySpec(
            spec_id="bad-title", question=question, intent="factoid",
            constraints=(QueryConstraint("tender_title", "eq", "Bridge repair"),),
            answer_operation="select_unique", answer_field="buyer_name", answer_value_type="string",
            dedupe_key="contract_node_id",
        )
        return (CandidatePlan(plan_id="p0", query_spec=spec, status="planned", confidence=0.9),)


class _RecordingVerificationAnalyzer:
    def analyze_verification(self, **kwargs: Any) -> dict[str, Any]:
        return {"diagnosis": "recorded", "status": kwargs["result"].status}


class _RecordingReflectionAnalyzer:
    def analyze_reflection(self, **kwargs: Any) -> dict[str, Any]:
        return {"assessment": "recorded", "action": kwargs["action"].action}


class TestReasoningPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = TabularRuntimeBackend(mock_records())
        self.planner = RuleBasedDryRunPlanner()

    def test_count_end_to_end(self) -> None:
        pipeline = ReasoningPipeline(backend=self.backend, planner=self.planner)
        trace = pipeline.run("How many works contracts were published in 2024?")
        self.assertTrue(trace.execution.passed)
        self.assertEqual(trace.answer_card.answer, 2)
        self.assertEqual(trace.evidence_verdict.status, "kg_supported")
        self.assertEqual(trace.reflection.action, "no_repair_needed")
        self.assertIn("plans", trace.metadata)
        self.assertIn("attempts", trace.metadata)
        attempt = trace.metadata["attempts"][0]
        self.assertIn("pre_ground_spec", attempt)
        self.assertIn("grounding", attempt)
        self.assertTrue(attempt["grounding"]["ok"])
        self.assertIn("grounded_spec", attempt)
        self.assertIn("execution_checks", attempt)
        self.assertEqual(attempt["failed_checks"], [])

    def test_sum_additive_guarded_end_to_end(self) -> None:
        # The planner omits the additive guard; grounding adds it proactively (one clean attempt),
        # excluding non-additive c3 (999) so the total is 350. The reflector's fail-then-repair
        # path is unit-tested separately in TestReflector.
        pipeline = ReasoningPipeline(backend=self.backend, planner=_UnguardedSumPlanner())
        trace = pipeline.run("What is the total value of all contracts?")
        self.assertTrue(trace.execution.passed)
        self.assertEqual(str(trace.answer_card.answer), "350.00")
        self.assertEqual(len(trace.metadata["attempts"]), 1)
        self.assertIn("added value_is_additive guard for sum", trace.metadata["grounding_changes"])

    def test_no_results_relaxed_to_answer(self) -> None:
        pipeline = ReasoningPipeline(backend=self.backend, planner=self.planner)
        # No 2024 goods contracts exist; for a SUM (empty -> no_results) the reflector drops the
        # category filter and answers over 2024 additive rows (350). A count of 0 is NOT relaxed
        # (see test_zero_count_is_answered_not_relaxed) -- 0 is a valid answer, not a no_results.
        trace = pipeline.run("What is the total value of goods contracts published in 2024?")
        self.assertTrue(trace.execution.passed)
        self.assertEqual(str(trace.answer_card.answer), "350.00")
        self.assertTrue(any("auto-repair" in lim for lim in trace.answer_card.limitations))

    def test_zero_count_is_answered_not_relaxed(self) -> None:
        pipeline = ReasoningPipeline(backend=self.backend, planner=self.planner)
        # A count over an empty match set is 0 (a valid answer), NOT a no_results to be relaxed into
        # a different question. This is what makes compare / bridge sub-counts correct.
        trace = pipeline.run("How many goods contracts were published in 2024?")
        self.assertTrue(trace.execution.passed)
        self.assertEqual(trace.answer_card.answer, 0)
        self.assertFalse(any("auto-repair" in lim for lim in trace.answer_card.limitations))

    def test_unsupported_question_not_answered(self) -> None:
        pipeline = ReasoningPipeline(backend=self.backend, planner=self.planner)
        trace = pipeline.run("Which contract had the strongest carbon reduction clause?")
        self.assertEqual(trace.reflection.action, "mark_unsupported")
        self.assertEqual(trace.answer_card.confidence_label, "not_answered")
        self.assertIsNone(trace.answer_card.answer)

    def test_ambiguous_question_asks_clarifying(self) -> None:
        pipeline = ReasoningPipeline(backend=self.backend, planner=self.planner)
        trace = pipeline.run("Tell me about procurement.")
        self.assertEqual(trace.reflection.action, "ask_clarifying_question")

    def test_semantic_candidate_repair_after_no_results(self) -> None:
        backend = TabularRuntimeBackend([
            {"contract_node_id": "c1", "tender_title": "Bridge repairs and maintenance",
             "buyer_name": "Alpha Council", "value_is_additive": True},
        ])
        retriever = LexicalTopKCandidateRetriever({"tender_title": ("Bridge repairs and maintenance",)})
        pipeline = ReasoningPipeline(
            backend=backend,
            planner=_BadTitlePlanner(),
            candidate_retriever=retriever,
            candidate_selector=TopScoreCandidateSelector(min_score=0.1),
            max_rounds=2,
        )
        trace = pipeline.run("Who bought the bridge repair contract?")
        self.assertTrue(trace.execution.passed)
        self.assertEqual(trace.answer_card.answer, "Alpha Council")
        self.assertTrue(trace.metadata["attempts"][0]["semantic_candidate_repair"]["repaired"])

    def test_llm_diagnostic_hooks_are_advisory_trace_only(self) -> None:
        pipeline = ReasoningPipeline(
            backend=self.backend,
            planner=self.planner,
            verification_analyzer=_RecordingVerificationAnalyzer(),
            reflection_analyzer=_RecordingReflectionAnalyzer(),
        )
        # Use a SUM (empty -> no_results) so the failure hooks fire; a count of 0 now passes cleanly.
        trace = pipeline.run("What is the total value of goods contracts published in 2024?")
        first = trace.metadata["attempts"][0]
        self.assertEqual(first["verifier_analysis"]["status"], "no_results")
        self.assertEqual(first["reflector_analysis"]["action"], "relax_non_answer_constraints")
        self.assertTrue(trace.execution.passed)


# --- KG adapter ----------------------------------------------------------------------

class TestKGBackendAdapter(unittest.TestCase):
    def _backend(self):
        import pandas as pd

        from procurement_graph.qa.benchmark.kg_interface import ParquetKGQueryBackend
        from procurement_graph.reasoning.kg_backend import RecordsOrgResolver, RuntimeKGBackend

        df = pd.DataFrame(
            [
                {"contract_node_id": "c1", "release_year": 2024, "tender_category": "works",
                 "buyer_name": "Alpha Council", "supplier_name": "BuildCo", "value_amount": "100", "value_is_additive": True},
                {"contract_node_id": "c2", "release_year": 2025, "tender_category": "goods",
                 "buyer_name": "Beta NHS", "supplier_name": "MedSupply", "value_amount": "200", "value_is_additive": True},
            ]
        )
        pb = ParquetKGQueryBackend(records_df=df)
        return RuntimeKGBackend(pb), RecordsOrgResolver(pb)

    def test_translates_eq_and_runs_count(self) -> None:
        backend, _ = self._backend()
        spec = RuntimeQuerySpec(spec_id="s", question="q", intent="aggregation_count",
                                constraints=(QueryConstraint("tender_category", "eq", "works"),),
                                answer_operation="count", answer_field="contract_node_id",
                                answer_value_type="integer", dedupe_key="contract_node_id")
        result = execute_query_spec(backend, spec)
        self.assertTrue(result.passed)
        self.assertEqual(result.answer, 1)

    def test_translates_between_to_gte_lte(self) -> None:
        backend, _ = self._backend()
        rows = backend.query((QueryConstraint("release_year", "between", [2024, 2025]),))
        self.assertEqual(len(rows), 2)

    def test_org_resolver_exact_and_missing(self) -> None:
        _, resolver = self._backend()
        exact = resolver.resolve("alpha council")
        self.assertEqual(exact[0].linked_label, "Alpha Council")
        self.assertEqual(resolver.resolve("Nonexistent Org"), [])


# --- LLM planner adapter -------------------------------------------------------------

class _MockChat:
    def __init__(self, parsed: Any) -> None:
        self._parsed = parsed

    def complete_json(self, *, model: str, system: str, user: str) -> Any:
        class _R:
            parsed = self._parsed
        return _R()


class TestLLMPlanner(unittest.TestCase):
    def test_valid_payload_becomes_executable_plan(self) -> None:
        from procurement_graph.reasoning.llm_planner import LLMReasoningPlanner
        from procurement_graph.reasoning.planner import PLANNER_SCHEMA_VERSION

        payload = {
            "schema_version": PLANNER_SCHEMA_VERSION,
            "plans": [{
                "intent": "aggregation_count", "answer_operation": "count",
                "answer_field": "contract_node_id", "answer_value_type": "integer",
                "constraints": [{"field": "release_year", "op": "eq", "value": 2024},
                                {"field": "tender_category", "op": "eq", "value": "works"}],
                "requires_exhaustive_retrieval": True, "confidence": 0.9,
            }],
        }
        planner = LLMReasoningPlanner(client=_MockChat(payload), model="mock")
        [plan] = planner.plan("How many works contracts in 2024?")
        self.assertEqual(plan.query_spec.answer_operation, "count")
        self.assertEqual(plan.planner_source, "llm")
        self.assertFalse(plan.fallback_used)
        self.assertEqual(plan.raw_response, payload)
        result = execute_query_spec(TabularRuntimeBackend(mock_records()), plan.query_spec)
        self.assertEqual(result.answer, 2)

    def test_malformed_payload_falls_back_to_rules(self) -> None:
        from procurement_graph.reasoning.llm_planner import LLMReasoningPlanner

        planner = LLMReasoningPlanner(client=_MockChat("not-json-dict"), model="mock")
        plans = planner.plan("How many works contracts were published in 2024?")
        self.assertTrue(plans)
        self.assertEqual(plans[0].query_spec.answer_operation, "count")  # rule-based fallback
        self.assertEqual(plans[0].planner_source, "llm")
        self.assertTrue(plans[0].fallback_used)
        self.assertEqual(plans[0].fallback_reason, "parse_error")
        self.assertEqual(plans[0].fallback_planner, "rule_based_planner_v1")


if __name__ == "__main__":
    unittest.main()
