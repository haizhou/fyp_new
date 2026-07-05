"""Tests for the runtime reasoning closed loop: linking, retrieval, reflector, evidence,
the end-to-end ReasoningPipeline, the KG adapter, and the LLM planner adapter."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from procurement_graph.reasoning import (
    CandidatePlan,
    FallbackChainPlanner,
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
from procurement_graph.reasoning.graph_planning import compile_graph_plan
from procurement_graph.reasoning.pipeline import _feedback_from_trace
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

    def test_links_year_range_as_between(self) -> None:
        link = link_question("How many services contracts were published between 2022 and 2024?")
        self.assertIn(("release_year", "between", [2022, 2024]),
                      [(c.field, c.op, c.value) for c in link.constraints])

    def test_links_multiple_bare_years_as_in_union(self) -> None:
        link = link_question("How many services contracts were published in 2022 or 2023?")
        self.assertIn(("release_year", "in", [2022, 2023]),
                      [(c.field, c.op, c.value) for c in link.constraints])

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

    def test_low_confidence_org_link_is_not_silently_constrained(self) -> None:
        class _Resolver:
            def resolve(self, mention: str) -> list[EntityLinkCandidate]:
                return [EntityLinkCandidate(mention, "canon:weak", "BuildCo Facilities Ltd",
                                            "organization", 0.42, source="records_substring")]

        link = link_question("How many contracts awarded to BuildCo in 2024?", org_resolver=_Resolver())
        self.assertFalse([c for c in link.constraints if c.field == "supplier_name"])
        self.assertTrue(any("low-confidence" in note for note in link.notes))


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

    def test_eq_conflicts_merge_to_in_union_not_first_value(self) -> None:
        spec = self._spec((QueryConstraint("release_year", "eq", 2022),
                           QueryConstraint("release_year", "eq", 2023)))
        action = reflect(self._fail("constraint_conflict", spec), CandidatePlan(plan_id="p", query_spec=spec))
        self.assertEqual(action.action, "replan_query")
        self.assertEqual([(c.field, c.op, c.value) for c in action.suggested_constraints],
                         [("release_year", "in", [2022, 2023])])

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


class _TwoPlanFallbackPlanner:
    def plan(self, question: str) -> tuple[CandidatePlan, ...]:
        bad = RuntimeQuerySpec(
            spec_id="bad-first", question=question, intent="factoid",
            constraints=(QueryConstraint("tender_category", "eq", "nonexistent"),),
            answer_operation="select_unique", answer_field="buyer_name", answer_value_type="string",
            dedupe_key="contract_node_id",
        )
        good = RuntimeQuerySpec(
            spec_id="good-second", question=question, intent="aggregation_count",
            constraints=(QueryConstraint("release_year", "eq", 2024),
                         QueryConstraint("tender_category", "eq", "works")),
            answer_operation="count", answer_field="contract_node_id", answer_value_type="integer",
            dedupe_key="contract_node_id", requires_exhaustive_retrieval=True,
        )
        return (
            CandidatePlan(plan_id="bad", query_spec=bad, status="planned", confidence=0.99, planner_source="typed_llm"),
            CandidatePlan(plan_id="good", query_spec=good, status="planned", confidence=0.40, planner_source="rule"),
        )


class _FeedbackReplanPlanner:
    def __init__(self):
        self.feedback_seen = None

    def plan(self, question: str) -> tuple[CandidatePlan, ...]:
        # buyer matches nothing but passes grounding, so the failure surfaces at the EXECUTOR
        # (an impossible tender_category value would now be rejected at grounding instead).
        bad = RuntimeQuerySpec(
            spec_id="feedback-bad", question=question, intent="factoid",
            constraints=(QueryConstraint("buyer_name", "eq", "Nonexistent Buyer Ltd"),),
            answer_operation="select_unique", answer_field="buyer_name", answer_value_type="string",
            dedupe_key="contract_node_id",
        )
        return (CandidatePlan(plan_id="feedback-bad", query_spec=bad, status="planned", confidence=0.8),)

    def replan_with_feedback(self, question: str, feedback: dict[str, Any]) -> tuple[CandidatePlan, ...]:
        self.feedback_seen = feedback
        good = RuntimeQuerySpec(
            spec_id="feedback-good", question=question, intent="aggregation_count",
            constraints=(QueryConstraint("release_year", "eq", 2024),
                         QueryConstraint("tender_category", "eq", "works")),
            answer_operation="count", answer_field="contract_node_id", answer_value_type="integer",
            dedupe_key="contract_node_id", requires_exhaustive_retrieval=True,
        )
        return (CandidatePlan(plan_id="feedback-good", query_spec=good, status="planned", confidence=0.7),)


class _PreExecutionReviewPlanner:
    def __init__(self, *, repair_plans: bool = True):
        self.feedback_seen = None
        self.repair_plans = repair_plans

    def _works_count(self, plan_id: str) -> CandidatePlan:
        spec = RuntimeQuerySpec(
            spec_id=plan_id, question="q", intent="aggregation_count",
            constraints=(QueryConstraint("release_year", "eq", 2024),
                         QueryConstraint("tender_category", "eq", "works")),
            answer_operation="count", answer_field="contract_node_id", answer_value_type="integer",
            dedupe_key="contract_node_id", requires_exhaustive_retrieval=True,
        )
        return CandidatePlan(
            plan_id=plan_id,
            query_spec=spec,
            status="planned",
            confidence=0.7,
            raw_response={"plan_review": {"verdict": "mismatch", "reason": "review thinks the type is wrong"}},
        )

    def plan(self, question: str) -> tuple[CandidatePlan, ...]:
        if self.repair_plans:
            bad = RuntimeQuerySpec(
                spec_id="pre-review-bad", question=question, intent="factoid",
                constraints=(QueryConstraint("buyer_name", "eq", "Nonexistent Buyer Ltd"),),
                answer_operation="select_unique", answer_field="buyer_name", answer_value_type="string",
                dedupe_key="contract_node_id",
            )
            return (CandidatePlan(
                plan_id="pre-review-bad",
                query_spec=bad,
                status="planned",
                confidence=0.8,
                raw_response={"plan_review": {"verdict": "mismatch", "reason": "wrong entity role"}},
            ),)
        return (self._works_count("pre-review-original"),)

    def replan_with_feedback(self, question: str, feedback: dict[str, Any]) -> tuple[CandidatePlan, ...]:
        self.feedback_seen = feedback
        if self.repair_plans:
            return (self._works_count("pre-review-good"),)
        not_planned = self._works_count("pre-review-invalid")
        return (replace(not_planned, status="ambiguous", rationale="repair could not produce a plan"),)


class _TwoRoundFeedbackPlanner:
    """Initial plan fails, repair 1 fails again, repair 2 verifies — exercises the attempt loop."""

    def __init__(self):
        self.feedbacks: list[dict[str, Any]] = []

    def _org_probe(self, plan_id: str, buyer: str) -> CandidatePlan:
        spec = RuntimeQuerySpec(
            spec_id=plan_id, question="q", intent="factoid",
            constraints=(QueryConstraint("buyer_name", "eq", buyer),),
            answer_operation="select_unique", answer_field="buyer_name", answer_value_type="string",
            dedupe_key="contract_node_id",
        )
        return CandidatePlan(plan_id=plan_id, query_spec=spec, status="planned", confidence=0.8)

    def plan(self, question: str) -> tuple[CandidatePlan, ...]:
        return (self._org_probe("fr-bad0", "Nonexistent Buyer Ltd"),)

    def replan_with_feedback(self, question: str, feedback: dict[str, Any]) -> tuple[CandidatePlan, ...]:
        self.feedbacks.append(feedback)
        if len(self.feedbacks) == 1:
            return (self._org_probe("fr-bad1", "Still Missing Ltd"),)
        good = RuntimeQuerySpec(
            spec_id="fr-good", question=question, intent="aggregation_count",
            constraints=(QueryConstraint("release_year", "eq", 2024),
                         QueryConstraint("tender_category", "eq", "works")),
            answer_operation="count", answer_field="contract_node_id", answer_value_type="integer",
            dedupe_key="contract_node_id", requires_exhaustive_retrieval=True,
        )
        return (CandidatePlan(plan_id="fr-good", query_spec=good, status="planned", confidence=0.7),)


class _OnePlan:
    def __init__(self, plan_id: str, confidence: float):
        self.plan_id = plan_id
        self.confidence = confidence
        self.planner_version = plan_id

    def plan(self, question: str) -> tuple[CandidatePlan, ...]:
        spec = RuntimeQuerySpec(
            spec_id=self.plan_id, question=question, intent="factoid", constraints=(),
            answer_operation="count", answer_field="contract_node_id", answer_value_type="integer",
        )
        return (CandidatePlan(plan_id=self.plan_id, query_spec=spec, status="planned",
                              confidence=self.confidence, planner_source=self.plan_id),)


class _RecordingVerificationAnalyzer:
    def analyze_verification(self, **kwargs: Any) -> dict[str, Any]:
        return {"diagnosis": "recorded", "status": kwargs["result"].status}


class _RecordingReflectionAnalyzer:
    def analyze_reflection(self, **kwargs: Any) -> dict[str, Any]:
        return {"assessment": "recorded", "action": kwargs["action"].action}


class _RecordingTraceReflector:
    def __init__(self):
        self.logged = None

    def reflect_trace(self, trace):
        from procurement_graph.reasoning.trace_reflector import TraceReflection
        return TraceReflection(faithfulness="faithful", faithfulness_checks=(),
                               plan_valid=True, plan_issues=(), action="no_action", reason="ok")

    def log_preference(self, trace, reflection, *, oracle_match=None):
        self.logged = oracle_match
        return {"oracle_match": oracle_match}


class _GraphPlanner:
    def plan(self, question: str) -> tuple[CandidatePlan, ...]:
        payload = {
            "understanding_network": {
                "answer": {"description": "count contracts in C", "answer_kind": "count",
                           "operation": "count", "depends_on": ["C"]},
                "variables": {
                    "A": {"description": "buyer Alpha Council", "kind": "entity",
                          "role": "buyer_or_authority", "known_surface": "Alpha Council",
                          "depends_on": []},
                    "B": {"description": "suppliers that worked with A", "kind": "entity_set",
                          "role": "supplier", "depends_on": ["A"]},
                    "C": {"description": "works contracts in 2024 awarded to suppliers in B",
                          "kind": "record_set", "role": "contract_records", "depends_on": ["B"]},
                },
                "goal_tree": {"id": "answer", "description": "count C", "requires": [
                    {"id": "C", "description": "find C", "requires": [
                        {"id": "B", "description": "find B", "requires": [
                            {"id": "A", "description": "resolve A", "requires": []}
                        ]}
                    ]}
                ]},
                "given_facts": [{"kind": "entity", "surface": "Alpha Council", "assigned_to": "A"},
                                {"kind": "year", "surface": "2024", "assigned_to": "C"},
                                {"kind": "category", "surface": "works", "assigned_to": "C"}],
                "procedure": [
                    {"step_id": "u1", "operation_unit": "resolve_entity",
                     "inputs": ["Alpha Council"], "output": "A",
                     "description": "resolve the buyer/entity mentioned as Alpha Council"},
                    {"step_id": "u2", "operation_unit": "find_entity_set",
                     "inputs": ["A"], "output": "B",
                     "description": "find suppliers that worked with A"},
                    {"step_id": "u3", "operation_unit": "find_record_set",
                     "inputs": ["B", "2024", "works"], "output": "C",
                     "description": "find works contracts in 2024 awarded to B"},
                    {"step_id": "u4", "operation_unit": "aggregate_count",
                     "inputs": ["C"], "output": "answer",
                     "description": "count records in C"},
                ],
                "reasoning_order": ["A", "B", "C", "answer"],
            },
            "graph_plan": {
                "variables": {
                    "A": {"kind": "entity", "description": "buyer Alpha Council",
                          "role": "buyer_or_authority",
                          "type_candidates": [{"type": "buyer", "confidence": 0.9}],
                          "grounding": {"surface": "Alpha Council"}, "depends_on": []},
                    "B": {"kind": "entity_set", "description": "suppliers that worked with A",
                          "role": "supplier",
                          "type_candidates": [{"type": "supplier", "confidence": 0.9}],
                          "depends_on": ["A"]},
                    "C": {"kind": "record_set",
                          "description": "works contracts in 2024 awarded to suppliers in B",
                          "type_candidates": [{"type": "contract", "confidence": 0.9}],
                          "filters": [
                              {"slot": "year", "surface": "2024", "value": 2024},
                              {"slot": "category", "surface": "works", "value": "works"},
                          ],
                          "depends_on": ["B"]},
                },
                "relations": [
                    {"from": "A", "to": "B",
                     "relation_candidates": [{"relation": "buyer_to_supplier", "confidence": 0.8}],
                     "direction": "outgoing", "surface_evidence": "worked with"},
                    {"from": "B", "to": "C",
                     "relation_candidates": [{"relation": "supplier_of_contract", "confidence": 0.8}],
                     "direction": "incoming_or_outgoing", "surface_evidence": "awarded to suppliers"},
                ],
                "operation_units": [
                    {"step_id": "u1", "operation_unit": "resolve_entity",
                     "inputs": ["Alpha Council"], "output": "A",
                     "uses": {"filters": ["Alpha Council"], "relations": []},
                     "description": "resolve the buyer/entity mentioned as Alpha Council"},
                    {"step_id": "u2", "operation_unit": "find_entity_set",
                     "inputs": ["A"], "output": "B",
                     "uses": {"filters": [], "relations": ["buyer_to_supplier"]},
                     "description": "find suppliers that worked with A"},
                    {"step_id": "u3", "operation_unit": "find_record_set",
                     "inputs": ["B", "2024", "works"], "output": "C",
                     "uses": {"filters": ["2024", "works"], "relations": ["supplier_of_contract"]},
                     "description": "find works contracts in 2024 awarded to B"},
                    {"step_id": "u4", "operation_unit": "aggregate_count",
                     "inputs": ["C"], "output": "answer",
                     "uses": {"filters": [], "relations": []},
                     "description": "count records in C"},
                ],
                "return": {"operation": "count", "input": "C", "field": "contracts", "guards": []},
            },
        }
        graph, reason = compile_graph_plan(question, payload)
        if reason:
            raise AssertionError(reason)
        spec = RuntimeQuerySpec("graph-placeholder", question, "role_path", (), "count", "", "integer",
                                dedupe_key="contract_node_id")
        return (CandidatePlan("graph:p0", spec, status="planned", confidence=0.7,
                              planner_source="graph_test", raw_response=payload, graph_plan=graph),)


class _GraphPayloadPlanner:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload

    def plan(self, question: str) -> tuple[CandidatePlan, ...]:
        graph, reason = compile_graph_plan(question, self.payload)
        if reason:
            raise AssertionError(reason)
        spec = RuntimeQuerySpec("graph-placeholder", question, "role_path", (), "count", "", "integer",
                                dedupe_key="contract_node_id")
        return (CandidatePlan("graph:p0", spec, status="planned", confidence=0.7,
                              planner_source="graph_test", raw_response=self.payload, graph_plan=graph),)


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
        guards = trace.metadata["attempts"][0]["deterministic_guards_added"]
        self.assertEqual(guards[0]["field"], "value_is_additive")
        self.assertEqual(guards[0]["reason"], "added value_is_additive guard for sum")

    def test_graph_plan_executes_variables_in_reasoning_order(self) -> None:
        pipeline = ReasoningPipeline(backend=self.backend, planner=_GraphPlanner())
        trace = pipeline.run("How many works contracts in 2024 went to suppliers that worked with Alpha Council?")
        self.assertTrue(trace.execution.passed)
        self.assertEqual(trace.answer_card.answer, 2)
        graph = trace.metadata["graph"]
        self.assertEqual([v["var_id"] for v in graph["variables"]], ["A", "B", "C"])
        self.assertEqual(graph["variables"][1]["emitted_field"], "supplier_name")
        self.assertEqual(graph["variables"][1]["operation_unit"], "find_entity_set")
        self.assertEqual(graph["operation_units"][2]["output"], "C")
        self.assertEqual(trace.metadata["attempts"][0]["graph_execution"]["status"], "passed")

    def test_graph_rank_top_k_operation_unit(self) -> None:
        payload = {
            "understanding_network": {"reasoning_order": ["C", "answer"]},
            "graph_plan": {
                "variables": {
                    "C": {"kind": "record_set", "description": "2024 works contracts",
                          "filters": [{"slot": "year", "surface": "2024", "value": 2024},
                                      {"slot": "category", "surface": "works", "value": "works"}],
                          "depends_on": []}
                },
                "operation_units": [
                    {"step_id": "u1", "operation_unit": "find_record_set",
                     "inputs": ["2024", "works"], "output": "C"},
                    {"step_id": "u2", "operation_unit": "rank_top_k",
                     "inputs": ["C"], "output": "answer"}
                ],
                "return": {"operation": "top_k", "input": "C", "field": "contracts",
                           "group_by": "buyer", "metric": "count", "k": 1, "guards": []},
            },
        }
        trace = ReasoningPipeline(backend=self.backend, planner=_GraphPayloadPlanner(payload)).run(
            "Which buyer has the most 2024 works contracts?"
        )
        self.assertTrue(trace.execution.passed)
        self.assertEqual(trace.answer_card.answer, [["Alpha Council", 2]])
        self.assertEqual(trace.metadata["graph"]["operation_units"][1]["operation_unit"], "rank_top_k")

    def test_graph_overdecomposed_conjunctive_filter_intersects(self) -> None:
        # Grok over-decomposes "works AND 2024" into two chained record_set variables. Binding on
        # the source emit_field (contract_node_id) must intersect them -> 2, not bind on
        # supplier_name and return no_results.
        payload = {
            "understanding_network": {"reasoning_order": ["A", "B", "answer"]},
            "graph_plan": {
                "variables": {
                    "A": {"kind": "record_set", "description": "works contracts",
                          "filters": [{"slot": "category", "surface": "works", "value": "works"}],
                          "depends_on": []},
                    "B": {"kind": "record_set", "description": "works contracts in 2024",
                          "filters": [{"slot": "year", "surface": "2024", "value": 2024}],
                          "depends_on": ["A"]},
                },
                "relations": [{"from": "A", "to": "B",
                               "relation_candidates": [{"relation": "contract has release date"}],
                               "direction": "outgoing", "surface_evidence": "in 2024"}],
                "operation_units": [
                    {"step_id": "u1", "operation_unit": "find_record_set", "inputs": [], "output": "A"},
                    {"step_id": "u2", "operation_unit": "filter_records", "inputs": ["A"], "output": "B"},
                    {"step_id": "u3", "operation_unit": "aggregate_count", "inputs": ["B"], "output": "answer"},
                ],
                "return": {"operation": "count", "input": "B", "field": "contracts", "guards": []},
            },
        }
        trace = ReasoningPipeline(backend=self.backend, planner=_GraphPayloadPlanner(payload)).run(
            "How many works contracts were published in 2024?")
        self.assertTrue(trace.execution.passed)
        self.assertEqual(trace.answer_card.answer, 2)

    def test_graph_plan_variables_as_array_compiles(self) -> None:
        # strict json_schema returns variables as an ARRAY (arbitrary-key maps are unsupported);
        # it must compile to the same executable plan as the legacy dict form.
        payload = {
            "graph_plan": {
                "question_type": "count", "operation": "count",
                "variables": [
                    {"var_id": "A", "kind": "record_set", "role": "contract_records",
                     "filters": [{"slot": "year", "value": "2024", "operator": "eq"},
                                 {"slot": "category", "value": "works", "operator": "eq"}],
                     "depends_on": []},
                ],
                "return": {"operation": "count", "input": "A", "field": "contract"},
            },
        }
        trace = ReasoningPipeline(backend=self.backend, planner=_GraphPayloadPlanner(payload)).run(
            "How many works contracts were published in 2024?")
        self.assertTrue(trace.execution.passed)
        self.assertEqual(trace.answer_card.answer, 2)

    def test_graph_tree_coded_var_ids_derive_dependencies(self) -> None:
        from procurement_graph.reasoning.graph_planning import compile_graph_plan
        payload = {
            "graph_plan": {
                "question_type": "count", "operation": "count",
                "variables": [
                    {"var_id": "a1", "kind": "record_set", "role": "contract_records",
                     "filters": [{"slot": "year", "value": "2025", "operator": "eq"}],
                     "depends_on": []},
                    {"var_id": "b1a1", "kind": "entity_set", "role": "supplier",
                     "filters": [{"slot": "year", "value": "2024", "operator": "eq"},
                                 {"slot": "category", "value": "services", "operator": "eq"}],
                     "depends_on": []},
                ],
                "return": {"operation": "count", "input": "a1", "field": "contract"},
            },
        }
        graph, reason = compile_graph_plan(
            "How many 2025 contracts went to suppliers who won services contracts in 2024?",
            payload,
        )
        self.assertEqual(reason, "")
        deps = {var.var_id: var.depends_on for var in graph.variables}
        self.assertEqual(deps["a1"], ("b1a1",))
        self.assertEqual(deps["b1a1"], ())
        self.assertEqual(graph.reasoning_order, ("b1a1", "a1", "answer"))

    def test_graph_normalise_var_reference_filter_becomes_bind(self) -> None:
        # T1/T2: a filter whose value NAMES another variable is a dependency edge; an empty org
        # filter on a dependent variable is dropped (the bind carries the role). Observed L2 idiom:
        # filtered_notices had supplier='' plus a filter value 'winners_suppliers'.
        from procurement_graph.reasoning.graph_planning import normalise_graph_plan
        graph = {
            "variables": [
                {"var_id": "records_a", "kind": "record_set", "role": "contract_records",
                 "filters": [{"slot": "year", "value": 2024, "operator": "eq"},
                             {"slot": "category", "value": "works", "operator": "eq"}], "depends_on": []},
                {"var_id": "winners", "kind": "entity_set", "role": "supplier",
                 "filters": [{"slot": "category", "value": "works", "operator": "eq"}], "depends_on": []},
                {"var_id": "filtered", "kind": "record_set", "role": "contract_records",
                 "filters": [{"slot": "supplier", "value": "", "operator": "eq"},
                             {"slot": "supplier", "value": "winners", "operator": "eq"}],
                 "depends_on": ["records_a"]},
            ],
            "return": {"operation": "count", "input": "filtered"},
        }
        cleaned = normalise_graph_plan("How many notices went to winning suppliers?", graph)
        by_id = {v["var_id"]: v for v in cleaned["variables"]}
        self.assertEqual(by_id["filtered"]["filters"], [])
        self.assertIn("winners", by_id["filtered"]["depends_on"])
        self.assertIn("records_a", by_id["filtered"]["depends_on"])

    def test_graph_compile_provenance_naming_does_not_cycle(self) -> None:
        # The 0416 bridge: grok names the derived CPV set "b1a1" (provenance: FROM a1) and states
        # the real dataflow in depends_on. The naming convention alone would add the reverse edge
        # (a1 consumes b1a1) and compile a 2-cycle; the explicit edge must win.
        from procurement_graph.reasoning.graph_planning import _execution_levels, compile_graph_plan
        payload = {
            "variables": [
                {"var_id": "a1", "kind": "record_set", "role": "contract_records",
                 "filters": [{"slot": "buyer", "value": "UNIVERSITY OF MANCHESTER", "operator": "eq"}],
                 "depends_on": []},
                {"var_id": "b1a1", "kind": "entity_set", "role": "cpv", "filters": [],
                 "depends_on": ["a1"]},
                {"var_id": "a2", "kind": "record_set", "role": "contract_records",
                 "filters": [{"slot": "cpv", "value": "b1a1", "operator": "in"}],
                 "depends_on": ["b1a1"]},
            ],
            "return": {"operation": "count", "input": "a2"},
        }
        plan, reason = compile_graph_plan("How many notices use CPV codes the university has used?", payload)
        self.assertEqual(reason, "")
        levels, level_reason = _execution_levels(plan)
        self.assertEqual(level_reason, "")
        order = [vid for level in levels for vid in level]
        self.assertLess(order.index("a1"), order.index("b1a1"))
        self.assertLess(order.index("b1a1"), order.index("a2"))

    def test_graph_normalise_drops_anchor_echo_literal(self) -> None:
        # T9 (nano 1761): "suppliers who worked with HIE" — the planner binds the derived supplier
        # set into a1 AND echoes HIE as a literal supplier filter; the intersection is empty (HIE
        # is the buyer). The bind carries the meaning; the echoed literal must be dropped.
        from procurement_graph.reasoning.graph_planning import normalise_graph_plan
        graph = {
            "variables": [
                {"var_id": "a1", "kind": "record_set", "role": "contract_records",
                 "filters": [{"slot": "supplier", "value": "Highlands and Islands Enterprise",
                              "operator": "eq"}],
                 "depends_on": []},
                {"var_id": "b1", "kind": "entity_set", "role": "supplier",
                 "filters": [{"slot": "buyer", "value": "Highlands and Islands Enterprise",
                              "operator": "eq"}],
                 "depends_on": []},
            ],
            "relations": [{"from": "b1", "to": "a1", "bind_field": "supplier"}],
            "return": {"operation": "sum", "input": "a1", "field": "value"},
        }
        cleaned = normalise_graph_plan(
            "What total value went to suppliers who have also worked with Highlands and Islands Enterprise?",
            graph)
        by_id = {v["var_id"]: v for v in cleaned["variables"]}
        self.assertEqual(by_id["a1"]["filters"], [])          # echoed anchor dropped
        self.assertEqual(len(by_id["b1"]["filters"]), 1)      # the real anchor stays on the source

    def test_clean_no_results_is_not_replanned(self) -> None:
        # Every literal in the plan comes from the question and execution was clean-empty: the
        # emptiness IS the answer. The repair loop must not answer-shop (measured: it turned 6
        # correct abstentions into hallucinations).
        from procurement_graph.reasoning.models import CandidatePlan as _CP
        from procurement_graph.reasoning.models import QueryConstraint as _QC
        from procurement_graph.reasoning.models import RuntimeQuerySpec as _Spec

        calls: list[dict[str, Any]] = []

        class _CleanNoResultsPlanner:
            def plan(self, question):
                spec = _Spec(spec_id="x", question=question, intent="count",
                             constraints=(_QC("buyer_name", "eq", "Alpha Council"),
                                          _QC("release_year", "eq", 2031)),
                             answer_operation="select_unique", answer_field="supplier_name",
                             answer_value_type="string", requires_exhaustive_retrieval=True)
                return (_CP(plan_id="p0", query_spec=spec, status="planned",
                            rationale="typed plan", planner_source="typed_llm"),)

            def replan_with_feedback(self, question, feedback):
                calls.append(feedback)
                return ()

        pipeline = ReasoningPipeline(backend=TabularRuntimeBackend(mock_records()),
                                     planner=_CleanNoResultsPlanner(), max_feedback_replans=1)
        trace = pipeline.run("Which supplier won the Alpha Council contract in 2031?")
        self.assertEqual(calls, [])
        self.assertEqual(trace.metadata.get("feedback_replan", {}).get("skipped"),
                         "clean_no_results_is_an_answer")

    def test_no_results_with_invented_literal_is_replanned(self) -> None:
        # The planner invented year 0 (the question says non-zero): a diagnosable defect, so the
        # repair loop SHOULD fire.
        from procurement_graph.reasoning.models import CandidatePlan as _CP
        from procurement_graph.reasoning.models import QueryConstraint as _QC
        from procurement_graph.reasoning.models import RuntimeQuerySpec as _Spec

        calls: list[dict[str, Any]] = []

        class _InventedLiteralPlanner:
            def plan(self, question):
                spec = _Spec(spec_id="x", question=question, intent="count",
                             constraints=(_QC("buyer_name", "eq", "Alpha Council"),
                                          _QC("release_year", "eq", 1999)),
                             answer_operation="select_unique", answer_field="supplier_name",
                             answer_value_type="string", requires_exhaustive_retrieval=True)
                return (_CP(plan_id="p0", query_spec=spec, status="planned",
                            rationale="typed plan", planner_source="typed_llm"),)

            def replan_with_feedback(self, question, feedback):
                calls.append(feedback)
                return ()

        pipeline = ReasoningPipeline(backend=TabularRuntimeBackend(mock_records()),
                                     planner=_InventedLiteralPlanner(), max_feedback_replans=1)
        pipeline.run("Which supplier won the Alpha Council contract?")
        # the invented year is repairable: either the relax path or the replan gate fires
        self.assertEqual(len(calls), 1)

    def test_feedback_from_compile_failure_carries_failed_plan(self) -> None:
        # A plan rejected at compile/consistency never executes; the repair feedback must still
        # carry the failed plan and a plan_compile stage, not failed_plan=None + stage "verifier".
        from procurement_graph.reasoning.models import CandidatePlan as _CP
        from procurement_graph.reasoning.models import RuntimeQuerySpec as _Spec

        captured: dict[str, Any] = {}

        class _RecordingPlanner:
            def plan(self, question):
                spec = _Spec(spec_id="x", question=question, intent="count", constraints=(),
                             answer_operation="count", answer_field="contract_node_id",
                             answer_value_type="count", requires_exhaustive_retrieval=True)
                return (_CP(plan_id="p0", query_spec=spec, status="ambiguous",
                            rationale="invalid_graph_plan:invalid_graph_structure:cycle among: a1,b1a1",
                            planner_source="typed_llm"),)

            def replan_with_feedback(self, question, feedback):
                captured.update(feedback)
                return ()

        pipeline = ReasoningPipeline(backend=TabularRuntimeBackend(mock_records()),
                                     planner=_RecordingPlanner(), max_feedback_replans=1)
        pipeline.run("How many notices share a CPV with the university's notices?")
        self.assertEqual(captured.get("failure_stage"), "plan_compile")
        self.assertIn("invalid_graph_structure", str(captured.get("failure_reason")))
        self.assertIsNotNone(captured.get("failed_plan"))
        self.assertEqual(captured["failed_plan"]["plan_id"], "p0")

    def test_graph_compile_relation_edge_becomes_dependency(self) -> None:
        # A relation "from b1 to a1" is dataflow: a1 consumes b1's output. Binds only apply per
        # depends_on entry, so without this merge nano's relation-style bridges never bound and
        # executed unfiltered (1761 summed the whole KG scope instead of the derived suppliers).
        from procurement_graph.reasoning.graph_planning import compile_graph_plan
        payload = {
            "variables": [
                {"var_id": "a1", "kind": "record_set", "role": "contract_records",
                 "filters": [], "depends_on": []},
                {"var_id": "b1", "kind": "entity_set", "role": "supplier",
                 "filters": [{"slot": "buyer", "value": "Highlands and Islands Enterprise",
                              "operator": "eq"}],
                 "depends_on": []},
            ],
            "relations": [{"from": "b1", "to": "a1", "bind_field": "supplier"}],
            "return": {"operation": "sum", "input": "a1", "field": "value"},
        }
        plan, reason = compile_graph_plan("Total value to suppliers who worked with HIE?", payload)
        self.assertEqual(reason, "")
        deps = {var.var_id: var.depends_on for var in plan.variables}
        self.assertIn("b1", deps["a1"])

    def test_graph_normalise_singular_question_rewrites_set_to_select(self) -> None:
        # T10 (1887 vs 2329): "which buyer ..." promises ONE entity; distinct_set would launder
        # uniqueness (a 30-item hallucinated "answer"). Rewriting to select lets run-time
        # uniqueness adjudicate: one match answers (2329 stays alive), many -> multiple_answers.
        from procurement_graph.reasoning.graph_planning import normalise_graph_plan
        graph = {
            "variables": [
                {"var_id": "a1", "kind": "record_set", "role": "contract_records",
                 "filters": [{"slot": "supplier", "value": "Access UK Ltd", "operator": "eq"}],
                 "depends_on": []},
                {"var_id": "a2", "kind": "entity_set", "role": "buyer", "filters": [],
                 "depends_on": ["a1"]},
            ],
            "return": {"operation": "distinct_set", "input": "a2", "field": "buyer"},
        }
        singular = normalise_graph_plan(
            "In the procurement records, which buyer awarded a contract to Access UK Ltd?", graph)
        self.assertEqual(singular["return"]["operation"], "select")
        plural = normalise_graph_plan(
            "Which buyers awarded contracts to Access UK Ltd?",
            {**graph, "return": dict(graph["return"])})
        self.assertEqual(plural["return"]["operation"], "distinct_set")

    def test_graph_compare_metric_sum_overrides_missing_field(self) -> None:
        # 0050: return said metric=sum but field=none; participants were COUNTED against a
        # 10M threshold. return.metric is authoritative for participant aggregation.
        from procurement_graph.reasoning.graph_planning import compile_graph_plan, execute_graph_plan
        payload = {
            "variables": [
                {"var_id": "a1", "kind": "record_set", "role": "contract_records",
                 "filters": [{"slot": "year", "value": 2024, "operator": "eq"}], "depends_on": []},
            ],
            "return": {"operation": "compare", "input": "a1", "field": "none",
                       "metric": "sum", "comparator": "gt", "left": "a1", "right": "GBP 200"},
        }
        plan, reason = compile_graph_plan("Was the 2024 total value more than GBP 200?", payload)
        self.assertEqual(reason, "")
        res = execute_graph_plan(TabularRuntimeBackend(mock_records()), plan)
        self.assertEqual(res.status, "passed")
        self.assertTrue(res.answer["answer"])  # 2024 additive sum 350 > 200

    def test_graph_compare_date_literal_side(self) -> None:
        # 0944: right side was a scalar/date var executed as a record query (counted 359 rows
        # signed that day). T12 folds it to the ISO literal; the left side fetches the record's
        # signed date; comparison is by date, not by count.
        from procurement_graph.reasoning.graph_planning import compile_graph_plan, execute_graph_plan
        backend = TabularRuntimeBackend(mock_records() + [{
            "contract_node_id": "cd", "ocid": "ocds-d", "buyer_name": "Alpha Council",
            "supplier_name": "DateCo", "release_year": 2024, "tender_category": "works",
            "tender_title": "Unique Dated Contract", "value_amount": "10",
            "value_is_additive": True, "award_date_signed": "2022-07-27T00:00:00+01:00",
        }])
        payload = {
            "variables": [
                {"var_id": "a1", "kind": "record_set", "role": "contract_records",
                 "filters": [{"slot": "title", "value": "Unique Dated Contract", "operator": "eq"}],
                 "depends_on": []},
                {"var_id": "a2", "kind": "scalar", "role": "date",
                 "filters": [{"slot": "date", "value": "1 May 2025", "operator": "eq"}],
                 "depends_on": []},
            ],
            "return": {"operation": "compare", "input": "a1", "field": "date",
                       "comparator": "gt", "left": "a1", "right": "a2"},
        }
        plan, reason = compile_graph_plan(
            'Was "Unique Dated Contract" signed after 1 May 2025?', payload)
        self.assertEqual(reason, "")
        res = execute_graph_plan(backend, plan)
        self.assertEqual(res.status, "passed", res.reason)
        self.assertFalse(res.answer["answer"])  # 2022-07-27 is not after 2025-05-01

    def test_graph_compile_second_chance_drops_naming_edges_on_cycle(self) -> None:
        # A sample whose naming edges are irreducibly self-contradictory must retry compile with
        # explicit/relation edges only, not die (the 0416 optlean variant).
        from procurement_graph.reasoning.graph_planning import compile_graph_plan, execute_graph_plan
        payload = {
            "variables": [
                {"var_id": "a1", "kind": "record_set", "role": "contract_records",
                 "filters": [{"slot": "buyer", "value": "Alpha Council", "operator": "eq"}],
                 "depends_on": []},
                # naming says b2a1 feeds a1 AND c1b2a1 feeds b2..., while explicit edges say the
                # reverse for one pair -> only dropping naming edges yields a DAG
                {"var_id": "b2a1", "kind": "entity_set", "role": "supplier", "filters": [],
                 "depends_on": ["a1"]},
                {"var_id": "a3", "kind": "record_set", "role": "contract_records",
                 "filters": [{"slot": "supplier", "value": "b2a1", "operator": "in"}],
                 "depends_on": ["b2a1", "a1"]},
            ],
            "return": {"operation": "count", "input": "a3"},
        }
        plan, reason = compile_graph_plan("How many notices from Alpha Council suppliers?", payload)
        self.assertIsNotNone(plan, reason)

    def test_org_variant_equivalence_resolves_to_in_filter(self) -> None:
        # BEIS: "The X" and "X (BEIS)" are decoration variants of ONE org whose rows are split
        # across surfaces; resolution yields an IN over all variants instead of abstaining.
        from procurement_graph.reasoning.entity_resolution import resolve_confident_org
        from procurement_graph.reasoning.models import EntityLinkCandidate

        class _TwoVariantResolver:
            def resolve(self, mention):
                return [
                    EntityLinkCandidate(mention=mention,
                                        linked_id="The Department for Business, Energy and Industrial Strategy",
                                        linked_label="The Department for Business, Energy and Industrial Strategy",
                                        entity_type="organisation", score=0.9, source="records_substring"),
                    EntityLinkCandidate(mention=mention,
                                        linked_id="Department for Business, Energy and Industrial Strategy (BEIS)",
                                        linked_label="Department for Business, Energy and Industrial Strategy (BEIS)",
                                        entity_type="organisation", score=0.88, source="records_substring"),
                ]

        res = resolve_confident_org(_TwoVariantResolver(), "Department for Business, Energy and Industrial Strategy")
        self.assertTrue(res.ok, res.reason)
        self.assertEqual(len(res.variants), 2)

    def test_graph_compile_rejects_unconstrained_bind_source(self) -> None:
        # unanswerable_1161: grok bound the records to an entity_set with NO filters and NO
        # dependencies -- the whole supplier universe -- silently replacing the question's own
        # constraints and hallucinating a buyer list. A consumed set must be defined by something.
        from procurement_graph.reasoning.graph_planning import compile_graph_plan
        payload = {
            "variables": [
                {"var_id": "b2", "kind": "entity_set", "role": "supplier", "filters": [],
                 "depends_on": []},
                {"var_id": "a1", "kind": "record_set", "role": "contract_records", "filters": [],
                 "depends_on": ["b2"]},
                {"var_id": "a2", "kind": "entity_set", "role": "buyer", "filters": [],
                 "depends_on": ["a1"]},
            ],
            "return": {"operation": "distinct_set", "input": "a2", "field": "buyer"},
        }
        plan, reason = compile_graph_plan("Who was the buyer for School Run Ltd's 2031 goods contract?", payload)
        self.assertIsNone(plan)
        self.assertIn("unconstrained_bind_source:b2", reason)

    def test_graph_compile_rejects_true_dependency_cycle(self) -> None:
        # DAG validity is an intent-program property: an irreducible cycle (explicit edges in both
        # directions, no naming heuristic to drop) must fail at COMPILE with structured feedback,
        # not surface later as an opaque executor error.
        from procurement_graph.reasoning.graph_planning import compile_graph_plan
        payload = {
            "variables": [
                {"var_id": "x1", "kind": "record_set", "role": "contract_records",
                 "filters": [{"slot": "year", "value": 2024, "operator": "eq"}],
                 "depends_on": ["y1"]},
                {"var_id": "y1", "kind": "record_set", "role": "contract_records",
                 "filters": [{"slot": "category", "value": "works", "operator": "eq"}],
                 "depends_on": ["x1"]},
            ],
            "return": {"operation": "count", "input": "x1"},
        }
        plan, reason = compile_graph_plan("How many 2024 works notices?", payload)
        self.assertIsNone(plan)
        self.assertIn("invalid_graph_structure", reason)

    def test_graph_normalise_entity_with_dependency_becomes_entity_set(self) -> None:
        # T8 (0079): kind=entity only grounds a literal value; "the buyer(s) of those records"
        # has a dependency and no literal, so it must run as a derived entity_set — unless it is
        # itself the returned answer, where singularity is the point.
        from procurement_graph.reasoning.graph_planning import normalise_graph_plan
        graph = {
            "variables": [
                {"var_id": "a1", "kind": "record_set", "role": "contract_records",
                 "filters": [{"slot": "supplier", "value": "SciMed Ltd", "operator": "eq"}],
                 "depends_on": []},
                {"var_id": "b1a1", "kind": "entity", "role": "buyer", "filters": [],
                 "depends_on": ["a1"]},
                {"var_id": "a2", "kind": "record_set", "role": "contract_records", "filters": [],
                 "depends_on": ["b1a1"]},
            ],
            "return": {"operation": "count", "input": "a2"},
        }
        cleaned = normalise_graph_plan("How many notices did SciMed's buyers publish?", graph)
        by_id = {v["var_id"]: v for v in cleaned["variables"]}
        self.assertEqual(by_id["b1a1"]["kind"], "entity_set")

    def test_graph_normalise_drops_unquoted_title_wrapper(self) -> None:
        # 1761: grok turned the scope phrase "matching procurement records" into a title filter,
        # which can only zero the match set. Same rule as the typed path: only a quoted span of
        # the question is a real title value.
        from procurement_graph.reasoning.graph_planning import normalise_graph_plan
        graph = {
            "variables": [
                {"var_id": "a1", "kind": "record_set", "role": "contract_records",
                 "filters": [{"slot": "title", "value": "matching procurement records", "operator": "in"},
                             {"slot": "title", "value": "Railhead Adhesion System", "operator": "eq"}],
                 "depends_on": []},
            ],
            "return": {"operation": "sum", "input": "a1", "field": "value"},
        }
        cleaned = normalise_graph_plan(
            'Looking only at the matching procurement records, what is the total for "Railhead Adhesion System"?',
            graph)
        filters = cleaned["variables"][0]["filters"]
        self.assertEqual([item["value"] for item in filters], ["Railhead Adhesion System"])

    def test_grounding_normalises_written_award_date(self) -> None:
        # 0944: '1 May 2025' in a date slot must ground to 2025-05-01, not fail the constraint.
        from procurement_graph.reasoning.grounding import ground_spec
        from procurement_graph.reasoning.models import QueryConstraint as _QC
        from procurement_graph.reasoning.models import RuntimeQuerySpec as _Spec
        spec = _Spec(spec_id="d", question="signed after 1 May 2025?", intent="boolean",
                     constraints=(_QC("award_date_signed", "eq", "1 May 2025"),),
                     answer_operation="exists", answer_field="contract_node_id",
                     answer_value_type="boolean", requires_exhaustive_retrieval=True)
        grounding = ground_spec(spec)
        self.assertTrue(grounding.ok, grounding.reason)
        value = next(c.value for c in grounding.spec.constraints if c.field == "award_date_signed")
        self.assertEqual(value, "2025-05-01")

    def test_graph_normalise_folds_same_role_entity_set_chain(self) -> None:
        # T7: entity_set C re-filtering entity_set B is not a second hop — C's record-level filter
        # folds into the root record_set (the 0102 set failure: Softcat leaked in via re-filter).
        from procurement_graph.reasoning.graph_planning import normalise_graph_plan
        graph = {
            "variables": [
                {"var_id": "A", "kind": "record_set", "role": "contract_records",
                 "filters": [{"slot": "buyer", "value": "PFCC Essex", "operator": "eq"}], "depends_on": []},
                {"var_id": "B", "kind": "entity_set", "role": "supplier", "filters": [], "depends_on": ["A"]},
                {"var_id": "C", "kind": "entity_set", "role": "supplier",
                 "filters": [{"slot": "category", "value": "works", "operator": "eq"}], "depends_on": ["B"]},
            ],
            "return": {"operation": "distinct_set", "input": "C", "field": "supplier"},
        }
        cleaned = normalise_graph_plan("Which suppliers show up on works notices issued by PFCC Essex?", graph)
        by_id = {v["var_id"]: v for v in cleaned["variables"]}
        self.assertNotIn("C", by_id)
        self.assertEqual(cleaned["return"]["input"], "B")
        a_filters = [(f["slot"], f["value"]) for f in by_id["A"]["filters"]]
        self.assertIn(("category", "works"), a_filters)

    def test_graph_normalise_executor_capability_cleanup(self) -> None:
        from procurement_graph.reasoning.graph_planning import compile_graph_plan, normalise_graph_plan
        graph = {
            "variables": [
                {"var_id": "a1", "kind": "record_set", "role": "contract_records",
                 "filters": [
                     {"slot": "date", "value": "2024", "operator": "eq"},
                     {"slot": "category", "value": "goods notices", "operator": "eq"},
                     {"slot": "category", "value": "matching procurement records", "operator": "eq"},
                 ], "depends_on": []},
            ],
            "return": {"operation": "count", "input": "a1", "field": "none"},
        }
        cleaned = normalise_graph_plan("How many 2024 goods notices?", graph)
        filters = [(f["slot"], f["value"]) for f in cleaned["variables"][0]["filters"]]
        self.assertIn(("year", "2024"), filters)
        self.assertIn(("category", "goods"), filters)
        self.assertNotIn(("category", "matching procurement records"), filters)
        compiled, reason = compile_graph_plan("How many 2024 goods notices?", {"graph_plan": cleaned})
        self.assertEqual(reason, "")
        constraints = [(c.field, c.op, c.value) for c in compiled.variables[0].constraints]
        self.assertIn(("release_year", "eq", 2024), constraints)
        self.assertIn(("tender_category", "eq", "goods"), constraints)

    def test_graph_compile_unknown_none_return_input_is_structured_failure(self) -> None:
        from procurement_graph.reasoning.graph_planning import compile_graph_plan
        payload = {
            "graph_plan": {
                "variables": [
                    {"var_id": "a1", "kind": "record_set", "role": "contract_records",
                     "filters": [], "depends_on": []},
                ],
                "return": {"operation": "count", "input": "none", "field": "none"},
            },
        }
        graph, reason = compile_graph_plan("How many notices?", payload)
        self.assertIsNone(graph)
        self.assertEqual(reason, "unknown_return_input:none")

    def test_graph_compare_count_wrappers_resolve_to_variable_outputs(self) -> None:
        payload = {
            "graph_plan": {
                "variables": [
                    {"var_id": "a1", "kind": "record_set", "role": "contract_records",
                     "filters": [{"slot": "buyer", "value": "Alpha Council", "operator": "eq"},
                                 {"slot": "year", "value": "2024", "operator": "eq"}], "depends_on": []},
                    {"var_id": "a2", "kind": "record_set", "role": "contract_records",
                     "filters": [{"slot": "buyer", "value": "Beta NHS Trust", "operator": "eq"},
                                 {"slot": "year", "value": "2024", "operator": "eq"}], "depends_on": []},
                ],
                "return": {"operation": "compare", "input": "a1", "field": "none",
                           "metric": "count", "comparator": "gt", "left": "count(a1)", "right": "count(a2)"},
            },
        }
        trace = ReasoningPipeline(backend=self.backend, planner=_GraphPayloadPlanner(payload)).run(
            "Did Alpha Council publish more contract notices than Beta NHS Trust in 2024?")
        self.assertTrue(trace.execution.passed)
        self.assertTrue(trace.answer_card.answer["answer"])

    def test_graph_compare_with_literal_threshold_side(self) -> None:
        # 0050: return.right was the literal "GBP 10 million"; 0749: left was missing entirely.
        payload = {
            "graph_plan": {
                "question_type": "comparison", "operation": "compare",
                "variables": [
                    {"var_id": "A", "kind": "scalar", "role": "value",
                     "filters": [{"slot": "year", "value": 2024, "operator": "eq"},
                                 {"slot": "category", "value": "works", "operator": "eq"}],
                     "depends_on": []},
                ],
                "return": {"operation": "compare", "input": "A", "field": "value",
                           "comparator": "gt", "right": "GBP 300"},
            },
        }
        trace = ReasoningPipeline(backend=self.backend, planner=_GraphPayloadPlanner(payload)).run(
            "Was the total value of 2024 works contracts more than GBP 300?")
        self.assertTrue(trace.execution.passed)
        self.assertTrue(trace.answer_card.answer["answer"])  # 350 > 300

    def test_extreme_exclude_zero(self) -> None:
        from procurement_graph.reasoning.models import RuntimeQuerySpec as _Spec
        backend = TabularRuntimeBackend(mock_records() + [{
            "contract_node_id": "c0", "ocid": "ocds-0", "buyer_name": "Alpha Council",
            "supplier_name": "ZeroCo", "release_year": 2024, "tender_category": "works",
            "value_amount": "0.00", "value_is_additive": True,
        }])
        spec = _Spec(spec_id="x", question="lowest non-zero", intent="min_max", constraints=(),
                     answer_operation="argmin", answer_field="contract_node_id",
                     answer_value_type="string", sort_field="value_amount",
                     requires_exhaustive_retrieval=True, metadata={"exclude_zero": True})
        result = execute_query_spec(backend, spec)
        self.assertTrue(result.passed)
        self.assertEqual(result.answer, "c1")  # 100 is the lowest NON-ZERO, not c0's 0

    def test_graph_plan_trace_is_json_serializable(self) -> None:
        # graph_plan_trace flows into the reflector feedback prompt; a compiled low_level_spec used
        # to carry QueryConstraint objects that crashed json.dumps and broke the whole repair loop.
        import json as _json
        from procurement_graph.reasoning.graph_planning import compile_graph_plan, graph_plan_trace
        payload = {
            "graph_plan": {
                "variables": {"A": {"kind": "record_set", "description": "2024 works",
                                    "filters": [{"slot": "year", "surface": "2024", "value": 2024},
                                                {"slot": "category", "surface": "works", "value": "works"}],
                                    "depends_on": []}},
                "return": {"operation": "count", "input": "A", "field": "contracts", "guards": []},
            },
        }
        graph, reason = compile_graph_plan("How many works contracts in 2024?", payload)
        self.assertEqual(reason, "")
        trace = graph_plan_trace(graph)
        _json.dumps(trace)  # must not raise
        self.assertEqual(trace["variables"][0]["low_level_spec"]["answer_operation"], "count")

    def test_graph_cpv_label_misslotted_as_category_is_dropped(self) -> None:
        # "under CPV 90000000 (works cleanup services)" — the parenthetical is the CPV label, not a
        # procurement category. A category filter outside goods|services|works is dropped so the
        # query runs on the real filters instead of erroring at grounding.
        from procurement_graph.reasoning.graph_planning import compile_graph_plan as _cgp
        payload = {
            "graph_plan": {
                "variables": {
                    "A": {"kind": "record_set", "description": "2024 works contracts",
                          "filters": [{"slot": "year", "surface": "2024", "value": 2024},
                                      {"slot": "category", "surface": "works", "value": "works"},
                                      {"slot": "category", "surface": "Civic-amenity services",
                                       "value": "Civic-amenity services"}],
                          "depends_on": []},
                },
                "return": {"operation": "count", "input": "A", "field": "contracts", "guards": []},
            },
        }
        graph, reason = _cgp("How many works contracts in 2024 under a civic label?", payload)
        self.assertEqual(reason, "")
        cats = [c.value for v in graph.variables for c in v.constraints if c.field == "tender_category"]
        self.assertEqual(cats, ["works"])  # the CPV-label category was dropped, real one kept

    def test_graph_find_extreme_operation_unit(self) -> None:
        payload = {
            "understanding_network": {"reasoning_order": ["C", "answer"]},
            "graph_plan": {
                "variables": {
                    "C": {"kind": "record_set", "description": "2024 works contracts",
                          "filters": [{"slot": "year", "surface": "2024", "value": 2024},
                                      {"slot": "category", "surface": "works", "value": "works"}],
                          "depends_on": []}
                },
                "operation_units": [
                    {"step_id": "u1", "operation_unit": "find_record_set",
                     "inputs": ["2024", "works"], "output": "C"},
                    {"step_id": "u2", "operation_unit": "find_extreme",
                     "inputs": ["C"], "output": "answer", "uses": {"extreme": "max"}}
                ],
                "return": {"operation": "argmax", "input": "C", "field": "contract value", "guards": []},
            },
        }
        trace = ReasoningPipeline(backend=self.backend, planner=_GraphPayloadPlanner(payload)).run(
            "Which 2024 works contract has the highest value?"
        )
        self.assertTrue(trace.execution.passed)
        self.assertEqual(trace.answer_card.answer, "c2")

    def test_graph_compare_operation_unit_combines_scalar_outputs(self) -> None:
        payload = {
            "understanding_network": {"reasoning_order": ["X", "Y", "answer"]},
            "graph_plan": {
                "variables": {
                    "X": {"kind": "scalar", "description": "count of 2024 works contracts",
                          "filters": [{"slot": "year", "surface": "2024", "value": 2024},
                                      {"slot": "category", "surface": "works", "value": "works"}],
                          "depends_on": []},
                    "Y": {"kind": "scalar", "description": "count of 2025 goods contracts",
                          "filters": [{"slot": "year", "surface": "2025", "value": 2025},
                                      {"slot": "category", "surface": "goods", "value": "goods"}],
                          "depends_on": []},
                },
                "operation_units": [
                    {"step_id": "u1", "operation_unit": "aggregate_count",
                     "inputs": ["2024", "works"], "output": "X"},
                    {"step_id": "u2", "operation_unit": "aggregate_count",
                     "inputs": ["2025", "goods"], "output": "Y"},
                    {"step_id": "u3", "operation_unit": "compare",
                     "inputs": ["X", "Y"], "output": "answer"}
                ],
                "return": {"operation": "compare", "input": "answer", "left": "X", "right": "Y",
                           "comparator": "gt", "field": "count", "guards": []},
            },
        }
        trace = ReasoningPipeline(backend=self.backend, planner=_GraphPayloadPlanner(payload)).run(
            "Were there more 2024 works contracts than 2025 goods contracts?"
        )
        self.assertTrue(trace.execution.passed)
        self.assertEqual(trace.answer_card.answer["answer"], True)
        self.assertEqual(trace.answer_card.answer["X"], 2)
        self.assertEqual(trace.answer_card.answer["Y"], 1)
        levels = {v["var_id"]: v["execution_level"] for v in trace.metadata["graph"]["variables"]}
        self.assertEqual(levels["X"], levels["Y"])
        self.assertEqual(trace.metadata["graph"]["execution_levels"], 1)

    def test_graph_empty_result_feedback_names_failed_operation_unit(self) -> None:
        payload = {
            "understanding_network": {"reasoning_order": ["C", "answer"]},
            "graph_plan": {
                "variables": {
                    "C": {"kind": "record_set", "description": "missing 2030 goods contracts",
                          "filters": [{"slot": "year", "surface": "2030", "value": 2030},
                                      {"slot": "category", "surface": "goods", "value": "goods"}],
                          "depends_on": []}
                },
                "operation_units": [
                    {"step_id": "u1", "operation_unit": "find_record_set",
                     "inputs": ["2030", "goods"], "output": "C"},
                    {"step_id": "u2", "operation_unit": "aggregate_count",
                     "inputs": ["C"], "output": "answer"}
                ],
                "return": {"operation": "count", "input": "C", "field": "contracts", "guards": []},
            },
        }
        trace = ReasoningPipeline(backend=self.backend, planner=_GraphPayloadPlanner(payload)).run(
            "How many goods contracts were published in 2030?"
        )
        feedback = _feedback_from_trace(trace)
        self.assertEqual(feedback["graph_failure_kind"], "empty_result")
        self.assertEqual(feedback["failed_operation_unit"]["operation_unit"], "find_record_set")

    def test_no_results_with_question_literals_is_an_abstention(self) -> None:
        pipeline = ReasoningPipeline(backend=self.backend, planner=self.planner)
        # Policy change 2026-07-04: no 2024 goods contracts exist and BOTH filters are stated in
        # the question, so the emptiness IS the answer. The old behaviour dropped the category
        # filter and answered a broadened question -- measured on dev_smoke, that relax->answer
        # chain turned 6 correct abstentions into hallucinations (84% -> 66%).
        trace = pipeline.run("What is the total value of goods contracts published in 2024?")
        self.assertFalse(trace.execution.passed)
        self.assertIsNone(trace.answer_card.answer)
        self.assertEqual(trace.reflection.action, "report_insufficient_evidence")

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

    def test_pipeline_falls_back_to_later_plan_after_unanswered_first_plan(self) -> None:
        pipeline = ReasoningPipeline(backend=self.backend, planner=_TwoPlanFallbackPlanner(), max_rounds=1)
        trace = pipeline.run("How many works contracts were published in 2024?")
        self.assertTrue(trace.execution.passed)
        self.assertEqual(trace.selected_plan_id, "good")
        self.assertEqual(trace.answer_card.answer, 2)
        self.assertEqual(trace.metadata["plan_fallbacks"][0]["plan_id"], "good")

    def test_fallback_chain_preserves_policy_order_not_confidence_order(self) -> None:
        chain = FallbackChainPlanner((_OnePlan("typed", 0.2), _OnePlan("rule", 0.95)))
        plans = chain.plan("q")
        self.assertEqual([plan.plan_id for plan in plans], ["typed", "rule"])
        self.assertIn("fallback_chain:0:typed", plans[0].warnings)
        self.assertIn("fallback_chain:1:rule", plans[1].warnings)

    def test_pipeline_can_replan_once_with_failure_feedback(self) -> None:
        planner = _FeedbackReplanPlanner()
        pipeline = ReasoningPipeline(backend=self.backend, planner=planner, max_rounds=1)
        trace = pipeline.run("How many works contracts were published in 2024?")
        self.assertTrue(trace.execution.passed)
        self.assertEqual(trace.selected_plan_id, "feedback-good")
        self.assertEqual(trace.answer_card.answer, 2)
        self.assertEqual(planner.feedback_seen["execution_status"], "no_results")
        self.assertEqual(planner.feedback_seen["failure_stage"], "executor")
        self.assertEqual(planner.feedback_seen["failure_reason"], "no_results")
        self.assertIn("failed_plan", planner.feedback_seen)
        self.assertIn("fix_question_type", planner.feedback_seen["allowed_repair_actions"])
        self.assertTrue(trace.metadata["feedback_replan"]["answered"])

    def test_pre_execution_review_mismatch_triggers_soft_repair(self) -> None:
        planner = _PreExecutionReviewPlanner(repair_plans=True)
        pipeline = ReasoningPipeline(backend=self.backend, planner=planner, max_rounds=1)
        trace = pipeline.run("How many works contracts were published in 2024?")
        self.assertTrue(trace.execution.passed)
        self.assertEqual(trace.selected_plan_id, "pre-review-good")
        self.assertEqual(trace.answer_card.answer, 2)
        self.assertEqual(planner.feedback_seen["failure_stage"], "pre_execution_review")
        self.assertEqual(planner.feedback_seen["reflection_action"], "pre_execution_review")
        review = trace.metadata["workflow"]["pre_execution_review"]
        self.assertTrue(review["triggered"])
        self.assertTrue(review["repair_planned"])
        self.assertEqual(review["selected_repair_plan_id"], "pre-review-good")

    def test_pre_execution_review_does_not_veto_original_when_repair_fails(self) -> None:
        planner = _PreExecutionReviewPlanner(repair_plans=False)
        pipeline = ReasoningPipeline(backend=self.backend, planner=planner, max_rounds=1)
        trace = pipeline.run("How many works contracts were published in 2024?")
        self.assertTrue(trace.execution.passed)
        self.assertEqual(trace.selected_plan_id, "pre-review-original")
        self.assertEqual(trace.answer_card.answer, 2)
        self.assertEqual(planner.feedback_seen["failure_stage"], "pre_execution_review")
        review = trace.metadata["workflow"]["pre_execution_review"]
        self.assertTrue(review["triggered"])
        self.assertTrue(review["repair_attempted"])
        self.assertFalse(review["repair_planned"])

    def test_feedback_replan_loops_until_verified(self) -> None:
        # attempt protocol: repair 1 fails, repair 2 passes full ground->execute->verify;
        # per-attempt records enable Repair@k metrics and nearest-failure DPO pairing.
        planner = _TwoRoundFeedbackPlanner()
        pipeline = ReasoningPipeline(backend=self.backend, planner=planner,
                                     max_rounds=1, max_feedback_replans=3)
        trace = pipeline.run("How many works contracts were published in 2024?")
        self.assertEqual(trace.answer_card.answer, 2)
        meta = trace.metadata["feedback_replan"]
        self.assertEqual(meta["first_verified_attempt"], 2)
        self.assertEqual([a["attempt"] for a in meta["attempts"]], [1, 2])
        self.assertTrue(meta["attempts"][1]["answered"])
        self.assertFalse(meta["attempts"][0]["answered"])
        self.assertEqual(planner.feedbacks[0]["attempt_index"], 1)
        self.assertEqual(planner.feedbacks[1]["attempt_index"], 2)
        # the second feedback must describe the FIRST REPAIR's failure, not the initial plan's
        self.assertEqual(planner.feedbacks[1]["failure_reason"], "no_results")

    def test_trace_preference_logging_receives_oracle_match_callback(self) -> None:
        reflector = _RecordingTraceReflector()
        pipeline = ReasoningPipeline(
            backend=self.backend,
            planner=self.planner,
            trace_reflector=reflector,
            oracle_matcher=lambda _question, trace: trace.answer_card.answer == 2,
        )
        pipeline.run("How many works contracts were published in 2024?")
        self.assertIs(reflector.logged, True)

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
        # policy change 2026-07-04: question-literal filters are never relaxed
        self.assertEqual(first["reflector_analysis"]["action"], "report_insufficient_evidence")
        self.assertFalse(trace.execution.passed)


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
