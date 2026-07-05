"""Trace-aware reflector (Stage 8): faithfulness verification, plan validity, targeted repair,
advisor validation boundary, and preference logging. Uses the tabular mock backend."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from procurement_graph.reasoning import ReasoningPipeline
from procurement_graph.reasoning.models import (
    AnswerCard,
    CandidatePlan,
    EntityLinkCandidate,
    EvidenceBundle,
    ExecutionResult,
    QueryConstraint,
    ReasoningTrace,
    RuntimeQuerySpec,
)
from procurement_graph.reasoning.planner_decomposition import DecompositionAwarePlanner
from procurement_graph.reasoning.trace_reflector import REPAIR_ACTIONS, TraceReflector
from tests.test_reasoning_runtime import TabularRuntimeBackend


def _spec(op="count", constraints=(), answer_field="contract_node_id", value_type="integer"):
    return RuntimeQuerySpec(spec_id="s", question="q", intent="x",
                            constraints=tuple(QueryConstraint(*c) for c in constraints),
                            answer_operation=op, answer_field=answer_field,
                            answer_value_type=value_type, requires_exhaustive_retrieval=True)


def _trace(question, spec, *, answer, status="passed", evidence_count=None, checks=(),
           metrics=None, metadata=None):
    evidence = EvidenceBundle(fields={"evidence_count": evidence_count} if evidence_count is not None else {})
    execution = ExecutionResult(query_spec=spec, status=status, answer=answer, evidence=evidence,
                                checks=tuple(checks), metrics=metrics or {})
    card = AnswerCard(question=question, answer=answer, answer_text=str(answer),
                      query_spec=spec, execution=execution)
    plan = CandidatePlan(plan_id="p0", query_spec=spec, status="planned", confidence=0.8,
                         rationale="test", planner_source="rule")
    return ReasoningTrace(question=question, plans=(plan,), selected_plan_id="p0",
                          execution=execution, answer_card=card, trace_id="t", metadata=metadata or {})


class _Resolver:
    """Two candidates for the same casefolded mention -> exercises re-linking."""

    def __init__(self, mapping):
        self.mapping = mapping  # mention.casefold() -> [names]

    def resolve(self, mention):
        names = self.mapping.get(str(mention).strip().casefold(), [])
        return [EntityLinkCandidate(mention=mention, linked_id=n, linked_label=n,
                                    entity_type="organization", score=1.0 - 0.1 * i,
                                    source="records_exact" if i == 0 else "records_substring")
                for i, n in enumerate(names)]


class _WeakAlternativeResolver:
    def resolve(self, mention):
        return [
            EntityLinkCandidate(mention=mention, linked_id="ACME LTD", linked_label="ACME LTD",
                                entity_type="organization", score=1.0, source="records_exact"),
            EntityLinkCandidate(mention=mention, linked_id="Acme Facilities Ltd",
                                linked_label="Acme Facilities Ltd", entity_type="organization",
                                score=0.42, source="records_substring"),
        ]


class TestFaithfulness(unittest.TestCase):
    def setUp(self):
        self.reflector = TraceReflector()

    def test_count_matching_evidence_is_faithful(self):
        trace = _trace("How many contract notices were published in 2024?",
                       _spec("count"), answer=3, evidence_count=3)
        reflection = self.reflector.reflect_trace(trace)
        self.assertEqual(reflection.faithfulness, "faithful")
        self.assertEqual(reflection.action, "no_action")

    def test_count_disagreeing_with_evidence_is_suspicious_and_abstains(self):
        trace = _trace("How many contract notices were published in 2024?",
                       _spec("count"), answer=5, evidence_count=3)
        reflection = self.reflector.reflect_trace(trace)
        self.assertEqual(reflection.faithfulness, "suspicious")
        self.assertEqual(reflection.action, "abstain")

    def test_sum_without_contributors_is_suspicious(self):
        trace = _trace("What is the total value of contracts in 2024?",
                       _spec("sum", answer_field="value_amount", value_type="currency"),
                       answer=100, metrics={"contributor_count": 0})
        reflection = self.reflector.reflect_trace(trace)
        self.assertEqual(reflection.faithfulness, "suspicious")

    def test_abstention_is_not_suspicious(self):
        trace = _trace("How many notices?", _spec("count"), answer=None, status="no_results")
        reflection = self.reflector.reflect_trace(trace)
        self.assertEqual(reflection.faithfulness, "abstained")

    def test_failed_decomposition_hop_is_suspicious(self):
        trace = _trace("bridge question: how many notices by buyers of S1?", _spec("count"),
                       answer=7, evidence_count=7,
                       metadata={"decomposition": {"hops": [{"status": "passed"}, {"status": "no_results"}]}})
        reflection = self.reflector.reflect_trace(trace)
        self.assertEqual(reflection.faithfulness, "suspicious")


class TestPlanValidityAndRepair(unittest.TestCase):
    def test_type_mismatch_is_flagged_and_retyped_on_failure(self):
        # question asks a count, executed as sum, and failed -> re_plan_question_type
        trace = _trace("How many contract notices were published in 2024?",
                       _spec("sum", answer_field="value_amount", value_type="currency"),
                       answer=None, status="incomplete_evidence")
        reflection = TraceReflector().reflect_trace(trace)
        self.assertFalse(reflection.plan_valid)
        self.assertEqual(reflection.action, "re_plan_question_type")
        self.assertEqual(reflection.repair_spec.answer_operation, "count")

    def test_multiple_answers_marks_ambiguous(self):
        trace = _trace("Who was the buyer for the 2024 contract with S?",
                       _spec("select_unique", answer_field="buyer_name", value_type="string"),
                       answer=None, status="multiple_answers",
                       checks=({"check": "select_unique", "passed": False, "unique_values": ["A", "B"]},))
        reflection = TraceReflector().reflect_trace(trace)
        self.assertEqual(reflection.action, "mark_ambiguous")
        self.assertEqual(reflection.metadata["distinct_values"], ["A", "B"])

    def test_highest_total_value_reads_as_superlative_before_sum(self):
        trace = _trace("Which contract had the highest total value?",
                       _spec("argmax", answer_field="contract_node_id", value_type="string"),
                       answer="c1", evidence_count=1)
        reflection = TraceReflector().reflect_trace(trace)
        self.assertTrue(reflection.plan_valid, reflection.plan_issues)

    def test_no_results_with_alternative_candidate_relinks(self):
        resolver = _Resolver({"acme ltd": ["ACME LTD", "Acme Ltd."]})
        spec = _spec("count", constraints=(("supplier_name", "eq", "ACME LTD"),))
        trace = _trace("How many notices were awarded to Acme Ltd?", spec, answer=None, status="no_results")
        reflection = TraceReflector(org_resolver=resolver).reflect_trace(trace)
        self.assertEqual(reflection.action, "re_link_entity")
        swapped = [c for c in reflection.repair_spec.constraints if c.field == "supplier_name"]
        self.assertEqual(swapped[0].value, "Acme Ltd.")

    def test_no_results_weak_alternative_does_not_relink(self):
        spec = _spec("count", constraints=(("supplier_name", "eq", "ACME LTD"),))
        trace = _trace("How many notices were awarded to Acme Ltd?", spec, answer=None, status="no_results")
        reflection = TraceReflector(org_resolver=_WeakAlternativeResolver()).reflect_trace(trace)
        self.assertEqual(reflection.action, "relax_constraints")
        self.assertIsNone(reflection.repair_spec)

    def test_no_results_without_alternative_relaxes(self):
        spec = _spec("count", constraints=(("release_year", "eq", 2024), ("tender_category", "eq", "goods")))
        trace = _trace("How many goods notices in 2024?", spec, answer=None, status="no_results")
        reflection = TraceReflector().reflect_trace(trace)
        self.assertEqual(reflection.action, "relax_constraints")


class _Advisor:
    def __init__(self, verdict):
        self.verdict = verdict
        self.calls = []

    def advise(self, payload):
        self.calls.append(payload)
        return self.verdict


class TestAdvisorBoundary(unittest.TestCase):
    def _failed_trace(self):
        return _trace("How many notices in 2024?", _spec("count"), answer=None, status="error")

    def test_invalid_advisor_action_is_ignored(self):
        advisor = _Advisor({"recommended_action": "just_answer_42"})
        reflection = TraceReflector(advisor=advisor).reflect_trace(self._failed_trace())
        self.assertEqual(reflection.action, "abstain")  # deterministic decision stands
        self.assertEqual(len(advisor.calls), 1)

    def test_advisor_may_reroute_between_conservative_actions(self):
        advisor = _Advisor({"recommended_action": "mark_ambiguous"})
        reflection = TraceReflector(advisor=advisor).reflect_trace(self._failed_trace())
        self.assertEqual(reflection.action, "mark_ambiguous")
        self.assertIn("advisor", reflection.reason)

    def test_advisor_entity_choice_must_come_from_candidates(self):
        resolver = _Resolver({"acme ltd": ["ACME LTD", "Acme Ltd."]})
        spec = _spec("count", constraints=(("supplier_name", "eq", "ACME LTD"),))
        trace = _trace("notices awarded to Acme Ltd?", spec, answer=None, status="error")
        advisor = _Advisor({"recommended_action": "re_link_entity",
                            "entity_choice": {"field": "supplier_name", "value": "Totally Made Up Org"}})
        reflection = TraceReflector(org_resolver=resolver, advisor=advisor).reflect_trace(trace)
        self.assertIsNone(reflection.repair_spec)  # invalid choice -> no repair adopted
        self.assertEqual(reflection.action, "abstain")

    def test_advisor_not_consulted_on_success(self):
        advisor = _Advisor({"recommended_action": "abstain"})
        trace = _trace("How many notices?", _spec("count"), answer=2, evidence_count=2)
        TraceReflector(advisor=advisor).reflect_trace(trace)
        self.assertEqual(advisor.calls, [])


class TestPreferenceLogging(unittest.TestCase):
    def test_good_bad_and_safe_abstain_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "pref.jsonl"
            reflector = TraceReflector(preference_log=log)
            good = _trace("How many notices?", _spec("count"), answer=2, evidence_count=2)
            reflector.log_preference(good, reflector.reflect_trace(good))
            bad = _trace("How many notices?", _spec("count"), answer=9, evidence_count=2)
            reflector.log_preference(bad, reflector.reflect_trace(bad))
            abstain = _trace("How many notices?", _spec("count"), answer=None, status="no_results")
            reflector.log_preference(abstain, reflector.reflect_trace(abstain))
            oracle_bad = _trace("How many notices?", _spec("count"), answer=2, evidence_count=2)
            reflector.log_preference(oracle_bad, reflector.reflect_trace(oracle_bad), oracle_match=False)

            rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["label"] for row in rows], ["good", "bad", "safe_abstain", "bad"])
            self.assertTrue(all(row["chosen_plan"]["plan_id"] == "p0" for row in rows))

    def test_logging_disabled_returns_none(self):
        reflector = TraceReflector()
        trace = _trace("How many notices?", _spec("count"), answer=2, evidence_count=2)
        self.assertIsNone(reflector.log_preference(trace, reflector.reflect_trace(trace)))


class TestPipelineIntegration(unittest.TestCase):
    """End-to-end: a wrong entity variant yields no_results; the trace reflector re-links to the
    resolver's alternative candidate and the bounded repair round produces a verified answer."""

    def setUp(self):
        records = [
            {"contract_node_id": f"c{i}", "buyer_name": "B1", "supplier_name": "Acme Ltd.",
             "value_amount": "10", "value_is_additive": True, "release_year": 2024,
             "tender_category": "goods"}
            for i in range(3)
        ]
        self.backend = TabularRuntimeBackend(records)
        self.resolver = _Resolver({"acme ltd": ["ACME LTD", "Acme Ltd."],
                                   "acme ltd.": ["Acme Ltd."], "b1": ["B1"]})

    def test_trace_repair_recovers_answer(self):
        class _WrongVariantPlanner:
            def plan(self, question):
                spec = _spec("count", constraints=(("supplier_name", "eq", "ACME LTD"),))
                return (CandidatePlan(plan_id="p0", query_spec=spec, status="planned",
                                      confidence=0.9, rationale="test", planner_source="rule"),)

        pipe = ReasoningPipeline(backend=self.backend, planner=_WrongVariantPlanner(),
                                 org_resolver=self.resolver,
                                 trace_reflector=TraceReflector(org_resolver=self.resolver))
        trace = pipe.run("How many contract notices were awarded to Acme Ltd?")
        self.assertEqual(trace.answer_card.answer, 3)
        self.assertEqual(trace.metadata["trace_repair"]["status"], "passed")
        self.assertEqual(trace.metadata["trace_reflection"]["action"], "re_link_entity")
        self.assertIn("trace-reflector repair applied",
                      " ".join(trace.answer_card.limitations))

    def test_without_reflector_behaviour_unchanged(self):
        resolver = self.resolver
        pipe = ReasoningPipeline(backend=self.backend,
                                 planner=DecompositionAwarePlanner(org_resolver=resolver),
                                 org_resolver=resolver)
        trace = pipe.run("How many contract notices were published in 2024?")
        self.assertEqual(trace.answer_card.answer, 3)
        self.assertNotIn("trace_reflection", trace.metadata)


if __name__ == "__main__":
    unittest.main()
