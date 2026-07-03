"""Verify-then-escalate cascade: escalation is triggered by deterministic verification (probe
execution failure or question-type contradiction), never by rule coverage; and Level-4
plan-verbalization must not leak the template question to the LLM."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from procurement_graph.reasoning.models import CandidatePlan, QueryConstraint, RuntimeQuerySpec
from procurement_graph.reasoning.planner_decomposition import VerifyingHybridPlanner
from tests.test_reasoning_runtime import TabularRuntimeBackend


def _records():
    return [{"contract_node_id": f"c{i}", "buyer_name": "B1", "supplier_name": "S1",
             "value_amount": "10", "value_is_additive": True, "release_year": 2024,
             "tender_category": "goods"} for i in range(4)]


def _cand(op="count", constraints=(), status="planned", source="rule", answer_field="contract_node_id",
          value_type="integer"):
    spec = RuntimeQuerySpec(spec_id="s", question="q", intent="x",
                            constraints=tuple(QueryConstraint(*c) for c in constraints),
                            answer_operation=op, answer_field=answer_field,
                            answer_value_type=value_type, requires_exhaustive_retrieval=True)
    return CandidatePlan(plan_id=f"{source}0", query_spec=spec, status=status, confidence=0.8,
                         rationale=source, planner_source=source)


class _StubPlanner:
    def __init__(self, cands):
        self.cands, self.calls = tuple(cands), 0

    def plan(self, question):
        self.calls += 1
        return self.cands


class TestVerifyingHybrid(unittest.TestCase):
    def setUp(self):
        self.backend = TabularRuntimeBackend(_records())

    def test_verified_rule_plan_never_consults_llm(self):
        rule = _StubPlanner([_cand(constraints=(("release_year", "eq", 2024),))])
        llm = _StubPlanner([_cand(source="llm")])
        planner = VerifyingHybridPlanner(rule=rule, backend=self.backend, llm=llm)
        [top] = planner.plan("How many contract notices were published in 2024?")[:1]
        self.assertEqual(top.planner_source, "rule")
        self.assertEqual(llm.calls, 0)

    def test_hard_probe_failure_escalates(self):
        # rule misfires confidently: a SUM over a filter matching nothing -> no_results (hard)
        rule = _StubPlanner([_cand(op="sum", answer_field="value_amount",
                                   constraints=(("buyer_name", "eq", "NOPE"),))])
        good = _cand(constraints=(("buyer_name", "eq", "B1"),), source="llm")
        llm = _StubPlanner([good])
        planner = VerifyingHybridPlanner(rule=rule, backend=self.backend, llm=llm)
        [top] = planner.plan("What is the total value the Birmingham body awarded?")[:1]
        self.assertEqual(top.planner_source, "llm+escalated")
        self.assertTrue(any(w.startswith("escalated_because:probe_no_results") for w in top.warnings))

    def test_valid_zero_and_false_do_not_escalate(self):
        # post-mortem regression class: oracle=False predicates were escalated and broken by the
        # aggressive degenerate-zero trigger. 0/False are valid answers -> NO LLM call.
        rule = _StubPlanner([_cand(constraints=(("buyer_name", "eq", "NOPE"),))])  # count -> 0
        llm = _StubPlanner([_cand(constraints=(("buyer_name", "eq", "B1"),), source="llm")])
        planner = VerifyingHybridPlanner(rule=rule, backend=self.backend, llm=llm)
        [top] = planner.plan("How many contract notices did NOPE place in 2024?")[:1]
        self.assertEqual(top.planner_source, "rule")
        self.assertEqual(llm.calls, 0)

    def test_multiple_answers_does_not_escalate(self):
        # underdetermined factoid: abstention/clarification is the correct outcome, not an LLM call
        rule = _StubPlanner([_cand(op="select_unique", answer_field="contract_node_id",
                                   value_type="string", constraints=(("buyer_name", "eq", "B1"),))])
        llm = _StubPlanner([_cand(source="llm")])
        planner = VerifyingHybridPlanner(rule=rule, backend=self.backend, llm=llm)
        [top] = planner.plan("Which contract did B1 award?")[:1]
        self.assertEqual(top.planner_source, "rule")
        self.assertEqual(llm.calls, 0)

    def test_llm_plan_failing_probe_keeps_rule_plan(self):
        rule = _StubPlanner([_cand(op="sum", answer_field="value_amount",
                                   constraints=(("buyer_name", "eq", "NOPE"),))])
        llm = _StubPlanner([_cand(op="sum", answer_field="value_amount",
                                  constraints=(("buyer_name", "eq", "ALSO_NOPE"),), source="llm")])
        planner = VerifyingHybridPlanner(rule=rule, backend=self.backend, llm=llm)
        [top] = planner.plan("What is the total value that body awarded?")[:1]
        self.assertEqual(top.planner_source, "rule")  # deterministic, auditable failure preferred

    def test_unsupported_never_escalates(self):
        rule = _StubPlanner([_cand(status="unsupported")])
        llm = _StubPlanner([_cand(source="llm")])
        planner = VerifyingHybridPlanner(rule=rule, backend=self.backend, llm=llm)
        planner.plan("What were the social value commitments?")
        self.assertEqual(llm.calls, 0)


class TestPlanVerbalization(unittest.TestCase):
    ROW = {
        "id": "x", "subset": "coverage_fixed", "expected_status": "answerable",
        "question": "TEMPLATE-SENTINEL should never reach the LLM?",
        "answer_operation": "count", "answer_type": "count", "oracle_answer": 4,
        "constraints": [{"field": "release_year", "op": "eq", "value": 2024},
                        {"field": "buyer_name", "op": "eq", "value": "B1"}],
    }

    def test_template_question_is_withheld(self):
        from procurement_graph.qa.multilevel import l2_rewrite_messages, required_atoms, verbalize_messages
        system, user = verbalize_messages(self.ROW, required_atoms(self.ROW))
        self.assertNotIn("TEMPLATE-SENTINEL", user)
        self.assertNotIn("TEMPLATE-SENTINEL", system)
        self.assertIn("B1", user)
        self.assertIn("2024", user)
        self.assertNotIn('"field"', user)
        self.assertNotIn("release_year exactly", user)
        _, l2_user = l2_rewrite_messages(self.ROW, required_atoms(self.ROW), persona="citizen")
        self.assertIn("task", l2_user)
        self.assertIn("examples_by_identity", l2_user)
        self.assertIn("note", l2_user)
        self.assertIn("TEMPLATE-SENTINEL", l2_user)
        self.assertIn("obvious abbreviation", l2_user)

    def test_l2_skips_unanswerable_and_persona_checker_flow(self):
        from build_multilevel_qa import rewrite_surfaces
        from procurement_graph.qa.multilevel import checker_accepts

        class _Chat:
            def __init__(self):
                self.models = []

            def complete_json(self, *, model, system, user):
                self.models.append(model)

                class _R:
                    parsed = ({"variants": [
                        "What social value commitments were recorded for B1 in 2024?"
                    ]} if "variants" in system and "social value" in user else
                              {"variants": ["How many contract notices did B1 publish in 2024?"]}
                              if "variants" in system else
                              {"matches_original_plan": True, "can_be_answered_by_original_plan": True,
                               "same_meaning_as_source_question": True,
                               "preserves_unanswerable_status": True, "mismatch_reason": None})
                return _R()

        chat = _Chat()
        unanswerable = {**self.ROW, "id": "y", "expected_status": "unsupported",
                        "question": "What social value commitments were recorded for B1 in 2024?"}
        accepted, _ = rewrite_surfaces([unanswerable], level=2, chat=chat, model="stub",
                                       retries=0, variants=1, org_resolver=None, known_orgs=None,
                                       checker_model="check")
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["l2_generation_mode"], "l1_persona_rewrite")
        accepted, _ = rewrite_surfaces([self.ROW], level=2, chat=chat, model="gen", retries=0,
                                       variants=1, org_resolver=None, known_orgs=None,
                                       checker_model="check")
        self.assertEqual(len(accepted), 1)
        self.assertIn(accepted[0]["persona"], ("citizen", "policy_analyst", "auditor", "journalist"))
        self.assertTrue(accepted[0]["checker"]["matches_original_plan"])
        self.assertIn("check", chat.models)  # the independent checker was consulted
        # checker rejection path
        from procurement_graph.qa.multilevel import checker_messages
        c_system, c_user = checker_messages("How many contract notices did B1 publish in 2024?",
                                            self.ROW)
        self.assertIn("government consultation adviser", c_system)
        self.assertIn("can_derive_reference_answer", c_user)
        self.assertIn("Question asks for goods", c_user)
        ok, why = checker_accepts({"matches_original_plan": False, "mismatch_reason": "role flipped"})
        self.assertFalse(ok)
        self.assertIn("role flipped", why)
        ok, why = checker_accepts({"same_meaning_as_source_question": True,
                                   "preserves_unanswerable_status": True},
                                  expected_status="unsupported")
        self.assertTrue(ok, why)


if __name__ == "__main__":
    unittest.main()
