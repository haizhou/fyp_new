"""Typed planner DSL: LLM fills slots, deterministic checks protect semantics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from procurement_graph.reasoning.typed_planning import (
    TypedLLMPlanner,
    compile_typed_plan,
    plan_consistency_check,
    question_understanding_messages,
    typed_plan_messages,
)


class _Hit:
    linked_id = "BIRMINGHAM CITY COUNCIL"
    linked_label = "BIRMINGHAM CITY COUNCIL"
    source = "records_exact"
    score = 1.0


class _Resolver:
    def resolve(self, mention):
        return [_Hit()] if str(mention).casefold() == "birmingham city council" else []


class _Response:
    def __init__(self, parsed):
        self.parsed = parsed


class _Chat:
    def __init__(self, parsed):
        self.parsed = parsed
        self.calls = 0

    def complete_json(self, *, model, system, user):
        self.calls += 1
        return _Response(self.parsed)


class TestTypedPlanningMessages(unittest.TestCase):
    def test_understanding_prompt_is_not_compiler_slots_first(self):
        system, user = question_understanding_messages(
            "How many notices did BIRMINGHAM CITY COUNCIL publish in 2024?")
        self.assertIn("understand", system)
        self.assertIn("needs_to_return", user)
        self.assertIn("known_information", user)
        self.assertIn("reasoning_chain", user)
        self.assertIn("Do not answer", system)

    def test_prompt_exposes_dsl_not_answer_request(self):
        system, user = typed_plan_messages("How many notices did BIRMINGHAM CITY COUNCIL publish in 2024?")
        self.assertIn("fixed reasoning DSL", system)
        self.assertIn("question_type", user)
        self.assertNotIn("oracle", user.casefold())


class TestPlanConsistencyCheck(unittest.TestCase):
    def test_rejects_invented_year(self):
        payload = {
            "question_type": "count",
            "operation": "count",
            "constraints": [
                {"slot": "buyer", "surface": "BIRMINGHAM CITY COUNCIL",
                 "value": "BIRMINGHAM CITY COUNCIL"},
                {"slot": "year", "surface": "2024", "value": 2025},
            ],
        }
        verdict = plan_consistency_check(
            "How many notices did BIRMINGHAM CITY COUNCIL publish in 2024?", payload)
        self.assertFalse(verdict.ok)
        self.assertIn("invented_number:2025", verdict.issues)

    def test_rejects_buyer_supplier_role_flip(self):
        payload = {
            "question_type": "count",
            "operation": "count",
            "constraints": [
                {"slot": "buyer", "surface": "BIRMINGHAM CITY COUNCIL",
                 "value": "BIRMINGHAM CITY COUNCIL"},
            ],
        }
        verdict = plan_consistency_check(
            "How many notices were awarded to BIRMINGHAM CITY COUNCIL?", payload)
        self.assertFalse(verdict.ok)
        self.assertTrue(any(issue.startswith("role_flipped:buyer") for issue in verdict.issues))

    def test_rejects_operation_outside_question_type(self):
        payload = {"question_type": "count", "operation": "sum", "constraints": []}
        verdict = plan_consistency_check("How many notices were published?", payload)
        self.assertFalse(verdict.ok)
        self.assertIn("operation_outside_type:sum!in:count", verdict.issues)


class TestCompileTypedPlan(unittest.TestCase):
    def test_count_plan_compiles_and_resolves_org(self):
        payload = {
            "question_type": "count",
            "operation": "count",
            "constraints": [
                {"slot": "buyer", "surface": "Birmingham City Council",
                 "value": "Birmingham City Council"},
                {"slot": "year", "surface": "2024", "value": "2024"},
            ],
        }
        candidate = compile_typed_plan(
            "How many notices did Birmingham City Council publish in 2024?",
            payload,
            org_resolver=_Resolver(),
        )
        self.assertEqual(candidate.status, "planned")
        self.assertEqual(candidate.query_spec.answer_operation, "count")
        self.assertEqual(
            [(c.field, c.op, c.value) for c in candidate.query_spec.constraints],
            [("buyer_name", "eq", "BIRMINGHAM CITY COUNCIL"), ("release_year", "eq", 2024)],
        )

    def test_unresolved_org_becomes_structured_ambiguity(self):
        payload = {
            "question_type": "count",
            "operation": "count",
            "constraints": [{"slot": "buyer", "surface": "Unknown Body", "value": "Unknown Body"}],
        }
        candidate = compile_typed_plan("How many notices did Unknown Body publish?", payload,
                                       org_resolver=_Resolver())
        self.assertEqual(candidate.status, "ambiguous")
        self.assertIn("entity_not_found", candidate.rationale)


class TestTypedLLMPlanner(unittest.TestCase):
    def test_consistency_failure_never_compiles_as_planned(self):
        payload = {
            "question_type": "count",
            "operation": "count",
            "constraints": [{"slot": "year", "surface": "2024", "value": 2025}],
        }
        planner = TypedLLMPlanner(client=_Chat(payload), model="stub")
        [candidate] = planner.plan("How many notices were published in 2024?")
        self.assertEqual(candidate.status, "ambiguous")
        self.assertIn("plan_semantic_mismatch", candidate.rationale)

    def test_valid_payload_returns_typed_candidate(self):
        payload = {
            "question_type": "count",
            "operation": "count",
            "constraints": [{"slot": "year", "surface": "2024", "value": "2024"}],
        }
        chat = _Chat(payload)
        planner = TypedLLMPlanner(client=chat, model="stub")
        [candidate] = planner.plan("How many notices were published in 2024?")
        self.assertEqual(candidate.status, "planned")
        self.assertEqual(candidate.planner_source, "typed_llm")
        self.assertEqual(chat.calls, 2)
        self.assertIn("understanding", candidate.raw_response)
        self.assertIn("typed_plan", candidate.raw_response)


if __name__ == "__main__":
    unittest.main()
