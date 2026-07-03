"""Tests for runtime reasoning planner contracts."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from procurement_graph.reasoning import (
    PLANNER_SCHEMA_VERSION,
    QueryConstraint,
    RuleBasedDryRunPlanner,
    candidate_from_payload,
    execute_query_spec,
    planner_output_schema,
)
from tests.test_reasoning_runtime import TabularRuntimeBackend, mock_records


class TestReasoningPlanner(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = RuleBasedDryRunPlanner()

    def test_rule_planner_count_with_year_category_and_cpv(self) -> None:
        [plan] = self.planner.plan("How many works contracts were published in 2024 under CPV 45000000?")

        spec = plan.query_spec

        self.assertEqual(plan.status, "planned")
        self.assertEqual(spec.answer_operation, "count")
        self.assertEqual(spec.intent, "categorical_cpv")
        self.assertIn(QueryConstraint("release_year", "eq", 2024, source_text="2024"), spec.constraints)
        self.assertIn(QueryConstraint("tender_category", "eq", "works", source_text="works contracts"), spec.constraints)
        self.assertIn(QueryConstraint("tender_cpv_id", "eq", "45000000", source_text="CPV 45000000"), spec.constraints)

    def test_rule_planner_sum_adds_hidden_additive_constraint(self) -> None:
        [plan] = self.planner.plan("What is the total value of works contracts published in 2024?")

        spec = plan.query_spec
        hidden = [constraint for constraint in spec.constraints if constraint.field == "value_is_additive"]

        self.assertEqual(spec.answer_operation, "sum")
        self.assertEqual(spec.answer_field, "value_amount")
        self.assertEqual(len(hidden), 1)
        self.assertFalse(hidden[0].visible_to_user)

    def test_rule_planner_unsupported_terms(self) -> None:
        [plan] = self.planner.plan("Which contract had the strongest carbon reduction clause?")

        self.assertEqual(plan.status, "unsupported")
        self.assertEqual(plan.query_spec.intent, "unsupported")
        self.assertIn("carbon", plan.rationale)

    def test_rule_planner_ambiguous_question(self) -> None:
        [plan] = self.planner.plan("Tell me about procurement activity.")

        self.assertEqual(plan.status, "ambiguous")
        self.assertEqual(plan.query_spec.answer_operation, "unsupported")

    def test_planner_payload_schema_to_candidate(self) -> None:
        payload = {
            "schema_version": PLANNER_SCHEMA_VERSION,
            "plans": [
                {
                    "intent": "aggregation_count",
                    "answer_operation": "count",
                    "answer_field": "contract_node_id",
                    "answer_value_type": "integer",
                    "constraints": [{"field": "release_year", "op": "eq", "value": 2024, "source_text": "2024"}],
                    "dedupe_key": "contract_node_id",
                    "target_node_type": "contract",
                    "requires_exhaustive_retrieval": True,
                    "rationale": "count contracts in the year",
                    "confidence": 0.9,
                }
            ],
        }

        plan = candidate_from_payload("How many contracts were published in 2024?", payload)

        self.assertEqual(plan.status, "planned")
        self.assertEqual(plan.query_spec.constraints[0].field, "release_year")
        self.assertEqual(plan.confidence, 0.9)
        self.assertEqual(plan.planner_source, "llm")
        self.assertFalse(plan.fallback_used)

    def test_planner_output_schema_mentions_schema_version(self) -> None:
        schema = planner_output_schema()

        self.assertEqual(schema["schema_version"], PLANNER_SCHEMA_VERSION)
        self.assertIn("plans", schema)

    def test_rule_planner_output_executes_against_mock_backend(self) -> None:
        [plan] = self.planner.plan("How many works contracts were published in 2024?")
        backend = TabularRuntimeBackend(mock_records())

        result = execute_query_spec(backend, plan.query_spec)

        self.assertTrue(result.passed)
        self.assertEqual(result.answer, 2)


if __name__ == "__main__":
    unittest.main()
