"""Tests for the runtime reasoning skeleton."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from procurement_graph.reasoning import QueryConstraint, RuntimeQuerySpec, build_answer_card, execute_query_spec


class TabularRuntimeBackend:
    def __init__(self, records: list[dict[str, Any]], id_field: str = "contract_node_id") -> None:
        self.records = records
        self.id_field = id_field

    def fields(self) -> set[str]:
        return {key for record in self.records for key in record}

    def record_id(self, record: dict[str, Any]) -> str:
        return str(record.get(self.id_field, ""))

    def query(self, constraints: tuple[QueryConstraint, ...]) -> list[dict[str, Any]]:
        return [record for record in self.records if all(_matches(record, constraint) for constraint in constraints)]

    def count(self, constraints: tuple[QueryConstraint, ...], *, dedupe_field: str = "contract_node_id") -> int:
        rows = self.query(constraints)
        if not dedupe_field:
            return len(rows)
        seen: set[str] = set()
        total = 0
        for record in rows:
            key = str(record.get(dedupe_field, ""))
            if not key or key in seen:
                continue
            seen.add(key)
            total += 1
        return total

    def sample(self, constraints: tuple[QueryConstraint, ...], n: int) -> list[dict[str, Any]]:
        return self.query(constraints)[: max(0, n)]

    def project(self, constraints: tuple[QueryConstraint, ...], fields: list[str]) -> list[dict[str, Any]]:
        return [{k: record.get(k) for k in fields if k in record} for record in self.query(constraints)]


def _matches(record: dict[str, Any], constraint: QueryConstraint) -> bool:
    value = record.get(constraint.field)
    target = constraint.value
    if constraint.op == "eq":
        return value == target
    if constraint.op == "in":
        return value in set(target or [])
    if constraint.op == "contains":
        return str(target).casefold() in str(value or "").casefold()
    if constraint.op == "exists":
        return value not in (None, "")
    if constraint.op == "gte":
        return value is not None and value >= target
    if constraint.op == "lte":
        return value is not None and value <= target
    if constraint.op == "between":
        low, high = constraint.value
        return value is not None and low <= value <= high
    raise AssertionError(f"unsupported op in test backend: {constraint.op}")


def mock_records() -> list[dict[str, Any]]:
    return [
        {
            "contract_node_id": "c1",
            "ocid": "ocds-1",
            "buyer_name": "Alpha Council",
            "supplier_name": "BuildCo",
            "release_year": 2024,
            "tender_category": "works",
            "value_amount": "100.00",
            "value_is_additive": True,
        },
        {
            "contract_node_id": "c2",
            "ocid": "ocds-2",
            "buyer_name": "Alpha Council",
            "supplier_name": "RoadCo",
            "release_year": 2024,
            "tender_category": "works",
            "value_amount": "250.00",
            "value_is_additive": True,
        },
        {
            "contract_node_id": "c3",
            "ocid": "ocds-3",
            "buyer_name": "Beta NHS Trust",
            "supplier_name": "MedSupply",
            "release_year": 2025,
            "tender_category": "goods",
            "value_amount": "999.00",
            "value_is_additive": False,
        },
    ]


class TestReasoningRuntime(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = TabularRuntimeBackend(mock_records())

    def test_select_unique_factoid_passes(self) -> None:
        spec = RuntimeQuerySpec(
            spec_id="supplier-c3",
            question="Who supplied the Beta NHS Trust contract in 2025?",
            intent="factoid",
            constraints=(QueryConstraint("contract_node_id", "eq", "c3"),),
            answer_operation="select_unique",
            answer_field="supplier_name",
            answer_value_type="string",
            dedupe_key="contract_node_id",
        )

        result = execute_query_spec(self.backend, spec)

        self.assertTrue(result.passed)
        self.assertEqual(result.answer, "MedSupply")
        self.assertEqual(result.evidence.evidence_ids, ("c3",))

    def test_select_unique_rejects_multiple_answers(self) -> None:
        spec = RuntimeQuerySpec(
            spec_id="alpha-supplier",
            question="Who supplied Alpha Council works contracts in 2024?",
            intent="factoid",
            constraints=(
                QueryConstraint("buyer_name", "eq", "Alpha Council"),
                QueryConstraint("release_year", "eq", 2024),
            ),
            answer_operation="select_unique",
            answer_field="supplier_name",
            answer_value_type="string",
            dedupe_key="contract_node_id",
        )

        result = execute_query_spec(self.backend, spec)

        self.assertEqual(result.status, "multiple_answers")
        self.assertFalse(result.passed)

    def test_count_uses_complete_matching_rows(self) -> None:
        spec = RuntimeQuerySpec(
            spec_id="works-2024-count",
            question="How many works contracts were published in 2024?",
            intent="aggregation_count",
            constraints=(
                QueryConstraint("release_year", "eq", 2024),
                QueryConstraint("tender_category", "eq", "works"),
            ),
            answer_operation="count",
            answer_field="contract_node_id",
            answer_value_type="integer",
            dedupe_key="contract_node_id",
            requires_exhaustive_retrieval=True,
        )

        result = execute_query_spec(self.backend, spec)

        self.assertTrue(result.passed)
        self.assertEqual(result.answer, 2)
        self.assertEqual(set(result.evidence.evidence_ids), {"c1", "c2"})

    def test_sum_rejects_non_additive_rows(self) -> None:
        spec = RuntimeQuerySpec(
            spec_id="all-value-sum",
            question="What is the total value of all contracts?",
            intent="aggregation_sum",
            constraints=(QueryConstraint("contract_node_id", "exists"),),
            answer_operation="sum",
            answer_field="value_amount",
            answer_value_type="currency",
            dedupe_key="contract_node_id",
            requires_exhaustive_retrieval=True,
        )

        result = execute_query_spec(self.backend, spec)

        self.assertEqual(result.status, "incomplete_evidence")
        self.assertFalse(result.passed)
        self.assertIn("c3", str(result.checks))

    def test_sum_passes_for_additive_rows(self) -> None:
        spec = RuntimeQuerySpec(
            spec_id="works-value-sum",
            question="What is the total value of works contracts in 2024?",
            intent="aggregation_sum",
            constraints=(
                QueryConstraint("release_year", "eq", 2024),
                QueryConstraint("tender_category", "eq", "works"),
            ),
            answer_operation="sum",
            answer_field="value_amount",
            answer_value_type="currency",
            dedupe_key="contract_node_id",
            requires_exhaustive_retrieval=True,
        )

        result = execute_query_spec(self.backend, spec)

        self.assertTrue(result.passed)
        self.assertEqual(str(result.answer), "350.00")

    def test_constraint_conflict_rejected_before_query(self) -> None:
        spec = RuntimeQuerySpec(
            spec_id="conflicting-year",
            question="How many contracts were in 2024 and 2025?",
            intent="aggregation_count",
            constraints=(
                QueryConstraint("release_year", "eq", 2024),
                QueryConstraint("release_year", "eq", 2025),
            ),
            answer_operation="count",
            answer_field="contract_node_id",
            answer_value_type="integer",
        )

        result = execute_query_spec(self.backend, spec)

        self.assertEqual(result.status, "constraint_conflict")

    def test_answer_card_does_not_change_answer(self) -> None:
        spec = RuntimeQuerySpec(
            spec_id="works-2024-count-card",
            question="How many works contracts were published in 2024?",
            intent="aggregation_count",
            constraints=(QueryConstraint("release_year", "eq", 2024),),
            answer_operation="count",
            answer_field="contract_node_id",
            answer_value_type="integer",
        )
        result = execute_query_spec(self.backend, spec)

        card = build_answer_card(result)

        self.assertEqual(card.answer, result.answer)
        self.assertIn(str(result.answer), card.answer_text)


if __name__ == "__main__":
    unittest.main()
