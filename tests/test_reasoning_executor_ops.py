"""Executor extensions specified by the targeted-v2 benchmark: exists / argmax / argmin /
distinct_set / top_k (single-hop reduction ops added to the shared executor). compare and
in_subquery are intentionally NOT here — they need the decomposition planner."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from procurement_graph.reasoning import QueryConstraint, RuntimeQuerySpec, ground_spec
from procurement_graph.reasoning.executor import execute_query_spec
from tests.test_reasoning_runtime import TabularRuntimeBackend


def _records():
    return [
        {"contract_node_id": "c1", "release_year": 2024, "tender_category": "services",
         "buyer_name": "B1", "supplier_name": "S1", "value_amount": "100", "value_is_additive": True},
        {"contract_node_id": "c2", "release_year": 2024, "tender_category": "services",
         "buyer_name": "B1", "supplier_name": "S2", "value_amount": "300", "value_is_additive": True},
        {"contract_node_id": "c3", "release_year": 2024, "tender_category": "services",
         "buyer_name": "B2", "supplier_name": "S1", "value_amount": "50", "value_is_additive": True},
    ]


class TestExecutorOps(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = TabularRuntimeBackend(_records())
        self.allowed = frozenset(self.backend.fields())

    def _run(self, op, *, answer_field="", value_type="string", sort_field="", metadata=None,
             constraints=(("release_year", "eq", 2024), ("tender_category", "eq", "services"))):
        spec = RuntimeQuerySpec(
            spec_id="s", question="q", intent="x",
            constraints=tuple(QueryConstraint(f, o, v) for f, o, v in constraints),
            answer_operation=op, answer_field=answer_field, answer_value_type=value_type,
            sort_field=sort_field, requires_exhaustive_retrieval=True, metadata=metadata or {},
        )
        grounded = ground_spec(spec, allowed_fields=self.allowed)
        self.assertTrue(grounded.ok, grounded.reason)
        return execute_query_spec(self.backend, grounded.spec)

    def test_exists_true_and_false(self) -> None:
        self.assertEqual(self._run("exists").answer, True)
        empty = self._run("exists", constraints=(("release_year", "eq", 2099),))
        self.assertEqual(empty.status, "passed")
        self.assertEqual(empty.answer, False)

    def test_argmax_returns_highest_value_contract(self) -> None:
        result = self._run("argmax", answer_field="contract_node_id", sort_field="value_amount")
        self.assertEqual(result.status, "passed")
        self.assertEqual(result.answer, "c2")

    def test_argmin_returns_lowest_value_contract(self) -> None:
        self.assertEqual(self._run("argmin", answer_field="contract_node_id", sort_field="value_amount").answer, "c3")

    def test_distinct_set_of_suppliers(self) -> None:
        result = self._run("distinct_set", answer_field="supplier_name")
        self.assertEqual(result.status, "passed")
        self.assertEqual(result.answer, ["S1", "S2"])

    def test_distinct_set_requires_answer_field(self) -> None:
        grounded = ground_spec(RuntimeQuerySpec(spec_id="s", question="q", intent="x", constraints=(),
                                                answer_operation="distinct_set", answer_field="",
                                                answer_value_type="string", requires_exhaustive_retrieval=True),
                               allowed_fields=self.allowed)
        self.assertFalse(grounded.ok)

    def test_top_k_buyers_by_count(self) -> None:
        result = self._run("top_k", metadata={"group_by_field": "buyer_name", "k": 2, "metric": "count"})
        self.assertEqual(result.status, "passed")
        self.assertEqual(result.answer, [["B1", 2], ["B2", 1]])

    def test_top_k_by_sum_gets_additive_guard(self) -> None:
        spec = self._spec_top_k("buyer_name")
        spec = RuntimeQuerySpec(**{**spec.__dict__, "metadata": {"group_by_field": "buyer_name", "k": 2, "metric": "sum"}})
        grounded = ground_spec(spec, allowed_fields=self.allowed)
        self.assertTrue(grounded.ok, grounded.reason)
        self.assertTrue(any(c.field == "value_is_additive" and c.op == "eq" and c.value is True
                            for c in grounded.spec.constraints))

    def test_top_k_by_sum_rejects_non_additive_population_without_guard(self) -> None:
        backend = TabularRuntimeBackend(_records() + [
            {"contract_node_id": "ceiling", "release_year": 2024, "tender_category": "services",
             "buyer_name": "B3", "supplier_name": "S3", "value_amount": "999999999",
             "value_is_additive": False},
        ])
        spec = RuntimeQuerySpec(spec_id="s", question="q", intent="top_k",
                                constraints=(QueryConstraint("release_year", "eq", 2024),),
                                answer_operation="top_k", answer_field="", answer_value_type="string",
                                requires_exhaustive_retrieval=True,
                                metadata={"group_by_field": "buyer_name", "k": 3, "metric": "sum"})
        result = execute_query_spec(backend, spec)
        self.assertEqual(result.status, "incomplete_evidence")
        self.assertTrue(any(c.get("check") == "sum_additive_safety" and not c.get("passed")
                            for c in result.checks))

    def test_top_k_unknown_group_field_rejected(self) -> None:
        grounded = ground_spec(self._spec_top_k("not_a_field"), allowed_fields=self.allowed)
        self.assertFalse(grounded.ok)

    def _spec_top_k(self, group_by):
        return RuntimeQuerySpec(spec_id="s", question="q", intent="x",
                                constraints=(QueryConstraint("release_year", "eq", 2024),),
                                answer_operation="top_k", answer_field="", answer_value_type="string",
                                requires_exhaustive_retrieval=True, metadata={"group_by_field": group_by, "k": 3})

    def test_compare_still_unsupported_until_decomposition(self) -> None:
        grounded = ground_spec(RuntimeQuerySpec(spec_id="s", question="q", intent="x", constraints=(),
                                                answer_operation="compare", answer_field="", answer_value_type="string",
                                                requires_exhaustive_retrieval=True), allowed_fields=self.allowed)
        self.assertFalse(grounded.ok)
        self.assertIn("decomposition", grounded.reason)


class _RecordingProjectBackend(TabularRuntimeBackend):
    def __init__(self, records):
        super().__init__(records)
        self.project_fields: list[list[str]] = []

    def project(self, constraints, fields):
        self.project_fields.append(list(fields))
        return super().project(constraints, fields)


class TestExecutorPerformanceGuards(unittest.TestCase):
    def test_distinct_set_uses_projection(self):
        backend = _RecordingProjectBackend(_records())
        spec = RuntimeQuerySpec(spec_id="s", question="q", intent="set",
                                constraints=(QueryConstraint("release_year", "eq", 2024),),
                                answer_operation="distinct_set", answer_field="supplier_name",
                                answer_value_type="string", requires_exhaustive_retrieval=True)
        result = execute_query_spec(backend, spec)
        self.assertEqual(result.status, "passed")
        self.assertTrue(backend.project_fields)
        self.assertIn("supplier_name", backend.project_fields[0])
        self.assertNotIn("value_amount", backend.project_fields[0])

    def test_evidence_rows_are_capped_but_count_is_exact(self):
        rows = [
            {"contract_node_id": f"c{i}", "release_year": 2024, "supplier_name": f"S{i % 3}"}
            for i in range(250)
        ]
        backend = TabularRuntimeBackend(rows)
        spec = RuntimeQuerySpec(spec_id="s", question="q", intent="set",
                                constraints=(QueryConstraint("release_year", "eq", 2024),),
                                answer_operation="distinct_set", answer_field="supplier_name",
                                answer_value_type="string", requires_exhaustive_retrieval=True)
        result = execute_query_spec(backend, spec)
        self.assertEqual(result.evidence.fields["evidence_count"], 250)
        self.assertEqual(len(result.evidence.rows), 200)
        self.assertTrue(result.evidence.fields["rows_capped"])


def _predicate_records():
    return [
        {"contract_node_id": "p1", "tender_title": "Alpha Framework", "tender_category": "services",
         "value_amount": "5000000", "value_is_additive": True, "release_year": 2026, "tender_cpv_id": "45000000",
         "award_date_signed": "2025-06-01T00:00:00+01:00"},
        {"contract_node_id": "p2", "tender_title": "Beta Supply", "tender_category": "goods",
         "value_amount": "8000000", "value_is_additive": True, "release_year": 2026, "tender_cpv_id": "33600000",
         "award_date_signed": "2025-04-10T00:00:00+01:00"},
    ]


class TestPredicateOp(unittest.TestCase):
    def setUp(self):
        self.backend = TabularRuntimeBackend(_predicate_records())
        self.allowed = frozenset(self.backend.fields())

    def _run(self, constraints, metadata):
        spec = RuntimeQuerySpec(spec_id="s", question="q", intent="boolean",
                                constraints=tuple(QueryConstraint(*c) for c in constraints),
                                answer_operation="predicate", answer_field="value_amount",
                                answer_value_type="boolean", requires_exhaustive_retrieval=True, metadata=metadata)
        grounded = ground_spec(spec, allowed_fields=self.allowed)
        self.assertTrue(grounded.ok, grounded.reason)
        return execute_query_spec(self.backend, grounded.spec)

    def test_field_equality_true(self):
        r = self._run((("tender_title", "eq", "Alpha Framework"),),
                      {"predicate_subject": "field", "predicate_field": "tender_category",
                       "comparator": "eq", "threshold": "services", "threshold_type": "string"})
        self.assertEqual(r.answer, True)

    def test_field_equality_false(self):
        r = self._run((("tender_title", "eq", "Beta Supply"),),
                      {"predicate_subject": "field", "predicate_field": "tender_category",
                       "comparator": "eq", "threshold": "services", "threshold_type": "string"})
        self.assertEqual(r.answer, False)

    def test_date_relation_after(self):
        r = self._run((("tender_title", "eq", "Alpha Framework"),),
                      {"predicate_subject": "field", "predicate_field": "award_date_signed",
                       "comparator": "after", "threshold": "2025-05-01", "threshold_type": "date"})
        self.assertEqual(r.answer, True)   # 2025-06-01 after 2025-05-01

    def test_numeric_threshold_sum(self):
        r = self._run((("release_year", "eq", 2026), ("value_is_additive", "eq", True)),
                      {"predicate_subject": "sum", "comparator": "gt", "threshold": 10_000_000, "threshold_type": "number"})
        self.assertEqual(r.answer, True)   # 5M + 8M = 13M > 10M

    def test_predicate_sum_gets_additive_guard(self):
        spec = RuntimeQuerySpec(spec_id="s", question="q", intent="boolean",
                                constraints=(QueryConstraint("release_year", "eq", 2026),),
                                answer_operation="predicate", answer_field="",
                                answer_value_type="boolean", requires_exhaustive_retrieval=True,
                                metadata={"predicate_subject": "sum", "comparator": "gt",
                                          "threshold": 10_000_000, "threshold_type": "number"})
        grounded = ground_spec(spec, allowed_fields=self.allowed)
        self.assertTrue(grounded.ok, grounded.reason)
        self.assertEqual(grounded.spec.answer_field, "value_amount")
        self.assertTrue(any(c.field == "value_is_additive" and c.op == "eq" and c.value is True
                            for c in grounded.spec.constraints))

    def test_predicate_sum_rejects_non_additive_population_without_guard(self):
        backend = TabularRuntimeBackend(_predicate_records() + [
            {"contract_node_id": "p3", "tender_title": "Gamma Ceiling", "tender_category": "goods",
             "value_amount": "999999999", "value_is_additive": False, "release_year": 2026,
             "tender_cpv_id": "33600000", "award_date_signed": "2025-04-10T00:00:00+01:00"},
        ])
        spec = RuntimeQuerySpec(spec_id="s", question="q", intent="boolean",
                                constraints=(QueryConstraint("release_year", "eq", 2026),),
                                answer_operation="predicate", answer_field="value_amount",
                                answer_value_type="boolean", requires_exhaustive_retrieval=True,
                                metadata={"predicate_subject": "sum", "comparator": "gt",
                                          "threshold": 10_000_000, "threshold_type": "number"})
        result = execute_query_spec(backend, spec)
        self.assertEqual(result.status, "incomplete_evidence")
        self.assertTrue(any(c.get("check") == "sum_additive_safety" and not c.get("passed")
                            for c in result.checks))

    def test_planner_recognises_all_three(self):
        from procurement_graph.reasoning import ReasoningPipeline
        from procurement_graph.reasoning.planner_decomposition import DecompositionAwarePlanner
        pipe = ReasoningPipeline(backend=self.backend, planner=DecompositionAwarePlanner())
        self.assertEqual(pipe.run('Was the contract titled "Alpha Framework" categorized as services?').answer_card.answer, True)
        self.assertEqual(pipe.run('Was the award for "Alpha Framework" signed after 1 May 2025?').answer_card.answer, True)
        self.assertEqual(pipe.run("Was the total value of goods notices published in 2026 above GBP 5 million?").answer_card.answer, True)


if __name__ == "__main__":
    unittest.main()
