"""Smoke tests for connecting QA benchmark logic to the real KG v0.1 files."""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from procurement_graph.qa.benchmark.executor import execute_answer_spec
from procurement_graph.qa.benchmark.gates import run_gate_a
from procurement_graph.qa.benchmark.kg_interface import ParquetKGQueryBackend
from procurement_graph.qa.benchmark.models import AnswerSpec, Constraint
from procurement_graph.qa.benchmark.reference_index import ReferenceKGIndex


ROOT = Path(__file__).resolve().parents[1]
KG_DIR = ROOT / "data" / "kg"


@unittest.skipUnless((KG_DIR / "nodes" / "contract_nodes.parquet").exists(), "real KG files are not built")
class TestRealKGBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = ParquetKGQueryBackend.from_directory(KG_DIR, include_evidence=False)
        cls.reference = ReferenceKGIndex.from_directory(KG_DIR)

    def test_reference_index_agrees_with_backend_on_real_data(self):
        # The independent re-derivation must equal the backend on good data, otherwise
        # Gate A completeness would raise false failures. Cover scalar + edge-derived
        # (supplier_count) constraint fields.
        for constraints in (
            (Constraint("release_year", "eq", 2025), Constraint("tender_category", "eq", "services")),
            (Constraint("release_year", "eq", 2024), Constraint("value_is_additive", "eq", True)),
            (Constraint("tender_category", "eq", "goods"), Constraint("supplier_count", "gte", 1)),
        ):
            backend_ids = {
                self.backend.record_id(row)
                for row in self.backend.query(constraints)
                if self.backend.record_id(row)
            }
            reference_ids = self.reference.matching_ids(constraints)
            self.assertEqual(reference_ids, backend_ids, msg=str(constraints))

    def test_exact_contract_spec_passes_gate_a(self):
        seed = self.backend.records_df[self.backend.records_df["value_is_additive"]].iloc[0].to_dict()
        spec = AnswerSpec(
            spec_id="real-contract-value-source",
            constraints=(Constraint("contract_node_id", "eq", seed["contract_node_id"]),),
            answer_operation="select_unique",
            answer_field="value_source",
            answer_value_type="string",
            sampled_evidence_ids=(seed["contract_node_id"],),
            logic_chain=("contract_node_id exact match", "select value_source"),
        )

        completeness, uniqueness = run_gate_a(self.backend, spec)

        self.assertTrue(completeness.passed)
        self.assertTrue(uniqueness.passed)
        self.assertIn(execute_answer_spec(self.backend, spec), {"award", "contract", "tender"})

    def test_feature_count_on_real_kg_is_deterministic(self):
        rows = self.backend.query(
            (
                Constraint("release_year", "eq", 2025),
                Constraint("tender_category", "eq", "services"),
            )
        )
        sampled_ids = tuple(sorted(row["contract_node_id"] for row in rows))
        spec = AnswerSpec(
            spec_id="real-2025-services-count",
            constraints=(
                Constraint("release_year", "eq", 2025),
                Constraint("tender_category", "eq", "services"),
            ),
            answer_operation="count",
            answer_field="contract_node_id",
            answer_value_type="integer",
            sampled_evidence_ids=sampled_ids,
            logic_chain=("release_year=2025", "tender_category=services", "count contracts"),
        )

        completeness, uniqueness = run_gate_a(self.backend, spec)

        self.assertTrue(completeness.passed)
        self.assertTrue(uniqueness.passed)
        self.assertEqual(execute_answer_spec(self.backend, spec), len(rows))

    def test_full_graph_query_performance_smoke(self):
        start = time.perf_counter()
        rows = self.backend.query(
            (
                Constraint("release_year", "eq", 2025),
                Constraint("tender_category", "eq", "services"),
                Constraint("value_is_additive", "eq", True),
            )
        )
        elapsed = time.perf_counter() - start

        self.assertGreater(len(rows), 0)
        self.assertLess(elapsed, 10.0)


class TestConstraintTranslation(unittest.TestCase):
    """Runtime -> KG constraint translation (no KG load needed; pure function)."""

    def _translate(self, field, op, value):
        from procurement_graph.reasoning.kg_backend import _translate
        from procurement_graph.reasoning.models import QueryConstraint

        return [(c.field, c.op, c.value) for c in _translate(QueryConstraint(field, op, value))]

    def test_date_only_eq_on_timestamp_field_becomes_contains(self):
        # regression for hard100 factoid_0021: award_date_signed eq '2021-09-28' could never
        # eq-match the stored ISO timestamp; translate to a by-day contains match.
        self.assertEqual(self._translate("award_date_signed", "eq", "2021-09-28"),
                         [("award_date_signed", "contains", "2021-09-28")])

    def test_full_timestamp_eq_is_unchanged(self):
        self.assertEqual(self._translate("award_date_signed", "eq", "2021-09-28T00:00:00+01:00"),
                         [("award_date_signed", "eq", "2021-09-28T00:00:00+01:00")])

    def test_non_timestamp_field_eq_is_unchanged(self):
        self.assertEqual(self._translate("release_year", "eq", 2025), [("release_year", "eq", 2025)])

    def test_between_splits_into_gte_lte(self):
        self.assertEqual(self._translate("release_year", "between", [2023, 2025]),
                         [("release_year", "gte", 2023), ("release_year", "lte", 2025)])


class TestRecordsOrgResolverVariants(unittest.TestCase):
    """The KG stores org names unnormalised, so one org can appear under several case variants.

    Regression for bridge_join_0902: the resolver returned the first-seen variant ('... upon ...',
    13 rows) for a mention that exactly matched another variant ('... Upon ...', 177 rows), so the
    bridge hop-1 anchored on the wrong subset. resolve() must prefer the mention's exact surface
    form, and fall back to the variant covering the most rows.
    """

    def _resolver(self):
        import pandas as pd
        from procurement_graph.reasoning.kg_backend import RecordsOrgResolver

        class _Stub:
            records_df = pd.DataFrame({
                "buyer_name": ["Newcastle upon Tyne"] * 2 + ["Newcastle Upon Tyne"] * 5,
                "supplier_name": ["S"] * 7,
            })

        return RecordsOrgResolver(_Stub())

    def test_exact_surface_form_wins(self):
        [hit] = self._resolver().resolve("Newcastle upon Tyne")
        self.assertEqual(hit.linked_id, "Newcastle upon Tyne")
        self.assertEqual(hit.source, "records_exact")

    def test_unseen_casing_maps_to_highest_coverage_variant(self):
        [hit] = self._resolver().resolve("NEWCASTLE UPON TYNE")
        self.assertEqual(hit.linked_id, "Newcastle Upon Tyne")  # 5 rows beats 2 rows


if __name__ == "__main__":
    unittest.main()
