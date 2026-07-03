"""Unit tests for the QA benchmark framework using a mock KG."""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from procurement_graph.qa.benchmark.executor import AnswerExecutionError, execute_answer_spec
from procurement_graph.qa.benchmark.gates import run_gate_a
from procurement_graph.qa.benchmark.generation import contract_field_spec, feature_set_spec
from procurement_graph.qa.benchmark.kg_interface import TabularQueryBackend
from procurement_graph.qa.benchmark.mock_data import mock_contract_records
from procurement_graph.qa.benchmark.models import AnswerSpec, Constraint
from procurement_graph.qa.benchmark.pipeline import (
    BenchmarkPipeline,
    ExactMatchSemanticVerifier,
    PromptOnlyQuestionGenerator,
)
from procurement_graph.qa.benchmark.reference_index import ReferenceKGIndex, UnsupportedReferenceOp
from procurement_graph.qa.benchmark.stage1 import _completeness_gate


class TestQABenchmarkPipeline(unittest.TestCase):
    def setUp(self):
        self.backend = TabularQueryBackend(mock_contract_records(), id_field="record_id")

    def test_gate_a_passes_complete_unique_spec(self):
        spec = contract_field_spec(
            self.backend,
            spec_id="supplier-for-contract",
            contract_id_field="ocid",
            contract_id="ocds-mock-001",
            answer_field="supplier_name",
            answer_value_type="string",
        )

        completeness, uniqueness = run_gate_a(self.backend, spec)

        self.assertTrue(completeness.passed)
        self.assertTrue(uniqueness.passed)
        self.assertEqual(execute_answer_spec(self.backend, spec), "MedSupply Ltd")

    def test_gate_a_completeness_fails_when_subgraph_missed_full_graph_record(self):
        spec = AnswerSpec(
            spec_id="alpha-supplier-incomplete",
            constraints=(Constraint("buyer_name", "eq", "Alpha NHS Trust"),),
            answer_operation="count",
            answer_field="ocid",
            answer_value_type="integer",
            sampled_evidence_ids=("r1",),
            logic_chain=("buyer_name=Alpha NHS Trust", "count contracts"),
        )

        completeness, uniqueness = run_gate_a(self.backend, spec)

        self.assertFalse(completeness.passed)
        self.assertIn("r2", completeness.metrics["missing_from_sample"])
        self.assertTrue(uniqueness.passed)

    def test_gate_a_uniqueness_fails_for_ambiguous_answer_field(self):
        spec = AnswerSpec(
            spec_id="alpha-supplier-ambiguous",
            constraints=(Constraint("buyer_name", "eq", "Alpha NHS Trust"),),
            answer_operation="select_unique",
            answer_field="supplier_name",
            answer_value_type="string",
            sampled_evidence_ids=("r1", "r2"),
            logic_chain=("buyer_name=Alpha NHS Trust", "select supplier_name"),
        )

        completeness, uniqueness = run_gate_a(self.backend, spec)

        self.assertTrue(completeness.passed)
        self.assertFalse(uniqueness.passed)
        with self.assertRaises(AnswerExecutionError):
            execute_answer_spec(self.backend, spec)

    def test_pipeline_writes_example_only_when_gate_b_passes(self):
        spec = contract_field_spec(
            self.backend,
            spec_id="award-value-for-contract",
            contract_id_field="ocid",
            contract_id="ocds-mock-003",
            answer_field="award_value",
            answer_value_type="integer",
        )
        pipeline = BenchmarkPipeline(
            backend=self.backend,
            question_generator=PromptOnlyQuestionGenerator(),
            semantic_verifier=ExactMatchSemanticVerifier(),
        )

        example = pipeline.build_one(spec)

        self.assertIsNotNone(example)
        self.assertEqual(example.golden_answer, 7500)
        self.assertEqual(example.evidence_ids, ("r3",))
        self.assertTrue(all(report.passed for report in example.gate_reports))

    def test_pipeline_discards_example_when_independent_verifier_disagrees(self):
        spec = contract_field_spec(
            self.backend,
            spec_id="supplier-for-contract",
            contract_id_field="ocid",
            contract_id="ocds-mock-004",
            answer_field="supplier_name",
            answer_value_type="string",
        )
        pipeline = BenchmarkPipeline(
            backend=self.backend,
            question_generator=PromptOnlyQuestionGenerator(),
            semantic_verifier=ExactMatchSemanticVerifier(predicted_answer="Wrong Supplier"),
        )

        self.assertIsNone(pipeline.build_one(spec))

    def test_feature_set_start_can_count_multiple_matching_contracts(self):
        spec = feature_set_spec(
            self.backend,
            spec_id="alpha-contract-count",
            constraints=(Constraint("buyer_name", "eq", "Alpha NHS Trust"),),
            answer_operation="count",
            answer_field="ocid",
            answer_value_type="integer",
            logic_label="contracts bought by Alpha NHS Trust",
        )

        completeness, uniqueness = run_gate_a(self.backend, spec)

        self.assertTrue(completeness.passed)
        self.assertTrue(uniqueness.passed)
        self.assertEqual(spec.sampled_evidence_ids, ("r1", "r2"))
        self.assertEqual(execute_answer_spec(self.backend, spec), 2)

    def test_feature_set_start_can_sum_over_filtered_contracts(self):
        spec = feature_set_spec(
            self.backend,
            spec_id="uki-award-value-sum",
            constraints=(Constraint("region", "eq", "UKI"),),
            answer_operation="sum",
            answer_field="award_value",
            answer_value_type="currency",
            logic_label="contracts in UKI",
        )

        completeness, uniqueness = run_gate_a(self.backend, spec)

        self.assertTrue(completeness.passed)
        self.assertTrue(uniqueness.passed)
        self.assertEqual(str(execute_answer_spec(self.backend, spec)), "3500")


class _RecordingMockClient:
    """Offline stand-in for ChatClient that records prompts and returns a canned reply."""

    def __init__(self, response):
        self.response = response
        self.calls = []  # (system, user) pairs

    def complete_json(self, *, model, system, user):
        from procurement_graph.qa.benchmark.chat import ChatResult

        self.calls.append((system, user))
        return ChatResult(parsed=self.response, raw_text="{}", model=model, usage={}, attempts=1)


class TestStage2LLM(unittest.TestCase):
    """Lock in the two Stage 2 invariants: leak-free generation prompt + Gate B logic."""

    def _factoid_spec(self):
        return AnswerSpec(
            spec_id="f1",
            constraints=(Constraint("contract_node_id", "eq", "contract:abc:1"),),
            answer_operation="select_unique",
            answer_field="buyer_name",
            answer_value_type="string",
        )

    def _sum_spec(self):
        return AnswerSpec(
            spec_id="s1",
            constraints=(
                Constraint("release_year", "eq", 2024),
                Constraint("tender_cpv_id", "eq", "79623000"),
                Constraint("value_is_additive", "eq", True),
            ),
            answer_operation="sum",
            answer_field="value_amount",
            answer_value_type="currency",
        )

    def test_generation_prompt_never_leaks_the_answer(self):
        from procurement_graph.qa.benchmark.question_gen import LLMQuestionGenerator

        # Factoid: the golden buyer name must not appear anywhere in the prompt.
        client = _RecordingMockClient({"question": "Who is the buyer for this contract?", "names_entity": False})
        gen = LLMQuestionGenerator(client=client, model="gpt-5.4-nano")
        outcome = gen.generate(self._factoid_spec())
        self.assertTrue(outcome.ok)
        system, user = client.calls[0]
        # The answer VALUE must never appear (the answer_field NAME legitimately may).
        self.assertNotIn("Fife Council", system + user)

        # Aggregation: the golden total must not appear in the prompt.
        sum_client = _RecordingMockClient({"question": "What is the total value of those contracts?"})
        sum_gen = LLMQuestionGenerator(client=sum_client, model="gpt-5.4-nano")
        sum_gen.generate(self._sum_spec())
        _, sum_user = sum_client.calls[0]
        self.assertNotIn("1912915", sum_user)

    def test_factoid_anchor_prompt_excludes_id_and_answer(self):
        from procurement_graph.qa.benchmark.prompts import build_factoid_generation_messages

        # Asking for the buyer: the anchor must carry natural attributes but neither an
        # internal id nor the buyer (answer) value.
        anchor = {
            "supplier": "Bellcare Ltd",
            "cpv_code": "85000000",
            "cpv_description": "health and social work services",
            "year": 2023,
            "category": "services",
        }
        system, user = build_factoid_generation_messages("buyer_name", anchor)
        blob = system + user
        self.assertNotIn("contract_node_id", blob)
        self.assertNotIn("contract:ocds", blob)
        self.assertIn("Bellcare Ltd", user)
        self.assertNotIn("buyer", anchor)  # answer field is excluded from the anchor

    def test_gate_b_recompute_pass_and_fail(self):
        from procurement_graph.qa.benchmark.gate_b import LLMGateBVerifier

        spec = self._factoid_spec()
        evidence = [{"contract_node_id": "contract:abc:1", "buyer_name": "Fife Council"}]

        ok = LLMGateBVerifier(_RecordingMockClient({"answer": "Fife Council", "reason": "in evidence"}), "grok")
        self.assertTrue(ok.verify_recompute(spec, "Who is the buyer?", evidence, "Fife Council").verified)

        wrong = LLMGateBVerifier(_RecordingMockClient({"answer": "Other Council", "reason": "x"}), "grok")
        self.assertFalse(wrong.verify_recompute(spec, "Who is the buyer?", evidence, "Fife Council").verified)

        unsure = LLMGateBVerifier(_RecordingMockClient({"answer": "uncertain", "reason": "ambiguous"}), "grok")
        self.assertFalse(unsure.verify_recompute(spec, "Who is the buyer?", evidence, "Fife Council").verified)

    def test_gate_b_faithfulness_pass_and_fail(self):
        from procurement_graph.qa.benchmark.gate_b import LLMGateBVerifier

        spec = self._sum_spec()
        good = {
            "operation": "sum",
            "filters": [{"field": "release_year", "value": 2024}, {"field": "tender_cpv_id", "value": "79623000"}],
            "ambiguous": False,
        }
        self.assertTrue(
            LLMGateBVerifier(_RecordingMockClient(good), "grok").verify_faithfulness(spec, "total value ...?").verified
        )

        wrong_op = {**good, "operation": "count"}
        self.assertFalse(
            LLMGateBVerifier(_RecordingMockClient(wrong_op), "grok").verify_faithfulness(spec, "how many ...?").verified
        )

        ambiguous = {**good, "ambiguous": True}
        self.assertFalse(
            LLMGateBVerifier(_RecordingMockClient(ambiguous), "grok").verify_faithfulness(spec, "vague ...?").verified
        )

    def test_gate_b_rejects_hidden_answer_changing_constraints(self):
        from procurement_graph.qa.benchmark.gate_b import LLMGateBVerifier

        spec = AnswerSpec(
            spec_id="bad-hidden",
            constraints=(
                Constraint("release_year", "eq", 2023),
                Constraint("tender_category", "eq", "goods"),
                Constraint("supplier_count", "eq", 9),
            ),
            answer_operation="count",
            answer_field="contract_node_id",
            answer_value_type="integer",
        )
        parsed = {
            "operation": "count",
            "filters": [
                {"field": "release_year", "value": 2023},
                {"field": "tender_category", "value": "goods"},
            ],
            "ambiguous": False,
        }

        outcome = LLMGateBVerifier(_RecordingMockClient(parsed), "grok").verify_faithfulness(
            spec, "How many goods contracts were published in 2023?"
        )

        self.assertFalse(outcome.verified)
        self.assertIn("hidden semantic constraint", outcome.reason)

    def test_question_id_leak_guard_allows_only_cpv_like_ids(self):
        from procurement_graph.qa.benchmark.stage2 import _question_id_leak_reason

        self.assertEqual(_question_id_leak_reason("How many contracts are under CPV 45000000?"), "")
        self.assertIn(
            "forbidden_identifier",
            _question_id_leak_reason("Who is the buyer for contract:ocds-h6vhtk-04dafb:054242-2025-4?"),
        )
        self.assertIn(
            "forbidden_identifier",
            _question_id_leak_reason("Who is the supplier for OCDS notice ocds-h6vhtk-04dafb?"),
        )
        self.assertIn(
            "forbidden_identifier",
            _question_id_leak_reason("Which buyer is linked to canonical_id GB-COH-12345678?"),
        )
        self.assertIn(
            "forbidden_identifier",
            _question_id_leak_reason("Who is the supplier for the contract with reference number ABC-123?"),
        )
        self.assertIn(
            "forbidden_identifier",
            _question_id_leak_reason("Who is the buyer for company number 12345678?"),
        )

    def test_strict_prompt_variant_requires_all_filters(self):
        from procurement_graph.qa.benchmark.prompts import build_generation_messages

        system, user = build_generation_messages(self._sum_spec(), variant="strict_filters")

        self.assertIn("MUST be explicitly verbalised", system)
        self.assertIn('"prompt_variant": "strict_filters"', user)


class TestIndependentCompletenessGate(unittest.TestCase):
    """The Stage 1 completeness gate must cross-check against an independent
    re-derivation, so an incomplete evidence set is rejected -- which the old
    same-query gate could not detect."""

    def setUp(self):
        self.reference = ReferenceKGIndex(
            contracts=pd.DataFrame(
                {
                    "contract_node_id": ["c1", "c2", "c3", "c4"],
                    "release_year": [2025, 2025, 2024, 2025],
                    "tender_category": ["goods", "services", "goods", "goods"],
                    "supplier_count": [1, 2, 1, 0],
                    "buyer_count": [1, 1, 1, 1],
                }
            )
        )
        self.constraints = (
            Constraint("release_year", "eq", 2025),
            Constraint("tender_category", "eq", "goods"),
        )

    def test_matching_ids_is_independent_of_sampler(self):
        self.assertEqual(self.reference.matching_ids(self.constraints), {"c1", "c4"})

    def test_complete_evidence_passes(self):
        ids = {"c1", "c4"}
        gate = _completeness_gate(self.reference, self.constraints, ids, ids, duplicate_rows=0)
        self.assertEqual(gate.effective_status, "PASS")

    def test_incomplete_evidence_fails_against_reference(self):
        ids = {"c1"}  # c4 silently dropped by a hypothetical backend bug
        gate = _completeness_gate(self.reference, self.constraints, ids, ids, duplicate_rows=0)
        self.assertEqual(gate.effective_status, "FAIL")
        self.assertIn("c4", gate.metrics["missing_vs_reference"])

    def test_duplicate_rows_fail(self):
        ids = {"c1", "c4"}
        gate = _completeness_gate(self.reference, self.constraints, ids, ids, duplicate_rows=3)
        self.assertEqual(gate.effective_status, "FAIL")

    def test_unverifiable_op_warns_not_silent_pass(self):
        constraints = (Constraint("tender_category", "contains", "good"),)
        with self.assertRaises(UnsupportedReferenceOp):
            self.reference.matching_ids(constraints)
        ids = {"c1", "c4"}
        gate = _completeness_gate(self.reference, constraints, ids, ids, duplicate_rows=0)
        self.assertEqual(gate.effective_status, "WARN")


if __name__ == "__main__":
    unittest.main()
