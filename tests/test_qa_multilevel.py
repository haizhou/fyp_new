"""Multi-level QA generation: atom extraction, the deterministic surface gate, assembly, and a
stub-LLM rejection-sampling round trip. No KG or API needed."""

from __future__ import annotations

import sys
import unittest
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from procurement_graph.qa.multilevel import (
    check_surface, plan_bank_row, required_atoms, rewrite_messages, surface_row,
)

ROW = {
    "id": "extended_ops_0000",
    "subset": "extended_ops",
    "question": "Did MIDLANDS AND LANCASHIRE COMMISSIONING SUPPORT UNIT publish any services "
                "contract notices in 2022?",
    "answer_type": "boolean",
    "answer_operation": "exists",
    "expected_status": "answerable",
    "constraints": [
        {"field": "buyer_name", "op": "eq", "value": "MIDLANDS AND LANCASHIRE COMMISSIONING SUPPORT UNIT"},
        {"field": "release_year", "op": "eq", "value": 2022},
        {"field": "tender_category", "op": "eq", "value": "services"},
    ],
    "oracle_answer": True,
}


class TestAtoms(unittest.TestCase):
    def test_org_year_category_extracted(self):
        atoms = required_atoms(ROW)
        self.assertIn("MIDLANDS AND LANCASHIRE COMMISSIONING SUPPORT UNIT", atoms.required_texts)
        self.assertIn("services", atoms.required_texts)
        self.assertIn("2022", atoms.required_numbers)

    def test_hidden_guard_not_required(self):
        row = {**ROW, "constraints": ROW["constraints"]
               + [{"field": "value_is_additive", "op": "eq", "value": True, "visible_to_user": False}]}
        atoms = required_atoms(row)
        self.assertNotIn("True", atoms.required_texts)

    def test_cpv_and_date_become_numbers(self):
        row = {**ROW, "constraints": [{"field": "tender_cpv_id", "op": "eq", "value": "45000000"},
                                      {"field": "award_date_signed", "op": "eq", "value": "2024-12-05"}]}
        atoms = required_atoms(row)
        self.assertIn("45000000", atoms.required_numbers)
        self.assertIn("2024", atoms.required_numbers)

    def test_unanswerable_trigger_captured(self):
        row = {**ROW, "expected_status": "unsupported",
               "question": "What were the social value commitments for notices in 2022?"}
        self.assertEqual(required_atoms(row).unanswerable_trigger, "social value")


class TestSurfaceGate(unittest.TestCase):
    def setUp(self):
        self.atoms = required_atoms(ROW)

    def _check(self, surface, level=2, **kw):
        return check_surface(surface, self.atoms, ROW["question"], level=level, **kw)

    def test_good_paraphrase_accepted(self):
        verdict = self._check("During 2022, were any services contract notices put out by "
                              "MIDLANDS AND LANCASHIRE COMMISSIONING SUPPORT UNIT?")
        self.assertTrue(verdict.ok, verdict.reasons)

    def test_missing_org_is_checker_responsibility_at_l2(self):
        verdict = self._check("During 2022, were any services contract notices put out by the trust?")
        self.assertTrue(verdict.ok, verdict.reasons)
        strict = self._check("During 2022, were any services contract notices put out by the trust?",
                             level=3)
        self.assertIn("missing_text:MIDLANDS AND LANCASHIRE COMMISSIONING SUPPORT UNIT", strict.reasons)

    def test_new_number_rejected(self):
        verdict = self._check("Unlike the 37 works deals, did MIDLANDS AND LANCASHIRE COMMISSIONING "
                              "SUPPORT UNIT publish any services contract notices in 2022?")
        self.assertTrue(any(r.startswith("new_numbers") for r in verdict.reasons))

    def test_new_temporal_relation_rejected(self):
        row = {**ROW, "question": "How many notices were published by buyers who have awarded a contract to S1?",
               "constraints": [{"field": "buyer_name", "op": "in_subquery",
                                "value": {"resolve": "buyers_of_supplier", "supplier": "S1"}}]}
        atoms = required_atoms(row)
        verdict = check_surface("How many notices were published by buyers who later awarded a contract to S1?",
                                atoms, row["question"], level=2)
        self.assertIn("new_temporal_relation:later", verdict.reasons)

    def test_existing_temporal_relation_allowed(self):
        row = {**ROW, "question": "How many notices were published after 2022?",
               "constraints": [{"field": "release_year", "op": "gt", "value": 2022}]}
        atoms = required_atoms(row)
        verdict = check_surface("After 2022, how many notices were published?",
                                atoms, row["question"], level=2)
        self.assertNotIn("new_temporal_relation:after", verdict.reasons)

    def test_unchanged_template_rejected_at_l3_but_allowed_at_l2(self):
        # L3 is a paraphrase level (must differ); L2 is plan-generated (similarity legitimate)
        self.assertIn("not_actually_rewritten", self._check(ROW["question"], level=3).reasons)
        self.assertNotIn("not_actually_rewritten", self._check(ROW["question"], level=2).reasons)

    def test_two_questions_rejected(self):
        verdict = self._check("Did MIDLANDS AND LANCASHIRE COMMISSIONING SUPPORT UNIT publish any "
                              "services contract notices in 2022? Are you sure?")
        self.assertIn("not_a_single_question", verdict.reasons)

    def test_new_kg_org_rejected(self):
        class _Resolver:
            def resolve(self, mention):
                class _Hit:
                    source = "records_exact"
                return [_Hit()] if "Atos" in mention else []
        verdict = self._check("Just as Atos IT Services UK Limited did, did MIDLANDS AND LANCASHIRE "
                              "COMMISSIONING SUPPORT UNIT publish any services contract notices in 2022?",
                              level=3, org_resolver=_Resolver())
        self.assertTrue(any(r.startswith("new_org_mention") for r in verdict.reasons))

    def test_unanswerable_trigger_is_checker_responsibility_at_l2(self):
        row = {**ROW, "expected_status": "unsupported",
               "constraints": [{"field": "release_year", "op": "eq", "value": 2022}],
               "question": "What were the social value commitments for the 2022 notices?"}
        atoms = required_atoms(row)
        verdict = check_surface("What were the community commitments for the 2022 notices?",
                                atoms, row["question"], level=2)
        self.assertTrue(verdict.ok, verdict.reasons)
        strict = check_surface("What were the community commitments for the 2022 notices?",
                               atoms, row["question"], level=3)
        self.assertIn("missing_unanswerable_trigger:social value", strict.reasons)
        good = check_surface("For the 2022 notices, what social value commitments were recorded?",
                             atoms, row["question"], level=2)
        self.assertTrue(good.ok, good.reasons)


class TestAssemblyAndPrompts(unittest.TestCase):
    def test_plan_bank_row_is_language_free(self):
        bank = plan_bank_row(ROW)
        self.assertEqual(bank["plan_id"], "extended_ops_0000")
        self.assertNotIn("question", bank)
        self.assertEqual(bank["oracle_answer"], True)

    def test_surface_row_shares_oracle_and_ids(self):
        verdict = check_surface("In 2022, did MIDLANDS AND LANCASHIRE COMMISSIONING SUPPORT UNIT "
                                "put out any services contract notices?", required_atoms(ROW),
                                ROW["question"], level=2)
        row = surface_row(ROW, level=2, question="q?", origin="stub", variant=0, verdict=verdict)
        self.assertEqual(row["id"], "extended_ops_0000#L2a")
        self.assertEqual(row["plan_id"], "extended_ops_0000")
        self.assertEqual(row["oracle_answer"], ROW["oracle_answer"])
        l1 = surface_row(ROW, level=1, question=ROW["question"], origin="template")
        self.assertEqual(l1["id"], "extended_ops_0000#L1")

    def test_rewrite_messages_forbid_answering_and_list_atoms(self):
        system, user = rewrite_messages(ROW, required_atoms(ROW), level=3)
        self.assertIn("Return strict JSON", system)
        self.assertIn("MIDLANDS AND LANCASHIRE COMMISSIONING SUPPORT UNIT", user)
        self.assertIn("2022", user)
        self.assertIn("Do NOT answer the question.", user)


class _StubChat:
    """First variant violates the numeric gate, second passes -> exercises rejection sampling."""

    def __init__(self):
        self.calls = 0

    def complete_json(self, *, model, system, user):
        self.calls += 1
        bad = "Did the trust publish any services notices in 2022 after 37 meetings?"
        good = ("Looking at 2022, did MIDLANDS AND LANCASHIRE COMMISSIONING SUPPORT UNIT put out "
                "any services contract notices?")

        class _R:
            parsed = {"variants": [bad] if self.calls == 1 else [good]}
        return _R()


class TestRejectionSampling(unittest.TestCase):
    def test_stub_llm_round_trip(self):
        from build_multilevel_qa import rewrite_surfaces
        chat = _StubChat()
        accepted, rejected = rewrite_surfaces([ROW], level=2, chat=chat, model="stub", retries=2,
                                              variants=1, org_resolver=None, known_orgs=None)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["plan_id"], "extended_ops_0000")
        self.assertEqual(accepted[0]["level"], 2)
        self.assertTrue(rejected and rejected[0]["reasons"])
        self.assertEqual(chat.calls, 2)

    def test_checkpoint_resume_skips_completed_plan(self):
        from build_multilevel_qa import rewrite_surfaces

        class _GoodChat:
            def __init__(self):
                self.calls = 0

            def complete_json(self, *, model, system, user):
                self.calls += 1

                class _R:
                    parsed = {"variants": [
                        "Looking at 2022, did MIDLANDS AND LANCASHIRE COMMISSIONING SUPPORT UNIT "
                        "put out any services contract notices?"
                    ]}
                return _R()

        class _NoCallChat:
            def complete_json(self, *, model, system, user):  # pragma: no cover - should not run
                raise AssertionError("resume should skip completed plans")

        tmp_root = Path(__file__).parent.parent / "tmp" / "qa_multilevel_checkpoint_test"
        tmp_root.mkdir(parents=True, exist_ok=True)
        run_id = uuid.uuid4().hex
        accepted_path = tmp_root / f"{run_id}.surfaces.L2.jsonl"
        rejected_path = tmp_root / f"{run_id}.surfaces.L2.rejected.jsonl"

        chat = _GoodChat()
        accepted, _ = rewrite_surfaces(
            [ROW], level=2, chat=chat, model="stub", retries=0, variants=1,
            org_resolver=None, known_orgs=None, accepted_path=accepted_path,
            rejected_path=rejected_path, resume=False, workers=2, rpm=0,
        )
        self.assertEqual(len(accepted), 1)
        self.assertEqual(chat.calls, 1)
        self.assertEqual(len(accepted_path.read_text(encoding="utf-8").splitlines()), 1)

        accepted, rejected = rewrite_surfaces(
            [ROW], level=2, chat=_NoCallChat(), model="stub", retries=0, variants=1,
            org_resolver=None, known_orgs=None, accepted_path=accepted_path,
            rejected_path=rejected_path, resume=True, workers=2, rpm=0,
        )
        self.assertEqual(len(accepted), 1)
        self.assertEqual(rejected, [])


if __name__ == "__main__":
    unittest.main()
