"""Multi-level QA generation: atom extraction, the deterministic surface gate, assembly, and a
stub-LLM rejection-sampling round trip. No KG or API needed."""

from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from procurement_graph.qa.multilevel import (
    bridge_drift_reasons, check_surface, plan_bank_row, required_atoms, rewrite_messages, surface_row,
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

    def test_buyers_of_supplier_bridge_drift_rejected(self):
        row = {**ROW, "question": "How many notices were published by buyers who have awarded a contract to S1?",
               "constraints": [{"field": "buyer_name", "op": "in_subquery",
                                "value": {"resolve": "buyers_of_supplier", "supplier": "S1"}}]}
        bad = "How many contract notices did buyers publish where the contract was awarded to S1?"
        self.assertIn("bridge_relation_drift:buyers_of_supplier_as_direct_supplier_filter",
                      bridge_drift_reasons(bad, row))
        good = "How many contract notices were published by buyers that have awarded a contract to S1?"
        self.assertEqual(bridge_drift_reasons(good, row), ())

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


class _AutoStubChat:
    """Deterministic stand-in for the generator+checker models used by ``run_l2_auto``.

    The generator embeds a variant marker before the trailing '?'; the checker accepts once the
    marker index reaches a per-plan threshold read off ``accept_from`` (keyed by the plan's L1
    question, which the real prompts pass through verbatim). A threshold higher than
    ``max_candidates_per_plan`` can ever reach simulates a permanently unresolvable plan. No
    network calls -- this only exercises the round/ceiling control flow in `build_multilevel_qa`.
    """

    def __init__(self, accept_from: dict[str, int], *, initial_index: dict[str, int] | None = None):
        self.accept_from = accept_from
        # seeds the per-plan counter for candidates that were written to disk directly (bypassing
        # this stub) so a freshly-generated candidate's embedded index still matches the script's
        # own running `offset` for that plan.
        self._gen_counts: dict[str, int] = dict(initial_index or {})
        self.gen_calls = 0
        self.chk_calls = 0

    def complete_json(self, *, model: str, system: str, user: str):
        payload = json.loads(user)
        if model == "gen":
            self.gen_calls += 1
            l1 = payload["task"]["l1_question"]
            idx = self._gen_counts.get(l1, 0)
            self._gen_counts[l1] = idx + 1
            text = re.sub(r"\?\s*$", f" [[v{idx}]]?", l1)

            class _R:
                parsed = {"variants": [text]}
            return _R()

        self.chk_calls += 1
        question, source = payload["question"], payload["source_question"]
        match = re.search(r"\[\[v(\d+)\]\]", question)
        idx = int(match.group(1)) if match else -1
        ok = idx >= self.accept_from.get(source, 10**9)

        class _R:
            parsed = {"matches_original_plan": ok, "can_derive_reference_answer": ok,
                      "same_meaning_as_source_question": ok, "preserves_unanswerable_status": True,
                      "mismatch_reason": None if ok else "stub_reject"}
        return _R()


def _auto_row(name: str, *, question: str) -> dict[str, Any]:
    return {
        "id": name, "subset": "test", "question": question,
        "answer_type": "count", "answer_operation": "count", "expected_status": "answerable",
        "constraints": [{"field": "release_year", "op": "eq", "value": 2024, "visible_to_user": True}],
        "oracle_answer": 3,
    }


class TestL2AutoDriver(unittest.TestCase):
    """`run_l2_auto` is the unattended overnight driver: it must converge plans that need more
    than the starting candidate budget, and must STOP (not spin forever) on plans that never
    pass, reporting them by id with their reject reasons."""

    def test_ceiling_raise_and_stuck_plan_report(self):
        from build_multilevel_qa import run_l2_auto

        rows = [
            _auto_row("easy", question="How many contract notices were published for the easy plan in 2024?"),
            _auto_row("retry", question="How many contract notices were published for the retry plan in 2024?"),
            _auto_row("impossible",
                     question="How many contract notices were published for the impossible plan in 2024?"),
        ]
        accept_from = {rows[0]["question"]: 0, rows[1]["question"]: 2, rows[2]["question"]: 999}
        chat = _AutoStubChat(accept_from)

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            report = run_l2_auto(
                rows, chat=chat, model="gen", checker_model="chk", out_dir=out_dir,
                variants=1, base_candidates_per_plan=2, candidate_top_up=2,
                max_candidates_per_plan=4, max_rounds=3, org_resolver=None, known_orgs=None,
                seed=1, progress_every=0, resume=True, workers=1, rpm=0,
            )
            surfaces = [json.loads(line) for line in
                       (out_dir / "surfaces.L2.jsonl").read_text(encoding="utf-8").splitlines()]

        self.assertEqual(report["total_plans"], 3)
        self.assertEqual(report["accepted_plans"], 2)
        self.assertEqual(report["unresolved_plan_ids"], ["impossible"])
        self.assertTrue(any("stub_reject" in reason
                            for reason in report["unresolved_reject_reasons"]["impossible"]))
        # capped by max_candidates_per_plan (4), not by max_rounds (3) -- confirms the ceiling
        # cap, not just the round cap, can end the loop.
        self.assertEqual(report["rounds_run"], 2)
        self.assertEqual(report["final_ceiling"], 4)
        self.assertEqual(chat.gen_calls, 10)  # 3 plans x 2 (round1) + 2 still-unresolved x 2 (round2)
        self.assertEqual({s["plan_id"] for s in surfaces}, {"easy", "retry"})
        self.assertEqual(len(surfaces), 2)  # exactly `variants`=1 accepted surface per resolved plan

    def test_resume_seeds_ceiling_from_existing_candidates(self):
        from build_multilevel_qa import candidate_path, run_l2_auto, write_jsonl

        row = _auto_row("resumed",
                        question="How many contract notices were published for the resumed plan in 2024?")
        # the 5 pre-seeded candidates below are written straight to disk, bypassing the stub, so
        # its internal counter must be told to resume from index 5 -- otherwise a freshly
        # generated candidate would be mislabelled index 0 and never reach the index-5 threshold.
        chat = _AutoStubChat({row["question"]: 5}, initial_index={row["question"]: 5})

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            # Simulate 5 already-generated (not yet checked) candidates from an earlier, interrupted
            # run, so resume must start the ceiling at >=5, not replay 2 -> 4 -> 6 from scratch.
            write_jsonl(candidate_path(out_dir, 2), [
                {"candidate_id": f"resumed#L2c{i}", "plan_id": "resumed", "level": 2,
                 "candidate_index": i, "question": re.sub(r"\?\s*$", f" [[v{i}]]?", row["question"]),
                 "persona": "citizen", "surface_origin": "gen"}
                for i in range(5)
            ])
            report = run_l2_auto(
                [row], chat=chat, model="gen", checker_model="chk", out_dir=out_dir,
                variants=1, base_candidates_per_plan=2, candidate_top_up=2,
                max_candidates_per_plan=9, max_rounds=6, org_resolver=None, known_orgs=None,
                seed=1, progress_every=0, resume=True, workers=1, rpm=0,
            )

        self.assertEqual(report["accepted_plans"], 1)
        self.assertEqual(report["final_ceiling"], 7)
        # only the 2 candidates past the resumed high-water mark were freshly generated --
        # the 5 pre-seeded ones were reused from disk, not regenerated.
        self.assertEqual(chat.gen_calls, 2)

    def test_one_attempt_per_round_three_distinct_personas_then_give_up(self):
        """Stated production policy: at most 3 attempts per plan, exactly 1 candidate generated
        per round, a different persona each of the first 3 attempts (seed+plan_id determined,
        reproducible), abandon after 3. Realised by base_candidates_per_plan=1,
        candidate_top_up=1, max_candidates_per_plan=3, max_rounds=3 -- no special-casing needed;
        this locks in that exact parameterisation against regressions."""
        from build_multilevel_qa import candidate_path, run_l2_auto

        row = _auto_row("gives_up_after_3",
                        question="How many contract notices were published for the giveup plan in 2024?")
        chat = _AutoStubChat({row["question"]: 10**9})  # never accepts

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            report = run_l2_auto(
                [row], chat=chat, model="gen", checker_model="chk", out_dir=out_dir,
                variants=1, base_candidates_per_plan=1, candidate_top_up=1,
                max_candidates_per_plan=3, max_rounds=3, org_resolver=None, known_orgs=None,
                seed=20260702, progress_every=0, resume=True, workers=1, rpm=0,
            )
            candidates = [json.loads(line) for line in
                         candidate_path(out_dir, 2).read_text(encoding="utf-8").splitlines()]

        self.assertEqual(report["rounds_run"], 3)
        self.assertEqual(report["final_ceiling"], 3)
        self.assertEqual(report["unresolved_plan_ids"], ["gives_up_after_3"])
        self.assertEqual(len(candidates), 3)  # exactly one attempt generated per round, not a batch
        self.assertEqual(len({c["persona"] for c in candidates}), 3)  # no persona repeated

    def test_checker_rpm_and_workers_are_independent_of_generator(self):
        """Regression: generator (e.g. nano, high rpm) and checker (e.g. grok, ~50 rpm) are
        commonly different deployments with different limits -- a shared cap would either
        throttle the generator or exceed the checker's limit. `run_l2_auto` must forward
        `checker_workers`/`checker_rpm` to the check phase only, leaving generate untouched."""
        from unittest.mock import patch
        import build_multilevel_qa as bmq

        row = _auto_row("easy", question="How many contract notices were published for the easy plan in 2024?")
        chat = _AutoStubChat({row["question"]: 0})
        calls: list[tuple[int, float]] = []
        real_run_concurrent = bmq.run_concurrent

        def spy(items, fn, *, workers, rpm, on_result):
            calls.append((workers, rpm))
            return real_run_concurrent(items, fn, workers=workers, rpm=rpm, on_result=on_result)

        with tempfile.TemporaryDirectory() as tmp, patch.object(bmq, "run_concurrent", spy):
            bmq.run_l2_auto(
                [row], chat=chat, model="gen", checker_model="chk", out_dir=Path(tmp),
                variants=1, base_candidates_per_plan=2, candidate_top_up=2,
                max_candidates_per_plan=4, max_rounds=3, org_resolver=None, known_orgs=None,
                seed=1, progress_every=0, resume=True, workers=8, rpm=300,
                checker_workers=4, checker_rpm=40,
            )
        # one round resolves this plan: [0]=generate call, [1]=check call
        self.assertEqual(calls, [(8, 300), (4, 40)])


if __name__ == "__main__":
    unittest.main()
