#!/usr/bin/env python3
"""Audit how hard-grounding feedback is represented in training and later plan behaviour.

This script does not call a model.  It links four existing artifact layers:

1. grounding-related failures and repaired targets in ``teacher_full_v1/repair_sft.jsonl``;
2. the same targets' exposure as direct plan-SFT, feedback-conditioned repair-SFT, and DPO;
3. conservative semantic-fidelity checks against the source benchmark constraints; and
4. in-sample behaviour of saved student harvests plus aggregate held-out checkpoint results.

The student-harvest comparison is descriptive, not causal: checkpoints, Step-1 models, data pools,
and optimisation objectives differ.  A matched data ablation is still required for attribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_grounding_impact import _grounding_failure_family  # noqa: E402


FIELD_TO_SLOT = {
    "release_year": "year",
    "tender_category": "category",
    "tender_cpv_id": "cpv",
    "buyer_name": "buyer",
    "supplier_name": "supplier",
    "tender_title": "title",
    "has_award_signed_date": "has_signed_date",
}

RESOLVER_SHAPES = {
    "cpvs_of_buyer": ("buyer", "cpv"),
    "suppliers_of_buyer": ("buyer", "supplier"),
    "buyers_of_supplier": ("supplier", "buyer"),
}

DEFAULT_STUDENTS = {
    "qwen_sft": Path("data/qa/rsft_qwen_r1"),
    "llama_sft": Path("data/qa/rsft_llama_r1"),
    "qwen_step1_dpo": Path("data/qa/rsft_qwen_r2"),
}

DEFAULT_HELDOUT = {
    "qwen_zeroshot": Path("outputs/eval/matrix_v2/cicada-qwen3-zeroshot/compare_cicada.summary.json"),
    "qwen_sft": Path("outputs/eval/matrix_v2/cicada-qwen3-sft/compare_cicada.summary.json"),
    "qwen_rsft": Path("outputs/eval/matrix_v2/cicada-qwen3-rsft/compare_cicada.summary.json"),
    "qwen_dpo": Path("outputs/eval/matrix_v2/cicada-qwen3-dpo/compare_cicada.summary.json"),
}

DEFAULT_PAIRED_RESULTS = {
    "qwen_zeroshot_to_sft": (
        Path("outputs/eval/matrix_v2/cicada-qwen3-zeroshot/compare_cicada.results.jsonl"),
        Path("outputs/eval/matrix_v2/cicada-qwen3-sft/compare_cicada.results.jsonl"),
    ),
    "qwen_sft_to_rsft": (
        Path("outputs/eval/matrix_v2/cicada-qwen3-sft/compare_cicada.results.jsonl"),
        Path("outputs/eval/matrix_v2/cicada-qwen3-rsft/compare_cicada.results.jsonl"),
    ),
    "qwen_rsft_to_dpo": (
        Path("outputs/eval/matrix_v2/cicada-qwen3-rsft/compare_cicada.results.jsonl"),
        Path("outputs/eval/matrix_v2/cicada-qwen3-dpo/compare_cicada.results.jsonl"),
    ),
    "llama_zeroshot_to_sft": (
        Path("outputs/eval/matrix_v2/cicada-llama31-zeroshot/compare_cicada.results.jsonl"),
        Path("outputs/eval/matrix_v2/cicada-llama31-sft/compare_cicada.results.jsonl"),
    ),
    "llama_sft_to_rsft": (
        Path("outputs/eval/matrix_v2/cicada-llama31-sft/compare_cicada.results.jsonl"),
        Path("outputs/eval/matrix_v2/cicada-llama31-rsft/compare_cicada.results.jsonl"),
    ),
    "llama_rsft_to_dpo": (
        Path("outputs/eval/matrix_v2/cicada-llama31-rsft/compare_cicada.results.jsonl"),
        Path("outputs/eval/matrix_v2/cicada-llama31-dpo/compare_cicada.results.jsonl"),
    ),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _normal(value: Any) -> str:
    return str(value).strip().casefold()


def _is_val(sample_id: str, val_frac: float) -> bool:
    return (int(hashlib.sha1(sample_id.encode()).hexdigest(), 16) % 1000) < val_frac * 1000


def _kept_plan_ids(verified: list[dict[str, Any]], qa_by_id: dict[str, dict[str, Any]],
                   *, family_cap: int, bucket_cap: int) -> set[str]:
    family_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    kept: set[str] = set()
    ordered = sorted(verified, key=lambda row: hashlib.sha1(str(row["id"]).encode()).hexdigest())
    for row in ordered:
        sample_id = str(row["id"])
        family = str(qa_by_id.get(sample_id, {}).get("template_family", "?"))
        bucket = str(row.get("bucket", "?"))
        if family_cap and family_counts[family] >= family_cap:
            continue
        if bucket_cap and bucket_counts[bucket] >= bucket_cap:
            continue
        family_counts[family] += 1
        bucket_counts[bucket] += 1
        kept.add(sample_id)
    return kept


def _plan_surface(plan: dict[str, Any] | None) -> tuple[set[tuple[str, str, str]], set[str]]:
    filters: set[tuple[str, str, str]] = set()
    entity_roles: set[str] = set()
    for variable in (plan or {}).get("variables", ()):
        if not isinstance(variable, dict):
            continue
        if str(variable.get("kind", "")) in {"entity", "entity_set", "node_set"}:
            entity_roles.add(_normal(variable.get("role", "")))
        for item in variable.get("filters", ()):
            if isinstance(item, dict):
                filters.add((_normal(item.get("slot", "")),
                             _normal(item.get("operator", "eq")),
                             _normal(item.get("value"))))
    return filters, entity_roles


def _literal_covered(expected: tuple[str, str, str], actual: set[tuple[str, str, str]]) -> bool:
    slot, operator, value = expected
    if expected in actual:
        return True
    # Several accepted normalised graphs encode a scalar equality as an ``in`` filter.  Count it
    # as covered only when the value is identical; this is a representation equivalence, not a
    # semantic relaxation.
    return operator == "eq" and (slot, "in", value) in actual


def assess_constraint_fidelity(source: dict[str, Any], plan: dict[str, Any] | None) -> dict[str, Any]:
    """Conservative gold-constraint coverage, including the three common bridge resolvers.

    This is deliberately not called semantic equivalence: entity aliases and more complex graph
    rewrites may be valid even when literal triples differ.  Missing category/year/CPV literals,
    however, are useful shortcut-risk flags for manual review.
    """
    actual, roles = _plan_surface(plan)
    evaluable = 0
    covered = 0
    missing: list[dict[str, Any]] = []
    for constraint in source.get("constraints", ()):
        if not isinstance(constraint, dict) or constraint.get("field") not in FIELD_TO_SLOT:
            continue
        evaluable += 1
        field = str(constraint["field"])
        slot = FIELD_TO_SLOT[field]
        operator = _normal(constraint.get("op", "eq"))
        value = constraint.get("value")
        ok = False
        if operator == "in_subquery" and isinstance(value, dict):
            shape = RESOLVER_SHAPES.get(_normal(value.get("resolve", "")))
            if shape is not None:
                source_slot, emitted_role = shape
                source_value = value.get(source_slot)
                ok = emitted_role in roles and (source_slot, "eq", _normal(source_value)) in actual
        else:
            ok = _literal_covered((slot, operator, _normal(value)), actual)
        if ok:
            covered += 1
        else:
            missing.append({"field": field, "slot": slot, "op": operator, "value": value})
    return {"evaluable": evaluable, "covered": covered, "missing": missing,
            "all_covered": bool(evaluable) and not missing}


def audit_exposure(grounding_rows: list[dict[str, Any]], verified: list[dict[str, Any]],
                   dpo: list[dict[str, Any]], qa_by_id: dict[str, dict[str, Any]],
                   *, family_cap: int, bucket_cap: int, val_frac: float) -> dict[str, Any]:
    grounding_ids = {str(row["id"]) for row in grounding_rows}
    verified_ids = {str(row["id"]) for row in verified}
    dpo_ids = {str(row["id"]) for row in dpo}
    kept = _kept_plan_ids(verified, qa_by_id, family_cap=family_cap, bucket_cap=bucket_cap)
    direct = grounding_ids & kept
    repair_train = {sample_id for sample_id in grounding_ids if not _is_val(sample_id, val_frac)}
    direct_train = {sample_id for sample_id in direct if not _is_val(sample_id, val_frac)}
    return {
        "grounding_repair_rows": len(grounding_rows),
        "also_in_raw_verified_sft": len(grounding_ids & verified_ids),
        "repair_sft_train": len(repair_train),
        "repair_sft_val": len(grounding_ids - repair_train),
        "also_direct_plan_sft_after_caps": len(direct),
        "direct_plan_sft_train": len(direct_train),
        "direct_plan_sft_val": len(direct - direct_train),
        "training_row_exposures": len(repair_train) + len(direct_train),
        "unique_training_questions": len(repair_train),
        "teacher_dpo_overlap": len(grounding_ids & dpo_ids),
        "interpretation": (
            "Direct plan-SFT trains p(plan|question); repair-SFT trains "
            "p(repaired_plan|question,failed_plan,hard_feedback). Counting both as one treatment "
            "would conflate first-pass planning with feedback-conditioned replanning."
        ),
    }


def audit_fidelity(grounding_rows: list[dict[str, Any]],
                   qa_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows_evaluable = 0
    rows_all_covered = 0
    constraints = 0
    covered = 0
    missing_by_slot: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    category_expected = 0
    category_missing = 0
    for row in grounding_rows:
        source = qa_by_id.get(str(row["id"]), {})
        result = assess_constraint_fidelity(source, row.get("target_graph_plan"))
        if result["evaluable"]:
            rows_evaluable += 1
            rows_all_covered += int(result["all_covered"])
        constraints += int(result["evaluable"])
        covered += int(result["covered"])
        for missing in result["missing"]:
            missing_by_slot[str(missing["slot"])] += 1
        expected_categories = [item for item in source.get("constraints", ())
                               if isinstance(item, dict) and item.get("field") == "tender_category"]
        category_expected += int(bool(expected_categories))
        category_missing += int(any(item["slot"] == "category" for item in result["missing"]))
        if result["missing"] and len(examples) < 12:
            examples.append({"id": row.get("id"), "question": row.get("question"),
                             "missing": result["missing"], "target_graph_plan": row.get("target_graph_plan")})
    return {
        "rows_with_evaluable_gold_constraints": rows_evaluable,
        "rows_covering_all_evaluable_gold_constraints": rows_all_covered,
        "rows_flagged_for_manual_review": rows_evaluable - rows_all_covered,
        "constraint_recall": covered / constraints if constraints else None,
        "missing_by_slot": dict(missing_by_slot.most_common()),
        "rows_with_category_constraint": category_expected,
        "rows_missing_category_literal": category_missing,
        "examples": examples,
        "interpretation": (
            "Every target was oracle-matched on the frozen KG, but oracle equality alone does not "
            "guarantee intensional plan fidelity. Missing literals can be extensionally redundant "
            "on the current data and therefore teach shortcut plans. Flags require review rather "
            "than automatic deletion because entity aliases and graph rewrites can be equivalent."
        ),
    }


def audit_student(label: str, directory: Path, grounding_ids: set[str],
                  qa_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    traces = {str(row["id"]): row for row in read_jsonl(directory / "traces.jsonl")}
    repair_rows = read_jsonl(directory / "repair_sft.jsonl")
    recurring = {str(row["id"]) for row in repair_rows
                 if _grounding_failure_family(row.get("failure_feedback"))}
    present = [traces[sample_id] for sample_id in grounding_ids if sample_id in traces]
    fidelity_evaluable = 0
    fidelity_all = 0
    missing_by_slot: Counter[str] = Counter()
    for row in present:
        result = assess_constraint_fidelity(qa_by_id.get(str(row["id"]), {}), row.get("graph_plan"))
        if result["evaluable"]:
            fidelity_evaluable += 1
            fidelity_all += int(result["all_covered"])
        missing_by_slot.update(str(item["slot"]) for item in result["missing"])
    return {
        "label": label,
        "artifact_dir": str(directory),
        "present": len(present),
        "initial_success": sum(row.get("attempt_of_success") == 0 for row in present),
        "still_needed_repair": sum(isinstance(row.get("attempt_of_success"), int)
                                   and row.get("attempt_of_success") > 0 for row in present),
        "eventual_verified": sum(row.get("verified") is True for row in present),
        "eventual_oracle_match": sum(row.get("oracle_match") is True for row in present),
        "grounding_failure_recurred_then_repaired": len(grounding_ids & recurring),
        "unverified": sum(row.get("verified") is not True for row in present),
        "gold_constraint_fidelity": {
            "evaluable": fidelity_evaluable,
            "all_covered": fidelity_all,
            "flagged": fidelity_evaluable - fidelity_all,
            "missing_by_slot": dict(missing_by_slot.most_common()),
        },
    }


def audit_heldout(paths: dict[str, Path]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, path in paths.items():
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        categories = payload.get("by_category") or {}
        out[label] = {
            "artifact": str(path),
            "overall": payload.get("overall"),
            "bridge_join": categories.get("v4:bridge_join"),
            "categorical": categories.get("v4:categorical"),
            "comparison": categories.get("v4:comparison"),
            "set": categories.get("v4:set"),
        }
    return out


def _exact_mcnemar_p(wrong_to_right: int, right_to_wrong: int) -> float:
    discordant = wrong_to_right + right_to_wrong
    if not discordant:
        return 1.0
    tail = min(wrong_to_right, right_to_wrong)
    probability = sum(math.comb(discordant, index) for index in range(tail + 1)) / (2 ** discordant)
    return min(1.0, 2 * probability)


def audit_paired_results(paths: dict[str, tuple[Path, Path]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, (before_path, after_path) in paths.items():
        if not before_path.exists() or not after_path.exists():
            continue
        before = {str(row["id"]): row for row in read_jsonl(before_path)}
        after = {str(row["id"]): row for row in read_jsonl(after_path)}
        shared = sorted(before.keys() & after.keys())
        wrong_to_right = sum(before[item].get("correct") is not True
                             and after[item].get("correct") is True for item in shared)
        right_to_wrong = sum(before[item].get("correct") is True
                             and after[item].get("correct") is not True for item in shared)
        out[label] = {
            "before_artifact": str(before_path),
            "after_artifact": str(after_path),
            "n": len(shared),
            "before_correct": sum(before[item].get("correct") is True for item in shared),
            "after_correct": sum(after[item].get("correct") is True for item in shared),
            "wrong_to_right": wrong_to_right,
            "right_to_wrong": right_to_wrong,
            "net_gain": wrong_to_right - right_to_wrong,
            "exact_mcnemar_p": _exact_mcnemar_p(wrong_to_right, right_to_wrong),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-dir", type=Path, default=Path("data/qa/teacher_full_v1"))
    parser.add_argument("--qa", type=Path, default=Path("data/qa/cicada_core_v4/train.jsonl"))
    parser.add_argument("--family-cap", type=int, default=150)
    parser.add_argument("--bucket-cap", type=int, default=400)
    parser.add_argument("--val-frac", type=float, default=0.02)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    qa_by_id = {str(row["id"]): row for row in read_jsonl(args.qa)}
    repairs = read_jsonl(args.teacher_dir / "repair_sft.jsonl")
    grounding_rows = [row for row in repairs if _grounding_failure_family(row.get("failure_feedback"))]
    grounding_ids = {str(row["id"]) for row in grounding_rows}
    report = {
        "scope": {
            "source_questions": len(qa_by_id),
            "all_repair_sft": len(repairs),
            "grounding_related_repair_sft": len(grounding_rows),
            "failure_families": dict(Counter(
                _grounding_failure_family(row.get("failure_feedback")) for row in grounding_rows
            ).most_common()),
        },
        "training_exposure": audit_exposure(
            grounding_rows,
            read_jsonl(args.teacher_dir / "verified_sft.jsonl"),
            read_jsonl(args.teacher_dir / "dpo_pairs.jsonl"),
            qa_by_id,
            family_cap=args.family_cap,
            bucket_cap=args.bucket_cap,
            val_frac=args.val_frac,
        ),
        "target_semantic_fidelity": audit_fidelity(grounding_rows, qa_by_id),
        "saved_student_behaviour_on_same_questions": {
            label: audit_student(label, directory, grounding_ids, qa_by_id)
            for label, directory in DEFAULT_STUDENTS.items() if directory.exists()
        },
        "heldout_aggregate_checkpoint_results": audit_heldout(DEFAULT_HELDOUT),
        "heldout_paired_transitions": audit_paired_results(DEFAULT_PAIRED_RESULTS),
        "causal_boundary": (
            "The same-question student harvest is in-sample and the held-out checkpoints differ in "
            "more than grounding data. These artifacts show exposure and behavioural compatibility, "
            "not the causal gain from hard-grounding examples. Use matched clean-SFT, non-grounding-"
            "repair, grounding-repair, and grounding-DPO arms for attribution."
        ),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2, default=str, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
