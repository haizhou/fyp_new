#!/usr/bin/env python3
"""Audit grounding interventions and their downstream effects without rerunning a model.

The audit has two complementary parts:

1. Saved runtime traces: compare the planner's ``pre_ground_spec`` with the saved
   ``grounded_spec``.  With ``--counterfactual``, execute both specifications over the same
   frozen KG and report whether grounding rescued, changed, or rejected the answer.
2. Teacher-data lineage: count successful repair-SFT examples whose preceding failure was
   caused by schema/value/entity grounding.  This measures how grounding feedback enters later
   optimisation; it is not a causal estimate of checkpoint accuracy.

No LLM is called.  Counterfactual execution is deterministic and reads the existing Parquet KG.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import fields
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from procurement_graph.reasoning.executor import execute_query_spec  # noqa: E402
from procurement_graph.reasoning.kg_backend import RuntimeKGBackend  # noqa: E402
from procurement_graph.reasoning.models import QueryConstraint, RuntimeQuerySpec  # noqa: E402


DEFAULT_TRACES = (
    Path("data/qa/eval/hard100_runtime_nano.jsonl"),
    Path("data/qa/eval/hard20_runtime_nano.jsonl"),
    Path("data/qa/eval/hard20_nlu_runtime_nano_semantic.jsonl"),
)

_SPEC_FIELDS = {item.name for item in fields(RuntimeQuerySpec)}
_CONSTRAINT_FIELDS = {item.name for item in fields(QueryConstraint)}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _spec_from_trace(payload: Any, *, question: str = "") -> RuntimeQuerySpec | None:
    if not isinstance(payload, dict):
        return None
    values = {key: value for key, value in payload.items() if key in _SPEC_FIELDS}
    values["question"] = str(values.get("question") or question)
    values["constraints"] = tuple(
        QueryConstraint(**{key: value for key, value in item.items() if key in _CONSTRAINT_FIELDS})
        for item in (payload.get("constraints") or ())
        if isinstance(item, dict)
    )
    values["relation_path"] = tuple(values.get("relation_path") or ())
    # Saved hard-set traces do not need linked entity objects for execution.  Do not attempt to
    # deserialize a provider-specific representation into the runtime dataclass.
    values["linked_entities"] = ()
    required_defaults = {
        "spec_id": "grounding-audit",
        "intent": "ambiguous",
        "answer_operation": "count",
        "answer_field": "",
        "answer_value_type": "string",
    }
    for key, value in required_defaults.items():
        values.setdefault(key, value)
    return RuntimeQuerySpec(**values)


def _decimal(value: Any) -> Decimal | None:
    try:
        parsed = Decimal(str(value).replace(",", ""))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return parsed if parsed.is_finite() else None


def answers_equal(left: Any, right: Any) -> bool:
    """Strict-enough artifact comparison without guessing a missing answer-type label."""
    if left is None or right is None:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            return False
        if "answer" in left or "answer" in right:
            return answers_equal(left.get("answer"), right.get("answer"))
        return left == right
    if isinstance(left, (list, tuple, set)) or isinstance(right, (list, tuple, set)):
        if not isinstance(left, (list, tuple, set)) or not isinstance(right, (list, tuple, set)):
            return False
        return sorted((str(item) for item in left)) == sorted((str(item) for item in right))
    lnum, rnum = _decimal(left), _decimal(right)
    if lnum is not None and rnum is not None:
        tolerance = max(Decimal("0.000001"), abs(rnum) * Decimal("0.000001"))
        return abs(lnum - rnum) <= tolerance
    return str(left).strip().casefold() == str(right).strip().casefold()


def _impact(raw_status: str, raw_answer: Any, grounded_status: str, grounded_answer: Any,
            oracle: Any) -> str:
    if oracle is None:
        # These rows encode unsupported/ambiguous outcomes at the full-pipeline level.  Replaying
        # the low-level query spec alone cannot reproduce the abstention policy, so do not fold
        # them into answer correctness.
        return "not_scored_no_oracle"
    raw_ok = raw_status == "passed" and answers_equal(raw_answer, oracle)
    grounded_ok = grounded_status == "passed" and answers_equal(grounded_answer, oracle)
    if not raw_ok and grounded_ok:
        return "rescued_to_correct"
    if raw_ok and not grounded_ok:
        return "degraded_from_correct"
    if raw_status != "passed" and grounded_status == "passed":
        return "made_executable_but_not_correct"
    if raw_status == "passed" and grounded_status != "passed":
        return "rejected_previously_executable"
    if raw_ok and grounded_ok:
        return "correct_both"
    if raw_status != grounded_status or not answers_equal(raw_answer, grounded_answer):
        return "changed_but_still_incorrect"
    return "no_observed_effect"


def audit_runtime(paths: Iterable[Path], *, backend: RuntimeKGBackend | None) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    all_changes: Counter[str] = Counter()
    all_impacts: Counter[str] = Counter()
    change_impacts: Counter[str] = Counter()
    all_counterfactual_rows = 0
    all_scored_counterfactual_rows = 0
    all_raw_correct = 0
    all_grounded_correct = 0
    examples: list[dict[str, Any]] = []
    replay_disagreements: list[dict[str, Any]] = []
    for path in paths:
        rows = read_jsonl(path)
        changes: Counter[str] = Counter()
        impacts: Counter[str] = Counter()
        changed_rows = 0
        changed_matches = 0
        grounding_failures = 0
        counterfactual_rows = 0
        scored_counterfactual_rows = 0
        raw_correct = 0
        grounded_correct = 0
        for row in rows:
            row_changes = [str(item) for item in (row.get("grounding_changes") or ())]
            changes.update(row_changes)
            all_changes.update(row_changes)
            changed = bool(row_changes)
            changed_rows += int(changed)
            changed_matches += int(changed and row.get("match") is True)
            grounding_failures += int((row.get("grounding") or {}).get("ok") is False)
            if backend is None:
                continue
            raw_spec = _spec_from_trace(row.get("pre_ground_spec"), question=str(row.get("question") or ""))
            grounded_spec = _spec_from_trace(row.get("grounded_spec"), question=str(row.get("question") or ""))
            if raw_spec is None or grounded_spec is None:
                continue
            raw = execute_query_spec(backend, raw_spec)
            grounded = execute_query_spec(backend, grounded_spec)
            impact = _impact(raw.status, raw.answer, grounded.status, grounded.answer, row.get("oracle"))
            impacts[impact] += 1
            all_impacts[impact] += 1
            raw_ok = raw.status == "passed" and answers_equal(raw.answer, row.get("oracle"))
            grounded_ok = grounded.status == "passed" and answers_equal(grounded.answer, row.get("oracle"))
            if row.get("oracle") is not None:
                raw_correct += int(raw_ok)
                grounded_correct += int(grounded_ok)
                all_raw_correct += int(raw_ok)
                all_grounded_correct += int(grounded_ok)
                scored_counterfactual_rows += 1
                all_scored_counterfactual_rows += 1
            counterfactual_rows += 1
            all_counterfactual_rows += 1
            for change in row_changes:
                change_impacts[f"{change} :: {impact}"] += 1
            if (row.get("oracle") is not None and grounded_ok != (row.get("match") is True)
                    and len(replay_disagreements) < 20):
                replay_disagreements.append({
                    "file": str(path),
                    "id": row.get("id"),
                    "saved_match": row.get("match"),
                    "replayed_grounded_status": grounded.status,
                    "replayed_grounded_answer": grounded.answer,
                    "oracle": row.get("oracle"),
                })
            if changed and len(examples) < 20:
                examples.append({
                    "file": str(path),
                    "id": row.get("id"),
                    "category": row.get("category"),
                    "changes": row_changes,
                    "raw_status": raw.status,
                    "raw_answer": raw.answer,
                    "grounded_status": grounded.status,
                    "grounded_answer": grounded.answer,
                    "oracle": row.get("oracle"),
                    "impact": impact,
                })
        files.append({
            "path": str(path),
            "n": len(rows),
            "saved_match": sum(row.get("match") is True for row in rows),
            "changed": changed_rows,
            "changed_and_saved_match": changed_matches,
            "grounding_failed": grounding_failures,
            "changes": dict(changes.most_common()),
            "counterfactual_n": counterfactual_rows,
            "scored_counterfactual_n": scored_counterfactual_rows,
            "raw_correct": raw_correct,
            "grounded_correct": grounded_correct,
            "counterfactual_impacts": dict(impacts.most_common()),
        })
    return {
        "files": files,
        "changes": dict(all_changes.most_common()),
        "counterfactual_impacts": dict(all_impacts.most_common()),
        "change_impacts": dict(change_impacts.most_common()),
        "counterfactual_n": all_counterfactual_rows,
        "scored_counterfactual_n": all_scored_counterfactual_rows,
        "raw_correct": all_raw_correct,
        "grounded_correct": all_grounded_correct,
        "absolute_accuracy_gain": (
            (all_grounded_correct - all_raw_correct) / all_scored_counterfactual_rows
            if all_scored_counterfactual_rows else None
        ),
        "saved_vs_current_replay_disagreements": replay_disagreements,
        "changed_examples": examples,
    }


def _grounding_failure_family(feedback: Any) -> str:
    if not isinstance(feedback, dict):
        return ""
    stage = str(feedback.get("failure_stage") or "").casefold()
    reason = str(feedback.get("failure_reason") or "").casefold()
    if any(token in reason for token in (
        "entity_not_found", "ambiguous_entity", "low_confidence", "unresolved organisation",
    )):
        return "entity_grounding"
    if any(token in reason for token in (
        "no_confident_schema_match", "ambiguous_schema_match", "type_gate_rejected",
        "unsupported_filter_slot", "schema_grounding",
    )):
        return "semantic_schema_grounding"
    if any(token in reason for token in (
        "ungroundable_variable", "constraint field", "answer_field", "group_by_field",
        "needs an explicit list of kg values", "release_year", "tender_category value",
        "award_date_signed needs", "dedupe_key", "sort_field",
    )):
        return "runtime_spec_grounding"
    if stage in {"grounding", "schema"}:
        return "runtime_schema_gate"
    return ""


def audit_teacher(teacher_dir: Path) -> dict[str, Any]:
    repairs = read_jsonl(teacher_dir / "repair_sft.jsonl")
    dpo = read_jsonl(teacher_dir / "dpo_pairs.jsonl")
    verified = read_jsonl(teacher_dir / "verified_sft.jsonl")
    hard_negatives = read_jsonl(teacher_dir / "hard_negatives.jsonl")
    abstain = read_jsonl(teacher_dir / "abstain_sft.jsonl")
    traces = read_jsonl(teacher_dir / "traces.jsonl")

    grounding_repairs: Counter[str] = Counter()
    grounding_examples: list[dict[str, Any]] = []
    for row in repairs:
        family = _grounding_failure_family(row.get("failure_feedback"))
        if not family:
            continue
        grounding_repairs[family] += 1
        if len(grounding_examples) < 20:
            grounding_examples.append({
                "id": row.get("id"),
                "family": family,
                "failure_stage": (row.get("failure_feedback") or {}).get("failure_stage"),
                "failure_reason": (row.get("failure_feedback") or {}).get("failure_reason"),
            })

    return {
        "teacher_dir": str(teacher_dir),
        "traces": len(traces),
        "verified_runtime": sum(row.get("verified") is True for row in traces),
        "verified_and_oracle": sum(row.get("verified") is True and row.get("oracle_match") is True for row in traces),
        "plan_sft": len(verified),
        "plan_sft_oracle_mismatch": sum(row.get("oracle_match") is not True for row in verified),
        "repair_sft": len(repairs),
        "grounding_related_successful_repairs": sum(grounding_repairs.values()),
        "grounding_repair_families": dict(grounding_repairs.most_common()),
        "dpo_pairs": len(dpo),
        "oracle_gated_dpo_pairs": sum(row.get("pair_kind") == "oracle_gated_repair" for row in dpo),
        "hard_negatives": len(hard_negatives),
        "abstain_sft": len(abstain),
        "grounding_repair_examples": grounding_examples,
        "interpretation": (
            "Grounding-related repair rows are successful repaired targets selected after the full "
            "pipeline and external training-time checks. They show a data-path influence on later "
            "optimisation, not the causal checkpoint gain from those rows."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-traces", nargs="+", type=Path, default=list(DEFAULT_TRACES))
    parser.add_argument("--teacher-dir", type=Path, default=Path("data/qa/teacher_full_v1"))
    parser.add_argument("--kg-dir", type=Path, default=Path("data/kg"))
    parser.add_argument("--counterfactual", action="store_true",
                        help="execute saved pre-ground and grounded specs over the frozen KG")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    backend = RuntimeKGBackend.from_directory(args.kg_dir) if args.counterfactual else None
    report = {
        "runtime": audit_runtime(args.runtime_traces, backend=backend),
        "teacher_data": audit_teacher(args.teacher_dir),
        "causal_boundary": (
            "The no-LLM counterfactual isolates deterministic execution-time grounding on saved "
            "plans. Existing SFT/RSFT/DPO checkpoint comparisons do not isolate the training effect "
            "of grounding-derived examples; that requires a controlled data ablation."
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
