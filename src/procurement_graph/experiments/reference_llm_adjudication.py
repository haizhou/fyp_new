"""LLM adjudication workflow for reference/API entity candidates.

This module never mutates `data/entities/*`. It builds auditable evidence
packets, validates LLM decisions, and applies deterministic safety gates to
produce dry-run merge approvals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "ablation" / "reference"
REPORT_DIR = ROOT / "reports" / "ablation" / "reference"
POLICY_CANDIDATES_PATH = DATA_DIR / "reference_policy_candidates.parquet"
QUEUE_PATH = DATA_DIR / "llm_review_queue.jsonl"
DECISIONS_PATH = DATA_DIR / "llm_decisions.jsonl"
TEMPLATE_DECISIONS_PATH = DATA_DIR / "llm_decisions.template.jsonl"
VALIDATED_DECISIONS_PATH = DATA_DIR / "llm_validated_decisions.parquet"
APPROVED_MERGES_PATH = DATA_DIR / "llm_approved_merges.parquet"
SUMMARY_PATH = REPORT_DIR / "llm_adjudication_summary.csv"
UNCERTAIN_PATH = REPORT_DIR / "llm_uncertain_cases.csv"
REJECTED_PATH = REPORT_DIR / "llm_rejected_or_blocked_cases.csv"

QUEUE_ACTIONS = {
    "manual_review_reference_collision",
    "manual_review_alias_or_name_mismatch",
    "manual_review_inactive_closed_or_medium",
}

ALLOWED_DECISIONS = {"merge_all", "merge_subset", "do_not_merge", "uncertain"}
REFERENCE_RISK_FLAGS = [
    "different_legal_entities",
    "subsidiary_possible",
    "parent_child_possible",
    "procurement_agent_possible",
    "different_geography",
    "role_conflict",
    "short_name_or_acronym_ambiguous",
    "historical_continuity_risk",
    "inactive_or_closed_reference",
    "medium_confidence_reference",
    "other_reference_risk",
    "insufficient_evidence",
    "none",
]
SYSTEM_RISK_FLAGS = {
    "dry_run",
    "invalid_json",
    "model_call_error",
    "template_decision",
    "unknown_task_id",
}
HARD_RISK_FLAGS = set(REFERENCE_RISK_FLAGS) - {"none"}
VALID_REFERENCE_RISK_FLAGS = set(REFERENCE_RISK_FLAGS) | SYSTEM_RISK_FLAGS

APPROVED_MERGE_COLUMNS = [
    "task_id",
    "old_canonical_id",
    "new_reference_id",
    "reference_source",
    "reference_matched_name",
    "decision",
    "llm_confidence",
    "reason",
    "post_gate_warning",
]

PROMPT_VERSION = "reference_v2"
SCHEMA_VERSION = "reference_decision_schema_v2"


@dataclass(frozen=True)
class SafetyPolicy:
    min_confidence: float = 0.85
    allow_medium_or_inactive: bool = False


def _stable_task_id(*parts: str) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _json_list(value) -> list:
    if isinstance(value, list):
        return value
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return []
    try:
        parsed = json.loads(value or "[]")
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _entity_payload(row: pd.Series, all_candidates: pd.DataFrame) -> dict[str, Any]:
    canonical_id = str(row["canonical_id"])
    cf_evidence = all_candidates[
        all_candidates["canonical_id"].eq(canonical_id)
        & all_candidates["reference_source"].eq("contracts_finder")
    ]
    return {
        "canonical_id": canonical_id,
        "canonical_name": row.get("canonical_name", ""),
        "er_status": row.get("er_status", ""),
        "n_aliases": int(row.get("n_aliases", 0) or 0),
        "org_type": row.get("org_type", ""),
        "org_category": row.get("org_category", ""),
        "address_region": row.get("address_region", ""),
        "alias_names": _json_list(row.get("alias_names"))[:20],
        "alias_raw_ids": _json_list(row.get("alias_raw_ids"))[:20],
        "match_features": {
            "exact_canonical_match": bool(row.get("exact_canonical_match", False)),
            "alias_match": bool(row.get("alias_match", False)),
            "reference_collision_size": int(row.get("reference_collision_size", 0) or 0),
        },
        "contracts_finder_evidence": [
            {
                "reference_canonical_id": cf_row.get("reference_canonical_id", ""),
                "matched_name": cf_row.get("reference_matched_name", ""),
                "status": cf_row.get("reference_status", ""),
            }
            for _, cf_row in cf_evidence.head(5).iterrows()
        ],
    }


def _expected_output_schema() -> dict[str, Any]:
    return {
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "task_id": "string copied from input",
        "decision": "merge_all | merge_subset | do_not_merge | uncertain",
        "confidence": "number from 0 to 1",
        "approved_entity_ids": ["canonical IDs approved for merge"],
        "excluded_entity_ids": ["canonical IDs explicitly excluded"],
        "canonical_reference_id": "official/reference ID copied from input",
        "canonical_name": "best display name after adjudication",
        "risk_flags": [f"zero or more of: {', '.join(REFERENCE_RISK_FLAGS)}"],
        "reason": "one concise explanation grounded in the evidence",
    }


def reference_decision_json_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "reference_entity_adjudication_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "prompt_version",
                "schema_version",
                "task_id",
                "decision",
                "confidence",
                "approved_entity_ids",
                "excluded_entity_ids",
                "canonical_reference_id",
                "canonical_name",
                "risk_flags",
                "reason",
            ],
            "properties": {
                "prompt_version": {"type": "string", "enum": [PROMPT_VERSION]},
                "schema_version": {"type": "string", "enum": [SCHEMA_VERSION]},
                "task_id": {"type": "string"},
                "decision": {"type": "string", "enum": ["merge_all", "merge_subset", "do_not_merge", "uncertain"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "approved_entity_ids": {"type": "array", "items": {"type": "string"}},
                "excluded_entity_ids": {"type": "array", "items": {"type": "string"}},
                "canonical_reference_id": {"type": "string"},
                "canonical_name": {"type": "string"},
                "risk_flags": {
                    "type": "array",
                    "items": {"type": "string", "enum": REFERENCE_RISK_FLAGS},
                },
                "reason": {"type": "string"},
            },
        },
    }


def _system_prompt() -> str:
    return (
        "You adjudicate whether procurement entity records refer to the same "
        "real-world organisation. Use only the evidence provided. Prefer "
        "uncertain or do_not_merge when there is a parent/subsidiary, agency, "
        "regional office, procurement agent, or historical-continuity risk. "
        "Return strict JSON matching the requested schema."
    )


def _user_prompt(packet: dict[str, Any]) -> str:
    return json.dumps(
        {
            "task": "Decide whether the candidate entities should be merged under the reference organisation.",
            "rules": [
                "merge_all only if every candidate entity is the same real-world organisation as the reference.",
                "merge_subset if only some candidates are safe to merge.",
                "do_not_merge if the evidence shows different organisations.",
                "uncertain if evidence is insufficient.",
                "Every candidate ID must appear in exactly one of approved_entity_ids or excluded_entity_ids.",
                "For do_not_merge or uncertain, approved_entity_ids must be empty and excluded_entity_ids must contain every candidate ID.",
                "risk_flags must describe residual risk in the approved merge only, not reasons for excluded candidates.",
                "Use risk_flags only from the allowed schema enum; use ['none'] only when the approved merge has no residual risk.",
                "Contracts Finder evidence is supporting provenance, not an independent legal authority.",
            ],
            "expected_output_schema": _expected_output_schema(),
            "evidence": packet["evidence"],
        },
        ensure_ascii=False,
        indent=2,
    )


def _packet(
    task_type: str,
    reference_source: str,
    reference_canonical_id: str,
    rows: pd.DataFrame,
    all_candidates: pd.DataFrame,
) -> dict[str, Any]:
    first = rows.iloc[0]
    task_id = _stable_task_id(task_type, reference_source, reference_canonical_id, "|".join(rows["canonical_id"].astype(str)))
    evidence = {
        "task_id": task_id,
        "task_type": task_type,
        "reference": {
            "source": reference_source,
            "canonical_id": reference_canonical_id,
            "matched_name": first.get("reference_matched_name", ""),
            "confidence": first.get("reference_confidence", ""),
            "status": first.get("reference_status", ""),
        },
        "candidate_entities": [
            _entity_payload(row, all_candidates) for _, row in rows.iterrows()
        ],
    }
    return {
            "task_id": task_id,
            "task_type": task_type,
            "prompt_version": PROMPT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "response_format": reference_decision_json_schema(),
        "evidence": evidence,
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": _user_prompt({"evidence": evidence})},
        ],
    }


def build_review_queue(
    policy_candidates: pd.DataFrame,
    max_tasks: int | None = None,
) -> list[dict[str, Any]]:
    """Build LLM review packets from policy-classified candidates."""
    candidates = policy_candidates[policy_candidates["policy_action"].isin(QUEUE_ACTIONS)].copy()
    all_candidates = policy_candidates.copy()
    packets: list[dict[str, Any]] = []

    collisions = candidates[candidates["policy_action"].eq("manual_review_reference_collision")]
    for (source, ref_id), grp in collisions.groupby(["reference_source", "reference_canonical_id"], sort=False):
        packets.append(_packet("reference_collision", source, ref_id, grp, all_candidates))
        if max_tasks and len(packets) >= max_tasks:
            return packets

    row_tasks = candidates[~candidates["policy_action"].eq("manual_review_reference_collision")]
    for _, row in row_tasks.iterrows():
        rows = pd.DataFrame([row])
        packets.append(_packet(str(row["policy_action"]), row["reference_source"], row["reference_canonical_id"], rows, all_candidates))
        if max_tasks and len(packets) >= max_tasks:
            return packets

    return packets


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                records.append({
                    "task_id": "",
                    "decision": "uncertain",
                    "confidence": 0.0,
                    "approved_entity_ids": [],
                    "excluded_entity_ids": [],
                    "canonical_reference_id": "",
                    "canonical_name": "",
                    "risk_flags": ["invalid_json"],
                    "reason": f"Invalid JSON on line {line_no}: {exc}",
                })
    return records


def build_template_decisions(queue_records: list[dict[str, Any]], max_tasks: int | None = None) -> list[dict[str, Any]]:
    """Build conservative placeholder decisions for API integration tests."""
    records = []
    for task in queue_records[:max_tasks]:
        reference = task.get("evidence", {}).get("reference", {})
        records.append({
            "prompt_version": PROMPT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "task_id": task["task_id"],
            "decision": "uncertain",
            "confidence": 0.0,
            "approved_entity_ids": [],
            "excluded_entity_ids": [
                entity["canonical_id"]
                for entity in task.get("evidence", {}).get("candidate_entities", [])
                if entity.get("canonical_id")
            ],
            "canonical_reference_id": reference.get("canonical_id", ""),
            "canonical_name": reference.get("matched_name", ""),
            "risk_flags": ["template_decision"],
            "reason": "Template placeholder; replace with an Azure model decision before approving merges.",
        })
    return records


def _queue_index(queue_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {record["task_id"]: record for record in queue_records}


def validate_decision(decision: dict[str, Any], task: dict[str, Any] | None) -> dict[str, Any]:
    """Validate one LLM decision and return a flat audit row."""
    errors: list[str] = []
    task_id = str(decision.get("task_id", ""))
    evidence = (task or {}).get("evidence", {})
    reference = evidence.get("reference", {})
    expected_prompt_version = str((task or {}).get("prompt_version", PROMPT_VERSION))
    expected_schema_version = str((task or {}).get("schema_version", SCHEMA_VERSION))
    candidate_ids = {
        str(entity.get("canonical_id"))
        for entity in evidence.get("candidate_entities", [])
        if entity.get("canonical_id")
    }

    raw_decision = str(decision.get("decision", "uncertain"))
    if raw_decision not in ALLOWED_DECISIONS:
        errors.append("invalid_decision")
        raw_decision = "uncertain"

    try:
        confidence = float(decision.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
        errors.append("invalid_confidence")
    if not 0 <= confidence <= 1:
        errors.append("confidence_out_of_range")
        confidence = max(0.0, min(1.0, confidence))

    approved = [str(x) for x in decision.get("approved_entity_ids") or []]
    excluded = [str(x) for x in decision.get("excluded_entity_ids") or []]
    if not set(approved).issubset(candidate_ids):
        errors.append("approved_ids_not_in_task")
    if not set(excluded).issubset(candidate_ids):
        errors.append("excluded_ids_not_in_task")
    if set(approved).intersection(set(excluded)):
        errors.append("approved_excluded_overlap")
    if raw_decision == "merge_all" and set(approved) != candidate_ids:
        errors.append("merge_all_does_not_approve_all_candidates")
    if raw_decision == "merge_all" and excluded:
        errors.append("merge_all_has_excluded_ids")
    if raw_decision == "merge_subset":
        if not approved:
            errors.append("merge_subset_without_approved_ids")
        if not excluded:
            errors.append("merge_subset_without_excluded_ids")
        if set(approved).union(set(excluded)) != candidate_ids:
            errors.append("merge_subset_does_not_partition_candidates")
    if raw_decision in {"do_not_merge", "uncertain"} and approved:
        errors.append("non_merge_decision_has_approved_ids")

    ref_id = str(decision.get("canonical_reference_id", ""))
    expected_ref_id = str(reference.get("canonical_id", ""))
    if ref_id != expected_ref_id:
        errors.append("reference_id_mismatch")
    if str(decision.get("prompt_version", "")) != expected_prompt_version:
        errors.append("prompt_version_mismatch")
    if str(decision.get("schema_version", "")) != expected_schema_version:
        errors.append("schema_version_mismatch")

    risk_flags = [str(flag) for flag in decision.get("risk_flags") or []]
    unknown_risk_flags = set(risk_flags) - VALID_REFERENCE_RISK_FLAGS
    if unknown_risk_flags:
        errors.append("unknown_risk_flags")
    if "none" in risk_flags and len(risk_flags) > 1:
        errors.append("none_risk_flag_with_other_flags")
    if raw_decision in {"do_not_merge", "uncertain"} and set(excluded) != candidate_ids:
        errors.append("non_merge_decision_does_not_exclude_all_candidates")
    return {
        "task_id": task_id,
        "task_type": evidence.get("task_type", ""),
        "prompt_version": str(decision.get("prompt_version", "")),
        "schema_version": str(decision.get("schema_version", "")),
        "schema_hash": str(decision.get("schema_hash", "")),
        "reference_source": reference.get("source", ""),
        "reference_canonical_id": expected_ref_id,
        "reference_matched_name": reference.get("matched_name", ""),
        "reference_status": reference.get("status", ""),
        "reference_confidence": reference.get("confidence", ""),
        "decision": raw_decision,
        "llm_confidence": confidence,
        "approved_entity_ids": json.dumps(approved),
        "excluded_entity_ids": json.dumps(excluded),
        "candidate_entity_ids": json.dumps(sorted(candidate_ids)),
        "risk_flags": json.dumps(risk_flags),
        "reason": str(decision.get("reason", "")),
        "validation_errors": json.dumps(errors),
        "is_valid_json_schema": len(errors) == 0,
    }


def validate_decisions(
    queue_records: list[dict[str, Any]],
    decision_records: list[dict[str, Any]],
) -> pd.DataFrame:
    queue = _queue_index(queue_records)
    rows = []
    for decision in decision_records:
        task = queue.get(str(decision.get("task_id", "")))
        if task is None:
            decision = dict(decision)
            decision.setdefault("risk_flags", [])
            decision["risk_flags"] = list(decision.get("risk_flags") or []) + ["unknown_task_id"]
        rows.append(validate_decision(decision, task))
    return pd.DataFrame(rows)


def _is_reference_status_allowed(row: pd.Series, policy: SafetyPolicy) -> bool:
    if policy.allow_medium_or_inactive:
        return True
    source = row.get("reference_source", "")
    status = str(row.get("reference_status", "")).lower()
    confidence = row.get("reference_confidence", "")
    if source == "govuk_orgs":
        return confidence == "high" and status in {"live", "exempt"}
    if source == "nhs_ods":
        return confidence == "high" and status == "active"
    return False


def apply_safety_gates(validated: pd.DataFrame, policy: SafetyPolicy = SafetyPolicy()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply deterministic gates and return `(approved_merges, blocked_cases)`."""
    approved_rows = []
    blocked_rows = []

    for _, row in validated.iterrows():
        validation_errors = _json_list(row.get("validation_errors"))
        risk_flags = _json_list(row.get("risk_flags"))
        approved_ids = _json_list(row.get("approved_entity_ids"))
        blockers: list[str] = []

        if validation_errors:
            blockers.append("validation_errors")
        if row.get("decision") not in {"merge_all", "merge_subset"}:
            blockers.append("decision_not_merge")
        if float(row.get("llm_confidence", 0.0)) < policy.min_confidence:
            blockers.append("low_confidence")
        if HARD_RISK_FLAGS.intersection(risk_flags):
            blockers.append("hard_risk_flags")
        if not _is_reference_status_allowed(row, policy):
            blockers.append("reference_status_not_allowed")
        if not approved_ids:
            blockers.append("no_approved_ids")

        if blockers:
            blocked = row.to_dict()
            blocked["safety_blockers"] = json.dumps(blockers)
            blocked_rows.append(blocked)
            continue

        for canonical_id in approved_ids:
            approved_rows.append({
                "task_id": row["task_id"],
                "old_canonical_id": canonical_id,
                "new_reference_id": row["reference_canonical_id"],
                "reference_source": row["reference_source"],
                "reference_matched_name": row["reference_matched_name"],
                "decision": row["decision"],
                "llm_confidence": row["llm_confidence"],
                "reason": row["reason"],
                "post_gate_warning": "",
            })

    approved = pd.DataFrame(approved_rows, columns=APPROVED_MERGE_COLUMNS)
    blocked = pd.DataFrame(blocked_rows)

    if not approved.empty and approved["old_canonical_id"].duplicated().any():
        dup_ids = set(approved.loc[approved["old_canonical_id"].duplicated(), "old_canonical_id"])
        approved["post_gate_warning"] = approved["old_canonical_id"].apply(
            lambda cid: "duplicate_old_canonical_id" if cid in dup_ids else ""
        )

    return approved, blocked


def build_queue(max_tasks: int | None = None) -> None:
    policy_candidates = pd.read_parquet(POLICY_CANDIDATES_PATH)
    records = build_review_queue(policy_candidates, max_tasks=max_tasks)
    write_jsonl(records, QUEUE_PATH)
    print(f"Written: {QUEUE_PATH} ({len(records):,} tasks)")


def write_template_decisions(
    queue_path: Path = QUEUE_PATH,
    output_path: Path = TEMPLATE_DECISIONS_PATH,
    max_tasks: int | None = None,
) -> None:
    queue_records = read_jsonl(queue_path)
    records = build_template_decisions(queue_records, max_tasks=max_tasks)
    write_jsonl(records, output_path)
    print(f"Written: {output_path} ({len(records):,} template decisions)")


def validate_and_gate(
    decisions_path: Path = DECISIONS_PATH,
    queue_path: Path = QUEUE_PATH,
    min_confidence: float = 0.85,
    allow_medium_or_inactive: bool = False,
) -> None:
    queue_records = read_jsonl(queue_path)
    decision_records = read_jsonl(decisions_path)
    validated = validate_decisions(queue_records, decision_records)
    approved, blocked = apply_safety_gates(
        validated,
        SafetyPolicy(
            min_confidence=min_confidence,
            allow_medium_or_inactive=allow_medium_or_inactive,
        ),
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    validated.to_parquet(VALIDATED_DECISIONS_PATH, index=False)
    approved.to_parquet(APPROVED_MERGES_PATH, index=False)
    blocked.to_csv(REJECTED_PATH, index=False)

    summary = pd.DataFrame([
        {"metric": "queue_tasks", "value": len(queue_records)},
        {"metric": "decision_rows", "value": len(decision_records)},
        {"metric": "validated_rows", "value": len(validated)},
        {"metric": "approved_merge_rows", "value": len(approved)},
        {"metric": "blocked_or_uncertain_rows", "value": len(blocked)},
    ])
    summary.to_csv(SUMMARY_PATH, index=False)

    uncertain = validated[validated["decision"].isin(["uncertain", "do_not_merge"])].copy()
    uncertain.to_csv(UNCERTAIN_PATH, index=False)

    print(f"Written: {VALIDATED_DECISIONS_PATH}")
    print(f"Written: {APPROVED_MERGES_PATH}")
    print(f"Written: {SUMMARY_PATH}")
    print(f"Written: {UNCERTAIN_PATH}")
    print(f"Written: {REJECTED_PATH}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/validate LLM adjudication for reference candidates")
    parser.add_argument("--build-queue", action="store_true", help="Build llm_review_queue.jsonl")
    parser.add_argument("--write-template-decisions", action="store_true", help="Write conservative decisions template JSONL")
    parser.add_argument("--validate-decisions", action="store_true", help="Validate existing llm_decisions.jsonl")
    parser.add_argument("--decisions", type=Path, default=DECISIONS_PATH, help="Path to LLM decisions JSONL")
    parser.add_argument("--queue", type=Path, default=QUEUE_PATH, help="Path to review queue JSONL")
    parser.add_argument("--template-decisions", type=Path, default=TEMPLATE_DECISIONS_PATH, help="Template decisions output path")
    parser.add_argument("--max-tasks", type=int, default=None, help="Optional cap for queue generation")
    parser.add_argument("--min-confidence", type=float, default=0.85, help="Safety gate confidence threshold")
    parser.add_argument(
        "--allow-medium-or-inactive",
        action="store_true",
        help="Allow medium/inactive reference statuses through safety gates",
    )
    return parser.parse_args(argv)


def cli_main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.build_queue:
        build_queue(max_tasks=args.max_tasks)
    if args.write_template_decisions:
        write_template_decisions(
            queue_path=args.queue,
            output_path=args.template_decisions,
            max_tasks=args.max_tasks,
        )
    if args.validate_decisions:
        validate_and_gate(
            decisions_path=args.decisions,
            queue_path=args.queue,
            min_confidence=args.min_confidence,
            allow_medium_or_inactive=args.allow_medium_or_inactive,
        )
    if not args.build_queue and not args.write_template_decisions and not args.validate_decisions:
        build_queue(max_tasks=args.max_tasks)


__all__ = [
    "SafetyPolicy",
    "apply_safety_gates",
    "build_queue",
    "build_review_queue",
    "build_template_decisions",
    "cli_main",
    "reference_decision_json_schema",
    "validate_and_gate",
    "validate_decision",
    "validate_decisions",
    "write_template_decisions",
]
