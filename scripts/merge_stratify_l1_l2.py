"""Merge clean L1 and checked L2 QA into grouped, stratified CICADA splits.

The script implements Phase A3 from docs/cicada_planner_training_plan.md:

- normalise L1/L2 rows into one schema with gold_plan + provenance;
- assign a stable near-duplicate group before splitting;
- split whole groups only, so near-duplicates do not cross train/dev/final;
- reserve an OOD slice enriched for L2 bridge/extended/abstain cases.

Default inputs reflect the current project state:
  L1: data/qa/generated_clean_l1/clean.jsonl
  L2: data/qa/multilevel_v2_full2k/surfaces.L2.jsonl + plan_bank.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_L1 = ROOT / "data" / "qa" / "generated_clean_l1" / "clean.jsonl"
DEFAULT_L2_DIR = ROOT / "data" / "qa" / "multilevel_v2_full2k"
DEFAULT_OUT = ROOT / "data" / "qa" / "cicada_merged_l1_l2_trainbalanced_v1"
ABSTAIN_STATUS = {"unsupported", "ambiguous", "no_results"}
ENTITY_FIELDS = {"buyer_name", "supplier_name", "contract_node_id", "tender_title", "award_title"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows),
                    encoding="utf-8")


def stable_hash(value: str, *, length: int = 16) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def constraints_key(constraints: list[dict[str, Any]]) -> dict[str, tuple[str, ...]]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for constraint in constraints or []:
        field = str(constraint.get("field", ""))
        if not field:
            continue
        value = constraint.get("value")
        if isinstance(value, (list, tuple)):
            rendered = ",".join(norm_text(v) for v in value)
        else:
            rendered = norm_text(value)
        if field in ENTITY_FIELDS:
            buckets["entities"].append(f"{field}={rendered}")
        elif field == "release_year":
            buckets["years"].append(f"{constraint.get('op', 'eq')}={rendered}")
        elif field == "tender_cpv_id":
            buckets["cpv"].append(rendered)
        elif field == "tender_category":
            buckets["category"].append(rendered)
    return {key: tuple(sorted(values)) for key, values in buckets.items()}


def infer_question_type(row: dict[str, Any], *, source: str) -> str:
    status = str(row.get("expected_status", "answerable"))
    if status in ABSTAIN_STATUS:
        return f"abstain_{status}"
    if source == "L1" and row.get("question_type"):
        return str(row.get("question_type"))
    op = str(row.get("answer_operation", ""))
    if op == "count":
        return "count"
    if op == "sum":
        return "sum"
    if op == "select_unique":
        return "factoid"
    if op == "exists":
        return "boolean"
    if op == "distinct_set":
        return "set"
    if op in {"argmax", "argmin"}:
        return "min_max"
    if op in {"rank_top_k", "top_k"}:
        return "top_k"
    if op == "compare":
        return "comparison"
    return op or status or "unknown"


def l1_row(row: dict[str, Any]) -> dict[str, Any]:
    spec_id = str(row.get("spec_id"))
    gold_plan = row.get("gold_plan") or {}
    constraints = list(gold_plan.get("constraints") or row.get("constraints") or [])
    question_type = infer_question_type(row, source="L1")
    template_family = str(row.get("operation_family") or gold_plan.get("metadata", {}).get("operation_family") or question_type)
    return {
        "id": f"L1::{spec_id}",
        "source": "L1",
        "level": 1,
        "source_dataset": "generated_clean_l1",
        "plan_id": spec_id,
        "question": row.get("question"),
        "question_type": question_type,
        "template_family": template_family,
        "answer_operation": row.get("answer_operation") or gold_plan.get("answer_operation"),
        "answer_field": gold_plan.get("answer_field", ""),
        "answer_type": row.get("answer_value_type") or gold_plan.get("answer_value_type"),
        "expected_status": "answerable",
        "oracle_answer": row.get("golden_answer"),
        "constraints": constraints,
        "difficulty": row.get("difficulty") or gold_plan.get("metadata", {}).get("difficulty", ""),
        "domain_slice": row.get("domain_slice") or gold_plan.get("metadata", {}).get("domain_slice", ""),
        "hop_class": row.get("hop_class") or gold_plan.get("metadata", {}).get("hop_class", ""),
        "generalization_class": row.get("generalization_class") or "iid",
        "gold_plan": gold_plan,
        "provenance": {
            "source": "L1",
            "source_file": "data/qa/generated_clean_l1/clean.jsonl",
            "cleaning": row.get("cleaning", {}),
            "generator_model": row.get("generation", {}).get("model", ""),
        },
        "metadata": {
            "operation_family": row.get("operation_family", ""),
            "source_stage1": row.get("source_stage1", {}),
            "value_sanity_status": row.get("value_sanity_status", ""),
        },
    }


def l2_row(row: dict[str, Any], plan_bank: dict[str, dict[str, Any]]) -> dict[str, Any]:
    plan = plan_bank.get(str(row.get("plan_id")), {})
    constraints = list(plan.get("constraints") or row.get("constraints") or [])
    question_type = infer_question_type(row, source="L2")
    template_family = str(plan.get("template_family") or row.get("subset") or question_type)
    answer_op = row.get("answer_operation") or plan.get("answer_operation")
    gold_plan = {
        "spec_id": row.get("plan_id"),
        "constraints": constraints,
        "answer_operation": answer_op,
        "answer_field": plan.get("answer_field", ""),
        "answer_value_type": row.get("answer_type") or plan.get("answer_type", ""),
        "dedupe_key": "contract_node_id",
        "metadata": {
            "question_type": question_type,
            "template_family": template_family,
            "source_subset": row.get("subset"),
            "expected_status": row.get("expected_status"),
        },
    }
    return {
        "id": f"L2::{row.get('id')}",
        "source": "L2",
        "level": 2,
        "source_dataset": "multilevel_v2_full2k",
        "plan_id": row.get("plan_id"),
        "question": row.get("question"),
        "question_type": question_type,
        "template_family": template_family,
        "answer_operation": answer_op,
        "answer_field": plan.get("answer_field", ""),
        "answer_type": row.get("answer_type") or plan.get("answer_type"),
        "expected_status": row.get("expected_status", "answerable"),
        "oracle_answer": row.get("oracle_answer"),
        "constraints": constraints,
        "difficulty": row.get("metadata", {}).get("difficulty", ""),
        "domain_slice": row.get("subset", ""),
        "hop_class": "multi-hop" if row.get("subset") == "bridge_join" else "",
        "generalization_class": "ood_candidate" if is_ood_candidate(row) else "iid",
        "gold_plan": gold_plan,
        "provenance": {
            "source": "L2",
            "source_file": "data/qa/multilevel_v2_full2k/surfaces.L2.jsonl",
            "generator_model": row.get("surface_origin", ""),
            "checker_model": (row.get("checker") or {}).get("model", ""),
            "persona": row.get("persona", ""),
            "candidate_id": row.get("candidate_id", ""),
        },
        "metadata": {
            "source_subset": row.get("subset"),
            "checker": row.get("checker", {}),
            "l2_generation_mode": row.get("l2_generation_mode", ""),
            "surface_checks": row.get("surface_checks", {}),
        },
    }


def is_ood_candidate(row: dict[str, Any]) -> bool:
    subset = str(row.get("subset", ""))
    op = str(row.get("answer_operation", ""))
    status = str(row.get("expected_status", "answerable"))
    return (
        row.get("source") == "L2"
        or subset in {"bridge_join", "unanswerable", "extended_ops"}
        or status in ABSTAIN_STATUS
        or op in {"compare", "exists", "distinct_set", "argmax", "argmin", "rank_top_k", "top_k"}
    )


def train_bucket(row: dict[str, Any]) -> str:
    """Training-balance bucket: coarser than template, finer than answer_operation."""
    status = str(row.get("expected_status", "answerable"))
    if status in ABSTAIN_STATUS:
        return f"abstain_{status}"
    subset = str(row.get("domain_slice") or row.get("metadata", {}).get("source_subset") or "")
    if subset == "bridge_join":
        return "bridge_join"
    qtype = str(row.get("question_type", ""))
    op = str(row.get("answer_operation", ""))
    if qtype in {"aggregation_count", "categorical_cpv", "temporal"} or op == "count":
        return "count"
    if qtype == "aggregation_sum" or op == "sum":
        return "sum"
    if op == "select_unique" or qtype == "factoid":
        return "factoid"
    if op == "exists":
        return "boolean"
    if op in {"compare"} or qtype == "comparison":
        return "comparison"
    if op in {"distinct_set", "set"}:
        return "set"
    if op in {"argmax", "argmin"}:
        return "min_max"
    if op in {"rank_top_k", "top_k"}:
        return "top_k"
    return qtype or op or "unknown"


def attach_group(row: dict[str, Any]) -> None:
    key_parts = constraints_key(row.get("constraints", []))
    has_anchor = any(key_parts.get(name) for name in ("entities", "years", "cpv", "category"))
    role_pattern = ",".join(sorted(
        field for field in ("buyer_name", "supplier_name")
        if any(c.get("field") == field for c in row.get("constraints", []))
    ))
    raw = json.dumps({
        "template_family": row.get("template_family"),
        "question_type": row.get("question_type"),
        "answer_operation": row.get("answer_operation"),
        "expected_status": row.get("expected_status"),
        "entities": key_parts.get("entities", ()),
        "years": key_parts.get("years", ()),
        "cpv": key_parts.get("cpv", ()),
        "category": key_parts.get("category", ()),
        "role_pattern": role_pattern,
        "fallback_anchor": "" if has_anchor else row.get("plan_id"),
    }, sort_keys=True, ensure_ascii=False, default=str)
    row["dedup_group"] = "dg_" + stable_hash(raw)
    row["stratum"] = "|".join([
        str(row.get("question_type", "")),
        str(row.get("template_family", "")),
        str(row.get("difficulty", "")),
        str(row.get("domain_slice", "")),
        str(row.get("expected_status", "")),
    ])
    row["train_bucket"] = train_bucket(row)


def grouped(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["dedup_group"]].append(row)
    return dict(groups)


def group_summary(group_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    statuses = Counter(str(r.get("expected_status")) for r in rows)
    sources = Counter(str(r.get("source")) for r in rows)
    return {
        "dedup_group": group_id,
        "size": len(rows),
        "stratum": first.get("stratum", ""),
        "template_family": first.get("template_family", ""),
        "question_type": first.get("question_type", ""),
        "train_bucket": first.get("train_bucket", ""),
        "expected_status": statuses.most_common(1)[0][0],
        "sources": dict(sources),
        "ood_candidate": any(is_ood_candidate(r) for r in rows),
        "hash": stable_hash(group_id),
    }


def take_groups(
    candidates: list[dict[str, Any]],
    used: set[str],
    target_rows: int,
    *,
    seed: int,
) -> list[str]:
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in candidates:
        if group["dedup_group"] not in used:
            by_stratum[group["stratum"]].append(group)
    for stratum, items in by_stratum.items():
        items.sort(key=lambda g: stable_hash(f"{seed}:{g['dedup_group']}"))

    picked: list[str] = []
    total = 0
    strata = sorted(by_stratum, key=lambda s: stable_hash(f"{seed}:stratum:{s}"))
    while total < target_rows:
        moved = False
        for stratum in strata:
            items = by_stratum[stratum]
            while items and items[0]["dedup_group"] in used:
                items.pop(0)
            if not items:
                continue
            group = items.pop(0)
            group_id = group["dedup_group"]
            if group_id in used:
                continue
            picked.append(group_id)
            used.add(group_id)
            total += int(group["size"])
            moved = True
            if total >= target_rows:
                break
        if not moved:
            break
    return picked


def take_train_balanced(
    candidates: list[dict[str, Any]],
    used: set[str],
    target_rows: int,
    *,
    seed: int,
    max_bucket_share: float = 0.80,
    tiny_bucket_all: int = 50,
    abstain_share: float = 0.10,
) -> list[str]:
    """Pick train first with bucket balance as the main objective.

    This intentionally lets scarce buckets (e.g. top_k) appear as examples without trying to force
    impossible quotas. Once a bucket is exhausted, the remaining train budget is shared across the
    still-available buckets.
    """
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in candidates:
        if group["dedup_group"] not in used:
            by_bucket[group["train_bucket"]].append(group)
    for bucket, items in by_bucket.items():
        items.sort(key=lambda g: stable_hash(f"{seed}:{bucket}:{g['dedup_group']}"))
    abstain_buckets = sorted(bucket for bucket in by_bucket if bucket.startswith("abstain_"))
    abstain_bucket_cap = max(1, int((target_rows * abstain_share) / max(len(abstain_buckets), 1)))
    bucket_caps: dict[str, int] = {}
    for bucket, items in by_bucket.items():
        available = sum(int(group["size"]) for group in items)
        if bucket.startswith("abstain_"):
            bucket_caps[bucket] = min(available, abstain_bucket_cap)
        elif available <= tiny_bucket_all:
            bucket_caps[bucket] = available
        else:
            bucket_caps[bucket] = max(1, int(available * max_bucket_share))
    picked_by_bucket: Counter[str] = Counter()

    bucket_order = sorted(by_bucket, key=lambda b: stable_hash(f"{seed}:bucket:{b}"))
    picked: list[str] = []
    total = 0
    while total < target_rows:
        moved = False
        for bucket in bucket_order:
            items = by_bucket[bucket]
            while items and items[0]["dedup_group"] in used:
                items.pop(0)
            if not items:
                continue
            group = items.pop(0)
            group_id = group["dedup_group"]
            if group_id in used:
                continue
            size = int(group["size"])
            if picked_by_bucket[bucket] >= bucket_caps[bucket]:
                continue
            picked.append(group_id)
            used.add(group_id)
            total += size
            picked_by_bucket[bucket] += size
            moved = True
            if total >= target_rows:
                break
        if not moved:
            break
    return picked


def assign_splits(rows: list[dict[str, Any]], *, seed: int, targets: dict[str, int],
                  train_abstain_share: float = 0.10) -> dict[str, list[str]]:
    groups_by_id = grouped(rows)
    summaries = [group_summary(group_id, members) for group_id, members in groups_by_id.items()]
    used: set[str] = set()
    split_groups: dict[str, list[str]] = {}
    final_reserved = [g for g in summaries if g["train_bucket"] == "top_k"]
    allocatable = [g for g in summaries if g["train_bucket"] != "top_k"]

    split_groups["train"] = take_train_balanced(allocatable, used, targets["train"], seed=seed + 11,
                                                abstain_share=train_abstain_share)

    remaining = [g for g in allocatable if g["dedup_group"] not in used]
    for split_name, offset in (("dev_smoke", 21), ("dev_tune", 31), ("dev_select", 41)):
        split_groups[split_name] = take_groups(remaining, used, targets[split_name], seed=seed + offset)
        remaining = [g for g in remaining if g["dedup_group"] not in used]

    split_groups["final_test"] = []
    reserved_rows = 0
    for group in final_reserved:
        group_id = group["dedup_group"]
        if group_id not in used:
            split_groups["final_test"].append(group_id)
            used.add(group_id)
            reserved_rows += int(group["size"])
    final_candidates = [g for g in summaries if g["dedup_group"] not in used]
    split_groups["final_test"].extend(
        take_groups(final_candidates, used, max(0, targets["final_test"] - reserved_rows), seed=seed + 51)
    )
    return split_groups


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split = Counter(str(r.get("split")) for r in rows)
    return {
        "total_rows": len(rows),
        "by_split": dict(by_split),
        "by_source": dict(Counter(str(r.get("source")) for r in rows)),
        "by_level": dict(Counter(str(r.get("level")) for r in rows)),
        "by_expected_status": dict(Counter(str(r.get("expected_status")) for r in rows)),
        "by_question_type": dict(Counter(str(r.get("question_type")) for r in rows)),
        "by_train_bucket": dict(Counter(str(r.get("train_bucket")) for r in rows)),
        "split_by_source": {
            split: dict(Counter(str(r.get("source")) for r in rows if r.get("split") == split))
            for split in sorted(by_split)
        },
        "split_by_expected_status": {
            split: dict(Counter(str(r.get("expected_status")) for r in rows if r.get("split") == split))
            for split in sorted(by_split)
        },
        "split_by_train_bucket": {
            split: dict(Counter(str(r.get("train_bucket")) for r in rows if r.get("split") == split))
            for split in sorted(by_split)
        },
        "final_test_buckets": dict(Counter(str(r.get("final_bucket", "")) for r in rows if r.get("split") == "final_test")),
        "dedup_groups": len({r["dedup_group"] for r in rows}),
        "max_group_size": max(Counter(r["dedup_group"] for r in rows).values()) if rows else 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--l1-clean", type=Path, default=DEFAULT_L1)
    ap.add_argument("--l2-dir", type=Path, default=DEFAULT_L2_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=20260703)
    ap.add_argument("--dev-smoke", type=int, default=50)
    ap.add_argument("--dev-tune", type=int, default=300)
    ap.add_argument("--dev-select", type=int, default=650)
    ap.add_argument("--final-test", type=int, default=10000)
    ap.add_argument("--train-target", type=int, default=0,
                    help="0 = total - dev targets - final-test target")
    ap.add_argument("--train-abstain-share", type=float, default=0.10,
                    help="Approximate train budget reserved for all abstain buckets combined")
    args = ap.parse_args()

    l1 = [l1_row(row) for row in read_jsonl(args.l1_clean)]
    plan_bank = {str(row.get("plan_id")): row for row in read_jsonl(args.l2_dir / "plan_bank.jsonl")}
    l2 = [l2_row(row, plan_bank) for row in read_jsonl(args.l2_dir / "surfaces.L2.jsonl")]
    rows = l1 + l2
    for row in rows:
        attach_group(row)

    dev_total = args.dev_smoke + args.dev_tune + args.dev_select
    train_target = args.train_target or max(0, len(rows) - dev_total - args.final_test)
    targets = {
        "train": train_target,
        "dev_smoke": args.dev_smoke,
        "dev_tune": args.dev_tune,
        "dev_select": args.dev_select,
        "final_test": args.final_test,
    }
    split_groups = assign_splits(rows, seed=args.seed, targets=targets,
                                 train_abstain_share=args.train_abstain_share)
    group_to_split = {group_id: split for split, group_ids in split_groups.items() for group_id in group_ids}
    for row in rows:
        row["split"] = group_to_split[row["dedup_group"]]
        row["split_family"] = "dev" if row["split"].startswith("dev_") else (
            "final" if row["split"].startswith("final_") else "train"
        )
        if row["split"] == "final_test":
            row["final_bucket"] = "ood" if is_ood_candidate(row) else "iid"
        else:
            row["final_bucket"] = ""

    rows.sort(key=lambda r: (r["split"], r["source"], str(r["id"])))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "all.jsonl", rows)
    for split in ("train", "dev_smoke", "dev_tune", "dev_select", "final_test"):
        write_jsonl(args.out_dir / f"{split}.jsonl", [r for r in rows if r["split"] == split])
    write_jsonl(args.out_dir / "final_iid.jsonl",
                [r for r in rows if r["split"] == "final_test" and r["final_bucket"] == "iid"])
    write_jsonl(args.out_dir / "final_ood.jsonl",
                [r for r in rows if r["split"] == "final_test" and r["final_bucket"] == "ood"])

    groups_by_id = grouped(rows)
    groups = [group_summary(group_id, members) | {"split": group_to_split[group_id]}
              for group_id, members in groups_by_id.items()]
    groups.sort(key=lambda g: (g["split"], g["dedup_group"]))
    write_jsonl(args.out_dir / "groups.jsonl", groups)

    summary = summarise(rows) | {
        "inputs": {
            "l1_clean": str(args.l1_clean),
            "l2_surfaces": str(args.l2_dir / "surfaces.L2.jsonl"),
            "l2_plan_bank": str(args.l2_dir / "plan_bank.jsonl"),
        },
        "targets": targets,
        "seed": args.seed,
        "outputs": {
            "all": str(args.out_dir / "all.jsonl"),
            "groups": str(args.out_dir / "groups.jsonl"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
