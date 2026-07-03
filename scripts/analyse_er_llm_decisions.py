"""Analyse full-corpus ER LLM decisions without mutating entity tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from procurement_graph.common.normalise import is_official, scheme_of, value_of


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "ablation" / "er"
REPORT_DIR = ROOT / "reports" / "ablation" / "er"


@dataclass
class UnionFind:
    parent: dict[str, str]

    def find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[max(root_left, root_right)] = min(root_left, root_right)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    try:
        parsed = json.loads(str(value) or "[]")
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def load_queue(queue_path: Path) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(queue_path):
        evidence = row.get("evidence") or {}
        task_id = str(row.get("task_id") or evidence.get("task_id") or "")
        if not task_id:
            continue
        entity_a = evidence.get("entity_a") or {}
        entity_b = evidence.get("entity_b") or {}
        features = evidence.get("features") or {}
        tasks[task_id] = {
            "task_id": task_id,
            "entity_a_id": str(entity_a.get("canonical_id") or ""),
            "entity_b_id": str(entity_b.get("canonical_id") or ""),
            "name_a": str(entity_a.get("name") or ""),
            "name_b": str(entity_b.get("name") or ""),
            "norm_name_a": str(entity_a.get("normalised_name") or ""),
            "norm_name_b": str(entity_b.get("normalised_name") or ""),
            **{f"feature_{key}": value for key, value in features.items()},
        }
    return tasks


def final_llm_decisions(decision_path: Path, queue_tasks: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    line_counts: Counter[str] = Counter()
    total_lines = 0
    for total_lines, row in enumerate(read_jsonl(decision_path), start=1):
        task_id = str(row.get("task_id") or "")
        if not task_id:
            continue
        row = dict(row)
        row["_decision_line"] = total_lines
        latest[task_id] = row
        line_counts[task_id] += 1

    rows: list[dict[str, Any]] = []
    for task_id, task in queue_tasks.items():
        decision = latest.get(task_id)
        if not decision:
            continue
        merged = {**task, **decision}
        merged["decision_source"] = "llm"
        merged["duplicate_decision_lines"] = line_counts[task_id]
        merged["is_schema_valid"] = validate_pair_decision(merged)
        rows.append(merged)

    metadata = {
        "decision_lines": total_lines,
        "decision_unique_task_ids": len(latest),
        "queue_task_ids": len(queue_tasks),
        "extra_decision_task_ids": len(set(latest) - set(queue_tasks)),
        "missing_decision_task_ids": len(set(queue_tasks) - set(latest)),
        "duplicate_task_ids": sum(1 for count in line_counts.values() if count > 1),
    }
    return rows, metadata


def validate_pair_decision(row: dict[str, Any]) -> bool:
    a, b = row_pair_ids(row)
    decision = row.get("decision")
    approved = {str(value) for value in row.get("approved_entity_ids") or []}
    excluded = {str(value) for value in row.get("excluded_entity_ids") or []}
    pair = {a, b}
    if not a or not b:
        return False
    if decision == "merge":
        return approved == pair and not excluded
    if decision in {"do_not_merge", "uncertain"}:
        return not approved and excluded == pair
    return False


def row_pair_ids(row: dict[str, Any]) -> tuple[str, str]:
    a = str(row.get("entity_a_id") or "")
    b = str(row.get("entity_b_id") or "")
    if a and b:
        return a, b
    ids = [str(value) for value in (row.get("approved_entity_ids") or row.get("excluded_entity_ids") or [])]
    if len(ids) >= 2:
        return ids[0], ids[1]
    return a, b


def load_rule_decisions(path: Path, queue_tasks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_jsonl(path):
        task_id = str(row.get("task_id") or "")
        task = queue_tasks.get(task_id, {})
        merged = {**task, **row}
        if not merged.get("entity_a_id") or not merged.get("entity_b_id"):
            left, right = row_pair_ids(merged)
            merged["entity_a_id"] = left
            merged["entity_b_id"] = right
        merged["decision_source"] = row.get("decision_source") or "deterministic_rule"
        merged["is_schema_valid"] = validate_pair_decision(merged)
        rows.append(merged)
    return rows


def decision_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    total = len(rows)
    by_decision = Counter(str(row.get("decision")) for row in rows)
    for decision, count in sorted(by_decision.items()):
        summary.append({"metric": f"decision::{decision}", "value": count, "share": count / total if total else 0})
    by_source = Counter(str(row.get("decision_source")) for row in rows)
    for source, count in sorted(by_source.items()):
        summary.append({"metric": f"decision_source::{source}", "value": count, "share": count / total if total else 0})
    valid = sum(1 for row in rows if row.get("is_schema_valid"))
    summary.append({"metric": "schema_valid", "value": valid, "share": valid / total if total else 0})
    summary.append({"metric": "schema_invalid", "value": total - valid, "share": (total - valid) / total if total else 0})
    return summary


def risk_flag_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    decision_counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        decision = str(row.get("decision"))
        for flag in row.get("risk_flags") or []:
            counts[str(flag)] += 1
            decision_counts[(decision, str(flag))] += 1
    output = []
    for flag, count in counts.most_common():
        item = {"risk_flag": flag, "count": count}
        for decision in ["merge", "do_not_merge", "uncertain"]:
            item[f"{decision}_count"] = decision_counts[(decision, flag)]
        output.append(item)
    return output


def pair_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted([str(left), str(right)]))  # type: ignore[return-value]


def load_auto_merge_edges(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    rows: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        rows.append(
            {
                "entity_a_id": str(row["entity_a_id"]),
                "entity_b_id": str(row["entity_b_id"]),
                "decision": "merge",
                "decision_source": "deterministic_auto_merge_candidate",
                "confidence": row.get("heuristic_score"),
                "risk_flags": json_list(row.get("risk_flags")),
                "reason": row.get("tier_reason", ""),
                "is_schema_valid": True,
                "name_a": row.get("name_a", ""),
                "name_b": row.get("name_b", ""),
                "feature_heuristic_score": row.get("heuristic_score"),
                "feature_strategies": json_list(row.get("strategies")),
            }
        )
    return rows


def load_entity_metadata(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    metadata: dict[str, dict[str, Any]] = {}
    for row in df.to_dict(orient="records"):
        canonical_id = str(row.get("canonical_id") or "")
        if canonical_id:
            metadata[canonical_id] = row
    return metadata


def official_ids_for(entity_id: str, metadata: dict[str, dict[str, Any]]) -> set[tuple[str, str]]:
    ids: set[tuple[str, str]] = set()
    row = metadata.get(entity_id, {})
    scheme = str(row.get("official_scheme") or "")
    value = str(row.get("official_id") or "")
    if scheme and value and is_official(scheme):
        ids.add((scheme, value))
    direct_scheme = scheme_of(entity_id)
    direct_value = value_of(entity_id)
    if direct_scheme and direct_value and is_official(direct_scheme):
        ids.add((direct_scheme, direct_value))
    for raw_id in json_list(row.get("alias_raw_ids")):
        alias_scheme = scheme_of(str(raw_id))
        alias_value = value_of(str(raw_id))
        if alias_scheme and alias_value and is_official(alias_scheme):
            ids.add((alias_scheme, alias_value))
    return ids


def build_cluster_reports(
    merge_edges: list[dict[str, Any]],
    hard_negative_edges: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    uf = UnionFind(parent={})
    for edge in merge_edges:
        left, right = row_pair_ids(edge)
        if left and right:
            uf.union(left, right)

    members_by_root: dict[str, set[str]] = defaultdict(set)
    for edge in merge_edges:
        for entity_id in row_pair_ids(edge):
            if entity_id:
                members_by_root[uf.find(entity_id)].add(entity_id)

    negative_by_pair = {
        pair_key(*row_pair_ids(edge)): edge
        for edge in hard_negative_edges
        if all(row_pair_ids(edge))
    }

    component_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    safe_edge_rows: list[dict[str, Any]] = []

    for root, members in sorted(members_by_root.items(), key=lambda item: (-len(item[1]), item[0])):
        sorted_members = sorted(members)
        official_by_scheme: dict[str, set[str]] = defaultdict(set)
        for member in sorted_members:
            for scheme, value in official_ids_for(member, metadata):
                official_by_scheme[scheme].add(value)

        official_conflicts = {
            scheme: sorted(values)
            for scheme, values in official_by_scheme.items()
            if len(values) > 1
        }
        internal_negative_edges = []
        for i, left in enumerate(sorted_members):
            for right in sorted_members[i + 1 :]:
                edge = negative_by_pair.get(pair_key(left, right))
                if edge:
                    internal_negative_edges.append(
                        {
                            "entity_a_id": left,
                            "entity_b_id": right,
                            "decision_source": edge.get("decision_source"),
                            "reason": edge.get("reason", ""),
                            "risk_flags": edge.get("risk_flags", []),
                        }
                    )

        component_edges = [
            edge
            for edge in merge_edges
            if row_pair_ids(edge)[0] in members and row_pair_ids(edge)[1] in members
        ]
        sources = sorted({str(edge.get("decision_source") or "") for edge in component_edges})
        component_id = root
        has_conflict = bool(official_conflicts or internal_negative_edges)
        component_row = {
            "component_id": component_id,
            "component_size": len(sorted_members),
            "member_ids": json.dumps(sorted_members, ensure_ascii=False),
            "member_names": json.dumps([str(metadata.get(member, {}).get("canonical_name") or "") for member in sorted_members], ensure_ascii=False),
            "merge_edge_count": len(component_edges),
            "decision_sources": json.dumps(sources, ensure_ascii=False),
            "official_ids_by_scheme": json.dumps({k: sorted(v) for k, v in official_by_scheme.items()}, ensure_ascii=False),
            "official_conflicts": json.dumps(official_conflicts, ensure_ascii=False),
            "internal_do_not_merge_edges": json.dumps(internal_negative_edges, ensure_ascii=False),
            "has_conflict": has_conflict,
        }
        component_rows.append(component_row)
        if has_conflict:
            conflict_rows.append(component_row)
        else:
            for edge in component_edges:
                safe_edge = dict(edge)
                safe_edge["component_id"] = component_id
                safe_edge["component_size"] = len(sorted_members)
                safe_edge_rows.append(safe_edge)

    return component_rows, conflict_rows, safe_edge_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sample_merge_rows(
    rows: list[dict[str, Any]],
    sample_size: int,
    safe_task_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    safe_task_ids = safe_task_ids or set()
    merge_rows = [row for row in rows if row.get("decision") == "merge" and row.get("is_schema_valid")]
    if not merge_rows:
        return []
    df = pd.DataFrame(merge_rows)
    df["_risk_flags_key"] = df["risk_flags"].map(lambda value: json.dumps(value or [], ensure_ascii=False, sort_keys=True))
    if len(df) <= sample_size:
        sample = df
    else:
        sample = (
            df.sort_values(["confidence", "task_id"], ascending=[True, True])
            .groupby("_risk_flags_key", dropna=False, group_keys=False)
            .head(max(1, sample_size // max(1, df["_risk_flags_key"].nunique())))
        )
        if len(sample) < sample_size:
            remaining = df.drop(sample.index, errors="ignore").sample(
                n=min(sample_size - len(sample), len(df) - len(sample)),
                random_state=42,
            )
            sample = pd.concat([sample, remaining], ignore_index=True)
    columns = [
        "task_id",
        "decision",
        "confidence",
        "safe_after_cluster_check",
        "entity_a_id",
        "entity_b_id",
        "name_a",
        "name_b",
        "norm_name_a",
        "norm_name_b",
        "feature_heuristic_score",
        "feature_risk_flags",
        "risk_flags",
        "reason",
    ]
    sample["safe_after_cluster_check"] = sample["task_id"].astype(str).isin(safe_task_ids)
    return sample[[column for column in columns if column in sample.columns]].to_dict(orient="records")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, default=DATA_DIR / "llm_review_queue_filtered.jsonl")
    parser.add_argument("--decisions", type=Path, default=DATA_DIR / "llm_decisions.jsonl")
    parser.add_argument("--rule-decisions", type=Path, default=DATA_DIR / "rule_decided_official_scheme_conflict.jsonl")
    parser.add_argument("--auto-merge", type=Path, default=DATA_DIR / "deterministic_auto_merge_candidates.parquet")
    parser.add_argument("--entities", type=Path, default=ROOT / "data" / "entities" / "canonical_orgs.parquet")
    parser.add_argument("--sample-size", type=int, default=100)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    queue_tasks = load_queue(args.queue)
    llm_rows, metadata = final_llm_decisions(args.decisions, queue_tasks)
    rule_rows = load_rule_decisions(args.rule_decisions, queue_tasks)
    combined_decisions = [*llm_rows, *rule_rows]

    final_jsonl = DATA_DIR / "final_llm_decisions.jsonl"
    write_jsonl(final_jsonl, llm_rows)
    pd.DataFrame(llm_rows).to_parquet(DATA_DIR / "final_llm_decisions.parquet", index=False)
    pd.DataFrame(combined_decisions).to_parquet(DATA_DIR / "final_er_review_decisions.parquet", index=False)

    summary_rows = [
        {"metric": key, "value": value, "share": ""}
        for key, value in metadata.items()
    ]
    summary_rows.extend(decision_summary(llm_rows))
    summary_rows.extend(
        {**row, "metric": f"combined::{row['metric']}"}
        for row in decision_summary(combined_decisions)
    )
    write_csv(REPORT_DIR / "llm_decision_summary.csv", summary_rows)
    write_csv(REPORT_DIR / "llm_decision_risk_flags.csv", risk_flag_summary(llm_rows))

    valid_llm_merges = [
        row for row in llm_rows if row.get("decision") == "merge" and row.get("is_schema_valid")
    ]
    auto_merge_edges = load_auto_merge_edges(args.auto_merge)
    merge_edges = [*auto_merge_edges, *valid_llm_merges]
    hard_negative_edges = [
        row
        for row in combined_decisions
        if row.get("decision") == "do_not_merge" and row.get("is_schema_valid")
    ]
    metadata_by_entity = load_entity_metadata(args.entities)
    components, conflicts, safe_edges = build_cluster_reports(merge_edges, hard_negative_edges, metadata_by_entity)
    safe_llm_task_ids = {
        str(edge.get("task_id"))
        for edge in safe_edges
        if edge.get("decision_source") == "llm" and edge.get("task_id")
    }

    pd.DataFrame(valid_llm_merges).to_parquet(DATA_DIR / "final_llm_merge_edges.parquet", index=False)
    pd.DataFrame(merge_edges).to_parquet(DATA_DIR / "proposed_merge_edges_before_cluster_check.parquet", index=False)
    pd.DataFrame(safe_edges).to_parquet(DATA_DIR / "safe_merge_edges_after_cluster_check.parquet", index=False)

    write_csv(REPORT_DIR / "llm_merge_sample.csv", sample_merge_rows(llm_rows, args.sample_size, safe_llm_task_ids))
    blocked_llm_merges = [
        row
        for row in valid_llm_merges
        if str(row.get("task_id")) not in safe_llm_task_ids
    ]
    write_csv(REPORT_DIR / "llm_merge_blocked_by_cluster_sample.csv", sample_merge_rows(blocked_llm_merges, args.sample_size, set()))
    write_csv(REPORT_DIR / "merge_components.csv", components)
    write_csv(REPORT_DIR / "merge_component_conflicts.csv", conflicts)
    write_csv(
        REPORT_DIR / "merge_cluster_check_summary.csv",
        [
            {"metric": "llm_valid_merge_edges", "value": len(valid_llm_merges)},
            {"metric": "deterministic_auto_merge_edges", "value": len(auto_merge_edges)},
            {"metric": "proposed_merge_edges_total", "value": len(merge_edges)},
            {"metric": "merge_components_total", "value": len(components)},
            {"metric": "merge_components_with_conflict", "value": len(conflicts)},
            {"metric": "safe_merge_edges_after_cluster_check", "value": len(safe_edges)},
        ],
    )

    print(json.dumps({
        **metadata,
        "llm_rows": len(llm_rows),
        "rule_rows": len(rule_rows),
        "llm_valid_merge_edges": len(valid_llm_merges),
        "deterministic_auto_merge_edges": len(auto_merge_edges),
        "merge_components_total": len(components),
        "merge_components_with_conflict": len(conflicts),
        "safe_merge_edges_after_cluster_check": len(safe_edges),
        "outputs": {
            "final_llm_decisions": str(final_jsonl),
            "summary": str(REPORT_DIR / "llm_decision_summary.csv"),
            "conflicts": str(REPORT_DIR / "merge_component_conflicts.csv"),
            "merge_sample": str(REPORT_DIR / "llm_merge_sample.csv"),
        },
    }, indent=2))


if __name__ == "__main__":
    main()
