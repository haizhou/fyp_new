#!/usr/bin/env python3
"""Reconcile extended_ops top_k oracles with the shared executor's deterministic tiebreak.

The v2 top_k oracles were computed with pandas `value_counts` (insertion-order tiebreak); the shared
executor ranks by `(-count, key)`. At a rank-k tie boundary the two can disagree. Per the
one-executor-two-consumers thesis, the benchmark oracle must equal the executor. This recomputes each
top_k row's `oracle_answer` by RUNNING THE EXECUTOR, in place. Only top_k rows change.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from procurement_graph.reasoning.kg_backend import RuntimeKGBackend
from procurement_graph.reasoning.models import QueryConstraint as QC, RuntimeQuerySpec
from procurement_graph.reasoning.grounding import ground_spec
from procurement_graph.reasoning.executor import execute_query_spec
from procurement_graph.reasoning.verifier import backend_fields

TARGET = ROOT / "data" / "qa" / "targeted_v2" / "full2k" / "extended_ops.full2k.accepted.jsonl"


def top_k_spec(row: dict) -> RuntimeQuerySpec:
    k = int((re.search(r"top\s+(\d+)", row["question"], re.I) or [None, "3"])[1])
    return RuntimeQuerySpec(
        spec_id=row["id"], question=row["question"], intent="top_k",
        constraints=tuple(QC(c["field"], c["op"], c["value"]) for c in row["constraints"]),
        answer_operation="top_k", answer_field="", answer_value_type="string",
        requires_exhaustive_retrieval=True,
        metadata={"group_by_field": row.get("group_by_field", "buyer_name"),
                  "metric": row.get("metric", "count"), "k": k},
    )


def main() -> None:
    backend = RuntimeKGBackend.from_directory(ROOT / "data" / "kg")
    allowed = frozenset(backend_fields(backend))
    rows = [json.loads(line) for line in TARGET.read_text(encoding="utf-8").splitlines() if line.strip()]
    changed = 0
    for row in rows:
        if row.get("answer_type") != "top_k":
            continue
        grounded = ground_spec(top_k_spec(row), allowed_fields=allowed)
        assert grounded.ok, grounded.reason
        result = execute_query_spec(backend, grounded.spec)
        assert result.status == "passed", result.status
        canonical = [[k, int(v)] for k, v in result.answer]
        if canonical != [list(x) for x in row["oracle_answer"]]:
            changed += 1
            row["oracle_answer"] = canonical
            row["generation_notes"] = (row.get("generation_notes", "") +
                                       " | top_k oracle reconciled to executor tiebreak (-count,key)").strip(" |")
    TARGET.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    total = sum(1 for r in rows if r.get("answer_type") == "top_k")
    print(f"top_k rows: {total} | reconciled (changed): {changed} | oracle now == executor for all top_k")


if __name__ == "__main__":
    main()
