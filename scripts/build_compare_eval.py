#!/usr/bin/env python3
"""Sample a comparison eval set: 20 questions per category across v1 (6 op-families) + v2 (5 subsets).

Read-only. Writes a unified `data/qa/eval/compare_set.jsonl` that both the KG-reasoning system and the
baseline RAG consume, so the comparison is apples-to-apples. Deterministic stride sampling.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "data" / "qa" / "generated"
V2 = ROOT / "data" / "qa" / "targeted_v2" / "full2k"
OUT = ROOT / "data" / "qa" / "eval" / "compare_set.jsonl"
N = 20
V1_FAMILIES = ["contract_factoid", "filtered_count", "additive_sum", "conjunction", "temporal_count", "cpv_slice"]
V2_SUBSETS = ["naturalized", "coverage_fixed", "unanswerable", "extended_ops", "bridge_join"]


def read_jsonl(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def stride(rows, n):
    if len(rows) <= n:
        return rows
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n)]


def main() -> None:
    out = []
    bench = read_jsonl(GEN / "benchmark.jsonl")
    for fam in V1_FAMILIES:
        pool = [b for b in bench if b["operation_family"] == fam]
        if fam == "contract_factoid":  # value_source is unsupported-by-design; exclude for a fair factoid set
            pool = [b for b in pool if not b["spec_id"].endswith("value_source")]
        pool = sorted(pool, key=lambda b: b["spec_id"])
        for b in stride(pool, N):
            out.append({"id": b["spec_id"], "dataset": "v1", "category": f"v1:{fam}",
                        "question": b["question"], "oracle_answer": b["golden_answer"],
                        "answer_type": b.get("answer_operation", ""), "expected_status": "answerable"})
    for subset in V2_SUBSETS:
        rows = read_jsonl(V2 / f"{subset}.full2k.accepted.jsonl")
        for r in stride(rows, N):
            out.append({"id": r["id"], "dataset": "v2", "category": f"v2:{subset}",
                        "question": r["question"], "oracle_answer": r.get("oracle_answer"),
                        "answer_type": r.get("answer_type", ""), "expected_status": r.get("expected_status", "answerable")})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8")
    import collections
    print(f"wrote {len(out)} questions to {OUT}")
    print("by category:", dict(collections.Counter(r["category"] for r in out)))


if __name__ == "__main__":
    main()
