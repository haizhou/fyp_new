#!/usr/bin/env python3
"""Salvage oracle-correct rows mis-routed to hard_negatives by the Decimal shape-gate bug.

The full harvest ran with a shape gate that rejected decimal.Decimal answers (money sums),
so verifier-passing AND oracle-matching rows were routed to hard_negatives with
reject_reason="shape_mismatch" (750 sums + Decimal-valued bridges, etc.). The plans are
correct; no API rerun is needed. This script:

  1. re-checks each shape_mismatch hard-negative against the FIXED gate + trace oracle_match,
  2. appends salvaged rows to verified_sft.jsonl (acceptance tagged "shape_salvage"),
  3. rewrites hard_negatives.jsonl without them (original backed up to *.pre_salvage.bak).

Idempotent: rows already present in verified_sft (by id) are never appended twice; a second
run finds no shape_mismatch rows left to salvage.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_teacher import shape_ok  # fixed gate (Decimal-aware)  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher-dir", type=Path, default=Path("data/qa/teacher_full_v1"))
    ap.add_argument("--qa", type=Path, default=Path("data/qa/cicada_core_v4/train.jsonl"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    traces = {str(t["id"]): t for t in read_jsonl(args.teacher_dir / "traces.jsonl")}
    answer_type = {str(r["id"]): str(r.get("answer_type", "")) for r in read_jsonl(args.qa)}
    hard = read_jsonl(args.teacher_dir / "hard_negatives.jsonl")
    already_verified = {str(r["id"]) for r in read_jsonl(args.teacher_dir / "verified_sft.jsonl")}

    keep, salvage = [], []
    per_bucket: Counter = Counter()
    for row in hard:
        qid = str(row.get("id"))
        tr = traces.get(qid)
        eligible = (
            row.get("reject_reason") == "shape_mismatch"
            and tr is not None and bool(tr.get("oracle_match"))
            and shape_ok(tr.get("answer"), answer_type.get(qid, ""))
            and row.get("rejected_graph_plan") is not None
            and qid not in already_verified
        )
        if eligible:
            salvage.append({
                "id": qid, "bucket": row.get("bucket"), "question": row.get("question"),
                "step1_briefing": row.get("step1_briefing"),
                "target_graph_plan": row.get("rejected_graph_plan"),
                "attempt": tr.get("attempt_of_success") or 0, "oracle_match": True,
                "teacher": {"acceptance": "executor_verifier+shape_salvage"},
            })
            per_bucket[str(row.get("bucket"))] += 1
        else:
            keep.append(row)

    print(f"hard_negatives: {len(hard)} -> keep {len(keep)}, salvage {len(salvage)}")
    for b, n in per_bucket.most_common():
        print(f"  {b}: {n}")
    if args.dry_run or not salvage:
        return 0

    hn_path = args.teacher_dir / "hard_negatives.jsonl"
    shutil.copy2(hn_path, hn_path.with_suffix(".jsonl.pre_salvage.bak"))
    hn_path.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in keep),
                       encoding="utf-8")
    with (args.teacher_dir / "verified_sft.jsonl").open("a", encoding="utf-8") as f:
        for r in salvage:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print("done: hard_negatives rewritten (backup .pre_salvage.bak), verified_sft appended")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
