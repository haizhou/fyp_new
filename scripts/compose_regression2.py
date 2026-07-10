#!/usr/bin/env python3
"""Regression gate, implementation #2: the independent evaluator must also
reproduce the frozen oracles on translated old questions. Same protocol as
compose_regression.py but answers come from scripts/compose_independent_eval.py
(raw-parquet universe, zero shared code with the runtime evaluator).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from procurement_graph.compose.from_gold import Unsupported, gold_plan_to_tree  # noqa: E402

_spec = importlib.util.spec_from_file_location("indep", ROOT / "scripts/compose_independent_eval.py")
indep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(indep)

_reg_spec = importlib.util.spec_from_file_location("reg1", ROOT / "scripts/compose_regression.py")
reg1 = importlib.util.module_from_spec(_reg_spec)
_reg_spec.loader.exec_module(reg1)
_match = reg1._match  # identical scoring for both implementations


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="final_test")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = []
    with (ROOT / f"data/qa/cicada_core_v4/{args.split}.jsonl").open() as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("expected_status") == "answerable":
                rows.append(row)
    if args.limit:
        rows = rows[: args.limit]
    print(f"{args.split}: {len(rows)} answerable rows (implementation #2)")

    df = indep.load_universe()
    stats = defaultdict(lambda: [0, 0, 0])
    mismatches, records = [], []
    for row in rows:
        family = str((row.get("gold_plan") or {}).get("metadata", {}).get("template_family")
                     or row.get("template_family") or "?")
        try:
            tree = gold_plan_to_tree(row)
        except Unsupported:
            stats[family][2] += 1
            continue
        result = indep.run_tree(df, tree)
        result.setdefault("status", "failed")
        wrapped = {"status": result["status"], "answer": result.get("answer")}
        ok = _match(row.get("oracle_answer"), wrapped)
        stats[family][0 if ok else 1] += 1
        records.append({"id": row["id"], "family": family, "ok": ok,
                        "oracle": row.get("oracle_answer"), "result": result})
        if not ok:
            mismatches.append({"id": row["id"], "family": family, "question": row["question"],
                               "oracle": row.get("oracle_answer"), "result": result})

    total_m = sum(v[0] for v in stats.values())
    total_x = sum(v[1] for v in stats.values())
    total_s = sum(v[2] for v in stats.values())
    covered = total_m + total_x
    print(f"\nIMPL#2 AGREEMENT: {total_m}/{covered} = {100.0 * total_m / max(1, covered):.2f}% "
          f"(skipped {total_s})")
    for family in sorted(stats):
        m, x, s = stats[family]
        if x:
            print(f"  MISMATCH family {family}: {x}")
    out = ROOT / f"data/qa/compose_regression2_{args.split}.json"
    out.write_text(json.dumps({"match": total_m, "mismatch": total_x, "skipped": total_s,
                               "mismatches": mismatches[:100]}, indent=1, default=str))
    print(f"wrote {out}")

    rng = random.Random(20260710)
    print("\n=== RAW SAMPLE (10, deterministic seed) ===")
    for r in rng.sample(records, min(10, len(records))):
        print(json.dumps(r, default=str)[:350])


if __name__ == "__main__":
    main()
