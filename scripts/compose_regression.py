#!/usr/bin/env python3
"""Regression gate for the compose algebra: every previously-solvable question
must be reproduced by the new evaluator.

Translates each answerable frozen gold plan into an algebra tree, evaluates it
with the runtime evaluator (implementation #1), and compares against the frozen
oracle answer. Reports agreement per template family, dumps every mismatch, and
prints 10 deterministic-random raw decisions for the mandatory human scan.

Usage:
  .venv/bin/python scripts/compose_regression.py [--split final_test] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from procurement_graph.compose.eval_runtime import RuntimeAlgebraEvaluator  # noqa: E402
from procurement_graph.compose.from_gold import Unsupported, gold_plan_to_tree  # noqa: E402
from procurement_graph.qa.benchmark.kg_interface import ParquetKGQueryBackend  # noqa: E402


def _as_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _match(oracle, result: dict) -> bool:
    if result.get("status") != "ok":
        return False
    answer = result.get("answer")
    if isinstance(oracle, dict) and "answer" in oracle:  # comparison oracles
        oracle = oracle["answer"]
    if isinstance(oracle, bool) or isinstance(answer, bool):
        return bool(oracle) == bool(answer)
    o_num, a_num = _as_number(oracle), _as_number(answer)
    if o_num is not None and a_num is not None:
        return abs(o_num - a_num) <= 0.01
    if isinstance(oracle, list) and isinstance(answer, list):
        # answer is ranking pairs but oracle stores bare ordered keys
        if answer and isinstance(answer[0], list) and (not oracle or not isinstance(oracle[0], list)):
            mine = [str(k) for k, _ in answer]
            theirs = [str(v) for v in oracle]
            if mine == theirs:
                return True
            # tie-equivalence: same key set and the oracle order is a valid
            # ordering under my values (non-increasing along oracle order)
            vals = {str(k): float(v) for k, v in answer}
            if set(mine) == set(theirs):
                seq = [vals[k] for k in theirs]
                return all(a >= b for a, b in zip(seq, seq[1:]))
            return False
        if oracle and isinstance(oracle[0], list):  # ranking [[key, value], ...]
            if len(oracle) != len(answer):
                return False
            for (ok, ov), (ak, av) in zip(oracle, answer):
                if str(ok) != str(ak) or abs(float(ov) - float(av)) > 0.01:
                    return False
            return True
        return sorted(str(v).strip() for v in oracle) == sorted(str(v).strip() for v in answer)
    return str(oracle).strip() == str(answer).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="final_test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = []
    path = ROOT / f"data/qa/cicada_core_v4/{args.split}.jsonl"
    with path.open() as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("expected_status") == "answerable":
                rows.append(row)
    if args.limit:
        rows = rows[: args.limit]
    print(f"{args.split}: {len(rows)} answerable rows")

    backend = ParquetKGQueryBackend.from_directory(ROOT / "data/kg", include_evidence=False)
    ev = RuntimeAlgebraEvaluator(backend)

    stats = defaultdict(lambda: [0, 0, 0])  # family -> [match, mismatch, skipped]
    mismatches, skipped, records = [], defaultdict(int), []
    for row in rows:
        family = str((row.get("gold_plan") or {}).get("metadata", {}).get("template_family")
                     or row.get("template_family") or "?")
        try:
            tree = gold_plan_to_tree(row)
        except Unsupported as exc:
            stats[family][2] += 1
            skipped[exc.reason] += 1
            continue
        result = ev.run(tree)
        ok = _match(row.get("oracle_answer"), result)
        stats[family][0 if ok else 1] += 1
        records.append({"id": row["id"], "family": family, "ok": ok,
                        "oracle": row.get("oracle_answer"), "result": result})
        if not ok:
            mismatches.append({"id": row["id"], "family": family, "question": row["question"],
                               "oracle": row.get("oracle_answer"), "result": result, "tree": tree})

    total_m = sum(v[0] for v in stats.values())
    total_x = sum(v[1] for v in stats.values())
    total_s = sum(v[2] for v in stats.values())
    covered = total_m + total_x
    print(f"\nAGREEMENT: {total_m}/{covered} = {100.0 * total_m / max(1, covered):.2f}% "
          f"(skipped {total_s} surface-borne/unsupported)")
    print(f"{'family':38s} {'match':>6s} {'miss':>5s} {'skip':>5s}")
    for family in sorted(stats):
        m, x, s = stats[family]
        flag = "  <-- MISMATCHES" if x else ""
        print(f"{family:38s} {m:6d} {x:5d} {s:5d}{flag}")
    if skipped:
        print("\nskip reasons:", dict(skipped))

    out = Path(args.out) if args.out else ROOT / f"data/qa/compose_regression_{args.split}.json"
    out.write_text(json.dumps({"split": args.split, "match": total_m, "mismatch": total_x,
                               "skipped": total_s, "per_family": {k: v for k, v in stats.items()},
                               "mismatches": mismatches[:200]}, indent=1, default=str))
    print(f"\nwrote {out}")

    # Iron rule: 10 deterministic-random raw decisions for human scan.
    rng = random.Random(20260710)
    sample = rng.sample(records, min(10, len(records)))
    print("\n=== RAW SAMPLE (10, deterministic seed, human-scan before citing) ===")
    for r in sample:
        print(json.dumps(r, default=str)[:400])


if __name__ == "__main__":
    main()
