#!/usr/bin/env python3
"""Screen-share demo: the deterministic core, with no LLM, GPU, or network.

    .venv/bin/python scripts/interview_demo.py            # all sections
    .venv/bin/python scripts/interview_demo.py --only kg  # one section

Sections: kg (graph scale), program (a gold program executed end to end),
results (sealed accuracy by data access).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULE = "=" * 72


def head(title: str) -> None:
    print(f"\n{RULE}\n  {title}\n{RULE}")


def section_kg() -> None:
    import pandas as pd

    head("1. THE KNOWLEDGE GRAPH  (166,277 OCDS releases, normalised)")
    for kind in ("nodes", "edges"):
        total = 0
        print(f"\n  {kind.upper()}")
        for f in sorted(glob.glob(str(ROOT / "data" / "kg" / kind / "*.parquet"))):
            n = len(pd.read_parquet(f).index)
            total += n
            print(f"    {os.path.basename(f)[:-8]:24s} {n:>10,}")
        print(f"    {'-' * 24} {'-' * 10}")
        print(f"    {'total ' + kind:24s} {total:>10,}")
    print("\n  Entity resolution is precision-first: 204,711 raw aliases collapse to")
    print("  131,502 canonical organisations. 82,013 singletons stay unmerged rather")
    print("  than be force-clustered. The Ministry of Defence alone appeared under 77")
    print("  platform identifiers in a single year.")


def section_program() -> None:
    head("2. ONE QUESTION, COMPILED AND EXECUTED  (no model in this path)")
    rows = [json.loads(l) for l in open(ROOT / "data/qa/cicada_core_v4/final_test.jsonl")]
    row = next(r for r in rows
               if r.get("gold_plan", {}).get("metadata", {}).get("source_subset") == "bridge_join"
               and r["answer_type"] == "count")
    print(f"\n  QUESTION\n    {row['question']}")
    print("\n  COMPILED PROGRAM (typed, closed algebra)")
    for line in json.dumps(row["gold_plan"], indent=2).splitlines():
        print(f"    {line}")
    print(f"\n  ORACLE ANSWER   {row['oracle_answer']:,}")
    print("\n  Note the in_subquery constraint. The answer needs every CPV code this")
    print("  buyer has ever used, then every notice under those codes. Top-k retrieval")
    print("  cannot enumerate that set; deterministic execution can.")


def section_results() -> None:
    head("3. SEALED TEST, BY DATA ACCESS  (n=2,285, one pass per frozen config)")
    AGG = {"count", "sum", "comparison", "set_list", "top_k", "min_max"}
    systems = [
        ("Closed book (no data)", "closed_book_grok/compare_cicada.results.jsonl"),
        ("Retrieval reader", "rag_grok/compare_rag_strong.results.jsonl"),
        ("Retrieval + gold plan", "plan_guided_rag_grok/compare_cicada.results.jsonl"),
        ("Qwen fully local", "fully_local_qwen/compare_cicada.results.jsonl"),
    ]
    print(f"\n  {'system':24s} {'overall':>8s} {'aggreg.':>8s} {'1-record':>9s} {'refusal':>8s}")
    print(f"  {'-' * 24} {'-' * 8} {'-' * 8} {'-' * 9} {'-' * 8}")
    for name, rel in systems:
        path = ROOT / "outputs/eval/final_test" / rel
        if not path.exists():
            print(f"  {name:24s}  (results file not found)")
            continue
        rows = [json.loads(l) for l in open(path)]
        ans = [r for r in rows if r.get("expected_status", "answerable") == "answerable"]
        ref = [r for r in rows if r.get("expected_status", "answerable") != "answerable"]
        agg = [r for r in ans if r["answer_type"] in AGG]
        one = [r for r in ans if r["answer_type"] not in AGG]
        pct = lambda xs: 100 * sum(r["correct"] for r in xs) / len(xs) if xs else float("nan")
        print(f"  {name:24s} {pct(rows):7.1f}% {pct(agg):7.1f}% {pct(one):8.1f}% {pct(ref):7.1f}%")
    print("\n  Retrieval is competent on single-record questions and collapses on")
    print("  exhaustive aggregation. Supplying the gold plan as the retrieval query")
    print("  does not rescue it, so the limit is enumeration, not query formulation.")


def section_audit(limit: int) -> None:
    head(f"4. INDEPENDENT ORACLE AUDIT, LIVE  ({limit} sealed rows recomputed)")
    print("\n  A second evaluator, sharing no execution code with the system, rebuilds")
    print("  each answer from the raw node and edge tables. Running it now...\n")
    cmd = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/qa_independent_eval.py"),
           "--qa", str(ROOT / "data/qa/cicada_core_v4/final_test.jsonl"),
           "--kg", str(ROOT / "data/kg"), "--out", "/tmp/interview_audit.json",
           "--limit", str(limit), "--convention", "flat"]
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "4"}
    out = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT).stdout
    for line in out.strip().splitlines():
        print(f"    {line}")
    print("\n  Over the full audited set this agreement is 99.88% (14,752 of 14,770).")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["kg", "program", "results", "audit"], default=None)
    ap.add_argument("--audit-rows", type=int, default=40)
    args = ap.parse_args()
    order = [("kg", section_kg), ("program", section_program),
             ("results", section_results), ("audit", lambda: section_audit(args.audit_rows))]
    for name, fn in order:
        if args.only in (None, name):
            fn()
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
