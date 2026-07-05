#!/usr/bin/env python3
"""Evaluate planners across the multi-level QA benchmark (the generalization-gap experiment).

Level 0 (plan bank, --mode executor) measures the EXECUTOR ceiling with language removed.
Levels 1-3 (surfaces.L*.jsonl, pipeline mode) measure language-to-plan generalization: every
surface of a plan shares one oracle, so the accuracy drop from L1 (template) to L2 (paraphrase)
to L3 (adversarial) isolates what each planner's language layer actually generalizes to.

The headline artifact is the level x planner matrix:

    python -B scripts/eval_multilevel.py --planner rule_decomp          # offline
    python -B scripts/eval_multilevel.py --planner hybrid               # needs API key
    python -B scripts/eval_multilevel.py --mode executor                # L0 ceiling

Scoring is shared with eval_targeted_v2 (type-aware answer match; abstention correctness for
unanswerable rows), so numbers are comparable with the earlier per-subset evals.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_targeted_v2 import eval_row, summarize  # noqa: E402  (shared scoring)

ML_DIR = ROOT / "data" / "qa" / "multilevel"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_pipeline(args, backend, resolver):
    from procurement_graph.reasoning import ReasoningPipeline, RuleBasedDryRunPlanner

    planner: Any = RuleBasedDryRunPlanner()
    if args.planner == "rule_decomp":
        from procurement_graph.reasoning.planner_decomposition import DecompositionAwarePlanner
        planner = DecompositionAwarePlanner(org_resolver=resolver)
    elif args.planner == "llm":
        from procurement_graph.reasoning.llm_planner import LLMReasoningPlanner
        planner = LLMReasoningPlanner.from_env(args.model, org_resolver=resolver)
    elif args.planner == "typed":
        from procurement_graph.qa.benchmark.chat import ChatClient
        from procurement_graph.reasoning.typed_planning import TypedLLMPlanner
        planner = TypedLLMPlanner(client=ChatClient.from_env(), model=args.model, org_resolver=resolver,
                                  understanding_model=args.understanding_model)
    elif args.planner == "hybrid":
        from procurement_graph.reasoning.llm_planner import LLMReasoningPlanner
        from procurement_graph.reasoning.planner_decomposition import DecompositionAwarePlanner, HybridPlanner
        planner = HybridPlanner(rule=DecompositionAwarePlanner(org_resolver=resolver),
                                llm=LLMReasoningPlanner.from_env(args.model, org_resolver=resolver))
    elif args.planner == "verified_hybrid":
        from procurement_graph.reasoning.llm_planner import LLMReasoningPlanner
        from procurement_graph.reasoning.planner_decomposition import (
            DecompositionAwarePlanner, VerifyingHybridPlanner)
        planner = VerifyingHybridPlanner(rule=DecompositionAwarePlanner(org_resolver=resolver),
                                         backend=backend,
                                         llm=LLMReasoningPlanner.from_env(args.model, org_resolver=resolver))
    elif args.planner == "verified_typed":
        from procurement_graph.qa.benchmark.chat import ChatClient
        from procurement_graph.reasoning.planner_decomposition import (
            DecompositionAwarePlanner, VerifyingHybridPlanner)
        from procurement_graph.reasoning.typed_planning import TypedLLMPlanner
        planner = VerifyingHybridPlanner(
            rule=DecompositionAwarePlanner(org_resolver=resolver),
            backend=backend,
            llm=TypedLLMPlanner(client=ChatClient.from_env(), model=args.model, org_resolver=resolver,
                                understanding_model=args.understanding_model),
        )
    trace_reflector = None
    if args.reflect == "on":
        from procurement_graph.reasoning.trace_reflector import TraceReflector
        trace_reflector = TraceReflector(org_resolver=resolver)
    return ReasoningPipeline(backend=backend, planner=planner, org_resolver=resolver,
                             trace_reflector=trace_reflector)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["pipeline", "executor"], default="pipeline")
    ap.add_argument("--planner", choices=["rule", "rule_decomp", "hybrid", "verified_hybrid", "llm",
                                          "typed", "verified_typed"],
                    default="rule_decomp")
    ap.add_argument("--model", default="gpt-5.4-nano")
    ap.add_argument("--understanding-model", default="",
                    help="optional Step-1 understanding model for typed planners; defaults to --model")
    ap.add_argument("--levels", default="1,2,3")
    ap.add_argument("--limit", type=int, default=0, help="stride-sample surfaces per level")
    ap.add_argument("--max-per-level", type=int, default=0,
                    help="random-sample at most N surfaces per level (debug runs; see --sample-seed)")
    ap.add_argument("--sample-seed", type=int, default=20260702,
                    help="seed for --max-per-level sampling (same seed = same items across planners)")
    ap.add_argument("--progress-every", type=int, default=25,
                    help="print live progress every N evaluated rows per level (0 disables)")
    ap.add_argument("--reflect", choices=["off", "on"], default="off")
    ap.add_argument("--wrong-answer-repair", choices=["off", "on"], default="off",
                    help="offline teacher mode: if an answered row mismatches the hidden oracle, "
                         "send wrong_answer feedback to the planner once (oracle is not shown)")
    ap.add_argument("--trace-log", choices=["off", "on"], default="off",
                    help="write per-row trace JSONL files next to the result files")
    ap.add_argument("--in-dir", default=str(ML_DIR))
    ap.add_argument("--out-dir", default=str(ML_DIR / "eval"))
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from procurement_graph.reasoning.grounding import ground_spec  # noqa: F401  (executor path deps)
    from procurement_graph.reasoning.kg_backend import RuntimeKGBackend
    from procurement_graph.reasoning.verifier import backend_fields

    print("[ml-eval] loading KG ...", flush=True)
    backend = RuntimeKGBackend.from_directory(ROOT / "data" / "kg")
    allowed = frozenset(backend_fields(backend))
    pipeline = None
    if args.mode == "pipeline":
        pipeline = build_pipeline(args, backend, backend.org_resolver())

    if args.mode == "executor":
        sources = {"L0": in_dir / "plan_bank.jsonl"}
    else:
        sources = {f"L{lvl}": in_dir / f"surfaces.L{lvl}.jsonl"
                   for lvl in (int(x) for x in args.levels.split(",") if x.strip())}

    label = f"{args.mode}.{args.planner}" + (".reflect" if args.reflect == "on" else "")
    matrix: dict[str, Any] = {}
    for level, path in sources.items():
        if not path.exists():
            print(f"  (skip {level}: {path} missing — build it first)")
            continue
        rows = read_jsonl(path)
        excluded_decomposition = 0
        if args.mode == "executor":
            for row in rows:  # plan-bank rows carry the plan id; executor scoring needs id+question
                row.setdefault("id", row.get("plan_id", ""))
                row.setdefault("question", row.get("canonical_question", ""))
                row.setdefault("subset", row.get("source_subset", ""))
            # decomposition-only plans (bridge/compare) are not single-spec executable by design;
            # they are the decomposition executor's job, not an L0 miss.
            keep = [r for r in rows
                    if r.get("answer_operation") not in {"compare", "in_subquery"}
                    and not str(r.get("executor_support", "")).startswith("needs_op:")]
            excluded_decomposition = len(rows) - len(keep)
            rows = keep
        if args.max_per_level and len(rows) > args.max_per_level:
            rng = random.Random(args.sample_seed)
            rows = sorted(rng.sample(rows, args.max_per_level), key=lambda r: str(r.get("id")))
        elif args.limit and len(rows) > args.limit:
            step = len(rows) / args.limit
            rows = [rows[int(i * step)] for i in range(args.limit)]
        results = []
        trace_records = []
        total = len(rows)
        for index, row in enumerate(rows, start=1):
            outcome = eval_row(
                row,
                mode=args.mode,
                pipeline=pipeline,
                backend=backend,
                allowed=allowed,
                wrong_answer_repair=args.wrong_answer_repair == "on",
                return_trace=args.trace_log == "on",
            )
            trace_record = outcome.pop("_trace_record", None)
            if trace_record is not None:
                trace_records.append(trace_record)
            results.append({**outcome, "id": row.get("id"), "plan_id": row.get("plan_id"),
                            "subset": row.get("subset", ""), "level": level})
            if args.progress_every and (index % args.progress_every == 0 or index == total):
                matched = sum(1 for r in results if r.get("scored") and r.get("match"))
                scored = sum(1 for r in results if r.get("scored"))
                print(f"    {level}: evaluated {index}/{total}, matched={matched}/{scored}",
                      flush=True)
        summary = summarize(level, results, args.mode)
        if excluded_decomposition:
            summary["excluded_decomposition_plans"] = excluded_decomposition
        by_subset = Counter((r["subset"], r["match"]) for r in results if r.get("scored"))
        summary["by_subset_accuracy"] = {
            s: round(by_subset[(s, True)] / max(1, by_subset[(s, True)] + by_subset[(s, False)]), 3)
            for s in sorted({r["subset"] for r in results})}
        matrix[level] = summary
        (out_dir / f"{level}.{label}.results.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in results),
            encoding="utf-8")
        if args.trace_log == "on":
            (out_dir / f"{level}.{label}.traces.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in trace_records),
                encoding="utf-8")
        print(f"  {level}: acc={summary['accuracy']:.0%} scored={summary['scored']} "
              f"gap={summary['planner_or_exec_gap']}", flush=True)

    (out_dir / f"matrix.{label}.json").write_text(json.dumps(matrix, indent=2, ensure_ascii=False),
                                                  encoding="utf-8")
    print(f"\nwrote {out_dir / f'matrix.{label}.json'}")
    if len(matrix) > 1:
        accs = {lvl: s["accuracy"] for lvl, s in matrix.items()}
        print("generalization profile:", " -> ".join(f"{lvl} {a:.0%}" for lvl, a in accs.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
