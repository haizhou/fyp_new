#!/usr/bin/env python3
"""Evaluate the runtime reasoning pipeline over the targeted-v2 subsets.

Two modes make the one-executor-two-consumers property measurable:
  --mode pipeline   full plan -> ground -> execute (measures PLANNER + executor, end-to-end).
  --mode executor   build the spec directly from each row -> ground -> execute (measures the
                    EXECUTOR ceiling, isolating planning). Answerable rows only.

Scoring by `expected_status`:
  answerable            -> answer matches oracle (type-aware: numeric / date / string / set / top_k / bool)
  unsupported/ambiguous/no_results -> the runtime must ABSTAIN (answer is None). A produced answer is wrong.

Reads `<in-dir>/<subset>.<tag>.accepted.jsonl`; writes results + summary under --out-dir.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from procurement_graph.reasoning import ReasoningPipeline, RuleBasedDryRunPlanner
from procurement_graph.reasoning.kg_backend import RuntimeKGBackend
from procurement_graph.reasoning.models import QueryConstraint as QC, RuntimeQuerySpec
from procurement_graph.reasoning.grounding import ground_spec
from procurement_graph.reasoning.executor import execute_query_spec
from procurement_graph.reasoning.verifier import backend_fields

SUBSETS = ["naturalized", "coverage_fixed", "unanswerable", "extended_ops", "bridge_join"]
ABSTAIN = {"unsupported", "ambiguous", "no_results"}
# the reflection action that abstains for the RIGHT reason per expected_status. Abstaining via any
# other action (e.g. ask_clarifying for an unsupported field) is "safe but soft"; producing an
# answer is a hallucination.
EXPECTED_ABSTENTION_ACTION = {
    "unsupported": {"mark_unsupported"},
    "ambiguous": {"ask_clarifying_question"},
    "no_results": {"report_insufficient_evidence", "report_no_results"},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _num(v: Any):
    if v in (None, ""):
        return None
    m = re.search(r"[-+]?\d+(?:\.\d+)?", str(v).replace(",", ""))
    return float(m.group(0)) if m else None


def _tobool(v: Any) -> bool:
    return str(v).strip().casefold() in {"true", "1", "yes"}


def answers_match(pred: Any, oracle: Any, answer_type: str) -> bool:
    if pred is None:
        return False
    if answer_type == "boolean":
        return _tobool(pred) == _tobool(oracle)
    if answer_type in {"set_list"}:
        return sorted(map(str, pred)) == sorted(map(str, oracle)) if isinstance(pred, (list, tuple)) else False
    if answer_type == "top_k":
        try:
            return [[str(a), int(b)] for a, b in pred] == [[str(a), int(b)] for a, b in oracle]
        except Exception:
            return False
    if answer_type == "comparison":
        return isinstance(pred, dict) and pred.get("answer") == oracle.get("answer")
    pn, on = _num(pred), _num(oracle)
    if pn is not None and on is not None:
        return abs(pn - on) <= max(1e-6, abs(on) * 1e-6)
    pd = re.search(r"\d{4}-\d{2}-\d{2}", str(pred)); od = re.search(r"\d{4}-\d{2}-\d{2}", str(oracle))
    if pd and od:
        return pd.group(0) == od.group(0)
    return str(pred).strip().casefold() == str(oracle).strip().casefold()


def build_exec_spec(row: dict[str, Any]) -> RuntimeQuerySpec | None:
    """Reconstruct an executable spec from a v2 row. Returns None for ops the current executor
    cannot run (compare / in_subquery -> need the decomposition planner)."""
    op = row["answer_operation"]
    if op in {"compare", "in_subquery"} or row.get("executor_support") in {"needs_op:compare", "needs_op:in_subquery"}:
        return None
    cons = tuple(QC(c["field"], c["op"], c["value"]) for c in row["constraints"])
    md: dict[str, Any] = {}
    answer_field, sort_field, value_type = "", "", "string"
    if op == "sum":
        value_type = "currency"
    elif op == "select_unique":
        answer_field = row.get("answer_field") or "contract_node_id"
    elif op in {"argmax", "argmin"}:
        sort_field = row.get("metric") or "value_amount"
        answer_field = "contract_node_id"
    elif op == "distinct_set":
        answer_field = "supplier_name"  # v2 set_list templates collect distinct suppliers
    elif op == "rank_top_k":
        op = "top_k"
        k = int((re.search(r"top\s+(\d+)", row["question"], re.I) or [None, "3"])[1])
        md = {"group_by_field": row.get("group_by_field", "buyer_name"), "metric": row.get("metric", "count"), "k": k}
    return RuntimeQuerySpec(spec_id=row["id"], question=row["question"], intent="x", constraints=cons,
                            answer_operation=op, answer_field=answer_field, answer_value_type=value_type,
                            sort_field=sort_field, requires_exhaustive_retrieval=True, metadata=md)


def eval_row(row, *, mode, pipeline, backend, allowed):
    status = row.get("expected_status", "answerable")
    oracle, atype = row.get("oracle_answer"), row.get("answer_type", "")
    if mode == "executor":
        if status != "answerable":
            return {"scored": False}
        spec = build_exec_spec(row)
        if spec is None:
            return {"scored": True, "answered": False, "match": False, "reason": "needs_decomposition"}
        grounded = ground_spec(spec, allowed_fields=allowed)
        if not grounded.ok:
            return {"scored": True, "answered": False, "match": False, "reason": f"grounding:{grounded.reason}"}
        result = execute_query_spec(backend, grounded.spec)
        pred = result.answer if result.passed else None
        return {"scored": True, "answered": pred is not None, "match": answers_match(pred, oracle, atype),
                "reason": result.status}
    # pipeline mode
    trace = pipeline.run(row["question"])
    pred = trace.answer_card.answer
    action = trace.reflection.action if trace.reflection else ""
    if status in ABSTAIN:
        abstained = pred is None  # primary correctness: did NOT hallucinate an answer
        return {"scored": True, "abstain_case": True, "status": status,
                "answered": pred is not None, "hallucinated": pred is not None,
                "match": abstained,  # safe abstention
                "reason_matched": abstained and action in EXPECTED_ABSTENTION_ACTION.get(status, set()),
                "reason": action}
    return {"scored": True, "answered": pred is not None, "match": answers_match(pred, oracle, atype),
            "reason": action}


def summarize(subset, results, mode):
    scored = [r for r in results if r["scored"]]
    matched = sum(1 for r in scored if r["match"])
    answered = sum(1 for r in scored if r.get("answered"))
    abstain = [r for r in scored if r.get("abstain_case")]
    ans_rows = [r for r in scored if not r.get("abstain_case")]
    abstention_by_status = {}
    for st in ("unsupported", "ambiguous", "no_results"):
        rows_st = [r for r in abstain if r.get("status") == st]
        if rows_st:
            abstention_by_status[st] = {
                "cases": len(rows_st),
                "abstained_safely": sum(1 for r in rows_st if r["match"]),
                "abstained_right_reason": sum(1 for r in rows_st if r.get("reason_matched")),
                "hallucinated": sum(1 for r in rows_st if r.get("hallucinated")),
            }
    return {
        "subset": subset, "mode": mode, "scored": len(scored),
        "matched": matched, "accuracy": round(matched / len(scored), 4) if scored else 0.0,
        "answerable_scored": len(ans_rows),
        "answerable_matched": sum(1 for r in ans_rows if r["match"]),
        "abstention_cases": len(abstain),
        # safe = did not hallucinate; right_reason = abstained via the expected mechanism.
        "abstained_safely": sum(1 for r in abstain if r["match"]),
        "abstained_right_reason": sum(1 for r in abstain if r.get("reason_matched")),
        "hallucinated": sum(1 for r in abstain if r.get("hallucinated")),
        "abstention_by_status": abstention_by_status,
        "answered_not_matched": sum(1 for r in ans_rows if r.get("answered") and not r["match"]),
        "planner_or_exec_gap": sum(1 for r in ans_rows if not r.get("answered")),
        "reason": dict(Counter(r.get("reason", "") for r in scored)),
    }


def run(args):
    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[v2-eval] loading KG ...", flush=True)
    backend = RuntimeKGBackend.from_directory(ROOT / "data" / "kg")
    allowed = frozenset(backend_fields(backend))
    pipeline = None
    if args.mode == "pipeline":
        resolver = backend.org_resolver()
        planner = RuleBasedDryRunPlanner()
        if args.planner == "rule_decomp":
            from procurement_graph.reasoning.planner_decomposition import DecompositionAwarePlanner
            planner = DecompositionAwarePlanner(org_resolver=resolver)
        elif args.planner == "llm":
            from procurement_graph.reasoning.llm_planner import LLMReasoningPlanner
            planner = LLMReasoningPlanner.from_env(args.model, org_resolver=resolver)
        elif args.planner == "typed":
            from procurement_graph.qa.benchmark.chat import ChatClient
            from procurement_graph.reasoning.typed_planning import TypedLLMPlanner
            planner = TypedLLMPlanner(client=ChatClient.from_env(), model=args.model, org_resolver=resolver)
        elif args.planner == "hybrid":
            from procurement_graph.reasoning.planner_decomposition import DecompositionAwarePlanner, HybridPlanner
            from procurement_graph.reasoning.llm_planner import LLMReasoningPlanner
            planner = HybridPlanner(rule=DecompositionAwarePlanner(org_resolver=resolver),
                                    llm=LLMReasoningPlanner.from_env(args.model, org_resolver=resolver))
        elif args.planner == "verified_typed":
            from procurement_graph.qa.benchmark.chat import ChatClient
            from procurement_graph.reasoning.planner_decomposition import (
                DecompositionAwarePlanner, VerifyingHybridPlanner)
            from procurement_graph.reasoning.typed_planning import TypedLLMPlanner
            planner = VerifyingHybridPlanner(
                rule=DecompositionAwarePlanner(org_resolver=resolver),
                backend=backend,
                llm=TypedLLMPlanner(client=ChatClient.from_env(), model=args.model, org_resolver=resolver),
            )
        trace_reflector = None
        if args.reflect == "on":
            from procurement_graph.reasoning.trace_reflector import TraceReflector
            advisor = None
            if args.reflect_advisor == "on":
                from procurement_graph.qa.benchmark.chat import ChatClient
                from procurement_graph.reasoning.trace_reflector import LLMTraceAdvisor
                advisor = LLMTraceAdvisor(client=ChatClient.from_env(), model=args.model)
            trace_reflector = TraceReflector(org_resolver=resolver, advisor=advisor,
                                             preference_log=out_dir / "preference_log.jsonl")
        pipeline = ReasoningPipeline(backend=backend, planner=planner, org_resolver=resolver,
                                     trace_reflector=trace_reflector)
    subsets = SUBSETS if args.subset == "all" else [args.subset]
    summaries = {}
    for subset in subsets:
        path = in_dir / f"{subset}.{args.tag}.accepted.jsonl"
        if not path.exists():
            print(f"  (skip {subset}: {path} missing)"); continue
        rows = read_jsonl(path)
        if args.limit and args.limit < len(rows):
            # stride sample so a limit spans the whole file (subsets are grouped by status/family)
            step = len(rows) / args.limit
            rows = [rows[int(i * step)] for i in range(args.limit)]
        results = []
        for row in rows:
            out = eval_row(row, mode=args.mode, pipeline=pipeline, backend=backend, allowed=allowed)
            out.update({"id": row["id"], "answer_type": row.get("answer_type"),
                        "expected_status": row.get("expected_status"), "template_family": row.get("template_family")})
            results.append(out)
        (out_dir / f"{subset}.{args.mode}.results.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n", encoding="utf-8")
        s = summarize(subset, results, args.mode)
        summaries[subset] = s
        abst = (f" | abstain: safe {s['abstained_safely']}/{s['abstention_cases']}, "
                f"right-reason {s['abstained_right_reason']}, hallucinated {s['hallucinated']}"
                if s["abstention_cases"] else "")
        print(f"  {subset:14s} [{args.mode}] acc={s['accuracy']:.0%} | answerable {s['answerable_matched']}/{s['answerable_scored']}"
              f"{abst} | gap={s['planner_or_exec_gap']}", flush=True)
    (out_dir / f"eval_summary.{args.mode}.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote results + eval_summary.{args.mode}.json to {out_dir}")
    return 0


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", choices=["all", *SUBSETS], default="all")
    ap.add_argument("--mode", choices=["pipeline", "executor"], default="pipeline")
    ap.add_argument("--planner", choices=["rule", "rule_decomp", "llm", "typed", "hybrid", "verified_typed"],
                    default="rule")
    ap.add_argument("--model", default="gpt-5.4-nano")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--in-dir", default=str(ROOT / "data" / "qa" / "targeted_v2" / "full2k"))
    ap.add_argument("--tag", default="full2k")
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "qa" / "targeted_v2" / "full2k" / "eval"))
    ap.add_argument("--reflect", choices=["off", "on"], default="off",
                    help="trace-aware reflector: verify faithfulness, bounded repair, preference log")
    ap.add_argument("--reflect-advisor", choices=["off", "on"], default="off",
                    help="consult the LLM advisor on uncertain traces (needs AZURE_OPENAI_API_KEY)")
    raise SystemExit(run(ap.parse_args()))


if __name__ == "__main__":
    main()
