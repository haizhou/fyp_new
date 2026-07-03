#!/usr/bin/env python3
"""SMOKE TEST ONLY: run the hard-20 eval questions through the runtime reasoning pipeline.

Loads the real KG once, runs each `hard20` question through `ReasoningPipeline`, compares the
predicted answer to the oracle answer, and saves per-question results + a summary. It NEVER touches
the QA generation / benchmark files and makes no benchmark or verification calls.

Planners:
  --planner rule            offline, deterministic, no API (default) -- the true smoke test
  --planner llm --model M   uses the LLM planner (e.g. gpt-5.4-nano) via AZURE_OPENAI_API_KEY

Outputs (under data/qa/eval/ by default):
  <out>.jsonl        one row per question (predicted, oracle, match, confidence, sanity, plan)
  <out>.summary.json aggregate accuracy + per-category + boundary/sanity checks
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from procurement_graph.reasoning import (
    LLMCandidateSelector,
    LLMReflectionAnalyzer,
    LLMVerificationAnalyzer,
    ReasoningPipeline,
    RuleBasedDryRunPlanner,
    TopScoreCandidateSelector,
)
from procurement_graph.reasoning.kg_backend import RuntimeKGBackend

DEFAULT_QUESTIONS = ROOT / "data" / "qa" / "eval" / "hard20_questions_only.jsonl"
DEFAULT_KEY = ROOT / "data" / "qa" / "eval" / "hard20_answer_key.jsonl"
DEFAULT_OUT = ROOT / "data" / "qa" / "eval" / "hard20_runtime_results.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _to_num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group(0)) if match else None


def _norm_date(text: str) -> str:
    match = re.search(r"\d{4}-\d{2}-\d{2}", str(text))
    return match.group(0) if match else ""


def answers_match(predicted: Any, oracle: Any) -> bool:
    """Type-flexible oracle comparison: numeric (tolerance) -> date -> casefold string."""
    if predicted is None:
        return False
    pred_s, oracle_s = str(predicted).strip(), str(oracle).strip()
    pn, on = _to_num(pred_s), _to_num(oracle_s)
    if pn is not None and on is not None:
        return abs(pn - on) <= max(1e-6, abs(on) * 1e-6)
    pd, od = _norm_date(pred_s), _norm_date(oracle_s)
    if pd and od:
        return pd == od
    return pred_s.casefold() == oracle_s.casefold()


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def build_planner(kind: str, model: str) -> Any:
    if kind == "llm":
        from procurement_graph.reasoning.llm_planner import LLMReasoningPlanner

        return LLMReasoningPlanner.from_env(model)
    return RuleBasedDryRunPlanner()


def build_selector(kind: str, model: str) -> Any:
    if kind == "llm":
        from procurement_graph.qa.benchmark.chat import ChatClient

        return LLMCandidateSelector(client=ChatClient.from_env(), model=model)
    if kind == "top_score":
        return TopScoreCandidateSelector()
    return None


def run(args: argparse.Namespace) -> int:
    questions = read_jsonl(Path(args.questions))
    key = {row["id"]: row for row in read_jsonl(Path(args.answer_key))}
    if args.limit:
        questions = questions[: args.limit]

    print(f"[smoke] loading KG backend from {args.kg_dir} (one-time, ~40-60s) ...", flush=True)
    backend = RuntimeKGBackend.from_directory(args.kg_dir)
    candidate_retriever = backend.candidate_retriever() if args.semantic_repair else None
    candidate_selector = build_selector(args.candidate_selector, args.model) if args.semantic_repair else None
    verification_analyzer = None
    reflection_analyzer = None
    if args.llm_diagnostics:
        from procurement_graph.qa.benchmark.chat import ChatClient

        diagnostic_client = ChatClient.from_env()
        verification_analyzer = LLMVerificationAnalyzer(client=diagnostic_client, model=args.model)
        reflection_analyzer = LLMReflectionAnalyzer(client=diagnostic_client, model=args.model)
    pipeline = ReasoningPipeline(
        backend=backend,
        planner=build_planner(args.planner, args.model),
        org_resolver=backend.org_resolver(),
        candidate_retriever=candidate_retriever,
        candidate_selector=candidate_selector,
        verification_analyzer=verification_analyzer,
        reflection_analyzer=reflection_analyzer,
        semantic_top_k=args.semantic_top_k,
    )
    print(f"[smoke] planner={args.planner}{' model=' + args.model if args.planner == 'llm' else ''}; "
          f"semantic_repair={bool(args.semantic_repair)} selector={args.candidate_selector}; "
          f"llm_diagnostics={bool(args.llm_diagnostics)}; running {len(questions)} questions\n", flush=True)
    out_name = Path(args.out).name.lower()
    if args.planner == "rule" and any(tag in out_name for tag in ("nano", "llm", "gpt", "grok")):
        print(f"[WARN] --planner is 'rule' but --out looks like an LLM run ({Path(args.out).name}). "
              "The rule planner CANNOT answer factoids (no NL anchor resolution). "
              "For factoids pass: --planner llm --model gpt-5.4-nano (needs AZURE_OPENAI_API_KEY).\n", flush=True)

    results: list[dict[str, Any]] = []
    for q in questions:
        qid = q["id"]
        oracle_row = key.get(qid, {})
        oracle = oracle_row.get("oracle_answer")
        started = time.perf_counter()
        try:
            trace = pipeline.run(q["question"])
            card = trace.answer_card
            spec = card.query_spec
            metadata = trace.metadata or {}
            attempts = metadata.get("attempts", [])
            last_attempt = attempts[-1] if attempts else {}
            plans_trace = metadata.get("plans", [])
            selected_plan_trace = _selected_plan_trace(trace.selected_plan_id, plans_trace)
            predicted = card.answer
            expected_answerable = bool(oracle_row.get("expected_answerable", True))
            if not expected_answerable:
                match = predicted is None
            else:
                match = answers_match(predicted, oracle) if predicted is not None else False
            planned_cons = [[c.field, c.op, c.value] for c in spec.constraints]
            gap_diagnosis = "" if match else _gap_diagnosis(
                planned_cons, oracle_row.get("constraints", []), predicted, oracle)
            row = {
                "id": qid,
                "category": oracle_row.get("category", ""),
                "question_type": q.get("question_type", ""),
                "expected_answerable": expected_answerable,
                "answered": predicted is not None,
                "match": match,
                "predicted": _jsonable(predicted),
                "oracle": oracle,
                "confidence": card.confidence_label,
                "sanity_flags": list(card.sanity_flags),
                "reflection": trace.reflection.action if trace.reflection else "",
                "execution_status": trace.execution.status if trace.execution else "not_run",
                "planned_operation": spec.answer_operation,
                "planned_answer_field": spec.answer_field,
                "planned_answer_value_type": spec.answer_value_type,
                "planned_dedupe_key": spec.dedupe_key,
                "planned_sort_field": spec.sort_field,
                "planned_sort_direction": spec.sort_direction,
                "planned_constraints": planned_cons,
                "gap_diagnosis": gap_diagnosis,
                "limitations": list(card.limitations),
                "planner_source": selected_plan_trace.get("planner_source", ""),
                "fallback_used": bool(selected_plan_trace.get("fallback_used", False)),
                "fallback_reason": selected_plan_trace.get("fallback_reason", ""),
                "fallback_planner": selected_plan_trace.get("fallback_planner", ""),
                "raw_planner_response": _jsonable(selected_plan_trace.get("raw_response")),
                "pre_ground_spec": _jsonable(last_attempt.get("pre_ground_spec", {})),
                "grounded_spec": _jsonable(last_attempt.get("grounded_spec", {})),
                "grounding": _jsonable(last_attempt.get("grounding", {})),
                "preflight": _jsonable(last_attempt.get("preflight", {})),
                "failed_checks": _jsonable(last_attempt.get("failed_checks", [])),
                "execution_checks": _jsonable(last_attempt.get("execution_checks", [])),
                "attempts": _jsonable(attempts),
                "workflow": _jsonable(metadata.get("workflow", {})),
                "grounding_changes": metadata.get("grounding_changes", []),
                "answer_text": card.answer_text,
                "elapsed_s": round(time.perf_counter() - started, 3),
                "error": "",
            }
        except Exception as exc:  # a smoke test must survive a single bad question
            expected_answerable = bool(oracle_row.get("expected_answerable", True))
            row = {"id": qid, "category": oracle_row.get("category", ""), "question_type": q.get("question_type", ""),
                   "expected_answerable": expected_answerable,
                   "answered": False, "match": not expected_answerable, "predicted": None, "oracle": oracle,
                   "confidence": "not_answered", "sanity_flags": [], "reflection": "", "execution_status": "exception",
                   "planned_operation": "", "planned_answer_field": "", "planned_answer_value_type": "",
                   "planned_dedupe_key": "", "planned_sort_field": "", "planned_sort_direction": "",
                   "planned_constraints": [], "gap_diagnosis": "", "limitations": [],
                   "planner_source": "", "fallback_used": False, "fallback_reason": "",
                   "fallback_planner": "", "raw_planner_response": None, "pre_ground_spec": {}, "grounded_spec": {},
                   "grounding": {}, "preflight": {}, "failed_checks": [], "execution_checks": [], "attempts": [],
                   "workflow": {},
                   "grounding_changes": [], "answer_text": "",
                   "elapsed_s": round(time.perf_counter() - started, 3), "error": repr(exc)}
        results.append(row)
        mark = "OK " if row["match"] else ("-- " if row["answered"] else "   ")
        print(f"[{mark}] {qid:36s} pred={str(row['predicted'])[:28]:28s} oracle={str(oracle)[:22]:22s} "
              f"conf={row['confidence']}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = _summarize(results)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== SMOKE SUMMARY ===")
    print(f"planner: {args.planner}" + (f" ({args.model})" if args.planner == "llm" else "")
          + f"  [planner_sources={summary.get('planner_sources')}]")
    print(f"answered: {summary['answered']}/{summary['total']}  |  matched oracle: {summary['matched']}/{summary['total']}"
          f"  (accuracy {summary['accuracy']:.0%})")
    print(f"by category: {summary['by_category']}")
    if summary.get("gap_diagnoses"):
        print(f"mismatch diagnoses: {summary['gap_diagnoses']}")
        if summary.get("benchmark_artifact_ids"):
            print(f"  -> {len(summary['benchmark_artifact_ids'])} are benchmark artifacts (hidden coverage guards), NOT planning errors")
    print(f"boundary sanity: {summary['boundary_sanity_flagged']}/{summary['boundary_total']} flagged "
          f"(expect all boundary sums flagged)")
    if summary["errors"]:
        print(f"errors: {summary['errors']}")
    print(f"\nresults : {out_path}")
    print(f"summary : {summary_path}")
    return 0


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    matched = sum(1 for r in results if r["match"])
    answered = sum(1 for r in results if r["answered"])
    by_cat: dict[str, dict[str, int]] = {}
    for r in results:
        bucket = by_cat.setdefault(r["category"] or "?", {"matched": 0, "total": 0})
        bucket["total"] += 1
        bucket["matched"] += int(r["match"])
    boundary = [r for r in results if r["category"] == "boundary"]
    return {
        "total": total,
        "answered": answered,
        "matched": matched,
        "accuracy": round(matched / total, 4) if total else 0.0,
        "by_category": {k: f"{v['matched']}/{v['total']}" for k, v in sorted(by_cat.items())},
        "confidence_counts": dict(Counter(r["confidence"] for r in results)),
        "expected_answerable": sum(1 for r in results if r.get("expected_answerable", True)),
        "expected_abstention": sum(1 for r in results if not r.get("expected_answerable", True)),
        "correct_abstentions": sum(
            1 for r in results if not r.get("expected_answerable", True) and not r.get("answered")
        ),
        "boundary_total": len(boundary),
        "boundary_sanity_flagged": sum(1 for r in boundary if r["sanity_flags"]),
        "planner_sources": dict(Counter(r.get("planner_source", "") or "unknown" for r in results)),
        "fallback_used": sum(1 for r in results if r.get("fallback_used")),
        "fallback_reasons": dict(Counter(r.get("fallback_reason", "") for r in results if r.get("fallback_used"))),
        "not_answered_ids": [r["id"] for r in results if not r["answered"]],
        "mismatched_ids": [r["id"] for r in results if r["answered"] and not r["match"]],
        "gap_diagnoses": dict(Counter(r.get("gap_diagnosis", "") for r in results if r.get("gap_diagnosis"))),
        "benchmark_artifact_ids": [r["id"] for r in results if str(r.get("gap_diagnosis", "")).startswith("golden_over_constrained")],
        "errors": {r["id"]: r["error"] for r in results if r["error"]},
    }


# Coverage guards a Stage-1 spec adds internally but the natural-language question never states.
_COVERAGE_GUARD_FIELDS = {"supplier_count", "buyer_count", "value_is_additive"}


def _gap_diagnosis(planned_constraints: list[list[Any]], key_constraints: list[dict[str, Any]],
                   predicted: Any, oracle: Any) -> str:
    """Classify a numeric reduction mismatch: benchmark artifact vs genuine planning error.

    The conjunction family fails because the GOLDEN spec carries hidden supplier/buyer coverage
    guards absent from the question; the runtime count is the faithful one (always higher). This
    labels that case so the report does not misattribute a benchmark defect to the planner.
    """
    pn, on = _to_num(predicted), _to_num(oracle)
    if pn is None or on is None or abs(pn - on) <= max(1e-6, abs(on) * 1e-6):
        return ""
    planned_fields = {c[0] for c in planned_constraints}
    hidden = sorted({
        str(c.get("field")) for c in key_constraints
        if c.get("field") in _COVERAGE_GUARD_FIELDS and c.get("field") not in planned_fields
    })
    if hidden and pn > on:
        return f"golden_over_constrained: oracle applies hidden coverage guards {hidden} not in the question (runtime count is faithful)"
    return "planning_gap"


def _selected_plan_trace(selected_plan_id: str, plans_trace: list[dict[str, Any]]) -> dict[str, Any]:
    if selected_plan_id:
        for plan in plans_trace:
            if plan.get("plan_id") == selected_plan_id:
                return plan
    return plans_trace[0] if plans_trace else {}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Smoke-test the hard-20 questions through the runtime pipeline.")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--answer-key", default=str(DEFAULT_KEY))
    parser.add_argument("--kg-dir", default=str(ROOT / "data" / "kg"))
    parser.add_argument("--planner", choices=["rule", "llm"], default="rule")
    parser.add_argument("--model", default="gpt-5.4-nano")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--limit", type=int, default=0, help="run only the first N questions (0 = all 20)")
    parser.add_argument("--semantic-repair", action="store_true",
                        help="after exact no-results, try top-k candidate retrieval and selector-based constraint repair")
    parser.add_argument("--candidate-selector", choices=["top_score", "llm"], default="top_score",
                        help="selector used for semantic repair candidates")
    parser.add_argument("--semantic-top-k", type=int, default=8)
    parser.add_argument("--llm-diagnostics", action="store_true",
                        help="ask an LLM to diagnose verifier/reflector failures; advisory only")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
