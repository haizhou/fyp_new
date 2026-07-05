#!/usr/bin/env python3
"""Run the trace-first teacher loop over a dev_smoke JSONL split.

This script is for offline data construction, not final evaluation. The LLM proposes typed plans
and repairs; the KG executor plus the hidden benchmark answer decide whether an attempt is accepted.
The hidden answer is never sent back to the LLM.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_targeted_v2 import answers_match, _jsonable, _plan_trace, _trace_payload  # noqa: E402
from procurement_graph.qa.benchmark.chat import ChatClient  # noqa: E402
from procurement_graph.reasoning import ReasoningPipeline  # noqa: E402
from procurement_graph.reasoning.kg_backend import RuntimeKGBackend  # noqa: E402
from procurement_graph.reasoning.pipeline import _feedback_from_trace  # noqa: E402
from procurement_graph.reasoning.typed_planning import TypedLLMPlanner  # noqa: E402


ABSTAIN_STATUSES = {"unsupported", "ambiguous", "no_results"}
OUTPUT_FILES = (
    "traces.jsonl",
    "reflector_inputs.jsonl",
    "reflector_outputs.jsonl",
    "verified_sft.jsonl",
    "repair_sft.jsonl",
    "dpo_pairs.jsonl",
    "failures.jsonl",
    "shape_failures.json",
)


class _StaticPlanner:
    def __init__(self, candidates: Iterable[Any]):
        self.candidates = tuple(candidates)

    def plan(self, question: str):
        return self.candidates


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    v = value.strip().casefold()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def trace_answer(trace: Any) -> Any:
    return trace.answer_card.answer if trace.answer_card is not None else None


def final_acceptance(row: dict[str, Any], trace: Any) -> tuple[bool, str]:
    pred = trace_answer(trace)
    status = row.get("expected_status", "answerable")
    if status in ABSTAIN_STATUSES:
        if pred is None:
            return True, "safe_abstention"
        return False, "hallucinated_answer"
    if pred is None:
        return False, "missing_answer"
    if answers_match(pred, row.get("oracle_answer"), row.get("answer_type", "")):
        return True, "answer_correct"
    return False, "wrong_answer"


def execute_candidate_trace(base_pipeline: ReasoningPipeline, question: str, plans: Iterable[Any]) -> Any:
    pipeline = replace(
        base_pipeline,
        planner=_StaticPlanner(tuple(plans)),
        trace_reflector=None,
        oracle_matcher=None,
        max_feedback_replans=0,
    )
    return pipeline.run(question)


def selected_plan_from_trace(trace: Any) -> dict[str, Any]:
    attempts = (trace.metadata or {}).get("attempts") or []
    for attempt in reversed(attempts):
        plan = attempt.get("selected_plan")
        if plan:
            return _jsonable(plan)
    plans = (trace.metadata or {}).get("plans") or []
    return _jsonable(plans[0]) if plans else {}


def last_attempt_from_trace(trace: Any) -> dict[str, Any]:
    attempts = (trace.metadata or {}).get("attempts") or []
    return _jsonable(attempts[-1]) if attempts else {}


def make_feedback(row: dict[str, Any], trace: Any, outcome: str, attempt_index: int) -> dict[str, Any]:
    feedback = _jsonable(_feedback_from_trace(trace))
    feedback["attempt_index"] = attempt_index + 1
    feedback["offline_verifier_outcome"] = outcome
    feedback["submitted_answer"] = _jsonable(trace_answer(trace))
    feedback["question_id"] = row.get("id")
    feedback["expected_status"] = row.get("expected_status")
    feedback["answer_type"] = row.get("answer_type")
    if not feedback.get("failed_plan"):
        feedback["failed_plan"] = selected_plan_from_trace(trace)
    feedback["failure_stage"] = "verifier" if outcome in {
        "wrong_answer", "hallucinated_answer", "missing_answer"
    } else feedback.get("failure_stage", "verifier")
    feedback["failure_reason"] = outcome
    feedback["notes"] = list(feedback.get("notes") or ()) + [
        "Offline verifier rejected this attempt.",
        "The hidden reference answer is intentionally not included.",
        "Repair or abstain using only the question and trace summary.",
    ]
    return feedback


def plan_shape_failed(plan: dict[str, Any]) -> bool:
    if not plan:
        return True
    status = str(plan.get("status", ""))
    rationale = str(plan.get("rationale", "")).casefold()
    warnings = " ".join(map(str, plan.get("warnings") or ())).casefold()
    raw = json.dumps(plan.get("raw_response") or {}, ensure_ascii=False).casefold()
    if status != "planned":
        return True
    shape_terms = (
        "unparseable",
        "missing_repaired_plan",
        "semantic_mismatch",
        "placeholder",
        "unknown_slot",
        "invalid_slot",
        "typed_feedback_error",
        "typed_inconsistent",
    )
    haystack = " ".join((rationale, warnings, raw))
    return any(term in haystack for term in shape_terms)


def attempt_record(
    *,
    row: dict[str, Any],
    trace: Any,
    attempt_index: int,
    source: str,
    accepted: bool,
    outcome: str,
    feedback: dict[str, Any] | None,
    replans: Iterable[Any] = (),
) -> dict[str, Any]:
    selected = selected_plan_from_trace(trace)
    last = last_attempt_from_trace(trace)
    return {
        "attempt_index": attempt_index,
        "source": source,
        "accepted": accepted,
        "outcome": outcome,
        "answer": _jsonable(trace_answer(trace)),
        "plan": selected,
        "feedback": _jsonable(feedback),
        "replans": [_plan_trace(plan) for plan in replans],
        "compiler_issues": compiler_issues(selected, last),
        "executor_verifier_result": executor_verifier_result(trace, last),
        "trace": _trace_payload(trace),
        "question_id": row.get("id"),
    }


def compiler_issues(plan: dict[str, Any], attempt: dict[str, Any]) -> dict[str, Any]:
    grounding = attempt.get("grounding") or {}
    return {
        "planner_status": plan.get("status"),
        "planner_rationale": plan.get("rationale"),
        "planner_warnings": plan.get("warnings") or [],
        "grounding_ok": grounding.get("ok"),
        "grounding_reason": grounding.get("reason"),
        "grounding_issues": grounding.get("issues") or [],
        "schema_errors": (attempt.get("preflight") or {}).get("failed_checks") or [],
        "deterministic_guards_added": attempt.get("deterministic_guards_added")
        or grounding.get("deterministic_guards_added")
        or [],
    }


def executor_verifier_result(trace: Any, attempt: dict[str, Any]) -> dict[str, Any]:
    return {
        "execution_status": trace.execution.status if trace.execution is not None else "not_run",
        "attempt_status": attempt.get("status") or attempt.get("execution_status"),
        "execution_checks": attempt.get("execution_checks") or [],
        "failed_checks": attempt.get("failed_checks") or [],
        "evidence_verdict": attempt.get("evidence_verdict") or {},
        "answer_sanity": attempt.get("answer_sanity") or {},
        "postflight": attempt.get("postflight") or {},
        "reflection": attempt.get("reflector") or _jsonable(getattr(trace.reflection, "__dict__", {})),
    }


def process_row(
    row: dict[str, Any],
    *,
    base_pipeline: ReasoningPipeline,
    initial_planner: TypedLLMPlanner,
    repair_planner: TypedLLMPlanner,
    max_feedback_replans: int,
    teacher: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    question = row["question"]
    attempts: list[dict[str, Any]] = []
    feedbacks: list[dict[str, Any]] = []
    first_verified_attempt: int | None = None

    initial_plans = tuple(initial_planner.plan(question))
    trace = execute_candidate_trace(base_pipeline, question, initial_plans)
    accepted, outcome = final_acceptance(row, trace)
    attempts.append(attempt_record(row=row, trace=trace, attempt_index=0, source="initial",
                                   accepted=accepted, outcome=outcome, feedback=None,
                                   replans=initial_plans))
    current_trace = trace

    if accepted:
        first_verified_attempt = 0
    else:
        for repair_idx in range(1, max_feedback_replans + 1):
            feedback = make_feedback(row, current_trace, outcome, repair_idx - 1)
            feedbacks.append(feedback)
            append_jsonl(out_dir / "reflector_inputs.jsonl", {
                "question_id": row.get("id"),
                "repair_attempt": repair_idx,
                "teacher": teacher,
                "feedback": feedback,
            })
            replans = tuple(repair_planner.replan_with_feedback(question, feedback))
            append_jsonl(out_dir / "reflector_outputs.jsonl", {
                "question_id": row.get("id"),
                "repair_attempt": repair_idx,
                "teacher": teacher,
                "replans": [_plan_trace(plan) for plan in replans],
            })
            current_trace = execute_candidate_trace(base_pipeline, question, replans)
            accepted, outcome = final_acceptance(row, current_trace)
            attempts.append(attempt_record(row=row, trace=current_trace, attempt_index=repair_idx,
                                           source="repair", accepted=accepted, outcome=outcome,
                                           feedback=feedback, replans=replans))
            if accepted:
                first_verified_attempt = repair_idx
                break

    final_attempt = attempts[-1]
    verified_plan = final_attempt["plan"] if first_verified_attempt is not None else None
    failed_attempts = [attempt for attempt in attempts if not attempt["accepted"]]
    record = {
        "question_id": row.get("id"),
        "plan_id": row.get("plan_id"),
        "subset": row.get("subset") or row.get("source_dataset") or row.get("source"),
        "source": row.get("source"),
        "level": row.get("level"),
        "split": row.get("split"),
        "train_bucket": row.get("train_bucket"),
        "question_type": row.get("question_type"),
        "expected_status": row.get("expected_status"),
        "question": question,
        "oracle_answer": _jsonable(row.get("oracle_answer")),
        "attempt_0": attempts[0],
        "repair_attempts": attempts[1:],
        "attempts": attempts,
        "first_verified_attempt": first_verified_attempt,
        "final_status": "verified" if first_verified_attempt is not None else "failed",
        "final_outcome": final_attempt["outcome"],
        "final_answer": final_attempt["answer"],
        "verified_plan": verified_plan,
        "failed_plans": [attempt["plan"] for attempt in failed_attempts],
        "feedbacks": feedbacks,
        "teacher": teacher,
        "compiler_issues": final_attempt["compiler_issues"],
        "executor_verifier_result": final_attempt["executor_verifier_result"],
        "shape_failed": any(plan_shape_failed(attempt["plan"]) for attempt in attempts),
    }
    write_artifacts(out_dir, row, record)
    return record


def write_artifacts(out_dir: Path, row: dict[str, Any], record: dict[str, Any]) -> None:
    append_jsonl(out_dir / "traces.jsonl", record)
    if record["shape_failed"]:
        append_jsonl(out_dir / "shape_failures.json", record)
    if record["final_status"] != "verified":
        append_jsonl(out_dir / "failures.jsonl", record)
        return

    verified = {
        "question_id": record["question_id"],
        "question": record["question"],
        "source": record.get("source"),
        "level": record.get("level"),
        "question_type": record.get("question_type"),
        "train_bucket": record.get("train_bucket"),
        "verified_plan": record["verified_plan"],
        "final_answer": record["final_answer"],
        "first_verified_attempt": record["first_verified_attempt"],
        "teacher": record["teacher"],
        "label_authority": "executor_verifier_hidden_oracle",
    }
    append_jsonl(out_dir / "verified_sft.jsonl", verified)

    k = record["first_verified_attempt"]
    if isinstance(k, int) and k > 0:
        chosen_attempt = record["attempts"][k]
        rejected_attempt = record["attempts"][k - 1]
        repair = {
            "question_id": record["question_id"],
            "question": record["question"],
            "failed_plan": rejected_attempt["plan"],
            "feedback": chosen_attempt["feedback"],
            "repaired_verified_plan": chosen_attempt["plan"],
            "teacher": record["teacher"],
            "repair_attempt": k,
        }
        append_jsonl(out_dir / "repair_sft.jsonl", repair)
        append_jsonl(out_dir / "dpo_pairs.jsonl", {
            "question_id": record["question_id"],
            "question": record["question"],
            "chosen": chosen_attempt["plan"],
            "rejected": rejected_attempt["plan"],
            "feedback": chosen_attempt["feedback"],
            "teacher": record["teacher"],
            "pair_type": "nearest_failed_vs_first_verified",
            "length_delta": len(json.dumps(chosen_attempt["plan"], ensure_ascii=False))
            - len(json.dumps(rejected_attempt["plan"], ensure_ascii=False)),
        })


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    verified = [r for r in records if r.get("final_status") == "verified"]
    answerable = [r for r in records if r.get("expected_status") not in ABSTAIN_STATUSES]
    abstain = [r for r in records if r.get("expected_status") in ABSTAIN_STATUSES]
    first_pass = [r for r in records if r.get("first_verified_attempt") == 0]
    repair_counts = Counter(r.get("first_verified_attempt") for r in records
                            if isinstance(r.get("first_verified_attempt"), int)
                            and r.get("first_verified_attempt") > 0)

    def group_by(field: str) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[str(record.get(field) or "unknown")].append(record)
        return {key: basic_metrics(rows) for key, rows in sorted(grouped.items())}

    summary = {
        "total": total,
        "verified": len(verified),
        "accuracy": ratio(len(verified), total),
        "answerable_total": len(answerable),
        "answerable_accuracy": ratio(sum(1 for r in answerable if r.get("final_status") == "verified"),
                                     len(answerable)),
        "abstain_total": len(abstain),
        "safe_abstention": sum(1 for r in abstain if r.get("final_outcome") == "safe_abstention"),
        "safe_abstention_rate": ratio(sum(1 for r in abstain if r.get("final_outcome") == "safe_abstention"),
                                      len(abstain)),
        "hallucination": sum(1 for r in abstain if r.get("final_outcome") == "hallucinated_answer"),
        "silent_wrong": sum(1 for r in answerable if r.get("final_outcome") == "wrong_answer"),
        "planner_or_exec_gap": sum(1 for r in answerable if r.get("final_outcome") == "missing_answer"),
        "first_pass_verified": len(first_pass),
        "first_pass_pass_rate": ratio(len(first_pass), total),
        "repair_at": {f"Repair@{k}": repair_counts.get(k, 0) for k in (1, 2, 3)},
        "shape_failures": sum(1 for r in records if r.get("shape_failed")),
        "shape_failure_rate": ratio(sum(1 for r in records if r.get("shape_failed")), total),
        "final_outcome": dict(Counter(str(r.get("final_outcome")) for r in records)),
        "by_subset": group_by("subset"),
        "by_question_type": group_by("question_type"),
        "by_train_bucket": group_by("train_bucket"),
    }
    return summary


def basic_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    verified = sum(1 for r in rows if r.get("final_status") == "verified")
    return {
        "total": len(rows),
        "verified": verified,
        "accuracy": ratio(verified, len(rows)),
        "first_pass": sum(1 for r in rows if r.get("first_verified_attempt") == 0),
        "repair_1": sum(1 for r in rows if r.get("first_verified_attempt") == 1),
        "repair_2": sum(1 for r in rows if r.get("first_verified_attempt") == 2),
        "repair_3": sum(1 for r in rows if r.get("first_verified_attempt") == 3),
        "shape_failures": sum(1 for r in rows if r.get("shape_failed")),
        "outcomes": dict(Counter(str(r.get("final_outcome")) for r in rows)),
    }


def ratio(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def write_summary(out_dir: Path, summary: dict[str, Any], args: argparse.Namespace, teacher: dict[str, Any]) -> None:
    (out_dir / "matrix.json").write_text(json.dumps({
        "teacher": teacher,
        "input": str(args.input),
        "summary": summary,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = [
        "# dev_smoke teacher run",
        "",
        f"- input: `{args.input}`",
        f"- teacher: understanding=`{teacher['understanding']}`, planning=`{teacher['planning']}`, "
        f"repair_understanding=`{teacher['repair_understanding']}`, "
        f"repair_planning=`{teacher['repair_planning']}`",
        f"- total: {summary['total']}",
        f"- verified: {summary['verified']} ({summary['accuracy']:.1%})",
        f"- answerable accuracy: {summary['answerable_accuracy']:.1%}",
        f"- safe abstention: {summary['safe_abstention']}/{summary['abstain_total']} ({summary['safe_abstention_rate']:.1%})",
        f"- hallucination / silent wrong: {summary['hallucination']} / {summary['silent_wrong']}",
        f"- planner_or_exec_gap: {summary['planner_or_exec_gap']}",
        f"- first-pass pass rate: {summary['first_pass_pass_rate']:.1%}",
        f"- Repair@1 / Repair@2 / Repair@3: {summary['repair_at']['Repair@1']} / {summary['repair_at']['Repair@2']} / {summary['repair_at']['Repair@3']}",
        f"- shape failure rate: {summary['shape_failure_rate']:.1%}",
        "",
        "| mix | accuracy | first-pass | Repair@1 | Repair@2 | Repair@3 | shape failure |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| {teacher['mix_name']} | {summary['accuracy']:.1%} | {summary['first_pass_pass_rate']:.1%} | "
        f"{summary['repair_at']['Repair@1']} | {summary['repair_at']['Repair@2']} | "
        f"{summary['repair_at']['Repair@3']} | {summary['shape_failure_rate']:.1%} |",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(rows), encoding="utf-8")


def append_worklog(summary: dict[str, Any], args: argparse.Namespace, teacher: dict[str, Any]) -> None:
    path = ROOT / "docs" / "cicada_worklog.md"
    if not path.exists():
        return
    entry = (
        f"\n## {datetime.now().strftime('%Y-%m-%d')} - dev_smoke teacher runner\n\n"
        f"**Task**: Run or prepare the trace-first teacher loop for `{args.input}`.\n"
        f"**Method**: `scripts/run_dev_smoke_teacher.py` used TypedLLMPlanner with "
        f"understanding=`{teacher['understanding']}`, planning=`{teacher['planning']}`, "
        f"repair_understanding=`{teacher['repair_understanding']}`, "
        f"repair_planning=`{teacher['repair_planning']}`, "
        f"max_feedback_replans={args.max_feedback_replans}. "
        "Each repair receives a structured trace summary without the hidden reference answer, "
        "then the repaired plan is re-grounded, re-executed, and re-verified.\n"
        f"**Result**: total={summary['total']}, verified={summary['verified']} "
        f"({summary['accuracy']:.1%}), first_pass={summary['first_pass_pass_rate']:.1%}, "
        f"Repair@1/2/3={summary['repair_at']['Repair@1']}/"
        f"{summary['repair_at']['Repair@2']}/{summary['repair_at']['Repair@3']}, "
        f"shape_failure={summary['shape_failure_rate']:.1%}.\n"
        "**Next**: Compare nano+nano with nano+Grok and inspect shape failures before scaling.\n"
    )
    text = path.read_text(encoding="utf-8")
    marker = "---\n"
    if marker in text:
        head, tail = text.split(marker, 1)
        path.write_text(head + marker + entry + tail, encoding="utf-8")
    else:
        path.write_text(entry + "\n" + text, encoding="utf-8")


def reset_outputs(out_dir: Path) -> None:
    for name in (*OUTPUT_FILES, "matrix.json", "summary.json", "summary.md"):
        path = out_dir / name
        if path.exists():
            path.unlink()


def run(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        reset_outputs(out_dir)

    rows = read_jsonl(input_path)
    if args.limit:
        rows = rows[:args.limit]

    existing = read_jsonl(out_dir / "traces.jsonl") if args.resume else []
    done = {r.get("question_id") for r in existing}
    rows = [row for row in rows if row.get("id") not in done]

    repair_understanding_model = args.repair_understanding_model or args.understanding_model
    repair_planning_model = args.repair_planning_model or args.repair_model or args.planning_model
    teacher = {
        "understanding": args.understanding_model if args.two_step else "",
        "planning": args.planning_model,
        "repair_understanding": repair_understanding_model if args.repair_two_step else "",
        "repair_planning": repair_planning_model,
        "repair": repair_planning_model,
        "two_step": args.two_step,
        "repair_two_step": args.repair_two_step,
        "mix_name": f"{args.understanding_model if args.two_step else 'single'}+{args.planning_model}+"
                    f"{repair_understanding_model if args.repair_two_step else 'single'}+{repair_planning_model}",
    }

    print("[teacher] loading KG ...", flush=True)
    backend = RuntimeKGBackend.from_directory(ROOT / "data" / "kg")
    resolver = backend.org_resolver()
    planning_client = ChatClient.from_env(temperature=args.temperature)
    understanding_client = ChatClient.from_env(temperature=args.temperature)
    repair_client = ChatClient.from_env(temperature=args.temperature)
    initial_planner = TypedLLMPlanner(
        client=planning_client,
        model=args.planning_model,
        org_resolver=resolver,
        two_step=args.two_step,
        understanding_client=understanding_client,
        understanding_model=args.understanding_model,
    )
    repair_planner = TypedLLMPlanner(
        client=repair_client,
        model=repair_planning_model,
        org_resolver=resolver,
        two_step=args.repair_two_step,
        understanding_client=understanding_client,
        understanding_model=repair_understanding_model,
    )
    base_pipeline = ReasoningPipeline(backend=backend, planner=initial_planner, org_resolver=resolver)

    print(f"[teacher] processing {len(rows)} rows (resume skipped {len(done)}) ...", flush=True)
    new_records: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        record = process_row(
            row,
            base_pipeline=base_pipeline,
            initial_planner=initial_planner,
            repair_planner=repair_planner,
            max_feedback_replans=args.max_feedback_replans,
            teacher=teacher,
            out_dir=out_dir,
        )
        new_records.append(record)
        if args.progress_every and idx % args.progress_every == 0:
            ok = sum(1 for r in new_records if r.get("final_status") == "verified")
            print(f"  processed {idx}/{len(rows)} new, verified={ok}/{idx}", flush=True)

    all_records = existing + new_records
    summary = summarize_records(all_records)
    write_summary(out_dir, summary, args, teacher)
    if args.worklog:
        append_worklog(summary, args, teacher)
    print(f"[teacher] wrote traces and teacher artifacts to {out_dir}", flush=True)
    print(f"[teacher] verified={summary['verified']}/{summary['total']} "
          f"first_pass={summary['first_pass_pass_rate']:.1%} "
          f"shape={summary['shape_failure_rate']:.1%}", flush=True)
    return 0


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Path to dev_smoke.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--understanding-model", default="gpt-5.4-nano")
    ap.add_argument("--planning-model", default="gpt-5.4-nano")
    ap.add_argument("--repair-model", default="", help="Deprecated alias for --repair-planning-model")
    ap.add_argument("--repair-understanding-model", default="", help="Defaults to --understanding-model")
    ap.add_argument("--repair-planning-model", default="", help="Defaults to --repair-model or --planning-model")
    ap.add_argument("--max-feedback-replans", type=int, default=3)
    ap.add_argument("--two-step", type=parse_bool, default=True)
    ap.add_argument("--repair-two-step", type=parse_bool, default=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--progress-every", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--worklog", type=parse_bool, default=True)
    raise SystemExit(run(ap.parse_args()))


if __name__ == "__main__":
    main()
