#!/usr/bin/env python3
"""Probe Step-2 graph planning on CACHED Step-1 understandings.

This consumes the outputs of `probe_understanding_step1.py` (each row carries the question and the
Step-1 briefing) and runs the REAL Step-2 path — `typed_plan_messages` -> planner LLM -> consistency
check -> `compile_typed_plan` -> graph execution + safety stack — with Step-1 injected from cache
instead of regenerated. That isolates Step-2: given a fixed (dense) Step-1, how good is the graph
plan, does it compile, execute, and match the oracle?

Gold answers are joined back from the original split by id (the Step-1 outputs do not carry oracle).
By default the bounded feedback replan is OFF so we measure Step-2 FIRST-PASS quality; --repair on
turns the verifier-guided repair loop back on.

    python -B scripts/probe_plan_step2.py \
        --step1 data/qa/understanding_probe/step1_nano_dense_v3_50/understanding.outputs.jsonl \
        --gold  data/qa/cicada_merged_l1_l2_trainbalanced_v1/dev_smoke.jsonl \
        --out-dir data/qa/plan_probe/step2_dense_v3_50 \
        --planning-model gpt-5.4-nano
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_targeted_v2 import (  # noqa: E402
    ABSTAIN, EXPECTED_ABSTENTION_ACTION, answers_match, _repair_wrong_answer,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _norm(question: str) -> str:
    return " ".join(str(question).split())


def _json_safe(value: Any) -> Any:
    """Recursively coerce to JSON-native (a repaired plan can carry compiled specs / dataclasses)."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    for attr in ("__dict__",):
        data = getattr(value, attr, None)
        if isinstance(data, dict):
            return _json_safe(data)
    return str(value)


class TimedChatClient:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.stats = Counter()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def _timed(self, method: str, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return getattr(self.inner, method)(**kwargs)
        finally:
            elapsed = time.perf_counter() - start
            self.stats["llm_calls"] += 1
            self.stats[f"{method}_calls"] += 1
            self.stats["llm_seconds"] += elapsed

    def complete_text(self, **kwargs: Any) -> Any:
        return self._timed("complete_text", **kwargs)

    def complete_json(self, **kwargs: Any) -> Any:
        return self._timed("complete_json", **kwargs)

    def complete_schema(self, **kwargs: Any) -> Any:
        return self._timed("complete_schema", **kwargs)


class TimedBackend:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.stats = Counter()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def _timed(self, method: str, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return getattr(self.inner, method)(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            self.stats["kg_calls"] += 1
            self.stats[f"{method}_calls"] += 1
            self.stats["kg_seconds"] += elapsed

    def fields(self) -> set[str]:
        return self.inner.fields()

    def record_id(self, record: dict[str, Any]) -> str:
        return self.inner.record_id(record)

    def org_resolver(self) -> Any:
        return self.inner.org_resolver()

    def query(self, *args: Any, **kwargs: Any) -> Any:
        return self._timed("query", *args, **kwargs)

    def count(self, *args: Any, **kwargs: Any) -> Any:
        return self._timed("count", *args, **kwargs)

    def sample(self, *args: Any, **kwargs: Any) -> Any:
        return self._timed("sample", *args, **kwargs)

    def project(self, *args: Any, **kwargs: Any) -> Any:
        return self._timed("project", *args, **kwargs)

    def distinct(self, *args: Any, **kwargs: Any) -> Any:
        return self._timed("distinct", *args, **kwargs)

    def top_k(self, *args: Any, **kwargs: Any) -> Any:
        return self._timed("top_k", *args, **kwargs)


def _counter_delta(after: Counter, before: Counter) -> dict[str, float]:
    keys = set(after) | set(before)
    return {key: after[key] - before[key] for key in keys if after[key] - before[key]}


def _step2_payload(trace) -> dict[str, Any]:
    """Pull the raw Step-2 graph-plan JSON + token usage from the selected plan's raw_response."""
    plans = trace.plans or ()
    plan = next((p for p in plans if p.plan_id == trace.selected_plan_id), plans[0] if plans else None)
    raw = getattr(plan, "raw_response", None) if plan is not None else None
    if not isinstance(raw, dict):
        return {}
    usage = ((raw.get("typed_plan_raw") or {}).get("usage")) or {}
    return {
        "graph_plan": (raw.get("typed_plan") or {}).get("graph_plan"),
        "understanding_network": (raw.get("typed_plan") or {}).get("understanding_network"),
        "teacher": raw.get("teacher"),
        "intent_issues": raw.get("intent_issues"),
        "plan_review": raw.get("plan_review"),
        "pre_execution_review": (trace.metadata.get("workflow") or {}).get("pre_execution_review"),
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
    }


def _score(gold: dict[str, Any], trace) -> dict[str, Any]:
    status = gold.get("expected_status", "answerable")
    oracle, atype = gold.get("oracle_answer"), gold.get("answer_type", "")
    card = trace.answer_card
    pred = card.answer if card else None
    action = trace.reflection.action if trace.reflection else ""
    exec_status = trace.execution.status if trace.execution is not None else "not_run"
    if status in ABSTAIN:
        abstained = pred is None
        return {"scored": True, "abstain_case": True, "expected_status": status,
                "answered": pred is not None, "hallucinated": pred is not None,
                "match": abstained,
                "reason_matched": abstained and action in EXPECTED_ABSTENTION_ACTION.get(status, set()),
                "reason": action, "exec_status": exec_status, "answer": pred}
    matched = answers_match(pred, oracle, atype)
    return {"scored": True, "abstain_case": False, "expected_status": status,
            "answered": pred is not None, "match": matched, "reason": action,
            "exec_status": exec_status, "answer": pred, "oracle": oracle}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step1", type=Path, required=True,
                    help="understanding.outputs.jsonl from probe_understanding_step1")
    ap.add_argument("--gold", type=Path,
                    default=ROOT / "data/qa/cicada_merged_l1_l2_trainbalanced_v1/dev_smoke.jsonl",
                    help="split with oracle answers, joined by id")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--planning-model", default="gpt-5.4-nano")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--repair", choices=["off", "on"], default="off",
                    help="verifier-guided feedback replan on NO-ANSWER failures (off = first-pass only)")
    ap.add_argument("--wrong-answer-repair", choices=["off", "on"], default="off",
                    help="offline teacher mode: also repair answered-but-wrong rows using the hidden "
                         "oracle (the oracle is NOT shown to the reflector); implies --repair on")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--two-step", choices=["on", "off"], default="on",
                    help="off = skip the Step-1 briefing entirely (schema-mode ablation)")
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent questions (LLM-bound; ChatClient is thread-safe)")
    ap.add_argument("--plan-review", choices=["on", "off"], default="off",
                    help="nano semantic review of the compiled plan (soft diagnostic; mismatch triggers repair)")
    ap.add_argument("--prompt-variant", choices=["card", "lean"], default=None,
                    help="Step-2 prompt: 'card' = capability card + template shells; 'lean' = v6-era "
                         "briefing-only. Default: lean for grok models, card otherwise (measured 2026-07-04).")
    ap.add_argument("--max-repairs", type=int, default=1,
                    help="feedback replan budget when --repair on (teacher runs: 2-3 for Repair@k / DPO depth)")
    ap.add_argument("--plan-samples", type=int, default=1,
                    help="structural resample budget: extra Step-2 samples when compile/consistency fails")
    ap.add_argument("--schema-variant", choices=["filler", "optional"], default=None,
                    help="json_schema: 'filler' = all-required official strict subset; 'optional' = v6-era "
                         "omit-unused. Default: optional for grok models, filler otherwise (nano endpoint enforces all-required).")
    args = ap.parse_args()

    from procurement_graph.qa.benchmark.chat import ChatClient
    from procurement_graph.reasoning import ReasoningPipeline
    from procurement_graph.reasoning.kg_backend import RuntimeKGBackend
    from procurement_graph.reasoning.typed_planning import TypedLLMPlanner, understanding_from_text

    step1_rows = read_jsonl(args.step1)
    if args.limit > 0:
        step1_rows = step1_rows[: args.limit]
    gold_by_id = {str(r.get("id")): r for r in read_jsonl(args.gold)}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    traces_path = args.out_dir / "step2.traces.jsonl"
    done: set[str] = set()
    if args.resume and traces_path.exists():
        done = {str(json.loads(l).get("id")) for l in traces_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    elif traces_path.exists():
        traces_path.unlink()

    print("[step2] loading KG ...", flush=True)
    backend = TimedBackend(RuntimeKGBackend.from_directory(ROOT / "data" / "kg"))
    resolver = backend.org_resolver()

    # Inject cached Step-1 keyed by normalised question; the planner skips the Step-1 LLM call.
    cache = {
        _norm(r["question"]): (
            r.get("intent_program")
            if isinstance(r.get("intent_program"), dict)
            else understanding_from_text(r.get("ascii_understanding") or r.get("raw_understanding") or "")
        )
        for r in step1_rows if r.get("question")
    }
    from procurement_graph.reasoning.typed_planning import resolve_planner_variants
    default_prompt, default_schema = resolve_planner_variants(args.planning_model)
    prompt_variant = args.prompt_variant or default_prompt
    schema_variant = args.schema_variant or default_schema
    print(f"[step2] prompt_variant={prompt_variant} schema_variant={schema_variant}", flush=True)
    chat = TimedChatClient(ChatClient.from_env(temperature=args.temperature))
    planner = TypedLLMPlanner(client=chat,
                              model=args.planning_model, org_resolver=resolver,
                              two_step=args.two_step == "on", understanding_cache=cache,
                              plan_review=args.plan_review == "on",
                              plan_prompt_variant=prompt_variant,
                              plan_schema_variant=schema_variant,
                              plan_samples=max(1, args.plan_samples))
    repair_on = args.repair == "on" or args.wrong_answer_repair == "on"
    pipeline = ReasoningPipeline(backend=backend, planner=planner, org_resolver=resolver,
                                 max_feedback_replans=max(1, args.max_repairs) if repair_on else 0)

    def process_row(s1: dict[str, Any]) -> dict[str, Any]:
        row_id = str(s1.get("id"))
        question = s1["question"]
        gold = gold_by_id.get(row_id, {"expected_status": s1.get("expected_status", "answerable"),
                                        "answer_type": s1.get("answer_type", ""), "oracle_answer": None})
        row_start = time.perf_counter()
        try:
            trace = pipeline.run(question)
        except Exception as exc:  # pragma: no cover - live API path
            return {"id": row_id, "bucket": s1.get("train_bucket"), "question": question,
                    "error": repr(exc), "scored": False, "infrastructure_error": True}
        score = _score(gold, trace)
        wrong_answer_repaired = False
        # offline teacher: an answered-but-wrong row can't be repaired at runtime (no oracle),
        # but here we DO have it — send a wrong_answer feedback (oracle hidden from the reflector),
        # re-run, and keep the repair if it now matches. This yields repair-SFT / DPO signal.
        if (args.wrong_answer_repair == "on" and not score.get("abstain_case")
                and score.get("answered") and not score.get("match")):
            question_row = {**gold, "question": question}
            repaired_trace, _rec = _repair_wrong_answer(question_row, trace, pipeline)
            if repaired_trace is not None:
                new_score = _score(gold, repaired_trace)
                if new_score.get("match"):
                    trace, score, wrong_answer_repaired = repaired_trace, new_score, True
        plan = trace.plans[0] if trace.plans else None
        step2 = _step2_payload(trace)
        rationale = plan.rationale if plan else ""
        rec = {
            "id": row_id,
            "bucket": s1.get("train_bucket"),
            "question": question,
            "plan_status": plan.status if plan else "none",
            "plan_rationale": rationale,
            "planner_source": plan.planner_source if plan else "",
            "understanding_source": (step2.get("teacher") or {}).get("understanding_source"),
            "wrong_answer_repaired": wrong_answer_repaired,
            # a dead LLM endpoint is an infrastructure failure, NOT a model decision — without
            # this tag an outage scores as "100% safe abstention"
            "infrastructure_error": str(rationale).startswith(
                ("llm_error", "llm_understanding_error", "llm_feedback_error")),
            **score,
            "graph_plan": step2.get("graph_plan"),
            "plan_review": step2.get("plan_review"),
            "pre_execution_review": step2.get("pre_execution_review"),
            "confidence_label": (trace.answer_card.confidence_label
                                 if trace.answer_card is not None else ""),
            "step2_completion_tokens": step2.get("completion_tokens"),
            "timing": {"total_seconds": round(time.perf_counter() - row_start, 4)},
        }
        return _json_safe(rec)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock

    pending = [s1 for s1 in step1_rows if str(s1.get("id")) not in done]
    total = len(pending)
    records: list[dict[str, Any]] = []
    write_lock = Lock()
    run_start = time.perf_counter()
    workers = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_row, s1): str(s1.get("id")) for s1 in pending}
        for future in as_completed(futures):
            rec = future.result()
            with write_lock:
                records.append(rec)
                with traces_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                done_n = len(records)
            if args.progress_every and (done_n % args.progress_every == 0 or done_n == total):
                matched = sum(1 for r in records if r.get("match"))
                print(f"[step2] {done_n}/{total} matched={matched}/{done_n}", flush=True)

    wall = time.perf_counter() - run_start
    all_recs = [json.loads(l) for l in traces_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    run_stats = {
        "wall_seconds": round(wall, 2),
        "workers": workers,
        "two_step": args.two_step,
        "llm_calls": int(chat.stats.get("llm_calls", 0)),
        "llm_seconds_total": round(chat.stats.get("llm_seconds", 0.0), 2),
        "kg_calls": int(backend.stats.get("kg_calls", 0)),
        "kg_seconds_total": round(backend.stats.get("kg_seconds", 0.0), 2),
    }
    _summarise(args.out_dir, all_recs, run_stats=run_stats)
    print(f"[step2] wall={wall:.1f}s llm_calls={run_stats['llm_calls']} "
          f"llm_time={run_stats['llm_seconds_total']}s", flush=True)
    print(f"[step2] wrote {args.out_dir}", flush=True)
    return 0


def _summarise(out_dir: Path, recs: list[dict[str, Any]], run_stats: dict[str, Any] | None = None) -> None:
    scored = [r for r in recs if r.get("scored")]
    infra = [r for r in recs if r.get("infrastructure_error")]
    # Infrastructure failures are not model decisions. Keep their count, but exclude them from
    # accuracy/planned-rate denominators so an API outage cannot look like safe abstention.
    model_scored = [r for r in scored if not r.get("infrastructure_error")]
    abstain = [r for r in model_scored if r.get("abstain_case")]
    answerable = [r for r in model_scored if not r.get("abstain_case")]
    planned = [r for r in model_scored if r.get("plan_status") == "planned"]
    toks = [r["step2_completion_tokens"] for r in recs if r.get("step2_completion_tokens")]
    timings = [r.get("timing") or {} for r in recs]
    def _timing_summary(key: str) -> dict[str, Any]:
        vals = [float(t[key]) for t in timings if t.get(key) not in (None, "")]
        return {
            "median": round(statistics.median(vals), 4) if vals else None,
            "mean": round(statistics.mean(vals), 4) if vals else None,
            "max": round(max(vals), 4) if vals else None,
        }
    exec_dist = Counter(r.get("exec_status") for r in model_scored)
    fail_reasons = Counter(r.get("plan_rationale", "")[:60] for r in model_scored
                           if r.get("plan_status") != "planned")
    pre_reviews = [r.get("pre_execution_review") or {} for r in model_scored]
    pre_review_triggered = [r for r in pre_reviews if r.get("triggered")]

    by_bucket: dict[str, Counter] = defaultdict(Counter)
    for r in model_scored:
        b = str(r.get("bucket"))
        by_bucket[b]["total"] += 1
        by_bucket[b]["planned"] += int(r.get("plan_status") == "planned")
        by_bucket[b]["match"] += int(bool(r.get("match")))

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "total": len(recs),
        "scored": len(scored),
        "model_scored": len(model_scored),
        "planned_rate": round(len(planned) / max(1, len(model_scored)), 3),
        "answer_accuracy": round(sum(1 for r in model_scored if r.get("match")) / max(1, len(model_scored)), 3),
        "answerable_accuracy": round(sum(1 for r in answerable if r.get("match")) / max(1, len(answerable)), 3),
        "abstain_cases": len(abstain),
        "abstained_safely": sum(1 for r in abstain if r.get("match")),
        "abstained_right_reason": sum(1 for r in abstain if r.get("reason_matched")),
        "hallucinated": sum(1 for r in answerable if False) + sum(1 for r in abstain if r.get("hallucinated")),
        "answered_not_matched": sum(1 for r in answerable if r.get("answered") and not r.get("match")),
        "planner_or_exec_gap": sum(1 for r in answerable if not r.get("answered")),
        "wrong_answer_repaired": sum(1 for r in scored if r.get("wrong_answer_repaired")),
        "pre_execution_review_triggered": len(pre_review_triggered),
        "pre_execution_review_repair_planned": sum(1 for r in pre_review_triggered if r.get("repair_planned")),
        "pre_execution_review_repair_attempted": sum(1 for r in pre_review_triggered if r.get("repair_attempted")),
        "infrastructure_errors": len(infra),
        "run_stats": run_stats or {},
        "exec_status_dist": dict(exec_dist),
        "compile_fail_reasons": dict(fail_reasons.most_common(12)),
        "step2_completion_tokens": {
            "median": int(statistics.median(toks)) if toks else None,
            "mean": int(statistics.mean(toks)) if toks else None,
            "max": max(toks) if toks else None,
        },
        "timing_seconds": {
            "total": _timing_summary("total_seconds"),
            "llm": _timing_summary("llm_seconds"),
            "kg": _timing_summary("kg_seconds"),
        },
        "timing_calls": {
            "llm": int(sum((t.get("llm_calls") or 0) for t in timings)),
            "kg": int(sum((t.get("kg_calls") or 0) for t in timings)),
            "complete_schema": int(sum((t.get("complete_schema_calls") or 0) for t in timings)),
            "complete_text": int(sum((t.get("complete_text_calls") or 0) for t in timings)),
            "query": int(sum((t.get("query_calls") or 0) for t in timings)),
            "count": int(sum((t.get("count_calls") or 0) for t in timings)),
            "sample": int(sum((t.get("sample_calls") or 0) for t in timings)),
            "project": int(sum((t.get("project_calls") or 0) for t in timings)),
            "distinct": int(sum((t.get("distinct_calls") or 0) for t in timings)),
            "top_k": int(sum((t.get("top_k_calls") or 0) for t in timings)),
        },
        "by_bucket": {b: dict(c) for b, c in sorted(by_bucket.items())},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Step-2 Graph-Plan Probe (cached Step-1)", "",
             f"- scored: {summary['scored']}",
             f"- planned rate: {summary['planned_rate']:.0%}",
             f"- answer accuracy: {summary['answer_accuracy']:.0%}",
             f"- answerable accuracy: {summary['answerable_accuracy']:.0%}",
             f"- answered-but-wrong: {summary['answered_not_matched']}",
             f"- planner/exec gap (no answer): {summary['planner_or_exec_gap']}",
             f"- abstain: {summary['abstained_safely']}/{summary['abstain_cases']} safe, "
             f"{summary['abstained_right_reason']} right-reason",
             f"- step2 completion tokens median: {summary['step2_completion_tokens']['median']}",
             f"- timing median: total={summary['timing_seconds']['total']['median']}s, "
             f"llm={summary['timing_seconds']['llm']['median']}s, "
             f"kg={summary['timing_seconds']['kg']['median']}s",
             f"- calls: llm={summary['timing_calls']['llm']}, kg={summary['timing_calls']['kg']} "
             f"(query={summary['timing_calls']['query']}, count={summary['timing_calls']['count']}, "
             f"distinct={summary['timing_calls']['distinct']}, top_k={summary['timing_calls']['top_k']})",
             "", "## By Bucket", "", "| bucket | planned | match | total |", "|---|---|---|---|"]
    for b, c in summary["by_bucket"].items():
        lines.append(f"| {b} | {c['planned']} | {c['match']} | {c['total']} |")
    lines += ["", "## Exec status", ""]
    for k, v in sorted(exec_dist.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {k}: {v}")
    lines += ["", "## Compile/consistency fail reasons", ""]
    for k, v in fail_reasons.most_common(12):
        lines.append(f"- ({v}) {k}")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
