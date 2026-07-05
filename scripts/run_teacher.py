#!/usr/bin/env python3
"""Teacher runner: attempt-protocol data generation over the QA train split.

Per the CICADA training plan (docs/cicada_planner_training_plan.md, user spec 2026-07-03):
  - teacher = live nano Step-1 (dense briefing) + grok Step-2 graph planning (lean prompt /
    optional schema, measured config) + verifier-guided repair loop (attempt protocol, k
    repairs, every repaired plan re-runs ground->execute->verify);
  - ONLY executor/verifier-accepted plans enter SFT as chosen targets — a teacher output is
    NEVER treated as gold by itself; the oracle match is recorded for analysis but acceptance
    is verifier-based (partial verifiability);
  - DPO pairs: chosen = first verified attempt, rejected = nearest preceding failed attempt of
    the SAME question; all-fail questions go to failures.jsonl only.

Outputs under --out-dir:
  traces.jsonl        full per-question attempt records (resume key: id)
  verified_sft.jsonl  Step-2 SFT rows: (question, step1_briefing) -> graph_plan JSON
  repair_sft.jsonl    repair SFT rows: (question, briefing, failure_feedback) -> repaired plan
  dpo_pairs.jsonl     preference pairs (chosen/rejected graph plans)
  failures.jsonl      all-fail questions with the final structured feedback
  summary.json        verified@1 / verified@k / repair gain / per-bucket matrix
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_targeted_v2 import _repair_wrong_answer, answers_match  # noqa: E402

# Numeric answers arrive as int/float OR decimal.Decimal (money sums come off the frames as
# Decimal; json serialisation masks this via default=str, so the trace shows a string while
# the live object is Decimal). The full-harvest run rejected 750 oracle-correct sums as
# "shape_mismatch" before Decimal was accepted here — keep Decimal in every numeric gate.
from decimal import Decimal  # noqa: E402

_NUM = (int, float, Decimal)
_SHAPE_OK = {
    "boolean": lambda a: isinstance(a, bool) or (isinstance(a, dict) and isinstance(a.get("answer"), bool)),
    "comparison": lambda a: isinstance(a, (bool, dict)),
    "count": lambda a: isinstance(a, _NUM) and not isinstance(a, bool),
    "integer": lambda a: isinstance(a, _NUM) and not isinstance(a, bool),
    "sum": lambda a: isinstance(a, (*_NUM, str)) and not isinstance(a, bool),
    "currency": lambda a: isinstance(a, (*_NUM, str)) and not isinstance(a, bool),
    "set_list": lambda a: isinstance(a, list),
    "top_k": lambda a: isinstance(a, list),
}


def shape_ok(answer, answer_type: str) -> bool:
    """Mechanical answer-SHAPE gate (needs no oracle): a boolean question answered with 849 is
    wrong regardless of execution success — the smoke run caught exactly this after a repair
    drifted the operation."""
    fn = _SHAPE_OK.get(str(answer_type))
    return True if fn is None else bool(fn(answer))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _graph_payload_of(plan: Any) -> dict[str, Any] | None:
    # authoritative: the COMPILED plan's normalised raw graph (post T-transforms — the cleaner
    # SFT target, and present regardless of which prompt/schema/repair path produced it)
    compiled = getattr(plan, "graph_plan", None)
    rawg = getattr(compiled, "raw_graph_plan", None)
    if isinstance(rawg, dict) and rawg.get("variables") is not None:
        return rawg
    raw = getattr(plan, "raw_response", None)
    if not isinstance(raw, dict):
        return None
    typed = raw.get("typed_plan")
    if isinstance(typed, dict) and isinstance(typed.get("graph_plan"), dict):
        return typed["graph_plan"]
    # the v6-era "optional" schema returns the graph at TOP level (no graph_plan wrapper)
    if isinstance(typed, dict) and isinstance(typed.get("variables"), (list, dict))             and isinstance(typed.get("return"), dict):
        return typed
    reflector = raw.get("reflector_output")
    if isinstance(reflector, dict) and isinstance(reflector.get("repaired_graph_plan"), dict):
        return reflector["repaired_graph_plan"]
    return None


def _briefing_of(plan: Any) -> Any:
    raw = getattr(plan, "raw_response", None)
    return raw.get("understanding") if isinstance(raw, dict) else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", type=Path, default=Path("data/qa/cicada_core_v4/train.jsonl"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=5,
                    help="spec says smoke with 5 first; 0 = full split")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--step1-model", default="gpt-5.4-nano")
    ap.add_argument("--plan-model", default="grok-4-1-fast-non-reasoning")
    ap.add_argument("--max-repairs", type=int, default=2)
    ap.add_argument("--plan-samples", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--plan-base-url", default=None,
                    help="OpenAI-compatible endpoint for Step-2 only (e.g. local vLLM "
                         "http://HOST:8000/v1). Step-1 stays on the default endpoint. "
                         "This is the RSFT self-harvest switch: point it at the SFT student.")
    ap.add_argument("--plan-api-key", default="local",
                    help="API key for --plan-base-url (vLLM default server ignores it)")
    ap.add_argument("--plan-temperature", type=float, default=0.0,
                    help="Step-2 sampling temperature. Teacher runs stay 0.0 (deterministic); "
                         "RSFT self-harvest needs >0 (e.g. 0.7 with --plan-samples 4) so "
                         "rejection sampling explores beyond what greedy decoding already solves.")
    args = ap.parse_args()

    from procurement_graph.qa.benchmark.chat import ChatClient
    from procurement_graph.reasoning import ReasoningPipeline
    from procurement_graph.reasoning.kg_backend import RuntimeKGBackend
    from procurement_graph.reasoning.typed_planning import (
        TypedLLMPlanner, resolve_planner_variants,
    )

    rows = read_jsonl(args.questions)[args.offset:]
    if args.limit:
        rows = rows[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    traces_path = args.out_dir / "traces.jsonl"
    done: set[str] = set()
    if args.resume and traces_path.exists():
        done = {str(json.loads(l).get("id")) for l in traces_path.read_text(encoding="utf-8").splitlines() if l.strip()}
        rows = [r for r in rows if str(r.get("id")) not in done]
    print(f"[teacher] questions={len(rows)} (resume skipped {len(done)})", flush=True)

    backend = RuntimeKGBackend.from_directory(ROOT / "data" / "kg")
    resolver = backend.org_resolver()
    chat = ChatClient.from_env(temperature=0.0)
    if args.plan_base_url:
        plan_chat = ChatClient(base_url=args.plan_base_url, api_key=args.plan_api_key,
                               temperature=args.plan_temperature)
    elif args.plan_temperature:
        plan_chat = ChatClient.from_env(temperature=args.plan_temperature)
    else:
        plan_chat = chat
    prompt_variant, schema_variant = resolve_planner_variants(args.plan_model)
    planner = TypedLLMPlanner(client=plan_chat, model=args.plan_model, org_resolver=resolver,
                              two_step=True,
                              understanding_client=chat, understanding_model=args.step1_model,
                              plan_prompt_variant=prompt_variant,
                              plan_schema_variant=schema_variant,
                              plan_samples=max(1, args.plan_samples))
    pipeline = ReasoningPipeline(backend=backend, planner=planner, org_resolver=resolver,
                                 max_feedback_replans=max(0, args.max_repairs))

    lock = threading.Lock()
    sinks = {name: (args.out_dir / f"{name}.jsonl").open("a", encoding="utf-8")
             for name in ("traces", "verified_sft", "repair_sft", "dpo_pairs", "failures",
                          "hard_negatives", "abstain_sft")}

    def emit(name: str, obj: dict[str, Any]) -> None:
        with lock:
            sinks[name].write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
            sinks[name].flush()

    stats = Counter()
    bucket_stats: dict[str, Counter] = {}

    def process(row: dict[str, Any]) -> dict[str, Any]:
        qid, question = str(row.get("id")), str(row.get("question"))
        bucket = str(row.get("train_bucket"))
        started = time.perf_counter()
        try:
            trace = pipeline.run(question)
        except Exception as exc:  # noqa: BLE001 - live boundary
            return {"id": qid, "bucket": bucket, "error": repr(exc)[:200]}

        selected = next((p for p in trace.plans if p.plan_id == trace.selected_plan_id),
                        trace.plans[0] if trace.plans else None)
        answer = trace.answer_card.answer if trace.answer_card else None
        verified = bool(trace.execution is not None and trace.execution.passed
                        and answer is not None)
        expected = str(row.get("expected_status", "answerable"))
        oracle_match = (answers_match(answer, row.get("oracle_answer"), str(row.get("answer_type", "")))
                        if expected == "answerable" else answer is None)

        replan_meta = trace.metadata.get("feedback_replan") or {}
        attempts = list(replan_meta.get("attempts") or ())
        first_verified_attempt = replan_meta.get("first_verified_attempt")
        attempt_index = int(first_verified_attempt or 0)  # 0 = initial plan
        briefing = _briefing_of(selected)
        final_graph = _graph_payload_of(selected)

        rec = {"id": qid, "bucket": bucket, "expected_status": expected,
               "verified": verified, "oracle_match": bool(oracle_match),
               "answer": answer, "oracle": row.get("oracle_answer"),
               "attempt_of_success": attempt_index if verified else None,
               "n_repair_attempts": len(attempts),
               "plan_status": getattr(selected, "status", ""),
               "graph_plan": final_graph,
               "briefing": briefing,
               "seconds": round(time.perf_counter() - started, 2)}
        emit("traces", rec)

        answer_shape_ok = shape_ok(answer, str(row.get("answer_type", "")))
        # ---- three-tier routing (teacher runs OFFLINE: oracle may gate/filter, never author) --
        if verified and final_graph is not None and not (oracle_match and answer_shape_ok):
            # verifier-passing but externally wrong: the MOST valuable rejected pool — plans the
            # runtime verifier cannot distinguish from correct ones. One oracle-GATED repair
            # attempt (feedback says only "failed external validation"; oracle content hidden).
            emit("hard_negatives", {
                "id": qid, "bucket": bucket, "question": question,
                "step1_briefing": briefing, "rejected_graph_plan": final_graph,
                "reject_reason": ("shape_mismatch" if not answer_shape_ok else "oracle_mismatch"),
                "answer": answer,
            })
            try:
                repaired_trace, _rec2 = _repair_wrong_answer(
                    {**row, "question": question}, trace, pipeline)
            except Exception:  # noqa: BLE001 - live boundary
                repaired_trace = None
            if repaired_trace is not None:
                r_ans = repaired_trace.answer_card.answer if repaired_trace.answer_card else None
                r_sel = next((p for p in repaired_trace.plans
                              if p.plan_id == repaired_trace.selected_plan_id), None)
                r_graph = _graph_payload_of(r_sel) if r_sel is not None else None
                if (r_graph is not None and r_ans is not None
                        and answers_match(r_ans, row.get("oracle_answer"), str(row.get("answer_type", "")))
                        and shape_ok(r_ans, str(row.get("answer_type", "")))):
                    emit("repair_sft", {
                        "id": qid, "question": question, "step1_briefing": briefing,
                        "failure_feedback": {"failure_stage": "external_validation",
                                             "failure_reason": "submitted answer failed external validation"},
                        "target_graph_plan": r_graph, "kind": "wrong_answer_repair",
                    })
                    emit("dpo_pairs", {
                        "id": qid, "question": question, "step1_briefing": briefing,
                        "chosen_graph_plan": r_graph, "rejected_graph_plan": final_graph,
                        "pair_kind": "oracle_gated_repair",
                    })
                    rec["wrong_answer_repaired"] = True
            return rec
        if expected != "answerable" and answer is None and final_graph is not None:
            # a correct abstention is a SUCCESS: teach the abstain/unanswerable plan shape
            emit("abstain_sft", {
                "id": qid, "bucket": bucket, "question": question,
                "step1_briefing": briefing, "target_graph_plan": final_graph,
                "expected_status": expected,
            })
            return rec
        if verified and final_graph is not None:
            emit("verified_sft", {
                "id": qid, "bucket": bucket, "question": question,
                "step1_briefing": briefing, "target_graph_plan": final_graph,
                "attempt": attempt_index, "oracle_match": bool(oracle_match),
                "teacher": {"step1": args.step1_model, "step2": args.plan_model,
                            "acceptance": "executor_verifier"},
            })
            if attempt_index > 0:
                # a repair succeeded: repair-SFT row + DPO pair vs the nearest failed attempt
                failed_prior = next((a for a in reversed(attempts)
                                     if int(a.get("attempt", 0)) < attempt_index), None)
                feedback = (failed_prior or {}).get("feedback") or (attempts[0].get("feedback")
                                                                    if attempts else None)
                emit("repair_sft", {
                    "id": qid, "question": question, "step1_briefing": briefing,
                    "failure_feedback": feedback, "target_graph_plan": final_graph,
                })
                rejected_graph = None
                for plan in trace.plans:
                    if plan.plan_id != trace.selected_plan_id:
                        g = _graph_payload_of(plan)
                        if g is not None:
                            rejected_graph = g  # last non-selected with a payload = nearest fail
                if rejected_graph is not None:
                    emit("dpo_pairs", {
                        "id": qid, "question": question, "step1_briefing": briefing,
                        "chosen_graph_plan": final_graph,
                        "rejected_graph_plan": rejected_graph,
                        "chosen_attempt": attempt_index,
                    })
        elif not verified and expected == "answerable":
            emit("failures", {"id": qid, "bucket": bucket, "question": question,
                              "plan_status": getattr(selected, "status", ""),
                              "rationale": str(getattr(selected, "rationale", ""))[:200],
                              "n_repair_attempts": len(attempts)})
        return rec

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process, r) for r in rows]
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            results.append(rec)
            if i % 25 == 0:
                print(f"[teacher] {i}/{len(rows)}", flush=True)
    for sink in sinks.values():
        sink.close()

    for rec in results:
        b = rec.get("bucket", "?")
        bucket_stats.setdefault(b, Counter())
        bucket_stats[b]["n"] += 1
        if rec.get("error"):
            stats["errors"] += 1
            continue
        stats["n"] += 1
        if rec.get("verified"):
            stats["verified"] += 1
            bucket_stats[b]["verified"] += 1
            if rec.get("attempt_of_success", 0) == 0:
                stats["verified_at_1"] += 1
            if rec.get("oracle_match"):
                stats["verified_and_oracle"] += 1
    summary = {
        "questions": len(results),
        "verified": stats["verified"],
        "verified_at_1": stats["verified_at_1"],
        "repair_gain": stats["verified"] - stats["verified_at_1"],
        "verified_and_oracle_match": stats["verified_and_oracle"],
        "errors": stats["errors"],
        "per_bucket": {b: dict(c) for b, c in sorted(bucket_stats.items())},
        "teacher": {"step1": args.step1_model, "step2": args.plan_model,
                    "step2_endpoint": args.plan_base_url or "default",
                    "plan_temperature": args.plan_temperature,
                    "prompt_variant": prompt_variant, "schema_variant": schema_variant,
                    "max_repairs": args.max_repairs, "plan_samples": args.plan_samples},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False),
                                               encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
