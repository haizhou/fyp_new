#!/usr/bin/env python3
"""CICADA demo server: ask a procurement question, get answer + evidence + reasoning chain.

    set -a && . ./.env && set +a && python scripts/serve_demo.py --port 8008
    -> open http://localhost:8008

Wraps the CURRENT (Gen-3) pipeline: nano Step-1 briefing + grok Step-2 graph plan +
deterministic compile/execute/verify + gated repair. The UI shows the full chain: briefing,
compiled plan, per-variable execution, verifier checks, evidence records, answer card.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="CICADA demo")
_STATE: dict[str, Any] = {}


class AskRequest(BaseModel):
    question: str


def _build_pipeline():
    from procurement_graph.qa.benchmark.chat import ChatClient
    from procurement_graph.reasoning import ReasoningPipeline
    from procurement_graph.reasoning.kg_backend import RuntimeKGBackend
    from procurement_graph.reasoning.typed_planning import TypedLLMPlanner, resolve_planner_variants

    print("[demo] loading KG ...", flush=True)
    backend = RuntimeKGBackend.from_directory(ROOT / "data" / "kg")
    resolver = backend.org_resolver()
    chat = ChatClient.from_env(temperature=0.0)
    plan_model = _STATE["plan_model"]
    pv, sv = resolve_planner_variants(plan_model)
    planner = TypedLLMPlanner(client=chat, model=plan_model, org_resolver=resolver, two_step=True,
                              understanding_client=chat, understanding_model=_STATE["step1_model"],
                              plan_prompt_variant=pv, plan_schema_variant=sv, plan_samples=2)
    pipeline = ReasoningPipeline(backend=backend, planner=planner, org_resolver=resolver,
                                 max_feedback_replans=1)
    _STATE["backend"] = backend
    _STATE["pipeline"] = pipeline
    print("[demo] ready", flush=True)


def _evidence_records(backend, evidence_ids, limit=5):
    if not evidence_ids:
        return []
    df = backend._backend.records_df
    sub = df[df["contract_node_id"].isin(list(evidence_ids)[:limit])]
    cols = ["contract_node_id", "buyer_name", "supplier_name", "release_year",
            "tender_category", "tender_cpv_id", "tender_title", "value_amount", "award_date_signed"]
    cols = [c for c in cols if c in sub.columns]
    return json.loads(sub[cols].head(limit).to_json(orient="records"))


def _jsonable(x: Any) -> Any:
    try:
        json.dumps(x)
        return x
    except TypeError:
        return json.loads(json.dumps(x, default=str))


def _mock_response(question: str) -> dict[str, Any]:
    """Canned but realistically-shaped payload so the UI can be previewed/styled without
    loading the KG or calling any LLM (`--mock`)."""
    return {
        "question": question,
        "answer": 42,
        "answer_text": "42 contract notices match these filters.",
        "confidence": "high",
        "limitations": ["Mock mode: canned response for UI preview - start without --mock for the live pipeline."],
        "abstained": False,
        "seconds": 3.21,
        "chain": {
            "step1_briefing": {
                "answer_type": "count", "query_template": "simple_filter_aggregate",
                "explicit_info": "- cpv = 85149000\n- year = 2024",
                "procedure": "Step 1: find contract notices with cpv 85149000\nStep 2: filter to year 2024\nStep 3: count",
                "targets": "none", "roles_direction": "none", "ambiguities": "none",
            },
            "compiled_plan": {
                "question_type": "count", "operation": "count",
                "variables": [{"var_id": "A", "kind": "record_set", "role": "contract_records",
                               "filters": [{"slot": "cpv", "operator": "eq", "value": "85149000"},
                                           {"slot": "year", "operator": "eq", "value": "2024"}],
                               "depends_on": []}],
                "return": {"operation": "count", "input": "A", "field": "none", "k": 0,
                           "group_by": "none", "metric": "none", "comparator": "none",
                           "left": "", "right": ""},
            },
            "execution_variables": [{"var_id": "A", "kind": "record_set", "status": "passed",
                                     "output_size": 42, "depends_on": [], "constraints": None}],
            "verifier_checks": [{"check": "plan_grounded", "passed": True},
                                {"check": "execution_complete", "passed": True},
                                {"check": "answer_shape_consistent", "passed": True},
                                {"check": "evidence_non_empty", "passed": True}],
            "repair": {"attempted": False, "skipped_reason": None, "attempts": 0},
        },
        "evidence": [
            {"tender_title": "Provision of community pharmacy services - North West region",
             "buyer_name": "NHS GREATER MANCHESTER", "supplier_name": "WELLCARE PHARMACY LTD",
             "release_year": 2024, "tender_category": "services", "tender_cpv_id": "85149000",
             "value_amount": 1250000.0},
            {"tender_title": "Pharmacy dispensing framework 2024-2027",
             "buyer_name": "LEEDS CITY COUNCIL", "supplier_name": "BOOTS UK LIMITED",
             "release_year": 2024, "tender_category": "services", "tender_cpv_id": "85149000",
             "value_amount": 890500.5},
            {"tender_title": "Out-of-hours pharmaceutical support",
             "buyer_name": "NHS DEVON", "supplier_name": None,
             "release_year": 2024, "tender_category": "services", "tender_cpv_id": "85149000",
             "value_amount": None},
        ],
        "evidence_count": 42,
    }


@app.post("/ask")
def ask(req: AskRequest):
    if _STATE.get("mock"):
        return _mock_response(req.question)
    pipeline = _STATE["pipeline"]
    backend = _STATE["backend"]
    t0 = time.perf_counter()
    trace = pipeline.run(req.question.strip())
    seconds = round(time.perf_counter() - t0, 2)

    card = trace.answer_card
    selected = next((p for p in trace.plans if p.plan_id == trace.selected_plan_id),
                    trace.plans[0] if trace.plans else None)
    raw = getattr(selected, "raw_response", None) or {}
    briefing = raw.get("understanding")
    attempts = trace.metadata.get("attempts") or []
    graph_exec = (attempts[-1].get("graph_execution") if attempts else {}) or {}
    variables = [{
        "var_id": v.get("var_id"), "kind": v.get("kind"), "status": v.get("status"),
        "output_size": v.get("output_size"), "depends_on": v.get("depends_on"),
        "constraints": v.get("constraints"),
    } for v in graph_exec.get("variables", [])]
    checks = (attempts[-1].get("execution_checks") if attempts else []) or []
    evidence_ids = tuple(trace.execution.evidence.evidence_ids) if (
        trace.execution is not None and trace.execution.evidence is not None) else ()
    replan = trace.metadata.get("feedback_replan") or {}

    return _jsonable({
        "question": req.question,
        "answer": card.answer if card else None,
        "answer_text": card.answer_text if card else "",
        "confidence": card.confidence_label if card else "",
        "limitations": list(card.limitations) if card else [],
        "abstained": card is None or card.answer is None,
        "seconds": seconds,
        "chain": {
            "step1_briefing": briefing,
            "compiled_plan": (attempts[-1].get("graph_plan") if attempts else None),
            "execution_variables": variables,
            "verifier_checks": checks,
            "repair": {"attempted": bool(replan.get("attempted")),
                       "skipped_reason": replan.get("skipped"),
                       "attempts": len(replan.get("attempts") or [])},
        },
        "evidence": _evidence_records(backend, evidence_ids),
        "evidence_count": len(evidence_ids),
    })


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "scripts" / "demo_ui.html").read_text(encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--step1-model", default="gpt-5.4-nano")
    ap.add_argument("--plan-model", default="grok-4-1-fast-non-reasoning")
    ap.add_argument("--mock", action="store_true",
                    help="serve canned responses (no KG load, no LLM calls) for UI preview")
    args = ap.parse_args()
    _STATE["step1_model"] = args.step1_model
    _STATE["plan_model"] = args.plan_model
    _STATE["mock"] = args.mock
    if args.mock:
        print("[demo] MOCK MODE - canned responses, pipeline not loaded", flush=True)
    else:
        _build_pipeline()
    uvicorn.run(app, host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
