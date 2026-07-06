#!/usr/bin/env python3
"""Schema-valid diagnostic (handoff Task 5): does the trained student emit valid plans
WITHOUT guided decoding?

Every ladder rung is served with vLLM guided-json, so format is free at eval time — rung
deltas measure planning skill. This diagnostic separates the other claim: SFT internalised
the output contract itself. Method: for N dev questions, produce the Step-1 briefing
(local step1 adapter, guided ON — it is not under test), then ask the Step-2 model for a
plan with NO response_format. Score raw text: (a) parses as JSON, (b) passes the same
plan-shape gate the runtime uses (compile-level acceptance).

Usage: schema_valid_diagnostic.py --plan-model cicada-qwen3-dpo --n 100 \
          --base-url http://localhost:8011/v1 [--zeroshot-model cicada-qwen3-zeroshot]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from procurement_graph.qa.benchmark.chat import ChatClient  # noqa: E402
from procurement_graph.reasoning.typed_planning import (  # noqa: E402
    _is_intent_program, question_intent_program_messages, typed_plan_messages,
)


def _extract_json(raw: str):
    """Fair extraction for un-guided output: direct parse, then ```json fences, then the FIRST
    balanced {...} object (brace counter, string-aware). The v1 greedy first-{ to last-} slice
    produced invalid concatenations when the model emitted several objects — over-counting
    failures on exactly the arm under test."""
    for cand in (raw, raw.strip()):
        try:
            o = json.loads(cand)
            return o if isinstance(o, dict) else None
        except Exception:
            pass
    import re
    for m in re.finditer(r"```(?:json)?\s*(.*?)```", raw, re.S):
        try:
            o = json.loads(m.group(1).strip())
            if isinstance(o, dict):
                return o
        except Exception:
            continue
    start = raw.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for j in range(start, len(raw)):
            ch = raw[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        o = json.loads(raw[start: j + 1])
                        if isinstance(o, dict):
                            return o
                    except Exception:
                        break
                    break
        start = raw.find("{", start + 1)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8011/v1")
    ap.add_argument("--step1-model", default="cicada-qwen3-step1")
    ap.add_argument("--plan-models", nargs="+",
                    default=["cicada-qwen3-dpo", "cicada-qwen3-zeroshot"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--questions", type=Path, default=ROOT / "data/qa/eval/compare_set_v4.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/eval/schema_valid_diagnostic.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.questions.read_text(encoding="utf-8").splitlines() if l.strip()]
    # stratified slice: every 260/n-th question keeps bucket coverage
    step = max(1, len(rows) // args.n)
    rows = rows[::step][: args.n]
    chat = ChatClient(base_url=args.base_url, api_key="local", temperature=0.0)

    report: dict[str, dict] = {}
    for model in args.plan_models:
        parse_ok = shape_ok = 0
        fails: list[str] = []
        for i, r in enumerate(rows):
            q = str(r["question"])
            u_sys, u_user = question_intent_program_messages(q)
            try:
                # Step-1 briefing guided (not under test)
                u = chat.complete_json(model=args.step1_model, system=u_sys, user=u_user)
                briefing = u.parsed if isinstance(u.parsed, dict) else {}
            except Exception:
                briefing = {}
            p_sys, p_user = typed_plan_messages(q, briefing, variant="lean")
            try:
                # THE TEST: no response_format / no guided decoding
                res = chat.complete_text(model=model, system=p_sys, user=p_user)
                raw = res.raw_text if hasattr(res, "raw_text") else str(res)
            except Exception as exc:
                fails.append({"id": r["id"], "kind": "call_error", "raw": repr(exc)[:200]})
                continue
            obj = _extract_json(raw)
            if obj is None:
                fails.append({"id": r["id"], "kind": "unparseable", "raw": raw})
                continue
            parse_ok += 1
            g = obj.get("graph_plan") if isinstance(obj.get("graph_plan"), dict) else obj
            if isinstance(g, dict) and (
                (isinstance(g.get("variables"), list) and isinstance(g.get("return"), dict))
                or str(g.get("question_type", "")) in {"unanswerable", "ambiguous"}
                or _is_intent_program(g)
            ):
                shape_ok += 1
            else:
                fails.append({"id": r["id"], "kind": "shape_miss", "raw": json.dumps(obj)[:400]})
            if (i + 1) % 25 == 0:
                print(f"[{model}] {i+1}/{len(rows)}", flush=True)
        report[model] = {"n": len(rows), "json_parse_ok": parse_ok, "plan_shape_ok": shape_ok,
                         "parse_rate": parse_ok / len(rows), "shape_rate": shape_ok / len(rows),
                         "failures": fails}
        print(f"[{model}] parse {parse_ok}/{len(rows)}  shape {shape_ok}/{len(rows)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
