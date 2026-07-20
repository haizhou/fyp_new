#!/usr/bin/env python3
"""Run frozen probe E (docs/paper_compose/probe_E.md) on frozen checkpoints.

Both arms, guided decoding with the procurement schema, trees executed on the
runtime evaluator over the real KG. Stores tree/abstain + answer envelope per
question; scoring against expected status/answers is a separate (manual) pass.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openai import OpenAI  # noqa: E402

from procurement_graph.compose.algebra import AlgebraError, validate_tree  # noqa: E402
from procurement_graph.compose.eval_runtime import RuntimeAlgebraEvaluator  # noqa: E402
from procurement_graph.compose.schema import algebra_json_schema  # noqa: E402
from procurement_graph.qa.benchmark.kg_interface import ParquetKGQueryBackend  # noqa: E402

PROMPT_PATH = ROOT / "docs/paper_compose/probe_E.md"

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("rcpe", ROOT / "scripts/run_compose_probe_eval.py")
_rcpe = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_rcpe)
SYSTEM = _rcpe.SYSTEM_PROMPT + (
    "\nIf the question cannot be answered from these fields and operators, or is "
    "ambiguous, abstain. The final plan must produce a scalar, list, boolean, "
    "groups, or ranking - never bare records."
)


def load_questions():
    rows = []
    for line in PROMPT_PATH.read_text().splitlines():
        m = re.match(r"\|\s*(\d+)\s*\|(.+?)\|(.+?)\|(.+?)\|(.+?)\|$", line)
        if m:
            rows.append({"idx": int(m.group(1)), "question": m.group(2).strip(),
                         "expected_ops": m.group(3).strip(),
                         "expected_status": m.group(4).strip(),
                         "seen_dist": m.group(5).strip()})
    return rows


def main() -> None:
    qs = load_questions()
    assert len(qs) == 40, f"expected 40 rows, parsed {len(qs)}"
    backend = ParquetKGQueryBackend.from_directory(ROOT / "data/kg", include_evidence=False)
    ev = RuntimeAlgebraEvaluator(backend)
    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local", timeout=300)
    schema = {"type": "json_schema", "json_schema": {
        "name": "algebra", "schema": algebra_json_schema(), "strict": True}}

    out = ROOT / "data/qa/compose_probe_v1/probe_E_results.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for model, arm in (("cicada-qwen3-composev3", "v3"), ("Qwen/Qwen3-8B", "base")):
            for q in qs:
                row = {"arm": arm, **q}
                try:
                    resp = client.chat.completions.create(
                        model=model, temperature=0.0, max_tokens=1400,
                        messages=[{"role": "system", "content": SYSTEM},
                                  {"role": "user", "content": q["question"]}],
                        response_format=schema)
                    payload = json.loads(resp.choices[0].message.content or "{}")
                except json.JSONDecodeError:
                    row.update(outcome="truncated")
                    fh.write(json.dumps(row, default=str) + "\n")
                    continue
                except Exception as exc:  # noqa: BLE001
                    row.update(outcome="api_error", detail=str(exc)[:120])
                    fh.write(json.dumps(row, default=str) + "\n")
                    continue
                if payload.get("abstain"):
                    row.update(outcome="abstain", reason=payload.get("reason", "")[:200])
                elif isinstance(payload.get("tree"), dict):
                    tree = payload["tree"]
                    row["tree"] = tree
                    try:
                        rtype = validate_tree(tree)
                        if rtype == "RECORDS":
                            row.update(outcome="records_root")
                            fh.write(json.dumps(row, default=str) + "\n")
                            continue
                        res = ev.run(tree)
                        row.update(outcome=f"tree_{res.get('status')}",
                                   answer=str(res.get("answer"))[:300],
                                   reason=str(res.get("reason", ""))[:120])
                    except AlgebraError as exc:
                        row.update(outcome="invalid_tree", reason=exc.reason[:120])
                else:
                    row.update(outcome="malformed")
                fh.write(json.dumps(row, default=str) + "\n")
                print(arm, q["idx"], row["outcome"], flush=True)
    print("done ->", out)


if __name__ == "__main__":
    main()
