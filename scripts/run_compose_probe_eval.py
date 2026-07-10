#!/usr/bin/env python3
"""Experiment 1: zero-shot symmetric tree-emission eval on the compose probe.

Protocol (pre-declared in the worklog before any model call):
- identical prompt for every arm: grammar spec + field catalogue + conventions
  + 2 format-only anchors (old-style single-op trees; no novel composition shown)
- single call per question, temperature 0, NO guided decoding (prompt-only JSON,
  parse + validate; symmetric across local and API arms), no repair loop
- answerable rows: correct iff the emitted tree validates, evaluates, and its
  answer matches the dual-verified oracle; abstention on answerable = wrong
- out-of-grammar controls (P1/P2): correct iff the model abstains
- report: accuracy per template family and distance band + tree-validity rate

Usage:
  .venv/bin/python scripts/run_compose_probe_eval.py --arm student \
      --base-url http://127.0.0.1:8000/v1 --model cicada-qwen3-dpo
  .venv/bin/python scripts/run_compose_probe_eval.py --arm teacher \
      --base-url https://.../openai/v1 --api-key-env AZURE_OPENAI_API_KEY \
      --model grok-4-1-fast-non-reasoning
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from procurement_graph.compose.algebra import AlgebraError, validate_tree  # noqa: E402
from procurement_graph.compose.eval_runtime import RuntimeAlgebraEvaluator  # noqa: E402
from procurement_graph.qa.benchmark.kg_interface import ParquetKGQueryBackend  # noqa: E402

_reg = importlib.util.spec_from_file_location("reg1", ROOT / "scripts/compose_regression.py")
reg1 = importlib.util.module_from_spec(_reg)
_reg.loader.exec_module(reg1)
_match = reg1._match

SYSTEM_PROMPT = """You answer questions about a knowledge graph of UK public procurement contract-award notices by writing a QUERY PLAN in a small typed algebra. You never answer from memory: you only write a plan; a deterministic engine executes it.

RECORD UNIVERSE: one row per contract-award notice. Queryable fields:
- contract_node_id (string id)
- buyer_name (string; canonical buyer organisation, first of sorted names)
- supplier_name (string; first-listed supplier organisation)
- release_year (integer, e.g. 2024)
- tender_category (one of: "goods", "services", "works")
- tender_cpv_id (8-digit CPV code as string, e.g. "45233000"; numeric range queries allowed)
- tender_title (string)
- value_amount (number, GBP)
- value_is_additive (boolean; true when the money value may be summed — ALWAYS filter on this before summing money)
- award_date_signed (ISO date string) and has_award_signed_date (boolean)

ALGEBRA (JSON). Types: RECORDS, VALUES (distinct scalars), GROUPS (key->number), NUMBER, VALUE, BOOL, RANKING.
Nodes:
  {"node":"filter","where":[PRED...]}                                   -> RECORDS (AND of predicates over the whole universe)
  {"node":"values","of":RECORDS,"field":F}                              -> VALUES (distinct non-empty)
  {"node":"count","of":RECORDS}                                         -> NUMBER (deduplicated record count)
  {"node":"size","of":VALUES}                                           -> NUMBER
  {"node":"sum","of":RECORDS,"field":F}                                 -> NUMBER
  {"node":"exists","of":RECORDS}                                        -> BOOL
  {"node":"select","of":RECORDS,"field":F}                              -> VALUE (must be unique)
  {"node":"extreme","of":RECORDS,"op":"argmax"|"argmin","field":F}      -> VALUE (contract_node_id of the extremum record)
  {"node":"groupby","of":RECORDS,"key":F,"metric":"count"|"sum","field":F-if-sum} -> GROUPS (empty keys excluded)
  {"node":"argext","of":GROUPS,"op":"argmax"|"argmin"}                  -> VALUE (the group key)
  {"node":"top","of":GROUPS,"k":int}                                    -> RANKING [[key,value],...]
  {"node":"num","value":number}                                         -> NUMBER (literal)
  {"node":"combine","op":"gt"|"lt"|"ge"|"le"|"eq","left":NUMBER,"right":NUMBER} -> BOOL
  {"node":"combine","op":"diff"|"ratio"|"add","left":NUMBER,"right":NUMBER}     -> NUMBER
  {"node":"vcompare","op":"gt"|"lt"|"ge"|"le"|"eq","of":VALUE,"value":literal,"normalize":"date"?} -> BOOL
  {"node":"setop","op":"union"|"intersect"|"difference","left":VALUES,"right":VALUES} -> VALUES
  {"node":"gcombine","op":"gt"|"diff"|"ratio","left":GROUPS,"right":GROUPS} -> GROUPS (aligned on keys, missing=0; "gt" gives 1.0/0.0)
  {"node":"keys_where","of":GROUPS,"op":"gt"|"ge"|"lt"|"le"|"eq","value":number} -> VALUES (keys passing the test)
Predicates (inside filter.where):
  {"field":F,"op":"eq"|"in"|"contains"|"gte"|"lte","value":V}
  {"field":F,"op":"exists"}
  {"op":"not","pred":PRED}
  {"op":"any","preds":[PRED,PRED,...]}          (logical OR)
  {"field":F,"op":"in_expr","expr":<VALUES subtree>}            (membership in a computed set)
  {"field":F,"op":"in_expr","expr":<VALUES subtree>,"negate":true} (NOT in the computed set)
Nodes compose freely as long as the types match. Max depth 16, max 64 nodes.

CONVENTIONS: counts are automatically deduplicated; grouping drops empty keys; money sums require an explicit {"field":"value_is_additive","op":"eq","value":true} predicate; ties rank by (-value, key).

OUTPUT CONTRACT: reply with ONE JSON object and nothing else.
- To answer: {"tree": {...}}
- If the question cannot be expressed in this algebra (no valid tree computes it): {"abstain": true, "reason": "<short reason>"}

FORMAT EXAMPLES (simple, for output shape only):
Q: How many services contract notices did Transport for London publish in 2023?
{"tree": {"node":"count","of":{"node":"filter","where":[{"field":"buyer_name","op":"eq","value":"Transport for London"},{"field":"tender_category","op":"eq","value":"services"},{"field":"release_year","op":"eq","value":2023}]}}}
Q: Which suppliers received contract notices from Leeds City Council in 2024?
{"tree": {"node":"values","of":{"node":"filter","where":[{"field":"buyer_name","op":"eq","value":"Leeds City Council"},{"field":"release_year","op":"eq","value":2024}]},"field":"supplier_name"}}"""


def _extract_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\s*|\s*```$", "", text, flags=re.S)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    depth, start = 0, -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    start = -1
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key-env", default="")
    ap.add_argument("--probe", default="data/qa/compose_probe_v1/probe.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--resume", action="store_true",
                    help="keep prior non-api_error results; redo only api_error rows")
    ap.add_argument("--guided", action="store_true",
                    help="supplementary protocol: enforce the recursive algebra schema "
                         "via guided decoding (local vLLM arms only; breaks teacher symmetry)")
    args = ap.parse_args()

    from openai import OpenAI
    api_key = os.getenv(args.api_key_env, "") if args.api_key_env else "local"
    client = OpenAI(base_url=args.base_url, api_key=api_key or "local", timeout=180)

    rows = [json.loads(line) for line in (ROOT / args.probe).open()]
    if args.limit:
        rows = rows[: args.limit]

    backend = ParquetKGQueryBackend.from_directory(ROOT / "data/kg", include_evidence=False)
    ev = RuntimeAlgebraEvaluator(backend)

    def ask(row: dict) -> dict:
        out = {"id": row["id"], "family": row["template_family"], "band": row["distance_band"]}
        extra = {}
        if args.guided:
            from procurement_graph.compose.schema import algebra_json_schema
            extra["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "algebra", "schema": algebra_json_schema(), "strict": True}}
        raw = None
        for attempt in range(5):
            try:
                resp = client.chat.completions.create(
                    model=args.model, temperature=0.0, max_tokens=args.max_tokens,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": row["question"]}], **extra)
                raw = resp.choices[0].message.content or ""
                break
            except Exception as exc:
                last_exc = exc
                import time
                time.sleep(min(60, 5 * 2 ** attempt))
        if raw is None:
            out.update(outcome="api_error", detail=str(last_exc)[:200], correct=False)
            return out
        out["raw"] = raw[:4000]
        payload = _extract_json(raw)
        expected_abstain = row["expected_status"] == "unanswerable_out_of_grammar"
        if payload is None:
            out.update(outcome="unparseable", correct=False)
            return out
        if payload.get("abstain"):
            out.update(outcome="abstain", correct=expected_abstain)
            return out
        tree = payload.get("tree") if isinstance(payload.get("tree"), dict) else (
            payload if payload.get("node") else None)
        if tree is None:
            out.update(outcome="no_tree", correct=False)
            return out
        try:
            validate_tree(tree)
        except AlgebraError as exc:
            out.update(outcome="invalid_tree", detail=exc.reason, correct=False, tree=tree)
            return out
        if expected_abstain:
            out.update(outcome="answered_out_of_grammar", correct=False, tree=tree)
            return out
        result = ev.run(tree)
        if result.get("status") != "ok":
            out.update(outcome=f"eval_{result.get('status')}", detail=result.get("reason", ""),
                       correct=False, tree=tree)
            return out
        ok = _match(row["oracle_answer"], result)
        out.update(outcome="answered", correct=bool(ok), tree=tree,
                   answer=result["answer"], oracle=row["oracle_answer"])
        return out

    out_dir = ROOT / "data/qa/compose_probe_v1"
    out_path = out_dir / f"eval_{args.arm}.jsonl"

    kept: dict[str, dict] = {}
    if args.resume and out_path.exists():
        for line in out_path.open():
            r = json.loads(line)
            if r.get("outcome") != "api_error":
                kept[r["id"]] = r
        print(f"resume: keeping {len(kept)} prior results, redoing {len(rows) - len(kept)}")
    todo = [r for r in rows if r["id"] not in kept]

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        fresh = list(pool.map(ask, todo))
    by_id = {**kept, **{r["id"]: r for r in fresh}}
    results = [by_id[r["id"]] for r in rows]
    with out_path.open("w") as fh:
        for r in results:
            fh.write(json.dumps(r, default=str) + "\n")

    fam = defaultdict(lambda: [0, 0])
    band = defaultdict(lambda: [0, 0])
    valid_trees = sum(1 for r in results if r.get("outcome") in ("answered", "answered_out_of_grammar"))
    for r in results:
        fam[r["family"]][0] += int(bool(r["correct"]))
        fam[r["family"]][1] += 1
        band[r["band"]][0] += int(bool(r["correct"]))
        band[r["band"]][1] += 1
    total_c = sum(v[0] for v in fam.values())
    total_n = sum(v[1] for v in fam.values())
    summary = {"arm": args.arm, "model": args.model,
               "accuracy": round(100 * total_c / max(1, total_n), 2),
               "n": total_n, "correct": total_c,
               "tree_valid_rate": round(100 * valid_trees / max(1, total_n), 2),
               "by_band": {k: f"{v[0]}/{v[1]}" for k, v in sorted(band.items())},
               "by_family": {k: f"{v[0]}/{v[1]}" for k, v in sorted(fam.items())},
               "outcomes": dict(sorted(__import__('collections').Counter(
                   r["outcome"] for r in results).items()))}
    (out_dir / f"summary_{args.arm}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
