#!/usr/bin/env python3
"""WTQ zero-shot arm: per-table dynamic schema, single call, no repair.

Scoring here is an INTERNAL denotation match (normalized string/number/list);
paper-grade numbers rerun through the official evaluator. Answers and trees are
stored per item for that rerun.

Usage:
  .venv/bin/python scripts/wtq/zero_shot.py --model cicada-qwen3-composev3 \
      --split random-split-1-dev --limit 300
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/wtq"))

from loader import WTQ_ROOT, catalog_text, load_universe  # noqa: E402

from procurement_graph.compose.algebra import AlgebraError, validate_tree  # noqa: E402
from wtq_eval import WTQEvaluator  # noqa: E402
from procurement_graph.compose.schema import algebra_json_schema  # noqa: E402

PROMPT = """You answer a question about ONE table by writing a QUERY PLAN in a small typed algebra. A deterministic engine executes the plan on the full table; you never answer from memory.

{catalog}

ALGEBRA (JSON). Types: RECORDS, VALUES, GROUPS, NUMBER, VALUE, BOOL, RANKING.
Nodes: {{"node":"filter","where":[PRED...]}} -> RECORDS; {{"node":"values","of":R,"field":F}} -> VALUES; {{"node":"count","of":R}}; {{"node":"size","of":V}}; {{"node":"sum","of":R,"field":F}}; {{"node":"exists","of":R}}; {{"node":"select","of":R,"field":F}} (unique value); {{"node":"extreme_rows","of":R,"op":"argmax"|"argmin","field":F}} -> RECORDS (the row(s) holding the extremum of F; then select/values another field from them); {{"node":"groupby","of":R,"key":F,"metric":"count"|"sum"[,"field":F]}} -> GROUPS; {{"node":"argext","of":G,"op":"argmax"|"argmin"}} -> key; {{"node":"top","of":G,"k":int}}; {{"node":"num","value":n}}; {{"node":"combine","op":"gt"|"lt"|"ge"|"le"|"eq"|"diff"|"ratio"|"add","left":N,"right":N}}; {{"node":"vcompare","op":...,"of":VALUE,"value":lit}}; {{"node":"setop","op":"union"|"intersect"|"difference","left":V,"right":V}}; {{"node":"gcombine","op":"gt"|"diff"|"ratio","left":G,"right":G}}; {{"node":"keys_where","of":G,"op":...,"value":n}} -> VALUES.
Preds: {{"field":F,"op":"eq"|"in"|"contains"|"gte"|"lte","value":v}}; {{"field":F,"op":"exists"}}; {{"op":"not","pred":P}}; {{"op":"any","preds":[...]}}; {{"field":F,"op":"in_expr","expr":<VALUES>[,"negate":true]}}.
To answer "which/what X ..." questions, prefer select/values/argext returning the VALUE from the right column (not a row id) unless the question asks for a row. Copy cell values EXACTLY as they appear in the samples. Reply with ONE JSON object: {{"tree": {{...}}}} or {{"abstain": true, "reason": "..."}}.

FORMAT EXAMPLE (row lookup is ONE step — filter the row, select the wanted column):
Q: what year did the team named Foo win?
{{"tree": {{"node":"select","of":{{"node":"filter","where":[{{"field":"team","op":"eq","value":"Foo"}}]}},"field":"year"}}}}

Question: {q}"""


def norm(x):
    s = str(x).strip().strip('"').casefold()
    s = re.sub(r"\s+", " ", s)
    try:
        f = float(s.replace(",", ""))
        return f"{f:g}"
    except ValueError:
        return s


def denotation_match(pred, targets: list) -> bool:
    if isinstance(pred, bool):  # deterministic rendering: BOOL -> yes/no
        pred = "yes" if pred else "no"
    if isinstance(pred, list) and pred and isinstance(pred[0], list):
        pred = [p[0] for p in pred]  # ranking -> keys
    preds = pred if isinstance(pred, list) else [pred]
    p = sorted(norm(x) for x in preds)
    t = sorted(norm(x) for x in targets)
    return p == t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", default=None)
    ap.add_argument("--split", default="random-split-1-dev")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--hints", choices=["none", "columns", "cells", "random"], default="none",
                    help="value-linker ablation: none | columns (ranked cols only) | "
                         "cells (column-aware candidates) | random (control: random cells)")
    ap.add_argument("--reflect", type=int, default=0,
                    help="max typed-feedback repair rounds on HARD failures only "
                         "(invalid_tree/eval_failed/truncated); empty results and "
                         "abstentions are never reflected")
    args = ap.parse_args()
    arm = args.arm or f"wtq_{args.model.split('/')[-1]}_{args.split}"

    rows = []
    with open(WTQ_ROOT / "data" / f"{args.split}.tsv") as fh:
        for rec in csv.DictReader(fh, delimiter="\t"):
            rows.append(rec)
    rows = rows[: args.limit]

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="local", timeout=180)

    def one(rec):
        out = {"id": rec["id"], "context": rec["context"], "target": rec["targetValue"]}
        try:
            shim, catalog = load_universe(rec["context"])
        except Exception as exc:
            out.update(outcome="table_load_error", detail=str(exc)[:80], correct=False)
            return out
        fields = [c[0] for c in catalog]  # synthetic row id stays internal
        schema = {"type": "json_schema", "json_schema": {
            "name": "algebra", "schema": algebra_json_schema(fields=fields), "strict": True}}
        cat_txt = catalog_text(catalog)
        if args.hints != "none":
            import random as _rnd
            from linker import link, render, TOP_COLS, TOP_CELLS
            links = link(rec["utterance"], shim.raw_df, catalog)
            if args.hints == "columns":
                links = [(c, []) for c, _ in links]
            elif args.hints == "random":
                rng = _rnd.Random(hash(rec["id"]) & 0xffff)
                cols = [c[0] for c in catalog]
                rng.shuffle(cols)
                links = []
                for c in cols[:TOP_COLS]:
                    vals = [v for v in shim.raw_df[c].dropna().astype(str) if v.strip()]
                    links.append((c, rng.sample(vals, min(TOP_CELLS, len(vals)))))
            hint = render(links)
            if hint:
                cat_txt = cat_txt + "\n\n" + hint
        prompt = PROMPT.format(catalog=cat_txt, q=rec["utterance"])
        ev = WTQEvaluator(shim)
        messages = [{"role": "user", "content": prompt}]
        rounds = 0
        while True:
            raw = None
            try:
                resp = client.chat.completions.create(
                    model=args.model, temperature=0.0, max_tokens=900,
                    messages=messages, response_format=schema)
                raw = resp.choices[0].message.content or "{}"
                payload = json.loads(raw)
            except json.JSONDecodeError:
                out.update(outcome="truncated", correct=False)
                payload = None
                feedback = ("Your previous plan was cut off before the JSON finished. "
                            "Write a SHORTER, simpler plan for the same question.")
            except Exception as exc:
                out.update(outcome="api_error", detail=str(exc)[:100], correct=False)
                return out
            if payload is not None:
                tree = payload.get("tree")
                if not isinstance(tree, dict):
                    # abstention is a legitimate final state: never reflected
                    out.update(outcome="abstain", correct=False, rounds=rounds)
                    return out
                try:
                    validate_tree(tree)
                except AlgebraError as exc:
                    out.update(outcome="invalid_tree", detail=exc.reason[:60],
                               correct=False, tree=tree)
                    feedback = (f"Your plan was REJECTED by the type checker: "
                                f"{exc.reason} at {exc.path}. Fix the plan.")
                else:
                    res = ev.run(tree)
                    if res.get("status") != "ok":
                        out.update(outcome="eval_failed",
                                   detail=str(res.get("reason"))[:60],
                                   correct=False, tree=tree)
                        feedback = (f"Your plan failed at execution: "
                                    f"{res.get('reason')}. Fix the plan.")
                    else:
                        # success — including legitimately empty answers: final
                        targets = rec["targetValue"].split("|")
                        ok = denotation_match(res["answer"], targets)
                        out.update(outcome="answered", correct=bool(ok), tree=tree,
                                   answer=str(res["answer"])[:200], rounds=rounds)
                        return out
            if rounds >= args.reflect:
                out["rounds"] = rounds
                return out
            rounds += 1
            messages = messages + [
                {"role": "assistant", "content": raw or "{}"},
                {"role": "user", "content": feedback},
            ]

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(one, rows))

    out_dir = ROOT / "data/qa/wtq"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"eval_{arm}.jsonl").open("w") as fh:
        for r in results:
            fh.write(json.dumps(r, default=str) + "\n")
    from collections import Counter
    n = len(results)
    c = sum(r["correct"] for r in results)
    print(f"{arm}: {c}/{n} = {100*c/n:.2f}% (internal denotation match)")
    print("outcomes:", dict(Counter(r["outcome"] for r in results)))


if __name__ == "__main__":
    main()
