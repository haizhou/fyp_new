#!/usr/bin/env python3
"""Deterministic reflector reranking — the authentic port of the main pipeline's
reflector philosophy (diagnosis is deterministic first; reflector.py, answer_sanity.py).

For each candidate that PASSES the admission gate (schema/validate/execute), score
mechanical fault features of (tree, executed answer, question):
  F1 zero-row filter: any filter subtree matches 0 rows            (+3)
  F2 empty/None answer                                             (+3)
  F3 answer-kind mismatch vs question word (how many -> NUMBER...) (+2)
  F4 answer echoes a literal already present in the question       (+2)
  F5 linked anchor column unused: linker grounded a question token
     to a column, but no pred references that column       (+1 each, cap 2)
  F6 count/sum question answering 0                                (+1)
Rank: lowest penalty; ties -> fewer nodes; ties -> earlier sample.
Arms on dev-300: k1 / first / reflector / oracle.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/wtq"))

from linker import link, render  # noqa: E402
from loader import WTQ_ROOT, catalog_text, load_universe  # noqa: E402
from wtq_eval import WTQEvaluator  # noqa: E402
from zero_shot import PROMPT, denotation_match  # noqa: E402

from procurement_graph.compose.algebra import AlgebraError, validate_tree  # noqa: E402
from procurement_graph.compose.schema import algebra_json_schema  # noqa: E402

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8003/v1", api_key="local", timeout=300)

NUM_Q = re.compile(r"^\s*(how many|how much|what is the (total|number|amount|sum|difference|average))", re.I)
BOOL_Q = re.compile(r"^\s*(is|are|was|were|did|does|do|has|have)\b", re.I)


def nodes_count(tree):
    if isinstance(tree, dict):
        return 1 + sum(nodes_count(v) for v in tree.values())
    if isinstance(tree, list):
        return sum(nodes_count(v) for v in tree)
    return 0


def iter_filters(tree):
    if isinstance(tree, dict):
        if tree.get("node") == "filter":
            yield tree
        for v in tree.values():
            yield from iter_filters(v)
    elif isinstance(tree, list):
        for v in tree:
            yield from iter_filters(v)


def pred_columns(tree):
    cols = set()
    def walk(n):
        if isinstance(n, dict):
            f = n.get("field")
            if isinstance(f, str):
                cols.add(f.split("__")[0])
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)
    walk(tree)
    return cols


def reflector_penalty(tree, ans, ttype, question, ev, links):
    p = 0
    for f in iter_filters(tree):
        try:
            r = ev.run({"node": "count", "of": f})
            if r.get("status") == "ok" and (r.get("answer") in (0, 0.0)):
                p += 3
                break
        except Exception:  # noqa: BLE001
            pass
    if ans is None or ans == [] or (isinstance(ans, str) and not ans.strip()):
        p += 3
    isnum = NUM_Q.search(question) is not None
    isbool = BOOL_Q.search(question) is not None
    if isnum and ttype not in ("NUMBER",):
        p += 2
    if isbool and ttype != "BOOL":
        p += 2
    if not isnum and not isbool and ttype == "NUMBER" and not re.search(r"\b(year|rank|score|number)\b", question, re.I):
        p += 1
    ql = question.casefold()
    vals = ans if isinstance(ans, list) else [ans]
    for v in vals[:3]:
        s = str(v).strip().casefold()
        if s and len(s) > 3 and s in ql:
            p += 2
            break
    used = pred_columns(tree)
    miss = 0
    for col, cells in links:
        if cells and col.split("__")[0] not in used:
            miss += 1
    p += min(miss, 2)
    if isnum and vals and str(vals[0]) in ("0", "0.0"):
        p += 1
    return p


def main() -> None:
    rows = []
    with open(WTQ_ROOT / "data/clean-eval-devfold.tsv") as fh:
        for rec in csv.DictReader(fh, delimiter="\t"):
            rows.append(rec)
    rows = rows[:300]

    def one(rec):
        try:
            shim, cat = load_universe(rec["context"])
        except Exception:  # noqa: BLE001
            return {"id": rec["id"], "k1": False, "first": False, "reflector": False, "oracle": False}
        fields = [c[0] for c in cat]
        schema = {"type": "json_schema", "json_schema": {
            "name": "algebra", "schema": algebra_json_schema(fields=fields), "strict": True}}
        links = link(rec["utterance"], shim.raw_df, cat)
        hint = render(links)
        cat_txt = catalog_text(cat) + (("\n\n" + hint) if hint else "")
        prompt = PROMPT.format(catalog=cat_txt, q=rec["utterance"])
        ev = WTQEvaluator(shim)
        targets = rec["targetValue"].split("|")

        def run_tree(tree):
            res = ev.run(tree)
            ok = res.get("status") == "ok"
            return ok, (res.get("answer") if ok else None), (denotation_match(res["answer"], targets) if ok else False)

        try:
            r1 = client.chat.completions.create(model="wtq_C6", temperature=0.0, max_tokens=900,
                messages=[{"role": "user", "content": prompt}], response_format=schema)
            t1 = json.loads(r1.choices[0].message.content or "{}").get("tree")
            k1 = isinstance(t1, dict) and run_tree(t1)[2]
        except Exception:  # noqa: BLE001
            k1 = False

        cands, seen = [], set()
        try:
            r4 = client.chat.completions.create(model="wtq_C6", temperature=1.0, n=4, max_tokens=900,
                messages=[{"role": "user", "content": prompt}], response_format=schema)
            choices = r4.choices
        except Exception:  # noqa: BLE001
            choices = []
        for i, ch in enumerate(choices):
            try:
                t = json.loads(ch.message.content or "{}").get("tree")
            except json.JSONDecodeError:
                continue
            if not isinstance(t, dict):
                continue
            key = json.dumps(t, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            try:
                ttype = validate_tree(t)
                if ttype == "RECORDS":
                    continue
            except AlgebraError:
                continue
            ok, ans, correct = run_tree(t)
            if not ok:
                continue
            cands.append({"tree": t, "ans": ans, "type": ttype, "correct": correct, "pos": i})
        if not cands:
            return {"id": rec["id"], "k1": k1, "first": False, "reflector": False, "oracle": False}
        first = cands[0]["correct"]
        oracle = any(c["correct"] for c in cands)
        for c in cands:
            c["pen"] = reflector_penalty(c["tree"], c["ans"], c["type"], rec["utterance"], ev, links)
        best = min(cands, key=lambda c: (c["pen"], nodes_count(c["tree"]), c["pos"]))
        return {"id": rec["id"], "k1": k1, "first": first, "reflector": best["correct"], "oracle": oracle}

    with ThreadPoolExecutor(max_workers=6) as pool:
        res = list(pool.map(one, rows))
    n = len(res)
    for arm in ("k1", "first", "reflector", "oracle"):
        v = sum(1 for r in res if r[arm])
        print(f"{arm:9}: {v}/{n} = {100*v/n:.2f}%")
    json.dump(res, open(ROOT / "data/qa/wtq/reflector_rerank.json", "w"), indent=1)


if __name__ == "__main__":
    main()
