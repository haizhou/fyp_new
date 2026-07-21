#!/usr/bin/env python3
"""Layer 2-A: denotation-only bootstrap harvest (main arm).

Pool purity: uses ONLY (question, table, gold answer) from WTQ training.tsv,
restricted to train-fold tables (Squall dev-fold tables excluded; pristine
test untouched). No gold SQL is read. Sampler: frozen composev3, temp 1.0,
n=8 guided samples per question; a candidate tree enters the pool iff its
executed denotation matches the gold answer. Identical trees deduped.

Usage: .venv/bin/python scripts/wtq/harvest_a.py --start 0 --limit 500
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/wtq"))

from loader import WTQ_ROOT, catalog_text, load_universe  # noqa: E402
from wtq_eval import WTQEvaluator  # noqa: E402
from zero_shot import PROMPT, denotation_match  # noqa: E402

from procurement_graph.compose.algebra import AlgebraError, validate_tree  # noqa: E402
from procurement_graph.compose.schema import algebra_json_schema  # noqa: E402

DEV_TBLS = set(json.load(open("/var/tmp/cicada/squall/squall-main/data/dev-0.ids")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--dev-fold", action="store_true")
    ap.add_argument("--outtag", default=None, help="output file tag (protects old pools from overwrite)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()

    rows = []
    with open(WTQ_ROOT / "data/training.tsv") as fh:
        for rec in csv.DictReader(fh, delimiter="\t"):
            tbl = rec["context"].removeprefix("csv/").replace("-csv/", "_").removesuffix(".csv")
            if (tbl in DEV_TBLS) == args.dev_fold:
                rows.append(rec)
    total_pool = len(rows)
    rows = rows[args.start: args.start + args.limit]

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="local", timeout=300)

    def one(rec):
        out = {"id": rec["id"], "context": rec["context"], "target": rec["targetValue"]}
        try:
            shim, catalog = load_universe(rec["context"])
        except Exception as exc:  # noqa: BLE001
            out.update(status="table_error", detail=str(exc)[:80])
            return out
        fields = [c[0] for c in catalog]
        schema = {"type": "json_schema", "json_schema": {
            "name": "algebra", "schema": algebra_json_schema(fields=fields), "strict": True}}
        prompt = PROMPT.format(catalog=catalog_text(catalog), q=rec["utterance"])
        try:
            resp = client.chat.completions.create(
                model="cicada-qwen3-composev3", temperature=1.0, n=args.n,
                max_tokens=900, messages=[{"role": "user", "content": prompt}],
                response_format=schema)
        except Exception as exc:  # noqa: BLE001
            out.update(status="api_error", detail=str(exc)[:80])
            return out
        ev = WTQEvaluator(shim)
        kept, seen = [], set()
        targets = rec["targetValue"].split("|")
        for ch in resp.choices:
            try:
                payload = json.loads(ch.message.content or "{}")
            except json.JSONDecodeError:
                continue
            tree = payload.get("tree")
            if not isinstance(tree, dict):
                continue
            key = json.dumps(tree, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            try:
                if validate_tree(tree) == "RECORDS":
                    continue
            except AlgebraError:
                continue
            res = ev.run(tree)
            if res.get("status") == "ok" and denotation_match(res["answer"], targets):
                kept.append(tree)
        out.update(status="ok", question=rec["utterance"], hits=len(kept), trees=kept)
        return out

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(one, rows))

    tag = args.outtag or ("dev" if args.dev_fold else "A")
    out_path = ROOT / f"data/qa/wtq/harvest_{tag}_{args.start}_{args.start+args.limit}.jsonl"
    with out_path.open("w") as fh:
        for r in results:
            fh.write(json.dumps(r, default=str) + "\n")
    n = len(results)
    hit = sum(1 for r in results if r.get("hits", 0) > 0)
    trees = sum(r.get("hits", 0) for r in results)
    print(f"pool tranche [{args.start}:{args.start+args.limit}] of {total_pool} train-fold questions")
    print(f"questions with >=1 verified tree: {hit}/{n} = {100*hit/max(1,n):.2f}%; total kept trees {trees}")


if __name__ == "__main__":
    main()
