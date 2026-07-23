#!/usr/bin/env python3
"""Holdout-20 local B-arm: C-v5 bare vs C-v5 + cached user-contract briefs.

Completes the three-lane holdout matrix (direct 7/20, two-step-fast 6/20, union 9/20).
Runs against a served wtq_C5 adapter (default port 8001); zero API cost.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/wtq"))

from linker import link, render_with_rows  # noqa: E402
from loader import catalog_text, load_universe  # noqa: E402
from tree_repair import repair_variants  # noqa: E402
from wtq_eval import WTQEvaluator  # noqa: E402
from zero_shot import PROMPT, denotation_match  # noqa: E402

from procurement_graph.compose.algebra import AlgebraError, validate_tree  # noqa: E402
from procurement_graph.compose.schema import algebra_json_schema  # noqa: E402

from openai import OpenAI

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8001/v1"
client = OpenAI(base_url=BASE, api_key="local", timeout=300)
for _ in range(60):
    try:
        client.models.list()
        break
    except Exception:  # noqa: BLE001
        time.sleep(10)

briefs = {r["id"]: r.get("brief") for r in json.load(open(ROOT / "data/qa/wtq/pilot_holdout_twostep.json"))}
rows = [json.loads(l) for l in open(ROOT / "data/qa/wtq/harvest_grok_pilot.jsonl")]
failed = [r for r in rows if r.get("status") == "ok" and int(r.get("hits", 0) or 0) == 0]
targets_set = failed[20:40]


def run_arm(rec, with_brief):
    try:
        shim, cat = load_universe(rec["context"])
    except Exception:  # noqa: BLE001
        return "table_error"
    fields = [c[0] for c in cat]
    schema = {"type": "json_schema", "json_schema": {
        "name": "algebra", "schema": algebra_json_schema(fields=fields), "strict": True}}
    links = link(rec["question"], shim.raw_df, cat)
    hint = render_with_rows(links, shim.raw_df)
    cat_txt = catalog_text(cat) + (("\n\n" + hint) if hint else "")
    if with_brief and briefs.get(rec["id"]):
        cat_txt += "\n\nANALYST BRIEF (specialist read the full table; follow it):\n" + briefs[rec["id"]]
    prompt = PROMPT.format(catalog=cat_txt, q=rec["question"])
    ev = WTQEvaluator(shim)
    targets = rec["target"].split("|")

    def ok(t):
        try:
            if validate_tree(t) == "RECORDS":
                return False
        except AlgebraError:
            return False
        res = ev.run(t)
        return res.get("status") == "ok" and denotation_match(res["answer"], targets)

    for _ in range(2):
        try:
            r = client.chat.completions.create(model="wtq_C5", temperature=1.0, max_tokens=900,
                messages=[{"role": "user", "content": prompt}], response_format=schema)
            tree = json.loads(r.choices[0].message.content or "{}").get("tree")
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(tree, dict):
            continue
        if ok(tree):
            return "verified"
        for v in repair_variants(tree, fields, [c for c, _ in links], max_variants=30):
            if ok(v):
                return "verified_repair"
        # keep last
    return "unrescued"


def one(rec):
    return {"id": rec["id"], "bare": run_arm(rec, False), "brief": run_arm(rec, True)}


with ThreadPoolExecutor(max_workers=6) as pool:
    res = list(pool.map(one, targets_set))

vb = sum(1 for r in res if r["bare"].startswith("verified"))
vB = sum(1 for r in res if r["brief"].startswith("verified"))
both = sum(1 for r in res if r["bare"].startswith("verified") and r["brief"].startswith("verified"))
print(f"HOLDOUT LOCAL: C-v5 bare {vb}/20; C-v5+brief {vB}/20 (both {both}, brief-only {vB-both}, bare-only {vb-both})")
for r in res:
    print(f"  {r['id']}: bare={r['bare']} brief={r['brief']}")
json.dump(res, open(ROOT / "data/qa/wtq/pilot_holdout_local.json", "w"), indent=1)
print("HOLDOUT LOCAL DONE")
