#!/usr/bin/env python3
"""$1 teacher-harvest pilot (ledger E13c design, user-approved spend).

Questions: 1,000 with ZERO local verified trees under the v3 grammar.
Sampler: grok via Azure, temperature 1.0, k=2 sequential calls, free decode
with tolerant JSON extraction (the API offers no grammar constraint).
A tree enters the pool only if it validates, is not a bare RECORDS root,
executes, and matches the gold answer. Every kept question is incremental
over the local sampler by construction.
Reported: incremental yield, calls, token estimate, cost per new trace.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/wtq"))

from linker import link, render  # noqa: E402
from loader import catalog_text, load_universe  # noqa: E402
from wtq_eval import WTQEvaluator  # noqa: E402
from zero_shot import PROMPT, denotation_match  # noqa: E402

from procurement_graph.compose.algebra import AlgebraError, validate_tree  # noqa: E402

_JSON_RE = re.compile(r"\{.*\}", re.S)


def extract_json(text: str):
    m = _JSON_RE.search(text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def main() -> None:
    zero_hit = []
    for f in sorted(glob.glob(str(ROOT / "data/qa/wtq/harvest_Av3_*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            if r.get("status") == "ok" and r.get("hits", 1) == 0:
                zero_hit.append(r)
    rows = zero_hit[:1000]
    print(f"zero-local-hit pool {len(zero_hit)}, pilot takes {len(rows)}")

    from openai import OpenAI
    client = OpenAI(
        base_url=os.environ.get("AZURE_OPENAI_BASE_URL",
                                "https://uceeh01-5458-resource.services.ai.azure.com/openai/v1"),
        api_key=os.environ["AZURE_OPENAI_API_KEY"], timeout=120)

    stats = {"calls": 0, "in_tok": 0, "out_tok": 0}

    def one(rec):
        out = {"id": rec["id"], "context": rec["context"], "target": rec["target"]}
        try:
            shim, catalog = load_universe(rec["context"])
        except Exception as exc:  # noqa: BLE001
            out.update(status="table_error", detail=str(exc)[:60])
            return out
        hint = render(link(rec["question"], shim.raw_df, catalog))
        cat_txt = catalog_text(catalog) + (("\n\n" + hint) if hint else "")
        prompt = PROMPT.format(catalog=cat_txt, q=rec["question"])
        ev = WTQEvaluator(shim)
        kept, seen = [], set()
        targets = rec["target"].split("|")
        for _ in range(2):
            try:
                resp = client.chat.completions.create(
                    model="grok-4-1-fast-non-reasoning", temperature=1.0,
                    max_tokens=900, messages=[{"role": "user", "content": prompt}])
            except Exception:  # noqa: BLE001
                continue
            stats["calls"] += 1
            u = getattr(resp, "usage", None)
            if u:
                stats["in_tok"] += u.prompt_tokens or 0
                stats["out_tok"] += u.completion_tokens or 0
            payload = extract_json(resp.choices[0].message.content or "")
            tree = (payload or {}).get("tree")
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
        out.update(status="ok", question=rec["question"], hits=len(kept), trees=kept)
        return out

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, rows))

    out_path = ROOT / "data/qa/wtq/harvest_grok_pilot.jsonl"
    with out_path.open("w") as fh:
        for r in results:
            fh.write(json.dumps(r, default=str) + "\n")
    n = len(results)
    hit = sum(1 for r in results if r.get("hits", 0) > 0)
    trees = sum(r.get("hits", 0) for r in results)
    cost = stats["in_tok"] / 1e6 * 0.20 + stats["out_tok"] / 1e6 * 0.50
    print(f"pilot: incremental questions {hit}/{n} = {100*hit/max(1,n):.2f}%; kept trees {trees}")
    print(f"calls {stats['calls']}, tokens in {stats['in_tok']:,} out {stats['out_tok']:,}, est cost ${cost:.2f}")
    if hit:
        print(f"cost per new verified question ${cost/hit:.4f}")

if __name__ == "__main__":
    main()
