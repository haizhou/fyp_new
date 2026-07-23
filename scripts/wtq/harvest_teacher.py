#!/usr/bin/env python3
"""Production cloud-teacher harvest over a zero-hit pool (Jobs A / A+).

Cascade position: runs AFTER the local stages (composev3 v4b reharvest, C-v5
self-harvest); reads the residue ids file the local cascade emits, or any
--ids-file. Teacher proposes, oracle filters, never authors — verification
chain identical to the local harvest (validate -> execute -> denotation match).

Crash-safe: appends one JSON line per finished question and skips ids already
present in the output file on restart. Prints running yield and cost.

  Job A  (fast sweep):   --deployment grok-4-1-fast-non-reasoning --k 2
  Job A+ (reasoning):    --deployment grok-4-20-reasoning --k 1 --max-tokens 6000
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import threading
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

_J = re.compile(r"\{.*\}", re.S)

PRICES = {  # $ per 1M tokens (in, out); reasoning thinking tokens bill as out
    "grok-4-1-fast-non-reasoning": (0.20, 0.50),
    "grok-4-20-reasoning": (0.20, 0.50),  # fast-tier assumption; dashboard shows GBP 0 so far
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deployment", required=True)
    ap.add_argument("--ids-file", default="/tmp/claude-1847/teacher_pool.ids")
    ap.add_argument("--out", default=None, help="output jsonl (default derives from deployment)")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--limit", type=int, default=10**9)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    tag = "fast" if "non-reasoning" in args.deployment else "reasoning"
    out_path = Path(args.out) if args.out else ROOT / f"data/qa/wtq/harvest_teacher_{tag}.jsonl"

    keep_ids = set(l.strip() for l in open(args.ids_file) if l.strip())
    done = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                done.add(json.loads(line)["id"])
            except Exception:  # noqa: BLE001
                pass
    rows = []
    with open(WTQ_ROOT / "data/training.tsv") as fh:
        for rec in csv.DictReader(fh, delimiter="\t"):
            if rec["id"] in keep_ids and rec["id"] not in done:
                rows.append(rec)
    rows = rows[: args.limit]
    print(f"pool {len(keep_ids)}, already done {len(done)}, this run {len(rows)}")

    env = dict(l.strip().split("=", 1) for l in open(ROOT / ".env") if "=" in l)
    from openai import OpenAI
    client = OpenAI(base_url=env["AZURE_OPENAI_BASE_URL"],
                    api_key=env["AZURE_OPENAI_API_KEY"], timeout=300)

    lock = threading.Lock()
    stats = {"n": 0, "hit": 0, "trees": 0, "in": 0, "out": 0}
    pi, po = PRICES.get(args.deployment, (0.20, 0.50))

    def one(rec):
        out = {"id": rec["id"], "context": rec["context"], "target": rec["targetValue"],
               "teacher": args.deployment}
        try:
            shim, catalog = load_universe(rec["context"])
        except Exception as exc:  # noqa: BLE001
            out.update(status="table_error", detail=str(exc)[:60])
            return out
        hint = render(link(rec["utterance"], shim.raw_df, catalog))
        cat_txt = catalog_text(catalog) + (("\n\n" + hint) if hint else "")
        prompt = PROMPT.format(catalog=cat_txt, q=rec["utterance"])
        ev = WTQEvaluator(shim)
        targets = rec["targetValue"].split("|")
        kept, seen = [], set()
        for _ in range(args.k):
            try:
                resp = client.chat.completions.create(
                    model=args.deployment, temperature=1.0, max_tokens=args.max_tokens,
                    messages=[{"role": "user", "content": prompt}])
            except Exception:  # noqa: BLE001
                continue
            u = getattr(resp, "usage", None)
            if u:
                with lock:
                    stats["in"] += u.prompt_tokens or 0
                    stats["out"] += u.completion_tokens or 0
                    det = getattr(u, "completion_tokens_details", None)
                    stats["out"] += (getattr(det, "reasoning_tokens", 0) or 0) if det else 0
            m = _J.search(resp.choices[0].message.content or "")
            if not m:
                continue
            try:
                tree = json.loads(m.group(0)).get("tree")
            except json.JSONDecodeError:
                continue
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

    with out_path.open("a") as fh, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for out in pool.map(one, rows):
            with lock:
                fh.write(json.dumps(out, default=str) + "\n")
                fh.flush()
                stats["n"] += 1
                if out.get("hits", 0):
                    stats["hit"] += 1
                    stats["trees"] += out["hits"]
                if stats["n"] % 200 == 0:
                    cost = stats["in"] / 1e6 * pi + stats["out"] / 1e6 * po
                    print(f"[{stats['n']}/{len(rows)}] yield {stats['hit']} "
                          f"({100*stats['hit']/max(1,stats['n']):.1f}%), trees {stats['trees']}, "
                          f"est ${cost:.2f}", flush=True)

    cost = stats["in"] / 1e6 * pi + stats["out"] / 1e6 * po
    print(f"DONE {stats['n']}: incremental {stats['hit']} "
          f"({100*stats['hit']/max(1,stats['n']):.1f}%), trees {stats['trees']}, "
          f"tokens in {stats['in']:,} out(+think) {stats['out']:,}, est ${cost:.2f}")


if __name__ == "__main__":
    main()
