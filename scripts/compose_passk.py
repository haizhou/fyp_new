#!/usr/bin/env python3
"""pass@k exploration probe: can the policy, sampling at temperature, EVER hit a
correct answer on the failing constructions? Gates the RLVR plan: pass@k = 0
means on-policy exploration cannot find the reward and RL has no signal.

Usage:
  .venv/bin/python scripts/compose_passk.py --model cicada-qwen3-compose \
      --probe data/qa/compose_probe_v1/probe_c3c5.jsonl --k 16 --temp 1.0
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from procurement_graph.compose.algebra import AlgebraError, validate_tree  # noqa: E402
from procurement_graph.compose.eval_runtime import RuntimeAlgebraEvaluator  # noqa: E402
from procurement_graph.compose.schema import algebra_json_schema  # noqa: E402
from procurement_graph.qa.benchmark.kg_interface import ParquetKGQueryBackend  # noqa: E402

_reg = importlib.util.spec_from_file_location("reg1", ROOT / "scripts/compose_regression.py")
reg1 = importlib.util.module_from_spec(_reg)
_reg.loader.exec_module(reg1)
_drv = importlib.util.spec_from_file_location("drv", ROOT / "scripts/run_compose_probe_eval.py")
drv = importlib.util.module_from_spec(_drv)
_drv.loader.exec_module(drv)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="local", timeout=300)
    rows = [json.loads(l) for l in (ROOT / args.probe).open()
            if json.loads(l).get("expected_status") == "answerable"]
    backend = ParquetKGQueryBackend.from_directory(ROOT / "data/kg", include_evidence=False)
    ev = RuntimeAlgebraEvaluator(backend)
    schema = {"type": "json_schema", "json_schema": {"name": "algebra",
                                                     "schema": algebra_json_schema(), "strict": True}}

    def one(row):
        hits = 0
        for _ in range(args.k):
            try:
                resp = client.chat.completions.create(
                    model=args.model, temperature=args.temp, max_tokens=1500,
                    messages=[{"role": "system", "content": drv.SYSTEM_PROMPT},
                              {"role": "user", "content": row["question"]}],
                    response_format=schema)
                payload = json.loads(resp.choices[0].message.content or "{}")
            except Exception:
                continue
            tree = payload.get("tree")
            if not isinstance(tree, dict):
                continue
            try:
                validate_tree(tree)
            except AlgebraError:
                continue
            res = ev.run(tree)
            if res.get("status") == "ok" and reg1._match(row["oracle_answer"], res):
                hits += 1
        return {"id": row["id"], "family": row["template_family"], "hits": hits}

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(one, rows))

    fam = defaultdict(lambda: [0, 0, 0])  # passed, total, total_hits
    for r in results:
        fam[r["family"]][0] += int(r["hits"] > 0)
        fam[r["family"]][1] += 1
        fam[r["family"]][2] += r["hits"]
    print(f"pass@{args.k} at temp {args.temp}, model {args.model}:")
    for family, (p, n, h) in sorted(fam.items()):
        print(f"  {family:28s} pass@{args.k}: {p}/{n} ({100*p/n:.0f}%)   mean hits/question: {h/n:.2f}")
    out = ROOT / f"data/qa/compose_probe_v1/passk_{Path(args.probe).stem}_{args.model.split('/')[-1]}.json"
    out.write_text(json.dumps({"k": args.k, "temp": args.temp, "results": results}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
