#!/usr/bin/env python3
"""Job B trial: grok-4-20-reasoning billing profile + incremental-rescue probe.

20 questions the fast pilot failed (hits=0 in harvest_grok_pilot.jsonl), k=1,
free decode with tolerant JSON extraction. Captures per-call usage including
reasoning tokens, reports yield and cost extrapolations under both plausible
price tiers (fast-reasoning $0.20/$0.50; full Grok-4 $5.5/$27.5 per M).
"""
from __future__ import annotations

import json
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
N_TRIAL = 20
MODEL = "grok-4-20-reasoning"


def main() -> None:
    env = dict(l.strip().split("=", 1) for l in open(ROOT / ".env") if "=" in l)
    from openai import OpenAI
    client = OpenAI(base_url=env["AZURE_OPENAI_BASE_URL"],
                    api_key=env["AZURE_OPENAI_API_KEY"], timeout=180)

    failed = [json.loads(l) for l in open(ROOT / "data/qa/wtq/harvest_grok_pilot.jsonl")]
    failed = [r for r in failed if r.get("status") == "ok" and int(r.get("hits", 0) or 0) == 0]
    rows = failed[:N_TRIAL]
    print(f"fast-pilot-failed pool {len(failed)}, trial takes {len(rows)}")

    def one(rec):
        out = {"id": rec["id"], "usage": {}}
        try:
            shim, catalog = load_universe(rec["context"])
        except Exception as exc:  # noqa: BLE001
            out.update(status="table_error", detail=str(exc)[:60])
            return out
        hint = render(link(rec["question"], shim.raw_df, catalog))
        cat_txt = catalog_text(catalog) + (("\n\n" + hint) if hint else "")
        prompt = PROMPT.format(catalog=cat_txt, q=rec["question"])
        try:
            resp = client.chat.completions.create(
                model=MODEL, max_tokens=3000,
                messages=[{"role": "user", "content": prompt}])
        except Exception as exc:  # noqa: BLE001
            out.update(status="api_error", detail=str(exc)[:100])
            return out
        u = resp.usage
        det = getattr(u, "completion_tokens_details", None)
        out["usage"] = {"in": u.prompt_tokens, "out": u.completion_tokens,
                        "reasoning": getattr(det, "reasoning_tokens", None) if det else None}
        m = _JSON_RE.search(resp.choices[0].message.content or "")
        tree = None
        if m:
            try:
                tree = json.loads(m.group(0)).get("tree")
            except json.JSONDecodeError:
                pass
        if not isinstance(tree, dict):
            out.update(status="no_tree")
            return out
        try:
            if validate_tree(tree) == "RECORDS":
                out.update(status="bare_records")
                return out
        except AlgebraError as exc:
            out.update(status="invalid", detail=str(exc)[:60])
            return out
        res = WTQEvaluator(shim).run(tree)
        ok = res.get("status") == "ok" and denotation_match(res["answer"], rec["target"].split("|"))
        out.update(status="verified" if ok else "wrong_answer")
        return out

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, rows))

    out_path = ROOT / "data/qa/wtq/pilot_grok_reasoning.jsonl"
    with out_path.open("w") as fh:
        for r in results:
            fh.write(json.dumps(r, default=str) + "\n")

    n = len(results)
    ver = sum(1 for r in results if r["status"] == "verified")
    usages = [r["usage"] for r in results if r.get("usage")]
    tin = sum(x["in"] or 0 for x in usages)
    tout = sum(x["out"] or 0 for x in usages)
    treas = sum(x["reasoning"] or 0 for x in usages if x.get("reasoning") is not None)
    print(f"verified {ver}/{n} on fast-failed questions (incremental rescue rate)")
    print(f"status mix: {dict((s, sum(1 for r in results if r['status'] == s)) for s in set(r['status'] for r in results))}")
    print(f"tokens: in {tin:,} out {tout:,} (reasoning within out: {treas:,})")
    if usages:
        per_in, per_out = tin / len(usages), tout / len(usages)
        for tier, pi, po in [("fast-reasoning $0.20/$0.50", 0.20, 0.50),
                             ("full Grok-4  $5.5/$27.5", 5.5, 27.5)]:
            per_q = per_in / 1e6 * pi + per_out / 1e6 * po
            print(f"  {tier}: ${per_q:.5f}/question -> full 8,049-pool ${per_q*8049:.2f}, 11,332 briefs ${per_q*11332:.2f}")


if __name__ == "__main__":
    main()
