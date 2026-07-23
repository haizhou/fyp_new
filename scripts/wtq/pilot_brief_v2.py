#!/usr/bin/env python3
"""Job B v2: faithful port of the main-experiment two-stage protocol to WTQ.

Step-1 (grok-4-20-reasoning): 8-section scaffold, QUESTION SPACE ONLY, closed
template vocabulary. Step-2 (grok-4-1-fast): structured JSON payload carrying
the briefing + template shell + catalog, k=2 temp 1.0, full verification.
Same 20 fast-failed questions as E13h for a paired comparison.
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
from zero_shot import denotation_match  # noqa: E402

from procurement_graph.compose.algebra import AlgebraError, validate_tree  # noqa: E402

_J = re.compile(r"\{.*\}", re.S)

TEMPLATES = {
    "row_lookup": '{"tree":{"node":"select","of":{"node":"filter","where":[...]},"field":F}}',
    "filter_aggregate": '{"tree":{"node":"count|sum|exists","of":{"node":"filter","where":[...]}}}',
    "superlative_row": '{"tree":{"node":"select|values","of":{"node":"extreme_rows","of":{"node":"filter","where":[...]} or omit filter,"op":"argmax|argmin","field":F},"field":F2}}',
    "grouped_extreme": '{"tree":{"node":"argext","of":{"node":"groupby","of":...,"key":F,"metric":"count|sum"},"op":"argmax|argmin"}} or top k',
    "comparison": '{"tree":{"node":"combine","op":"gt|lt|eq|diff|ratio","left":<NUMBER expr>,"right":<NUMBER expr>}} or vcompare on a VALUE',
    "set_operation": '{"tree":{"node":"size|...","of":{"node":"setop","op":"union|intersect|difference","left":<VALUES>,"right":<VALUES>}}}',
    "threshold_keys": '{"tree":{"node":"keys_where","of":{"node":"groupby",...},"op":"gte|gt|...","value":n}}',
    "ordered_navigation": 'CAUTION: next/previous/above/below row navigation is NOT expressible; if truly required, emit {"abstain": true}. Often the question can be recast as superlative_row over a filtered set instead.',
}

BRIEF_SYSTEM = ("You are the semantic reading layer for a table QA system. Output a DENSE scaffold "
    "for a downstream planner, not an essay. Every line must state a fact the planner cannot "
    "trivially re-derive. QUESTION SPACE ONLY: quote literals from the question verbatim; do NOT "
    "guess table column names. Do not answer the question. Do not write any program.")

BRIEF_USER = """Question:
{q}

Task: describe the solving procedure as a compact scaffold. Do NOT calculate or guess the answer.
Output exactly these 8 sections, OBEY line budgets, terse fragments:

1. Answer Type: one word (value | list | count | sum | boolean | ranking).
2. Query Template: one word (row_lookup | filter_aggregate | superlative_row | grouped_extreme | comparison | set_operation | threshold_keys | ordered_navigation).
3. Explicit Atoms: one bullet per literal condition as "concept = value", values VERBATIM from the question. Max 6. Do not echo the question.
4. Reverse Tree: dependency chain, max 3 lines (answer <- B <- A). If one filtered set, write "answer <- filtered set".
5. Procedure: max 5 lines, verb-first (filter, group, rank, compare, select).
6. Targets: intermediate sets needed for comparison/set ops, one line each, or exactly "none".
7. Order Semantics: one line, what first/last/most-recent/next means and over what quantity, or "none".
8. Traps: "none" by default; add a line ONLY for a real hazard stated in or implied by the question (ties, multiple entities with same name, totals rows).

Rules: ZERO hallucination; preserve every literal exactly; plain ASCII; no JSON; no column names."""

def main() -> None:
    env = dict(l.strip().split("=", 1) for l in open(ROOT / ".env") if "=" in l)
    from openai import OpenAI
    client = OpenAI(base_url=env["AZURE_OPENAI_BASE_URL"], api_key=env["AZURE_OPENAI_API_KEY"], timeout=300)

    rows = [json.loads(l) for l in open(ROOT / "data/qa/wtq/harvest_grok_pilot.jsonl")]
    failed = [r for r in rows if r.get("status") == "ok" and int(r.get("hits", 0) or 0) == 0][:20]

    def one(rec):
        out = {"id": rec["id"]}
        try:
            shim, cat = load_universe(rec["context"])
        except Exception:
            out["status"] = "table_error"; return out
        # Step-1: brief
        try:
            r1 = client.chat.completions.create(model="grok-4-20-reasoning", max_tokens=4000,
                messages=[{"role": "system", "content": BRIEF_SYSTEM},
                          {"role": "user", "content": BRIEF_USER.format(q=rec["question"])}])
        except Exception as e:
            out["status"] = "brief_api_error"; out["detail"] = str(e)[:80]; return out
        brief = r1.choices[0].message.content or ""
        out["brief"] = brief
        m = re.search(r"Query Template:\s*(\w+)", brief)
        template = (m.group(1) if m else "").strip()
        out["template"] = template
        shell = TEMPLATES.get(template, "")
        # Step-2: fast planner, structured payload
        hint = render(link(rec["question"], shim.raw_df, cat))
        cat_txt = catalog_text(cat) + (("\n\n" + hint) if hint else "")
        sys2 = ("You are the plan compiler for a table QA system. Turn the question and the Step-1 "
                "briefing into ONE executable plan in the typed algebra. The briefing carries the "
                "reading; the catalog carries the real column names and cell spellings; map the "
                "briefing's question-space atoms onto exact columns/cells yourself. Never answer "
                "from memory. Reply with ONE JSON object: {\"tree\": {...}} or {\"abstain\": true}.")
        payload = {
            "question": rec["question"],
            "step1_briefing": brief,
            "stage1_query_template": template,
            "selected_template_shell": shell,
            "catalog_and_cell_hints": cat_txt,
            "algebra_reference": "nodes: filter values count size sum exists select extreme extreme_rows groupby argext top num combine vcompare setop gcombine keys_where; preds: eq in contains gte lte exists not any in_expr",
            "process": [
                "Read stage1_query_template; fill ONLY the shape allowed by selected_template_shell.",
                "Map every Explicit Atom to an exact column and cell spelling from the catalog.",
                "Honor Order Semantics via extreme_rows/argext over the named quantity.",
                "Copy cell values EXACTLY as they appear in the catalog samples.",
            ],
        }
        ev = WTQEvaluator(shim)
        targets = rec["target"].split("|")
        for _ in range(2):
            try:
                r2 = client.chat.completions.create(model="grok-4-1-fast-non-reasoning",
                    temperature=1.0, max_tokens=900,
                    messages=[{"role": "system", "content": sys2},
                              {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
            except Exception:
                continue
            mm = _J.search(r2.choices[0].message.content or "")
            if not mm:
                continue
            try:
                tree = json.loads(mm.group(0)).get("tree")
            except json.JSONDecodeError:
                continue
            if not isinstance(tree, dict):
                continue
            try:
                if validate_tree(tree) == "RECORDS":
                    continue
            except AlgebraError:
                continue
            res = ev.run(tree)
            if res.get("status") == "ok" and denotation_match(res["answer"], targets):
                out["status"] = "verified"; out["tree"] = tree
                return out
        out["status"] = "unrescued"
        return out

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(one, failed))

    ver = sum(1 for r in results if r["status"] == "verified")
    mix = {s: sum(1 for r in results if r["status"] == s) for s in set(r["status"] for r in results)}
    tmpl = {}
    for r in results:
        t = r.get("template", "?")
        tmpl[t] = tmpl.get(t, 0) + 1
    print(f"two-stage v2 (faithful port): {ver}/20 rescued")
    print("mix:", mix)
    print("template distribution:", tmpl)
    print("\nSCOREBOARD same 20 fast-failed questions:")
    print("  fast alone k=2:              0/20")
    print("  fast + v1 brief k=2:         3/20")
    print(f"  two-stage v2 (card port):    {ver}/20")
    print("  reasoning direct k=1:        7/20")
    with open(ROOT / "data/qa/wtq/pilot_brief_v2.jsonl", "w") as fh:
        for r in results:
            fh.write(json.dumps(r, default=str) + "\n")

if __name__ == "__main__":
    main()
