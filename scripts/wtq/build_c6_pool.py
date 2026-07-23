#!/usr/bin/env python3
"""Assemble the C-v6 training pool (combined-fork stage 1).

Sources, in precedence order:
1. Fresh gold-translation base (data/training/wtq_sft_C/sft.jsonl, rebuilt under
   v4c views + first-line parse) — canonical C lineage.
2. C5 self-harvest supplement (harvest_C5self_*.jsonl): verified trees on
   zero-hit questions the gold pool never covered; dedup by (context, question)
   against the base; shortest-first, cap 2 per question (A-pool discipline).
3. Idiom exemplars: the hand/teacher-proven trees for the E13n/E13t blind-spot
   classes, replicated IDIOM_WEIGHT x so ~1% of the pool demonstrates each
   rare shape (gcombine rowwise arithmetic, keys_where tied extremum,
   row_index literal navigation, in_expr + setop difference self-exclusion,
   __norm/__noparen twin usage).

Output: data/training/wtq_sft_C6/{wtq_sft_train.json, wtq_sft_val.json,
dataset_info.json} in LLaMA-Factory alpaca format (instruction/input/output),
val = last VAL_N examples of the base (kept from base only, for comparability).
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/wtq"))

from linker import link, render  # noqa: E402
from loader import catalog_text, load_universe  # noqa: E402
from zero_shot import PROMPT  # noqa: E402

IDIOM_WEIGHT = 10
VAL_N = 200


def nodes(tree) -> int:
    if isinstance(tree, dict):
        return 1 + sum(nodes(v) for v in tree.values())
    if isinstance(tree, list):
        return sum(nodes(v) for v in tree)
    return 0


IDIOMS = [  # (context, question, tree) — all oracle-verified in E13n/E13o/E13p/E13m
    ("csv/203-csv/705.csv", "how many people stayed at least 3 years in office?",
     {"node": "size", "of": {"node": "keys_where", "of": {"node": "gcombine", "op": "diff",
      "left": {"node": "groupby", "of": {"node": "filter", "where": [{"field": "name", "op": "exists"}]}, "key": "name", "metric": "sum", "field": "left_office__min_year"},
      "right": {"node": "groupby", "of": {"node": "filter", "where": [{"field": "name", "op": "exists"}]}, "key": "name", "metric": "sum", "field": "took_office__min_year"}},
      "op": "ge", "value": 3}}),
    ("csv/204-csv/849.csv", "what counties had the least participants for the race?",
     {"node": "keys_where", "of": {"node": "groupby", "of": {"node": "filter", "where": [{"field": "nationality", "op": "exists"}]}, "key": "nationality", "metric": "count"}, "op": "le", "value": 1}),
    ("csv/203-csv/812.csv", "who ranked right after turkey?",
     {"node": "select", "of": {"node": "filter", "where": [{"field": "row_index", "op": "eq", "value": 7}]}, "field": "nation"}),
    ("csv/203-csv/743.csv", "how many beta versions were released before the first full release?",
     {"node": "count", "of": {"node": "filter", "where": [{"field": "row_index", "op": "lte", "value": 9}]}}),
    ("csv/203-csv/116.csv", "which players played the same position as ardo kreek?",
     {"node": "setop", "op": "difference",
      "left": {"node": "values", "of": {"node": "filter", "where": [{"field": "position__norm", "op": "in_expr", "expr": {"node": "values", "of": {"node": "filter", "where": [{"field": "player", "op": "eq", "value": "Ardo Kreek"}]}, "field": "position__norm"}}]}, "field": "player"},
      "right": {"node": "values", "of": {"node": "filter", "where": [{"field": "player", "op": "eq", "value": "Ardo Kreek"}]}, "field": "player"}}),
    ("csv/204-csv/361.csv", "who is the first away team on the chart",
     {"node": "select", "of": {"node": "filter", "where": [{"field": "row_index", "op": "eq", "value": 1}]}, "field": "away_team__noparen"}),
    ("csv/203-csv/577.csv", "what was the average number of years served by a coach?",
     {"node": "combine", "op": "ratio",
      "left": {"node": "sum", "of": {"node": "filter", "where": [{"op": "not", "pred": {"field": "tenure", "op": "eq", "value": "Totals"}}]}, "field": "years"},
      "right": {"node": "count", "of": {"node": "filter", "where": [{"op": "not", "pred": {"field": "tenure", "op": "eq", "value": "Totals"}}]}}}),
]


def render_example(context, question, tree, universes):
    if context not in universes:
        try:
            universes[context] = load_universe(context)
        except Exception:  # noqa: BLE001
            universes[context] = None
    if universes[context] is None:
        return None
    shim, catalog = universes[context]
    hint = render(link(question, shim.raw_df, catalog))
    cat_txt = catalog_text(catalog) + (("\n\n" + hint) if hint else "")
    return {"instruction": PROMPT.format(catalog=cat_txt, q=question), "input": "",
            "output": json.dumps({"tree": tree}, separators=(",", ":"))}


def main() -> None:
    universes: dict = {}
    base_rows = []
    seen_q = set()
    for line in open(ROOT / "data/training/wtq_sft_C/sft.jsonl"):
        r = json.loads(line)
        msgs = r["messages"]
        base_rows.append({"instruction": msgs[0]["content"], "input": "", "output": msgs[1]["content"]})
        q = msgs[0]["content"].rsplit("Question: ", 1)[-1].strip()
        seen_q.add(q)
    print(f"base (fresh gold-translation): {len(base_rows)} examples, {len(seen_q)} distinct questions")

    supp, per_q = [], {}
    for f in sorted(glob.glob(str(ROOT / "data/qa/wtq/harvest_C5self_*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            if r.get("status") != "ok" or not r.get("trees"):
                continue
            if r["question"].strip() in seen_q:
                continue
            per_q.setdefault((r["context"], r["question"]), []).extend(r["trees"])
    for (ctx, q), trees in per_q.items():
        for t in sorted(trees, key=nodes)[:2]:
            ex = render_example(ctx, q, t, universes)
            if ex:
                supp.append(ex)
    print(f"C5self supplement: {len(supp)} examples over {len(per_q)} new questions")

    idiom_rows = []
    for ctx, q, t in IDIOMS:
        ex = render_example(ctx, q, t, universes)
        if ex:
            idiom_rows.extend([ex] * IDIOM_WEIGHT)
    print(f"idiom exemplars: {len(IDIOMS)} shapes x{IDIOM_WEIGHT} = {len(idiom_rows)} examples")

    val = base_rows[-VAL_N:]
    train = base_rows[:-VAL_N] + supp + idiom_rows
    import random
    random.seed(7)
    random.shuffle(train)
    out = ROOT / "data/training/wtq_sft_C6"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(train, (out / "wtq_sft_train.json").open("w"), ensure_ascii=False)
    json.dump(val, (out / "wtq_sft_val.json").open("w"), ensure_ascii=False)
    json.dump({
        "wtq_sft_train": {"file_name": "wtq_sft_train.json"},
        "wtq_sft_val": {"file_name": "wtq_sft_val.json"},
    }, (out / "dataset_info.json").open("w"), indent=1)
    print(f"C6 pool: train {len(train)}, val {len(val)} -> {out}")


if __name__ == "__main__":
    main()
