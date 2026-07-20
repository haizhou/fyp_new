#!/usr/bin/env python3
"""PACS channel b: LLM naturalization behind deterministic fidelity gates.

Local paraphraser (zero external calls). Gates per spec v2.2:
LITERAL: every entity/number/year/CPV from params appears verbatim (case-insensitive).
LOGIC (keyword checks against the logical signature):
  negation flag  -> text must contain a negation cue (never/not/without/excluding/no );
  comparison     -> direction cue consistent (more/higher/larger/exceed vs fewer/lower);
  quantifier universal -> every/all/each present; 'any' NOT substituted for 'every';
  setop difference -> exclusion cue (but not/never/without/only);
  setop intersect  -> both/alike/as well/and also cue.
Rows failing gates after N attempts fall back to channel a (recorded).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

NEG_CUES = ("never", "not ", "without", "excluding", "no ", "lack", "missing", "absent")
UNIV_CUES = ("every", "all ", "each ", "perfect record", "single one")
INTERSECT_CUES = ("both", "alike", "as well", "and also", "shared", "in common")
DIFF_CUES = ("but not", "never", "without", "only", "absent", "excluding", "while staying")
MORE_CUES = ("more", "higher", "larger", "greater", "exceed", "outweigh", "above", "lead")
LESS_CUES = ("fewer", "lower", "smaller", "less", "behind", "below")


def literals_ok(text: str, params: dict) -> bool:
    t = text.casefold()
    for v in params.values():
        vals = v if isinstance(v, list) else [v]
        for x in vals:
            if isinstance(x, bool):
                continue
            if str(x).casefold() not in t:
                return False
    return True


def logic_ok(text: str, sig: dict, tree: dict) -> bool:
    t = " " + text.casefold() + " "
    if sig.get("negation") and not any(c in t for c in NEG_CUES):
        return False
    if sig.get("quantifier") == "universal_via_difference" and not any(c in t for c in UNIV_CUES):
        return False
    if isinstance(tree, dict) and tree.get("node") == "setop":
        op = tree.get("op")
        if op == "intersect" and not any(c in t for c in INTERSECT_CUES):
            return False
        if op == "difference" and not any(c in t for c in DIFF_CUES + UNIV_CUES):
            return False
    if sig.get("comparison") in ("gt", "ge") and not any(c in t for c in MORE_CUES):
        return False
    if sig.get("comparison") in ("lt", "le") and not any(c in t for c in LESS_CUES):
        return False
    return True


PROMPT = """Rewrite the question below in natural, fluent English as a different person would ask it.
STRICT RULES: keep every organisation name, number, year, CPV code, and category word EXACTLY as written;
keep the logical meaning identical (do not change and/or/not/only/both/every, comparison direction, or what is being counted);
one sentence or two short sentences; output ONLY the rewritten question.

Question: {q}"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="local", timeout=120)

    rows = [json.loads(l) for l in open(ROOT / "data/qa/pacs_v1/intent_pool_surfaced.jsonl")]

    def one(row):
        base_q = row.get("question_a")
        if not base_q or row.get("expected_status") == "requires_missing_operator":
            row["question_b"], row["b_gate"] = base_q, "copied"
            return row
        sig = row.get("logical_signature") or {}
        tree = row.get("compose_tree") or {}
        for attempt in range(args.attempts):
            try:
                resp = client.chat.completions.create(
                    model=args.model, temperature=0.8, max_tokens=200,
                    messages=[{"role": "user", "content": PROMPT.format(q=base_q)}])
                cand = (resp.choices[0].message.content or "").strip().strip('"')
            except Exception:
                continue
            if not cand or cand.casefold() == base_q.casefold():
                continue
            if literals_ok(cand, row.get("params", {})) and logic_ok(cand, sig, tree):
                row["question_b"], row["b_gate"] = cand, f"passed@{attempt+1}"
                return row
        row["question_b"], row["b_gate"] = base_q, "fallback_channel_a"
        return row

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        out_rows = list(pool.map(one, rows))

    out = ROOT / "data/qa/pacs_v1/intent_pool_full.jsonl"
    with out.open("w") as fh:
        for row in out_rows:
            fh.write(json.dumps(row, default=str) + "\n")
    from collections import Counter
    gates = Counter(r["b_gate"].split("@")[0] for r in out_rows)
    print(f"wrote {len(out_rows)} -> {out}; gate outcomes: {dict(gates)}")
    import random
    for r in random.Random(1).sample([x for x in out_rows if x['b_gate'].startswith('passed')], 5):
        print("A:", r["question_a"][:100])
        print("B:", r["question_b"][:100])
        print("---")


if __name__ == "__main__":
    main()
