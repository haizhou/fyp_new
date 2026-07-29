#!/usr/bin/env python3
"""Logit-expectation verifier reranking (LLM-as-a-Verifier port, arXiv 2607.05391).

Hierarchy preserved: deterministic checks (schema/validate/execute) remain the
ONLY admission gate; the probabilistic verifier ranks WITHIN the surviving set.
Score = mean over C criteria of E[digit] under the base model's first-token
distribution (1-9 scale, renormalized over digit tokens). Criteria follow the
E13t fault taxonomy: column choice, constraint completeness, answer shape.

Arms on clean-eval-devfold (n=300):
  k1        -- C6 single sample (ladder protocol reference)
  first     -- first verified of k=4 (no reranking control)
  verifier  -- argmax verifier score among verified k=4
  oracle    -- any verified sample correct (best-of-4 ceiling)
"""
from __future__ import annotations

import csv
import json
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

BASE = "http://127.0.0.1:8003/v1"
client = OpenAI(base_url=BASE, api_key="local", timeout=300)

CRITERIA = [
    ("columns", "Does every field in the plan reference the CORRECT column of this table for what the question asks (including numeric/normalized twin columns where appropriate)?"),
    ("constraints", "Does the plan express EVERY condition stated in the question as a filter, with values matching the table's exact cell spellings, and no invented conditions?"),
    ("shape", "Do the root operation and the answer column produce exactly the KIND of answer the question asks for (single value vs list vs count vs sum vs comparison)?"),
]
DIGITS = [str(d) for d in range(1, 10)]


def verifier_score(cat_txt, question, tree):
    total = 0.0
    for _, crit in CRITERIA:
        prompt = (f"{cat_txt}\n\nQuestion: {question}\n\nCandidate query plan (JSON):\n"
                  f"{json.dumps(tree, separators=(',', ':'))}\n\n"
                  f"Criterion: {crit}\nRate 1-9 (1 = clearly wrong, 5 = uncertain, 9 = clearly right). "
                  f"Reply with a single digit.")
        try:
            r = client.chat.completions.create(model="Qwen/Qwen3-8B", temperature=0.0,
                max_tokens=1, logprobs=True, top_logprobs=20,
                messages=[{"role": "user", "content": prompt}])
            top = r.choices[0].logprobs.content[0].top_logprobs
        except Exception:  # noqa: BLE001
            total += 5.0
            continue
        import math
        probs = {}
        for t in top:
            s = t.token.strip()
            if s in DIGITS:
                probs[s] = probs.get(s, 0.0) + math.exp(t.logprob)
        if not probs:
            total += 5.0
            continue
        z = sum(probs.values())
        total += sum(int(d) * p for d, p in probs.items()) / z
    return total / len(CRITERIA)


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
            return {"id": rec["id"], "k1": False, "first": False, "verifier": False, "oracle": False}
        fields = [c[0] for c in cat]
        schema = {"type": "json_schema", "json_schema": {
            "name": "algebra", "schema": algebra_json_schema(fields=fields), "strict": True}}
        hint = render(link(rec["utterance"], shim.raw_df, cat))
        cat_txt = catalog_text(cat) + (("\n\n" + hint) if hint else "")
        prompt = PROMPT.format(catalog=cat_txt, q=rec["utterance"])
        ev = WTQEvaluator(shim)
        targets = rec["targetValue"].split("|")

        def correct(tree):
            res = ev.run(tree)
            return res.get("status") == "ok" and denotation_match(res["answer"], targets)

        try:
            r1 = client.chat.completions.create(model="wtq_C6", temperature=0.0, max_tokens=900,
                messages=[{"role": "user", "content": prompt}], response_format=schema)
            t1 = json.loads(r1.choices[0].message.content or "{}").get("tree")
            k1 = isinstance(t1, dict) and correct(t1)
        except Exception:  # noqa: BLE001
            k1 = False

        cands, seen = [], set()
        try:
            r4 = client.chat.completions.create(model="wtq_C6", temperature=1.0, n=4, max_tokens=900,
                messages=[{"role": "user", "content": prompt}], response_format=schema)
            choices = r4.choices
        except Exception:  # noqa: BLE001
            choices = []
        for ch in choices:
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
                if validate_tree(t) == "RECORDS":
                    continue
            except AlgebraError:
                continue
            res = ev.run(t)
            if res.get("status") != "ok":
                continue
            cands.append((t, denotation_match(res["answer"], targets)))
        if not cands:
            return {"id": rec["id"], "k1": k1, "first": False, "verifier": False, "oracle": False}
        first = cands[0][1]
        oracle = any(ok for _, ok in cands)
        if len(cands) == 1:
            ver = cands[0][1]
        else:
            scored = [(verifier_score(cat_txt, rec["utterance"], t), ok) for t, ok in cands]
            ver = max(scored, key=lambda x: x[0])[1]
        return {"id": rec["id"], "k1": k1, "first": first, "verifier": ver, "oracle": oracle}

    with ThreadPoolExecutor(max_workers=6) as pool:
        res = list(pool.map(one, rows))
    n = len(res)
    for arm in ("k1", "first", "verifier", "oracle"):
        v = sum(1 for r in res if r[arm])
        print(f"{arm:9}: {v}/{n} = {100*v/n:.2f}%")
    json.dump(res, open(ROOT / "data/qa/wtq/verifier_rerank.json", "w"), indent=1)


if __name__ == "__main__":
    main()
