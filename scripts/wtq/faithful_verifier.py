#!/usr/bin/env python3
"""FAITHFUL port of LLM-as-a-Verifier (arXiv 2607.05391), per the official repo:

- pairwise A/B template ("expert reviewer... two trajectories"), free-form
  analysis FIRST, then scores in <score_A>/<score_B> tags;
- LETTER scale A-T (single token) mapped to 1-20 (A=1 incorrect, J=10
  borderline, T=20 correct); continuous score = expectation over the letter-token
  logprob distribution AT the tag position;
- "Trust observed output, NOT narration": each trajectory shows the plan AND
  its executed answer;
- judge: grok-4-1-fast (Flash-class), temperature 0, logprobs on.

Candidate score = mean expectation over its round-robin pairings.
Arms on dev-300 (C-v6 k=4, deterministic gate first): k1 / first / faithful / oracle.
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
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

env = dict(l.strip().split("=", 1) for l in open(ROOT / ".env") if "=" in l)
cloud = OpenAI(base_url=env["AZURE_OPENAI_BASE_URL"], api_key=env["AZURE_OPENAI_API_KEY"], timeout=180)
local = OpenAI(base_url="http://127.0.0.1:8003/v1", api_key="local", timeout=300)

LETTERS = [chr(ord("A") + i) for i in range(20)]  # A..T -> 1..20
VAL = {c: i + 1 for i, c in enumerate(LETTERS)}
TAG_A = re.compile(r"<score_A>\s*([A-T])\s*</score_A>")
TAG_B = re.compile(r"<score_B>\s*([A-T])\s*</score_B>")

JUDGE_TMPL = """You are an expert data-analysis reviewer. You will see a question about one table, \
and two candidate query-plan trajectories (each shows the executable plan and the answer it \
actually produced on the full table). Trust observed output, NOT narration.

Evaluation criteria: (1) every plan field references the correct table column for the question; \
(2) every condition stated in the question appears as a filter with exact cell spellings, none invented; \
(3) the root operation and answer column produce exactly the kind of answer asked for; \
(4) the observed answer is plausible for the question.

Table:
{catalog}

Question: {q}

Trajectory A:
plan: {plan_a}
observed answer: {ans_a}

Trajectory B:
plan: {plan_b}
observed answer: {ans_b}

Analyze both trajectories against the criteria. Then output your final ratings on an A-T letter \
scale (A = incorrect, J = borderline, T = correct), exactly in this format:
<score_A> LETTER </score_A> <score_B> LETTER </score_B>"""


def letter_expectation(resp, text, tag_re):
    m = tag_re.search(text)
    if not m:
        return None
    letter_off = m.start(1)
    toks = resp.choices[0].logprobs.content
    pos = 0
    for tk in toks:
        nxt = pos + len(tk.token)
        if pos <= letter_off < nxt:
            probs = {}
            for cand in tk.top_logprobs:
                s = cand.token.strip()
                if s in VAL:
                    probs[s] = probs.get(s, 0.0) + math.exp(cand.logprob)
            if probs:
                z = sum(probs.values())
                return sum(VAL[c] * p for c, p in probs.items()) / z
            return float(VAL[m.group(1)])
        pos = nxt
    return float(VAL[m.group(1)])


def judge_pair(cat_txt, q, a, b):
    prompt = JUDGE_TMPL.format(catalog=cat_txt, q=q,
                               plan_a=json.dumps(a["tree"], separators=(",", ":")), ans_a=str(a["ans"])[:120],
                               plan_b=json.dumps(b["tree"], separators=(",", ":")), ans_b=str(b["ans"])[:120])
    try:
        r = cloud.chat.completions.create(model="grok-4-1-fast-non-reasoning", temperature=0.0,
            max_tokens=700, logprobs=True, top_logprobs=20,
            messages=[{"role": "user", "content": prompt}])
        text = r.choices[0].message.content or ""
        sa = letter_expectation(r, text, TAG_A)
        sb = letter_expectation(r, text, TAG_B)
        return sa, sb
    except Exception:  # noqa: BLE001
        return None, None


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
            return {"id": rec["id"], "k1": False, "first": False, "faithful": False, "oracle": False}
        fields = [c[0] for c in cat]
        schema = {"type": "json_schema", "json_schema": {
            "name": "algebra", "schema": algebra_json_schema(fields=fields), "strict": True}}
        hint = render(link(rec["utterance"], shim.raw_df, cat))
        cat_txt = catalog_text(cat) + (("\n\n" + hint) if hint else "")
        prompt = PROMPT.format(catalog=cat_txt, q=rec["utterance"])
        ev = WTQEvaluator(shim)
        targets = rec["targetValue"].split("|")

        def run_tree(tree):
            res = ev.run(tree)
            ok = res.get("status") == "ok"
            return ok, (res.get("answer") if ok else None), (denotation_match(res["answer"], targets) if ok else False)

        try:
            r1 = local.chat.completions.create(model="wtq_C6", temperature=0.0, max_tokens=900,
                messages=[{"role": "user", "content": prompt}], response_format=schema)
            t1 = json.loads(r1.choices[0].message.content or "{}").get("tree")
            k1 = isinstance(t1, dict) and run_tree(t1)[2]
        except Exception:  # noqa: BLE001
            k1 = False

        cands, seen = [], set()
        try:
            r4 = local.chat.completions.create(model="wtq_C6", temperature=1.0, n=4, max_tokens=900,
                messages=[{"role": "user", "content": prompt}], response_format=schema)
            choices = r4.choices
        except Exception:  # noqa: BLE001
            choices = []
        for i, ch in enumerate(choices):
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
            ok, ans, correct = run_tree(t)
            if not ok:
                continue
            cands.append({"tree": t, "ans": ans, "correct": correct, "pos": i, "scores": []})
        if not cands:
            return {"id": rec["id"], "k1": k1, "first": False, "faithful": False, "oracle": False}
        first = cands[0]["correct"]
        oracle = any(c["correct"] for c in cands)
        if len(cands) == 1:
            faith = cands[0]["correct"]
        else:
            for a, b in combinations(range(len(cands)), 2):
                sa, sb = judge_pair(cat_txt, rec["utterance"], cands[a], cands[b])
                if sa is not None:
                    cands[a]["scores"].append(sa)
                if sb is not None:
                    cands[b]["scores"].append(sb)
            best = max(cands, key=lambda c: (sum(c["scores"]) / len(c["scores"]) if c["scores"] else 0.0, -c["pos"]))
            faith = best["correct"]
        return {"id": rec["id"], "k1": k1, "first": first, "faithful": faith, "oracle": oracle}

    with ThreadPoolExecutor(max_workers=6) as pool:
        res = list(pool.map(one, rows))
    n = len(res)
    for arm in ("k1", "first", "faithful", "oracle"):
        v = sum(1 for r in res if r[arm])
        print(f"{arm:9}: {v}/{n} = {100*v/n:.2f}%")
    print("(prior context: strawman LLM judge 56.67, deterministic reflector 59.33)")
    json.dump(res, open(ROOT / "data/qa/wtq/faithful_verifier.json", "w"), indent=1)


if __name__ == "__main__":
    main()
