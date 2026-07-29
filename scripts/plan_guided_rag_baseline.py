#!/usr/bin/env python3
"""Oracle-plan-guided retrieval baseline on the sealed final test (single pass).

The strongest possible form of plan-then-retrieve. The retrieval query is built
from the GOLD program's constraint literals, so question understanding and
planning are perfect by construction, and only the evidence mechanism differs
from the full system, top-k retrieval plus an LLM reader instead of exhaustive
deterministic execution. If this baseline still fails on aggregation, the
failure is attributable to retrieval itself, not to query formulation.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_compare import _BOOST, _jsonable, _rag_answer, _slim, is_correct  # noqa: E402

from procurement_graph.qa.benchmark.chat import ChatClient  # noqa: E402
from procurement_graph.qa.benchmark.kg_interface import ParquetKGQueryBackend  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "grok-4-1-fast-non-reasoning"
OUT = ROOT / "outputs/eval/final_test/plan_guided_rag_grok"
OUT.mkdir(parents=True, exist_ok=True)

rows = [json.loads(l) for l in open(ROOT / "data/qa/cicada_core_v4/final_test.jsonl")]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
print("loading KG + building TF-IDF index (strong config) ...", flush=True)
df = ParquetKGQueryBackend.from_directory(ROOT / "data" / "kg").records_df
cols = ["buyer_name", "supplier_name", "tender_category", "tender_cpv_id", "tender_cpv_description",
        "release_year", "value_amount", "tender_title", "award_date_signed"]
cols = [c for c in cols if c in df.columns]
sub = df[cols].fillna("")
texts, plain = [], []
for row in sub.itertuples(index=False, name=None):
    rowd = dict(zip(cols, row))
    plain.append(" | ".join(f"{c}: {rowd[c]}" for c in cols))
    texts.append(" ".join((f"{c}: {rowd[c]} " * _BOOST.get(c, 1)) for c in cols))
vec = TfidfVectorizer(max_features=120000, ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
matrix = vec.fit_transform(texts)
chat = ChatClient.from_env()
TOP_K = 40


def gold_query(row):
    parts = [row["question"]]
    for c in row.get("constraints") or []:
        v = c.get("value")
        if v is not None:
            parts.append(f"{c.get('field', '')} {v}")
    return " ".join(str(p) for p in parts)


def one(row):
    q = gold_query(row)
    sims = linear_kernel(vec.transform([q]), matrix).ravel()
    idx = sims.argsort()[::-1][:TOP_K]
    ctx = [plain[i] for i in idx]
    pred = _rag_answer(chat, MODEL, row["question"], ctx)
    return {**_slim(row), "predicted": _jsonable(pred), "correct": is_correct(pred, row),
            "retrieved": len(ctx)}


with ThreadPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(one, rows))

with (OUT / "compare_cicada.results.jsonl").open("w") as fh:
    for r in results:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
n = len(results)
c = sum(1 for r in results if r["correct"])
json.dump({"overall": {"total": n, "correct": c, "accuracy": round(c / n, 4)}},
          (OUT / "compare_cicada.summary.json").open("w"), indent=1)
print(f"plan-guided RAG {MODEL}: {c}/{n} = {100*c/n:.2f}%")
