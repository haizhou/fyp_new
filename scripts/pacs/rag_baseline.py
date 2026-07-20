#!/usr/bin/env python3
"""F: RAG baseline on PACS-test answerable rows (channel a).
Lexical retrieval over the flat record universe (entity/token match, top-k=20
rows serialized as context) + local 8B reader answering directly. Honest label:
this is the retrieval-and-read paradigm the intro contrasts; exhaustive
aggregation is impossible from a 20-row window by construction.
"""
from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import importlib.util

spec = importlib.util.spec_from_file_location("reg1", ROOT / "scripts/compose_regression.py")
reg1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reg1)
from procurement_graph.qa.benchmark.kg_interface import ParquetKGQueryBackend  # noqa: E402

K = 20
COLS = ["buyer_name", "supplier_name", "release_year", "tender_category",
        "tender_cpv_id", "value_amount", "value_is_additive"]


def main() -> None:
    backend = ParquetKGQueryBackend.from_directory(ROOT / "data/kg", include_evidence=False)
    df = backend.records_df
    rows = [json.loads(l) for l in open(ROOT / "data/qa/pacs_v1/test_channel_a.jsonl")]
    rows = [r for r in rows if r["expected_status"] == "answerable"]

    from openai import OpenAI
    client = OpenAI(base_url="http://127.0.0.1:8002/v1", api_key="local", timeout=180)

    names = set(df["buyer_name"]) | set(df["supplier_name"])

    def retrieve(q: str):
        hits = df
        matched = [n for n in names if n and len(n) > 4 and n.casefold() in q.casefold()]
        if matched:
            m = hits["buyer_name"].isin(matched) | hits["supplier_name"].isin(matched)
            hits = hits[m]
        years = re.findall(r"\b(202[2-6])\b", q)
        if years and len(hits) > K:
            hits = hits[hits["release_year"].isin([int(y) for y in years])]
        cpvs = re.findall(r"\b(\d{8})\b", q)
        if cpvs and len(hits) > K:
            hits = hits[hits["tender_cpv_id"].astype(str).isin(cpvs)]
        return hits.head(K)[COLS]

    def one(row):
        ctx = retrieve(row["question"]).to_csv(index=False)
        try:
            resp = client.chat.completions.create(
                model="Qwen/Qwen3-8B", temperature=0.0, max_tokens=200,
                messages=[{"role": "user", "content":
                           f"Records (CSV, up to {K} retrieved):\n{ctx}\n\n"
                           f"Question: {row['question']}\n"
                           "Answer with ONLY the final value (number, name, year, true/false, or a JSON list)."}])
            ans = (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            return {"id": row["id"], "correct": False, "error": str(exc)[:80]}
        try:
            parsed = json.loads(ans)
        except Exception:
            parsed = ans.strip('."')
        ok = reg1._match(row["oracle_answer"], {"status": "ok", "answer": parsed})
        return {"id": row["id"], "correct": bool(ok), "answer": str(parsed)[:120]}

    with ThreadPoolExecutor(max_workers=8) as pool:
        out = list(pool.map(one, rows))
    with (ROOT / "data/qa/pacs_v1/eval_rag_baseline.jsonl").open("w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    c = sum(r["correct"] for r in out)
    print(f"RAG baseline on PACS-test answerable: {c}/{len(out)} = {100*c/len(out):.2f}%")


if __name__ == "__main__":
    main()
