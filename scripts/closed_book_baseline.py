#!/usr/bin/env python3
"""Closed-book LLM baseline on the sealed final test (single pass, new frozen config).

No retrieval, no knowledge graph, no execution. The model answers each question
from parametric knowledge alone under the same strict JSON contract and the same
type-aware scorer as every other system. This anchors the floor of the ladder
and instantiates the fluent-wrong-answer failure the task definition warns of.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_compare import _jsonable, _slim, is_correct  # noqa: E402

from procurement_graph.qa.benchmark.chat import ChatClient  # noqa: E402

_ABSTAIN = {"unknown", "n/a", "none", "cannot answer", "not enough information"}

MODEL = sys.argv[1] if len(sys.argv) > 1 else "grok-4-1-fast-non-reasoning"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "outputs/eval/final_test/closed_book_grok"
OUT.mkdir(parents=True, exist_ok=True)

rows = [json.loads(l) for l in open(ROOT / "data/qa/cicada_core_v4/final_test.jsonl")]
chat = ChatClient.from_env()

SYSTEM = ("You answer questions about real UK public procurement contract award notices "
          "published between 2022 and 2026, using only your own knowledge. No records are "
          "provided. If you cannot know the answer, reply exactly {\"answer\": \"unknown\"}. "
          "Return strict JSON {\"answer\": <value>}.")


def one(row):
    try:
        parsed = getattr(chat.complete_json(model=MODEL, system=SYSTEM,
                                            user=json.dumps({"question": row["question"]},
                                                            ensure_ascii=False)),
                         "parsed", {}) or {}
        ans = parsed.get("answer") if isinstance(parsed, dict) else None
        if ans is not None and str(ans).strip().casefold() in _ABSTAIN:
            ans = None
    except Exception:  # noqa: BLE001
        ans = None
    return {**_slim(row), "predicted": _jsonable(ans), "correct": is_correct(ans, row)}


with ThreadPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(one, rows))

with (OUT / "compare_cicada.results.jsonl").open("w") as fh:
    for r in results:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
n = len(results)
c = sum(1 for r in results if r["correct"])
json.dump({"overall": {"total": n, "correct": c, "accuracy": round(c / n, 4)}},
          (OUT / "compare_cicada.summary.json").open("w"), indent=1)
print(f"closed-book {MODEL}: {c}/{n} = {100*c/n:.2f}%")
