#!/usr/bin/env python3
"""Run the comparison eval with either our KG-reasoning system or a baseline RAG.

  --system ours   ReasoningPipeline (plan -> ground -> exhaustive KG execute; abstains when unanswerable)
  --system rag    TF-IDF retrieve top-k contract records -> LLM answers from those records only

Both read `compare_set.jsonl` and are scored identically: answerable -> answer matches oracle
(type-aware); unanswerable -> correct iff the system does NOT produce an answer (abstains). Writes
`compare_<system>.results.jsonl` + `compare_<system>.summary.json` (accuracy by category).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_targeted_v2 import answers_match  # shared type-aware comparison

COMPARE_SET = ROOT / "data" / "qa" / "eval" / "compare_set.jsonl"
_ABSTAIN_WORDS = {"unknown", "none", "n/a", "na", "not available", "cannot determine", "not answerable", ""}


def read_jsonl(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def is_correct(pred: Any, row: dict) -> bool:
    if row.get("expected_status", "answerable") != "answerable":
        return pred is None  # abstain cases: correct = did NOT hallucinate an answer
    return answers_match(pred, row.get("oracle_answer"), row.get("answer_type", ""))


# ------------------------------------------------------------------ our system
def run_ours(rows, args):
    from procurement_graph.reasoning import ReasoningPipeline, RuleBasedDryRunPlanner
    from procurement_graph.reasoning.kg_backend import RuntimeKGBackend
    print("[compare/ours] loading KG ...", flush=True)
    backend = RuntimeKGBackend.from_directory(ROOT / "data" / "kg")
    resolver = backend.org_resolver()
    if args.planner == "rule_decomp":
        from procurement_graph.reasoning.planner_decomposition import DecompositionAwarePlanner
        planner = DecompositionAwarePlanner(org_resolver=resolver)
    elif args.planner == "hybrid":
        from procurement_graph.reasoning.planner_decomposition import DecompositionAwarePlanner, HybridPlanner
        from procurement_graph.reasoning.llm_planner import LLMReasoningPlanner
        planner = HybridPlanner(rule=DecompositionAwarePlanner(org_resolver=resolver),
                                llm=LLMReasoningPlanner.from_env(args.model, org_resolver=resolver))
    else:
        planner = RuleBasedDryRunPlanner()
    trace_reflector = None
    if args.reflect == "on":
        from procurement_graph.reasoning.trace_reflector import TraceReflector
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        trace_reflector = TraceReflector(org_resolver=resolver,
                                         preference_log=out_dir / "preference_log.jsonl")
    pipe = ReasoningPipeline(backend=backend, planner=planner, org_resolver=resolver,
                             trace_reflector=trace_reflector)
    out = []
    for row in rows:
        pred = pipe.run(row["question"]).answer_card.answer
        out.append({**_slim(row), "predicted": _jsonable(pred), "correct": is_correct(pred, row)})
    return out


# ------------------------------------------------------------------ baseline RAG (naive | strong)
# naive : plain per-record doc, default TF-IDF, top-10.
# strong: entity-boosted doc (buyer/supplier/CPV repeated) + sublinear TF-IDF with word bigrams +
#         larger context (top-40). A credible retrieval upgrade; the structural limits (exhaustive
#         aggregation, multi-hop joins, abstention) remain because RAG still reasons over a text subset.
_BOOST = {"buyer_name": 3, "supplier_name": 3, "tender_cpv_id": 3, "tender_cpv_description": 2}


def run_rag(rows, args):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    from procurement_graph.qa.benchmark.kg_interface import ParquetKGQueryBackend
    strong = args.rag_mode == "strong"
    print(f"[compare/rag:{args.rag_mode}] loading KG + building TF-IDF index ...", flush=True)
    df = ParquetKGQueryBackend.from_directory(ROOT / "data" / "kg").records_df
    cols = ["buyer_name", "supplier_name", "tender_category", "tender_cpv_id", "tender_cpv_description",
            "release_year", "value_amount", "tender_title", "award_date_signed"]
    cols = [c for c in cols if c in df.columns]
    sub = df[cols].fillna("")
    boost = _BOOST if strong else {}
    texts, plain = [], []
    for row in sub.itertuples(index=False, name=None):
        rowd = dict(zip(cols, row))
        plain.append(" | ".join(f"{c}: {rowd[c]}" for c in cols))
        texts.append(" ".join((f"{c}: {rowd[c]} " * boost.get(c, 1)) for c in cols) if strong else plain[-1])
    vec = (TfidfVectorizer(max_features=120000, ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
           if strong else TfidfVectorizer(max_features=50000, stop_words="english"))
    matrix = vec.fit_transform(texts)
    top_k = args.top_k if args.top_k else (40 if strong else 10)
    chat = None
    if args.rag_llm == "on":
        from procurement_graph.qa.benchmark.chat import ChatClient
        chat = ChatClient.from_env()

    def retrieve(question, k):
        sims = linear_kernel(vec.transform([question]), matrix).ravel()
        idx = sims.argsort()[::-1][:k]
        return [plain[i] for i in idx]  # show the LLM the clean per-record text

    out = []
    for row in rows:
        ctx = retrieve(row["question"], top_k)
        pred = _rag_answer(chat, args.model, row["question"], ctx) if chat else None
        out.append({**_slim(row), "predicted": _jsonable(pred), "correct": is_correct(pred, row),
                    "retrieved": len(ctx)})
    return out


def _rag_answer(chat, model, question, records):
    system = ("You answer UK public-procurement questions using ONLY the retrieved contract records "
              "below. Do not use outside knowledge. If the records do not contain enough to answer, "
              "reply exactly {\"answer\": \"unknown\"}. Return strict JSON {\"answer\": <value>}.")
    user = json.dumps({"question": question, "records": records}, ensure_ascii=False)
    try:
        parsed = getattr(chat.complete_json(model=model, system=system, user=user), "parsed", {}) or {}
    except Exception as exc:  # pragma: no cover - live only
        return None
    ans = parsed.get("answer") if isinstance(parsed, dict) else None
    if ans is None or str(ans).strip().casefold() in _ABSTAIN_WORDS:
        return None
    return ans


# ------------------------------------------------------------------ shared
def _slim(row):
    return {k: row.get(k) for k in ("id", "dataset", "category", "answer_type", "expected_status", "oracle_answer")}


def _jsonable(v):
    return v if isinstance(v, (str, int, float, bool, type(None), list, dict)) else str(v)


def summarize(results):
    by = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        b = by[r["category"]]
        b["total"] += 1
        b["correct"] += int(r["correct"])
    per_cat = {k: {"total": v["total"], "correct": v["correct"], "accuracy": round(v["correct"] / v["total"], 4)}
               for k, v in sorted(by.items())}
    total = len(results)
    correct = sum(r["correct"] for r in results)
    return {"overall": {"total": total, "correct": correct, "accuracy": round(correct / total, 4) if total else 0.0},
            "by_category": per_cat}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system", choices=["ours", "rag"], required=True)
    ap.add_argument("--planner", choices=["rule", "rule_decomp", "hybrid", "llm"], default="rule_decomp")
    ap.add_argument("--model", default="gpt-5.4-nano")
    ap.add_argument("--rag-llm", choices=["on", "off"], default="on")
    ap.add_argument("--rag-mode", choices=["naive", "strong"], default="naive")
    ap.add_argument("--reflect", choices=["off", "on"], default="off",
                    help="trace-aware reflector on our system (faithfulness check + bounded repair)")
    ap.add_argument("--top-k", type=int, default=0, help="0 = mode default (naive 10, strong 40)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--in", dest="in_path", default=str(COMPARE_SET))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "qa" / "eval" / "compare"))
    args = ap.parse_args()

    rows = read_jsonl(Path(args.in_path))
    if args.limit:
        rows = rows[: args.limit]
    results = run_ours(rows, args) if args.system == "ours" else run_rag(rows, args)
    label = "ours" if args.system == "ours" else f"rag_{args.rag_mode}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"compare_{label}.results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n", encoding="utf-8")
    summary = summarize(results)
    summary["label"] = label
    (out_dir / f"compare_{label}.summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[{label}] overall {summary['overall']['correct']}/{summary['overall']['total']} "
          f"({summary['overall']['accuracy']:.0%})")
    for cat, s in summary["by_category"].items():
        print(f"  {cat:24s} {s['correct']:2d}/{s['total']:2d} ({s['accuracy']:.0%})")


if __name__ == "__main__":
    main()
