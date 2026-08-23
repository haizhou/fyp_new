#!/usr/bin/env python3
"""Figure 1 evidence bundle: one multi-hop question, every system, one JSON.

Runs the panels of the introduction figure on a single bridge question and writes
everything the artwork needs (retrieved records, emitted programs, intermediate set
sizes, traces, verdicts) to data/qa/figures/fig1/fig1_trace.json.

Panels
  gt                 deterministic hop-by-hop denotation computed directly over U_G
  rag_k10 / rag_k40  TF-IDF retrieval over the record universe + a local LLM reader
  oneshot_base       untuned Qwen3-8B emits one algebra tree; validate + execute
  oneshot_tuned      compose_sft_v3 adapter emits one algebra tree; validate + execute
  pipeline_rule      full ReasoningPipeline, rule-decomposition planner
  pipeline_llm       full ReasoningPipeline, typed LLM planner on the local server

Every panel is independent: a failure in one is recorded and does not stop the rest.

usage:
  .venv/bin/python -B scripts/paper_figures/fig1_trace.py \
      --base-url http://127.0.0.1:8000/v1 --panels all
"""
from __future__ import annotations

import argparse
import dataclasses
import decimal
import importlib.util
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from procurement_graph.compose.algebra import validate_tree  # noqa: E402
from procurement_graph.compose.eval_runtime import RuntimeAlgebraEvaluator  # noqa: E402
from procurement_graph.qa.benchmark.kg_interface import ParquetKGQueryBackend  # noqa: E402

# the one-shot arm must use the *same* prompt and parser as the PACS protocol
_spec = importlib.util.spec_from_file_location("probe_eval", ROOT / "scripts/run_compose_probe_eval.py")
_probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_probe)
SYSTEM_PROMPT, _extract_json = _probe.SYSTEM_PROMPT, _probe._extract_json

# ---------------------------------------------------------------- the question
QUESTION = ("Follow LHC on behalf of the Scottish Procurement Alliance (SPA)'s lead suppliers "
            "outward: setting LHC on behalf of the Scottish Procurement Alliance (SPA) itself "
            "aside, which buyers do those same suppliers also work for?")
SOURCE_ID = "PACS::F6:L3:f6_other_buyers_via_suppliers:0030#a"   # PACS dev channel A
ANCHOR = "LHC on behalf of the Scottish Procurement Alliance (SPA)"
ORACLE = None            # filled from the graph below; a 19-member set of buyer strings
OUT = ROOT / "data/qa/figures/fig1"

_BOOST = {"buyer_name": 3, "supplier_name": 3, "tender_cpv_id": 3, "tender_cpv_description": 2}
_ABSTAIN = {"unknown", "none", "n/a", "na", "not available", "cannot determine", "", "null"}


def matches(ans, oracle) -> bool:
    """Set comparison for list answers, tolerant numeric comparison otherwise."""
    if isinstance(oracle, list):
        if not isinstance(ans, list):
            return False
        return sorted(str(x).strip() for x in ans) == sorted(str(x).strip() for x in oracle)
    if ans is None:
        return False
    try:
        return abs(float(ans) - float(oracle)) < 0.01
    except (TypeError, ValueError):
        return str(ans).strip() == str(oracle).strip()


def jsonable(v):
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if isinstance(v, decimal.Decimal):
        return float(v)
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return {f.name: jsonable(getattr(v, f.name)) for f in dataclasses.fields(v)}
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [jsonable(x) for x in v]
    return str(v)


def save(results: dict) -> Path:
    """Merge into the bundle on disk. Called after every panel so a slow or hanging
    panel can never hold the finished ones hostage."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig1_trace.json"
    merged = json.loads(path.read_text()) if path.exists() else {}
    merged.update(results)
    path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    return path


def panel(results: dict, name: str):
    """Run a panel, recording either its payload or its traceback."""
    def wrap(fn):
        print(f"\n=== panel: {name} ===", flush=True)
        try:
            results[name] = jsonable(fn())
            print(f"[{name}] ok", flush=True)
        except Exception:
            results[name] = {"error": traceback.format_exc()[-1500:]}
            print(f"[{name}] FAILED\n{traceback.format_exc()[-600:]}", flush=True)
        save(results)
        return fn
    return wrap


# ---------------------------------------------------------------- ground truth
def ground_truth(df):
    global ORACLE
    src = df[df.buyer_name == ANCHOR]
    suppliers = sorted(set(src.supplier_name.dropna()) - {""})
    tgt = df[df.supplier_name.isin(suppliers)]
    buyers_all = sorted(set(tgt.buyer_name.dropna()) - {""})
    buyers = [b for b in buyers_all if b != ANCHOR]
    if ORACLE is None:
        ORACLE = buyers
    return {
        "hop1_filter": f'buyer_name = "{ANCHOR}"',
        "hop1_source_records": int(src.contract_node_id.nunique()),
        "hop2_project": "values(supplier_name)",
        "hop2_bridge_values": len(suppliers),
        "hop2_bridge_all": suppliers,
        "hop3_filter": "supplier_name IN hop2",
        "hop3_target_records": int(tgt.contract_node_id.nunique()),
        "hop4_project": "values(buyer_name) minus the anchor",
        "buyers_before_exclusion": len(buyers_all),
        "answer_size": len(buyers),
        "answer": buyers,
        "anchor_excluded": ANCHOR in buyers_all,
        "near_variants_retained": [b for b in buyers if "LHC" in b],
        "record_universe_rows": int(len(df)),
    }


# ---------------------------------------------------------------- RAG panels
def build_index(df, strong: bool):
    from sklearn.feature_extraction.text import TfidfVectorizer
    cols = ["buyer_name", "supplier_name", "tender_category", "tender_cpv_id",
            "tender_cpv_description", "release_year", "value_amount", "tender_title",
            "award_date_signed"]
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
    return vec, vec.fit_transform(texts), plain


def run_rag(client, model, df, k: int, strong: bool):
    from sklearn.metrics.pairwise import linear_kernel
    vec, matrix, plain = build_index(df, strong)
    sims = linear_kernel(vec.transform([QUESTION]), matrix).ravel()
    idx = sims.argsort()[::-1][:k]
    ctx = [plain[i] for i in idx]
    system = ("You answer UK public-procurement questions using ONLY the retrieved contract records "
              "below. Do not use outside knowledge. If the records do not contain enough to answer, "
              "reply exactly {\"answer\": \"unknown\"}. Return strict JSON {\"answer\": <value>}.")
    user = json.dumps({"question": QUESTION, "records": ctx}, ensure_ascii=False)
    resp = client.chat.completions.create(
        model=model, temperature=0.0, max_tokens=600,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    raw = resp.choices[0].message.content or ""
    parsed = _extract_json(raw) or {}
    ans = parsed.get("answer") if isinstance(parsed, dict) else None
    if ans is not None and str(ans).strip().casefold() in _ABSTAIN:
        ans = None
    # how much of the true denotation did retrieval even see?
    retrieved_anchor = sum(1 for c in ctx if ANCHOR in c)
    return {
        "config": {"index": "strong-boosted TF-IDF" if strong else "plain TF-IDF", "top_k": k,
                   "reader_model": model, "temperature": 0.0},
        "retrieved_count": len(ctx),
        "retrieved_containing_anchor": retrieved_anchor,
        "retrieved_records": ctx,
        "reader_raw": raw[:2000],
        "answer": ans,
        "oracle": ORACLE,
        "correct": matches(ans, ORACLE),
        "cardinality_gap": {"retrieved": len(ctx), "true_denotation": ORACLE},
    }


# ---------------------------------------------------------------- LLM + KG, free-form SQL
_SQL_COLS = ["contract_node_id", "buyer_name", "supplier_name", "release_year",
             "tender_category", "tender_cpv_id", "tender_title", "value_amount",
             "value_currency", "value_is_additive", "award_date_signed"]


def run_kg_sql(client, model, df):
    """No fine-tuning and no house format: the model is given the record schema and writes
    ordinary SQL, which we execute. This is the fair 'LLM with access to the KG' baseline."""
    import sqlite3
    sub = df[[c for c in _SQL_COLS if c in df.columns]].copy()
    if "value_is_additive" in sub.columns:
        sub["value_is_additive"] = sub["value_is_additive"].astype("boolean").astype("Int64")
    con = sqlite3.connect(":memory:")
    sub.to_sql("records", con, index=False)
    schema = [f"{c} ({str(sub[c].dtype)})" for c in sub.columns]
    samples = {c: [str(v) for v in sub[c].dropna().unique()[:3]] for c in
               ("buyer_name", "supplier_name", "tender_category", "value_currency") if c in sub.columns}
    system = ("You answer questions about a UK public-procurement database by writing ONE SQLite "
              "SELECT statement over the table `records`, one row per contract-award notice. "
              "Return strict JSON {\"sql\": \"...\"}. No prose, no markdown, no explanation.")
    user = json.dumps({"question": QUESTION, "table": "records", "columns": schema,
                       "sample_values": samples,
                       "note": "value_is_additive is 1 when the amount may be summed."},
                      ensure_ascii=False)
    resp = client.chat.completions.create(
        model=model, temperature=0.0, max_tokens=700,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    raw = resp.choices[0].message.content or ""
    sql = (_extract_json(raw) or {}).get("sql")
    out = {"model": model, "protocol": "free-form SQL over the record table, no fine-tuning, "
                                       "no house output format",
           "raw": raw[:1500], "sql": sql, "n_rows": int(len(sub))}
    if not sql:
        return {**out, "outcome": "no_sql", "answer": None, "correct": False}
    try:
        rows = con.execute(sql).fetchall()
    except Exception as exc:
        return {**out, "outcome": "sql_error", "error": str(exc)[:300],
                "answer": None, "correct": False}
    if isinstance(ORACLE, list):
        ans = [r[0] for r in rows]
    else:
        ans = rows[0][0] if (len(rows) == 1 and len(rows[0]) == 1) else None
    out.update(outcome="executed", result_shape=[len(rows), len(rows[0]) if rows else 0],
               result_head=jsonable(rows[:5]), answer=jsonable(ans), oracle=ORACLE,
               correct=matches(ans, ORACLE))
    return out


# ---------------------------------------------------------------- one-shot algebra
def run_oneshot(client, model, ev):
    resp = client.chat.completions.create(
        model=model, temperature=0.0, max_tokens=1200,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": QUESTION}])
    raw = resp.choices[0].message.content or ""
    out = {"model": model, "protocol": "single decode, temperature 0, no guided JSON, no repair",
           "raw": raw[:4000]}
    payload = _extract_json(raw)
    if payload is None:
        return {**out, "outcome": "unparseable", "answer": None, "correct": False}
    if payload.get("abstain"):
        return {**out, "outcome": "abstain", "tree": payload, "answer": None, "correct": False}
    # same unwrapping rule as the PACS protocol: {"tree": {...}} or a bare node
    tree = payload.get("tree") if isinstance(payload.get("tree"), dict) else (
        payload if payload.get("node") else None)
    if tree is None:
        return {**out, "outcome": "no_tree", "payload": payload, "answer": None, "correct": False}
    out["tree"] = tree
    payload = tree
    try:
        out["static_validation"] = {"ok": True, "root_type": str(validate_tree(payload))}
    except Exception as exc:
        out["static_validation"] = {"ok": False, "error": str(exc)}
    env = ev.run(payload)                      # validates again, never raises
    answer = env.get("answer")
    out.update(outcome=env.get("status"), envelope=jsonable(env), answer=jsonable(answer),
               correct=matches(jsonable(answer), ORACLE), oracle=ORACLE)
    return out


# ---------------------------------------------------------------- full pipeline
def run_pipeline(planner_kind: str, base_url: str, model: str):
    from procurement_graph.reasoning import ReasoningPipeline
    from procurement_graph.reasoning.kg_backend import RuntimeKGBackend
    backend = RuntimeKGBackend.from_directory(ROOT / "data" / "kg")
    resolver = backend.org_resolver()
    if planner_kind == "rule":
        from procurement_graph.reasoning.planner_decomposition import DecompositionAwarePlanner
        planner = DecompositionAwarePlanner(org_resolver=resolver)
        meta = {"planner": "DecompositionAwarePlanner (deterministic, no LLM)"}
    else:
        from procurement_graph.qa.benchmark.chat import ChatClient
        from procurement_graph.reasoning.typed_planning import TypedLLMPlanner, resolve_planner_variants
        chat = ChatClient(base_url=base_url, api_key="local", temperature=0.0)
        pv, sv = resolve_planner_variants(model)
        planner = TypedLLMPlanner(client=chat, model=model, org_resolver=resolver, two_step=True,
                                  understanding_client=chat, understanding_model=model,
                                  plan_prompt_variant=pv, plan_schema_variant=sv, plan_samples=2)
        meta = {"planner": f"TypedLLMPlanner two-step on {model}"}
    pipe = ReasoningPipeline(backend=backend, planner=planner, org_resolver=resolver,
                             max_feedback_replans=1)
    trace = pipe.run(QUESTION)
    card = trace.answer_card
    answer = card.answer if card else None
    return {**meta, "trace_id": trace.trace_id, "selected_plan_id": trace.selected_plan_id,
            "plans": jsonable(trace.plans), "execution": jsonable(trace.execution),
            "evidence_verdict": jsonable(trace.evidence_verdict),
            "answer_card": jsonable(card), "metadata": jsonable(trace.metadata),
            "answer": jsonable(answer), "oracle": ORACLE,
            "correct": matches(jsonable(answer), ORACLE)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tuned-model", default="compose-v3")
    ap.add_argument("--panels", default="all")
    args = ap.parse_args()
    want = {p.strip() for p in args.panels.split(",")} if args.panels != "all" else None

    def on(name):
        return want is None or name in want

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="local", timeout=300)

    print("loading record universe ...", flush=True)
    df = ParquetKGQueryBackend.from_directory(ROOT / "data/kg", include_evidence=False).records_df
    ev = RuntimeAlgebraEvaluator(ParquetKGQueryBackend.from_directory(ROOT / "data/kg",
                                                                     include_evidence=False))

    results: dict = {
        "question": QUESTION,
        "source_id": SOURCE_ID,
        "source_note": ("wording taken from data/qa/eval/compare_set.jsonl with the template "
                        "preamble 'Looking only at the matching procurement records,' removed"),
        "oracle": ORACLE,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "kg_dir": str(ROOT / "data/kg"),
    }

    if on("gt"):
        panel(results, "gt")(lambda: ground_truth(df))
    if on("rag_k10"):
        panel(results, "rag_k10")(lambda: run_rag(client, args.base_model, df, 10, strong=False))
    if on("rag_k40"):
        panel(results, "rag_k40")(lambda: run_rag(client, args.base_model, df, 40, strong=True))
    if on("kg_sql"):
        panel(results, "kg_sql")(lambda: run_kg_sql(client, args.base_model, df))
    if on("oneshot_base"):
        panel(results, "oneshot_base")(lambda: run_oneshot(client, args.base_model, ev))
    if on("oneshot_tuned"):
        panel(results, "oneshot_tuned")(lambda: run_oneshot(client, args.tuned_model, ev))
    if on("pipeline_rule"):
        panel(results, "pipeline_rule")(lambda: run_pipeline("rule", args.base_url, args.base_model))
    if on("pipeline_llm"):
        panel(results, "pipeline_llm")(lambda: run_pipeline("llm", args.base_url, args.base_model))

    path = save(results)
    print(f"\nwrote {path}")
    print("\n--- summary ---")
    for k, v in results.items():
        if isinstance(v, dict):
            if "error" in v:
                print(f"  {k:16} FAILED")
            else:
                print(f"  {k:16} answer={v.get('answer', v.get('hop3_target_records'))} "
                      f"correct={v.get('correct', v.get('reproduces_oracle'))}")


if __name__ == "__main__":
    main()
