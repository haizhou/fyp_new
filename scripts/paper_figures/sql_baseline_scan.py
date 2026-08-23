#!/usr/bin/env python3
"""Scan a development slice with the free-form SQL baseline.

No fine-tuning and no house output format: the model is given the record schema and writes
ordinary SQLite, which we execute.  The point is to locate where that baseline breaks, so the
scan reports accuracy by question type and writes every emitted query for inspection.
"""
from __future__ import annotations
import argparse, json, random, sqlite3, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from procurement_graph.qa.benchmark.kg_interface import ParquetKGQueryBackend  # noqa: E402
import importlib.util  # noqa: E402
_s = importlib.util.spec_from_file_location("pe", ROOT / "scripts/run_compose_probe_eval.py")
_pe = importlib.util.module_from_spec(_s); _s.loader.exec_module(_pe)
_extract_json = _pe._extract_json

# expose every queryable field of the record view; withholding one silently handicaps the
# baseline and turns a schema gap into a fake capability result.
COLS = ["contract_node_id", "ocid", "award_id", "buyer_name", "supplier_name", "release_date",
        "release_year", "tender_title", "award_title", "award_status", "tender_method",
        "tender_category", "tender_cpv_id", "tender_cpv_description", "value_amount",
        "value_currency", "value_source", "value_is_additive", "award_date_signed",
        "award_period_start", "award_period_end", "tender_period_end", "contract_period_start",
        "contract_period_end", "has_award_signed_date", "has_contract_period",
        "days_release_to_award_signed", "contract_duration_days", "above_threshold"]

SYSTEM = ("You answer questions about a UK public-procurement database by writing ONE SQLite "
          "SELECT statement over the table `records`, one row per contract-award notice. "
          "If the question cannot be answered from the table, return {\"sql\": \"\"}. "
          "Return strict JSON {\"sql\": \"...\"}. No prose, no markdown.")


def norm(v):
    if v is None: return None
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)):
        return round(float(v), 2)
    s = str(v).strip()
    try: return round(float(s), 2)
    except ValueError: return s.casefold()


def extract(rows, oracle):
    """Read an answer out of a result set, interpreting it in the shape the oracle expects.
    Deliberately generous: the baseline is credited whenever a reasonable reading succeeds."""
    if isinstance(oracle, bool):
        if len(rows) == 1 and len(rows[0]) == 1:
            c = rows[0][0]
            if isinstance(c, (int, float)) and c in (0, 1):
                return bool(c)
            if isinstance(c, str) and c.casefold() in ("true", "false"):
                return c.casefold() == "true"
        return len(rows) > 0                       # existence reading
    if isinstance(oracle, list):
        return [r[0] for r in rows]                # first column as a list
    if not rows:
        return None
    return rows[0][0]                              # first cell


def match(ans, oracle):
    if oracle is None: return ans is None
    if isinstance(oracle, list):
        if not isinstance(ans, list): return False
        return sorted(map(str, map(norm, ans))) == sorted(map(str, map(norm, oracle)))
    if isinstance(oracle, dict):
        return norm(ans) == norm(oracle.get("answer"))
    a, o = norm(ans), norm(oracle)
    if isinstance(a, float) and isinstance(o, float): return abs(a - o) < 0.02
    return a == o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--in", dest="src", default="data/qa/cicada_core_v4/dev_select.jsonl")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="data/qa/figures/sql_baseline_scan.jsonl")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(ROOT / a.src)]
    rows = [r for r in rows if r.get("expected_status") == "answerable"]
    by = {}
    key = "question_type" if "question_type" in rows[0] else "distance_band"
    for r in rows: by.setdefault(r.get(key, "?"), []).append(r)
    rng = random.Random(a.seed)
    per = max(1, a.n // max(1, len(by)))
    pick = []
    for k in sorted(by):
        pick += rng.sample(by[k], min(per, len(by[k])))
    pick = pick[:a.n]
    print(f"{len(pick)} questions across {len(by)} types", flush=True)

    df = ParquetKGQueryBackend.from_directory(ROOT / "data/kg", include_evidence=False).records_df
    sub = df[[c for c in COLS if c in df.columns]].copy()
    for c in ("value_is_additive", "has_award_signed_date", "has_contract_period", "above_threshold"):
        if c in sub: sub[c] = sub[c].astype("boolean").astype("Int64")
    for c in sub.columns:
        if str(sub[c].dtype).startswith(("datetime", "period")):
            sub[c] = sub[c].astype(str)
    con = sqlite3.connect(":memory:")
    sub.to_sql("records", con, index=False)
    schema = [f"{c} ({sub[c].dtype})" for c in sub.columns]
    # low-cardinality categoricals: give the actual vocabulary, otherwise the baseline guesses
    # the casing and every filter silently misses.
    vocab = {}
    for c in sub.columns:
        try: u = sub[c].dropna().unique()
        except Exception: continue
        if 0 < len(u) <= 12 and sub[c].dtype == object:
            vocab[c] = sorted(str(x) for x in u)

    from openai import OpenAI
    client = OpenAI(base_url=a.base_url, api_key="local", timeout=180)
    out = (ROOT / a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open("w")
    ok = 0
    for i, r in enumerate(pick, 1):
        rec = {"id": r["id"], "question_type": r.get("question_type") or r.get("distance_band"),
               "answer_type": r.get("answer_type"), "question": r["question"],
               "oracle": r.get("oracle_answer")}
        try:
            resp = client.chat.completions.create(
                model=a.model, temperature=0.0, max_tokens=700,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": json.dumps(
                              {"question": r["question"], "table": "records", "columns": schema,
                               "categorical_values": vocab,
                               "note": "value_is_additive is 1 when the amount may be summed. "
                                       "String comparisons are case-sensitive; use the exact "
                                       "values listed in categorical_values."})}])
            raw = resp.choices[0].message.content or ""
            sql = (_extract_json(raw) or {}).get("sql") or ""
            rec["sql"] = sql
            if not sql:
                rec.update(outcome="no_sql", answer=None)
            else:
                res = con.execute(sql).fetchall()
                rec.update(outcome="executed", n_rows=len(res),
                           answer=extract(res, r.get("oracle_answer")))
        except Exception as exc:
            rec.update(outcome="error", error=str(exc)[:200], answer=None)
        rec["correct"] = bool(match(rec.get("answer"), rec["oracle"]))
        ok += rec["correct"]
        fh.write(json.dumps(rec, default=str) + "\n"); fh.flush()
        if i % 10 == 0: print(f"  {i}/{len(pick)}  running acc {ok/i:.1%}", flush=True)
    fh.close()
    print(f"\nSQL baseline: {ok}/{len(pick)} = {ok/len(pick):.1%}  -> {out}")


if __name__ == "__main__":
    main()
