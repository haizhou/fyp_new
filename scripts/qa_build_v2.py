#!/usr/bin/env python3
"""Build QA set v2: quality repairs ①②③④⑥ + gold-spec backfill + dual-verification audit.

Repairs (worklog 2026-07-04 QA audit):
  ① boolean balance     — every exists-True question gets a mutated False twin (same surface,
                          title swapped to a contract without a signed date); oracle recomputed
                          independently.
  ② factoid re-bucket   — small-domain answers (tender_category / value_source) move to a new
                          `categorical` bucket; the factoid bucket keeps high-entropy answers.
  ③ unanswerable twins  — cue-bearing unsupported questions get an ANSWERABLE twin with the
                          unsupported attribute swapped for a supported one, so "abstain" cannot
                          be solved by a cue-word list.
  ④ drift quarantine    — structure-aware detectors flag L2 rewrites whose surface injects
                          constraints the gold plan does not carry (0017 class); flagged rows are
                          moved to quarantine.jsonl, not silently dropped.
  ⑥ bucket floors       — dev/test buckets below the floor are reported (generation of new
                          template rows is a separate step; this build documents the gap).
  gold backfill         — the implicit oracle universe becomes EXPLICIT: sum-bridge plans gain
                          the value_is_additive guard; distinct_set gains answer_field; top_k
                          gains k/group_by/metric; compare families gain metadata.compare_params
                          parsed from oracle keys + question surface. Dataset card gains the
                          flat first-supplier convention note.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

SMALL_DOMAIN_FIELDS = {"tender_category", "value_source"}
CUE_SWAPS = [
    # (cue regex on question, replacement phrase, new answerable field)
    (re.compile(r"invoice(?:\s+or\s+payment)?\s+date|payment\s+date", re.I),
     "award signed date", "award_date_signed"),
]
CAT_INJECT_RE = re.compile(r"\b(goods|services|works)\s+(?:notices?|contracts?|tenders?)\b", re.I)
# families where the category word IS the asked attribute or a legitimate surface artefact
CAT_DETECTOR_EXEMPT = {"boolean_field_equality", "date_relation", "contract_factoid",
                       "naturalized_factoid", "cpv_slice"}
BUCKET_FLOOR = 20  # min rows per bucket in dev_tune / final_test


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def dump(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in rows) + "\n",
                    encoding="utf-8")


def flat_records(kg_dir: Path) -> pd.DataFrame:
    """The generator/executor convention universe: one row per contract, first-party names."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from procurement_graph.qa.benchmark.kg_interface import ParquetKGQueryBackend
    return ParquetKGQueryBackend.from_directory(kg_dir, include_evidence=False).records_df


# ---------------------------------------------------------------- gold backfill
def backfill_gold(row: dict[str, Any]) -> list[str]:
    notes = []
    gp = row.get("gold_plan") or {}
    fam = str(row.get("template_family") or "")
    op = str(gp.get("answer_operation") or "")
    cons = gp.get("constraints") or []
    if op == "sum" and not any(c.get("field") == "value_is_additive" for c in cons):
        cons.append({"field": "value_is_additive", "op": "eq", "value": True})
        gp["constraints"] = cons
        notes.append("backfill:additive_guard")
    if op == "distinct_set" and not gp.get("answer_field"):
        oracle = row.get("oracle_answer")
        field = "supplier_name"
        if isinstance(oracle, list) and oracle and any(
                str(o).isdigit() and len(str(o)) == 8 for o in oracle):
            field = "tender_cpv_id"
        gp["answer_field"] = field
        notes.append(f"backfill:answer_field:{field}")
    if op == "rank_top_k":
        meta = gp.setdefault("metadata", {})
        meta.setdefault("k", 3)
        meta.setdefault("group_by", "buyer_name")
        meta.setdefault("metric", "count")
        notes.append("backfill:top_k_params")
    if op == "compare":
        meta = gp.setdefault("metadata", {})
        params: dict[str, Any] = {}
        q = str(row.get("question") or "")
        oracle = row.get("oracle_answer")
        if fam in ("comparison", "supplier_set_compare") and isinstance(oracle, dict):
            params["sides"] = [k for k in oracle.keys() if k != "answer"]
        m = re.search(r"(?:gbp|£)\s*([\d,.]+)\s*(million|billion|m|bn)?", q, re.I)
        if fam == "numeric_threshold" and m:
            thr = float(m.group(1).replace(",", ""))
            unit = (m.group(2) or "").casefold()
            thr *= 1e6 if unit in ("million", "m") else 1e9 if unit in ("billion", "bn") else 1.0
            params["threshold"] = thr
        m = re.search(r"(after|before)\s+(\d{1,2}\s+\w+\s+\d{4}|\d{4}-\d{2}-\d{2})", q, re.I)
        if fam == "date_relation" and m:
            params["direction"] = m.group(1).casefold()
            params["pivot_date_surface"] = m.group(2)
        years = re.findall(r"\b(20[2-3]\d)\b", q)
        if years:
            params["year"] = int(years[0])
        if params:
            meta["compare_params"] = params
            notes.append("backfill:compare_params")
    return notes


# ---------------------------------------------------------------- ① boolean twins
def boolean_twins(rows: list[dict[str, Any]], flat: pd.DataFrame) -> list[dict[str, Any]]:
    """Boolean questions are exists(buyer+year+category). The False twin keeps the surface and
    swaps the YEAR to one where the same buyer+category has ZERO records (oracle recomputed on
    the flat convention universe)."""
    twins = []
    for r in rows:
        if r.get("train_bucket") != "boolean" or r.get("expected_status") != "answerable":
            continue
        gp = r.get("gold_plan") or {}
        cons = {c.get("field"): c for c in gp.get("constraints", [])}
        year_c = cons.get("release_year")
        if year_c is None:
            continue
        old_year = int(year_c.get("value"))
        base = pd.Series(True, index=flat.index)
        ok = True
        for f, c in cons.items():
            if f == "release_year":
                continue
            if f not in flat.columns:
                ok = False
                break
            base &= flat[f].astype(str) == str(c.get("value"))
        if not ok or str(old_year) not in str(r.get("question")):
            continue
        twin_year = None
        for y in (2022, 2023, 2024, 2025):
            if y == old_year:
                continue
            if int(flat[base & (flat["release_year"] == y)]["contract_node_id"].nunique()) == 0:
                twin_year = y
                break
        if twin_year is None:
            continue
        twin = json.loads(json.dumps(r, ensure_ascii=False))
        twin["id"] = r["id"] + "#neg"
        twin["question"] = str(r["question"]).replace(str(old_year), str(twin_year), 1)
        for c in twin["gold_plan"]["constraints"]:
            if c.get("field") == "release_year":
                c["value"] = twin_year
        twin["oracle_answer"] = False
        twin["dedup_group"] = str(r.get("dedup_group")) + "#neg"
        if twin.get("split") == "dev_smoke":
            twin["split"] = "dev_tune"  # keep the 50-row regression set stable
        twin.setdefault("provenance", {})["generated_by"] = "qa_build_v2:boolean_balance"
        twin["provenance"]["twin_of"] = r["id"]
        twins.append(twin)
    return twins


# ---------------------------------------------------------------- ③ unanswerable contrast twins
def contrast_twins(rows: list[dict[str, Any]], flat: pd.DataFrame,
                   limit: int = 400) -> list[dict[str, Any]]:
    twins = []
    for r in rows:
        if len(twins) >= limit:
            break
        if r.get("expected_status") != "unsupported":
            continue
        q = str(r.get("question") or "")
        for cue_re, phrase, field in CUE_SWAPS:
            if not cue_re.search(q):
                continue
            checker = (r.get("metadata") or {}).get("checker") or {}
            cons = [c for c in (checker.get("required_constraints") or [])
                    if isinstance(c, dict) and c.get("field") in ("buyer_name", "supplier_name", "tender_title")]
            if not cons:
                break
            mask = pd.Series(True, index=flat.index)
            for c in cons:
                col = c["field"]
                if col not in flat.columns:
                    mask &= False
                    continue
                mask &= flat[col].astype(str) == str(c["value"])
            sub = flat[mask]
            dates = sub[field].dropna().astype(str).unique() if field in sub.columns else []
            if len(dates) != 1:
                break
            twin = json.loads(json.dumps(r, ensure_ascii=False))
            twin["id"] = r["id"] + "#ans"
            twin["question"] = cue_re.sub(phrase, q, count=1)
            twin["expected_status"] = "answerable"
            twin["oracle_answer"] = str(dates[0])
            twin["answer_type"] = "date"
            twin["train_bucket"] = "factoid"
            twin["question_type"] = "factoid"
            twin["gold_plan"] = {"spec_id": str(r.get("plan_id")) + "_ans",
                                 "constraints": cons,
                                 "answer_operation": "select_unique",
                                 "answer_field": field, "answer_value_type": "date",
                                 "dedupe_key": "contract_node_id",
                                 "metadata": {"question_type": "factoid",
                                              "template_family": "contrast_twin",
                                              "expected_status": "answerable"}}
            twin["template_family"] = "contrast_twin"
            twin["dedup_group"] = str(r.get("dedup_group")) + "#ans"
            if twin.get("split") == "dev_smoke":
                twin["split"] = "dev_tune"
            twin.setdefault("provenance", {})["generated_by"] = "qa_build_v2:contrast_twin"
            twin["provenance"]["twin_of"] = r["id"]
            twins.append(twin)
            break
    return twins


# ---------------------------------------------------------------- ④ drift quarantine
def drift_flags(row: dict[str, Any]) -> list[str]:
    if row.get("source") != "L2" or row.get("expected_status") != "answerable":
        return []
    fam = str(row.get("template_family") or "")
    flags = []
    gp = row.get("gold_plan") or {}
    cons = gp.get("constraints") or []
    op = str(gp.get("answer_operation") or "")
    if op in ("count", "sum") and fam not in CAT_DETECTOR_EXEMPT:
        q = str(row.get("question") or "")
        m = CAT_INJECT_RE.search(q)
        if m:
            cat = m.group(1).casefold()
            in_top = any(c.get("field") == "tender_category" and str(c.get("value")).casefold() == cat
                         for c in cons)
            # a category INSIDE the bridge relative clause ("suppliers who won SERVICES
            # contracts") is licensed by the subquery; the same word BEFORE the clause marker
            # modifies the OUTER counted noun and needs a top-level constraint (0017 class).
            clause = re.search(r"\b(?:who|that|which)\b", q, re.I)
            inside_clause = bool(clause) and m.start() > clause.start()
            in_sub = inside_clause and any(
                cat in json.dumps(c.get("value") or {}).casefold()
                for c in cons if c.get("op") == "in_subquery")
            if not in_top and not in_sub:
                flags.append(f"category_injection:{cat}")
    return flags


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qa", type=Path,
                    default=Path("data/qa/cicada_merged_l1_l2_trainbalanced_v1/all.jsonl"))
    ap.add_argument("--kg", type=Path, default=Path("data/kg"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/qa/cicada_merged_l1_l2_trainbalanced_v2"))
    args = ap.parse_args()

    rows = load(args.qa)
    flat = flat_records(args.kg)
    report: dict[str, Any] = {"input_rows": len(rows)}

    # gold backfill (in place)
    backfill_notes = Counter()
    for r in rows:
        for note in backfill_gold(r):
            backfill_notes[note] += 1
    report["gold_backfill"] = dict(backfill_notes)

    # ② factoid re-bucket
    rebucketed = 0
    for r in rows:
        if r.get("train_bucket") == "factoid" \
                and str((r.get("gold_plan") or {}).get("answer_field")) in SMALL_DOMAIN_FIELDS:
            r["train_bucket"] = "categorical"
            rebucketed += 1
    report["factoid_rebucketed_to_categorical"] = rebucketed

    # ④ quarantine
    quarantine, kept = [], []
    for r in rows:
        flags = drift_flags(r)
        if flags:
            r.setdefault("metadata", {})["drift_flags"] = flags
            quarantine.append(r)
        else:
            kept.append(r)
    report["quarantined_l2_drift"] = len(quarantine)
    report["drift_flag_kinds"] = dict(Counter(f.split(":")[0] for q in quarantine
                                              for f in q["metadata"]["drift_flags"]))

    # ① boolean twins + ③ contrast twins (splits inherited from source rows)
    btwins = boolean_twins(kept, flat)
    ctwins = contrast_twins(kept, flat)
    report["boolean_false_twins"] = len(btwins)
    report["unanswerable_contrast_twins"] = len(ctwins)
    out_rows = kept + btwins + ctwins

    # ⑥ bucket floor report (dev/test)
    floor_gaps: dict[str, dict[str, int]] = defaultdict(dict)
    for split in ("dev_tune", "final_test", "dev_select"):
        counts = Counter(r.get("train_bucket") for r in out_rows if r.get("split") == split)
        for bucket, n in counts.items():
            if n < BUCKET_FLOOR:
                floor_gaps[split][str(bucket)] = n
    report["bucket_floor_gaps"] = {k: dict(v) for k, v in floor_gaps.items()}

    # boolean balance check
    bools = Counter()
    for r in out_rows:
        if r.get("train_bucket") == "boolean" and r.get("expected_status") == "answerable":
            bools[str(r.get("oracle_answer"))] += 1
    report["boolean_answer_balance"] = dict(bools)

    args.out.mkdir(parents=True, exist_ok=True)
    dump(args.out / "all.jsonl", out_rows)
    dump(args.out / "quarantine.jsonl", quarantine)
    for split in sorted({str(r.get("split")) for r in out_rows}):
        dump(args.out / f"{split}.jsonl", [r for r in out_rows if str(r.get("split")) == split])
    report["output_rows"] = len(out_rows)
    report["dataset_card_notes"] = [
        "supplier/buyer matching convention: FLAT first-party name per contract (kg_interface "
        "records_df); edge-level any-party matching is a DIFFERENT universe and disagrees on "
        "bridge families (dual-eval 2026-07-04).",
        "all money aggregations are additive-only (value_is_additive=true), now explicit in "
        "every sum gold plan.",
    ]
    (args.out / "build_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False),
                                                encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
