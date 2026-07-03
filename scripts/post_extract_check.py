"""
Post-extract sanity check.

Reads all pipeline outputs and produces a structured PASS/FAIL report.
Run from the project root (fyp_new/).
"""

import json
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
EXTRACTED = ROOT / "data" / "extracted"
ENTITIES  = ROOT / "data" / "entities"

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

checks = []   # list of (section, label, status, detail)


def record(section, label, status, detail=""):
    checks.append((section, label, status, detail))
    tag = {"PASS": "[PASS]", "FAIL": "[FAIL]", "WARN": "[WARN]"}[status]
    line = f"  {tag} {label}"
    if detail:
        line += f"  |  {detail}"
    print(line)


def section(title):
    print()
    print("─" * 70)
    print(f"  {title}")
    print("─" * 70)


# ─── 1. File existence + readability ────────────────────────────────────────

section("1. FILE EXISTENCE & READABILITY")

extracted_tables = {
    "tender_core":    EXTRACTED / "tender_core.parquet",
    "lots":           EXTRACTED / "lots.parquet",
    "award_criteria": EXTRACTED / "award_criteria.parquet",
    "awards":         EXTRACTED / "awards.parquet",
    "bid_stats":      EXTRACTED / "bid_stats.parquet",
    "text_evidence":  EXTRACTED / "text_evidence.parquet",
    "documents":      EXTRACTED / "documents.parquet",
}

entity_files = {
    "canonical_orgs": ENTITIES / "canonical_orgs.parquet",
    "alias_map":      ENTITIES / "alias_map.parquet",
    "er_audit":       ENTITIES / "er_audit.csv",
}

dfs = {}

for name, path in {**extracted_tables, **entity_files}.items():
    if not path.exists():
        record("files", name, FAIL, f"file not found: {path}")
        continue
    try:
        if path.suffix == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_parquet(path)
        dfs[name] = df
        record("files", name, PASS,
               f"{len(df):,} rows × {len(df.columns)} cols  ({path.stat().st_size // 1024:,} KB)")
    except Exception as e:
        record("files", name, FAIL, f"read error: {e}")


# ─── 2. Row counts + column inventory ────────────────────────────────────────

section("2. ROW COUNTS & KEY COLUMNS")

expected_key_cols = {
    "tender_core":    ["ocid"],
    "lots":           ["ocid", "lot_id"],
    "award_criteria": ["ocid", "lot_id", "criterion_index"],
    "awards":         ["ocid", "award_id", "award_value_amount", "award_value_currency",
                       "award_value_best_amount", "award_value_best_currency",
                       "award_date_signed", "supplier_raw_ids"],
    "bid_stats":      ["ocid", "measure", "stat_value"],
    "text_evidence":  ["ocid", "field_path", "text"],
    "documents":      ["ocid", "source", "document_type", "url"],
    "canonical_orgs": ["canonical_id", "canonical_name", "er_status"],
    "alias_map":      ["raw_id", "canonical_id", "alias_source"],
    "er_audit":       ["canonical_id", "canonical_name", "er_status"],
}

for name, key_cols in expected_key_cols.items():
    df = dfs.get(name)
    if df is None:
        continue
    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        record("columns", f"{name} key columns", FAIL, f"missing: {missing}")
    else:
        record("columns", f"{name} key columns", PASS,
               f"all present  ({', '.join(key_cols)})")


# ─── 3. Null rates on key ID columns ─────────────────────────────────────────

section("3. NULL RATES ON KEY ID COLUMNS")

null_checks = [
    ("tender_core",    "ocid"),
    ("lots",           "ocid"),
    ("award_criteria", "ocid"),
    ("awards",         "ocid"),
    ("awards",         "award_id"),
    ("bid_stats",      "ocid"),
    ("text_evidence",  "ocid"),
    ("documents",      "ocid"),
    ("canonical_orgs", "canonical_id"),
    ("alias_map",      "raw_id"),
    ("alias_map",      "canonical_id"),
]

for name, col in null_checks:
    df = dfs.get(name)
    if df is None or col not in df.columns:
        continue
    null_rate = df[col].isna().mean()
    if null_rate > 0.01:
        record("nulls", f"{name}.{col}", FAIL,
               f"null rate = {null_rate:.2%}  ({df[col].isna().sum():,} / {len(df):,})")
    elif null_rate > 0:
        record("nulls", f"{name}.{col}", WARN,
               f"null rate = {null_rate:.2%}  ({df[col].isna().sum():,} rows)")
    else:
        record("nulls", f"{name}.{col}", PASS, "0 nulls")


# ─── 4. Duplicate primary keys ────────────────────────────────────────────────

section("4. DUPLICATE PRIMARY KEYS")

pk_checks = [
    ("tender_core",    ["ocid"]),
    ("canonical_orgs", ["canonical_id"]),
    ("alias_map",      ["raw_id"]),
]

for name, pk_cols in pk_checks:
    df = dfs.get(name)
    if df is None:
        continue
    present = [c for c in pk_cols if c in df.columns]
    if len(present) < len(pk_cols):
        continue
    dup_count = df.duplicated(subset=present).sum()
    if dup_count > 0:
        record("duplicates", f"{name} PK ({'+'.join(pk_cols)})", FAIL,
               f"{dup_count:,} duplicate rows")
    else:
        record("duplicates", f"{name} PK ({'+'.join(pk_cols)})", PASS, "no duplicates")

# awards: (ocid, award_id) may legitimately repeat across OCID if award_id is local
aw = dfs.get("awards")
if aw is not None and "ocid" in aw.columns and "award_id" in aw.columns:
    dup = aw.duplicated(subset=["ocid", "award_id"]).sum()
    if dup > 0:
        record("duplicates", "awards PK (ocid+award_id)", WARN,
               f"{dup:,} duplicate (ocid, award_id) pairs")
    else:
        record("duplicates", "awards PK (ocid+award_id)", PASS, "no duplicates")


# ─── 5. Awards deep-dive ──────────────────────────────────────────────────────

section("5. AWARDS — VALUE, DATE, SUPPLIER LINKAGE")

aw = dfs.get("awards")
am = dfs.get("alias_map")
co = dfs.get("canonical_orgs")
interim = None
try:
    interim = pd.read_parquet(ROOT / "data" / "interim" / "releases.parquet")
except Exception:
    pass

if aw is not None:
    # value fields
    has_value   = "award_value_amount" in aw.columns
    has_best_value = "award_value_best_amount" in aw.columns
    has_gross   = "award_value_gross" in aw.columns
    has_currency = "award_value_currency" in aw.columns
    has_best_currency = "award_value_best_currency" in aw.columns
    record("awards", "award_value_amount present",  PASS if has_value else FAIL)
    record("awards", "award_value_best_amount present", PASS if has_best_value else FAIL)
    record("awards", "award_value_gross present",   PASS if has_gross else FAIL)
    record("awards", "award_value_currency present", PASS if has_currency else FAIL)
    record("awards", "award_value_best_currency present", PASS if has_best_currency else FAIL)

    # date fields
    has_period_start = "award_period_start" in aw.columns
    has_period_end   = "award_period_end" in aw.columns
    record("awards", "award_period_start present", PASS if has_period_start else FAIL)
    record("awards", "award_period_end present",   PASS if has_period_end else FAIL)
    has_signed = "award_date_signed" in aw.columns
    record("awards", "award_date_signed present", PASS if has_signed else FAIL)
    if has_signed:
        signed_non_null = aw["award_date_signed"].replace("", pd.NA).notna().sum()
        signed_pct = signed_non_null / len(aw) * 100 if len(aw) else 0
        signed_status = PASS if signed_pct >= 60 else WARN if signed_pct >= 20 else FAIL
        record("awards", "award_date_signed coverage", signed_status,
               f"{signed_non_null:,}/{len(aw):,} non-empty  ({signed_pct:.1f}%)")

    # non-null value coverage — award value is frequently omitted in OCDS when spread
    # across lots, or when the contract is a zero-value framework. 5% is a fail floor.
    if has_value:
        non_null = aw["award_value_amount"].notna().sum()
        pct = non_null / len(aw) * 100
        record("awards", "award_value_amount raw coverage", PASS,
               f"{non_null:,}/{len(aw):,} non-null  ({pct:.1f}%; raw award.value only)")

    if has_best_value:
        best_non_null = pd.to_numeric(aw["award_value_best_amount"], errors="coerce").notna().sum()
        best_pct = best_non_null / len(aw) * 100 if len(aw) else 0
        best_status = PASS if best_pct >= 60 else WARN if best_pct >= 20 else FAIL
        source_counts = aw.get("award_value_source", pd.Series(dtype=str)).replace("", pd.NA).value_counts().to_dict()
        record("awards", "award_value_best_amount coverage", best_status,
               f"{best_non_null:,}/{len(aw):,} non-null  ({best_pct:.1f}%)  sources={source_counts}")

    # supplier_raw_ids → alias_map join
    if "supplier_raw_ids" in aw.columns and am is not None and co is not None:
        # parse JSON lists, flatten
        import ast
        sample = aw["supplier_raw_ids"].dropna().head(5000)
        raw_ids = set()
        for v in sample:
            try:
                ids = json.loads(v) if isinstance(v, str) else []
                raw_ids.update(ids)
            except Exception:
                pass
        if raw_ids:
            alias_raw = set(am["raw_id"].values)
            matched = raw_ids & alias_raw
            match_rate = len(matched) / len(raw_ids) * 100
            status = PASS if match_rate >= 60 else WARN
            record("awards", "supplier_raw_ids → alias_map coverage", status,
                   f"{len(matched):,}/{len(raw_ids):,} unique raw_ids matched  ({match_rate:.1f}%)")
        else:
            record("awards", "supplier_raw_ids → alias_map coverage", WARN,
                   "no supplier_raw_ids found in sample")

    # 2025 stats — join to interim for date (date column may be datetime64 or string)
    if interim is not None and "ocid" in interim.columns and "date" in interim.columns:
        date_col = interim["date"]
        if hasattr(date_col, "dt"):
            mask_2025 = date_col.dt.year == 2025
        else:
            mask_2025 = date_col.astype(str).str.startswith("2025", na=False)
        interim_2025 = interim[mask_2025][["ocid"]]
        awards_2025 = aw[aw["ocid"].isin(interim_2025["ocid"])]
        n_2025 = len(awards_2025)
        if has_value:
            if "award_value_best_amount" in awards_2025.columns and "award_value_source" in awards_2025.columns:
                additive = awards_2025[awards_2025["award_value_source"].isin(["award", "contract"])]
                value_col = "award_value_best_amount"
                total_2025 = pd.to_numeric(additive[value_col], errors="coerce").sum()
                value_note = "award/contract sources only; tender fallback excluded from sum"
            else:
                value_col = "award_value_amount"
                total_2025 = pd.to_numeric(awards_2025[value_col], errors="coerce").sum()
                value_note = "raw award.value"
            record("awards", "2025 award records", PASS,
                   f"{n_2025:,} awards  |  total {value_col} = GBP {total_2025:,.0f} ({value_note})")
        else:
            record("awards", "2025 award records", PASS, f"{n_2025:,} awards")
    else:
        # fallback: filter by ocid year prefix if known
        aw_2025 = aw[aw["ocid"].str.contains("2025", na=False)]
        record("awards", "2025 award records (fallback filter)", WARN,
               f"{len(aw_2025):,} awards with '2025' in ocid (not date-filtered)")


# ─── 6. text_evidence — traceability + content check ─────────────────────────

section("6. TEXT_EVIDENCE — TRACEABILITY & CONTENT")

te = dfs.get("text_evidence")
tc = dfs.get("tender_core")

if te is not None:
    # join key to tender_core
    if tc is not None and "ocid" in te.columns and "ocid" in tc.columns:
        te_ocids = set(te["ocid"].unique())
        tc_ocids = set(tc["ocid"].unique())
        coverage = len(te_ocids & tc_ocids) / len(tc_ocids) * 100
        record("text_evidence", "ocid joins to tender_core", PASS if coverage >= 90 else WARN,
               f"{len(te_ocids & tc_ocids):,}/{len(tc_ocids):,} OCIDs have evidence  ({coverage:.1f}%)")

    # field_path variety
    if "field_path" in te.columns:
        paths = te["field_path"].value_counts()
        record("text_evidence", "field_path variety", PASS,
               f"{len(paths)} distinct paths  (top: {list(paths.index[:3])})")

    # lot_id traceability
    if "lot_id" in te.columns:
        has_lot = te["lot_id"].notna().sum()
        record("text_evidence", "lot_id present on rows",
               PASS, f"{has_lot:,}/{len(te):,} rows have lot_id  ({has_lot/len(te)*100:.1f}%)")

    # non-empty text
    if "text" in te.columns:
        empty = (te["text"].isna() | (te["text"].str.strip() == "")).sum()
        if empty > 0:
            record("text_evidence", "empty text strings", WARN,
                   f"{empty:,} rows have empty/null text  ({empty/len(te)*100:.2f}%)")
        else:
            record("text_evidence", "empty text strings", PASS, "none found")

        # sample 5
        sample_rows = te[te["text"].notna() & (te["text"].str.strip() != "")].sample(
            min(5, len(te)), random_state=42
        )
        print()
        print("  Random sample of 5 text_evidence rows:")
        for i, (_, row) in enumerate(sample_rows.iterrows(), 1):
            snippet = str(row["text"])[:120].replace("\n", " ")
            # Encode to ASCII with replacement so GBK console doesn't crash on £ etc.
            safe = snippet.encode("ascii", errors="replace").decode("ascii")
            print(f"    [{i}] ocid={row['ocid']}  path={row.get('field_path','?')}")
            print(f"        {safe!r}")


# ─── 7. documents — traceability + type variety ──────────────────────────────

section("7. DOCUMENTS — TRACEABILITY & CONTENT")

docs = dfs.get("documents")

if docs is not None:
    # ocid join
    if tc is not None and "ocid" in docs.columns and "ocid" in tc.columns:
        doc_ocids = set(docs["ocid"].unique())
        tc_ocids  = set(tc["ocid"].unique())
        cov = len(doc_ocids & tc_ocids) / len(tc_ocids) * 100
        record("documents", "ocid joins to tender_core", PASS if cov >= 80 else WARN,
               f"{len(doc_ocids & tc_ocids):,}/{len(tc_ocids):,} OCIDs have docs  ({cov:.1f}%)")

    # doc_id — present but expected sparse
    if "doc_id" in docs.columns:
        non_null_doc_id = docs["doc_id"].notna().sum()
        record("documents", "doc_id field present", PASS,
               f"{non_null_doc_id:,}/{len(docs):,} rows have doc_id  ({non_null_doc_id/len(docs)*100:.1f}%)")

    # source breakdown
    if "source" in docs.columns:
        src_counts = docs["source"].value_counts()
        record("documents", "source variety", PASS,
               "  ".join(f"{s}:{c:,}" for s, c in src_counts.items()))

    # document_type breakdown
    if "document_type" in docs.columns:
        dt_counts = docs["document_type"].value_counts().head(8)
        record("documents", "document_type top 8",
               PASS, "  ".join(f"{t}:{c:,}" for t, c in dt_counts.items()))

    # contactEmail rows are distinguishable
    if "document_type" in docs.columns:
        n_email = (docs["document_type"] == "contactEmail").sum()
        n_real  = (docs["document_type"] != "contactEmail").sum()
        record("documents", "contactEmail vs real docs", PASS,
               f"contactEmail: {n_email:,}  |  real docs: {n_real:,}")

    # url non-null for non-contactEmail rows
    if "url" in docs.columns and "document_type" in docs.columns:
        real = docs[docs["document_type"] != "contactEmail"]
        url_null = real["url"].isna().sum()
        url_empty = (real["url"].fillna("") == "").sum()
        pct_missing = url_empty / len(real) * 100
        status = PASS if pct_missing < 20 else WARN
        record("documents", "url non-empty (real docs)", status,
               f"{url_empty:,}/{len(real):,} empty/null  ({pct_missing:.1f}%)")


# ─── 8. Entity resolution summary ────────────────────────────────────────────

section("8. ENTITY RESOLUTION SUMMARY")

co = dfs.get("canonical_orgs")
am = dfs.get("alias_map")

if co is not None:
    status_counts = co["er_status"].value_counts()
    total = len(co)
    for s, c in status_counts.items():
        record("entities", f"er_status={s}", PASS, f"{c:,}  ({c/total*100:.1f}%)")

    # GB-FTS canonical_id is allowed ONLY for singletons (provisional, no merge possible).
    # It must never appear as canonical_id with a non-singleton er_status.
    fts_mask = co["canonical_id"].str.startswith("GB-FTS-", na=False)
    fts_non_singleton = co[fts_mask & (co["er_status"] != "singleton")]
    fts_singleton_count = co[fts_mask & (co["er_status"] == "singleton")].shape[0]
    if len(fts_non_singleton) > 0:
        record("entities", "GB-FTS as canonical_id (non-singleton)", FAIL,
               f"{len(fts_non_singleton)} GB-FTS IDs with er_status != singleton (must be 0)")
    else:
        record("entities", "GB-FTS as canonical_id (non-singleton)", PASS,
               f"0 non-singleton — {fts_singleton_count:,} GB-FTS singletons (provisional, expected)")

    # _gov_source column
    if "_gov_source" in co.columns:
        gov_rows = co["_gov_source"].notna().sum()
        record("entities", "_gov_source column present", PASS,
               f"{gov_rows:,} rows have _gov_source")

if am is not None:
    src_counts = am["alias_source"].value_counts()
    for s, c in src_counts.items():
        record("entities", f"alias_source={s}", PASS, f"{c:,}")
    # FTS stored as alias not canonical
    fts_alias = am[
        am["raw_id"].str.startswith("GB-FTS-", na=False) &
        (am["raw_id"] != am["canonical_id"])
    ]["raw_id"].nunique()
    record("entities", "GB-FTS IDs stored as aliases (not canonical)", PASS,
           f"{fts_alias:,}")


# ─── FINAL SUMMARY ────────────────────────────────────────────────────────────

print()
print("=" * 70)
print("  FINAL SUMMARY")
print("=" * 70)

n_pass = sum(1 for _, _, s, _ in checks if s == PASS)
n_warn = sum(1 for _, _, s, _ in checks if s == WARN)
n_fail = sum(1 for _, _, s, _ in checks if s == FAIL)
total_checks = len(checks)

print(f"  Total checks : {total_checks}")
print(f"  PASS         : {n_pass}")
print(f"  WARN         : {n_warn}")
print(f"  FAIL         : {n_fail}")
print()

if n_fail == 0 and n_warn == 0:
    verdict = "ALL PASS — pipeline outputs are clean."
elif n_fail == 0:
    verdict = f"PASS WITH WARNINGS — {n_warn} warning(s), no failures."
else:
    verdict = f"FAIL — {n_fail} failure(s) must be investigated."

print(f"  VERDICT: {verdict}")

if n_warn > 0:
    print()
    print("  Warnings:")
    for sec, label, status, detail in checks:
        if status == WARN:
            print(f"    [WARN] {label}  |  {detail}")

if n_fail > 0:
    print()
    print("  Failures:")
    for sec, label, status, detail in checks:
        if status == FAIL:
            print(f"    [FAIL] {label}  |  {detail}")

print("=" * 70)

sys.exit(0 if n_fail == 0 else 1)
