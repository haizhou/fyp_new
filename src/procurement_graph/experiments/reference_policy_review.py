"""Review policy for reference/API entity candidates."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from procurement_graph.common.normalise import normalise_name

ROOT = Path(__file__).resolve().parents[3]
CANDIDATES_PATH = ROOT / "data" / "ablation" / "reference_entity_candidates.parquet"
DATA_OUT_DIR = ROOT / "data" / "ablation" / "reference"
REPORT_OUT_DIR = ROOT / "reports" / "ablation" / "reference"


def _json_list(value) -> list:
    if isinstance(value, list):
        return value
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return []
    try:
        parsed = json.loads(value or "[]")
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def annotate_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Annotate all-reference candidates with review features and policy action."""
    df = candidates[candidates["variant"] == "all_references"].copy()
    df["canonical_norm"] = df["canonical_name"].apply(normalise_name)
    df["reference_norm"] = df["reference_matched_name"].apply(normalise_name)
    df["exact_canonical_match"] = df["canonical_norm"].eq(df["reference_norm"])
    df["alias_match"] = df.apply(
        lambda row: row["reference_norm"]
        in {normalise_name(name) for name in _json_list(row.get("alias_names"))},
        axis=1,
    )
    df["reference_status_norm"] = df["reference_status"].fillna("").astype(str).str.lower()
    df["reference_collision_size"] = (
        df.groupby("reference_canonical_id")["canonical_id"].transform("nunique")
    )

    external_high = (
        (
            df["reference_source"].eq("govuk_orgs")
            & df["reference_confidence"].eq("high")
            & df["reference_status_norm"].isin(["live", "exempt"])
        )
        | (
            df["reference_source"].eq("nhs_ods")
            & df["reference_confidence"].eq("high")
            & df["reference_status_norm"].eq("active")
        )
    )

    df["policy_action"] = "manual_review"
    df.loc[df["reference_source"].eq("contracts_finder"), "policy_action"] = (
        "candidate_only_contracts_finder"
    )
    df.loc[df["er_status"].eq("gov_lookup"), "policy_action"] = (
        "validate_existing_manual_gov_lookup"
    )
    df.loc[
        df["reference_source"].isin(["govuk_orgs", "nhs_ods"])
        & ~external_high,
        "policy_action",
    ] = "manual_review_inactive_closed_or_medium"
    df.loc[
        external_high & ~df["exact_canonical_match"],
        "policy_action",
    ] = "manual_review_alias_or_name_mismatch"
    df.loc[
        external_high & df["exact_canonical_match"] & df["reference_collision_size"].gt(1),
        "policy_action",
    ] = "manual_review_reference_collision"
    df.loc[
        external_high
        & df["exact_canonical_match"]
        & df["reference_collision_size"].eq(1)
        & ~df["er_status"].eq("gov_lookup"),
        "policy_action",
    ] = "auto_merge_candidate_external_unique"

    return df


def _write_markdown(
    annotated: pd.DataFrame,
    action_summary: pd.DataFrame,
    collision_summary: pd.DataFrame,
) -> None:
    def md_table(df: pd.DataFrame, limit: int | None = None) -> str:
        if df.empty:
            return "_No rows._"
        display = df.head(limit) if limit else df
        display = display.fillna("")
        cols = list(display.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in display.iterrows():
            values = [str(row[col]).replace("|", "\\|") for col in cols]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    auto = annotated[annotated["policy_action"] == "auto_merge_candidate_external_unique"]
    collisions = annotated[annotated["policy_action"] == "manual_review_reference_collision"]
    mismatch = annotated[annotated["policy_action"] == "manual_review_alias_or_name_mismatch"]
    medium = annotated[annotated["policy_action"] == "manual_review_inactive_closed_or_medium"]
    cf = annotated[annotated["policy_action"] == "candidate_only_contracts_finder"]

    lines = [
        "# Reference Candidate Policy Review",
        "",
        "This review classifies the `all_references` ablation output into",
        "auto-merge candidates, manual-review groups, and candidate-only evidence.",
        "",
        "## Recommended Policy",
        "",
        "- `contracts_finder`: keep as candidate/evidence only. It is derived from Contracts Finder buyer IDs, so it is useful coverage evidence but not an independent authority.",
        "- `govuk_orgs` high confidence with `live` or `exempt` status: eligible for automatic Phase 2 merge only when the canonical display name exactly matches the reference name and the reference ID is unique in the candidate set.",
        "- `nhs_ods` high confidence with `Active` status: same rule as GOV.UK.",
        "- `govuk_orgs` closed/joining and `nhs_ods` inactive: manual review only.",
        "- Any case where one reference ID matches multiple current canonical IDs: manual review first, because this is a consolidation decision across existing entities.",
        "- Any alias-only/name-mismatch hit: manual review first, even if the source is high confidence.",
        "",
        "## Action Summary",
        "",
        md_table(action_summary),
        "",
        "## Auto-Merge Candidate Summary",
        "",
        f"Rows: {len(auto):,}",
        f"Alias coverage: {int(auto['n_aliases'].sum()) if len(auto) else 0:,}",
        "",
        md_table(
            auto[
                [
                    "canonical_id",
                    "canonical_name",
                    "er_status",
                    "n_aliases",
                    "reference_source",
                    "reference_canonical_id",
                    "reference_matched_name",
                    "reference_status",
                ]
            ],
            limit=25,
        ),
        "",
        "## Collision Review Summary",
        "",
        f"Rows: {len(collisions):,}",
        f"Reference groups: {collisions['reference_canonical_id'].nunique():,}",
        f"Alias coverage: {int(collisions['n_aliases'].sum()) if len(collisions) else 0:,}",
        "",
        md_table(collision_summary, limit=40),
        "",
        "## Alias Or Name Mismatch Review",
        "",
        f"Rows: {len(mismatch):,}",
        "",
        "These are high-confidence source hits, but the canonical display name is not an exact match to the reference display name.",
        "",
        "## Inactive, Closed, Or Medium Confidence Review",
        "",
        f"Rows: {len(medium):,}",
        "",
        "These should not auto-merge until the status and historical continuity are reviewed.",
        "",
        "## Contracts Finder Candidate-Only",
        "",
        f"Rows: {len(cf):,}",
        f"Alias coverage: {int(cf['n_aliases'].sum()) if len(cf) else 0:,}",
    ]
    (REPORT_OUT_DIR / "reference_policy_review.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run() -> None:
    """Generate policy review artifacts for reference/API candidates."""
    DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_parquet(CANDIDATES_PATH)
    annotated = annotate_candidates(candidates)

    action_summary = (
        annotated.groupby(["policy_action", "reference_source", "reference_confidence"])
        .agg(
            rows=("canonical_id", "count"),
            canonical_entities=("canonical_id", "nunique"),
            reference_ids=("reference_canonical_id", "nunique"),
            alias_coverage=("n_aliases", "sum"),
        )
        .reset_index()
        .sort_values(["policy_action", "rows"], ascending=[True, False])
    )

    collisions = annotated[annotated["policy_action"] == "manual_review_reference_collision"]
    collision_summary = (
        collisions.groupby(["reference_source", "reference_canonical_id"])
        .agg(
            current_entities=("canonical_id", "nunique"),
            alias_coverage=("n_aliases", "sum"),
            sample_names=("canonical_name", lambda values: " | ".join(list(values)[:5])),
        )
        .reset_index()
        .sort_values(["current_entities", "alias_coverage"], ascending=False)
    )

    annotated.to_parquet(DATA_OUT_DIR / "reference_policy_candidates.parquet", index=False)
    action_summary.to_csv(REPORT_OUT_DIR / "reference_policy_action_summary.csv", index=False)
    collision_summary.to_csv(REPORT_OUT_DIR / "reference_collision_groups.csv", index=False)

    auto = annotated[annotated["policy_action"] == "auto_merge_candidate_external_unique"]
    auto.to_csv(REPORT_OUT_DIR / "auto_merge_candidate_external_unique.csv", index=False)

    review_cols = [
        "policy_action",
        "canonical_id",
        "canonical_name",
        "er_status",
        "n_aliases",
        "reference_source",
        "reference_confidence",
        "reference_canonical_id",
        "reference_matched_name",
        "reference_status",
        "reference_collision_size",
        "exact_canonical_match",
        "alias_match",
    ]
    annotated[review_cols].to_csv(
        REPORT_OUT_DIR / "reference_policy_candidates_review.csv",
        index=False,
    )
    _write_markdown(annotated, action_summary, collision_summary)

    print(f"Written: {DATA_OUT_DIR / 'reference_policy_candidates.parquet'}")
    print(f"Written: {REPORT_OUT_DIR / 'reference_policy_review.md'}")
    print(f"Written: {REPORT_OUT_DIR / 'reference_policy_action_summary.csv'}")
    print(f"Written: {REPORT_OUT_DIR / 'reference_collision_groups.csv'}")
    print(f"Written: {REPORT_OUT_DIR / 'auto_merge_candidate_external_unique.csv'}")
    print(f"Written: {REPORT_OUT_DIR / 'reference_policy_candidates_review.csv'}")


def cli_main() -> None:
    run()


__all__ = ["annotate_candidates", "cli_main", "run"]

