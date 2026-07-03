"""Full-corpus entity-resolution ablation framework.

This module is read-only over official entity outputs. It builds candidate
pairs, feature tables, blocking metrics, and risk reports so matchers can be
compared without mutating `data/entities/*`.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from procurement_graph.common.normalise import is_official, normalise_name, scheme_of

ROOT = Path(__file__).resolve().parents[3]
ENTITIES_DIR = ROOT / "data" / "entities"
EXTRACTED_DIR = ROOT / "data" / "extracted"
INTERIM_DIR = ROOT / "data" / "interim"
DATA_DIR = ROOT / "data" / "ablation" / "er"
REPORT_DIR = ROOT / "reports" / "ablation" / "er"

CANDIDATE_PAIRS_PATH = DATA_DIR / "candidate_pairs.parquet"
PAIR_FEATURES_PATH = DATA_DIR / "pair_features.parquet"
ENTITY_CONTEXT_PATH = DATA_DIR / "entity_context.parquet"
RISK_TIERS_PATH = DATA_DIR / "risk_tiers.parquet"
AUTO_MERGE_CANDIDATES_PATH = DATA_DIR / "deterministic_auto_merge_candidates.parquet"
MIDDLE_CANDIDATES_PATH = DATA_DIR / "middle_candidate_pairs.parquet"
REGION_QUALITY_CANDIDATES_PATH = DATA_DIR / "region_quality_candidates.parquet"
REGION_HIERARCHY_CANDIDATES_PATH = DATA_DIR / "region_hierarchy_candidates.parquet"
BLOCKING_METRICS_PATH = REPORT_DIR / "blocking_metrics.csv"
FEATURE_ABLATION_METRICS_PATH = REPORT_DIR / "feature_ablation_metrics.csv"
RISK_TIER_METRICS_PATH = REPORT_DIR / "risk_tier_metrics.csv"
BLOCK_DIAGNOSTICS_PATH = REPORT_DIR / "blocking_strategy_diagnostics.csv"
STRATEGY_UNIQUE_CONTRIBUTION_PATH = REPORT_DIR / "blocking_strategy_unique_contribution.csv"
HIGH_RISK_PAIRS_PATH = REPORT_DIR / "high_risk_pairs.csv"
HIGH_RISK_CLUSTERS_PATH = REPORT_DIR / "high_risk_clusters.csv"
SUMMARY_PATH = REPORT_DIR / "summary.md"
LLM_QUEUE_PATH = DATA_DIR / "llm_review_queue.jsonl"
ER_PAIR_PROMPT_VERSION = "er_pair_v2"
ER_PAIR_SCHEMA_VERSION = "er_pair_decision_schema_v2"
ER_PAIR_RISK_FLAGS = [
    "official_scheme_conflict",
    "trusted_id_conflict",
    "fts_official_mixed",
    "region_conflict",
    "role_conflict",
    "short_name_or_acronym_ambiguous",
    "parent_child_possible",
    "subsidiary_possible",
    "procurement_agent_possible",
    "other_er_risk",
    "insufficient_evidence",
    "none",
]

LLM_PRIORITY_RISK_FLAGS = {
    "official_scheme_conflict",
    "trusted_id_conflict",
    "fts_official_mixed",
    "short_name_or_acronym_ambiguous",
    "insufficient_evidence",
    "other_er_risk",
}

STOPWORDS = {
    "AND",
    "THE",
    "FOR",
    "WITH",
    "LIMITED",
    "LTD",
    "PLC",
    "LLP",
    "INC",
    "COMPANY",
    "GROUP",
    "SERVICES",
    "SERVICE",
    "TRUST",
    "COUNCIL",
    "AUTHORITY",
    "UNIVERSITY",
    "COLLEGE",
    "SCHOOL",
    "NATIONAL",
    "INTERNATIONAL",
}

PAIR_COLUMNS = [
    "entity_a_id",
    "entity_b_id",
    "strategies",
    "block_keys",
]

FEATURE_COLUMNS = [
    "entity_a_id",
    "entity_b_id",
    "name_a",
    "name_b",
    "norm_name_a",
    "norm_name_b",
    "strategies",
    "block_keys",
    "name_similarity",
    "token_jaccard",
    "alias_overlap_count",
    "address_region_a",
    "address_region_b",
    "region_relation",
    "shared_region",
    "shared_org_category",
    "shared_org_type",
    "official_scheme_conflict",
    "same_official_scheme",
    "shared_ocids",
    "shared_cpv2",
    "shared_years",
    "buyer_supplier_role_overlap",
    "heuristic_score",
    "risk_flags",
]


@dataclass(frozen=True)
class AblationConfig:
    max_pairs_total: int | None = None
    max_pairs_per_strategy: int | None = None
    max_block_size: int = 80
    max_prefix_block_size: int = 8
    max_token_frequency: int = 10
    max_alias_frequency: int = 60
    min_acronym_length: int = 3
    max_acronym_block_size: int = 20
    cluster_threshold: float = 0.88
    auto_merge_threshold: float = 0.88
    llm_max_tasks: int | None = None


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return []
    try:
        parsed = json.loads(value or "[]")
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _norm_tokens(value: str) -> set[str]:
    return {
        token
        for token in str(value or "").split()
        if len(token) >= 3 and token not in STOPWORDS
    }


def _blocking_tokens(value: str) -> list[str]:
    return sorted(token for token in _norm_tokens(value) if len(token) >= 4)


def _acronym(norm_name: str) -> str:
    tokens = [token for token in str(norm_name or "").split() if token not in STOPWORDS]
    if len(tokens) < 2:
        return ""
    acronym = "".join(token[0] for token in tokens if token)
    return acronym if 2 <= len(acronym) <= 8 else ""


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _region_relation(a: Any, b: Any) -> str:
    left = str(a or "").strip().upper()
    right = str(b or "").strip().upper()
    if not left or not right:
        return "missing"
    if left == right:
        return "same"
    if left.startswith("UK") and right.startswith("UK") and (left.startswith(right) or right.startswith(left)):
        return "compatible_hierarchy"
    return "conflict"


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _iter_group_pairs(ids: list[str]) -> Iterable[tuple[str, str]]:
    return itertools.combinations(sorted(set(ids)), 2)


def _limit_hit(value: int, limit: int | None) -> bool:
    return limit is not None and limit > 0 and value >= limit


def _theoretical_pairs(n: int) -> int:
    return int(n * (n - 1) / 2) if n >= 2 else 0


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    canonical = pd.read_parquet(ENTITIES_DIR / "canonical_orgs.parquet")
    alias_map = pd.read_parquet(ENTITIES_DIR / "alias_map.parquet")
    releases = pd.read_parquet(INTERIM_DIR / "releases.parquet")
    awards = pd.read_parquet(EXTRACTED_DIR / "awards.parquet")
    return canonical, alias_map, releases, awards


def build_entity_universe(canonical: pd.DataFrame) -> pd.DataFrame:
    entities = canonical.copy()
    entities["canonical_id"] = entities["canonical_id"].astype(str)
    entities["canonical_name"] = entities["canonical_name"].fillna("").astype(str)
    entities["norm_name"] = entities["canonical_name"].apply(normalise_name)
    entities["name_tokens"] = entities["norm_name"].apply(lambda value: sorted(_norm_tokens(value)))
    entities["blocking_tokens"] = entities["norm_name"].apply(_blocking_tokens)
    entities["acronym"] = entities["norm_name"].apply(_acronym)
    entities["scheme"] = entities["canonical_id"].apply(scheme_of)
    entities["alias_name_norms"] = entities["alias_names"].apply(
        lambda value: sorted({normalise_name(str(name)) for name in _json_list(value) if normalise_name(str(name))})
    )
    return entities


def build_entity_context(
    alias_map: pd.DataFrame,
    releases: pd.DataFrame,
    awards: pd.DataFrame,
    max_set_size: int = 200,
) -> pd.DataFrame:
    raw_to_canonical = dict(zip(alias_map["raw_id"].astype(str), alias_map["canonical_id"].astype(str)))
    context: dict[str, dict[str, set[str]]] = defaultdict(lambda: {
        "buyer_ocids": set(),
        "supplier_ocids": set(),
        "cpv2": set(),
        "years": set(),
    })

    release_meta = {}
    for row in releases[["ocid", "buyer_raw_id", "tender_cpv_id", "year"]].itertuples(index=False):
        ocid = str(row.ocid or "")
        cpv = str(row.tender_cpv_id or "")
        cpv2 = cpv[:2] if len(cpv) >= 2 else ""
        year = str(row.year or "")
        release_meta[ocid] = (cpv2, year)
        cid = raw_to_canonical.get(str(row.buyer_raw_id or ""))
        if cid:
            context[cid]["buyer_ocids"].add(ocid)
            if cpv2:
                context[cid]["cpv2"].add(cpv2)
            if year:
                context[cid]["years"].add(year)

    for row in awards[["ocid", "supplier_raw_ids"]].itertuples(index=False):
        ocid = str(row.ocid or "")
        cpv2, year = release_meta.get(ocid, ("", ""))
        for raw_id in _json_list(row.supplier_raw_ids):
            cid = raw_to_canonical.get(str(raw_id))
            if not cid:
                continue
            context[cid]["supplier_ocids"].add(ocid)
            if cpv2:
                context[cid]["cpv2"].add(cpv2)
            if year:
                context[cid]["years"].add(year)

    rows = []
    for cid, values in context.items():
        buyer_ocids = sorted(values["buyer_ocids"])[:max_set_size]
        supplier_ocids = sorted(values["supplier_ocids"])[:max_set_size]
        rows.append({
            "canonical_id": cid,
            "buyer_ocids": json.dumps(buyer_ocids),
            "supplier_ocids": json.dumps(supplier_ocids),
            "all_ocids": json.dumps(sorted(set(buyer_ocids) | set(supplier_ocids))[:max_set_size]),
            "cpv2": json.dumps(sorted(values["cpv2"])[:max_set_size]),
            "years": json.dumps(sorted(values["years"])[:max_set_size]),
            "buyer_ocid_count": len(values["buyer_ocids"]),
            "supplier_ocid_count": len(values["supplier_ocids"]),
        })
    return pd.DataFrame(rows)


def _add_block_pairs(
    pairs: dict[tuple[str, str], dict[str, set[str]]],
    metrics: list[dict[str, Any]],
    strategy: str,
    grouped: Iterable[tuple[str, list[str]]],
    config: AblationConfig,
    max_block_size: int | None = None,
) -> None:
    emitted = 0
    skipped_large = 0
    blocks = 0
    max_seen = 0
    hit_strategy_cap = False
    hit_global_cap = False

    for block_key, ids in grouped:
        ids = sorted(set(str(cid) for cid in ids if cid))
        if len(ids) < 2:
            continue
        blocks += 1
        max_seen = max(max_seen, len(ids))
        block_size_limit = max_block_size if max_block_size is not None else config.max_block_size
        if len(ids) > block_size_limit:
            skipped_large += 1
            continue
        for a, b in _iter_group_pairs(ids):
            key = _pair_key(a, b)
            record = pairs.setdefault(key, {"strategies": set(), "block_keys": set()})
            record["strategies"].add(strategy)
            record["block_keys"].add(f"{strategy}:{block_key}")
            emitted += 1
            hit_strategy_cap = _limit_hit(emitted, config.max_pairs_per_strategy)
            hit_global_cap = _limit_hit(len(pairs), config.max_pairs_total)
            if hit_strategy_cap or hit_global_cap:
                break
        if hit_strategy_cap or hit_global_cap:
            break

    metrics.append({
        "strategy": strategy,
        "blocks_seen": blocks,
        "pairs_emitted_before_dedup": emitted,
        "unique_pairs_after_strategy": len(pairs),
        "skipped_large_blocks": skipped_large,
        "max_block_size_seen": max_seen,
        "max_block_size_limit": max_block_size if max_block_size is not None else config.max_block_size,
        "max_pairs_per_strategy": config.max_pairs_per_strategy or 0,
        "max_pairs_total": config.max_pairs_total or 0,
        "hit_strategy_cap": hit_strategy_cap,
        "hit_global_cap": hit_global_cap,
        "is_truncated": hit_strategy_cap or hit_global_cap,
    })


def build_candidate_pairs(
    entities: pd.DataFrame,
    config: AblationConfig = AblationConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs: dict[tuple[str, str], dict[str, set[str]]] = {}
    metrics: list[dict[str, Any]] = []

    exact = (
        (key, grp["canonical_id"].tolist())
        for key, grp in entities[entities["norm_name"].ne("")].groupby("norm_name", sort=False)
    )
    _add_block_pairs(pairs, metrics, "exact_norm_name", exact, config)

    region_source = entities[
        entities["norm_name"].ne("") & entities["address_region"].fillna("").astype(str).ne("")
    ].copy()
    region_source["_region_block"] = region_source["norm_name"] + "|" + region_source["address_region"].astype(str)
    region = (
        (key, grp["canonical_id"].tolist())
        for key, grp in region_source.groupby("_region_block", sort=False)
    )
    _add_block_pairs(pairs, metrics, "name_region", region, config)

    prefix_source = entities[entities["norm_name"].str.len() >= 5].copy()
    prefix_source["_prefix3"] = prefix_source["norm_name"].str[:3]
    prefix = (
        (key, grp["canonical_id"].tolist())
        for key, grp in prefix_source.groupby("_prefix3", sort=False)
    )
    _add_block_pairs(pairs, metrics, "prefix3", prefix, config, max_block_size=config.max_prefix_block_size)

    token_members: dict[str, list[str]] = defaultdict(list)
    for _, row in entities.iterrows():
        for token in row["blocking_tokens"]:
            token_members[token].append(row["canonical_id"])
    rare_tokens = (
        (token, ids)
        for token, ids in token_members.items()
        if 2 <= len(set(ids)) <= config.max_token_frequency
    )
    _add_block_pairs(pairs, metrics, "rare_token", rare_tokens, config)

    alias_members: dict[str, list[str]] = defaultdict(list)
    for _, row in entities.iterrows():
        for alias_norm in row["alias_name_norms"]:
            alias_members[alias_norm].append(row["canonical_id"])
    alias_blocks = (
        (alias, ids)
        for alias, ids in alias_members.items()
        if 2 <= len(set(ids)) <= config.max_alias_frequency
    )
    _add_block_pairs(pairs, metrics, "alias_norm_overlap", alias_blocks, config)

    acronym_source = entities[entities["acronym"].ne("")].copy()
    acronym_source = acronym_source[
        acronym_source["acronym"].str.len().ge(config.min_acronym_length)
        & acronym_source["address_region"].fillna("").astype(str).ne("")
    ].copy()
    acronym_source["_acronym_region"] = (
        acronym_source["acronym"] + "|" + acronym_source["address_region"].astype(str)
    )
    acronym_region = (
        (key, grp["canonical_id"].tolist())
        for key, grp in acronym_source.groupby("_acronym_region", sort=False)
    )
    _add_block_pairs(
        pairs,
        metrics,
        "acronym_region",
        acronym_region,
        config,
        max_block_size=config.max_acronym_block_size,
    )

    rows = [
        {
            "entity_a_id": a,
            "entity_b_id": b,
            "strategies": json.dumps(sorted(payload["strategies"])),
            "block_keys": json.dumps(sorted(payload["block_keys"])[:20]),
        }
        for (a, b), payload in pairs.items()
    ]
    candidate_pairs = pd.DataFrame(rows, columns=PAIR_COLUMNS)
    metric_df = pd.DataFrame(metrics)
    n = len(entities)
    possible_pairs = n * (n - 1) / 2
    if not metric_df.empty:
        metric_df["entity_count"] = n
        metric_df["final_unique_pairs"] = len(candidate_pairs)
        metric_df["pair_reduction_ratio"] = 1.0 - (len(candidate_pairs) / possible_pairs if possible_pairs else 0.0)
    return candidate_pairs, metric_df


def _block_rows(
    strategy: str,
    grouped: Iterable[tuple[str, list[str]]],
    cap: int | None = None,
    block_size_limit: int | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for block_key, ids in grouped:
        unique_ids = sorted(set(str(cid) for cid in ids if cid))
        size = len(unique_ids)
        if size < 2:
            continue
        rows.append({
            "strategy": strategy,
            "block_key": str(block_key),
            "block_size": size,
            "theoretical_pairs": _theoretical_pairs(size),
            "generation_block_size_limit": block_size_limit or 0,
            "would_emit_under_current_rules": bool(block_size_limit is not None and size <= block_size_limit),
            "sample_entity_ids": json.dumps(unique_ids[:20]),
        })
    rows.sort(key=lambda row: (row["theoretical_pairs"], row["block_size"]), reverse=True)
    return rows[:cap] if cap else rows


def build_block_diagnostics(
    entities: pd.DataFrame,
    config: AblationConfig = AblationConfig(),
    cap_per_strategy: int = 200,
) -> pd.DataFrame:
    """Inspect which blocks dominate each blocking strategy before pair generation."""
    rows: list[dict[str, Any]] = []

    exact = (
        (key, grp["canonical_id"].tolist())
        for key, grp in entities[entities["norm_name"].ne("")].groupby("norm_name", sort=False)
    )
    rows.extend(_block_rows("exact_norm_name", exact, cap=cap_per_strategy, block_size_limit=config.max_block_size))

    region_source = entities[
        entities["norm_name"].ne("") & entities["address_region"].fillna("").astype(str).ne("")
    ].copy()
    region_source["_region_block"] = region_source["norm_name"] + "|" + region_source["address_region"].astype(str)
    region = (
        (key, grp["canonical_id"].tolist())
        for key, grp in region_source.groupby("_region_block", sort=False)
    )
    rows.extend(_block_rows("name_region", region, cap=cap_per_strategy, block_size_limit=config.max_block_size))

    prefix_source = entities[entities["norm_name"].str.len() >= 5].copy()
    prefix_source["_prefix3"] = prefix_source["norm_name"].str[:3]
    prefix = (
        (key, grp["canonical_id"].tolist())
        for key, grp in prefix_source.groupby("_prefix3", sort=False)
    )
    rows.extend(_block_rows("prefix3", prefix, cap=cap_per_strategy, block_size_limit=config.max_prefix_block_size))
    prefix_eligible = (
        (key, grp["canonical_id"].tolist())
        for key, grp in prefix_source.groupby("_prefix3", sort=False)
        if 2 <= len(set(grp["canonical_id"].tolist())) <= config.max_prefix_block_size
    )
    rows.extend(
        _block_rows(
            "prefix3_eligible",
            prefix_eligible,
            cap=cap_per_strategy,
            block_size_limit=config.max_prefix_block_size,
        )
    )

    token_members: dict[str, list[str]] = defaultdict(list)
    for _, row in entities.iterrows():
        for token in row["blocking_tokens"]:
            token_members[token].append(row["canonical_id"])
    token_blocks = ((token, ids) for token, ids in token_members.items() if len(set(ids)) >= 2)
    rows.extend(_block_rows("rare_token_all_frequencies", token_blocks, cap=cap_per_strategy))
    rare_token_eligible = (
        (token, ids)
        for token, ids in token_members.items()
        if 2 <= len(set(ids)) <= config.max_token_frequency
    )
    rows.extend(
        _block_rows(
            "rare_token_eligible",
            rare_token_eligible,
            cap=cap_per_strategy,
            block_size_limit=config.max_token_frequency,
        )
    )

    alias_members: dict[str, list[str]] = defaultdict(list)
    for _, row in entities.iterrows():
        for alias_norm in row["alias_name_norms"]:
            alias_members[alias_norm].append(row["canonical_id"])
    alias_blocks = ((alias, ids) for alias, ids in alias_members.items() if len(set(ids)) >= 2)
    rows.extend(
        _block_rows(
            "alias_norm_overlap",
            alias_blocks,
            cap=cap_per_strategy,
            block_size_limit=config.max_alias_frequency,
        )
    )

    acronym_source = entities[entities["acronym"].ne("")].copy()
    acronym = (
        (key, grp["canonical_id"].tolist())
        for key, grp in acronym_source.groupby("acronym", sort=False)
    )
    rows.extend(_block_rows("acronym", acronym, cap=cap_per_strategy))
    acronym_source = acronym_source[
        acronym_source["acronym"].str.len().ge(config.min_acronym_length)
        & acronym_source["address_region"].fillna("").astype(str).ne("")
    ].copy()
    acronym_source["_acronym_region"] = (
        acronym_source["acronym"] + "|" + acronym_source["address_region"].astype(str)
    )
    acronym_region = (
        (key, grp["canonical_id"].tolist())
        for key, grp in acronym_source.groupby("_acronym_region", sort=False)
    )
    rows.extend(
        _block_rows(
            "acronym_region_eligible",
            acronym_region,
            cap=cap_per_strategy,
            block_size_limit=config.max_acronym_block_size,
        )
    )

    diagnostics = pd.DataFrame(rows)
    if diagnostics.empty:
        return diagnostics
    diagnostics["exceeds_max_block_size_80"] = diagnostics["block_size"] > 80
    return diagnostics.sort_values(["strategy", "theoretical_pairs"], ascending=[True, False]).reset_index(drop=True)


def build_strategy_unique_contribution(candidate_pairs: pd.DataFrame) -> pd.DataFrame:
    """Summarise how much each strategy contributes only by itself."""
    rows = []
    if candidate_pairs.empty:
        return pd.DataFrame(columns=["strategy", "pairs_with_strategy", "pairs_only_strategy"])

    counters: Counter[str] = Counter()
    unique_counters: Counter[str] = Counter()
    for _, row in candidate_pairs.iterrows():
        strategies = _json_list(row.get("strategies"))
        for strategy in strategies:
            counters[str(strategy)] += 1
        if len(strategies) == 1:
            unique_counters[str(strategies[0])] += 1

    for strategy in sorted(counters):
        rows.append({
            "strategy": strategy,
            "pairs_with_strategy": counters[strategy],
            "pairs_only_strategy": unique_counters.get(strategy, 0),
            "only_strategy_rate": (
                unique_counters.get(strategy, 0) / counters[strategy]
                if counters[strategy]
                else 0.0
            ),
        })
    return pd.DataFrame(rows).sort_values("pairs_only_strategy", ascending=False)


def _set_intersection_size(a: Any, b: Any) -> int:
    return len(set(_json_list(a)).intersection(set(_json_list(b))))


def build_pair_features(
    entities: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
    context: pd.DataFrame,
) -> pd.DataFrame:
    if candidate_pairs.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    entity_lookup = entities.set_index("canonical_id")
    context_lookup = context.set_index("canonical_id") if not context.empty else pd.DataFrame()
    rows = []

    for _, pair in candidate_pairs.iterrows():
        a_id = pair["entity_a_id"]
        b_id = pair["entity_b_id"]
        if a_id not in entity_lookup.index or b_id not in entity_lookup.index:
            continue
        a = entity_lookup.loc[a_id]
        b = entity_lookup.loc[b_id]
        a_ctx = context_lookup.loc[a_id] if not context_lookup.empty and a_id in context_lookup.index else {}
        b_ctx = context_lookup.loc[b_id] if not context_lookup.empty and b_id in context_lookup.index else {}

        token_jaccard = 0.0
        tokens_a = set(a["name_tokens"])
        tokens_b = set(b["name_tokens"])
        if tokens_a or tokens_b:
            token_jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

        alias_overlap = len(set(a["alias_name_norms"]).intersection(set(b["alias_name_norms"])))
        name_similarity = _similarity(a["norm_name"], b["norm_name"])
        address_region_a = str(a.get("address_region") or "")
        address_region_b = str(b.get("address_region") or "")
        region_relation = _region_relation(address_region_a, address_region_b)
        shared_region = region_relation in {"same", "compatible_hierarchy"}
        shared_org_category = bool(a.get("org_category")) and a.get("org_category") == b.get("org_category")
        shared_org_type = bool(a.get("org_type")) and a.get("org_type") == b.get("org_type")
        a_scheme = str(a.get("scheme") or "")
        b_scheme = str(b.get("scheme") or "")
        a_trusted = is_official(a_scheme)
        b_trusted = is_official(b_scheme)
        same_official_scheme = a_trusted and b_trusted and a_scheme == b_scheme
        official_scheme_conflict = a_trusted and b_trusted and a_scheme != b_scheme
        trusted_id_conflict = same_official_scheme and a_id != b_id
        fts_official_mixed = (a_scheme == "GB-FTS" and b_trusted) or (b_scheme == "GB-FTS" and a_trusted)

        shared_ocids = _set_intersection_size(a_ctx.get("all_ocids", "[]"), b_ctx.get("all_ocids", "[]")) if isinstance(a_ctx, pd.Series) else 0
        shared_cpv2 = _set_intersection_size(a_ctx.get("cpv2", "[]"), b_ctx.get("cpv2", "[]")) if isinstance(a_ctx, pd.Series) else 0
        shared_years = _set_intersection_size(a_ctx.get("years", "[]"), b_ctx.get("years", "[]")) if isinstance(a_ctx, pd.Series) else 0
        buyer_supplier_role_overlap = (
            isinstance(a_ctx, pd.Series)
            and isinstance(b_ctx, pd.Series)
            and (
                (int(a_ctx.get("buyer_ocid_count", 0)) > 0 and int(b_ctx.get("buyer_ocid_count", 0)) > 0)
                or (int(a_ctx.get("supplier_ocid_count", 0)) > 0 and int(b_ctx.get("supplier_ocid_count", 0)) > 0)
            )
        )

        strategy_bonus = 0.04 if "alias_norm_overlap" in pair["strategies"] else 0.0
        context_bonus = min(0.08, 0.02 * shared_ocids + 0.015 * shared_cpv2 + 0.01 * shared_years)
        metadata_bonus = (0.03 if shared_region else 0.0) + (0.02 if shared_org_category else 0.0)
        conflict_penalty = (0.12 if official_scheme_conflict else 0.0) + (0.10 if trusted_id_conflict else 0.0)
        heuristic_score = max(
            0.0,
            min(1.0, 0.72 * name_similarity + 0.14 * token_jaccard + strategy_bonus + context_bonus + metadata_bonus - conflict_penalty),
        )

        risk_flags = []
        def add_risk(flag: str) -> None:
            if flag not in risk_flags:
                risk_flags.append(flag)

        if official_scheme_conflict:
            add_risk("official_scheme_conflict")
        if trusted_id_conflict:
            add_risk("trusted_id_conflict")
        if fts_official_mixed:
            add_risk("fts_official_mixed")
        if a["norm_name"] != b["norm_name"] and max(len(a["norm_name"]), len(b["norm_name"])) <= 5:
            add_risk("short_name_or_acronym_ambiguous")
        if region_relation == "conflict":
            add_risk("region_conflict")
        if any(str(strategy).startswith("acronym") for strategy in _json_list(pair["strategies"])) and name_similarity < 0.85:
            add_risk("short_name_or_acronym_ambiguous")
            add_risk("insufficient_evidence")
        if (int(a.get("n_aliases", 0)) >= 50 or int(b.get("n_aliases", 0)) >= 50) and name_similarity < 0.98:
            add_risk("other_er_risk")

        rows.append({
            "entity_a_id": a_id,
            "entity_b_id": b_id,
            "name_a": a["canonical_name"],
            "name_b": b["canonical_name"],
            "norm_name_a": a["norm_name"],
            "norm_name_b": b["norm_name"],
            "strategies": pair["strategies"],
            "block_keys": pair["block_keys"],
            "name_similarity": round(name_similarity, 4),
            "token_jaccard": round(token_jaccard, 4),
            "alias_overlap_count": alias_overlap,
            "address_region_a": address_region_a,
            "address_region_b": address_region_b,
            "region_relation": region_relation,
            "shared_region": shared_region,
            "shared_org_category": shared_org_category,
            "shared_org_type": shared_org_type,
            "official_scheme_conflict": official_scheme_conflict,
            "same_official_scheme": same_official_scheme,
            "shared_ocids": shared_ocids,
            "shared_cpv2": shared_cpv2,
            "shared_years": shared_years,
            "buyer_supplier_role_overlap": bool(buyer_supplier_role_overlap),
            "heuristic_score": round(heuristic_score, 4),
            "risk_flags": json.dumps(risk_flags),
        })

    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def build_feature_ablation_metrics(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame(columns=["variant", "candidate_pairs", "selected_pairs", "selection_rate"])

    variants = [
        ("name_only_096", features["name_similarity"] >= 0.96),
        ("name_alias_092", (features["name_similarity"] >= 0.92) & (features["alias_overlap_count"] > 0)),
        ("name_region_094", (features["name_similarity"] >= 0.94) & features["shared_region"]),
        ("name_context_090", (features["name_similarity"] >= 0.90) & ((features["shared_ocids"] > 0) | (features["shared_cpv2"] > 0))),
        ("full_heuristic_085", features["heuristic_score"] >= 0.85),
        ("full_heuristic_090", features["heuristic_score"] >= 0.90),
        ("high_conf_no_risk", (features["heuristic_score"] >= 0.88) & features["risk_flags"].eq("[]")),
    ]
    rows = []
    for name, mask in variants:
        selected = features[mask]
        rows.append({
            "variant": name,
            "candidate_pairs": len(features),
            "selected_pairs": len(selected),
            "selection_rate": float(len(selected) / len(features)) if len(features) else 0.0,
            "median_name_similarity": float(selected["name_similarity"].median()) if len(selected) else 0.0,
            "pairs_with_shared_context": int(((selected["shared_ocids"] > 0) | (selected["shared_cpv2"] > 0)).sum()) if len(selected) else 0,
            "pairs_with_risk_flags": int(selected["risk_flags"].ne("[]").sum()) if len(selected) else 0,
        })
    return pd.DataFrame(rows)


def classify_risk_tiers(features: pd.DataFrame, auto_merge_threshold: float = 0.88) -> pd.DataFrame:
    """Assign global ER candidate pairs to action tiers.

    The LLM tier is rule-derived, not a fixed top-N sample. Severe risk flags
    are routed to LLM review. Pure region conflicts are kept out of the priority
    queue because they often reflect address-region quality or hierarchy issues.
    High-scoring no-risk rows are deterministic candidates only when region
    evidence is identical or missing. Region-hierarchy-compatible rows are
    useful candidates, but are kept out of deterministic auto-merge because
    some chains and public-sector bodies reuse names across local records.
    """
    if features.empty:
        result = features.copy()
        result["risk_tier"] = pd.Series(dtype=str)
        result["tier_reason"] = pd.Series(dtype=str)
        return result

    result = features.copy()
    risk_sets = result["risk_flags"].apply(lambda value: set(_json_list(value)))
    has_risk = risk_sets.apply(bool)
    has_llm_priority_risk = risk_sets.apply(lambda flags: bool(flags & LLM_PRIORITY_RISK_FLAGS))
    pure_region_conflict = risk_sets.apply(lambda flags: flags == {"region_conflict"})
    exact_name = result["norm_name_a"].eq(result["norm_name_b"])
    exact_or_alias_strategy = result["strategies"].apply(
        lambda value: bool({"exact_norm_name", "alias_norm_overlap"} & set(_json_list(value)))
    )
    region_mismatch = result["address_region_a"].fillna("").astype(str).str.upper().ne(
        result["address_region_b"].fillna("").astype(str).str.upper()
    )
    region_hierarchy_candidate = (
        result["region_relation"].eq("compatible_hierarchy")
        & region_mismatch
        & (result["heuristic_score"] >= auto_merge_threshold)
    )
    high_score = result["heuristic_score"] >= auto_merge_threshold
    high_no_risk = ~has_risk & high_score & ~region_hierarchy_candidate
    region_quality_candidate = pure_region_conflict & exact_name & exact_or_alias_strategy & high_score

    result["risk_tier"] = "candidate_only"
    result["tier_reason"] = "below_auto_threshold_or_needs_more_evidence"
    result.loc[high_no_risk, "risk_tier"] = "deterministic_auto_merge_candidate"
    result.loc[high_no_risk, "tier_reason"] = f"no_risk_and_heuristic_score_ge_{auto_merge_threshold}"
    result.loc[region_quality_candidate, "risk_tier"] = "region_quality_candidate"
    result.loc[region_quality_candidate, "tier_reason"] = (
        "pure_region_conflict_exact_or_alias_high_score"
    )
    result.loc[region_hierarchy_candidate, "risk_tier"] = "region_hierarchy_candidate"
    result.loc[region_hierarchy_candidate, "tier_reason"] = (
        "region_hierarchy_compatible_high_score_needs_rule_validation"
    )
    result.loc[has_llm_priority_risk, "risk_tier"] = "llm_review"
    result.loc[has_llm_priority_risk, "tier_reason"] = "priority_risk_flags"
    return result


def build_risk_tier_metrics(risk_tiers: pd.DataFrame, blocking_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if risk_tiers.empty:
        return pd.DataFrame(columns=["metric", "value"])

    rows.extend([
        {"metric": "candidate_pairs_total", "value": len(risk_tiers)},
        {"metric": "llm_review_pairs", "value": int(risk_tiers["risk_tier"].eq("llm_review").sum())},
        {
            "metric": "deterministic_auto_merge_candidate_pairs",
            "value": int(risk_tiers["risk_tier"].eq("deterministic_auto_merge_candidate").sum()),
        },
        {
            "metric": "region_quality_candidate_pairs",
            "value": int(risk_tiers["risk_tier"].eq("region_quality_candidate").sum()),
        },
        {
            "metric": "region_hierarchy_candidate_pairs",
            "value": int(risk_tiers["risk_tier"].eq("region_hierarchy_candidate").sum()),
        },
        {"metric": "middle_candidate_pairs", "value": int(risk_tiers["risk_tier"].eq("candidate_only").sum())},
        {
            "metric": "blocking_any_cap_hit",
            "value": bool(not blocking_metrics.empty and blocking_metrics["is_truncated"].astype(bool).any()),
        },
    ])
    if "risk_flags" in risk_tiers.columns:
        for flags, count in risk_tiers["risk_flags"].value_counts().head(30).items():
            rows.append({"metric": f"risk_flags::{flags}", "value": int(count)})
    if not blocking_metrics.empty:
        for _, row in blocking_metrics.iterrows():
            rows.append({"metric": f"blocking::{row['strategy']}::is_truncated", "value": bool(row["is_truncated"])})
            rows.append({"metric": f"blocking::{row['strategy']}::pairs_emitted", "value": int(row["pairs_emitted_before_dedup"])})
    return pd.DataFrame(rows)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def build_cluster_report(features: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame(columns=["cluster_id", "cluster_size", "member_ids", "member_names", "risk_flags"])

    selected = features[features["heuristic_score"] >= threshold]
    uf = _UnionFind()
    name_by_id = {}
    risk_by_id: dict[str, set[str]] = defaultdict(set)
    for _, row in selected.iterrows():
        uf.union(row["entity_a_id"], row["entity_b_id"])
        name_by_id[row["entity_a_id"]] = row["name_a"]
        name_by_id[row["entity_b_id"]] = row["name_b"]
        for flag in _json_list(row["risk_flags"]):
            risk_by_id[row["entity_a_id"]].add(flag)
            risk_by_id[row["entity_b_id"]].add(flag)

    clusters: dict[str, list[str]] = defaultdict(list)
    for entity_id in name_by_id:
        clusters[uf.find(entity_id)].append(entity_id)

    rows = []
    for cluster_id, members in clusters.items():
        if len(members) < 2:
            continue
        flags = sorted(set().union(*(risk_by_id.get(member, set()) for member in members)))
        rows.append({
            "cluster_id": cluster_id,
            "cluster_size": len(members),
            "member_ids": json.dumps(sorted(members)[:30]),
            "member_names": json.dumps([name_by_id[mid] for mid in sorted(members)[:30]]),
            "risk_flags": json.dumps(flags),
        })
    return pd.DataFrame(rows).sort_values(["cluster_size", "cluster_id"], ascending=[False, True]) if rows else pd.DataFrame()


def high_risk_pairs(features: pd.DataFrame, limit: int = 500) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    risky = features[
        features["risk_flags"].ne("[]")
        | ((features["heuristic_score"] >= 0.82) & features["official_scheme_conflict"])
        | ((features["heuristic_score"] >= 0.86) & (features["shared_region"] == False))
    ].copy()
    return risky.sort_values(["heuristic_score", "name_similarity"], ascending=False).head(limit)


def _stable_task_id(*parts: str) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _er_pair_system_prompt() -> str:
    return (
        "You adjudicate whether two procurement organisation records refer to the same "
        "real-world organisation. Use only the evidence provided. Prefer uncertain or "
        "do_not_merge when there is a legal-identifier conflict, parent/subsidiary risk, "
        "regional-office risk, acronym ambiguity, or insufficient evidence. Return strict JSON."
    )


def _er_pair_user_prompt(evidence: dict[str, Any]) -> str:
    return json.dumps(
        {
            "task": "Decide whether these two procurement entity records should be merged.",
            "rules": [
                "merge only if both records are the same real-world legal/operational organisation.",
                "do_not_merge if official identifiers, geography, roles, or context imply different entities.",
                "uncertain if evidence is insufficient.",
                "If decision is merge, approved_entity_ids must contain both entity IDs and excluded_entity_ids must be empty.",
                "If decision is do_not_merge or uncertain, approved_entity_ids must be empty and excluded_entity_ids must contain both entity IDs.",
                "risk_flags must describe residual risk in the approved merge only; use ['none'] only when there is no residual risk.",
                "GB-FTS is a procurement-system identifier, not an independent official registry.",
                "Trusted registry ID conflicts are high risk and should usually not merge.",
            ],
            "expected_output_schema": {
                "prompt_version": ER_PAIR_PROMPT_VERSION,
                "schema_version": ER_PAIR_SCHEMA_VERSION,
                "task_id": "string copied from input",
                "decision": "merge | do_not_merge | uncertain",
                "confidence": "number from 0 to 1",
                "approved_entity_ids": ["two canonical IDs only when decision is merge"],
                "excluded_entity_ids": ["canonical IDs excluded from merge"],
                "risk_flags": [f"zero or more of: {', '.join(ER_PAIR_RISK_FLAGS)}"],
                "reason": "one concise explanation grounded in the evidence",
            },
            "evidence": evidence,
        },
        ensure_ascii=False,
        indent=2,
    )


def er_pair_decision_json_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "er_pair_adjudication_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "prompt_version",
                "schema_version",
                "task_id",
                "decision",
                "confidence",
                "approved_entity_ids",
                "excluded_entity_ids",
                "risk_flags",
                "reason",
            ],
            "properties": {
                "prompt_version": {"type": "string", "enum": [ER_PAIR_PROMPT_VERSION]},
                "schema_version": {"type": "string", "enum": [ER_PAIR_SCHEMA_VERSION]},
                "task_id": {"type": "string"},
                "decision": {"type": "string", "enum": ["merge", "do_not_merge", "uncertain"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "approved_entity_ids": {"type": "array", "items": {"type": "string"}},
                "excluded_entity_ids": {"type": "array", "items": {"type": "string"}},
                "risk_flags": {
                    "type": "array",
                    "items": {"type": "string", "enum": ER_PAIR_RISK_FLAGS},
                },
                "reason": {"type": "string"},
            },
        },
    }


def build_llm_review_queue(features: pd.DataFrame, max_tasks: int = 500) -> list[dict[str, Any]]:
    """Build LLM review tasks for globally risk-tiered ER candidate pairs."""
    if features.empty:
        return []
    if "risk_tier" in features.columns:
        review = features[features["risk_tier"].eq("llm_review")].copy()
    else:
        review = classify_risk_tiers(features)
        review = review[review["risk_tier"].eq("llm_review")].copy()
    review = review.sort_values(["heuristic_score", "name_similarity"], ascending=False)
    if max_tasks is not None and max_tasks > 0:
        review = review.head(max_tasks)
    records: list[dict[str, Any]] = []
    for _, row in review.iterrows():
        task_id = _stable_task_id("er_pair", row["entity_a_id"], row["entity_b_id"])
        evidence = {
            "task_id": task_id,
            "task_type": "er_pair_adjudication",
            "entity_a": {
                "canonical_id": row["entity_a_id"],
                "name": row["name_a"],
                "normalised_name": row["norm_name_a"],
            },
            "entity_b": {
                "canonical_id": row["entity_b_id"],
                "name": row["name_b"],
                "normalised_name": row["norm_name_b"],
            },
            "features": {
                "strategies": _json_list(row["strategies"]),
                "name_similarity": float(row["name_similarity"]),
                "token_jaccard": float(row["token_jaccard"]),
                "alias_overlap_count": int(row["alias_overlap_count"]),
                "address_region_a": str(row.get("address_region_a") or ""),
                "address_region_b": str(row.get("address_region_b") or ""),
                "region_relation": str(row.get("region_relation") or ""),
                "shared_region": bool(row["shared_region"]),
                "shared_org_category": bool(row["shared_org_category"]),
                "shared_org_type": bool(row["shared_org_type"]),
                "official_scheme_conflict": bool(row["official_scheme_conflict"]),
                "same_official_scheme": bool(row["same_official_scheme"]),
                "shared_ocids": int(row["shared_ocids"]),
                "shared_cpv2": int(row["shared_cpv2"]),
                "shared_years": int(row["shared_years"]),
                "buyer_supplier_role_overlap": bool(row["buyer_supplier_role_overlap"]),
                "heuristic_score": float(row["heuristic_score"]),
                "risk_flags": _json_list(row["risk_flags"]),
            },
        }
        records.append({
            "task_id": task_id,
            "task_type": "er_pair_adjudication",
            "prompt_version": ER_PAIR_PROMPT_VERSION,
            "schema_version": ER_PAIR_SCHEMA_VERSION,
            "response_format": er_pair_decision_json_schema(),
            "evidence": evidence,
            "failure_decision_template": {
                "task_id": task_id,
                "prompt_version": ER_PAIR_PROMPT_VERSION,
                "schema_version": ER_PAIR_SCHEMA_VERSION,
                "decision": "uncertain",
                "confidence": 0.0,
                "approved_entity_ids": [],
                "excluded_entity_ids": [row["entity_a_id"], row["entity_b_id"]],
                "risk_flags": ["model_call_error"],
                "reason": "Model call failed or returned invalid JSON.",
            },
            "messages": [
                {"role": "system", "content": _er_pair_system_prompt()},
                {"role": "user", "content": _er_pair_user_prompt(evidence)},
            ],
        })
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summary(
    blocking_metrics: pd.DataFrame,
    feature_metrics: pd.DataFrame,
    risk_tiers: pd.DataFrame,
    clusters: pd.DataFrame,
    llm_queue_size: int,
    block_diagnostics: pd.DataFrame | None = None,
    strategy_unique: pd.DataFrame | None = None,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Full-Corpus ER Ablation Summary",
        "",
        "This run is read-only over `data/entities/*` and writes experiment artifacts only.",
        "",
        "## Candidate Generation",
        "",
        f"- Candidate pairs: `{len(risk_tiers):,}`",
        f"- Blocking strategies: `{blocking_metrics['strategy'].nunique() if not blocking_metrics.empty else 0}`",
        f"- Any blocking cap hit: `{bool(not blocking_metrics.empty and blocking_metrics['is_truncated'].astype(bool).any())}`",
        "",
        "## Feature Ablation",
        "",
    ]
    if feature_metrics.empty:
        lines.append("_No feature metrics._")
    else:
        for _, row in feature_metrics.iterrows():
            lines.append(
                f"- `{row['variant']}`: {int(row['selected_pairs']):,} selected "
                f"({row['selection_rate']:.2%})"
            )
    lines.extend([
        "",
        "## Risk Tiers",
        "",
        f"- LLM review pairs: `{int(risk_tiers['risk_tier'].eq('llm_review').sum()) if not risk_tiers.empty else 0:,}`",
        f"- Deterministic auto-merge candidates: `{int(risk_tiers['risk_tier'].eq('deterministic_auto_merge_candidate').sum()) if not risk_tiers.empty else 0:,}`",
        f"- Region quality candidates: `{int(risk_tiers['risk_tier'].eq('region_quality_candidate').sum()) if not risk_tiers.empty else 0:,}`",
        f"- Region hierarchy candidates: `{int(risk_tiers['risk_tier'].eq('region_hierarchy_candidate').sum()) if not risk_tiers.empty else 0:,}`",
        f"- Middle candidate-only pairs: `{int(risk_tiers['risk_tier'].eq('candidate_only').sum()) if not risk_tiers.empty else 0:,}`",
        f"- LLM queue tasks written: `{llm_queue_size:,}`",
        "",
        "## Blocking Diagnostics",
        "",
        f"- Diagnostic rows: `{len(block_diagnostics) if block_diagnostics is not None else 0:,}`",
        f"- Largest diagnostic block size: `{int(block_diagnostics['block_size'].max()) if block_diagnostics is not None and not block_diagnostics.empty else 0}`",
        f"- Strategies with unique-only contribution rows: `{len(strategy_unique) if strategy_unique is not None else 0}`",
        "",
        "## Cluster Dry Run",
        "",
        f"- High-score clusters: `{len(clusters):,}`",
        f"- Largest cluster size: `{int(clusters['cluster_size'].max()) if not clusters.empty else 0}`",
        "",
        "## Safety",
        "",
        "No canonical IDs or alias maps are changed by this experiment.",
    ])
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config: AblationConfig = AblationConfig()) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    canonical, alias_map, releases, awards = load_inputs()
    entities = build_entity_universe(canonical)
    block_diagnostics = build_block_diagnostics(entities, config)
    context = build_entity_context(alias_map, releases, awards)
    candidate_pairs, blocking_metrics = build_candidate_pairs(entities, config)
    strategy_unique = build_strategy_unique_contribution(candidate_pairs)
    features = build_pair_features(entities, candidate_pairs, context)
    risk_tiers = classify_risk_tiers(features, auto_merge_threshold=config.auto_merge_threshold)
    feature_metrics = build_feature_ablation_metrics(features)
    risky_pairs = high_risk_pairs(features)
    clusters = build_cluster_report(features, threshold=config.cluster_threshold)
    llm_queue = build_llm_review_queue(risk_tiers, max_tasks=config.llm_max_tasks)
    risk_tier_metrics = build_risk_tier_metrics(risk_tiers, blocking_metrics)

    context.to_parquet(ENTITY_CONTEXT_PATH, index=False)
    candidate_pairs.to_parquet(CANDIDATE_PAIRS_PATH, index=False)
    features.to_parquet(PAIR_FEATURES_PATH, index=False)
    risk_tiers.to_parquet(RISK_TIERS_PATH, index=False)
    risk_tiers[risk_tiers["risk_tier"].eq("deterministic_auto_merge_candidate")].to_parquet(
        AUTO_MERGE_CANDIDATES_PATH,
        index=False,
    )
    risk_tiers[risk_tiers["risk_tier"].eq("region_quality_candidate")].to_parquet(
        REGION_QUALITY_CANDIDATES_PATH,
        index=False,
    )
    risk_tiers[risk_tiers["risk_tier"].eq("region_hierarchy_candidate")].to_parquet(
        REGION_HIERARCHY_CANDIDATES_PATH,
        index=False,
    )
    risk_tiers[risk_tiers["risk_tier"].eq("candidate_only")].to_parquet(MIDDLE_CANDIDATES_PATH, index=False)
    blocking_metrics.to_csv(BLOCKING_METRICS_PATH, index=False)
    feature_metrics.to_csv(FEATURE_ABLATION_METRICS_PATH, index=False)
    risk_tier_metrics.to_csv(RISK_TIER_METRICS_PATH, index=False)
    block_diagnostics.to_csv(BLOCK_DIAGNOSTICS_PATH, index=False)
    strategy_unique.to_csv(STRATEGY_UNIQUE_CONTRIBUTION_PATH, index=False)
    risky_pairs.to_csv(HIGH_RISK_PAIRS_PATH, index=False)
    clusters.to_csv(HIGH_RISK_CLUSTERS_PATH, index=False)
    write_jsonl(llm_queue, LLM_QUEUE_PATH)
    write_summary(
        blocking_metrics,
        feature_metrics,
        risk_tiers,
        clusters,
        len(llm_queue),
        block_diagnostics=block_diagnostics,
        strategy_unique=strategy_unique,
    )

    print(f"Written: {ENTITY_CONTEXT_PATH} ({len(context):,} entities with context)")
    print(f"Written: {CANDIDATE_PAIRS_PATH} ({len(candidate_pairs):,} pairs)")
    print(f"Written: {PAIR_FEATURES_PATH} ({len(features):,} pairs)")
    print(f"Written: {RISK_TIERS_PATH} ({len(risk_tiers):,} pairs)")
    print(f"Written: {AUTO_MERGE_CANDIDATES_PATH}")
    print(f"Written: {REGION_QUALITY_CANDIDATES_PATH}")
    print(f"Written: {REGION_HIERARCHY_CANDIDATES_PATH}")
    print(f"Written: {MIDDLE_CANDIDATES_PATH}")
    print(f"Written: {BLOCKING_METRICS_PATH}")
    print(f"Written: {FEATURE_ABLATION_METRICS_PATH}")
    print(f"Written: {RISK_TIER_METRICS_PATH}")
    print(f"Written: {BLOCK_DIAGNOSTICS_PATH}")
    print(f"Written: {STRATEGY_UNIQUE_CONTRIBUTION_PATH}")
    print(f"Written: {HIGH_RISK_PAIRS_PATH}")
    print(f"Written: {HIGH_RISK_CLUSTERS_PATH}")
    print(f"Written: {LLM_QUEUE_PATH} ({len(llm_queue):,} tasks)")
    print(f"Written: {SUMMARY_PATH}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-corpus ER ablation")
    parser.add_argument("--max-pairs-total", type=int, default=0, help="0 means no global pair cap")
    parser.add_argument("--max-pairs-per-strategy", type=int, default=0, help="0 means no per-strategy cap")
    parser.add_argument("--max-block-size", type=int, default=80)
    parser.add_argument("--max-prefix-block-size", type=int, default=8)
    parser.add_argument("--max-token-frequency", type=int, default=10)
    parser.add_argument("--max-alias-frequency", type=int, default=60)
    parser.add_argument("--min-acronym-length", type=int, default=3)
    parser.add_argument("--max-acronym-block-size", type=int, default=20)
    parser.add_argument("--cluster-threshold", type=float, default=0.88)
    parser.add_argument("--auto-merge-threshold", type=float, default=0.88)
    parser.add_argument("--llm-max-tasks", type=int, default=0, help="0 means write all LLM-review tier tasks")
    return parser.parse_args(argv)


def cli_main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run(
        AblationConfig(
            max_pairs_total=args.max_pairs_total or None,
            max_pairs_per_strategy=args.max_pairs_per_strategy or None,
            max_block_size=args.max_block_size,
            max_prefix_block_size=args.max_prefix_block_size,
            max_token_frequency=args.max_token_frequency,
            max_alias_frequency=args.max_alias_frequency,
            min_acronym_length=args.min_acronym_length,
            max_acronym_block_size=args.max_acronym_block_size,
            cluster_threshold=args.cluster_threshold,
            auto_merge_threshold=args.auto_merge_threshold,
            llm_max_tasks=args.llm_max_tasks or None,
        )
    )


__all__ = [
    "AblationConfig",
    "build_candidate_pairs",
    "build_block_diagnostics",
    "build_cluster_report",
    "build_entity_context",
    "build_entity_universe",
    "build_feature_ablation_metrics",
    "build_llm_review_queue",
    "build_risk_tier_metrics",
    "build_strategy_unique_contribution",
    "classify_risk_tiers",
    "build_pair_features",
    "cli_main",
    "run",
]
