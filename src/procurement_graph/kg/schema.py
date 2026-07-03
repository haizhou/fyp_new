"""Column contracts for the deterministic KG v0 build."""

from __future__ import annotations

ORG_NODE_COLUMNS = [
    "canonical_id",
    "canonical_name",
    "org_type",
    "address_region",
    "address_regions",
    "org_category",
    "official_scheme",
    "official_id",
    "er_status",
    "n_aliases",
    "alias_raw_id_count",
    "is_buyer",
    "is_supplier",
    "is_mixed_role",
    "buyer_contract_count",
    "supplier_contract_count",
    "buyer_additive_value_sum",
    "supplier_additive_value_sum",
    "first_activity_year",
    "last_activity_year",
]

CONTRACT_NODE_COLUMNS = [
    "contract_node_id",
    "ocid",
    "award_id",
    "contract_id",
    "release_date",
    "release_year",
    "buyer_raw_id",
    "tender_title",
    "award_title",
    "award_status",
    "tender_method",
    "tender_category",
    "tender_cpv_id",
    "tender_cpv_description",
    "value_amount",
    "value_currency",
    "value_source",
    "value_is_additive",
    "award_date_signed",
    "award_period_start",
    "award_period_end",
    "tender_period_end",
    "contract_period_start",
    "contract_period_end",
    "has_award_signed_date",
    "has_contract_period",
    "days_release_to_tender_end",
    "days_release_to_award_signed",
    "contract_duration_days",
    "related_lots",
    "above_threshold",
]

CPV_NODE_COLUMNS = [
    "cpv_code",
    "cpv_description",
    "cpv_division",
    "cpv_group",
    "cpv_class",
    "contract_count",
    "additive_value_sum",
    "first_activity_year",
    "last_activity_year",
]

EVIDENCE_NODE_COLUMNS = [
    "evidence_id",
    "ocid",
    "field_path",
    "lot_id",
    "text",
    "text_length",
]

BUYER_EDGE_COLUMNS = [
    "edge_id",
    "canonical_id",
    "contract_node_id",
    "raw_id",
    "ocid",
    "award_id",
    "edge_source",
]

SUPPLIER_EDGE_COLUMNS = BUYER_EDGE_COLUMNS.copy()

CATEGORIZED_BY_EDGE_COLUMNS = [
    "edge_id",
    "contract_node_id",
    "cpv_code",
    "ocid",
    "award_id",
    "edge_source",
]

EVIDENCE_FOR_EDGE_COLUMNS = [
    "edge_id",
    "evidence_id",
    "contract_node_id",
    "ocid",
    "award_id",
    "match_scope",
    "edge_source",
]

NODE_FILES = {
    "org_nodes": "nodes/org_nodes.parquet",
    "contract_nodes": "nodes/contract_nodes.parquet",
    "cpv_nodes": "nodes/cpv_nodes.parquet",
    "evidence_nodes": "nodes/evidence_nodes.parquet",
}

EDGE_FILES = {
    "buyer_of": "edges/buyer_of.parquet",
    "supplier_of": "edges/supplier_of.parquet",
    "categorized_by": "edges/categorized_by.parquet",
    "evidence_for": "edges/evidence_for.parquet",
}
