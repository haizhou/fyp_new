"""PACS identifier and signature machinery (spec v2.2, Isolation ledger).

Every intent carries seven identifiers. The five zero-overlap build gates and
the seen/unseen adjudication consume these; nothing is sealed without them.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import importlib.util

_spec = importlib.util.spec_from_file_location("bct", ROOT / "scripts/build_compose_train.py")
bct = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bct)

shape_signature = bct.signature  # canonical shape signature (skeleton, no literals)


def gold_tree_hash(tree: dict) -> str:
    return hashlib.sha1(json.dumps(tree, sort_keys=True).encode()).hexdigest()[:16]


def entity_anchor_signature(params: dict) -> str:
    """Sorted anchor tuple: which concrete entities/years/CPVs anchor the intent."""
    items = sorted((k, str(v)) for k, v in params.items())
    return hashlib.sha1(json.dumps(items).encode()).hexdigest()[:12]


def logical_signature(tree: dict) -> dict:
    """Exported logic profile used by the naturalization logic gates (spec v2.2):
    operation kind, negation flags, aggregation, comparison direction,
    quantifier, and left/right scope digests."""
    sig = {"operation": tree.get("node"), "op": tree.get("op"),
           "negation": False, "aggregation": None, "comparison": None,
           "quantifier": None, "left_scope": None, "right_scope": None}

    def scan(node, path=""):
        if isinstance(node, dict):
            if node.get("op") == "not" or node.get("negate"):
                sig["negation"] = True
            if node.get("node") in ("sum", "count", "size", "top", "argext", "extreme"):
                sig["aggregation"] = sig["aggregation"] or node["node"]
            if node.get("node") in ("combine", "vcompare", "gcombine", "keys_where") \
                    and node.get("op") in ("gt", "lt", "ge", "le"):
                sig["comparison"] = node["op"]
            if node.get("node") == "setop" and node.get("op") == "difference":
                sig["quantifier"] = "universal_via_difference"
            for key in ("left", "right"):
                if key in node and isinstance(node[key], dict):
                    digest = shape_signature(node[key])[:40]
                    sig[f"{key}_scope"] = digest
            for value in node.values():
                scan(value, path)
        elif isinstance(node, list):
            for value in node:
                scan(value, path)

    scan(tree)
    return sig


def make_identifiers(family: str, depth: str, template_id: str,
                     tree: dict, params: dict, index: int) -> dict:
    return {
        "intent_id": f"PACS::{family}:{depth}:{template_id}:{index:04d}",
        "gold_tree_hash": gold_tree_hash(tree),
        "shape_signature": shape_signature(tree),
        "template_id": template_id,
        "surface_grammar_id": None,  # set per rendered channel
        "entity_anchor_signature": entity_anchor_signature(params),
        "logical_signature": logical_signature(tree),
    }
