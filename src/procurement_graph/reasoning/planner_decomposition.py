"""Decomposition-aware rule planner (Phase 2, planner side).

Recognises the query shapes the Phase-2 executor/decomposition can run but the base rule planner
cannot emit: the filter-only single-hop ops (min_max, top_k) and the org-anchored shapes
(exists, set_list, bridge semijoins, compare). Filter-only ops need no entity extraction and are
robust; org-anchored shapes extract the anchor name by span + verify it against the OrgResolver,
which is deliberately benchmark-tuned — the GENERAL path for arbitrary NL is the LLM planner. Falls
back to the base `RuleBasedDryRunPlanner` for count / sum / explicit-id factoid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .decomposition import Binding, DecompositionPlan, SubQuery
from .entity_resolution import resolve_confident_org
from .linking import SUM_PHRASE_RE, OrgResolver, link_question
from .models import CandidatePlan, QueryConstraint, RuntimeQuerySpec
from .planner import RULE_BASED_PLANNER_VERSION, RuleBasedDryRunPlanner

_STOP = (r"(?:\s+publish\w*|\s+won\b|\s+also\b|\s+than\b|\s+in\s+\d{4}|\s+under\s+cpv"
         r"|\s+for\s+(?:goods|services|works)|\s+for\s+cpv\b|\s+for\s+the\b|\s+awarded\b|\s+to\b"
         r"|\s+that\b|\s+who\b|\s+has\b|\s+have\b|[?.,]|$)")
# 'what <field phrase> is recorded ...' -> select_unique over that field (longest alias first)
_FIELD_PHRASES = (
    ("award signed date", "award_date_signed", "string"),
    ("signed date", "award_date_signed", "string"),
    ("award date", "award_date_signed", "string"),
    ("tender category", "tender_category", "string"),
    ("procurement category", "tender_category", "string"),
    ("category", "tender_category", "string"),
    ("contract value", "value_amount", "currency"),
    ("award value", "value_amount", "currency"),
    ("value", "value_amount", "currency"),
    ("tender title", "tender_title", "string"),
    ("title", "tender_title", "string"),
    ("buyer", "buyer_name", "string"),
    ("supplier", "supplier_name", "string"),
)
_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], 1)}


def _quoted(text: str) -> str:
    m = re.search(r'[\"“]([^\"”]+)[\"”]', text)
    return m.group(1).strip() if m else ""


def _parse_date(text: str) -> str:
    m = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if m:
        return m.group(0)
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", text)
    if m and m.group(2).lower() in _MONTHS:
        return f"{m.group(3)}-{_MONTHS[m.group(2).lower()]:02d}-{int(m.group(1)):02d}"
    return ""


def _parse_money(text: str) -> float | None:
    m = re.search(r"(?:gbp|£)?\s*([\d,.]+)\s*(million|billion|m|bn)?", text, re.I)
    if not m:
        return None
    num = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or "").lower()
    return num * (1e6 if unit in {"million", "m"} else 1e9 if unit in {"billion", "bn"} else 1)


def _spec(question, op, constraints, *, answer_field="", value_type="string", sort_field="",
          intent="x", metadata=None) -> RuntimeQuerySpec:
    return RuntimeQuerySpec(spec_id=_sid(question, op), question=question, intent=intent,
                            constraints=tuple(constraints), answer_operation=op, answer_field=answer_field,
                            answer_value_type=value_type, sort_field=sort_field, dedupe_key="contract_node_id",
                            requires_exhaustive_retrieval=True, metadata=metadata or {},
                            planner_version=RULE_BASED_PLANNER_VERSION)


def _sid(question, op):
    import hashlib
    return "runtime_" + hashlib.sha1(f"{op}|{question}".encode("utf-8")).hexdigest()[:12]


def _cand(spec=None, *, decomposition=None, confidence=0.75, rationale="", intent_unsupported=False):
    base = spec or RuntimeQuerySpec(spec_id="d", question="", intent="decomposition", constraints=(),
                                    answer_operation="count", answer_field="", answer_value_type="string")
    return CandidatePlan(plan_id=f"{base.spec_id}:plan0", query_spec=base, status="planned",
                         confidence=confidence, rationale=rationale, planner_source="rule_decomposition",
                         decomposition=decomposition)


@dataclass
class DecompositionAwarePlanner:
    org_resolver: OrgResolver | None = None
    fallback: Any = field(default_factory=RuleBasedDryRunPlanner)

    def plan(self, question: str) -> tuple[CandidatePlan, ...]:
        text = " ".join(str(question or "").split())
        low = text.lower()
        link = link_question(text, org_resolver=self.org_resolver)
        # A non-KG concept (bidders, SME status, evaluation criteria, ...) makes the question
        # unsupported regardless of surface shape -> mark it before any recognizer can misfire.
        if link.unsupported_reason:
            return self.fallback.plan(question)
        filters = tuple(link.constraints)  # year/category/cpv extracted even when the base planner would bail

        for recognizer in (self._min_max, self._top_k, self._numeric_threshold, self._date_relation,
                           self._field_equality, self._role_factoid, self._exists, self._set_list,
                           self._compare, self._bridge):
            candidate = recognizer(text, low, filters)
            if candidate is not None:
                return (candidate,)
        return self.fallback.plan(question)

    # --- boolean value predicates -------------------------------------------------------
    def _numeric_threshold(self, text, low, filters):
        # '(?!\s+cpv)' keeps 'under CPV 42997300' from being read as a threshold comparator
        comparator_match = re.search(r"\b(above|below|more than|less than|over|under)\b(?!\s+cpv)", low)
        if not SUM_PHRASE_RE.search(low) or not comparator_match:
            return None
        threshold = _parse_money(low[comparator_match.end():])
        if threshold is None:
            return None
        comparator = "gt" if comparator_match.group(1) in ("above", "more than", "over") else "lt"
        cons = tuple(filters) + (QueryConstraint("value_is_additive", "eq", True, visible_to_user=False),)
        spec = _spec(text, "predicate", cons, answer_field="value_amount", value_type="boolean", intent="boolean",
                     metadata={"predicate_subject": "sum", "comparator": comparator, "threshold": threshold, "threshold_type": "number"})
        return _cand(spec, rationale="total-value-vs-threshold -> predicate(sum)")

    def _date_relation(self, text, low, filters):
        if "signed" not in low or ("after" not in low and "before" not in low):
            return None
        title = _quoted(text)
        date = _parse_date(text)
        if not title or not date:
            return None
        comparator = "after" if "after" in low else "before"
        spec = _spec(text, "predicate", (QueryConstraint("tender_title", "eq", title),), value_type="boolean", intent="boolean",
                     metadata={"predicate_subject": "field", "predicate_field": "award_date_signed",
                               "comparator": comparator, "threshold": date, "threshold_type": "date"})
        return _cand(spec, rationale="signed-date-relation -> predicate(field)")

    def _field_equality(self, text, low, filters):
        title = _quoted(text)
        if not title or not low.startswith("was"):
            return None
        m = re.search(r"categori[sz]ed as (goods|services|works)", low)
        if m:
            field, val = "tender_category", m.group(1)
        elif re.search(r"cpv\s*(\d{6,8})", low):
            field, val = "tender_cpv_id", re.search(r"cpv\s*(\d{6,8})", low).group(1)
        elif re.search(r"published in (\d{4})", low):
            field, val = "release_year", re.search(r"published in (\d{4})", low).group(1)
        else:
            return None
        spec = _spec(text, "predicate", (QueryConstraint("tender_title", "eq", title),), value_type="boolean", intent="boolean",
                     metadata={"predicate_subject": "field", "predicate_field": field, "comparator": "eq",
                               "threshold": val, "threshold_type": "string"})
        return _cand(spec, rationale="field-equality -> predicate(field)")

    # --- filter-only single-hop (robust; no entity extraction) --------------------------
    def _min_max(self, text, low, filters):
        if re.search(r"\bhighest(?:[- ]value|[- ]valued| value|est)?\b|most expensive|largest value", low):
            op = "argmax"
        elif re.search(r"\blowest(?:[- ]value| value| non-zero)?\b|cheapest|smallest value", low):
            op = "argmin"
        else:
            return None
        if "contract" not in low and "notice" not in low:
            return None
        spec = _spec(text, op, filters, answer_field="contract_node_id", sort_field="value_amount", intent="min_max")
        return _cand(spec, rationale="highest/lowest-value contract -> argmax/argmin")

    def _top_k(self, text, low, filters):
        m = re.search(r"top\s+(\d+)\s+(buyers|suppliers)", low)
        if not m:
            return None
        gbf = "buyer_name" if m.group(2) == "buyers" else "supplier_name"
        metric = "sum" if "value" in low or "total" in low else "count"
        spec = _spec(text, "top_k", filters, intent="top_k",
                     metadata={"group_by_field": gbf, "metric": metric, "k": int(m.group(1)), "metric_field": "value_amount"})
        return _cand(spec, rationale="top-N ranking -> top_k")

    # --- org-anchored single-hop --------------------------------------------------------
    def _role_factoid(self, text, low, filters):
        """Who was the buyer / which supplier / what category for THE contract matching the filters.

        A singular-role question is a select_unique (scalar), not a distinct_set (list); the
        executor's uniqueness check still guards against genuinely ambiguous matches.
        """
        answer_field, value_type = "", "string"
        recorded = re.search(r"\bwhat\s+([\w \-]+?)\s+is\s+recorded\b", low)
        if not recorded and not SUM_PHRASE_RE.search(low):
            # terse noun lead: "Award signed date for contract notices published ..."
            recorded = re.match(r"^([\w &\-]+?)\s+(?:recorded\s+)?for\s+contract\s+notices\b", low)
        if recorded:
            phrase = " ".join(recorded.group(1).split())
            for alias, field, vtype in _FIELD_PHRASES:
                if phrase == alias or phrase.endswith(" " + alias):
                    answer_field, value_type = field, vtype
                    break
        if not answer_field:
            if re.search(r"\b(?:who|what|which)\s+(?:(?:was|is|were)\s+the\s+)?buyer\b", low) \
                    or re.search(r"^(?:the\s+)?buyer\s+(?:is\s+)?recorded\b", low):
                answer_field = "buyer_name"
            elif re.search(r"\b(?:who|what|which)\s+(?:(?:was|is)\s+the\s+)?supplier\b", low) \
                    or re.search(r"^(?:the\s+)?supplier\s+(?:is\s+)?recorded\b", low):
                answer_field = "supplier_name"
            elif re.search(r"\bwhat\s+is\s+the\s+(?:procurement\s+)?category\b", low):
                answer_field = "tender_category"
            else:
                return None
        cons = list(filters)
        # a role named in quotes ('the supplier "uk car service"') beats keyword-span extraction
        supplier = buyer = ""
        quoted_role = re.search(r'\b(supplier|buyer)\s*,?\s+["“]([^"”]+)["”]', text, re.I)
        if quoted_role:
            resolved = self._resolve_org(quoted_role.group(2).strip())
            if quoted_role.group(1).lower() == "supplier":
                supplier = resolved
            else:
                buyer = resolved
        supplier = supplier or self._org_after(text, r"awarded\s+to|contract\s+with|for\s+the\s+supplier|with|to")
        buyer = buyer or self._org_after(text, r"awarded\s+by|published\s+by|by")
        # if linking already constrained a role, do not add a second (possibly punctuation-variant)
        # value for the same role — two eq constraints on one field can only contradict.
        have = {c.field for c in cons}
        if supplier and "supplier_name" not in have:
            cons.append(QueryConstraint("supplier_name", "eq", supplier, source_text=supplier))
        if buyer and "buyer_name" not in have:
            cons.append(QueryConstraint("buyer_name", "eq", buyer, source_text=buyer))
        has_org = any(c.field in ("supplier_name", "buyer_name") for c in cons)
        if not (has_org or filters):
            return None  # nothing anchors the contract; leave it to the fallback/LLM
        spec = _spec(text, "select_unique", tuple(cons), answer_field=answer_field,
                     value_type=value_type, intent="factoid")
        return _cand(spec, rationale=f"singular role question -> select_unique({answer_field})")

    def _exists(self, text, low, filters):
        if not re.search(r"\bdid\b.+\bany\b", low) and "is there" not in low and "are there any" not in low:
            return None
        org = self._org_after(text, r"did")
        cons = list(filters)
        if org:
            cons.append(QueryConstraint("buyer_name", "eq", org, source_text=org))
        spec = _spec(text, "exists", tuple(cons), value_type="boolean", intent="boolean")
        return _cand(spec, rationale="did ... any -> exists")

    def _set_list(self, text, low, filters):
        if not re.search(r"\b(which|list(?:\s+the)?(?:\s+distinct)?)\s+suppliers?\b", low):
            return None
        org = self._org_after(text, r"from|by")
        cons = list(filters)
        if org:
            cons.append(QueryConstraint("buyer_name", "eq", org, source_text=org))
        spec = _spec(text, "distinct_set", tuple(cons), answer_field="supplier_name", intent="set_list")
        return _cand(spec, rationale="which/list suppliers -> distinct_set")

    # --- compare (two independent sub-counts) -------------------------------------------
    def _compare(self, text, low, filters):
        if not re.search(r"\bdid\b.+\bmore\b.+\bthan\b", low):
            return None
        left = self._org_after(text, r"did")
        right = self._org_after(text, r"than")
        if not (left and right):
            return None
        year = [c for c in filters if c.field == "release_year"]
        base = tuple(year)
        a = SubQuery("a", _spec(text, "count", base + (QueryConstraint("buyer_name", "eq", left),)))
        b = SubQuery("b", _spec(text, "count", base + (QueryConstraint("buyer_name", "eq", right),)))
        plan = DecompositionPlan(_sid(text, "compare"), text, (a, b), combine="compare_gt", final_steps=("a", "b"))
        spec = _spec(text, "compare", base, intent="comparison")
        return _cand(spec, decomposition=plan, rationale="X more than Y -> compare (two sub-counts)")

    # --- bridge semijoins ---------------------------------------------------------------
    def _bridge(self, text, low, filters):
        if re.search(r"suppliers (?:who|that) also worked with", low):
            resolve, into, anchor_field, kw = "suppliers_of_buyer", "supplier_name", "buyer_name", r"worked with"
        elif re.search(r"suppliers (?:who|that) also won", low):
            resolve, into, anchor_field, kw = "suppliers_of_buyer", "supplier_name", "buyer_name", r"from"
        elif re.search(r"buyers (?:who|that) have awarded", low):
            resolve, into, anchor_field, kw = "buyers_of_supplier", "buyer_name", "supplier_name", r"to"
        elif "cpv codes that" in low and "has used" in low:
            resolve, into, anchor_field, kw = "cpvs_of_buyer", "tender_cpv_id", "buyer_name", r"that"
        else:
            return None
        anchor = self._org_after(text, kw)
        if not anchor:
            return None
        op = "sum" if ("total value" in low or "total contract value" in low) else "count"
        hop1 = SubQuery("h1", _spec(text, "count", (QueryConstraint(anchor_field, "eq", anchor),)),
                        kind="entity_set", emit_field=into)
        hop2_spec = _spec(text, "sum", filters, answer_field="value_amount", value_type="currency") if op == "sum" \
            else _spec(text, "count", filters)
        hop2 = SubQuery("ans", hop2_spec, binds=(Binding("h1", into),))
        plan = DecompositionPlan(_sid(text, "bridge"), text, (hop1, hop2), final_steps=("ans",))
        spec = _spec(text, op, filters, answer_field="value_amount" if op == "sum" else "",
                     value_type="currency" if op == "sum" else "string", intent="bridge")
        return _cand(spec, decomposition=plan, rationale=f"bridge {resolve}")

    # --- org span extraction + resolver verification ------------------------------------
    def _org_after(self, text: str, keyword_pat: str) -> str:
        m = re.search(rf"\b(?:{keyword_pat})\b", text, re.I)
        if not m:
            return ""
        tail = text[m.end():]
        stop = re.search(_STOP, tail, re.I)
        span = (tail[: stop.start()] if stop else tail).strip(" .?,")
        span = re.sub(r"^(a contract|contract|contracts)\s+(from|to)\s+", "", span, flags=re.I).strip()
        # org names can END in the stop punctuation ("EDS Ltd." vs a distinct org "EDS Ltd") --
        # try the dot-restored surface first so the exact variant wins over its truncation.
        if stop and tail[stop.start(): stop.start() + 1] == ".":
            restored = self._resolve_org(span + ".")
            if restored:
                return restored
        return self._resolve_org(span)

    def _resolve_org(self, span: str) -> str:
        if not span:
            return ""
        if self.org_resolver is None:
            return span
        for candidate_span in (span, span.rstrip("s")):
            resolved = resolve_confident_org(self.org_resolver, candidate_span)
            if resolved.ok and resolved.hit is not None:
                return resolved.hit.linked_id
        return ""


@dataclass
class HybridPlanner:
    """Rule-first, LLM-on-ambiguous (a cascade, not either/or).

    The rule `DecompositionAwarePlanner` handles the templated majority cheaply and precisely; when it
    cannot map the question (status 'ambiguous' — novel phrasing, an anchor it can't extract, an
    unrecognised shape), escalate to the LLM, which is strong at NLU + entity linking + decomposition.
    The LLM's plan is still grounded and executed deterministically (generate-then-ground; the shared
    executor is the sole answer authority). Unsupported stays unsupported (not escalated).

    Known blind spot (measured on the multilevel benchmark): a rule plan that MISFIRES confidently
    (status 'planned', wrong spec) never consults the LLM. `VerifyingHybridPlanner` closes it.
    """

    rule: Any
    llm: Any = None

    def plan(self, question: str) -> tuple[CandidatePlan, ...]:
        rule_cands = tuple(self.rule.plan(question))
        top = max(rule_cands, key=lambda c: c.confidence) if rule_cands else None
        if self.llm is None or top is None or top.status != "ambiguous":
            return rule_cands
        llm_cands = tuple(self.llm.plan(question))
        llm_top = max(llm_cands, key=lambda c: c.confidence) if llm_cands else None
        if llm_top is not None and llm_top.status == "planned":
            return llm_cands
        return rule_cands


@dataclass
class VerifyingHybridPlanner:
    """Conservative verify-then-escalate cascade.

    Policy (post-mortem-driven; see scripts/analyze_escalation.py and the escalation_report):
      - a rule plan that passes deterministic verification is ACCEPTED — no LLM call;
      - escalation happens ONLY on hard verification failure: the spec does not ground, or its
        probe execution finds no evidence at all (`no_results`). False/0 answers and
        `multiple_answers` are legitimate outcomes (oracle=False predicates; underdetermined
        factoids where abstention is correct), NOT escalation triggers — the earlier aggressive
        triggers caused every measured L1 regression and most wasted LLM calls;
      - a rule status of 'ambiguous' escalates exactly like the classic `HybridPlanner`;
      - the LLM plan is adopted only if it grounds and probes without hard failure; otherwise the
        rule plan (and its deterministic, auditable outcome) stands.

    Escalations are recorded on the adopted candidate (`planner_source: ...+escalated`,
    `warnings: escalated_because:*`), so the LLM contribution stays a per-question intervention
    ledger. Probing is read-only; the LLM never produces answers.
    """

    rule: Any
    backend: Any
    llm: Any = None
    probe_decompositions: bool = False  # bridge probes = full multi-hop runs; skip by default

    def plan(self, question: str) -> tuple[CandidatePlan, ...]:
        from dataclasses import replace as _replace

        rule_cands = tuple(self.rule.plan(question))
        top = max(rule_cands, key=lambda c: c.confidence) if rule_cands else None
        if self.llm is None or top is None:
            return rule_cands
        if top.status == "unsupported":
            return rule_cands  # unsupported never escalates: abstention is the correct output

        if top.status != "planned":
            # classic hybrid path: the rule layer itself says it cannot map the question
            needs_llm, why = True, "rule_ambiguous"
        else:
            needs_llm, why = self._probe(question, top)
        if not needs_llm:
            return rule_cands

        llm_cands = tuple(self.llm.plan(question))
        llm_top = max(llm_cands, key=lambda c: c.confidence) if llm_cands else None
        if llm_top is None or llm_top.status != "planned":
            return rule_cands
        if top.status == "planned":
            llm_bad, _llm_why = self._probe(question, llm_top)
            if llm_bad:
                return rule_cands  # fall back to the verified-able rule plan / its auditable failure
        annotated = _replace(llm_top,
                             planner_source=f"{llm_top.planner_source}+escalated",
                             warnings=llm_top.warnings + (f"escalated_because:{why}",),
                             rationale=f"{llm_top.rationale} (escalated: {why})")
        return (annotated,) + tuple(c for c in llm_cands if c is not llm_top)

    def _probe(self, question: str, candidate: CandidatePlan) -> tuple[bool, str]:
        """Read-only HARD verification of a planned candidate. Returns (should_escalate, reason).

        Hard failure = the plan cannot be grounded, or matches nothing at all. Valid-but-
        surprising outcomes (0, False, multiple candidate answers) never escalate.
        """
        from .executor import execute_query_spec
        from .grounding import ground_spec
        from .verifier import backend_fields

        if candidate.decomposition is not None and not self.probe_decompositions:
            return False, "decomposition_not_probed"
        grounded = ground_spec(candidate.query_spec, allowed_fields=frozenset(backend_fields(self.backend)))
        if not grounded.ok:
            return True, f"grounding_failed:{grounded.reason}"
        result = execute_query_spec(self.backend, grounded.spec)
        if result.status == "no_results":
            return True, "probe_no_results"
        return False, "probe_passed"


__all__ = ["DecompositionAwarePlanner", "HybridPlanner", "VerifyingHybridPlanner"]
