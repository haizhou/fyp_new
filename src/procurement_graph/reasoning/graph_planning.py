"""Graph-plan execution: resolve A/B/C intermediate targets before answering.

The LLM emits a semantic `understanding_network` plus a `graph_plan`. This module turns that graph
plan into a deterministic execution plan and runs each variable in reasoning order. The existing
RuntimeQuerySpec executor is used as the low-level table operation for each variable; it is no
longer the planner-facing representation.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field as dc_field
from typing import Any

from .decomposition import Binding, DecompositionPlan, SubQuery, execute_decomposition
from .entity_resolution import resolve_confident_org
from .executor import execute_query_spec
from .grounding import CANONICAL_FIELDS, ground_spec
from .models import ExecutionResult, QueryConstraint, RuntimeQuerySpec
from .verifier import RuntimeQueryBackend, backend_fields


@dataclass(frozen=True)
class GraphVariableSpec:
    var_id: str
    kind: str
    description: str = ""
    role: str = ""
    constraints: tuple[QueryConstraint, ...] = ()
    depends_on: tuple[str, ...] = ()
    emit_field: str = ""
    low_level_spec: RuntimeQuerySpec | None = None


@dataclass(frozen=True)
class GraphRelationSpec:
    source: str
    target: str
    relation: str = ""
    direction: str = "unknown"
    surface_evidence: str = ""
    bind_field: str = ""


@dataclass(frozen=True)
class GraphOperationUnitSpec:
    unit_id: str
    operation_unit: str
    inputs: tuple[str, ...] = ()
    output: str = ""
    description: str = ""
    uses: dict[str, Any] = dc_field(default_factory=dict)


@dataclass(frozen=True)
class GraphReturnSpec:
    operation: str
    input_id: str
    field: str = ""
    guards: tuple[str, ...] = ()
    parameters: dict[str, Any] = dc_field(default_factory=dict)


@dataclass(frozen=True)
class GraphExecutionPlan:
    plan_id: str
    question: str
    variables: tuple[GraphVariableSpec, ...]
    relations: tuple[GraphRelationSpec, ...]
    operation_units: tuple[GraphOperationUnitSpec, ...]
    return_spec: GraphReturnSpec
    reasoning_order: tuple[str, ...]
    understanding_network: dict[str, Any] = dc_field(default_factory=dict)
    raw_graph_plan: dict[str, Any] = dc_field(default_factory=dict)


@dataclass(frozen=True)
class GraphVariableTrace:
    var_id: str
    kind: str
    status: str
    output_size: int = 0
    depends_on: tuple[str, ...] = ()
    constraints: tuple[dict[str, Any], ...] = ()
    emitted_field: str = ""
    producer_unit: str = ""
    operation_unit: str = ""
    answer: Any = None
    reason: str = ""
    failure_kind: str = ""
    execution_level: int = 0


@dataclass(frozen=True)
class GraphExecutionResult:
    plan: GraphExecutionPlan
    status: str
    answer: Any = None
    variable_traces: tuple[GraphVariableTrace, ...] = ()
    low_level_result: ExecutionResult | None = None
    evidence_ids: tuple[str, ...] = ()
    reason: str = ""
    metrics: dict[str, Any] = dc_field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "passed"


_SLOT_FIELDS = {"buyer": "buyer_name", "supplier": "supplier_name", "year": "release_year",
                "year_range": "release_year", "years": "release_year",
                "cpv": "tender_cpv_id", "category": "tender_category", "title": "tender_title",
                "date": "award_date_signed", "has_signed_date": "has_award_signed_date"}


def variables_map(graph: dict[str, Any]) -> dict[str, Any]:
    """Normalise graph variables to a var_id->spec dict, accepting either the strict-schema ARRAY
    form ([{var_id, ...}]) or the legacy dict form ({A: {...}}). Arrays are required by the
    provider's strict json_schema (arbitrary-key maps are unsupported)."""
    variables = graph.get("variables")
    if isinstance(variables, dict):
        return {str(k): v for k, v in variables.items() if isinstance(v, dict)}
    if isinstance(variables, list):
        out: dict[str, Any] = {}
        for index, item in enumerate(variables):
            if isinstance(item, dict):
                out[str(item.get("var_id") or item.get("id") or index)] = item
        return out
    return {}


# Structure-encoding var_id convention: a variable's id carries its parent, so the dependency DAG
# is readable from a FLAT array (json_schema-friendly) and depends_on can be derived deterministically.
#   a1, a2            feed the answer directly
#   b1a1, b2a1        feed a1 (parent token = a1)
#   c1b1              feeds b1 (parent token = b1)
# id = <self-token><parent-token>, each token = one letter + digits; parent empty => feeds answer.
_VARID_RE = re.compile(r"^([a-z]\d+)([a-z]\d+)?", re.I)


def _parse_varid(var_id: str) -> tuple[str, str]:
    match = _VARID_RE.match(str(var_id).strip())
    if not match:
        return str(var_id), ""
    return match.group(1).lower(), (match.group(2) or "").lower()


def derive_depends_on(variables: dict[str, Any]) -> dict[str, list[str]]:
    """Derive depends_on from the id convention: a node with parent token P means the node whose
    SELF token is P consumes this node (P depends_on this). Ambiguous/unknown tokens are skipped
    (the emitted depends_on then carries the edge)."""
    by_self: dict[str, str] = {}
    for var_id in variables:
        self_tok, _ = _parse_varid(var_id)
        by_self.setdefault(self_tok, var_id)
    deps: dict[str, list[str]] = {var_id: [] for var_id in variables}
    for var_id in variables:
        self_tok, parent_tok = _parse_varid(var_id)
        if not parent_tok:
            continue
        parent_vid = by_self.get(parent_tok)
        if parent_vid and parent_vid != var_id and var_id not in deps[parent_vid]:
            deps[parent_vid].append(var_id)  # parent consumes this node
    return deps


_NON_ZERO_RE = re.compile(r"non[\s-]?zero", re.I)

# Singular interrogative / plural-answer cues for T10 (local copies: typed_planning imports this
# module, so the shared definitions cannot live there without a cycle).
_SINGULAR_Q_RE = re.compile(
    r"\b(which|what|who)\s+(?:is\s+|was\s+)?(?:the\s+)?(?:\w+\s+){0,2}?"
    r"(buyer|supplier|contracting authority|organisation|organization|company|winner|authority)\b")
_PLURAL_Q_CUE_RE = re.compile(
    r"\b(all|list|every|each|buyers|suppliers|authorities|organisations|organizations|"
    r"companies|winners)\b")


def normalise_graph_plan(question: str, graph: dict[str, Any]) -> dict[str, Any]:
    """Deterministic structural clean-up of a raw LLM graph plan, run BEFORE consistency/compile.

    Each transform converts an observed, mechanically-recognisable planner idiom into the
    executable structure it obviously means (never guessing semantics):
      T1  a filter whose VALUE names another variable is a dependency edge, not a constraint
      T2  an empty buyer/supplier filter is never executable — the bind (or nothing) carries it
      T3  zero-valued filters on 'non-zero' questions become return.exclude_zero
      T4  compare with a missing `left` compares its `input`
      T7  a same-role entity_set chained on an entity_set is a RE-FILTER of one set, not a second
          hop: its record-level filters fold into the root record_set
    """
    vars_map = variables_map(graph)
    if not vars_map:
        return graph
    out_vars: dict[str, dict[str, Any]] = {
        vid: {**var, "filters": [dict(f) for f in (var.get("filters") or ()) if isinstance(f, dict)],
              "depends_on": [str(d) for d in (var.get("depends_on") or ())]}
        for vid, var in vars_map.items()
    }
    self_tokens = {_parse_varid(vid)[0]: vid for vid in out_vars}
    non_zero_question = bool(_NON_ZERO_RE.search(question or ""))
    ret = dict(graph.get("return") or {})
    quoted_spans = {span.casefold() for span in re.findall(r'"([^"]{2,160})"', question or "")}

    for vid, var in out_vars.items():
        kept: list[dict[str, Any]] = []
        for item in var["filters"]:
            value = item.get("value", item.get("surface"))
            sval = str(value).strip() if value is not None else ""
            slot = str(item.get("slot") or "")
            if slot == "date" and re.fullmatch(r"20[2-3]\d", sval):
                item["slot"] = slot = "year"
            if slot == "category":
                normalised_category = _normalise_category(sval)
                if normalised_category is None:
                    # Wrapper phrases such as "matching procurement records" describe evaluation
                    # scope, not a KG category value. Keeping them as category filters zeros results.
                    continue
                item["value"] = normalised_category
                sval = normalised_category
            if slot == "title" and sval and sval.casefold() not in quoted_spans \
                    and not (sval in out_vars or sval in self_tokens):
                # Same rule as the typed path: a title value the question does not quote is L2
                # wrapper filler ("matching procurement records only"), not a KG title — drop it.
                continue
            if sval and sval != vid and (sval in out_vars or sval in self_tokens):
                dep = sval if sval in out_vars else self_tokens[sval]
                if dep != vid and dep not in var["depends_on"]:
                    var["depends_on"].append(dep)   # T1
                # the dropped filter's SLOT names the bind dimension (cpv IN b1 => b1 is a CPV
                # set); backfill the referenced variable's role so emit/bind derive correctly
                dep_var = out_vars.get(dep)
                if dep_var is not None and str(dep_var.get("role") or "none") in ("", "none") \
                        and slot in ("buyer", "supplier", "cpv", "category"):
                    dep_var["role"] = slot
                continue
            if not sval:
                continue                             # T2: empty-valued filter of ANY slot is never
                                                     # executable — bind or padding, drop it
                                                     # (grok pads unused enum slots with "")
            if sval in ("0", "0.0") and non_zero_question:
                ret["exclude_zero"] = True           # T3
                continue
            kept.append(item)
        var["filters"] = kept
        # T8: an `entity` variable with a dependency and no literal value can only run as a
        # DERIVED set (the entity branch grounds literal values, it never queries the KG).
        # Singularity only matters when the variable itself is the returned answer.
        if str(var.get("kind")) == "entity" and var["depends_on"] and not var["filters"] \
                and str(ret.get("input") or "") != vid:
            var["kind"] = "entity_set"

    # T9: anchor echo — a variable that RECEIVES a bind on an org slot must not also carry a
    # literal eq filter on that slot naming the bind source's own anchor. On "suppliers who
    # worked with ORG" bridges the planner echoes ORG into the supplier slot while binding the
    # derived supplier set; the intersection is the anchor itself or empty. The bind carries the
    # meaning — drop the echoed literal.
    _org_slots = ("buyer", "supplier")
    relations_raw = [rel for rel in (graph.get("relations") or ()) if isinstance(rel, dict)]
    for vid, var in out_vars.items():
        bind_sources: dict[str, list[dict[str, Any]]] = {}
        for rel in relations_raw:
            if str(rel.get("to")) == vid and str(rel.get("bind_field") or "") in _org_slots:
                src = out_vars.get(str(rel.get("from")))
                if src is not None:
                    bind_sources.setdefault(str(rel.get("bind_field")), []).append(src)
        for dep in var["depends_on"]:
            src = out_vars.get(dep)
            if src is not None and str(src.get("kind")) == "entity_set" \
                    and str(src.get("role") or "") in _org_slots:
                bind_sources.setdefault(str(src.get("role")), []).append(src)
        if not bind_sources:
            continue
        kept_filters: list[dict[str, Any]] = []
        for item in var["filters"]:
            slot = str(item.get("slot") or "")
            value = str(item.get("value", item.get("surface")) or "").strip().casefold()
            anchors = {str(f.get("value", f.get("surface")) or "").strip().casefold()
                       for src in bind_sources.get(slot, ())
                       for f in (src.get("filters") or ()) if isinstance(f, dict)}
            if value and value in anchors:
                continue
            kept_filters.append(item)
        var["filters"] = kept_filters

    # T7: fold C(entity_set role R, deps=[B]) with B(entity_set role R) — push C's record-level
    # filters into B's root record_set and redirect C -> B.
    redirect: dict[str, str] = {}
    for vid, var in list(out_vars.items()):
        deps = var["depends_on"]
        if str(var.get("kind")) != "entity_set" or len(deps) != 1:
            continue
        parent = out_vars.get(deps[0])
        if parent is None or str(parent.get("kind")) != "entity_set" \
                or str(parent.get("role") or "") != str(var.get("role") or ""):
            continue
        root_id, seen = deps[0], set()
        while root_id in out_vars and root_id not in seen:
            seen.add(root_id)
            node = out_vars[root_id]
            if str(node.get("kind")) == "record_set" or len(node["depends_on"]) != 1:
                break
            root_id = node["depends_on"][0]
        root = out_vars.get(root_id)
        if root is not None and str(root.get("kind")) == "record_set":
            root["filters"].extend(var["filters"])
            redirect[vid] = deps[0]
    if redirect:
        def _redir(name: str) -> str:
            while name in redirect:
                name = redirect[name]
            return name
        for vid in redirect:
            out_vars.pop(vid, None)
        for var in out_vars.values():
            var["depends_on"] = list(dict.fromkeys(
                _redir(dep) for dep in var["depends_on"] if _redir(dep) in out_vars))
        if ret.get("input"):
            ret["input"] = _redir(str(ret["input"]))

    # T11/T12: a compare SIDE that is a scalar variable holding one literal (an amount or a
    # date) is a THRESHOLD, not a query — executing it as a record filter counts records that
    # happen to share the value (0944 compared 1 title-match against "359 records signed on
    # 2025-05-01"). Fold the literal into the return side and drop the variable.
    if str(ret.get("operation") or "").casefold() in ("compare", "comparison"):
        for side in ("left", "right"):
            side_ref = str(ret.get(side) or "")
            side_var = out_vars.get(side_ref)
            if side_var is None or str(side_var.get("kind")) != "scalar" or side_var.get("depends_on"):
                continue
            filters = side_var.get("filters") or []
            if len(filters) != 1:
                continue
            slot = str(filters[0].get("slot") or "")
            literal = str(filters[0].get("value", filters[0].get("surface")) or "").strip()
            if not literal or slot not in ("date", "amount", "value", "threshold"):
                continue
            if slot == "date":
                from .grounding import _parse_written_date
                literal = _parse_written_date(literal) or literal
            ret[side] = literal
            consumed_elsewhere = any(side_ref in (v.get("depends_on") or ()) for v in out_vars.values())
            if not consumed_elsewhere and str(ret.get("input") or "") != side_ref:
                out_vars.pop(side_ref, None)

    # T10: a SINGULAR interrogative ("which buyer ...") answered with distinct_set launders the
    # uniqueness contract — the set op never trips multiple_answers, so an ambiguous question
    # gets a 30-item "answer". Rewrite to select: run-time uniqueness then adjudicates (one match
    # answers normally; several become multiple_answers -> clarify). Rewriting instead of
    # rejecting keeps legitimately-unique cases (2329) alive.
    if str(ret.get("operation") or "").casefold() in ("distinct_set", "distinct", "set") \
            and _SINGULAR_Q_RE.search(" ".join((question or "").casefold().split())) \
            and not _PLURAL_Q_CUE_RE.search(" ".join((question or "").casefold().split())):
        ret["operation"] = "select"

    if str(ret.get("operation") or "").casefold() in ("compare", "comparison") \
            and not ret.get("left") and ret.get("input"):
        ret["left"] = ret["input"]                   # T4
    if non_zero_question and str(ret.get("operation") or "").casefold() in (
            "argmin", "argmax", "min", "max"):
        ret["exclude_zero"] = True                   # T3 (question-cued, filter or not)

    return {**graph,
            "variables": [{**var, "var_id": vid} for vid, var in out_vars.items()],
            "return": ret}


def compile_graph_plan(question: str, payload: dict[str, Any], *, org_resolver: Any = None,
                       _use_naming_deps: bool = True) -> tuple[GraphExecutionPlan | None, str]:
    """Compile LLM graph JSON into a deterministic graph execution plan."""
    graph = payload.get("graph_plan") if isinstance(payload.get("graph_plan"), dict) else payload
    if not isinstance(graph, dict):
        return None, "missing_graph_plan"
    graph = normalise_graph_plan(question, graph)
    ret = graph.get("return") if isinstance(graph.get("return"), dict) else {}
    variables_raw = variables_map(graph)
    if not variables_raw:
        return None, "missing_graph_variables"
    operation = str(ret.get("operation") or "").strip()
    input_id = str(ret.get("input") or "").strip()
    if not operation or not input_id:
        return None, "missing_graph_return"

    id_dependencies = derive_depends_on(variables_raw) if _use_naming_deps else {}
    # The id convention reads "b1a1 feeds a1", but planners routinely use the same suffix for
    # provenance ("b1a1 is derived FROM a1") and then assert the true dataflow in depends_on.
    # When both directions are claimed, the explicit edge is the executable one — drop the
    # naming-derived reverse edge instead of compiling a 2-cycle.
    explicit_deps = {vid: {str(dep) for dep in (raw.get("depends_on") or ())}
                     for vid, raw in variables_raw.items() if isinstance(raw, dict)}
    id_dependencies = {parent: [child for child in children
                                if parent not in explicit_deps.get(child, ())]
                       for parent, children in id_dependencies.items()}
    # A relation "from S to T" IS a dataflow edge: T consumes S's output. Binds are applied per
    # depends_on entry, so a relation without a matching dependency would silently never bind
    # (nano's relation-style bridges executed unfiltered).
    relation_deps: dict[str, list[str]] = {}
    for raw_rel in (graph.get("relations") or ()):
        if not isinstance(raw_rel, dict):
            continue
        rel_src, rel_dst = str(raw_rel.get("from") or ""), str(raw_rel.get("to") or "")
        if rel_src and rel_dst and rel_src != rel_dst \
                and rel_src in variables_raw and rel_dst in variables_raw \
                and rel_dst not in explicit_deps.get(rel_src, ()):
            # planners write relation directions loosely; when the REVERSE edge is explicit in
            # depends_on, the explicit dataflow wins (same rule as the naming convention).
            relation_deps.setdefault(rel_dst, []).append(rel_src)
    variables: list[GraphVariableSpec] = []
    for var_id, raw in variables_raw.items():
        if not isinstance(raw, dict):
            continue
        constraints, reason = _constraints_from_graph_variable(raw, org_resolver=org_resolver)
        if reason:
            return None, f"{reason}:{var_id}"
        variables.append(GraphVariableSpec(
            var_id=str(var_id),
            kind=str(raw.get("kind") or ""),
            description=str(raw.get("description") or ""),
            role=str(raw.get("role") or ""),
            constraints=tuple(constraints),
            depends_on=tuple(_merge_dependencies(id_dependencies.get(str(var_id), ()),
                                                  list(raw.get("depends_on") or ())
                                                  + relation_deps.get(str(var_id), []))),
            emit_field=_emit_field_for_variable(raw),
        ))
    relations = tuple(_relation_from_raw(raw) for raw in (graph.get("relations") or ()) if isinstance(raw, dict))
    operation_units = tuple(
        _operation_unit_from_raw(raw)
        for raw in (graph.get("operation_units") or (payload.get("understanding_network") or {}).get("procedure") or ())
        if isinstance(raw, dict)
    )
    reasoning_order = tuple(str(item) for item in (
        (payload.get("understanding_network") or {}).get("reasoning_order")
        or graph.get("reasoning_order")
        or _default_reasoning_order({v.var_id: v for v in variables}, operation, input_id, ret) + ["answer"]
    ))
    if input_id not in {v.var_id for v in variables} and _runtime_op(operation) != "compare":
        return None, f"unknown_return_input:{input_id}"
    return_spec = GraphReturnSpec(operation=operation, input_id=input_id, field=str(ret.get("field") or ""),
                                  guards=tuple(str(g) for g in (ret.get("guards") or ())),
                                  parameters={k: v for k, v in ret.items()
                                              if k not in {"operation", "input", "field", "guards"}})
    plan_id = "graph_" + re.sub(r"\W+", "", operation.casefold())[:24]
    plan = GraphExecutionPlan(plan_id=plan_id, question=question, variables=tuple(variables),
                              relations=relations, operation_units=operation_units, return_spec=return_spec,
                              reasoning_order=reasoning_order,
                              understanding_network=payload.get("understanding_network") or {},
                              raw_graph_plan=graph)
    plan = _attach_low_level_specs(plan)
    # Grounding check covers the WHOLE program, not just the slice reachable from return.input.
    # Variables inside the execution slice surface their own grounding failures at run time, but a
    # DANGLING variable carrying an ungroundable literal ("invoice or payment" as a date) is how a
    # planner smuggles an unsupported concept past execution — its failure reason must fail the
    # plan (feedback/abstain), not vanish with the never-executed variable.
    executed = {item for item in plan.reasoning_order if item != "answer"}
    for var in plan.variables:
        if var.var_id in executed or var.low_level_spec is None:
            continue
        check = ground_spec(var.low_level_spec)
        if not check.ok:
            return None, f"ungroundable_variable:{var.var_id}:{check.reason}"
    # DAG validity is an intent-program property, not an executor outcome: an unknown dependency
    # or a cycle must fail COMPILE (structured, repairable feedback), never surface downstream as
    # an opaque executor error. execute_graph_plan keeps its own check for hand-built plans.
    _, level_reason = _execution_levels(plan)
    if level_reason:
        if _use_naming_deps and "cycle" in level_reason:
            # Second-chance compile: the naming-convention edges may be self-contradictory in
            # this sample; retry with explicit/relation edges only before failing the plan.
            retry_plan, retry_reason = compile_graph_plan(question, payload,
                                                          org_resolver=org_resolver,
                                                          _use_naming_deps=False)
            if retry_plan is not None:
                return retry_plan, retry_reason
        return None, f"invalid_graph_structure:{level_reason}"
    # A bind SOURCE with no filters and no dependencies is the whole dimension (every supplier in
    # the KG) — binding on it silently deletes the question's own constraints and answers a
    # different, universe-scoped question (the unanswerable_1161 hallucination). A universe set
    # that IS the returned answer stays legal ("how many distinct suppliers are there?").
    consumed = {dep for var in plan.variables for dep in var.depends_on}
    for var in plan.variables:
        if var.var_id in consumed and var.kind in {"entity_set", "node_set"} \
                and not var.constraints and not var.depends_on:
            return None, f"unconstrained_bind_source:{var.var_id}"
    return plan, ""


def _merge_dependencies(primary: Any, fallback: Any) -> list[str]:
    """Use structure-encoded var_id dependencies first, with explicit depends_on as a fallback."""
    merged: list[str] = []
    for dep in tuple(primary or ()) + tuple(fallback or ()):
        text = str(dep)
        if text and text not in merged:
            merged.append(text)
    return merged


def execute_graph_plan(backend: RuntimeQueryBackend, plan: GraphExecutionPlan, *, max_workers: int = 4) -> GraphExecutionResult:
    """Execute graph variables in order, then aggregate the requested answer."""
    allowed = frozenset(backend_fields(backend)) or CANONICAL_FIELDS
    traces: list[GraphVariableTrace] = []
    outputs: dict[str, Any] = {}
    levels, level_reason = _execution_levels(plan)
    if level_reason:
        return GraphExecutionResult(plan, "error", reason=level_reason, metrics={"failure_kind": "syntax_error"})
    for level_index, level in enumerate(levels):
        if max_workers > 1 and len(level) > 1:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(level))) as pool:
                futures = {pool.submit(_execute_variable, backend, allowed, plan, var_id, outputs, level_index): var_id
                           for var_id in level}
                level_results = [future.result() for future in as_completed(futures)]
            level_results.sort(key=lambda item: plan.reasoning_order.index(item[0]))
        else:
            level_results = [_execute_variable(backend, allowed, plan, var_id, outputs, level_index)
                             for var_id in level]
        for var_id, trace, value, low_result in level_results:
            traces.append(trace)
            if trace.status == "passed":
                outputs[var_id] = value
                continue
            status = trace.status
            return GraphExecutionResult(plan, status, variable_traces=tuple(traces),
                                        low_level_result=low_result,
                                        reason=trace.reason or f"variable {var_id!r}: {status}",
                                        metrics={"failure_kind": trace.failure_kind,
                                                 "failed_variable": var_id,
                                                 "failed_operation_unit": trace.operation_unit})

    if _runtime_op(plan.return_spec.operation) == "compare":
        answer, reason = _compare_outputs(plan, outputs)
        if reason:
            return GraphExecutionResult(plan, "error", variable_traces=tuple(traces), reason=reason)
        return GraphExecutionResult(plan, "passed", answer=answer, variable_traces=tuple(traces),
                                    metrics={"variables": len(traces), "combine": "compare",
                                             "execution_levels": len(levels)})

    terminal = _terminal_spec(plan, outputs)
    if terminal is None:
        return GraphExecutionResult(plan, "unsupported", variable_traces=tuple(traces),
                                    reason="could not build terminal graph answer spec")
    grounded = ground_spec(terminal, allowed_fields=allowed)
    if not grounded.ok:
        return GraphExecutionResult(plan, "unsupported", variable_traces=tuple(traces), reason=grounded.reason)
    result = execute_query_spec(backend, grounded.spec)
    evidence_ids = tuple(result.evidence.evidence_ids) if result.evidence else ()
    return GraphExecutionResult(plan, result.status, answer=result.answer if result.passed else None,
                                variable_traces=tuple(traces), low_level_result=result,
                                evidence_ids=evidence_ids,
                                reason="" if result.passed else result.status,
                                metrics={"variables": len(traces), "execution_levels": len(levels),
                                         "failure_kind": "" if result.passed else _failure_kind_for_status(result.status)})


def graph_to_decomposition(plan: GraphExecutionPlan) -> DecompositionPlan | None:
    """Best-effort adapter for existing pipeline trace path; graph_executor is preferred."""
    terminal_var = _variable(plan, plan.return_spec.input_id)
    if terminal_var is None or terminal_var.low_level_spec is None:
        return None
    steps: list[SubQuery] = []
    bindings: list[Binding] = []
    for dep_id in terminal_var.depends_on:
        dep = _variable(plan, dep_id)
        if dep is None or dep.low_level_spec is None or not dep.emit_field:
            continue
        steps.append(SubQuery(dep.var_id, dep.low_level_spec, kind="entity_set", emit_field=dep.emit_field))
        bind_field = _bind_field_for_dependency(plan, terminal_var, dep)
        bindings.append(Binding(dep.var_id, bind_field))
    steps.append(SubQuery("answer", terminal_var.low_level_spec, binds=tuple(bindings)))
    return DecompositionPlan(plan.plan_id, plan.question, tuple(steps), final_steps=("answer",))


def _attach_low_level_specs(plan: GraphExecutionPlan) -> GraphExecutionPlan:
    variables = []
    for var in plan.variables:
        unit = _producer_unit(plan, var.var_id)
        op = _runtime_op_for_variable(plan, var, unit)
        answer_field = ""
        value_type = "string"
        sort_field = ""
        if op == "sum":
            answer_field, value_type = "value_amount", "currency"
        elif op in {"distinct_set", "select_unique"}:
            answer_field = _answer_field(plan.return_spec.field, var)
        elif op in {"argmax", "argmin"}:
            answer_field = _answer_field(plan.return_spec.field, var)
            if answer_field == "value_amount":
                answer_field = "contract_node_id"
            sort_field = _sort_field(plan.return_spec.field)
        spec = RuntimeQuerySpec(
            spec_id=f"{plan.plan_id}_{var.var_id}",
            question=plan.question,
            intent="role_path" if var.depends_on else "typed",
            constraints=var.constraints,
            answer_operation=op,
            answer_field=answer_field,
            answer_value_type=value_type,
            dedupe_key="contract_node_id",
            requires_exhaustive_retrieval=True,
            sort_field=sort_field,
            metadata=_metadata_for_return(plan.return_spec) if var.var_id == plan.return_spec.input_id else {},
            planner_version="graph_planner_v1",
        )
        variables.append(GraphVariableSpec(**{**var.__dict__, "low_level_spec": spec}))
    return GraphExecutionPlan(**{**plan.__dict__, "variables": tuple(variables)})


def _constraints_from_graph_variable(raw: dict[str, Any], *, org_resolver: Any) -> tuple[list[QueryConstraint], str]:
    constraints: list[QueryConstraint] = []
    for item in raw.get("filters") or ():
        if not isinstance(item, dict):
            continue
        constraint, reason = _constraint_from_filter(item, org_resolver=org_resolver)
        if reason:
            return [], reason
        if constraint is not None:
            constraints.append(constraint)
    grounding = raw.get("grounding") if isinstance(raw.get("grounding"), dict) else {}
    surface = grounding.get("surface") or raw.get("known_surface")
    role_slot = _slot_from_graph_variable(raw)
    if surface and role_slot in {"buyer", "supplier"}:
        constraint, reason = _constraint_from_filter({"slot": role_slot, "surface": surface, "value": surface},
                                                     org_resolver=org_resolver)
        if reason:
            return [], reason
        if constraint is not None:
            constraints.append(constraint)
    return constraints, ""


_CATEGORY_ENUM = frozenset({"goods", "services", "works"})


def _normalise_category(value: Any) -> str | None:
    text = str(value).strip().casefold()
    for noise in ("notices", "notice", "tender", "contract", "contracts"):
        text = text.replace(noise, "")
    text = text.strip()
    singular = {"good": "goods", "service": "services", "work": "works"}.get(text, text)
    return singular if singular in _CATEGORY_ENUM else None


def _constraint_from_filter(item: dict[str, Any], *, org_resolver: Any) -> tuple[QueryConstraint | None, str]:
    slot = str(item.get("slot") or "").strip()
    if not slot:
        return None, ""
    field = _SLOT_FIELDS.get(slot, slot)
    value = item.get("value", item.get("surface"))
    op = str(item.get("operator", item.get("op", "eq")) or "eq")
    if field == "has_award_signed_date":
        # existence predicate: presence of a signed date, not a date VALUE to match. Grok used to be
        # forced to jam "signed award date" into the date slot -> a junk timestamp match -> error.
        return QueryConstraint(field, "eq", True, source_text=str(item.get("surface", value))), ""
    if field == "tender_category" and op == "eq":
        # Grok mis-slots a CPV parenthetical label ("Civic-amenity services") as a category.
        # A real category is one of goods|services|works; anything else is the CPV description,
        # so drop it (the cpv/year filters carry the query) rather than error at grounding.
        normalised = _normalise_category(value)
        if normalised is None:
            return None, ""
        value = normalised
    if field in {"buyer_name", "supplier_name"} and org_resolver is not None:
        resolved = resolve_confident_org(org_resolver, str(value))
        if not resolved.ok:
            return None, f"{resolved.reason}:{value}"
        if len(resolved.variants) > 1:
            # decoration variants of ONE org: the rows are split across surfaces, so filter
            # with an IN over all of them instead of silently picking one
            return QueryConstraint(field, "in", list(resolved.variants),
                                   source_text=str(item.get("surface", value))), ""
        value = resolved.hit.linked_id
    if field == "release_year":
        op, value = _year_constraint(slot, op, value)
    return QueryConstraint(field, op, value, source_text=str(item.get("surface", value))), ""


def _year_constraint(slot: str, op: str, value: Any) -> tuple[str, Any]:
    years = [int(y) for y in re.findall(r"\b20[2-3]\d\b", str(value))]
    if isinstance(value, (list, tuple)):
        years = []
        for item in value:
            years.extend(int(y) for y in re.findall(r"\b20[2-3]\d\b", str(item)))
    years = sorted(dict.fromkeys(years))
    if op == "between" or slot == "year_range":
        if len(years) >= 2:
            return "between", [years[0], years[-1]]
    if op == "in" or slot == "years" or len(years) > 1:
        return "in", years
    if years:
        return "eq", years[0]
    return op, value


def _terminal_spec(plan: GraphExecutionPlan, outputs: dict[str, Any]) -> RuntimeQuerySpec | None:
    var = _variable(plan, plan.return_spec.input_id)
    if var is None or var.low_level_spec is None:
        return None
    return _bind_variable_spec(var.low_level_spec, plan, var, outputs)


def _execute_variable(
    backend: RuntimeQueryBackend,
    allowed: frozenset[str],
    plan: GraphExecutionPlan,
    var_id: str,
    outputs: dict[str, Any],
    level_index: int,
) -> tuple[str, GraphVariableTrace, Any, ExecutionResult | None]:
    var = _variable(plan, var_id)
    if var is None:
        return var_id, GraphVariableTrace(var_id, "unknown", "error", reason=f"unknown variable {var_id!r}",
                                          failure_kind="syntax_error", execution_level=level_index), None, None
    unit = _producer_unit(plan, var_id)
    operation = unit.operation_unit if unit else _default_operation_for_var(var)
    if var.kind == "entity":
        value = _entity_value(var)
        status = "passed" if value else "no_results"
        return var_id, GraphVariableTrace(
            var.var_id, var.kind, status, output_size=1 if value else 0, depends_on=var.depends_on,
            constraints=_constraint_trace(var.constraints), producer_unit=unit.unit_id if unit else "",
            operation_unit=operation, answer=value, reason="" if value else "entity has no grounded value",
            failure_kind="" if value else "empty_result", execution_level=level_index,
        ), value, None
    if var.low_level_spec is None:
        return var_id, GraphVariableTrace(
            var.var_id, var.kind, "unsupported", depends_on=var.depends_on,
            constraints=_constraint_trace(var.constraints), producer_unit=unit.unit_id if unit else "",
            operation_unit=operation, reason="missing low-level spec", failure_kind="syntax_error",
            execution_level=level_index,
        ), None, None
    spec = _bind_variable_spec(var.low_level_spec, plan, var, outputs)
    grounded = ground_spec(spec, allowed_fields=allowed)
    if not grounded.ok:
        return var_id, GraphVariableTrace(
            var.var_id, var.kind, "unsupported", depends_on=var.depends_on,
            constraints=_constraint_trace(spec.constraints), producer_unit=unit.unit_id if unit else "",
            operation_unit=operation, reason=grounded.reason, failure_kind="syntax_error",
            execution_level=level_index,
        ), None, None
    # A compare participant must produce a SCALAR (its count/sum), never an emitted id set —
    # comparing frozensets silently coerced to NaN and returned a confident False.
    if var.kind in {"entity_set", "node_set", "record_set"} and var.emit_field \
            and var.var_id not in _compare_participants(plan):
        if hasattr(backend, "distinct"):
            values = frozenset(backend.distinct(grounded.spec.constraints, var.emit_field))
        else:
            rows = backend.query(grounded.spec.constraints)
            values = frozenset(row.get(var.emit_field) for row in rows if row.get(var.emit_field) not in (None, ""))
        return var_id, GraphVariableTrace(
            var.var_id, var.kind, "passed" if values else "no_results", output_size=len(values),
            depends_on=var.depends_on, constraints=_constraint_trace(grounded.spec.constraints),
            emitted_field=var.emit_field, producer_unit=unit.unit_id if unit else "",
            operation_unit=operation, reason="" if values else "operation unit returned Null/empty set",
            failure_kind="" if values else "empty_result", execution_level=level_index,
        ), values, None
    result = execute_query_spec(backend, grounded.spec)
    return var_id, GraphVariableTrace(
        var.var_id, var.kind, result.status, output_size=1 if result.passed else 0,
        depends_on=var.depends_on, constraints=_constraint_trace(grounded.spec.constraints),
        producer_unit=unit.unit_id if unit else "", operation_unit=operation,
        answer=result.answer, reason="" if result.passed else result.status,
        failure_kind="" if result.passed else _failure_kind_for_status(result.status),
        execution_level=level_index,
    ), result.answer, result


def _compare_participants(plan: GraphExecutionPlan) -> frozenset[str]:
    """var_ids whose OUTPUT feeds the compare combine (they must yield scalars, not sets)."""
    if _runtime_op(plan.return_spec.operation) != "compare":
        return frozenset()
    params = plan.return_spec.parameters
    names = {_var_ref(params.get("left")), _var_ref(params.get("right")),
             _var_ref(plan.return_spec.input_id)}
    unit = _producer_unit(plan, "answer")
    if unit is not None:
        names.update(_var_ref(item) for item in unit.inputs)
    return frozenset(name for name in names if name)


def _execution_levels(plan: GraphExecutionPlan) -> tuple[list[list[str]], str]:
    variables = {var.var_id: var for var in plan.variables}
    # reasoning_order may name steps compile deliberately dropped (cpv_label audit filters) or a
    # planner phantom; only actual variables execute. Referencing an unknown var as a DEPENDENCY
    # is still an error (the `missing` check below) — this was the probe's KeyError('B') crash.
    requested = [item for item in plan.reasoning_order if item != "answer" and item in variables]
    if not requested:
        requested = _topological_order(variables, plan.return_spec.input_id)
    requested_set = set(requested)
    missing = sorted(dep for var_id in requested for dep in variables.get(var_id, GraphVariableSpec(var_id, "")).depends_on
                     if dep not in variables)
    if missing:
        return [], "unknown dependency variables: " + ",".join(missing)
    done: set[str] = set()
    levels: list[list[str]] = []
    while len(done) < len(requested_set):
        ready = [var_id for var_id in requested if var_id not in done
                 and all(dep in done for dep in variables[var_id].depends_on)]
        if not ready:
            remaining = [var_id for var_id in requested if var_id not in done]
            return [], "cycle or unsatisfied dependencies among: " + ",".join(remaining)
        levels.append(ready)
        done.update(ready)
    return levels, ""


def _bind_variable_spec(spec: RuntimeQuerySpec, plan: GraphExecutionPlan, var: GraphVariableSpec,
                        outputs: dict[str, Any]) -> RuntimeQuerySpec:
    constraints = list(spec.constraints)
    for dep_id in var.depends_on:
        values = outputs.get(dep_id)
        if isinstance(values, frozenset):
            bind_field = _bind_field_for_dependency(plan, var, _variable(plan, dep_id))
            constraints.append(QueryConstraint(bind_field, "in", tuple(sorted(values, key=str)),
                                               visible_to_user=False, source_text=f"bound from {dep_id}"))
        elif values not in (None, ""):
            bind_field = _bind_field_for_dependency(plan, var, _variable(plan, dep_id))
            constraints.append(QueryConstraint(bind_field, "eq", values, visible_to_user=False,
                                               source_text=f"bound from {dep_id}"))
    return RuntimeQuerySpec(**{**spec.__dict__, "constraints": tuple(constraints)})


def _bind_field_for_dependency(plan: GraphExecutionPlan, target: GraphVariableSpec, source: GraphVariableSpec | None) -> str:
    # The source's emit_field is the authoritative bind key: an entity_set of suppliers emits
    # supplier_name (a real join / bridge), while a record_set of contracts emits contract_node_id.
    # Grok routinely OVER-DECOMPOSES a conjunctive filter (cpv AND year) into two chained
    # record_set variables; binding on the emit_field (contract_node_id) then intersects them
    # correctly, instead of the old default that bound on supplier_name and returned no_results.
    if source and source.emit_field:
        return source.emit_field
    if source and "supplier" in source.role.casefold():
        return "supplier_name"
    if source and ("buyer" in source.role.casefold() or "authority" in source.role.casefold()):
        return "buyer_name"
    for relation in plan.relations:
        if relation.source == (source.var_id if source else "") and relation.target == target.var_id and relation.bind_field:
            return relation.bind_field
        if relation.target == (source.var_id if source else "") and relation.source == target.var_id and relation.bind_field:
            return relation.bind_field
    return "supplier_name"


def _relation_from_raw(raw: dict[str, Any]) -> GraphRelationSpec:
    candidates = raw.get("relation_candidates") or ()
    relation = ""
    if candidates and isinstance(candidates, list) and isinstance(candidates[0], dict):
        relation = str(candidates[0].get("relation") or "")
    bind_field = _bind_field_from_relation(relation, raw)
    return GraphRelationSpec(source=str(raw.get("from") or ""), target=str(raw.get("to") or ""),
                             relation=relation, direction=str(raw.get("direction") or "unknown"),
                             surface_evidence=str(raw.get("surface_evidence") or ""),
                             bind_field=bind_field)


def _operation_unit_from_raw(raw: dict[str, Any]) -> GraphOperationUnitSpec:
    return GraphOperationUnitSpec(
        unit_id=str(raw.get("step_id") or raw.get("unit_id") or raw.get("id") or ""),
        operation_unit=str(raw.get("operation_unit") or raw.get("operation") or ""),
        inputs=tuple(str(item) for item in (raw.get("inputs") or ())),
        output=str(raw.get("output") or ""),
        description=str(raw.get("description") or ""),
        uses=raw.get("uses") if isinstance(raw.get("uses"), dict) else {},
    )


def _producer_unit(plan: GraphExecutionPlan, output: str) -> GraphOperationUnitSpec | None:
    return next((unit for unit in plan.operation_units if unit.output == output), None)


def _default_operation_for_var(var: GraphVariableSpec) -> str:
    if var.kind == "entity":
        return "resolve_entity"
    if var.kind == "entity_set":
        return "find_entity_set"
    if var.kind in {"node_set", "record_set"}:
        return "find_record_set"
    return "filter_records"


def _bind_field_from_relation(relation: str, raw: dict[str, Any]) -> str:
    text = " ".join(str(x) for x in (relation, raw.get("surface_evidence"), raw.get("description"))).casefold()
    if "cpv" in text or "classification" in text:
        return "tender_cpv_id"
    if "categor" in text:
        return "tender_category"
    if "supplier" in text or "award" in text or "worked with" in text:
        return "supplier_name"
    if "buyer" in text or "authority" in text or "published" in text:
        return "buyer_name"
    return ""


# An entity set is not only organisations: bridges also run over CPV codes ("notices under CPV
# codes the Trust has used before") and categories. The emit/bind dimension follows the role.
_ENTITY_DIMENSIONS = {"buyer": "buyer_name", "supplier": "supplier_name",
                      "cpv": "tender_cpv_id", "category": "tender_category"}


def _emit_field_for_variable(raw: dict[str, Any]) -> str:
    kind = str(raw.get("kind") or "")
    if kind in {"entity", "entity_set"}:
        slot = _slot_from_graph_variable(raw)
        if slot in _ENTITY_DIMENSIONS:
            return _ENTITY_DIMENSIONS[slot]
        # an untyped entity_set most often emits what its own filters mention
        for item in raw.get("filters") or ():
            if isinstance(item, dict) and str(item.get("slot")) == "cpv":
                return "tender_cpv_id"
    if kind in {"node_set", "record_set"}:
        return "contract_node_id"
    return ""


def _slot_from_graph_variable(raw: dict[str, Any]) -> str:
    text = " ".join(str(raw.get(k) or "") for k in ("role", "description")).casefold()
    if "cpv" in text or "classification" in text:
        return "cpv"
    if "supplier" in text or "winner" in text:
        return "supplier"
    if "buyer" in text or "authority" in text or "publisher" in text:
        return "buyer"
    if "categor" in text:
        return "category"
    candidates = raw.get("type_candidates") or raw.get("node_type_candidates") or ()
    if isinstance(candidates, list):
        for candidate in candidates:
            if isinstance(candidate, dict):
                typ = str(candidate.get("type") or "").casefold()
                if typ in {"buyer", "contracting_authority", "authority"}:
                    return "buyer"
                if typ in {"supplier", "winner", "award_winner"}:
                    return "supplier"
                if typ in {"cpv", "classification"}:
                    return "cpv"
                if typ in {"category", "procurement_category"}:
                    return "category"
    return ""


def _runtime_op(operation: str) -> str:
    op = operation.casefold()
    if op in {"sum", "total"}:
        return "sum"
    if op in {"count"}:
        return "count"
    if op in {"exists", "boolean", "yes_no"}:
        return "exists"
    if op in {"set", "distinct_set"}:
        return "distinct_set"
    if op in {"select", "factoid"}:
        return "select_unique"
    if op in {"top_k", "rank"}:
        return "top_k"
    if op in {"argmin", "min"}:
        return "argmin"
    if op in {"argmax", "max"}:
        return "argmax"
    if op in {"compare", "comparison"}:
        return "compare"
    return op or "count"


def _runtime_op_for_variable(
    plan: GraphExecutionPlan, var: GraphVariableSpec, unit: GraphOperationUnitSpec | None
) -> str:
    if var.var_id == plan.return_spec.input_id and _runtime_op(plan.return_spec.operation) != "compare":
        return _runtime_op(plan.return_spec.operation)
    unit_op = (unit.operation_unit if unit else "").casefold()
    if unit_op == "aggregate_sum":
        return "sum"
    if unit_op == "aggregate_count":
        return "count"
    if unit_op == "select_unique":
        return "select_unique"
    if unit_op == "rank_top_k":
        return "top_k"
    if unit_op == "find_extreme":
        extreme = str((unit.uses or {}).get("extreme") or plan.return_spec.parameters.get("extreme") or "").casefold()
        return "argmin" if extreme in {"min", "argmin", "lowest", "smallest"} else "argmax"
    if _runtime_op(plan.return_spec.operation) == "compare":
        # A compare participant's aggregate: return.metric is authoritative when set (0050 said
        # metric=sum but field=none and was counted); else return.field decides — comparing
        # VALUES means sum, comparing a DATE means fetching the record's signed date, anything
        # else (notices, contracts) is a count.
        left = str(plan.return_spec.parameters.get("left") or plan.return_spec.input_id)
        right = str(plan.return_spec.parameters.get("right") or "")
        if var.var_id in (left, right, plan.return_spec.input_id):
            metric = str(plan.return_spec.parameters.get("metric") or "").casefold()
            field = plan.return_spec.field.casefold()
            if metric == "sum" or "value" in field or "amount" in field:
                return "sum"
            if "date" in field:
                return "select_unique"
            return "count"
    if var.kind in {"entity_set", "node_set", "record_set"}:
        return "count"
    return "count"


def _answer_field(field: str, var: GraphVariableSpec) -> str:
    low = field.casefold()
    if "contract" in low or "record" in low:
        return "contract_node_id"
    if "buyer" in low or "authority" in low:
        return "buyer_name"
    if "supplier" in low or "winner" in low:
        return "supplier_name"
    if "date" in low:
        return "award_date_signed"
    if "value" in low or "amount" in low:
        return "value_amount"
    if "cpv" in low:
        return "tender_cpv_id"
    if "categor" in low:  # "procurement category" -> a factoid selecting goods|services|works
        return "tender_category"
    return var.emit_field or "supplier_name"


def _sort_field(field: str) -> str:
    low = field.casefold()
    if "date" in low:
        return "award_date_signed"
    return "value_amount"


def _metadata_for_return(ret: GraphReturnSpec) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if ret.parameters.get("exclude_zero"):
        metadata["exclude_zero"] = True  # "lowest NON-ZERO value": argmin/argmax skip zero values
    if _runtime_op(ret.operation) == "top_k":
        group_by = str(ret.parameters.get("group_by") or ret.parameters.get("group_by_field") or "buyer")
        metric = str(ret.parameters.get("metric") or "count")
        try:
            k = int(ret.parameters.get("k") or 3)
        except (TypeError, ValueError):
            k = 3
        metadata.update({
            "group_by_field": _SLOT_FIELDS.get(group_by, group_by if group_by.endswith("_name") else "buyer_name"),
            "metric": "sum" if metric.casefold() in {"sum", "value", "total_value"} else "count",
            "k": k,
            "metric_field": _sort_field(str(ret.parameters.get("metric_field") or ret.field))})
    return metadata


_ISO_DATE_PREFIX_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")


def _as_iso_date(value: Any) -> str:
    text = str(value).strip()
    match = _ISO_DATE_PREFIX_RE.match(text)
    if match:
        return match.group(1)
    from .grounding import _parse_written_date
    return _parse_written_date(text) or ""


def _compare_outputs(plan: GraphExecutionPlan, outputs: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    params = plan.return_spec.parameters
    left = _var_ref(params.get("left") or params.get("lhs") or "")
    right = _var_ref(params.get("right") or params.get("rhs") or "")
    if not left or not right:
        inputs = []
        unit = _producer_unit(plan, "answer")
        if unit is not None:
            inputs = list(unit.inputs)
        if len(inputs) >= 2:
            left, right = _var_ref(inputs[0]), _var_ref(inputs[1])
    if not left or not right:
        return None, "compare requires left/right variables"
    comparator = str(params.get("comparator") or params.get("operator") or "gt").casefold()
    a, b = outputs.get(left), outputs.get(right)
    # a side that is not a variable may be a LITERAL date or amount ("2025-05-01",
    # "GBP 10 million"): a threshold comparison, not a two-variable compare. Dates first —
    # _parse_amount would happily read "2025-05-01" as the number 2025.
    if a is None:
        a = _as_iso_date(left) or _parse_amount(left)
    if b is None:
        b = _as_iso_date(right) or _parse_amount(right)
    if a is None or b is None:
        return None, f"compare missing outputs: {left}/{right}"
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        # DATE comparison: a fetched signed date ("2024-12-19T00:00:00Z") vs an ISO/written
        # literal. ISO date strings order lexicographically, so compare the [:10] prefixes.
        da, db = _as_iso_date(a), _as_iso_date(b)
        if da and db and comparator not in {"eq", "=", "=="}:
            fa, fb = 0.0, 0.0  # placeholders; comparison below uses the date strings
            if comparator in {"gt", ">", "more", "greater", "after"}:
                return {"answer": da > db, left: a, right: b, "comparator": comparator}, ""
            if comparator in {"lt", "<", "less", "fewer", "before"}:
                return {"answer": da < db, left: a, right: b, "comparator": comparator}, ""
            if comparator in {"gte", ">="}:
                return {"answer": da >= db, left: a, right: b, "comparator": comparator}, ""
            if comparator in {"lte", "<="}:
                return {"answer": da <= db, left: a, right: b, "comparator": comparator}, ""
        if comparator not in {"eq", "=", "=="}:
            # a silent NaN comparison returned a confident False; incomparable sides must error
            return None, f"non-comparable compare sides: {a!r} vs {b!r}"
        fa = fb = float("nan")
    if comparator in {"gt", ">", "more", "greater"}:
        answer = fa > fb
    elif comparator in {"gte", ">="}:
        answer = fa >= fb
    elif comparator in {"lt", "<", "less", "fewer"}:
        answer = fa < fb
    elif comparator in {"lte", "<="}:
        answer = fa <= fb
    elif comparator in {"eq", "=", "=="}:
        answer = str(a) == str(b)
    else:
        return None, f"unknown comparator {comparator!r}"
    return {"answer": bool(answer), left: a, right: b, "comparator": comparator}, ""


def _var_ref(value: Any) -> str:
    text = str(value or "").strip()
    match = re.fullmatch(r"(?:count|sum)\(([^)]+)\)", text, re.I)
    if match:
        return match.group(1).strip()
    return "" if text == "none" else text


_AMOUNT_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(m|mn|million|bn|billion|k|thousand)?\b", re.I)
_AMOUNT_MULTIPLIERS = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6,
                       "bn": 1e9, "billion": 1e9}


def _parse_amount(text: Any) -> float | None:
    """Parse a literal money/number phrase ("GBP 10 million", "5") to a float, else None."""
    match = _AMOUNT_RE.search(str(text or ""))
    if not match:
        return None
    value = float(match.group(1).replace(",", ""))
    suffix = (match.group(2) or "").casefold()
    return value * _AMOUNT_MULTIPLIERS.get(suffix, 1.0)


def _failure_kind_for_status(status: str) -> str:
    if status in {"schema_error", "constraint_conflict", "unsupported_operation", "error", "unsupported"}:
        return "syntax_error"
    if status in {"no_results", "multiple_answers", "incomplete_evidence"}:
        return "empty_or_semantic_result"
    return "runtime_rejected"


def _variable(plan: GraphExecutionPlan, var_id: str) -> GraphVariableSpec | None:
    return next((var for var in plan.variables if var.var_id == var_id), None)


def _entity_value(var: GraphVariableSpec) -> Any:
    for constraint in var.constraints:
        if constraint.field in {"buyer_name", "supplier_name"}:
            return constraint.value
    return None


def _topological_order(variables: dict[str, GraphVariableSpec], root: str) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()

    def visit(var_id: str) -> None:
        if var_id in seen:
            return
        seen.add(var_id)
        var = variables.get(var_id)
        if var is None:
            return
        for dep in var.depends_on:
            visit(dep)
        order.append(var_id)

    visit(root)
    return order


def _default_reasoning_order(
    variables: dict[str, GraphVariableSpec], operation: str, input_id: str, ret: dict[str, Any]
) -> list[str]:
    roots = [input_id]
    if _runtime_op(operation) == "compare":
        roots.extend(_var_ref(ret.get(key)) for key in ("left", "right"))
    order: list[str] = []
    for root in roots:
        for var_id in _topological_order(variables, root):
            if var_id not in order:
                order.append(var_id)
    return order


def _constraint_trace(constraints: tuple[QueryConstraint, ...]) -> tuple[dict[str, Any], ...]:
    return tuple({"field": c.field, "op": c.op, "value": c.value, "source_text": c.source_text}
                 for c in constraints)


def _low_level_spec_trace(spec: RuntimeQuerySpec | None) -> dict[str, Any] | None:
    """JSON-native view of a variable's low-level spec. The raw ``__dict__`` holds QueryConstraint
    dataclass objects, which break json.dumps when this trace flows into the reflector feedback
    prompt (typed_replan_messages) — so materialise constraints as plain dicts here."""
    if spec is None:
        return None
    return {
        "spec_id": spec.spec_id,
        "intent": spec.intent,
        "answer_operation": spec.answer_operation,
        "answer_field": spec.answer_field,
        "answer_value_type": spec.answer_value_type,
        "dedupe_key": spec.dedupe_key,
        "sort_field": spec.sort_field,
        "constraints": list(_constraint_trace(spec.constraints)),
        "metadata": spec.metadata,
    }


def graph_plan_trace(plan: GraphExecutionPlan) -> dict[str, Any]:
    return {
        "plan_id": plan.plan_id,
        "reasoning_order": list(plan.reasoning_order),
        "variables": [
            {
                "var_id": var.var_id,
                "kind": var.kind,
                "description": var.description,
                "role": var.role,
                "depends_on": list(var.depends_on),
                "emit_field": var.emit_field,
                "constraints": list(_constraint_trace(var.constraints)),
                "low_level_spec": _low_level_spec_trace(var.low_level_spec),
            }
            for var in plan.variables
        ],
        "relations": [rel.__dict__ for rel in plan.relations],
        "operation_units": [unit.__dict__ for unit in plan.operation_units],
        "return": plan.return_spec.__dict__,
        "understanding_network": plan.understanding_network,
        "raw_graph_plan": plan.raw_graph_plan,
    }


def graph_execution_trace(result: GraphExecutionResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "answer": result.answer,
        "reason": result.reason,
        "failure_kind": result.metrics.get("failure_kind", ""),
        "execution_levels": result.metrics.get("execution_levels"),
        "variables": [trace.__dict__ for trace in result.variable_traces],
        "operation_units": [unit.__dict__ for unit in result.plan.operation_units],
        "low_level_result": {
            "status": result.low_level_result.status,
            "answer": result.low_level_result.answer,
            "checks": list(result.low_level_result.checks),
            "metrics": result.low_level_result.metrics,
        } if result.low_level_result is not None else None,
        "evidence_ids": list(result.evidence_ids),
        "metrics": result.metrics,
    }


__all__ = [
    "GraphExecutionPlan",
    "GraphExecutionResult",
    "GraphOperationUnitSpec",
    "GraphRelationSpec",
    "GraphReturnSpec",
    "GraphVariableSpec",
    "GraphVariableTrace",
    "compile_graph_plan",
    "execute_graph_plan",
    "graph_execution_trace",
    "graph_plan_trace",
    "graph_to_decomposition",
]
