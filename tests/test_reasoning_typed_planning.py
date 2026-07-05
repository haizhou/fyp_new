"""Typed planner DSL: LLM fills slots, deterministic checks protect semantics."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from procurement_graph.reasoning.typed_planning import (
    TypedLLMPlanner,
    compile_typed_plan,
    intent_program_schema,
    plan_consistency_check,
    question_intent_program_messages,
    question_understanding_messages,
    repair_understanding_messages,
    repair_graph_plan_schema,
    typed_plan_messages,
    typed_replan_messages,
)
from procurement_graph.reasoning.schema_grounding import ground_field_text, grounding_candidates
from procurement_graph.reasoning.models import EntityLinkCandidate


class _Hit:
    linked_id = "BIRMINGHAM CITY COUNCIL"
    linked_label = "BIRMINGHAM CITY COUNCIL"
    source = "records_exact"
    score = 1.0


class _Resolver:
    def resolve(self, mention):
        return [_Hit()] if str(mention).casefold() == "birmingham city council" else []


class _AmbiguousResolver:
    def resolve(self, mention):
        return [
            EntityLinkCandidate(mention, "BIRMINGHAM CITY COUNCIL", "BIRMINGHAM CITY COUNCIL",
                                "organization", 0.88, source="records_substring"),
            EntityLinkCandidate(mention, "BIRMINGHAM CHILDREN TRUST", "BIRMINGHAM CHILDREN TRUST",
                                "organization", 0.86, source="records_substring"),
        ]


class _Response:
    def __init__(self, parsed):
        self.parsed = parsed


class _Chat:
    def __init__(self, parsed):
        self.parsed = parsed
        self.calls = 0

    def complete_json(self, *, model, system, user):
        self.calls += 1
        return _Response(self.parsed)


class _TextThenJsonChat:
    def __init__(self, text, payload):
        self.text = text
        self.payload = payload
        self.calls = 0

    def complete_text(self, *, model, system, user):
        self.calls += 1
        response = _Response({})
        response.raw_text = self.text
        return response

    def complete_json(self, *, model, system, user):
        self.calls += 1
        return _Response(self.payload)


class _SchemaRepairChat:
    def __init__(self, repair_payload):
        self.repair_payload = repair_payload
        self.calls = 0
        self.schemas = []
        self.text_users = []

    def complete_text(self, *, model, system, user):
        self.calls += 1
        self.text_users.append(user)
        response = _Response({})
        response.raw_text = "1. Answer Type: count\n2. Explicit Info: year = 2024"
        return response

    def complete_schema(self, *, model, system, user, schema):
        self.calls += 1
        self.schemas.append(schema)
        return _Response(self.repair_payload)


class _SchemaFallbackChat:
    def __init__(self, *, schema_exc, fallback_payload):
        self.schema_exc = schema_exc
        self.fallback_payload = fallback_payload
        self.schema_calls = 0
        self.json_calls = 0

    def complete_schema(self, *, model, system, user, schema):
        self.schema_calls += 1
        raise self.schema_exc

    def complete_json(self, *, model, system, user):
        self.json_calls += 1
        return _Response(self.fallback_payload)


class _IntentSchemaChat:
    def __init__(self, intent_payload):
        self.intent_payload = intent_payload
        self.schema_calls = 0
        self.json_calls = 0

    def complete_schema(self, *, model, system, user, schema):
        self.schema_calls += 1
        return _Response(self.intent_payload)

    def complete_json(self, *, model, system, user):
        self.json_calls += 1
        return _Response({"should_not": "be_called"})


class TestTypedPlanningMessages(unittest.TestCase):
    def test_intent_program_prompt_and_schema_separate_label_from_category(self):
        system, user = question_intent_program_messages(
            "How many notices were published in 2024 under CPV 85149000 (Pharmacy services)?")
        schema = intent_program_schema()
        self.assertIn("typed intent programmer", system)
        self.assertIn("cpv_label", user)
        self.assertIn("procurement_category", user)
        self.assertIn("Do not put CPV labels in procurement_category", user)
        filter_props = schema["schema"]["properties"]["program"]["items"]["properties"]["args"]["properties"]["filters"]["items"]["properties"]
        self.assertIn("field_text", filter_props)
        self.assertNotIn("slot", filter_props)
        self.assertIn("answer_field_text", schema["schema"]["properties"]["answer_signature"]["properties"])

    def test_schema_grounding_uses_candidates_and_type_gate(self):
        grounded = ground_field_text("contract publication year", value="2024", value_type="year")
        self.assertEqual(grounded.slot, "year")
        self.assertTrue(grounded.candidates)

        rejected = ground_field_text("high-level procurement category",
                                     value="Pharmacy services", value_type="category")
        self.assertFalse(rejected.ok)
        self.assertIn("type_gate_rejected", rejected.reason)

    def test_schema_grounding_exposes_ranked_candidates(self):
        candidates = grounding_candidates("awarding authority")
        self.assertGreaterEqual(len(candidates), 2)
        self.assertEqual(candidates[0].slot, "buyer")

    def test_understanding_prompt_is_not_compiler_slots_first(self):
        system, user = question_understanding_messages(
            "How many notices did BIRMINGHAM CITY COUNCIL publish in 2024?")
        self.assertIn("semantic reading layer", system)
        self.assertIn("Answer Type", user)
        self.assertIn("Query Template", user)
        self.assertIn("Reverse Tree", user)
        self.assertIn("Procedure", user)
        self.assertNotIn("Retrieved local schema hints", user)
        self.assertNotIn("keyword_schema_slice_v1", user)
        self.assertIn("Targets (A, B, C)", user)
        # dense-scaffold prompt: hard line budgets + ASCII/no-JSON, no soft word limit
        self.assertIn("line budgets", user)
        self.assertIn("No JSON", user)
        self.assertNotIn("under 450 words", user)
        self.assertIn("do not answer", system.casefold())

    def test_prompt_is_lean_and_schema_enforced(self):
        # strict json_schema controls the OUTPUT shape, so the prompt is lean guidance only: no
        # nested schema dump, no understanding_network / goal_tree echo.
        from procurement_graph.reasoning.typed_planning import graph_plan_schema
        system, user = typed_plan_messages(
            "How many notices did BIRMINGHAM CITY COUNCIL publish in 2024?",
            understanding={"question_type": "count"},
        )
        payload = json.loads(user)
        self.assertIn("structured graph planner", system)
        self.assertIn("graph_plan", system)
        self.assertIn("variables", user)
        self.assertEqual(payload["stage1_query_template"], "simple_filter_aggregate")
        self.assertIn("selected_template_shell", payload)
        self.assertIn("template_specific_rules", payload)
        self.assertIn("retrieved_schema_context", payload)
        self.assertIn("executor_capability_card", payload)
        self.assertIn("instructions", payload)
        self.assertIn("distinct_set", payload["executor_capability_card"]["return_operations"])
        self.assertIn("query_semantics", payload["executor_capability_card"])
        self.assertIn("bridge_rules", payload["executor_capability_card"])
        self.assertIn("compare_two_counts", json.dumps(payload["executor_capability_card"]))
        self.assertIn("Copy stage1_query_template", user)
        self.assertNotIn("understanding_network", user)
        self.assertNotIn("goal_tree", user)
        self.assertNotIn("required_output_schema", user)
        self.assertNotIn("oracle", user.casefold())
        # the strict schema: variables is an ARRAY, slot/operation are enums
        schema = graph_plan_schema()
        self.assertTrue(schema["strict"])
        props = schema["schema"]["properties"]
        self.assertEqual(props["variables"]["type"], "array")
        self.assertFalse(schema["schema"]["additionalProperties"])
        self.assertIn("count", props["operation"]["enum"])
        self.assertIn("template", props)
        self.assertIn("bridge_join", props["template"]["enum"])
        slot_enum = props["variables"]["items"]["properties"]["filters"]["items"]["properties"]["slot"]["enum"]
        self.assertIn("cpv", slot_enum)

    def test_stage2_uses_stage1_query_template_shell(self):
        _system, user = typed_plan_messages(
            "How many contracts went to suppliers who worked with CHP?",
            understanding={"answer_type": "count", "query_template": "bridge_join"},
        )
        payload = json.loads(user)
        self.assertEqual(payload["stage1_query_template"], "bridge_join")
        self.assertIn("source records -> distinct entity_set -> target records", json.dumps(payload))

    def test_cached_understanding_parses_query_template(self):
        from procurement_graph.reasoning.typed_planning import understanding_from_text
        parsed = understanding_from_text("\n".join([
            "1. Answer Type: count",
            "2. Query Template: bridge_join",
            "3. Explicit Info: org name = CHP",
        ]))
        self.assertEqual(parsed["query_template"], "bridge_join")

    def test_schema_retrieval_uses_executor_operation_names(self):
        from procurement_graph.reasoning.schema_retrieval import retrieve_schema_context
        context = retrieve_schema_context("Did buyer A publish more notices than buyer B?")
        text = json.dumps(context)
        self.assertIn("count", context["allowed_operation_units"])
        self.assertIn("sum", context["allowed_operation_units"])
        self.assertNotIn("aggregate_count", text)
        self.assertNotIn("aggregate_sum", text)

    def test_feedback_replan_prompt_carries_failure_diagnosis(self):
        _system, user = typed_replan_messages("How many notices were published in 2024?",
                                              {"execution_status": "no_results"})
        payload = json.loads(user)
        self.assertIn("Feedback replan", user)
        self.assertIn("failure_feedback", user)
        self.assertIn("no_results", user)
        self.assertIn("fix_question_type", payload["allowed_repair_actions"])
        # graph-native reflector: repairs a graph plan, not the legacy type shell
        self.assertIn("repaired_graph_plan", payload["output_schema"])
        self.assertIn("retrieved_schema_context", payload)
        self.assertNotIn("all_type_shells", payload)
        schema = repair_graph_plan_schema()
        self.assertTrue(schema["strict"])
        props = schema["schema"]["properties"]
        self.assertIn("repaired_graph_plan", props)
        self.assertEqual(props["repaired_graph_plan"]["properties"]["variables"]["type"], "array")
        self.assertIn("fix_question_type", props["repair_action"]["enum"])

    def test_repair_understanding_sees_failed_stage2_but_not_oracle(self):
        _system, user = repair_understanding_messages(
            "How many notices were published in 2024?",
            {
                "failed_plan": {"graph_plan": {"variables": [{"var_id": "a1"}]}},
                "failure_reason": "no_results",
                "oracle_answer": 17,
                "reference_answer": 17,
            },
        )
        self.assertIn("failed Stage-2 graph plan", user)
        self.assertIn("no_results", user)
        self.assertIn("a1", user)
        self.assertNotIn("oracle_answer", user)
        self.assertNotIn("reference_answer", user)
        self.assertNotIn("17", user)


class TestPlanConsistencyCheck(unittest.TestCase):
    def test_rejects_invented_year(self):
        payload = {
            "question_type": "count",
            "operation": "count",
            "constraints": [
                {"slot": "buyer", "surface": "BIRMINGHAM CITY COUNCIL",
                 "value": "BIRMINGHAM CITY COUNCIL"},
                {"slot": "year", "surface": "2024", "value": 2025},
            ],
        }
        verdict = plan_consistency_check(
            "How many notices did BIRMINGHAM CITY COUNCIL publish in 2024?", payload)
        self.assertFalse(verdict.ok)
        self.assertIn("invented_number:2025", verdict.issues)

    def test_rejects_buyer_supplier_role_flip(self):
        payload = {
            "question_type": "count",
            "operation": "count",
            "constraints": [
                {"slot": "buyer", "surface": "BIRMINGHAM CITY COUNCIL",
                 "value": "BIRMINGHAM CITY COUNCIL"},
            ],
        }
        verdict = plan_consistency_check(
            "How many notices were awarded to BIRMINGHAM CITY COUNCIL?", payload)
        self.assertFalse(verdict.ok)
        self.assertTrue(any(issue.startswith("role_flipped:buyer") for issue in verdict.issues))

    def test_rejects_operation_outside_question_type(self):
        payload = {"question_type": "count", "operation": "sum", "constraints": []}
        verdict = plan_consistency_check("How many notices were published?", payload)
        self.assertFalse(verdict.ok)
        self.assertIn("operation_outside_type:sum!in:count", verdict.issues)

    def test_rejects_missing_explicit_cpv_year_and_category(self):
        payload = {
            "question_type": "count",
            "operation": "count",
            "constraints": [{"slot": "year", "surface": "2024", "value": 2024}],
        }
        verdict = plan_consistency_check(
            "How many goods notices published in 2024 under CPV 42996500 are recorded?", payload)
        self.assertFalse(verdict.ok)
        self.assertIn("missing_explicit_constraint:cpv:42996500", verdict.issues)
        self.assertIn("missing_explicit_constraint:category:goods", verdict.issues)

    def test_rejects_unsupported_cue_unless_unanswerable(self):
        payload = {"question_type": "factoid", "operation": "select_unique", "constraints": []}
        verdict = plan_consistency_check("What evaluation score was recorded for the contract?", payload)
        self.assertFalse(verdict.ok)
        self.assertIn("unsupported_cue_requires_unanswerable", verdict.issues)

    def test_accepts_year_range_when_both_years_are_in_question(self):
        payload = {
            "question_type": "count",
            "operation": "count",
            "constraints": [
                {"slot": "year_range", "surface": "between 2022 and 2024",
                 "operator": "between", "value": [2022, 2024]},
            ],
        }
        verdict = plan_consistency_check("How many notices were published between 2022 and 2024?", payload)
        self.assertTrue(verdict.ok, verdict.issues)

    def test_accepts_normalized_iso_date_from_written_date(self):
        payload = {
            "question_type": "boolean",
            "operation": "predicate",
            "constraints": [{"slot": "date", "surface": "1 June 2023", "value": "2023-06-01"}],
            "comparison": {"operator": "after", "threshold": "2023-06-01"},
        }
        verdict = plan_consistency_check("Was the award signed after 1 June 2023?", payload)
        self.assertTrue(verdict.ok, verdict.issues)

    def test_accepts_scaled_million_threshold(self):
        payload = {
            "question_type": "boolean",
            "operation": "predicate",
            "constraints": [],
            "comparison": {"operator": ">", "threshold": 1_500_000},
        }
        verdict = plan_consistency_check("Was the total value above £1.5m?", payload)
        self.assertTrue(verdict.ok, verdict.issues)

    def test_missing_operation_defaults_like_compile(self):
        # nano frequently omits `operation`; compile defaults it to the type's first operation,
        # so its absence is NOT a semantic mismatch (this killed 17/20 first-pass L2 plans).
        payload = {
            "question_type": "count",
            "constraints": [{"slot": "year", "surface": "2024", "value": 2024}],
        }
        verdict = plan_consistency_check("How many notices were published in 2024?", payload)
        self.assertTrue(verdict.ok, verdict.issues)

    def test_gt_cue_with_noun_phrase_between_more_and_than(self):
        payload = {
            "question_type": "comparison",
            "operation": "predicate",
            "constraints": [],
            "comparison": {"operator": ">", "threshold": 5},
        }
        verdict = plan_consistency_check(
            "Did ACME put out more contract notices than BETA in 2024? More than 5?", payload)
        self.assertNotIn("comparison_direction_gt_unsupported_by_text", verdict.issues)

    def test_bridge_cue_rejects_flat_count(self):
        payload = {
            "question_type": "count",
            "operation": "count",
            "constraints": [{"slot": "supplier", "surface": "Mills & Reeve LLP",
                             "value": "Mills & Reeve LLP"}],
        }
        verdict = plan_consistency_check(
            "How many contract notices did buyers publish where the awarded contract went to "
            "Mills & Reeve LLP?", payload)
        self.assertFalse(verdict.ok)
        self.assertTrue(any(issue.startswith("bridge_cue_requires_bridge_join") for issue in verdict.issues))

    def test_nested_shell_echo_unwrapped(self):
        # nano echoes its filled plan NESTED under the prompt's `selected_type_shell` key; the
        # checker must judge the unwrapped plan, not an empty top level.
        payload = {
            "question_type": "count",
            "selected_type_shell": {
                "question_type": "count", "operation": "count",
                "constraints": [
                    {"slot": "category", "surface": "services notices", "value": "services notices", "operator": "eq"},
                    {"slot": "year", "surface": "published in 2024", "value": "2024", "operator": "eq"},
                    {"slot": "cpv", "surface": "CPV 72221000", "value": "72221000", "operator": "eq"},
                ],
            },
        }
        verdict = plan_consistency_check(
            "What number of services notices in the dataset were published in 2024 under CPV 72221000?",
            payload)
        self.assertTrue(verdict.ok, verdict.issues)

    def test_bridge_type_not_flagged_by_bridge_cue(self):
        payload = {
            "question_type": "bridge_join",
            "operation": "count",
            "steps": [
                {"step": "anchor_set", "bind_slot": "buyer",
                 "constraints": [{"slot": "supplier", "surface": "Mills & Reeve LLP",
                                  "value": "Mills & Reeve LLP"}]},
                {"step": "answer", "from_step": "anchor_set", "constraints": []},
            ],
        }
        verdict = plan_consistency_check(
            "How many contract notices did buyers publish where the awarded contract went to "
            "Mills & Reeve LLP?", payload)
        self.assertTrue(verdict.ok, verdict.issues)


class TestCompileTypedPlan(unittest.TestCase):
    def test_count_plan_compiles_and_resolves_org(self):
        payload = {
            "question_type": "count",
            "operation": "count",
            "constraints": [
                {"slot": "buyer", "surface": "Birmingham City Council",
                 "value": "Birmingham City Council"},
                {"slot": "year", "surface": "2024", "value": "2024"},
            ],
        }
        candidate = compile_typed_plan(
            "How many notices did Birmingham City Council publish in 2024?",
            payload,
            org_resolver=_Resolver(),
        )
        self.assertEqual(candidate.status, "planned")
        self.assertEqual(candidate.query_spec.answer_operation, "count")
        self.assertEqual(
            [(c.field, c.op, c.value) for c in candidate.query_spec.constraints],
            [("buyer_name", "eq", "BIRMINGHAM CITY COUNCIL"), ("release_year", "eq", 2024)],
        )

    def test_unresolved_org_becomes_structured_ambiguity(self):
        payload = {
            "question_type": "count",
            "operation": "count",
            "constraints": [{"slot": "buyer", "surface": "Unknown Body", "value": "Unknown Body"}],
        }
        candidate = compile_typed_plan("How many notices did Unknown Body publish?", payload,
                                       org_resolver=_Resolver())
        self.assertEqual(candidate.status, "ambiguous")
        self.assertIn("entity_not_found", candidate.rationale)

    def test_ambiguous_org_candidates_do_not_silently_pick_top_hit(self):
        payload = {
            "question_type": "count",
            "operation": "count",
            "constraints": [{"slot": "buyer", "surface": "Birmingham", "value": "Birmingham"}],
        }
        candidate = compile_typed_plan("How many notices did Birmingham publish?", payload,
                                       org_resolver=_AmbiguousResolver())
        self.assertEqual(candidate.status, "ambiguous")
        self.assertIn("ambiguous_entity_candidates", candidate.rationale)

    def test_nested_shell_echo_compiles_with_constraints(self):
        # the L2 whole-KG-count failure: plan nested under selected_type_shell compiled to zero
        # constraints and ran unconstrained. Unwrap must recover the real constraints.
        payload = {
            "question_type": "count",
            "selected_type_shell": {
                "question_type": "count", "operation": "count",
                "constraints": [
                    {"slot": "year", "surface": "published in 2024", "value": "2024", "operator": "eq"},
                    {"slot": "cpv", "surface": "CPV 72221000", "value": "72221000", "operator": "eq"},
                ],
            },
        }
        candidate = compile_typed_plan(
            "How many notices were published in 2024 under CPV 72221000?", payload)
        self.assertEqual(candidate.status, "planned")
        self.assertEqual(
            [(c.field, c.op, c.value) for c in candidate.query_spec.constraints],
            [("release_year", "eq", 2024), ("tender_cpv_id", "eq", "72221000")],
        )

    def test_type_name_nesting_unwrapped(self):
        # nano's nesting key drifts: sometimes the plan sits under the TYPE NAME itself.
        payload = {
            "question_type": "count",
            "count": {
                "operation": "count",
                "constraints": [
                    {"slot": "year", "surface": "published in 2024", "value": "2024", "operator": "eq"},
                    {"slot": "cpv", "surface": "CPV 72221000", "value": "72221000", "operator": "eq"},
                ],
            },
        }
        candidate = compile_typed_plan(
            "How many notices were published in 2024 under CPV 72221000?", payload)
        self.assertEqual(candidate.status, "planned")
        self.assertEqual(
            [(c.field, c.op, c.value) for c in candidate.query_spec.constraints],
            [("release_year", "eq", 2024), ("tender_cpv_id", "eq", "72221000")],
        )

    def test_operation_alternation_echo_resolved(self):
        payload = {
            "question_type": "boolean",
            "operation": "exists|predicate",
            "constraints": [{"slot": "year", "surface": "2026", "value": 2026, "operator": "eq"}],
        }
        verdict = plan_consistency_check(
            "Were any notices published in 2026?", payload)
        self.assertTrue(verdict.ok, verdict.issues)
        candidate = compile_typed_plan("Were any notices published in 2026?", payload)
        self.assertEqual(candidate.status, "planned")
        self.assertEqual(candidate.query_spec.answer_operation, "exists")

    def test_placeholder_slot_with_splittable_value_expanded(self):
        # echoed placeholder slot whose value carries clean ';'-separated atoms: split and
        # re-slot deterministically (org role from the surface cue).
        payload = {
            "question_type": "factoid", "operation": "select_unique", "answer_field": "buyer",
            "constraints": [
                {"slot": "buyer|supplier|year|years|year_range|cpv|category|title|date",
                 "surface": "contract notices published in 2023 under CPV 33194220 by NHS Blood and Transplant",
                 "value": "NHS Blood and Transplant; 2023; CPV 33194220", "operator": "eq"},
                {"slot": "buyer|supplier|year|years|year_range|cpv|category|title|date",
                 "surface": "awarded to MacoPharma GMP.b", "value": "MacoPharma GMP.b", "operator": "eq"},
            ],
        }
        candidate = compile_typed_plan(
            "Which buyer was recorded for contract notices published in 2023 under CPV 33194220 "
            "awarded to MacoPharma GMP.b?", payload)
        self.assertEqual(candidate.status, "planned", candidate.rationale)
        compiled = {(c.field, c.op, c.value) for c in candidate.query_spec.constraints}
        self.assertIn(("buyer_name", "eq", "NHS Blood and Transplant"), compiled)
        self.assertIn(("release_year", "eq", 2023), compiled)
        self.assertIn(("tender_cpv_id", "eq", "33194220"), compiled)
        self.assertIn(("supplier_name", "eq", "MacoPharma GMP.b"), compiled)

    def test_placeholder_slot_is_structured_failure(self):
        payload = {
            "question_type": "factoid", "operation": "select_unique", "answer_field": "buyer",
            "constraints": [
                {"slot": "buyer|supplier|year|years|year_range|cpv|category|title|date",
                 "surface": "notices in 2023", "value": "NHS Blood and Transplant; 2023"},
            ],
        }
        candidate = compile_typed_plan(
            "Which buyer is recorded for notices in 2023?", payload)
        self.assertEqual(candidate.status, "ambiguous")
        self.assertIn("invalid_constraint_slot", candidate.rationale)

    def test_compiled_plan_dropping_atom_is_auto_completed(self):
        # a single year/CPV/category atom is an unambiguous regex extraction: instead of blocking,
        # compile completes the constraint deterministically and discloses it.
        payload = {
            "question_type": "count", "operation": "count",
            "notes": "CPV 72221000 mentioned",
            "constraints": [
                {"slot": "year", "surface": "published in 2024", "value": 2024, "operator": "eq"},
            ],
        }
        candidate = compile_typed_plan(
            "How many notices were published in 2024 under CPV 72221000?", payload)
        self.assertEqual(candidate.status, "planned", candidate.rationale)
        compiled = {(c.field, c.op, c.value) for c in candidate.query_spec.constraints}
        self.assertIn(("tender_cpv_id", "eq", "72221000"), compiled)
        self.assertTrue(any("auto-completed atoms" in w for w in candidate.warnings))

    def test_compiled_plan_dropping_quoted_title_is_blocked(self):
        # quoted titles are NOT auto-completable (tender vs award title is ambiguous): still block.
        payload = {
            "question_type": "count", "operation": "count",
            "constraints": [
                {"slot": "year", "surface": "2024", "value": 2024, "operator": "eq"},
            ],
        }
        candidate = compile_typed_plan(
            'How many notices titled "Temporary Chemical Dosing" were published in 2024?', payload)
        self.assertEqual(candidate.status, "ambiguous")
        self.assertIn("compiled_plan_dropped_atoms", candidate.rationale)
        self.assertIn("title:", candidate.rationale)

    def test_phantom_unquoted_title_filter_dropped(self):
        # L2 wrapper filler ("matching procurement records only") planned as a tender_title eq
        # filter zeroes out the match set; it is dropped with a disclosed note.
        payload = {
            "question_type": "count", "operation": "count",
            "constraints": [
                {"slot": "year", "surface": "published in 2023", "value": 2023, "operator": "eq"},
                {"slot": "cpv", "surface": "CPV 75251110", "value": "75251110", "operator": "eq"},
                {"slot": "title", "surface": "contract notices", "value": "contract notices", "operator": "eq"},
            ],
        }
        candidate = compile_typed_plan(
            "For the matching procurement records only, how many contract notices published in "
            "2023 under CPV 75251110 are recorded?", payload)
        self.assertEqual(candidate.status, "planned", candidate.rationale)
        fields = [c.field for c in candidate.query_spec.constraints]
        self.assertNotIn("tender_title", fields)
        self.assertTrue(any("phantom title" in w for w in candidate.warnings))

    def test_interrogative_answer_role_does_not_flip(self):
        # "which buyer ..." asks for the buyer while legitimately constraining the supplier.
        payload = {
            "question_type": "factoid", "operation": "select_unique", "answer_field": "buyer",
            "constraints": [
                {"slot": "supplier", "surface": "Nuclear Restoration Services Limited",
                 "value": "Nuclear Restoration Services Limited", "operator": "eq"},
                {"slot": "year", "surface": "2024", "value": 2024, "operator": "eq"},
            ],
        }
        verdict = plan_consistency_check(
            "For contract notices published in 2024, which buyer is recorded where Nuclear "
            "Restoration Services Limited was awarded the contract?", payload)
        self.assertNotIn("role_flipped:supplier:Nuclear Restoration Services", str(verdict.issues))

    def test_year_range_compiles_to_between_not_single_year(self):
        payload = {
            "question_type": "count",
            "operation": "count",
            "constraints": [
                {"slot": "year_range", "surface": "between 2022 and 2024",
                 "operator": "between", "value": [2022, 2024]},
            ],
        }
        candidate = compile_typed_plan("How many notices were published between 2022 and 2024?", payload)
        self.assertEqual(candidate.status, "planned")
        self.assertEqual(
            [(c.field, c.op, c.value) for c in candidate.query_spec.constraints],
            [("release_year", "between", [2022, 2024])],
        )

    def test_multiple_years_compile_to_in_not_conflicting_eqs(self):
        payload = {
            "question_type": "count",
            "operation": "count",
            "constraints": [
                {"slot": "years", "surface": "2022 or 2023",
                 "operator": "in", "value": [2022, 2023]},
            ],
        }
        candidate = compile_typed_plan("How many notices were published in 2022 or 2023?", payload)
        self.assertEqual(
            [(c.field, c.op, c.value) for c in candidate.query_spec.constraints],
            [("release_year", "in", [2022, 2023])],
        )


class TestIntentProgramCompiler(unittest.TestCase):
    def _args(self, filters=None, **extra):
        args = {"filters": filters or [], "field": "", "bind_field": "",
                "field_text": "", "bind_field_text": "", "metric": "", "metric_text": "",
                "comparator": "", "threshold": "", "k": 0}
        args.update(extra)
        return args

    def test_intent_program_count_compiles_and_ignores_cpv_label(self):
        payload = {
            "answer_signature": {"operation": "count", "value_type": "number", "answer_field": "contract"},
            "program": [
                {"id": "A", "op": "filter_records", "inputs": [],
                 "args": self._args([
                     {"slot": "year", "operator": "eq", "value": "2024"},
                     {"slot": "cpv_code", "operator": "eq", "value": "85149000"},
                     {"slot": "cpv_label", "operator": "eq", "value": "Pharmacy services"},
                 ]), "returns": "record_set"},
                {"id": "B", "op": "count", "inputs": ["A"], "args": self._args(), "returns": "number"},
            ],
            "answer_step": "B",
            "unsupported_or_ambiguous": [],
        }
        candidate = compile_typed_plan(
            "How many notices were published in 2024 under CPV 85149000 (Pharmacy services)?",
            payload,
        )
        self.assertEqual(candidate.status, "planned", candidate.rationale)
        compiled = {(c.field, c.op, c.value) for c in candidate.query_spec.constraints}
        self.assertIn(("release_year", "eq", 2024), compiled)
        self.assertIn(("tender_cpv_id", "eq", "85149000"), compiled)
        self.assertNotIn(("tender_category", "eq", "Pharmacy services"), compiled)

    def test_intent_program_rejects_cpv_label_as_procurement_category(self):
        payload = {
            "answer_signature": {"operation": "count", "value_type": "number", "answer_field": "contract"},
            "program": [
                {"id": "A", "op": "filter_records", "inputs": [],
                 "args": self._args([
                     {"slot": "procurement_category", "operator": "eq", "value": "Pharmacy services"},
                 ]), "returns": "record_set"},
                {"id": "B", "op": "count", "inputs": ["A"], "args": self._args(), "returns": "number"},
            ],
            "answer_step": "B",
            "unsupported_or_ambiguous": [],
        }
        candidate = compile_typed_plan("How many notices under CPV 85149000 (Pharmacy services)?", payload)
        self.assertEqual(candidate.status, "ambiguous")
        self.assertIn("invalid_intent_program", candidate.rationale)

    def test_intent_program_abstain_for_ambiguous_question(self):
        payload = {
            "answer_signature": {"operation": "abstain", "value_type": "unknown", "answer_field": "none"},
            "program": [],
            "answer_step": "",
            "unsupported_or_ambiguous": ["The question does not identify a unique contract."],
        }
        candidate = compile_typed_plan(
            "Which supplier is listed as the awarded supplier for a contract involving LiveWest Homes Limited?",
            payload,
        )
        self.assertEqual(candidate.status, "unsupported")

    def test_planner_uses_intent_program_without_stage2_llm(self):
        payload = {
            "answer_signature": {"operation": "count", "value_type": "number", "answer_field": "contract"},
            "program": [
                {"id": "A", "op": "filter_records", "inputs": [],
                 "args": self._args([{"slot": "year", "operator": "eq", "value": "2024"}]),
                 "returns": "record_set"},
                {"id": "B", "op": "count", "inputs": ["A"], "args": self._args(), "returns": "number"},
            ],
            "answer_step": "B",
            "unsupported_or_ambiguous": [],
        }
        chat = _IntentSchemaChat(payload)
        [candidate] = TypedLLMPlanner(client=chat, model="stub").plan(
            "How many notices were published in 2024?")
        self.assertEqual(candidate.status, "planned", candidate.rationale)
        self.assertEqual(chat.schema_calls, 1)
        self.assertEqual(chat.json_calls, 0)
        self.assertEqual(candidate.raw_response["teacher"]["plan"], "deterministic_intent_compiler")

    def test_intent_program_v1_field_text_grounds_to_canonical_filters(self):
        payload = {
            "answer_signature": {"operation": "count", "value_type": "number", "answer_field_text": ""},
            "program": [
                {"id": "A", "op": "filter_records", "inputs": [],
                 "args": self._args([
                     {"field_text": "published year", "operator": "eq", "value": "2024", "value_type": "year"},
                     {"field_text": "CPV code", "operator": "eq", "value": "85149000", "value_type": "cpv"},
                     {"field_text": "CPV label", "operator": "eq", "value": "Pharmacy services", "value_type": "text"},
                 ]), "returns": "record_set"},
                {"id": "B", "op": "count", "inputs": ["A"], "args": self._args(), "returns": "number"},
            ],
            "answer_step": "B",
            "unsupported_or_ambiguous": [],
        }
        candidate = compile_typed_plan(
            "How many notices were published in 2024 under CPV 85149000 (Pharmacy services)?",
            payload,
        )
        self.assertEqual(candidate.status, "planned", candidate.rationale)
        compiled = {(c.field, c.op, c.value) for c in candidate.query_spec.constraints}
        self.assertIn(("release_year", "eq", 2024), compiled)
        self.assertIn(("tender_cpv_id", "eq", "85149000"), compiled)
        self.assertFalse(any(c.field == "tender_category" for c in candidate.query_spec.constraints))

    def test_intent_program_count_over_final_filter_step_compiles(self):
        # nano's dominant idiom: the program ends at the filtered set and answer_signature names
        # the reduction. That is complete — the graph return carries the count; requiring a
        # ceremonial final count step rejected 21/50 of the v5 live run.
        payload = {
            "answer_signature": {"operation": "count", "value_type": "number", "answer_field_text": ""},
            "program": [
                {"id": "A", "op": "filter_records", "inputs": [],
                 "args": self._args([{"field_text": "published year", "operator": "eq",
                                      "value": "2024", "value_type": "year"}]),
                 "returns": "record_set"},
            ],
            "answer_step": "A",
            "unsupported_or_ambiguous": [],
        }
        candidate = compile_typed_plan("How many notices were published in 2024?", payload)
        self.assertEqual(candidate.status, "planned", candidate.rationale)
        self.assertIsNotNone(candidate.graph_plan)
        self.assertEqual(candidate.graph_plan.return_spec.operation, "count")
        self.assertEqual(candidate.graph_plan.return_spec.input_id, "A")

    def test_intent_program_select_step_may_return_entity_set(self):
        # select IS the unique-value reduction; runtime uniqueness enforces singularity. nano
        # labels the step returns=entity_set — shorthand, not a contract violation.
        payload = {
            "answer_signature": {"operation": "select", "value_type": "entity",
                                 "answer_field_text": "supplier"},
            "program": [
                {"id": "A", "op": "filter_records", "inputs": [],
                 "args": self._args([{"field_text": "CPV code", "operator": "eq",
                                      "value": "45432130", "value_type": "cpv"}]),
                 "returns": "record_set"},
                {"id": "B", "op": "select", "inputs": ["A"],
                 "args": self._args(field_text="supplier"), "returns": "entity_set"},
            ],
            "answer_step": "B",
            "unsupported_or_ambiguous": [],
        }
        candidate = compile_typed_plan(
            "Which supplier is listed for contracts under CPV 45432130?", payload)
        self.assertEqual(candidate.status, "planned", candidate.rationale)

    def test_intent_program_hedge_notes_do_not_force_abstain(self):
        # reasons + an executable program + non-abstain operation is a HEDGE: the plan proceeds
        # (executor/verifier judge it) and the note rides along as a caveat. Treating hedges as
        # abstain violations threw away 21/50 answerable rows in the v5/v6 live runs.
        payload = {
            "answer_signature": {"operation": "count", "value_type": "number", "answer_field_text": ""},
            "program": [
                {"id": "A", "op": "filter_records", "inputs": [],
                 "args": self._args([{"field_text": "published year", "operator": "eq",
                                      "value": "2024", "value_type": "year"}]),
                 "returns": "record_set"},
            ],
            "answer_step": "A",
            "unsupported_or_ambiguous": ["'only the matching procurement records' wording is vague"],
        }
        candidate = compile_typed_plan("In the matching records, how many notices in 2024?", payload)
        self.assertEqual(candidate.status, "planned", candidate.rationale)
        # the caveat must survive into the compiled plan for disclosure/teacher visibility
        network = candidate.graph_plan.understanding_network
        self.assertIn("wording is vague", json.dumps(network, default=str))

    def test_intent_program_rejects_invented_root_input(self):
        payload = {
            "answer_signature": {"operation": "count", "value_type": "number", "answer_field_text": ""},
            "program": [
                {"id": "A", "op": "filter_records", "inputs": ["contract_notices"],
                 "args": self._args([{"field_text": "published year", "operator": "eq",
                                      "value": "2024", "value_type": "year"}]),
                 "returns": "record_set"},
                {"id": "B", "op": "count", "inputs": ["A"], "args": self._args(), "returns": "number"},
            ],
            "answer_step": "B",
            "unsupported_or_ambiguous": [],
        }
        candidate = compile_typed_plan("How many notices were published in 2024?", payload)
        self.assertEqual(candidate.status, "ambiguous")
        self.assertIn("unknown_inputs:contract_notices", candidate.rationale)

    def test_abstain_with_candidate_program_is_rejected(self):
        payload = {
            "answer_signature": {"operation": "abstain", "value_type": "unknown", "answer_field_text": ""},
            "program": [
                {"id": "A", "op": "filter_records", "inputs": [], "args": self._args(), "returns": "record_set"},
            ],
            "answer_step": "",
            "unsupported_or_ambiguous": ["The question does not identify a unique contract."],
        }
        candidate = compile_typed_plan("Which supplier for a contract involving LiveWest Homes Limited?", payload)
        self.assertEqual(candidate.status, "ambiguous")
        self.assertIn("abstain_program_must_be_empty", candidate.rationale)

    def test_supplier_to_buyer_set_is_executable_not_ambiguous(self):
        payload = {
            "answer_signature": {"operation": "distinct_set", "value_type": "set",
                                 "answer_field_text": "buyer organisation"},
            "program": [
                {"id": "A", "op": "filter_records", "inputs": [],
                 "args": self._args([{"field_text": "supplier organisation", "operator": "eq",
                                      "value": "Access UK Ltd", "value_type": "entity"}]),
                 "returns": "record_set"},
                {"id": "B", "op": "distinct", "inputs": ["A"],
                 "args": self._args(field_text="buyer organisation"),
                 "returns": "set"},
            ],
            "answer_step": "B",
            "unsupported_or_ambiguous": [],
        }
        candidate = compile_typed_plan("Which buyer awarded a contract to Access UK Ltd?", payload)
        self.assertEqual(candidate.status, "planned", candidate.rationale)
        self.assertEqual(candidate.query_spec.answer_operation, "distinct_set")
        self.assertEqual(candidate.query_spec.answer_field, "buyer_name")

    def test_operation_contract_rejects_sum_over_entity_set_final_step(self):
        # Contract relaxed 2026-07-04: a reduction MAY run over the program's final SET step
        # (the ceremonial-final-step rule rejected 21/50 real nano programs in the v5 live run).
        # What stays rejected is a reduction the return spec cannot express: sum needs records.
        payload = {
            "answer_signature": {"operation": "sum", "value_type": "money", "answer_field_text": "contract value"},
            "program": [
                {"id": "A", "op": "filter_records", "inputs": [],
                 "args": self._args([{"field_text": "published year", "operator": "eq",
                                      "value": "2024", "value_type": "year"}]),
                 "returns": "record_set"},
                {"id": "B", "op": "distinct", "inputs": ["A"],
                 "args": self._args(field_text="supplier"), "returns": "entity_set"},
            ],
            "answer_step": "B",
            "unsupported_or_ambiguous": [],
        }
        candidate = compile_typed_plan("What is the total value won by 2024 suppliers?", payload)
        self.assertEqual(candidate.status, "ambiguous")
        self.assertIn("operation_contract:sum_requires_final_sum_got_distinct", candidate.rationale)
        issue = candidate.raw_response["intent_issues"][0]
        self.assertEqual(issue["stage"], "intent_program_validation")
        self.assertEqual(issue["error_type"], "answer_step_operation_mismatch")
        self.assertEqual(issue["answer_operation"], "sum")
        self.assertEqual(issue["answer_step"], "B")
        self.assertEqual(issue["answer_step_op"], "distinct")

    def test_operation_contract_requires_answer_step(self):
        payload = {
            "answer_signature": {"operation": "count", "value_type": "number", "answer_field_text": ""},
            "program": [
                {"id": "A", "op": "filter_records", "inputs": [],
                 "args": self._args([{"field_text": "published year", "operator": "eq",
                                      "value": "2024", "value_type": "year"}]),
                 "returns": "record_set"},
                {"id": "B", "op": "count", "inputs": ["A"], "args": self._args(), "returns": "number"},
            ],
            "answer_step": "",
            "unsupported_or_ambiguous": [],
        }
        candidate = compile_typed_plan("How many notices were published in 2024?", payload)
        self.assertEqual(candidate.status, "ambiguous")
        self.assertIn("operation_contract:missing_answer_step", candidate.rationale)

    def test_select_and_distinct_require_field_text(self):
        for operation, final_op in (("select", "select"), ("distinct_set", "distinct")):
            payload = {
                "answer_signature": {"operation": operation, "value_type": "set" if operation == "distinct_set" else "entity",
                                     "answer_field_text": ""},
                "program": [
                    {"id": "A", "op": "filter_records", "inputs": [],
                     "args": self._args([{"field_text": "published year", "operator": "eq",
                                          "value": "2024", "value_type": "year"}]),
                     "returns": "record_set"},
                    {"id": "B", "op": final_op, "inputs": ["A"], "args": self._args(),
                     "returns": "set" if operation == "distinct_set" else "entity"},
                ],
                "answer_step": "B",
                "unsupported_or_ambiguous": [],
            }
            candidate = compile_typed_plan("Which supplier was recorded in 2024?", payload)
            self.assertEqual(candidate.status, "ambiguous")
            self.assertIn(f"operation_contract:{final_op}_requires_field_text:B", candidate.rationale)

    def test_sum_requires_metric_text(self):
        payload = {
            "answer_signature": {"operation": "sum", "value_type": "money", "answer_field_text": "contract value"},
            "program": [
                {"id": "A", "op": "filter_records", "inputs": [],
                 "args": self._args([{"field_text": "published year", "operator": "eq",
                                      "value": "2024", "value_type": "year"}]),
                 "returns": "record_set"},
                {"id": "B", "op": "sum", "inputs": ["A"], "args": self._args(), "returns": "money"},
            ],
            "answer_step": "B",
            "unsupported_or_ambiguous": [],
        }
        candidate = compile_typed_plan("What is the total value in 2024?", payload)
        self.assertEqual(candidate.status, "ambiguous")
        self.assertIn("operation_contract:sum_requires_metric_text:B", candidate.rationale)

    def test_grounding_error_has_reflector_friendly_payload(self):
        payload = {
            "answer_signature": {"operation": "count", "value_type": "number", "answer_field_text": ""},
            "program": [
                {"id": "A", "op": "filter_records", "inputs": [],
                 "args": self._args([{"field_text": "high-level procurement category", "operator": "eq",
                                      "value": "Pharmacy services", "value_type": "category"}]),
                 "returns": "record_set"},
                {"id": "B", "op": "count", "inputs": ["A"], "args": self._args(), "returns": "number"},
            ],
            "answer_step": "B",
            "unsupported_or_ambiguous": [],
        }
        candidate = compile_typed_plan("How many notices under CPV 85149000 (Pharmacy services)?", payload)
        self.assertEqual(candidate.status, "ambiguous")
        issue = candidate.raw_response["intent_issues"][0]
        self.assertEqual(issue["stage"], "schema_grounding")
        self.assertEqual(issue["error_type"], "type_gate_rejected")
        self.assertEqual(issue["field_text"], "high-level procurement category")
        self.assertEqual(issue["candidate"], "procurement_category")
        self.assertIn("goods/services/works", issue["reason"])
        self.assertTrue(issue["suggested_actions"])


class TestTypedLLMPlanner(unittest.TestCase):
    def test_consistency_failure_never_compiles_as_planned(self):
        payload = {
            "question_type": "count",
            "operation": "count",
            "constraints": [{"slot": "year", "surface": "2024", "value": 2025}],
        }
        planner = TypedLLMPlanner(client=_Chat(payload), model="stub")
        [candidate] = planner.plan("How many notices were published in 2024?")
        self.assertEqual(candidate.status, "ambiguous")
        self.assertIn("plan_semantic_mismatch", candidate.rationale)

    def test_valid_payload_returns_typed_candidate(self):
        payload = {
            "question_type": "count",
            "operation": "count",
            "constraints": [{"slot": "year", "surface": "2024", "value": "2024"}],
        }
        chat = _Chat(payload)
        planner = TypedLLMPlanner(client=chat, model="stub")
        [candidate] = planner.plan("How many notices were published in 2024?")
        self.assertEqual(candidate.status, "planned")
        self.assertEqual(candidate.planner_source, "typed_llm")
        self.assertEqual(chat.calls, 2)
        self.assertIn("understanding", candidate.raw_response)
        self.assertIn("typed_plan", candidate.raw_response)

    def test_understanding_accepts_labelled_lines_before_json_plan(self):
        text = "\n".join([
            "QUESTION_TYPE: count",
            "NEEDS_TO_RETURN: number of notices",
            "KNOWN_INFORMATION: 2024",
            "ROLE_DIRECTION: no organisation role",
            "REASONING_CHAIN: count matching notices",
            "MISSING_OR_UNSUPPORTED_INFORMATION: null",
            "NOTES_FOR_PLANNER: preserve year",
        ])
        payload = {
            "question_type": "count",
            "operation": "count",
            "constraints": [{"slot": "year", "surface": "2024", "value": "2024"}],
        }
        chat = _TextThenJsonChat(text, payload)
        planner = TypedLLMPlanner(client=chat, model="stub")
        [candidate] = planner.plan("How many notices were published in 2024?")
        self.assertEqual(candidate.status, "planned")
        self.assertEqual(candidate.raw_response["understanding"]["question_type"], "count")

    def test_single_call_variant_skips_understanding_step(self):
        payload = {
            "question_type": "count",
            "operation": "count",
            "constraints": [{"slot": "year", "surface": "2024", "value": "2024"}],
        }
        chat = _Chat(payload)
        planner = TypedLLMPlanner(client=chat, model="stub", two_step=False)
        [candidate] = planner.plan("How many notices were published in 2024?")
        self.assertEqual(candidate.status, "planned")
        self.assertEqual(chat.calls, 1)
        self.assertIsNone(candidate.raw_response["understanding"])

    def test_feedback_replan_valid_payload_returns_candidate(self):
        payload = {
            "question_type": "count",
            "operation": "count",
            "constraints": [{"slot": "year", "surface": "2024", "value": "2024"}],
        }
        chat = _Chat(payload)
        planner = TypedLLMPlanner(client=chat, model="stub")
        [candidate] = planner.replan_with_feedback("How many notices were published in 2024?",
                                                   {"execution_status": "no_results"})
        self.assertEqual(candidate.status, "planned")
        self.assertTrue(candidate.plan_id.endswith(":feedback"))
        self.assertIn("feedback", candidate.raw_response)
        self.assertIn("repair_understanding", candidate.raw_response)
        self.assertEqual(chat.calls, 2)

    def test_feedback_replan_single_call_variant_skips_understanding_step(self):
        payload = {
            "question_type": "count",
            "operation": "count",
            "constraints": [{"slot": "year", "surface": "2024", "value": "2024"}],
        }
        chat = _Chat(payload)
        planner = TypedLLMPlanner(client=chat, model="stub", two_step=False)
        [candidate] = planner.replan_with_feedback("How many notices were published in 2024?",
                                                   {"execution_status": "no_results"})
        self.assertEqual(candidate.status, "planned")
        self.assertEqual(chat.calls, 1)
        self.assertIsNone(candidate.raw_response["repair_understanding"])

    def test_feedback_replan_wrapper_can_fix_question_type(self):
        payload = {
            "diagnosis": "The failed plan counted notices, but the question asks total value.",
            "repair_action": "fix_question_type",
            "repaired_plan": {
                "question_type": "sum",
                "operation": "sum",
                "constraints": [{"slot": "year", "surface": "2024", "value": "2024"}],
            },
            "changed_fields": ["question_type", "operation"],
            "unchanged_fields": ["constraints"],
        }
        planner = TypedLLMPlanner(client=_Chat(payload), model="stub")
        [candidate] = planner.replan_with_feedback("What was the total value in 2024?",
                                                   {"failure_stage": "verifier"})
        self.assertEqual(candidate.status, "planned")
        self.assertEqual(candidate.query_spec.answer_operation, "sum")
        self.assertEqual(candidate.raw_response["reflector_output"]["repair_action"], "fix_question_type")

    def test_feedback_replan_uses_repair_json_schema_when_available(self):
        payload = {
            "diagnosis": "The failed plan missed the year filter.",
            "repair_action": "repair_constraints",
            "repaired_understanding_briefing": "Answer Type: count; Explicit Info: year = 2024",
            "repaired_graph_plan": {
                "question_type": "count",
                "operation": "count",
                "variables": [
                    {"var_id": "a1", "kind": "record_set", "role": "contract_records",
                     "filters": [{"slot": "year", "value": "2024", "operator": "eq"}],
                     "depends_on": []},
                ],
                "return": {"operation": "count", "input": "a1", "field": "contract"},
            },
            "changed_fields": ["variables.a1.filters"],
            "unchanged_fields": ["return.operation"],
        }
        chat = _SchemaRepairChat(payload)
        planner = TypedLLMPlanner(client=chat, model="stub")
        [candidate] = planner.replan_with_feedback("How many notices were published in 2024?",
                                                   {"execution_status": "no_results"})
        self.assertEqual(candidate.status, "planned")
        self.assertEqual(candidate.raw_response["reflector_output"]["repair_action"], "repair_constraints")
        self.assertEqual(chat.schemas[-1]["name"], "procurement_graph_repair")
        self.assertIn("failed Stage-2 graph plan", chat.text_users[0])
        self.assertIn("no_results", chat.text_users[0])
        self.assertEqual(candidate.raw_response["typed_plan"]["graph_plan"]["variables"][0]["var_id"], "a1")

    def test_schema_auth_error_does_not_fallback_to_plain_json(self):
        payload = {
            "graph_plan": {
                "question_type": "count",
                "operation": "count",
                "variables": [
                    {"var_id": "a1", "kind": "record_set", "role": "contracts",
                     "filters": [{"slot": "year", "value": "2024"}]},
                ],
                "return": {"operation": "count", "input": "a1", "field": "contract"},
            }
        }
        chat = _SchemaFallbackChat(
            schema_exc=RuntimeError("Error code: 401 - invalid subscription key or wrong API endpoint"),
            fallback_payload=payload,
        )
        [candidate] = TypedLLMPlanner(client=chat, model="stub", two_step=False).plan(
            "How many notices were published in 2024?")
        self.assertEqual(candidate.status, "ambiguous")
        self.assertIn("llm_error", candidate.rationale)
        self.assertEqual(chat.schema_calls, 1)
        self.assertEqual(chat.json_calls, 0)

    def test_invalid_schema_error_does_not_fallback_to_plain_json(self):
        payload = {
            "graph_plan": {
                "question_type": "count",
                "operation": "count",
                "variables": [
                    {"var_id": "a1", "kind": "record_set", "role": "contracts",
                     "filters": [{"slot": "year", "value": "2024"}]},
                ],
                "return": {"operation": "count", "input": "a1", "field": "contract"},
            }
        }
        chat = _SchemaFallbackChat(
            schema_exc=RuntimeError("Error code: 400 - Invalid schema for response_format json_schema"),
            fallback_payload=payload,
        )
        [candidate] = TypedLLMPlanner(client=chat, model="stub", two_step=False).plan(
            "How many notices were published in 2024?")
        self.assertEqual(candidate.status, "ambiguous")
        self.assertIn("Invalid schema", candidate.rationale)
        self.assertEqual(chat.schema_calls, 1)
        self.assertEqual(chat.json_calls, 0)

    def test_recoverable_schema_error_can_fallback_to_plain_json(self):
        payload = {
            "graph_plan": {
                "question_type": "count",
                "operation": "count",
                "variables": [
                    {"var_id": "a1", "kind": "record_set", "role": "contracts",
                     "filters": [{"slot": "year", "value": "2024"}]},
                ],
                "return": {"operation": "count", "input": "a1", "field": "contract"},
            }
        }
        chat = _SchemaFallbackChat(schema_exc=RuntimeError("temporary schema failure"),
                                   fallback_payload=payload)
        [candidate] = TypedLLMPlanner(client=chat, model="stub", two_step=False).plan(
            "How many notices were published in 2024?")
        self.assertEqual(candidate.status, "planned")
        self.assertEqual(chat.schema_calls, 1)
        self.assertEqual(chat.json_calls, 1)


if __name__ == "__main__":
    unittest.main()
