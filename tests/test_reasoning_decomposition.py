"""General N-hop decomposition (Phase 2): 2-hop bridge, 3-hop chain, compare, depth guard,
empty-bind short-circuit. Oracles are hand-computed from the mock records."""

from __future__ import annotations

import sys
import unittest
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from procurement_graph.reasoning.decomposition import (
    Binding, DecompositionPlan, SubQuery, execute_decomposition,
)
from procurement_graph.reasoning.models import QueryConstraint, RuntimeQuerySpec
from tests.test_reasoning_runtime import TabularRuntimeBackend


def _records():
    # B1 uses S1,S2 ; S1 also -> B2 ; S2 also -> B3 ; B2,B3 also use S9
    common = {"value_is_additive": True, "release_year": 2024}
    return [
        {"contract_node_id": "c1", "buyer_name": "B1", "supplier_name": "S1", "value_amount": "100", "tender_category": "services", **common},
        {"contract_node_id": "c2", "buyer_name": "B1", "supplier_name": "S2", "value_amount": "200", "tender_category": "services", **common},
        {"contract_node_id": "c3", "buyer_name": "B2", "supplier_name": "S1", "value_amount": "50", "tender_category": "works", **common},
        {"contract_node_id": "c4", "buyer_name": "B3", "supplier_name": "S2", "value_amount": "70", "tender_category": "works", **common},
        {"contract_node_id": "c5", "buyer_name": "B2", "supplier_name": "S9", "value_amount": "10", "tender_category": "goods", **common},
        {"contract_node_id": "c6", "buyer_name": "B3", "supplier_name": "S9", "value_amount": "20", "tender_category": "goods", **common},
    ]


def _spec(op, *, constraints=(), answer_field="", value_type="string"):
    return RuntimeQuerySpec(spec_id="s", question="q", intent="x",
                            constraints=tuple(QueryConstraint(*c) for c in constraints),
                            answer_operation=op, answer_field=answer_field, answer_value_type=value_type,
                            requires_exhaustive_retrieval=True)


def _entity(step_id, emit_field, *, constraints=(), binds=()):
    return SubQuery(step_id=step_id, spec=_spec("count", constraints=constraints), binds=binds,
                    kind="entity_set", emit_field=emit_field)


class TestDecomposition(unittest.TestCase):
    def setUp(self):
        self.backend = TabularRuntimeBackend(_records())

    def test_2hop_bridge_sum(self):
        # B1 -> suppliers {S1,S2} -> sum value over ALL their contracts (c1..c4) = 420
        plan = DecompositionPlan("p", "q", steps=(
            _entity("suppliers", "supplier_name", constraints=(("buyer_name", "eq", "B1"),)),
            SubQuery("total", _spec("sum", answer_field="value_amount", value_type="currency"),
                     binds=(Binding("suppliers", "supplier_name"),)),
        ), final_steps=("total",))
        result = execute_decomposition(self.backend, plan)
        self.assertEqual(result.status, "passed")
        self.assertEqual(result.answer, Decimal("420"))
        self.assertEqual(len(result.hops), 2)

    def test_3hop_chain_count(self):
        # B1 -> suppliers {S1,S2} -> buyers of those {B1,B2,B3} -> count their contracts (all 6)
        plan = DecompositionPlan("p", "q", steps=(
            _entity("suppliers", "supplier_name", constraints=(("buyer_name", "eq", "B1"),)),
            _entity("buyers", "buyer_name", binds=(Binding("suppliers", "supplier_name"),)),
            SubQuery("n", _spec("count"), binds=(Binding("buyers", "buyer_name"),)),
        ), final_steps=("n",))
        result = execute_decomposition(self.backend, plan)
        self.assertEqual(result.status, "passed")
        self.assertEqual(result.answer, 6)
        self.assertEqual([h.kind for h in result.hops], ["entity_set", "entity_set", "answer"])

    def test_compare_two_independent_counts(self):
        # B2 has 2 (c3,c5), B3 has 2 (c4,c6) -> B2 > B3 is False
        plan = DecompositionPlan("p", "q", steps=(
            SubQuery("a", _spec("count", constraints=(("buyer_name", "eq", "B2"),))),
            SubQuery("b", _spec("count", constraints=(("buyer_name", "eq", "B3"),))),
        ), combine="compare_gt", final_steps=("a", "b"))
        result = execute_decomposition(self.backend, plan)
        self.assertEqual(result.status, "passed")
        self.assertEqual(result.answer["answer"], False)
        self.assertEqual((result.answer["a"], result.answer["b"]), (2, 2))

    def test_depth_guard(self):
        steps = tuple(SubQuery(f"s{i}", _spec("count")) for i in range(5))
        plan = DecompositionPlan("p", "q", steps=steps, max_hops=4)
        self.assertEqual(execute_decomposition(self.backend, plan).status, "unsupported")

    def test_empty_bind_short_circuits(self):
        plan = DecompositionPlan("p", "q", steps=(
            _entity("suppliers", "supplier_name", constraints=(("buyer_name", "eq", "NOPE"),)),
            SubQuery("total", _spec("sum", answer_field="value_amount", value_type="currency"),
                     binds=(Binding("suppliers", "supplier_name"),)),
        ), final_steps=("total",))
        result = execute_decomposition(self.backend, plan)
        self.assertEqual(result.status, "no_results")
        self.assertEqual(result.hops[0].status, "no_results")


class _Resolver:
    def __init__(self, names):
        self.names = set(names)

    def resolve(self, mention):
        from procurement_graph.reasoning.models import EntityLinkCandidate
        if mention in self.names:
            return [EntityLinkCandidate(mention=mention, linked_id=mention, linked_label=mention,
                                        entity_type="organization", score=1.0, source="records_exact")]
        return []


class TestDecompositionPlannerEndToEnd(unittest.TestCase):
    def setUp(self):
        from procurement_graph.reasoning import ReasoningPipeline
        from procurement_graph.reasoning.planner_decomposition import DecompositionAwarePlanner
        self.backend = TabularRuntimeBackend(_records())
        resolver = _Resolver(["B1", "B2", "B3", "S1", "S2"])
        self.pipeline = ReasoningPipeline(backend=self.backend,
                                          planner=DecompositionAwarePlanner(org_resolver=resolver),
                                          org_resolver=resolver)

    def test_argmax_end_to_end(self):
        trace = self.pipeline.run("Which contract has the highest value?")
        self.assertTrue(trace.execution.passed)
        self.assertEqual(trace.answer_card.answer, "c2")  # 200 is the max

    def test_top_k_end_to_end(self):
        trace = self.pipeline.run("What are the top 3 buyers by number of contract notices?")
        self.assertTrue(trace.execution.passed)
        self.assertEqual(len(trace.answer_card.answer), 3)

    def test_exists_end_to_end(self):
        trace = self.pipeline.run("Did B1 publish any contract notices?")
        self.assertTrue(trace.execution.passed)
        self.assertEqual(trace.answer_card.answer, True)

    def test_bridge_sum_end_to_end(self):
        trace = self.pipeline.run("What is the total value of contracts awarded to suppliers who also won a contract from B1?")
        self.assertTrue(trace.execution.passed)
        self.assertEqual(str(trace.answer_card.answer), "420")
        self.assertEqual(trace.metadata["decomposition"]["status"], "passed")

    def test_compare_end_to_end(self):
        trace = self.pipeline.run("Did B2 publish more contract notices than B3 in 2024?")
        self.assertTrue(trace.execution.passed)
        self.assertEqual(trace.answer_card.answer["answer"], False)  # 2 vs 2

    def test_exists_with_conversational_leadin(self):
        # 'did' is mid-sentence, not sentence-initial -> the recogniser must not require startswith
        trace = self.pipeline.run("I am checking the procurement data: did B1 publish any contract notices?")
        self.assertTrue(trace.execution.passed)
        self.assertEqual(trace.answer_card.answer, True)

    def test_compare_with_conversational_leadin(self):
        trace = self.pipeline.run("Looking only at the matching records, did B2 publish more contract notices than B3 in 2024?")
        self.assertTrue(trace.execution.passed)
        self.assertEqual(trace.answer_card.answer["answer"], False)

    def test_role_factoid_buyer_is_scalar(self):
        # singular role question -> select_unique scalar (not a distinct_set list)
        trace = self.pipeline.run("Who was the buyer for the works contract with S1?")
        self.assertTrue(trace.execution.passed)
        self.assertEqual(trace.answer_card.answer, "B2")  # c3 is S1's only works contract

    def test_role_factoid_supplier_is_scalar(self):
        trace = self.pipeline.run("Which supplier is recorded for goods contract notices by B2?")
        self.assertTrue(trace.execution.passed)
        self.assertEqual(trace.answer_card.answer, "S9")  # c5

    def test_role_factoid_ambiguous_abstains(self):
        # B1 awarded services contracts to S1 AND S2 -> select_unique must refuse, not guess
        trace = self.pipeline.run("Which supplier is recorded for services contract notices by B1?")
        self.assertIsNone(trace.answer_card.answer)

    def test_bridge_org_name_containing_category_word(self):
        # 'Services' inside the org name must NOT become a tender_category filter on hop 2
        backend = TabularRuntimeBackend([
            {**r, "supplier_name": ("Atos IT Services UK" if r["supplier_name"] == "S1" else r["supplier_name"])}
            for r in _records()])
        resolver = _Resolver(["B1", "B2", "B3", "Atos IT Services UK", "S2"])
        from procurement_graph.reasoning import ReasoningPipeline
        from procurement_graph.reasoning.planner_decomposition import DecompositionAwarePlanner
        pipeline = ReasoningPipeline(backend=backend, planner=DecompositionAwarePlanner(org_resolver=resolver),
                                     org_resolver=resolver)
        trace = pipeline.run("How many contract notices were published by buyers who have awarded a contract to Atos IT Services UK?")
        self.assertTrue(trace.execution.passed)
        # buyers of Atos = {B1, B2}; ALL their notices = c1,c2,c3,c5 (goods+works+services, no category filter)
        self.assertEqual(trace.answer_card.answer, 4)


class _StubDecompChat:
    """Stands in for the LLM: returns a canned decomposition payload (the JSON a nano would emit)."""

    def __init__(self, payload):
        self._payload = payload

    def complete_json(self, *, model, system, user):
        class _R:
            parsed = self._payload
        return _R()


def _bridge_payload():
    from procurement_graph.reasoning.planner import PLANNER_SCHEMA_VERSION
    return {"schema_version": PLANNER_SCHEMA_VERSION, "plans": [{
        "intent": "bridge", "answer_operation": "sum", "answer_field": "value_amount",
        "answer_value_type": "currency", "constraints": [],
        "decomposition": {
            "steps": [
                {"step_id": "h1", "kind": "entity_set", "emit_field": "supplier_name",
                 "constraints": [{"field": "buyer_name", "op": "eq", "value": "B1"}]},
                {"step_id": "ans", "kind": "answer", "answer_operation": "sum", "answer_field": "value_amount",
                 "constraints": [], "binds": [{"from_step": "h1", "into_field": "supplier_name"}]},
            ],
            "combine": "identity", "final_steps": ["ans"],
        }}]}


class TestLLMDecompositionAndHybrid(unittest.TestCase):
    def setUp(self):
        from procurement_graph.reasoning import ReasoningPipeline
        from procurement_graph.reasoning.llm_planner import LLMReasoningPlanner
        from procurement_graph.reasoning.planner_decomposition import DecompositionAwarePlanner, HybridPlanner
        self.backend = TabularRuntimeBackend(_records())
        self.resolver = _Resolver(["B1", "B2", "B3", "S1", "S2"])
        self.LLM = LLMReasoningPlanner
        self.Hybrid = HybridPlanner
        self.Rule = DecompositionAwarePlanner
        self.Pipeline = ReasoningPipeline

    def test_llm_emits_decomposition_and_executes(self):
        # The LLM (stub) proposes a bridge decomposition with a MENTION 'B1'; the engine grounds +
        # executes each hop. Sum over S1/S2 contracts = 420.
        llm = self.LLM(client=_StubDecompChat(_bridge_payload()), model="stub", org_resolver=self.resolver)
        cand = llm.plan("total value for vendors linked to B1")[0]
        self.assertIsNotNone(cand.decomposition)
        trace = self.Pipeline(backend=self.backend, planner=llm, org_resolver=self.resolver).run("total value for vendors linked to B1")
        self.assertTrue(trace.execution.passed)
        self.assertEqual(str(trace.answer_card.answer), "420")

    def test_hybrid_escalates_only_when_rule_ambiguous(self):
        llm = self.LLM(client=_StubDecompChat(_bridge_payload()), model="stub", org_resolver=self.resolver)
        hybrid = self.Hybrid(rule=self.Rule(org_resolver=self.resolver), llm=llm)
        # phrasing the rule recogniser does NOT map (no count/sum/factoid cue) -> ambiguous -> escalate
        cands = hybrid.plan("Tell me about the suppliers of B1")
        self.assertIsNotNone(cands[0].decomposition)
        # a shape the rule DOES handle -> no escalation (stays rule)
        rule_cands = hybrid.plan("Which contract has the highest value?")
        self.assertEqual(rule_cands[0].planner_source, "rule_decomposition")


if __name__ == "__main__":
    unittest.main()
