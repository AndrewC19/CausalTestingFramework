import unittest
import os
import shutil, tempfile
import pandas as pd
from itertools import combinations

from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.causal_specification import Scenario
from causal_testing.testing.metamorphic_relation import (
    ShouldCause,
    ShouldNotCause,
    generate_metamorphic_relations,
    generate_metamorphic_relation,
)
from causal_testing.specification.variable import Input, Output
from causal_testing.testing.base_test_case import BaseTestCase


class TestMetamorphicRelation(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir_path = tempfile.mkdtemp()
        self.dag_dot_path = os.path.join(self.temp_dir_path, "dag.dot")
        dag_dot = """digraph DAG { rankdir=LR; X1 -> Z; Z -> M; M -> Y; X2 -> Z; X3 -> M;}"""
        with open(self.dag_dot_path, "w") as f:
            f.write(dag_dot)
        self.dcg_dot_path = os.path.join(self.temp_dir_path, "dcg.dot")
        dcg_dot = """digraph dct { a -> b -> c -> d; d -> c; }"""
        with open(self.dcg_dot_path, "w") as f:
            f.write(dcg_dot)

        X1 = Input("X1", float)
        X2 = Input("X2", float)
        X3 = Input("X3", float)
        Z = Output("Z", float)
        M = Output("M", float)
        Y = Output("Y", float)
        self.scenario = Scenario(variables={X1, X2, X3, Z, M, Y})
        self.default_control_input_config = {"X1": 1, "X2": 2, "X3": 3}
        self.default_treatment_input_config = {"X1": 2, "X2": 3, "X3": 3}

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir_path)

    def test_should_not_cause_json_stub(self):
        """Test if the ShouldCause MR passes all metamorphic tests where the DAG perfectly represents the program
        and there is only a single input."""
        causal_dag = CausalDAG(self.dag_dot_path)
        causal_dag.graph.remove_nodes_from(["X2", "X3"])
        adj_set = list(causal_dag.direct_effect_adjustment_sets(["X1"], ["Z"])[0])
        should_not_cause_MR = ShouldNotCause(BaseTestCase("X1", "Z"), adj_set)
        self.assertEqual(
            should_not_cause_MR.to_json_stub(),
            {
                "effect": "direct",
                "estimate_type": "coefficient",
                "estimator": "LinearRegressionEstimator",
                "expected_effect": {"Z": "NoEffect"},
                "formula": "Z ~ X1",
                "mutations": ["X1"],
                "name": "X1 _||_ Z",
                "formula": "Z ~ X1",
                "alpha": 0.05,
                "skip": True,
            },
        )

    def test_should_cause_json_stub(self):
        """Test if the ShouldCause MR passes all metamorphic tests where the DAG perfectly represents the program
        and there is only a single input."""
        causal_dag = CausalDAG(self.dag_dot_path)
        causal_dag.graph.remove_nodes_from(["X2", "X3"])
        adj_set = list(causal_dag.direct_effect_adjustment_sets(["X1"], ["Z"])[0])
        should_cause_MR = ShouldCause(BaseTestCase("X1", "Z"), adj_set)
        self.assertEqual(
            should_cause_MR.to_json_stub(),
            {
                "effect": "direct",
                "estimate_type": "coefficient",
                "estimator": "LinearRegressionEstimator",
                "expected_effect": {"Z": "SomeEffect"},
                "formula": "Z ~ X1",
                "mutations": ["X1"],
                "name": "X1 --> Z",
                "skip": True,
            },
        )

    def test_all_metamorphic_relations_implied_by_dag(self):
        dag = CausalDAG(self.dag_dot_path)
        dag.add_edge("Z", "Y")  # Add a direct path from Z to Y so M becomes a mediator
        metamorphic_relations = generate_metamorphic_relations(dag)
        should_cause_relations = [mr for mr in metamorphic_relations if isinstance(mr, ShouldCause)]
        should_not_cause_relations = [mr for mr in metamorphic_relations if isinstance(mr, ShouldNotCause)]

        # Check all ShouldCause relations are present and no extra
        expected_should_cause_relations = [
            ShouldCause(BaseTestCase("X1", "Z"), []),
            ShouldCause(BaseTestCase("Z", "M"), []),
            ShouldCause(BaseTestCase("M", "Y"), ["Z"]),
            ShouldCause(BaseTestCase("Z", "Y"), ["M"]),
            ShouldCause(BaseTestCase("X2", "Z"), []),
            ShouldCause(BaseTestCase("X3", "M"), []),
        ]

        extra_sc_relations = [scr for scr in should_cause_relations if scr not in expected_should_cause_relations]
        missing_sc_relations = [escr for escr in expected_should_cause_relations if escr not in should_cause_relations]

        self.assertEqual(extra_sc_relations, [])
        self.assertEqual(missing_sc_relations, [])

        # Check all ShouldNotCause relations are present and no extra
        expected_should_not_cause_relations = [
            ShouldNotCause(BaseTestCase("X1", "X2"), []),
            ShouldNotCause(BaseTestCase("X1", "X3"), []),
            ShouldNotCause(BaseTestCase("X1", "M"), ["Z"]),
            ShouldNotCause(BaseTestCase("X1", "Y"), ["Z"]),
            ShouldNotCause(BaseTestCase("X2", "X3"), []),
            ShouldNotCause(BaseTestCase("X2", "M"), ["Z"]),
            ShouldNotCause(BaseTestCase("X2", "Y"), ["Z"]),
            ShouldNotCause(BaseTestCase("X3", "Y"), ["M", "Z"]),
            ShouldNotCause(BaseTestCase("Z", "X3"), []),
        ]

        extra_snc_relations = [
            sncr for sncr in should_not_cause_relations if sncr not in expected_should_not_cause_relations
        ]
        missing_snc_relations = [
            esncr for esncr in expected_should_not_cause_relations if esncr not in should_not_cause_relations
        ]

        self.assertEqual(extra_snc_relations, [])
        self.assertEqual(missing_snc_relations, [])

    def test_all_metamorphic_relations_implied_by_dag_parallel(self):
        dag = CausalDAG(self.dag_dot_path)
        dag.add_edge("Z", "Y")  # Add a direct path from Z to Y so M becomes a mediator
        metamorphic_relations = generate_metamorphic_relations(dag, threads=2)
        should_cause_relations = [mr for mr in metamorphic_relations if isinstance(mr, ShouldCause)]
        should_not_cause_relations = [mr for mr in metamorphic_relations if isinstance(mr, ShouldNotCause)]

        # Check all ShouldCause relations are present and no extra
        expected_should_cause_relations = [
            ShouldCause(BaseTestCase("X1", "Z"), []),
            ShouldCause(BaseTestCase("Z", "M"), []),
            ShouldCause(BaseTestCase("M", "Y"), ["Z"]),
            ShouldCause(BaseTestCase("Z", "Y"), ["M"]),
            ShouldCause(BaseTestCase("X2", "Z"), []),
            ShouldCause(BaseTestCase("X3", "M"), []),
        ]

        extra_sc_relations = [scr for scr in should_cause_relations if scr not in expected_should_cause_relations]
        missing_sc_relations = [escr for escr in expected_should_cause_relations if escr not in should_cause_relations]

        self.assertEqual(extra_sc_relations, [])
        self.assertEqual(missing_sc_relations, [])

        # Check all ShouldNotCause relations are present and no extra
        expected_should_not_cause_relations = [
            ShouldNotCause(BaseTestCase("X1", "X2"), []),
            ShouldNotCause(BaseTestCase("X1", "X3"), []),
            ShouldNotCause(BaseTestCase("X1", "M"), ["Z"]),
            ShouldNotCause(BaseTestCase("X1", "Y"), ["Z"]),
            ShouldNotCause(BaseTestCase("X2", "X3"), []),
            ShouldNotCause(BaseTestCase("X2", "M"), ["Z"]),
            ShouldNotCause(BaseTestCase("X2", "Y"), ["Z"]),
            ShouldNotCause(BaseTestCase("X3", "Y"), ["M", "Z"]),
            ShouldNotCause(BaseTestCase("Z", "X3"), []),
        ]

        extra_snc_relations = [
            sncr for sncr in should_not_cause_relations if sncr not in expected_should_not_cause_relations
        ]
        missing_snc_relations = [
            esncr for esncr in expected_should_not_cause_relations if esncr not in should_not_cause_relations
        ]

        self.assertEqual(extra_snc_relations, [])
        self.assertEqual(missing_snc_relations, [])

    def test_all_metamorphic_relations_implied_by_dag_ignore_cycles(self):
        dag = CausalDAG(self.dcg_dot_path, ignore_cycles=True)
        metamorphic_relations = generate_metamorphic_relations(dag, threads=2, nodes_to_ignore=set(dag.cycle_nodes()))
        should_cause_relations = [mr for mr in metamorphic_relations if isinstance(mr, ShouldCause)]
        should_not_cause_relations = [mr for mr in metamorphic_relations if isinstance(mr, ShouldNotCause)]

        # Check all ShouldCause relations are present and no extra

        self.assertEqual(
            should_cause_relations,
            [
                ShouldCause(BaseTestCase("a", "b"), []),
            ],
        )
        self.assertEqual(
            should_not_cause_relations,
            [],
        )

    def test_generate_metamorphic_relation_(self):
        dag = CausalDAG(self.dag_dot_path)
        [metamorphic_relation] = generate_metamorphic_relation(("X1", "Z"), dag)
        self.assertEqual(
            metamorphic_relation,
            ShouldCause(BaseTestCase("X1", "Z"), []),
        )

    def test_equivalent_metamorphic_relations(self):
        dag = CausalDAG(self.dag_dot_path)
        sc_mr_a = ShouldCause(BaseTestCase("X", "Y"), ["A", "B", "C"])
        sc_mr_b = ShouldCause(BaseTestCase("X", "Y"), ["A", "B", "C"])
        self.assertEqual(sc_mr_a == sc_mr_b, True)

    def test_equivalent_metamorphic_relations_empty_adjustment_set(self):
        dag = CausalDAG(self.dag_dot_path)
        sc_mr_a = ShouldCause(BaseTestCase("X", "Y"), [])
        sc_mr_b = ShouldCause(BaseTestCase("X", "Y"), [])
        self.assertEqual(sc_mr_a == sc_mr_b, True)

    def test_equivalent_metamorphic_relations_different_order_adjustment_set(self):
        dag = CausalDAG(self.dag_dot_path)
        sc_mr_a = ShouldCause(BaseTestCase("X", "Y"), ["A", "B", "C"])
        sc_mr_b = ShouldCause(BaseTestCase("X", "Y"), ["C", "A", "B"])
        self.assertEqual(sc_mr_a == sc_mr_b, True)

    def test_different_metamorphic_relations_empty_adjustment_set_different_outcome(self):
        dag = CausalDAG(self.dag_dot_path)
        sc_mr_a = ShouldCause(BaseTestCase("X", "Z"), [])
        sc_mr_b = ShouldCause(BaseTestCase("X", "Y"), [])
        self.assertEqual(sc_mr_a == sc_mr_b, False)

    def test_different_metamorphic_relations_empty_adjustment_set_different_treatment(self):
        dag = CausalDAG(self.dag_dot_path)
        sc_mr_a = ShouldCause(BaseTestCase("X", "Y"), [])
        sc_mr_b = ShouldCause(BaseTestCase("Z", "Y"), [])
        self.assertEqual(sc_mr_a == sc_mr_b, False)

    def test_different_metamorphic_relations_empty_adjustment_set_adjustment_set(self):
        dag = CausalDAG(self.dag_dot_path)
        sc_mr_a = ShouldCause(BaseTestCase("X", "Y"), ["A"])
        sc_mr_b = ShouldCause(BaseTestCase("X", "Y"), [])
        self.assertEqual(sc_mr_a == sc_mr_b, False)

    def test_different_metamorphic_relations_different_type(self):
        dag = CausalDAG(self.dag_dot_path)
        sc_mr_a = ShouldCause(BaseTestCase("X", "Y"), [])
        sc_mr_b = ShouldNotCause(BaseTestCase("X", "Y"), [])
        self.assertEqual(sc_mr_a == sc_mr_b, False)
