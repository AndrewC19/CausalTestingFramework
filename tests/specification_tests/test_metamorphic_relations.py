import unittest
import os

import pandas as pd
from itertools import combinations

from tests.test_helpers import create_temp_dir_if_non_existent
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.causal_specification import Scenario
from causal_testing.specification.metamorphic_relation import (ShouldCause, ShouldNotCause,
                                                               generate_metamorphic_relations)
from causal_testing.data_collection.data_collector import ExperimentalDataCollector
from causal_testing.specification.variable import Input, Output


def program_under_test(X1, X2, X3, Z=None, M=None, Y=None):
    if Z is None:
        Z = 2*X1 + -3*X2 + 10
    if M is None:
        M = 3*Z + X3
    if Y is None:
        Y = M/2
    return {'X1': X1, 'X2': X2, 'X3': X3, 'Z': Z, 'M': M, 'Y': Y}


def buggy_program_under_test(X1, X2, X3, Z=None, M=None, Y=None):
    if Z is None:
        Z = 2  # No effect of X1 or X2 on Z
    if M is None:
        M = 3*Z + X3
    if Y is None:
        Y = M/2
    return {'X1': X1, 'X2': X2, 'X3': X3, 'Z': Z, 'M': M, 'Y': Y}


class ProgramUnderTestEDC(ExperimentalDataCollector):

    def run_system_with_input_configuration(self, input_configuration: dict) -> pd.DataFrame:
        results_dict = program_under_test(**input_configuration)
        results_df = pd.DataFrame(results_dict, index=[0])
        return results_df


class BuggyProgramUnderTestEDC(ExperimentalDataCollector):

    def run_system_with_input_configuration(self, input_configuration: dict) -> pd.DataFrame:
        results_dict = buggy_program_under_test(**input_configuration)
        results_df = pd.DataFrame(results_dict, index=[0])
        return results_df


class TestMetamorphicRelation(unittest.TestCase):

    def setUp(self) -> None:
        temp_dir_path = create_temp_dir_if_non_existent()
        self.dag_dot_path = os.path.join(temp_dir_path, "dag.dot")
        dag_dot = """digraph DAG { rankdir=LR; X1 -> Z; Z -> M; M -> Y; X2 -> Z; X3 -> M;}"""
        with open(self.dag_dot_path, "w") as f:
            f.write(dag_dot)

        X1 = Input('X1', float)
        X2 = Input('X2', float)
        X3 = Input('X3', float)
        Z = Output('Z', float)
        M = Output('M', float)
        Y = Output('Y', float)
        self.scenario = Scenario(variables={X1, X2, X3, Z, M, Y})
        self.default_control_input_config = {'X1': 1, 'X2': 2, 'X3': 3}
        self.default_treatment_input_config = {'X1': 2, 'X2': 3, 'X3': 3}
        self.data_collector = ProgramUnderTestEDC(self.scenario,
                                                  self.default_control_input_config,
                                                  self.default_treatment_input_config)

    def test_should_cause_metamorphic_relations_correct_spec(self):
        """Test if the ShouldCause MR passes all metamorphic tests where the DAG perfectly represents the program."""
        causal_dag = CausalDAG(self.dag_dot_path)
        for edge in causal_dag.graph.edges:
            (u, v) = edge
            adj_set = list(causal_dag.direct_effect_adjustment_sets([u], [v])[0])
            should_cause_MR = ShouldCause(u, v, adj_set, causal_dag)
            should_cause_MR.generate_follow_up(10, -10.0, 10.0, 1)
            test_results = should_cause_MR.execute_tests(self.data_collector)
            should_cause_MR.test_oracle(test_results)

    def test_should_not_cause_metamorphic_relations_correct_spec(self):
        """Test if the ShouldNotCause MR passes all metamorphic tests where the DAG perfectly represents the program."""
        causal_dag = CausalDAG(self.dag_dot_path)
        for node_pair in combinations(causal_dag.graph.nodes, 2):
            (u, v) = node_pair
            # Get all pairs of nodes which don't form an edge
            if ((u, v) not in causal_dag.graph.edges) and ((v, u) not in causal_dag.graph.edges):
                # Check both directions if there is no causality
                # This can be done more efficiently by ignoring impossible directions (output --> input)
                adj_set_u_to_v = list(causal_dag.direct_effect_adjustment_sets([u], [v])[0])
                u_should_not_cause_v_MR = ShouldNotCause(u, v, adj_set_u_to_v, causal_dag)
                adj_set_v_to_u = list(causal_dag.direct_effect_adjustment_sets([v], [u])[0])
                v_should_not_cause_u_MR = ShouldNotCause(v, u, adj_set_v_to_u, causal_dag)
                u_should_not_cause_v_MR.generate_follow_up(10, -100, 100)
                v_should_not_cause_u_MR.generate_follow_up(10, -100, 100)
                u_should_not_cause_v_test_results = u_should_not_cause_v_MR.execute_tests(self.data_collector)
                v_should_not_cause_u_test_results = v_should_not_cause_u_MR.execute_tests(self.data_collector)
                u_should_not_cause_v_MR.test_oracle(u_should_not_cause_v_test_results)
                v_should_not_cause_u_MR.test_oracle(v_should_not_cause_u_test_results)

    def test_should_cause_metamorphic_relation_missing_relationship(self):
        """Test whether the ShouldCause MR catches missing relationships in the DAG."""
        causal_dag = CausalDAG(self.dag_dot_path)

        # Replace the data collector with one that runs a buggy program in which X1 and X2 do not affect Z
        self.data_collector = BuggyProgramUnderTestEDC(self.scenario,
                                                       self.default_control_input_config,
                                                       self.default_treatment_input_config)
        X1_should_cause_Z_MR = ShouldCause('X1', 'Z', None, causal_dag)
        X2_should_cause_Z_MR = ShouldCause('X2', 'Z', None, causal_dag)
        X1_should_cause_Z_MR.generate_follow_up(10, -100, 100, 1)
        X2_should_cause_Z_MR.generate_follow_up(10, -100, 100, 1)
        X1_should_cause_Z_test_results = X1_should_cause_Z_MR.execute_tests(self.data_collector)
        X2_should_cause_Z_test_results = X2_should_cause_Z_MR.execute_tests(self.data_collector)
        self.assertRaises(AssertionError, X1_should_cause_Z_MR.test_oracle, X1_should_cause_Z_test_results)
        self.assertRaises(AssertionError, X2_should_cause_Z_MR.test_oracle, X2_should_cause_Z_test_results)

    def test_all_metamorphic_relations_implied_by_dag(self):
        dag = CausalDAG(self.dag_dot_path)
        metamorphic_relations = generate_metamorphic_relations(dag)
        should_cause_relations = [mr for mr in metamorphic_relations if isinstance(mr, ShouldCause)]
        should_not_cause_relations = [mr for mr in metamorphic_relations if isinstance(mr, ShouldNotCause)]

        # Check all ShouldCause relations are present and no extra
        expected_should_cause_relations = [ShouldCause('X1', 'Z', [], dag),
                                           ShouldCause('Z', 'M', [], dag),
                                           ShouldCause('M', 'Y', [], dag),
                                           ShouldCause('X2', 'Z', [], dag),
                                           ShouldCause('X3', 'M', [], dag)]

        extra_sc_relations = [scr for scr in should_cause_relations if scr not in expected_should_cause_relations]
        missing_sc_relations = [escr for escr in expected_should_cause_relations if escr not in should_cause_relations]

        self.assertEqual(extra_sc_relations, [])
        self.assertEqual(missing_sc_relations, [])

        # Check all ShouldNotCause relations are present and no extra
        expected_should_not_cause_relations = [ShouldNotCause('X1', 'X2', [], dag),
                                               ShouldNotCause('X1', 'X3', [], dag),
                                               ShouldNotCause('X1', 'M', ['Z'], dag),
                                               ShouldNotCause('X1', 'Y', ['M'], dag),
                                               ShouldNotCause('X2', 'X3', [], dag),
                                               ShouldNotCause('X2', 'M', ['Z'], dag),
                                               ShouldNotCause('X2', 'Y', ['M'], dag),
                                               ShouldNotCause('X3', 'Y', ['M'], dag),
                                               ShouldNotCause('Z', 'Y', ['M'], dag),
                                               ShouldNotCause('Z', 'X3', [], dag)]

        extra_snc_relations = [sncr for sncr in should_not_cause_relations if sncr
                               not in expected_should_not_cause_relations]
        missing_snc_relations = [esncr for esncr in expected_should_not_cause_relations if esncr
                                 not in should_not_cause_relations]

        self.assertEqual(extra_snc_relations, [])
        self.assertEqual(missing_snc_relations, [])
