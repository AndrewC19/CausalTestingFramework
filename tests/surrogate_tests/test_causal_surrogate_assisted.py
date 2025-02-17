import unittest
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.variable import Input, Output
from causal_testing.surrogate.causal_surrogate_assisted import (
    SimulationResult,
    CausalSurrogateAssistedTestCase,
    Simulator,
)
from causal_testing.surrogate.surrogate_search_algorithms import GeneticSearchAlgorithm
from causal_testing.estimation.cubic_spline_estimator import CubicSplineRegressionEstimator

import os
import shutil, tempfile
import pandas as pd
import numpy as np


class TestSimulationResult(unittest.TestCase):

    def setUp(self):

        self.data = {"key": "value"}

    def test_inputs(self):

        fault_values = [True, False]

        relationship_values = ["positive", "negative", None]

        for fault in fault_values:

            for relationship in relationship_values:
                with self.subTest(fault=fault, relationship=relationship):
                    result = SimulationResult(data=self.data, fault=fault, relationship=relationship)

                    self.assertIsInstance(result.data, dict)

                    self.assertEqual(result.fault, fault)

                    self.assertEqual(result.relationship, relationship)


class TestCausalSurrogate(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.class_df = load_class_df()

    def setUp(self):
        self.temp_dir_path = tempfile.mkdtemp()
        self.dag_dot_path = os.path.join(self.temp_dir_path, "dag.dot")
        dag_dot = """digraph DAG { rankdir=LR; Z -> X; X -> M [included=1, expected=positive]; M -> Y [included=1, expected=negative]; Z -> M; }"""
        with open(self.dag_dot_path, "w") as f:
            f.write(dag_dot)

    def test_surrogate_model_generation(self):
        c_s_a_test_case = CausalSurrogateAssistedTestCase(None, None, None)

        df = self.class_df.copy()

        causal_dag = CausalDAG(self.dag_dot_path)
        z = Input("Z", int)
        x = Input("X", float)
        m = Input("M", int)
        y = Output("Y", float)
        scenario = Scenario(variables={z, x, m, y})
        specification = CausalSpecification(scenario, causal_dag)

        surrogate_models = c_s_a_test_case.generate_surrogates(specification, df)
        self.assertEqual(len(surrogate_models), 2)

        for surrogate in surrogate_models:
            self.assertIsInstance(surrogate, CubicSplineRegressionEstimator)
            self.assertNotEqual(surrogate.treatment, "Z")
            self.assertNotEqual(surrogate.outcome, "Z")

    def test_causal_surrogate_assisted_execution(self):
        df = self.class_df.copy()

        causal_dag = CausalDAG(self.dag_dot_path)
        z = Input("Z", int)
        x = Input("X", float)
        m = Input("M", int)
        y = Output("Y", float)
        scenario = Scenario(variables={z, x, m, y}, constraints={z <= 0, z >= 3, x <= 0, x >= 3, m <= 0, m >= 3})
        specification = CausalSpecification(scenario, causal_dag)

        search_algorithm = GeneticSearchAlgorithm(
            config={
                "parent_selection_type": "tournament",
                "K_tournament": 4,
                "mutation_type": "random",
                "mutation_percent_genes": 50,
                "mutation_by_replacement": True,
            }
        )
        simulator = TestSimulator()

        c_s_a_test_case = CausalSurrogateAssistedTestCase(specification, search_algorithm, simulator)

        result, iterations, result_data = c_s_a_test_case.execute(df)

        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(iterations, 1)
        self.assertEqual(len(result_data), 17)

    def test_causal_surrogate_assisted_execution_failure(self):
        df = self.class_df.copy()

        causal_dag = CausalDAG(self.dag_dot_path)
        z = Input("Z", int)
        x = Input("X", float)
        m = Input("M", int)
        y = Output("Y", float)
        scenario = Scenario(variables={z, x, m, y}, constraints={z <= 0, z >= 3, x <= 0, x >= 3, m <= 0, m >= 3})
        specification = CausalSpecification(scenario, causal_dag)

        search_algorithm = GeneticSearchAlgorithm(
            config={
                "parent_selection_type": "tournament",
                "K_tournament": 4,
                "mutation_type": "random",
                "mutation_percent_genes": 50,
                "mutation_by_replacement": True,
            }
        )
        simulator = TestSimulatorFailing()

        c_s_a_test_case = CausalSurrogateAssistedTestCase(specification, search_algorithm, simulator)

        result, iterations, result_data = c_s_a_test_case.execute(df, 1)

        self.assertIsInstance(result, str)
        self.assertEqual(iterations, 1)
        self.assertEqual(len(result_data), 17)

    def test_causal_surrogate_assisted_execution_custom_aggregator(self):
        df = self.class_df.copy()

        causal_dag = CausalDAG(self.dag_dot_path)
        z = Input("Z", int)
        x = Input("X", float)
        m = Input("M", int)
        y = Output("Y", float)
        scenario = Scenario(variables={z, x, m, y}, constraints={z <= 0, z >= 3, x <= 0, x >= 3, m <= 0, m >= 3})
        specification = CausalSpecification(scenario, causal_dag)

        search_algorithm = GeneticSearchAlgorithm(
            config={
                "parent_selection_type": "tournament",
                "K_tournament": 4,
                "mutation_type": "random",
                "mutation_percent_genes": 50,
                "mutation_by_replacement": True,
            }
        )
        simulator = TestSimulator()

        c_s_a_test_case = CausalSurrogateAssistedTestCase(specification, search_algorithm, simulator)

        result, iterations, result_data = c_s_a_test_case.execute(df, custom_data_aggregator=data_double_aggregator)

        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(iterations, 1)
        self.assertEqual(len(result_data), 18)

    def test_causal_surrogate_assisted_execution_incorrect_search_config(self):
        df = self.class_df.copy()

        causal_dag = CausalDAG(self.dag_dot_path)
        z = Input("Z", int)
        x = Input("X", float)
        m = Input("M", int)
        y = Output("Y", float)
        scenario = Scenario(variables={z, x, m, y}, constraints={z <= 0, z >= 3, x <= 0, x >= 3, m <= 0, m >= 3})
        specification = CausalSpecification(scenario, causal_dag)

        search_algorithm = GeneticSearchAlgorithm(
            config={
                "parent_selection_type": "tournament",
                "K_tournament": 4,
                "mutation_type": "random",
                "mutation_percent_genes": 50,
                "mutation_by_replacement": True,
                "gene_space": "Something",
            }
        )
        simulator = TestSimulator()

        c_s_a_test_case = CausalSurrogateAssistedTestCase(specification, search_algorithm, simulator)

        self.assertRaises(
            ValueError,
            c_s_a_test_case.execute,
            df=df,
            custom_data_aggregator=data_double_aggregator,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir_path)


def load_class_df():
    """Get the testing data and put into a dataframe."""

    class_df = pd.DataFrame(
        {"Z": np.arange(16), "X": np.arange(16), "M": np.arange(16, 32), "Y": np.arange(32, 16, -1)}
    )
    return class_df


class TestSimulator(Simulator):

    def run_with_config(self, configuration: dict) -> SimulationResult:
        return SimulationResult({"Z": 1, "X": 1, "M": 1, "Y": 1}, True, None)

    def startup(self):
        pass

    def shutdown(self):
        pass


class TestSimulatorFailing(Simulator):

    def run_with_config(self, configuration: dict) -> SimulationResult:
        return SimulationResult({"Z": 1, "X": 1, "M": 1, "Y": 1}, False, None)

    def startup(self):
        pass

    def shutdown(self):
        pass


def data_double_aggregator(data, new_data):
    """Previously used data.append(new_data), however, pandas version >2 requires pd.concat() since append is now a private method.
    Converting new_data to a pd.DataFrame is required to use pd.concat()."""
    new_data = pd.DataFrame([new_data])
    return pd.concat([data, new_data, new_data], ignore_index=True)
