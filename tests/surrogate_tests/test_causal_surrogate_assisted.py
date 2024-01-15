import unittest
from causal_testing.data_collection.data_collector import ObservationalDataCollector
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.variable import Input
from causal_testing.surrogate.causal_surrogate_assisted import SearchAlgorithm, SimulationResult, SearchFitnessFunction, CausalSurrogateAssistedTestCase, Simulator
from causal_testing.surrogate.surrogate_search_algorithms import GeneticSearchAlgorithm
from causal_testing.testing.estimators import CubicSplineRegressionEstimator
from tests.test_helpers import create_temp_dir_if_non_existent, remove_temp_dir_if_existent
import os
import pandas as pd
import numpy as np

class TestSimulationResult(unittest.TestCase):

    def setUp(self):

        self.data = {'key': 'value'}

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

class TestSearchFitnessFunction(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.class_df = load_class_df()

    def setUp(self):
        temp_dir_path = create_temp_dir_if_non_existent()
        self.dag_dot_path = os.path.join(temp_dir_path, "dag.dot")
        dag_dot = """digraph DAG { rankdir=LR; Z -> X; X -> M [included=1, expected=positive]; M -> Y [included=1, expected=negative]; Z -> M; }"""
        with open(self.dag_dot_path, "w") as f:
            f.write(dag_dot)

    def test_init_valid_values(self):

        test_function = lambda x: x **2

        surrogate_model = CubicSplineRegressionEstimator("", 0, 0, set(), "", 4)

        search_function = SearchFitnessFunction(fitness_function=test_function, surrogate_model=surrogate_model)

        self.assertTrue(callable(search_function.fitness_function))
        self.assertIsInstance(search_function.surrogate_model, CubicSplineRegressionEstimator)

    def test_surrogate_model_generation(self):
        c_s_a_test_case = CausalSurrogateAssistedTestCase(None, None, None)

        df = self.class_df.copy()

        causal_dag = CausalDAG(self.dag_dot_path)
        z = Input("Z", int)
        x = Input("X", int)
        m = Input("M", int)
        y = Input("Y", int)
        scenario = Scenario(variables={z, x, m, y})
        specification = CausalSpecification(scenario, causal_dag)

        surrogate_models = c_s_a_test_case.generate_surrogates(specification, ObservationalDataCollector(scenario, df))
        self.assertEqual(len(surrogate_models), 2)

        for surrogate in surrogate_models:
            self.assertIsInstance(surrogate, CubicSplineRegressionEstimator)
            self.assertNotEqual(surrogate.treatment, "Z")
            self.assertNotEqual(surrogate.outcome, "Z")

    def test_causal_surrogate_assisted_execution(self):
        df = self.class_df.copy()

        causal_dag = CausalDAG(self.dag_dot_path)
        z = Input("Z", int)
        x = Input("X", int)
        m = Input("M", int)
        y = Input("Y", int)
        scenario = Scenario(variables={z, x, m, y}, constraints={
            z <= 0, z >= 3,
            x <= 0, x >= 3,
            m <= 0, m >= 3
        })
        specification = CausalSpecification(scenario, causal_dag)

        search_algorithm = GeneticSearchAlgorithm(config= {
                "parent_selection_type": "tournament",
                "K_tournament": 4,
                "mutation_type": "random",
                "mutation_percent_genes": 50,
                "mutation_by_replacement": True,
            })
        simulator = TestSimulator()

        c_s_a_test_case = CausalSurrogateAssistedTestCase(specification, search_algorithm, simulator)

        result, iterations, result_data = c_s_a_test_case.execute(ObservationalDataCollector(scenario, df))

        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(iterations, 1)
        self.assertEqual(len(result_data), 17)

    def test_causal_surrogate_assisted_execution_failure(self):
        df = self.class_df.copy()

        causal_dag = CausalDAG(self.dag_dot_path)
        z = Input("Z", int)
        x = Input("X", int)
        m = Input("M", int)
        y = Input("Y", int)
        scenario = Scenario(variables={z, x, m, y}, constraints={
            z <= 0, z >= 3,
            x <= 0, x >= 3,
            m <= 0, m >= 3
        })
        specification = CausalSpecification(scenario, causal_dag)

        search_algorithm = GeneticSearchAlgorithm(config= {
                "parent_selection_type": "tournament",
                "K_tournament": 4,
                "mutation_type": "random",
                "mutation_percent_genes": 50,
                "mutation_by_replacement": True,
            })
        simulator = TestSimulatorFailing()

        c_s_a_test_case = CausalSurrogateAssistedTestCase(specification, search_algorithm, simulator)

        result, iterations, result_data = c_s_a_test_case.execute(ObservationalDataCollector(scenario, df), 1)

        self.assertIsInstance(result, str)
        self.assertEqual(iterations, 1)
        self.assertEqual(len(result_data), 17)

    def test_causal_surrogate_assisted_execution_custom_aggregator(self):
        df = self.class_df.copy()

        causal_dag = CausalDAG(self.dag_dot_path)
        z = Input("Z", int)
        x = Input("X", int)
        m = Input("M", int)
        y = Input("Y", int)
        scenario = Scenario(variables={z, x, m, y}, constraints={
            z <= 0, z >= 3,
            x <= 0, x >= 3,
            m <= 0, m >= 3
        })
        specification = CausalSpecification(scenario, causal_dag)

        search_algorithm = GeneticSearchAlgorithm(config= {
                "parent_selection_type": "tournament",
                "K_tournament": 4,
                "mutation_type": "random",
                "mutation_percent_genes": 50,
                "mutation_by_replacement": True,
            })
        simulator = TestSimulator()

        c_s_a_test_case = CausalSurrogateAssistedTestCase(specification, search_algorithm, simulator)

        result, iterations, result_data = c_s_a_test_case.execute(ObservationalDataCollector(scenario, df), 
                                                                    custom_data_aggregator=data_double_aggregator)

        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(iterations, 1)
        self.assertEqual(len(result_data), 18)

    def test_causal_surrogate_assisted_execution_incorrect_search_config(self):
        df = self.class_df.copy()

        causal_dag = CausalDAG(self.dag_dot_path)
        z = Input("Z", int)
        x = Input("X", int)
        m = Input("M", int)
        y = Input("Y", int)
        scenario = Scenario(variables={z, x, m, y}, constraints={
            z <= 0, z >= 3,
            x <= 0, x >= 3,
            m <= 0, m >= 3
        })
        specification = CausalSpecification(scenario, causal_dag)

        search_algorithm = GeneticSearchAlgorithm(config= {
                "parent_selection_type": "tournament",
                "K_tournament": 4,
                "mutation_type": "random",
                "mutation_percent_genes": 50,
                "mutation_by_replacement": True,
                "gene_space": "Something"
            })
        simulator = TestSimulator()

        c_s_a_test_case = CausalSurrogateAssistedTestCase(specification, search_algorithm, simulator)

        self.assertRaises(ValueError, c_s_a_test_case.execute, 
                          data_collector=ObservationalDataCollector(scenario, df),
                          custom_data_aggregator=data_double_aggregator)

    def tearDown(self) -> None:
        remove_temp_dir_if_existent()

def load_class_df():
    """Get the testing data and put into a dataframe."""

    class_df = pd.DataFrame({"Z": np.arange(16), "X": np.arange(16), "M": np.arange(16, 32), "Y": np.arange(32,16,-1)})
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
    return data.append(new_data, ignore_index=True).append(new_data, ignore_index=True)