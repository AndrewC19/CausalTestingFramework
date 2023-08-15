import unittest
from pathlib import Path
from statistics import StatisticsError
import scipy
import os

from causal_testing.testing.estimators import LinearRegressionEstimator
from causal_testing.testing.base_test_case import BaseTestCase
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_test_suite import CausalTestSuite
from causal_testing.testing.causal_test_adequacy import DAGAdequacy
from causal_testing.testing.causal_test_outcome import NoEffect, Positive
from tests.test_helpers import remove_temp_dir_if_existent
from causal_testing.json_front.json_class import JsonUtility, CausalVariables
from causal_testing.specification.variable import Input, Output, Meta
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.causal_specification import CausalSpecification


class TestCausalTestAdequacy(unittest.TestCase):
    """
    Test the causal test adequacy metrics. These provide metrics determine how comprehensive a test set is and,
    ultimately whether we can stop testing.
    """

    def setUp(self) -> None:
        json_file_name = "tests.json"
        dag_file_name = "dag.dot"
        data_file_name = "data_with_categorical.csv"
        test_data_dir_path = Path("tests/resources/data")
        self.json_path = str(test_data_dir_path / json_file_name)
        self.dag_path = str(test_data_dir_path / dag_file_name)
        self.data_path = [str(test_data_dir_path / data_file_name)]
        self.json_class = JsonUtility("temp_out.txt", True)
        self.example_distribution = scipy.stats.uniform(1, 10)
        self.input_dict_list = [
            {"name": "test_input", "datatype": float, "distribution": self.example_distribution},
            {"name": "test_input_no_dist", "datatype": float},
        ]
        self.output_dict_list = [{"name": "test_output", "datatype": float}]
        variables = CausalVariables(inputs=self.input_dict_list, outputs=self.output_dict_list, metas=[])
        self.scenario = Scenario(variables=variables, constraints=None)
        self.json_class.set_paths(self.json_path, self.dag_path, self.data_path)
        self.json_class.setup(self.scenario)

    def test_data_adequacy_numeric(self):
        example_test = {
            "tests": [
                {
                    "name": "test1",
                    "mutations": {"test_input": "Increase"},
                    "estimator": "LinearRegressionEstimator",
                    "estimate_type": "coefficient",
                    "effect_modifiers": [],
                    "expected_effect": {"test_output": "NoEffect"},
                    "coverage": True,
                    "skip": False,
                }
            ]
        }
        self.json_class.test_plan = example_test
        effects = {"NoEffect": NoEffect()}
        mutates = {
            "Increase": lambda x: self.json_class.scenario.treatment_variables[x].z3
            > self.json_class.scenario.variables[x].z3
        }
        estimators = {"LinearRegressionEstimator": LinearRegressionEstimator}

        test_results = self.json_class.run_json_tests(
            effects=effects, estimators=estimators, f_flag=False, mutates=mutates
        )
        self.assertEqual(
            test_results[0]["result"].adequacy.to_dict(),
            {"kurtosis": {"test_input": 0.0}, "bootstrap_size": 100, "passing": 100},
        )

    def test_data_adequacy_cateogorical(self):
        example_test = {
            "tests": [
                {
                    "name": "test1",
                    "mutations": ["test_input_no_dist"],
                    "estimator": "LinearRegressionEstimator",
                    "estimate_type": "coefficient",
                    "effect_modifiers": [],
                    "expected_effect": {"test_output": "NoEffect"},
                    "coverage": True,
                    "skip": False,
                }
            ]
        }
        self.json_class.test_plan = example_test
        effects = {"NoEffect": NoEffect()}
        mutates = {
            "Increase": lambda x: self.json_class.scenario.treatment_variables[x].z3
            > self.json_class.scenario.variables[x].z3
        }
        estimators = {"LinearRegressionEstimator": LinearRegressionEstimator}

        test_results = self.json_class.run_json_tests(
            effects=effects, estimators=estimators, f_flag=False, mutates=mutates
        )
        print("RESULT")
        print(test_results[0]["result"])
        self.assertEqual(
            test_results[0]["result"].adequacy.to_dict(),
            {"kurtosis": {"test_input_no_dist[T.b]": 0.0}, "bootstrap_size": 100, "passing": 100},
        )

    def test_dag_adequacy_dependent(self):
        base_test_case = BaseTestCase(
            treatment_variable="test_input",
            outcome_variable="B",
            effect=None,
        )
        causal_test_case = CausalTestCase(
            base_test_case=base_test_case,
            expected_causal_effect=None,
            estimate_type=None,
        )
        test_suite = CausalTestSuite()
        test_suite.add_test_object(base_test_case, causal_test_case, None, None)
        dag_adequacy = DAGAdequacy(self.json_class.causal_specification.causal_dag, test_suite)
        dag_adequacy.measure_adequacy()
        print(dag_adequacy.to_dict())
        self.assertEqual(
            dag_adequacy.to_dict(),
            {
                "causal_dag": self.json_class.causal_specification.causal_dag,
                "test_suite": test_suite,
                "tested_pairs": {("test_input", "B")},
                "pairs_to_test": {
                    ("B", "C"),
                    ("test_input_no_dist", "test_input"),
                    ("C", "test_output"),
                    ("test_input", "B"),
                    ("test_input_no_dist", "B"),
                    ("test_input", "test_output"),
                    ("test_input", "C"),
                    ("test_input_no_dist", "test_output"),
                    ("B", "test_output"),
                    ("test_input_no_dist", "C"),
                },
                "untested_pairs": {
                    ("B", "C"),
                    ("test_input_no_dist", "test_input"),
                    ("C", "test_output"),
                    ("test_input_no_dist", "B"),
                    ("test_input", "test_output"),
                    ("test_input", "C"),
                    ("test_input_no_dist", "test_output"),
                    ("B", "test_output"),
                    ("test_input_no_dist", "C"),
                },
                "dag_adequacy": 0.1,
            },
        )

    def test_dag_adequacy_independent(self):
        base_test_case = BaseTestCase(
            treatment_variable="test_input",
            outcome_variable="C",
            effect=None,
        )
        causal_test_case = CausalTestCase(
            base_test_case=base_test_case,
            expected_causal_effect=None,
            estimate_type=None,
        )
        test_suite = CausalTestSuite()
        test_suite.add_test_object(base_test_case, causal_test_case, None, None)
        dag_adequacy = DAGAdequacy(self.json_class.causal_specification.causal_dag, test_suite)
        dag_adequacy.measure_adequacy()
        print(dag_adequacy.to_dict())
        self.assertEqual(
            dag_adequacy.to_dict(),
            {
                "causal_dag": self.json_class.causal_specification.causal_dag,
                "test_suite": test_suite,
                "tested_pairs": {("test_input", "C")},
                "pairs_to_test": {
                    ("B", "C"),
                    ("test_input_no_dist", "test_input"),
                    ("C", "test_output"),
                    ("test_input", "B"),
                    ("test_input_no_dist", "B"),
                    ("test_input", "test_output"),
                    ("test_input", "C"),
                    ("test_input_no_dist", "test_output"),
                    ("B", "test_output"),
                    ("test_input_no_dist", "C"),
                },
                "untested_pairs": {
                    ("B", "C"),
                    ("test_input_no_dist", "test_input"),
                    ("C", "test_output"),
                    ("test_input_no_dist", "B"),
                    ("test_input", "test_output"),
                    ("test_input", "B"),
                    ("test_input_no_dist", "test_output"),
                    ("B", "test_output"),
                    ("test_input_no_dist", "C"),
                },
                "dag_adequacy": 0.1,
            },
        )

    def test_dag_adequacy_independent_other_way(self):
        base_test_case = BaseTestCase(
            treatment_variable="C",
            outcome_variable="test_input",
            effect=None,
        )
        causal_test_case = CausalTestCase(
            base_test_case=base_test_case,
            expected_causal_effect=None,
            estimate_type=None,
        )
        test_suite = CausalTestSuite()
        test_suite.add_test_object(base_test_case, causal_test_case, None, None)
        dag_adequacy = DAGAdequacy(self.json_class.causal_specification.causal_dag, test_suite)
        dag_adequacy.measure_adequacy()
        print(dag_adequacy.to_dict())
        self.assertEqual(
            dag_adequacy.to_dict(),
            {
                "causal_dag": self.json_class.causal_specification.causal_dag,
                "test_suite": test_suite,
                "tested_pairs": {("test_input", "C")},
                "pairs_to_test": {
                    ("B", "C"),
                    ("test_input_no_dist", "test_input"),
                    ("C", "test_output"),
                    ("test_input", "B"),
                    ("test_input_no_dist", "B"),
                    ("test_input", "test_output"),
                    ("test_input", "C"),
                    ("test_input_no_dist", "test_output"),
                    ("B", "test_output"),
                    ("test_input_no_dist", "C"),
                },
                "untested_pairs": {
                    ("B", "C"),
                    ("test_input_no_dist", "test_input"),
                    ("C", "test_output"),
                    ("test_input_no_dist", "B"),
                    ("test_input", "test_output"),
                    ("test_input", "B"),
                    ("test_input_no_dist", "test_output"),
                    ("B", "test_output"),
                    ("test_input_no_dist", "C"),
                },
                "dag_adequacy": 0.1,
            },
        )

    def tearDown(self) -> None:
        remove_temp_dir_if_existent()
        if os.path.exists("temp_out.txt"):
            os.remove("temp_out.txt")
