import unittest
from pathlib import Path
from statistics import StatisticsError
import scipy
import os

from causal_testing.estimation.linear_regression_estimator import LinearRegressionEstimator
from causal_testing.estimation.abstract_estimator import Estimator
from causal_testing.testing.causal_test_outcome import NoEffect, Positive
from causal_testing.json_front.json_class import JsonUtility, CausalVariables
from causal_testing.specification.variable import Input, Output, Meta
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.causal_specification import CausalSpecification


class TestJsonClass(unittest.TestCase):
    """Test the JSON frontend for the Causal Testing Framework (CTF)

    The JSON frontend is an alternative interface for the CTF where tests are specified in JSON format and ingested
    with the frontend. Tests involve testing that this correctly interfaces with the framework with some dummy data
    """

    def setUp(self) -> None:
        json_file_name = "tests.json"
        dag_file_name = "dag.dot"
        data_file_name = "data_with_meta.csv"
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
        self.meta_dict_list = [{"name": "test_meta", "datatype": float, "populate": populate_example}]
        variables = CausalVariables(
            inputs=self.input_dict_list, outputs=self.output_dict_list, metas=self.meta_dict_list
        )
        self.scenario = Scenario(variables=variables, constraints=None)
        self.json_class.set_paths(self.json_path, self.dag_path, self.data_path)
        self.json_class.setup(self.scenario)

    def test_setting_no_path(self):
        json_class = JsonUtility("temp_out.txt", True)
        json_class.set_paths(self.json_path, self.dag_path, None)
        self.assertEqual(json_class.input_paths.data_paths, [])  # Needs to be list of Paths

    def test_setting_paths(self):
        self.assertEqual(self.json_class.input_paths.json_path, Path(self.json_path))
        self.assertEqual(self.json_class.input_paths.dag_path, Path(self.dag_path))
        self.assertEqual(self.json_class.input_paths.data_paths, [Path(self.data_path[0])])  # Needs to be list of Paths

    def test_set_inputs(self):
        ctf_input = [Input("test_input", float, self.example_distribution)]
        self.assertEqual(ctf_input[0].name, self.json_class.scenario.variables["test_input"].name)
        self.assertEqual(ctf_input[0].datatype, self.json_class.scenario.variables["test_input"].datatype)
        self.assertEqual(ctf_input[0].distribution, self.json_class.scenario.variables["test_input"].distribution)

    def test_set_outputs(self):
        ctf_output = [Output("test_output", float)]
        self.assertEqual(ctf_output[0].name, self.json_class.scenario.variables["test_output"].name)
        self.assertEqual(ctf_output[0].datatype, self.json_class.scenario.variables["test_output"].datatype)

    def test_set_metas(self):
        ctf_meta = [Meta("test_meta", float, populate_example)]
        self.assertEqual(ctf_meta[0].name, self.json_class.scenario.variables["test_meta"].name)
        self.assertEqual(ctf_meta[0].datatype, self.json_class.scenario.variables["test_meta"].datatype)

    def test_argparse(self):
        args = self.json_class.get_args(["--data_path=data.csv", "--dag_path=dag.dot", "--json_path=tests.json"])
        self.assertEqual(args.data_path, ["data.csv"])
        self.assertEqual(args.dag_path, "dag.dot")
        self.assertEqual(args.json_path, "tests.json")

    def test_setup_scenario(self):
        self.assertIsInstance(self.json_class.scenario, Scenario)

    def test_setup_causal_specification(self):
        self.assertIsInstance(self.json_class.causal_specification, CausalSpecification)

    def test_f_flag(self):
        example_test = {
            "tests": [
                {
                    "name": "test1",
                    "mutations": {"test_input": "Increase"},
                    "estimator": "LinearRegressionEstimator",
                    "estimate_type": "ate",
                    "effect_modifiers": [],
                    "expected_effect": {"test_output": "NoEffect"},
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
        with self.assertRaises(StatisticsError):
            self.json_class.run_json_tests(effects, estimators, True, mutates)

    def test_generate_coefficient_tests_from_json(self):
        example_test = {
            "tests": [
                {
                    "name": "test1",
                    "mutations": ["test_input"],
                    "estimator": "LinearRegressionEstimator",
                    "estimate_type": "coefficient",
                    "effect_modifiers": [],
                    "expected_effect": {"test_output": "NoEffect"},
                    "skip": False,
                }
            ]
        }
        self.json_class.test_plan = example_test
        effects = {"NoEffect": NoEffect()}
        estimators = {"LinearRegressionEstimator": LinearRegressionEstimator}

        self.json_class.run_json_tests(effects=effects, mutates={}, estimators=estimators, f_flag=False)

        # Test that the final log message prints that failed tests are printed, which is expected behaviour for this scenario
        with open("temp_out.txt", "r") as reader:
            temp_out = reader.readlines()
        self.assertIn("FAILED", temp_out[-1])

    def test_run_json_tests_from_json(self):
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
        self.assertTrue(test_results[0]["failed"])

    def test_generate_tests_from_json_no_dist(self):
        example_test = {
            "tests": [
                {
                    "name": "test1",
                    "mutations": {"test_input_no_dist": "Increase"},
                    "estimator": "LinearRegressionEstimator",
                    "estimate_type": "ate",
                    "effect_modifiers": [],
                    "expected_effect": {"test_output": "NoEffect"},
                    "skip": False,
                }
            ]
        }
        self.json_class.test_plan = example_test
        effects = {"NoEffect": NoEffect()}
        mutates = {
            "Increase": lambda x: self.json_class.scenario.treatment_variables[x]
            > self.json_class.scenario.variables[x]
        }
        estimators = {"LinearRegressionEstimator": LinearRegressionEstimator}

        self.json_class.run_json_tests(effects=effects, mutates=mutates, estimators=estimators, f_flag=False)

        # Test that the final log message prints that failed tests are printed, which is expected behaviour for this scenario
        with open("temp_out.txt", "r") as reader:
            temp_out = reader.readlines()
        self.assertIn("failed", temp_out[-1])

    def test_formula_in_json_test(self):
        example_test = {
            "tests": [
                {
                    "name": "test1",
                    "mutations": {"test_input": "Increase"},
                    "estimator": "LinearRegressionEstimator",
                    "estimate_type": "ate",
                    "effect_modifiers": [],
                    "expected_effect": {"test_output": "Positive"},
                    "skip": False,
                    "formula": "test_output ~ test_input",
                }
            ]
        }
        self.json_class.test_plan = example_test
        effects = {"Positive": Positive()}
        mutates = {
            "Increase": lambda x: self.json_class.scenario.treatment_variables[x].z3
            > self.json_class.scenario.variables[x].z3
        }
        estimators = {"LinearRegressionEstimator": LinearRegressionEstimator}

        self.json_class.run_json_tests(effects=effects, mutates=mutates, estimators=estimators, f_flag=False)
        with open("temp_out.txt", "r") as reader:
            temp_out = reader.readlines()
        self.assertIn("test_output ~ test_input", "".join(temp_out))

    def test_run_concrete_json_testcase(self):
        example_test = {
            "tests": [
                {
                    "name": "test1",
                    "treatment_variable": "test_input",
                    "control_value": 0,
                    "treatment_value": 1,
                    "estimator": "LinearRegressionEstimator",
                    "estimate_type": "ate",
                    "expected_effect": {"test_output": "NoEffect"},
                    "skip": False,
                }
            ]
        }
        self.json_class.test_plan = example_test
        effects = {"NoEffect": NoEffect()}
        estimators = {"LinearRegressionEstimator": LinearRegressionEstimator}

        self.json_class.run_json_tests(effects=effects, estimators=estimators, f_flag=False)
        with open("temp_out.txt", "r") as reader:
            temp_out = reader.readlines()
        self.assertIn("FAILED", temp_out[-1])

    def test_concrete_generate_params(self):
        example_test = {
            "tests": [
                {
                    "name": "test1",
                    "mutations": {"test_input": "Increase"},
                    "estimator": "LinearRegressionEstimator",
                    "estimate_type": "ate",
                    "effect_modifiers": [],
                    "expected_effect": {"test_output": "NoEffect"},
                    "sample_size": 5,
                    "target_ks_score": 0.05,
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

        self.json_class.run_json_tests(effects=effects, estimators=estimators, f_flag=False, mutates=mutates)

        # Test that the final log message prints that failed tests are printed, which is expected behaviour for this
        # scenario
        with open("temp_out.txt", "r") as reader:
            temp_out = reader.readlines()
        self.assertIn("failed", temp_out[-1])

    def test_no_data_provided(self):
        example_test = {
            "tests": [
                {
                    "name": "test1",
                    "mutations": {"test_input": "Increase"},
                    "estimator": "LinearRegressionEstimator",
                    "estimate_type": "ate",
                    "effect_modifiers": [],
                    "expected_effect": {"test_output": "NoEffect"},
                    "skip": False,
                }
            ]
        }
        json_class = JsonUtility("temp_out.txt", True)
        json_class.set_paths(self.json_path, self.dag_path)

        with self.assertRaises(ValueError):
            json_class.setup(self.scenario)

    def test_estimator_formula_type_check(self):
        class ExampleEstimator(Estimator):
            def add_modelling_assumptions(self):
                pass

        example_test = {
            "tests": [
                {
                    "name": "test1",
                    "mutations": {"test_input": "Increase"},
                    "estimator": "ExampleEstimator",
                    "estimate_type": "ate",
                    "effect_modifiers": [],
                    "expected_effect": {"test_output": "Positive"},
                    "skip": False,
                    "formula": "test_output ~ test_input",
                }
            ]
        }
        self.json_class.test_plan = example_test
        effects = {"Positive": Positive()}
        mutates = {
            "Increase": lambda x: self.json_class.scenario.treatment_variables[x].z3
            > self.json_class.scenario.variables[x].z3
        }
        estimators = {"ExampleEstimator": ExampleEstimator}
        with self.assertRaises(TypeError):
            self.json_class.run_json_tests(effects=effects, mutates=mutates, estimators=estimators, f_flag=False)

    def tearDown(self) -> None:
        if os.path.exists("temp_out.txt"):
            os.remove("temp_out.txt")


def populate_example(*args, **kwargs):
    pass
