import unittest
from unittest.mock import patch
import argparse
from pathlib import Path
from causal_testing.testing.causal_test_outcome import Positive, Negative, NoEffect, SomeEffect
from causal_testing.estimation.linear_regression_estimator import LinearRegressionEstimator
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.variable import Input, Output
from dafni.src.main_dafni import get_args, parse_variables

# Base directory (relative to the current test file location)
BASE_DIR = Path(__file__).resolve().parent.parent  # Points to ./dafni
DATA_DIR = BASE_DIR / "data"  # Points to ./dafni/data

class TestGetArgs(unittest.TestCase):
    """Test the argparse functionality of the DAFNI entrypoint."""

    @patch("argparse.ArgumentParser.parse_args")
    def test_get_args_with_all_arguments(self, mock_parse_args):
        """Ensure all arguments work as expected"""

        mock_parse_args.return_value = argparse.Namespace(
            data_path=[str(DATA_DIR / "inputs/runtime_data.csv")],
            tests_path=str(DATA_DIR / "inputs/causal_tests.json"),
            ignore_cycles=False,
            dag_path=str(DATA_DIR / "inputs/dag.dot"),
            output_path=str(DATA_DIR / "outputs/causal_test_results.json"),
            f=False,
            w=False,
        )

        args = get_args()

        if args.output_path is None:
            self.assertEqual(args.output_path, Path(DATA_DIR / "outputs/causal_test_results.json"))
        else:
            self.assertIsInstance(args.output_path, Path)

        self.assertEqual(args.data_path, [str(DATA_DIR / "inputs/runtime_data.csv")])
        self.assertEqual(args.tests_path, Path(DATA_DIR / "inputs/causal_tests.json"))
        self.assertEqual(args.dag_path, Path(DATA_DIR / "inputs/dag.dot"))
        self.assertIsInstance(args.f, bool)
        self.assertIsInstance(args.w, bool)
        self.assertIsInstance(args.ignore_cycles, bool)


class TestParseVariables(unittest.TestCase):
    """Test the parse variables functionality of the DAFNI entrypoint."""

    @classmethod
    def setUpClass(cls):
        """Set up class method for the Causal Dag."""

        cls.causal_dag = CausalDAG(str(DATA_DIR / "inputs/dag.dot"))

    def test_causal_dag_instance(self):
        """Test if the input is an instance of CausalDAG."""
        self.assertIsInstance(self.causal_dag, CausalDAG)

    def test_dag_attributes(self):
        """Test if all attributes in the DAG are strings."""
        for _, attributes in self.causal_dag.graph.nodes(data=True):
            for key, value in attributes.items():
                self.assertIsInstance(value, str)
                self.assertIsInstance(key, str)

    def test_dag_inputs_outputs_constraints(self):
        """Test the DAG inputs, outputs and constraints."""

        try:
            inputs, outputs, constraints = parse_variables(self.causal_dag)

            for _input, _output, _constraint in zip(inputs, outputs, constraints):
                # Assert each input is an Input, Output or Set object
                self.assertIsInstance(_input, Input)
                self.assertIsInstance(_output, Output)
                self.assertIsInstance(constraints, set)

        except ValueError as e:
            self.fail(f"Unable to parse variables: {e}")


class TestMain(unittest.TestCase):
    """Test the main entrypoint."""

    def test_estimator(self):
        """Test estimators currently supported on DAFNI."""

        estimators = {"LinearRegressionEstimator": LinearRegressionEstimator}

        self.assertIsInstance(estimators, dict, "Estimators must be a dictionary.")

        self.assertIn(
            "LinearRegressionEstimator", estimators,
            "The key 'LinearRegressionEstimator' must exist in the dictionary."
        )

        self.assertEqual(
            estimators["LinearRegressionEstimator"],
            LinearRegressionEstimator,
            "The estimator should be LinearRegression.",
        )

        self.assertNotIn("LogisticRegressionEstimator", estimators,
                         "The dictionary should not contain other estimators.")

    def test_expected_outcome_effects_keys(self):
        """Check that the dictionary contains these keys only for DAFNI."""

        expected_outcome_effects = {
            "Positive": Positive(),
            "Negative": Negative(),
            "NoEffect": NoEffect(),
            "SomeEffect": SomeEffect(),
        }

        expected_keys = ["Positive", "Negative", "NoEffect", "SomeEffect"]

        actual_keys = list(expected_outcome_effects.keys())

        self.assertEqual(sorted(actual_keys), sorted(expected_keys),
                         "The dictionary keys do not match the expected keys.")

    def test_expected_outcome_effects_values(self):
        """Check that the dictionary contains the correct instances for DAFNI."""

        expected_outcome_effects = {
            "Positive": Positive(),
            "Negative": Negative(),
            "NoEffect": NoEffect(),
            "SomeEffect": SomeEffect(),
        }

        self.assertIsInstance(
            expected_outcome_effects["Positive"],
            Positive,
            "The value for 'Positive' should be an instance of Positive.",
        )
        self.assertIsInstance(
            expected_outcome_effects["Negative"],
            Negative,
            "The value for 'Negative' should be an instance of Negative.",
        )
        self.assertIsInstance(
            expected_outcome_effects["NoEffect"],
            NoEffect,
            "The value for 'NoEffect' should be an instance of NoEffect.",
        )
        self.assertIsInstance(
            expected_outcome_effects["SomeEffect"],
            SomeEffect,
            "The value for 'SomeEffect' should be an instance of SomeEffect.",
        )
