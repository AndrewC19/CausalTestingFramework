import unittest
import os
import tempfile
import numpy as np
import shutil
import pandas as pd
from causal_testing.testing.causal_test_suite import CausalTestSuite
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.base_test_case import BaseTestCase
from causal_testing.specification.variable import Input, Output
from causal_testing.testing.causal_test_outcome import ExactValue
from causal_testing.estimation.linear_regression_estimator import LinearRegressionEstimator
from causal_testing.estimation.logistic_regression_estimator import LogisticRegressionEstimator
from causal_testing.specification.causal_specification import CausalSpecification, Scenario
from causal_testing.specification.causal_dag import CausalDAG


class TestCausalTestSuite(unittest.TestCase):
    """Test the Test Suite object and it's implementation in the test engine using dummy data."""

    def setUp(self) -> None:
        # 1. Create dummy Scenario and BaseTestCase
        A = Input("A", float)
        self.A = A
        C = Output("C", float)
        self.C = C
        D = Output("D", float)
        self.D = D
        self.base_test_case = BaseTestCase(A, C)
        self.scenario = Scenario({A, C, D})

        # 2. Create DAG and dummy data and write to csvs
        self.temp_dir_path = tempfile.mkdtemp()
        dag_dot_path = os.path.join(self.temp_dir_path, "dag.dot")
        dag_dot = """digraph G { A -> C; D -> A; D -> C}"""
        with open(dag_dot_path, "w") as file:
            file.write(dag_dot)

        np.random.seed(1)
        df = pd.DataFrame({"D": list(np.random.normal(60, 10, 1000))})  # D = exogenous
        df["A"] = [1 if d > 50 else 0 for d in df["D"]]
        df["C"] = df["D"] + (4 * (df["A"] + 2))  # C = (4*(A+2)) + D
        self.df = df
        self.causal_dag = CausalDAG(dag_dot_path)

        # 3. Specify data structures required for test suite
        self.expected_causal_effect = ExactValue(4)
        test_list = [
            CausalTestCase(
                self.base_test_case,
                self.expected_causal_effect,
                0,
                1,
            ),
            CausalTestCase(self.base_test_case, self.expected_causal_effect, 0, 2),
        ]
        self.estimators = [LinearRegressionEstimator]

        # 3. Create test_suite and add a test
        self.test_suite = CausalTestSuite()
        self.test_suite.add_test_object(
            base_test_case=self.base_test_case, causal_test_case_list=test_list, estimators_classes=self.estimators
        )
        self.causal_specification = CausalSpecification(self.scenario, self.causal_dag)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir_path)

    def test_adding_test_object(self):
        "test an object can be added to the test_suite using the add_test_object function"
        test_suite = CausalTestSuite()
        test_list = [CausalTestCase(self.base_test_case, self.expected_causal_effect, 0, 1)]
        estimators = [LinearRegressionEstimator]
        test_suite.add_test_object(
            base_test_case=self.base_test_case, causal_test_case_list=test_list, estimators_classes=estimators
        )
        manual_test_object = {
            self.base_test_case: {"tests": test_list, "estimators": estimators, "estimate_type": "ate"}
        }
        self.assertEqual(test_suite, manual_test_object)

    def test_return_single_test_object(self):
        """Test that a single test case can be returned from the test_suite"""
        base_test_case = BaseTestCase(self.A, self.D)

        test_list = [CausalTestCase(self.base_test_case, self.expected_causal_effect, 0, 1)]
        estimators = [LinearRegressionEstimator]
        self.test_suite.add_test_object(
            base_test_case=base_test_case, causal_test_case_list=test_list, estimators_classes=estimators
        )

        manual_test_case = {"tests": test_list, "estimators": estimators, "estimate_type": "ate"}

        test_case = self.test_suite[base_test_case]

        self.assertEqual(test_case, manual_test_case)

    def test_execute_test_suite_single_base_test_case(self):
        """Check that the test suite can return the correct results from dummy data for a single base_test-case"""

        causal_test_results = self.test_suite.execute_test_suite(self.causal_specification, self.df)
        causal_test_case_result = causal_test_results[self.base_test_case]
        self.assertAlmostEqual(
            causal_test_case_result["LinearRegressionEstimator"][0].test_value.value[0], 4, delta=1e-10
        )
