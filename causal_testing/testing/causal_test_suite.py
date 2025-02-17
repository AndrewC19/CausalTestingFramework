"""This module contains the CausalTestSuite class, for details on using it:
https://causal-testing-framework.readthedocs.io/en/latest/test_suite.html"""

import logging
from typing import Type, Iterable
from collections import UserDict
import pandas as pd

from causal_testing.testing.base_test_case import BaseTestCase
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.estimation.abstract_estimator import Estimator
from causal_testing.testing.causal_test_result import CausalTestResult
from causal_testing.specification.causal_specification import CausalSpecification

logger = logging.getLogger(__name__)


class CausalTestSuite(UserDict):
    """
    A CausalTestSuite is an extension of the UserDict class, therefore it behaves as a python dictionary with the added
    functionality of this class.
    The dictionary structure should be the keys are base_test_cases representing the treatment and outcome Variables,
    and the values are test objects. Test Objects hold a causal_test_case_list which is a list of causal_test_cases
    which provide control and treatment values, and an iterator of Estimator Class References

    This dictionary can be fed into the execute_test_suite function which will iterate over all the
    base_test_case's and execute each causal_test_case with each iterator.
    """

    def add_test_object(
        self,
        base_test_case: BaseTestCase,
        causal_test_case_list: Iterable[CausalTestCase],
        estimators_classes: Iterable[Type[Estimator]],
        estimate_type: str = "ate",
    ):
        """
        A setter object to allow for the easy construction of the dictionary test suite structure

        :param base_test_case: A BaseTestCase object consisting of a treatment variable, outcome variable and effect
        :param causal_test_case_list: A list of causal test cases to be executed
        :param estimators_classes: A list of estimator class references, the execute_test_suite function in the
            TestEngine will produce a list of test results for each estimator
        :param estimate_type: A string which denotes the type of estimate to return
        """
        test_object = {"tests": causal_test_case_list, "estimators": estimators_classes, "estimate_type": estimate_type}
        self.data[base_test_case] = test_object

    def execute_test_suite(
        self, causal_specification: CausalSpecification, df: pd.DataFrame
    ) -> dict[str, CausalTestResult]:
        """Execute a suite of causal tests and return the results in a list
        :param causal_specification: A causal specification object which wraps up the scenario and causal DAG.
        :param df: A dataframe containing the test data.
        :return: A dictionary where each key is the name of the estimators specified and the values are lists of
                causal_test_result objects
        """
        test_suite_results = {}
        for edge in self:
            logger.info("treatment: %s", edge.treatment_variable)
            logger.info("outcome: %s", edge.outcome_variable)
            minimal_adjustment_set = causal_specification.causal_dag.identification(edge)
            minimal_adjustment_set = minimal_adjustment_set - set(edge.treatment_variable.name)
            minimal_adjustment_set = minimal_adjustment_set - set(edge.outcome_variable.name)

            estimators = self[edge]["estimators"]
            tests = self[edge]["tests"]
            results = {}
            for estimator_class in estimators:
                causal_test_results = []

                for test in tests:
                    estimator = estimator_class(
                        test.treatment_variable.name,
                        test.treatment_value,
                        test.control_value,
                        minimal_adjustment_set,
                        test.outcome_variable.name,
                        df=df,
                    )
                    causal_test_result = test.execute_test(estimator)
                    causal_test_results.append(causal_test_result)

                results[estimator_class.__name__] = causal_test_results
            test_suite_results[edge] = results
        return test_suite_results
