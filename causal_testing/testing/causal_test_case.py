"""This module contains the CausalTestCase class, a class that holds the information required for a causal test"""
import logging
from typing import Any

from causal_testing.specification.variable import Variable
from causal_testing.testing.causal_test_outcome import CausalTestOutcome
from causal_testing.testing.base_test_case import BaseTestCase
from causal_testing.testing.estimators import Estimator
from causal_testing.testing.causal_test_result import CausalTestResult, TestValue
from causal_testing.data_collection.data_collector import ObservationalDataCollector


logger = logging.getLogger(__name__)


class CausalTestCase:
    # pylint: disable=too-many-instance-attributes
    """
    A CausalTestCase extends the information held in a BaseTestCase. As well as storing the treatment and outcome
    variables, a CausalTestCase stores the values of these variables. Also the outcome variable and value are
    specified.

    The goal of a CausalTestCase is to test whether the intervention made to the control via the treatment causes the
    model-under-test to produce the expected change. The CausalTestCase structure is designed for execution using the
    CausalTestEngine, using either execute_test() function to execute a single test case or packing CausalTestCases into
    a CausalTestSuite and executing them as a batch using the execute_test_suite() function.
    """

    def __init__(
        # pylint: disable=too-many-arguments
        self,
        base_test_case: BaseTestCase,
        expected_causal_effect: CausalTestOutcome,
        control_value: Any = None,
        treatment_value: Any = None,
        estimate_type: str = "ate",
        estimate_params: dict = None,
        effect_modifier_configuration: dict[Variable:Any] = None,
    ):
        """
        :param base_test_case: A BaseTestCase object consisting of a treatment variable, outcome variable and effect
        :param expected_causal_effect: The expected causal effect (Positive, Negative, No Effect).
        :param control_value: The control value for the treatment variable (before intervention).
        :param treatment_value: The treatment value for the treatment variable (after intervention).
        :param estimate_type: A string which denotes the type of estimate to return
        :param effect_modifier_configuration:
        """
        self.base_test_case = base_test_case
        self.control_value = control_value
        self.expected_causal_effect = expected_causal_effect
        self.outcome_variable = base_test_case.outcome_variable
        self.treatment_variable = base_test_case.treatment_variable
        self.treatment_value = treatment_value
        self.estimate_type = estimate_type
        if estimate_params is None:
            self.estimate_params = {}
        self.effect = base_test_case.effect

        if effect_modifier_configuration:
            self.effect_modifier_configuration = effect_modifier_configuration
        else:
            self.effect_modifier_configuration = {}

    def get_treatment_variable(self):
        """Return the treatment variable name (as string) for this causal test case"""
        return self.treatment_variable.name

    def get_outcome_variable(self):
        """Return the outcome variable name (as string) for this causal test case."""
        return self.outcome_variable.name

    def get_control_value(self):
        """Return a the control value of the treatment variable in this causal test case."""
        return self.control_value

    def get_treatment_value(self):
        """Return the treatment value of the treatment variable in this causal test case."""
        return self.treatment_value

    def execute_test(self, estimator: type(Estimator), data_collector: ObservationalDataCollector) -> CausalTestResult:
        """Execute a causal test case and return the causal test result.

        :param estimator: A reference to an Estimator class.
        :param data_collector: The data collector to be used which provides a dataframe for the Estimator
        :return causal_test_result: A CausalTestResult for the executed causal test case.
        """
        if estimator.df is None:
            estimator.df = data_collector.collect_data()
        treatment_variable = self.treatment_variable
        treatments = treatment_variable.name
        outcome_variable = self.outcome_variable

        logger.info("treatments: %s", treatments)
        logger.info("outcomes: %s", outcome_variable)

        causal_test_result = self._return_causal_test_results(estimator)
        return causal_test_result

    def _return_causal_test_results(self, estimator) -> CausalTestResult:
        """Depending on the estimator used, calculate the 95% confidence intervals and return in a causal_test_result

        :param estimator: An Estimator class object
        :return: a CausalTestResult object containing the confidence intervals
        """
        if not hasattr(estimator, f"estimate_{self.estimate_type}"):
            raise AttributeError(f"{estimator.__class__} has no {self.estimate_type} method.")
        estimate_effect = getattr(estimator, f"estimate_{self.estimate_type}")
        effect, confidence_intervals = estimate_effect(**self.estimate_params)
        causal_test_result = CausalTestResult(
            estimator=estimator,
            test_value=TestValue(self.estimate_type, effect),
            effect_modifier_configuration=self.effect_modifier_configuration,
            confidence_intervals=confidence_intervals,
        )

        return causal_test_result

    def __str__(self):
        treatment_config = {self.treatment_variable.name: self.treatment_value}
        control_config = {self.treatment_variable.name: self.control_value}
        outcome_variable = {self.outcome_variable}
        return (
            f"Running {treatment_config} instead of {control_config} should cause the following "
            f"changes to {outcome_variable}: {self.expected_causal_effect}."
        )
