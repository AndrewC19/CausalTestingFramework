"""This module contains the CausalTestEngine class, a class responsible for the execution of causal tests"""

import logging

from causal_testing.data_collection.data_collector import DataCollector
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_test_result import CausalTestResult, TestValue
from causal_testing.testing.estimators import Estimator
from causal_testing.testing.base_test_case import BaseTestCase
from causal_testing.testing.causal_test_suite import CausalTestSuite

logger = logging.getLogger(__name__)


class CausalTestEngine:
    """
    Overarching workflow for Causal Testing. The CausalTestEngine proceeds in four steps.
    (1) Given a causal test case, specification, and (optionally) observational data, compute the causal estimand
        that, once estimated, yields the desired causal effect. This is essentially the recipe for a statistical
        procedure that, according to the assumptions encoded in the causal specification, estimates the casual effect
        of interest.
    (2) If using observational data, check whether the data is sufficient for estimating the causal effect of interest.
        If the data is insufficient, identify the missing data to guide the user towards un-exercised areas of the
        system-under-test. Else, if generating experimental data, run the model in the experimental conditions required
        to isolate the causal effect of interest.
    (3) Using the gathered data (whether observational or experimental), implement the statistical procedure prescribed
        by the causal estimand. For example, apply a linear regression model which includes a term for the set of
        variables which block (d-separate) all back-door paths. Return the causal estimate obtained following this
        procedure and, optionally, (depending on the estimator used) confidence intervals for this estimate. These are
        provided as an instance of the CausalTestResult class.
    (4) Define a test oracle procedure which uses the causal test results to determine whether the intervention has
        had the anticipated causal effect. This should assign a pass/fail value to the CausalTestResult.

    Data is loaded as part of the "__init__" function
    Data can be loaded in two ways:
            (1) Experimentally - the model is executed with the treatment and control input configurations under
                conditions that guarantee the observed change in outcome must be caused by the change in input
                (intervention).
            (2) Observationally - previous execution data is supplied in the form of a csv which is then filtered
                to remove any data corresponding to executions of a different scenario
                (i.e. not the scenario-under-test) and checked for positivity violations.

        After the data is loaded, both are treated in the same way and, provided the identifiability and modelling
        assumptions hold, can be used to estimate the causal effect for the causal test case.
    """

    def __init__(self, causal_specification: CausalSpecification, data_collector: DataCollector, **kwargs):
        self.causal_dag, self.scenario = (
            causal_specification.causal_dag,
            causal_specification.scenario,
        )
        self.data_collector = data_collector
        self.scenario_execution_data_df = self.data_collector.collect_data(**kwargs)

    def execute_test_suite(self, test_suite: CausalTestSuite) -> list[CausalTestResult]:
        """Execute a suite of causal tests and return the results in a list
        :param test_suite: CasualTestSuite object
        :return: A dictionary where each key is the name of the estimators specified and the values are lists of
                causal_test_result objects
        """
        if self.scenario_execution_data_df.empty:
            raise ValueError("No data has been loaded. Please call load_data prior to executing a causal test case.")
        test_suite_results = {}
        for edge in test_suite:
            print("edge: ")
            print(edge)
            logger.info("treatment: %s", edge.treatment_variable)
            logger.info("outcome: %s", edge.outcome_variable)
            minimal_adjustment_set = self.causal_dag.identification(edge)
            minimal_adjustment_set = minimal_adjustment_set - set(edge.treatment_variable.name)
            minimal_adjustment_set = minimal_adjustment_set - set(edge.outcome_variable.name)

            variables_for_positivity = list(minimal_adjustment_set) + [
                edge.treatment_variable.name,
                edge.outcome_variable.name,
            ]

            if self._check_positivity_violation(variables_for_positivity):
                raise ValueError("POSITIVITY VIOLATION -- Cannot proceed.")

            estimators = test_suite[edge]["estimators"]
            tests = test_suite[edge]["tests"]
            estimate_type = test_suite[edge]["estimate_type"]
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
                    )
                    if estimator.df is None:
                        estimator.df = self.scenario_execution_data_df
                    causal_test_result = self._return_causal_test_results(estimate_type, estimator, test)
                    causal_test_results.append(causal_test_result)

                results[estimator_class.__name__] = causal_test_results
            test_suite_results[edge] = results
        return test_suite_results

    def execute_test(
        self, estimator: type(Estimator), causal_test_case: CausalTestCase, estimate_type: str = "ate"
    ) -> CausalTestResult:
        """Execute a causal test case and return the causal test result.

        Test case execution proceeds with the following steps:
        (1) Check that data has been loaded using the method load_data
        (2) Check loaded data for any positivity violations and warn the user if so
        (3) Instantiate the estimator with the values of the causal test case.
        (4) Using the estimator, estimate the average treatment effect of the changing the treatment from control value
            to treatment value on the outcome of interest, adjusting for the identified adjustment set.
        (5) Depending on the estimator used, compute 95% confidence intervals for the estimate.
        (6) Store results in an instance of CausalTestResults.
        (7) Apply test oracle procedure to assign a pass/fail to the CausalTestResult and return.

        :param estimator: A reference to an Estimator class.
        :param causal_test_case: The CausalTestCase object to be tested
        :param estimate_type: A string which denotes the type of estimate to return, ATE or CATE.
        :return causal_test_result: A CausalTestResult for the executed causal test case.
        """
        if self.scenario_execution_data_df.empty:
            raise ValueError("No data has been loaded. Please call load_data prior to executing a causal test case.")
        if estimator.df is None:
            estimator.df = self.scenario_execution_data_df
        treatment_variable = causal_test_case.treatment_variable
        treatments = treatment_variable.name
        outcome_variable = causal_test_case.outcome_variable

        logger.info("treatments: %s", treatments)
        logger.info("outcomes: %s", outcome_variable)
        minimal_adjustment_set = self.causal_dag.identification(BaseTestCase(treatment_variable, outcome_variable))
        minimal_adjustment_set = minimal_adjustment_set - set(treatment_variable.name)
        minimal_adjustment_set = minimal_adjustment_set - set(outcome_variable.name)

        variables_for_positivity = list(minimal_adjustment_set) + [treatment_variable.name] + [outcome_variable.name]

        if self._check_positivity_violation(variables_for_positivity):
            raise ValueError("POSITIVITY VIOLATION -- Cannot proceed.")

        causal_test_result = self._return_causal_test_results(estimate_type, estimator, causal_test_case)
        return causal_test_result

    def _return_causal_test_results(self, estimate_type, estimator, causal_test_case):
        """Depending on the estimator used, calculate the 95% confidence intervals and return in a causal_test_result

        :param estimate_type: A string which denotes the type of estimate to return
        :param estimator: An Estimator class object
        :param causal_test_case: The concrete test case to be executed
        :return: a CausalTestResult object containing the confidence intervals
        """
        if estimate_type == "cate":
            logger.debug("calculating cate")
            if not hasattr(estimator, "estimate_cates"):
                raise NotImplementedError(f"{estimator.__class__} has no CATE method.")

            cates_df, confidence_intervals = estimator.estimate_cates()
            causal_test_result = CausalTestResult(
                estimator=estimator,
                test_value=TestValue("ate", cates_df),
                effect_modifier_configuration=causal_test_case.effect_modifier_configuration,
                confidence_intervals=confidence_intervals,
            )
        elif estimate_type == "risk_ratio":
            logger.debug("calculating risk_ratio")
            risk_ratio, confidence_intervals = estimator.estimate_risk_ratio()
            causal_test_result = CausalTestResult(
                estimator=estimator,
                test_value=TestValue("risk_ratio", risk_ratio),
                effect_modifier_configuration=causal_test_case.effect_modifier_configuration,
                confidence_intervals=confidence_intervals,
            )
        elif estimate_type == "coefficient":
            logger.debug("calculating coefficient")
            coefficient, confidence_intervals = estimator.estimate_unit_ate()
            causal_test_result = CausalTestResult(
                estimator=estimator,
                test_value=TestValue("coefficient", coefficient),
                effect_modifier_configuration=causal_test_case.effect_modifier_configuration,
                confidence_intervals=confidence_intervals,
            )
        elif estimate_type == "cate_coefficient":
            logger.debug("calculating CATE coefficients")
            coefficient, confidence_intervals = estimator.estimate_unit_cate()
            causal_test_result = CausalTestResult(
                estimator=estimator,
                test_value=TestValue("coefficient", coefficient),
                effect_modifier_configuration=causal_test_case.effect_modifier_configuration,
                confidence_intervals=confidence_intervals,
            )
        elif estimate_type == "ate":
            logger.debug("calculating ate")
            ate, confidence_intervals = estimator.estimate_ate()
            causal_test_result = CausalTestResult(
                estimator=estimator,
                test_value=TestValue("ate", ate),
                effect_modifier_configuration=causal_test_case.effect_modifier_configuration,
                confidence_intervals=confidence_intervals,
            )
            # causal_test_result = CausalTestResult(minimal_adjustment_set, ate, confidence_intervals)
            # causal_test_result.apply_test_oracle_procedure(self.causal_test_case.expected_causal_effect)
        elif estimate_type == "ate_calculated":
            logger.debug("calculating ate")
            ate, confidence_intervals = estimator.estimate_ate_calculated()
            causal_test_result = CausalTestResult(
                estimator=estimator,
                test_value=TestValue("ate", ate),
                effect_modifier_configuration=causal_test_case.effect_modifier_configuration,
                confidence_intervals=confidence_intervals,
            )
            # causal_test_result = CausalTestResult(minimal_adjustment_set, ate, confidence_intervals)
            # causal_test_result.apply_test_oracle_procedure(self.causal_test_case.expected_causal_effect)
        else:
            raise ValueError(f"Invalid estimate type {estimate_type}, expected 'ate', 'cate', or 'risk_ratio'")
        return causal_test_result

    def _check_positivity_violation(self, variables_list):
        """Check whether the dataframe has a positivity violation relative to the specified variables list.

        A positivity violation occurs when there is a stratum of the dataframe which does not have any data. Put simply,
        if we split the dataframe into covariate sub-groups, each sub-group must contain both a treated and untreated
        individual. If a positivity violation occurs, causal inference is still possible using a properly specified
        parametric estimator. Therefore, we should not throw an exception upon violation but raise a warning instead.

        :param variables_list: The list of variables for which positivity must be satisfied.
        :return: True if positivity is violated, False otherwise.
        """
        if not (set(variables_list) - {x.name for x in self.scenario.hidden_variables()}).issubset(
            self.scenario_execution_data_df.columns
        ):
            missing_variables = set(variables_list) - set(self.scenario_execution_data_df.columns)
            logger.warning(
                "Positivity violation: missing data for variables %s.\n"
                "Causal inference is only valid if a well-specified parametric model is used.\n"
                "Alternatively, consider restricting analysis to executions without the variables:"
                ".",
                missing_variables,
            )
            return True

        return False
