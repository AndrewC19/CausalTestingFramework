# -*- coding: utf-8 -*-
import os
import logging
import pandas as pd
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.variable import Input, Output
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_test_outcome import Positive, Negative, NoEffect
from causal_testing.estimation.linear_regression_estimator import LinearRegressionEstimator
from causal_testing.testing.base_test_case import BaseTestCase


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(message)s")
ROOT = os.path.realpath(os.path.dirname(__file__))


def setup_test_case(verbose: bool = False):
    """Run the causal test case for the effect of changing vaccine to prioritise elderly from observational
    data that was previously simulated.

    :param verbose: Whether to print verbose details (causal test results).
    :return results_dict: A dictionary containing ATE, 95% CIs, and Test Pass/Fail
    """

    # 1. Read in the Causal DAG
    causal_dag = CausalDAG(f"{ROOT}/dag.dot")

    # 2. Create variables
    pop_size = Input("pop_size", int)
    pop_infected = Input("pop_infected", int)
    n_days = Input("n_days", int)
    vaccine = Input("vaccine", int)
    cum_infections = Output("cum_infections", int)
    cum_vaccinations = Output("cum_vaccinations", int)
    cum_vaccinated = Output("cum_vaccinated", int)
    max_doses = Output("max_doses", int)

    # 3. Create scenario by applying constraints over a subset of the input variables
    scenario = Scenario(
        variables={
            pop_size,
            pop_infected,
            n_days,
            cum_infections,
            vaccine,
            cum_vaccinated,
            cum_vaccinations,
            max_doses,
        },
        constraints={pop_size.z3 == 50000, pop_infected.z3 == 1000, n_days.z3 == 50},
    )

    # 4. Construct a causal specification from the scenario and causal DAG
    causal_specification = CausalSpecification(scenario, causal_dag)

    # 5. Read the previously simulated data
    obs_df = pd.read_csv("simulated_data.csv")

    # 6. Express expected outcomes
    expected_outcome_effects = {
        cum_infections: Positive(),
        cum_vaccinations: Negative(),
        cum_vaccinated: Negative(),
        max_doses: NoEffect(),
    }
    results_dict = {"cum_infections": {}, "cum_vaccinations": {}, "cum_vaccinated": {}, "max_doses": {}}

    for outcome_variable, expected_effect in expected_outcome_effects.items():
        base_test_case = BaseTestCase(treatment_variable=vaccine, outcome_variable=outcome_variable)
        causal_test_case = CausalTestCase(
            base_test_case=base_test_case, expected_causal_effect=expected_effect, control_value=0, treatment_value=1
        )
        # 7. Obtain the minimal adjustment set for the causal test case from the causal DAG
        minimal_adjustment_set = causal_dag.identification(base_test_case)

        # 8. Build statistical model using the Linear Regression estimator
        linear_regression_estimator = LinearRegressionEstimator(
            treatment=vaccine.name,
            treatment_value=1,
            control_value=0,
            adjustment_set=minimal_adjustment_set,
            outcome=outcome_variable.name,
            df=obs_df,
        )

        # 9. Execute test and save results in dict
        causal_test_result = causal_test_case.execute_test(linear_regression_estimator, obs_df)

        if verbose:
            logging.info("Causation:\n%s", causal_test_result)

        results_dict[outcome_variable.name]["ate"] = causal_test_result.test_value.value

        results_dict[outcome_variable.name]["cis"] = causal_test_result.confidence_intervals

        results_dict[outcome_variable.name]["test_passes"] = causal_test_case.expected_causal_effect.apply(
            causal_test_result
        )

    return results_dict


if __name__ == "__main__":

    test_results = setup_test_case(verbose=True)

    logging.info("%s", test_results)
