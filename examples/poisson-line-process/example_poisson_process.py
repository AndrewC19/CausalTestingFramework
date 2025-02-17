from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.variable import Input, Output
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_test_outcome import ExactValue, Positive
from causal_testing.estimation.linear_regression_estimator import LinearRegressionEstimator
from causal_testing.estimation.abstract_estimator import Estimator
from causal_testing.testing.base_test_case import BaseTestCase

import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(message)s")


class EmpiricalMeanEstimator(Estimator):
    def add_modelling_assumptions(self):
        """
        Add modelling assumptions to the estimator. This is a list of strings which list the modelling assumptions that
        must hold if the resulting causal inference is to be considered valid.
        """
        self.modelling_assumptions += "The data must contain runs with the exact configuration of interest."

    def estimate_ate(self) -> float:
        """Estimate the outcomes under control and treatment.
        :return: The empirical average treatment effect.
        """
        control_results = self.df.where(self.df[self.treatment[0]] == self.control_value)[self.outcome].dropna()
        treatment_results = self.df.where(self.df[self.treatment[0]] == self.treatment_value)[self.outcome].dropna()
        return treatment_results.mean()[0] - control_results.mean()[0], None

    def estimate_risk_ratio(self) -> float:
        """Estimate the outcomes under control and treatment.
        :return: The empirical average treatment effect.
        """
        control_results = self.df.where(self.df[self.treatment[0]] == self.control_value)[self.outcome].dropna()
        treatment_results = self.df.where(self.df[self.treatment[0]] == self.treatment_value)[self.outcome].dropna()
        return treatment_results.mean()[0] / control_results.mean()[0], None


# 1. Read in the Causal DAG
ROOT = os.path.realpath(os.path.dirname(__file__))
causal_dag = CausalDAG(f"{ROOT}/dag.dot")

# 2. Create variables
width = Input("width", float)
height = Input("height", float)
intensity = Input("intensity", float)

num_lines_abs = Output("num_lines_abs", float)
num_lines_unit = Output("num_lines_unit", float)
num_shapes_abs = Output("num_shapes_abs", float)
num_shapes_unit = Output("num_shapes_unit", float)

# 3. Create scenario by applying constraints over a subset of the input variables
scenario = Scenario(
    variables={
        width,
        height,
        intensity,
        num_lines_abs,
        num_lines_unit,
        num_shapes_abs,
        num_shapes_unit,
    }
)

# 4. Construct a causal specification from the scenario and causal DAG
causal_specification = CausalSpecification(scenario, causal_dag)

observational_data_path = f"{ROOT}/data/random/data_random_1000.csv"


def causal_test_intensity_num_shapes(
    observational_data_path,
    causal_test_case,
    square_terms=[],
    inverse_terms=[],
    empirical=False,
):
    # 7. Obtain the minimal adjustment set for the causal test case from the causal DAG
    minimal_adjustment_set = causal_dag.identification(causal_test_case.base_test_case)

    # 8. Set up an estimator
    data = pd.read_csv(observational_data_path, index_col=0).astype(float)

    treatment = causal_test_case.treatment_variable.name
    outcome = causal_test_case.outcome_variable.name

    estimator = None
    if empirical:
        estimator = EmpiricalMeanEstimator(
            treatment=[treatment],
            control_value=causal_test_case.control_value,
            treatment_value=causal_test_case.treatment_value,
            adjustment_set=set(),
            outcome=[outcome],
            df=data,
            effect_modifiers=causal_test_case.effect_modifier_configuration,
        )
    else:
        square_terms = [f"I({t} ** 2)" for t in square_terms]
        inverse_terms = [f"I({t} ** -1)" for t in inverse_terms]
        estimator = LinearRegressionEstimator(
            treatment=treatment,
            control_value=causal_test_case.control_value,
            treatment_value=causal_test_case.treatment_value,
            adjustment_set=set(),
            outcome=outcome,
            df=data,
            effect_modifiers=causal_test_case.effect_modifier_configuration,
            formula=f"{outcome} ~ {treatment} + {'+'.join(square_terms + inverse_terms + list([e for e in causal_test_case.effect_modifier_configuration]))} -1",
        )

    # 9. Execute the test
    causal_test_result = causal_test_case.execute_test(estimator)

    return causal_test_result


def test_poisson_intensity_num_shapes(save=False):
    intensity_num_shapes_results = []
    for wh in range(1, 11):
        smt_data_path = f"{ROOT}/data/smt_100/data_smt_wh{wh}_100.csv"
        for control_value, treatment_value in [(1, 2), (2, 4), (4, 8), (8, 16)]:
            logger.info("%s CAUSAL TEST %s", "=" * 33, "=" * 33)
            logger.info("WIDTH = HEIGHT = %s", wh)
            logger.info("Identifying")
            base_test_case = BaseTestCase(treatment_variable=intensity, outcome_variable=num_shapes_unit)
            causal_test_case = CausalTestCase(
                base_test_case=base_test_case,
                expected_causal_effect=ExactValue(4, atol=0.5),
                treatment_value=treatment_value,
                control_value=control_value,
                estimate_type="risk_ratio",
            )
            obs_causal_test_result = causal_test_intensity_num_shapes(
                observational_data_path,
                causal_test_case,
                square_terms=["intensity"],
                empirical=False,
            )
            logger.info("Observational %s", obs_causal_test_result)
            smt_causal_test_result = causal_test_intensity_num_shapes(
                smt_data_path, causal_test_case, square_terms=["intensity"], empirical=True
            )
            logger.info("RCT %s", smt_causal_test_result)

            results = {
                "width": wh,
                "height": wh,
                "control": control_value,
                "treatment": treatment_value,
                "smt_risk_ratio": smt_causal_test_result.test_value.value,
                "obs_risk_ratio": obs_causal_test_result.test_value.value,
            }
            intensity_num_shapes_results.append(results)

    intensity_num_shapes_results = pd.DataFrame(intensity_num_shapes_results)
    if save:
        intensity_num_shapes_results.to_csv("intensity_num_shapes_results_random_1000.csv")
    logger.info("%s", intensity_num_shapes_results)


def test_poisson_width_num_shapes(save=False):
    width_num_shapes_results = []
    for i in range(17):
        for w in range(1, 10):
            logger.info("%s CAUSAL TEST %s", "=" * 33, "=" * 33)
            logger.info("Identifying")
            # 5. Create a causal test case
            control_value = float(w)
            treatment_value = w + 1.0
            base_test_case = BaseTestCase(treatment_variable=width, outcome_variable=num_shapes_unit)
            causal_test_case = CausalTestCase(
                base_test_case=base_test_case,
                expected_causal_effect=Positive(),
                control_value=control_value,
                treatment_value=treatment_value,
                estimate_type="ate_calculated",
                effect_modifier_configuration={"intensity": i},
            )
            causal_test_result = causal_test_intensity_num_shapes(
                observational_data_path,
                causal_test_case,
                square_terms=["intensity"],
                inverse_terms=["width"],
            )
            logger.info("%s", causal_test_result)
            results = {
                "control": control_value,
                "treatment": treatment_value,
                "intensity": i,
                "ate": causal_test_result.test_value.value,
                "ci_low": causal_test_result.confidence_intervals[0][0],
                "ci_high": causal_test_result.confidence_intervals[1][0],
            }
            width_num_shapes_results.append(results)
    width_num_shapes_results = pd.DataFrame(width_num_shapes_results)
    if save:
        width_num_shapes_results.to_csv("width_num_shapes_results_random_1000.csv")
    logger.info("%s", width_num_shapes_results)


if __name__ == "__main__":
    # test_poisson_intensity_num_shapes(save=True)
    test_poisson_width_num_shapes(save=True)
