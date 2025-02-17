import numpy as np
import pandas as pd
import scipy
import os

from causal_testing.estimation.linear_regression_estimator import LinearRegressionEstimator
from causal_testing.testing.causal_test_outcome import ExactValue, Positive, Negative, NoEffect, CausalTestOutcome
from causal_testing.testing.causal_test_result import CausalTestResult
from causal_testing.json_front.json_class import JsonUtility
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.variable import Input, Output, Meta

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(message)s")


class WidthHeightEstimator(LinearRegressionEstimator):
    """
    Extension of LinearRegressionEstimator class to include scenario specific user code
    """

    def add_modelling_assumptions(self):
        """
        Add modelling assumptions to the estimator. This is a list of strings which list the modelling assumptions that
        must hold if the resulting causal inference is to be considered valid.
        """
        self.modelling_assumptions += "The output varies according to 2i(w+h)"

    def estimate_ate(self) -> (float, [float, float], float):
        """Estimate the conditional average treatment effect of the treatment on the outcome. That is, the change
        in outcome caused by changing the treatment variable from the control value to the treatment value.
        :return: The conditional average treatment effect and the 95% Wald confidence intervals.
        """
        assert (
            self.effect_modifiers
        ), f"Must have at least one effect modifier to compute CATE - {self.effect_modifiers}."
        x = pd.DataFrame()
        x[self.treatment[0]] = [self.treatment_values, self.control_values]
        x["Intercept"] = 1
        for k, v in self.effect_modifiers.items():
            self.adjustment_set.add(k)
            x[k] = v
        if hasattr(self, "product_terms"):
            for a, b in self.product_terms:
                x[f"{a}*{b}"] = x[a] * x[b]

        x.drop(["width", "height"], axis=1, inplace=True)
        self.adjustment_set = {"width*intensity", "height*intensity"}

        logger.info("%s", x)
        logger.info("%s", self.adjustment_set)

        model = self._run_linear_regression()
        logger.info("%s", model.summary())
        y = model.predict(x)
        treatment_outcome = y.iloc[0]
        control_outcome = y.iloc[1]

        return treatment_outcome - control_outcome, None


class PoissonWidthHeight(CausalTestOutcome):
    """An extension of TestOutcome representing that the expected causal effect should be positive."""

    def __init__(self, atol=0.5):
        self.atol = atol
        self.i2c = None

    def apply(self, res: CausalTestResult) -> bool:
        # TODO: confidence intervals?
        logger.info("=== APPLYING ===")
        logger.info("effect_modifier_configuration", res.effect_modifier_configuration)
        effect_modifier_configuration = {k.name: v for k, v in res.effect_modifier_configuration.items()}
        c = res.treatment_value - res.control_value
        i = effect_modifier_configuration["intensity"]
        self.i2c = i * 2 * c
        logger.info("2ic: 2 * %s * %s = %s", i, c, self.i2c)
        logger.info("ate: %s", res.test_value.value)
        return np.isclose(res.test_value.value, self.i2c, atol=self.atol)

    def __str__(self):
        if self.i2c is None:
            return f"PoissonWidthHeight±{self.atol}"
        return f"PoissonWidthHeight:{self.i2c}±{self.atol}"


def populate_width_height(data):
    data["width_plus_height"] = data["width"] + data["height"]


def populate_num_lines_unit(data):
    area = data["width"] * data["height"]
    data["num_lines_unit"] = data["num_lines_abs"] / area


def populate_num_shapes_unit(data):
    area = data["width"] * data["height"]
    data["num_shapes_unit"] = data["num_shapes_abs"] / area


inputs = [
    {"name": "width", "datatype": float, "distribution": scipy.stats.uniform(0, 10)},
    {"name": "height", "datatype": float, "distribution": scipy.stats.uniform(0, 10)},
    {"name": "intensity", "datatype": float, "distribution": scipy.stats.uniform(0, 10)},
]

outputs = [{"name": "num_lines_abs", "datatype": float}, {"name": "num_shapes_abs", "datatype": float}]

metas = [
    {"name": "num_lines_unit", "datatype": float, "populate": populate_num_lines_unit},
    {"name": "num_shapes_unit", "datatype": float, "populate": populate_num_shapes_unit},
    {"name": "width_plus_height", "datatype": float, "populate": populate_width_height},
]

constraints = ["width > 0", "height > 0", "intensity > 0"]

effects = {
    "PoissonWidthHeight": PoissonWidthHeight(),
    "Positive": Positive(),
    "Negative": Negative(),
    "ExactValue4_05": ExactValue(4, atol=0.5),
    "NoEffect": NoEffect(),
}

estimators = {
    "WidthHeightEstimator": WidthHeightEstimator,
    "LinearRegressionEstimator": LinearRegressionEstimator,
}

# Create input structure required to create a modelling scenario
modelling_inputs = (
    [Input(i["name"], i["datatype"], i["distribution"]) for i in inputs]
    + [Output(i["name"], i["datatype"]) for i in outputs]
    + ([Meta(i["name"], i["datatype"], i["populate"]) for i in metas] if metas else list())
)

# Create modelling scenario to access z3 variable mirrors
modelling_scenario = Scenario(modelling_inputs, None)
modelling_scenario.setup_treatment_variables()

mutates = {
    "Increase": lambda x: modelling_scenario.treatment_variables[x].z3 > modelling_scenario.variables[x].z3,
    "ChangeByFactor(2)": lambda x: modelling_scenario.treatment_variables[x].z3
    == modelling_scenario.variables[x].z3 * 2,
}


def test_run_causal_tests():
    ROOT = os.path.realpath(os.path.dirname(__file__))

    log_path = f"{ROOT}/json_frontend.log"
    json_path = f"{ROOT}/causal_tests.json"
    dag_path = f"{ROOT}/dag.dot"
    data_path = f"{ROOT}/data.csv"

    json_utility = JsonUtility(log_path, output_overwrite=True)  # Create an instance of the extended JsonUtility class
    json_utility.set_paths(
        json_path, dag_path, [data_path]
    )  # Set the path to the data.csv, dag.dot and causal_tests.json file

    # Load the Causal Variables into the JsonUtility class ready to be used in the tests
    json_utility.setup(
        scenario=modelling_scenario
    )  # Sets up all the necessary parts of the json_class needed to execute tests

    json_utility.run_json_tests(effects=effects, mutates=mutates, estimators=estimators, f_flag=False)


if __name__ == "__main__":
    args = JsonUtility.get_args()
    json_utility = JsonUtility(args.log_path)  # Create an instance of the extended JsonUtility class
    json_utility.set_paths(
        args.json_path, args.dag_path, args.data_path
    )  # Set the path to the data.csv, dag.dot and causal_tests.json file

    # Load the Causal Variables into the JsonUtility class ready to be used in the tests
    json_utility.setup(
        scenario=modelling_scenario
    )  # Sets up all the necessary parts of the json_class needed to execute tests

    json_utility.run_json_tests(effects=effects, mutates=mutates, estimators=estimators, f_flag=args.f)
