"""This module contains the ExperimentalEstimator class for directly interacting with the system under test."""

from typing import Any
from abc import abstractmethod
import pandas as pd

from causal_testing.estimation.abstract_estimator import Estimator


class ExperimentalEstimator(Estimator):
    """A Logistic Regression Estimator is a parametric estimator which restricts the variables in the data to a linear
    combination of parameters and functions of the variables (note these functions need not be linear). It is designed
    for estimating categorical outcomes.
    """

    def __init__(
        # pylint: disable=too-many-arguments
        self,
        treatment: str,
        treatment_value: float,
        control_value: float,
        adjustment_set: dict[str:Any],
        outcome: str,
        effect_modifiers: dict[str:Any] = None,
        alpha: float = 0.05,
        repeats: int = 200,
    ):
        # pylint: disable=R0801
        super().__init__(
            treatment=treatment,
            treatment_value=treatment_value,
            control_value=control_value,
            adjustment_set=adjustment_set,
            outcome=outcome,
            effect_modifiers=effect_modifiers,
            alpha=alpha,
        )
        if effect_modifiers is None:
            self.effect_modifiers = {}
        self.repeats = repeats

    def add_modelling_assumptions(self):
        """
        Add modelling assumptions to the estimator. This is a list of strings which list the modelling assumptions that
        must hold if the resulting causal inference is to be considered valid.
        """
        self.modelling_assumptions.append(
            "The supplied number of repeats must be sufficient for statistical significance"
        )

    @abstractmethod
    def run_system(self, configuration: dict) -> dict:
        """
        Runs the system under test with the supplied configuration and supplies the outputs as a dict.
        :param configuration: The run configuration arguments.
        :returns: The resulting output as a dict.
        """

    def estimate_ate(self) -> tuple[pd.Series, list[pd.Series, pd.Series]]:
        """Estimate the average treatment effect of the treatment on the outcome. That is, the change in outcome caused
        by changing the treatment variable from the control value to the treatment value.

        :return: The average treatment effect and the bootstrapped confidence intervals.
        """
        control_configuration = self.adjustment_set | self.effect_modifiers | {self.treatment: self.control_value}
        treatment_configuration = self.adjustment_set | self.effect_modifiers | {self.treatment: self.treatment_value}

        control_outcomes = pd.DataFrame([self.run_system(control_configuration) for _ in range(self.repeats)])
        treatment_outcomes = pd.DataFrame([self.run_system(treatment_configuration) for _ in range(self.repeats)])

        difference = (treatment_outcomes[self.outcome] - control_outcomes[self.outcome]).sort_values().reset_index()

        ci_low_index = round(self.repeats * (self.alpha / 2))
        ci_low = difference.iloc[ci_low_index]
        ci_high = difference.iloc[self.repeats - ci_low_index]

        return pd.Series({self.treatment: difference.mean()[self.outcome]}), [
            pd.Series({self.treatment: ci_low[self.outcome]}),
            pd.Series({self.treatment: ci_high[self.outcome]}),
        ]

    def estimate_risk_ratio(self) -> tuple[pd.Series, list[pd.Series, pd.Series]]:
        """Estimate the risk ratio of the treatment on the outcome. That is, the change in outcome caused
        by changing the treatment variable from the control value to the treatment value.

        :return: The average treatment effect and the bootstrapped confidence intervals.
        """
        control_configuration = self.adjustment_set | self.effect_modifiers | {self.treatment: self.control_value}
        treatment_configuration = self.adjustment_set | self.effect_modifiers | {self.treatment: self.treatment_value}

        control_outcomes = pd.DataFrame([self.run_system(control_configuration) for _ in range(self.repeats)])
        treatment_outcomes = pd.DataFrame([self.run_system(treatment_configuration) for _ in range(self.repeats)])

        difference = (treatment_outcomes[self.outcome] / control_outcomes[self.outcome]).sort_values().reset_index()

        ci_low_index = round(self.repeats * (self.alpha / 2))
        ci_low = difference.iloc[ci_low_index]
        ci_high = difference.iloc[self.repeats - ci_low_index]

        return pd.Series({self.treatment: difference.mean()[self.outcome]}), [
            pd.Series({self.treatment: ci_low[self.outcome]}),
            pd.Series({self.treatment: ci_high[self.outcome]}),
        ]
