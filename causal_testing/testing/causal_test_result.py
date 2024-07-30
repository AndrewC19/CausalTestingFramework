"""This module contains the CausalTestResult class, which is a container for the results of a causal test, and the
CausalTestValue dataclass.
"""

from typing import Any
from dataclasses import dataclass
import pandas as pd

from causal_testing.testing.estimators import Estimator
from causal_testing.specification.variable import Variable


@dataclass
class CausalTestValue:
    """A dataclass to hold both the type and value of a causal test result"""

    type: str
    value: float


class CausalTestResult:
    """A container to hold the results of a causal test case. Every causal test case provides a point estimate of
    the ATE, given a particular treatment, outcome, and adjustment set. Some but not all estimators can provide
    confidence intervals."""

    def __init__(
        # pylint: disable=too-many-arguments
        self,
        estimator: Estimator,
        test_value: CausalTestValue,
        confidence_intervals: [pd.Series, pd.Series] = None,
        effect_modifier_configuration: {Variable: Any} = None,
        adequacy=None,
    ):
        self.estimator = estimator
        self.adequacy = adequacy
        if estimator.adjustment_set:
            self.adjustment_set = estimator.adjustment_set
        else:
            self.adjustment_set = set()
        self.test_value = test_value
        self.confidence_intervals = confidence_intervals

        if effect_modifier_configuration is not None:
            self.effect_modifier_configuration = effect_modifier_configuration
        else:
            self.effect_modifier_configuration = {}

    def __str__(self):
        def push(s, inc="  "):
            return inc + str(s).replace("\n", "\n" + inc)

        result_str = str(self.test_value.value)
        if "\n" in result_str:
            result_str = "\n" + push(self.test_value.value)
        base_str = (
            f"Causal Test Result\n==============\n"
            f"Treatment: {self.estimator.treatment}\n"
            f"Control value: {self.estimator.control_value}\n"
            f"Treatment value: {self.estimator.treatment_value}\n"
            f"Outcome: {self.estimator.outcome}\n"
            f"Adjustment set: {self.adjustment_set}\n"
        )
        if hasattr(self.estimator, "formula"):
            base_str += f"Formula: {self.estimator.formula}\n"
        base_str += f"{self.test_value.type}: {result_str}\n"
        confidence_str = ""
        if self.confidence_intervals:
            ci_str = " " + str(self.confidence_intervals)
            if "\n" in ci_str:
                ci_str = " " + push(pd.DataFrame(self.confidence_intervals).transpose().to_string(header=False))
            confidence_str += f"Confidence intervals:{ci_str}\n"
            confidence_str += f"Alpha:{self.estimator.alpha}\n"
        adequacy_str = ""
        if self.adequacy:
            adequacy_str = str(self.adequacy)
        return base_str + confidence_str + adequacy_str

    def to_dict(self, json=False):
        """Return result contents as a dictionary
        :return: Dictionary containing contents of causal_test_result
        """
        base_dict = {
            "treatment": self.estimator.treatment,
            "control_value": self.estimator.control_value,
            "treatment_value": self.estimator.treatment_value,
            "outcome": self.estimator.outcome,
            "adjustment_set": list(self.adjustment_set) if json else self.adjustment_set,
            "effect_measure": self.test_value.type,
            "effect_estimate": (
                self.test_value.value.to_dict()
                if json and hasattr(self.test_value.value, "to_dict")
                else self.test_value.value
            ),
            "ci_low": self.ci_low().to_dict() if json and hasattr(self.ci_low(), "to_dict") else self.ci_low(),
            "ci_high": self.ci_high().to_dict() if json and hasattr(self.ci_high(), "to_dict") else self.ci_high(),
        }
        if self.adequacy:
            base_dict["adequacy"] = self.adequacy.to_dict()
        return base_dict

    def ci_low(self):
        """Return the lower bracket of the confidence intervals."""
        if self.confidence_intervals:
            if isinstance(self.confidence_intervals[0], pd.Series):
                return self.confidence_intervals[0].to_list()
            return self.confidence_intervals[0]
        return None

    def ci_high(self):
        """Return the higher bracket of the confidence intervals."""
        if self.confidence_intervals:
            if isinstance(self.confidence_intervals[1], pd.Series):
                return self.confidence_intervals[1].to_list()
            return self.confidence_intervals[1]
        return None

    def ci_valid(self) -> bool:
        """Return whether or not the result has valid confidence invervals"""
        return self.ci_low() and (not pd.isnull(self.ci_low())) and self.ci_high() and (not pd.isnull(self.ci_high()))

    def summary(self):
        """Summarise the causal test result as an intuitive sentence."""
        print(
            f"The causal effect of changing {self.estimator.treatment} = {self.estimator.control_value} to "
            f"{self.estimator.treatment}' = {self.estimator.treatment_value} is {self.test_value.value}"
            f"(95% confidence intervals: {self.confidence_intervals})."
        )
