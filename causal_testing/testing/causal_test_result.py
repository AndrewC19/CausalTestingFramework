"""This module contains the CausalTestResult class, which is a container for the results of a causal test, and the
TestValue dataclass.
"""
from typing import Any
from dataclasses import dataclass
import pandas as pd
import json

from causal_testing.testing.estimators import Estimator
from causal_testing.specification.variable import Variable


@dataclass
class TestValue:
    """A dataclass to hold both the type and value of a causal test result"""

    type: str
    value: float


class CausalTestResult:
    """A container to hold the results of a causal test case. Every causal test case provides a point estimate of
    the ATE, given a particular treatment, outcome, and adjustment set. Some but not all estimators can provide
    confidence intervals."""

    def __init__(
        self,
        estimator: Estimator,
        test_value: TestValue,
        confidence_intervals: [float, float] = None,
        effect_modifier_configuration: {Variable: Any} = None,
    ):
        self.estimator = estimator
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
        )
        if self.estimator.control_value:
            base_str += f"Control value: {self.estimator.control_value.value}\n"
        if self.estimator.treatment_value:
            base_str += f"Treatment value: {self.estimator.treatment_value.value}\n"
        base_str += (
            f"Outcome: {self.estimator.outcome}\n"
            f"Adjustment set: {self.adjustment_set}\n"
            f"{self.test_value.type}: {result_str}\n"
        )
        confidence_str = ""
        if self.confidence_intervals:
            ci_str = " " + str(self.confidence_intervals)
            if "\n" in ci_str:
                ci_str = " " + push(pd.DataFrame(self.confidence_intervals).transpose().to_string(header=False))
            confidence_str += f"Confidence intervals:{ci_str}\n"
        return base_str + confidence_str

    def to_dict(self):
        """Return result contents as a dictionary
        :return: Dictionary containing contents of causal_test_result
        """
        base_dict = {
            "treatment": self.estimator.treatment,
            "outcome": self.estimator.outcome,
            "adjustment_set": sorted(list(self.adjustment_set)),
            }
        if self.estimator.control_value:
            base_dict["control_value"] = self.estimator.control_value.value
        if self.estimator.treatment_value:
            base_dict["treatment_value"] = self.estimator.treatment_value.value
        if self.ci_low() is not None:
            base_dict["ci_low"] = self.ci_low()
        if self.ci_high() is not None:
            base_dict["ci_high"] = self.ci_high()
        return base_dict

    def ci_low(self):
        """Return the lower bracket of the confidence intervals."""
        if self.confidence_intervals:
            return self.confidence_intervals[0]
        return None

    def ci_high(self):
        """Return the higher bracket of the confidence intervals."""
        if self.confidence_intervals:
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
