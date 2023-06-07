# pylint: disable=too-few-public-methods
"""This module contains the CausalTestOutcome abstract class, as well as the concrete extension classes:
ExactValue, Positive, Negative, SomeEffect, NoEffect"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
import numpy as np

from causal_testing.testing.causal_test_result import CausalTestResult


class CausalTestOutcome(ABC):
    """An abstract class representing an expected causal effect."""

    @abstractmethod
    def apply(self, res: CausalTestResult) -> bool:
        """Abstract apply method that should return a bool representing if the result meets the outcome
        :param res: CausalTestResult to be checked
        :return: Bool that is true if outcome is met
        """

    def __str__(self) -> str:
        return type(self).__name__


class SomeEffect(CausalTestOutcome):
    """An extension of TestOutcome representing that the expected causal effect should not be zero."""

    def apply(self, res: CausalTestResult) -> bool:
        if res.test_value.type == "ate":
            return (0 < res.ci_low() < res.ci_high()) or (res.ci_low() < res.ci_high() < 0)
        if "coefficient" in res.test_value.type:
            ci_low = res.ci_low() if isinstance(res.ci_low(), Iterable) else [res.ci_low()]
            ci_high = res.ci_high() if isinstance(res.ci_high(), Iterable) else [res.ci_high()]
            return any(0 < ci_low < ci_high or ci_low < ci_high < 0 for ci_low, ci_high in zip(ci_low, ci_high))
        if res.test_value.type == "risk_ratio":
            return (1 < res.ci_low() < res.ci_high()) or (res.ci_low() < res.ci_high() < 1)
        raise ValueError(f"Test Value type {res.test_value.type} is not valid for this TestOutcome")


class NoEffect(CausalTestOutcome):
    """An extension of TestOutcome representing that the expected causal effect should be zero."""

    def __init__(self, atol: float = 1e-10):
        self.atol = atol

    def apply(self, res: CausalTestResult) -> bool:
        if res.test_value.type == "ate":
            return (res.ci_low() < 0 < res.ci_high()) or (abs(res.test_value.value) < self.atol)
        if "coefficient" in res.test_value.type:
            ci_low = res.ci_low() if isinstance(res.ci_low(), Iterable) else [res.ci_low()]
            ci_high = res.ci_high() if isinstance(res.ci_high(), Iterable) else [res.ci_high()]
            value = res.test_value.value if isinstance(res.ci_high(), Iterable) else [res.test_value.value]
            return all(ci_low < 0 < ci_high for ci_low, ci_high in zip(ci_low, ci_high)) or all(
                abs(v) < self.atol for v in value
            )
        if res.test_value.type == "risk_ratio":
            return (res.ci_low() < 1 < res.ci_high()) or np.isclose(res.test_value.value, 1.0, atol=self.atol)
        raise ValueError(f"Test Value type {res.test_value.type} is not valid for this TestOutcome")


class ExactValue(SomeEffect):
    """An extension of TestOutcome representing that the expected causal effect should be a specific value."""

    def __init__(self, value: float, atol: float = None):
        self.value = value
        if atol is None:
            self.atol = value * 0.05
        else:
            self.atol = atol

    def apply(self, res: CausalTestResult) -> bool:
        if res.ci_valid():
            return super().apply(res) and np.isclose(res.test_value.value, self.value, atol=self.atol)
        return np.isclose(res.test_value.value, self.value, atol=self.atol)

    def __str__(self):
        return f"ExactValue: {self.value}±{self.atol}"


class Positive(SomeEffect):
    """An extension of TestOutcome representing that the expected causal effect should be positive."""

    def apply(self, res: CausalTestResult) -> bool:
        if res.ci_valid() and not super().apply(res):
            return False
        if res.test_value.type in {"ate", "coefficient"}:
            return res.test_value.value > 0
        if res.test_value.type == "risk_ratio":
            return res.test_value.value > 1
        # Dead code but necessary for pylint
        raise ValueError(f"Test Value type {res.test_value.type} is not valid for this TestOutcome")


class Negative(SomeEffect):
    """An extension of TestOutcome representing that the expected causal effect should be negative."""

    def apply(self, res: CausalTestResult) -> bool:
        if res.ci_valid() and not super().apply(res):
            return False
        if res.test_value.type in {"ate", "coefficient"}:
            return res.test_value.value < 0
        if res.test_value.type == "risk_ratio":
            return res.test_value.value < 1
        # Dead code but necessary for pylint
        raise ValueError(f"Test Value type {res.test_value.type} is not valid for this TestOutcome")
