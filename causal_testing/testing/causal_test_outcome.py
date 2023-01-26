from abc import ABC, abstractmethod
from causal_testing.testing.causal_test_result import CausalTestResult
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CausalTestOutcome(ABC):
    """An abstract class representing an expected causal effect."""

    @abstractmethod
    def apply(self, res: CausalTestResult) -> bool:
        pass

    def __str__(self) -> str:
        return type(self).__name__


class ExactValue(CausalTestOutcome):
    """An extension of TestOutcome representing that the expected causal effect should be a specific value."""

    def __init__(self, value: float, tolerance: float = None):
        self.value = value
        if tolerance is None:
            self.tolerance = value * 0.05
        else:
            self.tolerance = tolerance

    def apply(self, res: CausalTestResult) -> bool:
        return np.isclose(res.test_value.value, self.value, atol=self.tolerance)

    def __str__(self):
        return f"ExactValue: {self.value}±{self.tolerance}"


class Positive(CausalTestOutcome):
    """An extension of TestOutcome representing that the expected causal effect should be positive."""

    def apply(self, res: CausalTestResult) -> bool:
        # TODO: confidence intervals?
        if res.test_value.type == "ate":
            return res.test_value.value > 0
        elif res.test_value.type == "risk_ratio":
            return res.test_value.value > 1


class Negative(CausalTestOutcome):
    """An extension of TestOutcome representing that the expected causal effect should be negative."""

    def apply(self, res: CausalTestResult) -> bool:
        # TODO: confidence intervals?
        if res.test_value.type == "ate":
            return res.test_value.value < 0
        elif res.test_value.type == "risk_ratio":
            return res.test_value.value < 1


class SomeEffect(CausalTestOutcome):
    """An extension of TestOutcome representing that the expected causal effect should not be zero."""

    def apply(self, res: CausalTestResult) -> bool:
        if res.test_value.type == "ate":
            return (0 < res.ci_low() < res.ci_high()) or (res.ci_low() < res.ci_high() < 0)
        elif res.test_value.type == "risk_ratio":
            return (1 < res.ci_low() < res.ci_high()) or (res.ci_low() < res.ci_high() < 1)

    def __str__(self):
        return "Changed"


class NoEffect(CausalTestOutcome):
    """An extension of TestOutcome representing that the expected causal effect should be zero."""

    def apply(self, res: CausalTestResult) -> bool:
        if res.test_value.type == "ate":
            if res.ci_low() and res.ci_high():
                return (res.ci_low() < 0 < res.ci_high()) or (abs(res.test_value.value) < 1e-10)
            else:
                logger.warning("No confidence intervals for estimate. Using ATE alone.")
                return abs(res.test_value.value) < 1e-10
        elif res.test_value.type == "risk_ratio":
            return (res.ci_low() < 1 < res.ci_high()) or np.isclose(res.test_value.value, 1.0, atol=1e-10)

    def __str__(self):
        return "Unchanged"
