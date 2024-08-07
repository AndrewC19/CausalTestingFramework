"""This module contains the LinearRegressionEstimator for estimating continuous outcomes."""

import logging
from typing import Any

import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrix  # pylint: disable = no-name-in-module
from patsy import ModelDesc
from statsmodels.regression.linear_model import RegressionResultsWrapper

from causal_testing.specification.variable import Variable
from causal_testing.estimation.gp import GP
from causal_testing.estimation.estimator import Estimator

logger = logging.getLogger(__name__)


class LinearRegressionEstimator(Estimator):
    """A Linear Regression Estimator is a parametric estimator which restricts the variables in the data to a linear
    combination of parameters and functions of the variables (note these functions need not be linear).
    """

    def __init__(
        # pylint: disable=too-many-arguments
        self,
        treatment: str,
        treatment_value: float,
        control_value: float,
        adjustment_set: set,
        outcome: str,
        df: pd.DataFrame = None,
        effect_modifiers: dict[Variable:Any] = None,
        formula: str = None,
        alpha: float = 0.05,
        query: str = "",
    ):
        super().__init__(
            treatment,
            treatment_value,
            control_value,
            adjustment_set,
            outcome,
            df,
            effect_modifiers,
            alpha=alpha,
            query=query,
        )

        self.model = None
        if effect_modifiers is None:
            effect_modifiers = []

        if formula is not None:
            self.formula = formula
        else:
            terms = [treatment] + sorted(list(adjustment_set)) + sorted(list(effect_modifiers))
            self.formula = f"{outcome} ~ {'+'.join(terms)}"

        for term in self.effect_modifiers:
            self.adjustment_set.add(term)

    def gp_formula(
        self,
        ngen: int = 100,
        mu: int = 20,
        lambda_: int = 10,
        extra_operators: list = None,
        sympy_conversions: dict = None,
        seeds: list = None,
        seed: int = 0,
    ):
        # pylint: disable=too-many-arguments,invalid-name
        """
        Use Genetic Programming (GP) to infer the regression equation from the data.

        :param ngen: The maximum number of GP generations to run for.
        :param mu: The GP population size.
        :param lambda_: The number of offspring per generation.
        :param extra_operators: Additional operators for the GP (defaults are +, *, and 1/x). Operations should be of
                                the form (fun, numArgs), e.g. (add, 2).
        :param sympy_conversions: Dictionary of conversions of extra_operators for sympy,
                                  e.g. `"mul": lambda *args_: "Mul({},{})".format(*args_)`.
        :param seeds: Seed individuals for the population (e.g. if you think that the relationship between X and Y is
                      probably logarithmic, you can put that in).
        :param seed: Random seed for the GP.
        """
        gp = GP(
            df=self.df,
            features=sorted(list(self.adjustment_set.union([self.treatment]))),
            outcome=self.outcome,
            extra_operators=extra_operators,
            sympy_conversions=sympy_conversions,
            seed=seed,
        )
        formula = gp.run_gp(ngen=ngen, pop_size=mu, num_offspring=lambda_, seeds=seeds)
        formula = gp.simplify(formula)
        self.formula = f"{self.outcome} ~ I({formula}) - 1"

    def add_modelling_assumptions(self):
        """
        Add modelling assumptions to the estimator. This is a list of strings which list the modelling assumptions that
        must hold if the resulting causal inference is to be considered valid.
        """
        self.modelling_assumptions.append(
            "The variables in the data must fit a shape which can be expressed as a linear"
            "combination of parameters and functions of variables. Note that these functions"
            "do not need to be linear."
        )

    def estimate_coefficient(self) -> tuple[pd.Series, list[pd.Series, pd.Series]]:
        """Estimate the unit average treatment effect of the treatment on the outcome. That is, the change in outcome
        caused by a unit change in treatment.

        :return: The unit average treatment effect and the 95% Wald confidence intervals.
        """
        model = self._run_linear_regression()
        newline = "\n"
        patsy_md = ModelDesc.from_formula(self.treatment)

        if any(
            (
                self.df.dtypes[factor.name()] == "object"
                for factor in patsy_md.rhs_termlist[1].factors
                # We want to remove this long term as it prevents us from discovering categoricals within I(...) blocks
                if factor.name() in self.df.dtypes
            )
        ):
            design_info = dmatrix(self.formula.split("~")[1], self.df).design_info
            treatment = design_info.column_names[design_info.term_name_slices[self.treatment]]
        else:
            treatment = [self.treatment]
        assert set(treatment).issubset(
            model.params.index.tolist()
        ), f"{treatment} not in\n{'  ' + str(model.params.index).replace(newline, newline + '  ')}"
        unit_effect = model.params[treatment]  # Unit effect is the coefficient of the treatment
        [ci_low, ci_high] = self._get_confidence_intervals(model, treatment)
        return unit_effect, [ci_low, ci_high]

    def estimate_ate(self) -> tuple[pd.Series, list[pd.Series, pd.Series]]:
        """Estimate the average treatment effect of the treatment on the outcome. That is, the change in outcome caused
        by changing the treatment variable from the control value to the treatment value.

        :return: The average treatment effect and the 95% Wald confidence intervals.
        """
        model = self._run_linear_regression()

        # Create an empty individual for the control and treated
        individuals = pd.DataFrame(1, index=["control", "treated"], columns=model.params.index)

        # For Pandas version > 2, we need to explicitly state that the dataframe takes floating-point values
        individuals = individuals.astype(float)

        # It is ABSOLUTELY CRITICAL that these go last, otherwise we can't index
        # the effect with "ate = t_test_results.effect[0]"
        individuals.loc["control", [self.treatment]] = self.control_value
        individuals.loc["treated", [self.treatment]] = self.treatment_value

        # Perform a t-test to compare the predicted outcome of the control and treated individual (ATE)
        t_test_results = model.t_test(individuals.loc["treated"] - individuals.loc["control"])
        ate = pd.Series(t_test_results.effect[0])
        confidence_intervals = list(t_test_results.conf_int(alpha=self.alpha).flatten())
        confidence_intervals = [pd.Series(interval) for interval in confidence_intervals]
        return ate, confidence_intervals

    def estimate_control_treatment(self, adjustment_config: dict = None) -> tuple[pd.Series, pd.Series]:
        """Estimate the outcomes under control and treatment.

        :return: The estimated outcome under control and treatment in the form
        (control_outcome, treatment_outcome).
        """
        if adjustment_config is None:
            adjustment_config = {}
        model = self._run_linear_regression()

        x = pd.DataFrame(columns=self.df.columns)
        x[self.treatment] = [self.treatment_value, self.control_value]
        x["Intercept"] = 1  # self.intercept

        print(x[self.treatment])
        for k, v in adjustment_config.items():
            x[k] = v
        for k, v in self.effect_modifiers.items():
            x[k] = v
        x = dmatrix(self.formula.split("~")[1], x, return_type="dataframe")
        for col in x:
            if str(x.dtypes[col]) == "object":
                x = pd.get_dummies(x, columns=[col], drop_first=True)
        x = x[model.params.index]

        x[self.treatment] = [self.treatment_value, self.control_value]

        y = model.get_prediction(x).summary_frame()

        return y.iloc[1], y.iloc[0]

    def estimate_risk_ratio(self, adjustment_config: dict = None) -> tuple[pd.Series, list[pd.Series, pd.Series]]:
        """Estimate the risk_ratio effect of the treatment on the outcome. That is, the change in outcome caused
        by changing the treatment variable from the control value to the treatment value.

        :return: The average treatment effect and the 95% Wald confidence intervals.
        """
        if adjustment_config is None:
            adjustment_config = {}
        control_outcome, treatment_outcome = self.estimate_control_treatment(adjustment_config=adjustment_config)
        ci_low = pd.Series(treatment_outcome["mean_ci_lower"] / control_outcome["mean_ci_upper"])
        ci_high = pd.Series(treatment_outcome["mean_ci_upper"] / control_outcome["mean_ci_lower"])
        return pd.Series(treatment_outcome["mean"] / control_outcome["mean"]), [ci_low, ci_high]

    def estimate_ate_calculated(self, adjustment_config: dict = None) -> tuple[pd.Series, list[pd.Series, pd.Series]]:
        """Estimate the ate effect of the treatment on the outcome. That is, the change in outcome caused
        by changing the treatment variable from the control value to the treatment value. Here, we actually
        calculate the expected outcomes under control and treatment and divide one by the other. This
        allows for custom terms to be put in such as squares, inverses, products, etc.

        :return: The average treatment effect and the 95% Wald confidence intervals.
        """
        if adjustment_config is None:
            adjustment_config = {}
        control_outcome, treatment_outcome = self.estimate_control_treatment(adjustment_config=adjustment_config)
        ci_low = pd.Series(treatment_outcome["mean_ci_lower"] - control_outcome["mean_ci_upper"])
        ci_high = pd.Series(treatment_outcome["mean_ci_upper"] - control_outcome["mean_ci_lower"])
        return pd.Series(treatment_outcome["mean"] - control_outcome["mean"]), [ci_low, ci_high]

    def _run_linear_regression(self) -> RegressionResultsWrapper:
        """Run linear regression of the treatment and adjustment set against the outcome and return the model.

        :return: The model after fitting to data.
        """
        model = smf.ols(formula=self.formula, data=self.df).fit()
        self.model = model
        return model

    def _get_confidence_intervals(self, model, treatment):
        confidence_intervals = model.conf_int(alpha=self.alpha, cols=None)
        ci_low, ci_high = (
            pd.Series(confidence_intervals[0].loc[treatment]),
            pd.Series(confidence_intervals[1].loc[treatment]),
        )
        return [ci_low, ci_high]
