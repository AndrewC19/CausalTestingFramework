"""This module contains the Estimator abstract class, as well as its concrete extensions: LogisticRegressionEstimator,
LinearRegressionEstimator and CausalForestEstimator"""
import logging
from abc import ABC, abstractmethod
from typing import Any
from math import ceil

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from econml.dml import CausalForestDML
from patsy import dmatrix

from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from causal_testing.specification.variable import Variable

from itertools import product

logger = logging.getLogger(__name__)


class Estimator(ABC):
    # pylint: disable=too-many-instance-attributes
    """An estimator contains all of the information necessary to compute a causal estimate for the effect of changing
    a set of treatment variables to a set of values.

    All estimators must implement the following two methods:

    1) add_modelling_assumptions: The validity of a model-assisted causal inference result depends on whether
    the modelling assumptions imposed by a model actually hold. Therefore, for each model, is important to state
    the modelling assumption upon which the validity of the results depend. To achieve this, the estimator object
    maintains a list of modelling assumptions (as strings). If a user wishes to implement their own estimator, they
    must implement this method and add all assumptions to the list of modelling assumptions.

    2) estimate_ate: All estimators must be capable of returning the average treatment effect as a minimum. That is, the
    average effect of the intervention (changing treatment from control to treated value) on the outcome of interest
    adjusted for all confounders.
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
        effect_modifiers: dict[str:Any] = None,
    ):
        self.treatment = treatment
        self.treatment_value = treatment_value
        self.control_value = control_value
        self.adjustment_set = adjustment_set
        self.outcome = outcome
        self.df = df
        if effect_modifiers is None:
            self.effect_modifiers = {}
        elif isinstance(effect_modifiers, dict):
            self.effect_modifiers = effect_modifiers
        else:
            raise ValueError(f"Unsupported type for effect_modifiers {effect_modifiers}. Expected iterable")
        self.modelling_assumptions = []
        self.add_modelling_assumptions()
        logger.debug("Effect Modifiers: %s", self.effect_modifiers)

    @abstractmethod
    def add_modelling_assumptions(self):
        """
        Add modelling assumptions to the estimator. This is a list of strings which list the modelling assumptions that
        must hold if the resulting causal inference is to be considered valid.
        """

    def compute_confidence_intervals(self) -> list[float, float]:
        """
        Estimate the 95% Wald confidence intervals for the effect of changing the treatment from control values to
        treatment values on the outcome.
        :return: 95% Wald confidence intervals.
        """


class LogisticRegressionEstimator(Estimator):
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
        adjustment_set: set,
        outcome: str,
        df: pd.DataFrame = None,
        effect_modifiers: dict[str:Any] = None,
        formula: str = None,
    ):
        super().__init__(treatment, treatment_value, control_value, adjustment_set, outcome, df, effect_modifiers)

        self.model = None

        if formula is not None:
            self.formula = formula
        else:
            terms = [treatment] + sorted(list(adjustment_set)) + sorted(list(self.effect_modifiers))
            self.formula = f"{outcome} ~ {'+'.join(((terms)))}"

    def add_modelling_assumptions(self):
        """
        Add modelling assumptions to the estimator. This is a list of strings which list the modelling assumptions that
        must hold if the resulting causal inference is to be considered valid.
        """
        self.modelling_assumptions += (
            "The variables in the data must fit a shape which can be expressed as a linear"
            "combination of parameters and functions of variables. Note that these functions"
            "do not need to be linear."
        )
        self.modelling_assumptions += "The outcome must be binary."
        self.modelling_assumptions += "Independently and identically distributed errors."

    def _run_logistic_regression(self, data) -> RegressionResultsWrapper:
        """Run logistic regression of the treatment and adjustment set against the outcome and return the model.

        :return: The model after fitting to data.
        """
        # 1. Reduce dataframe to contain only the necessary columns
        reduced_df = data.copy()
        necessary_cols = [self.treatment] + list(self.adjustment_set) + [self.outcome]
        missing_rows = reduced_df[necessary_cols].isnull().any(axis=1)
        reduced_df = reduced_df[~missing_rows]
        reduced_df = reduced_df.sort_values([self.treatment])
        logger.debug(reduced_df[necessary_cols])

        # 2. Add intercept
        reduced_df["Intercept"] = 1  # self.intercept

        # 3. Estimate the unit difference in outcome caused by unit difference in treatment
        cols = [self.treatment]
        cols += [x for x in self.adjustment_set if x not in cols]
        treatment_and_adjustments_cols = reduced_df[cols + ["Intercept"]]
        for col in treatment_and_adjustments_cols:
            if str(treatment_and_adjustments_cols.dtypes[col]) == "object":
                treatment_and_adjustments_cols = pd.get_dummies(
                    treatment_and_adjustments_cols, columns=[col], drop_first=True
                )
        model = smf.logit(formula=self.formula, data=data).fit(disp=0)
        return model

    def estimate(self, data: pd.DataFrame, adjustment_config: dict = None) -> RegressionResultsWrapper:
        """add terms to the dataframe and estimate the outcome from the data
        :param data: A pandas dataframe containing execution data from the system-under-test.
        :param adjustment_config: Dictionary containing the adjustment configuration of the adjustment set
        """
        if adjustment_config is None:
            adjustment_config = {}
        if set(self.adjustment_set) != set(adjustment_config):
            raise ValueError(
                f"Invalid adjustment configuration {adjustment_config}. Must specify values for {self.adjustment_set}"
            )

        model = self._run_logistic_regression(data)
        self.model = model

        x = pd.DataFrame(columns=self.df.columns)
        x["Intercept"] = 1  # self.intercept
        x[self.treatment] = [self.treatment_value, self.control_value]
        for k, v in adjustment_config.items():
            x[k] = v
        for k, v in self.effect_modifiers.items():
            x[k] = v
        x = dmatrix(self.formula.split("~")[1], x, return_type="dataframe")
        for col in x:
            if str(x.dtypes[col]) == "object":
                x = pd.get_dummies(x, columns=[col], drop_first=True)
        # x = x[model.params.index]
        return model.predict(x)

    def estimate_control_treatment(self, bootstrap_size=100, adjustment_config=None) -> tuple[pd.Series, pd.Series]:
        """Estimate the outcomes under control and treatment.

        :return: The estimated control and treatment values and their confidence
        intervals in the form ((ci_low, control, ci_high), (ci_low, treatment, ci_high)).
        """

        y = self.estimate(self.df, adjustment_config=adjustment_config)

        try:
            bootstrap_samples = [
                self.estimate(self.df.sample(len(self.df), replace=True), adjustment_config=adjustment_config)
                for _ in range(bootstrap_size)
            ]
            control, treatment = zip(*[(x.iloc[1], x.iloc[0]) for x in bootstrap_samples])
        except PerfectSeparationError:
            logger.warning(
                "Perfect separation detected, results not available. Cannot calculate confidence intervals for such "
                "a small dataset."
            )
            return (y.iloc[1], None), (y.iloc[0], None)
        except np.linalg.LinAlgError:
            logger.warning("Singular matrix detected. Confidence intervals not available. Try with a larger data set")
            return (y.iloc[1], None), (y.iloc[0], None)

        # Delta method confidence intervals from
        # https://stackoverflow.com/questions/47414842/confidence-interval-of-probability-prediction-from-logistic-regression-statsmode
        # cov = model.cov_params()
        # gradient = (y * (1 - y) * x.T).T  # matrix of gradients for each observation
        # std_errors = np.array([np.sqrt(np.dot(np.dot(g, cov), g)) for g in gradient.to_numpy()])
        # c = 1.96  # multiplier for confidence interval
        # upper = np.maximum(0, np.minimum(1, y + std_errors * c))
        # lower = np.maximum(0, np.minimum(1, y - std_errors * c))

        return (y.iloc[1], np.array(control)), (y.iloc[0], np.array(treatment))

    def estimate_ate(self, bootstrap_size=100, adjustment_config=None) -> float:
        """Estimate the ate effect of the treatment on the outcome. That is, the change in outcome caused
        by changing the treatment variable from the control value to the treatment value. Here, we actually
        calculate the expected outcomes under control and treatment and take one away from the other. This
        allows for custom terms to be put in such as squares, inverses, products, etc.

        :return: The estimated average treatment effect and 95% confidence intervals
        """
        (control_outcome, control_bootstraps), (
            treatment_outcome,
            treatment_bootstraps,
        ) = self.estimate_control_treatment(bootstrap_size=bootstrap_size, adjustment_config=adjustment_config)
        estimate = treatment_outcome - control_outcome

        if control_bootstraps is None or treatment_bootstraps is None:
            return estimate, (None, None)

        bootstraps = sorted(list(treatment_bootstraps - control_bootstraps))
        bound = int((bootstrap_size * 0.05) / 2)
        ci_low = bootstraps[bound]
        ci_high = bootstraps[bootstrap_size - bound]

        logger.info(
            f"Changing {self.treatment} from {self.control_value} to {self.treatment_value} gives an estimated "
            f"ATE of {ci_low} < {estimate} < {ci_high}"
        )
        assert ci_low < estimate < ci_high, f"Expecting {ci_low} < {estimate} < {ci_high}"

        return estimate, (ci_low, ci_high)

    def estimate_risk_ratio(self, bootstrap_size=100, adjustment_config=None) -> float:
        """Estimate the ate effect of the treatment on the outcome. That is, the change in outcome caused
        by changing the treatment variable from the control value to the treatment value. Here, we actually
        calculate the expected outcomes under control and treatment and divide one by the other. This
        allows for custom terms to be put in such as squares, inverses, products, etc.

        :return: The estimated risk ratio and 95% confidence intervals.
        """
        (control_outcome, control_bootstraps), (
            treatment_outcome,
            treatment_bootstraps,
        ) = self.estimate_control_treatment(bootstrap_size=bootstrap_size, adjustment_config=adjustment_config)
        estimate = treatment_outcome / control_outcome

        if control_bootstraps is None or treatment_bootstraps is None:
            return estimate, (None, None)

        bootstraps = sorted(list(treatment_bootstraps / control_bootstraps))
        bound = ceil((bootstrap_size * 0.05) / 2)
        ci_low = bootstraps[bound]
        ci_high = bootstraps[bootstrap_size - bound]

        logger.info(
            f"Changing {self.treatment} from {self.control_value} to {self.treatment_value} gives an estimated "
            f"risk ratio of {ci_low} < {estimate} < {ci_high}"
        )
        assert ci_low < estimate < ci_high, f"Expecting {ci_low} < {estimate} < {ci_high}"

        return estimate, (ci_low, ci_high)

    def estimate_unit_odds_ratio(self) -> float:
        """Estimate the odds ratio of increasing the treatment by one. In logistic regression, this corresponds to the
        coefficient of the treatment of interest.

        :return: The odds ratio. Confidence intervals are not yet supported.
        """
        model = self._run_logistic_regression(self.df)
        return np.exp(model.params[self.treatment])


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
    ):
        super().__init__(treatment, treatment_value, control_value, adjustment_set, outcome, df, effect_modifiers)

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

    def add_modelling_assumptions(self):
        """
        Add modelling assumptions to the estimator. This is a list of strings which list the modelling assumptions that
        must hold if the resulting causal inference is to be considered valid.
        """
        self.modelling_assumptions += (
            "The variables in the data must fit a shape which can be expressed as a linear"
            "combination of parameters and functions of variables. Note that these functions"
            "do not need to be linear."
        )

    def estimate_unit_cate(self) -> float:
        """Estimate the unit average treatment effect of the treatment on the outcome. That is, the change in outcome
        caused by a unit change in treatment.

        :return: The unit average treatment effect and the 95% Wald confidence intervals.
        """
        categoricals = set(self.df.select_dtypes(object).columns).intersection(self.adjustment_set)
        categorical_values = [sorted(list(set(self.df[c]))) for c in categoricals]
        unit_effect = []
        ci_low = []
        ci_high = []
        for combination in product(*categorical_values):
            queries = [f"{c} == '{v}'" for c, v in zip(categoricals, combination)]
            df = self.df.query(" and ".join(queries))
            model = self._run_linear_regression(df)
            treatment = [self.treatment]
            if str(self.df.dtypes[self.treatment]) == "object":
                design_info = dmatrix(self.formula.split("~")[1], self.df).design_info
                treatment = design_info.column_names[design_info.term_name_slices[self.treatment]]
            newline = "\n"
            assert set(treatment).issubset(
                model.params.index.tolist()
            ), f"{treatment} not in\n{'  '+str(model.params.index).replace(newline, newline+'  ')}"
            unit_effect_ = model.params[treatment]  # Unit effect is the coefficient of the treatment
            [ci_low_, ci_high_] = self._get_confidence_intervals(model, treatment)
            if str(self.df.dtypes[self.treatment]) != "object":
                unit_effect.append(unit_effect_[0])
                ci_low.append(ci_low_[0])
                ci_high.append(ci_high_[0])
            else:
                unit_effect += unit_effect_
                ci_low += ci_low_
                ci_high += ci_high_

        return pd.Series(unit_effect, index=categorical_values), [ pd.Series(ci_low, index=categorical_values),  pd.Series(ci_high, index=categorical_values)]

    def estimate_unit_ate(self) -> float:
        """Estimate the unit average treatment effect of the treatment on the outcome. That is, the change in outcome
        caused by a unit change in treatment.

        :return: The unit average treatment effect and the 95% Wald confidence intervals.
        """
        model = self._run_linear_regression()
        newline = "\n"
        treatment = [self.treatment]
        if str(self.df.dtypes[self.treatment]) == "object":
            design_info = dmatrix(self.formula.split("~")[1], self.df).design_info
            treatment = design_info.column_names[design_info.term_name_slices[self.treatment]]
        assert set(treatment).issubset(
            model.params.index.tolist()
        ), f"{treatment} not in\n{'  '+str(model.params.index).replace(newline, newline+'  ')}"
        unit_effect = model.params[treatment]  # Unit effect is the coefficient of the treatment
        [ci_low, ci_high] = self._get_confidence_intervals(model, treatment)
        if str(self.df.dtypes[self.treatment]) != "object":
            unit_effect = unit_effect[0]
            ci_low = ci_low[0]
            ci_high = ci_high[0]
        return unit_effect, [ci_low, ci_high]

    def estimate_ate(self) -> tuple[float, list[float, float], float]:
        """Estimate the average treatment effect of the treatment on the outcome. That is, the change in outcome caused
        by changing the treatment variable from the control value to the treatment value.

        :return: The average treatment effect and the 95% Wald confidence intervals.
        """
        model = self._run_linear_regression()
        self.model = model

        # Create an empty individual for the control and treated
        individuals = pd.DataFrame(1, index=["control", "treated"], columns=model.params.index)

        # It is ABSOLUTELY CRITICAL that these go last, otherwise we can't index
        # the effect with "ate = t_test_results.effect[0]"
        individuals.loc["control", [self.treatment]] = self.control_value
        individuals.loc["treated", [self.treatment]] = self.treatment_value

        # Perform a t-test to compare the predicted outcome of the control and treated individual (ATE)
        t_test_results = model.t_test(individuals.loc["treated"] - individuals.loc["control"])
        ate = t_test_results.effect[0]
        confidence_intervals = list(t_test_results.conf_int().flatten())
        return ate, confidence_intervals

    def estimate_control_treatment(self, adjustment_config: dict = None) -> tuple[pd.Series, pd.Series]:
        """Estimate the outcomes under control and treatment.

        :return: The estimated outcome under control and treatment in the form
        (control_outcome, treatment_outcome).
        """
        if adjustment_config is None:
            adjustment_config = {}

        model = self._run_linear_regression()
        self.model = model

        x = pd.DataFrame(columns=self.df.columns)
        x[self.treatment] = [self.treatment_value, self.control_value]
        x["Intercept"] = 1  # self.intercept
        for k, v in adjustment_config.items():
            x[k] = v
        for k, v in self.effect_modifiers.items():
            x[k] = v
        x = dmatrix(self.formula.split("~")[1], x, return_type="dataframe")
        for col in x:
            if str(x.dtypes[col]) == "object":
                x = pd.get_dummies(x, columns=[col], drop_first=True)
        x = x[model.params.index]
        y = model.get_prediction(x).summary_frame()

        return y.iloc[1], y.iloc[0]

    def estimate_risk_ratio(self) -> tuple[float, list[float, float]]:
        """Estimate the risk_ratio effect of the treatment on the outcome. That is, the change in outcome caused
        by changing the treatment variable from the control value to the treatment value.

        :return: The average treatment effect and the 95% Wald confidence intervals.
        """
        control_outcome, treatment_outcome = self.estimate_control_treatment()
        ci_low = treatment_outcome["mean_ci_lower"] / control_outcome["mean_ci_upper"]
        ci_high = treatment_outcome["mean_ci_upper"] / control_outcome["mean_ci_lower"]

        return (treatment_outcome["mean"] / control_outcome["mean"]), [ci_low, ci_high]

    def estimate_ate_calculated(self, adjustment_config: dict = None) -> tuple[float, list[float, float]]:
        """Estimate the ate effect of the treatment on the outcome. That is, the change in outcome caused
        by changing the treatment variable from the control value to the treatment value. Here, we actually
        calculate the expected outcomes under control and treatment and divide one by the other. This
        allows for custom terms to be put in such as squares, inverses, products, etc.

        :return: The average treatment effect and the 95% Wald confidence intervals.
        """
        control_outcome, treatment_outcome = self.estimate_control_treatment(adjustment_config=adjustment_config)
        ci_low = treatment_outcome["mean_ci_lower"] - control_outcome["mean_ci_upper"]
        ci_high = treatment_outcome["mean_ci_upper"] - control_outcome["mean_ci_lower"]

        return (treatment_outcome["mean"] - control_outcome["mean"]), [ci_low, ci_high]

    def _run_linear_regression(self, df=None) -> RegressionResultsWrapper:
        """Run linear regression of the treatment and adjustment set against the outcome and return the model.

        :return: The model after fitting to data.
        """
        if df is None:
            df = self.df
        cols = [self.treatment]
        cols += [x for x in self.adjustment_set if x not in cols]
        model = smf.ols(formula=self.formula, data=df).fit()
        return model

    def _get_confidence_intervals(self, model, treatment):
        confidence_intervals = model.conf_int(alpha=0.05, cols=None)
        ci_low, ci_high = (
            confidence_intervals[0].loc[treatment],
            confidence_intervals[1].loc[treatment],
        )
        return [ci_low, ci_high]


class InstrumentalVariableEstimator(Estimator):
    """
    Carry out estimation using instrumental variable adjustment rather than conventional adjustment. This means we do
    not need to observe all confounders in order to adjust for them. A key assumption here is linearity.
    """

    def __init__(
        # pylint: disable=too-many-arguments
        self,
        treatment: str,
        treatment_value: float,
        control_value: float,
        adjustment_set: set,
        outcome: str,
        instrument: str,
        df: pd.DataFrame = None,
        intercept: int = 1,
        effect_modifiers: dict = None,  # Not used (yet?). Needed for compatibility
    ):
        super().__init__(treatment, treatment_value, control_value, adjustment_set, outcome, df, None)
        self.intercept = intercept
        self.model = None
        self.instrument = instrument

    def add_modelling_assumptions(self):
        """
        Add modelling assumptions to the estimator. This is a list of strings which list the modelling assumptions that
        must hold if the resulting causal inference is to be considered valid.
        """
        self.modelling_assumptions += """The instrument and the treatment, and the treatment and the outcome must be
        related linearly in the form Y = aX + b."""
        self.modelling_assumptions += """The three IV conditions must hold
            (i) Instrument is associated with treatment
            (ii) Instrument does not affect outcome except through its potential effect on treatment
            (iii) Instrument and outcome do not share causes
        """

    def estimate_coefficient(self, df):
        """
        Estimate the linear regression coefficient of the treatment on the
        outcome.
        """
        # Estimate the total effect of instrument I on outcome Y = abI + c1
        ab = sm.OLS(df[self.outcome], df[[self.instrument]]).fit().params[self.instrument]

        # Estimate the direct effect of instrument I on treatment X = aI + c1
        a = sm.OLS(df[self.treatment], df[[self.instrument]]).fit().params[self.instrument]

        # Estimate the coefficient of I on X by cancelling
        return ab / a

    def estimate_unit_ate(self, bootstrap_size=100):
        """
        Estimate the unit ate (i.e. coefficient) of the treatment on the
        outcome.
        """
        bootstraps = sorted(
            [self.estimate_coefficient(self.df.sample(len(self.df), replace=True)) for _ in range(bootstrap_size)]
        )
        bound = ceil((bootstrap_size * 0.05) / 2)
        ci_low = bootstraps[bound]
        ci_high = bootstraps[bootstrap_size - bound]

        return self.estimate_coefficient(self.df), (ci_low, ci_high)


class CausalForestEstimator(Estimator):
    """A causal random forest estimator is a non-parametric estimator which recursively partitions the covariate space
    to learn a low-dimensional representation of treatment effect heterogeneity. This form of estimator is best suited
    to the estimation of heterogeneous treatment effects i.e. the estimated effect for every sample rather than the
    population average.
    """

    def add_modelling_assumptions(self):
        """Add any modelling assumptions to the estimator.

        :return self: Update self.modelling_assumptions
        """
        self.modelling_assumptions += "Non-parametric estimator: no restrictions imposed on the data."

    def estimate_ate(self) -> float:
        """Estimate the average treatment effect.

        :return ate, confidence_intervals: The average treatment effect and 95% confidence intervals.
        """
        # Remove any NA containing rows
        reduced_df = self.df.copy()
        necessary_cols = [self.treatment] + list(self.adjustment_set) + [self.outcome]
        missing_rows = reduced_df[necessary_cols].isnull().any(axis=1)
        reduced_df = reduced_df[~missing_rows]

        # Split data into effect modifiers (X), confounders (W), treatments (T), and outcome (Y)
        if self.effect_modifiers:
            effect_modifier_df = reduced_df[list(self.effect_modifiers)]
        else:
            effect_modifier_df = reduced_df[list(self.adjustment_set)]
        confounders_df = reduced_df[list(self.adjustment_set)]
        treatment_df = np.ravel(reduced_df[[self.treatment]])
        outcome_df = np.ravel(reduced_df[[self.outcome]])

        # Fit the model to the data using a gradient boosting regressor for both the treatment and outcome model
        model = CausalForestDML(
            model_y=GradientBoostingRegressor(),
            model_t=GradientBoostingRegressor(),
        )
        model.fit(outcome_df, treatment_df, X=effect_modifier_df, W=confounders_df)

        # Obtain the ATE and 95% confidence intervals
        ate = model.ate(effect_modifier_df, T0=self.control_value, T1=self.treatment_value)
        ate_interval = model.ate_interval(effect_modifier_df, T0=self.control_value, T1=self.treatment_value)
        ci_low, ci_high = ate_interval[0], ate_interval[1]
        return ate, [ci_low, ci_high]

    def estimate_cates(self) -> pd.DataFrame:
        """Estimate the conditional average treatment effect for each sample in the data as a function of a set of
        covariates (X) i.e. effect modifiers. That is, the predicted change in outcome caused by the intervention
        (change in treatment from control to treatment value) for every execution of the system-under-test, taking into
        account the value of each effect modifier X. As a result, for every unique setting of the set of covariates X,
        we expect a different CATE.

        :return results_df: A dataframe containing a conditional average treatment effect, 95% confidence intervals, and
        the covariate (effect modifier) values for each sample.
        """

        # Remove any NA containing rows
        reduced_df = self.df.copy()
        necessary_cols = [self.treatment] + list(self.adjustment_set) + [self.outcome]
        missing_rows = reduced_df[necessary_cols].isnull().any(axis=1)
        reduced_df = reduced_df[~missing_rows]

        # Split data into effect modifiers (X), confounders (W), treatments (T), and outcome (Y)
        if self.effect_modifiers:
            effect_modifier_df = reduced_df[list(self.effect_modifiers)]
        else:
            raise ValueError("CATE requires the user to define a set of effect modifiers.")

        if self.adjustment_set:
            confounders_df = reduced_df[list(self.adjustment_set)]
        else:
            confounders_df = None
        treatment_df = reduced_df[[self.treatment]]
        outcome_df = reduced_df[[self.outcome]]

        # Fit a model to the data
        model = CausalForestDML(model_y=GradientBoostingRegressor(), model_t=GradientBoostingRegressor())
        model.fit(outcome_df, treatment_df, X=effect_modifier_df, W=confounders_df)

        # Obtain CATES and confidence intervals
        conditional_ates = model.effect(effect_modifier_df, T0=self.control_value, T1=self.treatment_value).flatten()
        [ci_low, ci_high] = model.effect_interval(
            effect_modifier_df, T0=self.control_value, T1=self.treatment_value, alpha=0.05
        )

        # Merge results into a dataframe (CATE, confidence intervals, and effect modifier values)
        results_df = pd.DataFrame(columns=["cate", "ci_low", "ci_high"])
        results_df["cate"] = list(conditional_ates)
        results_df["ci_low"] = list(ci_low.flatten())
        results_df["ci_high"] = list(ci_high.flatten())
        effect_modifier_df.reset_index(drop=True, inplace=True)
        results_df[list(self.effect_modifiers)] = effect_modifier_df
        results_df.sort_values(by=list(self.effect_modifiers), inplace=True)
        return results_df, None
