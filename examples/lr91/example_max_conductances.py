import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.variable import Input, Output
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_test_outcome import Positive, Negative, NoEffect
from causal_testing.estimation.linear_regression_estimator import LinearRegressionEstimator
from causal_testing.testing.base_test_case import BaseTestCase
from matplotlib.pyplot import rcParams

import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(message)s")

# Uncommenting the code below will make all graphs publication quality but requires a suitable latex installation

# rc_fonts = {
#     "font.size": 8,
#     "figure.figsize": (5, 4),
#     "text.usetex": True,
#     "font.family": "serif",
#     "text.latex.preamble": r"\usepackage{libertine}",
# }
# rcParams.update(rc_fonts)
ROOT = os.path.realpath(os.path.dirname(__file__))
OBSERVATIONAL_DATA_PATH = f"{ROOT}/data/normalised_results.csv"


def test_sensitivity_analysis():
    """Perform causal testing to evaluate the effect of six conductance inputs on one output, APD90, over the defined
    (normalised) design distribution to quantify the extent to which each input affects the output, and plot as a
    graph.
    """
    # Read in the 200 model runs and define mean value and expected effect
    model_runs = pd.read_csv(f"{ROOT}/data/results.csv")
    conductance_means = {
        "G_K": (0.5, Positive),
        "G_b": (0.5, Positive),
        "G_K1": (0.5, Positive),
        "G_si": (0.5, Negative),
        "G_Na": (0.5, NoEffect),
        "G_Kp": (0.5, NoEffect),
    }

    # Normalise the inputs as per the original study
    normalised_df = normalise_data(model_runs, columns=list(conductance_means.keys()))
    normalised_df.to_csv(f"{ROOT}/data/normalised_results.csv")

    # For each input, perform 10 causal tests that change the input from its mean value (0.5) to the equidistant values
    # [0, 0.1, 0.2, ..., 0.9, 1] over the input space of each input, as defined by the normalised design distribution.
    # For each input, this will yield 10 causal test results that measure the extent the input causes APD90 to change,
    # enabling us to compare the magnitude and direction of each inputs' effect.
    treatment_values = np.linspace(0, 1, 11)
    results = {"G_K": {}, "G_b": {}, "G_K1": {}, "G_si": {}, "G_Na": {}, "G_Kp": {}}
    for conductance_param, mean_and_oracle in conductance_means.items():
        average_treatment_effects = []
        confidence_intervals = []

        # Perform each causal test for the given input
        for treatment_value in treatment_values:
            mean, oracle = mean_and_oracle
            conductance_input = Input(conductance_param, float)
            ate, ci = effects_on_APD90(OBSERVATIONAL_DATA_PATH, conductance_input, 0.5, treatment_value, oracle)

            # Store results
            average_treatment_effects.append(ate)
            confidence_intervals.append(ci)
        results[conductance_param] = {"ate": average_treatment_effects, "cis": confidence_intervals}
    plot_ates_with_cis(results, treatment_values)


def effects_on_APD90(observational_data_path, treatment_var, control_val, treatment_val, expected_causal_effect):
    """Perform causal testing for the scenario in which we investigate the causal effect of a given input on APD90.

    :param observational_data_path: Path to observational data containing previous executions of the LR91 model.
    :param treatment_var: The input variable whose effect on APD90 we are interested in.
    :param control_val: The control value for the treatment variable (before intervention).
    :param treatment_val: The treatment value for the treatment variable (after intervention).
    :param expected_causal_effect: The expected causal effect (Positive, Negative, No Effect).
    :return: ATE for the effect of G_K on APD90
    """
    # 1. Define Causal DAG
    causal_dag = CausalDAG(f"{ROOT}/dag.dot")

    # 2. Specify all inputs
    g_na = Input("G_Na", float)
    g_si = Input("G_si", float)
    g_k = Input("G_K", float)
    g_k1 = Input("G_K1", float)
    g_kp = Input("G_Kp", float)
    g_b = Input("G_b", float)

    # 3. Specify all outputs
    max_voltage = Output("max_voltage", float)
    rest_voltage = Output("rest_voltage", float)
    max_voltage_gradient = Output("max_voltage_gradient", float)
    dome_voltage = Output("dome_voltage", float)
    apd50 = Output("APD50", int)
    apd90 = Output("APD90", int)

    # 4. Create scenario by applying constraints over a subset of the inputs
    scenario = Scenario(
        variables={
            g_na,
            g_si,
            g_k,
            g_k1,
            g_kp,
            g_b,
            max_voltage,
            rest_voltage,
            max_voltage_gradient,
            dome_voltage,
            apd50,
            apd90,
        },
        constraints=set(),
    )

    # 5. Create a causal specification from the scenario and causal DAG
    causal_specification = CausalSpecification(scenario, causal_dag)
    base_test_case = BaseTestCase(treatment_var, apd90)
    # 6. Create a causal test case
    causal_test_case = CausalTestCase(
        base_test_case=base_test_case,
        expected_causal_effect=expected_causal_effect,
        control_value=control_val,
        treatment_value=treatment_val,
    )

    # 8. Obtain the minimal adjustment set from the causal DAG
    minimal_adjustment_set = causal_dag.identification(base_test_case)
    linear_regression_estimator = LinearRegressionEstimator(
        treatment_var.name,
        treatment_val,
        control_val,
        minimal_adjustment_set,
        "APD90",
        df=pd.read_csv(observational_data_path),
    )

    # 9. Run the causal test and print results
    causal_test_result = causal_test_case.execute_test(linear_regression_estimator)
    logger.info("%s", causal_test_result)
    return causal_test_result.test_value.value, causal_test_result.confidence_intervals


def plot_ates_with_cis(results_dict: dict, xs: list, save: bool = False, show: bool = False):
    """Plot the average treatment effects for a given treatment against a list of x-values with confidence intervals.

    :param results_dict: A dictionary containing results for sensitivity analysis of each input parameter.
    :param xs: Values to be plotted on the x-axis.
    :param save: Whether to save the plot.
    """
    fig, axes = plt.subplots()
    input_colors = {"G_Na": "red", "G_si": "green", "G_K": "blue", "G_K1": "magenta", "G_Kp": "cyan", "G_b": "yellow"}
    for treatment, test_results in results_dict.items():
        ates = test_results["ate"]
        cis = test_results["cis"]
        before_underscore, after_underscore = treatment.split("_")
        after_underscore_braces = f"{{{after_underscore}}}"
        latex_compatible_treatment_str = rf"${before_underscore}_{after_underscore_braces}$"
        cis_low = [c[0][0] for c in cis]
        cis_high = [c[1][0] for c in cis]
        axes.fill_between(
            xs, cis_low, cis_high, alpha=0.2, color=input_colors[treatment], label=latex_compatible_treatment_str
        )
        axes.plot(xs, ates, color=input_colors[treatment], linewidth=1)
        axes.plot(xs, [0] * len(xs), color="black", alpha=0.5, linestyle="--", linewidth=1)
    axes.set_ylabel(r"ATE: Change in $APD_{90} (ms)$")
    axes.set_xlabel(r"Treatment value")
    axes.set_ylim(-80, 80)
    axes.set_xlim(min(xs), max(xs))
    box = axes.get_position()
    axes.set_position([box.x0, box.y0 + box.height * 0.3, box.width * 0.85, box.height * 0.7])
    plt.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fancybox=True, ncol=1, title=r"Input (95\% CIs)")
    if save:
        plt.savefig(f"APD90_sensitivity.pdf", format="pdf")
    if show:
        plt.show()


def normalise_data(df, columns=None):
    """Normalise the data in the dataframe into the range [0, 1]."""
    if columns:
        df[columns] = (df[columns] - df[columns].min()) / (df[columns].max() - df[columns].min())
        return df
    else:
        return (df - df.min()) / (df.max() - df.min())


if __name__ == "__main__":
    test_sensitivity_analysis()
