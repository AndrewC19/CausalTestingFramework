import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.variable import Input, Output
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.data_collection.data_collector import ObservationalDataCollector
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_test_outcome import Positive, Negative, NoEffect
from causal_testing.testing.causal_test_engine import CausalTestEngine
from causal_testing.testing.estimators import LinearRegressionEstimator
from causal_testing.testing.base_test_case import BaseTestCase
from causal_testing.testing.causal_test_suite import CausalTestSuite
from matplotlib.pyplot import rcParams

# Uncommenting the code below will make all graphs publication quality but requires a suitable latex installation

# rc_fonts = {
#     "font.size": 8,
#     "figure.figsize": (5, 4),
#     "text.usetex": True,
#     "font.family": "serif",
#     "text.latex.preamble": r"\usepackage{libertine}",
# }
# rcParams.update(rc_fonts)
OBSERVATIONAL_DATA_PATH = "./data/normalised_results.csv"


def causal_testing_sensitivity_analysis():
    """Perform causal testing to evaluate the effect of six conductance inputs on one output, APD90, over the defined
       (normalised) design distribution to quantify the extent to which each input affects the output, and plot as a
       graph.
    """
    # Read in the 200 model runs and define mean value and expected effect
    model_runs = pd.read_csv("data/results.csv")
    conductance_means = {'G_K': (.5, Positive),
                         'G_b': (.5, Positive),
                         'G_K1': (.5, Positive),
                         'G_si': (.5, Negative),
                         'G_Na': (.5, NoEffect),
                         'G_Kp': (.5, NoEffect)}

    # Normalise the inputs as per the original study
    normalised_df = normalise_data(model_runs, columns=list(conductance_means.keys()))
    normalised_df.to_csv("data/normalised_results.csv")

    # For each input, perform 10 causal tests that change the input from its mean value (0.5) to the equidistant values
    # [0, 0.1, 0.2, ..., 0.9, 1] over the input space of each input, as defined by the normalised design distribution.
    # For each input, this will yield 10 causal test results that measure the extent the input causes APD90 to change,
    # enabling us to compare the magnitude and direction of each inputs' effect.
    treatment_values = np.linspace(0, 1, 11)
    results = {'G_K': {},
               'G_b': {},
               'G_K1': {},
               'G_si': {},
               'G_Na': {},
               'G_Kp': {}}

    apd90 = Output('APD90', int)
    outcome_variable = apd90
    test_suite = CausalTestSuite()

    for conductance_param, mean_and_oracle in conductance_means.items():
        treatment_variable = Input(conductance_param, float)
        base_test_case = BaseTestCase(treatment_variable, outcome_variable)
        test_list = []
        control_value = 0.5
        mean, oracle = mean_and_oracle
        for treatment_value in treatment_values:
            test_list.append(CausalTestCase(base_test_case, oracle, control_value, treatment_value))
        test_suite.add_test_object(base_test_case, test_list, [LinearRegressionEstimator], 'ate')

    causal_test_results = effects_on_APD90(OBSERVATIONAL_DATA_PATH, test_suite)

    # Extract data from causal_test_results needed for plotting
    for base_test_case in causal_test_results:
        results[base_test_case.treatment_variable.name] = \
            {"ate": [result.ate for result in causal_test_results[base_test_case][0]],
             "cis": [result.confidence_intervals for result in
                     causal_test_results[base_test_case][0]]}

    plot_ates_with_cis(results, treatment_values)


def effects_on_APD90(observational_data_path, test_suite):
    """Perform causal testing for the scenario in which we investigate the causal effect of a given input on APD90.

    :param: test_suite: A CausalTestSuite object containing a dictionary of base_test_cases and the treatment/outcome
                        values to be tested
    :return: causal_test_results containing a list of causal_test_result objects
    """
    # 1. Define Causal DAG
    causal_dag = CausalDAG('./dag.dot')

    # 2. Specify all inputs
    g_na = Input('G_Na', float)
    g_si = Input('G_si', float)
    g_k = Input('G_K', float)
    g_k1 = Input('G_K1', float)
    g_kp = Input('G_Kp', float)
    g_b = Input('G_b', float)

    # 3. Specify all outputs
    max_voltage = Output('max_voltage', float)
    rest_voltage = Output('rest_voltage', float)
    max_voltage_gradient = Output('max_voltage_gradient', float)
    dome_voltage = Output('dome_voltage', float)
    apd50 = Output('APD50', int)
    apd90 = Output('APD90', int)

    # 4. Create scenario by applying constraints over a subset of the inputs
    scenario = Scenario(
        variables={g_na, g_si, g_k, g_k1, g_kp, g_b,
                   max_voltage, rest_voltage, max_voltage_gradient, dome_voltage, apd50, apd90},
        constraints=set()
    )

    # 5. Create a causal specification from the scenario and causal DAG
    causal_specification = CausalSpecification(scenario, causal_dag)

    # 7. Create a data collector
    data_collector = ObservationalDataCollector(scenario, observational_data_path)

    # 8. Create an instance of the causal test engine
    causal_test_engine = CausalTestEngine(causal_specification, data_collector)

    # 9. Run the causal test suite
    causal_test_results = causal_test_engine.execute_test_suite(test_suite)
    return causal_test_results


def plot_ates_with_cis(results_dict: dict, xs: list, save: bool = True):
    """Plot the average treatment effects for a given treatment against a list of x-values with confidence intervals.

    :param results_dict: A dictionary containing results for sensitivity analysis of each input parameter.
    :param xs: Values to be plotted on the x-axis.
    :param save: Whether to save the plot.
    """
    fig, axes = plt.subplots()
    input_colors = {'G_Na': 'red',
                    'G_si': 'green',
                    'G_K': 'blue',
                    'G_K1': 'magenta',
                    'G_Kp': 'cyan',
                    'G_b': 'yellow'}
    for treatment, test_results in results_dict.items():
        ates = test_results['ate']
        cis = test_results['cis']
        before_underscore, after_underscore = treatment.split('_')
        after_underscore_braces = f"{{{after_underscore}}}"
        latex_compatible_treatment_str = rf"${before_underscore}_{after_underscore_braces}$"
        cis_low = [c[0] for c in cis]
        cis_high = [c[1] for c in cis]
        axes.fill_between(xs, cis_low, cis_high, alpha=.2, color=input_colors[treatment],
                          label=latex_compatible_treatment_str)
        axes.plot(xs, ates, color=input_colors[treatment], linewidth=1)
        axes.plot(xs, [0] * len(xs), color='black', alpha=.5, linestyle='--', linewidth=1)
    axes.set_ylabel(r"ATE: Change in $APD_{90} (ms)$")
    axes.set_xlabel(r"Treatment value")
    axes.set_ylim(-80, 80)
    axes.set_xlim(min(xs), max(xs))
    box = axes.get_position()
    axes.set_position([box.x0, box.y0 + box.height * 0.3,
                       box.width * 0.85, box.height * 0.7])
    plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fancybox=True, ncol=1,
               title=r'Input (95\% CIs)')
    if save:
        plt.savefig(f"APD90_sensitivity.pdf", format="pdf")
    plt.show()


def normalise_data(df, columns=None):
    """ Normalise the data in the dataframe into the range [0, 1]. """
    if columns:
        df[columns] = (df[columns] - df[columns].min()) / (df[columns].max() - df[columns].min())
        return df
    else:
        return (df - df.min()) / (df.max() - df.min())


if __name__ == '__main__':
    causal_testing_sensitivity_analysis()
