import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob

from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.variable import Input, Output
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.data_collection.data_collector import ObservationalDataCollector
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_test_outcome import Positive
from causal_testing.testing.intervention import Intervention
from causal_testing.testing.causal_test_engine import CausalTestEngine
from causal_testing.testing.estimators import LinearRegressionEstimator, CausalForestEstimator


OBSERVATIONAL_DATA_PATH = f"./data/testing_with_different_rel_beta.csv"


def concatenate_csvs_in_directory(directory_path, output_path):
    """ Concatenate all csvs in a given directory, assuming all csvs share the same header. This will stack the csvs
    vertically and will not reset the index.
    """
    dfs = []
    for csv_name in glob.glob(directory_path):
        dfs.append(pd.read_csv(csv_name, index_col=0))
    full_df = pd.concat(dfs, ignore_index=True)
    full_df.to_csv(output_path)


def manual_CATE(observational_data_path):
    """ Compute the CATE for the effect of doubling beta across simulations with different age demographics.
    To compute the CATE, this method splits the observational data into high and low age data and computes the
    ATE using this data and a linear regression model.

    Since this method already adjusts for age, adding age as
    an adjustment to the LR model will have no impact. However, adding contacts as an adjustment should reduce
    bias and reveal the average causal effect of doubling beta in simulations of a particular age demographic. """
    past_execution_df = pd.read_csv(observational_data_path)

    # Find middle age
    min_age = np.floor(past_execution_df['avg_age'].min())
    max_age = np.ceil(past_execution_df['avg_age'].max())
    mid_age = (min_age + max_age) / 2

    # Split df into two age ranges
    younger_population_df = past_execution_df.loc[past_execution_df['avg_age'] <= mid_age]
    younger_population_df.to_csv("./data/bessemer/younger_population.csv")
    older_population_df = past_execution_df.loc[past_execution_df['avg_age'] > mid_age]
    older_population_df.to_csv("./data/bessemer/older_population.csv")

    # Apply causal forest model to both datasets separately
    separated_observational_data_paths = ["./data/bessemer/younger_population.csv",
                                          "./data/bessemer/older_population.csv"]

    for separated_observational_data_path in separated_observational_data_paths:
        separated_observational_df = pd.read_csv(separated_observational_data_path)
        separated_observational_df_avg_age = round(separated_observational_df["avg_age"].mean(), 1)
        _, causal_test_engine = identification(separated_observational_data_path)

        linear_regression_estimator = LinearRegressionEstimator(('beta',), 0.032, 0.016,
                                                                ('avg_age', 'contacts',),
                                                                ('cum_infections',))
        no_adjustment_linear_regression_estimator = LinearRegressionEstimator(('beta',), 0.032, 0.016,
                                                                              set(),
                                                                              ('cum_infections',))

        causal_test_result = causal_test_engine.execute_test(linear_regression_estimator, 'ate')
        association_test_result = causal_test_engine.execute_test(no_adjustment_linear_regression_estimator, 'ate')

        causal_ate = causal_test_result.ate
        percentage_causal_ate = round((causal_ate / separated_observational_df['cum_infections'].mean())*100, 3)
        causal_ate_cis = [causal_test_result.ci_low(), causal_test_result.ci_high()]
        percentage_causal_ate_cis = [str(round(((ci/separated_observational_df['cum_infections'].mean())*100), 3)) + '%'
                                     for ci in causal_ate_cis]

        non_causal_ate = association_test_result.ate
        percentage_non_causal_ate = round((non_causal_ate / separated_observational_df['cum_infections'].mean())*100, 3)
        non_causal_ate_cis = [association_test_result.ci_low(), association_test_result.ci_high()]
        percentage_non_causal_ate_cis = [str(round(((ci/separated_observational_df['cum_infections'].mean())*100), 3)) +
                                         '%' for ci in non_causal_ate_cis]

        print("==================================================")
        print(f"Condition: Executions with an average age of {separated_observational_df_avg_age}")
        print("==================================================")
        print(f"Causal Results: \n"
              f"ATE: {percentage_causal_ate}\n"
              f"CIs: {percentage_causal_ate_cis}")
        print("==================================================")
        print(f"Non-causal Results: \n"
              f"ATE: {percentage_non_causal_ate}\n"
              f"CIs: {percentage_non_causal_ate_cis}")
        print("==================================================")
        print("==================================================")


def identification(observational_data_path):
    # 1. Read in the Causal DAG
    causal_dag = CausalDAG('./dag.dot')

    # 2. Create variables
    pop_size = Input('pop_size', int)
    pop_infected = Input('pop_infected', int)
    n_days = Input('n_days', int)
    cum_infections = Output('cum_infections', int)
    cum_deaths = Output('cum_deaths', int)
    location = Input('location', str)
    variants = Input('variants', str)
    avg_age = Input('avg_age', float)
    beta = Input('beta', float)
    contacts = Input('contacts', float)

    # 3. Create scenario by applying constraints over a subset of the input variables
    scenario = Scenario(variables={pop_size, pop_infected, n_days, cum_infections, cum_deaths,
                                   location, variants, avg_age, beta, contacts},
                        constraints={pop_size.z3 == 51633, pop_infected.z3 == 1000, n_days.z3 == 216})

    # 4. Construct a causal specification from the scenario and causal DAG
    causal_specification = CausalSpecification(scenario, causal_dag)

    # 5. Create a causal test case
    causal_test_case = CausalTestCase(control_input_configuration={beta: 0.016},
                                      expected_causal_effect=Positive,
                                      outcome_variables={cum_infections},
                                      intervention=Intervention((beta,), (0.032,), ), )

    # 6. Create a data collector
    data_collector = ObservationalDataCollector(scenario, observational_data_path)

    # 7. Create an instance of the causal test engine
    causal_test_engine = CausalTestEngine(causal_test_case, causal_specification, data_collector)

    # 8. Obtain the minimal adjustment set for the causal test case from the causal DAG
    minimal_adjustment_set = causal_test_engine.load_data(index_col=0)

    return minimal_adjustment_set, causal_test_engine


def causal_forest_CATE(observational_data_path):
    _, causal_test_engine = identification(observational_data_path)
    causal_forest_estimator = CausalForestEstimator(('beta',), 0.032, 0.016, {'avg_age', 'contacts'},
                                                    ('cum_infections',), effect_modifiers=('avg_age',))
    causal_forest_estimator_no_adjustment = CausalForestEstimator(('beta',), 0.032, 0.016, set(),
                                                                  ('cum_infections',), effect_modifiers=('avg_age',))

    # 10. Execute the test case and compare the results
    causal_test_result = causal_test_engine.execute_test(causal_forest_estimator, 'cate')
    association_test_result = causal_test_engine.execute_test(causal_forest_estimator_no_adjustment, 'cate')
    observational_data = pd.read_csv(observational_data_path)
    plot_causal_forest_result(causal_test_result, observational_data, "Causal Forest Adjusted for Age and Contacts.")
    plot_causal_forest_result(association_test_result, observational_data, "Causal Forest Without Adjustment.")


def plot_causal_forest_result(causal_forest_test_result, previous_data_df, title=None, filter_data_by_variant=False):
    sorted_causal_forest_test_result = causal_forest_test_result.sort_index()
    no_avg_age_causal_forest_test_result = sorted_causal_forest_test_result.drop(columns='avg_age')
    observational_df_with_results = previous_data_df.join(no_avg_age_causal_forest_test_result)
    observational_df_with_results['percentage_increase'] = \
        (observational_df_with_results['cate'] / observational_df_with_results['cum_infections']) * 100
    fig, ax = plt.subplots()
    if filter_data_by_variant:
        observational_df_with_results = observational_df_with_results.loc[observational_df_with_results['variants']
                                                                          == 'beta']
    for location in observational_df_with_results.location.unique():
        location_variant_df = observational_df_with_results.loc[observational_df_with_results['location'] == location]
        xs = location_variant_df['avg_age']
        ys = location_variant_df['percentage_increase']
        ax.scatter(xs, ys, s=1, alpha=.3, label=location)
        ax.set_ylabel("% change in cumulative infections")
        ax.set_xlabel("Average age")
        ax.set_title(title)
    ax.set_ylim(0, 40)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, ncol=4)
    plt.show()


if __name__ == "__main__":
    # manual_CATE(OBSERVATIONAL_DATA_PATH)
    causal_forest_CATE(OBSERVATIONAL_DATA_PATH)
