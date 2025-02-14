"""Module containing classes to define and run causal surrogate assisted test cases"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable
import pandas as pd
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.testing.base_test_case import BaseTestCase
from causal_testing.estimation.cubic_spline_estimator import CubicSplineRegressionEstimator


@dataclass
class SimulationResult:
    """Data class holding the data and result metadata of a simulation"""

    data: dict
    fault: bool
    relationship: str

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the simulation result data to a pandas DataFrame"""
        data_as_lists = {k: v if isinstance(v, list) else [v] for k, v in self.data.items()}
        return pd.DataFrame(data_as_lists)


class SearchAlgorithm(ABC):  # pylint: disable=too-few-public-methods
    """Class to be inherited with the search algorithm consisting of a search function and the fitness function of the
    space to be searched"""

    @abstractmethod
    def search(
        self, surrogate_models: list[CubicSplineRegressionEstimator], specification: CausalSpecification
    ) -> list:
        """Function which implements a search routine which searches for the optimal fitness value for the specified
        scenario
        :param surrogate_models: The surrogate models to be searched
        :param specification:  The Causal Specification (combination of Scenario and Causal Dag)"""


class Simulator(ABC):
    """Class to be inherited with Simulator specific functions to start, shutdown and run the simulation with the give
    config file"""

    @abstractmethod
    def startup(self, **kwargs):
        """Function that when run, initialises and opens the Simulator"""

    @abstractmethod
    def shutdown(self, **kwargs):
        """Function to safely exit and shutdown the Simulator"""

    @abstractmethod
    def run_with_config(self, configuration: dict) -> SimulationResult:
        """Run the simulator with the given configuration and return the results in the structure of a
        SimulationResult
        :param configuration: The configuration required to initialise the Simulation
        :return: Simulation results in the structure of the SimulationResult data class"""


class CausalSurrogateAssistedTestCase:
    """A class representing a single causal surrogate assisted test case."""

    def __init__(
        self,
        specification: CausalSpecification,
        search_algorithm: SearchAlgorithm,
        simulator: Simulator,
    ):
        self.specification = specification
        self.search_algorithm = search_algorithm
        self.simulator = simulator

    def execute(
        self,
        data: pd.DataFrame,
        max_executions: int = 200,
        custom_data_aggregator: Callable[[dict, dict], dict] = None,
    ):
        """For this specific test case, a search algorithm is used to find the most contradictory point in the input
        space which is, therefore, most likely to indicate incorrect behaviour. This candidate test case is run against
        the simulator, checked for faults and the result returned with collected data
        :param data: DataFrame containing the initial data for analysis
        :param max_executions: Maximum number of simulator executions before exiting the search
        :param custom_data_aggregator: Optional function to customize how new data is aggregated with existing data
        :return: tuple containing SimulationResult or str, execution number and collected data"""
        collected_data = data.copy()

        for i in range(max_executions):
            surrogate_models = self.generate_surrogates(self.specification, collected_data)
            candidate_test_case, _, surrogate = self.search_algorithm.search(surrogate_models, self.specification)

            self.simulator.startup()
            test_result = self.simulator.run_with_config(candidate_test_case)
            test_result_df = test_result.to_dataframe()
            self.simulator.shutdown()

            if custom_data_aggregator is not None:
                collected_data = pd.DataFrame(custom_data_aggregator(collected_data.to_dict(), test_result.data))
            else:
                collected_data = pd.concat([collected_data, test_result_df], ignore_index=True)

            if test_result.fault:
                print(
                    f"Fault found between {surrogate.treatment} causing {surrogate.outcome}. Contradiction with "
                    f"expected {surrogate.expected_relationship}."
                )
                test_result.relationship = (
                    f"{surrogate.treatment} -> {surrogate.outcome} expected {surrogate.expected_relationship}"
                )
                return test_result, i + 1, collected_data

        print("No fault found")
        return "No fault found", i + 1, collected_data

    def generate_surrogates(
        self, specification: CausalSpecification, data: pd.DataFrame
    ) -> list[CubicSplineRegressionEstimator]:
        """Generate a surrogate model for each edge of the dag that specifies it is included in the DAG metadata.
        :param specification: The Causal Specification (combination of Scenario and Causal Dag)
        :param data: DataFrame containing the data for generating surrogate models
        :return: A list of surrogate models
        """
        surrogate_models = []

        for u, v in specification.causal_dag.graph.edges:
            edge_metadata = specification.causal_dag.graph.adj[u][v]
            if "included" in edge_metadata:
                from_var = specification.scenario.variables.get(u)
                to_var = specification.scenario.variables.get(v)
                base_test_case = BaseTestCase(from_var, to_var)

                minimal_adjustment_set = specification.causal_dag.identification(base_test_case, specification.scenario)

                surrogate = CubicSplineRegressionEstimator(
                    u,
                    0,
                    0,
                    minimal_adjustment_set,
                    v,
                    4,
                    df=data,
                    expected_relationship=edge_metadata["expected"],
                )
                surrogate_models.append(surrogate)

        return surrogate_models