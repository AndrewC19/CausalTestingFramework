"""
This module contains the ShouldCause and ShouldNotCause metamorphic relations as
defined in our ICST paper [https://eprints.whiterose.ac.uk/195317/].
"""

from dataclasses import dataclass
from abc import abstractmethod
from typing import Iterable
from itertools import combinations
import argparse
import logging
import json
import networkx as nx
import pandas as pd
import numpy as np

from causal_testing.specification.causal_specification import CausalDAG, Node
from causal_testing.data_collection.data_collector import ExperimentalDataCollector

logger = logging.getLogger(__name__)


@dataclass(order=True)
class MetamorphicRelation:
    """Class representing a metamorphic relation."""

    treatment_var: Node
    output_var: Node
    adjustment_vars: Iterable[Node]
    dag: CausalDAG
    tests: Iterable = None

    def generate_follow_up(self, n_tests: int, min_val: float, max_val: float, seed: int = 0):
        """Generate numerical follow-up input configurations."""
        np.random.seed(seed)

        # Get set of variables to change, excluding the treatment itself
        variables_to_change = {node for node in self.dag.graph.nodes if self.dag.graph.in_degree(node) == 0}
        if self.adjustment_vars:
            variables_to_change |= set(self.adjustment_vars)
        if self.treatment_var in variables_to_change:
            variables_to_change.remove(self.treatment_var)

        # Assign random numerical values to the variables to change
        test_inputs = pd.DataFrame(
            np.random.randint(min_val, max_val, size=(n_tests, len(variables_to_change))),
            columns=sorted(variables_to_change),
        )

        # Enumerate the possible source, follow-up pairs for the treatment
        candidate_source_follow_up_pairs = np.array(list(combinations(range(int(min_val), int(max_val + 1)), 2)))

        # Sample without replacement from the possible source, follow-up pairs
        sampled_source_follow_up_indices = np.random.choice(
            candidate_source_follow_up_pairs.shape[0], n_tests, replace=False
        )

        follow_up_input = f"{self.treatment_var}'"
        source_follow_up_test_inputs = pd.DataFrame(
            candidate_source_follow_up_pairs[sampled_source_follow_up_indices],
            columns=sorted([self.treatment_var] + [follow_up_input]),
        )
        self.tests = [
            MetamorphicTest(
                source_inputs,
                follow_up_inputs,
                other_inputs,
                self.output_var,
                str(self),
            )
            for source_inputs, follow_up_inputs, other_inputs in zip(
                source_follow_up_test_inputs[[self.treatment_var]].to_dict(orient="records"),
                source_follow_up_test_inputs[[follow_up_input]]
                .rename(columns={follow_up_input: self.treatment_var})
                .to_dict(orient="records"),
                test_inputs.to_dict(orient="records")
                if not test_inputs.empty
                else [{}] * len(source_follow_up_test_inputs),
            )
        ]

    def execute_tests(self, data_collector: ExperimentalDataCollector):
        """Execute the generated list of metamorphic tests, returning a dictionary of tests that pass and fail.

        :param data_collector: An experimental data collector for the system-under-test.
        """
        test_results = {"pass": [], "fail": []}
        for metamorphic_test in self.tests:
            # Update the control and treatment configuration to take generated values for source and follow-up tests
            control_input_config = metamorphic_test.source_inputs | metamorphic_test.other_inputs
            treatment_input_config = metamorphic_test.follow_up_inputs | metamorphic_test.other_inputs
            data_collector.control_input_configuration = control_input_config
            data_collector.treatment_input_configuration = treatment_input_config
            metamorphic_test_results_df = data_collector.collect_data()

            # Apply assertion to control and treatment outputs
            control_output = metamorphic_test_results_df.loc["control_0"][metamorphic_test.output]
            treatment_output = metamorphic_test_results_df.loc["treatment_0"][metamorphic_test.output]

            if not self.assertion(control_output, treatment_output):
                test_results["fail"].append(metamorphic_test)
            else:
                test_results["pass"].append(metamorphic_test)
        return test_results

    @abstractmethod
    def assertion(self, source_output, follow_up_output):
        """An assertion that should be applied to an individual metamorphic test run."""

    @abstractmethod
    def to_json_stub(self, skip=True) -> dict:
        """Convert to a JSON frontend stub string for user customisation"""

    @abstractmethod
    def test_oracle(self, test_results):
        """A test oracle that assert whether the MR holds or not based on ALL test results.

        This method must raise an assertion, not return a bool."""

    def __eq__(self, other):
        same_type = self.__class__ == other.__class__
        same_treatment = self.treatment_var == other.treatment_var
        same_output = self.output_var == other.output_var
        same_adjustment_set = set(self.adjustment_vars) == set(other.adjustment_vars)
        return same_type and same_treatment and same_output and same_adjustment_set


class ShouldCause(MetamorphicRelation):
    """Class representing a should cause metamorphic relation."""

    def assertion(self, source_output, follow_up_output):
        """If there is a causal effect, the outputs should not be the same."""
        return source_output != follow_up_output

    def test_oracle(self, test_results):
        """A single passing test is sufficient to show presence of a causal effect."""
        assert len(test_results["fail"]) < len(
            self.tests
        ), f"{str(self)}: {len(test_results['fail'])}/{len(self.tests)} tests failed."

    def to_json_stub(self, skip=True) -> dict:
        """Convert to a JSON frontend stub string for user customisation"""
        return {
            "name": str(self),
            "estimator": "LinearRegressionEstimator",
            "estimate_type": "coefficient",
            "effect": "direct",
            "mutations": [self.treatment_var],
            "expected_effect": {self.output_var: "SomeEffect"},
            "formula": f"{self.output_var} ~ {' + '.join([self.treatment_var] + self.adjustment_vars)}",
            "skip": skip,
        }

    def __str__(self):
        formatted_str = f"{self.treatment_var} --> {self.output_var}"
        if self.adjustment_vars:
            formatted_str += f" | {self.adjustment_vars}"
        return formatted_str


class ShouldNotCause(MetamorphicRelation):
    """Class representing a should cause metamorphic relation."""

    def assertion(self, source_output, follow_up_output):
        """If there is a causal effect, the outputs should not be the same."""
        return source_output == follow_up_output

    def test_oracle(self, test_results):
        """A single passing test is sufficient to show presence of a causal effect."""
        assert (
            len(test_results["fail"]) == 0
        ), f"{str(self)}: {len(test_results['fail'])}/{len(self.tests)} tests failed."

    def to_json_stub(self, skip=True) -> dict:
        """Convert to a JSON frontend stub string for user customisation"""
        return {
            "name": str(self),
            "estimator": "LinearRegressionEstimator",
            "estimate_type": "coefficient",
            "effect": "direct",
            "mutations": [self.treatment_var],
            "expected_effect": {self.output_var: "NoEffect"},
            "formula": f"{self.output_var} ~ {' + '.join([self.treatment_var] + self.adjustment_vars)}",
            "skip": skip,
        }

    def __str__(self):
        formatted_str = f"{self.treatment_var} _||_ {self.output_var}"
        if self.adjustment_vars:
            formatted_str += f" | {self.adjustment_vars}"
        return formatted_str


@dataclass(order=True)
class MetamorphicTest:
    """Class representing a metamorphic test case."""

    source_inputs: dict
    follow_up_inputs: dict
    other_inputs: dict
    output: str
    relation: str

    def __str__(self):
        return (
            f"Source inputs: {self.source_inputs}\n"
            f"Follow-up inputs: {self.follow_up_inputs}\n"
            f"Other inputs: {self.other_inputs}\n"
            f"Output: {self.output}"
            f"Metamorphic Relation: {self.relation}"
        )


def generate_metamorphic_relations(dag: CausalDAG) -> list[MetamorphicRelation]:
    """Construct a list of metamorphic relations implied by the Causal DAG.

    This list of metamorphic relations contains a ShouldCause relation for every edge, and a ShouldNotCause
    relation for every (minimal) conditional independence relation implied by the structure of the DAG.

    :param CausalDAG dag: Causal DAG from which the metamorphic relations will be generated.
    :return: A list containing ShouldCause and ShouldNotCause metamorphic relations.
    """
    metamorphic_relations = []
    for node_pair in combinations(dag.graph.nodes, 2):
        (u, v) = node_pair

        # Create a ShouldNotCause relation for each pair of nodes that are not directly connected
        if ((u, v) not in dag.graph.edges) and ((v, u) not in dag.graph.edges):
            # Case 1: U --> ... --> V
            if u in nx.ancestors(dag.graph, v):
                adj_set = list(dag.direct_effect_adjustment_sets([u], [v])[0])
                metamorphic_relations.append(ShouldNotCause(u, v, adj_set, dag))

            # Case 2: V --> ... --> U
            elif v in nx.ancestors(dag.graph, u):
                adj_set = list(dag.direct_effect_adjustment_sets([v], [u])[0])
                metamorphic_relations.append(ShouldNotCause(v, u, adj_set, dag))

            # Case 3: V _||_ U (No directed walk from V to U but there may be a back-door path e.g. U <-- Z --> V).
            # Only make one MR since V _||_ U == U _||_ V
            else:
                adj_set = list(dag.direct_effect_adjustment_sets([u], [v])[0])
                metamorphic_relations.append(ShouldNotCause(u, v, adj_set, dag))

        # Create a ShouldCause relation for each edge (u, v) or (v, u)
        elif (u, v) in dag.graph.edges:
            adj_set = list(dag.direct_effect_adjustment_sets([u], [v])[0])
            metamorphic_relations.append(ShouldCause(u, v, adj_set, dag))
        else:
            adj_set = list(dag.direct_effect_adjustment_sets([v], [u])[0])
            metamorphic_relations.append(ShouldCause(v, u, adj_set, dag))

    return metamorphic_relations


if __name__ == "__main__": # pragma: no cover
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="A script for generating metamorphic relations to test the causal relationships in a given DAG."
    )
    parser.add_argument(
        "--dag_path",
        "-d",
        help="Specify path to file containing the DAG, normally a .dot file.",
        required=True,
    )
    parser.add_argument(
        "--output_path",
        "-o",
        help="Specify path where tests should be saved, normally a .json file.",
        required=True,
    )
    args = parser.parse_args()

    causal_dag = CausalDAG(args.dag_path)
    relations = generate_metamorphic_relations(causal_dag)
    tests = [
        relation.to_json_stub(skip=False)
        for relation in relations
        if len(list(causal_dag.graph.predecessors(relation.output_var))) > 0
    ]

    logger.info(f"Generated {len(tests)} tests. Saving to {args.output_path}.")
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump({"tests": tests}, f, indent=2)
