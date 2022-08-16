import networkx as nx
import logging
from itertools import combinations
from random import sample
from typing import Union, TypeVar
from causal_testing.specification.conditional_independence import (
    ConditionalIndependence,
)
from math import inf

Node = Union[str, int]  # Node type hint: A node is a string or an int
CausalDAG = TypeVar("CausalDAG")

logger = logging.getLogger(__name__)
from .scenario import Scenario
from .variable import Output


def list_all_min_sep(
    graph: nx.Graph,
    treatment_node: Node,
    outcome_node: Node,
    treatment_node_set: set[Node],
    outcome_node_set: set[Node],
):
    """A backtracking algorithm for listing all minimal treatment-outcome separators in an undirected graph.

    Reference: (Space-optimal, backtracking algorithms to list the minimal vertex separators of a graph, Ken Takata,
    2013, p.5, ListMinSep procedure).

    :param graph: An undirected graph.
    :param treatment_node: The node corresponding to the treatment variable we wish to separate from the output.
    :param outcome_node: The node corresponding to the outcome variable we wish to separate from the input.
    :param treatment_node_set: Set of treatment nodes.
    :param outcome_node_set: Set of outcome nodes.
    :return: A list of minimal-sized sets of variables which separate treatment and outcome in the undirected graph.
    """
    # 1. Compute the close separator of the treatment set
    close_separator_set = close_separator(
        graph, treatment_node, outcome_node, treatment_node_set
    )

    # 2. Use the close separator to separate the graph and obtain the connected components (connected sub-graphs)
    components_graph = graph.copy()
    components_graph.remove_nodes_from(close_separator_set)
    graph_components = nx.connected_components(components_graph)

    # 3. Find the connected component that contains the treatment node
    treatment_connected_component_node_set = set()
    for component in graph_components:
        if treatment_node in component:
            treatment_connected_component_node_set = component

    # 4. Confirm that the connected component containing the treatment node is disjoint with the outcome node set
    if not treatment_connected_component_node_set.intersection(outcome_node_set):

        # 5. Update the treatment node set to the set of nodes in the connected component containing the treatment node
        treatment_node_set = treatment_connected_component_node_set

        # 6. Obtain the neighbours of the new treatment node set (this excludes the treatment nodes themselves)
        treatment_node_set_neighbours = (
            set.union(*[set(nx.neighbors(graph, node)) for node in treatment_node_set])
            - treatment_node_set
        )

        # 7. Check that there exists at least one neighbour of the treatment nodes that is not in the outcome node set
        if treatment_node_set_neighbours.difference(outcome_node_set):

            # 7.1. If so, sample a random node from the set of treatment nodes' neighbours not in the outcome node set
            node = set(
                sample(treatment_node_set_neighbours.difference(outcome_node_set), 1)
            )

            # 7.2. Add this node to the treatment node set and recurse (left branch)
            yield from list_all_min_sep(
                graph,
                treatment_node,
                outcome_node,
                treatment_node_set.union(node),
                outcome_node_set,
            )

            # 7.3. Add this node to the outcome node set and recurse (right branch)
            yield from list_all_min_sep(
                graph,
                treatment_node,
                outcome_node,
                treatment_node_set,
                outcome_node_set.union(node),
            )
        else:

            # 8. If all neighbours of the treatments nodes are in the outcome node set, return the set of treatment
            # node neighbours
            yield treatment_node_set_neighbours


def close_separator(
    graph: nx.Graph,
    treatment_node: Node,
    outcome_node: Node,
    treatment_node_set: set[Node],
) -> set[Node]:
    """Compute the close separator for a set of treatments in an undirected graph.

    A close separator (relative to a set of variables X) is a separator whose vertices are adjacent to those in X.
    An X-Y separator is a set of variables which, once deleted from a graph, create a subgraph in which X and Y
    are in different components.

    Reference: (Space-optimal, backtracking algorithms to list the minimal vertex separators of a graph, Ken Takata,
    2013, p.4, CloseSeparator procedure).

    :param graph: An undirected graph.
    :param treatment_node: A label for the treatment node (parent of treatments in undirected graph).
    :param outcome_node: A label for the outcome node (parent of outcomes in undirected graph).
    :param treatment_node_set: The set of variables containing the treatment node ({treatment_node}).
    :return: A treatment_node-outcome_node separator whose vertices are adjacent to those in treatments.
    """
    treatment_neighbours = set.union(
        *[set(nx.neighbors(graph, treatment)) for treatment in treatment_node_set]
    )
    components_graph = graph.copy()
    components_graph.remove_nodes_from(treatment_neighbours)
    graph_components = nx.connected_components(components_graph)
    for component in graph_components:
        if outcome_node in component:
            neighbours_of_variables_in_component = set.union(
                *[set(nx.neighbors(graph, variable)) for variable in component]
            )
            # For this algorithm, the neighbours of a node do not include the node itself
            neighbours_of_variables_in_component = (
                neighbours_of_variables_in_component.difference(component)
            )
            return neighbours_of_variables_in_component
    raise ValueError(f"No {treatment_node}-{outcome_node} separator in the graph.")


class CausalDAG(nx.DiGraph):
    """A causal DAG is a directed acyclic graph in which nodes represent random variables and edges represent causality
    between a pair of random variables. We implement a CausalDAG as a networkx DiGraph with an additional check that
    ensures it is acyclic. A CausalDAG must be specified as a dot file.
    """

    def __init__(self, dot_path: str = None, **attr):
        super().__init__(**attr)
        if dot_path:
            self.graph = nx.DiGraph(nx.drawing.nx_agraph.read_dot(dot_path))
        else:
            self.graph = nx.DiGraph()

        if not self.is_acyclic():
            raise nx.HasACycle("Invalid Causal DAG: contains a cycle.")

    def add_edge(self, u_of_edge: Node, v_of_edge: Node, **attr):
        """Add an edge to the causal DAG.

        Overrides the default networkx method to prevent users from adding a cycle.
        :param u_of_edge: From node
        :param v_of_edge: To node
        :param attr: Attributes
        """
        self.graph.add_edge(u_of_edge, v_of_edge, **attr)
        if not self.is_acyclic():
            raise nx.HasACycle("Invalid Causal DAG: contains a cycle.")

    def is_acyclic(self) -> bool:
        """Checks if the graph is acyclic.

        :return: True if acyclic, False otherwise.
        """
        return not list(nx.simple_cycles(self.graph))

    def get_proper_backdoor_graph(
        self, treatments: list[str], outcomes: list[str]
    ) -> CausalDAG:
        """Convert the causal DAG to a proper back-door graph.

        A proper back-door graph of a causal DAG is obtained by
        removing the first edge of every proper causal path from treatments to outcomes. A proper causal path from
        X to Y is a path of directed edges that starts from X and ends in Y.

        Reference: (Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework,
        Zander et al.,  2019, Definition 3, p.15)

        :param treatments: A list of treatment variables.
        :param outcomes: A list of outcomes.
        :return: A CausalDAG corresponding to the proper back-door graph.
        """
        for var in treatments + outcomes:
            if var not in self.graph.nodes:
                raise IndexError(
                    f"{var} not a node in Causal DAG.\nValid nodes are{self.graph.nodes}."
                )

        proper_backdoor_graph = self.copy()
        nodes_on_proper_causal_path = proper_backdoor_graph.proper_causal_pathway(
            treatments, outcomes
        )
        edges_to_remove = [
            (u, v)
            for (u, v) in proper_backdoor_graph.graph.out_edges(treatments)
            if v in nodes_on_proper_causal_path
        ]
        proper_backdoor_graph.graph.remove_edges_from(edges_to_remove)
        return proper_backdoor_graph

    def get_ancestor_graph(
        self, treatments: list[str], outcomes: list[str]
    ) -> CausalDAG:
        """Given a list of treament variables and a list of outcome variables, transform a CausalDAG into an ancestor
        graph.

        An ancestor graph G[An(W)] for a CausalDAG G is a subgraph of G consisting of only the vertices who are
        ancestors of the set of variables W and all edges between them. Note that a node is an ancestor of itself.

        Reference: (Adjustment Criteria in Causal Diagrams: An Algorithmic Perspective, Textor and Lískiewicz, 2012,
        p. 3 [Descendants and Ancestors]).

        :param treatments: A list of treatment variables to include in the ancestral graph (and their ancestors).
        :param outcomes: A list of outcome variables to include in the ancestral graph (and their ancestors).
        :return: An ancestral graph relative to the set of variables X union Y.
        """
        ancestor_graph = self.copy()
        treatment_ancestors = set.union(
            *[
                nx.ancestors(ancestor_graph.graph, treatment).union({treatment})
                for treatment in treatments
            ]
        )
        outcome_ancestors = set.union(
            *[
                nx.ancestors(ancestor_graph.graph, outcome).union({outcome})
                for outcome in outcomes
            ]
        )
        variables_to_keep = treatment_ancestors.union(outcome_ancestors)
        variables_to_remove = set(self.graph.nodes).difference(variables_to_keep)
        ancestor_graph.graph.remove_nodes_from(variables_to_remove)
        return ancestor_graph

    def get_indirect_graph(
        self, treatments: list[str], outcomes: list[str]
    ) -> CausalDAG:
        """
        This is the counterpart of the back-door graph for direct effects. We remove only edges pointing from X to Y.
        It is a Python implementation of the indirectGraph function from Dagitty.

        :param list[str] treatments: List of treatment names.
        :param list[str] outcomes: List of outcome names.
        :return: The indirect graph with edges pointing from X to Y removed.
        :rtype: CausalDAG
        """
        gback = self.copy()
        ee = []
        for s in treatments:
            for t in outcomes:
                if (s, t) in gback.graph.edges:
                    ee.append((s, t))
        for v1, v2 in ee:
            gback.graph.remove_edge(v1, v2)
        return gback

    def direct_effect_adjustment_sets(
        self, treatments: list[str], outcomes: list[str]
    ) -> list[set[str]]:
        """
        Get the smallest possible set of variables that blocks all back-door paths between all pairs of treatments
        and outcomes for DIRECT causal effect.

        This is an Python implementation of the listMsasTotalEffect function from Dagitty using Algorithms presented in
        Adjustment Criteria in Causal Diagrams: An Algorithmic Perspective, Textor and Lískiewicz, 2012 and extended in
        Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework, Zander et al.,
        2019. These works use the algorithm presented by Takata et al. in their work entitled: Space-optimal,
        backtracking algorithms to list the minimal vertex separators of a graph, 2013.

        :param list[str] treatments: List of treatment names.
        :param list[str] outcomes: List of outcome names.
        :return: A list of possible adjustment sets.
        :rtype: list[set[str]]
        """

        indirect_graph = self.get_indirect_graph(treatments, outcomes)
        ancestor_graph = indirect_graph.get_ancestor_graph(treatments, outcomes)
        gam = nx.moral_graph(ancestor_graph.graph)

        edges_to_add = [("TREATMENT", treatment) for treatment in treatments]
        edges_to_add += [("OUTCOME", outcome) for outcome in outcomes]
        gam.add_edges_from(edges_to_add)

        min_seps = list(
            list_all_min_sep(
                gam, "TREATMENT", "OUTCOME", set(treatments), set(outcomes)
            )
        )
        # min_seps.remove(set(outcomes))
        return min_seps

    def enumerate_minimal_adjustment_sets(
        self, treatments: list[str], outcomes: list[str]
    ) -> list[set[str]]:
        """Get the smallest possible set of variables that blocks all back-door paths between all pairs of treatments
        and outcomes.

        This is an implementation of the Algorithm presented in Adjustment Criteria in Causal Diagrams: An
        Algorithmic Perspective, Textor and Lískiewicz, 2012 and extended in Separators and adjustment sets in causal
        graphs: Complete criteria and an algorithmic framework, Zander et al.,  2019. These works use the algorithm
        presented by Takata et al. in their work entitled: Space-optimal, backtracking algorithms to list the minimal
        vertex separators of a graph, 2013.

        At a high-level, this algorithm proceeds as follows for a causal DAG G, set of treatments X, and set of
        outcomes Y):
        1). Transform G to a proper back-door graph G_pbd (remove the first edge from X on all proper causal paths).
        2). Transform G_pbd to the ancestor moral graph (G_pbd[An(X union Y)])^m.
        3). Apply Takata's algorithm to output all minimal X-Y separators in the graph.

        :param treatments: A list of strings representing treatments.
        :param outcomes: A list of strings representing outcomes.
        :return: A list of strings representing the minimal adjustment set.
        """
        # 1. Construct the proper back-door graph's ancestor moral graph
        proper_backdoor_graph = self.get_proper_backdoor_graph(treatments, outcomes)
        ancestor_proper_backdoor_graph = proper_backdoor_graph.get_ancestor_graph(
            treatments, outcomes
        )
        moralised_proper_backdoor_graph = nx.moral_graph(
            ancestor_proper_backdoor_graph.graph
        )

        # 2. Add an edge X^m to treatment nodes and Y^m to outcome nodes
        edges_to_add = [("TREATMENT", treatment) for treatment in treatments]
        edges_to_add += [("OUTCOME", outcome) for outcome in outcomes]
        moralised_proper_backdoor_graph.add_edges_from(edges_to_add)

        # 3. Remove treatment and outcome nodes from graph and connect neighbours
        treatment_neighbours = set.union(
            *[
                set(nx.neighbors(moralised_proper_backdoor_graph, treatment))
                for treatment in treatments
            ]
        ) - set(treatments)
        outcome_neighbours = set.union(
            *[
                set(nx.neighbors(moralised_proper_backdoor_graph, outcome))
                for outcome in outcomes
            ]
        ) - set(outcomes)

        neighbour_edges_to_add = list(combinations(treatment_neighbours, 2)) + list(
            combinations(outcome_neighbours, 2)
        )
        moralised_proper_backdoor_graph.add_edges_from(neighbour_edges_to_add)

        # 4.  Find all minimal separators of X^m and Y^m using Takata's algorithm for listing minimal separators
        treatment_node_set = {"TREATMENT"}
        outcome_node_set = set(
            nx.neighbors(moralised_proper_backdoor_graph, "OUTCOME")
        ).union({"OUTCOME"})
        minimum_adjustment_sets = list(
            list_all_min_sep(
                moralised_proper_backdoor_graph,
                "TREATMENT",
                "OUTCOME",
                treatment_node_set,
                outcome_node_set,
            )
        )
        return minimum_adjustment_sets

    def adjustment_set_is_minimal(
        self, treatments: list[str], outcomes: list[str], adjustment_set: set[str]
    ) -> bool:
        """Given a list of treatments X, a list of outcomes Y, and an adjustment set Z, determine whether Z is the
        smallest possible adjustment set.

        Z is the minimal adjustment set if no element of Z can be removed without breaking the constructive back-door
        criterion.

        Reference: Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework,
        Zander et al., 2019, Corollary 5, p.19)

        :param treatments: List of treatment variables.
        :param outcomes: List of outcome variables.
        :param adjustment_set: Set of adjustment variables.
        :return: True or False depending on whether the adjustment set is minimal.
        """
        proper_backdoor_graph = self.get_proper_backdoor_graph(treatments, outcomes)

        # Ensure that constructive back-door criterion is satisfied
        if not self.constructive_backdoor_criterion(
            proper_backdoor_graph, treatments, outcomes, adjustment_set
        ):
            raise ValueError(f"{adjustment_set} is not a valid adjustment set.")

        # Remove each variable one at a time and return false if constructive back-door criterion remains satisfied
        for variable in adjustment_set:
            smaller_adjustment_set = adjustment_set.copy()
            smaller_adjustment_set.remove(variable)
            if not smaller_adjustment_set:  # Treat None as the empty set
                smaller_adjustment_set = set()
            if self.constructive_backdoor_criterion(
                proper_backdoor_graph, treatments, outcomes, smaller_adjustment_set
            ):
                logger.info(
                    f"Z={adjustment_set} is not minimal because Z'=Z\\{{'{variable}'}}="
                    f"{smaller_adjustment_set} is also a valid adjustment set."
                )
                return False

        return True

    def constructive_backdoor_criterion(
        self,
        proper_backdoor_graph: CausalDAG,
        treatments: list[str],
        outcomes: list[str],
        covariates: list[str],
    ) -> bool:
        """A variation of Pearl's back-door criterion applied to a proper backdoor graph which enables more efficient
        computation of minimal adjustment sets for the effect of a set of treatments on a set of outcomes.

        The constructive back-door criterion is satisfied for a causal DAG G, a set of treatments X, a set of outcomes
        Y, and a set of covariates Z, if:
        (1) Z is not a descendent of any variable on a proper causal path between X and Y.
        (2) Z d-separates X and Y in the proper back-door graph relative to X and Y.

        Reference: (Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework,
        Zander et al.,  2019, Definition 4, p.16)

        :param proper_backdoor_graph: A proper back-door graph relative to the specified treatments and outcomes.
        :param treatments: A list of treatment variables that appear in the proper back-door graph.
        :param outcomes: A list of outcome variables that appear in the proper back-door graph.
        :param covariates: A list of variables that appear in the proper back-door graph that we will check against
        the constructive back-door criterion.
        :return: True or False, depending on whether the set of covariates satisfies the constructive back-door
        criterion.
        """
        # Condition (1)
        proper_causal_path_vars = self.proper_causal_pathway(treatments, outcomes)
        descendents_of_proper_casual_paths = set.union(
            *[
                set.union(
                    nx.descendants(self.graph, proper_causal_path_var),
                    {proper_causal_path_var},
                )
                for proper_causal_path_var in proper_causal_path_vars
            ]
        )

        if not set(covariates).issubset(
            set(self.graph.nodes).difference(descendents_of_proper_casual_paths)
        ):
            logger.info(
                f"Failed Condition 1: Z={covariates} **is** a descendent of some variable on a proper causal "
                f"path between X={treatments} and Y={outcomes}."
            )
            return False

        # Condition (2)
        if not nx.d_separated(
            proper_backdoor_graph.graph, set(treatments), set(outcomes), set(covariates)
        ):
            logger.info(
                f"Failed Condition 2: Z={covariates} **does not** d-separate X={treatments} and Y={outcomes} in"
                f" the proper back-door graph relative to X and Y."
            )
            return False

        return True

    def proper_causal_pathway(
        self, treatments: list[str], outcomes: list[str]
    ) -> list[str]:
        """Given a list of treatments and outcomes, compute the proper causal pathways between them.

        PCP(X, Y) = {DeX^(X) - X} intersect AnX_(Y)}, where:
        - DeX^(X) refers to the descendents of X in the graph obtained by deleting all edges into X.
        - AnX_(Y) refers to the ancestors of Y in the graph obtained by deleting all edges leaving X.

        :param treatments: A list of treatment variables in the causal DAG.
        :param outcomes: A list of outcomes in the causal DAG.
        :return vars_on_proper_causal_pathway: Return a list of the variables on the proper causal pathway between
        treatments and outcomes.
        """
        treatments_descendants = set.union(
            *[
                nx.descendants(self.graph, treatment).union({treatment})
                for treatment in treatments
            ]
        )
        treatments_descendants_without_treatments = set(
            treatments_descendants
        ).difference(treatments)
        backdoor_graph = self.get_backdoor_graph(set(treatments))
        outcome_ancestors = set.union(
            *[
                nx.ancestors(backdoor_graph, outcome).union({outcome})
                for outcome in outcomes
            ]
        )
        nodes_on_proper_causal_paths = (
            treatments_descendants_without_treatments.intersection(outcome_ancestors)
        )
        return nodes_on_proper_causal_paths

    def get_backdoor_graph(self, treatments: list[str]) -> CausalDAG:
        """A back-door graph is a graph for the list of treatments is a Causal DAG in which all edges leaving the
        treatment nodes are deleted.

        :param treatments: The set of treatments whose outgoing edges will be deleted.
        :return: A back-door graph corresponding to the given causal DAG and set of treatments.
        """
        outgoing_edges = self.graph.out_edges(treatments)
        backdoor_graph = self.graph.copy()
        backdoor_graph.remove_edges_from(outgoing_edges)
        return backdoor_graph

    def depends_on_outputs(self, node: Node, scenario: Scenario) -> bool:
        """Check whether a given node in a given scenario is or depends on a
        model output in the given scenario. That is, whether or not the model
        needs to be run to determine its value.

        NOTE: The graph must be acyclic for this to terminate.

        :param Node node: The node in the DAG representing the variable of interest.
        :param Scenario scenario: The modelling scenario.
        :return: Whether the given variable is or depends on an output.
        :rtype: bool
        """
        if isinstance(scenario.variables[node], Output):
            return True
        return any(
            [
                self.depends_on_outputs(n, scenario)
                for n in self.graph.predecessors(node)
            ]
        )

    def list_conditional_independence_relationships(
        self, search_heuristic: str = "minimal"
    ):
        """List all conditional independence relationships implied by a DAG.

        :param str search_heuristic: A string representing the choice of search heuristic to employ in order to reduce
                                     the number of listed conditional independence relationships. The following
                                     heuristics are available:
                                     minimal: output minimal sized conditional independence relationships for each
                                              unique pair of variables in the DAG.
                                     maximal: output maximal sized conditional independence relationships for each
                                              unique pair of variables in the DAG.
                                     all: output all conditional independence relationships for each unique pair of
                                          variables in the DAG. Warning: this can produce an extremely large number
                                          of conditional independence relationships.

        :return: A list of conditional independence relationships implied by the DAG.
        :rtype: list
        """
        conditional_independence_relationships = []
        if len(self.graph.nodes) < 2:
            return None

        for node_pair in combinations(self.graph.nodes, 2):
            x, y = node_pair
            nodes = set(self.graph.nodes) - set(node_pair)

            # Check unconditional independence relationships
            if nx.d_separated(self.graph, {x}, {y}, set()):
                conditional_independence_relationships.append(
                    ConditionalIndependence(x, y)
                )

            # Check conditional independence relationships by considering adjustment sets of increasing size
            current_adjustment_set_size = inf
            start = 1
            end = len(nodes) + 1
            step = 1

            # For the maximal setting, consider adjustment sets of decreasing size
            if search_heuristic == "maximal":
                step = -1
                start, end = end, start

            for adjustment_set_size in range(start, end, step):
                for zs in combinations(nodes, adjustment_set_size):
                    zs_set = set(list(zs))
                    if nx.d_separated(self.graph, {x}, {y}, zs_set):
                        conditional_independence_relationships.append(
                            ConditionalIndependence(x, y, zs_set)
                        )
                        current_adjustment_set_size = adjustment_set_size

                # To obtain minimal or maximal sized adjustment sets, stop searching after finding the first adjustment
                # set size that yields adjustments
                if (search_heuristic != "all") and (current_adjustment_set_size < inf):
                    break

        return conditional_independence_relationships

    def __str__(self):
        return f"Nodes: {self.graph.nodes}\nEdges: {self.graph.edges}"
