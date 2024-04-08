import unittest
import os
import networkx as nx
from causal_testing.specification.causal_dag import CausalDAG, close_separator, list_all_min_sep
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.variable import Input, Output
from causal_testing.testing.base_test_case import BaseTestCase
from tests.test_helpers import create_temp_dir_if_non_existent, remove_temp_dir_if_existent


class TestCausalDAGIssue90(unittest.TestCase):
    """
    Test the CausalDAG class for the resolution of Issue 90.
    """

    def setUp(self) -> None:
        temp_dir_path = create_temp_dir_if_non_existent()
        self.dag_dot_path = os.path.join(temp_dir_path, "dag.dot")
        dag_dot = """digraph DAG { rankdir=LR; Z -> X; X -> M; M -> Y; Z -> M; }"""
        with open(self.dag_dot_path, "w") as f:
            f.write(dag_dot)

    def test_enumerate_minimal_adjustment_sets(self):
        """Test whether enumerate_minimal_adjustment_sets lists all possible minimum sized adjustment sets."""
        causal_dag = CausalDAG(self.dag_dot_path)
        xs, ys = ["X"], ["Y"]
        adjustment_sets = causal_dag.enumerate_minimal_adjustment_sets(xs, ys)
        self.assertEqual([{"Z"}], adjustment_sets)

    def tearDown(self) -> None:
        remove_temp_dir_if_existent()


class TestIVAssumptions(unittest.TestCase):
    def setUp(self) -> None:
        temp_dir_path = create_temp_dir_if_non_existent()
        self.dag_dot_path = os.path.join(temp_dir_path, "dag.dot")
        dag_dot = """digraph G { I -> X; X -> Y; U -> X; U -> Y;}"""
        f = open(self.dag_dot_path, "w")
        f.write(dag_dot)
        f.close()

    def test_valid_iv(self):
        causal_dag = CausalDAG(self.dag_dot_path)
        self.assertTrue(causal_dag.check_iv_assumptions("X", "Y", "I"))

    def test_unrelated_instrument(self):
        causal_dag = CausalDAG(self.dag_dot_path)
        causal_dag.graph.remove_edge("I", "X")
        with self.assertRaises(ValueError):
            causal_dag.check_iv_assumptions("X", "Y", "I")

    def test_direct_cause(self):
        causal_dag = CausalDAG(self.dag_dot_path)
        causal_dag.graph.add_edge("I", "Y")
        with self.assertRaises(ValueError):
            causal_dag.check_iv_assumptions("X", "Y", "I")

    def test_common_cause(self):
        causal_dag = CausalDAG(self.dag_dot_path)
        causal_dag.graph.add_edge("U", "I")
        with self.assertRaises(ValueError):
            causal_dag.check_iv_assumptions("X", "Y", "I")


class TestCausalDAG(unittest.TestCase):
    """
    Test the CausalDAG class for creation of Causal Directed Acyclic Graphs (DAGs).

    In particular, confirm whether the Causal DAG class creates valid causal directed acyclic graphs (empty and directed
    graphs without cycles) and refuses to create invalid (cycle-containing) graphs.
    """

    def setUp(self) -> None:
        temp_dir_path = create_temp_dir_if_non_existent()
        self.dag_dot_path = os.path.join(temp_dir_path, "dag.dot")
        dag_dot = """digraph G { A -> B; B -> C; D -> A; D -> C;}"""
        f = open(self.dag_dot_path, "w")
        f.write(dag_dot)
        f.close()

    def test_valid_causal_dag(self):
        """Test whether the Causal DAG is valid."""
        causal_dag = CausalDAG(self.dag_dot_path)
        print(causal_dag)
        assert list(causal_dag.graph.nodes) == ["A", "B", "C", "D"] and list(causal_dag.graph.edges) == [
            ("A", "B"),
            ("B", "C"),
            ("D", "A"),
            ("D", "C"),
        ]

    def test_invalid_causal_dag(self):
        """Test whether a cycle-containing directed graph is an invalid causal DAG."""
        causal_dag = CausalDAG(self.dag_dot_path)
        self.assertRaises(nx.HasACycle, causal_dag.add_edge, "C", "A")

    def test_empty_casual_dag(self):
        """Test whether an empty dag can be created."""
        causal_dag = CausalDAG()
        assert list(causal_dag.graph.nodes) == [] and list(causal_dag.graph.edges) == []

    def test_to_dot_string(self):
        causal_dag = CausalDAG(self.dag_dot_path)
        self.assertEqual(causal_dag.to_dot_string(), """digraph G {\nA -> B;\nB -> C;\nD -> A;\nD -> C;\n}""")

    def tearDown(self) -> None:
        remove_temp_dir_if_existent()


class TestCyclicCausalDAG(unittest.TestCase):
    """
    Test the creation of a cyclic causal graph.
    """

    def setUp(self) -> None:
        temp_dir_path = create_temp_dir_if_non_existent()
        self.dag_dot_path = os.path.join(temp_dir_path, "dag.dot")
        dag_dot = """digraph G { A -> B; B -> C; D -> A; D -> C; C -> A;}"""
        f = open(self.dag_dot_path, "w")
        f.write(dag_dot)
        f.close()

    def test_invalid_causal_dag(self):
        self.assertRaises(nx.HasACycle, CausalDAG, self.dag_dot_path)

    def tearDown(self) -> None:
        remove_temp_dir_if_existent()


class TestDAGDirectEffectIdentification(unittest.TestCase):
    """
    Test the Causal DAG identification algorithms and supporting algorithms.
    """

    def setUp(self) -> None:
        temp_dir_path = create_temp_dir_if_non_existent()
        self.dag_dot_path = os.path.join(temp_dir_path, "dag.dot")
        dag_dot = """digraph G { X1->X2;X2->V;X2->D1;X2->D2;D1->Y;D1->D2;Y->D3;Z->X2;Z->Y;}"""
        f = open(self.dag_dot_path, "w")
        f.write(dag_dot)
        f.close()

    def test_direct_effect_adjustment_sets(self):
        causal_dag = CausalDAG(self.dag_dot_path)
        adjustment_sets = causal_dag.direct_effect_adjustment_sets(["X1"], ["Y"])
        self.assertEqual(list(adjustment_sets), [{"D1", "Z"}, {"X2", "Z"}])

    def test_direct_effect_adjustment_sets_no_adjustment(self):
        causal_dag = CausalDAG(self.dag_dot_path)
        adjustment_sets = causal_dag.direct_effect_adjustment_sets(["X2"], ["D1"])
        self.assertEqual(list(adjustment_sets), [set()])


class TestDAGIdentification(unittest.TestCase):
    """
    Test the Causal DAG identification algorithms and supporting algorithms.
    """

    def setUp(self) -> None:
        temp_dir_path = create_temp_dir_if_non_existent()
        self.dag_dot_path = os.path.join(temp_dir_path, "dag.dot")
        dag_dot = """digraph G { X1->X2;X2->V;X2->D1;X2->D2;D1->Y;D1->D2;Y->D3;Z->X2;Z->Y;}"""
        f = open(self.dag_dot_path, "w")
        f.write(dag_dot)
        f.close()

    def test_get_indirect_graph(self):
        causal_dag = CausalDAG(self.dag_dot_path)
        indirect_graph = causal_dag.get_indirect_graph(["D1"], ["Y"])
        original_edges = list(causal_dag.graph.edges)
        original_edges.remove(("D1", "Y"))
        self.assertEqual(list(indirect_graph.graph.edges), original_edges)
        self.assertEqual(indirect_graph.graph.nodes, causal_dag.graph.nodes)

    def test_proper_backdoor_graph(self):
        """Test whether converting a Causal DAG to a proper back-door graph works correctly."""
        causal_dag = CausalDAG(self.dag_dot_path)
        proper_backdoor_graph = causal_dag.get_proper_backdoor_graph(["X1", "X2"], ["Y"])
        edges = set(
            [
                ("X1", "X2"),
                ("X2", "V"),
                ("X2", "D2"),
                ("D1", "D2"),
                ("D1", "Y"),
                ("Y", "D3"),
                ("Z", "X2"),
                ("Z", "Y"),
            ]
        )
        self.assertTrue(set(proper_backdoor_graph.graph.edges).issubset(edges))

    def test_constructive_backdoor_criterion_should_hold(self):
        """Test whether the constructive criterion holds when it should."""
        causal_dag = CausalDAG(self.dag_dot_path)
        xs, ys, zs = ["X1", "X2"], ["Y"], ["Z"]
        proper_backdoor_graph = causal_dag.get_proper_backdoor_graph(xs, ys)
        self.assertTrue(causal_dag.constructive_backdoor_criterion(proper_backdoor_graph, xs, ys, zs))

    def test_constructive_backdoor_criterion_should_not_hold_not_d_separator_in_proper_backdoor_graph(
        self,
    ):
        """Test whether the constructive criterion fails when the adjustment set is not a d-separator."""
        causal_dag = CausalDAG(self.dag_dot_path)
        xs, ys, zs = ["X1", "X2"], ["Y"], ["V"]
        proper_backdoor_graph = causal_dag.get_proper_backdoor_graph(xs, ys)
        self.assertFalse(causal_dag.constructive_backdoor_criterion(proper_backdoor_graph, xs, ys, zs))

    def test_constructive_backdoor_criterion_should_not_hold_descendent_of_proper_causal_path(
        self,
    ):
        """Test whether the constructive criterion holds when the adjustment set Z contains a descendent of a variable
        on a proper causal path between X and Y."""
        causal_dag = CausalDAG(self.dag_dot_path)
        xs, ys, zs = ["X1", "X2"], ["Y"], ["D1"]
        proper_backdoor_graph = causal_dag.get_proper_backdoor_graph(xs, ys)
        self.assertFalse(causal_dag.constructive_backdoor_criterion(proper_backdoor_graph, xs, ys, zs))

    def test_is_min_adjustment_for_min_adjustment(self):
        """Test whether is_min_adjustment can correctly test whether the minimum adjustment set is minimal."""
        causal_dag = CausalDAG(self.dag_dot_path)
        xs, ys, zs = ["X1", "X2"], ["Y"], {"Z"}
        self.assertTrue(causal_dag.adjustment_set_is_minimal(xs, ys, zs))

    def test_is_min_adjustment_for_not_min_adjustment(self):
        """Test whether is_min_adjustment can correctly test whether the minimum adjustment set is not minimal."""
        causal_dag = CausalDAG(self.dag_dot_path)
        xs, ys, zs = ["X1", "X2"], ["Y"], {"Z", "V"}
        self.assertFalse(causal_dag.adjustment_set_is_minimal(xs, ys, zs))

    def test_is_min_adjustment_for_invalid_adjustment(self):
        """Test whether is min_adjustment can correctly identify that the minimum adjustment set is invalid."""
        causal_dag = CausalDAG(self.dag_dot_path)
        xs, ys, zs = ["X1", "X2"], ["Y"], set()
        self.assertRaises(ValueError, causal_dag.adjustment_set_is_minimal, xs, ys, zs)

    def test_get_ancestor_graph_of_causal_dag(self):
        """Test whether get_ancestor_graph converts a CausalDAG to the correct ancestor graph."""
        causal_dag = CausalDAG(self.dag_dot_path)
        xs, ys = ["X1", "X2"], ["Y"]
        ancestor_graph = causal_dag.get_ancestor_graph(xs, ys)
        self.assertEqual(list(ancestor_graph.graph.nodes), ["X1", "X2", "D1", "Y", "Z"])
        self.assertEqual(
            list(ancestor_graph.graph.edges),
            [("X1", "X2"), ("X2", "D1"), ("D1", "Y"), ("Z", "X2"), ("Z", "Y")],
        )

    def test_get_ancestor_graph_of_proper_backdoor_graph(self):
        """Test whether get_ancestor_graph converts a CausalDAG to the correct proper back-door graph."""
        causal_dag = CausalDAG(self.dag_dot_path)
        xs, ys = ["X1", "X2"], ["Y"]
        proper_backdoor_graph = causal_dag.get_proper_backdoor_graph(xs, ys)
        ancestor_graph = proper_backdoor_graph.get_ancestor_graph(xs, ys)
        self.assertEqual(list(ancestor_graph.graph.nodes), ["X1", "X2", "D1", "Y", "Z"])
        self.assertEqual(
            list(ancestor_graph.graph.edges),
            [("X1", "X2"), ("D1", "Y"), ("Z", "X2"), ("Z", "Y")],
        )

    def test_enumerate_minimal_adjustment_sets(self):
        """Test whether enumerate_minimal_adjustment_sets lists all possible minimum sized adjustment sets."""
        causal_dag = CausalDAG(self.dag_dot_path)
        xs, ys = ["X1", "X2"], ["Y"]
        adjustment_sets = causal_dag.enumerate_minimal_adjustment_sets(xs, ys)
        self.assertEqual([{"Z"}], adjustment_sets)

    def test_enumerate_minimal_adjustment_sets_multiple(self):
        """Test whether enumerate_minimal_adjustment_sets lists all minimum adjustment sets if multiple are possible."""
        causal_dag = CausalDAG()
        causal_dag.graph.add_edges_from(
            [
                ("X1", "X2"),
                ("X2", "V"),
                ("Z1", "X2"),
                ("Z1", "Z2"),
                ("Z2", "Z3"),
                ("Z3", "Y"),
                ("D1", "Y"),
                ("D1", "D2"),
                ("Y", "D3"),
            ]
        )
        xs, ys = ["X1", "X2"], ["Y"]
        adjustment_sets = causal_dag.enumerate_minimal_adjustment_sets(xs, ys)
        set_of_adjustment_sets = set(frozenset(min_separator) for min_separator in adjustment_sets)
        self.assertEqual(
            {frozenset({"Z1"}), frozenset({"Z2"}), frozenset({"Z3"})},
            set_of_adjustment_sets,
        )

    def test_enumerate_minimal_adjustment_sets_two_adjustments(self):
        """Test whether enumerate_minimal_adjustment_sets lists all possible minimum adjustment sets of arity two."""
        causal_dag = CausalDAG()
        causal_dag.graph.add_edges_from(
            [
                ("X1", "X2"),
                ("X2", "V"),
                ("Z1", "X2"),
                ("Z1", "Z2"),
                ("Z2", "Z3"),
                ("Z3", "Y"),
                ("D1", "Y"),
                ("D1", "D2"),
                ("Y", "D3"),
                ("Z4", "X1"),
                ("Z4", "Y"),
                ("X2", "D1"),
            ]
        )
        xs, ys = ["X1", "X2"], ["Y"]
        adjustment_sets = causal_dag.enumerate_minimal_adjustment_sets(xs, ys)
        set_of_adjustment_sets = set(frozenset(min_separator) for min_separator in adjustment_sets)
        self.assertEqual(
            {frozenset({"Z1", "Z4"}), frozenset({"Z2", "Z4"}), frozenset({"Z3", "Z4"})},
            set_of_adjustment_sets,
        )

    def test_dag_with_non_character_nodes(self):
        """Test identification for a DAG whose nodes are not just characters (strings of length greater than 1)."""
        causal_dag = CausalDAG()
        causal_dag.graph.add_edges_from(
            [
                ("va", "ba"),
                ("ba", "ia"),
                ("ba", "da"),
                ("ba", "ra"),
                ("la", "va"),
                ("la", "aa"),
                ("aa", "ia"),
                ("aa", "da"),
                ("aa", "ra"),
            ]
        )
        xs, ys = ["ba"], ["da"]
        adjustment_sets = causal_dag.enumerate_minimal_adjustment_sets(xs, ys)
        self.assertEqual(adjustment_sets, [{"aa"}, {"la"}, {"va"}])

    def tearDown(self) -> None:
        remove_temp_dir_if_existent()


class TestDependsOnOutputs(unittest.TestCase):
    """
    Test the depends_on_outputs method.
    """

    def setUp(self) -> None:
        from scipy.stats import uniform
        from causal_testing.specification.variable import Input, Output, Meta
        from causal_testing.specification.scenario import Scenario

        temp_dir_path = create_temp_dir_if_non_existent()
        self.dag_dot_path = os.path.join(temp_dir_path, "dag.dot")
        dag_dot = """digraph G { A -> B; B -> C; D -> A; D -> C}"""
        f = open(self.dag_dot_path, "w")
        f.write(dag_dot)
        f.close()

        D = Input("D", float, uniform(0, 1))
        A = Meta("A", float, uniform(0, 1))
        B = Output("B", float, uniform(0, 1))
        C = Meta("C", float, uniform(0, 1))

        self.scenario = Scenario({A, B, C, D})

    def test_depends_on_outputs_output(self):
        causal_dag = CausalDAG(self.dag_dot_path)
        print("nodes:", causal_dag.nodes())
        print("graph:", causal_dag)
        self.assertTrue(causal_dag.depends_on_outputs("B", self.scenario))

    def test_depends_on_outputs_output_meta(self):
        causal_dag = CausalDAG(self.dag_dot_path)
        print("nodes:", causal_dag.nodes())
        print("graph:", causal_dag)
        self.assertTrue(causal_dag.depends_on_outputs("C", self.scenario))

    def test_depends_on_outputs_input_meta(self):
        causal_dag = CausalDAG(self.dag_dot_path)
        print("nodes:", causal_dag.nodes())
        print("graph:", causal_dag)
        self.assertFalse(causal_dag.depends_on_outputs("A", self.scenario))

    def test_depends_on_outputs_input(self):
        causal_dag = CausalDAG(self.dag_dot_path)
        print("nodes:", causal_dag.nodes())
        print("graph:", causal_dag)
        self.assertFalse(causal_dag.depends_on_outputs("D", self.scenario))

    def tearDown(self) -> None:
        remove_temp_dir_if_existent()


class TestUndirectedGraphAlgorithms(unittest.TestCase):
    """
    Test the graph algorithms designed for the undirected graph variants of a Causal DAG.
    During the identification process, a Causal DAG is converted into several forms of undirected graph which allow for
    more efficient computation of minimal separators. This suite of tests covers the two main algorithms applied to
    these forms of undirected graphs (ancestor and moral graphs): close_separator and list_all_min_sep."""

    def setUp(self) -> None:
        self.graph = nx.Graph()
        self.graph.add_edges_from([("a", 2), ("a", 3), (2, 4), (3, 5), (3, 4), (4, "b"), (5, "b")])
        self.treatment_node = "a"
        self.outcome_node = "b"
        self.treatment_node_set = {"a"}
        self.outcome_node_set = set(nx.neighbors(self.graph, "b"))
        self.outcome_node_set.add("b")

    def test_close_separator(self):
        """Test whether close_separator correctly identifies the close separator of {2,3} in the undirected graph."""
        result = close_separator(self.graph, self.treatment_node, self.outcome_node, self.treatment_node_set)
        self.assertEqual({2, 3}, result)

    def test_list_all_min_sep(self):
        """Test whether list_all_min_sep finds all minimal separators for the undirected graph relative to a and b."""
        min_separators = list(
            list_all_min_sep(
                self.graph,
                self.treatment_node,
                self.outcome_node,
                self.treatment_node_set,
                self.outcome_node_set,
            )
        )

        # Convert list of sets to set of frozen sets for comparison
        min_separators = set(frozenset(min_separator) for min_separator in min_separators)
        self.assertEqual({frozenset({2, 3}), frozenset({3, 4}), frozenset({4, 5})}, min_separators)

    def tearDown(self) -> None:
        remove_temp_dir_if_existent()


class TestHiddenVariableDAG(unittest.TestCase):
    """
    Test the CausalDAG identification for the exclusion of hidden variables.
    """

    def setUp(self) -> None:
        temp_dir_path = create_temp_dir_if_non_existent()
        self.dag_dot_path = os.path.join(temp_dir_path, "dag.dot")
        dag_dot = """digraph DAG { rankdir=LR; Z -> X; X -> M; M -> Y; Z -> M; }"""
        with open(self.dag_dot_path, "w") as f:
            f.write(dag_dot)

    def test_hidden_varaible_adjustment_sets(self):
        """Test whether identification produces different adjustment sets depending on if a variable is hidden."""
        causal_dag = CausalDAG(self.dag_dot_path)
        z = Input("Z", int)
        x = Input("X", int)
        m = Input("M", int)

        adjustment_sets = causal_dag.identification(BaseTestCase(x, m))

        adjustment_sets_with_hidden = causal_dag.identification(BaseTestCase(x, m), hidden_variables=["Z"])

        self.assertNotEqual(adjustment_sets, adjustment_sets_with_hidden)

    def tearDown(self) -> None:
        remove_temp_dir_if_existent()
