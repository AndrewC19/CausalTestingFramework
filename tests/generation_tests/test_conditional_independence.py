import unittest
from causal_testing.specification.conditional_independence import ConditionalIndependence
from causal_testing.specification.variable import Input, Output


class TestConditionalIndependence(unittest.TestCase):
    """
        Testing Conditional Independence container object.
    """

    def test_print_conditional_independence_no_adjustment_set(self):
        cis = ConditionalIndependence("X", "Y")
        self.assertEqual("X⫫Y", str(cis))

    def test_print_conditional_independence_with_unary_adjustment_set(self):
        cis = ConditionalIndependence("X", "Y", "Z")
        self.assertEqual("X⫫Y|{Z}", str(cis))

    def test_print_conditional_independence_with_binary_adjustment_set(self):
        cis = ConditionalIndependence("X", "Y", {"Z1", "Z2"})
        self.assertEqual("X⫫Y|{Z1, Z2}", str(cis))

