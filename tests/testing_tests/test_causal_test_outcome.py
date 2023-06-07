import unittest
from causal_testing.testing.causal_test_outcome import ExactValue, SomeEffect, Positive, Negative, NoEffect
from causal_testing.testing.causal_test_result import CausalTestResult, TestValue
from causal_testing.testing.estimators import LinearRegressionEstimator
from causal_testing.utils.validation import CausalValidator


class TestCausalTestOutcome(unittest.TestCase):
    """Test the TestCausalTestOutcome basic methods."""

    def setUp(self) -> None:
        self.estimator = LinearRegressionEstimator(
            treatment="A",
            outcome="A",
            treatment_value=TestValue(int, 1),
            control_value=TestValue(int, 0),
            adjustment_set={},
        )

    def test_None_ci(self):
        test_value = TestValue(type="ate", value=0)
        ctr = CausalTestResult(
            estimator=self.estimator,
            test_value=test_value,
            confidence_intervals=[None, None],
            effect_modifier_configuration=None,
        )

        self.assertIsNone(ctr.ci_low())
        self.assertIsNone(ctr.ci_high())
        self.assertEqual(
            ctr.to_dict(),
            {
                "treatment": "A",
                "control_value": 0,
                "treatment_value": 1,
                "outcome": "A",
                "adjustment_set": [],
            },
        )

    def test_empty_adjustment_set(self):
        test_value = TestValue(type="ate", value=0)
        ctr = CausalTestResult(
            estimator=self.estimator,
            test_value=test_value,
            confidence_intervals=None,
            effect_modifier_configuration=None,
        )

        self.assertIsNone(ctr.ci_low())
        self.assertIsNone(ctr.ci_high())
        self.assertEqual(
            str(ctr),
            (
                "Causal Test Result\n==============\n"
                "Treatment: A\n"
                "Control value: 0\n"
                "Treatment value: 1\n"
                "Outcome: A\n"
                "Adjustment set: set()\n"
                "ate: 0\n"
            ),
        )

    def test_Positive_ate_pass(self):
        test_value = TestValue(type="ate", value=5.05)
        ctr = CausalTestResult(
            estimator=self.estimator,
            test_value=test_value,
            confidence_intervals=None,
            effect_modifier_configuration=None,
        )
        ev = Positive()
        self.assertTrue(ev.apply(ctr))

    def test_Positive_risk_ratio_pass(self):
        test_value = TestValue(type="risk_ratio", value=2)
        ctr = CausalTestResult(
            estimator=self.estimator,
            test_value=test_value,
            confidence_intervals=None,
            effect_modifier_configuration=None,
        )
        ev = Positive()
        self.assertTrue(ev.apply(ctr))

    def test_Positive_fail(self):
        test_value = TestValue(type="ate", value=0)
        ctr = CausalTestResult(
            estimator=self.estimator,
            test_value=test_value,
            confidence_intervals=None,
            effect_modifier_configuration=None,
        )
        ev = Positive()
        self.assertFalse(ev.apply(ctr))

    def test_Positive_fail_ci(self):
        test_value = TestValue(type="ate", value=0)
        ctr = CausalTestResult(
            estimator=self.estimator,
            test_value=test_value,
            confidence_intervals=[-1, 1],
            effect_modifier_configuration=None,
        )
        ev = Positive()
        self.assertFalse(ev.apply(ctr))

    def test_Negative_ate_pass(self):
        test_value = TestValue(type="ate", value=-5.05)
        ctr = CausalTestResult(
            estimator=self.estimator,
            test_value=test_value,
            confidence_intervals=None,
            effect_modifier_configuration=None,
        )
        ev = Negative()
        self.assertTrue(ev.apply(ctr))

    def test_Negative_risk_ratio_pass(self):
        test_value = TestValue(type="risk_ratio", value=0.2)
        ctr = CausalTestResult(
            estimator=self.estimator,
            test_value=test_value,
            confidence_intervals=None,
            effect_modifier_configuration=None,
        )
        ev = Negative()
        self.assertTrue(ev.apply(ctr))

    def test_Negative_fail(self):
        test_value = TestValue(type="ate", value=0)
        ctr = CausalTestResult(
            estimator=self.estimator,
            test_value=test_value,
            confidence_intervals=None,
            effect_modifier_configuration=None,
        )
        ev = Negative()
        self.assertFalse(ev.apply(ctr))

    def test_Negative_fail_ci(self):
        test_value = TestValue(type="ate", value=0)
        ctr = CausalTestResult(
            estimator=self.estimator,
            test_value=test_value,
            confidence_intervals=[-1, 1],
            effect_modifier_configuration=None,
        )
        ev = Negative()
        self.assertFalse(ev.apply(ctr))

    def test_exactValue_pass(self):
        test_value = TestValue(type="ate", value=5.05)
        ctr = CausalTestResult(
            estimator=self.estimator,
            test_value=test_value,
            confidence_intervals=None,
            effect_modifier_configuration=None,
        )
        ev = ExactValue(5, 0.1)
        self.assertTrue(ev.apply(ctr))

    def test_exactValue_pass_ci(self):
        test_value = TestValue(type="ate", value=5.05)
        ctr = CausalTestResult(
            estimator=self.estimator,
            test_value=test_value,
            confidence_intervals=[4, 6],
            effect_modifier_configuration=None,
        )
        ev = ExactValue(5, 0.1)
        self.assertTrue(ev.apply(ctr))

    def test_exactValue_fail(self):
        test_value = TestValue(type="ate", value=0)
        ctr = CausalTestResult(
            estimator=self.estimator,
            test_value=test_value,
            confidence_intervals=None,
            effect_modifier_configuration=None,
        )
        ev = ExactValue(5, 0.1)
        self.assertFalse(ev.apply(ctr))

    def test_invalid(self):
        test_value = TestValue(type="invalid", value=5.05)
        ctr = CausalTestResult(
            estimator=self.estimator,
            test_value=test_value,
            confidence_intervals=[4.8, 6.7],
            effect_modifier_configuration=None,
        )
        with self.assertRaises(ValueError):
            SomeEffect().apply(ctr)
        with self.assertRaises(ValueError):
            NoEffect().apply(ctr)
        with self.assertRaises(ValueError):
            Positive().apply(ctr)
        with self.assertRaises(ValueError):
            Negative().apply(ctr)

    def test_someEffect_pass_coefficient(self):
        test_value = TestValue(type="coefficient", value=5.05)
        ctr = CausalTestResult(
            estimator=self.estimator,
            test_value=test_value,
            confidence_intervals=[4.8, 6.7],
            effect_modifier_configuration=None,
        )
        self.assertTrue(SomeEffect().apply(ctr))
        self.assertFalse(NoEffect().apply(ctr))

    def test_someEffect_pass_ate(self):
        test_value = TestValue(type="ate", value=5.05)
        ctr = CausalTestResult(
            estimator=self.estimator,
            test_value=test_value,
            confidence_intervals=[4.8, 6.7],
            effect_modifier_configuration=None,
        )
        self.assertTrue(SomeEffect().apply(ctr))
        self.assertFalse(NoEffect().apply(ctr))

    def test_someEffect_pass_rr(self):
        test_value = TestValue(type="risk_ratio", value=5.05)
        ctr = CausalTestResult(
            estimator=self.estimator,
            test_value=test_value,
            confidence_intervals=[4.8, 6.7],
            effect_modifier_configuration=None,
        )
        self.assertTrue(SomeEffect().apply(ctr))
        self.assertFalse(NoEffect().apply(ctr))

    def test_someEffect_fail(self):
        test_value = TestValue(type="ate", value=0)
        ctr = CausalTestResult(
            estimator=self.estimator,
            test_value=test_value,
            confidence_intervals=[-0.1, 0.2],
            effect_modifier_configuration=None,
        )
        self.assertFalse(SomeEffect().apply(ctr))
        self.assertTrue(NoEffect().apply(ctr))

    def test_someEffect_dict(self):
        test_value = TestValue(type="ate", value=0)
        ctr = CausalTestResult(
            estimator=self.estimator,
            test_value=test_value,
            confidence_intervals=[-0.1, 0.2],
            effect_modifier_configuration=None,
        )
        ev = SomeEffect()
        self.assertEqual(
            ctr.to_dict(),
            {
                "treatment": "A",
                "control_value": 0,
                "treatment_value": 1,
                "outcome": "A",
                "adjustment_set": [],
                "ci_low": -0.1,
                "ci_high": 0.2,
            },
        )

    def test_positive_risk_ratio_e_value(self):
        cv = CausalValidator()
        e_value = cv.estimate_e_value(1.5)
        self.assertEqual(round(e_value, 4), 2.366)

    def test_positive_risk_ratio_e_value_using_ci(self):
        cv = CausalValidator()
        e_value = cv.estimate_e_value_using_ci(1.5, [1.2, 1.8])
        self.assertEqual(round(e_value, 4), 1.6899)

    def test_negative_risk_ratio_e_value(self):
        cv = CausalValidator()
        e_value = cv.estimate_e_value(0.8)
        self.assertEqual(round(e_value, 4), 1.809)

    def test_negative_risk_ratio_e_value_using_ci(self):
        cv = CausalValidator()
        e_value = cv.estimate_e_value_using_ci(0.8, [0.2, 0.9])
        self.assertEqual(round(e_value, 4), 1.4625)
