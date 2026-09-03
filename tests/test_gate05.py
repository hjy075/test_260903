import unittest

from src.gate05 import (
    active_in_window,
    classify_future_continuation,
    gate05_decision,
    standardized_mean_difference,
    window_bounds,
)


class Gate05DefinitionTests(unittest.TestCase):
    def test_active_in_window_requires_minimum_observed_weeks(self):
        observed = [True, True, True, True, False, False]
        self.assertTrue(active_in_window(observed, min_weeks=4))
        self.assertFalse(active_in_window(observed, min_weeks=5))

    def test_future_continuation_uses_only_future_availability_window(self):
        future_observed = [True, True, True, False, True, True]
        self.assertTrue(classify_future_continuation(future_observed, min_weeks=5))
        self.assertFalse(classify_future_continuation(future_observed, min_weeks=6))

    def test_empty_window_is_not_active(self):
        self.assertFalse(active_in_window([], min_weeks=1))


class Gate05AnalysisTests(unittest.TestCase):
    def test_window_bounds_for_past_and_future(self):
        self.assertEqual(window_bounds(origin=100, width=13, offset_end=-1), (87, 100))
        self.assertEqual(window_bounds(origin=100, width=13, offset_end=52), (140, 153))

    def test_standardized_mean_difference_detects_group_shift(self):
        a = [0.0, 0.0, 0.0, 0.0]
        b = [1.0, 1.0, 1.0, 1.0]
        self.assertGreater(standardized_mean_difference(a, b), 0.0)

    def test_gate05_decision_passes_with_balanced_groups_and_separation(self):
        out = gate05_decision(
            minority_shares=[0.30, 0.25, 0.20],
            median_abs_smd_by_feature={"zero_rate": 0.35, "momentum": 0.10},
            aucs=[0.58, 0.62, 0.64],
        )
        self.assertEqual(out["status"], "PASS_GATE_0_5")

    def test_gate05_decision_hard_kills_when_noncontinuing_group_is_tiny(self):
        out = gate05_decision(
            minority_shares=[0.02, 0.03, 0.04],
            median_abs_smd_by_feature={"zero_rate": 0.60},
            aucs=[0.80],
        )
        self.assertEqual(out["status"], "HARD_KILL")


if __name__ == "__main__":
    unittest.main()
