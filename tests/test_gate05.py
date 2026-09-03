import unittest

from src.gate05 import active_in_window, classify_future_continuation


class Gate05DefinitionTests(unittest.TestCase):
    def test_active_in_window_requires_minimum_observed_weeks(self):
        observed = [True, True, True, True, False, False]
        self.assertTrue(active_in_window(observed, min_weeks=4))
        self.assertFalse(active_in_window(observed, min_weeks=5))

    def test_future_continuation_uses_only_future_availability_window(self):
        future_observed = [True, True, True, False, True, True]
        self.assertTrue(
            classify_future_continuation(future_observed, min_weeks=5)
        )
        self.assertFalse(
            classify_future_continuation(future_observed, min_weeks=6)
        )

    def test_empty_window_is_not_active(self):
        self.assertFalse(active_in_window([], min_weeks=1))


if __name__ == "__main__":
    unittest.main()
