import unittest

import numpy as np

from autotcr.aggregation import aggregate_probabilities, normalize_abundance
from autotcr.exceptions import InputFormatError


class AggregationTests(unittest.TestCase):
    def test_normalized_abundance_sums_to_one(self):
        observed = normalize_abundance([1, 2, 7])
        np.testing.assert_allclose(observed, [0.1, 0.2, 0.7])

    def test_ars_is_abundance_weighted_mean(self):
        metrics = aggregate_probabilities([0.1, 0.9], [1, 3])
        self.assertAlmostEqual(metrics["ars"], 0.7)

    def test_top_rules_are_deterministic(self):
        metrics = aggregate_probabilities(np.linspace(0, 1, 100), np.ones(100))
        self.assertAlmostEqual(metrics["top100_mean"], 0.5)
        self.assertGreater(metrics["top_diff_mean"], 0)

    def test_invalid_abundance_is_rejected(self):
        with self.assertRaises(InputFormatError):
            aggregate_probabilities([0.2, 0.8], [0, 0])


if __name__ == "__main__":
    unittest.main()

