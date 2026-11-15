import math
import unittest

import numpy as np

import helpers  # noqa: F401
from cover.metrics import auroc, average_metrics, fisher_mean, polarity, regression_metrics, wilson_interval


class TestMetrics(unittest.TestCase):
    def test_polarity(self):
        self.assertEqual(polarity([-0.3, 0.0, 2.0]).tolist(), [-1, 0, 1])

    def test_fisher_mean(self):
        rs = [0.2, 0.5, 0.8]
        expected = math.tanh(np.mean([math.atanh(r) for r in rs]))
        self.assertAlmostEqual(fisher_mean(rs), expected, places=10)
        self.assertGreater(fisher_mean(rs), np.mean(rs))      # the z-transform weighs high correlations more
        self.assertAlmostEqual(fisher_mean([0.4] * 7), 0.4, places=10)

    def test_regression_metrics_exclude_neutral(self):
        labels = np.array([1.0, -1.0, 0.0, 2.0])
        preds = np.array([0.5, 0.5, -3.0, 1.0])            # the neutral sample is wrong in polarity but excluded
        m = regression_metrics(preds, labels)
        self.assertAlmostEqual(m["Acc2"], 2 / 3)
        self.assertAlmostEqual(m["MAE"], np.mean(np.abs(preds - labels)))
        self.assertAlmostEqual(m["Corr"], np.corrcoef(preds, labels)[0, 1])

    def test_average_uses_fisher_for_corr_only(self):
        rows = [{"Acc2": 0.5, "Corr": 0.1}, {"Acc2": 0.7, "Corr": 0.9}]
        avg = average_metrics(rows)
        self.assertAlmostEqual(avg["Acc2"], 0.6)
        self.assertAlmostEqual(avg["Corr"], fisher_mean([0.1, 0.9]))

    def test_wilson_interval(self):
        lo, hi = wilson_interval(310, 620)
        self.assertAlmostEqual((lo + hi) / 2, 50.0, places=6)
        self.assertLess(hi - lo, 2 * 3.95)                 # at most +-3.9 points with 620 clips
        lo, hi = wilson_interval(55, 110)
        self.assertAlmostEqual((hi - lo) / 2, 9.2, delta=0.2)

    def test_auroc(self):
        self.assertEqual(auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]), 1.0)
        self.assertEqual(auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]), 0.0)
        self.assertEqual(auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]), 0.5)


if __name__ == "__main__":
    unittest.main()
