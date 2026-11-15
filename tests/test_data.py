import os
import tempfile
import unittest
from collections import Counter

import numpy as np

import helpers
from cover.data import (INCOMPLETE_PATTERNS, PATTERNS, SentimentDataset, pattern_name, permanent_masks, rate_masks)


class TestPatterns(unittest.TestCase):
    def test_seven_patterns_complete_first(self):
        self.assertEqual([pattern_name(p) for p in PATTERNS], ["TAV", "TA", "TV", "AV", "T", "A", "V"])
        self.assertEqual(len(INCOMPLETE_PATTERNS), 6)

    def test_permanent_masks_balanced_and_fixed(self):
        a = permanent_masks(700, "train", mask_seed=0)
        b = permanent_masks(700, "train", mask_seed=0)
        np.testing.assert_array_equal(a, b)                              # one assignment for the whole run
        counts = Counter(pattern_name(m) for m in a.tolist())
        self.assertEqual(set(counts.values()), {100})                    # balanced over the seven patterns
        self.assertAlmostEqual(a.mean(), 4 / 7, places=6)                # a modality is observed with probability 4/7
        self.assertFalse(np.array_equal(a, permanent_masks(700, "valid", mask_seed=0)))
        inc = permanent_masks(600, "train", incomplete_only=True)
        self.assertTrue((inc.sum(1) < 3).all())                          # no sample is ever complete

    def test_rate_masks(self):
        m = rate_masks(2000, 0.7, seed=1)
        self.assertTrue((m.sum(1) > 0).all())                            # at least one modality is kept
        np.testing.assert_array_equal(m, rate_masks(2000, 0.7, seed=1))  # deterministic evaluation


class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = helpers.make_corpus(os.path.join(cls.tmp.name, "sims.pt"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_labels_normalised_to_unit_interval(self):
        ds = SentimentDataset(self.path, "train")
        item = ds[0]
        self.assertTrue(0.0 <= float(item["y"]) <= 1.0)
        self.assertAlmostEqual(float(ds.denorm_label(item["y"])), float(item["y_raw"]), places=5)
        self.assertEqual(item["mask"].tolist(), [1.0, 1.0, 1.0])

    def test_dynamic_masking_redraws_patterns(self):
        ds = SentimentDataset(self.path, "train", missing="dynamic", seed=3)
        ds.set_epoch(0)
        first = [pattern_name(ds[i]["mask"].int().tolist()) for i in range(len(ds))]
        ds.set_epoch(1)
        second = [pattern_name(ds[i]["mask"].int().tolist()) for i in range(len(ds))]
        self.assertNotEqual(first, second)                               # the pattern of a sample changes between visits
        self.assertGreaterEqual(len(set(first + second)), 6)

    def test_permanent_missingness_hides_features_and_labels(self):
        ds = SentimentDataset(self.path, "train", missing="permanent")
        for i in range(len(ds)):
            item = ds[i]
            self.assertEqual(item["mask"].int().tolist(), ds.obs[i].tolist())
            for j, m in enumerate(("t", "a", "v")):
                if not ds.obs[i][j]:
                    self.assertEqual(float(item[m].abs().sum()), 0.0)    # unobserved features are never read
                    self.assertEqual(float(item[f"y_{m}"]), 0.5)

    def test_fixed_pattern(self):
        ds = SentimentDataset(self.path, "test")
        ds.set_pattern([0, 1, 1])
        self.assertEqual(ds[5]["mask"].tolist(), [0.0, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
