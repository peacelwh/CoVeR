import glob
import os
import unittest

import numpy as np

import helpers  # noqa: F401
from cover import mrbench_a, mrbench_h

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


class TestCaseTyping(unittest.TestCase):
    def test_rules(self):
        kappa = 0.2
        #            y     u_t   u_a   u_v
        cases = [(0.6, 0.4, 0.2, 0.8, "consensus"),         # all judgments share the polarity of the label
                 (0.6, -0.4, 0.4, 0.0, "conflict"),         # text confidently wrong, audio confidently right
                 (0.6, -0.4, 0.1, 0.1, "other"),            # text wrong but no confident non-verbal evidence
                 (0.6, -0.1, 0.4, 0.4, "other"),            # text wrong without confidence
                 (0.6, 0.4, -0.4, 0.4, "weak_conflict"),    # text right, audio confidently wrong
                 (0.0, 0.4, 0.4, 0.4, "other"),             # neutral label
                 (-0.6, 0.4, -0.3, -0.3, "conflict")]
        y, t, a, v, expected = zip(*cases)
        self.assertEqual(mrbench_a.type_cases(y, t, a, v, kappa).tolist(), list(expected))

    def test_confidence_threshold(self):
        self.assertEqual(mrbench_a.confidence_threshold([-1.0, 1.0]), 0.2)
        self.assertEqual(mrbench_a.confidence_threshold([-3.0, 3.0]), 0.5)

    def test_case_metrics(self):
        rows = mrbench_a.annotate(["a", "b", "c", "d"], [0.6, 0.6, -0.6, 0.6], [0.4, -0.4, 0.4, -0.4],
                                  [0.4, 0.4, -0.4, 0.4], [0.4, 0.0, -0.4, 0.4], 0.2, keep_judgments=True)
        self.assertEqual([r["case_type"] for r in rows], ["consensus", "conflict", "conflict", "conflict"])
        m = mrbench_a.case_metrics(np.array([0.5, 0.5, 0.5, -0.5]), rows)   # follows the words on c and d
        self.assertEqual(m["Cons"], 1.0)
        self.assertAlmostEqual(m["Conf"], 1 / 3)
        self.assertAlmostEqual(m["ShortErr"], 2 / 3)


class TestReleasedAnnotations(unittest.TestCase):
    """The MR-Bench-A files shipped in data/mrbench_a carry the counts of the paper (train / validation / test)."""
    COUNTS = {"mosi": {"samples": (1284, 229, 686), "consensus": (428, 92, 187), "conflict": (55, 9, 36), "weak_conflict": (297, 55, 186)},
              "mosei": {"samples": (16326, 1871, 4659), "consensus": (6387, 675, 1669), "conflict": (68, 4, 20), "weak_conflict": (675, 83, 265)},
              "sims": {"samples": (1368, 456, 457), "consensus": (621, 211, 221), "conflict": (105, 37, 31), "weak_conflict": (99, 38, 32)},
              "simsv2": {"samples": (2722, 647, 1034), "consensus": (1423, 326, 457), "conflict": (148, 55, 73), "weak_conflict": (146, 42, 54)}}

    def test_counts(self):
        directory = os.path.join(ROOT, "data", "mrbench_a")
        if not glob.glob(os.path.join(directory, "*.jsonl")):
            self.skipTest("annotation files not present")
        for ds, expected in self.COUNTS.items():
            for i, split in enumerate(("train", "valid", "test")):
                c = mrbench_a.counts(mrbench_a.load(directory, ds, split))
                for key, values in expected.items():
                    self.assertEqual(c[key], values[i], f"{ds}/{split}/{key}")


class TestMRBenchH(unittest.TestCase):
    def setUp(self):
        self.items = {f"u{i}": {"uid": f"u{i}", "source": "sims", "category": "L" if i < 6 else "V",
                                "label_polarity": 1, "distractor_polarity": -1} for i in range(10)}

    def rows(self, model, view, seed, wrong, invalid=()):
        return [{"uid": u, "model": model, "view": view, "seed": seed,
                 "polarity": 0 if u in invalid else (-1 if u in wrong else 1)} for u in self.items]

    def test_score(self):
        rows = self.rows("m", "full", 42, wrong={"u0", "u1", "u6"}, invalid={"u2"}) + self.rows("m", "full", 43, wrong={"u0"})
        r = mrbench_h.score(self.items, rows)["m|full"]
        self.assertEqual(r["seeds"], [42, 43])
        self.assertAlmostEqual(r["LS"]["acc"], np.mean([50.0, 100 * 5 / 6]))          # mean over seeds
        self.assertAlmostEqual(r["LS"]["std"], np.std([50.0, 100 * 5 / 6]))
        self.assertAlmostEqual(r["LS"]["ShortErr"], np.mean([100 * 2 / 6, 100 / 6]))  # an invalid answer is wrong but not a shortcut error
        self.assertAlmostEqual(r["NS"]["acc"], np.mean([75.0, 100.0]))

    def test_missing_prediction_counts_as_wrong(self):
        rows = [r for r in self.rows("m", "full", 0, wrong=set()) if r["uid"] != "u0"]
        r = mrbench_h.score(self.items, rows)["m|full"]
        self.assertAlmostEqual(r["LS"]["acc"], 100 * 5 / 6)
        self.assertEqual(r["missing"], 1)

    def test_text_override_rate(self):
        rows = self.rows("m", "nonverbal", 0, wrong={"u5"}) + self.rows("m", "full", 0, wrong={"u0", "u1", "u5"})
        o = mrbench_h.text_override_rate(self.items, rows, "m")
        self.assertAlmostEqual(o["rate"], 100 * 2 / 5)      # 5 LS clips solved without text, 2 made wrong by the text

    def test_processed_split(self):
        f = mrbench_h.processed_split
        self.assertEqual(f({"source": "mosi", "source_split": "valid"}), "valid")
        self.assertEqual(f({"source": "iemocap", "source_split": "Session5"}), "test")
        self.assertEqual(f({"source": "iemocap", "source_split": "Session2"}), "pool")
        self.assertEqual(f({"source": "meld", "source_split": "dev"}), "pool")
        self.assertEqual(f({"source": "mustardpp", "source_split": "all"}), "pool")


if __name__ == "__main__":
    unittest.main()
