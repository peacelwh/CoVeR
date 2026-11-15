import os
import tempfile
import unittest

import numpy as np

import helpers  # noqa: F401
from cover import features


class TestFeatures(unittest.TestCase):
    def test_pool_to_fifty_steps(self):
        self.assertEqual(tuple(features.pool_T(np.random.randn(173, 65)).shape), (50, 65))
        self.assertEqual(tuple(features.pool_T(np.random.randn(7, 65)).shape), (50, 65))
        x = np.random.randn(50, 4).astype(np.float32)
        np.testing.assert_allclose(features.pool_T(x).numpy(), x)

    def test_speaker_face_from_the_side_of_the_female_actor(self):
        left, right, none = np.ones((3, features.VIS_DIM), np.float32), 2 * np.ones((4, features.VIS_DIM), np.float32), np.zeros((0, features.VIS_DIM), np.float32)
        dialogue = {"Ses01F_impro01_F000": {"L": left, "R": right},
                    "Ses01F_impro01_M000": {"L": left, "R": right},
                    "Ses01F_impro01_M001": {"L": left, "R": none},      # the speaker's side has no detected face
                    "Ses01F_impro01_F001": {"L": none, "R": none}}
        seqs, info = features.speaker_sequences(dialogue, female_side="L")
        self.assertEqual(info["female_side"], "L")
        self.assertEqual(float(seqs["Ses01F_impro01_F000"][0].mean()), 1.0)      # female speaker: left face
        self.assertEqual(float(seqs["Ses01F_impro01_M000"][0].mean()), 2.0)      # male speaker: right face
        self.assertEqual(seqs["Ses01F_impro01_M001"][1], "other_side")
        self.assertEqual(seqs["Ses01F_impro01_F001"][1], "no_face")

    def test_read_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ids.txt")
            with open(path, "w") as f:
                f.write("a\n\nb \n")
            self.assertEqual(features.read_ids(path), {"a", "b"})
            self.assertEqual(features.read_ids(os.path.join(tmp, "missing.txt")), set())

    def test_released_holdout_lists(self):
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
        if not os.path.exists(os.path.join(root, "mrbench_h_holdout", "iemocap.txt")):
            self.skipTest("hold-out lists not present")
        ids = features.read_ids(os.path.join(root, "mrbench_h_holdout", "iemocap.txt"))
        self.assertEqual(len(ids), 512)                                        # candidates of Sessions 1-4
        self.assertFalse(any(i.startswith("Ses05") for i in ids))


if __name__ == "__main__":
    unittest.main()
