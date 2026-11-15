"""Minimal end-to-end run on a synthetic corpus with a tiny randomly initialised decoder: training of LLM-Fusion and
CoVeR, model selection, strict checkpoint loading, and evaluation in accuracy and robust mode."""
import json
import os
import tempfile
import unittest
from unittest import mock

import torch

import helpers
from cover import mrbench_a, mrbench_h, train as T
from cover.data import SentimentDataset
from cover.evaluate import evaluate_patterns, predict
from cover.steering import ActivationSteering, predict as steer_predict


class TestTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = cls.tmp.name
        cls.data_dir, cls.bench_dir, cls.out_root = (os.path.join(root, d) for d in ("processed", "mrbench_a", "runs"))
        os.makedirs(cls.data_dir)
        path = helpers.make_corpus(os.path.join(cls.data_dir, "sims.pt"))
        blob = torch.load(path, weights_only=False)
        for split in ("train", "valid", "test"):
            d = blob[split]
            rows = mrbench_a.annotate(d["id"], d["y"].numpy(), d["y_t"].numpy(), d["y_a"].numpy(), d["y_v"].numpy(), 0.2, False)
            mrbench_a.save(rows, cls.bench_dir, "sims", split)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_training(self, *extra):
        argv = ["--dataset", "sims", "--epochs", "2", "--batch", "8", "--num_workers", "0", "--eval_batch", "16",
                "--data_dir", self.data_dir, "--mrbench_dir", self.bench_dir, "--out_root", self.out_root, *extra]
        def build(cfg, meta, device):
            return helpers.tiny_model(lora_r=cfg.lora_r, gate_alpha=getattr(cfg, "gate_alpha", None)).to(device)
        with mock.patch.object(T, "build_model", build):
            return T.main(argv)

    def test_llm_fusion(self):
        final = self.run_training("--method", "llm_fusion", "--recipe", "default")
        self.assertEqual(set(final), {"accuracy"})                       # no views: accuracy mode only
        self.assertEqual(set(final["accuracy"]["patterns"]), {"TAV", "TA", "TV", "AV", "T", "A", "V", "Avg"})
        self.assertIn("Rate", final["accuracy"]["rates"])
        self.assertIn("ShortErr", final["accuracy"]["mrbench_a"])
        run = os.path.join(self.out_root, "sims", "llm_fusion_default_s42")
        for f in ("config.json", "history.json", "best.pt", "results.json"):
            self.assertTrue(os.path.exists(os.path.join(run, f)), f)
        history = json.load(open(os.path.join(run, "history.json")))
        self.assertEqual(set(history[0]["train"]), {"task", "con", "reg", "total"})

    def test_cover(self):
        final = self.run_training("--method", "cover", "--recipe", "default", "--ema", "0.9")
        self.assertEqual(set(final), {"accuracy", "robust"})
        self.assertEqual(final["robust"]["tau_test"], 0.3)
        run = os.path.join(self.out_root, "sims", "cover_default_s42")
        history = json.load(open(os.path.join(run, "history.json")))
        self.assertIn("uni", history[0]["train"])                        # the judges are supervised
        cfg = json.load(open(os.path.join(run, "config.json")))
        self.assertEqual((cfg["tau"], cfg["lambda_uni"]), (0.3, 1.0))

        # the stored checkpoint reproduces the stored results; a different architecture is rejected
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ds = SentimentDataset(os.path.join(self.data_dir, "sims.pt"), "test")
        model = helpers.tiny_model(lora_r=cfg["lora_r"]).to(device)
        T.load_checkpoint(model, os.path.join(run, "best.pt"), device)
        again = evaluate_patterns(model, ds, 16, device, num_workers=0)
        self.assertAlmostEqual(again["Avg"]["Corr"], final["accuracy"]["patterns"]["Avg"]["Corr"], places=5)
        with self.assertRaises(RuntimeError):
            T.load_checkpoint(helpers.tiny_model(lora_r=cfg["lora_r"] * 2).to(device), os.path.join(run, "best.pt"), device)

        # robust mode differs from accuracy mode only where a judge confidently disagrees
        ds.set_pattern([1, 1, 1])
        acc = predict(model, ds, 16, device, num_workers=0)
        rob = predict(model, ds, 16, device, mode="robust", tau=10.0, num_workers=0)   # no view is confident enough
        self.assertTrue(abs(acc["pred"] - rob["pred"]).max() < 1e-5)

        # activation steering with a threshold that never triggers leaves the predictions unchanged
        st = steer_predict(ActivationSteering(model, alpha=1.0, tau_d=float("inf")), ds, 16, device, num_workers=0)
        self.assertTrue(abs(acc["pred"] - st["pred"]).max() < 1e-5)
        st = steer_predict(ActivationSteering(model, alpha=1.0, tau_d=-1.0), ds, 16, device, num_workers=0)
        self.assertTrue(abs(acc["pred"] - st["pred"]).max() > 0)

        # item-level predictions on benchmark clips of this corpus, in every view and both modes, and their scores
        path = os.path.join(self.data_dir, "sims.pt")
        clips = [{"uid": f"sims-{k:05d}", "source": "sims", "source_split": split, "source_id": f"{split}_{i:04d}",
                  "category": "L" if k % 2 else "V", "label_polarity": 1, "distractor_polarity": -1}
                 for k, (split, i) in enumerate([("test", 3), ("valid", 1), ("test", 0), ("valid", 7)])]
        rows = mrbench_h.predict_items(model, path, clips, device, "cover", 42, views=["full", "nonverbal"], batch_size=16)
        rows += mrbench_h.predict_items(model, path, clips, device, "cover-robust", 42, "robust", 0.3, ["full"], 16)
        self.assertEqual(len(rows), 4 * 3)
        full = SentimentDataset(path, "test")
        full.set_pattern([1, 1, 1])
        ref = predict(model, full, 16, device, num_workers=0)["pred"]
        by_uid = {r["uid"]: r for r in rows if r["model"] == "cover" and r["view"] == "full"}
        self.assertAlmostEqual(by_uid["sims-00000"]["score"], float(ref[3]), places=3)      # clip <-> sample mapping
        self.assertAlmostEqual(by_uid["sims-00002"]["score"], float(ref[0]), places=3)
        scores = mrbench_h.score({c["uid"]: c for c in clips}, rows)
        self.assertEqual(set(scores), {"cover|full", "cover|nonverbal", "cover-robust|full"})
        self.assertIn("rate", mrbench_h.text_override_rate({c["uid"]: c for c in clips}, rows, "cover"))

    def test_variants(self):
        final = self.run_training("--method", "cover", "--recipe", "default", "--no_gate")
        self.assertEqual(set(final), {"accuracy"})                       # judges without the gate
        final = self.run_training("--method", "cover", "--missing", "permanent", "--no_rates", "--name", "cover_permanent")
        self.assertNotIn("rates", final["accuracy"])
        final = self.run_training("--method", "llm_fusion", "--recipe", "default", "--conflict_weight", "5",
                                  "--name", "reweight")
        self.assertIn("accuracy", final)

    def test_learned_gate(self):
        final = self.run_training("--method", "learned_gate")
        self.assertEqual(set(final), {"accuracy"})
        run = os.path.join(self.out_root, "sims", "learned_gate_s42")
        history = json.load(open(os.path.join(run, "history.json")))
        self.assertIn("gate", history[0]["train"])                       # the gates are supervised
        cfg = json.load(open(os.path.join(run, "config.json")))
        self.assertEqual((cfg["gate_alpha"], cfg["gate_targets"], cfg["gate_agreement"]), (0.5, "unimodal", "polarity"))
        state = torch.load(os.path.join(run, "best.pt"), map_location="cpu", weights_only=True)
        self.assertTrue(any(k.startswith("gates.t.") for k in state))
        # the gate only moves the fused readout: with strength 0 the prediction is that of the plain backbone
        model = helpers.tiny_model(lora_r=cfg["lora_r"], gate_alpha=0.0)
        plain = helpers.tiny_model(lora_r=cfg["lora_r"])
        plain.load_state_dict(model.state_dict(), strict=False)
        ds = SentimentDataset(os.path.join(self.data_dir, "sims.pt"), "test")
        ds.set_pattern([1, 1, 0])
        device = torch.device("cpu")
        a = predict(model, ds, 16, device, num_workers=0)["pred"]
        b = predict(plain, ds, 16, device, num_workers=0)["pred"]
        self.assertTrue(abs(a - b).max() < 1e-5)


if __name__ == "__main__":
    unittest.main()
