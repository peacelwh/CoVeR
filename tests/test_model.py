import os
import tempfile
import unittest

import torch

import helpers
from cover.data import MODALITIES, SentimentDataset, collate
from cover.losses import FusionObjective, token_diversity_loss
from cover.model import LLMFusion, disagreement_gate


def batch_of(ds, idx, mask=None):
    b = collate([ds[i] for i in idx])
    if mask is not None:
        b["mask"] = torch.tensor([mask] * len(idx), dtype=torch.float32)
    return b


class TestGate(unittest.TestCase):
    def test_eq4_and_budget(self):
        y = {"t": torch.tensor([0.8, 0.8, 0.8, 0.8, 0.2]),
             "a": torch.tensor([0.2, 0.45, 0.9, 0.2, 0.9]),
             "v": torch.tensor([0.1, 0.9, 0.9, 0.1, 0.9])}
        mask = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]], dtype=torch.float32)
        w = disagreement_gate(y, mask, tau=0.4)             # confident: |y - 0.5| >= 0.2
        self.assertEqual(w[:, 0].tolist(), [0.5, 0.5])        # both views confidently disagree: budget shared
        self.assertEqual(w[:, 1].tolist(), [0.0, 0.0])        # audio disagrees without confidence, visual agrees
        self.assertEqual(w[:, 2].tolist(), [0.0, 0.0])        # no disagreement
        self.assertEqual(w[:, 3].tolist(), [0.0, 1.0])        # a missing modality cannot fire
        self.assertEqual(w[:, 4].tolist(), [0.5, 0.5])

    def test_gated_prediction_is_mean_of_disagreeing_views(self):
        y_hat = torch.tensor([0.7])
        y = {"t": torch.tensor([0.8]), "a": torch.tensor([0.2]), "v": torch.tensor([0.1])}
        w = disagreement_gate(y, torch.ones(1, 3), tau=0.3)
        y_tilde = y_hat + sum(w[i] * (y[m] - y_hat) for i, m in enumerate(("a", "v")))
        self.assertAlmostEqual(float(y_tilde), 0.15, places=6)


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = helpers.make_corpus(os.path.join(cls.tmp.name, "sims.pt"))
        cls.ds = SentimentDataset(cls.path, "train")
        cls.model = helpers.tiny_model()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def feats(self, b):
        return {m: b[m] for m in MODALITIES}

    def test_prompt_states_availability(self):
        p = LLMFusion.prompt_text([1, 0, 1])
        self.assertIn("available text, visual features", p)
        self.assertIn("The audio modality is missing", p)

    def test_views_share_one_token_layout(self):
        b = batch_of(self.ds, range(4), [1, 1, 0])
        feat = self.model.project(self.feats(b))
        embeds, attn, pos, layout = self.model.assemble(feat, b["mask"], ["full", "t", "a", "v", "cf"])
        S = attn.shape[1]
        self.assertEqual(embeds.shape[:2], (5 * 4, S))
        self.assertEqual(layout["fusion"], (S - self.model.K, S))     # fusion tokens at the same positions in every view
        C = layout["C"].view(5, 4, 3)
        self.assertEqual(C[0, 0].tolist(), [1, 1, 0])                 # full view: content = availability
        self.assertEqual(C[1, 0].tolist(), [1, 0, 0])                 # text-only view
        self.assertEqual(C[3, 0].tolist(), [0, 0, 0])                 # view of a missing modality carries no content
        self.assertEqual(C[4, 0].tolist(), [0, 0, 0])                 # content-free view

    def test_full_view_prediction_does_not_depend_on_extra_views(self):
        self.model.eval()
        b = batch_of(self.ds, range(6), [1, 1, 1])
        with torch.no_grad():
            alone = self.model(self.feats(b), b["mask"])
            judges = self.model(self.feats(b), b["mask"], views="judges")
            every = self.model(self.feats(b), b["mask"], views="all", tau=0.3)
        self.assertTrue(torch.allclose(alone["y"], judges["y"], atol=1e-5))
        self.assertTrue(torch.allclose(alone["y"], every["y"], atol=1e-5))
        self.assertTrue(torch.equal(alone["y"], alone["y_tilde"]))    # accuracy mode: one view, no gate
        self.assertEqual(set(judges["y_views"]), {"t", "a", "v"})
        self.assertTrue(((judges["y"] >= 0) & (judges["y"] <= 1)).all())

    def test_gate_withholds_the_fused_task_gradient(self):
        """If a judge confidently disagrees, y_tilde is a constant and the fused path receives no task gradient."""
        self.model.eval()
        b = batch_of(self.ds, range(8), [1, 1, 1])
        out = self.model(self.feats(b), b["mask"], views="judges", tau=0.0)   # tau = 0: every disagreement fires
        fired = out["gate_weights"].sum(0) > 0
        grad = torch.autograd.grad(((out["y_tilde"] - b["y"]) ** 2).sum(), out["y"], retain_graph=True)[0]
        self.assertTrue(torch.all(grad[fired].abs() < 1e-7))
        if (~fired).any():
            self.assertTrue(torch.all(grad[~fired].abs() > 0))
        mean_views = sum(out["gate_weights"][i] * out["y_views"][m] for i, m in enumerate(("a", "v")))
        self.assertTrue(torch.allclose(out["y_tilde"][fired], mean_views[fired], atol=1e-6))

    def test_unimodal_loss_ignores_unobserved_modalities(self):
        self.model.eval()
        b = batch_of(self.ds, range(6), [1, 0, 1])
        out = self.model(self.feats(b), b["mask"], views="judges")
        loss = FusionObjective(lambda_uni=1.0)
        base = loss.unimodal_loss(out, b)
        b2 = dict(b)
        b2["y_a"] = 1.0 - b["y_a"]                                    # labels of the missing modality are never read
        self.assertAlmostEqual(float(base), float(loss.unimodal_loss(out, b2)), places=7)
        expected = (((out["y_views"]["t"] - b["y_t"]) ** 2).sum() + ((out["y_views"]["v"] - b["y_v"]) ** 2).sum()) / 12
        self.assertAlmostEqual(float(base), float(expected), places=6)

    def test_token_diversity_loss(self):
        x = torch.nn.functional.one_hot(torch.arange(3), 3).float()[None].repeat(4, 1, 1)   # distinct, consistent tokens
        self.assertAlmostEqual(float(token_diversity_loss(x)), 0.0, places=6)
        self.assertGreater(float(token_diversity_loss(torch.ones(4, 3, 3))), 0.5)            # collapsed tokens


if __name__ == "__main__":
    unittest.main()
