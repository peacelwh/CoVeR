"""Discrepancy-gated activation steering of a trained LLM-Fusion (Sec. 3.2; inference-time comparator).

Nothing is trained. A restricted view p of the same sample (text-only, audio-only, visual-only or content-free) is
run first; when the fused state and the state of that view differ by more than a threshold,

    d_p = 1 - cos(h^f, h^p) > tau_d        (means over the fusion tokens, first steered layer),

the fusion-token states of the full view are pushed away from the view in the last quarter of the decoder layers:

    h^f <- h^f + alpha * sum_p w_p * r_p,   r_p = (h^f - h^p) rescaled to the norm of h^f,

with budgeted weights w_p (sum <= 1). Two uses:

* the fixed-threshold comparison of the paper (tau_d = 0.15, alpha = 0.1, all restricted views);
* the conflict-triggered state steering compared on MR-Bench-H, which steers away from the text view only: the
  threshold triggers on 15% of the validation samples and the strength alpha in {0.1, 0.3, 1} is chosen by validation
  Acc2 (:func:`calibrate`).
"""
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cover.data import MODALITIES, collate, to_device
from cover.metrics import regression_metrics

RESTRICTED_VIEWS = ("t", "a", "v", "cf")


class ActivationSteering:
    def __init__(self, model, alpha: float = 0.1, tau_d: float = 0.15, views: Sequence[str] = RESTRICTED_VIEWS,
                 layers: Optional[Sequence[int]] = None):
        self.model, self.alpha, self.tau_d, self.views = model, alpha, tau_d, tuple(views)
        L = model.num_layers
        self.layers = sorted(layers) if layers is not None else list(range(L - L // 4, L))  # last quarter
        self._ctx = None

    def _hook(self, layer_idx):
        def hook(module, args, output):
            ctx = self._ctx
            if ctx is None:
                return None
            hs = output[0] if isinstance(output, tuple) else output
            fs, fe = ctx["fusion"]
            h_full = hs[:, fs:fe, :].float()
            states = {p: ctx["states"][layer_idx][i].float() for i, p in enumerate(self.views)}
            if layer_idx == self.layers[0]:  # the gate is decided once, at the first steered layer
                for p, h_p in states.items():
                    d = 1.0 - F.cosine_similarity(h_full.mean(1), h_p.mean(1), dim=-1)
                    ctx["d"][p], ctx["s"][p] = d, (d > self.tau_d).float()
            w = torch.stack([ctx["s"][p] * ctx["valid"][p] for p in self.views], 0)
            w = w / w.sum(0, keepdim=True).clamp(min=1.0)
            norm_full = h_full.flatten(1).norm(dim=-1)
            delta = torch.zeros_like(h_full)
            for i, p in enumerate(self.views):
                r = h_full - states[p]
                r = r * (norm_full / r.flatten(1).norm(dim=-1).clamp(min=1e-6)).view(-1, 1, 1)
                delta = delta + (self.alpha * w[i]).view(-1, 1, 1) * r
            hs_new = hs.clone()
            hs_new[:, fs:fe, :] = (h_full + delta).to(hs.dtype)
            return (hs_new,) + tuple(output[1:]) if isinstance(output, tuple) else hs_new
        return hook

    @torch.no_grad()
    def forward(self, feats, mask) -> Dict[str, torch.Tensor]:
        """Steered prediction ``y`` of the full view, the discrepancies ``d`` and the gates ``fired`` per view."""
        model, B = self.model, mask.shape[0]
        out, layout = model.run_views(feats, mask, self.views, return_hidden=True)
        fs, fe = layout["fusion"]
        states = {li: out.hidden_states[li + 1][:, fs:fe, :].view(len(self.views), B, model.K, -1) for li in self.layers}
        valid = {p: (torch.ones(B, device=mask.device) if p == "cf" else mask[:, MODALITIES.index(p)]) for p in self.views}
        self._ctx = {"fusion": (fs, fe), "states": states, "valid": valid, "d": {}, "s": {}}
        hooks = [model.decoder_layers()[li].register_forward_hook(self._hook(li)) for li in self.layers]
        try:
            out, layout = model.run_views(feats, mask, ["full"])
        finally:
            for h in hooks:
                h.remove()
            ctx, self._ctx = self._ctx, None
        y = model.head(out.last_hidden_state, layout)["y"]
        return {"y": y, "d": ctx["d"], "fired": {p: ctx["s"][p] * valid[p] for p in self.views}}


@torch.no_grad()
def predict(steering: ActivationSteering, dataset, batch_size: int, device, num_workers: int = 2) -> Dict[str, np.ndarray]:
    """Steered predictions over ``dataset`` (original label scale) with, per restricted view, d and the gate."""
    steering.model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)
    rec = {}
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = steering.forward({m: batch[m] for m in MODALITIES}, batch["mask"])
        rows = {"y_raw": batch["y_raw"], "pred": dataset.denorm_label(out["y"].float())}
        for p in steering.views:
            rows[f"d_{p}"], rows[f"fired_{p}"] = out["d"][p], out["fired"][p]
        rec.setdefault("idx", []).append(batch["idx"].cpu().numpy())
        for k, v in rows.items():
            rec.setdefault(k, []).append(v.float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in rec.items()}


def calibrate(model, valid_ds, batch_size: int, device, alphas: Sequence[float] = (0.1, 0.3, 1.0),
              trigger: float = 0.15) -> ActivationSteering:
    """Conflict-triggered steering away from the text view, calibrated on the validation split with all modalities
    observed: the threshold is the quantile of d_t that triggers on ``trigger`` of the samples and alpha is the
    candidate with the best validation Acc2 (ties: the smaller alpha). Test labels are never used."""
    valid_ds.set_pattern([1, 1, 1])
    steering = ActivationSteering(model, alpha=alphas[0], tau_d=float("inf"), views=("t",))
    rec = predict(steering, valid_ds, batch_size, device)  # no sample exceeds the threshold: plain model, and d_t
    steering.tau_d = float(np.quantile(rec["d_t"], 1.0 - trigger))
    best = None
    for alpha in alphas:
        steering.alpha = alpha
        r = predict(steering, valid_ds, batch_size, device)
        acc = regression_metrics(r["pred"], r["y_raw"])["Acc2"]
        if best is None or acc > best[0] + 1e-9:
            best = (acc, alpha)
    steering.alpha = best[1]
    return steering
