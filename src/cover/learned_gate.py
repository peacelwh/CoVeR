"""Learned reliability gate: a training-time correction compared with CoVeR (Sec. 4.3, "+ learned reliability gate").

The counterfactual views of a sample (text-, audio-, visual-only and content-free) are run first, without gradient.
For every view p, a small gate reads the fused readout, the readout of the view and the views' predictions and
outputs a strength s_p in (0, 1); the fused readout is then pushed away from the views,

    z^f <- z^f + alpha * sum_p w_p * r_p,   r_p = (z^f - z^p) rescaled to the norm of z^f,

with budgeted weights w_p (s_p of the observed views, sum <= 1). The gates of the single-modality views are trained
to fire on unreliable views: the target of s_j is 1 if the judgment of modality j disagrees with the label and 0
otherwise, where the judgment is the human unimodal label (CH-SIMS, CH-SIMSv2) or the prediction of the view itself.
The model needs the views at test time as well.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cover.data import MODALITIES
from cover.losses import FusionObjective
from cover.model import LLMFusion, NEUTRAL

GATED_VIEWS = ("t", "a", "v", "prior")  # 'prior': the content-free view


class ReliabilityGate(nn.Module):
    """s = sigmoid(MLP([proj(r), d, log-norm ratio, availability, prediction cues])) for one view."""

    def __init__(self, hidden: int, proj_dim: int = 32, mlp_hidden: int = 64, init_bias: float = -2.0, extra_dim: int = 7):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.proj = nn.Linear(hidden, proj_dim)
        self.mlp = nn.Sequential(nn.Linear(proj_dim + 3 + extra_dim, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, 1))
        nn.init.constant_(self.mlp[-1].bias, init_bias)  # start with a weak gate

    def forward(self, z_full, z_view, avail, extra):
        r = z_full - z_view
        d = 1.0 - F.cosine_similarity(z_full, z_view, dim=-1)
        ratio = torch.log((r.norm(dim=-1) + 1e-6) / (z_full.norm(dim=-1) + 1e-6))
        logit = self.mlp(torch.cat([self.proj(self.norm(r)), d[:, None], ratio[:, None], avail[:, None], extra], -1)).squeeze(-1)
        return torch.sigmoid(logit), logit


class LearnedGateFusion(LLMFusion):
    """LLM-Fusion with a learned reliability gate on the fused readout."""

    def __init__(self, *args, gate_alpha: float = 0.15, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate_alpha = gate_alpha
        self.gates = nn.ModuleDict({p: ReliabilityGate(self.D) for p in GATED_VIEWS})

    def forward(self, feats: Dict[str, torch.Tensor], mask: torch.Tensor, views: Optional[str] = None,
                tau: Optional[float] = None, gate_scale: float = 1.0, return_hidden: bool = False):
        B = mask.shape[0]
        feat = self.project(feats)
        # ---- pass 1: the views, without gradient. Autocast caches weight casts; casts made under no_grad would be
        # reused by the gradient pass and cut its graph, so the cache is disabled here and cleared afterwards.
        ac_on = torch.is_autocast_enabled()
        ac_dtype = torch.get_autocast_dtype("cuda") if ac_on else torch.bfloat16
        with torch.no_grad(), torch.autocast("cuda", dtype=ac_dtype, enabled=ac_on, cache_enabled=False):
            out_v, layout_v = self.run_views(feats, mask, ["t", "a", "v", "cf"], feat=feat.detach())
            hv = self.head(out_v.last_hidden_state, layout_v)
            y_v = hv["y"].view(4, B).float()
            z_v = hv["readout"].view(4, B, -1)
            cues = torch.stack([2 * (y_v[0] - NEUTRAL), 2 * (y_v[1] - NEUTRAL), 2 * (y_v[2] - NEUTRAL), 2 * (y_v[3] - NEUTRAL),
                                (y_v[0] - y_v[1]).abs(), (y_v[0] - y_v[2]).abs(), (y_v[1] - y_v[2]).abs()], -1)
        torch.clear_autocast_cache()
        valid = torch.cat([mask.t(), torch.ones(1, B, device=mask.device)], 0)  # [4, B]: t, a, v observed; content-free
        logits = {}

        def adjust(z):
            s = []
            for i, p in enumerate(GATED_VIEWS):
                sp, logits[p] = self.gates[p](z.detach(), z_v[i], valid[i], cues)  # the gate reads detached states
                s.append(sp * valid[i])
            w = torch.stack(s, 0)
            w = w / w.sum(0, keepdim=True).clamp(min=1.0)
            norm_full = z.norm(dim=-1)
            delta = torch.zeros_like(z)
            for i in range(len(GATED_VIEWS)):
                r = z - z_v[i]
                r = r * (norm_full / r.norm(dim=-1).clamp(min=1e-6))[:, None]
                delta = delta + (self.gate_alpha * gate_scale * w[i])[:, None] * r
            return z + delta

        # ---- pass 2: the full view, with the gated readout
        out, layout = self.run_views(feats, mask, ["full"], return_hidden, feat=feat)
        h = self.head(out.last_hidden_state, layout, adjust_readout=adjust)
        res = {k: h[k] for k in ("y", "y_main", "y_aux", "fusion")}
        res.update(y_tilde=h["y"], layout=layout, view_valid=mask.t(), gate_logits=logits,
                   y_views={m: y_v[j] for j, m in enumerate(MODALITIES)}, y_cf=y_v[3])
        if return_hidden:
            res["hidden_states"] = out.hidden_states
        return res


class GateObjective(FusionObjective):
    """L_L plus the supervision of the gates: binary cross-entropy between the gate of every observed
    single-modality view and its target, 1 if the judgment of that modality disagrees with the label.

    targets:   'unimodal'  the judgment is the unimodal label y^(j) of the batch
               'views'     the judgment is the prediction of the view itself
    agreement: 'polarity'  the judgment agrees with the label if both have the same polarity
               'close'     same polarity and |judgment - label| < delta (normalised label scale)
    """

    def __init__(self, targets: str = "views", agreement: str = "close", delta: float = 0.15, lambda_gate: float = 1.0, **kw):
        super().__init__(**kw)
        assert targets in ("unimodal", "views") and agreement in ("polarity", "close")
        self.targets, self.agreement, self.delta, self.lambda_gate = targets, agreement, delta, lambda_gate

    def forward(self, out, batch):
        total, parts = super().forward(out, batch)
        y, valid = batch["y"], out["view_valid"]
        loss, n = torch.zeros((), device=y.device), torch.zeros((), device=y.device)
        for j, m in enumerate(MODALITIES):
            judgment = batch[f"y_{m}"] if self.targets == "unimodal" else out["y_views"][m].detach()
            agree = ((judgment - NEUTRAL) * (y - NEUTRAL)) > 0
            if self.agreement == "close":
                agree = agree & ((judgment - y).abs() < self.delta)
            bce = F.binary_cross_entropy_with_logits(out["gate_logits"][m].float(), 1.0 - agree.float(), reduction="none")
            loss, n = loss + (bce * valid[j]).sum(), n + valid[j].sum()
        gate = loss / n.clamp(min=1)
        total = total + self.lambda_gate * gate
        parts.update(gate=float(gate.detach()), total=float(total.detach()))
        return total, parts
