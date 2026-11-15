"""Training objectives (Sec. 2.1, Eqs. 5 and 6 of the paper).

LLM-Fusion is trained with L_L = lambda_task * l(y_hat, y) + lambda_con * L_con + lambda_reg * L_reg, where ``l`` is
the squared error (L1 on request), ``L_con`` ties the two decoding paths of the head and ``L_reg`` keeps the fusion
tokens diverse. CoVeR (Eq. 6) puts the task loss on the gated prediction ``y_tilde`` of Eq. (4) instead of ``y_hat``
and adds the unimodal-judge loss ``L_uni`` of Eq. (5), weighted by lambda_uni.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from cover.data import MODALITIES


def token_diversity_loss(fusion: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """L_reg: batch-hardest triplet loss on the cosine distance between fusion-token states [B, K, D].

    Anchor = token k of a sample, positives = token k of the other samples, negatives = the other tokens: every
    fusion token stays consistent across samples and distinct from the other tokens.
    """
    B, K, D = fusion.shape
    x = F.normalize(fusion.float().reshape(B * K, D), dim=-1)
    dist = 1.0 - x @ x.t()
    slot = torch.arange(K, device=x.device).repeat(B)
    same = slot[:, None] == slot[None, :]
    eye = torch.eye(B * K, device=x.device, dtype=torch.bool)
    hardest_pos = (dist * (same & ~eye).float()).max(1).values
    hardest_neg = dist.masked_fill(same, 10.0).min(1).values
    return F.relu(hardest_pos - hardest_neg + margin).mean()


class FusionObjective(nn.Module):
    """L_L (``lambda_uni = 0``) and the CoVeR objective of Eq. (6) (``lambda_uni > 0``).

    Eq. (6): L = lambda_task * [l(y_tilde, y) + lambda_uni * L_uni] + lambda_con * L_con + lambda_reg * L_reg.
    """

    def __init__(self, lambda_task: float = 10.0, lambda_con: float = 1.0, lambda_reg: float = 1.0,
                 lambda_uni: float = 0.0, regression: str = "mse", margin: float = 1.0):
        super().__init__()
        assert regression in ("mse", "l1")
        self.lambda_task, self.lambda_con, self.lambda_reg, self.lambda_uni = lambda_task, lambda_con, lambda_reg, lambda_uni
        self.regression, self.margin = regression, margin

    def _err(self, pred, target):
        return (pred - target).abs() if self.regression == "l1" else (pred - target) ** 2

    def unimodal_loss(self, out, batch) -> torch.Tensor:
        """Eq. (5): L_uni = (1 / sum_ij m_ij) * sum_i sum_j m_ij * l(y_hat_i^(j), y_i^(j)).

        The mask m_ij restricts the loss to observed modalities, so a masked modality supplies neither an input nor
        a unimodal target at that visit.
        """
        valid = out["view_valid"]  # [3, B]
        total = torch.zeros((), device=valid.device)
        for j, m in enumerate(MODALITIES):
            total = total + (self._err(out["y_views"][m], batch[f"y_{m}"]) * valid[j]).sum()
        return total / valid.sum().clamp(min=1)

    def forward(self, out, batch):
        y = batch["y"]
        err = self._err(out["y_tilde"], y)
        w = batch.get("w")  # optional per-sample weights (conflict re-weighting)
        parts = {
            "task": err.mean() if w is None else (err * w).sum() / w.sum().clamp(min=1e-6),
            "con": F.mse_loss(out["y_main"], out["y_aux"]),
            "reg": token_diversity_loss(out["fusion"], self.margin),
        }
        total = self.lambda_task * parts["task"] + self.lambda_con * parts["con"] + self.lambda_reg * parts["reg"]
        if self.lambda_uni > 0:
            parts["uni"] = self.unimodal_loss(out, batch)
            total = total + self.lambda_task * self.lambda_uni * parts["uni"]
        parts["total"] = total
        return total, {k: float(v.detach()) for k, v in parts.items()}
