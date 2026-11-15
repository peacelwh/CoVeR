"""Diagnosis of dominant-modality collapse (Sec. 3.2 of the paper).

Prior-centred reliance (Eq. 2): since the prompt and the special tokens dominate the states, raw cosine similarities
between views are close to one at every layer; we measure how much the content-driven part of the fused state aligns
with that of a single-modality view,

    rho_j^l = cos( h_bar^f_l - h_bar^0_l ,  h_bar^j_l - h_bar^0_l ),

where h_bar^V_l is the mean over the fusion tokens of the layer-l states of view V and 0 is the content-free view.
rho -> 1 means that the content-driven components become directionally aligned; it is not a measure of causal
contribution.

Conflict signals: AUROC for separating conflict from consensus cases with the hidden-state discrepancy
d = 1 - cos(h^f, h^t) and with the prediction-level disagreement between the audio / visual and the text views.
"""
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cover.data import MODALITIES, collate, to_device
from cover.metrics import auroc, polarity

CASES = ("consensus", "conflict", "all")


@torch.no_grad()
def diagnose(model, dataset, rows: List[dict], batch_size: int, device, pattern: Sequence[int] = (1, 1, 1),
             num_workers: int = 2) -> Dict[str, dict]:
    """Run the full, single-modality and content-free views of every sample under ``pattern`` and report

    ``reliance``    {case: {j: [L + 1]}}  prior-centred reliance rho_j^l per layer (embeddings first), averaged over
                    the consensus cases, the conflict cases and all samples
    ``auroc``       {"state": AUROC of d = 1 - cos(h^f, h^t), "prediction": AUROC of the view disagreement}
    ``disagreement`` share of conflict / consensus cases on which the visual-only view disagrees in polarity with
                    the text-only view

    ``rows``: MR-Bench-A annotations aligned with ``dataset``.
    """
    model.eval()
    dataset.set_pattern(list(pattern))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)
    rho, idx, d_state, views = {m: [] for m in MODALITIES}, [], [], {m: [] for m in MODALITIES}
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model({m: batch[m] for m in MODALITIES}, batch["mask"], views="all", return_hidden=True)
        fs, fe = out["layout"]["fusion"]
        B, V = batch["mask"].shape[0], len(out["layout"]["views"])  # full, t, a, v, cf
        per_layer = []
        for h in out["hidden_states"]:
            H = h[:, fs:fe, :].float().mean(1).view(V, B, -1)
            per_layer.append(torch.stack([F.cosine_similarity(H[0] - H[4], H[1 + j] - H[4], dim=-1) for j in range(3)], 0))
        per_layer = torch.stack(per_layer, 0).cpu().numpy()  # [L + 1, 3, B]
        for j, m in enumerate(MODALITIES):
            rho[m].append(per_layer[:, j, :])
            views[m].append(dataset.denorm_label(out["y_views"][m].float()).cpu().numpy())
        H = out["hidden_states"][-1][:, fs:fe, :].float().mean(1).view(V, B, -1)
        d_state.append((1.0 - F.cosine_similarity(H[0], H[1], dim=-1)).cpu().numpy())
        idx.append(batch["idx"].cpu().numpy())
    order = np.argsort(np.concatenate(idx))
    types = np.array([r["case_type"] for r in rows])
    sel = {"consensus": types == "consensus", "conflict": types == "conflict", "all": np.ones(len(types), bool)}
    reliance = {}
    for case in CASES:
        if sel[case].any():
            reliance[case] = {m: np.concatenate(rho[m], 1)[:, order][:, sel[case]].mean(1).round(4).tolist() for m in MODALITIES}
    t, a, v = (np.concatenate(views[m])[order] for m in MODALITIES)
    d = np.concatenate(d_state)[order]
    both = sel["consensus"] | sel["conflict"]
    disagreement = np.maximum(-a * np.sign(t), -v * np.sign(t))  # how strongly a non-text view opposes the text view
    v_vs_t = polarity(v) * polarity(t) < 0
    return {
        "pattern": "".join(m.upper() for m, b in zip(MODALITIES, pattern) if b),
        "n": {case: int(sel[case].sum()) for case in CASES},
        "reliance": reliance,
        "auroc": {"state": auroc(d[both], sel["conflict"][both]), "prediction": auroc(disagreement[both], sel["conflict"][both])},
        "disagreement": {case: float(np.mean(v_vs_t[sel[case]])) for case in ("conflict", "consensus") if sel[case].any()},
    }


def late_layer_reliance(values: Sequence[float]) -> Dict[str, float]:
    """Mean reliance over the first two thirds and over the last third of the decoder layers."""
    L = len(values) - 1  # the first entry is the embedding output
    cut = 1 + (2 * L) // 3
    return {"early": float(np.mean(values[1:cut])), "late": float(np.mean(values[cut:]))}
