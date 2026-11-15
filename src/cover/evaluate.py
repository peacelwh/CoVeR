"""Evaluation protocol (Sec. 4.1): every availability pattern and random missing rates on the complete test split.

Acc2, weighted F1, MAE and Pearson correlation are reported per pattern and averaged over the seven patterns (Avg) or
over the missing rates r in {0.1, ..., 0.7} (Rate); the MR-Bench-A metrics use all modalities observed.

Inference modes of a trained model (Sec. 4.3):
    ``accuracy``  the prediction of the full view in one forward pass (architecture and inference cost of LLM-Fusion)
    ``robust``    the gated prediction of Eqs. (3, 4) with a test-time margin tau: disagreeing views override the
                  fused decision. Applied to a trained LLM-Fusion this is the test-time override of Sec. 4.2.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from cover import mrbench_a
from cover.data import MISSING_RATES, MODALITIES, PATTERNS, SentimentDataset, collate, pattern_name, to_device
from cover.metrics import average_metrics, polarity, regression_metrics

MODES = ("accuracy", "robust")


@torch.no_grad()
def predict(model, dataset: SentimentDataset, batch_size: int, device, mode: str = "accuracy",
            tau: Optional[float] = None, views: Optional[str] = None, num_workers: int = 2) -> Dict[str, np.ndarray]:
    """Run ``model`` over ``dataset`` under its current availability protocol. Returns per-sample arrays on the
    original label scale: ``pred`` (and ``pred_t`` / ``pred_a`` / ``pred_v`` [/ ``pred_cf``] when views are run)."""
    assert mode in MODES, mode
    if mode == "robust":
        assert tau is not None, "robust mode needs the test-time margin tau"
        views = views or "judges"
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)
    rec: Dict[str, List[np.ndarray]] = {}

    def add(key, value):
        rec.setdefault(key, []).append(value.float().cpu().numpy() if isinstance(value, torch.Tensor) else value)

    for batch in loader:
        batch = to_device(batch, device)
        feats = {m: batch[m] for m in MODALITIES}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(feats, batch["mask"], views=views, tau=tau if mode == "robust" else None)
        rec.setdefault("idx", []).append(batch["idx"].cpu().numpy())
        add("y_raw", batch["y_raw"])
        add("mask", batch["mask"])
        add("pred", dataset.denorm_label(out["y_tilde" if mode == "robust" else "y"].float()))
        if "y_views" in out:
            for m in MODALITIES:
                add(f"pred_{m}", dataset.denorm_label(out["y_views"][m].float()))
        if "y_cf" in out:
            add("pred_cf", dataset.denorm_label(out["y_cf"].float()))
    return {k: np.concatenate(v) for k, v in rec.items()}


def evaluate_patterns(model, dataset: SentimentDataset, batch_size: int, device, mode: str = "accuracy",
                      tau: Optional[float] = None, patterns: Sequence[Sequence[int]] = None, num_workers: int = 2,
                      keep_records: bool = False, views: Optional[str] = None):
    """Metrics under each availability pattern and their average over the patterns ('Avg')."""
    patterns = [list(p) for p in (patterns or PATTERNS)]
    results, records = {}, {}
    for pat in patterns:
        dataset.set_pattern(pat)
        r = predict(model, dataset, batch_size, device, mode=mode, tau=tau, views=views, num_workers=num_workers)
        name = pattern_name(pat)
        results[name] = regression_metrics(r["pred"], r["y_raw"])
        if keep_records:
            records[name] = r
    results["Avg"] = average_metrics([results[pattern_name(p)] for p in patterns])
    return (results, records) if keep_records else results


def evaluate_rates(model, dataset: SentimentDataset, batch_size: int, device, mode: str = "accuracy",
                   tau: Optional[float] = None, rates: Sequence[float] = MISSING_RATES, seed: int = 12345,
                   num_workers: int = 2, mask_block: int = 32):
    """Metrics under random missing rates and their average over the rates ('Rate'). The masks of a rate are fixed
    by (seed, mask_block, num_workers), see :func:`cover.data.rate_masks`."""
    results = {}
    for r in rates:
        dataset.set_rate(r, seed, block=mask_block, streams=max(num_workers, 1))
        rec = predict(model, dataset, batch_size, device, mode=mode, tau=tau, num_workers=num_workers)
        results[f"rate_{r:.1f}"] = regression_metrics(rec["pred"], rec["y_raw"])
    results["Rate"] = average_metrics([results[f"rate_{r:.1f}"] for r in rates])
    return results


def evaluate_permanent(model, dataset: SentimentDataset, batch_size: int, device, mode: str = "accuracy",
                       tau: Optional[float] = None, num_workers: int = 2):
    """Permanent-missingness validation: every sample is scored once, under its own permanent pattern. 'Avg' holds
    the metrics pooled over the split (the model-selection score)."""
    rec = predict(model, dataset, batch_size, device, mode=mode, tau=tau, num_workers=num_workers)
    return {"Avg": regression_metrics(rec["pred"], rec["y_raw"])}


def evaluate_mrbench_a(model, dataset: SentimentDataset, rows: List[dict], batch_size: int, device,
                       mode: str = "accuracy", tau: Optional[float] = None, num_workers: int = 2) -> Dict[str, float]:
    """MR-Bench-A statistics of a model with all modalities observed: Acc2 on consensus (Cons) and conflict (Conf)
    cases, ShortErr, and the conflict Acc2 of the same model without text (``Conf_wo_t``) and of its visual-only
    view (``Conf_by_v``). ``rows``: the annotations of the split, aligned with ``dataset``."""
    dataset.set_pattern([1, 1, 1])
    rec = predict(model, dataset, batch_size, device, mode=mode, tau=tau, views="judges", num_workers=num_workers)
    order = np.argsort(rec["idx"])
    out = mrbench_a.case_metrics(rec["pred"][order], rows)
    conflict = np.array([r["case_type"] == "conflict" for r in rows])
    label = np.array([r["label_polarity"] for r in rows])
    if conflict.any():
        out["Conf_by_v"] = float(np.mean(polarity(rec["pred_v"][order])[conflict] == label[conflict]))
        dataset.set_pattern([0, 1, 1])
        rec_av = predict(model, dataset, batch_size, device, mode=mode, tau=tau, num_workers=num_workers)
        pred_av = rec_av["pred"][np.argsort(rec_av["idx"])]
        out["Conf_wo_t"] = float(np.mean(polarity(pred_av)[conflict] == label[conflict]))
    return out


def summarize(results: Dict[str, dict], keys=("Acc2", "F1", "MAE", "Corr")) -> str:
    return "\n".join(f"  {name:>9s}: " + "  ".join(f"{k}={m[k]:.4f}" for k in keys if k in m)
                     for name, m in results.items())
