"""Metrics of the paper (Sec. 4.1 and supplementary Sec. A, "Metrics")."""
import math

import numpy as np


def polarity(y, neutral_band: float = 0.0) -> np.ndarray:
    """Polarity p(y) = sgn(y) in {-1, 0, +1} of scores on a zero-centred label scale."""
    y = np.asarray(y, dtype=np.float64)
    p = np.zeros(y.shape, dtype=np.int64)
    p[y > neutral_band] = 1
    p[y < -neutral_band] = -1
    return p


def regression_metrics(preds, labels) -> dict:
    """Acc2, weighted F1, MAE and Pearson correlation of sentiment scores on the original label scale.

    Acc2 and F1 exclude neutral labels (label == 0); F1 is scikit-learn's weighted F1 with the true labels as
    reference.
    """
    from sklearn.metrics import f1_score

    preds = np.asarray(preds, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    out = {"MAE": float(np.mean(np.abs(preds - labels)))}
    if np.std(preds) < 1e-8 or np.std(labels) < 1e-8:
        out["Corr"] = 0.0
    else:
        out["Corr"] = float(np.corrcoef(preds, labels)[0][1])
    keep = labels != 0
    if keep.any():
        p, l = preds[keep] > 0, labels[keep] > 0
        out["Acc2"] = float(np.mean(p == l))
        out["F1"] = float(f1_score(l, p, average="weighted"))
    else:
        out["Acc2"] = out["F1"] = float("nan")
    return out


def fisher_mean(correlations) -> float:
    """Average of correlations through Fisher's z-transform: tanh(mean(arctanh(r_k)))."""
    rs = [min(max(float(r), -0.999999), 0.999999) for r in correlations]
    return float(math.tanh(float(np.mean([math.atanh(r) for r in rs]))))


def average_metrics(rows) -> dict:
    """Average a list of metric dicts (over availability patterns or missing rates): arithmetic mean, except for
    the correlation, which is averaged through Fisher's z-transform."""
    return {k: fisher_mean([r[k] for r in rows]) if k == "Corr" else float(np.nanmean([r[k] for r in rows]))
            for k in rows[0]}


def wilson_interval(k: float, n: int, z: float = 1.96):
    """95% Wilson interval of an accuracy k / n, in percent."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (centre - half), 100 * (centre + half)


def auroc(scores, positives) -> float:
    """Area under the ROC curve of `scores` for the binary indicator `positives` (rank statistic, ties averaged)."""
    scores = np.asarray(scores, dtype=np.float64)
    positives = np.asarray(positives, dtype=bool)
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
