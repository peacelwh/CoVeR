"""MR-Bench-A: every sample typed by its unimodal judgments (Sec. 2.1, Sec. 2.2 and supplementary Sec. B).

Every sample receives a judgment u^(j) per modality that is independent of the model under study: the human unimodal
labels on CH-SIMS / CH-SIMSv2 and, on the other corpora, the predictions of single-modality BiLSTM experts (three
seeds, out-of-fold on the training split). A judgment is confident if |u^(j)| >= kappa (0.2 on a [-1, 1] label scale,
0.5 on the wider scales of CMU-MOSI / CMU-MOSEI). With polarity p(.) = sgn(.), among samples with p(y) != 0:

    consensus      p(u_t) = p(u_a) = p(u_v) = p(y)
    conflict       the text judgment is confidently wrong and an audio or visual judgment is confidently right
    weak_conflict  the text judgment is right and an audio or visual judgment is confidently wrong
    other          otherwise

A conflict case on which a model predicts the polarity of the text judgment is a language-shortcut error; ShortErr is
the fraction of conflict cases on which a model makes it.

Annotation files: ``<dir>/<dataset>_<split>.jsonl``, one object per sample:
    id                  sample identifier of the processed corpus
    case_type           consensus | conflict | weak_conflict | other
    label_polarity      p(y)
    polarity            {"t", "a", "v"}: p(u^(j))
    shortcut_polarity   p(u_t), the answer of a model that follows the words
    unimodal            {"t", "a", "v"}: expert judgments on the label scale (corpora without human unimodal labels)
"""
import json
import os
from typing import Dict, List, Sequence

import numpy as np

from cover.metrics import polarity

CASE_TYPES = ("consensus", "conflict", "weak_conflict", "other")


def confidence_threshold(label_range: Sequence[float]) -> float:
    """kappa: 0.2 on a [-1, 1] label scale (CH-SIMS, CH-SIMSv2), 0.5 on wider scales (CMU-MOSI, CMU-MOSEI)."""
    half_range = (label_range[1] - label_range[0]) / 2.0
    return 0.2 if half_range <= 1 else 0.5


def type_cases(y, u_t, u_a, u_v, kappa: float) -> np.ndarray:
    """Case type of every sample from its label ``y`` and unimodal judgments (all on the zero-centred label scale)."""
    y, u_t, u_a, u_v = (np.asarray(x, dtype=np.float64) for x in (y, u_t, u_a, u_v))
    py, pt, pa, pv = polarity(y), polarity(u_t), polarity(u_a), polarity(u_v)
    ct, ca, cv = np.abs(u_t) >= kappa, np.abs(u_a) >= kappa, np.abs(u_v) >= kappa
    labelled = py != 0
    t_right = (pt == py) & labelled
    t_wrong = (pt != py) & (pt != 0) & ct & labelled
    a_right, v_right = (pa == py) & ca & labelled, (pv == py) & cv & labelled
    a_wrong, v_wrong = (pa != py) & (pa != 0) & ca & labelled, (pv != py) & (pv != 0) & cv & labelled
    types = np.array(["other"] * len(y), dtype=object)
    types[(pt == py) & (pa == py) & (pv == py) & labelled] = "consensus"
    types[t_wrong & (a_right | v_right)] = "conflict"
    types[t_right & (a_wrong | v_wrong)] = "weak_conflict"
    return types


def annotate(ids, y, u_t, u_a, u_v, kappa: float, keep_judgments: bool) -> List[dict]:
    """Annotation rows of one split. ``keep_judgments``: store the judgments themselves (expert-judged corpora)."""
    types = type_cases(y, u_t, u_a, u_v, kappa)
    py, pt, pa, pv = polarity(y), polarity(u_t), polarity(u_a), polarity(u_v)
    rows = []
    for i, sid in enumerate(ids):
        row = {"id": str(sid), "case_type": str(types[i]), "label_polarity": int(py[i]),
               "polarity": {"t": int(pt[i]), "a": int(pa[i]), "v": int(pv[i])}, "shortcut_polarity": int(pt[i])}
        if keep_judgments:
            row["unimodal"] = {"t": round(float(u_t[i]), 4), "a": round(float(u_a[i]), 4), "v": round(float(u_v[i]), 4)}
        rows.append(row)
    return rows


def path_of(directory: str, dataset: str, split: str) -> str:
    return os.path.join(directory, f"{dataset}_{split}.jsonl")


def save(rows: List[dict], directory: str, dataset: str, split: str) -> str:
    os.makedirs(directory, exist_ok=True)
    dst = path_of(directory, dataset, split)
    with open(dst, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return dst


def load(directory: str, dataset: str, split: str, ids: Sequence[str] = None) -> List[dict]:
    """Annotation rows of one split; with ``ids``, re-ordered to (and checked against) the processed corpus."""
    with open(path_of(directory, dataset, split), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if ids is None:
        return rows
    by_id = {r["id"]: r for r in rows}
    missing = [i for i in ids if str(i) not in by_id]
    if missing or len(by_id) != len(rows):
        raise ValueError(f"MR-Bench-A {dataset}/{split} does not match the processed corpus "
                         f"({len(missing)} samples without annotation, e.g. {missing[:3]})")
    return [by_id[str(i)] for i in ids]


def counts(rows: List[dict]) -> Dict[str, int]:
    out = {t: sum(r["case_type"] == t for r in rows) for t in CASE_TYPES}
    out["samples"] = len(rows)
    return out


def unimodal_labels(rows: List[dict]) -> Dict[str, np.ndarray]:
    """Expert judgments of the rows as unimodal labels y^(j) (label scale)."""
    return {m: np.asarray([r["unimodal"][m] for r in rows], dtype=np.float32) for m in ("t", "a", "v")}


def case_metrics(pred, rows: List[dict]) -> Dict[str, float]:
    """MR-Bench-A metrics of predictions ``pred`` (zero-centred label scale, aligned with ``rows``): Acc2 on
    consensus (Cons) and conflict (Conf) cases and ShortErr, as fractions, with the case counts."""
    pp = polarity(pred)
    types = np.array([r["case_type"] for r in rows])
    label = np.array([r["label_polarity"] for r in rows])
    shortcut = np.array([r["shortcut_polarity"] for r in rows])
    cons, conf = types == "consensus", types == "conflict"
    out = {"n_consensus": int(cons.sum()), "n_conflict": int(conf.sum())}
    out["Cons"] = float(np.mean(pp[cons] == label[cons])) if cons.any() else float("nan")
    out["Conf"] = float(np.mean(pp[conf] == label[conf])) if conf.any() else float("nan")
    out["ShortErr"] = float(np.mean(pp[conf] == shortcut[conf])) if conf.any() else float("nan")
    return out
