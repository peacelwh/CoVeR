"""MR-Bench-H: fusion-model predictions and scoring on the 900 human-validated clips (Sec. 2.3, supplementary Sec. C).

MR-Bench-H contains 620 language-shortcut (LS) and 280 non-verbal-shortcut (NS) clips from seven sources. Every clip
asks whether the speaker's sentiment is positive or negative. Every model is scored on all 900 clips; an answer that
is not a valid polarity counts as wrong, and ShortErr is the share of LS clips answered with the misleading polarity
of the words.

A fusion model reads features, so each clip is scored, with all modalities observed, by the model trained on the
training split of its source corpus; no benchmark clip is used to update a model. Fusion results are the mean and
standard deviation over seeds of the accuracy pooled over the seven sources.

Prediction files are JSONL with one object per (clip, view): ``uid``, ``model``, ``view``, ``seed``, ``score``
(prediction minus the neutral point, on the source label scale) and ``polarity``. The same files are accepted by the
scorer shipped with the benchmark (``tools/evaluate.py``).
"""
import gzip
import json
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

from cover.evaluate import predict
from cover.metrics import wilson_interval

#: views of a feature-input model = availability patterns (t, a, v)
VIEWS = {"full": [1, 1, 1], "transcript": [1, 0, 0], "nonverbal": [0, 1, 1], "audio": [0, 1, 0], "muted": [0, 0, 1]}
SOURCES = ("mosi", "mosei", "sims", "simsv2", "mustardpp", "meld", "iemocap")


def read_jsonl(path: str) -> List[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_items(path: str, sources: Optional[Iterable[str]] = None) -> Dict[str, dict]:
    """The released item list (``data/test.jsonl`` of the benchmark), keyed by uid."""
    items = {}
    for it in read_jsonl(path):
        if sources is None or it["source"] in sources:
            items[it["uid"]] = it
    return items


def processed_split(item: dict) -> str:
    """Split of the processed corpus that holds the features of a clip. The clips of IEMOCAP Sessions 1-4, of the
    MELD development split and of MUStARD++ are held out of training in the 'pool' split."""
    src, split = item["source"], str(item["source_split"])
    if src == "iemocap":
        return "test" if split.lower().replace(" ", "") in ("session5", "test") else "pool"
    if src == "meld":
        return "test" if split == "test" else "pool"
    if src == "mustardpp":
        return "pool"
    return split


def _polarity(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def subset(dataset, idx):
    """Restrict a :class:`cover.data.SentimentDataset` to the samples ``idx`` (in place)."""
    ix = torch.as_tensor(idx, dtype=torch.long)
    dataset.t, dataset.a, dataset.v, dataset.y = dataset.t[ix], dataset.a[ix], dataset.v[ix], dataset.y[ix]
    dataset.y_uni = {m: y[ix] for m, y in dataset.y_uni.items()}
    dataset.ids = [dataset.ids[i] for i in idx]
    return dataset


def predict_items(model, data_path: str, items: List[dict], device, name: str, seed: int, mode: str = "accuracy",
                  tau: Optional[float] = None, views: Iterable[str] = ("full",), batch_size: int = 32,
                  predict_fn=None) -> List[dict]:
    """Predictions of one source-trained fusion model on the benchmark clips of its source.

    ``items``: the clips of this source; ``views``: names in :data:`VIEWS`. ``predict_fn`` replaces
    :func:`cover.evaluate.predict` (used by the steering comparator)."""
    from cover.data import SentimentDataset
    rows = []
    for split in sorted({processed_split(it) for it in items}):
        sel = [it for it in items if processed_split(it) == split]
        ds = SentimentDataset(data_path, split)
        position = {sid: k for k, sid in enumerate(ds.ids)}
        missing = [it["source_id"] for it in sel if str(it["source_id"]) not in position]
        if missing:
            raise KeyError(f"{len(missing)} benchmark clips not found in {data_path}[{split}], e.g. {missing[:3]}")
        subset(ds, [position[str(it["source_id"])] for it in sel])
        for view in views:
            ds.set_pattern(VIEWS[view])
            if predict_fn is not None:
                rec = predict_fn(model, ds)
            else:
                rec = predict(model, ds, batch_size, device, mode=mode, tau=tau)
            pred = np.empty(len(ds))
            pred[rec["idx"]] = rec["pred"]
            for k, it in enumerate(sel):
                rows.append({"uid": it["uid"], "model": name, "view": view, "seed": seed,
                             "score": round(float(pred[k]), 4), "polarity": _polarity(float(pred[k]))})
    return rows


def write_predictions(rows: List[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _score_one(pred: Dict[str, int], items: Dict[str, dict], uids: List[str]) -> dict:
    k = sum(pred.get(u) == items[u]["label_polarity"] for u in uids)
    short = sum(pred.get(u) == items[u]["distractor_polarity"] for u in uids)
    n = len(uids)
    return {"k": k, "n": n, "acc": 100 * k / n, "short": 100 * short / n}


def score(items: Dict[str, dict], rows: Iterable[dict]) -> Dict[str, dict]:
    """Score predictions on the released clips.

    Per (model, view): LS and NS accuracy (%), the 95% Wilson interval of the LS accuracy, LS ShortErr (%) and the
    LS accuracy per source. With several seeds: mean and (population) standard deviation over seeds of the pooled
    per-seed accuracies. A missing or invalid answer counts as wrong (``missing``: clips without a prediction).
    """
    LS = [u for u, it in items.items() if it["category"] == "L"]
    NS = [u for u, it in items.items() if it["category"] == "V"]
    preds = defaultdict(lambda: defaultdict(dict))  # (model, view) -> seed -> uid -> polarity
    for r in rows:
        if r["uid"] in items:
            preds[(r["model"], r["view"])][r.get("seed", 0)][r["uid"]] = r["polarity"]
    out = {}
    for (model, view), by_seed in sorted(preds.items()):
        ls = [_score_one(p, items, LS) for _, p in sorted(by_seed.items())]
        ns = [_score_one(p, items, NS) for _, p in sorted(by_seed.items())]
        rec = {"seeds": sorted(by_seed), "LS": {}, "NS": {},
               "missing": max(sum(u not in p for u in items) for p in by_seed.values())}
        for key, runs, uids in (("LS", ls, LS), ("NS", ns, NS)):
            accs = [r["acc"] for r in runs]
            rec[key] = {"n": len(uids), "acc": float(np.mean(accs)), "std": float(np.std(accs)),
                        "ShortErr": float(np.mean([r["short"] for r in runs]))}
        lo, hi = wilson_interval(round(float(np.mean([r["k"] for r in ls]))), len(LS))
        rec["LS"]["ci95"] = [lo, hi]
        rec["LS"]["by_source"] = {}
        for src in SOURCES:
            uids = [u for u in LS if items[u]["source"] == src]
            if uids:
                rec["LS"]["by_source"][src] = float(np.mean([_score_one(p, items, uids)["acc"] for p in by_seed.values()]))
        out[f"{model}|{view}"] = rec
    return out


def text_override_rate(items: Dict[str, dict], rows: Iterable[dict], model: str,
                       without_text: str = "nonverbal", with_text: str = "full") -> dict:
    """Paired views of one fusion model: among the LS clips it solves from audio and video alone, the share made
    wrong by adding the text (per seed, then mean and standard deviation over seeds)."""
    LS = [u for u, it in items.items() if it["category"] == "L"]
    preds = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r["model"] == model and r["uid"] in items:
            preds[r["view"]][r.get("seed", 0)][r["uid"]] = r["polarity"]
    rates = []
    for seed in sorted(preds[with_text]):
        av, full = preds[without_text].get(seed, {}), preds[with_text][seed]
        solved = [u for u in LS if av.get(u) == items[u]["label_polarity"]]
        flipped = [u for u in solved if full.get(u) != items[u]["label_polarity"]]
        rates.append(100 * len(flipped) / max(len(solved), 1))
    return {"rate": float(np.mean(rates)), "std": float(np.std(rates)), "seeds": len(rates)}
