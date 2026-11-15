"""Single-modality experts: unimodal judgments for corpora without human unimodal labels (supplementary Sec. A).

Each expert is a LayerNorm, a bidirectional LSTM (hidden size 128) with mean pooling and a two-layer MLP head
(dropout 0.3), trained with AdamW (learning rate 1e-3, weight decay 1e-3, batch 64) for 40 epochs on one modality.
Three seeds are averaged. Validation and test judgments come from experts trained on the training split; on the
training split we use five-fold out-of-fold predictions, which serve as the unimodal labels of CoVeR's judges.

Under permanent missingness the expert of a modality is trained and applied only on the samples that observe it.
"""
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cover.data import MODALITIES, permanent_masks
from cover.metrics import regression_metrics


class UnimodalExpert(nn.Module):
    def __init__(self, d_in: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.lstm = nn.LSTM(d_in, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, 1))

    def forward(self, x):
        h, _ = self.lstm(self.norm(x))
        return self.head(h.mean(1)).squeeze(-1)


def _seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_expert(blob, m: str, seed: int, device, epochs: int = 40, lr: float = 1e-3, batch: int = 64,
                 train_idx: Optional[torch.Tensor] = None, verbose: bool = True) -> Dict[str, np.ndarray]:
    """Train one expert of modality ``m`` on ``blob['train'][train_idx]`` (all samples if None), keep the epoch with
    the best validation correlation and return its predictions (label scale) for the three splits."""
    _seed(seed)
    tr, va = blob["train"], blob["valid"]
    model = UnimodalExpert(tr[m].shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    X, Y = tr[m].float(), tr["y"].float()
    if train_idx is not None:
        X, Y = X[train_idx], Y[train_idx]
    loader = DataLoader(TensorDataset(X, Y), batch_size=batch, shuffle=True)

    def predict(split):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(split["y"]), 256):
                out.append(model(split[m][i:i + 256].float().to(device)).cpu())
        return torch.cat(out).numpy()

    best, best_state, best_epoch = -1e9, None, -1
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = nn.functional.l1_loss(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        score = regression_metrics(predict(va), va["y"].numpy())["Corr"]
        if score > best:
            best, best_epoch = score, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    preds = {s: predict(blob[s]) for s in ("train", "valid", "test")}
    if verbose:
        mt = regression_metrics(preds["test"], blob["test"]["y"].numpy())
        print(f"  [{m}] seed {seed}: best epoch {best_epoch}, valid Corr {best:.3f} | test Corr {mt['Corr']:.3f} Acc2 {mt['Acc2']:.3f}")
    return preds


def expert_judgments(blob, device, seeds: Sequence[int] = (0, 1, 2), kfold: int = 5, epochs: int = 40,
                     verbose: bool = True) -> Dict[str, Dict[str, np.ndarray]]:
    """Unimodal judgments u^(j) of every split, averaged over ``seeds``.

    valid / test: experts trained on the whole training split. train: ``kfold``-fold out-of-fold predictions
    (in-sample predictions on the training split would be over-fitted and contain almost no conflicts).
    Returns {split: {modality: [N]}} on the label scale.
    """
    out = {s: {} for s in ("train", "valid", "test")}
    n_train = len(blob["train"]["y"])
    for m in MODALITIES:
        acc = {s: [] for s in out}
        for seed in seeds:
            full = train_expert(blob, m, seed, device, epochs=epochs, verbose=verbose)
            acc["valid"].append(full["valid"])
            acc["test"].append(full["test"])
            folds = np.array_split(np.random.RandomState(seed).permutation(n_train), kfold)
            oof = np.zeros(n_train, dtype=np.float32)
            for k in range(kfold):
                train_idx = np.concatenate([folds[j] for j in range(kfold) if j != k])
                preds = train_expert(blob, m, seed * 100 + k, device, epochs=epochs,
                                     train_idx=torch.tensor(train_idx), verbose=False)
                oof[folds[k]] = preds["train"][folds[k]]
            acc["train"].append(oof)
        for s in out:
            out[s][m] = np.mean(acc[s], 0)
        if verbose:
            for s in ("train", "test"):
                mt = regression_metrics(out[s][m], blob[s]["y"].numpy())
                print(f"[{m}] {s} ({'out-of-fold' if s == 'train' else 'ensemble'}): Corr {mt['Corr']:.3f} Acc2 {mt['Acc2']:.3f}")
    return out


def _take(split, idx):
    ix = torch.as_tensor(idx, dtype=torch.long)
    return {k: (v[ix] if isinstance(v, torch.Tensor) else [v[i] for i in idx]) for k, v in split.items()}


def permanent_expert_judgments(blob, device, missing: str = "permanent", mask_seed: int = 0,
                               seeds: Sequence[int] = (0, 1, 2), kfold: int = 5, epochs: int = 40,
                               verbose: bool = True) -> Dict[str, np.ndarray]:
    """Out-of-fold training judgments under permanent missingness: the expert of modality m is trained, validated and
    applied only on the samples whose permanent pattern contains m, so the features of unobserved modalities are
    never read (the test split is not touched). Entries of unobserved modalities are 0 and never used, because the
    corresponding view is masked out. Returns {modality: [N_train]}."""
    incomplete = missing == "permanent_incomplete"
    n_train, n_valid = len(blob["train"]["y"]), len(blob["valid"]["y"])
    obs_train = permanent_masks(n_train, "train", mask_seed, incomplete)
    obs_valid = permanent_masks(n_valid, "valid", mask_seed, incomplete)
    out = {}
    for j, m in enumerate(MODALITIES):
        vis_train, vis_valid = np.where(obs_train[:, j] == 1)[0], np.where(obs_valid[:, j] == 1)[0]
        valid = _take(blob["valid"], vis_valid)
        sub = {"meta": blob["meta"], "train": _take(blob["train"], vis_train), "valid": valid, "test": valid}
        acc = []
        for seed in seeds:
            folds = np.array_split(np.random.RandomState(seed).permutation(len(vis_train)), kfold)
            oof = np.zeros(len(vis_train), dtype=np.float32)
            for k in range(kfold):
                train_idx = np.concatenate([folds[i] for i in range(kfold) if i != k])
                preds = train_expert(sub, m, seed * 100 + k, device, epochs=epochs, train_idx=torch.tensor(train_idx),
                                     verbose=False)
                oof[folds[k]] = preds["train"][folds[k]]
            acc.append(oof)
        oof = np.mean(acc, 0)
        if verbose:
            mt = regression_metrics(oof, sub["train"]["y"].numpy())
            print(f"[{m}] observed on {len(vis_train)}/{n_train} training samples; out-of-fold Corr {mt['Corr']:.3f}")
        full = np.zeros(n_train, dtype=np.float32)
        full[vis_train] = oof
        out[m] = full
    return out
