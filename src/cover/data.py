"""Datasets with per-sample availability masks (Sec. 2.1 and Sec. 4.1 of the paper).

A sample consists of modality features X = {X^t, X^a, X^v}, an availability mask m in {0,1}^3 \\ {0} and a sentiment
score y. Observations are incomplete at training time: a missing modality is unavailable both as an input and as a
target.

* Main protocol (``dynamic``): one of the seven non-empty patterns, the complete one included, is drawn uniformly at
  each visit of a training sample.
* Permanent missingness (``permanent`` / ``permanent_incomplete``): every training and validation sample is assigned
  one pattern once, before training, and keeps it for the whole run; the features of its unobserved modalities are
  zeroed when the sample is loaded, so no component can read them (supplementary Sec. E).
* Evaluation: every availability pattern (``fixed``) and random missing rates (``rate``) on the complete test split.
"""
import itertools
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

MODALITIES = ("t", "a", "v")
MODALITY_NAMES = {"t": "text", "a": "audio", "v": "visual"}

#: the seven non-empty availability patterns (t, a, v), the complete pattern first
PATTERNS: List[List[int]] = sorted(
    (list(c) for c in itertools.product([0, 1], repeat=3) if sum(c) > 0),
    key=lambda c: (-sum(c), [-x for x in c]))
INCOMPLETE_PATTERNS = [c for c in PATTERNS if sum(c) < 3]
MISSING_RATES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)

PERMANENT = ("permanent", "permanent_incomplete")
_SPLIT_SALT = {"train": 1, "valid": 2, "test": 3}


def pattern_name(mask: Sequence[int]) -> str:
    """[1, 0, 1] -> 'TV'."""
    return "".join(m.upper() for m, b in zip(MODALITIES, mask) if b)


def permanent_masks(n: int, split: str, mask_seed: int = 0, incomplete_only: bool = False) -> np.ndarray:
    """Permanent-missingness assignment: one availability pattern per sample, drawn once.

    The assignment is a function of the split, the number of samples and ``mask_seed`` only (it is independent of the
    training seed and of the epoch) and is balanced over the patterns: the seven non-empty patterns, or the six
    incomplete ones so that no sample is ever complete. Returns an int array [n, 3].
    """
    patterns = INCOMPLETE_PATTERNS if incomplete_only else PATTERNS
    rng = np.random.RandomState(7919 * int(mask_seed) + _SPLIT_SALT.get(split, 9) + (50 if incomplete_only else 0))
    idx = np.resize(np.arange(len(patterns)), n)
    rng.shuffle(idx)
    return np.asarray(patterns, dtype=np.int64)[idx]


def rate_masks(n: int, rate: float, seed: int, block: Optional[int] = None, streams: int = 1) -> np.ndarray:
    """Random-missing-rate evaluation: each modality is dropped independently with probability ``rate`` and at least
    one modality is kept.

    The masks are a deterministic function of the arguments. Blocks of ``block`` consecutive samples are served in
    turn by ``streams`` generators with the same seed; the evaluation of the paper uses blocks of 32 samples and two
    streams (four on the larger corpora), see ``cover.recipes``.
    """
    out = np.zeros((n, 3), dtype=np.int64)
    block = block or n
    generators = [np.random.RandomState(seed) for _ in range(max(streams, 1))]
    for b, start in enumerate(range(0, n, block)):
        rng = generators[b % len(generators)]
        for i in range(start, min(start + block, n)):
            while True:
                m = (rng.rand(3) >= rate).astype(np.int64)
                if m.sum() > 0:
                    out[i] = m
                    break
    return out


class SentimentDataset(Dataset):
    """One split of a processed corpus (``cover.preprocess``) with an availability protocol.

    missing:
      ``dynamic``              a pattern drawn uniformly from the seven non-empty ones at every visit (training)
      ``permanent``            the pattern assigned to the sample by :func:`permanent_masks` (training / validation)
      ``permanent_incomplete`` as ``permanent`` with the six incomplete patterns only
      ``fixed``                every sample uses ``fixed_mask`` (evaluation under one availability pattern)
      ``rate``                 the masks set by :meth:`set_rate` (evaluation under a random missing rate)
      ``none``                 all modalities available
    """

    def __init__(self, path: str, split: str, missing: str = "none", fixed_mask: Optional[Sequence[int]] = None,
                 seed: int = 0, mask_seed: int = 0):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.meta = blob["meta"]
        self.name = self.meta["dataset"]
        d = blob[split]
        self.split = split
        self.t, self.a, self.v = d["t"], d["a"], d["v"]
        self.y = d["y"].float()
        self.ids = [str(i) for i in d["id"]]
        #: unimodal labels y^(j) on the label scale: human annotations where the corpus has them (CH-SIMS, CH-SIMSv2);
        #: expert judgments are attached with :meth:`set_unimodal_labels`
        self.y_uni: Dict[str, torch.Tensor] = {m: d[f"y_{m}"].float() for m in MODALITIES if f"y_{m}" in d}
        self.lo, self.hi = self.meta["label_range"]
        self.missing = missing
        self.fixed_mask = list(fixed_mask) if fixed_mask is not None else None
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.masks: Optional[np.ndarray] = None  # per-sample masks of the ``rate`` protocol
        self.obs: Optional[np.ndarray] = None    # per-sample permanent patterns
        if missing in PERMANENT:
            self.obs = permanent_masks(len(self.y), split, mask_seed, incomplete_only=(missing == "permanent_incomplete"))
        self.sample_weight: Optional[torch.Tensor] = None

    def __len__(self):
        return len(self.y)

    # ------------------------------------------------------------------ labels
    def norm_label(self, y):
        """Labels are normalised to [0, 1] with the label range of the corpus, so the neutral point is c = 0.5."""
        return (y - self.lo) / (self.hi - self.lo)

    def denorm_label(self, y01):
        return y01 * (self.hi - self.lo) + self.lo

    def set_unimodal_labels(self, labels: Dict[str, Sequence[float]]):
        """Attach unimodal judgments (label scale) as the targets of the unimodal judges, e.g. the out-of-fold expert
        predictions of MR-Bench-A on CMU-MOSI / CMU-MOSEI."""
        for m in MODALITIES:
            y = torch.as_tensor(np.asarray(labels[m], dtype=np.float32))
            assert len(y) == len(self.y), f"unimodal labels of '{m}': {len(y)} values for {len(self.y)} samples"
            self.y_uni[m] = y

    # ------------------------------------------------------------------ masks
    def set_epoch(self, epoch: int):
        """Re-seed the pattern sampler of the dynamic protocol (call before iterating over an epoch)."""
        self.rng = np.random.RandomState((self.seed * 1000003 + epoch * 7919) % (2 ** 31 - 1))

    def set_pattern(self, mask: Sequence[int]):
        self.missing, self.fixed_mask = "fixed", list(mask)

    def set_rate(self, rate: float, seed: int = 12345, block: Optional[int] = None, streams: int = 1):
        self.missing, self.masks = "rate", rate_masks(len(self), rate, seed, block, streams)

    def sample_mask(self, i: int) -> List[int]:
        if self.missing in PERMANENT:
            return self.obs[i].tolist()
        if self.missing == "none":
            return [1, 1, 1]
        if self.missing == "fixed":
            return list(self.fixed_mask)
        if self.missing == "rate":
            return self.masks[i].tolist()
        if self.missing == "dynamic":
            return list(PATTERNS[self.rng.randint(len(PATTERNS))])
        raise ValueError(f"unknown missing protocol: {self.missing}")

    # ------------------------------------------------------------------ items
    def __getitem__(self, i):
        item = {"t": self.t[i].float(), "a": self.a[i].float(), "v": self.v[i].float(),
                "y": self.norm_label(self.y[i]), "y_raw": self.y[i],
                "mask": torch.tensor(self.sample_mask(i), dtype=torch.float32), "idx": i}
        permanent = self.missing in PERMANENT
        if permanent:  # the unobserved modalities of this sample do not exist
            for j, m in enumerate(MODALITIES):
                if not self.obs[i][j]:
                    item[m] = torch.zeros_like(item[m])
        for j, m in enumerate(MODALITIES):
            if m in self.y_uni:
                item[f"y_{m}"] = self.norm_label(self.y_uni[m][i])
                if permanent and not self.obs[i][j]:
                    item[f"y_{m}"] = torch.tensor(0.5)  # never used: the view of an unobserved modality is masked out
        if self.sample_weight is not None:
            item["w"] = self.sample_weight[i]
        return item


def worker_init_fn(worker_id: int):
    """Give every data-loader worker its own pattern stream."""
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    state = ds.rng.randint(2 ** 31 - 1)
    ds.rng = np.random.RandomState((state + 104729 * (worker_id + 1)) % (2 ** 31 - 1))


def collate(batch):
    out = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals)
        elif isinstance(vals[0], (int, float)):
            out[k] = torch.tensor(vals)
        else:
            out[k] = vals
    return out


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
