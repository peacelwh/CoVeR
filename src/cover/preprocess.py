"""Convert MMSA-style feature pickles into the tensors used for training (Sec. 4.1, supplementary Sec. A).

CMU-MOSI, CMU-MOSEI, CH-SIMS and CH-SIMSv2 use the standard MMSA features. Each modality sequence is resampled to
T = 50 steps by adaptive average pooling over its valid (unpadded) length; audio and visual features are z-scored with
training-split statistics. IEMOCAP, MELD and MUStARD++ use pickles in the same format (BERT / openSMILE ComParE /
ViT facial-expression features; see data/README.md).

    python scripts/preprocess.py --raw_dir data/raw --datasets mosi mosei sims simsv2

Output: ``<out_dir>/<dataset>.pt``, a dict
    meta:  dataset, label_range [lo, hi], T, dims {t, a, v}, has_unimodal_labels, norm (training statistics)
    train / valid / test [/ pool]:
           t, a, v   float16 [N, T, d]        y   float32 [N]        id   list of str
           y_t, y_a, y_v  float32 [N]  (human unimodal labels, CH-SIMS / CH-SIMSv2 only)
``pool`` holds clips that are held out of training for MR-Bench-H (IEMOCAP, MELD, MUStARD++).
"""
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F

DATASET_FILES = {
    "mosi": "MOSI/aligned_50.pkl",
    "mosei": "MOSEI/aligned_50.pkl",
    "sims": "SIMS/unaligned_39.pkl",
    "simsv2": "SIMSv2/sims_unaligned.pkl",
    "iemocap": "IEMOCAP/iemocap_50.pkl",
    "meld": "MELD/meld_50.pkl",
    "mustardpp": "MUSTARDPP/mustardpp_50.pkl",
}

#: label scale of each corpus: IEMOCAP = averaged valence (1-5) minus 3, MELD = sentiment in {-1, 0, 1},
#: MUStARD++ = (valence - 5) / 4
LABEL_RANGE = {
    "mosi": (-3.0, 3.0), "mosei": (-3.0, 3.0), "sims": (-1.0, 1.0), "simsv2": (-1.0, 1.0),
    "iemocap": (-2.0, 2.0), "meld": (-1.0, 1.0), "mustardpp": (-1.0, 1.0),
}


def resample(x, lengths, T: int) -> torch.Tensor:
    """x: [N, L, d]; lengths: valid lengths or None. Adaptive average pooling over the valid length -> [N, T, d]."""
    x = torch.from_numpy(np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0))
    N, L, d = x.shape
    out = torch.empty(N, T, d, dtype=torch.float32)
    for i in range(N):
        li = max(1, min(int(lengths[i]) if lengths is not None else L, L))
        seq = x[i, :li].T.unsqueeze(0)  # [1, d, li]
        out[i] = seq[0].T if li == T else F.adaptive_avg_pool1d(seq, T)[0].T
    return out


def convert(dataset: str, raw_dir: str, out_dir: str, T: int = 50, clip: float = 10.0) -> str:
    path = os.path.join(raw_dir, DATASET_FILES[dataset])
    print(f"[{dataset}] loading {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)

    out = {"meta": {"dataset": dataset, "label_range": list(LABEL_RANGE[dataset]), "T": T}}
    splits = ["train", "valid", "test"] + (["pool"] if "pool" in data and len(data["pool"]["regression_labels"]) else [])
    for split in splits:
        d = data[split]
        # text length: number of non-pad BERT tokens (text_bert[:, 1, :] is the attention mask)
        t_len = np.asarray(d["text_bert"])[:, 1, :].sum(1).astype(np.int64) if "text_bert" in d else None
        s = {
            "t": resample(d["text"], t_len, T).clamp(-clip * 4, clip * 4).half(),
            "a": resample(d["audio"], d.get("audio_lengths"), T),
            "v": resample(d["vision"], d.get("vision_lengths"), T),
            "y": torch.tensor(np.asarray(d["regression_labels"], dtype=np.float32)),
            "id": [str(x) for x in d["id"]],
        }
        for m in ("T", "A", "V"):
            if f"regression_labels_{m}" in d:
                s[f"y_{m.lower()}"] = torch.tensor(np.asarray(d[f"regression_labels_{m}"], dtype=np.float32))
        out[split] = s
        print(f"  {split}: N={len(s['y'])} t={tuple(s['t'].shape)} a={tuple(s['a'].shape)} v={tuple(s['v'].shape)}")

    # per-dimension z-score of the audio / visual features with training-split statistics
    out["meta"]["norm"] = {}
    for m in ("a", "v"):
        x = out["train"][m].float()
        mean = x.reshape(-1, x.shape[-1]).mean(0)
        std = x.reshape(-1, x.shape[-1]).std(0).clamp(min=1e-3)
        out["meta"]["norm"][m] = {"mean": mean, "std": std}
        for split in splits:
            out[split][m] = ((out[split][m].float() - mean) / std).clamp(-clip, clip).half()
    out["meta"]["dims"] = {m: int(out["train"][m].shape[-1]) for m in ("t", "a", "v")}
    out["meta"]["has_unimodal_labels"] = "y_t" in out["train"]
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, f"{dataset}.pt")
    torch.save(out, dst)
    print(f"[{dataset}] saved -> {dst} ({os.path.getsize(dst) / 1e9:.2f} GB)")
    return dst
