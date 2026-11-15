#!/usr/bin/env python
"""Build the MR-Bench-A annotations of a corpus: unimodal judgments and case types for every sample.

CH-SIMS / CH-SIMSv2 use their human unimodal labels. The other corpora use single-modality BiLSTM experts (three
seeds; out-of-fold on the training split), whose judgments are stored with the annotations and also serve as the
unimodal labels of CoVeR's judges.

    python scripts/build_mrbench_a.py --dataset sims
    python scripts/build_mrbench_a.py --dataset mosi
    # expert judgments of the training split under permanent missingness (visible samples only):
    python scripts/build_mrbench_a.py --dataset mosi --missing permanent --out_dir data/mrbench_a_permanent
"""
import argparse
import json
import os

import torch

import _path  # noqa: F401
from cover import mrbench_a
from cover.experts import expert_judgments, permanent_expert_judgments
from cover.recipes import DATASETS, HUMAN_UNIMODAL


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--out_dir", default="data/mrbench_a")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--missing", default="dynamic", choices=["dynamic", "permanent", "permanent_incomplete"])
    ap.add_argument("--mask_seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(os.path.join(args.data_dir, f"{args.dataset}.pt"), map_location="cpu", weights_only=False)
    kappa = mrbench_a.confidence_threshold(blob["meta"]["label_range"])

    if args.missing != "dynamic":
        assert args.dataset not in HUMAN_UNIMODAL, "corpora with human unimodal labels need no expert judgments"
        u = permanent_expert_judgments(blob, device, args.missing, args.mask_seed, args.seeds, args.kfold, args.epochs)
        rows = [{"id": str(sid), "unimodal": {m: round(float(u[m][i]), 4) for m in ("t", "a", "v")}}
                for i, sid in enumerate(blob["train"]["id"])]
        print("saved ->", mrbench_a.save(rows, args.out_dir, args.dataset, "train"))
        return

    human = blob["meta"]["has_unimodal_labels"]
    judgments = None if human else expert_judgments(blob, device, args.seeds, args.kfold, args.epochs)
    stats = {"dataset": args.dataset, "kappa": kappa, "judgments": "human unimodal labels" if human else "out-of-fold experts"}
    for split in ("train", "valid", "test"):
        d = blob[split]
        if human:
            u_t, u_a, u_v = d["y_t"].numpy(), d["y_a"].numpy(), d["y_v"].numpy()
        else:
            u_t, u_a, u_v = (judgments[split][m] for m in ("t", "a", "v"))
        rows = mrbench_a.annotate(d["id"], d["y"].numpy(), u_t, u_a, u_v, kappa, keep_judgments=not human)
        mrbench_a.save(rows, args.out_dir, args.dataset, split)
        stats[split] = mrbench_a.counts(rows)
        print(f"[{args.dataset}/{split}] {stats[split]}")
    with open(os.path.join(args.out_dir, f"{args.dataset}_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)


if __name__ == "__main__":
    main()
