#!/usr/bin/env python
"""Convert the feature pickles of the corpora into training tensors (T = 50 tokens per modality).

    python scripts/preprocess.py --raw_dir data/raw --datasets mosi mosei sims simsv2
"""
import argparse

import _path  # noqa: F401
from cover.preprocess import DATASET_FILES, convert

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["mosi", "mosei", "sims", "simsv2"], choices=list(DATASET_FILES))
    ap.add_argument("--raw_dir", default="data/raw", help="folder with the MMSA-style pickles (see data/README.md)")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--T", type=int, default=50)
    args = ap.parse_args()
    for ds in args.datasets:
        convert(ds, args.raw_dir, args.out_dir, args.T)
