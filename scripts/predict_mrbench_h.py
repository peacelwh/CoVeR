#!/usr/bin/env python
"""Item-level predictions of a trained fusion model on the MR-Bench-H clips of its source corpus.

A fusion model reads features, so each clip is scored by the model trained on the training split of its source
corpus. Run this script once per source and seed, then score the pooled files with scripts/score_mrbench_h.py.
CoVeR is exported in accuracy mode (``cover``) and in robust mode (``cover-robust``); ``--steering`` exports the
conflict-triggered state-steering comparator of a trained LLM-Fusion (``llm_fusion-steering``).

    python scripts/predict_mrbench_h.py --run results/runs/sims/cover_s42 --items /path/to/MR-Bench-H/data/test.jsonl
"""
import argparse
import json
import os
from types import SimpleNamespace

import torch

import _path  # noqa: F401
from cover import mrbench_h, steering
from cover.data import SentimentDataset
from cover.train import build_model, load_checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory with config.json and best.pt")
    ap.add_argument("--items", required=True, help="item list of the benchmark (data/test.jsonl)")
    ap.add_argument("--out_dir", default="results/mrbench_h")
    ap.add_argument("--views", nargs="+", default=["full", "transcript", "nonverbal"], choices=list(mrbench_h.VIEWS))
    ap.add_argument("--steering", action="store_true", help="conflict-triggered state steering (full view only)")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--llm_path", default=None)
    args = ap.parse_args()
    with open(os.path.join(args.run, "config.json")) as f:
        cfg = SimpleNamespace(**json.load(f))
    cfg.data_dir, cfg.llm_path = args.data_dir or cfg.data_dir, args.llm_path or cfg.llm_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = os.path.join(cfg.data_dir, f"{cfg.dataset}.pt")
    items = list(mrbench_h.load_items(args.items, sources=[cfg.dataset]).values())
    if not items:
        raise SystemExit(f"no MR-Bench-H clip comes from '{cfg.dataset}'")
    meta = SentimentDataset(path, "test").meta
    model = build_model(cfg, meta, device)
    load_checkpoint(model, os.path.join(args.run, "best.pt"), device)

    rows = []
    if args.steering:
        st = steering.calibrate(model, SentimentDataset(path, "valid"), cfg.eval_batch, device)
        print(f"[steering] threshold {st.tau_d:.4f}, alpha {st.alpha}")
        rows += mrbench_h.predict_items(model, path, items, device, f"{cfg.method}-steering", cfg.seed, views=["full"],
                                        predict_fn=lambda _, ds: steering.predict(st, ds, cfg.eval_batch, device))
    else:
        rows += mrbench_h.predict_items(model, path, items, device, cfg.method, cfg.seed, "accuracy", None, args.views,
                                        cfg.eval_batch)
        if cfg.tau is not None:
            rows += mrbench_h.predict_items(model, path, items, device, f"{cfg.method}-robust", cfg.seed, "robust",
                                            cfg.tau, args.views, cfg.eval_batch)
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{cfg.method}{'-steering' if args.steering else ''}_{cfg.dataset}_s{cfg.seed}"
    dst = os.path.join(args.out_dir, f"{tag}.jsonl")
    mrbench_h.write_predictions(rows, dst)
    print(f"{len(rows)} predictions ({len(items)} clips) -> {dst}")


if __name__ == "__main__":
    main()
