#!/usr/bin/env python
"""Where the decision collapses: prior-centred reliance per decoder layer and conflict-detection AUROC of a run.

    python scripts/diagnose.py --run results/runs/mosi/llm_fusion_default_s42
    python scripts/plot_reliance.py --runs results/runs/mosi/llm_fusion_default_s4* --out results/figures/reliance_mosi.pdf
"""
import argparse
import json
import os
from types import SimpleNamespace

import torch

import _path  # noqa: F401
from cover import mrbench_a
from cover.data import SentimentDataset
from cover.diagnosis import diagnose, late_layer_reliance
from cover.train import build_model, load_checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--pattern", type=int, nargs=3, default=[1, 1, 1], help="availability pattern (t a v)")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--mrbench_dir", default=None)
    ap.add_argument("--llm_path", default=None)
    args = ap.parse_args()
    with open(os.path.join(args.run, "config.json")) as f:
        cfg = SimpleNamespace(**json.load(f))
    cfg.data_dir, cfg.mrbench_dir = args.data_dir or cfg.data_dir, args.mrbench_dir or cfg.mrbench_dir
    cfg.llm_path = args.llm_path or cfg.llm_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = SentimentDataset(os.path.join(cfg.data_dir, f"{cfg.dataset}.pt"), "test")
    rows = mrbench_a.load(cfg.mrbench_dir, cfg.dataset, "test", ds.ids)
    model = build_model(cfg, ds.meta, device)
    load_checkpoint(model, os.path.join(args.run, "best.pt"), device)
    res = diagnose(model, ds, rows, cfg.eval_batch, device, args.pattern)
    for case, r in res["reliance"].items():
        print(f"[{case}] n={res['n'][case]} " + " ".join(
            f"rho_{m}: {late_layer_reliance(v)['early']:.2f} -> {late_layer_reliance(v)['late']:.2f}" for m, v in r.items()))
    print(f"AUROC conflict vs consensus: state discrepancy {res['auroc']['state']:.2f}, "
          f"prediction disagreement {res['auroc']['prediction']:.2f}")
    dst = os.path.join(args.run, f"diagnosis_{res['pattern']}.json")
    with open(dst, "w") as f:
        json.dump(res, f, indent=1)
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
