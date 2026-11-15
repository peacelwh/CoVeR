#!/usr/bin/env python
"""Evaluate a finished run: the seven availability patterns, random missing rates and MR-Bench-A.

Accuracy mode outputs the prediction of the full view in one forward pass; robust mode applies the disagreement gate
at test time with margin tau_test. On a trained LLM-Fusion, ``--tau_test`` gives the test-time override.

    python scripts/evaluate.py --run results/runs/mosi/cover_s42
    python scripts/evaluate.py --run results/runs/mosi/llm_fusion_default_s42 --tau_test 0.3      # test-time override
    python scripts/evaluate.py --run results/runs/mosi/llm_fusion_default_s42 --steering          # activation steering
"""
import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import torch

import _path  # noqa: F401
from cover import mrbench_a
from cover.data import PATTERNS, SentimentDataset, pattern_name
from cover.metrics import average_metrics, regression_metrics
from cover.steering import ActivationSteering, predict as steer_predict
from cover.train import build_model, evaluate_run, load_checkpoint


def evaluate_steering(run, alpha, tau_d, data_dir=None, mrbench_dir=None, llm_path=None):
    """Discrepancy-gated activation steering of the run's checkpoint (fixed threshold, all restricted views)."""
    with open(os.path.join(run, "config.json")) as f:
        cfg = SimpleNamespace(**json.load(f))
    cfg.data_dir, cfg.mrbench_dir = data_dir or cfg.data_dir, mrbench_dir or cfg.mrbench_dir
    cfg.llm_path = llm_path or cfg.llm_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = SentimentDataset(os.path.join(cfg.data_dir, f"{cfg.dataset}.pt"), "test")
    model = build_model(cfg, ds.meta, device)
    load_checkpoint(model, os.path.join(run, "best.pt"), device)
    steering = ActivationSteering(model, alpha=alpha, tau_d=tau_d)
    res, fired = {"patterns": {}}, []
    for pat in PATTERNS:
        ds.set_pattern(pat)
        rec = steer_predict(steering, ds, cfg.eval_batch, device)
        res["patterns"][pattern_name(pat)] = regression_metrics(rec["pred"], rec["y_raw"])
        if pat[0]:  # text-present patterns: how often the text gate fires
            fired.append(rec["fired_t"])
        if pat == [1, 1, 1] and os.path.exists(mrbench_a.path_of(cfg.mrbench_dir, cfg.dataset, "test")):
            rows = mrbench_a.load(cfg.mrbench_dir, cfg.dataset, "test", ds.ids)
            res["mrbench_a"] = mrbench_a.case_metrics(rec["pred"][np.argsort(rec["idx"])], rows)
    res["patterns"]["Avg"] = average_metrics([res["patterns"][pattern_name(p)] for p in PATTERNS])
    res["text_gate_rate"] = float(np.mean(np.concatenate(fired)))
    res.update(alpha=alpha, tau_d=tau_d)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory with config.json and best.pt")
    ap.add_argument("--tau_test", type=float, default=None, help="test-time margin of robust mode / the override")
    ap.add_argument("--no_rates", action="store_true")
    ap.add_argument("--steering", action="store_true", help="evaluate discrepancy-gated activation steering instead")
    ap.add_argument("--alpha", type=float, default=0.1, help="steering strength")
    ap.add_argument("--tau_d", type=float, default=0.15, help="threshold on the state discrepancy d")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--mrbench_dir", default=None)
    ap.add_argument("--llm_path", default=None)
    ap.add_argument("--out", default=None, help="output JSON (default: <run>/results.json or results_steering.json)")
    args = ap.parse_args()
    if args.steering:
        res = evaluate_steering(args.run, args.alpha, args.tau_d, args.data_dir, args.mrbench_dir, args.llm_path)
        a, m = res["patterns"]["Avg"], res.get("mrbench_a", {})
        print(f"[steering] Avg Corr={a['Corr']:.4f} Acc2={a['Acc2']:.4f} | text gate fires on {100 * res['text_gate_rate']:.1f}%"
              + (f" | Cons={100 * m['Cons']:.1f} Conf={100 * m['Conf']:.1f} ShortErr={100 * m['ShortErr']:.1f}" if m else ""))
        dst = args.out or os.path.join(args.run, "results_steering.json")
    else:
        _, res = evaluate_run(args.run, args.tau_test, not args.no_rates, data_dir=args.data_dir,
                              mrbench_dir=args.mrbench_dir, llm_path=args.llm_path)
        dst = args.out or os.path.join(args.run, "results.json")
    with open(dst, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[done] {dst}")


if __name__ == "__main__":
    main()
