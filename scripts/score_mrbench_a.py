#!/usr/bin/env python
"""Score the predictions of any model on MR-Bench-A: Acc2 on consensus and conflict cases and ShortErr.

Predictions: a JSONL file with one object {"id": ..., "pred": ...} per test sample, where ``pred`` is the predicted
sentiment score on the label scale of the corpus (its sign is the predicted polarity), with all modalities observed.

    python scripts/score_mrbench_a.py --dataset sims --pred my_predictions.jsonl
"""
import argparse
import json

import _path  # noqa: F401
from cover import mrbench_a


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--mrbench_dir", default="data/mrbench_a")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    rows = mrbench_a.load(args.mrbench_dir, args.dataset, args.split)
    preds = {}
    with open(args.pred, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                preds[str(r["id"])] = float(r["pred"])
    missing = [r["id"] for r in rows if r["id"] not in preds]
    if missing:
        raise SystemExit(f"{len(missing)} samples without a prediction, e.g. {missing[:3]}")
    m = mrbench_a.case_metrics([preds[r["id"]] for r in rows], rows)
    print(f"{args.dataset}/{args.split}: consensus Acc2 {100 * m['Cons']:.1f} ({m['n_consensus']} cases) | "
          f"conflict Acc2 {100 * m['Conf']:.1f} | ShortErr {100 * m['ShortErr']:.1f} ({m['n_conflict']} cases)")


if __name__ == "__main__":
    main()
