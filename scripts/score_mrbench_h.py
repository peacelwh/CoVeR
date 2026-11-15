#!/usr/bin/env python
"""Score prediction files on the 900 clips of MR-Bench-H (620 LS, 280 NS).

Every model is scored on all clips; an answer that is not a valid polarity counts as wrong. Reported per model and
view: LS and NS accuracy, the 95% Wilson interval of the LS accuracy and LS ShortErr; for fusion models, the mean and
standard deviation over seeds of the accuracy pooled over the seven sources, and the share of LS clips solved from
audio and video alone that adding the text makes wrong.

    python scripts/score_mrbench_h.py --items /path/to/MR-Bench-H/data/test.jsonl --pred results/mrbench_h/*.jsonl
"""
import argparse
import glob
import json

import _path  # noqa: F401
from cover import mrbench_h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True)
    ap.add_argument("--pred", nargs="+", required=True, help="prediction files (JSONL, optionally .gz) or globs")
    ap.add_argument("--out", default="results/mrbench_h.json")
    args = ap.parse_args()
    items = mrbench_h.load_items(args.items)
    n_ls = sum(it["category"] == "L" for it in items.values())
    print(f"items: {len(items)} ({n_ls} LS, {len(items) - n_ls} NS)")
    rows = [r for pat in args.pred for f in sorted(glob.glob(pat)) for r in mrbench_h.read_jsonl(f)]
    res = mrbench_h.score(items, rows)
    print("| model | view | seeds | LS acc | 95% CI | NS acc | LS ShortErr |\n|---|---|---|---|---|---|---|")
    for key, r in res.items():
        model, view = key.split("|")
        many = len(r["seeds"]) > 1
        ls = f"{r['LS']['acc']:.1f}" + (f"±{r['LS']['std']:.1f}" if many else "")
        ns = f"{r['NS']['acc']:.1f}" + (f"±{r['NS']['std']:.1f}" if many else "")
        ci = "--" if many else f"[{r['LS']['ci95'][0]:.1f}, {r['LS']['ci95'][1]:.1f}]"
        print(f"| {model} | {view} | {len(r['seeds'])} | {ls} | {ci} | {ns} | {r['LS']['ShortErr']:.1f} |")
    incomplete = {key: r["missing"] for key, r in res.items() if r["missing"]}
    if incomplete:
        print("note: clips without a prediction count as wrong; pool the files of all seven sources before scoring. "
              + ", ".join(f"{k}: {n} missing" for k, n in incomplete.items()))
    views = {}
    for key in res:
        views.setdefault(key.split("|")[0], set()).add(key.split("|")[1])
    for model, vs in views.items():
        if {"nonverbal", "full"} <= vs:
            o = mrbench_h.text_override_rate(items, rows, model)
            res[f"{model}|text_override"] = o
            print(f"{model}: of the LS clips solved from audio and video alone, adding the text makes {o['rate']:.1f}% wrong")
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
