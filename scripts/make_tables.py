#!/usr/bin/env python
"""Collect the runs under results/runs into tables (mean over seeds), written to results/tables.

    main.md       Acc2 / F1 / MAE / Corr averaged over the seven availability patterns (Avg) and over the random
                  missing rates (Rate), accuracy mode
    mrbench_a.md  Avg Corr (x100) and, with all modalities observed, Acc2 on consensus (Cons.) and conflict (Conf.)
                  cases and ShortErr (SE), in %, for accuracy mode and robust mode
    patterns.md   Acc2 / MAE / Corr per availability pattern, accuracy mode
    summary.json  the same numbers, machine readable

    python scripts/make_tables.py --runs results/runs --out results/tables
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

PATTERNS = ["TAV", "TA", "TV", "AV", "T", "A", "V"]
DATASETS = ["mosi", "mosei", "sims", "simsv2", "iemocap", "meld", "mustardpp"]


def mean(values):
    return float(np.mean(values)) if values else float("nan")


def collect(root):
    """{dataset: {run name: {seed: results}}}"""
    runs = defaultdict(lambda: defaultdict(dict))
    for path in sorted(glob.glob(os.path.join(root, "*", "*", "results.json"))):
        m = re.match(r"(.+)_s(\d+)$", os.path.basename(os.path.dirname(path)))
        if m:
            with open(path) as f:
                runs[os.path.basename(os.path.dirname(os.path.dirname(path)))][m.group(1)][int(m.group(2))] = json.load(f)
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="results/tables")
    args = ap.parse_args()
    runs = collect(args.runs)
    os.makedirs(args.out, exist_ok=True)
    summary = {}
    main_md = ["| Dataset | Run | Seeds | Acc2 | F1 | MAE | Corr | Rate Acc2 | Rate MAE | Rate Corr |", "|---|---|---|---|---|---|---|---|---|---|"]
    bench_md = ["| Dataset | Run | Mode | Seeds | Corr | Cons. | Conf. | SE |", "|---|---|---|---|---|---|---|---|"]
    pat_md = ["| Dataset | Run | Pattern | Acc2 | MAE | Corr |", "|---|---|---|---|---|---|"]
    for ds in [d for d in DATASETS if d in runs] + sorted(set(runs) - set(DATASETS)):
        for name, by_seed in sorted(runs[ds].items()):
            seeds = sorted(by_seed)
            acc = [by_seed[s]["accuracy"] for s in seeds]
            avg = {k: mean([r["patterns"]["Avg"][k] for r in acc]) for k in ("Acc2", "F1", "MAE", "Corr")}
            rate = {k: mean([r["rates"]["Rate"][k] for r in acc if "rates" in r]) for k in ("Acc2", "MAE", "Corr")}
            entry = {"seeds": seeds, "Avg": avg, "Rate": rate, "patterns": {}, "mrbench_a": {}}
            main_md.append(f"| {ds} | {name} | {len(seeds)} | {100 * avg['Acc2']:.1f} | {100 * avg['F1']:.1f} | {avg['MAE']:.3f} | "
                           f"{avg['Corr']:.3f} | {100 * rate['Acc2']:.1f} | {rate['MAE']:.3f} | {rate['Corr']:.3f} |")
            for p in PATTERNS:
                pm = {k: mean([r["patterns"][p][k] for r in acc]) for k in ("Acc2", "MAE", "Corr")}
                entry["patterns"][p] = pm
                pat_md.append(f"| {ds} | {name} | {p} | {100 * pm['Acc2']:.1f} | {pm['MAE']:.3f} | {pm['Corr']:.3f} |")
            for mode in ("accuracy", "robust"):
                rs = [by_seed[s][mode] for s in seeds if mode in by_seed[s] and "mrbench_a" in by_seed[s][mode]]
                if not rs:
                    continue
                bm = {"Corr": mean([r["patterns"]["Avg"]["Corr"] for r in rs]),
                      **{k: mean([r["mrbench_a"][k] for r in rs]) for k in ("Cons", "Conf", "ShortErr")}}
                entry["mrbench_a"][mode] = bm
                bench_md.append(f"| {ds} | {name} | {mode} | {len(rs)} | {100 * bm['Corr']:.1f} | {100 * bm['Cons']:.1f} | "
                                f"{100 * bm['Conf']:.1f} | {100 * bm['ShortErr']:.1f} |")
            summary[f"{ds}/{name}"] = entry
    for fname, lines in (("main.md", main_md), ("mrbench_a.md", bench_md), ("patterns.md", pat_md)):
        with open(os.path.join(args.out, fname), "w") as f:
            f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("\n".join(main_md) + "\n\n" + "\n".join(bench_md) + f"\n\n-> {args.out}")


if __name__ == "__main__":
    main()
