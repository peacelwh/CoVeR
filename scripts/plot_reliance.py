#!/usr/bin/env python
"""Plot the prior-centred reliance of the fused state on the text-, audio- and visual-only views across the decoder
layers, on conflict (solid) and consensus (dashed) cases, averaged over the given runs (output of diagnose.py).

    python scripts/plot_reliance.py --runs results/runs/mosi/llm_fusion_default_s4* --out results/figures/reliance_mosi.pdf
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--pattern", default="TAV")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [json.load(open(os.path.join(r, f"diagnosis_{args.pattern}.json"))) for r in args.runs]
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    colors = {"t": "#C0392B", "a": "#2E86C1", "v": "#1E7B34"}
    names = {"t": "text view", "a": "audio view", "v": "visual view"}
    for m in ("t", "a", "v"):
        for case, style in (("conflict", "-"), ("consensus", "--")):
            curves = [r["reliance"][case][m] for r in runs if case in r["reliance"]]
            if curves:
                y = np.mean(curves, 0)
                ax.plot(range(len(y)), y, style, color=colors[m], label=f"{names[m]}, {case}")
    ax.set_xlabel("decoder layer")
    ax.set_ylabel("prior-centred reliance")
    ax.legend(fontsize=6, ncol=2, frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
