#!/usr/bin/env python
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def reliability_curve(probs, y_true, n_bins=15):
    conf = probs
    yhat = (probs >= 0.5).astype(int)
    correct = (yhat == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    idx = np.digitize(conf, bins, right=True)
    bin_conf, bin_acc, counts = [], [], []
    for b in range(1, n_bins+1):
        m = (idx==b)
        if m.sum()==0:
            bin_conf.append(np.nan); bin_acc.append(np.nan); counts.append(0)
        else:
            bin_conf.append(conf[m].mean()); bin_acc.append(correct[m].mean()); counts.append(int(m.sum()))
    return np.array(bin_conf), np.array(bin_acc), np.array(counts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/processed/eval/predictions_external.csv")
    ap.add_argument("--out", default="docs/paper/external/figures/fig_reliability_external_cal.png")
    ap.add_argument("--title", default="Reliability (External, calibrated)")
    ap.add_argument("--n_bins", type=int, default=15)
    args = ap.parse_args()
    df = pd.read_csv(args.csv, index_col=0)
    p = df["p_productive"].values; y = df["true_label"].values.astype(int)
    c, a, _ = reliability_curve(p, y, n_bins=args.n_bins)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(4,4))
    plt.plot([0,1],[0,1])
    plt.plot(c, a, marker="o")
    plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.title(args.title)
    plt.tight_layout(); fig.savefig(args.out, dpi=300)
    print(f"Wrote {args.out}")
if __name__ == "__main__":
    main()
