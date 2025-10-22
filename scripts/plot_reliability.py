#!/usr/bin/env python
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def reliability_curve(probs, y_true, n_bins=15):
    # probs: (n, ) for binary; else (n, K) we use max prob & correctness
    if probs.ndim == 2:
        conf = probs.max(1)
        yhat = probs.argmax(1)
        correct = (yhat == y_true).astype(float)
    else:
        conf = probs
        yhat = (probs >= 0.5).astype(int)
        correct = (yhat == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins+1)
    idx = np.digitize(conf, bins, right=True)
    bin_conf, bin_acc, counts = [], [], []
    for b in range(1, n_bins+1):
        mask = (idx==b)
        if mask.sum()==0:
            bin_conf.append(np.nan); bin_acc.append(np.nan); counts.append(0)
            continue
        bin_conf.append(conf[mask].mean())
        bin_acc.append(correct[mask].mean())
        counts.append(int(mask.sum()))
    return np.array(bin_conf), np.array(bin_acc), np.array(counts)

def ece(conf, acc, counts):
    m = counts.sum()
    w = np.where(counts==0, 0.0, counts / m)
    return float(np.nansum(np.abs(acc - conf) * w))

def load_binary_preds(csv):
    df = pd.read_csv(csv, index_col=0)
    return df['p_productive'].values, df['true_label'].values.astype(int)

def load_multiclass_preds(csv, classes):
    df = pd.read_csv(csv, index_col=0)
    probs = df[classes].values
    y = pd.Categorical(df['true_label'], categories=classes).codes
    return probs, y

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", default="data/processed/eval")
    ap.add_argument("--outdir", default="docs/paper/figures")
    ap.add_argument("--classes", nargs="+", default=["inducible","latent","productive"])
    ap.add_argument("--n_bins", type=int, default=15)
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Multiclass (val)
    probs_mc, y_mc = load_multiclass_preds(Path(args.eval_dir)/"predictions_val.csv", args.classes)
    c_conf, c_acc, c_cnt = reliability_curve(probs_mc, y_mc, n_bins=args.n_bins)
    fig = plt.figure(figsize=(4,4))
    plt.plot([0,1],[0,1])
    plt.plot(c_conf, c_acc, marker="o")
    plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.title("Reliability (Validation, calibrated)")
    plt.tight_layout(); fig.savefig(outdir/"fig_reliability_val_cal.png", dpi=300)

    # Binary (donor)
    p_prod, yb = load_binary_preds(Path(args.eval_dir)/"predictions_donor.csv")
    b_conf, b_acc, b_cnt = reliability_curve(p_prod, yb, n_bins=args.n_bins)
    fig = plt.figure(figsize=(4,4))
    plt.plot([0,1],[0,1])
    plt.plot(b_conf, b_acc, marker="o")
    plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.title("Reliability (Donor, calibrated)")
    plt.tight_layout(); fig.savefig(outdir/"fig_reliability_donor_cal.png", dpi=300)

    # Try uncalibrated if available
    unc_val = Path(args.eval_dir)/"predictions_val_uncal.csv"
    if unc_val.exists():
        probs_mc_u, y_mc_u = load_multiclass_preds(unc_val, args.classes)
        u_conf, u_acc, u_cnt = reliability_curve(probs_mc_u, y_mc_u, n_bins=args.n_bins)
        fig = plt.figure(figsize=(4,4))
        plt.plot([0,1],[0,1])
        plt.plot(u_conf, u_acc, marker="o")
        plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.title("Reliability (Validation, uncalibrated)")
        plt.tight_layout(); fig.savefig(outdir/"fig_reliability_val_uncal.png", dpi=300)

    unc_don = Path(args.eval_dir)/"predictions_donor_uncal.csv"
    if unc_don.exists():
        p_prod_u, yb_u = load_binary_preds(unc_don)
        u_conf, u_acc, u_cnt = reliability_curve(p_prod_u, yb_u, n_bins=args.n_bins)
        fig = plt.figure(figsize=(4,4))
        plt.plot([0,1],[0,1])
        plt.plot(u_conf, u_acc, marker="o")
        plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.title("Reliability (Donor, uncalibrated)")
        plt.tight_layout(); fig.savefig(outdir/"fig_reliability_donor_uncal.png", dpi=300)

    # Save small JSON with ECE values
    out = {
      "val_ece_cal": ece(c_conf, c_acc, c_cnt),
      "donor_ece_cal": ece(b_conf, b_acc, b_cnt)
    }
    Path(outdir/"reliability_ece.json").write_text(json.dumps(out, indent=2))
