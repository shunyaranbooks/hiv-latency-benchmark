#!/usr/bin/env python
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def bootstrap_binary(y, p, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    n_s = len(y)
    aurocs, auprcs, briers = [], [], []
    for _ in range(n):
        idx = rng.integers(0, n_s, size=n_s)
        yt = y[idx]; pt = p[idx]
        if len(np.unique(yt))<2:
            continue
        aurocs.append(roc_auc_score(yt, pt))
        auprcs.append(average_precision_score(yt, pt))
        briers.append(brier_score_loss(yt, pt))
    def ci(a): 
        return float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))
    return {
        "auroc": {"mean": float(np.mean(aurocs)), "ci95": ci(aurocs)},
        "auprc": {"mean": float(np.mean(auprcs)), "ci95": ci(auprcs)},
        "brier": {"mean": float(np.mean(briers)), "ci95": ci(briers)},
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", default="data/processed/eval")
    ap.add_argument("--out", default="docs/paper/tables/bootstrap_donor_binary.json")
    args = ap.parse_args()

    # donor
    df = pd.read_csv(Path(args.eval_dir)/"predictions_donor.csv", index_col=0)
    p = df["p_productive"].values
    y = df["true_label"].values.astype(int)
    out = bootstrap_binary(y, p)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
