#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from hivlat.config import Config
from hivlat.models.baseline import load as load_model, predict_proba
from hivlat.models.metrics import expected_calibration_error_multiclass as ece_mc

VALID_LAT = {"latent","inducible","productive"}

def binarize(label: str) -> int:
    # latent+inducible = 0; productive = 1
    return 1 if label=="productive" else 0

def prior_correct(p_prod: np.ndarray, pi_s: float, pi_t: float) -> np.ndarray:
    eps = 1e-6
    p = np.clip(p_prod, eps, 1-eps)
    return (pi_t/pi_s) * p / ((pi_t/pi_s)*p + ((1-pi_t)/(1-pi_s))*(1-p))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cohort", default="GSE180133")
    args = ap.parse_args()

    cfg = Config(args.config)
    processed = Path(cfg.get('paths','processed'))
    interim   = Path(cfg.get('paths','interim'))
    outdir    = processed / 'eval'
    outdir.mkdir(parents=True, exist_ok=True)

    bundle = load_model(args.model)
    classes = bundle['classes']

    # Load cohort
    X_ext = pd.read_parquet(interim / f"{args.cohort}.parquet")
    labs  = pd.read_csv(interim / f"{args.cohort}_labels.csv")
    labs = labs.set_index("cell_id")["label"]

    # keep only labeled, valid cells
    keep = labs.index.intersection(X_ext.columns)
    y_str = labs.loc[keep].astype(str).str.strip().str.lower()
    # map empty to NaN, filter to known set
    y_str = y_str.replace({"":np.nan}).dropna()
    y_str = y_str[y_str.isin(VALID_LAT)]
    cells = y_str.index
    X_ext = X_ext.loc[:, cells]

    # predict
    proba_full = predict_proba(bundle, X_ext)
    if "productive" not in classes or "latent" not in classes or "inducible" not in classes:
        raise RuntimeError("Model classes must include latent, inducible, productive.")
    p_prod = proba_full[:, classes.index("productive")]
    # binary ground truth
    y_bin = np.array([binarize(s) for s in y_str.values], dtype=int)

    # prior-shift correction (pi_s from validation file if present; else estimate from probs)
    y_val_path = processed / "y_val.csv"
    if y_val_path.exists():
        y_val = pd.read_csv(y_val_path, index_col=0)['label']
        pi_s = float((y_val == "productive").mean())
    else:
        pi_s = float(np.clip(p_prod.mean(), 1e-6, 1-1e-6))
    pi_t = float(np.mean(y_bin))
    p_prod_corr = prior_correct(p_prod, pi_s, pi_t)

    # metrics
    auroc = float(roc_auc_score(y_bin, p_prod_corr))
    auprc = float(average_precision_score(y_bin, p_prod_corr))
    proba2 = np.vstack([1.0 - p_prod_corr, p_prod_corr]).T
    ece = float(ece_mc(proba2, y_bin, n_bins=15))

    # save predictions
    dfp = pd.DataFrame({"p_latent": 1.0 - p_prod_corr, "p_productive": p_prod_corr, "true_label": y_bin}, index=cells)
    dfp.to_csv(outdir / "predictions_external.csv")

    rep = {"external": {"cohort": args.cohort, "auroc_binary": auroc, "auprc_binary": auprc, "ece_maxprob": ece}}
    (outdir / "report_external.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
if __name__ == "__main__":
    main()
