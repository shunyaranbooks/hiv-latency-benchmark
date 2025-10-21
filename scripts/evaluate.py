#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from hivlat.config import Config
from hivlat.logging import get_logger
from hivlat.models.baseline import load as load_model, eval_multiclass, predict_proba
from hivlat.models.metrics import expected_calibration_error_multiclass as ece_mc

def evaluate_set(bundle, X: pd.DataFrame, y: pd.Series, name: str, outdir: Path):
    metrics, proba, y_true = eval_multiclass(bundle, X, y)
    ece = ece_mc(proba, y_true, n_bins=15)
    metrics['ece_maxprob'] = float(ece)
    # Save per-cell predictions
    df_pred = pd.DataFrame(proba, index=X.columns, columns=bundle['classes'])
    df_pred['true_label'] = y.values
    df_pred.to_csv(outdir / f'predictions_{name}.csv')
    return metrics

def main(cfg_path: str, model_path: str):
    cfg = Config(cfg_path)
    log = get_logger()
    processed = Path(cfg.get('paths','processed'))
    interim = Path(cfg.get('paths','interim'))
    outdir = processed / 'eval'
    outdir.mkdir(parents=True, exist_ok=True)

    bundle = load_model(model_path)

    # Validation (multiclass)
    X_val = pd.read_parquet(processed / 'X_val.parquet')
    y_val = pd.read_csv(processed / 'y_val.csv', index_col=0)['label']
    m_val = evaluate_set(bundle, X_val, y_val, 'val', outdir)

    # Donor (binary latent vs productive)
    donor_metrics = {}
    f_don = interim / 'GSE111727_donors.parquet'
    f_lab = interim / 'GSE111727_donors_labels.csv'
    if f_don.exists() and f_lab.exists():
        X_don = pd.read_parquet(f_don)
        y_don = pd.read_csv(f_lab, index_col=0)['label']
        mask = y_don.isin(['donor_untreated','donor_tcr'])
        X_don = X_don.loc[:, mask.index[mask]]
        y_bin_labels = y_don[mask].map({'donor_untreated':'latent', 'donor_tcr':'productive'})
        proba_full = predict_proba(bundle, X_don)
        classes = bundle['classes']
        if 'productive' in classes and 'latent' in classes:
            p_prod = proba_full[:, classes.index('productive')]
            y_bin = (y_bin_labels.values == 'productive').astype(int)
            if len(set(y_bin)) == 2:
                donor_metrics = {
                    'auroc_binary': float(roc_auc_score(y_bin, p_prod)),
                    'auprc_binary': float(average_precision_score(y_bin, p_prod))
                }
                proba2 = np.vstack([1.0 - p_prod, p_prod]).T
                donor_metrics['ece_maxprob'] = float(ece_mc(proba2, y_bin, n_bins=15))
            else:
                donor_metrics = {'note': 'Only one donor class present; AUROC/AUPRC undefined.'}
        else:
            donor_metrics = {'note': 'Model classes missing latent/productive.'}

    report = {'val': m_val, 'donor': donor_metrics}
    with open(outdir / 'report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--model', required=True)
    args = ap.parse_args()
    main(args.config, args.model)
