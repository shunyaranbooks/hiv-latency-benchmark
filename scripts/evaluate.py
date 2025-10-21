#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from hivlat.config import Config
from hivlat.logging import get_logger
from hivlat.models.baseline import load as load_model, eval_multiclass
from hivlat.models.metrics import expected_calibration_error_multiclass as ece_mc
from hivlat.data.labeling import apply_labels

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

    # Validation (held-out from latency model)
    X_val = pd.read_parquet(processed / 'X_val.parquet')
    y_val = pd.read_csv(processed / 'y_val.csv', index_col=0)['label']
    m_val = evaluate_set(bundle, X_val, y_val, 'val', outdir)

    # Donor external set (if available)
    donor_metrics = {}
    f_don = interim / 'GSE111727_donors.parquet'
    f_lab = interim / 'GSE111727_donors_labels.csv'
    if f_don.exists() and f_lab.exists():
        X_don = pd.read_parquet(f_don)
        y_don = pd.read_csv(f_lab, index_col=0)['label']
        # Keep only donor_untreated / donor_tcr; map to latent/productive for a sanity check
        mask = y_don.isin(['donor_untreated','donor_tcr'])
        X_don = X_don.loc[:, mask.index[mask]]
        y_map = y_don[mask].map({'donor_untreated':'latent', 'donor_tcr':'productive'})
        donor_metrics = evaluate_set(bundle, X_don, y_map, 'donor', outdir)

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
