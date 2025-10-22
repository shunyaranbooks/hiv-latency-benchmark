#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from hivlat.models.baseline import load as load_model, predict_proba
from hivlat.config import Config

def main(cfg_path: str, model_uncal: str):
    cfg = Config(cfg_path)
    processed = Path(cfg.get('paths','processed'))
    interim   = Path(cfg.get('paths','interim'))
    outdir    = processed / 'eval'

    bundle = load_model(model_uncal)
    # val (multiclass)
    X_val = pd.read_parquet(processed / 'X_val.parquet')
    y_val = pd.read_csv(processed / 'y_val.csv', index_col=0)['label']
    proba = predict_proba(bundle, X_val)
    dfp = pd.DataFrame(proba, index=X_val.columns, columns=bundle['classes'])
    dfp['true_label'] = y_val.values
    dfp.to_csv(outdir/'predictions_val_uncal.csv')

    # donor (binary projection)
    f_don = interim / 'GSE111727_donors.parquet'
    f_lab = interim / 'GSE111727_donors_labels.csv'
    if f_don.exists() and f_lab.exists():
        X_don = pd.read_parquet(f_don)
        y_don = pd.read_csv(f_lab, index_col=0)['label']
        mask = y_don.isin(['donor_untreated','donor_tcr'])
        X_don = X_don.loc[:, mask.index[mask]]
        y_bin = (y_don[mask].map({'donor_untreated':'latent','donor_tcr':'productive'}).values == 'productive').astype(int)

        proba_full = predict_proba(bundle, X_don)
        classes = bundle['classes']
        if 'productive' in classes:
            p_prod = proba_full[:, classes.index('productive')]
            dfp = pd.DataFrame({'p_latent': 1.0 - p_prod, 'p_productive': p_prod, 'true_label': y_bin}, index=X_don.columns)
            dfp.to_csv(outdir/'predictions_donor_uncal.csv')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--model-uncal', required=True)
    args = ap.parse_args()
    main(args.config, args.model_uncal)
