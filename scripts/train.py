#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from hivlat.config import Config
from hivlat.logging import get_logger
from hivlat.models.baseline import save as save_model
from hivlat.models.baseline import _align_to_genes  # reuse align function
from hivlat.models.baseline import train_model as _train_model  # fallback if needed

def select_hvgs(df_train: pd.DataFrame, n: int) -> list[str]:
    # df_train: genes x cells (normalized log1p already)
    # Use variance across cells as a simple HVG criterion
    v = df_train.var(axis=1)
    v = v.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    keep = v.sort_values(ascending=False).head(n).index.tolist()
    return keep

def main(cfg_path: str, model_out: str, model_type: str = 'logistic'):
    cfg = Config(cfg_path)
    log = get_logger()
    processed = Path(cfg.get('paths','processed'))

    X_train = pd.read_parquet(processed / 'X_train.parquet')   # genes x cells
    y_train = pd.read_csv(processed / 'y_train.csv', index_col=0)['label']
    X_val   = pd.read_parquet(processed / 'X_val.parquet')
    y_val   = pd.read_csv(processed / 'y_val.csv', index_col=0)['label']

    # --- HVGs from TRAIN only ---
    hvg_n = cfg.get('prep','hvg_n', default=2000)
    hvgs = select_hvgs(X_train, hvg_n)
    X_train = X_train.loc[hvgs]
    # align val to hvgs
    X_val = _align_to_genes(X_val, hvgs)

    # --- Train class-balanced logistic ---
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler(with_mean=False)
    Xs_train = scaler.fit_transform(X_train.T)

    clf = LogisticRegression(
        max_iter=1000,
        random_state=cfg.get('model','random_state', default=42),
        class_weight='balanced',      # helps with class imbalance
        C=1.0,                        # can tune later
        multi_class='auto'
    )
    clf.fit(Xs_train, pd.Categorical(y_train).codes)

    # --- Isotonic calibration on VAL ---
    Xs_val = scaler.transform(X_val.T)
    cal = CalibratedClassifierCV(clf, method='isotonic', cv='prefit')
    cal.fit(Xs_val, pd.Categorical(y_val).codes)

    bundle = {
        'model': cal,         # calibrated model
        'scaler': scaler,
        'classes': sorted(y_train.unique().tolist()),
        'genes': hvgs         # HVG gene list used everywhere
    }

    out_path = Path(model_out); out_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(bundle, out_path)
    log.info(f"Saved calibrated model → {out_path} | HVGs={len(hvgs)}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--model-out', required=True)
    ap.add_argument('--model-type', default='logistic', choices=['logistic','gbm'])
    args = ap.parse_args()
    main(args.config, args.model_out, args.model_type)
