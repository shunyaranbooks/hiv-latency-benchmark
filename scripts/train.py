#!/usr/bin/env python
from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from hivlat.config import Config
from hivlat.logging import get_logger
from hivlat.models.baseline import save as save_model, _align_to_genes

def select_hvgs(df: pd.DataFrame, n: int) -> list[str]:
    v = df.var(axis=1).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    return v.sort_values(ascending=False).head(n).index.tolist()

def make_weights(y: pd.Series) -> np.ndarray:
    vc = y.value_counts()
    w = {c: (1.0/len(vc)) / (vc[c]/len(y)) for c in vc.index}  # inverse-freq normalized
    return y.map(w).values

def maybe_intersection(train: pd.DataFrame, cfg) -> list[str] | None:
    """Return a train∩donor gene list if enabled and donor interim exists; else None."""
    if not cfg.get('prep', 'hvg_intersection_with_donor', default=False):
        return None
    interim = Path(cfg.get('paths', 'interim'))
    f_don = interim / 'GSE111727_donors.parquet'
    if not f_don.exists():
        return None
    don = pd.read_parquet(f_don)  # genes x cells
    inter = train.index.intersection(don.index)
    return inter.tolist()

def main(cfg_path: str, model_out: str, model_type: str = 'logistic'):
    cfg = Config(cfg_path); log = get_logger()
    processed = Path(cfg.get('paths','processed'))

    X_train = pd.read_parquet(processed/'X_train.parquet')   # genes x cells
    y_train = pd.read_csv(processed/'y_train.csv', index_col=0)['label']
    X_val   = pd.read_parquet(processed/'X_val.parquet')
    y_val   = pd.read_csv(processed/'y_val.csv', index_col=0)['label']

    # Gene set for HVG selection
    inter = maybe_intersection(X_train, cfg)
    base = X_train if inter is None else X_train.loc[inter]

    hvg_n = cfg.get('prep','hvg_n', default=2000)
    hvgs = select_hvgs(base, min(hvg_n, base.shape[0]))
    # apply HVGs
    X_train = X_train.loc[hvgs]
    X_val   = _align_to_genes(X_val, hvgs)

    scaler = StandardScaler(with_mean=False)
    Xs_train = scaler.fit_transform(X_train.T)
    Xs_val   = scaler.transform(X_val.T)

    if model_type == 'gbm':
        est = GradientBoostingClassifier(random_state=cfg.get('model','random_state', default=42))
        est.fit(Xs_train, pd.Categorical(y_train).codes, sample_weight=make_weights(y_train))
    else:
        C = float(os.getenv('HIVLAT_C', '1.0'))
        est = LogisticRegression(max_iter=1000, random_state=cfg.get('model','random_state', default=42),
                                 class_weight='balanced', C=C, multi_class='auto')
        est.fit(Xs_train, pd.Categorical(y_train).codes)

    cal = CalibratedClassifierCV(est, method='isotonic', cv='prefit')
    cal.fit(Xs_val, pd.Categorical(y_val).codes)

    bundle = {'model': cal, 'scaler': scaler, 'classes': sorted(y_train.unique().tolist()), 'genes': hvgs}
    out = Path(model_out); out.parent.mkdir(parents=True, exist_ok=True)
    save_model(bundle, out)
    log.info(f"Saved calibrated {model_type} → {out} | HVGs={len(hvgs)} | intersection={inter is not None}")
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--model-out', required=True)
    ap.add_argument('--model-type', default='logistic', choices=['logistic','gbm'])
    args = ap.parse_args()
    main(args.config, args.model_out, args.model_type)
