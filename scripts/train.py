#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from hivlat.config import Config
from hivlat.logging import get_logger
from hivlat.models.baseline import train_model, save as save_model

def main(cfg_path: str, model_out: str, model_type: str = 'logistic'):
    cfg = Config(cfg_path)
    log = get_logger()
    processed = Path(cfg.get('paths','processed'))

    X_train = pd.read_parquet(processed / 'X_train.parquet')
    y_train = pd.read_csv(processed / 'y_train.csv', index_col=0)['label']

    bundle = train_model(
        X_train, y_train,
        model_type=model_type,
        random_state=cfg.get('model','random_state', default=42),
        max_iter=cfg.get('model','max_iter', default=800)
    )
    out_path = Path(model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(bundle, out_path)
    log.info(f"Saved model → {out_path}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--model-out', required=True)
    ap.add_argument('--model-type', default='logistic', choices=['logistic','gbm'])
    args = ap.parse_args()
    main(args.config, args.model_out, args.model_type)
