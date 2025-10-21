#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from hivlat.config import Config
from hivlat.logging import get_logger

def main(cfg_path: str):
    cfg = Config(cfg_path)
    log = get_logger()
    interim = Path(cfg.get('paths','interim'))
    processed = Path(cfg.get('paths','processed'))
    processed.mkdir(parents=True, exist_ok=True)

    X = pd.read_parquet(interim / 'GSE111727_latency_model.parquet')
    y = pd.read_csv(interim / 'GSE111727_latency_model_labels.csv', index_col=0)['label']

    # Keep only labeled latency model cells
    keep = y.isin(['latent','inducible','productive'])
    X = X.loc[:, keep.index[keep]]
    y = y[keep]

    # Stratified split: 80% train / 20% val
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(y.index, test_size=0.2, random_state=42, stratify=y.values)

    # Save splits as lists of cell IDs
    splits = {'train': list(train_idx), 'val': list(val_idx)}
    (processed / 'splits').mkdir(exist_ok=True, parents=True)
    with open(processed / 'splits' / 'latency_gse111727.json', 'w') as f:
        json.dump(splits, f, indent=2)

    # Save convenience tables
    X.loc[:, train_idx].to_parquet(processed / 'X_train.parquet')
    X.loc[:, val_idx].to_parquet(processed / 'X_val.parquet')
    y.loc[train_idx].to_frame('label').to_csv(processed / 'y_train.csv')
    y.loc[val_idx].to_frame('label').to_csv(processed / 'y_val.csv')
    log.info(f"Train/val split saved. Train cells={len(train_idx)}, Val cells={len(val_idx)}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml')
    args = ap.parse_args()
    main(args.config)
