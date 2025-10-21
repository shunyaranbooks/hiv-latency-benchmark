#!/usr/bin/env python
from __future__ import annotations
import argparse, json, sys
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

    # keep only cells we actually have in X
    y = y.loc[X.columns]

    # filter to three training classes
    keep = y.isin(['latent','inducible','productive'])
    X = X.loc[:, keep.index[keep]]
    y = y[keep]

    if X.shape[1] == 0:
        log.error("No cells available after QC/labeling. "
                  "Try loosening QC in configs/default.yaml (e.g., min_genes_per_cell: 50, min_cells_per_gene: 1).")
        sys.exit(2)

    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(y.index, test_size=0.2, random_state=42, stratify=y.values)

    # save splits
    (processed / 'splits').mkdir(parents=True, exist_ok=True)
    with open(processed / 'splits' / 'latency_gse111727.json', 'w') as f:
        json.dump({'train': list(train_idx), 'val': list(val_idx)}, f, indent=2)

    X.loc[:, train_idx].to_parquet(processed / 'X_train.parquet')
    X.loc[:, val_idx].to_parquet(processed / 'X_val.parquet')
    y.loc[train_idx].to_frame('label').to_csv(processed / 'y_train.csv')
    y.loc[val_idx].to_frame('label').to_csv(processed / 'y_val.csv')
    log.info(f"Train/val split saved. Train={len(train_idx)}, Val={len(val_idx)}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml')
    args = ap.parse_args()
    main(args.config)
