#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from hivlat.config import Config
from hivlat.logging import get_logger
from hivlat.data.loaders import load_counts_txt_gz, save_parquet, save_csv
from hivlat.data.labeling import apply_labels
from hivlat.data.preprocess import qc_filter, normalize_log1p

def main(cfg_path: str):
    cfg = Config(cfg_path)
    log = get_logger()
    raw = Path(cfg.get('paths','raw'))
    interim = Path(cfg.get('paths','interim'))
    interim.mkdir(parents=True, exist_ok=True)

    # Load latency model raw counts
    f_lat = raw / 'GSE111727' / 'GSE111727_sc_lat_model_raw_counts_genes.txt.gz'
    if not f_lat.exists():
        raise SystemExit(f"Missing file: {f_lat}. Run scripts/download_geo.py first.")
    df_lat = load_counts_txt_gz(f_lat)
    labels_lat = pd.Series(apply_labels(df_lat.columns), index=df_lat.columns, name='label')

    # Basic QC + normalization
    df_lat_qc = qc_filter(df_lat,
                          min_genes_per_cell=cfg.get('prep','min_genes_per_cell', default=200),
                          min_cells_per_gene=cfg.get('prep','min_cells_per_gene', default=3))
    df_lat_norm = normalize_log1p(df_lat_qc)

    # Save interim matrices
    save_parquet(df_lat_norm, interim / 'GSE111727_latency_model.parquet')
    labels_lat.to_frame().to_csv(interim / 'GSE111727_latency_model_labels.csv')
    log.info(f"Saved latency model matrix: {df_lat_norm.shape}, with labels: {labels_lat.value_counts().to_dict()}")

    # Donor set (external)
    f_don = raw / 'GSE111727' / 'GSE111727_sc_donors_raw_counts_genes.txt.gz'
    if f_don.exists():
        df_don = load_counts_txt_gz(f_don)
        labels_don = pd.Series(apply_labels(df_don.columns), index=df_don.columns, name='label')
        df_don_qc = qc_filter(df_don,
                              min_genes_per_cell=cfg.get('prep','min_genes_per_cell', default=200),
                              min_cells_per_gene=cfg.get('prep','min_cells_per_gene', default=3))
        df_don_norm = normalize_log1p(df_don_qc)
        save_parquet(df_don_norm, interim / 'GSE111727_donors.parquet')
        labels_don.to_frame().to_csv(interim / 'GSE111727_donors_labels.csv')
        log.info(f"Saved donor matrix: {df_don_norm.shape}, labels: {labels_don.value_counts().to_dict()} ")
    else:
        log.warning("Donor file not found; skipping donor set.")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml', help='Path to YAML config')
    args = ap.parse_args()
    main(args.config)
