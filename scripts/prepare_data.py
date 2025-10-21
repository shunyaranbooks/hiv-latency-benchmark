#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
from hivlat.config import Config
from hivlat.logging import get_logger
from hivlat.data.loaders import load_counts_txt_gz, save_parquet
from hivlat.data.labeling import apply_labels
from hivlat.data.preprocess import qc_filter, normalize_log1p

VALID_LAT = {"latent","inducible","productive"}
VALID_DON = {"donor_untreated","donor_tcr"}

def save_matrix_and_labels(df, labels, path_matrix, path_labels, log, tag):
    path_matrix.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(df, path_matrix)
    labels.to_frame("label").to_csv(path_labels)
    log.info(f"[{tag}] cells: {df.shape[1]} | labels: {labels.value_counts().to_dict()}")

def ensure_mapping_template(cols, out_csv, valid_set, tag, log):
    out = Path(out_csv)
    if not out.exists():
        df = pd.DataFrame({"cell_id": list(cols), "label": [""]*len(cols)})
        df.to_csv(out, index=False)
        log.warning(f"Created label template for {tag}: {out}\n"
                    f"  → Fill 'label' with one of {sorted(valid_set)} and re-run.")
        return False
    lab = pd.read_csv(out)
    if not {"cell_id","label"}.issubset(lab.columns):
        raise SystemExit(f"{out} must have columns: cell_id,label")
    lab = lab.set_index("cell_id").reindex(cols)
    if lab["label"].isna().any() or (lab["label"].astype(str).str.len()==0).any():
        missing = int((lab["label"].astype(str).str.len()==0).sum())
        log.warning(f"{missing} empty labels in {out}. Please complete them.")
        return False
    bad = set(lab["label"]) - valid_set
    if bad:
        raise SystemExit(f"Invalid labels in {out}: {bad} ; allowed={valid_set}")
    return lab["label"]

def main(cfg_path: str):
    cfg = Config(cfg_path)
    log = get_logger()
    raw = Path(cfg.get('paths','raw'))
    interim = Path(cfg.get('paths','interim')); interim.mkdir(parents=True, exist_ok=True)

    # --- Latency model ---
    f_lat = raw / 'GSE111727' / 'GSE111727_sc_lat_model_raw_counts_genes.txt.gz'
    if not f_lat.exists():
        raise SystemExit(f"Missing file: {f_lat}. Run scripts/download_geo.py first.")
    df_lat = load_counts_txt_gz(f_lat)
    df_lat_qc = qc_filter(df_lat,
        min_genes_per_cell=cfg.get('prep','min_genes_per_cell', default=50),
        min_cells_per_gene=cfg.get('prep','min_cells_per_gene', default=1))
    df_lat_norm = normalize_log1p(df_lat_qc)

    # 1) try heuristic labels
    labels_lat = pd.Series(apply_labels(df_lat_norm.columns), index=df_lat_norm.columns, name="label")
    # 2) if unknown dominates, require mapping template
    if (labels_lat == "unknown").mean() > 0.5:
        tmpl = raw / 'GSE111727' / 'GSE111727_latency_model_labels_template.csv'
        lab = ensure_mapping_template(df_lat_norm.columns, tmpl, VALID_LAT, "latency model", log)
        if lab is False:
            sys.exit(2)  # stop and let user fill the CSV
        labels_lat = lab

    save_matrix_and_labels(df_lat_norm, labels_lat,
        interim / 'GSE111727_latency_model.parquet',
        interim / 'GSE111727_latency_model_labels.csv', log, "latency_model")

    # --- Donor set (external) ---
    f_don = raw / 'GSE111727' / 'GSE111727_sc_donors_raw_counts_genes.txt.gz'
    if f_don.exists():
        df_don = load_counts_txt_gz(f_don)
        df_don_qc = qc_filter(df_don,
            min_genes_per_cell=cfg.get('prep','min_genes_per_cell', default=50),
            min_cells_per_gene=cfg.get('prep','min_cells_per_gene', default=1))
        df_don_norm = normalize_log1p(df_don_qc)

        labels_don = pd.Series(apply_labels(df_don_norm.columns), index=df_don_norm.columns, name="label")
        if (labels_don == "unknown").mean() > 0.5:
            tmpl = raw / 'GSE111727' / 'GSE111727_donors_labels_template.csv'
            lab = ensure_mapping_template(df_don_norm.columns, tmpl, VALID_DON, "donors", log)
            if lab is False:
                sys.exit(2)
            labels_don = lab

        save_matrix_and_labels(df_don_norm, labels_don,
            interim / 'GSE111727_donors.parquet',
            interim / 'GSE111727_donors_labels.csv', log, "donor")
    else:
        log.warning("Donor file not found; skipping donor set.")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml')
    args = ap.parse_args()
    main(args.config)
