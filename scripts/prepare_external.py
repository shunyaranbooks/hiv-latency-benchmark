#!/usr/bin/env python
from __future__ import annotations
import argparse, re, sys, gzip
from pathlib import Path
import numpy as np, pandas as pd, yaml

def _read_any(path: Path) -> pd.DataFrame:
    p=str(path)
    if p.endswith(".gz"):
        with gzip.open(p, "rt") as f:
            try: return pd.read_csv(f, sep="\t", index_col=0)
            except: 
                f.seek(0)
                return pd.read_csv(f, sep=",", index_col=0)
    else:
        try: return pd.read_csv(p, sep="\t", index_col=0)
        except: return pd.read_csv(p, sep=",", index_col=0)

def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    # Keep genes x cells, numeric entries
    return df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

def _fix_orientation(df: pd.DataFrame) -> pd.DataFrame:
    # Choose orientation where "library size per cell" (column sum) has positive median
    col_sum = df.sum(axis=0)
    med0 = np.median(col_sum[col_sum>0]) if (col_sum>0).any() else 0.0

    row_sum = df.sum(axis=1)
    med1 = np.median(row_sum[row_sum>0]) if (row_sum>0).any() else 0.0

    # If columns look dead and rows look alive, transpose to make columns cells
    if med0 == 0.0 and med1 > 0.0:
        return df.T
    return df

def _qc(df: pd.DataFrame, mgpc: int, mcpg: int) -> pd.DataFrame:
    keep_cells = (df > 0).sum(axis=0) >= mgpc
    keep_genes = (df > 0).sum(axis=1) >= mcpg
    return df.loc[keep_genes, keep_cells]

def _size_factor_norm(df: pd.DataFrame) -> pd.DataFrame:
    lib = df.sum(axis=0).astype(float)
    pos = lib > 0
    if not pos.any():  # degenerate; skip normalization
        return df
    med = np.median(lib[pos])
    if not np.isfinite(med) or med == 0:
        return df
    return df.divide((lib/med).replace({0:np.nan}).fillna(1.0), axis=1)

def _log1p(df: pd.DataFrame) -> pd.DataFrame:
    return np.log1p(df)

def _guess_label(cid: str) -> str:
    s = str(cid).lower()
    if re.search(r'\bpma\b|tcr|cd3|ionomycin|pha', s): return "productive"
    if re.search(r'saha|vorinostat|hdac|lra', s):     return "inducible"
    if re.search(r'dmso|control|ctrl|untreated|(^|_)ut(_|$)|vehicle', s): return "latent"
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--cohort", default="GSE180133")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    raw = Path(cfg["paths"]["raw"]); interim = Path(cfg["paths"]["interim"])
    interim.mkdir(parents=True, exist_ok=True)
    coh_dir = raw / a.cohort

    # Find a matrix
    cand = list(coh_dir.glob("*exprMatrix*.tsv*")) + list(coh_dir.glob("*.tsv*")) + list(coh_dir.glob("*.csv*"))
    if not cand:
        print(f"[prepare_external] Put matrix under {coh_dir}/ (e.g., GSE180133_exprMatrix.tsv.gz)", file=sys.stderr)
        sys.exit(0)
    mat = sorted(cand)[0]

    df = _read_any(mat)
    df = _coerce_numeric(df)
    df = _fix_orientation(df)  # ensure genes x cells (columns are cells)

    # QC + normalization
    mgpc = int(cfg.get("prep",{}).get("min_genes_per_cell", 100))
    mcpg = int(cfg.get("prep",{}).get("min_cells_per_gene", 3))
    df = _qc(df, mgpc, mcpg)

    if cfg.get("prep",{}).get("normalize","size_factor") == "size_factor":
        df = _size_factor_norm(df)
    if bool(cfg.get("prep",{}).get("log1p", True)):
        df = _log1p(df)

    # Save harmonized matrix
    out_mat = interim / f"{a.cohort}.parquet"
    df.to_parquet(out_mat)

    # Labels
    tmpl = coh_dir / f"{a.cohort}_labels_template.csv"
    if not tmpl.exists():
        labs = pd.DataFrame({"cell_id": df.columns, "label": [ _guess_label(c) for c in df.columns ]})
        labs["cell_id"] = labs["cell_id"].astype(str)
        labs["label"] = labs["label"].astype(str)
        labs.to_csv(tmpl, index=False)
        print(f"[prepare_external] wrote template → {tmpl}")

    # Prefer full labels if present, else use template
    full = coh_dir / f"{a.cohort}_labels.csv"
    labs = pd.read_csv(full if full.exists() else tmpl)
    labs["cell_id"] = labs["cell_id"].astype(str)
    labs["label"] = labs["label"].astype(str)
    labs.to_csv(interim / f"{a.cohort}_labels.csv", index=False)
    print(f"[prepare_external] saved matrix {out_mat.name} & labels to interim.")
if __name__ == "__main__":
    main()
