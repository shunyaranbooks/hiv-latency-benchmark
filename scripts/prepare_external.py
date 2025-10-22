#!/usr/bin/env python
from __future__ import annotations
import argparse, gzip, io, re, sys
from pathlib import Path
import numpy as np, pandas as pd

def _read_any_matrix(path: Path) -> pd.DataFrame:
    p = str(path)
    openf = gzip.open if p.endswith(".gz") else open
    with openf(p, "rt", encoding="utf-8", errors="ignore") as f:
        # try TSV, then CSV
        try:
            df = pd.read_csv(f, sep="\t", index_col=0)
        except Exception:
            f.seek(0) if hasattr(f, "seek") else None
            df = pd.read_csv(path, sep=",", index_col=0)
    # standardize index name
    if df.index.name is None or df.index.name.lower().startswith("unnamed"):
        df.index.name = "gene"
    return df

def _size_factor_norm(df: pd.DataFrame) -> pd.DataFrame:
    # df: genes x cells (counts)
    lib = df.sum(axis=0).replace(0, np.nan)
    sf = lib / np.median(lib.dropna())
    sf = sf.replace(0, np.nan).fillna(1.0)
    return (df / sf)

def _log1p(df: pd.DataFrame) -> pd.DataFrame:
    return np.log1p(df)

def _qc(df: pd.DataFrame, min_genes_per_cell: int, min_cells_per_gene: int) -> pd.DataFrame:
    # cells with >= min genes, genes detected in >= min cells
    cell_kept = (df > 0).sum(axis=0) >= min_genes_per_cell
    gene_kept = (df > 0).sum(axis=1) >= min_cells_per_gene
    return df.loc[gene_kept, cell_kept]

def _guess_label_from_name(cid: str) -> str | None:
    s = cid.lower()
    # very conservative heuristics
    if re.search(r"(tcr|cd3|pma|pha|ionomycin|il2|stim)", s):
        return "productive"
    if re.search(r"(saha|vorinostat|hdaci|lra)", s):
        return "inducible"
    if re.search(r"(untreated|control|ctrl|ut\b|vehicle)", s):
        return "latent"
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--cohort", default="GSE180133")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    raw = Path(cfg["paths"]["raw"]); interim = Path(cfg["paths"]["interim"])
    interim.mkdir(parents=True, exist_ok=True)
    cohort_dir = raw / args.cohort
    if not cohort_dir.exists():
        cohort_dir.mkdir(parents=True, exist_ok=True)

    # find a plausible matrix
    candidates = list(cohort_dir.glob("*genes*.txt*")) + list(cohort_dir.glob("*.csv")) + list(cohort_dir.glob("*.tsv"))
    if not candidates:
        print(f"[prepare_external] Place a gene-by-cell count matrix under {cohort_dir}/ (e.g., *_genes.txt.gz).", file=sys.stderr)
        sys.exit(0)

    mat = sorted(candidates)[0]
    df = _read_any_matrix(mat)
    # ensure numeric
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # QC + normalize + log1p
    qc = cfg.get("prep", {})
    df = _qc(df, qc.get("min_genes_per_cell", 100), qc.get("min_cells_per_gene", 3))
    if qc.get("normalize","size_factor") == "size_factor":
        df = _size_factor_norm(df)
    if qc.get("log1p", True):
        df = _log1p(df)

    # Save harmonized matrix
    out_mat = interim / f"{args.cohort}.parquet"
    df.to_parquet(out_mat)

    # Make label template (cell_id,label) and try conservative auto-fill
    tmpl = cohort_dir / f"{args.cohort}_labels_template.csv"
    if not tmpl.exists():
        labs = []
        for cid in df.columns:
            labs.append([cid, _guess_label_from_name(cid) or ""])
        pd.DataFrame(labs, columns=["cell_id","label"]).to_csv(tmpl, index=False)
        print(f"[prepare_external] Wrote label template → {tmpl}")

    # Copy to interim (if user later edits template, we use that)
    # If a full labels csv already exists beside template, prefer it
    lbl_full = cohort_dir / f"{args.cohort}_labels.csv"
    if lbl_full.exists():
        labs = pd.read_csv(lbl_full)
    else:
        labs = pd.read_csv(tmpl)
    labs.to_csv(interim / f"{args.cohort}_labels.csv", index=False)
    print(f"[prepare_external] Saved harmonized matrix {out_mat.name} and labels to interim.")
if __name__ == "__main__":
    main()
