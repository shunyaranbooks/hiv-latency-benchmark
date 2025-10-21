from __future__ import annotations
import numpy as np
import pandas as pd

def qc_filter(df: pd.DataFrame, min_genes_per_cell=200, min_cells_per_gene=3):
    cells_detected = (df > 0).sum(axis=0)
    genes_detected = (df > 0).sum(axis=1)
    keep_cells = cells_detected >= min_genes_per_cell
    keep_genes = genes_detected >= min_cells_per_gene
    return df.loc[keep_genes, keep_cells]

def normalize_log1p(df: pd.DataFrame):
    size = df.sum(axis=0).replace(0, np.nan)
    sf = 1e4 * (df / size)
    return np.log1p(sf.fillna(0.0))
