from __future__ import annotations
from pathlib import Path
import gzip
import pandas as pd

def load_counts_txt_gz(path: str | Path) -> pd.DataFrame:
    """Load a GEO-style gzipped TSV with genes as rows and cells as columns."""
    path = Path(path)
    with gzip.open(path, 'rt') as f:
        df = pd.read_csv(f, sep='\t', index_col=0)
    return df  # rows=genes, cols=cells

def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)

def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
