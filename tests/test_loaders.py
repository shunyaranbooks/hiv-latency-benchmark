from pathlib import Path
from hivlat.data.loaders import save_parquet
import pandas as pd

def test_save_parquet(tmp_path: Path):
    df = pd.DataFrame([[1,0],[0,1]], index=['g1','g2'], columns=['c1','c2'])
    out = tmp_path / 'x.parquet'
    save_parquet(df, out)
    assert out.exists()
