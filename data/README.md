This folder stores datasets in three stages:

- `raw/`       : Original GEO/IPDA files (NOT committed to git).
- `interim/`   : Cleaned/normalized matrices (Parquet/H5AD).
- `processed/` : Canonical splits, trained models, and evaluation outputs.

Use `scripts/download_geo.py` to fetch raw files. The script records `manifest.json` with URL, bytes, and SHA-256.
