#!/usr/bin/env python
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import requests
from hivlat.utils.io import ensure_dir, update_manifest

FILES = [
  # (subdir, filename, url)
  ("GSE111727", "GSE111727_sc_lat_model_raw_counts_genes.txt.gz",
   "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE111nnn/GSE111727/suppl/GSE111727_sc_lat_model_raw_counts_genes.txt.gz"),
  ("GSE111727", "GSE111727_sc_donors_raw_counts_genes.txt.gz",
   "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE111nnn/GSE111727/suppl/GSE111727_sc_donors_raw_counts_genes.txt.gz"),
  ("GSE111727", "GSE111727_sc_lat_model_raw_counts_spikes.txt.gz",
   "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE111nnn/GSE111727/suppl/GSE111727_sc_lat_model_raw_counts_spikes.txt.gz"),
]

def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()

def main(raw_root: str = "data/raw", manifest_path: str = "data/manifest.json"):
    raw_root = Path(raw_root)
    ensure_dir(raw_root)
    for subdir, fname, url in FILES:
        outdir = ensure_dir(raw_root / subdir)
        out = outdir / fname
        print(f"Downloading {fname} ...")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        out.write_bytes(r.content)
        digest = sha256sum(out)
        update_manifest(manifest_path, fname, url, digest, out.stat().st_size)
        print(f"  ↳ saved {out} ({out.stat().st_size/1e6:.2f} MB), sha256={digest[:12]}…")
    print("All downloads complete. Edit FILES to add externals (GSE199727, GSE180133) when ready.")

if __name__ == "__main__":
    sys.exit(main())
