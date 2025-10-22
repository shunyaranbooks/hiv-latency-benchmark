#!/usr/bin/env python
from __future__ import annotations
import re, gzip
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw/GSE180133")
INTERIM = Path("data/interim")

def read_tsv_any(path: Path) -> pd.DataFrame:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return pd.read_csv(f, sep="\t", index_col=0)
    return pd.read_csv(path, sep="\t", index_col=0)

def read_samplesheet(path: Path) -> pd.DataFrame | None:
    try:
        if str(path).endswith(".gz"):
            with gzip.open(path, "rt") as f:
                return pd.read_csv(f, sep="\t")
        return pd.read_csv(path, sep="\t")
    except Exception:
        return None

def heur_label_from_text(text: str) -> str | None:
    s = text.lower()
    if re.search(r"\bpma\b|tcr|cd3|ionomycin|pha", s):
        return "productive"
    if re.search(r"saha|vorinostat|hdac|lra", s):
        return "inducible"
    if re.search(r"dmso|control|ctrl|untreated|(^|_)ut(_|$)|vehicle", s):
        return "latent"
    return None

def label_cells():
    # 1) Load matrix columns (cell IDs) from the *raw* expr matrix.
    mat = RAW_DIR / "GSE180133_exprMatrix.tsv.gz"
    if not mat.exists():
        print("[label_gse180133] Missing matrix:", mat)
        return 2
    df = read_tsv_any(mat)
    cell_ids = [str(c) for c in df.columns]

    # 2) Load sample sheet (optional) and flatten to searchable text
    ss_path = RAW_DIR / "GSE180133_sampleSheet.tsv.gz"
    ss = read_samplesheet(ss_path)
    joined_rows: list[str] = []
    if ss is not None and not ss.empty:
        str_ss = ss.astype(str)
        joined_rows = [" ".join(row.values) for _, row in str_ss.iterrows()]

    # 3) Label each cell using: cell name → heuristics; else sampleSheet match → heuristics; else blank
    rows = []
    for cid in cell_ids:
        lbl = heur_label_from_text(cid)
        if lbl is None and joined_rows:
            # try to match tokens from cell id inside any sample row text
            toks = re.split(r"[_\-\.:]+", cid.lower())
            cand = []
            for t in toks:
                if not t or t.isdigit() or len(t) < 2:
                    continue
                for r in joined_rows:
                    if t in r.lower():
                        cand.append(r)
            if cand:
                # vote by heuristics on concatenated text
                vote_lbl = heur_label_from_text(" ".join(cand))
                if vote_lbl is not None:
                    lbl = vote_lbl
        rows.append((cid, lbl or ""))

    labs = pd.DataFrame(rows, columns=["cell_id", "label"])
    labs.to_csv(RAW_DIR / "GSE180133_labels.csv", index=False)
    # Also copy to interim for the pipeline
    (INTERIM / "GSE180133_labels.csv").write_text((RAW_DIR / "GSE180133_labels.csv").read_text())
    print("[label_gse180133] Label counts:", labs["label"].astype(str).value_counts(dropna=False).to_dict())
    return 0

if __name__ == "__main__":
    raise SystemExit(label_cells())
