#!/usr/bin/env python
from __future__ import annotations
import re, sys, os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "GSE111727"

LAT_CSV = RAW / "GSE111727_latency_model_labels_template.csv"
DON_CSV = RAW / "GSE111727_donors_labels_template.csv"

# === Ground-truth mapping (from authors' README / Zenodo) ===
# Fluidigm C1 runs: smart33 = untreated, smart34 = SAHA, smart35 = TCR
# We map first token in cell_id (e.g. "33_...") to those runs.
RUN2COND = {33: "latent", 34: "inducible", 35: "productive"}

def label_latency(cell_id: str) -> str:
    # Expect formats like "33_0_C02", "34_1_C15", "35_2_C09"
    try:
        run = int(cell_id.split("_", 1)[0])
    except Exception:
        return ""
    return RUN2COND.get(run, "")  # empty if unknown run

def label_donor(cell_id: str) -> str:
    # Donor files contain untreated vs TCR-treated cells.
    # Middle token encodes treatment replicate: 0 ≈ untreated; 1/2 ≈ TCR.
    # Accept suffixes like '1d','2d'.
    m = re.search(r'_(\d+)[A-Za-z]*_', cell_id)
    if not m:
        # If there's an explicit '_d_' with no number, treat as untreated conservatively.
        return "donor_untreated" if "_d_" in cell_id else ""
    code = int(m.group(1))
    if code == 0:
        return "donor_untreated"
    if code in (1, 2):
        return "donor_tcr"
    return ""

def fill_labels(template_csv: Path, mapper, valid: set[str], name: str):
    if not template_csv.exists():
        print(f"[WARN] Template not found: {template_csv}")
        return 0
    df = pd.read_csv(template_csv)
    if "cell_id" not in df.columns or "label" not in df.columns:
        raise SystemExit(f"{template_csv} must have columns: cell_id,label")

    # Fill only missing/blank labels to avoid overwriting any manual corrections.
    s = df["label"].astype(str).str.strip()
    missing = s.eq("") | s.eq("nan") | s.eq("NaN")
    df.loc[missing, "label"] = df.loc[missing, "cell_id"].map(mapper)

    # Validate
    bad = ~df["label"].isin(valid)
    n_bad = int(bad.sum())
    if n_bad:
        print(f"[ERROR] {name}: {n_bad} labels unresolved or invalid. Showing first 15:")
        print(df.loc[bad, "cell_id"].head(15).to_list())
        print(f"→ Please correct these in {template_csv} (allowed={sorted(valid)})")
    else:
        print(f"[OK] {name}: all labels valid. Counts:", df["label"].value_counts().to_dict())

    df.to_csv(template_csv, index=False)
    return n_bad

def main():
    n_err = 0
    n_err += fill_labels(
        LAT_CSV,
        label_latency,
        {"latent", "inducible", "productive"},
        "Latency model"
    )
    n_err += fill_labels(
        DON_CSV,
        label_donor,
        {"donor_untreated", "donor_tcr"},
        "Donors"
    )
    if n_err:
        sys.exit(2)

if __name__ == "__main__":
    main()
