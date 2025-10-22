#!/usr/bin/env python3
"""
Normalize evaluation CSVs into a standard schema the paper builder expects.

Outputs (overwrites by default unless --out-* given):
  - Validation: columns ['y_true','latent','inducible','productive'] with probs in [0,1] summing ~1
  - Donor:      columns ['y_true','p_productive'] with probs in [0,1]

Heuristics:
- Label column: prefers one of y_true,label,y,target,state,class,truth
- Maps donor labels like 'donor_untreated'→'latent', 'donor_tcr'→'productive'
- Probability columns:
    * Validation: tries to find columns that contain 'latent','inducible','productive' (any case)
                  If only two are found, computes the missing one as 1 - sum(others)
                  If none/one found → warns and skips multiclass probs
    * Donor:      tries 'p_productive','prob_productive','productive','productive_proba','p1','proba_productive'
                  If 'latent' and 'productive' exist, uses 'productive'
                  If only logits like 'logit_productive' exist, applies sigmoid
                  If only predicted label exists, uses 0/1 fallback (warns)
"""
import argparse, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

LABEL_CAND = ["y_true","label","y","target","state","class","truth"]
VAL_CLASS_NAMES = ["latent","inducible","productive"]
DONOR_POS = "productive"

def find_label_col(df: pd.DataFrame):
    for c in LABEL_CAND:
        if c in df.columns: return c
    return None

def map_donor_labels_to_binary(y: pd.Series):
    s = y.astype(str).str.lower()
    # Common donor tokens
    s = s.replace({
        "productive": "productive",
        "latent": "latent",
        "donor_tcr": "productive",
        "tcr": "productive",
        "donor_untreated": "latent",
        "untreated": "latent",
        "ctrl": "latent",
        "control": "latent"
    })
    return s

def detect_val_prob_matrix(df: pd.DataFrame):
    cols_lower = {c.lower(): c for c in df.columns}
    hits = {}
    for cls in VAL_CLASS_NAMES:
        # exact or contains
        exact = cols_lower.get(cls)
        if exact is not None:
            hits[cls] = exact
            continue
        # try fuzzy contains (e.g., prob_latent, latent_proba)
        cand = [c for c in df.columns if re.search(rf"{cls}", c, re.I)]
        if cand:
            hits[cls] = cand[0]
    return hits

def renorm_rows(P: np.ndarray):
    P = np.clip(P, 0, 1)
    row_sum = P.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return P / row_sum

def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1/(1+np.exp(-x))

def normalize_validation(path_in: Path, path_out: Path):
    if not path_in or not path_in.exists(): 
        print(f"[skip] validation file not found: {path_in}")
        return False
    df = pd.read_csv(path_in)
    # label
    ycol = find_label_col(df)
    if ycol is None:
        print(f"[warn] validation: no label column found in {path_in.name}")
        y = None
    else:
        y = df[ycol].astype(str).str.lower()
        # Map obvious donor-like tokens back to canonical classes (if present here)
        y = y.replace({"untreated":"latent","ctrl":"latent","control":"latent",
                       "tcr":"productive","saha":"inducible"})
    # probs
    hits = detect_val_prob_matrix(df)
    if len(hits) == 0:
        print(f"[warn] validation: could not find any class-prob columns in {path_in.name}")
        # write minimal output to help debugging
        out = pd.DataFrame()
        if y is not None: out["y_true"] = y
        out.to_csv(path_out, index=False)
        return y is not None
    # Build matrix; compute missing if exactly two are present
    P_cols = []
    for cls in VAL_CLASS_NAMES:
        if cls in hits:
            P_cols.append(df[hits[cls]].astype(float).values)
        else:
            P_cols.append(None)
    n = len(df)
    # If exactly one missing, derive it
    if sum(p is None for p in P_cols) == 1:
        miss_idx = [i for i,p in enumerate(P_cols) if p is None][0]
        present = np.vstack([p for p in P_cols if p is not None]).T
        rest_sum = present.sum(axis=1)
        P_cols[miss_idx] = np.clip(1.0 - rest_sum, 0, 1)
    # If more than one missing, bail with warning
    if any(p is None for p in P_cols):
        print(f"[warn] validation: insufficient probability columns to form 3-class matrix in {path_in.name}")
        out = pd.DataFrame()
        if y is not None: out["y_true"] = y
        out.to_csv(path_out, index=False)
        return y is not None
    P = np.vstack(P_cols).T
    P = renorm_rows(P)
    out = pd.DataFrame({
        "latent": P[:,0],
        "inducible": P[:,1],
        "productive": P[:,2],
    })
    if y is not None:
        out.insert(0, "y_true", y.values)
    out.to_csv(path_out, index=False)
    print(f"[ok] normalized validation → {path_out}")
    return True

def normalize_donor(path_in: Path, path_out: Path):
    if not path_in or not path_in.exists(): 
        print(f"[skip] donor file not found: {path_in}")
        return False
    df = pd.read_csv(path_in)
    ycol = find_label_col(df)
    y = df[ycol] if ycol else None
    if y is not None:
        y = map_donor_labels_to_binary(y)
    # choose prob column
    candidates = ["p_productive","prob_productive","productive","productive_proba","p1","proba_productive"]
    pcol = next((c for c in candidates if c in df.columns), None)
    if pcol is None:
        # If we have latent/productive pair, use 'productive'
        if "productive" in df.columns: pcol = "productive"
        elif "logit_productive" in df.columns:
            df["p_productive"] = sigmoid(df["logit_productive"].values)
            pcol = "p_productive"
        elif "score_productive" in df.columns:
            # min-max to [0,1] as last resort
            s = df["score_productive"].astype(float).values
            s = (s - s.min())/(s.max()-s.min() + 1e-12)
            df["p_productive"] = s
            pcol = "p_productive"
    if pcol is None and y is not None:
        # last fallback: convert predicted label to hard prob
        hard = (y == DONOR_POS).astype(int).values
        df["p_productive"] = hard
        pcol = "p_productive"
        print("[warn] donor: only hard labels found; using 0/1 as probabilities (uncalibrated).")
    if pcol is None:
        print(f"[warn] donor: no usable probability column in {path_in.name}")
        out = pd.DataFrame()
        if y is not None: out["y_true"] = y.values
        out.to_csv(path_out, index=False)
        return y is not None

    p = df[pcol].astype(float).clip(0,1).values
    out = pd.DataFrame({"p_productive": p})
    if y is not None:
        # convert to 0/1 or strings accepted by downstream
        ybin = (y == DONOR_POS).astype(int)
        out.insert(0, "y_true", ybin.values)  # builder accepts either int or strings
    out.to_csv(path_out, index=False)
    print(f"[ok] normalized donor → {path_out}")
    return True

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val", type=Path, default=Path("data/processed/eval/predictions_val.csv"))
    p.add_argument("--donor", type=Path, default=Path("data/processed/eval/predictions_donor.csv"))
    p.add_argument("--val-uncal", type=Path, default=None)
    p.add_argument("--donor-uncal", type=Path, default=None)
    p.add_argument("--out-val", type=Path, default=None)
    p.add_argument("--out-donor", type=Path, default=None)
    p.add_argument("--out-val-uncal", type=Path, default=None)
    p.add_argument("--out-donor-uncal", type=Path, default=None)
    args = p.parse_args()

    out_val   = args.out_val or args.val
    out_donor = args.out_donor or args.donor
    out_val_u = args.out_val_uncal or args.val_uncal
    out_don_u = args.out_donor_uncal or args.donor_uncal

    ok_val = normalize_validation(args.val, out_val)
    ok_don = normalize_donor(args.donor, out_donor)

    if args.val_uncal:
        normalize_validation(args.val_uncal, out_val_u)

    if args.donor_uncal:
        normalize_donor(args.donor_uncal, out_don_u)

    if not ok_val:
        print("[note] Validation normalization missed labels and/or probs. Open the CSV and check columns.")
    if not ok_don:
        print("[note] Donor normalization missed labels and/or probs. Open the CSV and check columns.")

if __name__ == "__main__":
    main()
