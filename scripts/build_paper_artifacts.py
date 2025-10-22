#!/usr/bin/env python3
"""
Build all manuscript artifacts (figures + tables) from predictions and report.json.

What it makes (into docs/paper/*):
  figures/
    fig_reliability_val_cal.png       # validation reliability (if probs available)
    fig_reliability_val_uncal.png     # optional, if --val-uncal is provided
    fig_reliability_donor_cal.png     # donor reliability
    fig_reliability_donor_uncal.png   # optional, if --donor-uncal is provided
    fig_val_roc_latent.png            # OvR ROC per class (validation)
    fig_val_pr_latent.png             # OvR PR per class (validation)
    fig_val_roc_inducible.png
    fig_val_pr_inducible.png
    fig_val_roc_productive.png
    fig_val_pr_productive.png
    fig_donor_roc.png                 # donor ROC
    fig_donor_pr.png                  # donor PR
    reliability_ece.json              # ECE summary

  tables/
    table1_dataset_summary.csv        # static scaffold (edit if needed)
    table2_validation_metrics.csv     # from report.json or computed
    table3_donor_metrics.csv          # from report.json or computed
    table3_donor_bootstrap.csv        # bootstrap 95% CI (if --bootstrap>0)
    appendix_prediction_counts.csv    # row counts per prediction file
"""
import argparse, json, math, os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    log_loss
)

# ---------------------- I/O helpers ---------------------- #

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_csv_lenient(p: Optional[Path]) -> Optional[pd.DataFrame]:
    if p is None: return None
    if not p.exists(): return None
    try:
        return pd.read_csv(p)
    except Exception:
        try:
            return pd.read_csv(p, sep="\t")
        except Exception:
            return None

# ---------------------- detection utilities ---------------------- #

_LABEL_CANDIDATES = ["y_true","label","y","target"]
# prefer explicit probability names first
_PROB_PREFS = [
    "p_productive","prob_productive","productive_proba","productive","p1","proba_productive"
]
_CLASS_NAMES_PREFS = ["latent","inducible","productive"]

def detect_label_col(df: pd.DataFrame) -> Optional[str]:
    for c in _LABEL_CANDIDATES:
        if c in df.columns:
            return c
    return None

def detect_binary_prob_col(df: pd.DataFrame, ycol: Optional[str]) -> Optional[str]:
    for c in _PROB_PREFS:
        if c in df.columns:
            return c
    # fallback: any numeric 0..1 column not the label
    num = df.select_dtypes(include="number")
    cands = [c for c in num.columns if c != ycol and num[c].between(0,1).all()]
    return cands[0] if cands else None

def detect_multiclass_proba_matrix(df: pd.DataFrame, class_names: Optional[List[str]] = None) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """
    Try to find class-prob columns for latent/inducible/productive.
    Strategy:
      1) exact matches if class_names given
      2) try lowercased exact class names
      3) choose 3 numeric 0..1 columns that sum ~1 across rows
    Returns (P, classes) where P shape is (n, K)
    """
    class_names = class_names or _CLASS_NAMES_PREFS
    low_cols = {c.lower(): c for c in df.columns}
    # exact names
    hit = []
    for cls in class_names:
        if cls in low_cols:
            hit.append(low_cols[cls])
    if len(hit) == len(class_names):
        P = df[hit].to_numpy(dtype=float)
        return P, class_names

    # search for likely prob triplets
    num = df.select_dtypes(include="number")
    if num.shape[1] >= 3:
        cols = list(num.columns)
        # heuristic: choose the 3 with mean row sum closest to 1
        best = None
        best_trip = None
        m = min(10, len(cols))
        # try first m columns combos for speed
        for i in range(m):
            for j in range(i+1, m):
                for k in range(j+1, m):
                    trip = [cols[i], cols[j], cols[k]]
                    X = num[trip].to_numpy()
                    row_sum = X.sum(axis=1)
                    closeness = abs(row_sum.mean() - 1.0) + row_sum.std()
                    if best is None or closeness < best:
                        best, best_trip = closeness, trip
        if best_trip:
            P = num[best_trip].to_numpy(dtype=float)
            # names unknown; map to prefs length
            classes = class_names[:3]
            return P, classes
    return None, None

def binarize_labels(y: pd.Series, positive="productive") -> np.ndarray:
    if y.dtype == object:
        return (y.str.lower() == positive).astype(int).to_numpy()
    arr = y.to_numpy()
    # assume 0/1 integers
    return (arr == 1).astype(int)

# ---------------------- metrics ---------------------- #

def ece_maxprob_binary(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """ECE using max-prob for binary (p is prob of positive)."""
    bins = np.linspace(0, 1, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (p >= lo) & (p < hi) if i < n_bins-1 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        conf = p[mask].mean()
        acc = ((p[mask] >= 0.5).astype(int) == y[mask]).mean()
        ece += abs(acc - conf) * (mask.sum() / len(p))
    return float(ece)

def ece_maxprob_multiclass(y: np.ndarray, P: np.ndarray, n_bins: int = 15) -> float:
    """ECE on max-prob for multiclass with integer labels 0..K-1."""
    maxp = P.max(axis=1)
    yhat = P.argmax(axis=1)
    bins = np.linspace(0, 1, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (maxp >= lo) & (maxp < hi) if i < n_bins-1 else (maxp >= lo) & (maxp <= hi)
        if not mask.any():
            continue
        conf = maxp[mask].mean()
        acc = (yhat[mask] == y[mask]).mean()
        ece += abs(acc - conf) * (mask.sum() / len(maxp))
    return float(ece)

def brier_binary(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y)**2))

def nll_binary(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(log_loss(y, np.vstack([1-p, p]).T, labels=[0,1]))

def macro_ovr_auc_ap(y_str: pd.Series, P: np.ndarray, classes: List[str]) -> Tuple[float, float]:
    """Macro average over OvR AUROC and AP."""
    y_str = y_str.astype(str).str.lower()
    aucs, aps = [], []
    for idx, cls in enumerate(classes):
        y_bin = (y_str == cls).astype(int).to_numpy()
        p = P[:, idx]
        try:
            aucs.append(roc_auc_score(y_bin, p))
        except ValueError:
            pass
        try:
            aps.append(average_precision_score(y_bin, p))
        except ValueError:
            pass
    auc_macro = float(np.mean(aucs)) if aucs else np.nan
    ap_macro  = float(np.mean(aps))  if aps  else np.nan
    return auc_macro, ap_macro

# ---------------------- plots ---------------------- #

def plot_reliability_binary(y: np.ndarray, p: np.ndarray, title: str, ofile: Path, n_bins: int = 15) -> float:
    ece = ece_maxprob_binary(y, p, n_bins=n_bins)
    bins = np.linspace(0, 1, n_bins+1)
    xs, ys, ws = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (p >= lo) & (p < hi) if i < n_bins-1 else (p >= lo) & (p <= hi)
        if not mask.any(): continue
        conf = p[mask].mean()
        acc = ((p[mask] >= 0.5).astype(int) == y[mask]).mean()
        xs.append(conf); ys.append(acc); ws.append(mask.sum())
    plt.figure()
    plt.plot([0,1],[0,1],'--')
    plt.scatter(xs, ys, s=(np.array(ws)*2).astype(float), alpha=0.7)
    plt.xlabel("Predicted confidence")
    plt.ylabel("Empirical accuracy")
    plt.title(f"{title}\nECE={ece:.3f}")
    plt.tight_layout(); plt.savefig(ofile, dpi=300); plt.close()
    return ece

def plot_reliability_multiclass(y_int: np.ndarray, P: np.ndarray, title: str, ofile: Path, n_bins: int = 15) -> float:
    ece = ece_maxprob_multiclass(y_int, P, n_bins=n_bins)
    maxp = P.max(axis=1)
    yhat = P.argmax(axis=1)
    bins = np.linspace(0, 1, n_bins+1)
    xs, ys, ws = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (maxp >= lo) & (maxp < hi) if i < n_bins-1 else (maxp >= lo) & (maxp <= hi)
        if not mask.any(): continue
        conf = maxp[mask].mean()
        acc = (yhat[mask] == y_int[mask]).mean()
        xs.append(conf); ys.append(acc); ws.append(mask.sum())
    plt.figure()
    plt.plot([0,1],[0,1],'--')
    plt.scatter(xs, ys, s=(np.array(ws)*2).astype(float), alpha=0.7)
    plt.xlabel("Predicted confidence")
    plt.ylabel("Empirical accuracy")
    plt.title(f"{title}\nECE={ece:.3f}")
    plt.tight_layout(); plt.savefig(ofile, dpi=300); plt.close()
    return ece

def plot_roc_pr_binary(y: np.ndarray, p: np.ndarray, prefix: Path, title: str):
    fpr,tpr,_ = roc_curve(y,p); auc_val = roc_auc_score(y,p)
    prec,rec,_ = precision_recall_curve(y,p); ap = average_precision_score(y,p)

    plt.figure()
    plt.plot(fpr,tpr,label=f"AUC={auc_val:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC — {title}"); plt.legend()
    plt.tight_layout(); plt.savefig(prefix.parent / f"{prefix.name}_roc.png", dpi=300); plt.close()

    plt.figure()
    plt.plot(rec,prec,label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR — {title}"); plt.legend()
    plt.tight_layout(); plt.savefig(prefix.parent / f"{prefix.name}_pr.png", dpi=300); plt.close()

def plot_ovr_validation(y_str: pd.Series, P: np.ndarray, classes: List[str], out_dir: Path):
    y_str = y_str.astype(str).str.lower()
    for i, cls in enumerate(classes):
        y_bin = (y_str == cls).astype(int).to_numpy()
        p = P[:, i]
        plot_roc_pr_binary(y_bin, p, out_dir / f"fig_val_{cls}", title=f"Validation — {cls} (OvR)")

# ---------------------- bootstrap ---------------------- #

def bootstrap_binary(y: np.ndarray, p: np.ndarray, n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metrics = []
    for _ in range(n):
        idx = rng.integers(0, len(y), size=len(y))
        yb, pb = y[idx], p[idx]
        try:
            auroc = roc_auc_score(yb, pb)
        except ValueError:
            auroc = np.nan
        try:
            auprc = average_precision_score(yb, pb)
        except ValueError:
            auprc = np.nan
        brier = brier_binary(yb, pb)
        metrics.append((auroc, auprc, brier))
    arr = np.array(metrics, dtype=float)
    df = pd.DataFrame(arr, columns=["auroc","auprc","brier"])
    out = []
    for m in ["auroc","auprc","brier"]:
        series = df[m].dropna()
        if series.empty:
            mean = lo = hi = np.nan
        else:
            mean = float(series.mean())
            lo, hi = float(series.quantile(0.025)), float(series.quantile(0.975))
        out.append({"metric": m.upper(), "mean": mean, "CI95_lower": lo, "CI95_upper": hi})
    return pd.DataFrame(out)

# ---------------------- main ---------------------- #

def main():
    ap = argparse.ArgumentParser(description="Generate paper figures and tables from predictions.")
    ap.add_argument("--val", type=Path, default=Path("data/processed/eval/predictions_val.csv"))
    ap.add_argument("--val-uncal", type=Path, default=None)
    ap.add_argument("--donor", type=Path, default=Path("data/processed/eval/predictions_donor.csv"))
    ap.add_argument("--donor-uncal", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=Path("data/processed/eval/report.json"))
    ap.add_argument("--out-fig", type=Path, default=Path("docs/paper/figures"))
    ap.add_argument("--out-tab", type=Path, default=Path("docs/paper/tables"))
    ap.add_argument("--n-bins", type=int, default=15)
    ap.add_argument("--bootstrap", type=int, default=2000, help="0 to skip bootstrap CI")
    args = ap.parse_args()

    ensure_dir(args.out_fig); ensure_dir(args.out_tab)

    # ---------- Table 1 scaffold (edit if you like) ----------
    table1 = pd.DataFrame([
        {"Accession":"GSE111727","Role in Benchmark":"Train/Validation (latency model)","Organism":"Homo sapiens","Experiment":"scRNA-seq","Public Since":"2018-03-13"},
        {"Accession":"GSE111727 (donors)","Role in Benchmark":"External test (donor cells)","Organism":"Homo sapiens","Experiment":"scRNA-seq","Public Since":"2018-03-13"},
        {"Accession":"GSE180133","Role in Benchmark":"External stress-test (latency reversal)","Organism":"Homo sapiens","Experiment":"scRNA-seq","Public Since":"2021-07-15"},
        {"Accession":"GSE199727","Role in Benchmark":"External stress-test (ex vivo reservoir)","Organism":"Homo sapiens","Experiment":"multi-omic (varies)","Public Since":"2022-10-21"},
    ])
    table1.to_csv(args.out_tab/"table1_dataset_summary.csv", index=False)

    # ---------- Load CSVs ----------
    df_val   = read_csv_lenient(args.val)
    df_val_u = read_csv_lenient(args.val_uncal) if args.val_uncal else None
    df_don   = read_csv_lenient(args.donor)
    df_don_u = read_csv_lenient(args.donor_uncal) if args.donor_uncal else None

    counts = []
    for name, df, p in [
        ("Validation (cal)", df_val, args.val),
        ("Validation (uncal)", df_val_u, args.val_uncal),
        ("Donor (cal)", df_don, args.donor),
        ("Donor (uncal)", df_don_u, args.donor_uncal),
    ]:
        if df is not None:
            counts.append({"Split/File": name, "Rows": int(df.shape[0]), "File": str(p)})
    pd.DataFrame(counts).to_csv(args.out_tab/"appendix_prediction_counts.csv", index=False)

    # ---------- Validation (multiclass) ----------
    val_metrics = {"auroc_macro_ovr": None, "auprc_macro_ovr": None, "ece_maxprob": None}
    if df_val is not None:
        ycol = detect_label_col(df_val)
        if ycol is None:
            print("[warn] validation: no label column found; skipping.")
        else:
            P, classes = detect_multiclass_proba_matrix(df_val, _CLASS_NAMES_PREFS)
            if P is not None:
                # Map y to integers 0..K-1 according to 'classes'
                y_str = df_val[ycol].astype(str).str.lower()
                cls_to_idx = {c:i for i,c in enumerate(classes)}
                y_int = y_str.map(cls_to_idx).to_numpy()
                # Drop rows with unknown labels, if any
                mask = ~np.isnan(y_int)
                y_int = y_int[mask].astype(int); P = P[mask]
                # metrics
                auc_macro, ap_macro = macro_ovr_auc_ap(y_str[mask], P, classes)
                ece = ece_maxprob_multiclass(y_int, P, n_bins=args.n_bins)
                val_metrics.update({"auroc_macro_ovr": auc_macro, "auprc_macro_ovr": ap_macro, "ece_maxprob": ece})
                # plots (OvR ROC/PR per class)
                plot_ovr_validation(y_str[mask], P, classes, args.out_fig)
                # reliability
                plot_reliability_multiclass(y_int, P, "Validation (calibrated)", args.out_fig/"fig_reliability_val_cal.png", n_bins=args.n_bins)
            else:
                print("[warn] validation: could not detect multiclass probabilities; skipping plots/metrics.")

    # Optional: validation uncalibrated reliability if provided
    if df_val_u is not None:
        ycol_u = detect_label_col(df_val_u)
        P_u, classes_u = detect_multiclass_proba_matrix(df_val_u, _CLASS_NAMES_PREFS)
        if ycol_u and P_u is not None:
            y_str_u = df_val_u[ycol_u].astype(str).str.lower()
            y_int_u = y_str_u.map({c:i for i,c in enumerate(classes_u)}).to_numpy()
            mask = ~np.isnan(y_int_u); y_int_u = y_int_u[mask].astype(int); P_u = P_u[mask]
            plot_reliability_multiclass(y_int_u, P_u, "Validation (uncalibrated)", args.out_fig/"fig_reliability_val_uncal.png", n_bins=args.n_bins)

    # ---------- Donor (binary) ----------
    donor_metrics = {"auroc_binary": None, "auprc_binary": None, "ece_maxprob": None, "brier": None, "nll": None}
    donor_boot_df = None

    if df_don is not None:
        ycol = detect_label_col(df_don)
        pcol = detect_binary_prob_col(df_don, ycol)
        if ycol and pcol:
            y = binarize_labels(df_don[ycol])
            p = df_don[pcol].astype(float).clip(1e-9, 1-1e-9).to_numpy()
            try:
                donor_metrics["auroc_binary"] = float(roc_auc_score(y, p))
            except ValueError:
                donor_metrics["auroc_binary"] = np.nan
            try:
                donor_metrics["auprc_binary"] = float(average_precision_score(y, p))
            except ValueError:
                donor_metrics["auprc_binary"] = np.nan
            donor_metrics["ece_maxprob"] = ece_maxprob_binary(y, p, n_bins=args.n_bins)
            donor_metrics["brier"] = brier_binary(y, p)
            donor_metrics["nll"]   = nll_binary(y, p)

            # plots
            plot_reliability_binary(y, p, "Donor (calibrated)", args.out_fig/"fig_reliability_donor_cal.png", n_bins=args.n_bins)
            plot_roc_pr_binary(y, p, args.out_fig/"fig_donor", "Donor")

            # bootstrap CIs
            if args.bootstrap > 0:
                donor_boot_df = bootstrap_binary(y, p, n=args.bootstrap, seed=42)
                donor_boot_df.to_csv(args.out_tab/"table3_donor_bootstrap.csv", index=False)
        else:
            print("[warn] donor: could not detect label/prob columns; skipping donor metrics.")

    if df_don_u is not None:
        ycol = detect_label_col(df_don_u)
        pcol = detect_binary_prob_col(df_don_u, ycol)
        if ycol and pcol:
            y = binarize_labels(df_don_u[ycol])
            p = df_don_u[pcol].astype(float).clip(1e-9, 1-1e-9).to_numpy()
            plot_reliability_binary(y, p, "Donor (uncalibrated)", args.out_fig/"fig_reliability_donor_uncal.png", n_bins=args.n_bins)

    # ---------- Write ECE summary JSON ----------
    ece_json = {}
    if val_metrics["ece_maxprob"] is not None:
        ece_json["val_ece_cal"] = val_metrics["ece_maxprob"]
    if donor_metrics["ece_maxprob"] is not None:
        ece_json["donor_ece_cal"] = donor_metrics["ece_maxprob"]
    if ece_json:
        (args.out_fig/"reliability_ece.json").write_text(json.dumps(ece_json, indent=2))

    # ---------- Tables 2 and 3 from report.json (or computed fallback) ----------
    rep = {}
    if args.report.exists():
        try:
            rep = json.loads(args.report.read_text())
        except Exception:
            rep = {}

    # Table 2: validation
    val_row = {
        "Split": "Validation (multiclass OvR)",
        "AUROC_macro": rep.get("val",{}).get("auroc_macro_ovr", val_metrics["auroc_macro_ovr"]),
        "AUPRC_macro": rep.get("val",{}).get("auprc_macro_ovr", val_metrics["auprc_macro_ovr"]),
        "ECE_maxprob": rep.get("val",{}).get("ece_maxprob",      val_metrics["ece_maxprob"]),
    }
    pd.DataFrame([val_row]).to_csv(args.out_tab/"table2_validation_metrics.csv", index=False)

    # Table 3: donor
    don_row = {
        "Split": "Donor (binary; prior-corrected if applicable)",
        "AUROC": rep.get("donor",{}).get("auroc_binary", donor_metrics["auroc_binary"]),
        "AUPRC": rep.get("donor",{}).get("auprc_binary", donor_metrics["auprc_binary"]),
        "ECE_maxprob": rep.get("donor",{}).get("ece_maxprob", donor_metrics["ece_maxprob"]),
        "Brier": donor_metrics["brier"],
        "NLL": donor_metrics["nll"],
    }
    pd.DataFrame([don_row]).to_csv(args.out_tab/"table3_donor_metrics.csv", index=False)

    print(f"[ok] Figures → {args.out_fig}")
    print(f"[ok] Tables  → {args.out_tab}")

if __name__ == "__main__":
    main()
