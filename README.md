# HIV Latency Benchmark: Cross-Study, Uncertainty-Aware Classifier

A reproducible, **laptop-friendly** benchmark and baseline model that distinguishes **latent ↔ inducible-latent ↔ productive** HIV-infected CD4 T cells from single-cell RNA-seq, reporting **uncertainty (calibration)** alongside accuracy.

> **Why this matters.** The latent reservoir is the key barrier to an HIV cure. A **generalizable, calibrated** classifier and a clean evaluation protocol help method developers and translational teams compare ideas fairly and prioritize strategies.


---

## Table of Contents
- [What’s in this repo](#whats-in-this-repo)
- [Data sources](#data-sources)
- [Install & quick start](#install--quick-start)
- [Label provenance (transparent & reproducible)](#label-provenance-transparent--reproducible)
- [Configuration](#configuration)
- [Modeling defaults](#modeling-defaults)
- [What the evaluation reports](#what-the-evaluation-reports)
- [Build paper artifacts (figures + tables)](#build-paper-artifacts-figures--tables)
- [Optional: ROC/PR charts](#optional-rocpr-charts)
- [Optional: Donor bootstrap → CSV](#optional-donor-bootstrap--csv)
- [Repo layout](#repo-layout)
- [Current run snapshot](#current-run-snapshot)
- [Troubleshooting](#troubleshooting)
- [Scope, safety, and ethics](#scope-safety-and-ethics)
- [Citation](#citation)
- [Maintainer](#maintainer)
- [License](#license)

---

## What’s in this repo

**CLI programs** (`scripts/`):
- `download_geo.py` – fetch exact GEO supplementary files (e.g., GSE111727 raw matrices) and log checksums to `data/manifest.json`.
- `prepare_data.py` – QC + normalization + labeling → harmonized matrices.
- `split_dataset.py` – canonical train/validation (on the latency model only); donor/external sets held out.
- `train.py` – train multiclass baseline (**Logistic** or **GBM**) with **isotonic calibration**.
- `evaluate.py` – report **AUROC/AUPRC** and **Expected Calibration Error (ECE)** for validation; **binary AUROC/AUPRC/ECE** for donor/external sets.

**Auto-labeling**:
- `auto_label_gse111727.py` – fills label templates using dataset-provenance rules so you don’t hand-edit CSVs.

**Stack**: Python 3.10+, pandas, numpy, scikit-learn, scipy, **pyarrow** (for Parquet). No GPU required.

---

## Data sources

Public datasets (not shipped in this repo):

- **GSE111727** — primary latency model (untreated / SAHA / TCR) **and** HIV+ donor cells (external test).
- **GSE199727** — reservoir CD4+ T cells (planned external transfer test).
- **GSE180133** — single-cell latency reversal (planned stress test).
- **IPDA cohorts** — intact vs defective provirus (planned patient-level anchor).

Use `scripts/download_geo.py` to populate `data/raw/...`. File sizes and SHA-256 checksums are recorded in `data/manifest.json`.

---

## Install & quick start

> **Tip:** Install from the **project root** to register the `hivlat` package; this prevents `ModuleNotFoundError` in scripts.

```bash
# 0) Create environment
python -m venv .venv
source .venv/bin/activate

# 1) Install package + CLI deps (installs src/hivlat)
pip install -e .

# (if Parquet error) ensure pyarrow
pip install pyarrow

# 2) Download public files (starts with GSE111727)
python scripts/download_geo.py

# 3) Create label templates & auto-fill from provenance rules
python scripts/auto_label_gse111727.py

# 4) Prepare, split, train, evaluate
python scripts/prepare_data.py --config configs/default.yaml
python scripts/split_dataset.py   --config configs/default.yaml
python scripts/train.py           --config configs/default.yaml --model-out data/processed/models/logreg.joblib
python scripts/evaluate.py        --config configs/default.yaml --model data/processed/models/logreg.joblib
```

Outputs land under `data/processed/`:
- `X_train.parquet`, `X_val.parquet`, `y_train.csv`, `y_val.csv`
- `models/*.joblib`
- `eval/report.json`, `eval/predictions_val.csv`, `eval/predictions_donor.csv`

---

## Label provenance (transparent & reproducible)

This repo **does not guess** labels; it encodes published provenance rules in code (`scripts/auto_label_gse111727.py`) and writes them into two templates:

- `data/raw/GSE111727/GSE111727_latency_model_labels_template.csv`
- `data/raw/GSE111727/GSE111727_donors_labels_template.csv`

**GSE111727 – Latency model (train/val)**  
Column IDs carry **Fluidigm C1 run IDs**:
- `33_*` → **untreated** → label **latent**
- `34_*` → **SAHA/Vorinostat** → label **inducible**
- `35_*` → **TCR stimulation** → label **productive**

**GSE111727 – Donors (external test)**  
The **middle token** in the cell ID encodes treatment replicate:
- `_0_` → **donor_untreated**
- `_1_` or `_2_` → **donor_tcr**
- `_d_` (no number) → defaults to **donor_untreated** (conservative)

> The auto-label script fills **only** missing/blank entries so you can override any cell by hand if needed.

**Splits.** Train/validation are **only** from the latency model; all donor/external cells are never used for training.

---

## Configuration

All knobs live in `configs/default.yaml`. Important ones:

```yaml
paths:
  raw: data/raw
  interim: data/interim
  processed: data/processed

prep:
  min_genes_per_cell: 100
  min_cells_per_gene: 3
  normalize: size_factor
  log1p: true
  hvg_n: 2000        # try 1000/2000/3000
  # Reduce domain shift by using only genes present in train ∩ donor
  hvg_intersection_with_donor: true

model:
  random_state: 42
```

Tune logistic regularization via an env var (kept out of YAML for easy sweeps):
```bash
export HIVLAT_C=0.5  # try 0.25, 0.5, 1.0, 2.0
```

Train GBM instead of Logistic:
```bash
python scripts/train.py --config configs/default.yaml --model-type gbm --model-out data/processed/models/gbm.joblib
```

---

## Modeling defaults

- **QC**: `min_genes_per_cell=100`, `min_cells_per_gene=3`, library-size normalize, `log1p`.
- **Features**: top **HVGs** chosen **on training only** (default `hvg_n=2000–3000`).  
  *Recommended:* `prep.hvg_intersection_with_donor: true` to reduce cross-study drift.
- **Classifier**: Logistic Regression (`class_weight='balanced'`, `C` from `HIVLAT_C`) **or** Gradient Boosting (GBM).
- **Calibration**: **Isotonic** on the validation split (reduces ECE / over-confidence).
- **Donor prior shift**: at evaluation apply **Saerens prior correction** using source (val) and target (donor) class priors.

---

## What the evaluation reports

Running `python scripts/evaluate.py ...` prints and writes:

- **Validation (multiclass)**:  
  - `auroc_macro_ovr`  
  - `auprc_macro_ovr`  
  - `ece_maxprob` (Expected Calibration Error)

- **Donor / external (binary: latent vs productive)**:  
  - `auroc_binary`, `auprc_binary`  
  - `ece_maxprob` (after prior correction)

Per-cell probabilities:  
- `data/processed/eval/predictions_val.csv`  
- `data/processed/eval/predictions_donor.csv`

---

## Build paper artifacts (figures + tables)

**Recommended schema for CSVs**  
- `predictions_val.csv` must have: `y_true`, `latent`, `inducible`, `productive` (probs in [0,1]).  
- `predictions_donor.csv` must have: `y_true` (0/1 or latent/productive) and `p_productive`.  

> If your files use different column names, see **[Troubleshooting → Normalizing eval CSVs]** below.

**Generate artifacts (safe line breaks — no comments after `\`):**
```bash
python scripts/build_paper_artifacts.py   --val data/processed/eval/predictions_val.csv   --donor data/processed/eval/predictions_donor.csv   --report data/processed/eval/report.json   --bootstrap 2000   --n-bins 15
```

**If you also have uncalibrated files:**
```bash
python scripts/build_paper_artifacts.py   --val data/processed/eval/predictions_val.csv   --val-uncal data/processed/eval/predictions_val_uncal.csv   --donor data/processed/eval/predictions_donor.csv   --donor-uncal data/processed/eval/predictions_donor_uncal.csv   --report data/processed/eval/report.json   --bootstrap 2000   --n-bins 15
```

Artifacts are written to:
- **Figures:** `docs/paper/figures/fig_reliability_val_cal.png`, `fig_reliability_val_uncal.png`, `fig_reliability_donor_cal.png`, `fig_reliability_donor_uncal.png`, plus `reliability_ece.json`.
- **Tables:** `docs/paper/tables/table1_dataset_summary.csv`, `table2_validation_metrics.csv`, `table3_donor_metrics.csv`, `appendix_prediction_counts.csv`, and (optional) `bootstrap_donor_binary.json`.

---

## Optional: ROC/PR charts

If your journal wants ROC/PR figures too, use the helper in `notebooks/02_train_baseline.ipynb` **or** run the standalone plotter (example below assumes standard column names):

```bash
python - <<'PY'
import pandas as pd, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

figdir = Path("docs/paper/figures"); figdir.mkdir(parents=True, exist_ok=True)
def roc_pr_binary(y, p, prefix, title):
    fpr,tpr,_ = roc_curve(y,p); roc_auc = auc(fpr,tpr)
    prec,rec,_ = precision_recall_curve(y,p); ap = average_precision_score(y,p)
    plt.figure(); plt.plot(fpr,tpr,label=f"AUC={roc_auc:.3f}"); plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title(f"ROC — {title}"); plt.legend(); plt.tight_layout()
    plt.savefig(figdir/f"{prefix}_roc.png", dpi=300); plt.close()
    plt.figure(); plt.plot(rec,prec,label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR — {title}"); plt.legend(); plt.tight_layout()
    plt.savefig(figdir/f"{prefix}_pr.png", dpi=300); plt.close()
    print(f"[ok] {title}: ROC/PR saved → {prefix}_*.png")

# Donor (binary)
don = pd.read_csv("data/processed/eval/predictions_donor.csv")
y = don.get("y_true")
if y is not None:
    y = (y.astype(str).str.lower().isin(["1","productive","true"])).astype(int).values
pcol = next((c for c in ["p_productive","prob_productive","productive","productive_proba","p1","proba_productive"] if c in don.columns), None)
if pcol:
    p = don[pcol].astype(float).clip(1e-9,1-1e-9).values
    roc_pr_binary(y, p, "fig_donor", "Donor")

# Validation (OvR)
val = pd.read_csv("data/processed/eval/predictions_val.csv")
ycol = next((c for c in ["y_true","label","y","target"] if c in val.columns), None)
if ycol and all(c in val.columns for c in ["latent","inducible","productive"]):
    y_str = val[ycol].astype(str).str.lower()
    for cls in ["latent","inducible","productive"]:
        y_bin = (y_str == cls).astype(int).values
        p = val[cls].astype(float).clip(0,1).values
        roc_pr_binary(y_bin, p, f"fig_val_{cls}", f"Validation — {cls} (OvR)")
PY
```

---

## Optional: Donor bootstrap → CSV

Some journals require CIs in tables. Convert bootstraps like this:

```bash
python - <<'PY'
import json, csv
from pathlib import Path
src = Path("docs/paper/tables/bootstrap_donor_binary.json")
dst = Path("docs/paper/tables/table3_donor_bootstrap.csv")
if not src.exists():
    raise SystemExit(f"[error] not found: {src}")
data = json.loads(src.read_text())
rows = []
def add(metric, mean=None, ci=None):
    lo, hi = (ci or [None, None])
    rows.append([metric, mean, lo, hi])

if isinstance(data, list):
    for item in data:
        if isinstance(item, dict):
            add(item.get("metric"), item.get("mean"), item.get("ci95"))
        elif isinstance(item, (list, tuple)):
            add(item[0], item[1] if len(item)>1 else None,
                [item[2] if len(item)>2 else None, item[3] if len(item)>3 else None])
elif isinstance(data, dict):
    for k,v in data.items():
        if isinstance(v, dict):
            add(k, v.get("mean"), v.get("ci95"))
        else:
            add(k, v, None)
else:
    add("unknown", None, None)

with dst.open("w", newline="") as f:
    w = csv.writer(f); w.writerow(["Metric","Mean","CI95_lower","CI95_upper"]); w.writerows(rows)
print(f"[ok] wrote {dst}")
PY
```

---

## Repo layout

```
hiv-latency-benchmark/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ pyproject.toml
├─ configs/
│  └─ default.yaml
├─ data/
│  ├─ README.md
│  ├─ manifest.json           # checksums + source URLs
│  ├─ raw/                    # (ignored) original GEO files
│  │  └─ GSE111727/
│  ├─ interim/                # cleaned, harmonized matrices
│  └─ processed/              # splits, models, evals
├─ notebooks/
│  ├─ 01_explore_gse111727.ipynb
│  └─ 02_train_baseline.ipynb
├─ scripts/
│  ├─ download_geo.py
│  ├─ auto_label_gse111727.py
│  ├─ prepare_data.py
│  ├─ split_dataset.py
│  ├─ train.py
│  └─ evaluate.py
├─ src/
│  └─ hivlat/
│     ├─ __init__.py
│     ├─ config.py
│     ├─ logging.py
│     ├─ data/
│     │  ├─ loaders.py
│     │  ├─ labeling.py
│     │  └─ preprocess.py
│     ├─ models/
│     │  ├─ baseline.py
│     │  ├─ calibrate.py
│     │  └─ metrics.py
│     └─ utils/
│        └─ io.py
└─ tests/
   ├─ test_labeling.py
   └─ test_loaders.py
```

---

## Current run snapshot

*(Replace with your own when you re-run; these match a recent calibrated baseline on GSE111727 with donors held out.)*

- **Validation (multiclass, OvR)** — Logistic + isotonic, HVG=2000:  
  **AUROC_macro = 0.7631**, **AUPRC_macro = 0.5802**, **ECE = 0.0824**

- **Donor (binary)** — with HVG intersection + prior correction:  
  **AUROC = 0.6535**, **AUPRC = 0.8597**, **ECE = 0.0691**

---

## Troubleshooting

**1) `ModuleNotFoundError: No module named 'hivlat'`**  
Run `pip install -e .` **from the project root** (where `pyproject.toml` lives). Avoid running scripts from outside the repo without installing.

**2) “unrecognized arguments” after a long command**  
In bash, a line-continuation `\` must be the **last character** on the line. **Do not** add spaces or comments after it.

**3) Filenames with spaces (e.g., “(1)”)**  
Quote them:
```bash
python scripts/build_paper_artifacts.py   --val "data/processed/eval/predictions_val (1).csv"   --donor "data/processed/eval/predictions_donor (1).csv"   --report "data/processed/eval/report (2).json"
```

**4) Normalizing eval CSVs (schema mismatch)**  
Expected schema:
- Validation: `y_true`, `latent`, `inducible`, `productive`
- Donor: `y_true`, `p_productive`

If your columns differ, minimally rename columns or adapt with a small Python snippet before running the builder.

**5) Parquet engine error** (`pyarrow`/`fastparquet` missing)  
`pip install pyarrow` (this project prefers pyarrow).

**6) Empty splits**  
Loosen QC in `configs/default.yaml` (e.g., `min_genes_per_cell: 50`, `min_cells_per_gene: 1`) just to verify the pipeline; restore stricter QC for real runs.

---

## Scope, safety, and ethics

- This is **computational research**. No wet-lab protocols, no clinical advice, and no identifiable data.
- Use only **public, de-identified datasets** or synthetic data.  
- Include a **Model Card** and **Data Card** with any release or manuscript.

---

## Citation

Akhtar, M. A. K. (2025). *HIV Latency Benchmark: Cross-Study, Uncertainty-Aware Classifier*. Usha Martin University.  
GitHub: https://github.com/shunyaranbooks/hiv-latency-benchmark

---

## Maintainer

**Dr. Mohammad Amir Khusru Akhtar**  
Dean Research, Usha Martin University  
Email: amir@umu.ac.in· https://doi.org/10.5281/zenodo.17413071

---

## License

Released under the **MIT License** (see `LICENSE`).

