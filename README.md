# HIV Latency Benchmark: Cross-Study, Uncertainty-Aware Classifier

A reproducible, **laptop-friendly** benchmark and baseline model that distinguishes **latent ↔ inducible-latent ↔ productive** HIV-infected CD4 T cells from single-cell RNA-seq, reporting **uncertainty (calibration)** alongside accuracy. This is the first artifact in a planned open series.

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
- [Repo layout](#repo-layout)
- [Results (example placeholder)](#results-example-placeholder)
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

```bash
# 0) Create environment
python -m venv .venv
source .venv/bin/activate

# 1) Install package + CLI deps
pip install -e .

# 2) Download public files (starts with GSE111727)
python scripts/download_geo.py

# 3) Create label templates & auto-fill from provenance rules
python scripts/auto_label_gse111727.py

# 4) Prepare, split, train, evaluate
python scripts/prepare_data.py --config configs/default.yaml
python scripts/split_dataset.py   --config configs/default.yaml

# Tip: enable HVG intersection to reduce domain shift (see “Configuration”)
python scripts/train.py           --config configs/default.yaml --model-out data/processed/models/logreg.joblib
python scripts/evaluate.py        --config configs/default.yaml --model data/processed/models/logreg.joblib
```

Outputs land under `data/processed/`:
- `X_train.parquet`, `X_val.parquet`, `y_train.csv`, `y_val.csv`
- `models/*.joblib`
- `eval/report.json`, `eval/predictions_val.csv`, `eval/predictions_donor.csv`

---

## Label provenance (transparent & reproducible)

This repo **does not guess** labels from vague prefixes; it encodes published provenance rules in code (`scripts/auto_label_gse111727.py`) and writes them into two templates:

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
  # Recommended: reduce domain shift by using only genes present in train ∩ donor
  hvg_intersection_with_donor: true

model:
  random_state: 42
```

You can tune logistic regularization via an env var (kept out of the YAML to make sweeps easy):
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
- **Donor prior shift**: during evaluation we apply **Saerens prior correction** using the source (val) and target (donor) class priors to improve calibration under domain shift.

---

## What the evaluation reports

`python scripts/evaluate.py ...` prints and writes:

- **Validation (multiclass)**:  
  - `auroc_macro_ovr`  
  - `auprc_macro_ovr`  
  - `ece_maxprob` (Expected Calibration Error, lower is better)

- **Donor / external (binary: latent vs productive)**:  
  - `auroc_binary`, `auprc_binary`  
  - `ece_maxprob` (on a 2-class probability matrix after prior correction)

Per-cell probabilities:  
- `data/processed/eval/predictions_val.csv`  
- `data/processed/eval/predictions_donor.csv`

> **Planned add-ons:** reliability diagrams and split-conformal coverage plots (`scripts/plot_reliability.py`, `scripts/plot_conformal.py`). PRs welcome.

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

## Results (example placeholder)

*(Replace with your own numbers; these are indicative from early runs with GSE111727 only.)*

- **Validation (multiclass)** — Logistic + isotonic, HVG=2000:  
  AUROC ≈ **0.74**, AUPRC ≈ **0.58**, **ECE ≈ 0.06**

- **Donor (binary)** — with HVG intersection + prior correction:  
  AUROC ≈ **0.68**, AUPRC ≈ **0.88**, **ECE** decreases vs. uncorrected

- **GBM comparator** — often best in-study AUROC but weaker cross-study AUROC; we include it as a robustness comparator.

Next planned: add **GSE180133 / GSE199727** as external tests and publish uncertainty plots.

---

## Troubleshooting

- **Parquet engine error** (`pyarrow`/`fastparquet` missing):  
  `pip install pyarrow` (the project’s `pyproject.toml` should already include it).
- **“X does not have valid feature names” warning**: harmless; the code keeps DataFrame feature names during scaling to silence it.
- **Empty splits**: loosen QC in `configs/default.yaml` (e.g., `min_genes_per_cell: 50`, `min_cells_per_gene: 1`) just to verify the pipeline, then restore stricter QC for real runs.

---

## Scope, safety, and ethics

- This is **computational research**. No wet-lab protocols, no clinical advice, and no identifiable data.
- Use only public, de-identified datasets or synthetic data.  
- Please include a **Model Card** and **Data Card** with any release or manuscript.

---

## Citation

> Akhtar, M. A. K. (2025). *HIV Latency Benchmark: Cross-Study, Uncertainty-Aware Classifier*. Usha Martin University.  
> GitHub: https://github.com/shunyaranbooks

---

## Maintainer

**Dr. Mohammad Amir Khusru Akhtar**  
Dean (Research), Usha Martin University  
Email: *(add your email)* · GitHub: https://github.com/shunyaranbooks

---

## License

This project is released under the **MIT License** (see `LICENSE`).
