# HIV Latency Benchmark: Cross-Study, Uncertainty‑Aware Classifier

**Goal:** Build and share a *reproducible*, laptop‑friendly benchmark and baseline model that distinguishes **latent ↔ inducible‑latent ↔ productive** HIV‑infected CD4 T‑cells from single‑cell RNA‑seq, and links a **patient‑level score** to **intact proviral DNA (IPDA)** counts where available. This repository is the starting point for a series of publishable, open science artifacts.

> **Why this matters:** The latent reservoir is the key barrier to an HIV cure. A generalizable, *calibrated* classifier and a robust surrogate for the reservoir would help **prioritize cure strategies** and **accelerate trials**.

---

## What’s inside (quick tour)

- **Five CLI programs** (see `scripts/`):
  1. `download_geo.py` — fetches exact GEO supplementary files (e.g., GSE111727 raw matrices) and logs checksums.
  2. `prepare_data.py` — QC + normalization + labeling → harmonized matrices.
  3. `split_dataset.py` — canonical train/val split (GSE111727 model) and reserved external test sets.
  4. `train.py` — trains a **multiclass** baseline (Logistic/GBM) and saves the model.
  5. `evaluate.py` — reports **AUROC/AUPRC**, **Expected Calibration Error**, and **conformal coverage** on held‑out donor and external sets.

- **Reproducible layout**: `data/` (never commit raw), `configs/` (YAML), `src/` (library code), `notebooks/` (exploration), `tests/` (sanity).

- **Laptop‑friendly stack**: Python 3.10+, pandas/numpy/scikit‑learn/scipy, optional scanpy/anndata. No GPUs required.

---

## Datasets (public)

- **GSE111727** — *primary* latency model (untreated/SAHA/TCR) + HIV+ donor cells. Use the two raw matrices listed below (download script included).
- **GSE199727** — reservoir CD4+ T‑cells with multi‑omics (external transfer test).
- **GSE180133** — single‑cell latency reversal (external stress test).
- **IPDA cohorts** — intact vs. defective provirus counts (patient‑level anchor).

> The repository **does not** ship data. Use `scripts/download_geo.py` to fetch files into `data/raw/…`. The script records file size and SHA‑256 in `data/manifest.json`.

---

## One‑line reproduction (after cloning)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .

python scripts/download_geo.py
python scripts/prepare_data.py --config configs/default.yaml
python scripts/split_dataset.py   --config configs/default.yaml
python scripts/train.py           --config configs/default.yaml --model-out data/processed/models/logreg.joblib
python scripts/evaluate.py        --config configs/default.yaml --model data/processed/models/logreg.joblib
```

**Outputs** land under `data/processed/` (splits, trained models, and evaluation reports).

---

## Labels and splits (transparent)

- From **GSE111727** column names:
  - `Latency_model_untreated_*` → **latent**
  - `Latency_model_SAHA_*`      → **inducible** (weak supervision)
  - `Latency_model_TCR_*`       → **productive**
  - `Donor*_untreated_*` / `Donor*_TCR_*` → donor sets for external validation

- **Training/validation**: latency model only (no donor cells).  
- **External tests**: donor cells; plus `GSE199727` and `GSE180133` when available.

---

## Safety, scope, and ethics

- This is **computational** research only. No wet‑lab protocols, no clinical advice, no identifiable data.  
- Use **public, de‑identified** datasets or synthetic data.  
- Include a **Model Card** and **Data Card** in your paper/release.

---

## Citation (suggested)

> Akhtar, M. A. K. (2025). *HIV Latency Benchmark: Cross‑Study, Uncertainty‑Aware Classifier*. Usha Martin University.

---

## Maintainer & Contact

**Dr. Mohammad Amir Khusru Akhtar**  
*Dean (Research), Usha Martin University*  
Email: (add your email) · GitHub: https://github.com/shunyaranbooks

