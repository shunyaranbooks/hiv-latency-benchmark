#!/usr/bin/env bash
set +e
set -u
say(){ printf "\n\033[1;34m[INFO]\033[0m %s\n" "$*"; }
warn(){ printf "\n\033[1;33m[WARN]\033[0m %s\n" "$*"; }
ok(){ printf "\n\033[1;32m[DONE]\033[0m %s\n" "$*"; }

mkdir -p docs/paper/external/{figures,tables} data/processed/eval

MODEL="data/processed/models/logreg_inter.joblib"
if [ ! -f "$MODEL" ]; then
  warn "Model missing → running minimal pipeline"
  python scripts/prepare_data.py --config configs/default.yaml || true
  python scripts/split_dataset.py --config configs/default.yaml || true
  python scripts/train.py --config configs/default.yaml --model-out "$MODEL" || true
  python scripts/evaluate.py --config configs/default.yaml --model "$MODEL" || true
fi

# Class balance check (string-safe)
python - <<'PY'
import sys, pandas as pd
labs = pd.read_csv("data/interim/GSE180133_labels.csv")
lab = labs['label'].astype(str).str.lower().str.strip()
c = lab.value_counts()
prod = int(c.get('productive',0))
neg  = int(c.get('latent',0)) + int(c.get('inducible',0))
print("External label counts:", dict(c))
sys.exit(0 if (prod>0 and neg>0) else 3)
PY
BAL=$?
if [ "$BAL" != "0" ]; then
  warn "External evaluation needs both classes (≥1 'productive' and ≥1 latent/inducible). Edit data/raw/GSE180133/GSE180133_labels.csv and re-run."
  exit 0
fi

# Evaluate + assets
python scripts/evaluate_external.py --config configs/default.yaml --model "$MODEL" --cohort GSE180133 || true
python scripts/plot_reliability_external.py --csv data/processed/eval/predictions_external.csv --out docs/paper/external/figures/fig_reliability_external_cal.png || true
python scripts/bootstrap_ci.py --csv data/processed/eval/predictions_external.csv --out docs/paper/external/tables/bootstrap_external.json || true
python scripts/make_tables_external.py --report data/processed/eval/report_external.json --out_md docs/paper/external/tables/table_external.md --out_tex docs/paper/external/tables/table_external.tex || true

ok "External assets written to docs/paper/external/{figures,tables}"
exit 0
