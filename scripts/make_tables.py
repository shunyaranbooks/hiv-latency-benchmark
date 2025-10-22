#!/usr/bin/env python
import argparse, json
from pathlib import Path

def fmt(x, nd=3): 
    return f"{x:.{nd}f}"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="data/processed/eval/report.json")
    ap.add_argument("--fallback", default="")
    ap.add_argument("--out_md", default="docs/paper/tables/table_main.md")
    ap.add_argument("--out_tex", default="docs/paper/tables/table_main.tex")
    args = ap.parse_args()

    # Fallback (from env via small JSON string)
    fb = {}
    if args.fallback:
        fb = json.loads(args.fallback)

    if Path(args.report).exists():
        rep = json.loads(Path(args.report).read_text())
        v = rep["val"]; d = rep["donor"]
        val_auroc, val_auprc, val_ece = v["auroc_macro_ovr"], v["auprc_macro_ovr"], v["ece_maxprob"]
        don_auroc = d.get("auroc_binary", fb.get("don_auroc")); 
        don_auprc = d.get("auprc_binary", fb.get("don_auprc")); 
        don_ece   = d.get("ece_maxprob", fb.get("don_ece"))
    else:
        val_auroc, val_auprc, val_ece = fb["val_auroc"], fb["val_auprc"], fb["val_ece"]
        don_auroc, don_auprc, don_ece = fb["don_auroc"], fb["don_auprc"], fb["don_ece"]

    md = f"""| Split | Metric | Value |
|---|---|---|
| Validation (multiclass) | AUROC (macro OVR) | {fmt(val_auroc)} |
| Validation (multiclass) | AUPRC (macro OVR) | {fmt(val_auprc)} |
| Validation (multiclass) | ECE (max prob) | {fmt(val_ece)} |
| Donor (binary) | AUROC | {fmt(don_auroc)} |
| Donor (binary) | AUPRC | {fmt(don_auprc)} |
| Donor (binary) | ECE (max prob) | {fmt(don_ece)} |
"""
    Path(args.out_md).write_text(md)

    tex = r"""\begin{table}[ht]\centering
\caption{Performance summary (calibrated probabilities).}
\begin{tabular}{l l c}
\hline
Split & Metric & Value \\
\hline
Validation (multiclass) & AUROC (macro OVR) & """ + fmt(val_auroc) + r""" \\
Validation (multiclass) & AUPRC (macro OVR) & """ + fmt(val_auprc) + r""" \\
Validation (multiclass) & ECE (max prob) & """ + fmt(val_ece) + r""" \\
Donor (binary) & AUROC & """ + fmt(don_auroc) + r""" \\
Donor (binary) & AUPRC & """ + fmt(don_auprc) + r""" \\
Donor (binary) & ECE (max prob) & """ + fmt(don_ece) + r""" \\
\hline
\end{tabular}
\end{table}
"""
    Path(args.out_tex).write_text(tex)
    print("Wrote:", args.out_md, "and", args.out_tex)
