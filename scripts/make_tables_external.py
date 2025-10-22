#!/usr/bin/env python
import argparse, json
from pathlib import Path

def fmt(x, nd=3): 
    return f"{x:.{nd}f}"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="data/processed/eval/report_external.json")
    ap.add_argument("--out_md", default="docs/paper/external/tables/table_external.md")
    ap.add_argument("--out_tex", default="docs/paper/external/tables/table_external.tex")
    args = ap.parse_args()

    rep = json.loads(Path(args.report).read_text())["external"]
    md = f"""| Cohort | Metric | Value |
|---|---|---|
| {rep['cohort']} | AUROC (binary) | {fmt(rep['auroc_binary'])} |
| {rep['cohort']} | AUPRC (binary) | {fmt(rep['auprc_binary'])} |
| {rep['cohort']} | ECE (max prob) | {fmt(rep['ece_maxprob'])} |
"""
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(md)

    tex = r"""\begin{table}[ht]\centering
\caption{External cohort performance (calibrated probabilities).}
\begin{tabular}{l l c}
\hline
Cohort & Metric & Value \\
\hline
""" + f"""{rep['cohort']} & AUROC (binary) & {fmt(rep['auroc_binary'])} \\\n""" + \
        f"""{rep['cohort']} & AUPRC (binary) & {fmt(rep['auprc_binary'])} \\\n""" + \
        f"""{rep['cohort']} & ECE (max prob) & {fmt(rep['ece_maxprob'])} \\\n""" + r"""
\hline
\end{tabular}
\end{table}
"""
    Path(args.out_tex).write_text(tex)
    print("Wrote:", args.out_md, "and", args.out_tex)
