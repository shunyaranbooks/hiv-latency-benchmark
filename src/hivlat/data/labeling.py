import re
from typing import List

def label_from_column(col: str) -> str:
    c = col.lower()

    # donor mapping first (external validation)
    if "donor" in c:
        if "tcr" in c:
            return "donor_tcr"
        if any(k in c for k in ["untreated", "unstim", "ctrl", "control"]):
            return "donor_untreated"

    # explicit latency_model prefixes (original file style)
    if "latency_model_untreated" in c:
        return "latent"
    if "latency_model_saha" in c:
        return "inducible"
    if "latency_model_tcr" in c:
        return "productive"

    # generic fallbacks (in case columns are simpler like 'Untreated_cell_1')
    if any(k in c for k in ["tcr", "cd3", "activation"]):
        return "productive"
    if any(k in c for k in ["saha", "vorinostat", "lra"]):
        return "inducible"
    if any(k in c for k in ["untreated", "unstim", "ctrl", "control"]):
        return "latent"

    return "unknown"

def apply_labels(columns: List[str]) -> List[str]:
    return [label_from_column(c) for c in columns]
