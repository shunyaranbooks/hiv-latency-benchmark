import re
from typing import List

def label_from_column(col: str) -> str:
    c = col.lower()
    if 'latency_model_untreated' in c:
        return 'latent'
    if 'latency_model_saha' in c:
        return 'inducible'   # weak label
    if 'latency_model_tcr' in c:
        return 'productive'
    if re.search(r'donor\d+_untreated', c):
        return 'donor_untreated'
    if re.search(r'donor\d+_tcr', c):
        return 'donor_tcr'
    return 'unknown'

def apply_labels(columns: List[str]) -> List[str]:
    return [label_from_column(c) for c in columns]
