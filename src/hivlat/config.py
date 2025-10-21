from __future__ import annotations
from pathlib import Path
import yaml

class Config:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        with open(self.path, 'r') as f:
            self.cfg = yaml.safe_load(f)
    def get(self, *keys, default=None):
        cur = self.cfg
        for k in keys:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return default
        return cur
