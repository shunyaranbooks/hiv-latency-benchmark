from __future__ import annotations
from pathlib import Path
import json

def ensure_dir(p: str | Path):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def update_manifest(path: str | Path, entry_key: str, url: str, sha256: str, nbytes: int):
    path = Path(path)
    data = {}
    if path.exists():
        data = json.loads(path.read_text())
    data[entry_key] = {'url': url, 'sha256': sha256, 'bytes': nbytes}
    path.write_text(json.dumps(data, indent=2))
