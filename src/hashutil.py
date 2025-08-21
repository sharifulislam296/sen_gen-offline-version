from __future__ import annotations
import hashlib, json
from typing import Any

def stable_hash(obj: Any) -> str:
    """
    Deterministic short hash for caching keys.
    """
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
