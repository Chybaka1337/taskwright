"""Shared helper: coerce arbitrary values into JSON-serializable Python types.

Used by both persistence (``manifest.json``) and logging (JSON Lines) so that
numpy scalars produced by sklearn metrics serialize cleanly.
"""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any


def json_safe(obj: Any) -> Any:
    """Recursively convert ``obj`` into JSON-serializable Python primitives."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return json_safe(obj.item())
    except Exception:
        pass
    try:
        return float(obj)
    except Exception:
        return str(obj)
