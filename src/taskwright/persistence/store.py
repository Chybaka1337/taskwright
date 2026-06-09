"""Persist a run as a joblib artifact alongside a JSON manifest."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

from .._jsonutil import json_safe
from ..result import TaskCategory


@dataclass(frozen=True)
class RunArtifacts:
    """Filesystem references produced by :func:`save_run`."""

    run_dir: Path
    artifact_path: Path
    manifest_path: Path
    timestamp: datetime


def save_run(
    *,
    base_dir: Path,
    task_name: str,
    category: TaskCategory,
    dataset: str,
    payload: Any,
    metrics: dict[str, Any],
    timestamp: datetime,
) -> RunArtifacts:
    """Dump ``payload`` (joblib) and a ``manifest.json`` under a unique run dir."""
    base_dir = Path(base_dir)
    stamp = timestamp.strftime("%Y%m%dT%H%M%S_%f")
    run_dir = base_dir / f"{stamp}__{category.value}__{task_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = run_dir / "artifact.joblib"
    joblib.dump(payload, artifact_path)

    manifest_path = run_dir / "manifest.json"
    manifest = {
        "timestamp": timestamp.isoformat(),
        "task_name": task_name,
        "category": category.value,
        "dataset": dataset,
        "metrics": json_safe(metrics),
        "artifact": artifact_path.name,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return RunArtifacts(
        run_dir=run_dir,
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        timestamp=timestamp,
    )
