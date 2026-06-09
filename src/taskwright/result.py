"""TaskResult — the MVP result object returned by :func:`taskwright.run`."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class TaskCategory(str, Enum):
    """The three analytics task categories the framework contracts cover."""

    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    DETERMINISTIC = "deterministic"


@dataclass(frozen=True)
class TaskResult:
    """Outcome of a single Runtime execution (architecture.md, "TaskResult").

    MVP minimum: a reference to the saved artifact, the computed metrics, and run
    metadata. Richer per-run material (ROC data, cluster assignments, retention
    series, ...) is intentionally *not* part of this object — per architecture.md
    it lives in the apробation repository, not in the library API.
    """

    task_name: str
    category: TaskCategory
    dataset: str
    timestamp: datetime
    metrics: dict[str, Any]
    artifact_path: Path
    manifest_path: Path
