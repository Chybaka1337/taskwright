"""taskwright — typed contracts + Runtime for game-telemetry analytics tasks.

The framework is *contracts + Runtime* in equal measure. Contracts declare WHAT
(data types, method signatures); the Runtime owns HOW (validation, orchestration,
metrics, persistence, logging). Public API:

* Telemetry contract (Level 1): :class:`BaseEvent`, :class:`MatchFinished`
* Task contracts (Level 2): :class:`SupervisedTask`, :class:`UnsupervisedTask`,
  :class:`DeterministicTask`
* Runtime (Level 3): :func:`run`
* Result: :class:`TaskResult`, :class:`TaskCategory`
"""
from __future__ import annotations

from .exceptions import (
    ContractConsistencyError,
    SanityCheckError,
    TaskwrightError,
    TelemetryValidationError,
)
from .result import TaskCategory, TaskResult
from .runtime.runner import run
from .tasks.deterministic import KPI, DeterministicTask
from .tasks.supervised import SupervisedTask
from .tasks.unsupervised import ClusterProfile, UnsupervisedTask
from .telemetry.events import BaseEvent, MatchFinished

__version__ = "0.1.0"

__all__ = [
    "BaseEvent",
    "MatchFinished",
    "SupervisedTask",
    "UnsupervisedTask",
    "DeterministicTask",
    "ClusterProfile",
    "KPI",
    "run",
    "TaskResult",
    "TaskCategory",
    "TaskwrightError",
    "TelemetryValidationError",
    "ContractConsistencyError",
    "SanityCheckError",
]
