"""Level 3 — Runtime: ``run(task, events)`` and the three orchestration branches.

Dispatch is by contract type. Every branch shares the same envelope:

    validate events -> (branch-specific pipeline) -> metrics
    -> persist (joblib + manifest.json) -> log (JSON Lines) -> TaskResult

The branches differ in orchestration exactly as architecture.md §"Уровень 3"
prescribes: deterministic has no model and no split; unsupervised has no ``y``.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
from pydantic import ValidationError
from sklearn.model_selection import train_test_split

from ..exceptions import (
    ContractConsistencyError,
    SanityCheckError,
    TelemetryValidationError,
)
from ..logging_setup import get_run_logger
from ..metrics.core import (
    clustering_metrics,
    deterministic_metrics,
    supervised_metrics,
)
from ..persistence.store import save_run
from ..result import TaskCategory, TaskResult
from ..tasks.deterministic import DeterministicTask, TimeWindow
from ..tasks.supervised import SupervisedTask
from ..tasks.unsupervised import UnsupervisedTask
from ..telemetry.events import BaseEvent

_DEFAULT_OUTPUT_DIR = "taskwright_runs"
_DEFAULT_TEST_SIZE = 0.25
_DEFAULT_RANDOM_STATE = 0


def run(
    task: Any,
    events: Iterable[Any],
    *,
    dataset: Optional[str] = None,
    window: Optional[TimeWindow] = None,
    output_dir: Optional[Any] = None,
) -> TaskResult:
    """Execute ``task`` over ``events`` and return a :class:`TaskResult`.

    Args:
        task: an instance of one of the three task contracts.
        events: stream of telemetry (``BaseEvent`` instances or mappings).
        dataset: dataset name recorded in run metadata.
        window: ``(start, end)`` window for a :class:`DeterministicTask` (ignored
            by the other branches). ``None`` means "all events".
        output_dir: base directory for artifacts and the JSON Lines log
            (default ``taskwright_runs`` in the current working directory).

    Raises:
        TelemetryValidationError: an event does not conform to the contract.
        ContractConsistencyError: a task returns inconsistent artifacts.
        SanityCheckError: a deterministic KPI fails its sanity predicate.
        TypeError: ``task`` is not a recognized contract.
    """
    dataset_name = dataset or "unspecified"
    base_dir = Path(output_dir) if output_dir is not None else Path(_DEFAULT_OUTPUT_DIR)

    # --- common prefix: validate events; abort on the first invalid one ---
    valid_events = _validate_events(events)

    if isinstance(task, SupervisedTask):
        return _run_supervised(task, valid_events, dataset_name, base_dir)
    if isinstance(task, UnsupervisedTask):
        return _run_unsupervised(task, valid_events, dataset_name, base_dir)
    if isinstance(task, DeterministicTask):
        return _run_deterministic(task, valid_events, window, dataset_name, base_dir)
    raise TypeError(
        f"Unsupported task {type(task)!r}: expected a SupervisedTask, "
        f"UnsupervisedTask, or DeterministicTask."
    )


# --------------------------------------------------------------------------- #
# common prefix
# --------------------------------------------------------------------------- #
def _validate_events(events: Iterable[Any]) -> list[BaseEvent]:
    """Validate each event against the telemetry contract.

    Valid <=> conforms to the Pydantic schema. ``BaseEvent`` instances are already
    schema-valid (validated at construction); mappings are validated against
    ``BaseEvent``; anything else is invalid and aborts the run.
    """
    validated: list[BaseEvent] = []
    for i, event in enumerate(events):
        if isinstance(event, BaseEvent):
            validated.append(event)
        elif isinstance(event, Mapping):
            try:
                validated.append(BaseEvent.model_validate(dict(event)))
            except ValidationError as exc:
                raise TelemetryValidationError(
                    f"Event at index {i} does not conform to the telemetry contract: {exc}"
                ) from exc
        else:
            raise TelemetryValidationError(
                f"Event at index {i} is not a telemetry event (got {type(event)!r})."
            )
    return validated


# --------------------------------------------------------------------------- #
# supervised branch
# --------------------------------------------------------------------------- #
def _run_supervised(
    task: SupervisedTask, events: list[BaseEvent], dataset: str, base_dir: Path
) -> TaskResult:
    X = task.build_features(events)
    y = task.build_labels(events)
    _check_xy_consistency(X, y)

    X_train, X_test, y_train, y_test = _split(X, y, time_aware=bool(getattr(task, "time_aware", False)))

    model = task.build_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metric_values = supervised_metrics(model, X_test, y_test, y_pred)

    return _finalize(
        task=task,
        category=TaskCategory.SUPERVISED,
        payload=model,
        metrics=metric_values,
        dataset=dataset,
        base_dir=base_dir,
    )


def _check_xy_consistency(X: Any, y: Any) -> None:
    if not isinstance(X, pd.DataFrame):
        raise ContractConsistencyError(f"build_features must return a DataFrame, got {type(X)!r}.")
    if not isinstance(y, pd.Series):
        raise ContractConsistencyError(f"build_labels must return a Series, got {type(y)!r}.")
    if len(X) != len(y):
        raise ContractConsistencyError(f"X has {len(X)} rows but y has {len(y)}.")
    if not X.index.equals(y.index):
        raise ContractConsistencyError("X and y indices are not aligned.")


def _split(X: pd.DataFrame, y: pd.Series, *, time_aware: bool):
    n = len(X)
    if n < 2:
        raise ContractConsistencyError(f"Need at least 2 samples to split, got {n}.")
    if time_aware:
        # Ordered split: assume rows already chronological; last fraction = test.
        n_test = min(max(1, round(n * _DEFAULT_TEST_SIZE)), n - 1)
        return X.iloc[:-n_test], X.iloc[-n_test:], y.iloc[:-n_test], y.iloc[-n_test:]
    return train_test_split(
        X, y, test_size=_DEFAULT_TEST_SIZE, random_state=_DEFAULT_RANDOM_STATE, shuffle=True
    )


# --------------------------------------------------------------------------- #
# unsupervised branch
# --------------------------------------------------------------------------- #
def _run_unsupervised(
    task: UnsupervisedTask, events: list[BaseEvent], dataset: str, base_dir: Path
) -> TaskResult:
    X = task.build_features(events)
    if not isinstance(X, pd.DataFrame):
        raise ContractConsistencyError(f"build_features must return a DataFrame, got {type(X)!r}.")

    X_norm = _normalize(task, X)
    model = task.build_model()
    labels = _fit_assign(model, X_norm)
    metric_values = clustering_metrics(X_norm, labels)
    profile = task.interpret(labels, X)

    payload = {"model": model, "labels": labels, "profile": profile}
    return _finalize(
        task=task,
        category=TaskCategory.UNSUPERVISED,
        payload=payload,
        metrics=metric_values,
        dataset=dataset,
        base_dir=base_dir,
    )


def _normalize(task: UnsupervisedTask, X: pd.DataFrame):
    """Apply the Runtime's default normalization (overridable via ``task.scaler``)."""
    scaler_factory = getattr(task, "scaler", None)
    if scaler_factory is None:
        return X.to_numpy()
    scaler = scaler_factory() if isinstance(scaler_factory, type) else scaler_factory
    return scaler.fit_transform(X)


def _fit_assign(model: Any, X: Any):
    """Return cluster assignments.

    architecture.md illustrates ``fit_transform``; in sklearn the cluster
    *assignments* are produced by ``fit_predict`` / ``labels_``, which is what we
    use here.
    """
    if hasattr(model, "fit_predict"):
        return model.fit_predict(X)
    model.fit(X)
    if hasattr(model, "labels_"):
        return model.labels_
    raise ContractConsistencyError("Clusterer exposes neither fit_predict nor labels_.")


# --------------------------------------------------------------------------- #
# deterministic branch
# --------------------------------------------------------------------------- #
def _run_deterministic(
    task: DeterministicTask,
    events: list[BaseEvent],
    window: Optional[TimeWindow],
    dataset: str,
    base_dir: Path,
) -> TaskResult:
    deduped = _dedup_by_event_id(events)
    windowed = _filter_by_window(deduped, window)
    kpi = task.aggregate(windowed, window)

    is_valid = bool(task.is_valid(kpi))
    if not is_valid:
        raise SanityCheckError(f"KPI failed its sanity predicate sigma(kappa): {kpi!r}")
    metric_values = deterministic_metrics(is_valid)

    return _finalize(
        task=task,
        category=TaskCategory.DETERMINISTIC,
        payload=kpi,
        metrics=metric_values,
        dataset=dataset,
        base_dir=base_dir,
    )


def _dedup_by_event_id(events: list[BaseEvent]) -> list[BaseEvent]:
    seen: set[str] = set()
    out: list[BaseEvent] = []
    for event in events:
        if event.event_id in seen:
            continue
        seen.add(event.event_id)
        out.append(event)
    return out


def _filter_by_window(
    events: list[BaseEvent], window: Optional[TimeWindow]
) -> list[BaseEvent]:
    if window is None:
        return list(events)
    start, end = window
    return [e for e in events if start <= e.event_timestamp <= end]


# --------------------------------------------------------------------------- #
# common terminal
# --------------------------------------------------------------------------- #
def _finalize(
    *,
    task: Any,
    category: TaskCategory,
    payload: Any,
    metrics: dict[str, Any],
    dataset: str,
    base_dir: Path,
) -> TaskResult:
    task_name = type(task).__name__
    timestamp = datetime.now(timezone.utc)

    artifacts = save_run(
        base_dir=base_dir,
        task_name=task_name,
        category=category,
        dataset=dataset,
        payload=payload,
        metrics=metrics,
        timestamp=timestamp,
    )

    logger = get_run_logger(base_dir)
    logger.info(
        "task_completed",
        extra={
            "context": {
                "task_name": task_name,
                "category": category.value,
                "dataset": dataset,
                "metrics": metrics,
                "artifact_path": str(artifacts.artifact_path),
                "manifest_path": str(artifacts.manifest_path),
            }
        },
    )

    return TaskResult(
        task_name=task_name,
        category=category,
        dataset=dataset,
        timestamp=timestamp,
        metrics=metrics,
        artifact_path=artifacts.artifact_path,
        manifest_path=artifacts.manifest_path,
    )
