"""Exception hierarchy raised by the taskwright Runtime."""
from __future__ import annotations


class TaskwrightError(Exception):
    """Base class for all taskwright Runtime errors."""


class TelemetryValidationError(TaskwrightError):
    """An event does not conform to the telemetry contract.

    The Runtime aborts processing on the first invalid event.
    """


class ContractConsistencyError(TaskwrightError):
    """A task contract returned inconsistent artifacts.

    Example: a :class:`~taskwright.SupervisedTask` whose ``build_features`` (X)
    and ``build_labels`` (y) disagree on row count or index.
    """


class SanityCheckError(TaskwrightError):
    """A :class:`~taskwright.DeterministicTask` KPI failed its sanity predicate."""
