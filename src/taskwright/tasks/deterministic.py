"""DeterministicTask contract (architecture.md, Level 2)."""
from __future__ import annotations

import math
import numbers
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Iterable, Optional, Tuple, TypeAlias

import pandas as pd

from ..telemetry.events import BaseEvent

KPI: TypeAlias = Any
"""Developer-shaped KPI returned by :meth:`DeterministicTask.aggregate`.

A number, a mapping, a pandas object — whatever the metric is. The framework does
not prescribe its structure.
"""

TimeWindow: TypeAlias = Tuple[datetime, datetime]
"""Inclusive ``(start, end)`` window the Runtime filters events to."""


class DeterministicTask(ABC):
    """Contract for deterministic KPI computations (e.g. T12 retention / KPI).

    No model and no train/test split: the Runtime deduplicates by ``event_id``,
    filters to the time window, calls :meth:`aggregate`, then checks the result
    with :meth:`is_valid`.
    """

    @abstractmethod
    def aggregate(
        self, events: Iterable[BaseEvent], window: Optional[TimeWindow]
    ) -> KPI:
        """Compute the KPI over ``events`` within ``window``."""

    def is_valid(self, kpi: KPI) -> bool:
        """Sanity predicate sigma(kappa): is the KPI in its admissible region?

        Default: every numeric component is finite and non-null. Override to
        narrow the admissible region (e.g. a retention rate in ``[0, 1]``).
        """
        return _all_finite(kpi)


def _all_finite(kpi: Any) -> bool:
    """Best-effort finiteness / non-null check across common KPI shapes."""
    if kpi is None:
        return False
    if isinstance(kpi, numbers.Real):
        return math.isfinite(float(kpi))
    if isinstance(kpi, pd.Series):
        return bool(kpi.notna().all()) and all(_all_finite(v) for v in kpi.tolist())
    if isinstance(kpi, pd.DataFrame):
        return bool(kpi.notna().all().all())
    if isinstance(kpi, dict):
        return all(_all_finite(v) for v in kpi.values())
    if isinstance(kpi, (list, tuple, set)):
        return all(_all_finite(v) for v in kpi)
    return True
