"""SupervisedTask contract (architecture.md, Level 2)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

import pandas as pd

from ..telemetry.events import BaseEvent


class SupervisedTask(ABC):
    """Contract for supervised learning tasks (e.g. T2 churn, T4 win prediction).

    The developer declares *what* the task is (features, labels, estimator); the
    Runtime owns *how* it runs (validation, split, fit, predict, metrics, save).

    Attributes:
        time_aware: split selector read by the Runtime. ``False`` (default) ->
            random train/test split; ``True`` -> ordered temporal split (the last
            fraction of rows becomes the test set). When ``True`` the rows of
            ``build_features`` / ``build_labels`` must already be in chronological
            order — temporal ordering is the developer's responsibility.
    """

    time_aware: bool = False

    @abstractmethod
    def build_features(self, events: Iterable[BaseEvent]) -> pd.DataFrame:
        """Return the feature matrix X (one row per training example)."""

    @abstractmethod
    def build_labels(self, events: Iterable[BaseEvent]) -> pd.Series:
        """Return the target vector y, index-aligned with X."""

    @abstractmethod
    def build_model(self) -> Any:
        """Return an unfitted estimator (sklearn-style ``fit`` / ``predict``)."""
