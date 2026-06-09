"""UnsupervisedTask contract (architecture.md, Level 2)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, TypeAlias

import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..telemetry.events import BaseEvent

ClusterProfile: TypeAlias = Any
"""Developer-shaped interpretation of clusters returned by :meth:`UnsupervisedTask.interpret`.

The framework does not prescribe its structure (e.g. per-cluster summary stats);
it is whatever the task needs to describe its segments.
"""


class UnsupervisedTask(ABC):
    """Contract for clustering / segmentation tasks (e.g. T1 player segmentation).

    Attributes:
        scaler: feature-normalization strategy applied by the Runtime before
            clustering. Default :class:`~sklearn.preprocessing.StandardScaler`.
            Set to ``None`` to disable, or assign any sklearn transformer class /
            instance to override. Normalization is the Runtime's responsibility,
            not a contract method — this attribute only swaps the default.
    """

    scaler: Any = StandardScaler

    @abstractmethod
    def build_features(self, events: Iterable[BaseEvent]) -> pd.DataFrame:
        """Return the feature matrix X to cluster."""

    @abstractmethod
    def build_model(self) -> Any:
        """Return an unfitted clusterer (sklearn-style ``fit_predict``)."""

    @abstractmethod
    def interpret(self, clusters: Any, X: pd.DataFrame) -> ClusterProfile:
        """Describe the discovered clusters given assignments and (original) X."""
