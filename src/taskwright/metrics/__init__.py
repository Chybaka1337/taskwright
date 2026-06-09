"""Metrics — thin standardized wrappers over :mod:`sklearn.metrics`.

The framework does NOT choose the success metric for the developer (out of scope);
it standardizes the *call* of the appropriate sklearn metrics per problem type.
Formulas are never reimplemented.
"""
from __future__ import annotations

from .core import (
    classification_metrics,
    clustering_metrics,
    deterministic_metrics,
    regression_metrics,
    supervised_metrics,
)

__all__ = [
    "supervised_metrics",
    "classification_metrics",
    "regression_metrics",
    "clustering_metrics",
    "deterministic_metrics",
]
