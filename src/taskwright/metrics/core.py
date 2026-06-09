"""Standardized metric computation — wrappers over :mod:`sklearn.metrics`.

Problem type for the supervised branch is detected with sklearn's own
``type_of_target`` so routing stays consistent with the metric backend; this
picks the standard descriptive *panel* (classification vs regression) without
choosing which single metric defines task success (that is the developer's call).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.metrics import (
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.utils.multiclass import type_of_target

_CLASSIFICATION_TARGETS = ("binary", "multiclass", "multilabel-indicator")
_REGRESSION_TARGETS = ("continuous", "continuous-multioutput")


def supervised_metrics(model: Any, X_test: Any, y_true: Any, y_pred: Any) -> dict[str, Any]:
    """Route to the classification or regression panel based on ``y_true``'s type.

    Note: ``type_of_target`` treats *integer-valued* targets (including integer
    floats like ``5.0``) as classification. A regression task whose targets are
    counts should keep them continuous, or it will be routed to the
    classification panel.
    """
    target_type = type_of_target(y_true)
    if target_type in _CLASSIFICATION_TARGETS:
        y_score = _safe_scores(model, X_test, target_type)
        return classification_metrics(y_true, y_pred, y_score=y_score, target_type=target_type)
    if target_type in _REGRESSION_TARGETS:
        return regression_metrics(y_true, y_pred)
    # Unknown target type: report it rather than guessing a panel.
    return {"target_type": target_type}


def classification_metrics(
    y_true: Any,
    y_pred: Any,
    *,
    y_score: Any = None,
    target_type: Optional[str] = None,
) -> dict[str, Any]:
    """precision / recall / F1 (+ ROC AUC when class scores are available)."""
    if target_type is None:
        target_type = type_of_target(y_true)
    average = "binary" if target_type == "binary" else "macro"
    out: dict[str, Any] = {
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }
    if y_score is not None:
        out["roc_auc"] = _safe_roc_auc(y_true, y_score, target_type)
    return out


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    """Mean absolute error (sklearn)."""
    return {"mae": float(mean_absolute_error(y_true, y_pred))}


def clustering_metrics(X: Any, labels: Any) -> dict[str, Any]:
    """silhouette + Davies-Bouldin; ``None`` when undefined (e.g. < 2 clusters)."""
    labels = np.asarray(labels)
    n_samples = labels.shape[0]
    n_labels = len(set(labels.tolist()))
    out: dict[str, Any] = {"n_clusters": int(n_labels)}
    if 2 <= n_labels <= n_samples - 1:
        try:
            out["silhouette"] = float(silhouette_score(X, labels))
            out["davies_bouldin"] = float(davies_bouldin_score(X, labels))
        except Exception:
            out["silhouette"] = None
            out["davies_bouldin"] = None
    else:
        out["silhouette"] = None
        out["davies_bouldin"] = None
    return out


def deterministic_metrics(is_valid: bool) -> dict[str, Any]:
    """Correctness predicate sigma(kappa) in {0, 1}."""
    return {"sigma_kappa": int(bool(is_valid))}


def _safe_scores(model: Any, X: Any, target_type: str) -> Any:
    """Best-effort class scores for AUC; ``None`` if the model exposes none."""
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            if target_type == "binary" and getattr(proba, "ndim", 1) == 2 and proba.shape[1] == 2:
                return proba[:, 1]
            return proba
        except Exception:
            return None
    if hasattr(model, "decision_function"):
        try:
            return model.decision_function(X)
        except Exception:
            return None
    return None


def _safe_roc_auc(y_true: Any, y_score: Any, target_type: Optional[str]) -> Optional[float]:
    """ROC AUC when computable (binary needs both classes present in y_true)."""
    try:
        if target_type == "binary":
            return float(roc_auc_score(y_true, y_score))
        return float(roc_auc_score(y_true, y_score, multi_class="ovr"))
    except Exception:
        return None
