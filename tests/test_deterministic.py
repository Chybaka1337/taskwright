"""Deterministic branch — KPI, dedup, time window, sanity predicate."""
from __future__ import annotations

from datetime import datetime

import joblib
import pytest

import taskwright as tw
from taskwright import TaskCategory
from taskwright.exceptions import SanityCheckError


def test_deterministic_end_to_end(events, det_task, tmp_path):
    result = tw.run(det_task, events, dataset="synthetic", output_dir=tmp_path)

    assert result.category is TaskCategory.DETERMINISTIC
    assert result.task_name == "DummyDeterministic"
    assert result.metrics["sigma_kappa"] == 1
    assert result.artifact_path.exists()

    kpi = joblib.load(result.artifact_path)
    assert 0.0 <= kpi <= 1.0


def test_deterministic_dedup_by_event_id(events, det_task, tmp_path):
    # Duplicate every event (same event_id) -> dedup must collapse, KPI stays valid.
    result = tw.run(det_task, list(events) + list(events), dataset="synthetic", output_dir=tmp_path)
    assert result.metrics["sigma_kappa"] == 1


def test_deterministic_window_filter(events, det_task, tmp_path):
    window = (datetime(2025, 1, 1, 0), datetime(2025, 1, 1, 3))  # first 4 events
    result = tw.run(det_task, events, dataset="synthetic", window=window, output_dir=tmp_path)
    assert result.metrics["sigma_kappa"] == 1


def test_deterministic_sanity_failure_out_of_range(events, tmp_path):
    class OutOfRange(tw.DeterministicTask):
        def aggregate(self, events, window):
            return 1.5  # retention must be <= 1

        def is_valid(self, kpi):
            return 0.0 <= kpi <= 1.0

    with pytest.raises(SanityCheckError):
        tw.run(OutOfRange(), events, dataset="synthetic", output_dir=tmp_path)


def test_deterministic_default_sanity_rejects_nan(events, tmp_path):
    class NanKPI(tw.DeterministicTask):
        def aggregate(self, events, window):
            return float("nan")  # default is_valid checks finiteness

    with pytest.raises(SanityCheckError):
        tw.run(NanKPI(), events, dataset="synthetic", output_dir=tmp_path)
