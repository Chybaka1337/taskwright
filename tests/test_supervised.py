"""Supervised branch — end-to-end, X/y consistency, split modes, regression panel."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import taskwright as tw
from taskwright import TaskCategory
from taskwright.exceptions import ContractConsistencyError


def test_supervised_end_to_end(events, sup_task, tmp_path):
    result = tw.run(sup_task, events, dataset="synthetic", output_dir=tmp_path)

    assert isinstance(result, tw.TaskResult)
    assert result.category is TaskCategory.SUPERVISED
    assert result.task_name == "DummySupervised"
    assert result.dataset == "synthetic"
    assert {"precision", "recall", "f1"} <= set(result.metrics)

    # Artifact + manifest persisted.
    assert result.artifact_path.exists()
    assert result.manifest_path.exists()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["category"] == "supervised"
    assert manifest["dataset"] == "synthetic"
    assert manifest["task_name"] == "DummySupervised"

    # JSON Lines log written.
    log_path = Path(tmp_path) / "taskwright.jsonl"
    assert log_path.exists()
    lines = [ln for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines
    record = json.loads(lines[-1])
    assert record["event"] == "task_completed"
    assert record["category"] == "supervised"


def test_supervised_regression_panel(events, tmp_path):
    class Regression(tw.SupervisedTask):
        def build_features(self, events):
            return pd.DataFrame([{"kills": e.kills} for e in events])

        def build_labels(self, events):
            # Non-integer targets so type_of_target routes to the regression panel
            # (sklearn treats integer-valued targets as classification).
            return pd.Series([e.duration_sec + 0.5 for e in events])

        def build_model(self):
            return DecisionTreeRegressor(random_state=0)

    result = tw.run(Regression(), events, dataset="synthetic", output_dir=tmp_path)
    assert "mae" in result.metrics
    assert "precision" not in result.metrics


def test_supervised_xy_length_mismatch_aborts(events, tmp_path):
    class BadLength(tw.SupervisedTask):
        def build_features(self, events):
            return pd.DataFrame([{"k": e.kills} for e in events])

        def build_labels(self, events):
            return pd.Series([1 for _ in events][:-1])  # one short

        def build_model(self):
            return DecisionTreeClassifier()

    with pytest.raises(ContractConsistencyError):
        tw.run(BadLength(), events, dataset="synthetic", output_dir=tmp_path)


def test_supervised_xy_index_mismatch_aborts(events, tmp_path):
    class BadIndex(tw.SupervisedTask):
        def build_features(self, events):
            return pd.DataFrame([{"k": e.kills} for e in events])  # RangeIndex 0..n-1

        def build_labels(self, events):
            n = len(events)
            return pd.Series([1] * n, index=range(100, 100 + n))  # misaligned index

        def build_model(self):
            return DecisionTreeClassifier()

    with pytest.raises(ContractConsistencyError):
        tw.run(BadIndex(), events, dataset="synthetic", output_dir=tmp_path)


def test_supervised_time_aware_split_runs(events, tmp_path):
    class TimeAware(tw.SupervisedTask):
        time_aware = True

        def build_features(self, events):
            return pd.DataFrame([{"kills": e.kills, "deaths": e.deaths} for e in events])

        def build_labels(self, events):
            return pd.Series([1 if e.kills >= e.deaths else 0 for e in events])

        def build_model(self):
            return DecisionTreeClassifier(random_state=0)

    result = tw.run(TimeAware(), events, dataset="synthetic", output_dir=tmp_path)
    assert result.category is TaskCategory.SUPERVISED
    assert result.artifact_path.exists()
