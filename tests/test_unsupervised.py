"""Unsupervised branch — end-to-end clustering, quality metrics, persisted payload."""
from __future__ import annotations

from pathlib import Path

import joblib

import taskwright as tw
from taskwright import TaskCategory


def test_unsupervised_end_to_end(events, unsup_task, tmp_path):
    result = tw.run(unsup_task, events, dataset="synthetic", output_dir=tmp_path)

    assert result.category is TaskCategory.UNSUPERVISED
    assert result.task_name == "DummyUnsupervised"
    assert "silhouette" in result.metrics
    assert "davies_bouldin" in result.metrics
    assert result.metrics["n_clusters"] == 2

    assert result.artifact_path.exists()
    assert result.manifest_path.exists()

    payload = joblib.load(result.artifact_path)
    assert set(payload) >= {"model", "labels", "profile"}
    assert len(payload["labels"]) == len(events)
    assert payload["profile"]["sizes"]  # interpret() output present

    log_path = Path(tmp_path) / "taskwright.jsonl"
    assert log_path.exists()


def test_unsupervised_scaler_override_none(events, tmp_path):
    class NoScale(tw.UnsupervisedTask):
        scaler = None  # disable Runtime normalization

        def build_features(self, events):
            import pandas as pd

            return pd.DataFrame([{"kills": e.kills, "deaths": e.deaths} for e in events])

        def build_model(self):
            from sklearn.cluster import KMeans

            return KMeans(n_clusters=2, n_init=10, random_state=0)

        def interpret(self, clusters, X):
            return {"n": len(set(clusters.tolist()))}

    result = tw.run(NoScale(), events, dataset="synthetic", output_dir=tmp_path)
    assert result.category is TaskCategory.UNSUPERVISED
