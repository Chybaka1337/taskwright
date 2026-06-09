"""Shared synthetic fixtures and dummy contract implementations for the tests.

No real datasets — a tiny in-memory event stream plus a trivial implementation of
each of the three task contracts, enough to exercise the Runtime end-to-end.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier

import taskwright as tw


@pytest.fixture
def events():
    """A tiny synthetic stream of ``match_finished`` events (two latent classes)."""
    t0 = datetime(2025, 1, 1)
    return [
        tw.MatchFinished(
            user_id=f"u{i % 4}",
            event_timestamp=t0 + timedelta(hours=i),
            event_id=f"e{i}",
            match_id=f"m{i}",
            duration_sec=1500 + 10 * i,
            hero="axe" if i % 2 == 0 else "lina",
            kills=i % 6,
            deaths=(i + 3) % 6,
        )
        for i in range(12)
    ]


class DummySupervised(tw.SupervisedTask):
    """Trivial classification task: 'win' iff kills >= deaths."""

    def build_features(self, events) -> pd.DataFrame:
        return pd.DataFrame(
            [{"kills": e.kills, "deaths": e.deaths, "dur": e.duration_sec} for e in events]
        )

    def build_labels(self, events) -> pd.Series:
        return pd.Series([1 if e.kills >= e.deaths else 0 for e in events], name="win")

    def build_model(self):
        return DecisionTreeClassifier(random_state=0)


class DummyUnsupervised(tw.UnsupervisedTask):
    """Trivial 2-cluster segmentation over (kills, deaths)."""

    def build_features(self, events) -> pd.DataFrame:
        return pd.DataFrame([{"kills": e.kills, "deaths": e.deaths} for e in events])

    def build_model(self):
        return KMeans(n_clusters=2, n_init=10, random_state=0)

    def interpret(self, clusters, X) -> dict:
        return {"sizes": pd.Series(clusters).value_counts().to_dict()}


class DummyDeterministic(tw.DeterministicTask):
    """Trivial KPI: fraction of users with at least one kill (in [0, 1])."""

    def aggregate(self, events, window) -> float:
        users = {e.user_id for e in events}
        active = {e.user_id for e in events if e.kills > 0}
        return len(active) / len(users) if users else 0.0

    def is_valid(self, kpi) -> bool:
        return isinstance(kpi, float) and 0.0 <= kpi <= 1.0


@pytest.fixture
def sup_task():
    return DummySupervised()


@pytest.fixture
def unsup_task():
    return DummyUnsupervised()


@pytest.fixture
def det_task():
    return DummyDeterministic()
