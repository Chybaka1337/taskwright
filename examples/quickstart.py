"""Minimal synthetic quickstart for taskwright (illustrative, not apробation).

Run from the repo root::

    poetry run python examples/quickstart.py

It builds a tiny in-memory event stream, declares a trivial ``SupervisedTask``,
and lets the Runtime orchestrate validation -> split -> fit -> predict -> metrics
-> persistence -> logging, returning a ``TaskResult``.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

import taskwright as tw


class WinFromKills(tw.SupervisedTask):
    """Predict a synthetic 'win' label (kills >= deaths) from match_finished events."""

    def build_features(self, events) -> pd.DataFrame:
        return pd.DataFrame([{"kills": e.kills, "deaths": e.deaths} for e in events])

    def build_labels(self, events) -> pd.Series:
        return pd.Series([1 if e.kills >= e.deaths else 0 for e in events], name="win")

    def build_model(self):
        return DecisionTreeClassifier(random_state=0)


def synthetic_events():
    t0 = datetime(2025, 1, 1)
    return [
        tw.MatchFinished(
            user_id=f"u{i % 3}",
            event_timestamp=t0 + timedelta(hours=i),
            event_id=f"m{i}",
            match_id=f"match{i}",
            duration_sec=1800,
            hero="axe",
            kills=i % 5,
            deaths=(i + 2) % 5,
        )
        for i in range(12)
    ]


def main() -> None:
    result = tw.run(WinFromKills(), synthetic_events(), dataset="synthetic-quickstart")
    print("category:", result.category.value)
    print("metrics :", result.metrics)
    print("artifact:", result.artifact_path)
    print("manifest:", result.manifest_path)


if __name__ == "__main__":
    main()
