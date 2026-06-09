# taskwright

Typed **contracts + Runtime** for categories of game-telemetry analytics tasks.

`taskwright` is a game-agnostic Python library (a dependency, like Hibernate /
Spring Data / scikit-learn): the developer brings their own telemetry and a small
declarative task implementation, and the framework supplies typed contracts plus a
Runtime that orchestrates validation, metrics, persistence and logging.

This repository is the **library** only. Reference task implementations, datasets
and the Streamlit demo live in a separate apробation repository (`taskwright-demo`).
The canonical specification is [`architecture.md`](architecture.md).

## Install (development)

The project targets **Python 3.11+** and uses **Poetry**.

```bash
poetry install
poetry run pytest
```

## The three contracts

| Level | What it declares |
|---|---|
| **Telemetry** (`BaseEvent`) | Pydantic v2 base with fixed fields `user_id`, `event_timestamp`, `event_id`, `event_name`; subclass it for your events. |
| **Task** (3 ABCs) | `SupervisedTask` (`build_features` / `build_labels` / `build_model`), `UnsupervisedTask` (`build_features` / `build_model` / `interpret`), `DeterministicTask` (`aggregate`). |
| **Runtime** (`run`) | `run(task, events)` dispatches by contract type into one of three orchestration branches and returns a `TaskResult`. |

A contract declares **what** the task is; the Runtime owns **how** it runs.

## Quickstart

```python
from datetime import datetime, timedelta
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import taskwright as tw

class WinFromKills(tw.SupervisedTask):
    def build_features(self, events):
        return pd.DataFrame([{"kills": e.kills, "deaths": e.deaths} for e in events])
    def build_labels(self, events):
        return pd.Series([1 if e.kills >= e.deaths else 0 for e in events])
    def build_model(self):
        return DecisionTreeClassifier(random_state=0)

t0 = datetime(2025, 1, 1)
events = [
    tw.MatchFinished(
        user_id=f"u{i%3}", event_timestamp=t0 + timedelta(hours=i),
        event_id=f"m{i}", match_id=f"match{i}", duration_sec=1800,
        hero="axe", kills=i % 5, deaths=(i + 2) % 5,
    )
    for i in range(12)
]

result = tw.run(WinFromKills(), events, dataset="synthetic-quickstart")
print(result.category.value, result.metrics, result.artifact_path)
```

See [`examples/quickstart.py`](examples/quickstart.py) for the runnable version.

## What the Runtime does (and does not)

**Does:** declares contracts (Pydantic + ABC); validates telemetry; orchestrates
the pipeline per category; standardizes metrics (wrappers over `sklearn.metrics`)
and persistence (`joblib` + `manifest.json`); logs in JSON Lines.

**Does not:** choose the success metric or the algorithm (the developer's call);
serve HTTP; build dashboards; talk to a database; ship trained models or datasets.
