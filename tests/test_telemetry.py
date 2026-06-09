"""Level 1 — telemetry contract validation."""
from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

import taskwright as tw


def test_baseevent_valid():
    e = tw.BaseEvent(
        user_id="u1", event_timestamp=datetime(2025, 1, 1), event_id="e1", event_name="x"
    )
    assert e.user_id == "u1"
    assert e.event_name == "x"


def test_matchfinished_event_name_is_fixed():
    e = tw.MatchFinished(
        user_id="u1",
        event_timestamp=datetime(2025, 1, 1),
        event_id="e1",
        match_id="m1",
        duration_sec=100,
        hero="axe",
        kills=3,
        deaths=1,
    )
    assert e.event_name == "match_finished"


def test_missing_required_field_is_invalid():
    with pytest.raises(ValidationError):
        tw.BaseEvent(user_id="u1", event_id="e1", event_name="x")  # no event_timestamp


def test_wrong_type_is_invalid():
    with pytest.raises(ValidationError):
        tw.MatchFinished(
            user_id="u1",
            event_timestamp=datetime(2025, 1, 1),
            event_id="e1",
            match_id="m1",
            duration_sec="not-an-int",
            hero="axe",
            kills=3,
            deaths=1,
        )
