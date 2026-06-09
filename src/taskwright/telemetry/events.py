"""Level 1 — telemetry contract (Pydantic v2).

The framework fixes a small mandatory base (:class:`BaseEvent`); the developer
extends it with subclasses that declare game-specific fields. "Validation" means
exactly: does an event conform to this Pydantic schema (valid <=> conforms).
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class BaseEvent(BaseModel):
    """Mandatory telemetry base — the framework's fixed vocabulary.

    Attributes:
        user_id: stable identifier of the player/account.
        event_timestamp: when the event occurred.
        event_id: unique event identifier, used for idempotent deduplication.
        event_name: event type; subclasses narrow it to a ``Literal``.
    """

    user_id: str
    event_timestamp: datetime
    event_id: str
    event_name: str


class MatchFinished(BaseEvent):
    """Example developer-defined event (illustrative, not apробation).

    Demonstrates how a subclass narrows ``event_name`` to a ``Literal`` and adds
    its own game-specific fields on top of the fixed base.
    """

    event_name: Literal["match_finished"] = "match_finished"
    match_id: str
    duration_sec: int
    hero: str
    kills: int
    deaths: int
