"""Common prefix — telemetry validation aborts; dispatch rejects unknown tasks."""
from __future__ import annotations

import pytest

import taskwright as tw
from taskwright.exceptions import TelemetryValidationError


def test_invalid_event_mapping_aborts(events, det_task, tmp_path):
    # A mapping missing required contract fields is invalid -> abort before work.
    bad_stream = list(events) + [{"user_id": "u", "event_id": "x"}]
    with pytest.raises(TelemetryValidationError):
        tw.run(det_task, bad_stream, dataset="synthetic", output_dir=tmp_path)


def test_invalid_event_object_aborts(events, sup_task, tmp_path):
    bad_stream = list(events) + [42]  # not an event, not a mapping
    with pytest.raises(TelemetryValidationError):
        tw.run(sup_task, bad_stream, dataset="synthetic", output_dir=tmp_path)


def test_unknown_task_type_raises(events, tmp_path):
    with pytest.raises(TypeError):
        tw.run(object(), events, output_dir=tmp_path)
