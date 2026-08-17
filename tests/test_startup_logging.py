# SPDX-License-Identifier: Apache-2.0
"""Contract tests for structured startup phase events."""

import json

import pytest

from vllm_torchtpu import startup_logging


class _RecordingLogger:

    def __init__(self) -> None:
        self.info_lines: list[str] = []
        self.warning_lines: list[str] = []

    def info(self, message: str, *args: object) -> None:
        self.info_lines.append(message % args)

    def warning(self, message: str, *args: object) -> None:
        self.warning_lines.append(message % args)


def _configure_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "VLLM_STARTUP_ID": "1234567890abcdef1234567890abcdef",
        "VLLM_SERVICE_ID": "q397-decode",
        "VLLM_ROLE": "decode",
        "VLLM_MODEL_ALIAS": "Qwen3.5-397B-A17B-FP8",
        "VLLM_REQUEST_ID": "1234567890abcdef1234567890abcdef",
        "VLLM_INVOCATION_ID": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "VLLM_STARTUP_ORIGIN_MONOTONIC_NS": "1000000000",
        "VLLM_STARTUP_ORIGIN_UNIX_NS": "101000000000",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def _events(logger: _RecordingLogger) -> list[dict[str, object]]:
    return [
        json.loads(line.removeprefix(startup_logging.EVENT_PREFIX))
        for line in logger.info_lines
    ]


def test_startup_span_is_disabled_without_wrapper_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _RecordingLogger()
    monkeypatch.setattr(startup_logging, "logger", logger)
    monkeypatch.delenv("VLLM_STARTUP_ID", raising=False)

    with startup_logging.startup_span(
            "model_load", rank=0, world_size=8):
        pass

    assert logger.info_lines == []
    assert logger.warning_lines == []


def test_process_startup_identity_uses_slice_global_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("WORLD_SIZE", "8")

    assert startup_logging.process_startup_identity() == (7, 8)


def test_startup_span_emits_paired_monotonic_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _RecordingLogger()
    _configure_identity(monkeypatch)
    monkeypatch.setattr(startup_logging, "logger", logger)
    monotonic_values = iter(
        [2_000_000_000, 2_000_000_000, 5_000_000_000])
    unix_values = iter([102_000_000_000, 105_000_000_000])
    monkeypatch.setattr(startup_logging.time, "monotonic_ns",
                        lambda: next(monotonic_values))
    monkeypatch.setattr(startup_logging.time, "time_ns",
                        lambda: next(unix_values))

    with startup_logging.startup_span(
            "compile_subgraph",
            rank=3,
            world_size=8,
            detail="model backbone"):
        pass

    start, end = _events(logger)
    assert start["state"] == "start"
    assert end["state"] == "end"
    assert start["span_id"] == end["span_id"]
    assert end["duration_ms"] == 3000.0
    assert start["elapsed_ms"] == 1000.0
    assert end["elapsed_ms"] == 4000.0
    assert end["rank"] == 3
    assert end["world_size"] == 8
    assert end["detail"] == "model backbone"


def test_startup_span_preserves_error_and_async_pairing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _RecordingLogger()
    _configure_identity(monkeypatch)
    monkeypatch.setattr(startup_logging, "logger", logger)
    monotonic_values = iter([
        2_000_000_000,
        2_000_000_000,
        3_000_000_000,
        4_000_000_000,
        4_000_000_000,
        6_000_000_000,
    ])
    unix_values = iter([
        102_000_000_000,
        103_000_000_000,
        104_000_000_000,
        106_000_000_000,
    ])
    monkeypatch.setattr(startup_logging.time, "monotonic_ns",
                        lambda: next(monotonic_values))
    monkeypatch.setattr(startup_logging.time, "time_ns",
                        lambda: next(unix_values))

    with pytest.raises(RuntimeError, match="boom"):
        with startup_logging.startup_span(
                "profile_run", rank=1, world_size=8):
            raise RuntimeError("boom")

    async_span = startup_logging.begin_startup_span(
        "raiden_worker_register", rank=1, world_size=8)
    startup_logging.finish_startup_span(async_span)

    start, error, async_start, async_end = _events(logger)
    assert error["state"] == "error"
    assert error["span_id"] == start["span_id"]
    assert error["error_type"] == "RuntimeError"
    assert async_end["state"] == "end"
    assert async_end["span_id"] == async_start["span_id"]
    assert async_end["duration_ms"] == 2000.0
