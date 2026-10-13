# SPDX-License-Identifier: Apache-2.0
"""Unit tests for vllm_torchtpu.distributed.kv_transfer.raiden.telemetry_utils."""

from __future__ import annotations

import sys
import types
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "vllm_torchtpu"
if (_SRC_ROOT / "distributed" / "kv_transfer" / "raiden").is_dir():
    for _pkg_name, _pkg_dir in (
        ("vllm_torchtpu", _SRC_ROOT),
        ("vllm_torchtpu.distributed", _SRC_ROOT / "distributed"),
        (
            "vllm_torchtpu.distributed.kv_transfer",
            _SRC_ROOT / "distributed" / "kv_transfer",
        ),
        (
            "vllm_torchtpu.distributed.kv_transfer.raiden",
            _SRC_ROOT / "distributed" / "kv_transfer" / "raiden",
        ),
    ):
        _pkg = types.ModuleType(_pkg_name)
        _pkg.__path__ = [str(_pkg_dir)]
        sys.modules.setdefault(_pkg_name, _pkg)

from vllm_torchtpu.distributed.kv_transfer.raiden.telemetry_utils import (  # noqa: E402
    _UNESCAPE_PATTERN,
    RAIDEN_METRIC_PREFIX,
    _unescape_match,
    normalize_raiden_metric_name,
    parse_raiden_metric_key,
)


def test_parse_raiden_metric_key_unlabeled_and_invalid() -> None:
    """Verify unlabeled, empty-braces, whitespace, and malformed metric keys."""
    parse_raiden_metric_key.cache_clear()
    assert parse_raiden_metric_key("tpu_raiden_h2d_transfer_time_ms") == (
        "tpu_raiden_h2d_transfer_time_ms",
        {},
    )
    assert parse_raiden_metric_key("  tpu_raiden:latency:seconds  ") == (
        "tpu_raiden:latency:seconds",
        {},
    )
    assert parse_raiden_metric_key("tpu_raiden_metric{}") == (
        "tpu_raiden_metric",
        {},
    )
    assert parse_raiden_metric_key("tpu_raiden_metric{   }") == (
        "tpu_raiden_metric",
        {},
    )
    assert parse_raiden_metric_key("  123_invalid{unclosed  ") == (
        "123_invalid{unclosed",
        {},
    )
    assert parse_raiden_metric_key("") == ("", {})


def test_parse_raiden_metric_key_labeled_fast_and_escaped_paths() -> None:
    """Verify fast no-backslash path and mixed escaped/unescaped label values."""
    parse_raiden_metric_key.cache_clear()
    base, labels = parse_raiden_metric_key(
        'tpu_raiden_transfer_duration_ms{direction="pull", mode="direct"}'
    )
    assert base == "tpu_raiden_transfer_duration_ms"
    assert labels == {"direction": "pull", "mode": "direct"}

    raw_mixed_escape = (
        r'tpu_raiden_failures{plain="ok",err="msg=\"timeout\"\nline2\tcol\r",'
        r'path="C:\\dir"}'
    )
    base_esc, labels_esc = parse_raiden_metric_key(raw_mixed_escape)
    assert base_esc == "tpu_raiden_failures"
    assert labels_esc == {
        "plain": "ok",
        "err": 'msg="timeout"\nline2\tcol\r',
        "path": "C:\\dir",
    }

    match_quote = _UNESCAPE_PATTERN.search(r"\"")
    assert match_quote is not None
    assert _unescape_match(match_quote) == '"'


def test_normalize_raiden_metric_name_and_prefix() -> None:
    """Verify prefix normalization and LRU cache hits."""
    normalize_raiden_metric_name.cache_clear()
    assert RAIDEN_METRIC_PREFIX == "tpu_raiden_"
    assert (
        normalize_raiden_metric_name("tpu_raiden_sent_bytes_total")
        == "tpu_raiden_sent_bytes_total"
    )
    assert (
        normalize_raiden_metric_name("sent_bytes_total")
        == "tpu_raiden_sent_bytes_total"
    )
    assert (
        normalize_raiden_metric_name("sent_bytes_total")
        == "tpu_raiden_sent_bytes_total"
    )
