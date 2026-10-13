# SPDX-License-Identifier: Apache-2.0
"""Telemetry utility functions for TPU Raiden metrics."""

import functools
import re

# Metric name prefix
RAIDEN_METRIC_PREFIX = "tpu_raiden_"

_LABELED_METRIC_KEY_REGEX = re.compile(r"([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?")
_LABEL_PAIR_REGEX = re.compile(
    r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"([^"\\]*(?:\\.[^"\\]*)*)"'
)
_UNESCAPE_PATTERN = re.compile(r'\\([\\n"tr])')
_UNESCAPE_MAP = {"n": "\n", "t": "\t", "r": "\r"}
_KEY_FULLMATCH = _LABELED_METRIC_KEY_REGEX.fullmatch
_PAIR_FINDALL = _LABEL_PAIR_REGEX.findall
_UNESCAPE_SUB = _UNESCAPE_PATTERN.sub


def _unescape_match(match: re.Match) -> str:
    escaped = match.group(1)
    return _UNESCAPE_MAP.get(escaped, escaped)


@functools.lru_cache(maxsize=512)
def parse_raiden_metric_key(raw_key: str) -> tuple[str, dict[str, str]]:
    """Parses Prometheus text-formatted keys emitted by Raiden telemetry, e.g.:

    tpu_raiden_sent_bytes_total{direction="push"} into base name and labels
    dict.
    """
    stripped = raw_key.strip()
    match = _KEY_FULLMATCH(stripped)
    if not match:
        return stripped, {}

    base_name, raw_labels = match.group(1, 2)
    if not raw_labels:
        return base_name, {}

    if "\\" not in raw_labels:
        return base_name, dict(_PAIR_FINDALL(raw_labels))

    return base_name, {
        k: (_UNESCAPE_SUB(_unescape_match, raw_val) if "\\" in raw_val else raw_val)
        for k, raw_val in _PAIR_FINDALL(raw_labels)
    }


@functools.lru_cache(maxsize=512)
def normalize_raiden_metric_name(name: str) -> str:
    """Ensures a Raiden metric name starts with RAIDEN_METRIC_PREFIX."""
    return (
        name
        if name.startswith(RAIDEN_METRIC_PREFIX)
        else f"{RAIDEN_METRIC_PREFIX}{name}"
    )
