# Copyright 2026 Google LLC
"""Helper utilities for request ID and KV cache profiler tracing."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def trim_request_id_suffix(request_id: str) -> str:
    if not request_id.startswith("cmpl-"):
        return request_id
    parts = request_id.split("-", 6)
    if len(parts) >= 6:
        return "-".join(parts[:6])
    return request_id


def extract_request_ids_for_tracing(
    req_ids: list[Any], start_index: int = 0, num_reqs: int = -1
) -> dict[str, str]:
    try:
        active_ids = (
            req_ids[start_index : start_index + num_reqs] if num_reqs != -1 else req_ids
        )
        trimmed_req_ids = [
            trim_request_id_suffix(str(rid)) for rid in active_ids if rid is not None
        ]
        return {f"request_id{i}": rid for i, rid in enumerate(trimmed_req_ids, start=1)}
    except Exception as e:
        logger.warning("Failed to extract request IDs for tracing: %s", e)
        return {}


def extract_kv_lens_for_tracing(
    num_computed_tokens: Any, start_index: int = 0, num_reqs: int = -1
) -> dict[str, int]:
    end_index = start_index + num_reqs if num_reqs != -1 else None
    active_slice = num_computed_tokens[start_index:end_index]
    if len(active_slice) == 0:
        return {}

    result: dict[str, int] = {}
    first_len = int(active_slice[0])
    min_kv_len = first_len
    max_kv_len = first_len
    sum_kv_len = 0
    for i, raw_len in enumerate(active_slice):
        kv_len = int(raw_len)
        result[f"kv_len{i + 1}"] = kv_len
        if kv_len < min_kv_len:
            min_kv_len = kv_len
        elif kv_len > max_kv_len:
            max_kv_len = kv_len
        sum_kv_len += kv_len

    result["min_kv_len"] = min_kv_len
    result["max_kv_len"] = max_kv_len
    result["avg_kv_len"] = sum_kv_len // len(active_slice)
    return result
