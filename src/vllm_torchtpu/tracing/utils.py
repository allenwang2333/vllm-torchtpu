# Copyright 2026 Google LLC
"""Helper utilities for request ID and KV cache profiler tracing."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


_REQ_ID_KEYS: tuple[str, ...] = tuple(f"request_id{i}" for i in range(1, 4097))
_KV_LEN_KEYS: tuple[str, ...] = tuple(f"kv_len{i}" for i in range(1, 4097))
_TRIM_CACHE: dict[str, str] = {}


def trim_request_id_suffix(request_id: str) -> str:
    cached = _TRIM_CACHE.get(request_id)
    if cached is not None:
        return cached
    if not request_id.startswith("cmpl-"):
        _TRIM_CACHE[request_id] = request_id
        return request_id
    parts = request_id.split("-", 6)
    res = request_id[: -len(parts[6]) - 1] if len(parts) > 6 else request_id
    _TRIM_CACHE[request_id] = res
    return res


def extract_request_ids_for_tracing(
    req_ids: list[Any], start_index: int = 0, num_reqs: int = -1
) -> dict[str, str]:
    try:
        active_ids = (
            req_ids[start_index : start_index + num_reqs] if num_reqs != -1 else req_ids
        )
        out: dict[str, str] = {}
        keys = _REQ_ID_KEYS
        cache = _TRIM_CACHE
        idx = 0
        for rid in active_ids:
            if rid is not None:
                s = rid if type(rid) is str else str(rid)
                trimmed = cache.get(s)
                if trimmed is None:
                    trimmed = trim_request_id_suffix(s)
                out[keys[idx]] = trimmed
                idx += 1
        return out
    except Exception as e:
        logger.warning("Failed to extract request IDs for tracing: %s", e)
        return {}


def extract_kv_lens_for_tracing(
    num_computed_tokens: Any, start_index: int = 0, num_reqs: int = -1
) -> dict[str, int]:
    end_index = start_index + num_reqs if num_reqs != -1 else None
    active_slice = num_computed_tokens[start_index:end_index]
    n = len(active_slice)
    if n == 0:
        return {}

    vals: list[int] = (
        active_slice.tolist()
        if hasattr(active_slice, "tolist")
        else [int(x) for x in active_slice]
    )
    result = dict(zip(_KV_LEN_KEYS, vals))
    result["min_kv_len"] = min(vals)
    result["max_kv_len"] = max(vals)
    result["avg_kv_len"] = sum(vals) // n
    return result
