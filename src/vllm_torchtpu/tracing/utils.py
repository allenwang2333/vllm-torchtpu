# Copyright 2026 Google LLC
"""Helper utilities for request ID and KV cache profiler tracing."""
import logging
from typing import Any

logger = logging.getLogger(__name__)

def trim_request_id_suffix(request_id: str, log_this_frame: bool = False) -> str:
    original = request_id
    parts = request_id.split("-")
    if len(parts) >= 6 and parts[0] == "cmpl":
        request_id = "-".join(parts[:6])
    if log_this_frame:
        logger.info(f"Custom Debug: [TRIM REQUEST ID] Before: '{original}' | After: '{request_id}'")
    return request_id

def extract_request_ids_for_tracing(req_ids: list[Any], start_index: int = 0, num_reqs: int = -1, log_this_frame: bool = False) -> dict[str, str]:
    if log_this_frame:
        logger.info("Custom Debug: extract_request_ids_for_tracing() called")
    req_id_kwargs = {}
    try:
        active_ids = req_ids[start_index:start_index+num_reqs] if num_reqs != -1 else req_ids
        trimmed_req_ids = [trim_request_id_suffix(str(rid), log_this_frame) for rid in active_ids if rid is not None]
        for i, rid in enumerate(trimmed_req_ids):
            req_id_kwargs[f"request_id{i+1}"] = rid
    except Exception as e:
        logger.warning(f"Failed to extract request IDs for tracing: {e}")
    
    if log_this_frame:
        logger.info(f"Custom Debug: extract_request_ids_for_tracing returned {req_id_kwargs}")
    return req_id_kwargs
