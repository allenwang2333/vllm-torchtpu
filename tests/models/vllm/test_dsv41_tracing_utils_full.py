# Copyright 2026 Google LLC
"""Full 100% statement and branch coverage tests for tracing.utils."""

from unittest.mock import patch

import numpy as np
import pytest

from vllm_torchtpu.tracing.utils import (
    _TRIM_CACHE,
    extract_kv_lens_for_tracing,
    extract_request_ids_for_tracing,
    trim_request_id_suffix,
)

pytestmark = pytest.mark.cpu_test


def test_trim_request_id_suffix_all_branches() -> None:
    _TRIM_CACHE.clear()
    # Non-cmpl prefix (uncached then cached)
    assert trim_request_id_suffix("req-abc-1-2-3-4-5-6-7") == "req-abc-1-2-3-4-5-6-7"
    assert trim_request_id_suffix("req-abc-1-2-3-4-5-6-7") == "req-abc-1-2-3-4-5-6-7"

    # Short cmpl prefix (<= 6 parts)
    assert trim_request_id_suffix("cmpl-1234") == "cmpl-1234"
    assert trim_request_id_suffix("cmpl-1-2-3-4-5") == "cmpl-1-2-3-4-5"

    # Long cmpl prefix (> 6 parts, uncached then cached)
    assert trim_request_id_suffix("cmpl-1-2-3-4-5-6-7-8") == "cmpl-1-2-3-4-5"
    assert trim_request_id_suffix("cmpl-1-2-3-4-5-6-7-8") == "cmpl-1-2-3-4-5"


def test_extract_request_ids_for_tracing_all_branches() -> None:
    _TRIM_CACHE.clear()
    req_ids = ["cmpl-a-b-c-d-e-f-g", None, 12345, "custom-1"]

    # Slice with start_index and num_reqs != -1 (uncached and cached paths)
    res1 = extract_request_ids_for_tracing(req_ids, start_index=0, num_reqs=3)
    assert res1 == {"request_id1": "cmpl-a-b-c-d-e", "request_id2": "12345"}

    # Full list with num_reqs == -1 (hits cached entries + new entry)
    res2 = extract_request_ids_for_tracing(req_ids)
    assert res2 == {
        "request_id1": "cmpl-a-b-c-d-e",
        "request_id2": "12345",
        "request_id3": "custom-1",
    }

    # Exception handling branch
    class _BrokenStr:
        def __str__(self) -> str:
            raise RuntimeError("str conversion failed")

    with patch("vllm_torchtpu.tracing.utils.logger.warning") as mock_warn:
        assert extract_request_ids_for_tracing([_BrokenStr()], 0, 1) == {}
        mock_warn.assert_called_once()


def test_extract_kv_lens_for_tracing_all_branches() -> None:
    # Empty slice
    assert extract_kv_lens_for_tracing([], start_index=0, num_reqs=2) == {}

    # Python list with num_reqs == -1
    res_list = extract_kv_lens_for_tracing([200, 100, 300])
    assert res_list == {
        "kv_len1": 200,
        "kv_len2": 100,
        "kv_len3": 300,
        "min_kv_len": 100,
        "max_kv_len": 300,
        "avg_kv_len": 200,
    }

    # NumPy array with explicit slice
    arr = np.array([64, 128, 256, 512], dtype=np.int32)
    res_np = extract_kv_lens_for_tracing(arr, start_index=1, num_reqs=2)
    assert res_np == {
        "kv_len1": 128,
        "kv_len2": 256,
        "min_kv_len": 128,
        "max_kv_len": 256,
        "avg_kv_len": 192,
    }
