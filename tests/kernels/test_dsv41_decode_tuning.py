# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config misses must preserve request boundaries without padding the batch."""

from unittest.mock import Mock

import pytest

from vllm_torchtpu.kernels.deepseek_v41.index_selection import (
    tuned_block_sizes as tuning,
)


def key(device_kind="test-device"):
    return tuning.DecodeTuningKey(device_kind, 6, 32, 131072, 128, 512, 2)


@pytest.mark.cpu_test
@pytest.mark.parametrize("requests, expected", [(48, 16), (24, 16), (7, 7), (17, 16)])
def test_decode_tuning_miss_preserves_whole_requests(monkeypatch, requests, expected):
    log = Mock()
    monkeypatch.setattr(tuning, "logger", log)
    lookup = key()
    assert tuning.get_decode_request_tile(lookup, requests) == expected
    log.warning_once.assert_called_once_with(
        "[decode_indexer] No tuned request_tile for %s; using %d", lookup, expected
    )


@pytest.mark.cpu_test
def test_decode_tuning_matches_exact_hardware_and_shape(monkeypatch):
    log = Mock()
    monkeypatch.setattr(tuning, "logger", log)
    monkeypatch.setitem(tuning.DECODE_REQUEST_TILES, key(), 24)
    assert tuning.get_decode_request_tile(key(), 48) == 24
    log.warning_once.assert_not_called()
    assert tuning.get_decode_request_tile(key(device_kind="other-device"), 48) == 16
    assert tuning.get_decode_request_tile(key(), 96) == 24


@pytest.mark.cpu_test
@pytest.mark.parametrize("tile", [0, -1, True])
def test_decode_tuning_rejects_invalid_table_entries(monkeypatch, tile):
    monkeypatch.setitem(tuning.DECODE_REQUEST_TILES, key(), tile)
    with pytest.raises(ValueError, match="Invalid decode request_tile"):
        tuning.get_decode_request_tile(key(), 48)


@pytest.mark.cpu_test
def test_decode_tuning_rejects_empty_batch():
    with pytest.raises(ValueError, match="requests must be positive"):
        tuning.get_decode_request_tile(key(), 0)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "requests,queries,capacity",
    [
        (24, 6, 49152),
        (64, 6, 49152),
        (96, 6, 49152),
        (48, 6, 98304),
        (48, 3, 98304),
        (7, 6, 49152),
    ],
)
def test_sparse_mqa_row_tiles_bound_scores(monkeypatch, requests, queries, capacity):
    monkeypatch.setattr(tuning, "logger", Mock())
    rows = requests * queries
    lookup = tuning.SparseMQATuningKey("TPU7x", 32, capacity, 8192, 512)
    tile = tuning.get_sparse_mqa_block_sizes(lookup, rows).row_tile
    assert tile <= rows
    assert tile * capacity <= tuning.SPARSE_MQA_SCORE_ELEMENTS


@pytest.mark.cpu_test
def test_sparse_mqa_exact_tuning_precedes_default(monkeypatch):
    lookup = tuning.SparseMQATuningKey("TPU7x", 32, 49152, None, 512)
    config = tuning.SparseMQABlockSizes(4096, 12)
    monkeypatch.setitem(tuning.SPARSE_MQA_BLOCK_SIZES, lookup, config)
    assert tuning.get_sparse_mqa_block_sizes(lookup, 24) == config
