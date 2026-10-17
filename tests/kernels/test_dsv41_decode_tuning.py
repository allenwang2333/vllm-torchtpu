# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config misses must preserve request boundaries without padding the batch."""

from unittest.mock import Mock

import pytest

from vllm_torchtpu.kernels.deepseek_v41.index_selection import (
    tuned_block_sizes as tuning,
)


def key(requests=48, device_kind="test-device"):
    return tuning.DecodeTuningKey(
        device_kind, requests, 6, 32, 131072, 128, 4096, 512, 2
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize("requests, expected", [(48, 16), (24, 12), (7, 7), (17, 1)])
def test_decode_tuning_miss_preserves_whole_requests(monkeypatch, requests, expected):
    log = Mock()
    monkeypatch.setattr(tuning, "logger", log)
    lookup = key(requests)
    assert tuning.get_decode_request_tile(lookup) == expected
    log.warning_once.assert_called_once_with(
        "[decode_indexer] No tuned request_tile for %s; using %d", lookup, expected
    )


@pytest.mark.cpu_test
def test_decode_tuning_matches_exact_hardware_and_shape(monkeypatch):
    log = Mock()
    monkeypatch.setattr(tuning, "logger", log)
    monkeypatch.setitem(tuning.DECODE_REQUEST_TILES, key(), 24)
    assert tuning.get_decode_request_tile(key()) == 24
    log.warning_once.assert_not_called()
    assert tuning.get_decode_request_tile(key(device_kind="other-device")) == 16
    assert tuning.get_decode_request_tile(key(requests=96)) == 16


@pytest.mark.cpu_test
@pytest.mark.parametrize("tile", [0, 5, True])
def test_decode_tuning_rejects_invalid_table_entries(monkeypatch, tile):
    monkeypatch.setitem(tuning.DECODE_REQUEST_TILES, key(), tile)
    with pytest.raises(ValueError, match="Invalid decode request_tile"):
        tuning.get_decode_request_tile(key())


@pytest.mark.cpu_test
def test_decode_tuning_rejects_empty_batch():
    with pytest.raises(ValueError, match="requests must be positive"):
        tuning.get_decode_request_tile(key(requests=0))


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "device,requests,queries,capacity,k,packed",
    [
        ("TPU7x", 48, 6, 49152, 512, False),
        ("TPU7x", 48, 6, 49152, 512, True),
        ("other", 48, 6, 49152, 512, False),
        ("TPU7x", 48, 5, 49152, 512, False),
        ("TPU7x", 24, 6, 12288, 512, False),
        ("TPU7x", 24, 6, 6144, 512, False),
        ("TPU7x", 48, 12, 49152, 512, False),
        ("TPU7x", 48, 6, 49152, 256, False),
    ],
)
def test_candidate_tiles_preserve_aligned_windows(
    device, requests, queries, capacity, k, packed
):
    lookup = tuning.CandidateTuningKey(
        device, requests, queries, 32, capacity, None, k, packed
    )
    config = tuning.get_candidate_block_sizes(lookup)
    assert 0 < config.key_tile <= capacity
    assert capacity % config.key_tile == 0
    assert 0 < config.request_tile <= requests
    assert requests % config.request_tile == 0


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
def test_candidate_default_groups_bound_scores_without_splitting_requests(
    requests, queries, capacity
):
    lookup = tuning.CandidateTuningKey(
        "TPU7x", requests, queries, 32, capacity, 8192, 512
    )
    tile = tuning.get_candidate_block_sizes(lookup).request_tile
    assert 0 < tile <= requests
    assert requests % tile == 0
    assert tile * queries * capacity <= tuning.CANDIDATE_SCORE_ELEMENTS


@pytest.mark.cpu_test
def test_candidate_exact_tuning_precedes_default(monkeypatch):
    lookup = tuning.CandidateTuningKey("TPU7x", 24, 6, 32, 49152, None, 512)
    config = tuning.CandidateBlockSizes(4096, 12)
    monkeypatch.setitem(tuning.CANDIDATE_BLOCK_SIZES, lookup, config)
    assert tuning.get_candidate_block_sizes(lookup) == config
