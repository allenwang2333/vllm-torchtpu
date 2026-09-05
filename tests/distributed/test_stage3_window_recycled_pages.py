# SPDX-License-Identifier: Apache-2.0
"""Stage 3 must not transfer a sliding-window page the allocator has taken back.

Both peers work out which leading pages of a windowed group are dead from
``num_tokens``, the transfer prefix. The allocator, meanwhile, freed pages
against wherever the producer actually stopped. Those two numbers agree only
while the producer halts at the handoff token.

Let them drift and the send lists a page the allocator has already pointed at
the null block. One dead page arrives as a hole in the decode replica's window
and nothing complains. Two collide into a duplicate destination, which Raiden
rejects by raising out of ``execute_model`` on the producer worker, killing the
engine and every other in-flight request with it.

``_stage3_truncates_prompt`` is what stops the drift: the producer prefills
N-1 tokens, so the allocator and the connector read the same number and the
two page counts cannot disagree. The dead-page count remains as a guard for
the case truncation cannot reach, a client that hits the prefill port directly
and generates past the handoff.

The tables below are verbatim ``group_block_ids`` captured from a live
DP4/DP4 DeepSeek-V4-Flash pair, so the geometry and the null runs are the real
ones rather than a reconstruction.

Deviceless: this is block-table arithmetic and never touches a TPU.
"""

from __future__ import annotations

import pytest

from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (
    TPURaidenConnectorScheduler, _stage3_recycled_transfer_pages,
    _stage3_skipped_window_pages, _Stage3TransferGroup)

# DeepSeek-V4-Flash compressor-state geometry: a 16-token page under an
# 8-token window, the tightest ratio the model has and so the first to drift.
_STATE_CSA_PAGE = 16
_STATE_CSA_WINDOW = 8


def test_handoff_length_transfers_only_live_pages():
    """The normal P/D case: the producer stops one token past the prefix."""
    # computed=114, num_tokens=113. The allocator freed six pages and the
    # connector skips six, so both live pages go over the wire.
    ids = [0, 0, 0, 0, 0, 0, 286, 58]
    assert _stage3_skipped_window_pages(num_tokens=113,
                                        page_tokens=_STATE_CSA_PAGE,
                                        window_tokens=_STATE_CSA_WINDOW) == 6
    assert _stage3_recycled_transfer_pages(
        ids,
        num_tokens=113,
        page_tokens=_STATE_CSA_PAGE,
        window_tokens=_STATE_CSA_WINDOW) == 0


def test_one_page_of_drift_is_caught_rather_than_shipped_dead():
    """computed=375 against num_tokens=374, which happens on the proxy path.

    The allocator freed 23 pages, the connector skips 22, so page 22 is the
    null block. This one is silent without the check: a single dead page in an
    8-token window rarely changes the answer, so it survives an accuracy sweep
    and corrupts the decode replica's state anyway.
    """
    ids = [0] * 23 + [13]
    assert _stage3_skipped_window_pages(num_tokens=374,
                                        page_tokens=_STATE_CSA_PAGE,
                                        window_tokens=_STATE_CSA_WINDOW) == 22
    assert _stage3_recycled_transfer_pages(
        ids,
        num_tokens=374,
        page_tokens=_STATE_CSA_PAGE,
        window_tokens=_STATE_CSA_WINDOW) == 1


def test_producer_that_kept_generating_is_caught():
    """The engine-killing case: a 114-token prompt run out to 161 tokens.

    This is what a request sent straight at the producer does, bypassing the
    proxy's ``max_tokens=1`` clamp. The allocator freed nine of the eleven
    pages it had by then, but the transfer prefix is only eight pages long, so
    ``_stage3_normalize_producer_blocks`` keeps ``[0] * 8`` and throws away the
    two live ones at the tail. Skipping six leaves two null blocks, which is
    the duplicate destination that killed the engine.
    """
    # [0] * 9 + [250, 18], normalized down to the eight-page prefix.
    ids = [0] * 8
    assert _stage3_recycled_transfer_pages(
        ids,
        num_tokens=113,
        page_tokens=_STATE_CSA_PAGE,
        window_tokens=_STATE_CSA_WINDOW) == 2


def test_truncation_removes_the_drift_at_its_source():
    """Stopping the producer at N-1 puts both formulas on the same input.

    This is the repair the guard above only reports. The 375-token prompt that
    drifts by a page becomes a 374-token prefill, so the allocator frees the
    same 22 pages the connector skips and no dead page is listed at all.
    """
    truncated = 374
    freed = max(0, truncated - _STATE_CSA_WINDOW + 1) // _STATE_CSA_PAGE
    assert freed == _stage3_skipped_window_pages(
        num_tokens=truncated,
        page_tokens=_STATE_CSA_PAGE,
        window_tokens=_STATE_CSA_WINDOW)
    ids = [0] * freed + [13, 41]
    assert _stage3_recycled_transfer_pages(
        ids,
        num_tokens=truncated,
        page_tokens=_STATE_CSA_PAGE,
        window_tokens=_STATE_CSA_WINDOW) == 0


@pytest.mark.parametrize(
    "mamba_indices,windows,expected",
    [
        # Qwen3.5 and GLM: one full-attention group, so P may run to the end
        # of the prompt.
        ([], [None], False),
        # A windowed group recycles pages against how far P ran.
        ([], [None, 128, 8], True),
        # The original Mamba reason, unchanged.
        ([2], [None], True),
        # Both at once still truncates exactly once.
        ([2], [None, 8], True),
    ],
)
def test_which_models_truncate_the_prompt(mamba_indices, windows, expected):
    scheduler = TPURaidenConnectorScheduler.__new__(
        TPURaidenConnectorScheduler)
    scheduler._stage3_mamba_group_indices = list(mamba_indices)
    scheduler._stage3_transfer_groups = tuple(
        _Stage3TransferGroup(
            cache_group_index=index, page_tokens=16, window_tokens=window)
        for index, window in enumerate(windows))
    assert scheduler._stage3_truncates_prompt() is expected


@pytest.mark.parametrize(
    "ids,num_tokens,page_tokens,window_tokens,expected",
    [
        # dsv4.state.hca.g5, computed=287: a 32-token page under a 128-token
        # window drifts the same way, just less often.
        ([0, 0, 0, 0, 0, 7, 5, 4, 2], 286, 32, 128, 1),
        # dsv4.state.idx.g3, computed=327.
        ([0] * 10 + [280], 326, 32, 8, 1),
        # dsv4.swa.g1, computed=2175. A 128-token page under a 128-token
        # window is the least exposed and still drifts.
        ([0] * 16 + [270], 2174, 128, 128, 1),
        # A full-attention group has no window, so nothing is ever skipped and
        # nothing is ever dead.
        ([84, 85, 242, 279, 109], 4371, 1024, None, 0),
    ],
)
def test_captured_group_tables(ids, num_tokens, page_tokens, window_tokens,
                               expected):
    assert _stage3_recycled_transfer_pages(
        ids,
        num_tokens=num_tokens,
        page_tokens=page_tokens,
        window_tokens=window_tokens) == expected
