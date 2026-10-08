# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from vllm_torchtpu.runner.owner_token_layout import build_owner_token_layout


def _check_layout(
    layout,
    owners,
    tokens,
    tp,
    paddings,
    *,
    num_padded=None,
    reqs_padded=None,
    multiple=1,
):
    """The invariants the attention kernels rely on."""
    owners = np.asarray(owners)
    tokens = np.asarray(tokens)
    num_reqs = owners.size
    total = int(tokens.sum())
    seg = layout.segment_tokens
    counts = np.bincount(owners, weights=tokens, minlength=tp).astype(np.int64)
    fits = [p for p in paddings if p % (tp * multiple) == 0 and p // tp >= counts.max()]
    assert layout.padded_tokens == min(fits) == tp * seg
    np.testing.assert_array_equal(layout.owner_tokens, counts)

    qsl = layout.query_start_loc
    assert qsl.shape == (tp, (reqs_padded or num_reqs) + 1)
    starts = np.cumsum(tokens) - tokens
    for o in range(tp):
        # Owner o's tokens fill the front of segment o, requests whole and
        # in batch order.
        expected = [
            starts[i] + t
            for i in range(num_reqs)
            if owners[i] == o
            for t in range(tokens[i])
        ]
        np.testing.assert_array_equal(
            layout.perm[o * seg : o * seg + counts[o]], expected
        )
        lengths = np.diff(qsl[o])
        np.testing.assert_array_equal(
            lengths[:num_reqs], np.where(owners == o, tokens, 0)
        )
        assert not lengths[num_reqs:].any()
        for i in np.flatnonzero(owners == o):
            np.testing.assert_array_equal(
                layout.perm[o * seg + qsl[o, i] : o * seg + qsl[o, i + 1]],
                starts[i] + np.arange(tokens[i]),
            )

    batch = layout.inv.size
    assert batch == (num_padded or layout.padded_tokens)
    np.testing.assert_array_equal(layout.perm[layout.inv[:total]], np.arange(total))
    assert layout.perm.min() >= 0 and layout.perm.max() < batch
    assert layout.inv.min() >= 0 and layout.inv.max() < layout.padded_tokens
    if batch == layout.padded_tokens:
        np.testing.assert_array_equal(layout.perm[layout.inv], np.arange(batch))
    assert layout.perm.dtype == layout.inv.dtype == qsl.dtype == np.int32
    assert layout.owner_tokens.dtype == np.int32


def test_example_layout():
    layout = build_owner_token_layout(
        [1, 0, 1], [2, 1, 3], 2, [4, 8, 16], num_reqs_padded=4
    )

    # Owner 1 has 5 tokens, so 8 rows per owner do not fit them.
    assert (layout.padded_tokens, layout.segment_tokens) == (16, 8)
    np.testing.assert_array_equal(
        layout.perm, [2, 6, 7, 8, 9, 10, 11, 12, 0, 1, 3, 4, 5, 13, 14, 15]
    )
    np.testing.assert_array_equal(
        layout.inv, [8, 9, 0, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 13, 14, 15]
    )
    np.testing.assert_array_equal(
        layout.query_start_loc, [[0, 0, 1, 1, 1], [0, 2, 2, 5, 5]]
    )
    np.testing.assert_array_equal(layout.owner_tokens, [1, 5])


def test_balanced_owner_major_batch_matches_positional_split():
    owners = [0, 0, 1, 2, 2, 3]
    tokens = [3, 1, 4, 2, 2, 4]

    layout = build_owner_token_layout(owners, tokens, 4, [16, 32])

    np.testing.assert_array_equal(layout.perm, np.arange(16))
    np.testing.assert_array_equal(layout.inv, np.arange(16))
    qsl = np.concatenate(([0], np.cumsum(tokens)))
    for o in range(4):
        # The positional split's per-rank query_start_loc.
        np.testing.assert_array_equal(
            layout.query_start_loc[o], np.clip(qsl - o * 4, 0, 4)
        )


def test_random_layouts_keep_requests_whole_on_their_owner():
    rng = np.random.default_rng(0)
    paddings = [16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024, 4096]
    for _ in range(300):
        tp = int(rng.choice([1, 2, 4, 8]))
        num_reqs = int(rng.integers(1, 41))
        owners = rng.integers(0, tp, num_reqs)
        tokens = rng.integers(0, 7, num_reqs)
        tokens[rng.integers(num_reqs)] += 1
        total = int(tokens.sum())
        num_padded = [None, total, total + 5][int(rng.integers(3))]
        reqs_padded = [None, num_reqs + 3][int(rng.integers(2))]
        multiple = int(rng.choice([1, 2]))

        layout = build_owner_token_layout(
            owners,
            tokens,
            tp,
            paddings,
            num_padded_tokens=num_padded,
            num_reqs_padded=reqs_padded,
            segment_multiple=multiple,
        )

        _check_layout(
            layout,
            owners,
            tokens,
            tp,
            paddings,
            num_padded=num_padded,
            reqs_padded=reqs_padded,
            multiple=multiple,
        )


def test_batch_padding_differs_from_permuted_padding():
    # More batch padding tokens than padding rows: they share the rows.
    layout = build_owner_token_layout([0, 1], [3, 1], 2, [8], num_padded_tokens=12)
    _check_layout(layout, [0, 1], [3, 1], 2, [8], num_padded=12)
    np.testing.assert_array_equal(layout.perm, [0, 1, 2, 4, 3, 5, 6, 7])
    np.testing.assert_array_equal(layout.inv[4:], [3, 5, 6, 7, 3, 5, 6, 7])

    # No batch padding tokens: padding rows read token 0.
    layout = build_owner_token_layout([0, 1], [3, 1], 2, [8], num_padded_tokens=4)
    np.testing.assert_array_equal(layout.perm, [0, 1, 2, 0, 3, 0, 0, 0])
    np.testing.assert_array_equal(layout.inv, [0, 1, 2, 4])

    # No padding rows: batch padding tokens read row 0.
    layout = build_owner_token_layout([0, 1], [2, 2], 2, [4], num_padded_tokens=6)
    np.testing.assert_array_equal(layout.perm, [0, 1, 2, 3])
    np.testing.assert_array_equal(layout.inv, [0, 1, 2, 3, 0, 0])


def test_segment_multiple_picks_a_larger_padding():
    layout = build_owner_token_layout([0, 1], [1, 1], 2, [2, 4, 16], segment_multiple=8)

    assert (layout.padded_tokens, layout.segment_tokens) == (16, 8)
    _check_layout(layout, [0, 1], [1, 1], 2, [2, 4, 16], multiple=8)


@pytest.mark.parametrize(
    ("owners", "tokens", "tp", "kwargs", "match"),
    [
        ([0, 1], [1], 2, {}, "one owner per request"),
        ([[0, 1]], [[1, 1]], 2, {}, "one owner per request"),
        ([0], [1], 0, {}, "must be positive"),
        ([0], [1], 2, {"segment_multiple": 0}, "must be positive"),
        ([0, 2], [1, 1], 2, {}, r"Owners must be in \[0, 2\)"),
        ([-1, 0], [1, 1], 2, {}, r"Owners must be in \[0, 2\)"),
        ([0, 1], [1, -1], 2, {}, "non-negative"),
        ([0, 1], [0, 0], 2, {}, "no scheduled tokens"),
        ([], [], 2, {}, "no scheduled tokens"),
        ([0, 1], [1, 1], 2, {"num_reqs_padded": 1}, "num_reqs_padded"),
        ([0, 0], [4, 4], 2, {}, "No token padding holds 16 tokens"),
        ([0, 1], [4, 4], 2, {"num_padded_tokens": 7}, "num_padded_tokens"),
    ],
)
def test_invalid_inputs(owners, tokens, tp, kwargs, match):
    with pytest.raises(ValueError, match=match):
        build_owner_token_layout(owners, tokens, tp, [8, 12], **kwargs)
