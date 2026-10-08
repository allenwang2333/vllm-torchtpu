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


def test_extract_request_owners_and_prepare_kv_shard_layout():
    from types import SimpleNamespace

    import torch
    import vllm  # noqa: F401

    from vllm_torchtpu.core.owner_affine_block_pool import OwnerLayout
    from vllm_torchtpu.runner.owner_token_layout import (
        attach_owner_token_layout,
        extract_request_owners_from_block_table,
        owner_layout_tensors,
        prepare_kv_shard_owner_token_layout,
    )

    owner_layout = OwnerLayout(num_replicated=4, num_owners=4, blocks_per_owner=5)
    # Replicated: [0, 4). Owners 0..3: [4, 9), [9, 14), [14, 19), [19, 24).
    # Table 0 (SWA): only replicated blocks [1, 2, 0]
    swa_tbl = SimpleNamespace(
        get_cpu_tensor=lambda: torch.tensor(
            [[1, 2, 0], [3, 0, 0], [2, 1, 0], [0, 0, 0]], dtype=torch.int32
        )
    )
    # Table 1 (Main sharded): req 0 -> 2, req 1 -> 0, req 2 -> 3, req 3 -> default
    main_tbl = np.array([[1, 15, 16], [5, 6, 0], [2, 21, 9], [1, 2, 0]], dtype=np.int32)

    owners = extract_request_owners_from_block_table(
        [swa_tbl, main_tbl],
        [2, 2, 2, 1],
        owner_layout,
        default_owner=1,
    )
    np.testing.assert_array_equal(owners, [2, 0, 3, 1])

    # 2-arg form (block_table, owner_layout) with metadata dict & start_index/num_reqs
    meta_swa = SimpleNamespace(
        block_tables=torch.tensor([[1, 2, 0], [3, 0, 0], [2, 1, 0]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
    )
    meta_main = SimpleNamespace(
        block_tables=torch.tensor(
            [[1, 15, 16], [5, 6, 0], [2, 21, 9]], dtype=torch.int32
        ),
        query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
    )
    meta_dict = {"swa": meta_swa, "main": meta_main}
    owners_slice = extract_request_owners_from_block_table(
        meta_dict,
        owner_layout,
        num_reqs=2,
        num_blocks_per_req=[2, 2],
        start_index=1,
    )
    np.testing.assert_array_equal(owners_slice, [0, 3])

    # Single metadata object with .block_tables and prepare_kv_shard_owner_token_layout
    layout = prepare_kv_shard_owner_token_layout(
        meta_dict,
        owner_layout,
        torch.tensor([1, 2, 1], dtype=torch.int32),
        [8, 16],
        num_padded_tokens=6,
    )
    assert meta_swa.owner_token_layout is layout
    assert meta_main.owner_token_layout is layout
    assert meta_main.owner_perm.dtype == torch.int32
    assert meta_main.owner_inv.dtype == torch.int32
    assert meta_main.owner_q_loc.shape == (4, 4)
    # Real tokens = 4, num_padded_tokens = 6 -> inv[4:] masked to -1
    assert torch.all(meta_main.owner_inv[4:] == -1)
    # Segment size = 2 across 4 owners: counts are [2, 0, 1, 1]
    np.testing.assert_array_equal(
        meta_main.owner_perm.numpy(), [1, 2, -1, -1, 0, -1, 3, -1]
    )

    # Also test with raw block_table and explicit attn_metadata / None
    single_meta = SimpleNamespace(query_start_loc=None)
    layout2 = prepare_kv_shard_owner_token_layout(
        main_tbl[:3],
        owner_layout,
        [1, 2, 1],
        [8],
        attn_metadata=single_meta,
        device="cpu",
    )
    assert single_meta.owner_token_layout is layout2
    layout3 = prepare_kv_shard_owner_token_layout(
        main_tbl[:3],
        owner_layout,
        [1, 2, 1],
        [8],
    )
    perm_t, inv_t, q_loc_t = owner_layout_tensors(layout3)
    assert perm_t.shape == (8,) and inv_t.shape == (8,) and q_loc_t.shape == (4, 4)
    assert attach_owner_token_layout({}, layout3) == {}
    assert attach_owner_token_layout(single_meta, layout3) is single_meta

    # Error guards in _to_2d_numpy and extract_request_owners_from_block_table
    with pytest.raises(ValueError, match="Expected 2D block table"):
        extract_request_owners_from_block_table(
            np.zeros((4,), dtype=np.int32), owner_layout
        )
    with pytest.raises(ValueError, match="Expected a valid OwnerLayout"):
        extract_request_owners_from_block_table(main_tbl, None)
    with pytest.raises(ValueError, match="Expected at least one block table"):
        extract_request_owners_from_block_table([], owner_layout)
    with pytest.raises(ValueError, match="Invalid request slice"):
        extract_request_owners_from_block_table(main_tbl, owner_layout, num_reqs=5)
    with pytest.raises(ValueError, match="Expected num_blocks_per_req of shape"):
        extract_request_owners_from_block_table(main_tbl, [1, 2], owner_layout)
