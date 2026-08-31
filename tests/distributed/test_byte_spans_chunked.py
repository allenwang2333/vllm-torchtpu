# SPDX-License-Identifier: Apache-2.0
"""Unit tests for chunked sub-window byte span lowering in Stage-3 Raiden."""

from __future__ import annotations

import pytest
from vllm_torchtpu.distributed.kv_transfer.raiden import byte_spans


def test_owned_token_ranges_aligned_strides_pcp4():
    """Verifies token range queries across PCP=4 with interleave 256."""
    parallelism = 4
    interleave_tokens = 256

    # Test window [0, 1024)
    expected_0_1024 = {
        0: [(0, 256)],
        1: [(256, 512)],
        2: [(512, 768)],
        3: [(768, 1024)],
    }
    for rank in range(parallelism):
        ranges = byte_spans.owned_token_ranges(
            token_base=0,
            num_tokens=1024,
            transfer_rank=rank,
            parallelism=parallelism,
            interleave_tokens=interleave_tokens,
        )
        assert ranges == expected_0_1024[rank]
        assert sum(end - start for start, end in ranges) == 256

    # Test window [1024, 2048)
    expected_1024_2048 = {
        0: [(1024, 1280)],
        1: [(1280, 1536)],
        2: [(1536, 1792)],
        3: [(1792, 2048)],
    }
    for rank in range(parallelism):
        ranges = byte_spans.owned_token_ranges(
            token_base=1024,
            num_tokens=1024,
            transfer_rank=rank,
            parallelism=parallelism,
            interleave_tokens=interleave_tokens,
        )
        assert ranges == expected_1024_2048[rank]
        assert sum(end - start for start, end in ranges) == 256

    # Test window [2048, 4096) (2 strides of 1024)
    expected_2048_4096 = {
        0: [(2048, 2304), (3072, 3328)],
        1: [(2304, 2560), (3328, 3584)],
        2: [(2560, 2816), (3584, 3840)],
        3: [(2816, 3072), (3840, 4096)],
    }
    for rank in range(parallelism):
        ranges = byte_spans.owned_token_ranges(
            token_base=2048,
            num_tokens=2048,
            transfer_rank=rank,
            parallelism=parallelism,
            interleave_tokens=interleave_tokens,
        )
        assert ranges == expected_2048_4096[rank]
        assert sum(end - start for start, end in ranges) == 512
        for start, end in ranges:
            assert end - start == 256


def test_owned_token_ranges_aligned_strides_pcp2():
    """Verifies token range queries for PCP=2 with interleave 256."""
    parallelism = 2
    interleave_tokens = 256

    # Test window [512, 1024)
    ranges_rank0 = byte_spans.owned_token_ranges(
        token_base=512,
        num_tokens=512,
        transfer_rank=0,
        parallelism=parallelism,
        interleave_tokens=interleave_tokens,
    )
    ranges_rank1 = byte_spans.owned_token_ranges(
        token_base=512,
        num_tokens=512,
        transfer_rank=1,
        parallelism=parallelism,
        interleave_tokens=interleave_tokens,
    )
    assert ranges_rank0 == [(512, 768)]
    assert ranges_rank1 == [(768, 1024)]


def test_owned_token_ranges_aligned_strides_pcp8():
    """Verifies token range queries for PCP=8 with interleave 128."""
    parallelism = 8
    interleave_tokens = 128
    token_base = 1024
    num_tokens = 1024

    for rank in range(parallelism):
        ranges = byte_spans.owned_token_ranges(
            token_base=token_base,
            num_tokens=num_tokens,
            transfer_rank=rank,
            parallelism=parallelism,
            interleave_tokens=interleave_tokens,
        )
        expected_start = token_base + rank * interleave_tokens
        expected_end = expected_start + interleave_tokens
        assert ranges == [(expected_start, expected_end)]


def test_owned_token_ranges_negative_token_base_raises_value_error():
    """Asserts ValueError is raised when token_base < 0."""
    with pytest.raises(ValueError, match="token_base must be non-negative"):
        byte_spans.owned_token_ranges(
            token_base=-1,
            num_tokens=1024,
            transfer_rank=0,
            parallelism=4,
            interleave_tokens=256,
        )

    with pytest.raises(ValueError, match="token_base must be non-negative"):
        byte_spans.owned_token_ranges(
            token_base=-1024,
            num_tokens=1024,
            transfer_rank=0,
            parallelism=4,
            interleave_tokens=256,
        )


def test_owned_token_ranges_unaligned_cycle_raises_value_error():
    """Asserts ValueError is raised when token_base is not cycle-aligned."""
    parallelism = 4
    interleave_tokens = 256  # cycle = 1024
    unaligned_bases = [1, 100, 256, 512, 768, 1025, 1536]
    for unaligned_base in unaligned_bases:
        with pytest.raises(ValueError, match="must be aligned to PCP cycle"):
            byte_spans.owned_token_ranges(
                token_base=unaligned_base,
                num_tokens=1024,
                transfer_rank=0,
                parallelism=parallelism,
                interleave_tokens=interleave_tokens,
            )


def test_owned_token_ranges_invalid_arguments_raise_value_error():
    """Asserts ValueError is raised for invalid domain parameters."""
    with pytest.raises(ValueError, match="num_tokens must be positive"):
        byte_spans.owned_token_ranges(
            token_base=1024,
            num_tokens=0,
            transfer_rank=0,
            parallelism=4,
            interleave_tokens=256,
        )

    with pytest.raises(ValueError, match="parallelism must be positive"):
        byte_spans.owned_token_ranges(
            token_base=1024,
            num_tokens=1024,
            transfer_rank=0,
            parallelism=0,
            interleave_tokens=256,
        )

    with pytest.raises(ValueError, match="outside parallelism"):
        byte_spans.owned_token_ranges(
            token_base=1024,
            num_tokens=1024,
            transfer_rank=4,
            parallelism=4,
            interleave_tokens=256,
        )

    with pytest.raises(ValueError, match="interleave_tokens must be positive"):
        byte_spans.owned_token_ranges(
            token_base=1024,
            num_tokens=1024,
            transfer_rank=0,
            parallelism=4,
            interleave_tokens=0,
        )


def test_owned_token_ranges_backward_compatibility():
    """Verifies default token_base=0 matches explicit token_base=0."""
    ranges_default = byte_spans.owned_token_ranges(
        num_tokens=1024,
        transfer_rank=1,
        parallelism=4,
        interleave_tokens=256,
    )
    ranges_explicit = byte_spans.owned_token_ranges(
        token_base=0,
        num_tokens=1024,
        transfer_rank=1,
        parallelism=4,
        interleave_tokens=256,
    )
    assert ranges_default == ranges_explicit == [(256, 512)]


def test_owned_token_ranges_parallelism_one():
    """Verifies parallelism=1 returns full token range offset by token_base."""
    ranges = byte_spans.owned_token_ranges(
        token_base=1024,
        num_tokens=2048,
        transfer_rank=0,
        parallelism=1,
        interleave_tokens=256,
    )
    assert ranges == [(1024, 3072)]


def test_lower_fa_spans_destination_offsets():
    """Verifies sub-chunk destination byte offsets match monolithic slice."""
    parallelism = 4
    interleave_tokens = 256
    page_tokens = 1024
    token_bytes = 1024  # 1 KB per token

    for rank in range(parallelism):
        mono_reg = byte_spans.lower_fa_spans(
            token_base=0,
            num_tokens=2048,
            transfer_rank=rank,
            parallelism=parallelism,
            interleave_tokens=interleave_tokens,
            page_tokens=page_tokens,
            token_bytes=token_bytes,
            block_ids=[100],
        )

        chunk0_reg = byte_spans.lower_fa_spans(
            token_base=0,
            num_tokens=1024,
            transfer_rank=rank,
            parallelism=parallelism,
            interleave_tokens=interleave_tokens,
            page_tokens=page_tokens,
            token_bytes=token_bytes,
            block_ids=[200],
        )

        chunk1_reg = byte_spans.lower_fa_spans(
            token_base=1024,
            num_tokens=1024,
            transfer_rank=rank,
            parallelism=parallelism,
            interleave_tokens=interleave_tokens,
            page_tokens=page_tokens,
            token_bytes=token_bytes,
            block_ids=[300],
        )

        assert len(mono_reg.spans) == 2
        assert len(chunk0_reg.spans) == 1
        assert len(chunk1_reg.spans) == 1

        # Chunk 0 span matches monolithic span 0 dst_offset_bytes & size_bytes.
        assert (
            chunk0_reg.spans[0].dst_offset_bytes
            == mono_reg.spans[0].dst_offset_bytes
        )
        assert chunk0_reg.spans[0].size_bytes == mono_reg.spans[0].size_bytes
        assert chunk0_reg.spans[0].src_block_ordinal == 0
        assert chunk0_reg.spans[0].src_offset_bytes == 0
        assert chunk0_reg.spans[0].dst_block_index == 0
        assert chunk0_reg.spans[0].size_bytes == 256 * token_bytes
        assert chunk0_reg.declared_bytes == 256 * token_bytes
        assert chunk0_reg.dst_space_version == 1
        assert chunk0_reg.tag == "fa"
        assert chunk0_reg.block_ids == (200,)

        # Chunk 1 span matches monolithic span 1 dst_offset_bytes & size_bytes.
        assert (
            chunk1_reg.spans[0].dst_offset_bytes
            == mono_reg.spans[1].dst_offset_bytes
        )
        assert chunk1_reg.spans[0].size_bytes == mono_reg.spans[1].size_bytes
        assert chunk1_reg.spans[0].src_block_ordinal == 0
        assert chunk1_reg.spans[0].src_offset_bytes == 0
        assert chunk1_reg.spans[0].dst_block_index == 0
        expected_dst_offset = (1024 + rank * 256) * token_bytes
        assert chunk1_reg.spans[0].dst_offset_bytes == expected_dst_offset
        assert chunk1_reg.spans[0].size_bytes == 256 * token_bytes
        assert chunk1_reg.declared_bytes == 256 * token_bytes
        assert chunk1_reg.dst_space_version == 1
        assert chunk1_reg.tag == "fa"
        assert chunk1_reg.block_ids == (300,)


def test_lower_fa_spans_backward_compatibility():
    """Verifies lower_fa_spans without token_base maintains default behavior."""
    reg_default = byte_spans.lower_fa_spans(
        num_tokens=1024,
        transfer_rank=2,
        parallelism=4,
        interleave_tokens=256,
        page_tokens=1024,
        token_bytes=512,
        block_ids=[42],
    )
    reg_explicit = byte_spans.lower_fa_spans(
        token_base=0,
        num_tokens=1024,
        transfer_rank=2,
        parallelism=4,
        interleave_tokens=256,
        page_tokens=1024,
        token_bytes=512,
        block_ids=[42],
    )
    assert reg_default == reg_explicit
    assert reg_default.tag == "fa"
    assert reg_default.block_ids == (42,)
    assert reg_default.spans == (
        byte_spans.PoolByteSpan(
            src_block_ordinal=0,
            src_offset_bytes=0,
            dst_block_index=0,
            dst_offset_bytes=512 * 512,
            size_bytes=256 * 512,
        ),
    )
    assert reg_default.declared_bytes == 256 * 512
    assert reg_default.dst_space_version == 1


def test_lower_fa_spans_parallelism_one():
    """Verifies lower_fa_spans with parallelism=1 and token_base > 0."""
    reg = byte_spans.lower_fa_spans(
        token_base=2048,
        num_tokens=2048,
        transfer_rank=0,
        parallelism=1,
        interleave_tokens=256,
        page_tokens=1024,
        token_bytes=512,
        block_ids=[10, 11],
    )
    assert reg.tag == "fa"
    assert reg.block_ids == (10, 11)
    assert reg.declared_bytes == 2048 * 512
    assert reg.dst_space_version == 1
    assert reg.spans == (
        byte_spans.PoolByteSpan(
            src_block_ordinal=0,
            src_offset_bytes=0,
            dst_block_index=0,
            dst_offset_bytes=2048 * 512,
            size_bytes=1024 * 512,
        ),
        byte_spans.PoolByteSpan(
            src_block_ordinal=1,
            src_offset_bytes=0,
            dst_block_index=0,
            dst_offset_bytes=3072 * 512,
            size_bytes=1024 * 512,
        ),
    )


def test_lower_fa_spans_multi_page_pagination():
    """Verifies lower_fa_spans splits spans across multiple source pages.

    4096 tokens starting at token_base=1024 on PCP=4 with interleave 256.
    Each rank owns 4 * 256 = 1024 tokens.
    With page_tokens=512, each rank allocates 2 pages (block_ids=[101, 102]).
    """
    token_bytes = 1024
    reg = byte_spans.lower_fa_spans(
        token_base=1024,
        num_tokens=4096,
        transfer_rank=0,
        parallelism=4,
        interleave_tokens=256,
        page_tokens=512,
        token_bytes=token_bytes,
        block_ids=[101, 102],
    )
    assert reg.tag == "fa"
    assert reg.block_ids == (101, 102)
    assert reg.declared_bytes == 1024 * token_bytes
    assert reg.dst_space_version == 1
    assert len(reg.spans) == 4

    # Rank 0 owned ranges:
    # [(1024, 1280), (2048, 2304), (3072, 3328), (4096, 4352)]
    # First 512 owned tokens map to source page 0 (block ordinal 0)
    # Next 512 owned tokens map to source page 1 (block ordinal 1)
    assert reg.spans == (
        byte_spans.PoolByteSpan(
            src_block_ordinal=0,
            src_offset_bytes=0,
            dst_block_index=0,
            dst_offset_bytes=1024 * token_bytes,
            size_bytes=256 * token_bytes,
        ),
        byte_spans.PoolByteSpan(
            src_block_ordinal=0,
            src_offset_bytes=256 * token_bytes,
            dst_block_index=0,
            dst_offset_bytes=2048 * token_bytes,
            size_bytes=256 * token_bytes,
        ),
        byte_spans.PoolByteSpan(
            src_block_ordinal=1,
            src_offset_bytes=0,
            dst_block_index=0,
            dst_offset_bytes=3072 * token_bytes,
            size_bytes=256 * token_bytes,
        ),
        byte_spans.PoolByteSpan(
            src_block_ordinal=1,
            src_offset_bytes=256 * token_bytes,
            dst_block_index=0,
            dst_offset_bytes=4096 * token_bytes,
            size_bytes=256 * token_bytes,
        ),
    )


def test_lower_fa_spans_page_boundary_crossing_within_stride():
    """Verifies lower_fa_spans splits a stride across source pages.

    1024 tokens at token_base=1024 on PCP=2 with interleave 512.
    Each rank owns 512 tokens.
    With page_tokens=256, each 512-token stride crosses 2 source pages.
    """
    token_bytes = 512
    reg = byte_spans.lower_fa_spans(
        token_base=1024,
        num_tokens=1024,
        transfer_rank=0,
        parallelism=2,
        interleave_tokens=512,
        page_tokens=256,
        token_bytes=token_bytes,
        block_ids=[201, 202],
    )
    assert reg.tag == "fa"
    assert reg.block_ids == (201, 202)
    assert reg.declared_bytes == 512 * token_bytes
    assert reg.dst_space_version == 1
    assert len(reg.spans) == 2
    assert reg.spans == (
        byte_spans.PoolByteSpan(
            src_block_ordinal=0,
            src_offset_bytes=0,
            dst_block_index=0,
            dst_offset_bytes=1024 * token_bytes,
            size_bytes=256 * token_bytes,
        ),
        byte_spans.PoolByteSpan(
            src_block_ordinal=1,
            src_offset_bytes=0,
            dst_block_index=0,
            dst_offset_bytes=(1024 + 256) * token_bytes,
            size_bytes=256 * token_bytes,
        ),
    )


def test_lower_fa_spans_validation_failures():
    """Verifies lower_fa_spans input validation and error rejection."""
    with pytest.raises(ValueError, match="page_tokens must be positive"):
        byte_spans.lower_fa_spans(
            token_base=1024,
            num_tokens=1024,
            transfer_rank=0,
            parallelism=4,
            interleave_tokens=256,
            page_tokens=0,
            token_bytes=512,
            block_ids=[1],
        )

    with pytest.raises(ValueError, match="token_bytes must be positive"):
        byte_spans.lower_fa_spans(
            token_base=1024,
            num_tokens=1024,
            transfer_rank=0,
            parallelism=4,
            interleave_tokens=256,
            page_tokens=1024,
            token_bytes=0,
            block_ids=[1],
        )

    with pytest.raises(ValueError, match="token_base must be non-negative"):
        byte_spans.lower_fa_spans(
            token_base=-1024,
            num_tokens=1024,
            transfer_rank=0,
            parallelism=4,
            interleave_tokens=256,
            page_tokens=1024,
            token_bytes=512,
            block_ids=[1],
        )

    with pytest.raises(ValueError, match="must be aligned to PCP cycle"):
        byte_spans.lower_fa_spans(
            token_base=512,
            num_tokens=1024,
            transfer_rank=0,
            parallelism=4,
            interleave_tokens=256,
            page_tokens=1024,
            token_bytes=512,
            block_ids=[1],
        )

    # Test block_ids count mismatch: expected 1 block, passed 0 or 2 blocks
    with pytest.raises(
        ValueError, match="block_ids do not match dense PCP rank packing"
    ):
        byte_spans.lower_fa_spans(
            token_base=1024,
            num_tokens=1024,
            transfer_rank=0,
            parallelism=4,
            interleave_tokens=256,
            page_tokens=1024,
            token_bytes=512,
            block_ids=[],
        )

    with pytest.raises(
        ValueError, match="block_ids do not match dense PCP rank packing"
    ):
        byte_spans.lower_fa_spans(
            token_base=1024,
            num_tokens=1024,
            transfer_rank=0,
            parallelism=4,
            interleave_tokens=256,
            page_tokens=1024,
            token_bytes=512,
            block_ids=[1, 2],
        )
