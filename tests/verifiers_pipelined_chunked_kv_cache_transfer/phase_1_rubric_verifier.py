# SPDX-License-Identifier: Apache-2.0
"""Phase 1 Rubric Verifier — Span Range Algebra & PCP Interleave Geometry.

Validates all Phase 1 Rubric items for chunked KV cache transfer:
- R1.1 — Span Additivity Invariant: For any prompt length N and any legal
  stride boundary K, the union of byte span descriptors for [0, K) and [K, N)
  is identical in byte offsets, counts, and strides to the monolithic span
  descriptor for [0, N) across all context parallel ranks.
- R1.2 — Interleave Stride Alignment Enforcement: Range queries starting at
  token offsets that do not align with the context parallel interleave stride
  are rejected with an explicit validation error.
- R1.3 — Legacy Backward Compatibility: Calling span generation functions
  without specifying a base offset defaults to token index zero, maintaining
  full compatibility with existing single-shot registrations.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

import pytest
from vllm_torchtpu.distributed.kv_transfer.raiden import byte_spans


def _token_payload(token_id: int, token_bytes: int) -> bytes:
    """Generates a deterministic unique byte pattern for a given token.

    Encodes the token identifier as a big-endian 4-byte integer and tiles it
    across the requested byte width, creating a distinct payload for verifying
    memory copy fidelity.

    Args:
        token_id: Integer index identifying the token within the global prompt.
        token_bytes: Total number of bytes per token payload.

    Returns:
        A bytes object of length token_bytes populated with tiled token data.
    """
    pattern = token_id.to_bytes(4, byteorder="big", signed=False)
    repeat_count = token_bytes // 4
    remainder = token_bytes % 4
    return (pattern * repeat_count) + pattern[:remainder]


def _execute_span_copy(
    *,
    registration: byte_spans.PoolSpanRegistration,
    source_blocks: Sequence[bytearray],
    dst_buffer: bytearray,
) -> None:
    """Copies bytes from source blocks into dst_buffer per span descriptors.

    Iterates over every PoolByteSpan descriptor in the registration, extracting
    the source memory slice from the designated source block ordinal and writing
    it directly into the destination buffer at the target byte offset.

    Args:
        registration: PoolSpanRegistration holding copy span descriptors.
        source_blocks: Sequence of allocated source memory bytearrays.
        dst_buffer: Mutable destination bytearray receiving transferred bytes.
    """
    for span in registration.spans:
        src = source_blocks[span.src_block_ordinal]
        src_start = span.src_offset_bytes
        dst_start = span.dst_offset_bytes
        size = span.size_bytes
        dst_buffer[dst_start : dst_start + size] = src[
            src_start : src_start + size
        ]


def _build_populated_source_blocks(
    *,
    owned_ranges: Sequence[tuple[int, int]],
    page_tokens: int,
    token_bytes: int,
    num_blocks: int,
) -> list[bytearray]:
    """Allocates and populates source blocks with deterministic token data.

    Creates a list of zeroed bytearray source blocks and populates them with
    deterministic token payload bytes for each token index across the provided
    PCP-owned half-open token ranges.

    Args:
        owned_ranges: Sequence of (start_token, end_token) half-open index
          pairs.
        page_tokens: Capacity of each source page block measured in tokens.
        token_bytes: Byte size allocated for a single token's KV cache.
        num_blocks: Total number of source blocks to instantiate.

    Returns:
        A list of populated bytearray objects representing source pages.
    """
    block_size = page_tokens * token_bytes
    blocks = [bytearray(block_size) for _ in range(num_blocks)]
    local_token_idx = 0
    for start, end in owned_ranges:
        for token_id in range(start, end):
            block_idx = local_token_idx // page_tokens
            offset = (local_token_idx % page_tokens) * token_bytes
            payload = _token_payload(token_id, token_bytes)
            blocks[block_idx][offset : offset + token_bytes] = payload
            local_token_idx += 1
    return blocks


def _populate_and_execute_chunk_copies(
    *,
    chunk_sizes: Sequence[int],
    chunk_regs: Sequence[byte_spans.PoolSpanRegistration],
    transfer_rank: int,
    parallelism: int,
    interleave_tokens: int,
    page_tokens: int,
    token_bytes: int,
    dst_buffer: bytearray,
) -> int:
    """Simulates chunked memory copy execution for a specific rank.

    Computes owned token ranges for each sub-chunk in order, allocates and
    populates source blocks, executes span copies into destination memory, and
    accumulates declared byte totals.

    Args:
        chunk_sizes: Sequence of token chunk lengths.
        chunk_regs: Sequence of lowered PoolSpanRegistrations for each chunk.
        transfer_rank: Zero-based PCP rank index.
        parallelism: Context parallelism degree (PCP size).
        interleave_tokens: Token count per rank stride.
        page_tokens: Tokens stored per physical KV page block.
        token_bytes: Bytes per token KV representation.
        dst_buffer: Mutable destination buffer receiving copy bytes.

    Returns:
        The total number of declared bytes copied across all sub-chunks.
    """
    current_base = 0
    total_declared_bytes = 0
    for chunk_idx, chunk_size in enumerate(chunk_sizes):
        chunk_ranges = byte_spans.owned_token_ranges(
            token_base=current_base,
            num_tokens=chunk_size,
            transfer_rank=transfer_rank,
            parallelism=parallelism,
            interleave_tokens=interleave_tokens,
        )
        chunk_owned_tokens = sum(end - start for start, end in chunk_ranges)
        chunk_num_blocks = (
            chunk_owned_tokens + page_tokens - 1
        ) // page_tokens
        chunk_blocks = _build_populated_source_blocks(
            owned_ranges=chunk_ranges,
            page_tokens=page_tokens,
            token_bytes=token_bytes,
            num_blocks=chunk_num_blocks,
        )
        _execute_span_copy(
            registration=chunk_regs[chunk_idx],
            source_blocks=chunk_blocks,
            dst_buffer=dst_buffer,
        )
        total_declared_bytes += chunk_regs[chunk_idx].declared_bytes
        current_base += chunk_size
    return total_declared_bytes


def _verify_span_decomposition_for_rank(
    *,
    chunk_sizes: Sequence[int],
    transfer_rank: int,
    parallelism: int,
    interleave_tokens: int,
    page_tokens: int,
    token_bytes: int,
) -> tuple[
    byte_spans.PoolSpanRegistration, list[byte_spans.PoolSpanRegistration]
]:
    """Verifies that sub-chunk span lowering decomposes losslessly for a rank.

    Compares the monolithic span registration for the full prompt against the
    concatenation of sub-chunk span registrations. Validates declared byte
    conservation, destination offset alignment, span descriptor equality, and
    bit-for-bit memory copy fidelity.

    Args:
        chunk_sizes: Sequence of token chunk lengths forming the full prompt.
        transfer_rank: Zero-based Context Parallel rank.
        parallelism: Context Parallelism degree (PCP size).
        interleave_tokens: Number of tokens per rank stride.
        page_tokens: Tokens stored per physical KV page block.
        token_bytes: Byte size per token representation.

    Returns:
        A tuple of (monolithic_registration, list_of_chunk_registrations).
    """
    total_tokens = sum(chunk_sizes)
    cycle = parallelism * interleave_tokens
    for size in chunk_sizes:
        assert (
            size % cycle == 0
        ), f"Chunk size {size} must align to cycle {cycle}"

    # 1. Monolithic Lowering
    mono_ranges = byte_spans.owned_token_ranges(
        token_base=0,
        num_tokens=total_tokens,
        transfer_rank=transfer_rank,
        parallelism=parallelism,
        interleave_tokens=interleave_tokens,
    )
    mono_owned_tokens = sum(end - start for start, end in mono_ranges)
    mono_num_blocks = (mono_owned_tokens + page_tokens - 1) // page_tokens
    mono_block_ids = [1000 + block_idx for block_idx in range(mono_num_blocks)]

    mono_reg = byte_spans.lower_fa_spans(
        token_base=0,
        num_tokens=total_tokens,
        transfer_rank=transfer_rank,
        parallelism=parallelism,
        interleave_tokens=interleave_tokens,
        page_tokens=page_tokens,
        token_bytes=token_bytes,
        block_ids=mono_block_ids,
    )

    # 2. Chunked Lowering
    chunk_regs: list[byte_spans.PoolSpanRegistration] = []
    current_base = 0
    for chunk_idx, chunk_size in enumerate(chunk_sizes):
        chunk_ranges = byte_spans.owned_token_ranges(
            token_base=current_base,
            num_tokens=chunk_size,
            transfer_rank=transfer_rank,
            parallelism=parallelism,
            interleave_tokens=interleave_tokens,
        )
        chunk_owned_tokens = sum(end - start for start, end in chunk_ranges)
        chunk_num_blocks = (
            chunk_owned_tokens + page_tokens - 1
        ) // page_tokens
        chunk_block_ids = [
            2000 + chunk_idx * 100 + block_idx
            for block_idx in range(chunk_num_blocks)
        ]

        chunk_reg = byte_spans.lower_fa_spans(
            token_base=current_base,
            num_tokens=chunk_size,
            transfer_rank=transfer_rank,
            parallelism=parallelism,
            interleave_tokens=interleave_tokens,
            page_tokens=page_tokens,
            token_bytes=token_bytes,
            block_ids=chunk_block_ids,
        )
        chunk_regs.append(chunk_reg)
        current_base += chunk_size

    # 3. Structural Equivalence Assertions
    # Declared bytes additivity
    total_chunk_declared = sum(
        chunk_reg.declared_bytes for chunk_reg in chunk_regs
    )
    assert total_chunk_declared == mono_reg.declared_bytes, (
        f"Total chunk declared bytes ({total_chunk_declared}) != "
        f"monolithic declared bytes ({mono_reg.declared_bytes})"
    )
    assert total_chunk_declared == mono_owned_tokens * token_bytes

    # Destination spans equality
    all_chunk_spans = [
        span for chunk_reg in chunk_regs for span in chunk_reg.spans
    ]
    assert len(all_chunk_spans) == len(mono_reg.spans), (
        f"Concatenated chunk spans count ({len(all_chunk_spans)}) != "
        f"monolithic spans count ({len(mono_reg.spans)})"
    )

    for chunk_span, mono_span in zip(all_chunk_spans, mono_reg.spans):
        assert chunk_span.dst_block_index == mono_span.dst_block_index == 0
        assert chunk_span.dst_offset_bytes == mono_span.dst_offset_bytes, (
            f"dst_offset_bytes mismatch: chunk={chunk_span.dst_offset_bytes}, "
            f"mono={mono_span.dst_offset_bytes}"
        )
        assert chunk_span.size_bytes == mono_span.size_bytes, (
            f"size_bytes mismatch: chunk={chunk_span.size_bytes}, "
            f"mono={mono_span.size_bytes}"
        )
        assert chunk_span.src_stride_bytes == mono_span.src_stride_bytes == 0
        assert chunk_span.dst_stride_bytes == mono_span.dst_stride_bytes == 0
        assert chunk_span.count == mono_span.count == 1

    # Registration metadata invariant
    assert mono_reg.tag == "fa"
    assert mono_reg.dst_space_version == 1
    for chunk_reg in chunk_regs:
        assert chunk_reg.tag == "fa"
        assert chunk_reg.dst_space_version == 1

    # 4. Bit-for-bit Memory Copy Simulation
    mono_source_blocks = _build_populated_source_blocks(
        owned_ranges=mono_ranges,
        page_tokens=page_tokens,
        token_bytes=token_bytes,
        num_blocks=mono_num_blocks,
    )
    dst_mono = bytearray(total_tokens * token_bytes)
    _execute_span_copy(
        registration=mono_reg,
        source_blocks=mono_source_blocks,
        dst_buffer=dst_mono,
    )

    dst_chunked = bytearray(total_tokens * token_bytes)
    _populate_and_execute_chunk_copies(
        chunk_sizes=chunk_sizes,
        chunk_regs=chunk_regs,
        transfer_rank=transfer_rank,
        parallelism=parallelism,
        interleave_tokens=interleave_tokens,
        page_tokens=page_tokens,
        token_bytes=token_bytes,
        dst_buffer=dst_chunked,
    )

    assert dst_chunked == dst_mono, "Bit-for-bit memory copy mismatch on rank!"

    return mono_reg, chunk_regs


def _verify_full_collective_decomposition(
    *,
    chunk_sizes: Sequence[int],
    parallelism: int,
    interleave_tokens: int,
    page_tokens: int,
    token_bytes: int,
) -> None:
    """Verifies span decomposition across all ranks and global coverage.

    Simulates parallel copy execution across all ranks in the Context Parallel
    group into a global destination buffer, and asserts that the reconstructed
    buffer matches the expected prompt token payload bit-for-bit.

    Args:
        chunk_sizes: Sequence of token chunk lengths forming the full prompt.
        parallelism: Context Parallelism degree (PCP size).
        interleave_tokens: Number of tokens per rank stride.
        page_tokens: Tokens stored per physical KV page block.
        token_bytes: Byte size per token representation.
    """
    total_tokens = sum(chunk_sizes)
    global_dest_chunked = bytearray(total_tokens * token_bytes)
    global_dest_expected = bytearray(total_tokens * token_bytes)

    # Populate expected global buffer
    for token_id in range(total_tokens):
        payload = _token_payload(token_id, token_bytes)
        offset = token_id * token_bytes
        global_dest_expected[offset : offset + token_bytes] = payload

    total_collective_bytes = 0
    for rank in range(parallelism):
        _, chunk_regs = _verify_span_decomposition_for_rank(
            chunk_sizes=chunk_sizes,
            transfer_rank=rank,
            parallelism=parallelism,
            interleave_tokens=interleave_tokens,
            page_tokens=page_tokens,
            token_bytes=token_bytes,
        )
        total_collective_bytes += _populate_and_execute_chunk_copies(
            chunk_sizes=chunk_sizes,
            chunk_regs=chunk_regs,
            transfer_rank=rank,
            parallelism=parallelism,
            interleave_tokens=interleave_tokens,
            page_tokens=page_tokens,
            token_bytes=token_bytes,
            dst_buffer=global_dest_chunked,
        )

    assert (
        total_collective_bytes == total_tokens * token_bytes
    ), (
        f"Total collective bytes ({total_collective_bytes}) != "
        f"{total_tokens * token_bytes}"
    )
    assert (
        global_dest_chunked == global_dest_expected
    ), "Global multi-rank reconstructed buffer does not match expected prompt!"


# ==============================================================================
# Rubric Test 1: R1.1 — Span Additivity Invariant
# ==============================================================================


@pytest.mark.parametrize(
    (
        "parallelism,interleave_tokens,page_tokens,token_bytes,"
        "prompt_tokens,split_k"
    ),
    [
        # PCP=2 binary splits
        (2, 256, 256, 512, 2048, 1024),
        (2, 256, 512, 512, 4096, 2048),
        (2, 512, 512, 256, 4096, 1024),
        (2, 512, 512, 256, 8192, 3072),
        # PCP=4 binary splits
        (4, 256, 256, 512, 4096, 1024),
        (4, 256, 512, 512, 4096, 2048),
        (4, 256, 512, 512, 8192, 3072),
        (4, 256, 1024, 1024, 16384, 8192),
        # PCP=8 binary splits
        (8, 256, 512, 512, 4096, 2048),
        (8, 256, 512, 512, 8192, 4096),
        (8, 256, 1024, 512, 16384, 6144),
        (8, 128, 512, 1024, 8192, 2048),
        # PCP=1 single rank
        (1, 256, 512, 512, 4096, 1024),
    ],
)
def test_rubric_r1_1_binary_split_additivity(
    parallelism: int,
    interleave_tokens: int,
    page_tokens: int,
    token_bytes: int,
    prompt_tokens: int,
    split_k: int,
) -> None:
    """Verifies R1.1: [0, K) U [K, N) == [0, N) across all PCP ranks."""
    chunk_sizes = [split_k, prompt_tokens - split_k]
    _verify_full_collective_decomposition(
        chunk_sizes=chunk_sizes,
        parallelism=parallelism,
        interleave_tokens=interleave_tokens,
        page_tokens=page_tokens,
        token_bytes=token_bytes,
    )


def test_rubric_r1_1_heterogeneous_multi_chunk_additivity() -> None:
    """Verifies R1.1: Multi-chunk partitions match monolithic references."""
    test_cases = [
        (4, 256, 512, 1024, [1024, 2048, 3072, 4096]),
        (4, 256, 1024, 512, [2048, 1024, 2048, 3072, 1024]),
        (4, 256, 256, 512, [3072, 1024, 2048, 4096, 2048]),
        (4, 256, 512, 256, [1024, 1024, 1024, 2048, 1024]),
        (8, 256, 512, 512, [2048, 4096, 2048, 6144]),
        (8, 256, 1024, 1024, [4096, 2048, 4096, 2048]),
        (8, 128, 512, 512, [1024, 2048, 3072, 1024]),
        (2, 512, 512, 512, [1024, 2048, 1024, 3072]),
    ]
    for (
        parallelism,
        interleave_tokens,
        page_tokens,
        token_bytes,
        chunk_sizes,
    ) in test_cases:
        _verify_full_collective_decomposition(
            chunk_sizes=chunk_sizes,
            parallelism=parallelism,
            interleave_tokens=interleave_tokens,
            page_tokens=page_tokens,
            token_bytes=token_bytes,
        )


# ==============================================================================
# Rubric Test 2: R1.2 — Interleave Stride Alignment Enforcement
# ==============================================================================


def test_rubric_r1_2_interleave_stride_alignment_enforcement() -> None:
    """Verifies R1.2: Unaligned token offsets are rejected with ValueError."""
    test_configs = [
        # (parallelism, interleave_tokens, unaligned_bases)
        (2, 256, [1, 64, 128, 256, 511, 513, 768]),
        (4, 256, [1, 128, 256, 512, 768, 1023, 1025, 1280, 1536]),
        (8, 256, [1, 256, 512, 1024, 1536, 2047, 2049, 3072]),
        (8, 128, [1, 128, 256, 512, 768, 1023, 1025, 1536]),
    ]

    for parallelism, interleave_tokens, unaligned_bases in test_configs:
        cycle = parallelism * interleave_tokens
        for token_base in unaligned_bases:
            assert token_base % cycle != 0

            # 1. Test owned_token_ranges rejects unaligned token_base
            with pytest.raises(
                ValueError, match=r"must be aligned to PCP cycle"
            ):
                byte_spans.owned_token_ranges(
                    token_base=token_base,
                    num_tokens=cycle,
                    transfer_rank=0,
                    parallelism=parallelism,
                    interleave_tokens=interleave_tokens,
                )

            # 2. Test lower_fa_spans rejects unaligned token_base
            with pytest.raises(
                ValueError, match=r"must be aligned to PCP cycle"
            ):
                byte_spans.lower_fa_spans(
                    token_base=token_base,
                    num_tokens=cycle,
                    transfer_rank=0,
                    parallelism=parallelism,
                    interleave_tokens=interleave_tokens,
                    page_tokens=512,
                    token_bytes=512,
                    block_ids=[100],
                )

    # 3. Test negative token_base rejection
    with pytest.raises(ValueError, match=r"token_base must be non-negative"):
        byte_spans.owned_token_ranges(
            token_base=-1024,
            num_tokens=1024,
            transfer_rank=0,
            parallelism=4,
            interleave_tokens=256,
        )

    with pytest.raises(ValueError, match=r"token_base must be non-negative"):
        byte_spans.lower_fa_spans(
            token_base=-1024,
            num_tokens=1024,
            transfer_rank=0,
            parallelism=4,
            interleave_tokens=256,
            page_tokens=512,
            token_bytes=512,
            block_ids=[100],
        )

    # 4. Test invalid input parameter rejection
    with pytest.raises(ValueError, match=r"page_tokens must be positive"):
        byte_spans.lower_fa_spans(
            token_base=1024,
            num_tokens=1024,
            transfer_rank=0,
            parallelism=4,
            interleave_tokens=256,
            page_tokens=0,
            token_bytes=512,
            block_ids=[100],
        )

    with pytest.raises(ValueError, match=r"token_bytes must be positive"):
        byte_spans.lower_fa_spans(
            token_base=1024,
            num_tokens=1024,
            transfer_rank=0,
            parallelism=4,
            interleave_tokens=256,
            page_tokens=512,
            token_bytes=0,
            block_ids=[100],
        )

    with pytest.raises(
        ValueError, match=r"block_ids do not match dense PCP rank packing"
    ):
        byte_spans.lower_fa_spans(
            token_base=1024,
            num_tokens=1024,
            transfer_rank=0,
            parallelism=4,
            interleave_tokens=256,
            page_tokens=512,
            token_bytes=512,
            block_ids=[],  # Expected 1 block, passed 0
        )


# ==============================================================================
# Rubric Test 3: R1.3 — Legacy Backward Compatibility
# ==============================================================================


def test_rubric_r1_3_legacy_backward_compatibility() -> None:
    """Verifies R1.3: Omitting token_base preserves legacy behavior."""
    test_cases = [
        # (parallelism, interleave_tokens, page_tokens, token_bytes, num_tokens)
        (1, 256, 512, 512, 2048),
        (2, 256, 512, 512, 2048),
        (4, 256, 512, 512, 4096),
        (4, 256, 256, 1024, 4096),
        (8, 256, 512, 512, 8192),
        (8, 128, 512, 512, 4096),
    ]

    for (
        parallelism,
        interleave_tokens,
        page_tokens,
        token_bytes,
        num_tokens,
    ) in test_cases:
        for rank in range(parallelism):
            # 1. owned_token_ranges without token_base kwarg
            ranges_legacy = byte_spans.owned_token_ranges(
                num_tokens=num_tokens,
                transfer_rank=rank,
                parallelism=parallelism,
                interleave_tokens=interleave_tokens,
            )
            ranges_explicit = byte_spans.owned_token_ranges(
                token_base=0,
                num_tokens=num_tokens,
                transfer_rank=rank,
                parallelism=parallelism,
                interleave_tokens=interleave_tokens,
            )
            assert (
                ranges_legacy == ranges_explicit
            ), f"owned_token_ranges legacy mismatch on rank {rank}"

            # 2. lower_fa_spans without token_base kwarg
            owned_tokens = sum(end - start for start, end in ranges_legacy)
            num_blocks = (owned_tokens + page_tokens - 1) // page_tokens
            block_ids = [5000 + block_idx for block_idx in range(num_blocks)]

            reg_legacy = byte_spans.lower_fa_spans(
                num_tokens=num_tokens,
                transfer_rank=rank,
                parallelism=parallelism,
                interleave_tokens=interleave_tokens,
                page_tokens=page_tokens,
                token_bytes=token_bytes,
                block_ids=block_ids,
            )
            reg_explicit = byte_spans.lower_fa_spans(
                token_base=0,
                num_tokens=num_tokens,
                transfer_rank=rank,
                parallelism=parallelism,
                interleave_tokens=interleave_tokens,
                page_tokens=page_tokens,
                token_bytes=token_bytes,
                block_ids=block_ids,
            )

            assert reg_legacy.tag == reg_explicit.tag == "fa"
            assert reg_legacy.block_ids == reg_explicit.block_ids
            assert reg_legacy.declared_bytes == reg_explicit.declared_bytes
            assert (
                reg_legacy.dst_space_version
                == reg_explicit.dst_space_version
                == 1
            )
            assert len(reg_legacy.spans) == len(reg_explicit.spans)
            for span_legacy, span_explicit in zip(
                reg_legacy.spans, reg_explicit.spans
            ):
                assert span_legacy == span_explicit


def run_all_rubric_checks() -> bool:
    """Executes all Phase 1 Rubric checks and prints formatted results.

    Runs comprehensive verifications for R1.1 (Span Additivity Invariant),
    R1.2 (Interleave Stride Alignment Enforcement), and R1.3 (Legacy Backward
    Compatibility), reporting pass/fail outcomes for each item.

    Returns:
        True if all rubric verification checks passed successfully, False
        otherwise.
    """
    banner = "=" * 60
    print(f"\n{banner}")
    print("PHASE 1 RUBRIC VERIFICATION SUITE — vLLM-TorchTPU Raiden Stage-3")
    print(banner)

    all_passed = True

    # R1.1
    print("\n--- Checking R1.1: Span Additivity Invariant ---")
    try:
        # Run binary split test cases
        binary_cases = [
            (2, 256, 256, 512, 2048, 1024),
            (2, 256, 512, 512, 4096, 2048),
            (2, 512, 512, 256, 4096, 1024),
            (2, 512, 512, 256, 8192, 3072),
            (4, 256, 256, 512, 4096, 1024),
            (4, 256, 512, 512, 4096, 2048),
            (4, 256, 512, 512, 8192, 3072),
            (4, 256, 1024, 1024, 16384, 8192),
            (8, 256, 512, 512, 4096, 2048),
            (8, 256, 512, 512, 8192, 4096),
            (8, 256, 1024, 512, 16384, 6144),
            (8, 128, 512, 1024, 8192, 2048),
            (1, 256, 512, 512, 4096, 1024),
        ]
        for (
            parallelism,
            interleave_tokens,
            page_tokens,
            token_bytes,
            prompt_tokens,
            split_k,
        ) in binary_cases:
            test_rubric_r1_1_binary_split_additivity(
                parallelism,
                interleave_tokens,
                page_tokens,
                token_bytes,
                prompt_tokens,
                split_k,
            )

        test_rubric_r1_1_heterogeneous_multi_chunk_additivity()
        print(
            "[PASS] R1.1 — Span Additivity Invariant: Verified lossless "
            "decomposition and byte-for-byte span identity for all [0, K) + "
            "[K, N) partitions across PCP=1,2,4,8 and prompt lengths up to "
            "16,384 tokens."
        )
    except Exception as error:
        print(f"[FAIL] R1.1 — Span Additivity Invariant failed: {error}")
        all_passed = False

    # R1.2
    print("\n--- Checking R1.2: Interleave Stride Alignment Enforcement ---")
    try:
        test_rubric_r1_2_interleave_stride_alignment_enforcement()
        print(
            "[PASS] R1.2 — Interleave Stride Alignment Enforcement: Verified "
            "explicit ValueError rejection on unaligned token bases, negative "
            "offsets, and invalid parameters across all PCP interleave cycles."
        )
    except Exception as error:
        print(
            f"[FAIL] R1.2 — Interleave Stride Alignment Enforcement failed: "
            f"{error}"
        )
        all_passed = False

    # R1.3
    print("\n--- Checking R1.3: Legacy Backward Compatibility ---")
    try:
        test_rubric_r1_3_legacy_backward_compatibility()
        print(
            "[PASS] R1.3 — Legacy Backward Compatibility: Verified that "
            "omitting token_base defaults to 0 and maintains identical "
            "single-shot registration behavior across all configurations."
        )
    except Exception as error:
        print(f"[FAIL] R1.3 — Legacy Backward Compatibility failed: {error}")
        all_passed = False

    print(f"\n{banner}")
    if all_passed:
        print("ALL PHASE 1 RUBRIC CHECKS PASSED PERFECTLY!")
        print(banner)
        return True
    else:
        print("PHASE 1 RUBRIC VERIFICATION FAILED.")
        print(banner)
        return False


if __name__ == "__main__":
    success = run_all_rubric_checks()
    sys.exit(0 if success else 1)
