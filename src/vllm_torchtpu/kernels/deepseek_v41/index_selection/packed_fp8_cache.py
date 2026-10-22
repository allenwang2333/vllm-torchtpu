# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 page format: 32 data word rows and one FP32 scale row.

For token t and feature d within a 128x128 page, byte d%4 of word [d//4,t]
contains K[t,d]. Row 32 holds scales[t] as FP32 bits. Page IDs and scale order
are unchanged by the feature/token transpose. Unwritten columns are masked by
sequence lengths; they need no initialization.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.deepseek_v41.index_quantization import quantize_index_rows

from ._packed_fp8_write import write_packed_pages

PAGE_RECORDS = 128
PAGE_ROWS = 33


@dataclass(frozen=True)
class PackedFP8Layout:
    """Location of this layer's pages inside one or two shared word-row owners."""

    pages: int
    page_stride: int = PAGE_ROWS
    page_offset: int = 0

    def validate(self, bank_rows: tuple[int, ...]) -> None:
        if len(bank_rows) not in (1, 2) or any(rows < PAGE_ROWS for rows in bank_rows):
            raise ValueError("Packed FP8 cache requires one or two nonempty banks")
        if self.pages < 1 or self.page_stride < PAGE_ROWS or self.page_offset < 0:
            raise ValueError("Packed FP8 pages require nonoverlapping 33-row slots")
        end = self.page_offset + (self.pages - 1) * self.page_stride + PAGE_ROWS
        if end > sum(bank_rows) or max(bank_rows) >= 1 << 29:
            raise ValueError("Packed FP8 page addresses exceed their owners")
        if len(bank_rows) == 2:
            page = min(
                self.pages - 1,
                max(0, (bank_rows[0] - self.page_offset) // self.page_stride),
            )
            start = self.page_offset + page * self.page_stride
            if start < bank_rows[0] < start + PAGE_ROWS:
                raise ValueError("A packed FP8 page must not cross the bank boundary")


def _record_addresses(records: jax.Array, layout: PackedFP8Layout):
    valid = (records >= 0) & (records < layout.pages * PAGE_RECORDS)
    safe = jnp.clip(records, 0, layout.pages * PAGE_RECORDS - 1)
    base = layout.page_offset + (safe // PAGE_RECORDS) * layout.page_stride
    return base, safe % PAGE_RECORDS, valid


def write_fp8_records(
    banks: tuple[jax.Array, ...],
    records: jax.Array,
    keys: jax.Array,
    layout: PackedFP8Layout,
) -> tuple[jax.Array, ...]:
    """Quantize only new keys and update their columns; valid slots are unique."""
    layout.validate(tuple(bank.shape[0] for bank in banks))
    if keys.shape != (records.size, PAGE_RECORDS) or records.ndim != 1:
        raise ValueError("Expected one logical record slot per D128 key")
    quantized, scales, _ = quantize_index_rows(keys)
    bits = jax.lax.bitcast_convert_type(quantized, jnp.uint8).astype(jnp.uint32)
    bits = bits.reshape(-1, 32, 4)
    words = bits[:, :, 0] | (bits[:, :, 1] << 8)
    words |= (bits[:, :, 2] << 16) | (bits[:, :, 3] << 24)
    values = jnp.concatenate(
        (
            words.astype(jnp.int32),
            jax.lax.bitcast_convert_type(scales, jnp.int32)[:, None],
        ),
        axis=1,
    )
    base, columns, valid = _record_addresses(records, layout)
    updated = []
    offset = 0
    for bank in banks:
        if jax.default_backend() == "tpu":
            updated.append(
                write_packed_pages(
                    bank, jnp.where(valid, base - offset, -1), columns, values
                )
            )
        else:
            rows = base[:, None] + jnp.arange(PAGE_ROWS)[None, :]
            local = rows - offset
            active = valid[:, None] & (local >= 0) & (local < bank.shape[0])
            local = jnp.where(active, local, bank.shape[0])
            updated.append(bank.at[local, columns[:, None]].set(values, mode="drop"))
        offset += bank.shape[0]
    return tuple(updated)


def read_fp8_records(
    banks: tuple[jax.Array, ...], records: jax.Array, layout: PackedFP8Layout
) -> jax.Array:
    """Decode selected records to BF16 for eager consumers; invalid slots are zero."""
    layout.validate(tuple(bank.shape[0] for bank in banks))
    shape = records.shape
    base, columns, valid = _record_addresses(records.reshape(-1), layout)
    rows = base[:, None] + jnp.arange(PAGE_ROWS)[None, :]
    words = jnp.zeros(rows.shape, jnp.int32)
    offset = 0
    for bank in banks:
        local = rows - offset
        active = valid[:, None] & (local >= 0) & (local < bank.shape[0])
        gathered = bank[jnp.clip(local, 0, bank.shape[0] - 1), columns[:, None]]
        words = jnp.where(active, gathered, words)
        offset += bank.shape[0]
    bits = (words[:, :32, None].astype(jnp.uint32) >> (jnp.arange(4) * 8)) & 255
    values = jax.lax.bitcast_convert_type(
        bits.astype(jnp.uint8).reshape(-1, PAGE_RECORDS), jnp.float8_e4m3fn
    )
    scales = jax.lax.bitcast_convert_type(words[:, 32], jnp.float32)
    result = (values.astype(jnp.float32) * scales[:, None]).astype(jnp.bfloat16)
    return jnp.where(valid[:, None], result, 0).reshape(*shape, PAGE_RECORDS)


@jax.jit
def pack_fp8_pages(keys: jax.Array, scales: jax.Array) -> jax.Array:
    """Pack complete pages into 33 int32 rows, preserving every FP8 bit."""
    if keys.ndim != 2 or keys.shape[1] != 128 or keys.shape[0] % 128:
        raise ValueError("Expected complete 128-record pages with dimension 128")
    if keys.dtype != jnp.float8_e4m3fn:
        raise ValueError("Expected float8_e4m3fn keys")
    if scales.shape != (keys.shape[0],) or scales.dtype != jnp.float32:
        raise ValueError("Expected one float32 scale per key record")
    values = jax.lax.bitcast_convert_type(keys, jnp.uint8)
    values = values.reshape(-1, 128, 128).transpose(0, 2, 1)
    values = values.astype(jnp.uint32).reshape(-1, 32, 4, 128)
    words = values[:, :, 0] | (values[:, :, 1] << 8)
    words |= (values[:, :, 2] << 16) | (values[:, :, 3] << 24)
    scale_words = jax.lax.bitcast_convert_type(scales, jnp.int32).reshape(-1, 1, 128)
    return jnp.concatenate((words.astype(jnp.int32), scale_words), axis=1).reshape(
        -1, 128
    )
