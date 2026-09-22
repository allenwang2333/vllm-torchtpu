# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The one place the package reads the environment: a Config from the
FUSED_EP_* variables, for a caller at a process boundary such as the
serving adapter or the bench arm. Every field has a default and every
field is overridable. Callers inside a process construct a Config
directly."""
import os

import jax.numpy as jnp

from . import config as defaults


def config_from_env(environ=None):
    """The Config the FUSED_EP_* variables describe.

    FUSED_EP_RESULT_SLOTS, FUSED_EP_ROW_AHEAD, FUSED_EP_WEIGHT_SLOTS,
    FUSED_EP_WEIGHT_AHEAD, FUSED_EP_TABLES_BLOCKS (routing blocks of 128
    pairs per tables-kernel step, 0 for the layer's choice),
    FUSED_EP_STREAM_BLOCK (intermediate columns per streamed weight block,
    0 for the layer's choice), FUSED_EP_STREAM_ROWS (rows per streamed
    group) and FUSED_EP_ACTIVATION_BLOCK (values per fp8 scale of a row, 0
    for one scale per row) are integers; FUSED_EP_VMEM_FRACTION is the
    share of the chip's VMEM the kernels plan for. FUSED_EP_RESULT_ROWS is
    "fp8" or "bf16"; FUSED_EP_ELEMENTWISE and FUSED_EP_COMBINE are "float32"
    or "bfloat16". FUSED_EP_SCALE_BF16 and FUSED_EP_BOUNDS_CHECKS are 0 or
    1. The programs' geometry, integers all: FUSED_EP_GATHER_BLOCK and
    FUSED_EP_GATHER_RING (the gather's rows per DMA and DMAs in flight),
    FUSED_EP_SCATTER_CHUNK and FUSED_EP_SCATTER_RING (the row-tables
    scatter's pairs per chunk and chunks in flight), FUSED_EP_COMBINE_TILE
    and FUSED_EP_COMBINE_UNROLL (the weighted sum's tokens per step and
    unroll), FUSED_EP_ACTIVATION_COLUMNS (intermediate columns per
    activation pass), FUSED_EP_TABLES_ZERO_WORDS (words zeroed per tables
    step) and FUSED_EP_WEIGHT_DMA_PRIORITY (0 or 1). FUSED_EP_TOKEN_BLOCK,
    when set, is the token rows' own rounding block (0 for one scale per
    row); unset, the token rows follow FUSED_EP_ACTIVATION_BLOCK.
    FUSED_EP_SORTED_ROWS is auto, 0 or 1: the routed rows sorted into
    expert order by a sparse-core gather before the FFN kernel, which then
    fetches each tile as one copy (1), one copy per row (0), or the choice
    by the shard's shape (auto, the default).
    """
    env = os.environ if environ is None else environ

    def switch(name, default="0"):
        value = env.get(name, default)
        if value not in ("0", "1"):
            raise ValueError(f"{name}={value!r}: a switch is 0 or 1")
        return value == "1"

    def tristate(name):
        value = env.get(name, "auto")
        if value not in ("auto", "0", "1"):
            raise ValueError(f"{name}={value!r}: auto, 0 or 1")
        return None if value == "auto" else value == "1"

    def integer(name, default):
        return int(env.get(name, default))

    def choice(name, options, default):
        value = env.get(name, default)
        if value not in options:
            raise ValueError(f"{name}={value!r}, one of {options}")
        return value

    return defaults.Config(
        result_slots=integer("FUSED_EP_RESULT_SLOTS",
                             defaults.RESULT_SLOTS_DEFAULT),
        row_fetch_ahead=integer("FUSED_EP_ROW_AHEAD",
                                defaults.ROW_FETCH_AHEAD_DEFAULT),
        weight_slots=integer("FUSED_EP_WEIGHT_SLOTS",
                             defaults.WEIGHT_SLOTS_DEFAULT),
        weight_prefetch=integer("FUSED_EP_WEIGHT_AHEAD",
                                defaults.WEIGHT_PREFETCH_DEFAULT),
        tables_blocks_per_step=integer(
            "FUSED_EP_TABLES_BLOCKS",
            defaults.TABLES_BLOCKS_PER_STEP_AUTOMATIC),
        stream_block=integer("FUSED_EP_STREAM_BLOCK",
                             defaults.STREAM_BLOCK_AUTOMATIC),
        stream_rows=integer("FUSED_EP_STREAM_ROWS",
                            defaults.STREAM_ROWS_DEFAULT),
        vmem_fraction=float(
            env.get("FUSED_EP_VMEM_FRACTION", defaults.VMEM_FRACTION_DEFAULT)),
        activation_block=integer("FUSED_EP_ACTIVATION_BLOCK", 0),
        token_block=(None if "FUSED_EP_TOKEN_BLOCK" not in env else int(
            env["FUSED_EP_TOKEN_BLOCK"])),
        result_rows=choice("FUSED_EP_RESULT_ROWS", defaults.RESULT_ROWS,
                           "fp8"),
        intermediate=choice("FUSED_EP_INTERMEDIATE", defaults.INTERMEDIATES,
                            "fp8"),
        token_rows=choice("FUSED_EP_TOKEN_ROWS", defaults.TOKEN_ROWS, "fp8"),
        elementwise_dtype=choice("FUSED_EP_ELEMENTWISE",
                                 defaults.ELEMENTWISE_DTYPES, "bfloat16"),
        combine_dtype=choice("FUSED_EP_COMBINE", defaults.ELEMENTWISE_DTYPES,
                             "float32"),
        row_scale_dtype=(jnp.bfloat16
                         if switch("FUSED_EP_SCALE_BF16") else jnp.float32),
        bounds_checks=switch("FUSED_EP_BOUNDS_CHECKS"),
        gather_block_rows=integer("FUSED_EP_GATHER_BLOCK",
                                  defaults.GATHER_BLOCK_ROWS_DEFAULT),
        gather_ring=integer("FUSED_EP_GATHER_RING",
                            defaults.GATHER_RING_DEFAULT),
        scatter_chunk_pairs=integer("FUSED_EP_SCATTER_CHUNK",
                                    defaults.SCATTER_CHUNK_PAIRS_DEFAULT),
        scatter_ring=integer("FUSED_EP_SCATTER_RING",
                             defaults.SCATTER_RING_DEFAULT),
        combine_tokens_per_tile=integer(
            "FUSED_EP_COMBINE_TILE", defaults.COMBINE_TOKENS_PER_TILE_DEFAULT),
        combine_unroll=integer("FUSED_EP_COMBINE_UNROLL",
                               defaults.COMBINE_UNROLL_DEFAULT),
        activation_columns_per_pass=integer(
            "FUSED_EP_ACTIVATION_COLUMNS",
            defaults.ACTIVATION_COLUMNS_PER_PASS_DEFAULT),
        tables_zero_words=integer("FUSED_EP_TABLES_ZERO_WORDS",
                                  defaults.TABLES_ZERO_WORDS_DEFAULT),
        weight_dma_priority=integer("FUSED_EP_WEIGHT_DMA_PRIORITY",
                                    defaults.WEIGHT_DMA_PRIORITY_DEFAULT),
        sorted_rows=tristate("FUSED_EP_SORTED_ROWS"))
