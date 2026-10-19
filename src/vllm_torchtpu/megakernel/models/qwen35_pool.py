# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 raw-pool page packing, layout, and Pallas GDN gather kernel.

Each of the `g.NATT` attention layers owns one raw pool of shape
`[R, num_pages, 2, 64, 4, page_size] fp8_e4m3fn` (stored rank-folded as
`[R * num_pages, 2, 64, 4, page_size]`). A single pool holds:

1. **KV cache pages** (`1` page per `page_size=128` tokens per sequence):
   `[0]` = K page `(64, 4, 128) fp8`, `[1]` = V page `(64, 4, 128) fp8`, where
   `(64, 4)` packs `HD = 256` into 32-bit words along sublanes so a `bitcast` to
   `uint32` yields `(64, page_size) u32` ready for `pltpu.bitcast(..., fp8)` on
   TPU without lane transposes.
2. **GDN recurrent + conv checkpoint slots** (`5` consecutive pages per slot for
   each of the `G_PER_ATT = att_every - 1` GDN layers preceding the attention
   layer):
   - Pages `0..3`: recurrent SSM state `(vh, KD=128, VD=128) bf16` packed as
     `u32` pairs into `(4, 2, 64, 4, 128) fp8`.
   - Page `4`: causal conv1d 3-tap window `(3, QKV) bf16` packed into
     `(2, 64, 4, 128) fp8`.
"""

from __future__ import annotations

from typing import Any

import jax

from ..blocks.gdn import GDNBlock
from ..ops import rup
from ..pool import (
    conv_to_raw_page,
    default_pool_layout,
    gather_gdn_states,
    kv_to_raw_page,
    pool_to_states,
    raw_page_to_conv,
    raw_page_to_kv,
    raw_pages_to_ssm,
    scatter_pool_states,
    ssm_to_raw_pages,
    states_to_pool,
)
from ..spec import ExecSpec, PoolBackend


def qwen35_pool_backend(g: Any) -> PoolBackend:
    """Create the model-injected `PoolBackend` for `Qwen35Geometry`."""
    has_gdn = g.NGDN > 0

    def _gather(
        ex_paged: ExecSpec,
        raw_pools: tuple[jax.Array, ...],
        ckpt_indices: jax.Array | None,
        read_offsets: jax.Array | None,
        active_seqs: Any,
    ) -> dict[str, Any]:
        if not has_gdn or ex_paged.GPR:
            return {}
        assert ckpt_indices is not None and read_offsets is not None
        conv, rec = gather_gdn_states(
            g,
            ex_paged,
            raw_pools,
            ckpt_indices,
            read_offsets,
            active_seqs,
            interpret=ex_paged.interpret,
        )
        return {"conv": conv, "rec": rec}

    def _scatter(
        ex_paged: ExecSpec,
        raw_pools: tuple[jax.Array, ...],
        pos: jax.Array,
        page_table: jax.Array,
        out_map: dict[str, Any],
        ckpt_indices: jax.Array | None,
        active_seqs: Any,
    ) -> tuple[jax.Array, ...]:
        conv_out = out_map.get("conv_ckpt" if ex_paged.VER else "conv")
        rec_out = out_map.get("rec_ckpt" if ex_paged.VER else "rec")
        return scatter_pool_states(
            g,
            ex_paged,
            raw_pools,
            pos,
            page_table,
            out_map["knew"],
            out_map["vnew"],
            conv_out if has_gdn else None,
            rec_out if has_gdn else None,
            ckpt_indices if has_gdn else None,
            active_seqs,
            interpret=ex_paged.interpret,
            conv_partial=ex_paged.GPR,
        )

    def _can_pool_gather(ex_paged: ExecSpec) -> bool:
        if not has_gdn or ex_paged.page_size != 128 or ex_paged.T <= 1:
            return False
        if ex_paged.pool_gather in (False, "split"):
            return False
        if ex_paged.pool_gather is None and not ex_paged.WIDE:
            return False
        tiles = ex_paged.resolve_tiles(g)
        cb = tiles.gdn_rows
        if cb <= 0 or cb % 8 != 0 or cb % ex_paged.T != 0:
            return False
        _, tr = divmod(ex_paged.B, cb)
        nch = ex_paged.B // cb + int(tr > 0)
        nrb = min(tiles.gdn_bufs, nch)
        if nch > 1 and nrb < 2:
            return False
        lay = GDNBlock.lin_in_layout(g, ex_paged)
        hr = rup(3 + ex_paged.T, 8)
        if (hr * lay.QKV * 2) % (4 * 128) != 0:
            return False
        gconv_rows = (hr * lay.QKV * 2) // (4 * 128)
        return gconv_rows <= 64

    return PoolBackend(
        num_pools=g.NATT,
        has_recurrent_pool=has_gdn,
        recurrent_slots_per_pool=max(1, g.att_every - 1),
        pool_state_name="raw",
        page_table_state_name="page_table",
        gather_states=_gather,
        scatter_states=_scatter,
        supports_fused_io=True,
        supports_pool_gather=True,
        ckpt_pages_per_slot=5,
        kernel_names=("gather_gdn_states", "scatter_pool_states"),
        can_pool_gather=_can_pool_gather,
    )


__all__ = [
    "conv_to_raw_page",
    "default_pool_layout",
    "gather_gdn_states",
    "kv_to_raw_page",
    "pool_to_states",
    "qwen35_pool_backend",
    "raw_page_to_conv",
    "raw_page_to_kv",
    "raw_pages_to_ssm",
    "scatter_pool_states",
    "ssm_to_raw_pages",
    "states_to_pool",
]
