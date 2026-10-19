# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 raw-pool layout: `PoolLayout` / `PoolBackend` from `Qwen35Geometry`.

Each of the `g.NATT` attention layers owns one raw pool holding its KV pages and
the checkpoint slots of the `att_every - 1` GDN layers preceding it (see
`pool/layout.py`). The pool kernels and packers live in the generic `pool/`
package; the `g`-taking wrappers below keep the public Qwen3.5 API.
"""

from __future__ import annotations

from typing import Any

import jax

from .. import pool
from ..pool import (
    PoolLayout,
    conv_to_raw_page,
    kv_to_raw_page,
    raw_page_to_conv,
    raw_page_to_kv,
    raw_pages_to_ssm,
    ssm_to_raw_pages,
)
from ..spec import ExecSpec, PoolBackend


def qwen35_pool_layout(g: Any) -> PoolLayout:
    """`PoolLayout` of `Qwen35Geometry`: one pool per attention layer, one KV head
    pair per rank and page, `att_every - 1` GDN checkpoint slots per pool."""
    return PoolLayout(
        num_pools=g.NATT,
        slots_per_pool=g.att_every - 1,
        hd=g.HD,
        kvh=1,
        kh=g.KH,
        vh=g.VH,
        KD=g.KD,
        VD=g.VD,
    )


def qwen35_pool_backend(g: Any) -> PoolBackend:
    """Create the model-injected `PoolBackend` for `Qwen35Geometry`."""
    return pool.make_pool_backend(
        qwen35_pool_layout(g), check=getattr(g, "check", None)
    )


def default_pool_layout(g: Any, ex: ExecSpec) -> tuple[Any, ...]:
    """`pool.allocate_test_layout` for `g`: `(page_table, ckpt, roff, min_pages)`."""
    return pool.allocate_test_layout(qwen35_pool_layout(g), ex)


def states_to_pool(
    g: Any, ex: ExecSpec, *args: Any, **kw: Any
) -> tuple[jax.Array, ...]:
    return pool.states_to_pool(qwen35_pool_layout(g), ex, *args, **kw)


def pool_to_states(g: Any, ex: ExecSpec, *args: Any, **kw: Any) -> dict[str, jax.Array]:
    return pool.pool_to_states(qwen35_pool_layout(g), ex, *args, **kw)


def gather_gdn_states(g: Any, ex: ExecSpec, *args: Any, **kw: Any) -> Any:
    if hasattr(g, "check"):
        g.check(ex)
    return pool.gather_gdn_states(qwen35_pool_layout(g), ex, *args, **kw)


def scatter_pool_states(g: Any, ex: ExecSpec, *args: Any, **kw: Any) -> Any:
    if hasattr(g, "check"):
        g.check(ex)
    return pool.scatter_pool_states(qwen35_pool_layout(g), ex, *args, **kw)


__all__ = [
    "conv_to_raw_page",
    "default_pool_layout",
    "gather_gdn_states",
    "kv_to_raw_page",
    "pool_to_states",
    "qwen35_pool_backend",
    "qwen35_pool_layout",
    "raw_page_to_conv",
    "raw_page_to_kv",
    "raw_pages_to_ssm",
    "scatter_pool_states",
    "ssm_to_raw_pages",
    "states_to_pool",
]
