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
"""Shared pool checks: active-sequence modes and checkpoint-slot bounds."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from ..ops import I32


def _resolve_active_seqs(
    active_seqs: jax.Array | str | None,
    *,
    default_mode: str = "from_ckpt",
) -> tuple[str, jax.Array | None]:
    """Normalize `active_seqs` into `(act_mode, active_mask_array)`."""
    if active_seqs is None:
        return default_mode, None
    if isinstance(active_seqs, str):
        if active_seqs in ("from_ckpt", "all"):
            return active_seqs, None
        raise ValueError(
            f"Unsupported active_seqs mode {active_seqs!r}; "
            "expected 'from_ckpt', 'all', None, or an int32/bool mask array"
        )
    return "ref", jnp.asarray(active_seqs, dtype=I32)


def _check_ckpt_bounds(
    ckpt_indices: Any,
    num_pages: int,
    g_per_att: int,
    *,
    act_mode: str = "from_ckpt",
    active_mask: Any = None,
) -> None:
    """Validate concrete (eager) active GDN checkpoint slots fit in `[0, num_pages)`.

    Under `jax.jit` (`Tracer` inputs), this eager host check is a no-op and bounds
    are checked inside the Pallas kernel via `pl.debug_check` when Pallas debug
    checks are enabled (`jax_pallas_enable_debug_checks=True`).
    """
    if (
        ckpt_indices is None
        or g_per_att <= 0
        or isinstance(ckpt_indices, jax.core.Tracer)
    ):
        return
    ckpt_np = np.asarray(ckpt_indices)
    is_3d = ckpt_np.ndim == 3
    ns = ckpt_np.shape[1] if is_3d else ckpt_np.shape[0]
    mask_np = (
        None
        if (active_mask is None or isinstance(active_mask, jax.core.Tracer))
        else np.asarray(active_mask)
    )
    for s in range(ns):
        if act_mode == "from_ckpt":
            active = int(ckpt_np[0, s, 0] if is_3d else ckpt_np[s, 0]) > 0
        elif act_mode == "ref" and mask_np is not None:
            active = int(mask_np[s]) != 0
        else:
            active = True
        if not active:
            continue
        seen_slots: dict[int, int] = {}
        for sub in range(g_per_att):
            slots = ckpt_np[sub, s] if is_3d else ckpt_np[s]
            for slot in np.ravel(slots):
                slot_i = int(slot)
                pb = slot_i * 5 if is_3d else slot_i * (5 * g_per_att) + sub * 5
                if pb < 0 or pb + 5 > num_pages:
                    raise ValueError(
                        f"GDN checkpoint slot {slot_i} (sub={sub}, s={s}) "
                        f"maps to pages [{pb}, {pb + 5}), which exceeds pool "
                        f"capacity [0, {num_pages})"
                    )
                if is_3d:
                    prev_sub = seen_slots.get(slot_i)
                    if prev_sub is not None and prev_sub != sub:
                        raise ValueError(
                            f"3D ckpt_indices slot {slot_i} is shared between "
                            f"sub={prev_sub} and sub={sub} for active sequence "
                            f"s={s}; 3D ckpt_indices requires disjoint slots "
                            f"across sub-layers (use 2D ckpt_indices if slots "
                            f"are shared across sub-layers)"
                        )
                    seen_slots[slot_i] = sub


def _kernel_debug_check(cond: Any, msg: str, *, interpret: bool) -> None:
    """Emit `pl.debug_check` and enforce it in CPU `interpret` mode when enabled."""
    pl.debug_check(cond, msg)
    if interpret and pl.enable_debug_checks.value:

        def _cb(ok: Any) -> None:
            if not bool(ok):
                raise RuntimeError(msg)

        jax.debug.callback(_cb, cond)
