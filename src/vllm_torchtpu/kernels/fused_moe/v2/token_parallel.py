# SPDX-License-Identifier: Apache-2.0
"""Disjoint token ownership within groups of replicated MoE inputs."""

import functools
from dataclasses import dataclass, field
from typing import Self

import jax
import numpy as np
from jax import lax


@functools.lru_cache(maxsize=64)
def _slots_i32(slots: tuple[int, ...]) -> np.ndarray:
    return np.asarray(slots, dtype=np.int32)


@dataclass(frozen=True, slots=True)
class TokenReplicaLayout:
    """Groups are ordered mesh coordinates, independent of expert ownership.

    Every member supplies identical input rows. Each computes one contiguous
    slice, then all members recover the original row order. Singleton groups
    leave the input and output unchanged, including unaligned row counts.
    """

    groups: tuple[tuple[int, ...], ...]
    slots: tuple[int, ...]
    input_rows: int
    local_rows: int
    _singleton: bool = field(default=False, repr=False, compare=False)

    @classmethod
    def create(
        cls,
        *,
        ep: int,
        input_rows: int,
        groups: tuple[tuple[int, ...], ...] | None = None,
        row_alignment: int = 1,
    ) -> Self:
        if ep < 1 or input_rows < 1:
            raise ValueError("EP width and input row count must be positive")
        if row_alignment < 1 or row_alignment & (row_alignment - 1):
            raise ValueError("row alignment must be a positive power of two")
        if groups is None:
            return cls(
                tuple((i,) for i in range(ep)),
                (0,) * ep,
                input_rows,
                input_rows,
                True,
            )
        if not groups or not (first_group := groups[0]):
            raise ValueError(
                "token replica groups must partition the mesh "
                "into equally sized, nonempty ordered groups"
            )
        width = len(first_group)
        if len(groups) * width != ep:
            raise ValueError(
                "token replica groups must partition the mesh "
                "into equally sized, nonempty ordered groups"
            )
        slots = [-1] * ep
        norm_groups = []
        for group in groups:
            if len(group) != width:
                raise ValueError(
                    "token replica groups must partition the mesh "
                    "into equally sized, nonempty ordered groups"
                )
            norm_group = []
            for slot, r in enumerate(group):
                rank = int(r)
                if rank < 0 or rank >= ep or slots[rank] != -1:
                    raise ValueError(
                        "token replica groups must partition the mesh "
                        "into equally sized, nonempty ordered groups"
                    )
                slots[rank] = slot
                norm_group.append(rank)
            norm_groups.append(tuple(norm_group))
        local_rows = (input_rows + width - 1) // width
        # Pallas routing tiles need aligned row blocks. Below one block,
        # keep a power-of-two whole-array tile rather than widening to eight.
        if width > 1:
            alignment = min(row_alignment, 1 << (local_rows - 1).bit_length())
            local_rows = (local_rows + alignment - 1) & -alignment
        return cls(
            tuple(norm_groups),
            tuple(slots),
            input_rows,
            local_rows,
            width == 1,
        )

    def split(
        self, rows: jax.Array, rank: jax.Array, *, fill_value: float = 0
    ) -> jax.Array:
        if self._singleton:
            return rows
        local_rows = self.local_rows
        padding = local_rows * len(self.groups[0]) - self.input_rows
        if padding:
            rows = lax.pad(
                rows,
                lax.convert_element_type(fill_value, rows.dtype),
                ((0, padding, 0), (0, 0, 0)),
            )
        slot = lax.dynamic_index_in_dim(_slots_i32(self.slots), rank, 0, False)
        return lax.dynamic_slice_in_dim(rows, slot * local_rows, local_rows, axis=0)

    def restore(self, rows: jax.Array, axis: str) -> jax.Array:
        if self._singleton:
            return rows
        gathered = lax.all_gather(
            rows, axis, axis=0, tiled=True, axis_index_groups=self.groups
        )
        input_rows = self.input_rows
        return (
            gathered
            if gathered.shape[0] == input_rows
            else lax.slice_in_dim(gathered, 0, input_rows)
        )
