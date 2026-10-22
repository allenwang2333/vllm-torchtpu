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
"""Declarative per-layer block scheduling (`Schedule`, `Every`, `FromHFLayerTypes`)."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .decls import (
    REQUIRED,
    BlockBinding,
    FrozenDict,
    KVCacheDecl,
    RecurrentCacheDecl,
    _lookup_raw_key,
)
from .errors import MegakernelConfigError
from .spec import Block

BlockOrBinding = BlockBinding | Block
_VALID_OVERRIDE_SLOTS = frozenset({"mixer", "ffn", "pre_norm", "post_norm"})


def _is_block_or_binding(x: Any) -> bool:
    return isinstance(x, (BlockBinding, Block))


def _block_cls_of(b: BlockOrBinding) -> type[Block]:
    return b.block_cls if isinstance(b, BlockBinding) else type(b)


def _block_name_of(b: BlockOrBinding) -> str:
    return b.block_cls.name if isinstance(b, BlockBinding) else b.name


@dataclass(frozen=True)
class FromHFLayerTypes:
    """Resolve per-layer block bindings from an HF config layer-type list field."""

    field: str = "layer_types"
    mapping: Mapping[str, BlockOrBinding] = dataclasses.field(
        default_factory=FrozenDict
    )

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or not self.field.strip():
            raise MegakernelConfigError(
                f"FromHFLayerTypes: field must be non-empty str, got {self.field!r}"
            )
        if not isinstance(self.mapping, Mapping) or not self.mapping:
            raise MegakernelConfigError(
                "FromHFLayerTypes: mapping must be a non-empty {layer_type: Block} dict"
            )
        for k, v in self.mapping.items():
            if not isinstance(k, str) or not k:
                raise MegakernelConfigError(
                    f"FromHFLayerTypes: key must be a non-empty string, got {k!r}"
                )
            if not _is_block_or_binding(v):
                raise MegakernelConfigError(
                    f"FromHFLayerTypes: {k!r} (layer 0) has no valid block ({v!r})"
                )
        object.__setattr__(self, "mapping", FrozenDict(self.mapping))

    def resolve(
        self, hf_config: Any, num_layers: int, *, slot_name: str = "mixer"
    ) -> tuple[BlockOrBinding, ...]:
        """Resolve a length-`num_layers` tuple of block bindings from `hf_config`."""
        raw = _lookup_raw_key(hf_config, self.field)
        if raw is REQUIRED or raw is None:
            raise MegakernelConfigError(
                f"Schedule.{slot_name}: missing field {self.field!r} "
                f"for layers 0..{max(0, num_layers - 1)}"
            )
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise MegakernelConfigError(
                f"Schedule.{slot_name}: {self.field!r} must be a sequence, "
                f"got {raw!r} at layer 0"
            )
        if len(raw) != num_layers:
            raise MegakernelConfigError(
                f"Schedule.{slot_name}: {self.field!r} has length {len(raw)}, expected "
                f"num_layers={num_layers} (at layer {min(len(raw), num_layers)})"
            )
        out: list[BlockOrBinding] = []
        supported = sorted(self.mapping)
        for idx, lt in enumerate(raw):
            if lt not in self.mapping:
                raise MegakernelConfigError(
                    f"Schedule.{slot_name}: unknown layer type {lt!r} at layer {idx}; "
                    f"supported: {supported}"
                )
            out.append(self.mapping[lt])
        return tuple(out)


@dataclass(frozen=True)
class Every:
    """Periodic selector: `last` on every `n`-th layer, `rest` on the others."""

    n: int
    last: BlockOrBinding
    rest: BlockOrBinding | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.n, int) or isinstance(self.n, bool) or self.n < 1:
            raise MegakernelConfigError(
                f"Every: period n must be an integer >= 1, got {self.n!r} at layer 0"
            )
        if not _is_block_or_binding(self.last):
            raise MegakernelConfigError(
                f"Every(n={self.n}): layer {self.n - 1} ('last') "
                f"has no valid block (got {self.last!r})"
            )
        if self.n > 1 and not _is_block_or_binding(self.rest):
            raise MegakernelConfigError(
                f"Every(n={self.n}): layer 0 ('rest') "
                f"has no valid block (got {self.rest!r})"
            )
        if self.n == 1 and self.rest is not None:
            raise MegakernelConfigError(
                f"Every(n=1): 'rest' must be None when period n=1, "
                f"got {self.rest!r} at layer 0"
            )

    def resolve(
        self, hf_config: Any, num_layers: int, *, slot_name: str = "mixer"
    ) -> tuple[BlockOrBinding, ...]:
        del hf_config
        if num_layers <= 0 or num_layers % self.n != 0:
            raise MegakernelConfigError(
                f"Schedule.{slot_name}: num_layers={num_layers} is not divisible by "
                f"Every(n={self.n}) period (layer {max(0, num_layers - 1)})"
            )
        rest_blk = self.last if self.n == 1 else self.rest
        if rest_blk is None:
            raise MegakernelConfigError(
                f"Every(n={self.n}): missing 'rest' block at layer 0"
            )
        return tuple(
            self.last if (idx + 1) % self.n == 0 else rest_blk
            for idx in range(num_layers)
        )


def _validate_slot_spec(spec: Any, slot_name: str, *, allow_none: bool = False) -> None:
    if spec is None:
        if allow_none:
            return
        raise MegakernelConfigError(
            f"Schedule.{slot_name}: layer 0 has no block assigned (got None)"
        )
    if isinstance(spec, (BlockBinding, Block, FromHFLayerTypes, Every)):
        return
    if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)) and spec:
        for idx, item in enumerate(spec):
            if not _is_block_or_binding(item):
                raise MegakernelConfigError(
                    f"Schedule.{slot_name}: layer {idx} has no valid block ({item!r})"
                )
        return
    raise MegakernelConfigError(
        f"Schedule.{slot_name}: unsupported schedule specification {spec!r} at layer 0"
    )


def _resolve_slot_spec(
    spec: Any,
    hf_config: Any,
    num_layers: int,
    slot_name: str,
    *,
    allow_none: bool = False,
) -> list[BlockOrBinding | None]:
    if spec is None:
        if allow_none:
            return [None] * num_layers
        raise MegakernelConfigError(
            f"Schedule.{slot_name}: layer 0 has no block assigned (got None)"
        )
    if _is_block_or_binding(spec):
        return [spec] * num_layers
    if isinstance(spec, (FromHFLayerTypes, Every)):
        return list(spec.resolve(hf_config, num_layers, slot_name=slot_name))
    if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
        if len(spec) != num_layers:
            raise MegakernelConfigError(
                f"Schedule.{slot_name}: sequence has length {len(spec)}, expected "
                f"num_layers={num_layers} (at layer {min(len(spec), num_layers)})"
            )
        for idx, item in enumerate(spec):
            if not _is_block_or_binding(item):
                raise MegakernelConfigError(
                    f"Schedule.{slot_name}: layer {idx} has no valid block ({item!r})"
                )
        return list(spec)
    raise MegakernelConfigError(
        f"Schedule.{slot_name}: unsupported schedule specification {spec!r} at layer 0"
    )


def _derive_att_every_and_branch_indices(
    mixer: Sequence[BlockOrBinding],
) -> tuple[int, tuple[int, ...]]:
    """Derive periodic `att_every` and per-layer mixer branch ordinals.

    In the hybrid attention / linear-attention family, each cycle of `period`
    layers must be either a single `gqa` layer (`period == 1`) or `period - 1`
    `gdn` layers followed by one `gqa` layer.
    """
    num_layers = len(mixer)
    if num_layers <= 0:
        raise MegakernelConfigError("Schedule.mixer: num_layers must be >= 1, got 0")
    counts: dict[str, int] = {}
    branch_indices: list[int] = []
    names: list[str] = []
    for lyr, b in enumerate(mixer):
        if not _is_block_or_binding(b):
            raise MegakernelConfigError(
                f"Schedule.mixer: layer {lyr} has no valid block (got {b!r})"
            )
        b_name = _block_name_of(b)
        if b_name not in ("gqa", "gdn"):
            raise MegakernelConfigError(
                f"Schedule.mixer: layer {lyr} has unsupported mixer block {b_name!r} "
                f"({b!r}); expected 'gdn' or 'gqa'"
            )
        idx = counts.get(b_name, 0)
        counts[b_name] = idx + 1
        branch_indices.append(idx)
        names.append(b_name)

    if set(names) == {"gqa"}:
        return 1, tuple(branch_indices)
    if "gqa" not in names or names[0] != "gdn":
        raise MegakernelConfigError(
            f"Schedule.mixer: layer 0 has {names[0]!r} ({mixer[0]!r}); "
            "hybrid GDN+GQA schedule requires 'gdn' layers preceding 'gqa'"
        )
    period = names.index("gqa") + 1
    if num_layers % period != 0:
        raise MegakernelConfigError(
            f"Schedule.mixer: num_layers={num_layers} is not a multiple of "
            f"mixer period={period} (at layer {num_layers - 1}: {names[-1]!r})"
        )
    for lyr, n in enumerate(names):
        exp = "gqa" if (lyr + 1) % period == 0 else "gdn"
        if n != exp:
            raise MegakernelConfigError(
                f"Schedule.mixer: layer {lyr} has {n!r} ({mixer[lyr]!r}), "
                f"expected {exp!r}"
            )
    return period, tuple(branch_indices)


@dataclass(frozen=True)
class ResolvedSchedule:
    """Resolved per-layer schedule for a concrete `num_layers` model."""

    num_layers: int
    mixer: tuple[BlockOrBinding, ...]
    ffn: tuple[BlockOrBinding, ...]
    pre_norm: tuple[BlockOrBinding | None, ...]
    post_norm: tuple[BlockOrBinding | None, ...]
    att_every: int
    mixer_branch_indices: tuple[int, ...]

    def layer_cache_decls(self, g: Any) -> list[KVCacheDecl | RecurrentCacheDecl]:
        """Return the per-layer cache declarations along `self.mixer`."""
        out: list[KVCacheDecl | RecurrentCacheDecl] = []
        for lyr, b in enumerate(self.mixer):
            blk = (
                b.block_cls(geometry=g, prefix=b.prefix, **dict(b.options))
                if isinstance(b, BlockBinding)
                else b
            )
            decl = blk.cache_decl(g)
            if decl is None:
                raise MegakernelConfigError(
                    f"Schedule.mixer: {blk.name!r} at layer {lyr} has no cache_decl()"
                )
            out.append(decl)
        return out


@dataclass(frozen=True)
class Schedule:
    """Declarative layer schedule (`mixer`, `ffn`, norms, and per-layer `overrides`)."""

    mixer: Any
    ffn: Any
    pre_norm: BlockOrBinding | None = None
    post_norm: BlockOrBinding | None = None
    overrides: Mapping[int, Mapping[str, Any]] = dataclasses.field(
        default_factory=FrozenDict
    )

    def __post_init__(self) -> None:
        _validate_slot_spec(self.mixer, "mixer", allow_none=False)
        _validate_slot_spec(self.ffn, "ffn", allow_none=False)
        _validate_slot_spec(self.pre_norm, "pre_norm", allow_none=True)
        _validate_slot_spec(self.post_norm, "post_norm", allow_none=True)
        if not isinstance(self.overrides, Mapping):
            raise MegakernelConfigError(
                f"Schedule.overrides must be a mapping, got {self.overrides!r}"
            )
        frozen_ov: dict[int, FrozenDict] = {}
        for lyr, ov in self.overrides.items():
            if not isinstance(lyr, int) or isinstance(lyr, bool) or lyr < 0:
                raise MegakernelConfigError(
                    f"Schedule.overrides: layer index {lyr!r} is out of range (>= 0)"
                )
            if not isinstance(ov, Mapping) or not ov:
                raise MegakernelConfigError(
                    f"Schedule.overrides[{lyr}]: value {ov!r} must be non-empty"
                )
            for slot_k, slot_v in ov.items():
                if slot_k not in _VALID_OVERRIDE_SLOTS:
                    raise MegakernelConfigError(
                        f"Schedule.overrides[{lyr}]: unknown {slot_k!r} ({slot_v!r})"
                    )
                allow_none = slot_k in ("pre_norm", "post_norm")
                if (not allow_none and not _is_block_or_binding(slot_v)) or (
                    allow_none
                    and slot_v is not None
                    and not _is_block_or_binding(slot_v)
                ):
                    raise MegakernelConfigError(
                        f"Schedule.overrides[{lyr}][{slot_k!r}]: "
                        f"layer {lyr} has no valid block (got {slot_v!r})"
                    )
            frozen_ov[lyr] = FrozenDict(ov)
        object.__setattr__(self, "overrides", FrozenDict(frozen_ov))

    def validate_bounds(self, num_layers: int) -> None:
        """Validate `num_layers` and `overrides` layer indices in `0..num_layers-1`."""
        if (
            not isinstance(num_layers, int)
            or isinstance(num_layers, bool)
            or num_layers < 1
        ):
            raise MegakernelConfigError(
                f"Schedule: num_layers must be an integer >= 1, got {num_layers!r}"
            )
        for lyr, ov in self.overrides.items():
            if lyr < 0 or lyr >= num_layers:
                raise MegakernelConfigError(
                    f"Schedule.overrides: layer index {lyr} ({dict(ov)!r}) is out of "
                    f"range for num_layers={num_layers} (0..{num_layers - 1})"
                )

    def resolve(self, hf_config: Any, num_layers: int) -> ResolvedSchedule:
        """Resolve per-layer bindings for `num_layers` layers against `hf_config`."""
        self.validate_bounds(num_layers)
        mix_list = _resolve_slot_spec(self.mixer, hf_config, num_layers, "mixer")
        ffn_list = _resolve_slot_spec(self.ffn, hf_config, num_layers, "ffn")
        pre_list = _resolve_slot_spec(
            self.pre_norm, hf_config, num_layers, "pre_norm", allow_none=True
        )
        post_list = _resolve_slot_spec(
            self.post_norm, hf_config, num_layers, "post_norm", allow_none=True
        )
        for lyr, ov in self.overrides.items():
            if "mixer" in ov:
                mix_list[lyr] = ov["mixer"]
            if "ffn" in ov:
                ffn_list[lyr] = ov["ffn"]
            if "pre_norm" in ov:
                pre_list[lyr] = ov["pre_norm"]
            if "post_norm" in ov:
                post_list[lyr] = ov["post_norm"]
        mix_tuple = tuple(b for b in mix_list if b is not None)
        ffn_tuple = tuple(b for b in ffn_list if b is not None)
        att_every, branch_indices = _derive_att_every_and_branch_indices(mix_tuple)
        return ResolvedSchedule(
            num_layers, mix_tuple, ffn_tuple, tuple(pre_list), tuple(post_list),
            att_every, branch_indices,
        )  # fmt: skip


__all__ = ["Every", "FromHFLayerTypes", "ResolvedSchedule", "Schedule"]
