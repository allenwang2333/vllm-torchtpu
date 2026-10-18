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
"""Named execution profiles: per-bucket tuning bundles selected by name.

A `Profile` groups everything a performance engineer tunes for a deployment
(tiles per bucket, opt-in perf flags, bucket lists, pool-I/O mode) so model
files and library defaults stay general. The `default` profile is the
frozen-parity path (`spec.default_tiles`, split pool I/O, no GPR). Adapters
(vLLM, SGLang, plain JAX) select a profile by name and apply it per bucket via
`Profile.exec_for`; nothing in the core changes behaviour based on profiles.

Numerics classes:
- `exact`: must reproduce the `default` profile bit-exactly.
- `approx`: changes numerics (e.g. `kv_sub`, `route_vec`, `dense_acc_pair`,
  hierarchical all-reduce variants); must pass the accuracy ladder.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from .spec import ExecSpec, Tiles, default_tiles

# Tile rule: `(B, *, moe_tp, R) -> Tiles` for a bucket of `B` rows.
TileRule = Callable[..., Tiles]


@dataclass(frozen=True)
class Baseline:
    """Provenance of the reference a profile is tuned against."""

    ref: str
    commit: str = ""
    job: str = ""
    config: str = ""
    bench: str = ""


@dataclass(frozen=True)
class Profile:
    """Named bundle of execution tuning (tiles, flags, buckets, pool I/O)."""

    name: str
    model: str | None = None  # `ModelSpec.name` this profile targets (None: any)
    numerics: Literal["exact", "approx"] = "exact"
    tiles: TileRule = default_tiles
    buckets: tuple[int, ...] | None = None  # T=1 decode buckets (None: adapter default)
    verify_buckets: tuple[int, ...] | None = None  # T>1 buckets (None: adapter default)
    # Experimental 1-launch fused pool I/O: False | True | "auto" (see
    # `cache_experimental.should_fuse_pool_io`). Requires pool donation.
    fused_pool_io: bool | str = False
    fused_pool_io_max_b: int = 24
    fused_pool_io_min_b_hi: int = 56
    pool_gather: bool | None = False  # In-kernel GDN gather: False | None (auto) | True
    baseline: Baseline | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ValueError(f"invalid profile name {self.name!r}")
        if self.numerics not in ("exact", "approx"):
            raise ValueError(
                f"profile {self.name!r}: numerics must be 'exact' or 'approx', "
                f"got {self.numerics!r}"
            )
        if isinstance(self.fused_pool_io, str) and self.fused_pool_io != "auto":
            raise ValueError(
                f"profile {self.name!r}: fused_pool_io must be a bool or 'auto', "
                f"got {self.fused_pool_io!r}"
            )

    def tiles_for(self, B: int, *, R: int, moe_tp: bool = False) -> Tiles:
        """Return this profile's tiles for a bucket of `B` rows on `R` ranks."""
        return self.tiles(B, moe_tp=moe_tp, R=R)

    def exec_for(self, base: ExecSpec, *, B: int, T: int, g: Any = None) -> ExecSpec:
        """Specialize `base` for bucket `(T, B)` under this profile.

        An explicit `base.tiles` always wins (expert / test override); otherwise
        the profile's tile rule supplies them. `pool_gather` comes from the
        profile. Fused pool I/O is resolved by the caller through
        `should_fuse_pool_io` because it also depends on the pool backend.
        """
        moe_tp = bool(getattr(g, "moe_tp", False)) if g is not None else False
        tiles = (
            base.tiles
            if base.tiles is not None
            else self.tiles_for(B, R=base.R, moe_tp=moe_tp)
        )
        return dataclasses.replace(
            base, B=B, T=T, tiles=tiles, pool_gather=self.pool_gather
        )


DEFAULT_PROFILE = Profile(
    name="default",
    numerics="exact",
    description=(
        "General frozen-parity path: spec.default_tiles, split pool I/O, no GPR, "
        "all opt-in perf flags off."
    ),
)

_REGISTRY: dict[str, Profile] = {DEFAULT_PROFILE.name: DEFAULT_PROFILE}
_BUILTINS_LOADED = False


def register_profile(profile: Profile, *, replace: bool = False) -> Profile:
    """Register `profile` by name (`replace=True` to overwrite an existing one)."""
    if profile.name in _REGISTRY and not replace:
        if _REGISTRY[profile.name] is profile:
            return profile
        raise ValueError(f"profile {profile.name!r} is already registered")
    _REGISTRY[profile.name] = profile
    return profile


def _load_builtins() -> None:
    global _BUILTINS_LOADED
    if not _BUILTINS_LOADED:
        _BUILTINS_LOADED = True
        from . import profiles  # noqa: F401  (registers checked-in profiles)


def list_profiles() -> tuple[str, ...]:
    """Return the names of all registered profiles (built-ins included)."""
    _load_builtins()
    return tuple(sorted(_REGISTRY))


def get_profile(name: str | Profile | None) -> Profile:
    """Resolve a profile by name (`None` / `""` -> `default`)."""
    if isinstance(name, Profile):
        return name
    key = (name or "default").strip()
    _load_builtins()
    if key not in _REGISTRY:
        raise KeyError(
            f"unknown megakernel profile {key!r}; available: {list(list_profiles())}"
        )
    return _REGISTRY[key]


__all__ = [
    "DEFAULT_PROFILE",
    "Baseline",
    "Profile",
    "TileRule",
    "get_profile",
    "list_profiles",
    "register_profile",
]
