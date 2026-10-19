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

Numerics classes (enforced, not just labels):
- `exact`: must reproduce the `default` profile bit-exactly.
- `approx`: changes numerics (e.g. `kv_sub`, `route_vec`, `dense_acc_pair`,
  head-major GDN, hierarchical all-reduce variants, fused pool I/O); must pass
  the accuracy ladder.

Every `Tiles` field and pool-I/O mode declares its class in
`spec.TILE_NUMERICS` / `spec.POOL_IO_NUMERICS`. `approx_features(profile)`
lists the approx features a profile turns on relative to the `default` tile
rule; a profile declared `exact` that turns any of them on is rejected at
construction, and `Profile.with_overrides` re-labels (`<name>+overrides`, class
derived) instead of silently staying `exact`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from .spec import POOL_IO_NUMERICS, TILE_NUMERICS, ExecSpec, Tiles, default_tiles

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
    # In-kernel GDN gather (GPR): False (split) | None (auto: WIDE verify buckets
    # where eligible; exact) | True (every bucket where eligible, never a T=1
    # bucket; approx, see `spec.POOL_IO_NUMERICS`).
    pool_gather: bool | None = False
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
        if self.pool_gather not in (False, None, True):
            raise ValueError(
                f"profile {self.name!r}: pool_gather must be False, None or True, "
                f"got {self.pool_gather!r}"
            )
        if self.numerics == "exact":
            feats = approx_features(self)
            if feats:
                raise ValueError(
                    f"profile {self.name!r} is declared numerics='exact' but turns on "
                    f"approx features {list(feats)}; declare numerics='approx'"
                )

    def with_overrides(self, **overrides: Any) -> Profile:
        """Return this profile with `overrides` applied, re-labelled if changed.

        Overrides equal to the current values are ignored. Any real change
        renames the profile to `<name>+overrides` and derives its numerics
        class: it stays `exact` only if this profile is `exact` and the result
        turns on no approx feature (it never silently stays `exact`).
        """
        changed = {k: v for k, v in overrides.items() if getattr(self, k) != v}
        if not changed:
            return self
        name = (
            self.name if self.name.endswith("+overrides") else f"{self.name}+overrides"
        )
        cand = dataclasses.replace(self, name=name, numerics="approx", **changed)
        exact = self.numerics == "exact" and not approx_features(cand)
        return dataclasses.replace(cand, numerics="exact" if exact else "approx")

    def tiles_for(self, B: int, *, R: int, moe_tp: bool = False) -> Tiles:
        """Return this profile's tiles for a bucket of `B` rows on `R` ranks."""
        return self.tiles(B, moe_tp=moe_tp, R=R)

    def exec_for(self, base: ExecSpec, *, B: int, T: int, g: Any = None) -> ExecSpec:
        """Specialize `base` for bucket `(T, B)` under this profile.

        Explicit settings on `base` win (expert / test overrides), the profile
        fills the rest:
        - `tiles`: `base.tiles` if set, else the profile's tile rule;
        - `pool_gather`: `base.pool_gather` unless it is the library default
          `False` ("no explicit choice"), else the profile's: `None` -> auto,
          `True` -> `"eligible"` (gather wherever the pool backend can, e.g.
          never on a `T=1` bucket). Pass `pool_gather="split"` to force split
          under a GPR profile.
        Fused pool I/O is resolved by the caller through `should_fuse_pool_io`
        because it also depends on the pool backend.
        """
        moe_tp = bool(getattr(g, "moe_tp", False)) if g is not None else False
        tiles = (
            base.tiles
            if base.tiles is not None
            else self.tiles_for(B, R=base.R, moe_tp=moe_tp)
        )
        if base.pool_gather is not False:
            pool_gather = base.pool_gather
        else:
            pool_gather = "eligible" if self.pool_gather is True else self.pool_gather
        return dataclasses.replace(base, B=B, T=T, tiles=tiles, pool_gather=pool_gather)


# Bucket grid on which a profile's tile rule is compared to the default rule.
_NUMERICS_GRID_B = range(1, 65)
_NUMERICS_GRID_R = (1, 2, 4, 8, 16)


def approx_features(profile: Profile) -> tuple[str, ...]:
    """Approx-class features `profile` turns on relative to the `default` rule.

    A tile field counts when the profile's rule gives it a different value
    than `default_tiles` on any bucket of the grid (`B` in 1..64, `R` in
    {1, 2, 4, 8, 16}, EP and TP MoE) and the field is `approx` in
    `TILE_NUMERICS`; pool-I/O modes count per `POOL_IO_NUMERICS`.
    """
    feats: set[str] = set()
    names = [f.name for f in dataclasses.fields(Tiles)]
    approx_names = [n for n in names if TILE_NUMERICS[n] == "approx"]
    for R in _NUMERICS_GRID_R:
        for moe_tp in (False, True):
            for B in _NUMERICS_GRID_B:
                t = profile.tiles(B, moe_tp=moe_tp, R=R)
                d = default_tiles(B, moe_tp=moe_tp, R=R)
                feats.update(
                    f"tiles.{n}" for n in approx_names if getattr(t, n) != getattr(d, n)
                )
    if (
        profile.fused_pool_io is not False
        and POOL_IO_NUMERICS["fused_pool_io"] == "approx"
    ):
        feats.add("fused_pool_io")
    if profile.pool_gather is not False:
        mode = "pool_gather_eligible" if profile.pool_gather is True else "pool_gather"
        if POOL_IO_NUMERICS[mode] == "approx":
            feats.add(mode)
    return tuple(sorted(feats))


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
        if _REGISTRY[profile.name] == profile:
            return _REGISTRY[profile.name]  # idempotent (e.g. re-import)
        raise ValueError(f"profile {profile.name!r} is already registered")
    _REGISTRY[profile.name] = profile
    return profile


def _load_builtins() -> None:
    global _BUILTINS_LOADED
    if not _BUILTINS_LOADED:
        from . import profiles

        profiles.load_builtin_profiles()  # registers checked-in profiles
        _BUILTINS_LOADED = True  # only after a successful import


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
    "approx_features",
    "get_profile",
    "list_profiles",
    "register_profile",
]
