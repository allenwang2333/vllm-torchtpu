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
"""Declarative `ModelDef` base class, derived artifacts, and model registry."""

from __future__ import annotations  # noqa: I001

from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar

import jax

from . import pool
from .blocks.gdn import GDNBlock
from .decls import (
    BlockBinding, FrozenDict, KVCacheDecl, RecurrentCacheDecl,
    _lookup_raw_key, _sub_cfg,
)  # fmt: skip
from .errors import MegakernelConfigError
from .exec_spec import ExecSpec
from .geometry import FromHF, LMHead, ModelGeometry, _BoundFromHF
from .pool import PoolBackend, PoolLayout
from .profile import Profile, get_profile, model_default_profile, register_profile
from .reference import (
    composed_reference_step, derive_model_geometry, derive_model_spec,
    resolve_model_schedule, shrink_model_hf_config,
)  # fmt: skip
from .schedule import (
    BlockOrBinding, Every, FromHFLayerTypes, ResolvedSchedule, Schedule,
)  # fmt: skip
from .spec import Block, ModelSpec


class ModelDef:
    """Declarative model definition base class for the hybrid GQA/GDN + MoE family."""

    _IS_MODEL_DEF: ClassVar[bool] = True
    name: ClassVar[str] = ""
    hf_architectures: ClassVar[tuple[str, ...]] = ()
    hf_model_type: ClassVar[tuple[str, ...]] = ()
    layer_type_name: ClassVar[str | None] = None
    geometry: ClassVar[Any] = None
    schedule: ClassVar[Schedule | None] = None
    embed: ClassVar[Any] = None
    final_norm: ClassVar[BlockOrBinding | None] = None
    lm_head: ClassVar[LMHead | None] = None
    profiles: ClassVar[Mapping[str, Profile]] = FrozenDict()
    default_hf_config: ClassVar[Mapping[str, Any] | None] = None
    pool_layout_override: ClassVar[Callable[..., PoolLayout] | None] = None
    geometry_cls: ClassVar[type[ModelGeometry]] = ModelGeometry

    @classmethod
    def _from_hf_decl(cls) -> FromHF:
        raw = cls.__dict__.get("geometry") or getattr(cls, "geometry", None)
        if isinstance(raw, _BoundFromHF):
            return raw._from_hf
        if isinstance(raw, FromHF):
            return raw
        raise MegakernelConfigError(
            f"ModelDef {cls.__name__} must declare `geometry = FromHF(...)`"
        )

    @classmethod
    def _effective_hf_config(cls, hf_config: Any = None) -> Any:
        if hf_config is not None:
            return hf_config
        if cls.default_hf_config is not None:
            return cls.default_hf_config
        raise MegakernelConfigError(
            f"ModelDef {cls.name or cls.__name__!r} has no default_hf_config; "
            "pass an explicit hf_config"
        )

    @classmethod
    def _find_binding(
        cls, target_cls: type[Block], sched: ResolvedSchedule
    ) -> BlockBinding | None:
        for seq in (sched.mixer, sched.ffn):
            for b in seq:
                if isinstance(b, BlockBinding) and issubclass(b.block_cls, target_cls):
                    return b
        return None

    @classmethod
    def derive_geometry(
        cls, hf_config: Any = None, *,
        moe_tp: bool | None = None, e_slots: int | None = None,
    ) -> ModelGeometry:  # fmt: skip
        """Derive `ModelGeometry` from `cls.geometry`, bindings, and `schedule`."""
        return derive_model_geometry(cls, hf_config, moe_tp=moe_tp, e_slots=e_slots)

    @classmethod
    def resolve_schedule(cls, hf_config_or_geom: Any = None) -> ResolvedSchedule:
        """Resolve per-layer schedule for `hf_config_or_geom`."""
        return resolve_model_schedule(cls, hf_config_or_geom)

    @classmethod
    def layer_cache_decls(
        cls, hf_config_or_geom: Any = None
    ) -> list[KVCacheDecl | RecurrentCacheDecl]:
        """Return the per-layer cache declaration sequence for `hf_config_or_geom`."""
        g = cls.derive_geometry(hf_config_or_geom)
        return cls.resolve_schedule(g).layer_cache_decls(g)

    @classmethod
    def pool_layout(cls, hf_config_or_geom: Any = None, *, R: int) -> PoolLayout:
        """Derive `PoolLayout` from per-layer `cache_decl()` + schedule grouping."""
        g = cls.derive_geometry(hf_config_or_geom)
        if cls.pool_layout_override is not None:
            if not callable(cls.pool_layout_override):
                raise MegakernelConfigError(
                    f"ModelDef {cls.name or cls.__name__!r}: pool_layout_override "
                    "must be a callable (g, *, R=...) -> PoolLayout"
                )
            return cls.pool_layout_override(g, R=R)
        return pool.derive_pool_layout(cls.layer_cache_decls(g), R=R)

    @classmethod
    def pool_backend(cls, hf_config_or_geom: Any = None, *, R: int) -> PoolBackend:
        """Create the `PoolBackend` for `hf_config_or_geom`."""
        g = cls.derive_geometry(hf_config_or_geom)
        return pool.make_pool_backend(
            cls.pool_layout(g, R=R),
            check=getattr(g, "check", None),
            rec_chunk=GDNBlock.chunk_rows,
        )

    @classmethod
    def to_model_spec(
        cls, hf_config_or_geom: Any = None, *, gdn_block: Block | None = None,
        moe_tp: bool | None = None, e_slots: int | None = None, R: int = 8,
    ) -> ModelSpec:  # fmt: skip
        """Derive `ModelSpec` (`LayerType` + `SwitchSlot`s, deduping shared blocks)."""
        return derive_model_spec(
            cls, hf_config_or_geom, gdn_block=gdn_block,
            moe_tp=moe_tp, e_slots=e_slots, R=R,
        )  # fmt: skip

    @classmethod
    def shrink_hf_config(
        cls, hf_config: Any = None, *,
        periods: int = 2, R: int = 8, ex: ExecSpec | None = None,
    ) -> dict[str, Any]:  # fmt: skip
        """Return a shrunk HF config dict (`periods` cycles, legal heads/experts)."""
        return shrink_model_hf_config(cls, hf_config, periods=periods, R=R, ex=ex)

    @classmethod
    def shrink(
        cls, hf_config_or_geom: Any = None, *,
        periods: int = 2, R: int = 8, ex: ExecSpec | None = None,
    ) -> ModelGeometry:  # fmt: skip
        """Return a tiny `ModelGeometry` with `periods` cycles that passes checks."""
        return cls.derive_geometry(hf_config_or_geom).shrink(
            periods=periods, R=R, ex=ex
        )

    @classmethod
    def default_profile(cls) -> Profile:
        """Return the registered default `Profile` for this model."""
        return model_default_profile(cls.name)

    @classmethod
    def get_profile(cls, name: str | None = None) -> Profile:
        """Resolve a profile by `cls.profiles` alias key or global profile name."""
        key = (name or "default").strip()
        if key in (cls.profiles or {}):
            return cls.profiles[key]
        return cls.default_profile() if key == "default" else get_profile(key)

    @classmethod
    def reference_step(
        cls, g: ModelGeometry, ex: ExecSpec, pos: jax.Array, hidden: jax.Array,
        logical_weights: dict[str, Any], logical_states: dict[str, Any],
        *, paged_kv: bool | None = None, pool_io: bool = False,
    ) -> tuple[jax.Array, dict[str, Any]]:  # fmt: skip
        """Composed pure-JAX multi-rank reference along `schedule`."""
        return composed_reference_step(
            cls.resolve_schedule(g), cls.final_norm, g, ex, pos, hidden,
            logical_weights, logical_states, paged_kv=paged_kv, pool_io=pool_io,
        )  # fmt: skip


_MODEL_REGISTRY: dict[str, type[ModelDef]] = {}
_ARCH_REGISTRY: dict[str, type[ModelDef]] = {}
_MODEL_TYPE_REGISTRY: dict[str, type[ModelDef]] = {}


def _validate_model_def(
    cls: type[ModelDef], name: str, archs: tuple[str, ...], model_types: tuple[str, ...]
) -> None:
    if not isinstance(name, str) or not name.strip() or name.strip() != name:
        raise MegakernelConfigError(f"register_model: invalid model name {name!r}")
    if not issubclass(cls, ModelDef):
        raise MegakernelConfigError(
            f"register_model({name!r}): {cls.__name__} must subclass ModelDef"
        )
    from_hf = cls._from_hf_decl()
    for req_key in ("hidden", "layers", "eps"):
        if req_key not in from_hf.fields:
            raise MegakernelConfigError(
                f"register_model({name!r}): FromHF is missing required key {req_key!r}"
            )
    if not isinstance(cls.schedule, Schedule):
        raise MegakernelConfigError(
            f"register_model({name!r}): `schedule` must be a Schedule instance, "
            f"got {cls.schedule!r}"
        )
    if cls.pool_layout_override is not None and not callable(cls.pool_layout_override):
        raise MegakernelConfigError(
            f"register_model({name!r}): pool_layout_override must be callable"
        )

    static_l: int | None = None
    layers_spec = from_hf.fields.get("layers")
    if cls.default_hf_config is not None:
        raw_l = _lookup_raw_key(
            cls.default_hf_config,
            layers_spec[0] if isinstance(layers_spec, tuple) else str(layers_spec),
        )
        if isinstance(raw_l, int) and not isinstance(raw_l, bool):
            static_l = raw_l
    if static_l is None:
        if isinstance(layers_spec, int) and not isinstance(layers_spec, bool):
            static_l = layers_spec
        elif (
            isinstance(layers_spec, tuple)
            and isinstance(layers_spec[1], int)
            and not isinstance(layers_spec[1], bool)
        ):
            static_l = layers_spec[1]
        elif isinstance(cls.schedule.mixer, Sequence) and not isinstance(
            cls.schedule.mixer, (str, bytes)
        ):
            static_l = len(cls.schedule.mixer)

    if static_l is not None:
        cls.schedule.validate_bounds(static_l)
        if cls.default_hf_config is not None or not isinstance(
            cls.schedule.mixer, FromHFLayerTypes
        ):
            cls.schedule.resolve(cls.default_hf_config or {}, static_l)

    if cls.default_hf_config is not None:
        try:
            g_def = derive_model_geometry(cls, cls.default_hf_config)
            cls.pool_layout(g_def, R=8)
        except MegakernelConfigError as exc:
            if name not in str(exc):
                raise MegakernelConfigError(f"register_model({name!r}): {exc}") from exc
            raise
        except Exception as exc:
            raise MegakernelConfigError(
                f"register_model({name!r}): invalid default_hf_config: {exc}"
            ) from exc

    for alias, prof in (cls.profiles or {}).items():
        if not isinstance(alias, str) or not alias.strip() or alias.strip() != alias:
            raise MegakernelConfigError(
                f"register_model({name!r}): invalid profile alias key {alias!r}"
            )
        if not isinstance(prof, Profile):
            raise MegakernelConfigError(
                f"register_model({name!r}): profiles[{alias!r}] must be a Profile, "
                f"got {prof!r}"
            )
        if prof.model is not None and prof.model != name:
            raise MegakernelConfigError(
                f"register_model({name!r}): profile {prof.name!r} targets model "
                f"{prof.model!r}, expected {name!r}"
            )
    for kind, seq in (("hf_architectures", archs), ("hf_model_type", model_types)):
        for entry in seq:
            if not isinstance(entry, str) or not entry.strip():
                raise MegakernelConfigError(
                    f"register_model({name!r}): invalid {kind} entry {entry!r}"
                )


def register_model(
    *, name: str, hf_architectures: Sequence[str] = (),
    hf_model_type: str | Sequence[str] | None = None, replace: bool = False,
) -> Callable[[type[ModelDef]], type[ModelDef]]:  # fmt: skip
    """Class decorator registering a declarative `ModelDef` subclass."""
    archs = tuple(hf_architectures)
    if hf_model_type is None:
        mtypes: tuple[str, ...] = ()
    else:
        mtypes = (
            (hf_model_type,) if isinstance(hf_model_type, str) else tuple(hf_model_type)
        )

    def _decorator(cls: type[ModelDef]) -> type[ModelDef]:
        if name in _MODEL_REGISTRY and not replace:
            if _MODEL_REGISTRY[name] is cls:
                return cls
            prev_cls = _MODEL_REGISTRY[name].__name__
            raise MegakernelConfigError(
                f"Model {name!r} is already registered by {prev_cls}"
            )
        for kind, reg, keys in (
            ("HF architecture", _ARCH_REGISTRY, archs),
            ("HF model_type", _MODEL_TYPE_REGISTRY, mtypes),
        ):
            for k in keys:
                prev = reg.get(k)
                if (
                    prev is not None
                    and prev is not cls
                    and prev.name != name
                    and not replace
                ):  # noqa: E501
                    raise MegakernelConfigError(
                        f"{kind} {k!r} is already registered to {prev.name!r}"
                    )
        _validate_model_def(cls, name, archs, mtypes)
        if replace:
            old_cls = _MODEL_REGISTRY.get(name)
            for reg in (_ARCH_REGISTRY, _MODEL_TYPE_REGISTRY):
                for k in [rk for rk, rv in reg.items() if rv in (cls, old_cls)]:
                    del reg[k]
        cls.name = name
        cls.hf_architectures = archs
        cls.hf_model_type = mtypes
        cls.profiles = FrozenDict(cls.profiles or {})
        for prof in cls.profiles.values():
            register_profile(prof, replace=replace)
        _MODEL_REGISTRY[name] = cls
        for a in archs:
            _ARCH_REGISTRY[a] = cls
        for mt in mtypes:
            _MODEL_TYPE_REGISTRY[mt] = cls
        return cls

    return _decorator


def list_models() -> tuple[str, ...]:
    """Return sorted names of all registered `ModelDef`s."""
    return tuple(sorted(_MODEL_REGISTRY))


def find_model(target: Any) -> type[ModelDef] | None:
    """Look up a `ModelDef` by name, HF architecture, `model_type`, or `hf_config`."""
    if isinstance(target, type) and issubclass(target, ModelDef):
        return target
    if isinstance(target, str):
        key = target.strip()
        return (
            _MODEL_REGISTRY.get(key)
            or _ARCH_REGISTRY.get(key)
            or _MODEL_TYPE_REGISTRY.get(key)
        )
    if target is not None:
        cands: list[str] = []
        archs = _sub_cfg(target, "architectures") or ()
        if isinstance(archs, Sequence) and not isinstance(archs, (str, bytes)):
            cands.extend(str(a) for a in archs if a)
        for root in (target, _sub_cfg(target, "text_config")):
            if root is not None and (mt := _sub_cfg(root, "model_type")):
                cands.append(str(mt))
        for cand in cands:
            if (found := find_model(cand)) is not None:
                return found
    return None


def get_model(target: Any) -> type[ModelDef]:
    """Resolve a `ModelDef` by name, HF architecture, `model_type`, or `hf_config`."""
    found = find_model(target)
    if found is None:
        raise KeyError(
            f"Unknown megakernel model {target!r}; available: {list(list_models())}"
        )
    return found


__all__ = [
    "Every", "FromHF", "FromHFLayerTypes", "LMHead", "ModelDef", "ModelGeometry",
    "ResolvedSchedule", "Schedule", "find_model", "get_model", "list_models",
    "register_model",
]  # fmt: skip
