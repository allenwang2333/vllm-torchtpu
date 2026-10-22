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
"""Composed reference step and `ModelDef` derivation for hybrid GQA/GDN + MoE models."""

from __future__ import annotations  # noqa: I001

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from .blocks.gdn import GDNBlock
from .blocks.gqa import GQABlock
from .blocks.moe import MoEBlock
from .blocks.norm import NormBlock
from .decls import REQUIRED, BlockBinding, _lookup_raw_key
from .errors import MegakernelConfigError
from .exec_spec import ExecSpec
from .geometry import ModelGeometry, _merged_cfg
from .ops import BF16, F32, rnd, rup
from .pool import PoolLayout
from .pool.config import cache_exec
from .schedule import (
    BlockOrBinding, FromHFLayerTypes, ResolvedSchedule, Schedule, _block_cls_of,
)  # fmt: skip
from .spec import Block, LayerType, ModelSpec, SwitchSlot


def instantiate_spec_block(
    b: BlockOrBinding, *, pool_layout: PoolLayout | None = None
) -> Block:
    if isinstance(b, Block):
        return b
    opts = {k: v for k, v in b.options.items() if v is not None}
    if issubclass(b.block_cls, GQABlock):
        return b.block_cls(pool_layout=pool_layout, prefix=b.prefix, **opts)
    return b.block_cls(prefix=b.prefix, **opts)


def _norm_zc_and_eps(
    b: BlockOrBinding, merged: dict[str, Any] | None, default_eps: float
) -> tuple[bool, float]:
    if not issubclass(_block_cls_of(b), NormBlock):
        raise MegakernelConfigError(f"Expected NormBlock binding, got {b!r}")
    zc = bool(b.options.get("zero_centered", False))
    if isinstance(b, BlockBinding) and "eps" in b.explicit_fields:
        if merged is not None:
            return zc, float(b.resolve(merged).eps)
        if isinstance(b.fields.get("eps"), (int, float)):
            return zc, float(b.fields["eps"])
    elif not isinstance(b, BlockBinding) and getattr(b, "eps", None) is not None:
        return zc, float(b.eps)
    return zc, default_eps


def _validate_norms(
    cls: Any, sched: ResolvedSchedule, merged: dict[str, Any] | None,
    eps: float, expected_zc: bool | None = None,
) -> bool:  # fmt: skip
    entries = [
        (f"layer {i} slot 'pre_norm'", b) for i, b in enumerate(sched.pre_norm) if b
    ] + [(f"layer {i} slot 'post_norm'", b) for i, b in enumerate(sched.post_norm) if b]
    if cls.final_norm is not None:
        entries.append(("slot 'final_norm'", cls.final_norm))
    if not entries:
        return False if expected_zc is None else expected_zc
    first_loc, first_b = entries[0]
    first_zc, _ = _norm_zc_and_eps(first_b, merged, eps)
    target_zc = first_zc if expected_zc is None else expected_zc
    mname = cls.name or cls.__name__
    for loc, b in entries:
        zc_i, eps_i = _norm_zc_and_eps(b, merged, eps)
        if zc_i != target_zc:
            raise MegakernelConfigError(
                f"ModelDef {mname!r}: {loc} ({b!r}) has zero_centered={zc_i}, "
                f"disagreeing with {first_loc} zero_centered={target_zc}"
            )
        if eps_i != eps:
            raise MegakernelConfigError(
                f"ModelDef {mname!r}: {loc} ({b!r}) has eps={eps_i}, "
                f"disagreeing with model eps={eps}"
            )
    return target_zc


def derive_model_geometry(
    cls: Any, hf_config: Any = None, *,
    moe_tp: bool | None = None, e_slots: int | None = None,
) -> ModelGeometry:  # fmt: skip
    """Derive hybrid GQA/GDN + MoE `ModelGeometry` from `cls` and `hf_config`."""
    if isinstance(hf_config, ModelGeometry):
        rep: dict[str, Any] = {}
        if moe_tp is not None:
            rep["moe_tp"] = bool(moe_tp)
        if e_slots is not None:
            rep["e_slots"] = int(e_slots)
        g = dataclasses.replace(hf_config, **rep)
        object.__setattr__(g, "_model_def", cls)
        return g

    mname = cls.name or cls.__name__
    if not isinstance(cls.schedule, Schedule):
        raise MegakernelConfigError(f"ModelDef {mname!r}: `schedule` must be Schedule")
    cfg = cls._effective_hf_config(hf_config)
    top = cls._from_hf_decl().resolve(cfg)
    L, H, eps = int(top["layers"]), int(top["hidden"]), float(top["eps"])
    gdn_l2_eps = float(top.get("gdn_l2_eps", eps))
    sched = cls.schedule.resolve(cfg, L)
    merged = _merged_cfg(
        cfg,
        dict(
            hidden_size=H, rms_norm_eps=eps, rope_theta=top.get("rope_theta"),
            partial_rotary_factor=top.get("rope_partial"),
        ),
    )  # fmt: skip
    gqa_b = cls._find_binding(GQABlock, sched)
    if gqa_b is None:
        raise MegakernelConfigError(f"ModelDef {mname!r} has no GQABlock in schedule")
    gqa_g = gqa_b.resolve(merged)
    NH, NKV, HD = int(gqa_g.NH), int(gqa_g.NKV), int(gqa_g.HD)
    if "rot" in gqa_b.explicit_fields or "rope_partial" not in top:
        ROT = int(gqa_g.ROT)
    else:
        ROT = int(round(HD * float(top["rope_partial"])))
    theta = float(top["rope_theta"]) if "rope_theta" in top else float(gqa_g.theta)

    gdn_b = cls._find_binding(GDNBlock, sched)
    if gdn_b is not None:
        gdn_g = gdn_b.resolve(merged)
        KH, VH, KD, VD = int(gdn_g.KH), int(gdn_g.VH), int(gdn_g.KD), int(gdn_g.VD)
        if "gdn_l2_eps" in gdn_b.explicit_fields:
            gdn_l2_eps = float(gdn_g.gdn_l2_eps)
    else:
        KH = VH = KD = VD = 0

    moe_b = cls._find_binding(MoEBlock, sched)
    if moe_b is None:
        raise MegakernelConfigError(f"ModelDef {mname!r} has no MoEBlock in schedule")
    exp_spec = moe_b.fields.get("experts", "num_experts")
    exp_key = exp_spec[0] if isinstance(exp_spec, tuple) else exp_spec
    if isinstance(exp_key, str):
        raw_e = _lookup_raw_key(merged, exp_key)
        if raw_e is REQUIRED or raw_e is None or (
            isinstance(raw_e, (int, float)) and raw_e <= 0
        ):  # fmt: skip
            raise MegakernelConfigError(
                f"{cls.__name__} supports MoE checkpoints only "
                f"(missing or non-positive {exp_key!r})"
            )
    moe_g = moe_b.resolve(merged)
    eff_tp = bool(moe_g.moe_tp) if moe_tp is None else bool(moe_tp)
    eff_slots = int(moe_g.e_slots) if e_slots is None else int(e_slots)
    zc = _validate_norms(cls, sched, merged, eps)
    g = cls.geometry_cls(
        L=L, att_every=sched.att_every, H=H, NH=NH, NKV=NKV, HD=HD, ROT=ROT,
        KH=KH, VH=VH, KD=KD, VD=VD, E=int(moe_g.E), TOPK=int(moe_g.TOPK),
        I=int(moe_g.I), IS=int(moe_g.IS), moe_tp=eff_tp, e_slots=eff_slots,
        eps=eps, gdn_l2_eps=gdn_l2_eps, theta=theta, zero_centered_norm=zc,
    )  # fmt: skip
    object.__setattr__(g, "_model_def", cls)
    return g


def derive_model_spec(
    cls: Any, hf_config_or_geom: Any = None, *, gdn_block: Block | None = None,
    moe_tp: bool | None = None, e_slots: int | None = None, R: int = 8,
) -> ModelSpec:  # fmt: skip
    """Derive `ModelSpec` (`LayerType` + `SwitchSlot`s), failing closed on overrides."""
    g = cls.derive_geometry(hf_config_or_geom, moe_tp=moe_tp, e_slots=e_slots)
    sched = cls.resolve_schedule(g)
    mname = cls.name or cls.__name__
    ovs = cls.schedule.overrides if isinstance(cls.schedule, Schedule) else {}
    _validate_norms(cls, sched, None, g.eps, expected_zc=g.zero_centered_norm)
    for slot_name, seq in (("pre_norm", sched.pre_norm), ("post_norm", sched.post_norm)):  # fmt: skip # noqa: E501
        canon_n = next(
            (b for i, b in enumerate(seq) if slot_name not in ovs.get(i, {})), seq[0]
        )
        for lyr, b in enumerate(seq):
            if b != canon_n:
                raise MegakernelConfigError(
                    f"ModelDef {mname!r}: layer {lyr} slot {slot_name!r} binding "
                    f"{b!r} cannot be represented in ModelSpec (expected {canon_n!r})"
                )
    p = sched.att_every
    canon_att = next((sched.mixer[i] for i in range(p - 1, g.L, p) if "mixer" not in ovs.get(i, {})), sched.mixer[p - 1])  # fmt: skip # noqa: E501
    canon_rec = next((sched.mixer[i] for i in range(g.L) if (i + 1) % p != 0 and "mixer" not in ovs.get(i, {})), sched.mixer[0])  # fmt: skip # noqa: E501
    for lyr, b in enumerate(sched.mixer):
        exp_m = canon_att if (lyr + 1) % p == 0 else canon_rec
        if b != exp_m:
            raise MegakernelConfigError(
                f"ModelDef {mname!r}: layer {lyr} slot 'mixer' binding {b!r} "
                f"cannot be represented in ModelSpec (expected {exp_m!r})"
            )
    canon_ffn = next(
        (
            b for i, b in enumerate(sched.ffn)
            if "ffn" not in ovs.get(i, {}) and issubclass(_block_cls_of(b), MoEBlock)
            and getattr(_block_cls_of(b), "routed_experts", True)
        ),
        None,
    )  # fmt: skip
    for lyr, b in enumerate(sched.ffn):
        if canon_ffn is None or b != canon_ffn:
            raise MegakernelConfigError(
                f"ModelDef {mname!r}: layer {lyr} slot 'ffn' binding {b!r} "
                "cannot be represented in ModelSpec"
            )
    play = cls.pool_layout(g, R=R)
    branches, counts = [], []
    for b in sched.mixer[:p]:
        blk = gdn_block if gdn_block is not None and issubclass(_block_cls_of(b), GDNBlock) else instantiate_spec_block(b, pool_layout=play)  # fmt: skip # noqa: E501
        if branches and branches[-1].name == blk.name:
            counts[-1] += 1
        else:
            branches.append(blk)
            counts.append(1)
    mixer_slot = SwitchSlot("mixer", tuple(branches), tuple(counts))
    ffn_blk = instantiate_spec_block(canon_ffn)
    lt_name = cls.layer_type_name or f"{cls.name.replace('_', '')}_layer"
    return ModelSpec(
        name=cls.name, config=g,
        layer_type=LayerType(name=lt_name, slots=(mixer_slot, ffn_blk)),
        cache_backend=cls.pool_backend(g, R=R),
    )  # fmt: skip


def _synth_layer_types(cls: Any, g: ModelGeometry) -> tuple[str, list[str]] | None:
    if not isinstance(cls.schedule, Schedule) or not isinstance(
        cls.schedule.mixer, FromHFLayerTypes
    ):
        return None
    if g.L <= 0 or g.att_every <= 0 or g.L % g.att_every != 0:
        raise MegakernelConfigError(f"Invalid layers L={g.L}, att_every={g.att_every}")
    m = cls.schedule.mixer.mapping
    att_k = next((k for k, b in m.items() if issubclass(_block_cls_of(b), GQABlock)), None)  # fmt: skip # noqa: E501
    rec_k = next((k for k, b in m.items() if issubclass(_block_cls_of(b), GDNBlock)), None)  # fmt: skip # noqa: E501
    if att_k is None or (g.att_every > 1 and rec_k is None):
        raise MegakernelConfigError(
            f"FromHFLayerTypes mapping {dict(m)!r} cannot build att_every={g.att_every}"
        )
    seq = ([rec_k or att_k] * (g.att_every - 1) + [att_k]) * (g.L // g.att_every)
    return cls.schedule.mixer.field, seq


def resolve_model_schedule(cls: Any, hf_config_or_geom: Any = None) -> ResolvedSchedule:
    """Resolve per-layer schedule for `hf_config_or_geom`."""
    if not isinstance(cls.schedule, Schedule):
        raise MegakernelConfigError(
            f"ModelDef {cls.name or cls.__name__!r}: `schedule` must be a Schedule"
        )
    if isinstance(hf_config_or_geom, ModelGeometry):
        g = hf_config_or_geom
        synth = _synth_layer_types(cls, g)
        return cls.schedule.resolve({synth[0]: synth[1]} if synth else {}, g.L)
    cfg = cls._effective_hf_config(hf_config_or_geom)
    return cls.schedule.resolve(cfg, int(cls._from_hf_decl().resolve(cfg)["layers"]))


def shrink_model_hf_config(
    cls: Any, hf_config: Any = None, *, periods: int = 2, R: int = 8,
    ex: ExecSpec | None = None,
) -> dict[str, Any]:  # fmt: skip
    """Return a shrunk HF config dict (`periods` cycles, legal heads/experts for `R`)."""  # noqa: E501
    cfg = cls._effective_hf_config(hf_config)
    g_tiny = cls.derive_geometry(cfg).shrink(periods=periods, R=R, ex=ex)
    out = _merged_cfg(cfg, {})
    for k, v in cls._from_hf_decl().fields.items():
        hf_k = v[0] if isinstance(v, tuple) else v
        if isinstance(hf_k, str) and k in ("layers", "hidden"):
            out[hf_k] = g_tiny.L if k == "layers" else g_tiny.H
    sched = cls.resolve_schedule(g_tiny)
    gqa_map = dict(heads=g_tiny.NH, kv_heads=g_tiny.NKV, head_dim=g_tiny.HD)
    gdn_map = dict(k_heads=g_tiny.KH, v_heads=g_tiny.VH, k_dim=g_tiny.KD, v_dim=g_tiny.VD)  # fmt: skip # noqa: E501
    moe_map = dict(experts=g_tiny.E, top_k=g_tiny.TOPK, inter=g_tiny.I, shared_inter=g_tiny.IS)  # fmt: skip # noqa: E501
    for b, mapping in (
        (cls._find_binding(GQABlock, sched), gqa_map),
        (cls._find_binding(GDNBlock, sched), gdn_map),
        (cls._find_binding(MoEBlock, sched), moe_map),
    ):
        if b is not None:
            for canon, val in mapping.items():
                spec_v = b.fields.get(canon)
                hf_k = spec_v[0] if isinstance(spec_v, tuple) else spec_v
                if isinstance(hf_k, str):
                    out[hf_k] = val
    if (synth := _synth_layer_types(cls, g_tiny)) is not None:
        out[synth[0]] = synth[1]
    return out


def _make_ref_block(b: BlockOrBinding | None, g: ModelGeometry) -> Block | None:
    if b is None or isinstance(b, Block):
        return b
    if not isinstance(b, BlockBinding):
        raise MegakernelConfigError(f"Expected Block or BlockBinding, got {b!r}")
    if issubclass(b.block_cls, NormBlock):
        zc, eps_val = _norm_zc_and_eps(b, None, g.eps)
        return b.block_cls(b.prefix, geometry=g, zero_centered=zc, eps=eps_val)
    return b.block_cls(
        geometry=g, prefix=b.prefix,
        **{k: v for k, v in b.options.items() if v is not None},
    )  # fmt: skip


def composed_reference_step(
    sched: ResolvedSchedule, final_norm: BlockOrBinding | None,
    g: ModelGeometry, ex: ExecSpec, pos: jax.Array, hidden: jax.Array,
    logical_weights: dict[str, Any], logical_states: dict[str, Any],
    *, paged_kv: bool | None = None, pool_io: bool = False,
) -> tuple[jax.Array, dict[str, Any]]:  # fmt: skip
    """Execute hybrid GQA/GDN + MoE block `reference()` along `sched`."""
    R, B, T, NS = ex.R, ex.B, ex.T, ex.NS
    L, ngdn = g.L, g.NGDN
    use_paged_kv = cache_exec(ex).paged_kv if paged_kv is None else bool(paged_kv)
    step_state: dict[str, Any] = {"kc": logical_states["kc"], "vc": logical_states["vc"]}  # fmt: skip # noqa: E501
    CR, HR = rup(3 * B, 8), rup(3 + T, 8)
    vh = KD = VD = QKV = 0
    conv_in = rec_in = None
    if ngdn > 0:
        lay_gdn = GDNBlock.lin_in_layout(g, ex)
        vh, KD, VD, QKV = lay_gdn.vh, lay_gdn.KD, lay_gdn.VD, lay_gdn.QKV
        conv_in = logical_states.get("conv", jnp.zeros((R, ngdn, NS, 3, QKV), dtype=F32))  # fmt: skip # noqa: E501
        if pool_io:
            conv_in = conv_in.astype(BF16).astype(F32)
        rec_in = logical_states.get(
            "rec", jnp.zeros((R, ngdn, NS, vh, KD, VD), dtype=BF16)
        ).reshape(R, ngdn, NS, vh, KD, VD)
        step_state["conv"], step_state["rec"] = conv_in, rec_in
        step_state["conv_out"] = jnp.zeros(
            (R, ngdn, CR, QKV) if T == 1 else (R, ngdn, NS, HR, QKV), dtype=F32
        )
        step_state["rec_out"] = jnp.zeros((R, ngdn, B * vh, KD, VD), dtype=BF16)

    res = hidden[:B].astype(F32)
    for l in range(L):  # noqa: E741
        pre_blk = _make_ref_block(sched.pre_norm[l], g)
        x_mix = pre_blk.reference(logical_weights["in_norm"][l], res, g=g)[0] if pre_blk is not None else res  # fmt: skip # noqa: E501
        mix_blk = _make_ref_block(sched.mixer[l], g)
        if mix_blk is None:
            raise MegakernelConfigError(f"Layer {l} has no mixer block")
        b_idx = sched.mixer_branch_indices[l]
        if isinstance(mix_blk, GQABlock):
            mix_kw = dict(pos=pos, att_idx=b_idx, layer_idx=l, paged_kv=use_paged_kv)
        elif isinstance(mix_blk, GDNBlock):
            mix_kw = dict(gdn_idx=b_idx, layer_idx=l, multi_layer=True)
        else:
            mix_kw = dict(branch_idx=b_idx, layer_idx=l)
        mix_delta, step_state = mix_blk.reference(
            logical_weights, x_mix, step_state, g=g, ex=ex, **mix_kw
        )
        res = rnd(res + mix_delta)
        post_blk = _make_ref_block(sched.post_norm[l], g)
        x_ffn = post_blk.reference(logical_weights["post_norm"][l], res, g=g)[0] if post_blk is not None else res  # fmt: skip # noqa: E501
        ffn_blk = _make_ref_block(sched.ffn[l], g)
        if ffn_blk is None:
            raise MegakernelConfigError(f"Layer {l} has no ffn block")
        ffn_delta, step_state = ffn_blk.reference(
            logical_weights, x_ffn, step_state, g=g, ex=ex, layer_idx=l
        )
        res = rnd(res + ffn_delta)

    fn_blk = _make_ref_block(final_norm, g)
    out_b = fn_blk.reference(logical_weights["final_norm"], res, g=g, out_dtype=BF16)[0] if fn_blk is not None else res.astype(BF16)  # fmt: skip # noqa: E501
    out_states: dict[str, Any] = {"kc": step_state["kc"], "vc": step_state["vc"]}
    if ngdn > 0 and conv_in is not None and rec_in is not None:
        conv_out, rec_out = step_state["conv_out"], step_state["rec_out"]
        if pool_io:
            conv_out = conv_out.astype(BF16).astype(F32)
        if ex.VER:
            conv_pad = (
                jnp.pad(
                    conv_in,
                    ((0, 0), (0, 0), (0, 0), (0, HR - conv_in.shape[-2]), (0, 0)),
                )  # fmt: skip # noqa: E501
                if conv_in.ndim == 5 and conv_in.shape[-2] < HR
                else conv_in
            )
            out_states["conv"] = conv_pad
            out_states["rec"] = rec_in.reshape(R, ngdn, NS * vh, KD, VD)
            out_states["conv_ckpt"], out_states["rec_ckpt"] = conv_out, rec_out
        else:
            out_states["conv"], out_states["rec"] = conv_out, rec_out
    return out_b, out_states
