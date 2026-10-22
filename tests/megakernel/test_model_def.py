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
"""Unit and CPU interpret tests for `ModelDef`, `Schedule`, and `FromHF`."""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import vllm_torchtpu.megakernel as mk
from vllm_torchtpu.megakernel import (
    geometry,
    model_def,
    profile,
    reference,
    schedule,
    spec,
)
from vllm_torchtpu.megakernel.blocks import (
    GDN,
    GQA,
    Embedding,
    GDNBlock,
    GQABlock,
    MoE,
    MoEBlock,
    Norm,
)
from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.model_def import (
    Every,
    FromHF,
    FromHFLayerTypes,
    LMHead,
    ModelDef,
    Schedule,
    find_model,
    get_model,
    list_models,
    register_model,
)
from vllm_torchtpu.megakernel.models import (
    qwen35,
    qwen35_pack,
    qwen35_pool,
    qwen35_weights,
)
from vllm_torchtpu.megakernel.models.qwen35 import DEFAULT_HF_CONFIG, Qwen35
from vllm_torchtpu.megakernel.models.qwen35_profile import (
    PROFILE,
    QWEN35_MODEL_NAME,
    TINY_TILES,
)
from vllm_torchtpu.megakernel.ops import BF16, F32, rnd
from vllm_torchtpu.megakernel.profile import DEFAULT_PROFILE, Profile


@pytest.fixture()
def isolated_model_registry(monkeypatch: pytest.MonkeyPatch):
    """Snapshot and restore model and profile registries around mutation tests."""
    monkeypatch.setattr(model_def, "_MODEL_REGISTRY", dict(model_def._MODEL_REGISTRY))
    monkeypatch.setattr(model_def, "_ARCH_REGISTRY", dict(model_def._ARCH_REGISTRY))
    monkeypatch.setattr(
        model_def, "_MODEL_TYPE_REGISTRY", dict(model_def._MODEL_TYPE_REGISTRY)
    )
    monkeypatch.setattr(profile, "_REGISTRY", dict(profile._REGISTRY))
    monkeypatch.setattr(profile, "_MODEL_DEFAULTS", dict(profile._MODEL_DEFAULTS))


def assert_blocks_equal(ba: mk.Block, bb: mk.Block) -> None:
    """Assert structural equality of two `Block` instances including `prefix`."""
    assert type(ba) is type(bb)
    assert (
        ba.name,
        ba.reduction,
        ba.input_norm,
        ba.prefix,
        ba.options,
    ) == (
        bb.name,
        bb.reduction,
        bb.input_norm,
        bb.prefix,
        bb.options,
    )
    if isinstance(ba, GQABlock):
        assert isinstance(bb, GQABlock)
        assert (ba.pool_layout, ba.qk_norm, ba.output_gate) == (
            bb.pool_layout,
            bb.qk_norm,
            bb.output_gate,
        )
    elif isinstance(ba, GDNBlock):
        assert isinstance(bb, GDNBlock)
        assert ba.prefix == bb.prefix
    elif isinstance(ba, MoEBlock):
        assert isinstance(bb, MoEBlock)
        assert (
            ba.shared_gate,
            ba._explicit_moe_tp,
            ba._explicit_e_slots,
            ba.moe_tp,
            ba.e_slots,
        ) == (
            bb.shared_gate,
            bb._explicit_moe_tp,
            bb._explicit_e_slots,
            bb.moe_tp,
            bb.e_slots,
        )
    elif isinstance(ba, Norm):
        assert isinstance(bb, Norm)
        assert (ba.zero_centered, ba.eps, ba.param_name) == (
            bb.zero_centered,
            bb.eps,
            bb.param_name,
        )
    elif isinstance(ba, Embedding):
        assert isinstance(bb, Embedding)
        assert (ba.param_name, ba.dtype) == (bb.param_name, bb.dtype)


def assert_specs_equal(spec_a: mk.ModelSpec, spec_b: mk.ModelSpec) -> None:
    """Assert structural equality of two `ModelSpec` instances."""
    assert type(spec_a) is type(spec_b) is mk.ModelSpec
    assert spec_a.name == spec_b.name
    assert type(spec_a.config) is type(spec_b.config)
    assert spec_a.config == spec_b.config
    assert spec_a.cache_backend == spec_b.cache_backend
    assert spec_a.layer_type.name == spec_b.layer_type.name
    assert len(spec_a.layer_type.slots) == len(spec_b.layer_type.slots)
    for sa, sb in zip(spec_a.layer_type.slots, spec_b.layer_type.slots, strict=True):
        if isinstance(sa, mk.SwitchSlot):
            assert isinstance(sb, mk.SwitchSlot)
            assert sa.name == sb.name
            assert sa.branch_counts == sb.branch_counts
            assert sa.period == sb.period
            for ba, bb in zip(sa.branches, sb.branches, strict=True):
                assert_blocks_equal(ba, bb)
        else:
            assert isinstance(sa, mk.Block) and isinstance(sb, mk.Block)
            assert_blocks_equal(sa, sb)
    for ba, bb in zip(spec_a.all_blocks(), spec_b.all_blocks(), strict=True):
        assert_blocks_equal(ba, bb)


class _DenseMLPBlock(MoEBlock):
    """Test-only dense-FFN override block (shared SwiGLU MLP without routed experts)."""

    routed_experts = False

    def reference(
        self,
        params: Mapping[str, Any],
        x: Any,
        state: Any = None,
        *,
        g: Any = None,
        ex: ExecSpec,
        layer_idx: int = 0,
        moe_tp: bool | None = None,
    ) -> tuple[jax.Array, Any]:
        del moe_tp
        geom = self._require_geometry(g)
        R, B = ex.R, ex.B
        ISR = int(geom.IS) // R
        x_f32 = x[:B].astype(F32)
        xb = x_f32.astype(BF16).astype(F32)
        sh_gate_l = params["sh_gate"][:, layer_idx]
        sh13_l = params["sh13"][:, layer_idx]
        sh13_s_l = params["sh13_s"][:, layer_idx]
        sh2_l = params["sh2"][:, layer_idx]
        sh2_s_l = params["sh2_s"][:, layer_idx]
        partials = []
        for r in range(R):
            sg = jax.nn.sigmoid(
                jnp.sum(x_f32 * sh_gate_l[r][None, :], axis=-1, keepdims=True)
            )
            g13 = (xb @ sh13_l[r].astype(F32)) * sh13_s_l[r]
            mid = (
                rnd(rnd(jax.nn.silu(rnd(g13[:, :ISR]))) * rnd(g13[:, ISR:]))
                .astype(BF16)
                .astype(F32)
            )
            part_r = (mid @ sh2_l[r].astype(F32)) * sh2_s_l[r] * sg
            partials.append(part_r.astype(BF16).astype(F32))
        return jnp.sum(jnp.stack(partials, axis=0), axis=0), state


def test_core_model_def_modules_have_no_qwen_literals_and_fit_line_budget():
    """Core `ModelDef` modules have no Qwen names and are <= 400 lines."""
    for mod in (geometry, schedule, reference, model_def, spec):
        path = Path(mod.__file__)
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        assert len(lines) <= 400, f"{path.name} has {len(lines)} lines (> 400)"
        if mod is not spec:
            low = text.lower()
            assert "qwen" not in low, f"{path.name} must not mention Qwen"

    # Block equality is identity-based (not value equality across prefixes)
    b1 = MoEBlock(prefix="mlp")
    b2 = MoEBlock(prefix="mlp")
    b3 = MoEBlock(prefix="other")
    assert b1 != b2 and b1 != b3
    assert_blocks_equal(b1, b2)
    with pytest.raises(AssertionError):
        assert_blocks_equal(b1, b3)


def test_from_hf_strict_resolution_and_nested_text_config():
    """`FromHF` requires known keys, declared HF field names, and `text_config`."""
    with pytest.raises(MegakernelConfigError, match="rope_thta"):
        FromHF(
            hidden="hidden_size",
            layers="num_hidden_layers",
            eps="rms_norm_eps",
            rope_thta="rope_theta",
        )

    decl = FromHF(
        hidden="custom_hidden_size",
        layers="custom_num_layers",
        vocab=("custom_vocab_size", 32000),
        eps="rms_norm_eps",
        rope_theta="rope_theta",
        rope_partial=("partial_rotary_factor", 0.5),
    )
    # Canonical name alone ("hidden") does not satisfy explicit "custom_hidden_size"
    with pytest.raises(ValueError, match="custom_hidden_size"):
        decl.resolve({"hidden": 2048, "custom_num_layers": 8, "rms_norm_eps": 1e-5})

    nested_cfg = {
        "model_type": " wrapper",
        "text_config": {
            "custom_hidden_size": 2048,
            "custom_num_layers": 8,
            "rms_norm_eps": 1e-5,
            "rope_parameters": {"rope_theta": 500_000.0},
        },
    }
    res = decl.resolve(nested_cfg)
    assert dict(res) == {
        "hidden": 2048,
        "layers": 8,
        "vocab": 32000,
        "eps": 1e-5,
        "rope_theta": 500_000.0,
        "rope_partial": 0.5,
    }


def test_schedule_validation_errors_name_layer_index_and_value(isolated_model_registry):
    """Unknown layer types, bad mixer order, norm mismatches, and OOB overrides fail."""
    attn = GQA.bind("self_attn")
    gdn = GDN.bind("linear_attn")
    moe = MoE.bind("mlp")

    # 1. Layers without a block in Every / FromHFLayerTypes / Schedule / overrides
    with pytest.raises(MegakernelConfigError, match=r"layer 0.*None"):
        Every(n=2, last=attn, rest=None)
    with pytest.raises(MegakernelConfigError, match=r"layer 1.*None"):
        Every(n=2, last=None, rest=gdn)  # type: ignore[arg-type]
    with pytest.raises(MegakernelConfigError, match=r"Every\(n=1\).*rest"):
        Every(n=1, last=attn, rest=gdn)
    with pytest.raises(MegakernelConfigError, match=r"layer 0.*None"):
        FromHFLayerTypes(
            field="layer_types",
            mapping={"full_attention": attn, "linear_attention": None},  # type: ignore[dict-item]
        )
    with pytest.raises(MegakernelConfigError, match=r"layer 1.*None"):
        Schedule(mixer=[gdn, None], ffn=moe)
    with pytest.raises(MegakernelConfigError, match=r"layer 2.*None"):
        Schedule(
            mixer=Every(n=2, last=attn, rest=gdn),
            ffn=moe,
            overrides={2: {"ffn": None}},
        )

    # 2. Reversed mixer order (GQA before GDN) fails closed naming layer 0
    with pytest.raises(MegakernelConfigError, match=r"layer 0.*gqa"):
        Schedule(mixer=Every(n=4, last=gdn, rest=attn), ffn=moe).resolve({}, 8)

    # 3. Unknown layer type in FromHFLayerTypes names the layer index and bad value
    sched_hf = Schedule(
        mixer=FromHFLayerTypes(
            field="layer_types",
            mapping={"full_attention": attn, "linear_attention": gdn},
        ),
        ffn=moe,
    )
    with pytest.raises(MegakernelConfigError, match=r"sliding_attention.*layer 2"):
        sched_hf.resolve(
            {
                "layer_types": [
                    "linear_attention",
                    "full_attention",
                    "sliding_attention",
                    "full_attention",
                ]
            },
            4,
        )

    # 4. Override indices out of range (negative or >= num_layers)
    with pytest.raises(MegakernelConfigError, match=r"-1.*out of range"):
        Schedule(
            mixer=Every(n=2, last=attn, rest=gdn),
            ffn=moe,
            overrides={-1: {"ffn": moe}},
        )
    sched_oob = Schedule(
        mixer=Every(n=2, last=attn, rest=gdn),
        ffn=moe,
        overrides={6: {"ffn": moe}},
    )
    with pytest.raises(MegakernelConfigError, match=r"layer index 6.*num_layers=4"):
        sched_oob.resolve({}, 4)

    # 5. Registration-time validation: prefers default_hf_config over FromHF tuple
    #    default for static_l, derives geometry/pool_layout, and catches norm mismatches
    valid_hf_cfg = {
        "hidden_size": 1024,
        "num_hidden_layers": 2,
        "rms_norm_eps": 1e-6,
        "rope_theta": 500_000.0,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "k_heads": 8,
        "v_heads": 16,
        "k_dim": 128,
        "v_dim": 128,
        "conv_kernel": 4,
        "num_experts": 128,
        "num_experts_per_tok": 4,
        "moe_intermediate_size": 1024,
        "shared_expert_intermediate_size": 1024,
        "layer_types": ["linear_attention", "full_attention"],
    }
    with pytest.raises(MegakernelConfigError, match=r"layer index 3.*num_layers=2"):

        @register_model(name="bad_oob_model")
        class _BadOOBModel(ModelDef):
            default_hf_config = valid_hf_cfg
            geometry = FromHF(
                hidden=("hidden_size", 1024),
                layers=("num_hidden_layers", 4),
                eps=("rms_norm_eps", 1e-6),
            )
            schedule = Schedule(
                mixer=Every(n=2, last=attn, rest=gdn),
                ffn=moe,
                overrides={3: {"ffn": moe}},
            )

    with pytest.raises(MegakernelConfigError, match=r"mamba2.*layer 1"):

        @register_model(name="bad_lt_model")
        class _BadLTModel(ModelDef):
            default_hf_config = {
                **valid_hf_cfg,
                "layer_types": ["linear_attention", "mamba2"],
            }
            geometry = FromHF(
                hidden="hidden_size",
                layers="num_hidden_layers",
                eps="rms_norm_eps",
            )
            schedule = sched_hf

    with pytest.raises(
        MegakernelConfigError, match=r"bad_missing_field_model.*num_attention_heads"
    ):

        @register_model(name="bad_missing_field_model")
        class _BadMissingFieldModel(ModelDef):
            default_hf_config = {
                "hidden_size": 1024,
                "num_hidden_layers": 2,
                "rms_norm_eps": 1e-6,
            }
            geometry = FromHF(
                hidden="hidden_size",
                layers="num_hidden_layers",
                eps="rms_norm_eps",
            )
            schedule = Schedule(mixer=Every(n=2, last=attn, rest=gdn), ffn=moe)

    with pytest.raises(MegakernelConfigError, match=r"zero_centered=False"):

        @register_model(name="bad_norm_zc_model")
        class _BadNormZCModel(ModelDef):
            default_hf_config = valid_hf_cfg
            geometry = FromHF(
                hidden="hidden_size",
                layers="num_hidden_layers",
                eps="rms_norm_eps",
            )
            schedule = Schedule(
                mixer=Every(n=2, last=attn, rest=gdn),
                ffn=moe,
                pre_norm=Norm.bind("input_layernorm", zero_centered=True),
                post_norm=Norm.bind("post_attention_layernorm", zero_centered=False),
            )


def test_model_registry_lookups_duplicates_and_profile_integration(
    isolated_model_registry,
):
    """Test `register_model`, `get_model`, `find_model`, `replace`, and profiles."""
    attn = GQA.bind("self_attn")
    gdn = GDN.bind("linear_attn")
    moe = MoE.bind("mlp")
    synth_prof = Profile(
        name="synth-default",
        model="synth_reg_test",
        numerics="exact",
        model_default=True,
    )

    @register_model(
        name="synth_reg_test",
        hf_architectures=("SynthForCausalLM", "SynthForConditionalGeneration"),
        hf_model_type=("synth_moe", "synth"),
    )
    class SynthRegModel(ModelDef):
        geometry = FromHF(
            hidden="hidden_size",
            layers="num_hidden_layers",
            eps="rms_norm_eps",
        )
        schedule = Schedule(mixer=Every(n=2, last=attn, rest=gdn), ffn=moe)
        profiles = {"default": synth_prof}

    # Idempotent re-registration of the same class
    assert (
        register_model(
            name="synth_reg_test",
            hf_architectures=("SynthForCausalLM", "SynthForConditionalGeneration"),
            hf_model_type=("synth_moe", "synth"),
        )(SynthRegModel)
        is SynthRegModel
    )

    assert "synth_reg_test" in list_models()
    assert get_model("synth_reg_test") is SynthRegModel
    assert get_model("SynthForCausalLM") is SynthRegModel
    assert get_model("synth_moe") is SynthRegModel
    assert (
        get_model({"architectures": ["SynthForConditionalGeneration"]}) is SynthRegModel
    )
    assert get_model({"text_config": {"model_type": "synth"}}) is SynthRegModel
    assert SynthRegModel.default_profile() is synth_prof
    assert SynthRegModel.get_profile("default") is synth_prof
    assert find_model("nonexistent_model") is None
    with pytest.raises(KeyError, match="Unknown megakernel model"):
        get_model("nonexistent_model")

    # Duplicate registration by a different class fails closed
    with pytest.raises(MegakernelConfigError, match="already registered"):

        @register_model(name="synth_reg_test")
        class _DuplicateModel(ModelDef):
            geometry = SynthRegModel.geometry._from_hf
            schedule = SynthRegModel.schedule

    # replace=True cleans up stale architecture and model_type keys
    @register_model(
        name="synth_reg_test",
        hf_architectures=("SynthV2ForCausalLM",),
        hf_model_type="synth_v2",
        replace=True,
    )
    class SynthRegModelV2(ModelDef):
        geometry = SynthRegModel.geometry._from_hf
        schedule = SynthRegModel.schedule

    assert get_model("synth_reg_test") is SynthRegModelV2
    assert get_model("SynthV2ForCausalLM") is SynthRegModelV2
    assert get_model("synth_v2") is SynthRegModelV2
    assert find_model("SynthForCausalLM") is None
    assert find_model("synth_moe") is None


def test_synthetic_model_def_every2_dense_override_cpu_kernel_vs_reference(
    isolated_model_registry,
):
    """Synthetic `ModelDef` (`Every(n=2)`) CPU kernel vs ref + override fail-closed."""
    synth_hf_cfg = {
        "model_type": "synth_hybrid",
        "architectures": ["SynthHybridForCausalLM"],
        "vocab_size": 64000,
        "hidden_size": 2048,
        "num_hidden_layers": 12,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500_000.0,
        "partial_rotary_factor": 0.25,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 8,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 256,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
    }

    @register_model(
        name="synth_hybrid",
        hf_architectures=("SynthHybridForCausalLM",),
        hf_model_type="synth_hybrid",
    )
    class SynthHybrid(ModelDef):
        default_hf_config = synth_hf_cfg
        geometry = FromHF(
            hidden="hidden_size",
            layers="num_hidden_layers",
            vocab="vocab_size",
            eps="rms_norm_eps",
            rope_theta="rope_theta",
            rope_partial=("partial_rotary_factor", 1.0),
        )
        attn = GQA.bind(
            prefix="self_attn",
            heads="num_attention_heads",
            kv_heads="num_key_value_heads",
            head_dim="head_dim",
            qk_norm=True,
            output_gate=True,
        )
        linear = GDN.bind(
            prefix="linear_attn",
            k_heads="linear_num_key_heads",
            v_heads="linear_num_value_heads",
            k_dim="linear_key_head_dim",
            v_dim="linear_value_head_dim",
            conv_kernel="linear_conv_kernel_dim",
        )
        moe = MoE.bind(
            prefix="mlp",
            experts="num_experts",
            top_k="num_experts_per_tok",
            inter="moe_intermediate_size",
            shared_inter="shared_expert_intermediate_size",
            shared_gate=True,
        )
        schedule = Schedule(
            mixer=Every(n=2, last=attn, rest=linear),
            ffn=moe,
            pre_norm=Norm.bind("input_layernorm", zero_centered=True),
            post_norm=Norm.bind("post_attention_layernorm", zero_centered=True),
        )
        embed = Embedding("model.embed_tokens")
        final_norm = Norm.bind("model.norm", zero_centered=True)
        lm_head = LMHead("lm_head", tied_to=None)

    assert SynthHybrid.default_profile() is DEFAULT_PROFILE
    g_full = SynthHybrid.geometry()
    assert (g_full.L, g_full.att_every, g_full.H, g_full.NH, g_full.E, g_full.TOPK) == (
        12,
        2,
        2048,
        4,
        256,
        2,
    )
    assert (g_full.NGDN, g_full.NATT) == (6, 6)

    ex = ExecSpec(
        R=2,
        B=2,
        T=1,
        ctx=256,
        interpret=True,
        tiles=dataclasses.replace(TINY_TILES, bank=512 << 10, nbd=6),
    )
    # shrink(R=2) preserves SynthHybrid's distinct NH=4, KH=4, VH=8, TOPK=2, I=IS=512
    g_tiny = SynthHybrid.shrink(R=ex.R)
    assert (
        g_tiny.L,
        g_tiny.att_every,
        g_tiny.NH,
        g_tiny.KH,
        g_tiny.VH,
        g_tiny.TOPK,
        g_tiny.I,
        g_tiny.IS,
    ) == (4, 2, 4, 4, 8, 2, 512, 512)
    g_tiny.check(ex)

    # Calling to_model_spec on an unbound ModelGeometry does not mutate it
    g_unbound = dataclasses.replace(g_tiny)
    assert getattr(g_unbound, "_model_def", None) is None
    ms = SynthHybrid.to_model_spec(g_unbound, R=ex.R)
    assert getattr(g_unbound, "_model_def", None) is None
    assert ms.name == "synth_hybrid"
    mixer_slot, ffn_slot = ms.layer_type.slots
    assert isinstance(mixer_slot, mk.SwitchSlot)
    assert mixer_slot.branch_counts == (1, 1) and mixer_slot.period == 2
    assert isinstance(ffn_slot, MoEBlock)

    play = SynthHybrid.pool_layout(g_tiny, R=ex.R)
    assert play.num_pools == 2 and play.recurrent_slots_per_pool == 1

    # Run CPU kernel vs composed reference on unmodified random weights
    pos, hid_l, w_l, st_l = qwen35.random_logical_inputs(
        jax.random.PRNGKey(42), g_tiny, ex
    )
    ref_out, ref_st = SynthHybrid.reference_step(g_tiny, ex, pos, hid_l, w_l, st_l)

    devs = jax.devices("cpu")[: ex.R]
    mesh = jax.sharding.Mesh(np.asarray(devs), ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = qwen35.pack_inputs(
        pos, hid_l, w_l, st_l, g_tiny, ex, mesh=mesh
    )
    step_fn, _ = mk.make_step(mesh, ms, ex)
    out_mk, st_out_mk = step_fn(pos_mk, hid_mk, w_mk, st_mk)
    out_mk_unfolded = mk.unfold(out_mk, ex.R)
    for r in range(ex.R):
        out_r = out_mk_unfolded[r, : ex.B].astype(F32)
        ref_f32 = ref_out.astype(F32)
        max_diff = float(jnp.max(jnp.abs(out_r - ref_f32)))
        cos = float(
            jnp.sum(out_r * ref_f32)
            / (jnp.linalg.norm(out_r) * jnp.linalg.norm(ref_f32))
        )
        assert max_diff < 0.05, f"rank {r}: max_diff={max_diff}"
        assert cos > 0.9999, f"rank {r}: cos={cos}"

    kc_mk = mk.unfold(st_out_mk["kc"], ex.R)
    vc_mk = mk.unfold(st_out_mk["vc"], ex.R)
    conv_mk = mk.unfold(st_out_mk["conv"], ex.R)
    rec_mk = mk.unfold(st_out_mk["rec"], ex.R)
    assert jnp.allclose(
        kc_mk.astype(F32), ref_st["kc"].astype(F32), atol=2e-2, rtol=1e-2
    )
    assert jnp.allclose(
        vc_mk.astype(F32), ref_st["vc"].astype(F32), atol=2e-2, rtol=1e-2
    )
    assert jnp.allclose(conv_mk, ref_st["conv"], atol=1e-2, rtol=1e-2)
    assert jnp.allclose(
        rec_mk.astype(F32), ref_st["rec"].astype(F32), atol=5e-2, rtol=2e-2
    )

    # Per-layer overrides that the single-LayerType kernel cannot represent fail closed
    # in to_model_spec naming the layer, the slot, and the binding.
    dense_mlp = _DenseMLPBlock.bind(
        prefix="mlp",
        experts="num_experts",
        top_k="num_experts_per_tok",
        inter="moe_intermediate_size",
        shared_inter="shared_expert_intermediate_size",
        shared_gate=True,
    )
    alt_linear = GDN.bind(
        prefix="alt_linear_attn",
        k_heads="linear_num_key_heads",
        v_heads="linear_num_value_heads",
        k_dim="linear_key_head_dim",
        v_dim="linear_value_head_dim",
        conv_kernel="linear_conv_kernel_dim",
    )

    @register_model(name="synth_dense_ov")
    class SynthDenseOverride(SynthHybrid):
        schedule = Schedule(
            mixer=Every(n=2, last=SynthHybrid.attn, rest=SynthHybrid.linear),
            ffn=SynthHybrid.moe,
            pre_norm=Norm.bind("input_layernorm", zero_centered=True),
            post_norm=Norm.bind("post_attention_layernorm", zero_centered=True),
            overrides={0: {"ffn": dense_mlp}},
        )

    with pytest.raises(
        MegakernelConfigError, match=r"layer 0.*slot 'ffn'.*_DenseMLPBlock"
    ):
        SynthDenseOverride.to_model_spec(g_tiny, R=ex.R)

    @register_model(name="synth_mixer_ov")
    class SynthMixerOverride(SynthHybrid):
        schedule = Schedule(
            mixer=Every(n=2, last=SynthHybrid.attn, rest=SynthHybrid.linear),
            ffn=SynthHybrid.moe,
            pre_norm=Norm.bind("input_layernorm", zero_centered=True),
            post_norm=Norm.bind("post_attention_layernorm", zero_centered=True),
            overrides={0: {"mixer": alt_linear}},
        )

    with pytest.raises(
        MegakernelConfigError, match=r"layer 0.*slot 'mixer'.*alt_linear_attn"
    ):
        SynthMixerOverride.to_model_spec(g_tiny, R=ex.R)


def _independent_qwen35_pool_layout(
    g: qwen35.Qwen35Geometry, *, R: int = 8
) -> mk.PoolLayout:
    """Build expected `PoolLayout` from `cache_decl()` + `derive_pool_layout`."""
    kv_decl = GQABlock(geometry=g).cache_decl(g)
    assert kv_decl is not None
    if g.att_every == 1:
        decls: list[Any] = [kv_decl] * g.NATT
    else:
        rec_decl = GDNBlock(geometry=g).cache_decl(g)
        assert rec_decl is not None
        decls = ([rec_decl] * (g.att_every - 1) + [kv_decl]) * g.NATT
    return mk.pool.derive_pool_layout(decls, R=R)


def _independent_qwen35_spec(
    g: qwen35.Qwen35Geometry, *, gdn_block: mk.Block | None = None, R: int = 8
) -> mk.ModelSpec:
    """Build expected `ModelSpec` independently of `Qwen35.to_model_spec`."""
    play = _independent_qwen35_pool_layout(g, R=R)
    gqa_blk = GQABlock(
        pool_layout=play, prefix="self_attn", qk_norm=True, output_gate=True
    )
    moe_blk = MoEBlock(prefix="mlp", shared_gate=True)
    if g.att_every == 1:
        mixer = mk.SwitchSlot("mixer", branches=(gqa_blk,), branch_counts=(1,))
    else:
        gdn_blk = GDNBlock(prefix="linear_attn") if gdn_block is None else gdn_block
        mixer = mk.SwitchSlot(
            "mixer", branches=(gdn_blk, gqa_blk), branch_counts=(g.att_every - 1, 1)
        )
    lt = mk.LayerType(name="qwen35_layer", slots=(mixer, moe_blk))
    backend = mk.pool.make_pool_backend(
        play, check=g.check, rec_chunk=GDNBlock.chunk_rows
    )
    return mk.ModelSpec(
        name=QWEN35_MODEL_NAME,
        config=g,
        layer_type=lt,
        cache_backend=backend,
    )


def test_qwen35_model_def_line_budget_docstrings_and_runtime_registry():
    """`models/qwen35.py` is <= 160 lines, keeps docstrings/kwargs, pins driver keys."""
    from vllm_torchtpu.megakernel_runtime import driver as driver_mod

    qwen_lines = Path(qwen35.__file__).read_text(encoding="utf-8").splitlines()
    assert len(qwen_lines) <= 160, f"qwen35.py has {len(qwen_lines)} lines (> 160)"
    for mod in (qwen35_weights, qwen35_pack, qwen35_pool):
        p = Path(mod.__file__)
        n_lines = len(p.read_text(encoding="utf-8").splitlines())
        assert n_lines <= 400, f"{p.name} has {n_lines} lines (> 400)"

    # Docstrings and explicit keyword parameters are preserved on Qwen3.5 entry points
    for fn in (
        qwen35.Qwen35Geometry.to_model_spec,
        qwen35.qwen35_spec,
        qwen35.build,
        qwen35.make_step,
        qwen35.make_pooled_step,
    ):
        assert fn.__doc__ and fn.__doc__.strip(), f"{fn.__name__} missing docstring"
    assert "checkpoint slots" in (qwen35_pool.__doc__ or "")
    assert "donate_states" in inspect.signature(qwen35.make_step).parameters
    pooled_params = inspect.signature(qwen35.make_pooled_step).parameters
    for kw_name in ("donate_pools", "fused_pool_io", "max_b_low", "min_b_hi"):
        assert kw_name in pooled_params

    # Model registry and profile lookups
    assert get_model(QWEN35_MODEL_NAME) is Qwen35
    assert get_model("Qwen3_5MoeForConditionalGeneration") is Qwen35
    assert get_model("Qwen3_5ForConditionalGeneration") is Qwen35
    assert get_model("qwen3_5_moe") is Qwen35
    assert get_model("qwen3_5_moe_text") is Qwen35
    assert (
        get_model(
            {
                "model_type": "qwen3_5_moe",
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "text_config": DEFAULT_HF_CONFIG,
            }
        )
        is Qwen35
    )
    assert Qwen35.default_profile() is PROFILE
    assert Qwen35.get_profile("candidate30").name == "qwen35-candidate30"

    # Runtime driver registry is pinned to the 4 serving keys
    assert set(driver_mod._MODEL_DRIVER_REGISTRY) == {
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
        "qwen3_5_moe",
        "qwen3_5",
    }

    # Dense Qwen3.5 config fails closed with "Qwen35 supports MoE checkpoints only"
    dense_cfg = {k: v for k, v in DEFAULT_HF_CONFIG.items() if k != "num_experts"}
    with pytest.raises(
        MegakernelConfigError, match="Qwen35 supports MoE checkpoints only"
    ):
        Qwen35.derive_geometry(dense_cfg)


def test_qwen35_model_def_derived_geometry_spec_and_pool_layout_equivalence():
    """Derived `Qwen35` geometry, `ModelSpec`, and `PoolLayout` match ground truth."""
    nested_vlm_cfg = {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": dict(DEFAULT_HF_CONFIG),
    }
    expected_full = qwen35.Qwen35Geometry()
    for cfg in (DEFAULT_HF_CONFIG, nested_vlm_cfg):
        g_derived = Qwen35.derive_geometry(cfg)
        assert type(g_derived) is qwen35.Qwen35Geometry
        assert g_derived == expected_full
        for f in dataclasses.fields(qwen35.Qwen35Geometry):
            assert getattr(g_derived, f.name) == getattr(expected_full, f.name), f.name

    tiny_hf_cfg = {
        **DEFAULT_HF_CONFIG,
        "num_hidden_layers": 4,
        "hidden_size": 1024,
        "num_attention_heads": 8,
        "linear_num_key_heads": 8,
        "linear_num_value_heads": 16,
        "num_experts": 128,
        "num_experts_per_tok": 4,
        "moe_intermediate_size": 1024,
        "shared_expert_intermediate_size": 1024,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
    }
    expected_tiny = qwen35.Qwen35Geometry(
        L=4, H=1024, NH=8, KH=8, VH=16, E=128, TOPK=4, I=1024, IS=1024
    )
    g_tiny_from_hf = Qwen35.derive_geometry(tiny_hf_cfg)
    assert type(g_tiny_from_hf) is qwen35.Qwen35Geometry
    assert g_tiny_from_hf == expected_tiny
    assert expected_tiny == qwen35.TINY
    for f in dataclasses.fields(qwen35.Qwen35Geometry):
        assert getattr(g_tiny_from_hf, f.name) == getattr(expected_tiny, f.name), f.name
        assert getattr(qwen35.TINY, f.name) == getattr(expected_tiny, f.name), f.name

    # shrink() is R-aware across R in {2, 4, 8}, including EPLB e_slots=80
    g_ep_eplb = dataclasses.replace(qwen35.FULL_EP, e_slots=80)
    for r_val in (2, 4, 8):
        for base_g in (qwen35.FULL, qwen35.FULL_EP, g_ep_eplb):
            g_shrunk = base_g.shrink(R=r_val)
            assert type(g_shrunk) is qwen35.Qwen35Geometry
            assert (g_shrunk.L, g_shrunk.att_every) == (8, 4)
            g_shrunk.check(ExecSpec(R=r_val, B=r_val, T=1, ctx=256))

    # Independent PoolLayout + ModelSpec equivalence across FULL, TINY, FULL_EP,
    # EPLB e_slots=72, and att_every=1
    att_only_tiny = dataclasses.replace(
        qwen35.TINY, att_every=1, KH=0, VH=0, KD=0, VD=0
    )
    att_only_hf_cfg = {**tiny_hf_cfg, "layer_types": ["full_attention"] * 4}
    eplb72 = dataclasses.replace(qwen35.FULL_EP, e_slots=72)
    spec_cases = [
        (qwen35.FULL, DEFAULT_HF_CONFIG, True, 0),
        (qwen35.TINY, tiny_hf_cfg, True, 0),
        (qwen35.FULL_EP, DEFAULT_HF_CONFIG, False, 0),
        (eplb72, DEFAULT_HF_CONFIG, False, 72),
        (att_only_tiny, att_only_hf_cfg, True, 0),
    ]
    for g, cfg, moe_tp_val, e_slots_val in spec_cases:
        expected_layout = _independent_qwen35_pool_layout(g, R=8)
        assert Qwen35.pool_layout(cfg, R=8) == expected_layout
        assert Qwen35.pool_layout(g, R=8) == expected_layout
        assert qwen35_pool.qwen35_pool_layout(g, R=8) == expected_layout

        expected_spec = _independent_qwen35_spec(g, R=8)
        spec_from_cfg = Qwen35.to_model_spec(
            cfg, moe_tp=moe_tp_val, e_slots=e_slots_val
        )
        spec_from_g = qwen35.qwen35_spec(g)
        assert_specs_equal(spec_from_cfg, expected_spec)
        assert_specs_equal(spec_from_g, expected_spec)

    g_att1 = dataclasses.replace(qwen35.TINY, att_every=1)
    assert qwen35_pool.qwen35_pool_layout(
        g_att1, R=8
    ) == _independent_qwen35_pool_layout(g_att1, R=8)
    assert_specs_equal(
        qwen35.qwen35_spec(g_att1), _independent_qwen35_spec(g_att1, R=8)
    )


@pytest.mark.parametrize(
    "t_val,moe_tp,pool_io",
    [
        (1, True, False),
        pytest.param(4, True, False, marks=pytest.mark.slow),
        (1, False, True),
    ],
)
def test_qwen35_model_def_composed_reference_matches_block_loop(
    t_val: int, moe_tp: bool, pool_io: bool
):
    """Composed `Qwen35.reference_step` is bitwise identical to explicit block loop."""
    g = dataclasses.replace(qwen35.TINY, moe_tp=moe_tp)
    ex = ExecSpec(R=2, B=2 * t_val, T=t_val, ctx=128)
    pos, hid, w_l, st_l = qwen35.random_logical_inputs(
        jax.random.PRNGKey(17 + t_val), g, ex
    )
    out_composed, st_composed = Qwen35.reference_step(
        g, ex, pos, hid, w_l, st_l, pool_io=pool_io
    )

    # Verify against an independent explicit layer loop (both out and final states)
    in_norm_blk = Norm("input_layernorm", geometry=g, zero_centered=True, eps=g.eps)
    post_norm_blk = Norm(
        "post_attention_layernorm", geometry=g, zero_centered=True, eps=g.eps
    )
    final_norm_blk = Norm("model.norm", geometry=g, zero_centered=True, eps=g.eps)
    gqa_blk = GQABlock(geometry=g)
    gdn_blk = GDNBlock(geometry=g)
    moe_blk = MoEBlock(geometry=g)
    lay_gdn = GDNBlock.lin_in_layout(g, ex)
    cr, hr = mk.rup(3 * ex.B, 8), mk.rup(3 + ex.T, 8)
    conv_in = st_l["conv"].astype(BF16).astype(F32) if pool_io else st_l["conv"]
    rec_in = st_l["rec"].reshape(
        ex.R, g.NGDN, ex.NS, lay_gdn.vh, lay_gdn.KD, lay_gdn.VD
    )
    step_state: dict[str, Any] = {
        "kc": st_l["kc"],
        "vc": st_l["vc"],
        "conv": conv_in,
        "rec": rec_in,
        "conv_out": (
            jnp.zeros((ex.R, g.NGDN, cr, lay_gdn.QKV), dtype=F32)
            if ex.T == 1
            else jnp.zeros((ex.R, g.NGDN, ex.NS, hr, lay_gdn.QKV), dtype=F32)
        ),
        "rec_out": jnp.zeros(
            (ex.R, g.NGDN, ex.B * lay_gdn.vh, lay_gdn.KD, lay_gdn.VD), dtype=BF16
        ),
    }
    res = hid[: ex.B].astype(F32)
    for lyr in range(g.L):
        x_m, _ = in_norm_blk.reference(w_l["in_norm"][lyr], res, g=g)
        if (lyr + 1) % g.att_every == 0:
            d_m, step_state = gqa_blk.reference(
                w_l,
                x_m,
                step_state,
                g=g,
                ex=ex,
                pos=pos,
                att_idx=lyr // g.att_every,
                layer_idx=lyr,
                paged_kv=False,
            )
        else:
            d_m, step_state = gdn_blk.reference(
                w_l,
                x_m,
                step_state,
                g=g,
                ex=ex,
                gdn_idx=lyr - (lyr + 1) // g.att_every,
                layer_idx=lyr,
                multi_layer=True,
            )
        res = rnd(res + d_m)
        x_f, _ = post_norm_blk.reference(w_l["post_norm"][lyr], res, g=g)
        d_f, step_state = moe_blk.reference(
            w_l, x_f, step_state, g=g, ex=ex, layer_idx=lyr
        )
        res = rnd(res + d_f)
    out_hand, _ = final_norm_blk.reference(w_l["final_norm"], res, g=g, out_dtype=BF16)
    conv_out = (
        step_state["conv_out"].astype(BF16).astype(F32)
        if pool_io
        else step_state["conv_out"]
    )
    rec_out = step_state["rec_out"]
    st_hand: dict[str, Any] = {"kc": step_state["kc"], "vc": step_state["vc"]}
    if ex.VER:
        conv_pad = (
            jnp.pad(
                conv_in,
                ((0, 0), (0, 0), (0, 0), (0, hr - conv_in.shape[-2]), (0, 0)),
            )
            if conv_in.ndim == 5 and conv_in.shape[-2] < hr
            else conv_in
        )
        st_hand["conv"] = conv_pad
        st_hand["rec"] = rec_in.reshape(
            ex.R, g.NGDN, ex.NS * lay_gdn.vh, lay_gdn.KD, lay_gdn.VD
        )
        st_hand["conv_ckpt"] = conv_out
        st_hand["rec_ckpt"] = rec_out
    else:
        st_hand["conv"] = conv_out
        st_hand["rec"] = rec_out

    assert jnp.array_equal(out_composed, out_hand)
    assert set(st_composed) == set(st_hand)
    for k in st_hand:
        assert jnp.array_equal(st_composed[k], st_hand[k]), k
