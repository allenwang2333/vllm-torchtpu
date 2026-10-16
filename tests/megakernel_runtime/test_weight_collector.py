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
"""Unit tests for `megakernel_runtime.weight_collector`."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.megakernel import planner
from vllm_torchtpu.megakernel.blocks.gdn import GDNBlock
from vllm_torchtpu.megakernel.models import qwen35
from vllm_torchtpu.megakernel.spec import TINY_TILES, ExecSpec
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.megakernel_runtime.weight_collector import (
    _own,
    _stack,
    find_backbone,
    pack_weights,
    reshard_layer_experts_ep_to_tp,
)


def _make_fp8_linear(
    in_dim: int, out_dim: int, *, flipped: bool = True
) -> SimpleNamespace:
    if flipped:
        w = torch.randint(1, 120, (in_dim, out_dim), dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
    else:
        w = torch.randint(1, 120, (out_dim, in_dim), dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
    s = torch.rand((out_dim, 1), dtype=torch.float32) + 0.1
    ns = SimpleNamespace(
        weight=torch.nn.Parameter(w, requires_grad=False),
        weight_scale=torch.nn.Parameter(s, requires_grad=False),
    )
    setattr(ns, WEIGHT_FLIPPED_ATTR, flipped)
    return ns


def make_synthetic_qwen35_backbone(
    g: qwen35.Qwen35Geometry,
    ex: ExecSpec,
    *,
    fused_mlp: bool = True,
    fused_qkv: bool = False,
) -> SimpleNamespace:
    """Construct a synthetic Qwen3.5 backbone matching `g` and `ex`."""
    gdn_lay = GDNBlock.lin_in_layout(g, ex)
    nh = g.nh(ex.R)
    nai = g.NAI(ex.R)
    q_dim = 2 * nh * g.HD
    kv_dim = (nai - q_dim) // 2
    ew = g.EW(ex.R)
    ir = g.IR(ex.R)
    layers = []

    for l_idx in range(g.L):
        is_att = (l_idx % g.att_every) == (g.att_every - 1)
        ltype = "full_attention" if is_att else "linear_attention"

        if is_att:
            if fused_qkv:
                self_attn = SimpleNamespace(
                    qkv_proj=_make_fp8_linear(g.H, nai),
                    o_proj=_make_fp8_linear(nh * g.HD, g.H),
                    q_norm=SimpleNamespace(
                        weight=torch.nn.Parameter(
                            torch.randn(g.HD, dtype=torch.bfloat16), requires_grad=False
                        )
                    ),
                    k_norm=SimpleNamespace(
                        weight=torch.nn.Parameter(
                            torch.randn(g.HD, dtype=torch.bfloat16), requires_grad=False
                        )
                    ),
                    attn=SimpleNamespace(
                        layer_name=f"model.layers.{l_idx}.self_attn.attn", kv_cache=[]
                    ),
                )
            else:
                self_attn = SimpleNamespace(
                    q_proj=_make_fp8_linear(g.H, q_dim),
                    k_proj=_make_fp8_linear(g.H, kv_dim, flipped=False),
                    v_proj=_make_fp8_linear(g.H, kv_dim),
                    o_proj=_make_fp8_linear(nh * g.HD, g.H),
                    q_norm=SimpleNamespace(
                        weight=torch.nn.Parameter(
                            torch.randn(g.HD, dtype=torch.bfloat16), requires_grad=False
                        )
                    ),
                    k_norm=SimpleNamespace(
                        weight=torch.nn.Parameter(
                            torch.randn(g.HD, dtype=torch.bfloat16), requires_grad=False
                        )
                    ),
                    attn=SimpleNamespace(
                        layer_name=f"model.layers.{l_idx}.self_attn.attn", kv_cache=[]
                    ),
                )
            linear_attn = None
        else:
            in_qkvz = _make_fp8_linear(g.H, gdn_lay.QKV + gdn_lay.Z)
            in_ba = SimpleNamespace(
                weight=torch.nn.Parameter(
                    torch.randn((2 * gdn_lay.vh, g.H), dtype=torch.bfloat16) * 0.05,
                    requires_grad=False,
                )
            )
            linear_attn = SimpleNamespace(
                prefix=f"model.layers.{l_idx}.linear_attn",
                in_proj_qkvz=in_qkvz,
                in_proj_ba=in_ba,
                out_proj=_make_fp8_linear(gdn_lay.Z, g.H),
                conv1d=SimpleNamespace(
                    weight=torch.nn.Parameter(
                        torch.randn((gdn_lay.QKV, 1, 4), dtype=torch.bfloat16),
                        requires_grad=False,
                    )
                ),
                A_log=torch.nn.Parameter(
                    torch.randn(gdn_lay.vh, dtype=torch.float32), requires_grad=False
                ),
                dt_bias=torch.nn.Parameter(
                    torch.randn(gdn_lay.vh, dtype=torch.bfloat16), requires_grad=False
                ),
                norm=SimpleNamespace(
                    weight=torch.nn.Parameter(
                        torch.randn(gdn_lay.VD, dtype=torch.float32),
                        requires_grad=False,
                    )
                ),
                kv_cache=[],
            )
            self_attn = None

        w13 = torch.randint(1, 120, (ew, g.H, 2 * ir), dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
        w2 = torch.randint(1, 120, (ew, ir, g.H), dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
        w13_s = torch.rand((ew, 1, 2 * ir), dtype=torch.float32) + 0.1
        w2_s = torch.rand((ew, 1, g.H), dtype=torch.float32) + 0.1
        experts = SimpleNamespace(
            w13_weight=torch.nn.Parameter(w13, requires_grad=False),
            w2_weight=torch.nn.Parameter(w2, requires_grad=False),
            w13_weight_scale=torch.nn.Parameter(w13_s, requires_grad=False),
            w2_weight_scale=torch.nn.Parameter(w2_s, requires_grad=False),
        )

        if fused_mlp:
            fused_gu_cols = g.E + 2 * g.IS
            fused_w = torch.randint(
                1, 120, (g.H, fused_gu_cols), dtype=torch.uint8
            ).view(torch.float8_e4m3fn)
            fused_s = torch.rand((1, fused_gu_cols), dtype=torch.float32) + 0.1
            shared_expert = SimpleNamespace(
                down_proj=_make_fp8_linear(g.IS, g.H),
            )
            shared_expert_gate = SimpleNamespace(
                weight=torch.nn.Parameter(
                    torch.randn((1, g.H), dtype=torch.bfloat16), requires_grad=False
                )
            )
            mlp = SimpleNamespace(
                experts=experts,
                _fused_gate_gu_fp8_weight=fused_w,
                _fused_gate_gu_fp8_scale=fused_s,
                shared_expert=shared_expert,
                shared_expert_gate=shared_expert_gate,
            )
        else:
            gate = SimpleNamespace(
                weight=torch.nn.Parameter(
                    torch.randn((g.E, g.H), dtype=torch.bfloat16), requires_grad=False
                )
            )
            shared_expert = SimpleNamespace(
                gate_up_proj=_make_fp8_linear(g.H, 2 * g.IS),
                down_proj=_make_fp8_linear(g.IS, g.H, flipped=False),
            )
            shared_expert_gate = SimpleNamespace(
                weight=torch.nn.Parameter(
                    torch.randn((1, g.H), dtype=torch.bfloat16), requires_grad=False
                )
            )
            mlp = SimpleNamespace(
                experts=experts,
                gate=gate,
                shared_expert=shared_expert,
                shared_expert_gate=shared_expert_gate,
            )

        layer = SimpleNamespace(
            layer_type=ltype,
            input_layernorm=SimpleNamespace(
                weight=torch.nn.Parameter(
                    torch.randn(g.H, dtype=torch.bfloat16), requires_grad=False
                )
            ),
            post_attention_layernorm=SimpleNamespace(
                weight=torch.nn.Parameter(
                    torch.randn(g.H, dtype=torch.bfloat16), requires_grad=False
                )
            ),
            linear_attn=linear_attn,
            self_attn=self_attn,
            mlp=mlp,
        )
        layers.append(layer)

    embed = torch.nn.Embedding(64, g.H, dtype=torch.bfloat16)
    norm = SimpleNamespace(
        weight=torch.nn.Parameter(
            torch.randn(g.H, dtype=torch.bfloat16), requires_grad=False
        )
    )
    return SimpleNamespace(embed_tokens=embed, layers=layers, norm=norm)


def test_own_and_stack_allocate_base_buffers() -> None:
    """Verify `_own` and `_stack` return non-view base buffers (`offset == 0`)."""
    base = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    sliced = base[1:3, 2:6]
    assert sliced._is_view() or sliced.storage_offset() != 0

    owned = _own(sliced)
    assert not owned._is_view()
    assert owned.storage_offset() == 0
    assert torch.equal(owned, sliced)

    stacked = _stack([sliced, sliced])
    assert not stacked._is_view()
    assert stacked.storage_offset() == 0
    assert stacked.shape == (2, 2, 4)


def test_find_backbone_wrappers() -> None:
    """Verify `find_backbone` locates `(layers, norm)` across nested vLLM wrappers."""
    inner = SimpleNamespace(layers=[1], norm=object())
    for wrapped in (
        inner,
        SimpleNamespace(model=inner),
        SimpleNamespace(model=SimpleNamespace(model=inner)),
        SimpleNamespace(language_model=SimpleNamespace(model=inner)),
        SimpleNamespace(
            model=SimpleNamespace(language_model=SimpleNamespace(model=inner))
        ),
    ):
        assert find_backbone(wrapped) is inner

    with pytest.raises(ValueError, match="Cannot locate decoder backbone"):
        find_backbone(SimpleNamespace(foo=1))


@pytest.mark.parametrize("stack_e,fused_mlp", [(False, True), (True, False)])
def test_pack_weights_shapes_base_buffers_and_expert_aliasing(
    stack_e: bool, fused_mlp: bool
) -> None:
    """Verify `pack_weights` shapes, base buffers, EPLB `ecopy`, and expert aliasing."""
    g = qwen35.Qwen35Geometry(
        L=4,
        att_every=4,
        H=256,
        NH=4,
        NKV=2,
        HD=256,
        ROT=64,
        KH=2,
        VH=4,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=256,
        IS=256,
        moe_tp=False,
        e_slots=68,
    )
    tiles = replace(TINY_TILES, bank=512 << 10)
    ex = ExecSpec(
        R=2,
        B=4,
        T=1,
        ctx=256,
        paged_kv=True,
        num_pages=64,
        max_pages_per_seq=2,
        stack_e=stack_e,
        tiles=tiles,
        interpret=True,
    )
    backbone = make_synthetic_qwen35_backbone(
        g, ex, fused_mlp=fused_mlp, fused_qkv=not fused_mlp
    )
    packed = pack_weights(backbone, g, ex, tp_rank=1)
    plan: planner.Plan = packed["plan"]
    w_shapes = planner.weight_shapes(qwen35.qwen35_spec(g), ex, plan)

    for k in plan.w_names:
        val = packed[k]
        if k in plan.dynamic_kinds and not stack_e:
            assert isinstance(val, tuple) and len(val) == g.L
            for l_idx in range(g.L):
                t_l = val[l_idx]
                orig = (
                    backbone.layers[l_idx].mlp.experts.w13_weight
                    if k == "e13"
                    else backbone.layers[l_idx].mlp.experts.w2_weight
                )
                assert t_l.data_ptr() == orig.data_ptr()
                assert t_l.storage_offset() == 0
                assert t_l.ndim == 3
        else:
            exp_shape, _ = w_shapes[k]
            assert isinstance(val, torch.Tensor)
            assert not val._is_view()
            assert val.storage_offset() == 0
            assert tuple(val.shape) == exp_shape

    # Verify `ecopy` is populated in `smalls` when `e_slots > ER`
    ecopy_off, ecopy_w = plan.smalls_layout["ecopy"]
    ecopy_slice = packed["smalls"][:, 0, ecopy_off : ecopy_off + ecopy_w]
    assert torch.any(ecopy_slice > 0.0)


def test_reshard_layer_experts_ep_to_tp() -> None:
    """Verify `reshard_layer_experts_ep_to_tp` slices experts and updates MoE flags."""
    g = qwen35.Qwen35Geometry(
        L=2,
        att_every=2,
        H=256,
        NH=4,
        NKV=2,
        HD=256,
        ROT=64,
        KH=2,
        VH=4,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=256,
        IS=256,
        moe_tp=True,
        e_slots=0,
    )
    ex_ep = ExecSpec(
        R=2, B=4, T=1, ctx=256, stack_e=False, tiles=TINY_TILES, interpret=True
    )
    g_ep = replace(g, moe_tp=False, e_slots=68)
    backbone = make_synthetic_qwen35_backbone(g_ep, ex_ep)
    mlp = backbone.layers[0].mlp
    mlp.experts.moe_config = SimpleNamespace(
        use_ep=True,
        is_sequence_parallel=True,
        moe_parallel_config=SimpleNamespace(
            use_ep=True,
            ep_size=2,
            ep_rank=1,
            tp_size=1,
            tp_rank=0,
            sp_size=2,
            is_sequence_parallel=True,
        ),
    )

    # Simulate 2-rank all_gather by repeating primary `ER=64` experts twice -> `E=128`
    reshard_layer_experts_ep_to_tp(
        mlp,
        tp_rank=1,
        g=g,
        ex=ex_ep,
        all_gather_fn=lambda t: torch.cat([t, t], dim=0),
    )
    ir = g.I // ex_ep.R
    assert tuple(mlp.experts.w13_weight.shape) == (g.E, g.H, 2 * ir)
    assert tuple(mlp.experts.w2_weight.shape) == (g.E, ir, g.H)
    assert not mlp.experts.w13_weight.data._is_view()
    assert not mlp.experts.w2_weight.data._is_view()
    assert mlp.experts.moe_config.use_ep is False
    assert mlp.experts.moe_config.moe_parallel_config.tp_size == 2
    assert mlp.experts.moe_config.moe_parallel_config.tp_rank == 1
    assert mlp.experts.moe_config.moe_parallel_config.ep_size == 1
