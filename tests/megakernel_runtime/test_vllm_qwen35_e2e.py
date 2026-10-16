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
"""End-to-end CPU tests for `MegakernelDriver`, metadata prep, and speculative slots."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_force_host_platform_device_count" not in _xla_flags:
    os.environ["XLA_FLAGS"] = (
        f"{_xla_flags} --xla_force_host_platform_device_count=8".strip()
    )

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

import vllm_torchtpu.megakernel_runtime.driver as driver_mod  # noqa: E402
from tests.megakernel_runtime.test_weight_collector import (  # noqa: E402
    _jax_to_torch,
    make_synthetic_qwen35_backbone,
)
from vllm_torchtpu.megakernel.models import qwen35, qwen35_pool  # noqa: E402
from vllm_torchtpu.megakernel.spec import TINY_TILES, ExecSpec  # noqa: E402
from vllm_torchtpu.megakernel_runtime.driver import (  # noqa: E402
    MegakernelDriver,
    MKABCalibrator,
    parse_megakernel_config,
    parse_mk_buckets,
    parse_mk_verify_buckets,
    prepare_step_inputs_eager,
    prepare_step_metadata,
    select_bucket,
)
from vllm_torchtpu.megakernel_runtime.weight_collector import (  # noqa: E402
    WEIGHT_FLIPPED_ATTR,
)


def _cpu_mesh(r: int = 1) -> Mesh:
    return Mesh(np.array(jax.devices("cpu")[:r]), ("tp",))


def _tiny_geom(L: int = 4, att_every: int = 4) -> qwen35.Qwen35Geometry:
    return qwen35.Qwen35Geometry(
        L=L,
        att_every=att_every,
        H=256,
        NH=4,
        NKV=1,
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
        e_slots=0,
    )


def _tiny_exec(
    R: int = 1,
    B: int = 2,
    T: int = 1,
    ctx: int = 256,
    num_pages: int = 96,
) -> ExecSpec:
    tiles = replace(TINY_TILES, bank=512 << 10, kv_blk=min(TINY_TILES.kv_blk, ctx))
    return ExecSpec(
        R=R,
        B=B,
        T=T,
        ctx=ctx,
        paged_kv=True,
        num_pages=num_pages,
        max_pages_per_seq=2,
        stack_e=False,
        tiles=tiles,
        interpret=True,
    )


def _torch_to_jax(t: torch.Tensor, sharding: NamedSharding) -> jax.Array:
    if t.dtype == torch.float8_e4m3fn:
        arr = jnp.asarray(t.view(torch.uint8).numpy()).view(jnp.float8_e4m3fn)
    elif t.dtype == torch.bfloat16:
        arr = jnp.asarray(t.view(torch.uint16).numpy()).view(jnp.bfloat16)
    else:
        arr = jnp.asarray(t.numpy())
    return jax.device_put(arr, sharding)


def _allocate_aliased_pools(
    backbone: Any, g: qwen35.Qwen35Geometry, num_pages: int
) -> list[torch.Tensor]:
    raw_pools_t: list[torch.Tensor] = []
    for r in range(g.NATT):
        pool_u8 = torch.zeros((num_pages, 2, 64, 4, 128), dtype=torch.uint8)
        pool = pool_u8.view(torch.float8_e4m3fn)
        att_layer = backbone.layers[g.att_every * r + 3]
        att_layer.self_attn.attn.kv_cache = [pool]
        for g_idx in range(3):
            la_layer = backbone.layers[g.att_every * r + g_idx]
            la_layer.linear_attn.kv_cache = [pool]
        raw_pools_t.append(pool)
    return raw_pools_t


def _make_backbone_from_logical_weights(
    g: qwen35.Qwen35Geometry,
    ex: ExecSpec,
    w_logical: dict[str, Any],
    tp_rank: int = 0,
) -> SimpleNamespace:
    """Build a synthetic PyTorch backbone whose weights match `w_logical`."""
    gdn_lay = qwen35.GDNBlock.lin_in_layout(g, ex)
    qkvz_dim = gdn_lay.QKV + gdn_lay.Z
    layers: list[Any] = []
    for l_idx in range(g.L):
        is_att = (l_idx % g.att_every) == (g.att_every - 1)
        ltype = "full_attention" if is_att else "linear_attention"
        if is_att:
            a_idx = l_idx // g.att_every
            qkv_proj = SimpleNamespace(
                weight=torch.nn.Parameter(
                    _jax_to_torch(w_logical["att_in"][tp_rank, a_idx]),
                    requires_grad=False,
                ),
                weight_scale=torch.nn.Parameter(
                    _jax_to_torch(w_logical["att_in_s"][tp_rank, a_idx]),
                    requires_grad=False,
                ),
            )
            setattr(qkv_proj, WEIGHT_FLIPPED_ATTR, True)
            o_proj = SimpleNamespace(
                weight=torch.nn.Parameter(
                    _jax_to_torch(w_logical["att_out"][tp_rank, a_idx]),
                    requires_grad=False,
                ),
                weight_scale=torch.nn.Parameter(
                    _jax_to_torch(w_logical["att_out_s"][tp_rank, a_idx]),
                    requires_grad=False,
                ),
            )
            setattr(o_proj, WEIGHT_FLIPPED_ATTR, True)
            self_attn = SimpleNamespace(
                qkv_proj=qkv_proj,
                o_proj=o_proj,
                q_norm=SimpleNamespace(
                    weight=torch.nn.Parameter(
                        _jax_to_torch(w_logical["qn"][tp_rank, a_idx]),
                        requires_grad=False,
                    )
                ),
                k_norm=SimpleNamespace(
                    weight=torch.nn.Parameter(
                        _jax_to_torch(w_logical["kn"][tp_rank, a_idx]),
                        requires_grad=False,
                    )
                ),
                attn=SimpleNamespace(
                    layer_name=f"model.layers.{l_idx}.self_attn.attn", kv_cache=[]
                ),
            )
            linear_attn = None
        else:
            o_gdn = l_idx - (l_idx + 1) // g.att_every
            w_lin_full = _jax_to_torch(w_logical["lin_in"][tp_rank, o_gdn])
            s_lin_full = _jax_to_torch(w_logical["lin_in_s"][tp_rank, o_gdn])
            w_qkvz = w_lin_full[:, :qkvz_dim].contiguous()
            s_qkvz = s_lin_full[:qkvz_dim].contiguous()
            w_b = w_lin_full[:, gdn_lay.BA : gdn_lay.BA + gdn_lay.vh]
            w_a = w_lin_full[:, gdn_lay.A_OFF : gdn_lay.A_OFF + gdn_lay.vh]
            w_ba = torch.cat([w_b, w_a], dim=1).contiguous()
            s_b = s_lin_full[gdn_lay.BA : gdn_lay.BA + gdn_lay.vh]
            s_a = s_lin_full[gdn_lay.A_OFF : gdn_lay.A_OFF + gdn_lay.vh]
            s_ba = torch.cat([s_b, s_a], dim=0).contiguous()

            in_proj_qkvz = SimpleNamespace(
                weight=torch.nn.Parameter(w_qkvz, requires_grad=False),
                weight_scale=torch.nn.Parameter(s_qkvz, requires_grad=False),
            )
            setattr(in_proj_qkvz, WEIGHT_FLIPPED_ATTR, True)
            in_proj_ba = SimpleNamespace(
                weight=torch.nn.Parameter(w_ba, requires_grad=False),
                weight_scale=torch.nn.Parameter(s_ba, requires_grad=False),
            )
            setattr(in_proj_ba, WEIGHT_FLIPPED_ATTR, True)
            out_proj = SimpleNamespace(
                weight=torch.nn.Parameter(
                    _jax_to_torch(w_logical["lin_out"][tp_rank, o_gdn]),
                    requires_grad=False,
                ),
                weight_scale=torch.nn.Parameter(
                    _jax_to_torch(w_logical["lin_out_s"][tp_rank, o_gdn]),
                    requires_grad=False,
                ),
            )
            setattr(out_proj, WEIGHT_FLIPPED_ATTR, True)
            conv_4_qkv = _jax_to_torch(w_logical["conv"][tp_rank, o_gdn]).reshape(
                4, gdn_lay.QKV
            )
            conv_w = conv_4_qkv.transpose(0, 1).unsqueeze(1).contiguous()
            linear_attn = SimpleNamespace(
                prefix=f"model.layers.{l_idx}.linear_attn",
                in_proj_qkvz=in_proj_qkvz,
                in_proj_ba=in_proj_ba,
                out_proj=out_proj,
                conv1d=SimpleNamespace(
                    weight=torch.nn.Parameter(conv_w, requires_grad=False)
                ),
                A_log=torch.nn.Parameter(
                    _jax_to_torch(w_logical["alog"][tp_rank, o_gdn]),
                    requires_grad=False,
                ),
                dt_bias=torch.nn.Parameter(
                    _jax_to_torch(w_logical["dtb"][tp_rank, o_gdn]),
                    requires_grad=False,
                ),
                norm=SimpleNamespace(
                    weight=torch.nn.Parameter(
                        _jax_to_torch(w_logical["gnorm"][tp_rank, o_gdn]),
                        requires_grad=False,
                    )
                ),
                kv_cache=[],
            )
            self_attn = None

        experts = SimpleNamespace(
            w13_weight=torch.nn.Parameter(
                _jax_to_torch(w_logical["e13"][tp_rank, l_idx]), requires_grad=False
            ),
            w2_weight=torch.nn.Parameter(
                _jax_to_torch(w_logical["e2"][tp_rank, l_idx]), requires_grad=False
            ),
            w13_weight_scale=torch.nn.Parameter(
                _jax_to_torch(w_logical["e13_s"][tp_rank, l_idx]), requires_grad=False
            ),
            w2_weight_scale=torch.nn.Parameter(
                _jax_to_torch(w_logical["e2_s"][tp_rank, l_idx]), requires_grad=False
            ),
        )
        gu_proj = SimpleNamespace(
            weight=torch.nn.Parameter(
                _jax_to_torch(w_logical["sh13"][tp_rank, l_idx]), requires_grad=False
            ),
            weight_scale=torch.nn.Parameter(
                _jax_to_torch(w_logical["sh13_s"][tp_rank, l_idx]), requires_grad=False
            ),
        )
        setattr(gu_proj, WEIGHT_FLIPPED_ATTR, True)
        down_proj = SimpleNamespace(
            weight=torch.nn.Parameter(
                _jax_to_torch(w_logical["sh2"][tp_rank, l_idx]), requires_grad=False
            ),
            weight_scale=torch.nn.Parameter(
                _jax_to_torch(w_logical["sh2_s"][tp_rank, l_idx]), requires_grad=False
            ),
        )
        setattr(down_proj, WEIGHT_FLIPPED_ATTR, True)
        mlp = SimpleNamespace(
            experts=experts,
            gate=SimpleNamespace(
                weight=torch.nn.Parameter(
                    _jax_to_torch(w_logical["router"][l_idx])
                    .transpose(0, 1)
                    .contiguous(),
                    requires_grad=False,
                )
            ),
            shared_expert=SimpleNamespace(gate_up_proj=gu_proj, down_proj=down_proj),
            shared_expert_gate=SimpleNamespace(
                weight=torch.nn.Parameter(
                    _jax_to_torch(w_logical["sh_gate"][tp_rank, l_idx]),
                    requires_grad=False,
                )
            ),
        )
        layers.append(
            SimpleNamespace(
                layer_type=ltype,
                input_layernorm=SimpleNamespace(
                    weight=torch.nn.Parameter(
                        _jax_to_torch(w_logical["in_norm"][l_idx]), requires_grad=False
                    )
                ),
                post_attention_layernorm=SimpleNamespace(
                    weight=torch.nn.Parameter(
                        _jax_to_torch(w_logical["post_norm"][l_idx]),
                        requires_grad=False,
                    )
                ),
                linear_attn=linear_attn,
                self_attn=self_attn,
                mlp=mlp,
            )
        )
    embed = torch.nn.Embedding(64, g.H, dtype=torch.bfloat16)
    norm = SimpleNamespace(
        weight=torch.nn.Parameter(
            _jax_to_torch(w_logical["final_norm"]), requires_grad=False
        )
    )
    return SimpleNamespace(embed_tokens=embed, layers=layers, norm=norm)


def test_bucket_parsing_and_selection() -> None:
    """Verify decode/verify bucket parsing, sorting, config parsing, and selection."""
    assert parse_mk_buckets("16, 4, 8, 4") == (4, 8, 16)
    assert parse_mk_buckets("16:4:8") == (4, 8, 16)
    assert parse_mk_buckets("0") == ()
    assert parse_mk_buckets("off") == ()
    with pytest.raises(ValueError, match="invalid decode buckets"):
        parse_mk_buckets("128")

    assert parse_mk_verify_buckets("16, 4, 8") == (4, 8, 16)
    assert parse_mk_verify_buckets("16:4:8") == (4, 8, 16)
    assert parse_mk_verify_buckets("none") == ()
    with pytest.raises(ValueError, match="invalid verify_buckets"):
        parse_mk_verify_buckets("6")

    # Single VLLM_TPU_MEGAKERNEL string forms + additional_config["megakernel"]
    assert not parse_megakernel_config(raw="")["enabled"]
    assert not parse_megakernel_config(raw="0")["enabled"]
    cfg_on = parse_megakernel_config(raw="1")
    assert cfg_on["enabled"] and cfg_on["mode"] == ""
    cfg_tp = parse_megakernel_config(raw="tp")
    assert cfg_tp["enabled"] and cfg_tp["mode"] == "tp"
    cfg_kv = parse_megakernel_config(
        raw="mode=tp,buckets=8:16:32,verify_buckets=4:8,ab=1,stats_period_s=12.5,compile_hlo_check=1"
    )
    assert cfg_kv["enabled"]
    assert cfg_kv["mode"] == "tp"
    assert cfg_kv["buckets"] == (8, 16, 32)
    assert cfg_kv["verify_buckets"] == (4, 8)
    assert cfg_kv["ab"] is True
    assert cfg_kv["stats_period_s"] == pytest.approx(12.5)
    assert cfg_kv["compile_hlo_check"] is True

    cfg_json = parse_megakernel_config(
        raw='{"mode": "ep", "buckets": [4, 8], "verify_buckets": [4], "ab": false}'
    )
    assert cfg_json["enabled"]
    assert cfg_json["mode"] == "ep"
    assert cfg_json["buckets"] == (4, 8)
    assert cfg_json["verify_buckets"] == (4,)

    fake_runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            additional_config={
                "megakernel": {"mode": "tp", "buckets": [16, 32], "ab": True}
            }
        )
    )
    cfg_runner = parse_megakernel_config(runner=fake_runner, raw="")
    assert cfg_runner["enabled"]
    assert cfg_runner["mode"] == "tp"
    assert cfg_runner["buckets"] == (16, 32)
    assert cfg_runner["ab"] is True

    buckets = (2, 4, 8)
    assert select_bucket(1, buckets) == 2
    assert select_bucket(4, buckets) == 4
    assert select_bucket(5, buckets) == 8
    with pytest.raises(ValueError, match="exceeds max bucket"):
        select_bucket(9, buckets)


def test_ab_calibrator() -> None:
    """Verify `MKABCalibrator` alternation and latency summary formatting."""
    off = MKABCalibrator(enabled=False, period_s=0.0)
    assert off.choose("mk_t1_b2", True) == (True, "mk_t1_b2", "mk")
    assert off.choose("stock_mixed", False) == (False, "stock_mixed", "stock")
    assert off.record(1.5, padded_tokens=2) is None

    ab = MKABCalibrator(enabled=True, period_s=0.0)
    assert ab.choose("mk_t1_b2", True) == (True, "mk_t1_b2_ab_mk", "mk")
    res1 = ab.record(1.2, padded_tokens=2)
    assert res1 is not None

    assert ab.choose("mk_t1_b2", True) == (False, "mk_t1_b2_ab_stock", "stock")
    res2 = ab.record(2.2, padded_tokens=2)
    assert res2 is not None
    _, _, cum_json = res2
    parsed = json.loads(cum_json)
    entry = parsed["mk_t1_b2_p2"]
    assert entry["mk"][0] == 1
    assert entry["stock"][0] == 1
    assert entry["d_mean_ms"] == pytest.approx(-1.0)


def test_prepare_step_metadata_decode_and_verify() -> None:
    """Verify `prepare_step_metadata` for `T=1` decode, `T=4` verify, and `roff`."""
    attn_md = SimpleNamespace(
        block_tables=torch.tensor([[16, 17], [18, 19], [0, 0]], dtype=torch.int32),
        seq_lens=torch.tensor([10, 25, 0], dtype=torch.int32),
    )
    # 2 active sequences: base slots are [1, 5] and 4-slot checkpoint windows are
    # [[1, 2, 3, 4], [5, 6, 7, 8]]. In TPUModelRunner, mamba_slot_read_offsets is
    # indexed by the group-0 base slot id (1 and 5), with roff[1]=2 and roff[5]=1.
    roff_slot_indexed = torch.zeros((16,), dtype=torch.int32)
    roff_slot_indexed[1] = 2
    roff_slot_indexed[5] = 1
    mamba_mds = [
        SimpleNamespace(
            mamba_state_indices=torch.tensor([1, 5, 0], dtype=torch.int32),
            mamba_ckpt_indices=torch.tensor(
                [[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 0, 0]], dtype=torch.int32
            ),
            mamba_slot_read_offsets=roff_slot_indexed,
        )
        for _ in range(3)
    ]

    # 1. T=1 decode with speculative read offsets:
    #    ckpt_1 has shape (3, 4, 2) where col 0 is read_slot c[s, roff[s]] ->
    #    [3, 6, 0, 0] and col 1 is write_slot s (base slot) -> [1, 5, 0, 0].
    pos_1, pt_1, ckpt_1, roff_1 = prepare_step_metadata(
        attn_md=attn_md,
        mamba_mds=mamba_mds,
        positions=torch.tensor([9, 24], dtype=torch.int32),
        num_reqs=2,
        bucket_t=4,
        max_pages_per_seq=2,
        tokens_per_seq=1,
    )
    assert pos_1.shape == (4,)
    assert pos_1.tolist() == [9, 24, 0, 0]
    assert pt_1.shape == (4, 2)
    assert pt_1[:2].tolist() == [[16, 17], [18, 19]]
    assert ckpt_1.shape == (3, 4, 2)
    assert ckpt_1[0, :, 0].tolist() == [3, 6, 0, 0]
    assert ckpt_1[0, :, 1].tolist() == [1, 5, 0, 0]
    assert roff_1.shape == (3, 4)
    assert roff_1[0].tolist() == [0, 0, 0, 0]

    # 2. T=4 MTP verify with speculative read offsets: preserves 4-col ckpt & roff
    pos_4, pt_4, ckpt_4, roff_4 = prepare_step_metadata(
        attn_md=attn_md,
        mamba_mds=mamba_mds,
        positions=torch.tensor([9, 10, 11, 12, 24, 25, 26, 27], dtype=torch.int32),
        num_reqs=2,
        bucket_t=8,
        max_pages_per_seq=2,
        tokens_per_seq=4,
    )
    assert pos_4.shape == (8,)
    assert pos_4.tolist() == [9, 10, 11, 12, 24, 25, 26, 27]
    assert pt_4.shape == (2, 2)
    assert ckpt_4.shape == (3, 2, 4)
    assert ckpt_4[0].tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert roff_4.shape == (3, 2)
    assert roff_4[0].tolist() == [2, 1]


def test_block_tables_1d_reshape_most_model_len() -> None:
    """Verify 1-D `block_tables` reshapes by `s0.shape[0]` (`target_num_reqs`) first."""
    target_num_reqs = 16
    target_num_blocks = 2
    max_pages_per_seq = 4
    bt_2d = torch.arange(
        100,
        100 + target_num_reqs * target_num_blocks,
        dtype=torch.int32,
    ).reshape(target_num_reqs, target_num_blocks)
    bt_1d = bt_2d.reshape(-1)
    assert bt_1d.numel() == 32
    assert bt_1d.numel() % max_pages_per_seq == 0

    s_padded = torch.zeros((target_num_reqs,), dtype=torch.int32)
    s_padded[0] = 1
    s_padded[1] = 2
    zero_roff = torch.zeros((1,), dtype=torch.int32)

    _, _, pt_b, _, _ = prepare_step_inputs_eager(
        torch.tensor([11, 22], dtype=torch.int32),
        torch.tensor([5, 9], dtype=torch.int32),
        bt_1d,
        s_padded,
        s_padded,
        zero_roff,
        s_padded,
        s_padded,
        zero_roff,
        s_padded,
        s_padded,
        zero_roff,
        bucket_t=4,
        tokens_per_seq=1,
        max_pages_per_seq=max_pages_per_seq,
        has_roff=False,
        use_ckpt_2d=False,
    )
    assert pt_b.shape == (4, max_pages_per_seq)
    assert pt_b[0].tolist() == [100, 101, 0, 0]
    assert pt_b[1].tolist() == [102, 103, 0, 0]


def test_bind_caches_rejects_compact_mamba_and_non_aliasing_la_cache() -> None:
    """Verify `bind_caches` rejects compact-mamba or non-aliasing `la.kv_cache`."""
    mesh = _cpu_mesh(1)
    g = _tiny_geom(L=4, att_every=4)
    ex_base = _tiny_exec(R=1, B=2, T=1, ctx=256, num_pages=96)
    backbone = make_synthetic_qwen35_backbone(g, ex_base)
    model = SimpleNamespace(model=backbone)
    raw_pools_t = _allocate_aliased_pools(backbone, g, 96)

    driver = MegakernelDriver(
        model=model,
        mesh=mesh,
        tp_rank=0,
        mode="ep",
        geom=g,
        base_exec=ex_base,
        buckets=(2,),
        verify_buckets=(),
    )

    runner_compact = SimpleNamespace(model=model, _unified_kv_layout=False)
    with pytest.raises(ValueError, match="_unified_kv_layout must be True"):
        driver.bind_caches(runner_compact)

    backbone.layers[0].linear_attn.kv_cache = []
    runner_ok = SimpleNamespace(model=model, _unified_kv_layout=True)
    with pytest.raises(ValueError, match="must alias pool"):
        driver.bind_caches(runner_ok)

    separate_pool = torch.zeros_like(raw_pools_t[0])
    backbone.layers[0].linear_attn.kv_cache = [separate_pool]
    with pytest.raises(ValueError, match="must alias pool"):
        driver.bind_caches(runner_ok)

    backbone.layers[0].linear_attn.kv_cache = [raw_pools_t[0]]
    driver.bind_caches(runner_ok)
    assert driver._bound


def test_can_run_real_tpu_runner_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `can_run` against real `TPUModelRunner` and `AttentionMetadata` fields."""
    mesh = _cpu_mesh(1)
    g = _tiny_geom(L=4, att_every=4)
    ex_base = _tiny_exec(R=1, B=2, T=1, ctx=256, num_pages=96)
    backbone = make_synthetic_qwen35_backbone(g, ex_base)
    model = SimpleNamespace(model=backbone)
    _allocate_aliased_pools(backbone, g, 96)

    driver = MegakernelDriver(
        model=model,
        mesh=mesh,
        tp_rank=0,
        mode="ep",
        geom=g,
        base_exec=ex_base,
        buckets=(2, 4),
        verify_buckets=(4, 8),
    )

    qsl_np = np.zeros(130, dtype=np.int32)
    req_dist_cpu = torch.zeros(3, dtype=torch.int32)
    comb_dist_cpu = torch.zeros(6, dtype=torch.int32)
    runner = SimpleNamespace(
        model=model,
        mesh=mesh,
        max_num_blocks_per_req=2,
        num_tokens_paddings=[2, 4, 8],
        _unified_kv_layout=True,
        lora_config=None,
        _pp_is_first=True,
        _pp_is_last=True,
        _dp_lockstep_enabled=lambda: False,
        _in_dummy_run=False,
        _attn_metadata_builder_ctx=SimpleNamespace(num_reqs=2),
        _request_distribution_cpu=req_dist_cpu,
        _combined_request_distribution_cpu=comb_dist_cpu,
        _cached_comb_key=None,
        query_start_loc_np=qsl_np,
        mamba_slot_read_offsets=None,
        mrope_positions=None,
    )
    driver.bind_caches(runner)

    max_reqs_padded = 128
    attn_md = SimpleNamespace(
        block_tables=torch.zeros((max_reqs_padded * 2,), dtype=torch.int32),
        seq_lens=torch.ones((max_reqs_padded,), dtype=torch.int32),
        query_start_loc=None,
        request_distribution=None,
    )
    mamba_mds = [
        SimpleNamespace(
            mamba_state_indices=torch.zeros((max_reqs_padded,), dtype=torch.int32),
            mamba_ckpt_indices=None,
            mamba_slot_read_offsets=None,
            mamba_request_distribution=None,
        )
        for _ in range(3)
    ]
    md_dict = {driver.attn_layer_name: attn_md}
    for name, m_md in zip(driver.mamba_layer_names, mamba_mds, strict=True):
        md_dict[name] = m_md

    monkeypatch.setattr(
        driver_mod,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=md_dict),
    )

    ids_2 = torch.zeros((2,), dtype=torch.int32)
    pos_2 = torch.zeros((2,), dtype=torch.int32)
    ids_8 = torch.zeros((8,), dtype=torch.int32)
    pos_8 = torch.zeros((8,), dtype=torch.int32)

    # 1. Pure T=1 decode (2 reqs, 1 token each) -> True (mk_t1_b2)
    runner._attn_metadata_builder_ctx.num_reqs = 2
    req_dist_cpu.copy_(torch.tensor([2, 2, 2], dtype=torch.int32))
    qsl_np[:3] = [0, 1, 2]
    assert driver.can_run(ids_2, pos_2, runner=runner)
    assert driver._prepared_step is not None
    assert driver._prepared_step[5:8] == (2, 1, 2)

    # 2. Single-request prefill (1 req, 8 tokens) with mamba_state_indices -> False
    runner._attn_metadata_builder_ctx.num_reqs = 1
    req_dist_cpu.copy_(torch.tensor([0, 1, 1], dtype=torch.int32))
    qsl_np[:2] = [0, 8]
    assert not driver.can_run(ids_8, pos_8, runner=runner)
    req_dist_cpu.copy_(torch.tensor([1, 1, 1], dtype=torch.int32))
    assert not driver.can_run(ids_8, pos_8, runner=runner)

    # 3. Mixed decode + prefill (2 reqs, [1, 7] tokens) -> False
    runner._attn_metadata_builder_ctx.num_reqs = 2
    req_dist_cpu.copy_(torch.tensor([1, 2, 2], dtype=torch.int32))
    qsl_np[:3] = [0, 1, 8]
    assert not driver.can_run(ids_8, pos_8, runner=runner)
    req_dist_cpu.copy_(torch.tensor([2, 2, 2], dtype=torch.int32))
    assert not driver.can_run(ids_8, pos_8, runner=runner)

    # 4. Spec decode enabled (`mamba_slot_read_offsets` present, 4-col ckpt):
    #    4a. Uniform T=4 verify (1 req, 4 tokens) -> True (mk_t4_b4)
    runner.mamba_slot_read_offsets = torch.zeros((64,), dtype=torch.int32)
    for m_md in mamba_mds:
        m_md.mamba_ckpt_indices = torch.zeros((max_reqs_padded, 4), dtype=torch.int32)
        m_md.mamba_slot_read_offsets = runner.mamba_slot_read_offsets
    runner._attn_metadata_builder_ctx.num_reqs = 1
    comb_dist_cpu.copy_(torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.int32))
    qsl_np[:2] = [0, 4]
    ids_4 = torch.zeros((4,), dtype=torch.int32)
    pos_4 = torch.zeros((4,), dtype=torch.int32)
    assert driver.can_run(ids_4, pos_4, runner=runner)
    assert driver._prepared_step is not None
    assert driver._prepared_step[5:8] == (1, 4, 4)

    #    4b. T=1 plain decode under spec decode (`ckpt.shape == (128, 4)`, 2 active
    #        reqs with 1 token each): must dispatch as (num_reqs=2, T=1, bkt=2), NOT
    #        T=4 and NOT num_reqs=128!
    runner._attn_metadata_builder_ctx.num_reqs = 2
    comb_dist_cpu.copy_(torch.tensor([2, 2, 2, 2, 2, 2], dtype=torch.int32))
    qsl_np[:3] = [0, 1, 2]
    assert driver.can_run(ids_2, pos_2, runner=runner)
    assert driver._prepared_step is not None
    assert driver._prepared_step[5:8] == (2, 1, 2)

    #    4c. Variable draft lengths under spec decode ([4, 2] tokens, both windowed
    #        so comb_dist_cpu[3] == 2 == num_reqs) -> False (stock_mixed)
    runner._attn_metadata_builder_ctx.num_reqs = 2
    comb_dist_cpu.copy_(torch.tensor([0, 0, 2, 2, 2, 2], dtype=torch.int32))
    qsl_np[:3] = [0, 4, 6]
    assert not driver.can_run(ids_8, pos_8, runner=runner)

    # 5. Missing host `query_start_loc` must fail closed (`stock_no_qsl`), never
    #    falling back to `mamba_state_indices.shape[0]` (128) or `ckpt.shape[1]` (4).
    runner.query_start_loc_np = None
    assert not driver.can_run(ids_2, pos_2, runner=runner)
    runner.query_start_loc_np = qsl_np
    qsl_np[:3] = [0, 1, 2]
    assert driver.can_run(ids_2, pos_2, runner=runner)

    # 6. Dummy runs (`bctx.position_ids_override is not None` or `_in_dummy_run=True`)
    #    gate off MK without needing runner-side flags in `_dummy_run`
    runner._attn_metadata_builder_ctx.position_ids_override = pos_2
    assert not driver.can_run(ids_2, pos_2, runner=runner)
    hit_dummy, out_dummy = driver.try_forward(runner, ids_2, pos_2)
    assert not hit_dummy and out_dummy is None
    runner._attn_metadata_builder_ctx.position_ids_override = None
    assert driver.can_run(ids_2, pos_2, runner=runner)

    runner._in_dummy_run = True
    assert not driver.can_run(ids_2, pos_2, runner=runner)
    runner._in_dummy_run = False

    # 7. mrope 2D positions `(3, tok_pad)`: rejected when rows differ, accepted when ==
    pos_mrope_diff = torch.tensor([[5, 9], [5, 10], [5, 9]], dtype=torch.int32)
    assert not driver.can_run(ids_2, pos_mrope_diff, runner=runner)
    pos_mrope_eq = torch.tensor([[5, 9], [5, 9], [5, 9]], dtype=torch.int32)
    assert driver.can_run(ids_2, pos_mrope_eq, runner=runner)

    # 8. Unsupported runner/backbone configurations gate off `can_run`
    runner._unified_kv_layout = False
    assert not driver.can_run(ids_2, pos_2, runner=runner)
    runner._unified_kv_layout = True

    runner.lora_config = object()
    assert not driver.can_run(ids_2, pos_2, runner=runner)
    runner.lora_config = None

    runner._pp_is_last = False
    assert not driver.can_run(ids_2, pos_2, runner=runner)
    runner._pp_is_last = True

    runner._dp_lockstep_enabled = lambda: True
    assert not driver.can_run(ids_2, pos_2, runner=runner)
    runner._dp_lockstep_enabled = lambda: False

    backbone.aux_hidden_state_layers = (1,)
    assert not driver.can_run(ids_2, pos_2, runner=runner)
    backbone.aux_hidden_state_layers = ()

    # 9. `megakernel_runtime.maybe_create` model-architecture registry check
    non_qwen_runner = SimpleNamespace(
        model=SimpleNamespace(),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["LlamaForCausalLM"], model_type="llama"
            )
        ),
    )
    assert driver_mod.maybe_create(non_qwen_runner) is None


def test_megakernel_driver_e2e_decode_verify_and_spec_accept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `MegakernelDriver` warmup, `T=1`, `T=4`, and speculative accept."""
    mesh = _cpu_mesh(1)
    tp = NamedSharding(mesh, P("tp"))
    unsh = NamedSharding(mesh, P())
    g = _tiny_geom(L=4, att_every=4)
    num_pages = 96
    ex_base = _tiny_exec(R=1, B=2, T=1, ctx=256, num_pages=num_pages)

    # Generate independent logical weights and packed MK reference weights via pure-JAX
    # `qwen35` helpers, then construct the PyTorch backbone from `w_logical`.
    torch.manual_seed(42)
    hf = qwen35.random_hf_weights(jax.random.key(42), g)
    w_logical = qwen35.split_hf_weights(hf, g, ex_base)
    pos_init, hid_l_init, _, st_l = qwen35.random_logical_inputs(
        jax.random.key(77), g, ex_base
    )
    st_l = dict(
        st_l,
        conv=st_l["conv"].astype(jnp.bfloat16).astype(jnp.float32),
    )
    _, _, w_mk_ref, _ = qwen35.pack_inputs(
        pos_init,
        hid_l_init,
        w_logical,
        st_l,
        g,
        ex_base,
        mesh=mesh,
        build_default_pool=True,
    )

    backbone = _make_backbone_from_logical_weights(g, ex_base, w_logical, tp_rank=0)
    model = SimpleNamespace(model=backbone)
    raw_pools_t = _allocate_aliased_pools(backbone, g, num_pages)

    qsl_np = np.zeros(16, dtype=np.int32)
    req_dist_cpu = torch.zeros(3, dtype=torch.int32)
    comb_dist_cpu = torch.zeros(6, dtype=torch.int32)
    roff_slot_buf = torch.zeros((16,), dtype=torch.int32)
    runner = SimpleNamespace(
        model=model,
        mesh=mesh,
        max_num_blocks_per_req=2,
        num_reqs_max_model_len=4,
        num_tokens_paddings=[2, 4, 8],
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        _mamba_ckpt_window=4,
        _unified_kv_layout=True,
        lora_config=None,
        _pp_is_first=True,
        _pp_is_last=True,
        _dp_lockstep_enabled=lambda: False,
        _in_dummy_run=False,
        _attn_metadata_builder_ctx=SimpleNamespace(num_reqs=2),
        _request_distribution_cpu=req_dist_cpu,
        _combined_request_distribution_cpu=comb_dist_cpu,
        _cached_comb_key=None,
        query_start_loc_np=qsl_np,
        mamba_slot_read_offsets=roff_slot_buf,
    )

    driver = MegakernelDriver(
        model=model,
        mesh=mesh,
        tp_rank=0,
        mode="ep",
        geom=g,
        base_exec=ex_base,
        buckets=(2,),
        verify_buckets=(4,),
    )
    orig_data_ptrs = [p.data_ptr() for p in raw_pools_t]
    orig_storage_ptrs = [p.untyped_storage().data_ptr() for p in raw_pools_t]

    # 1. `bind_and_warmup` binds pools, compiles (1, 2) and (4, 4), runs sentinel check,
    #    and restores pools
    driver.bind_and_warmup(runner)
    for idx, p in enumerate(raw_pools_t):
        assert p.data_ptr() == orig_data_ptrs[idx]
        assert p.untyped_storage().data_ptr() == orig_storage_ptrs[idx]
        assert torch.count_nonzero(p.view(torch.uint8)).item() == 0

    # Pack initial logical KV and GDN states into Raiden pools via `qwen35_pool`
    pt_init = jnp.asarray([[60, 61], [62, 63]], dtype=jnp.int32)
    ckpt_init = jnp.asarray([[[1], [5]], [[1], [5]], [[1], [5]]], dtype=jnp.int32)
    roff_init = jnp.zeros((3, 2), dtype=jnp.int32)
    init_pools_j = qwen35_pool.states_to_pool(
        g, ex_base, st_l, pt_init, ckpt_init, roff_init
    )
    for p_t, p_j in zip(raw_pools_t, init_pools_j, strict=True):
        p_t.view(torch.uint8).copy_(
            torch.from_numpy(np.asarray(p_j).view(np.uint8).copy())
        )

    # Set up serving forward context with 1-D flattened block_tables and padded mamba
    # indices (matching TPUModelRunner's exact tensor shapes).
    attn_md = SimpleNamespace(
        block_tables=torch.tensor([60, 61, 62, 63, 0, 0, 0, 0], dtype=torch.int32),
        seq_lens=torch.tensor([6, 10, 0, 0], dtype=torch.int32),
    )
    mamba_mds = [
        SimpleNamespace(
            mamba_state_indices=torch.tensor([1, 5, 0, 0], dtype=torch.int32),
            mamba_ckpt_indices=torch.tensor(
                [[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 0, 0], [0, 0, 0, 0]],
                dtype=torch.int32,
            ),
            mamba_slot_read_offsets=roff_slot_buf,
        )
        for _ in range(3)
    ]
    md_dict = {driver.attn_layer_name: attn_md}
    for name, m_md in zip(driver.mamba_layer_names, mamba_mds, strict=True):
        md_dict[name] = m_md

    monkeypatch.setattr(
        driver_mod,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=md_dict),
    )

    # -------------------------------------------------------------------------
    # Step A: T=1 decode (B=2, NS=2, T=1) vs both `qwen35.reference_step` and
    #         `qwen35.make_pooled_step` (with independently built metadata & w_mk_ref).
    # -------------------------------------------------------------------------
    runner._attn_metadata_builder_ctx.num_reqs = 2
    comb_dist_cpu.copy_(torch.tensor([2, 2, 2, 2, 2, 2], dtype=torch.int32))
    qsl_np[:3] = [0, 1, 2]
    roff_slot_buf.zero_()

    input_ids_t1 = torch.tensor([5, 12], dtype=torch.int32)
    positions_t1 = torch.tensor([5, 9], dtype=torch.int32)
    ref_pools_j = tuple(_torch_to_jax(p.clone(), tp) for p in raw_pools_t)

    hit_t1, out_t1 = driver.try_forward(runner, input_ids_t1, positions_t1)
    assert hit_t1 and out_t1 is not None
    assert out_t1.shape == (2, g.H)
    for idx, p in enumerate(raw_pools_t):
        assert p.data_ptr() == orig_data_ptrs[idx]

    # Independent pure-JAX `qwen35.reference_step` check
    hid_t1_bf16 = driver.embed_tokens(input_ids_t1).to(torch.bfloat16)
    hid_l_t1 = _torch_to_jax(hid_t1_bf16, unsh)
    pos_j_t1 = jnp.asarray([5, 9], dtype=jnp.int32)
    ref_out_t1, _ = qwen35.reference_step(
        g, ex_base, pos_j_t1, hid_l_t1, w_logical, st_l, paged_kv=True, pool_io=True
    )
    np.testing.assert_allclose(
        out_t1.float().numpy(),
        np.asarray(ref_out_t1, dtype=np.float32),
        atol=8e-2,
        rtol=8e-2,
    )

    # Independent `qwen35.make_pooled_step` check (using `w_mk_ref` from `pack_inputs`)
    ex_t1 = replace(driver.base_exec, B=2, T=1)
    step_ref_t1, _ = qwen35.make_pooled_step(mesh, g, ex_t1, donate_pools=False)
    hid_in_t1_pad = torch.nn.functional.pad(hid_t1_bf16, (0, 0, 0, ex_t1.M - 2))
    ckpt_ref_t1 = jnp.asarray(
        [[[1, 1], [5, 5]], [[1, 1], [5, 5]], [[1, 1], [5, 5]]], dtype=jnp.int32
    )
    roff_ref_t1 = jnp.zeros((3, 2), dtype=jnp.int32)
    hid_ref_t1, ref_pools_j = step_ref_t1(
        pos_j_t1,
        _torch_to_jax(hid_in_t1_pad, tp),
        w_mk_ref,
        ref_pools_j,
        pt_init,
        ckpt_ref_t1,
        roff_ref_t1,
    )
    np.testing.assert_array_equal(
        out_t1.view(torch.uint8).numpy(),
        np.asarray(hid_ref_t1[:2]).view(np.uint8),
    )
    for p_t, p_ref in zip(raw_pools_t, ref_pools_j, strict=True):
        np.testing.assert_array_equal(
            p_t.view(torch.uint8).numpy(),
            np.asarray(p_ref).view(np.uint8),
        )

    # -------------------------------------------------------------------------
    # Step B: T=4 MTP verify (B=4, NS=1, T=4) writing contiguous slots [1, 2, 3, 4]
    # -------------------------------------------------------------------------
    runner._attn_metadata_builder_ctx.num_reqs = 1
    comb_dist_cpu.copy_(torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.int32))
    qsl_np[:2] = [0, 4]
    roff_slot_buf.zero_()
    attn_md.block_tables = torch.tensor([60, 61, 0, 0, 0, 0, 0, 0], dtype=torch.int32)
    attn_md.seq_lens = torch.tensor([9, 0, 0, 0], dtype=torch.int32)
    for m_md in mamba_mds:
        m_md.mamba_state_indices = torch.tensor([1, 0, 0, 0], dtype=torch.int32)
        m_md.mamba_ckpt_indices = torch.tensor(
            [[1, 2, 3, 4], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=torch.int32,
        )

    input_ids_t4 = torch.tensor([11, 12, 13, 14], dtype=torch.int32)
    positions_t4 = torch.tensor([6, 7, 8, 9], dtype=torch.int32)
    assert driver.can_run(input_ids_t4, positions_t4, runner=runner)
    out_t4 = driver(input_ids_t4, positions_t4, num_tokens=4, runner=runner)
    assert out_t4.shape == (4, g.H)

    ex_t4 = replace(driver.base_exec, B=4, T=4)
    step_ref_t4, _ = qwen35.make_pooled_step(mesh, g, ex_t4, donate_pools=False)
    hid_t4_bf16 = driver.embed_tokens(input_ids_t4).to(torch.bfloat16)
    hid_in_t4_pad = torch.nn.functional.pad(hid_t4_bf16, (0, 0, 0, ex_t4.M - 4))
    pos_j_t4 = jnp.asarray([6, 7, 8, 9], dtype=jnp.int32)
    pt_ref_t4 = jnp.asarray([[60, 61]], dtype=jnp.int32)
    ckpt_ref_t4 = jnp.asarray(
        [[[1, 2, 3, 4]], [[1, 2, 3, 4]], [[1, 2, 3, 4]]], dtype=jnp.int32
    )
    roff_ref_t4 = jnp.zeros((3, 1), dtype=jnp.int32)
    hid_ref_t4, ref_pools_j = step_ref_t4(
        pos_j_t4,
        _torch_to_jax(hid_in_t4_pad, tp),
        w_mk_ref,
        ref_pools_j,
        pt_ref_t4,
        ckpt_ref_t4,
        roff_ref_t4,
    )
    np.testing.assert_array_equal(
        out_t4.view(torch.uint8).numpy(),
        np.asarray(hid_ref_t4[:4]).view(np.uint8),
    )
    for p_t, p_ref in zip(raw_pools_t, ref_pools_j, strict=True):
        np.testing.assert_array_equal(
            p_t.view(torch.uint8).numpy(),
            np.asarray(p_ref).view(np.uint8),
        )

    # -------------------------------------------------------------------------
    # Step C: T=1 plain decode after T=4 verify with `roff[1] = 2` (accepting 2
    #         draft tokens so the step reads slot `ckpt[0, 2] == 3` and writes
    #         the updated GDN state back to base slot `s[0] == 1`).
    # -------------------------------------------------------------------------
    runner._attn_metadata_builder_ctx.num_reqs = 1
    comb_dist_cpu.copy_(torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.int32))
    qsl_np[:2] = [0, 1]
    roff_slot_buf.zero_()
    roff_slot_buf[1] = 2  # base slot 1 -> read offset 2 (slot 3)

    # Snapshot slot 3 (pages 15..19) and slot 1 (pages 5..9) before Step C
    slot3_before = [p[15:19].view(torch.uint8).clone() for p in raw_pools_t]
    slot1_before = [p[5:9].view(torch.uint8).clone() for p in raw_pools_t]

    input_ids_t1_after_t4 = torch.tensor([19, 0], dtype=torch.int32)
    positions_t1_after_t4 = torch.tensor([8, 0], dtype=torch.int32)
    assert driver.can_run(input_ids_t1_after_t4, positions_t1_after_t4, runner=runner)
    assert driver._prepared_step is not None
    assert driver._prepared_step[5:8] == (1, 1, 2)
    out_t1_after_t4 = driver(
        input_ids_t1_after_t4, positions_t1_after_t4, num_tokens=2, runner=runner
    )

    # Slot 1 (base slot, pages 5..9) is updated; slot 3 (pages 15..19) is untouched
    for idx, p in enumerate(raw_pools_t):
        assert not torch.equal(p[5:9].view(torch.uint8), slot1_before[idx])
        assert torch.equal(p[15:19].view(torch.uint8), slot3_before[idx])

    # Independent `make_pooled_step` oracle with read_slot=3, write_slot=1 for seq 0
    hid_c_bf16 = driver.embed_tokens(input_ids_t1_after_t4).to(torch.bfloat16)
    hid_in_c_pad = torch.nn.functional.pad(hid_c_bf16, (0, 0, 0, ex_t1.M - 2))
    pos_j_c = jnp.asarray([8, 0], dtype=jnp.int32)
    pt_ref_c = jnp.asarray([[60, 61], [0, 0]], dtype=jnp.int32)
    ckpt_ref_c = jnp.asarray(
        [[[3, 1], [0, 0]], [[3, 1], [0, 0]], [[3, 1], [0, 0]]], dtype=jnp.int32
    )
    roff_ref_c = jnp.zeros((3, 2), dtype=jnp.int32)
    hid_ref_c, ref_pools_j = step_ref_t1(
        pos_j_c,
        _torch_to_jax(hid_in_c_pad, tp),
        w_mk_ref,
        ref_pools_j,
        pt_ref_c,
        ckpt_ref_c,
        roff_ref_c,
    )
    np.testing.assert_array_equal(
        out_t1_after_t4[:1].view(torch.uint8).numpy(),
        np.asarray(hid_ref_c[:1]).view(np.uint8),
    )
    for p_t, p_ref in zip(raw_pools_t, ref_pools_j, strict=True):
        np.testing.assert_array_equal(
            p_t.view(torch.uint8).numpy(),
            np.asarray(p_ref).view(np.uint8),
        )

    # -------------------------------------------------------------------------
    # Step D: Subsequent T=4 MTP verify with reset `roff[1] = 0` (reads base slot 1
    #         written by Step C) followed by Step E with `roff[1] = 2`.
    # -------------------------------------------------------------------------
    runner._attn_metadata_builder_ctx.num_reqs = 1
    comb_dist_cpu.copy_(torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.int32))
    qsl_np[:2] = [0, 4]
    roff_slot_buf.zero_()  # reset to 0 after the 1-token step

    input_ids_t4_d = torch.tensor([21, 22, 23, 24], dtype=torch.int32)
    positions_t4_d = torch.tensor([9, 10, 11, 12], dtype=torch.int32)
    assert driver.can_run(input_ids_t4_d, positions_t4_d, runner=runner)
    out_t4_d = driver(input_ids_t4_d, positions_t4_d, num_tokens=4, runner=runner)

    hid_d_bf16 = driver.embed_tokens(input_ids_t4_d).to(torch.bfloat16)
    hid_in_d_pad = torch.nn.functional.pad(hid_d_bf16, (0, 0, 0, ex_t4.M - 4))
    pos_j_d = jnp.asarray([9, 10, 11, 12], dtype=jnp.int32)
    hid_ref_d, ref_pools_j = step_ref_t4(
        pos_j_d,
        _torch_to_jax(hid_in_d_pad, tp),
        w_mk_ref,
        ref_pools_j,
        pt_ref_t4,
        ckpt_ref_t4,
        roff_ref_t4,
    )
    np.testing.assert_array_equal(
        out_t4_d.view(torch.uint8).numpy(),
        np.asarray(hid_ref_d[:4]).view(np.uint8),
    )
    for p_t, p_ref in zip(raw_pools_t, ref_pools_j, strict=True):
        np.testing.assert_array_equal(
            p_t.view(torch.uint8).numpy(),
            np.asarray(p_ref).view(np.uint8),
        )

    # Step E: Speculative acceptance on next T=4 step via `roff_slot_buf[1] = 2`
    roff_slot_buf[1] = 2
    input_ids_t4_acc = torch.tensor([31, 32, 33, 34], dtype=torch.int32)
    positions_t4_acc = torch.tensor([11, 12, 13, 14], dtype=torch.int32)
    assert driver.can_run(input_ids_t4_acc, positions_t4_acc, runner=runner)
    out_t4_acc = driver(input_ids_t4_acc, positions_t4_acc, num_tokens=4, runner=runner)

    hid_e_bf16 = driver.embed_tokens(input_ids_t4_acc).to(torch.bfloat16)
    hid_in_e_pad = torch.nn.functional.pad(hid_e_bf16, (0, 0, 0, ex_t4.M - 4))
    pos_j_e = jnp.asarray([11, 12, 13, 14], dtype=jnp.int32)
    roff_ref_e = jnp.asarray([[2], [2], [2]], dtype=jnp.int32)
    hid_ref_e, ref_pools_j = step_ref_t4(
        pos_j_e,
        _torch_to_jax(hid_in_e_pad, tp),
        w_mk_ref,
        ref_pools_j,
        pt_ref_t4,
        ckpt_ref_t4,
        roff_ref_e,
    )
    np.testing.assert_array_equal(
        out_t4_acc.view(torch.uint8).numpy(),
        np.asarray(hid_ref_e[:4]).view(np.uint8),
    )
    for p_t, p_ref in zip(raw_pools_t, ref_pools_j, strict=True):
        np.testing.assert_array_equal(
            p_t.view(torch.uint8).numpy(),
            np.asarray(p_ref).view(np.uint8),
        )
