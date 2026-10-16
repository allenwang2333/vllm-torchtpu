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
    make_synthetic_qwen35_backbone,
)
from vllm_torchtpu.megakernel.models import qwen35  # noqa: E402
from vllm_torchtpu.megakernel.spec import TINY_TILES, ExecSpec  # noqa: E402
from vllm_torchtpu.megakernel_runtime.driver import (  # noqa: E402
    MegakernelDriver,
    MKABCalibrator,
    parse_mk_buckets,
    parse_mk_verify_buckets,
    prepare_step_inputs_eager,
    prepare_step_metadata,
    select_bucket,
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


def test_bucket_parsing_and_selection() -> None:
    """Verify decode/verify bucket parsing, sorting, and smallest-fit selection."""
    assert parse_mk_buckets("16, 4, 8, 4") == (4, 8, 16)
    assert parse_mk_buckets("0") == ()
    assert parse_mk_buckets("off") == ()
    with pytest.raises(ValueError, match="invalid VLLM_TPU_QWEN35_MK_BUCKETS"):
        parse_mk_buckets("128")

    assert parse_mk_verify_buckets("16, 4, 8") == (4, 8, 16)
    assert parse_mk_verify_buckets("none") == ()
    with pytest.raises(ValueError, match="invalid VLLM_TPU_QWEN35_MK_VERIFY_BUCKETS"):
        parse_mk_verify_buckets("6")

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
    # 2 active sequences, each with a 4-slot checkpoint window; roff selects slot
    mamba_mds = [
        SimpleNamespace(
            mamba_state_indices=torch.tensor(
                [[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32
            ),
            mamba_ckpt_indices=torch.tensor(
                [[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32
            ),
            mamba_slot_read_offsets=torch.tensor([2, 1], dtype=torch.int32),
        )
        for _ in range(3)
    ]

    # 1. T=1 decode with speculative read offsets: gathers slot c[s, roff[s]] -> [3, 6]
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
    assert ckpt_1.shape == (3, 4, 1)
    assert ckpt_1[0, :, 0].tolist() == [3, 6, 0, 0]
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


def test_megakernel_driver_e2e_decode_verify_and_spec_accept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `MegakernelDriver` warmup, `T=1`, `T=4`, and spec accept parity."""
    mesh = _cpu_mesh(1)
    tp = NamedSharding(mesh, P("tp"))
    unsh = NamedSharding(mesh, P())
    g = _tiny_geom(L=4, att_every=4)
    num_pages = 96
    ex_base = _tiny_exec(R=1, B=2, T=1, ctx=256, num_pages=num_pages)

    torch.manual_seed(42)
    backbone = make_synthetic_qwen35_backbone(g, ex_base)
    model = SimpleNamespace(model=backbone)

    # Allocate Raiden-shaped float8_e4m3fn pools and alias linear_attn.kv_cache
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

    runner = SimpleNamespace(
        model=model,
        mesh=mesh,
        max_num_blocks_per_req=2,
        num_tokens_paddings=[2, 4, 8],
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        _mamba_ckpt_window=4,
        input_batch=SimpleNamespace(num_reqs=2),
        _last_num_scheduled_tokens=[1, 1],
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
    driver.bind_caches(runner)
    orig_data_ptrs = [p.data_ptr() for p in raw_pools_t]
    orig_storage_ptrs = [p.untyped_storage().data_ptr() for p in raw_pools_t]

    # 1. Warmup compiles (1, 2) and (4, 4), runs sentinel check, and restores pools
    driver.warmup(runner)
    for idx, p in enumerate(raw_pools_t):
        assert p.data_ptr() == orig_data_ptrs[idx]
        assert p.untyped_storage().data_ptr() == orig_storage_ptrs[idx]
        assert torch.count_nonzero(p.view(torch.uint8)).item() == 0

    # Seed non-zero initial KV and GDN state in the bound pools
    rng = np.random.default_rng(77)
    for p in raw_pools_t:
        rand_bytes = rng.integers(1, 100, size=tuple(p.shape), dtype=np.uint8)
        p.view(torch.uint8).copy_(torch.from_numpy(rand_bytes))

    # Prepare JAX weights dict for the pure-JAX reference step
    w_mk_j: dict[str, Any] = {}
    for k in driver.weights["plan"].w_names:
        v = driver.weights[k]
        if isinstance(v, tuple):
            w_mk_j[k] = tuple(_torch_to_jax(x, tp) for x in v)
        else:
            w_mk_j[k] = _torch_to_jax(v, tp)

    # Set up fake forward context metadata
    attn_md = SimpleNamespace(
        block_tables=torch.tensor([[60, 61], [62, 63]], dtype=torch.int32),
        seq_lens=torch.tensor([6, 10], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        num_seqs=2,
    )
    mamba_mds = [
        SimpleNamespace(
            mamba_state_indices=torch.tensor([1, 2], dtype=torch.int32),
            mamba_ckpt_indices=torch.tensor([[1], [2]], dtype=torch.int32),
            mamba_slot_read_offsets=None,
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

    # 2. Check `can_run` eligibility rules
    input_ids_t1 = torch.tensor([5, 12], dtype=torch.int32)
    positions_t1 = torch.tensor([5, 9], dtype=torch.int32)
    assert driver.can_run(input_ids_t1, positions_t1, runner=runner)
    assert not driver.can_run(
        input_ids_t1,
        positions_t1,
        inputs_embeds=torch.zeros((2, g.H), dtype=torch.bfloat16),
        runner=runner,
    )
    runner._last_num_scheduled_tokens = [1, 4]
    assert not driver.can_run(input_ids_t1, positions_t1, runner=runner)
    runner._last_num_scheduled_tokens = [1, 1]
    assert driver.can_run(input_ids_t1, positions_t1, runner=runner)

    # 3. Step A: T=1 decode (B=2, NS=2, T=1) vs pure-JAX `qwen35.make_pooled_step`
    ref_pools_j = tuple(_torch_to_jax(p.clone(), tp) for p in raw_pools_t)
    out_t1 = driver(input_ids_t1, positions_t1, num_tokens=2, runner=runner)
    assert out_t1.shape == (2, g.H)
    for idx, p in enumerate(raw_pools_t):
        assert p.data_ptr() == orig_data_ptrs[idx]

    ex_t1 = replace(driver.base_exec, B=2, T=1)
    step_ref_t1, _ = qwen35.make_pooled_step(mesh, g, ex_t1, donate_pools=False)
    ids_b, pos_b, pt_b, ckpt_b, roff_b = prepare_step_inputs_eager(
        input_ids_t1,
        positions_t1,
        attn_md.block_tables,
        mamba_mds[0].mamba_state_indices,
        mamba_mds[0].mamba_ckpt_indices,
        torch.zeros((1,), dtype=torch.int32),
        mamba_mds[1].mamba_state_indices,
        mamba_mds[1].mamba_ckpt_indices,
        torch.zeros((1,), dtype=torch.int32),
        mamba_mds[2].mamba_state_indices,
        mamba_mds[2].mamba_ckpt_indices,
        torch.zeros((1,), dtype=torch.int32),
        bucket_t=2,
        tokens_per_seq=1,
        max_pages_per_seq=2,
        has_roff=False,
        use_ckpt_2d=False,
    )
    hid_in_t1 = torch.nn.functional.pad(
        driver.embed_tokens(ids_b).to(torch.bfloat16), (0, 0, 0, ex_t1.M - 2)
    )
    hid_ref_t1, ref_pools_j = step_ref_t1(
        _torch_to_jax(pos_b, unsh),
        _torch_to_jax(hid_in_t1, tp),
        w_mk_j,
        ref_pools_j,
        _torch_to_jax(pt_b, unsh),
        _torch_to_jax(ckpt_b, unsh),
        _torch_to_jax(roff_b, unsh),
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

    # 4. Step B: T=4 MTP verify (B=4, NS=1, T=4) writing contiguous slots [1, 2, 3, 4]
    runner.input_batch.num_reqs = 1
    runner._last_num_scheduled_tokens = [4]
    attn_md.block_tables = torch.tensor([[60, 61]], dtype=torch.int32)
    attn_md.seq_lens = torch.tensor([9], dtype=torch.int32)
    for m_md in mamba_mds:
        m_md.mamba_state_indices = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
        m_md.mamba_ckpt_indices = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
        m_md.mamba_slot_read_offsets = torch.tensor([0, 0, 0], dtype=torch.int32)

    input_ids_t4 = torch.tensor([11, 12, 13, 14], dtype=torch.int32)
    positions_t4 = torch.tensor([6, 7, 8, 9], dtype=torch.int32)
    assert driver.can_run(input_ids_t4, positions_t4, runner=runner)
    out_t4 = driver(input_ids_t4, positions_t4, num_tokens=4, runner=runner)
    assert out_t4.shape == (4, g.H)

    ex_t4 = replace(driver.base_exec, B=4, T=4)
    step_ref_t4, _ = qwen35.make_pooled_step(mesh, g, ex_t4, donate_pools=False)
    ids_b4, pos_b4, pt_b4, ckpt_b4, roff_b4 = prepare_step_inputs_eager(
        input_ids_t4,
        positions_t4,
        attn_md.block_tables,
        mamba_mds[0].mamba_state_indices,
        mamba_mds[0].mamba_ckpt_indices,
        mamba_mds[0].mamba_slot_read_offsets,
        mamba_mds[1].mamba_state_indices,
        mamba_mds[1].mamba_ckpt_indices,
        mamba_mds[1].mamba_slot_read_offsets,
        mamba_mds[2].mamba_state_indices,
        mamba_mds[2].mamba_ckpt_indices,
        mamba_mds[2].mamba_slot_read_offsets,
        bucket_t=4,
        tokens_per_seq=4,
        max_pages_per_seq=2,
        has_roff=True,
        use_ckpt_2d=True,
    )
    hid_in_t4 = torch.nn.functional.pad(
        driver.embed_tokens(ids_b4).to(torch.bfloat16), (0, 0, 0, ex_t4.M - 4)
    )
    hid_ref_t4, ref_pools_j = step_ref_t4(
        _torch_to_jax(pos_b4, unsh),
        _torch_to_jax(hid_in_t4, tp),
        w_mk_j,
        ref_pools_j,
        _torch_to_jax(pt_b4, unsh),
        _torch_to_jax(ckpt_b4, unsh),
        _torch_to_jax(roff_b4, unsh),
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

    # 5. Step C: Speculative acceptance on next T=4 step via `mamba_slot_read_offsets`
    # Accept 2 draft tokens (`roff[base_slot=1]=2`, reading `ckpt[0, 2] == 3` in-place)
    for m_md in mamba_mds:
        m_md.mamba_slot_read_offsets = torch.tensor([0, 2, 0], dtype=torch.int32)
    input_ids_t4_acc = torch.tensor([21, 22, 23, 24], dtype=torch.int32)
    positions_t4_acc = torch.tensor([8, 9, 10, 11], dtype=torch.int32)
    assert driver.can_run(input_ids_t4_acc, positions_t4_acc, runner=runner)
    out_t4_acc = driver(input_ids_t4_acc, positions_t4_acc, num_tokens=4, runner=runner)

    ids_ba, pos_ba, pt_ba, ckpt_ba, roff_ba = prepare_step_inputs_eager(
        input_ids_t4_acc,
        positions_t4_acc,
        attn_md.block_tables,
        mamba_mds[0].mamba_state_indices,
        mamba_mds[0].mamba_ckpt_indices,
        mamba_mds[0].mamba_slot_read_offsets,
        mamba_mds[1].mamba_state_indices,
        mamba_mds[1].mamba_ckpt_indices,
        mamba_mds[1].mamba_slot_read_offsets,
        mamba_mds[2].mamba_state_indices,
        mamba_mds[2].mamba_ckpt_indices,
        mamba_mds[2].mamba_slot_read_offsets,
        bucket_t=4,
        tokens_per_seq=4,
        max_pages_per_seq=2,
        has_roff=True,
        use_ckpt_2d=True,
    )
    assert roff_ba[0, 0].item() == 2
    hid_in_ta = torch.nn.functional.pad(
        driver.embed_tokens(ids_ba).to(torch.bfloat16), (0, 0, 0, ex_t4.M - 4)
    )
    hid_ref_ta, ref_pools_j = step_ref_t4(
        _torch_to_jax(pos_ba, unsh),
        _torch_to_jax(hid_in_ta, tp),
        w_mk_j,
        ref_pools_j,
        _torch_to_jax(pt_ba, unsh),
        _torch_to_jax(ckpt_ba, unsh),
        _torch_to_jax(roff_ba, unsh),
    )
    np.testing.assert_array_equal(
        out_t4_acc.view(torch.uint8).numpy(),
        np.asarray(hid_ref_ta[:4]).view(np.uint8),
    )
    for p_t, p_ref in zip(raw_pools_t, ref_pools_j, strict=True):
        np.testing.assert_array_equal(
            p_t.view(torch.uint8).numpy(),
            np.asarray(p_ref).view(np.uint8),
        )
