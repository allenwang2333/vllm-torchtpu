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
"""CPU tests for the Qwen3.5 megakernel bucket policy: verify buckets, per-bucket tiles,
fused-vs-3-stage pool I/O, tile env switches, the fused pool I/O fallback, MK_AB hooks and the
rank-ordered mesh."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import time

import pytest
import torch
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.qwen35_megakernel import megakernel as K
from vllm_torchtpu.kernels.qwen35_megakernel import mtp_adapter
from vllm_torchtpu.kernels.qwen35_megakernel import vllm_adapter as VA

VERIFY_ENV = (
    "VLLM_TPU_QWEN35_MK_VERIFY_BUCKETS",
    "VLLM_TPU_QWEN35_MK_VERIFY_48",
    "VLLM_TPU_QWEN35_MK_VERIFY_64",
)
POOL_IO_ENV = (
    "VLLM_TPU_QWEN35_MK_FUSED_POOL_IO",
    "VLLM_TPU_QWEN35_MK_FUSED_POOL_IO_MAX_B",
    "VLLM_TPU_QWEN35_MK_FUSED_POOL_IO_MIN_B_HI",
)
TILE_ENV = (
    "VLLM_TPU_QWEN35_MK_ROUTE_VEC",
    "VLLM_TPU_QWEN35_MK_DENSE_ACC",
    "VLLM_TPU_QWEN35_MK_R2_DIRECT",
    "VLLM_TPU_QWEN35_MK_AMASK",
    "VLLM_TPU_QWEN35_MK_ECOPY_OPT",
    "VLLM_TPU_QWEN35_MK_NBE_SMALL",
)
ALL_VERIFY = (4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64)

# The v21 per-bucket tiles at R=8: B -> (nbd, nbe in EP mode, nbe in TP mode, nkv, hier, rsag).
R8_TILES = {
    4: (14, 20, 10, 3, 1, False),
    8: (14, 20, 10, 3, 2, True),
    12: (14, 12, 10, 4, 2, True),
    16: (14, 12, 10, 4, 2, True),
    20: (12, 12, 10, 3, 2, True),
    24: (12, 12, 10, 3, 2, True),
    32: (10, 12, 10, 3, 2, True),
    40: (8, 12, 10, 3, 2, True),
    48: (8, 12, 10, 3, 2, True),
    56: (8, 12, 10, 3, 2, True),
    64: (7, 12, 10, 3, 2, True),
}


@pytest.fixture
def reload_adapter(monkeypatch):
    """Re-import vllm_adapter under the test's env; restore the default module afterwards."""
    for name in VERIFY_ENV:
        monkeypatch.delenv(name, raising=False)
    yield lambda: importlib.reload(VA)
    monkeypatch.undo()
    importlib.reload(VA)


def _bare_mk(monkeypatch, **attrs) -> VA.Qwen35Megakernel:
    """A `Qwen35Megakernel` without `__init__` (no runner / model / TPU)."""
    for name in POOL_IO_ENV:
        monkeypatch.delenv(name, raising=False)
    mk = object.__new__(VA.Qwen35Megakernel)
    mk.fused_pool_io = True
    mk.fused_pool_io_max_b = VA.parse_mk_fused_pool_io_max_b()
    mk.fused_pool_io_min_b_hi = VA.parse_mk_fused_pool_io_min_b_hi()
    mk._bucket_steps, mk._bucket_ops, mk._bucket_out_rows = {}, {}, {}
    mk._bucket_fused, mk._prep_steps = {}, {}
    for k, v in attrs.items():
        setattr(mk, k, v)
    return mk


def test_default_verify_buckets_cover_16_sequences(reload_adapter, monkeypatch):
    mod = reload_adapter()
    assert mod._DEFAULT_VERIFY_BUCKETS == ALL_VERIFY
    assert mod.parse_mk_verify_buckets() == ALL_VERIFY
    assert mod.parse_mk_verify_buckets("4,8,64") == (4, 8, 64)

    monkeypatch.setenv("VLLM_TPU_QWEN35_MK_VERIFY_64", "0")
    assert reload_adapter()._DEFAULT_VERIFY_BUCKETS == ALL_VERIFY[:9]
    monkeypatch.setenv("VLLM_TPU_QWEN35_MK_VERIFY_48", "0")
    assert reload_adapter()._DEFAULT_VERIFY_BUCKETS == ALL_VERIFY[:7]


@pytest.mark.parametrize("b", sorted(R8_TILES))
def test_default_tiles_r8_match_v21(b):
    nbd, nbe_ep, nbe_tp, nkv, hier, rsag = R8_TILES[b]
    for base, nbe in ((K.FULL_EP, nbe_ep), (K.FULL, nbe_tp)):
        t = K.default_tiles(dataclasses.replace(base, B=b, T=4))
        assert (t.nbd, t.nbe, t.nkv, t.hier, t.rsag) == (nbd, nbe, nkv, hier, rsag), b
        assert t.ecopy_opt == 1 and t.amask == 1 and t.kv_sub == 1024
        assert t.route_vec and t.dense_acc_pair and t.r2_direct_store
        assert t.kv_ering and t.pair_kv


def test_default_tiles_r16_only_change_the_all_reduce():
    for b in sorted(R8_TILES):
        for base in (K.FULL_EP, K.FULL):
            t8 = K.default_tiles(dataclasses.replace(base, B=b))
            t16 = K.default_tiles(dataclasses.replace(base, R=16, B=b))
            assert t16.hier == (3 if b <= 32 else 0), b
            assert dataclasses.replace(t16, hier=t8.hier) == t8, b
            if b > 32:
                assert t16.rsag


def test_tp8_geometry_is_unpadded_and_tp16_scratch_fits():
    """R=8 keeps v21's checkpoint geometry and unpadded widths; at R=16 the B > 32 verify
    buckets (flat rsag all-reduce) need no more VMEM scratch than at R=8."""
    for base in (K.FULL_EP, K.FULL):
        c8 = dataclasses.replace(base, paged_kv=True)
        assert (c8.rec_pages, c8.gdn_pages) == (4, 5)
        assert c8.ISR == c8.IS // 8 and c8.IR == (c8.I // 8 if c8.moe_tp else c8.I)
    for b in (40, 64):
        for base in (K.FULL_EP, K.FULL):
            vm = {}
            for r in (8, 16):
                fused = b >= 56
                c = dataclasses.replace(base, R=r, B=b, T=4, paged_kv=True,
                                        fused_pool_io=fused, ckpt_k=4)
                tiles = K.default_tiles(c)
                _, info = K.build(c, tiles, pool_gather=not fused)
                vm[r] = info["vmem_scratch_bytes"]
                assert vm[r] <= tiles.vmem_mib * 2**20, (r, b)
            assert vm[16] <= vm[8], (b, vm)


def test_fused_pool_io_bucket_selection(monkeypatch):
    mk = _bare_mk(monkeypatch)
    assert (mk.fused_pool_io_max_b, mk.fused_pool_io_min_b_hi) == (24, 56)
    assert [b for b in ALL_VERIFY if not mk._bucket_is_fused(b)] == [32, 40, 48]
    mk.fused_pool_io_min_b_hi = VA.parse_mk_fused_pool_io_min_b_hi("0")
    assert [b for b in ALL_VERIFY if mk._bucket_is_fused(b)] == [4, 8, 12, 16, 20, 24]
    mk.fused_pool_io_max_b = VA.parse_mk_fused_pool_io_max_b("32")
    assert [b for b in ALL_VERIFY if mk._bucket_is_fused(b)][-1] == 32
    assert VA.parse_mk_fused_pool_io("0") is False
    assert VA.parse_mk_zero_copy(None) is False


def test_tile_env_overrides(monkeypatch):
    for name in TILE_ENV:
        monkeypatch.delenv(name, raising=False)
    base = K.default_tiles(dataclasses.replace(K.FULL_EP, B=8))
    assert VA.apply_mk_tile_env_overrides(base, 8) == base
    monkeypatch.setenv("VLLM_TPU_QWEN35_MK_ROUTE_VEC", "0")
    monkeypatch.setenv("VLLM_TPU_QWEN35_MK_DENSE_ACC", "off")
    monkeypatch.setenv("VLLM_TPU_QWEN35_MK_AMASK", "2")
    monkeypatch.setenv("VLLM_TPU_QWEN35_MK_NBE_SMALL", "12")
    t8 = VA.apply_mk_tile_env_overrides(base, 8)
    assert (t8.route_vec, t8.dense_acc_pair, t8.r2_direct_store) == (False, False, True)
    assert (t8.amask, t8.nbe) == (2, 12)
    # NBE_SMALL only applies to the 1..8-row buckets
    assert VA.apply_mk_tile_env_overrides(base, 12).nbe == base.nbe


def test_fused_pool_io_fallback_rebuilds_only_fused_buckets(monkeypatch):
    mk = _bare_mk(monkeypatch, buckets=(8,), verify_buckets=(4, 32, 64))
    built = []

    def fake_build(b, tokens_per_seq=1, out_pad_rows=None):
        fused = bool(mk.fused_pool_io and mk._bucket_is_fused(b))
        mk._bucket_fused[(tokens_per_seq, b)] = fused
        built.append((tokens_per_seq, b, fused))
        return object()

    mk._build_bucket_op = fake_build
    mk._build_all_buckets()
    assert built == [(1, 8, True), (4, 4, True), (4, 32, False), (4, 64, True)]
    kept = mk._bucket_steps[(4, 32)]
    assert mk._bucket_steps[8] is mk._bucket_steps[(1, 8)]

    built.clear()
    mk.fused_pool_io = False
    mk._build_all_buckets(only_fused=True)
    assert built == [(1, 8, False), (4, 4, False), (4, 64, False)]
    assert mk._bucket_steps[(4, 32)] is kept
    assert not any(mk._bucket_fused.values())


def test_build_bucket_op_names_and_hlo_fallback(monkeypatch):
    """Op names carry `_3stage` exactly for 3-stage buckets, including a bucket that falls
    back from fused pool I/O when the HLO guard fails."""
    c = dataclasses.replace(K.FULL_EP, stack_e=False, e_slots=72, paged_kv=True,
                            num_pages=4096, max_pages_per_seq=64, ctx=8192)
    n_w, n_p = 20, c.NATT
    names, made = [], []

    def fake_step(mesh, cfg, tiles, fused_pool_io=None):
        made.append((cfg.B, bool(fused_pool_io), cfg.fused_pool_io))
        n_in = 2 + n_w + n_p + 3
        return (lambda *a: a), (P("tp"),) * n_in, (P("tp"),) * (1 + n_p), ()

    def fake_callable(name, fn, **kw):
        names.append(name)
        return fn

    monkeypatch.setattr(VA.pool_io, "make_flat_pooled_step", fake_step)
    monkeypatch.setattr(VA, "sharded_jax_callable", fake_callable)
    monkeypatch.setattr(torch, "compile", lambda f, **kw: f)
    for mode, hlo_ok in (("utok", True), ("utfb", False)):
        names.clear()
        made.clear()
        mk = _bare_mk(
            monkeypatch,
            mode=mode,
            base_cfg=c,
            mesh=None,
            w_flat=[None] * n_w,
            raw_pools=[None] * n_p,
            max_pages_per_seq=c.max_pages_per_seq,
            zero_copy=False,
            _hlo_checked=False,
        )
        mk._verify_fused_pool_hlo = lambda *a, ok=hlo_ok: (ok, "forced")
        for b in (4, 32, 64):
            mk._build_bucket_op(b, tokens_per_seq=4, out_pad_rows=b)
        suffix = {b: n.endswith("_3stage") for b, n in zip((4, 32, 64), names)}
        if hlo_ok:
            assert suffix == {4: False, 32: True, 64: False}
            assert mk._bucket_fused == {(4, 4): True, (4, 32): False, (4, 64): True}
            assert mk.fused_pool_io
        else:
            # B=4 is rebuilt 3-stage after the failed guard and named as such
            assert suffix == {4: True, 32: True, 64: True}
            assert made[:2] == [(4, True, True), (4, False, False)]
            assert not any(mk._bucket_fused.values()) and not mk.fused_pool_io


def test_mk_ab_calibration_hooks(monkeypatch):
    mk = _bare_mk(
        monkeypatch,
        tp_rank=0,
        _step_mix={},
        _step_mix_t0=time.monotonic(),
        _step_mix_period_s=1e9,
        ab_calibrator=VA.MKABCalibrator(enabled=True, period_s=1e9),
    )
    assert mk._note_step("mk_t4_b16", True) is True
    assert mk._note_step("mk_t4_b16", True) is False  # alternate step on stock
    assert mk._note_step("stock_t4_n13", False) is False
    assert mk._step_mix == {"mk_t4_b16": 1, "ab_stock_t4_b16": 1, "stock_t4_n13": 1}
    mk.record_ab_step(12.5, padded_tokens=16)
    mk.reset_stats()
    assert mk._step_mix == {} and mk.ab_calibrator.counts == {}


def test_megakernel_meshes_are_rank_ordered():
    """Every megakernel mesh comes from `_build_tp_rank_mesh` (device-id sorted meshes put
    `axis_index('tp')` != tp_rank on tpu7x)."""
    for mod in (VA, mtp_adapter):
        assert not hasattr(mod, "build_ep_mesh")
        assert "build_ep_mesh(" not in inspect.getsource(mod)
    for cls in (VA.Qwen35Megakernel, mtp_adapter.Qwen35MtpMegakernel):
        assert "_build_tp_rank_mesh(self.tp_size)" in inspect.getsource(cls.__init__)
