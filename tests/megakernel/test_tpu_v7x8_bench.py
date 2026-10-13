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
"""On-TPU v7x-8 (2x2x1, 8 TensorCores) FULL_EP parity & latency benchmark."""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import time
from typing import Any

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SRC_DIR = str(_REPO_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from jax import lax  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.megakernel import kernel, planner  # noqa: E402
from vllm_torchtpu.megakernel.blocks.gdn import GDNBlock  # noqa: E402
from vllm_torchtpu.megakernel.models import qwen35  # noqa: E402
from vllm_torchtpu.megakernel.ops import (  # noqa: E402
    BF16,
    F32,
    FP8,
    I32,
    rup,
)
from vllm_torchtpu.megakernel.spec import ExecSpec  # noqa: E402


def _has_tpu_v7x8() -> bool:
    try:
        return len(jax.devices("tpu")) >= 8
    except RuntimeError:
        return False


def _load_frozen_mk():
    env_dir = os.environ.get("MK_FROZEN_QWEN35_DIR")
    frozen_dir = pathlib.Path(env_dir) if env_dir else (_REPO_ROOT / "frozen_qwen35")
    if not frozen_dir.exists():
        pytest.skip(
            f"Set MK_FROZEN_QWEN35_DIR to frozen qwen35 path (checked {frozen_dir})"
        )
    if str(frozen_dir) not in sys.path:
        sys.path.insert(0, str(frozen_dir))
    spec_f = importlib.util.spec_from_file_location(
        "_frozen_qwen35_mk", frozen_dir / "megakernel.py"
    )
    assert spec_f is not None and spec_f.loader is not None
    frozen_mk = importlib.util.module_from_spec(spec_f)
    sys.modules[spec_f.name] = frozen_mk
    spec_f.loader.exec_module(frozen_mk)
    return frozen_mk


def _init_shared_weights_on_tpu(
    mesh: Mesh,
    g: qwen35.Qwen35Geometry,
    ex: ExecSpec,
    f_lay: dict[str, tuple[int, int]],
    f_S: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Allocate 60-layer hybrid FULL_EP (att_every=4) weights on TPU HBM once."""
    ms = qwen35.qwen35_spec(g)
    p = planner.plan(ms, ex)
    nt, tk = p.nt, p.tk
    l_lay = p.smalls_layout
    l_S = p.smalls_width

    R = ex.R
    L, H, E, HD, VD = g.L, g.H, g.E, g.HD, g.VD
    nh = g.nh(R)
    G = g.G(R)
    vh = g.vh(R)
    qkv = g.QKV(R)
    nli = g.NLI(R)
    nai = g.NAI(R)
    in_s_w = max(nli, nai)
    isr, ir, ew = g.ISR(R), g.IR(R), g.EW(R)
    ngdn = g.NGDN
    natt = g.NATT
    koff = nh * 2 * HD

    out_specs = {
        "lin_in": P("tp"),
        "lin_out": P("tp"),
        "att_in": P("tp"),
        "att_out": P("tp"),
        "router": P("tp"),
        "sh13": P("tp"),
        "sh2": P("tp"),
        "e13": P("tp"),
        "e2": P("tp"),
        "es": P("tp"),
        "smalls_lib": P("tp"),
        "smalls_f": P("tp"),
        "final_norm": P("tp"),
    }

    def _local_init():
        me = lax.axis_index("tp")
        kv_grp = lax.div(me, G)
        root = jax.random.PRNGKey(20261013)

        def key_for(tag: int, rank_dep: Any = None):
            k = jax.random.fold_in(root, tag)
            if rank_dep is not None:
                k = jax.random.fold_in(k, rank_dep)
            return k

        lin_in = (
            jax.random.normal(
                key_for(101, me),
                (1, ngdn, nt["lin_in"], tk["lin_in"], nli),
                dtype=BF16,
            )
            * jnp.bfloat16(0.125)
        ).astype(FP8)
        lin_out = (
            jax.random.normal(
                key_for(102, me),
                (1, ngdn, nt["lin_out"], tk["lin_out"], H),
                dtype=BF16,
            )
            * jnp.bfloat16(0.125)
        ).astype(FP8)

        att_q = (
            jax.random.normal(
                key_for(1, me),
                (1, natt, nt["att_in"], tk["att_in"], koff),
                dtype=BF16,
            )
            * jnp.bfloat16(0.125)
        ).astype(FP8)
        att_kv = (
            jax.random.normal(
                key_for(2, kv_grp),
                (1, natt, nt["att_in"], tk["att_in"], 2 * HD),
                dtype=BF16,
            )
            * jnp.bfloat16(0.125)
        ).astype(FP8)
        att_in = jnp.concatenate([att_q, att_kv], axis=-1)

        att_out = (
            jax.random.normal(
                key_for(3, me),
                (1, natt, nt["att_out"], tk["att_out"], H),
                dtype=BF16,
            )
            * jnp.bfloat16(0.125)
        ).astype(FP8)

        router = jax.random.normal(
            key_for(4),
            (1, L, nt["router"], tk["router"], E),
            dtype=BF16,
        ) * jnp.bfloat16(0.05)

        sh13 = (
            jax.random.normal(
                key_for(5, me),
                (1, L, nt["sh13"], tk["sh13"], 2 * isr),
                dtype=BF16,
            )
            * jnp.bfloat16(0.125)
        ).astype(FP8)
        sh2 = (
            jax.random.normal(
                key_for(6, me),
                (1, L, nt["sh2"], tk["sh2"], H),
                dtype=BF16,
            )
            * jnp.bfloat16(0.125)
        ).astype(FP8)

        k_e13_base = key_for(7, me)

        def _gen_e13(_, lyr):
            kl = jax.random.fold_in(k_e13_base, lyr)
            w = (
                jax.random.normal(kl, (ew, nt["e13"], tk["e13"], 2 * ir), dtype=BF16)
                * jnp.bfloat16(0.125)
            ).astype(FP8)
            return None, w

        _, e13_layers = lax.scan(_gen_e13, None, jnp.arange(L, dtype=I32))
        e13 = e13_layers[None, ...]

        k_e2_base = key_for(8, me)

        def _gen_e2(_, lyr):
            kl = jax.random.fold_in(k_e2_base, lyr)
            w = (
                jax.random.normal(kl, (ew, nt["e2"], tk["e2"], H), dtype=BF16)
                * jnp.bfloat16(0.125)
            ).astype(FP8)
            return None, w

        _, e2_layers = lax.scan(_gen_e2, None, jnp.arange(L, dtype=I32))
        e2 = e2_layers[None, ...]

        es = (
            jnp.abs(
                jax.random.normal(key_for(9, me), (1, L, ew, 1, 2 * ir + H), dtype=F32)
            )
            * 0.008
            + 0.004
        )

        in_s_parts = [
            jnp.abs(jax.random.normal(key_for(13, me), (1, L, koff), dtype=F32)) * 0.008
            + 0.004,
            jnp.abs(jax.random.normal(key_for(14, kv_grp), (1, L, 2 * HD), dtype=F32))
            * 0.008
            + 0.004,
        ]
        if in_s_w > nai:
            in_s_parts.append(
                jnp.abs(
                    jax.random.normal(key_for(103, me), (1, L, in_s_w - nai), dtype=F32)
                )
                * 0.008
                + 0.004
            )
        alog_vh = jax.random.normal(key_for(105, me), (1, L, vh), dtype=F32) * 0.1 - 1.0
        dtb_vh = jax.random.normal(key_for(106, me), (1, L, vh), dtype=F32) * 0.1
        fields = {
            "in_norm": jax.random.normal(key_for(10), (1, L, H), dtype=F32) * 0.02,
            "post_norm": jax.random.normal(key_for(11), (1, L, H), dtype=F32) * 0.02,
            "sh_gate": jax.random.normal(key_for(12), (1, L, H), dtype=F32) * 0.05,
            "in_s": jnp.concatenate(in_s_parts, axis=-1),
            "out_s": jnp.abs(jax.random.normal(key_for(15, me), (1, L, H), dtype=F32))
            * 0.008
            + 0.004,
            "conv": jax.random.normal(key_for(104, me), (1, L, 4 * qkv), dtype=F32)
            * 0.05,
            "alog": jnp.pad(alog_vh, ((0, 0), (0, 0), (0, 128 - vh))),
            "dtb": jnp.pad(dtb_vh, ((0, 0), (0, 0), (0, 128 - vh))),
            "gnorm": jnp.ones((1, L, VD), dtype=F32)
            + jax.random.normal(key_for(107, me), (1, L, VD), dtype=F32) * 0.02,
            "qn": jax.random.normal(key_for(16, me), (1, L, HD), dtype=F32) * 0.02,
            "kn": jax.random.normal(key_for(17, kv_grp), (1, L, HD), dtype=F32) * 0.02,
            "sh13_s": jnp.abs(
                jax.random.normal(key_for(18, me), (1, L, 2 * isr), dtype=F32)
            )
            * 0.008
            + 0.004,
            "sh2_s": jnp.abs(jax.random.normal(key_for(19, me), (1, L, H), dtype=F32))
            * 0.008
            + 0.004,
        }

        sm_lib = jnp.zeros((1, L, 1, l_S), dtype=F32)
        sm_f = jnp.zeros((1, L, 1, f_S), dtype=F32)
        for name, (lo, w) in l_lay.items():
            fo, _ = f_lay[name]
            val = fields[name][:, :, :w]
            sm_lib = sm_lib.at[:, :, 0, lo : lo + w].set(val)
            sm_f = sm_f.at[:, :, 0, fo : fo + w].set(val)

        final_norm = jax.random.normal(key_for(20), (1, 1, H), dtype=F32) * 0.02

        return {
            "lin_in": lin_in,
            "lin_out": lin_out,
            "att_in": att_in,
            "att_out": att_out,
            "router": router,
            "sh13": sh13,
            "sh2": sh2,
            "e13": e13,
            "e2": e2,
            "es": es,
            "smalls_lib": sm_lib,
            "smalls_f": sm_f,
            "final_norm": final_norm,
        }

    init_fn = jax.jit(
        jax.shard_map(
            _local_init,
            mesh=mesh,
            in_specs=(),
            out_specs=out_specs,
            check_vma=False,
        )
    )
    raw = init_fn()
    jax.block_until_ready(raw["e13"])

    w_lib = {
        "lin_in": raw["lin_in"],
        "lin_out": raw["lin_out"],
        "att_in": raw["att_in"],
        "att_out": raw["att_out"],
        "router": raw["router"],
        "sh13": raw["sh13"],
        "sh2": raw["sh2"],
        "e13": raw["e13"],
        "e2": raw["e2"],
        "es": raw["es"],
        "smalls": raw["smalls_lib"],
        "final_norm": raw["final_norm"],
    }
    w_f = {
        "lin_in": raw["lin_in"],
        "lin_out": raw["lin_out"],
        "att_in": raw["att_in"],
        "att_out": raw["att_out"],
        "router": raw["router"],
        "sh13": raw["sh13"],
        "sh2": raw["sh2"],
        "e13": raw["e13"],
        "e2": raw["e2"],
        "es": raw["es"],
        "smalls": raw["smalls_f"],
        "final_norm": raw["final_norm"],
    }
    return w_lib, w_f


def _init_bucket_inputs_on_tpu(
    mesh: Mesh,
    g: qwen35.Qwen35Geometry,
    ex: ExecSpec,
) -> tuple[jax.Array, jax.Array, dict[str, Any]]:
    """Allocate `(pos, hid, states)` for a single `(T, B)` hybrid bucket on TPU HBM."""
    rep = NamedSharding(mesh, P())
    R, B, T, M, NS, ctx = ex.R, ex.B, ex.T, ex.M, ex.NS, ex.ctx
    G = g.G(R)
    ngdn, natt, H, HD, KD, VD = g.NGDN, g.NATT, g.H, g.HD, g.KD, g.VD
    qkv = g.QKV(R)
    vh = g.vh(R)
    conv_shape = (
        (ngdn, rup(3 * B, 8), qkv) if T == 1 else (ngdn, NS, rup(3 + T, 8), qkv)
    )

    base_pos = jnp.int32(2048) + jnp.arange(NS, dtype=I32) * jnp.int32(16)
    if T == 1:
        pos_host = base_pos
    else:
        pos_host = (base_pos[:, None] + jnp.arange(T, dtype=I32)[None, :]).reshape(B)
    pos = jax.device_put(pos_host, rep)

    st_specs = {
        "conv": P("tp"),
        "rec": P("tp"),
        "kc": P("tp"),
        "vc": P("tp"),
    }

    def _local_bucket():
        me = lax.axis_index("tp")
        kv_grp = lax.div(me, G)
        root = jax.random.PRNGKey(99000 + T * 100 + B)
        k_hid = jax.random.fold_in(root, 1)
        k_kc_base = jax.random.fold_in(jax.random.fold_in(root, 2), kv_grp)
        k_vc_base = jax.random.fold_in(jax.random.fold_in(root, 3), kv_grp)
        k_conv = jax.random.fold_in(jax.random.fold_in(root, 4), me)
        k_rec_base = jax.random.fold_in(jax.random.fold_in(root, 5), me)

        hid_b = jax.random.normal(k_hid, (B, H), dtype=BF16) * jnp.bfloat16(0.1)
        hid_m = jnp.pad(hid_b, ((0, M - B), (0, 0)))

        conv = jax.random.normal(k_conv, conv_shape, dtype=F32) * jnp.float32(0.05)

        def _gen_rec(_, lyr):
            kr = jax.random.fold_in(k_rec_base, lyr)
            rec_l = jax.random.normal(kr, (NS * vh, KD, VD), dtype=BF16) * jnp.bfloat16(
                0.02
            )
            return None, rec_l

        _, rec = lax.scan(_gen_rec, None, jnp.arange(ngdn, dtype=I32))

        def _gen_kv(_, lyr):
            kk = jax.random.fold_in(k_kc_base, lyr)
            kv = jax.random.fold_in(k_vc_base, lyr)
            kc_l = jax.random.normal(kk, (NS, ctx, HD), dtype=BF16) * jnp.bfloat16(0.08)
            vc_l = jax.random.normal(kv, (NS, ctx, HD), dtype=BF16) * jnp.bfloat16(0.08)
            return None, (kc_l, vc_l)

        _, (kc, vc) = lax.scan(_gen_kv, None, jnp.arange(natt, dtype=I32))
        st: dict[str, Any] = {
            "conv": conv,
            "rec": rec,
            "kc": kc,
            "vc": vc,
        }
        return hid_m, st

    init_fn = jax.jit(
        jax.shard_map(
            _local_bucket,
            mesh=mesh,
            in_specs=(),
            out_specs=(P("tp"), st_specs),
            check_vma=False,
        )
    )
    hid, st = init_fn()
    jax.block_until_ready(hid)
    return pos, hid, st


def _count_routed_experts(
    hid: jax.Array,
    router_w: jax.Array,
    B: int,
    H: int,
    E: int,
    TOPK: int,
    R: int,
) -> tuple[float, float]:
    """Compute mean distinct routed experts hit across all 60 layers."""
    hid_0 = np.asarray(jax.device_get(kernel.unfold(hid, R)[0, :B])).astype(np.float32)
    router_0 = np.asarray(jax.device_get(router_w[0])).astype(np.float32)
    L = router_0.shape[0]
    w_r = router_0.reshape(L, H, E)
    logits = np.einsum("bh,lhe->lbe", hid_0, w_r)
    top_idx = np.argpartition(-logits, TOPK, axis=-1)[:, :, :TOPK]
    er = E // R
    uniq_global = [len(np.unique(top_idx[lyr])) for lyr in range(L)]
    uniq_per_rank = [
        np.mean(
            [len(np.unique(top_idx[lyr][(top_idx[lyr] // er) == r])) for r in range(R)]
        )
        for lyr in range(L)
    ]
    return float(np.mean(uniq_global)), float(np.mean(uniq_per_rank))


def _time_step_fn(
    step_fn,
    pos: jax.Array,
    hid: jax.Array,
    weights: dict[str, Any],
    states: dict[str, Any],
    *,
    T: int,
    warmup: int = 15,
    batches: int = 10,
    steps_per_batch: int = 10,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Warm up and measure step latency over `batches * steps_per_batch` iterations."""
    for _ in range(warmup):
        out_hid, out_st = step_fn(pos, hid, weights, states)
        if T == 1:
            states = out_st
    jax.block_until_ready(out_hid)

    batch_us = []
    for _ in range(batches):
        t0 = time.perf_counter()
        for _ in range(steps_per_batch):
            out_hid, out_st = step_fn(pos, hid, weights, states)
            if T == 1:
                states = out_st
        jax.block_until_ready(out_hid)
        dt_us = (time.perf_counter() - t0) * 1e6 / steps_per_batch
        batch_us.append(dt_us)

    arr = np.asarray(batch_us, dtype=np.float64)
    stats = {
        "median_us": float(np.median(arr)),
        "mean_us": float(np.mean(arr)),
        "p10_us": float(np.percentile(arr, 10)),
        "p90_us": float(np.percentile(arr, 90)),
        "min_us": float(np.min(arr)),
    }
    return stats, states


def test_m2_tpu_v7x8_full_ep_parity_and_latency():
    """Run FULL_EP (L=60, att_every=4, R=8, ctx=4096) on-TPU v7x-8 benchmark."""
    if not _has_tpu_v7x8():
        pytest.skip("Requires 8 TPU TensorCores (v7x-8)")
    tpu_devs = jax.devices("tpu")
    frozen_mk = _load_frozen_mk()
    mesh = Mesh(tpu_devs[:8], ("tp",))
    rep = NamedSharding(mesh, P())

    g = dataclasses.replace(qwen35.FULL_EP, L=60, att_every=4)
    ex_init = ExecSpec(B=4, T=1, R=8, ctx=4096, stack_e=True)
    c_f_init = frozen_mk.MKConfig(
        R=8,
        B=4,
        T=1,
        L=60,
        att_every=4,
        moe_tp=False,
        ctx=4096,
        stack_e=True,
    )
    f_lay, f_S = frozen_mk.smalls_layout(c_f_init)

    t_w0 = time.perf_counter()
    w_lib, w_f = _init_shared_weights_on_tpu(mesh, g, ex_init, f_lay, f_S)
    dt_w = time.perf_counter() - t_w0
    print(
        f"\n[M2 TPU v7x-8 Bench] Allocated shared 60-layer hybrid FULL_EP "
        f"(att_every=4) weights on 8 TPU cores in {dt_w:.2f}s"
    )

    buckets = [
        (1, 4),
        (1, 8),
        (1, 16),
        (1, 32),
        (1, 64),
        (4, 4),
        (4, 8),
        (4, 16),
        (4, 32),
        (4, 64),
    ]
    results = []

    for T, B in buckets:
        ex = ExecSpec(B=B, T=T, R=8, ctx=4096, stack_e=True)
        c_f = frozen_mk.MKConfig(
            R=8,
            B=B,
            T=T,
            L=60,
            att_every=4,
            moe_tp=False,
            ctx=4096,
            stack_e=True,
        )
        tiles_f = frozen_mk.default_tiles(c_f)
        if B > 48:
            tiles_f = dataclasses.replace(tiles_f, nbe=6)
        donate = T == 1
        conv_key = "conv" if T == 1 else "conv_ckpt"
        rec_key = "rec" if T == 1 else "rec_ckpt"

        # --- 1. Library megakernel: compile, parity snapshot, and latency ---
        pos, hid_lib, st_lib = _init_bucket_inputs_on_tpu(mesh, g, ex)
        uniq_glob, uniq_rank = _count_routed_experts(
            hid_lib, w_lib["router"], B, g.H, g.E, g.TOPK, 8
        )
        step_lib, info_lib = qwen35.make_step(mesh, g, ex, donate_states=donate)
        vmem_lib = info_lib["vmem_scratch_bytes"]
        comp_lib = step_lib.lower(pos, hid_lib, w_lib, st_lib).compile()
        lib_sha = hashlib.sha256(comp_lib.as_text().encode("utf-8")).hexdigest()[:16]
        del comp_lib

        out_hid_lib, out_st_lib = step_lib(pos, hid_lib, w_lib, st_lib)
        jax.block_until_ready(out_hid_lib)
        if donate:
            st_lib = out_st_lib
        elif T > 1:
            acc_dev = jax.device_put((jnp.arange(ex.NS, dtype=I32) % T) + 1, rep)
            vh = g.vh(8)
            acc_st = jax.jit(
                lambda s, a, _t=T, _vh=vh: GDNBlock.accept(s, s, a, T=_t, vh=_vh)
            )(out_st_lib, acc_dev)
            jax.block_until_ready(acc_st["rec"])
            assert acc_st["conv"].shape == st_lib["conv"].shape
            assert acc_st["rec"].shape == st_lib["rec"].shape
            del acc_st, acc_dev

        hid_lib_np = np.asarray(jax.device_get(kernel.unfold(out_hid_lib, 8)))
        conv_unfold_lib = kernel.unfold(out_st_lib[conv_key], 8)
        rec_unfold_lib = kernel.unfold(out_st_lib[rec_key], 8)
        kc_unfold_lib = kernel.unfold(out_st_lib["kc"], 8)
        vc_unfold_lib = kernel.unfold(out_st_lib["vc"], 8)
        conv_snap_lib = np.asarray(jax.device_get(conv_unfold_lib[:, :4, ...]))
        rec_snap_lib = np.asarray(jax.device_get(rec_unfold_lib[:, :4, :, :16, :64]))
        kc_snap_lib = np.asarray(
            jax.device_get(kc_unfold_lib[:, :4, 0, 2048 : 2048 + T, :])
        )
        vc_snap_lib = np.asarray(
            jax.device_get(vc_unfold_lib[:, :4, 0, 2048 : 2048 + T, :])
        )
        del conv_unfold_lib, rec_unfold_lib, kc_unfold_lib, vc_unfold_lib, out_st_lib

        lib_stats, st_lib = _time_step_fn(step_lib, pos, hid_lib, w_lib, st_lib, T=T)
        del pos, hid_lib, st_lib, out_hid_lib, step_lib
        gc.collect()

        # --- 2. Frozen megakernel: compile, parity snapshot, and latency ---
        pos_f, hid_f, st_f = _init_bucket_inputs_on_tpu(mesh, g, ex)
        raw_step_f, info_f = frozen_mk.make_step(mesh, c_f, tiles_f)
        vmem_f = info_f["vmem_scratch_bytes"]
        step_f = jax.jit(raw_step_f, donate_argnums=(3,) if donate else ())
        comp_f = step_f.lower(pos_f, hid_f, w_f, st_f).compile()
        f_sha = hashlib.sha256(comp_f.as_text().encode("utf-8")).hexdigest()[:16]
        del comp_f

        out_hid_f, out_st_f = step_f(pos_f, hid_f, w_f, st_f)
        jax.block_until_ready(out_hid_f)
        if donate:
            st_f = out_st_f
        hid_f_np = np.asarray(jax.device_get(kernel.unfold(out_hid_f, 8)))
        conv_unfold_f = kernel.unfold(out_st_f["conv"], 8)
        rec_unfold_f = kernel.unfold(out_st_f["rec"], 8)
        kc_unfold_f = kernel.unfold(out_st_f["kc"], 8)
        vc_unfold_f = kernel.unfold(out_st_f["vc"], 8)
        conv_snap_f = np.asarray(jax.device_get(conv_unfold_f[:, :4, ...]))
        rec_snap_f = np.asarray(jax.device_get(rec_unfold_f[:, :4, :, :16, :64]))
        kc_snap_f = np.asarray(
            jax.device_get(kc_unfold_f[:, :4, 0, 2048 : 2048 + T, :])
        )
        vc_snap_f = np.asarray(
            jax.device_get(vc_unfold_f[:, :4, 0, 2048 : 2048 + T, :])
        )
        del conv_unfold_f, rec_unfold_f, kc_unfold_f, vc_unfold_f, out_st_f

        frozen_stats, st_f = _time_step_fn(step_f, pos_f, hid_f, w_f, st_f, T=T)
        del pos_f, hid_f, st_f, out_hid_f, step_f, raw_step_f
        gc.collect()

        # --- 3. Parity verification ---
        live_lib = hid_lib_np[:, :B, :].astype(np.float32)
        live_f = hid_f_np[:, :B, :].astype(np.float32)
        assert np.all(np.isfinite(live_lib)), (
            f"Non-finite output in library at T={T}, B={B}"
        )
        assert np.std(live_lib) > 1e-4, (
            f"Trivial zero output in library at T={T}, B={B}"
        )

        hid_bit_exact = bool(np.array_equal(hid_lib_np[:, :B, :], hid_f_np[:, :B, :]))
        conv_bit_exact = bool(np.array_equal(conv_snap_lib, conv_snap_f))
        rec_bit_exact = bool(np.array_equal(rec_snap_lib, rec_snap_f))
        kc_bit_exact = bool(np.array_equal(kc_snap_lib, kc_snap_f))
        vc_bit_exact = bool(np.array_equal(vc_snap_lib, vc_snap_f))
        max_abs_diff = float(np.max(np.abs(live_lib - live_f)))
        denom = float(np.linalg.norm(live_lib) * np.linalg.norm(live_f))
        cos_sim = float(np.sum(live_lib * live_f) / denom) if denom > 0 else 1.0

        delta_pct = (
            (lib_stats["median_us"] - frozen_stats["median_us"])
            / frozen_stats["median_us"]
            * 100.0
        )
        row = {
            "T": T,
            "B": B,
            "NS": B // T,
            "GCH": B > 16,
            "uniq_experts_global": round(uniq_glob, 1),
            "uniq_experts_per_rank": round(uniq_rank, 1),
            "lib_median_us": round(lib_stats["median_us"], 1),
            "frozen_median_us": round(frozen_stats["median_us"], 1),
            "delta_pct": round(delta_pct, 2),
            "lib_mean_us": round(lib_stats["mean_us"], 1),
            "frozen_mean_us": round(frozen_stats["mean_us"], 1),
            "vmem_lib_mib": round(vmem_lib / (1024 * 1024), 2),
            "vmem_frozen_mib": round(vmem_f / (1024 * 1024), 2),
            "hid_bit_exact": hid_bit_exact,
            "conv_bit_exact": conv_bit_exact,
            "rec_bit_exact": rec_bit_exact,
            "kc_bit_exact": kc_bit_exact,
            "vc_bit_exact": vc_bit_exact,
            "max_abs_diff": max_abs_diff,
            "cos_sim": cos_sim,
            "lib_sha256": lib_sha,
            "frozen_sha256": f_sha,
            "sha256_match": lib_sha == f_sha,
        }
        results.append(row)
        print(
            f"[Bucket T={T} B={B:2d} (NS={B // T:2d}, GCH={B > 16}, "
            f"experts={row['uniq_experts_global']:.1f}/{g.E} global, "
            f"{row['uniq_experts_per_rank']:.1f}/{g.ER(8)} per-rank)] "
            f"lib={row['lib_median_us']:6.1f} us vs "
            f"frozen={row['frozen_median_us']:6.1f} us ({row['delta_pct']:+.2f}%) | "
            f"VMEM {row['vmem_lib_mib']} vs {row['vmem_frozen_mib']} MiB | "
            f"bit_exact(hid={hid_bit_exact}, conv={conv_bit_exact}, "
            f"rec={rec_bit_exact}, kc={kc_bit_exact}, vc={vc_bit_exact}) "
            f"sha_match={row['sha256_match']} "
            f"max_diff={max_abs_diff:.3e} cos={cos_sim:.8f}",
            flush=True,
        )

    print("\n=== M2_TPU_V7X8_BENCH_JSON_START ===")
    print(json.dumps(results, indent=2))
    print("=== M2_TPU_V7X8_BENCH_JSON_END ===", flush=True)

    for r in results:
        T, B = r["T"], r["B"]
        assert r["hid_bit_exact"], (
            f"Hidden state not bit-exact at T={T}, B={B}: max_diff={r['max_abs_diff']}"
        )
        assert r["conv_bit_exact"], f"Conv state not bit-exact at T={T}, B={B}"
        assert r["rec_bit_exact"], f"Rec state not bit-exact at T={T}, B={B}"
        assert r["kc_bit_exact"], f"KC state not bit-exact at T={T}, B={B}"
        assert r["vc_bit_exact"], f"VC state not bit-exact at T={T}, B={B}"

    # --- 4. Also measure M1 (att_every=1) B=64 (T=1, T=4) for PR #1837 ---
    g_m1 = dataclasses.replace(qwen35.FULL_EP, L=60, att_every=1)
    p_m1 = planner.plan(qwen35.qwen35_spec(g_m1), ex_init)
    m1_lay = p_m1.smalls_layout
    m1_S = p_m1.smalls_width

    def _build_m1_weights(att_in_15, att_out_15, sm_f_all):
        att_in_60 = jnp.tile(att_in_15, (1, 4, 1, 1, 1))
        att_out_60 = jnp.tile(att_out_15, (1, 4, 1, 1, 1))
        sm_m1 = jnp.zeros((1, 60, 1, m1_S), dtype=F32)
        for name, (lo, w) in m1_lay.items():
            fo, _ = f_lay[name]
            sm_m1 = sm_m1.at[:, :, 0, lo : lo + w].set(sm_f_all[:, :, 0, fo : fo + w])
        lin_in_0 = jnp.zeros((1, 1, 8, 512, g.NLI(8)), dtype=FP8)
        lin_out_0 = jnp.zeros((1, 1, 2, 512, g.H), dtype=FP8)
        return att_in_60, att_out_60, sm_m1, lin_in_0, lin_out_0

    att_in_60, att_out_60, sm_m1, lin_in_0, lin_out_0 = jax.jit(
        jax.shard_map(
            _build_m1_weights,
            mesh=mesh,
            in_specs=(P("tp"), P("tp"), P("tp")),
            out_specs=(P("tp"), P("tp"), P("tp"), P("tp"), P("tp")),
            check_vma=False,
        )
    )(w_lib["att_in"], w_lib["att_out"], w_f["smalls"])
    jax.block_until_ready(att_in_60)

    del w_lib["lin_in"], w_lib["lin_out"], w_f["lin_in"], w_f["lin_out"]
    w_m1_lib = {
        "att_in": att_in_60,
        "att_out": att_out_60,
        "router": w_lib["router"],
        "sh13": w_lib["sh13"],
        "sh2": w_lib["sh2"],
        "e13": w_lib["e13"],
        "e2": w_lib["e2"],
        "es": w_lib["es"],
        "smalls": sm_m1,
        "final_norm": w_lib["final_norm"],
    }
    w_m1_f = {
        **w_m1_lib,
        "lin_in": lin_in_0,
        "lin_out": lin_out_0,
        "smalls": w_f["smalls"],
    }

    m1_results = []
    for T, B in [(1, 64), (4, 64)]:
        ex_m1 = ExecSpec(B=B, T=T, R=8, ctx=4096, stack_e=True)
        c_f_m1 = frozen_mk.MKConfig(
            R=8,
            B=B,
            T=T,
            L=60,
            att_every=1,
            moe_tp=False,
            ctx=4096,
            stack_e=True,
        )
        tiles_f_m1 = frozen_mk.default_tiles(c_f_m1)
        if B > 48:
            tiles_f_m1 = dataclasses.replace(tiles_f_m1, nbe=6)
        pos_m1, hid_m1, st_2 = _init_bucket_inputs_on_tpu(mesh, g, ex_m1)

        def _tile_kv(kc_15, vc_15, _b=B, _t=T):
            kc_60 = jnp.tile(kc_15, (4, 1, 1, 1))
            vc_60 = jnp.tile(vc_15, (4, 1, 1, 1))
            conv_0 = (
                jnp.zeros((1, rup(3 * _b, 8), g.QKV(8)), dtype=F32)
                if _t == 1
                else jnp.zeros((1, _b // _t, rup(3 + _t, 8), g.QKV(8)), dtype=F32)
            )
            rec_0 = jnp.zeros((1, (_b // _t) * g.vh(8), g.KD, g.VD), dtype=BF16)
            return kc_60, vc_60, conv_0, rec_0

        kc_60, vc_60, conv_0, rec_0 = jax.jit(
            jax.shard_map(
                _tile_kv,
                mesh=mesh,
                in_specs=(P("tp"), P("tp")),
                out_specs=(P("tp"), P("tp"), P("tp"), P("tp")),
                check_vma=False,
            )
        )(st_2["kc"], st_2["vc"])
        del st_2
        st_m1_lib = {"kc": kc_60, "vc": vc_60}
        step_m1_lib, info_m1_lib = qwen35.make_step(
            mesh, g_m1, ex_m1, donate_states=(T == 1)
        )
        out_hid_m1_lib, out_st_m1_lib = step_m1_lib(pos_m1, hid_m1, w_m1_lib, st_m1_lib)
        jax.block_until_ready(out_hid_m1_lib)
        if T == 1:
            st_m1_lib = out_st_m1_lib
        hid_m1_lib_np = np.asarray(jax.device_get(kernel.unfold(out_hid_m1_lib, 8)))
        del out_st_m1_lib
        m1_lib_stats, st_m1_lib = _time_step_fn(
            step_m1_lib, pos_m1, hid_m1, w_m1_lib, st_m1_lib, T=T
        )
        del st_m1_lib, out_hid_m1_lib, step_m1_lib
        gc.collect()

        pos_m1_f, hid_m1_f, st_2_f = _init_bucket_inputs_on_tpu(mesh, g, ex_m1)
        kc_60_f, vc_60_f, conv_0_f, rec_0_f = jax.jit(
            jax.shard_map(
                _tile_kv,
                mesh=mesh,
                in_specs=(P("tp"), P("tp")),
                out_specs=(P("tp"), P("tp"), P("tp"), P("tp")),
                check_vma=False,
            )
        )(st_2_f["kc"], st_2_f["vc"])
        del st_2_f
        st_m1_f = {
            "conv": conv_0_f,
            "rec": rec_0_f,
            "kc": kc_60_f,
            "vc": vc_60_f,
        }
        raw_step_m1_f, info_m1_f = frozen_mk.make_step(mesh, c_f_m1, tiles_f_m1)
        step_m1_f = jax.jit(raw_step_m1_f, donate_argnums=(3,) if T == 1 else ())
        out_hid_m1_f, out_st_m1_f = step_m1_f(pos_m1_f, hid_m1_f, w_m1_f, st_m1_f)
        jax.block_until_ready(out_hid_m1_f)
        if T == 1:
            st_m1_f = out_st_m1_f
        hid_m1_f_np = np.asarray(jax.device_get(kernel.unfold(out_hid_m1_f, 8)))
        del out_st_m1_f
        m1_f_stats, st_m1_f = _time_step_fn(
            step_m1_f, pos_m1_f, hid_m1_f, w_m1_f, st_m1_f, T=T
        )
        del (
            pos_m1,
            hid_m1,
            pos_m1_f,
            hid_m1_f,
            st_m1_f,
            out_hid_m1_f,
            step_m1_f,
            raw_step_m1_f,
        )
        gc.collect()

        m1_bit_exact = bool(
            np.array_equal(hid_m1_lib_np[:, :B, :], hid_m1_f_np[:, :B, :])
        )
        m1_delta = (
            (m1_lib_stats["median_us"] - m1_f_stats["median_us"])
            / m1_f_stats["median_us"]
            * 100.0
        )
        m1_row = {
            "T": T,
            "B": B,
            "NS": B // T,
            "lib_median_us": round(m1_lib_stats["median_us"], 1),
            "frozen_median_us": round(m1_f_stats["median_us"], 1),
            "delta_pct": round(m1_delta, 2),
            "lib_mean_us": round(m1_lib_stats["mean_us"], 1),
            "frozen_mean_us": round(m1_f_stats["mean_us"], 1),
            "vmem_lib_mib": round(info_m1_lib["vmem_scratch_bytes"] / (1024 * 1024), 2),
            "vmem_frozen_mib": round(
                info_m1_f["vmem_scratch_bytes"] / (1024 * 1024), 2
            ),
            "hid_bit_exact": m1_bit_exact,
        }
        m1_results.append(m1_row)
        print(
            f"[M1 (att_every=1) Bucket T={T} B={B:2d}] "
            f"lib={m1_row['lib_median_us']:6.1f} us vs "
            f"frozen={m1_row['frozen_median_us']:6.1f} us "
            f"({m1_row['delta_pct']:+.2f}%) | "
            f"VMEM {m1_row['vmem_lib_mib']} vs {m1_row['vmem_frozen_mib']} MiB | "
            f"bit_exact={m1_bit_exact}",
            flush=True,
        )
        assert m1_bit_exact, f"M1 B={B}, T={T} not bit-exact"

    print("\n=== M1_B64_TPU_V7X8_BENCH_JSON_START ===")
    print(json.dumps(m1_results, indent=2))
    print("=== M1_B64_TPU_V7X8_BENCH_JSON_END ===", flush=True)
