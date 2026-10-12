# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full statement and branch coverage tests for common/host_row_table.py."""

import os
import sys
import tempfile
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

# Provide lightweight stubs for `jax`,
# `vllm_torchtpu.kernels.deepseek_v41.host_row_gather`,
# `torch._functorch._activation_offloading.offload_ops`, and `torch_tpu` before
# import.
if "jax" not in sys.modules:
    jax_stub = types.ModuleType("jax")
    jax_sharding = types.ModuleType("jax.sharding")

    class _SingleDeviceSharding:
        def __init__(self, device, memory_kind="device"):
            self.device = device
            self.memory_kind = memory_kind

    class _ShapeDtypeStruct:
        def __init__(self, shape, dtype, sharding=None):
            self.shape = shape
            self.dtype = dtype
            self.sharding = sharding

    jax_sharding.SingleDeviceSharding = _SingleDeviceSharding
    jax_stub.sharding = jax_sharding
    jax_stub.ShapeDtypeStruct = _ShapeDtypeStruct
    jax_stub.Array = object
    jax_stub.Device = object
    jax_stub.devices = lambda: ["tpu0"]
    jax_src = types.ModuleType("jax._src")
    jax_src.xla_bridge = SimpleNamespace(
        get_backend=lambda: SimpleNamespace(platform="cpu"),
        backends_are_initialized=lambda: False,
    )
    jax_stub._src = jax_src
    sys.modules["jax._src"] = jax_src
    sys.modules["jax"] = jax_stub
    sys.modules["jax.sharding"] = jax_sharding

import jax  # noqa: E402

hrg_stub = types.ModuleType("vllm_torchtpu.kernels.deepseek_v41.host_row_gather")
hrg_stub.SUBLANES = 8
hrg_stub.MAJOR_LANES = 1024
hrg_stub.TILES_PER_MAJOR = 8
hrg_stub.row_words = lambda row_bytes: row_bytes // 4
hrg_stub.rows_per_tile = lambda row_bytes: 1024 // (row_bytes // 4)


def _fake_pack_rows(arr: np.ndarray, majors: int) -> np.ndarray:
    out = np.zeros((majors, 8, 1024), dtype=np.int32)
    flat = arr.reshape(-1)
    out.reshape(-1)[: flat.size] = flat
    return out


hrg_stub.pack_rows = _fake_pack_rows
hrg_stub.gather_rows_jax = lambda row_ids, *chunks, row_bytes: (
    "gather",
    row_ids,
    chunks,
    row_bytes,
)
hrg_stub.lookup_and_dequant_rows_jax = (
    lambda row_ids, valid, weight_scale_inv, *chunks, row_bytes, block_size: (
        "lookup",
        row_ids,
        valid,
        weight_scale_inv,
        chunks,
        row_bytes,
        block_size,
    )
)
sys.modules["vllm_torchtpu.kernels.deepseek_v41.host_row_gather"] = hrg_stub

# Ensure torch.ops.ao.offload and torch_tpu modules exist for _install().
if not hasattr(torch.ops, "ao") or not hasattr(torch.ops.ao, "offload"):
    ao_ns = SimpleNamespace(offload=lambda x: x)
    torch.ops.ao = ao_ns  # type: ignore[attr-defined]

offload_mod = types.ModuleType("torch._functorch._activation_offloading.offload_ops")
sys.modules["torch._functorch._activation_offloading.offload_ops"] = offload_mod

tpu_pkg = types.ModuleType("torch_tpu")
tpu_internal = types.ModuleType("torch_tpu._internal")
tpu_compile = types.ModuleType("torch_tpu._internal.compile")
tpu_backend = types.ModuleType("torch_tpu._internal.compile._backend")
tpu_compiler = types.ModuleType("torch_tpu._internal.compile.compiler")
tpu_pallas_pkg = types.ModuleType("torch_tpu._internal.pallas")
tpu_pallas = types.ModuleType("torch_tpu._internal.pallas.pallas")


def _base_placeholder(tensor, mesh=None, partition_spec=None):
    if getattr(tensor, "non_struct", False):
        return "raw_placeholder"
    return jax.ShapeDtypeStruct(tensor.shape, tensor.dtype)


tpu_pallas.jax_placeholder = _base_placeholder
tpu_backend._mark_pinned_host_inputs = lambda gm, ex: None
tpu_compiler.is_pinned_host_tensor = lambda t: bool(
    getattr(t, "_is_pinned_host", False)
)


def _fake_jax_op(name, fn):
    class _Op:
        def __init__(self):
            self.name = name
            self.fn = fn
            self.fake_fn = None

        def register_fake(self, f):
            self.fake_fn = f

        def __call__(self, *args):
            # Exercise inner `fn` when explicitly requested, otherwise return tensor via
            # `fake_fn`.
            if getattr(args[0], "_invoke_jax_fn", False):
                return self.fn(*args)
            assert self.fake_fn is not None
            return self.fake_fn(*args)

        def __str__(self):
            return self.name

    return _Op()


tpu_pallas.jax_op = _fake_jax_op
tpu_pallas_pkg.pallas = tpu_pallas
tpu_compile._backend = tpu_backend
tpu_compile.compiler = tpu_compiler
tpu_internal.compile = tpu_compile
tpu_internal.pallas = tpu_pallas_pkg
tpu_pkg._internal = tpu_internal
for mod_name, mod_obj in [
    ("torch_tpu", tpu_pkg),
    ("torch_tpu._internal", tpu_internal),
    ("torch_tpu._internal.compile", tpu_compile),
    ("torch_tpu._internal.compile._backend", tpu_backend),
    ("torch_tpu._internal.compile.compiler", tpu_compiler),
    ("torch_tpu._internal.pallas", tpu_pallas_pkg),
    ("torch_tpu._internal.pallas.pallas", tpu_pallas),
]:
    sys.modules[mod_name] = mod_obj

for mod_name in (
    "vllm",
    "vllm.logger",
    "vllm.model_executor",
    "vllm.model_executor.models",
    "vllm.model_executor.models.config",
):
    sys.modules.setdefault(mod_name, types.ModuleType(mod_name))
sys.modules["vllm"].logger = sys.modules["vllm.logger"]  # type: ignore[attr-defined]
sys.modules["vllm.logger"].init_logger = lambda name: SimpleNamespace(  # type: ignore[attr-defined]
    info=lambda *a, **kw: None
)
sys.modules["vllm.logger"]._print_warning_once = SimpleNamespace(  # type: ignore[attr-defined]
    cache_clear=lambda: None
)
sys.modules["vllm.model_executor.models.config"].VerifyAndUpdateConfig = object  # type: ignore[attr-defined]

from vllm_torchtpu.models.vllm.deepseek_v41.common import (  # noqa: E402
    host_row_table as hrt,
)


class _FakeGroup:
    def __init__(
        self,
        rank: int = 0,
        world_size: int = 2,
        peer_errors: list[str | None] | None = None,
    ):
        self.rank_in_group = rank
        self.world_size = world_size
        self.cpu_group = "fake_cpu_group"
        self.peer_errors = peer_errors
        self.preset_path: str | None = None
        self.preset_error: str | None = None

    def broadcast_object(self, obj):
        if self.rank_in_group == 0:
            return obj
        return (self.preset_path, self.preset_error)


def test_allocate_shared_host_storage_all_branches(monkeypatch):
    def fake_all_gather(out_list, local_err, group=None):
        out_list[0] = local_err
        if len(out_list) > 1 and out_list[1] is None:
            out_list[1] = None

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather)

    # 1. Leader rank 0 happy path with check_free_space callback.
    with tempfile.TemporaryDirectory() as tmpdir:
        checked = []
        grp = _FakeGroup(rank=0, world_size=2)
        tensor = hrt.allocate_shared_host_storage(
            64,
            grp,
            shm_dir=tmpdir,
            check_free_space=lambda n: checked.append(n),
        )
        assert checked == [64]
        assert tensor.dtype == torch.uint8
        assert tensor.numel() == 64
        tensor_no_check = hrt.allocate_shared_host_storage(
            32, grp, shm_dir=tmpdir, check_free_space=None
        )
        assert tensor_no_check.numel() == 32

    # 2. Leader rank 0 failure in check_free_space -> raises RuntimeError on broadcast
    # error.
    with tempfile.TemporaryDirectory() as tmpdir:
        grp_err = _FakeGroup(rank=0, world_size=2)
        with pytest.raises(RuntimeError, match="creation failed on EDP rank 0"):
            hrt.allocate_shared_host_storage(
                64,
                grp_err,
                shm_dir=tmpdir,
                check_free_space=lambda n: (_ for _ in ()).throw(OSError("disk full")),
            )

    # 3. Follower rank 1 with missing file -> raises RuntimeError after
    # all_gather_object fence.
    grp_follower = _FakeGroup(rank=1, world_size=2)
    grp_follower.preset_path = "/tmp/nonexistent_vllm_tpu_shm_file_xyz"
    grp_follower.preset_error = None
    with pytest.raises(RuntimeError, match="initialization failed"):
        hrt.allocate_shared_host_storage(64, grp_follower)

    # 4. Follower rank 1 mapping valid existing file succeeds cleanly.
    with tempfile.NamedTemporaryFile(prefix="vllm_tpu_test_", delete=False) as tf:
        tf.truncate(32)
        tf_name = tf.name
    try:
        grp_ok = _FakeGroup(rank=1, world_size=2)
        grp_ok.preset_path = tf_name
        grp_ok.preset_error = None
        mapped = hrt.allocate_shared_host_storage(32, grp_ok)
        assert mapped.numel() == 32
    finally:
        if os.path.exists(tf_name):
            os.unlink(tf_name)


def test_install_and_placeholder_marking_branches():
    hrt._install.cache_clear()
    hrt._pinned_sharding.cache_clear()
    hrt._install()

    # Exercise jax_placeholder branches:
    t_pinned = torch.zeros(4, 4)
    t_pinned._is_pinned_host = True  # type: ignore[attr-defined]
    res_pinned = tpu_pallas.jax_placeholder(t_pinned, mesh=None)
    assert res_pinned.sharding is not None
    assert res_pinned.sharding.memory_kind == "pinned_host"

    res_mesh = tpu_pallas.jax_placeholder(t_pinned, mesh="some_mesh")
    assert res_mesh.sharding is None

    t_unpinned = torch.zeros(4, 4)
    res_unpinned = tpu_pallas.jax_placeholder(t_unpinned, mesh=None)
    assert res_unpinned.sharding is None

    # Exercise mark_pinned_host_inputs across all graph node variants:
    class _Node:
        def __init__(self, op, target, args=()):
            self.op = op
            self.target = target
            self.args = args

    p0 = _Node(op="placeholder", target="p0")
    p1 = _Node(op="placeholder", target="p1")
    p2 = _Node(op="placeholder", target="p2")
    p3 = _Node(op="placeholder", target="p3")
    p4 = _Node(op="placeholder", target="p4")
    p5 = _Node(op="placeholder", target="p5")
    non_call = _Node(op="output", target="out", args=())
    unrelated_call = _Node(op="call_function", target="aten::add", args=(p0, p1))
    non_placeholder_arg = _Node(op="get_attr", target="const", args=())
    gather_call = _Node(
        op="call_function",
        target=f"{hrt.HOST_OP_NAMESPACE}::gather_c2_b16",
        args=(p0, p1, non_placeholder_arg, p5),
    )
    lookup_call = _Node(
        op="call_function",
        target=f"{hrt.HOST_OP_NAMESPACE}::lookup_and_dequant_c1_b16_s4",
        args=(p0, p1, p2, p3),
    )
    gm = SimpleNamespace(
        graph=SimpleNamespace(
            nodes=[
                p0,
                p1,
                p2,
                p3,
                p4,
                p5,
                non_call,
                unrelated_call,
                gather_call,
                lookup_call,
            ]
        )
    )
    ex0 = torch.zeros(2)
    ex1 = torch.zeros(2)
    ex2 = torch.zeros(2)
    ex3 = torch.zeros(2)
    ex4 = torch.zeros(2)
    ex5 = "not_a_tensor"
    tpu_backend._mark_pinned_host_inputs(gm, [ex0, ex1, ex2, ex3, ex4, ex5])
    assert getattr(ex1, "_is_pinned_host", False) is True
    assert getattr(ex3, "_is_pinned_host", False) is True
    assert getattr(ex0, "_is_pinned_host", False) is False


def test_host_row_table_gather_and_lookup_branches():
    hrt._gather_op.cache_clear()
    hrt._lookup_and_dequant_op.cache_clear()
    raw_full = torch.arange(0, 64, dtype=torch.uint8).reshape(4, 16)
    # Non-contiguous slice view to test contiguous conversion in __init__.
    rows = raw_full[:, :8]
    table = hrt.HostRowTable(rows, device=torch.device("cpu"), capacity_rows=8)
    assert table.num_rows == 4
    assert table.row_bytes == 8

    # Exercise gather with both int64 (triggers cast) and int32 (direct).
    ids_i64 = torch.tensor([0, 2], dtype=torch.int64)
    g1 = table.gather(ids_i64)
    assert g1.dtype == torch.uint8
    assert g1.shape == (2, 8)

    ids_i32 = torch.tensor([1, 3], dtype=torch.int32)
    g2 = table.gather(ids_i32)
    assert g2.dtype == torch.uint8

    # Invoke underlying jax wrapper functions directly via `_invoke_jax_fn = True`.
    probe = torch.tensor([0, 1], dtype=torch.int32)
    probe._invoke_jax_fn = True  # type: ignore[attr-defined]
    raw_res = table._op(probe, *table.chunks)
    assert raw_res[0] == "gather"

    # Exercise lookup_and_dequant with uncached vs cached block_size and int64/uint8 vs
    # int32/bool.
    valid_u8 = torch.tensor([1, 0], dtype=torch.uint8)
    scale = torch.ones((4, 2), dtype=torch.uint8)
    out1 = table.lookup_and_dequant(ids_i64, valid_u8, scale, block_size=4)
    assert out1.dtype == torch.bfloat16
    assert out1.shape == (2, 8)

    valid_bool = torch.tensor([True, True], dtype=torch.bool)
    out2 = table.lookup_and_dequant(ids_i32, valid_bool, scale, block_size=4)
    assert out2.dtype == torch.bfloat16

    raw_lookup = table._lookup_ops[4](probe, valid_bool, scale, *table.chunks)
    assert raw_lookup[0] == "lookup"
