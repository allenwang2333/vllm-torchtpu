# Copyright 2026 Google LLC
"""Full 100% statement and branch coverage tests for tpu_multiproc_executor."""

from __future__ import annotations

import importlib
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_SRC_DIR = str(Path(__file__).resolve().parents[3] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

for _pkg_name, _rel_parts in (
    ("vllm_torchtpu", ("vllm_torchtpu",)),
    ("vllm_torchtpu.executors", ("vllm_torchtpu", "executors")),
):
    _pkg = sys.modules.get(_pkg_name)
    if _pkg is None:
        _pkg = types.ModuleType(_pkg_name)
        sys.modules[_pkg_name] = _pkg
    _pkg.__path__ = [str(Path(_SRC_DIR, *_rel_parts))]

if not hasattr(sys.modules["vllm_torchtpu"], "_patch_multiproc_worker_global_rank_env"):
    sys.modules["vllm_torchtpu"]._patch_multiproc_worker_global_rank_env = lambda: None  # type: ignore[attr-defined]

sys.modules.pop("vllm_torchtpu.executors.tpu_multiproc_executor", None)


class _StubMultiprocExecutor:
    def _init_executor(self) -> None:
        pass

    def get_kv_cache_specs(self) -> list[dict[str, Any]]:
        return []

    def initialize_from_config(self, kv_cache_configs: list[Any]) -> None:
        pass


for _mod_name in (
    "vllm",
    "vllm.logger",
    "vllm.v1",
    "vllm.v1.executor",
    "vllm.v1.executor.multiproc_executor",
    "vllm.v1.kv_cache_interface",
):
    if _mod_name not in sys.modules:
        _m = types.ModuleType(_mod_name)
        _m.__path__ = []  # type: ignore[attr-defined]
        sys.modules[_mod_name] = _m

_vlog_mod = sys.modules["vllm.logger"]
sys.modules["vllm"].logger = _vlog_mod  # type: ignore[attr-defined]
if not hasattr(_vlog_mod, "init_logger"):
    _vlog_mod.init_logger = logging.getLogger  # type: ignore[attr-defined]
if not hasattr(_vlog_mod, "_print_warning_once"):
    _vlog_mod._print_warning_once = SimpleNamespace(cache_clear=lambda: None)  # type: ignore[attr-defined]

_mp_mod = sys.modules["vllm.v1.executor.multiproc_executor"]
if not hasattr(_mp_mod, "MultiprocExecutor"):
    _mp_mod.MultiprocExecutor = _StubMultiprocExecutor  # type: ignore[attr-defined]

_kv_mod = sys.modules["vllm.v1.kv_cache_interface"]
if not hasattr(_kv_mod, "KVCacheConfig"):
    _kv_mod.KVCacheConfig = object  # type: ignore[attr-defined]
if not hasattr(_kv_mod, "KVCacheSpec"):
    _kv_mod.KVCacheSpec = object  # type: ignore[attr-defined]

from vllm_torchtpu.executors import tpu_multiproc_executor  # noqa: E402

pytestmark = pytest.mark.cpu_test


def test_tpu_multiproc_executor_all_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = importlib.reload(tpu_multiproc_executor)

    patch_calls: list[str] = []
    monkeypatch.setattr(
        mod.vllm_torchtpu,
        "_patch_multiproc_worker_global_rank_env",
        lambda: patch_calls.append("patched"),
    )
    monkeypatch.setattr(
        mod.MultiprocExecutor,
        "_init_executor",
        lambda self: patch_calls.append("super_init"),
    )
    sentinel_specs: list[dict[str, Any]] = [{"layer.0": object()}]
    monkeypatch.setattr(
        mod.MultiprocExecutor,
        "get_kv_cache_specs",
        lambda self: sentinel_specs,
    )
    init_cfg_calls: list[Any] = []
    monkeypatch.setattr(
        mod.MultiprocExecutor,
        "initialize_from_config",
        lambda self, cfgs: init_cfg_calls.append(cfgs),
    )

    executor = mod.TpuMultiprocExecutor.__new__(mod.TpuMultiprocExecutor)

    # 1. _init_executor
    executor._init_executor()
    assert patch_calls == ["patched", "super_init"]

    # 2. get_kv_cache_specs with existing engine override (not None)
    executor.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks_override=2048),
        parallel_config=SimpleNamespace(world_size=8, pipeline_parallel_size=1),
    )
    rpc_calls: list[tuple[str, Any]] = []
    executor.collective_rpc = lambda method, args=(): rpc_calls.append((method, args))  # type: ignore[method-assign]
    assert executor.get_kv_cache_specs() is sentinel_specs
    assert rpc_calls == []
    assert executor.vllm_config.cache_config.num_gpu_blocks_override == 2048

    # 3. get_kv_cache_specs with None engine override and workers returning None
    executor.vllm_config.cache_config.num_gpu_blocks_override = None
    executor.collective_rpc = lambda method, args=(): [None] * 8  # type: ignore[method-assign]
    assert executor.get_kv_cache_specs() is sentinel_specs
    assert executor.vllm_config.cache_config.num_gpu_blocks_override is None

    # 4. get_kv_cache_specs with None engine override and workers agreeing on 4096
    executor.vllm_config.parallel_config = SimpleNamespace(
        world_size=8, pipeline_parallel_size=2
    )
    executor.collective_rpc = lambda method, args=(): [4096] * 4 + [4098] * 4  # type: ignore[method-assign]
    assert executor.get_kv_cache_specs() is sentinel_specs
    assert executor.vllm_config.cache_config.num_gpu_blocks_override == 4096

    # 5. initialize_from_config with empty shapes ([])
    executor.collective_rpc = lambda method, args=(): []  # type: ignore[method-assign]
    executor.initialize_from_config([])
    assert len(init_cfg_calls) == 1

    # 6. initialize_from_config with empty shapes[0] ([[]])
    executor.collective_rpc = lambda method, args=(): [[]]  # type: ignore[method-assign]
    executor.initialize_from_config([])
    assert len(init_cfg_calls) == 2

    # 7. initialize_from_config with non-empty shapes[0] ([[16, 32, 64]])
    prewarm_calls: list[tuple[str, Any]] = []

    def _prewarm_rpc(method: str, args: Any = ()) -> Any:
        prewarm_calls.append((method, args))
        if method == "get_kv_prewarm_shapes":
            return [[16, 32, 64]]
        return None

    executor.collective_rpc = _prewarm_rpc  # type: ignore[method-assign]
    executor.initialize_from_config([])
    assert prewarm_calls == [
        ("get_kv_prewarm_shapes", ()),
        ("prewarm_kv_offload_shape", (16,)),
        ("prewarm_kv_offload_shape", (32,)),
        ("prewarm_kv_offload_shape", (64,)),
    ]
