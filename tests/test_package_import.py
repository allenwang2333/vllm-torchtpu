import runpy
import subprocess
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest
import torch

import vllm_torchtpu


def test_package_imports_before_vllm():
    result = subprocess.run(
        [sys.executable, "-c", "import vllm_torchtpu; import vllm"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_submodule_imports_before_vllm_initialization_completes():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import vllm_torchtpu.layers.adapter.quantization.fp8; import vllm",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_patch_hybrid_kv_load_failure_recovery_and_mamba_split(monkeypatch):
    from vllm.v1.core.sched.scheduler import Scheduler

    monkeypatch.setattr(
        Scheduler,
        "_update_requests_with_invalid_blocks",
        getattr(Scheduler, "_update_requests_with_invalid_blocks", None),
        raising=False,
    )
    monkeypatch.setattr(
        Scheduler,
        "_tpu_hybrid_kv_load_failure_patch",
        getattr(Scheduler, "_tpu_hybrid_kv_load_failure_patch", False),
        raising=False,
    )

    # 1. original_update is None -> early return
    monkeypatch.delattr(
        Scheduler, "_update_requests_with_invalid_blocks", raising=False
    )
    monkeypatch.setattr(
        Scheduler, "_tpu_hybrid_kv_load_failure_patch", False, raising=False
    )
    vllm_torchtpu._patch_vllm_hybrid_kv_load_failure_recovery()

    # 2. Install patch and test idempotency
    orig_calls = []

    def _orig_update(
        self, requests, invalid_block_ids, num_scheduled_tokens, evict_blocks=True
    ):
        orig_calls.append(
            (requests, invalid_block_ids, num_scheduled_tokens, evict_blocks)
        )
        return ({"orig"}, 1, {99})

    monkeypatch.setattr(
        Scheduler, "_update_requests_with_invalid_blocks", _orig_update, raising=False
    )
    monkeypatch.setattr(
        Scheduler, "_tpu_hybrid_kv_load_failure_patch", False, raising=False
    )
    vllm_torchtpu._patch_vllm_hybrid_kv_load_failure_recovery()
    vllm_torchtpu._patch_vllm_hybrid_kv_load_failure_recovery()

    fn = Scheduler._update_requests_with_invalid_blocks
    # Single group -> delegates to original_update
    sched_single = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=["g0"])
    )
    assert fn(sched_single, [], {1}, {}, evict_blocks=False) == ({"orig"}, 1, {99})
    assert len(orig_calls) == 1

    # Hybrid (>1 group): empty, unaffected, and affected requests
    block_map = {
        "r_ok": [[0, 1], [2]],
        "r_bad": [[0, 5], []],
    }
    sched_hybrid = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=["g0", "g1"]),
        kv_cache_manager=SimpleNamespace(
            block_pool=SimpleNamespace(null_block=SimpleNamespace(block_id=0)),
            get_block_ids=lambda req_id: block_map[req_id],
        ),
    )
    assert fn(sched_hybrid, [], {5}, {}) == (set(), 0, set())

    r_ok = SimpleNamespace(request_id="r_ok", num_computed_tokens=10)
    r_bad = SimpleNamespace(request_id="r_bad", num_computed_tokens=12)
    affected, total_toks, to_evict = fn(
        sched_hybrid, [r_ok, r_bad], {5}, {"r_bad": 2}, evict_blocks=True
    )
    assert affected == {"r_bad"}
    assert total_toks == 10
    assert to_evict == {5}
    assert r_bad.num_computed_tokens == 0

    r_bad2 = SimpleNamespace(request_id="r_bad", num_computed_tokens=8)
    affected2, total_toks2, to_evict2 = fn(
        sched_hybrid, [r_bad2], {5}, {}, evict_blocks=False
    )
    assert affected2 == {"r_bad"}
    assert total_toks2 == 8
    assert to_evict2 == set()

    # _patch_vllm_mamba_split_scheduler_block_size idempotency & both branches
    monkeypatch.setattr(
        Scheduler,
        "_mamba_block_aligned_split",
        lambda self, *a, **k: self.cache_config.block_size,
    )
    vllm_torchtpu._patch_vllm_mamba_split_scheduler_block_size()
    vllm_torchtpu._patch_vllm_mamba_split_scheduler_block_size()
    s_eq = SimpleNamespace(cache_config=SimpleNamespace(block_size=16), block_size=16)
    assert Scheduler._mamba_block_aligned_split(s_eq) == 16
    s_ne = SimpleNamespace(cache_config=SimpleNamespace(block_size=16), block_size=32)
    assert Scheduler._mamba_block_aligned_split(s_ne) == 32
    assert s_ne.cache_config.block_size == 16


def test_patch_aot_compile_hash_and_config_hash_and_moe_pcp(monkeypatch):
    from vllm.compilation import caching
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.fused_moe import FusedMoEParallelConfig

    # 1. _patch_vllm_aot_compile_cache_key
    monkeypatch.setattr(caching, "aot_compile_hash_factors", lambda cfg: ["base"])
    monkeypatch.setattr(
        "vllm_torchtpu.compilation.tpu_compiler.compute_tpu_compilation_hash",
        lambda cfg: "tpu_hash",
    )
    monkeypatch.setattr(
        "vllm_torchtpu.compilation.shape_variants.aot_cache_tag",
        lambda: "tag1",
    )
    vllm_torchtpu._patch_vllm_aot_compile_cache_key()
    vllm_torchtpu._patch_vllm_aot_compile_cache_key()
    assert caching.aot_compile_hash_factors("cfg") == ["base", "tpu_hash", "tag1"]

    # 2. _patch_vllm_config_hash_ignore_diagnostics
    monkeypatch.setattr(
        VllmConfig, "_tpu_additional_config_hash_patch", False, raising=False
    )
    monkeypatch.setattr(
        VllmConfig,
        "compute_hash",
        lambda self: f"hash:{self.additional_config}",
    )
    vllm_torchtpu._patch_vllm_config_hash_ignore_diagnostics()
    vllm_torchtpu._patch_vllm_config_hash_ignore_diagnostics()

    class _DummyCfg:
        def __init__(self, additional_config):
            self.additional_config = additional_config

    assert VllmConfig.compute_hash(_DummyCfg(None)) == "hash:None"
    assert VllmConfig.compute_hash(_DummyCfg({"keep": 1})) == "hash:{'keep': 1}"
    from vllm_torchtpu.runner.utils import HASH_IGNORED_ADDITIONAL_CONFIG_KEYS

    ignored_key = next(iter(HASH_IGNORED_ADDITIONAL_CONFIG_KEYS))
    assert (
        VllmConfig.compute_hash(_DummyCfg({"keep": 1, ignored_key: "ignore"}))
        == "hash:{'keep': 1}"
    )

    # 3. _patch_moe_explicit_pcp_collectives
    monkeypatch.setattr(
        FusedMoEParallelConfig,
        "_tpu_explicit_pcp_collectives_patch",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        FusedMoEParallelConfig,
        "use_all2all_kernels",
        property(lambda self: True),
    )
    vllm_torchtpu._patch_moe_explicit_pcp_collectives()
    vllm_torchtpu._patch_moe_explicit_pcp_collectives()
    monkeypatch.setattr(vllm_torchtpu.envs, "TPU_MOE_COLLECTION_CHUNK_SIZE", 0)
    pcp_ep_obj = SimpleNamespace(
        use_ep=True, pcp_size=2, dp_size=1, is_sequence_parallel=False
    )
    assert FusedMoEParallelConfig.use_all2all_kernels.fget(pcp_ep_obj) is False
    non_pcp_obj = SimpleNamespace(
        use_ep=True, pcp_size=1, dp_size=1, is_sequence_parallel=False
    )
    assert FusedMoEParallelConfig.use_all2all_kernels.fget(non_pcp_obj) is True


def test_patch_expert_map_host_lookup_and_compile_ranges(monkeypatch):
    from vllm.compilation import piecewise_backend as pb_mod
    from vllm.config.compilation import CompilationConfig
    from vllm.config.parallel import ParallelConfig
    from vllm.model_executor.layers.fused_moe import expert_map_manager as _em
    from vllm.v1.executor.multiproc_executor import WorkerProc

    cls = _em.ExpertMapManager
    for attr in (
        "_calculate_expert_maps",
        "map_global_to_local",
        "get_compressed_map_string",
        "_tpu_expert_map_source",
        "_tpu_expert_map_version",
        "_tpu_expert_map_host",
        "_tpu_host_expert_map_patch",
    ):
        if hasattr(cls, attr):
            monkeypatch.setattr(cls, attr, getattr(cls, attr), raising=False)
    monkeypatch.setattr(cls, "_tpu_host_expert_map_patch", False, raising=False)

    vllm_torchtpu._patch_expert_map_host_lookup()
    vllm_torchtpu._patch_expert_map_host_lookup()

    # Non-None expert_map and expert_mask
    monkeypatch.setattr(
        _em,
        "determine_expert_map",
        lambda **kw: (
            2,
            torch.tensor([0, -1, 1], dtype=torch.int32),
            torch.tensor([1, 0, 1]),
        ),
    )
    mgr = SimpleNamespace(
        ep_size=2,
        ep_rank=0,
        global_num_experts=3,
        _placement_strategy="linear",
        num_fused_shared_experts=0,
        rocm_aiter_enabled=True,
        _tpu_expert_map_source=None,
        _tpu_expert_map_version=None,
        _tpu_expert_map_host=None,
    )
    cls._calculate_expert_maps(mgr)
    assert cls.map_global_to_local(mgr, 0) == 0
    assert cls.map_global_to_local(mgr, 1) == -1
    assert cls.get_compressed_map_string(mgr) == "0->0, 1->2"

    # Mutate _expert_map in-place to bump _version and trigger host cache refresh
    mgr._expert_map.add_(0)
    assert cls.map_global_to_local(mgr, 2) == 1

    # None expert_map branch
    monkeypatch.setattr(_em, "determine_expert_map", lambda **kw: (3, None, None))
    cls._calculate_expert_maps(mgr)
    assert cls.map_global_to_local(mgr, 2) == 2
    assert cls.get_compressed_map_string(mgr) == "[0..2]"

    # _patch_vllm_disable_compile_ranges
    monkeypatch.setattr(
        CompilationConfig, "get_compile_ranges", CompilationConfig.get_compile_ranges
    )
    vllm_torchtpu._patch_vllm_disable_compile_ranges()
    assert CompilationConfig.get_compile_ranges(SimpleNamespace()) == []

    # _patch_vllm_compile_all_ranges
    monkeypatch.setattr(
        pb_mod.PiecewiseBackend,
        "compile_all_ranges",
        pb_mod.PiecewiseBackend.compile_all_ranges,
    )
    monkeypatch.setattr(pb_mod, "create_concrete_args", lambda g, s: ["concrete", s])
    monkeypatch.setattr(pb_mod, "get_fake_args_from_graph", lambda g: ["fake"])
    monkeypatch.setattr(vllm_torchtpu.envs, "TPU_PARALLEL_PRECOMPILE", False)
    vllm_torchtpu._patch_vllm_compile_all_ranges()

    monkeypatch.setattr(vllm_torchtpu.envs, "TPU_PARALLEL_PRECOMPILE", True)
    vllm_torchtpu._patch_vllm_compile_all_ranges()
    monkeypatch.setenv("LOCAL_RANK", "0")

    # Empty range_entries
    pb_empty = SimpleNamespace(graph="g", range_entries={})
    pb_mod.PiecewiseBackend.compile_all_ranges(pb_empty)

    # 3 range entries (compiled=True, single_size=True, single_size=False)
    e0 = SimpleNamespace(
        compiled=True,
        compile_range=SimpleNamespace(is_single_size=lambda: True, start=8),
        runnable=None,
    )
    e1 = SimpleNamespace(
        compiled=False,
        compile_range=SimpleNamespace(is_single_size=lambda: True, start=16),
        runnable=None,
    )
    e2 = SimpleNamespace(
        compiled=False,
        compile_range=SimpleNamespace(is_single_size=lambda: False, start=32),
        runnable=None,
    )
    pb_inst = SimpleNamespace(
        graph="g",
        range_entries={0: e0, 1: e1, 2: e2},
        _log_compile_start=lambda r: None,
        piecewise_compile_index=0,
        total_piecewise_compiles=1,
        compilation_config="ccfg",
        vllm_backend=SimpleNamespace(
            inductor_config="icfg",
            is_encoder=False,
            compiler_manager=SimpleNamespace(
                compile=lambda *a, **kw: ("compiled", a[1])
            ),
        ),
    )
    pb_mod.PiecewiseBackend.compile_all_ranges(pb_inst)
    assert e1.compiled is True and e1.runnable == ("compiled", ["concrete", 16])
    assert e2.compiled is True and e2.runnable == ("compiled", ["fake"])

    # _patch_disable_sequence_parallel_moe
    monkeypatch.setattr(ParallelConfig, "_tpu_no_sp_moe_patch", False, raising=False)
    monkeypatch.setattr(
        ParallelConfig,
        "use_sequence_parallel_moe",
        ParallelConfig.use_sequence_parallel_moe,
    )
    vllm_torchtpu._patch_disable_sequence_parallel_moe()
    vllm_torchtpu._patch_disable_sequence_parallel_moe()
    assert ParallelConfig.use_sequence_parallel_moe.fget(SimpleNamespace()) is False

    # _patch_multiproc_worker_global_rank_env
    monkeypatch.setattr(WorkerProc, "_tpu_global_rank_env_patch", False, raising=False)
    monkeypatch.setattr(
        WorkerProc, "make_worker_process", staticmethod(lambda *a, **k: "ok")
    )
    vllm_torchtpu._patch_multiproc_worker_global_rank_env()
    vllm_torchtpu._patch_multiproc_worker_global_rank_env()
    for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        monkeypatch.setenv(k, "")
        monkeypatch.delenv(k, raising=False)
    fake_vllm_cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(
            world_size=2,
            local_world_size=2,
            prefill_context_parallel_size=1,
            data_parallel_size=1,
            data_parallel_index=0,
            enable_expert_parallel=False,
        )
    )
    assert WorkerProc.make_worker_process(fake_vllm_cfg, 1, 1) == "ok"


def test_patch_offloading_kimi_kda_and_hybrid_pcp(monkeypatch):
    from vllm.config.model import ModelConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading import (
        config as offloading_config_module,
    )
    from vllm.v1.core import kv_cache_coordinator, kv_cache_utils
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheSpec,
        MambaSpec,
        UniformTypeKVCacheSpecs,
    )

    # 1. _patch_vllm_offloading_config_build
    @dataclass
    class _Group:
        tokens_per_block: int

    @dataclass
    class _Cache:
        tokens_per_hash: int

    @dataclass
    class _OffloadCfg:
        groups: tuple[_Group, ...]
        cache: _Cache

    monkeypatch.setattr(
        offloading_config_module, "_tpu_offloading_config_patch", False, raising=False
    )
    monkeypatch.setattr(
        offloading_config_module,
        "build_offloading_config",
        lambda vcfg, kcfg: _OffloadCfg(groups=(_Group(16),), cache=_Cache(16)),
    )
    monkeypatch.setattr(
        kv_cache_utils, "resolve_kv_cache_block_sizes", lambda kcfg, vcfg: (32, 32)
    )
    fake_conn_mod = ModuleType("offloading_connector")
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector",
        fake_conn_mod,
    )
    vllm_torchtpu._patch_vllm_offloading_config_build()
    vllm_torchtpu._patch_vllm_offloading_config_build()
    assert hasattr(fake_conn_mod, "build_offloading_config")

    vcfg_pcp1 = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1)
    )
    assert (
        offloading_config_module.build_offloading_config(vcfg_pcp1, "kcfg")
        .groups[0]
        .tokens_per_block
        == 16
    )
    vcfg_pcp2 = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=2)
    )
    res_pcp2 = offloading_config_module.build_offloading_config(vcfg_pcp2, "kcfg")
    assert res_pcp2.groups[0].tokens_per_block == 32
    assert res_pcp2.cache.tokens_per_hash == 32

    # Also test when offloading_connector is not in sys.modules
    monkeypatch.delitem(
        sys.modules,
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector",
        raising=False,
    )
    monkeypatch.setattr(
        offloading_config_module, "_tpu_offloading_config_patch", False, raising=False
    )
    vllm_torchtpu._patch_vllm_offloading_config_build()

    # 2. _patch_vllm_kimi_kda_layer_counts
    monkeypatch.setattr(
        ModelConfig, "_tpu_kimi_kda_layer_counts_patched", False, raising=False
    )

    def _orig_layers(self, pc, block_type="attention"):
        if getattr(self, "_raise_val_err", False):
            raise ValueError("unknown layer type")
        return 12

    monkeypatch.setattr(ModelConfig, "get_num_layers_by_block_type", _orig_layers)
    vllm_torchtpu._patch_vllm_kimi_kda_layer_counts()
    vllm_torchtpu._patch_vllm_kimi_kda_layer_counts()

    m_ok = SimpleNamespace(_raise_val_err=False)
    assert ModelConfig.get_num_layers_by_block_type(m_ok, "pc") == 12

    m_kda = SimpleNamespace(
        _raise_val_err=True,
        get_layers_start_end_indices=lambda pc: (0, 4),
        hf_text_config=SimpleNamespace(linear_attn_config={"kda_layers": [1, 3]}),
    )
    assert ModelConfig.get_num_layers_by_block_type(m_kda, "pc", "attention") == 2

    m_fail = SimpleNamespace(
        _raise_val_err=True,
        get_layers_start_end_indices=lambda pc: (0, 4),
        hf_text_config=SimpleNamespace(linear_attn_config=None),
    )
    with pytest.raises(ValueError, match="unknown layer type"):
        ModelConfig.get_num_layers_by_block_type(m_fail, "pc", "attention")

    # 3. _patch_vllm_hybrid_pcp_block_sizes remaining branches
    from vllm.v1.core import kv_cache_manager
    from vllm.v1.engine import core as engine_core_mod
    from vllm.v1.engine.core import EngineCoreProc

    monkeypatch.setattr(
        kv_cache_utils,
        "_tpu_original_resolve_kv_cache_block_sizes",
        lambda kcfg, vcfg: (16, 16),
        raising=False,
    )
    monkeypatch.setattr(
        kv_cache_utils, "_tpu_hybrid_pcp_block_sizes_patch", False, raising=False
    )
    monkeypatch.setattr(
        kv_cache_utils,
        "resolve_kv_cache_block_sizes",
        lambda kcfg, vcfg: (16, 16),
    )
    monkeypatch.setattr(
        kv_cache_utils,
        "_promote_local_kv_cache_specs",
        lambda specs: (_ for _ in ()).throw(ValueError("heterogeneous"))
        if specs == "bad"
        else {"ok": 1},
        raising=False,
    )
    monkeypatch.setattr(
        kv_cache_coordinator,
        "get_kv_cache_coordinator",
        lambda *a, **kw: ("coord", a, kw),
    )
    monkeypatch.setattr(
        kv_cache_manager,
        "get_kv_cache_coordinator",
        kv_cache_manager.get_kv_cache_coordinator,
    )
    monkeypatch.setattr(
        engine_core_mod,
        "resolve_kv_cache_block_sizes",
        engine_core_mod.resolve_kv_cache_block_sizes,
        raising=False,
    )
    monkeypatch.setattr(
        engine_core_mod,
        "_promote_local_kv_cache_specs",
        getattr(engine_core_mod, "_promote_local_kv_cache_specs", None),
        raising=False,
    )
    monkeypatch.setattr(
        EngineCoreProc,
        "_tpu_engine_core_patch_wrapper",
        getattr(EngineCoreProc, "_tpu_engine_core_patch_wrapper", False),
        raising=False,
    )
    monkeypatch.setattr(
        EngineCoreProc,
        "_tpu_original_run_engine_core",
        getattr(
            EngineCoreProc,
            "_tpu_original_run_engine_core",
            EngineCoreProc.run_engine_core,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        EngineCoreProc, "run_engine_core", EngineCoreProc.run_engine_core
    )
    # Remove vllm.v1.kv_offload.base temporarily to hit 895->893 False branch
    saved_offload_base = sys.modules.pop("vllm.v1.kv_offload.base", None)
    try:
        vllm_torchtpu._patch_vllm_hybrid_pcp_block_sizes()
    finally:
        if saved_offload_base is not None:
            sys.modules["vllm.v1.kv_offload.base"] = saved_offload_base

    assert kv_cache_utils._promote_local_kv_cache_specs("good") == {"ok": 1}
    assert kv_cache_utils._promote_local_kv_cache_specs("bad") == {}

    @dataclass
    class _GroupSpec:
        kv_cache_spec: KVCacheSpec

    @dataclass
    class _KVCfg:
        kv_cache_groups: list[_GroupSpec]

    attn_leaf = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16
    )
    uniform_attn = UniformTypeKVCacheSpecs(
        block_size=16, kv_cache_specs={"l0": attn_leaf}
    )
    mamba_leaf = MambaSpec(shapes=((4, 8),), dtypes=(torch.float32,), block_size=16)
    custom_leaf = KVCacheSpec(block_size=16)
    hybrid_groups = [
        _GroupSpec(kv_cache_spec=uniform_attn),
        _GroupSpec(kv_cache_spec=mamba_leaf),
        _GroupSpec(kv_cache_spec=custom_leaf),
    ]
    kcfg_hybrid = _KVCfg(kv_cache_groups=hybrid_groups)
    vcfg_hybrid = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=True, prefix_match_unit=16),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, prefill_context_parallel_size=2
        ),
        kv_transfer_config=None,
    )
    assert kv_cache_utils.resolve_kv_cache_block_sizes(kcfg_hybrid, vcfg_hybrid) == (
        32,
        16,
    )

    vcfg_hybrid_bad_unit = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=True, prefix_match_unit=15),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, prefill_context_parallel_size=2
        ),
        kv_transfer_config=None,
    )
    with pytest.raises(ValueError, match="Invalid prefix_match_unit"):
        kv_cache_utils.resolve_kv_cache_block_sizes(kcfg_hybrid, vcfg_hybrid_bad_unit)

    # Coordinator: non-divisible scheduler_block_size -> ValueError (line 815)
    with pytest.raises(ValueError, match="Logical scheduler block size"):
        kv_cache_coordinator.get_kv_cache_coordinator(
            kcfg_hybrid, 128, 128, False, True, False, 1, 1, 25, 16
        )

    # Coordinator: logical_pcp_world_size > 1 and dcp_world_size != 1 -> ValueError
    with pytest.raises(ValueError, match="does not support DCP"):
        kv_cache_coordinator.get_kv_cache_coordinator(
            kcfg_hybrid, 128, 128, False, True, False, 2, 2, 32, 16
        )

    # Coordinator: logical_pcp_world_size > 1 with UniformType + custom KVCacheSpec
    coord_res = kv_cache_coordinator.get_kv_cache_coordinator(
        kcfg_hybrid, 128, 128, False, True, False, 1, 2, 32, 16
    )
    assert coord_res[0] == "coord"


def test_patch_dcp_fine_mm_compile_dflash_triton_v1_lifo_vpe_and_proxy(monkeypatch):
    import vllm.compilation.wrapper as wrapper_mod
    import vllm.model_executor.layers.vocab_parallel_embedding as vpe
    import vllm.model_executor.models.utils as vllm_utils
    from vllm.compilation.piecewise_backend import PiecewiseBackend
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
    from vllm.config.vllm import VllmConfig
    from vllm.v1.core import kv_cache_coordinator, kv_cache_utils
    from vllm.v1.core.block_pool import BlockPool

    # 1. _patch_vllm_dcp_fine_prefix_hits error branches (lines 1097, 1102)
    cls_coord = kv_cache_coordinator.UnitaryKVCacheCoordinator
    monkeypatch.setattr(cls_coord, "_tpu_dcp_fine_prefix_patch", False, raising=False)
    monkeypatch.setattr(
        cls_coord, "find_longest_cache_hit", cls_coord.find_longest_cache_hit
    )
    monkeypatch.setattr(
        kv_cache_utils,
        "resolve_kv_cache_block_sizes",
        kv_cache_utils.resolve_kv_cache_block_sizes,
    )
    for mod_name in ("vllm.v1.engine.core", "vllm.v1.kv_offload.base"):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "resolve_kv_cache_block_sizes"):
            monkeypatch.setattr(
                mod,
                "resolve_kv_cache_block_sizes",
                mod.resolve_kv_cache_block_sizes,
            )

    def _fake_unitary_init(
        self,
        kv_cache_config,
        max_model_len,
        max_in_flight_tokens,
        use_eagle,
        enable_caching,
        enable_kv_cache_events,
        dcp_world_size,
        pcp_world_size,
        scheduler_block_size,
        hash_block_size,
        metrics_collector=None,
        num_prefill_lookahead=0,
    ):
        pass

    def _fake_base_init(self, **kw):
        self.single_type_managers = [self._mgr]
        self.eagle_group_ids = []

    monkeypatch.setattr(cls_coord, "__init__", _fake_unitary_init)
    monkeypatch.setattr(
        kv_cache_coordinator.KVCacheCoordinator, "__init__", _fake_base_init
    )
    vllm_torchtpu._patch_vllm_dcp_fine_prefix_hits()

    class _UnsupportedMgr:
        supports_fine_grained_hash_lookup = False
        dcp_world_size = 2
        block_size = 32

    class _SupportedMgr:
        supports_fine_grained_hash_lookup = True
        dcp_world_size = 2
        block_size = 32

    kcfg_1g = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec="s")])

    with pytest.raises(ValueError, match="does not support prefix hits finer"):
        cls_coord.__init__(
            SimpleNamespace(_mgr=_UnsupportedMgr()),
            kcfg_1g,
            128,
            128,
            False,
            True,
            False,
            2,
            1,
            32,
            16,
        )

    with pytest.raises(ValueError, match="must divide the DCP manager block size"):
        cls_coord.__init__(
            SimpleNamespace(_mgr=_SupportedMgr()),
            kcfg_1g,
            128,
            128,
            False,
            True,
            False,
            2,
            1,
            32,
            15,
        )

    # 2. _patch_vllm_merge_multimodal_embeddings
    monkeypatch.setattr(
        vllm_utils,
        "_merge_multimodal_embeddings",
        vllm_utils._merge_multimodal_embeddings,
    )
    monkeypatch.setattr(
        vllm_utils,
        "_original_merge_multimodal_embeddings",
        getattr(
            vllm_utils,
            "_original_merge_multimodal_embeddings",
            vllm_utils._merge_multimodal_embeddings,
        ),
        raising=False,
    )
    for mod_name, mod in list(sys.modules.items()):
        if mod_name.startswith("vllm.model_executor.models") and hasattr(
            mod, "_merge_multimodal_embeddings"
        ):
            monkeypatch.setattr(
                mod, "_merge_multimodal_embeddings", mod._merge_multimodal_embeddings
            )
    monkeypatch.setattr(vllm_utils, "_tpu_static_merge_mm_patch", False, raising=False)
    vllm_torchtpu._patch_vllm_merge_multimodal_embeddings()
    merge_fn = vllm_utils._merge_multimodal_embeddings
    inputs_embeds = torch.zeros((3, 2))
    is_mm = torch.tensor([False, True, False])
    assert merge_fn(inputs_embeds, [], is_mm) is inputs_embeds
    monkeypatch.setattr(
        vllm_utils, "_flatten_embeddings", lambda mm: torch.empty((0, 2))
    )
    assert merge_fn(inputs_embeds, [torch.empty((0, 2))], is_mm) is inputs_embeds
    monkeypatch.setattr(
        vllm_utils, "_flatten_embeddings", lambda mm: torch.ones((1, 2))
    )
    merged = merge_fn(inputs_embeds, [torch.ones((1, 2))], is_mm)
    assert merged[1].tolist() == [1.0, 1.0]

    # 3. Compile wrapper, piecewise backend, and compile prefix isolation patches
    reset_calls = []
    monkeypatch.setattr(
        wrapper_mod, "reset_compile_wrapper", lambda m: reset_calls.append(m)
    )
    vllm_torchtpu._patch_vllm_reset_compile_wrapper()
    vllm_torchtpu._patch_vllm_reset_compile_wrapper()
    wrapper_mod.reset_compile_wrapper(None)

    class _FakeWrapper(TorchCompileWithNoGuardsWrapper, torch.nn.Module):
        def __init__(self):
            torch.nn.Module.__init__(self)

    w_direct = _FakeWrapper()
    wrapper_mod.reset_compile_wrapper(w_direct)
    parent_with_child = torch.nn.Sequential(_FakeWrapper())
    wrapper_mod.reset_compile_wrapper(parent_with_child)
    plain_mod = torch.nn.Linear(2, 2)
    wrapper_mod.reset_compile_wrapper(plain_mod)
    assert len(reset_calls) == 3

    monkeypatch.setattr(PiecewiseBackend, "__call__", lambda self, *a, **k: "orig")
    vllm_torchtpu._patch_vllm_piecewise_backend()
    vllm_torchtpu._patch_vllm_piecewise_backend()
    re_compiled = SimpleNamespace(
        compiled=True, compile_range="r", runnable=lambda *a, **k: "ran"
    )
    pb_sym = SimpleNamespace(
        sym_shape_indices=[0],
        compile_ranges="cr",
        _find_range_for_shape=lambda s: re_compiled,
    )
    assert PiecewiseBackend.__call__(pb_sym, 8) == "ran"
    pb_tensor = SimpleNamespace(
        sym_shape_indices=[],
        range_entries={0: re_compiled, 1: re_compiled},
        compile_sizes=[8, 16],
        compile_ranges="cr",
        _find_range_for_shape=lambda s: re_compiled,
    )
    assert PiecewiseBackend.__call__(pb_tensor, torch.zeros((8, 2))) == "ran"
    pb_no_tensor = SimpleNamespace(
        sym_shape_indices=[],
        range_entries={0: re_compiled, 1: SimpleNamespace(compiled=False)},
        compile_sizes=[8, 16],
    )
    assert PiecewiseBackend.__call__(pb_no_tensor, "not_a_tensor") == "ran"
    pb_single = SimpleNamespace(
        sym_shape_indices=[],
        range_entries={0: re_compiled},
        compile_sizes=[],
    )
    assert PiecewiseBackend.__call__(pb_single) == "ran"

    monkeypatch.setattr(
        TorchCompileWithNoGuardsWrapper,
        "__init__",
        lambda self, compile_prefix="", is_encoder=False: setattr(
            self, "saved_prefix", compile_prefix
        ),
    )
    vllm_torchtpu._patch_vllm_compile_prefix_isolation()
    vllm_torchtpu._patch_vllm_compile_prefix_isolation()
    inst_pref = SimpleNamespace(prefix="layer.0")
    TorchCompileWithNoGuardsWrapper.__init__(inst_pref)
    assert inst_pref.saved_prefix == "layer.0"
    TorchCompileWithNoGuardsWrapper.__init__(inst_pref, compile_prefix="explicit")
    assert inst_pref.saved_prefix == "explicit"
    inst_nopref = SimpleNamespace()
    TorchCompileWithNoGuardsWrapper.__init__(inst_nopref)
    assert inst_nopref.saved_prefix.startswith("_tpu_instance_")

    # 4. _patch_dflash_bypass_v2_runner_check
    import vllm.model_executor.models as vllm_models_pkg

    monkeypatch.delattr(vllm_models_pkg, "qwen3_dflash", raising=False)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.qwen3_dflash", None)
    vllm_torchtpu._patch_dflash_bypass_v2_runner_check()

    fake_dflash = ModuleType("vllm.model_executor.models.qwen3_dflash")
    fake_dflash._resolve_layer_attention = lambda cfg, idx: "resolved"  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.models.qwen3_dflash", fake_dflash
    )
    monkeypatch.setattr(vllm_models_pkg, "qwen3_dflash", fake_dflash, raising=False)
    vllm_torchtpu._patch_dflash_bypass_v2_runner_check()
    vllm_torchtpu._patch_dflash_bypass_v2_runner_check()

    class _BaseCfg:
        use_v2_model_runner = property(lambda self: False)

    class _SubCfg(_BaseCfg):
        pass

    class _NoFlagCfg:
        pass

    monkeypatch.setattr("vllm.config.get_current_vllm_config", lambda: _SubCfg())
    assert fake_dflash._resolve_layer_attention("cfg", 0) == "resolved"
    assert "use_v2_model_runner" not in _SubCfg.__dict__

    monkeypatch.setattr("vllm.config.get_current_vllm_config", lambda: _NoFlagCfg())
    with pytest.raises(RuntimeError, match="no longer defines"):
        fake_dflash._resolve_layer_attention("cfg", 0)

    # 5. _patch_vllm_config_triton_tpu & _patch_vllm_force_v1_runner_tpu
    monkeypatch.setattr(VllmConfig, "_tpu_triton_patch_applied", False, raising=False)
    monkeypatch.setattr(
        VllmConfig, "_validate_v2_model_runner", lambda self: "orig_validated"
    )
    vllm_torchtpu._patch_vllm_config_triton_tpu()
    vllm_torchtpu._patch_vllm_config_triton_tpu()
    assert (
        VllmConfig._validate_v2_model_runner(
            SimpleNamespace(device_config=SimpleNamespace(device_type="tpu"))
        )
        is None
    )
    assert (
        VllmConfig._validate_v2_model_runner(
            SimpleNamespace(device_config=SimpleNamespace(device_type="cpu"))
        )
        == "orig_validated"
    )

    monkeypatch.setattr(VllmConfig, "_tpu_force_v1_runner_patch", False, raising=False)
    monkeypatch.setattr(VllmConfig, "use_v2_model_runner", "not_a_property")
    vllm_torchtpu._patch_vllm_force_v1_runner_tpu()

    # 6. _patch_vllm_block_pool_lifo_free
    monkeypatch.setattr(BlockPool, "_tpu_lifo_free_patch", False, raising=False)
    monkeypatch.setattr(BlockPool, "free_blocks", BlockPool.free_blocks, raising=False)
    vllm_torchtpu._patch_vllm_block_pool_lifo_free()
    vllm_torchtpu._patch_vllm_block_pool_lifo_free()
    prepended, appended = [], []
    pool = SimpleNamespace(
        free_block_queue=SimpleNamespace(
            prepend_n=lambda b: prepended.extend(b),
            append_n=lambda b: appended.extend(b),
        )
    )
    b_no_hash = SimpleNamespace(ref_cnt=1, is_null=False, block_hash=None)
    b_with_hash = SimpleNamespace(ref_cnt=1, is_null=False, block_hash="h")
    BlockPool.free_blocks(pool, [b_no_hash, b_with_hash])
    assert prepended == [b_no_hash]
    assert appended == [b_with_hash]

    # 7. _patch_vllm_vocab_parallel_embedding
    monkeypatch.setattr(
        vpe.VocabParallelEmbedding,
        "_tpu_vocab_parallel_embedding_patch",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        vpe.VocabParallelEmbedding,
        "__init__",
        lambda self, *a, **k: torch.nn.Module.__init__(self),
    )
    monkeypatch.setattr(
        vpe.VocabParallelEmbedding,
        "forward",
        vpe.VocabParallelEmbedding.forward,
    )
    monkeypatch.setattr(
        vpe,
        "get_masked_input_and_mask",
        vpe.get_masked_input_and_mask,
    )
    vllm_torchtpu._patch_vllm_vocab_parallel_embedding()
    vllm_torchtpu._patch_vllm_vocab_parallel_embedding()

    class _DummyVPE(vpe.VocabParallelEmbedding):
        def __init__(self, preset_buf=False):
            self.shard_indices = SimpleNamespace(
                org_vocab_start_index=0,
                org_vocab_end_index=4,
                num_org_vocab_padding=0,
                added_vocab_start_index=4,
                added_vocab_end_index=6,
            )
            if preset_buf:
                self.org_vocab_start_index = torch.tensor(0)
            super().__init__()
            self.quant_method = SimpleNamespace(
                embedding=lambda mod, x: x.to(torch.float32).unsqueeze(-1)
            )

    vpe_inst = _DummyVPE(preset_buf=False)
    for buf_name in (
        "org_vocab_start_index",
        "org_vocab_end_index",
        "num_org_vocab_padding",
        "added_vocab_start_index",
        "added_vocab_end_index",
    ):
        assert getattr(vpe_inst, buf_name).dtype == torch.int32
    _DummyVPE(preset_buf=True)

    vpe_inst.tp_size = 1
    out_tp1 = vpe_inst.forward(torch.tensor([1, 2]))
    assert out_tp1.shape == (2, 1)

    vpe_inst.tp_size = 2
    monkeypatch.setattr(
        "vllm.distributed.tensor_model_parallel_all_reduce", lambda x: x
    )
    out_tp2 = vpe_inst.forward(torch.tensor([1, 5], dtype=torch.int32))
    assert out_tp2.shape == (2, 1)

    # Also test patched_get_masked_input_and_mask with plain int arguments
    masked_in, mask = vpe.get_masked_input_and_mask(torch.tensor([1, 5]), 0, 4, 0, 4, 6)
    assert masked_in.shape == (2,) and mask.shape == (2,)
    assert masked_in.dtype == torch.int64

    masked_in_i32, mask_i32 = vpe.get_masked_input_and_mask(
        torch.tensor([1, 5], dtype=torch.int32),
        vpe_inst.org_vocab_start_index,
        vpe_inst.org_vocab_end_index,
        vpe_inst.num_org_vocab_padding,
        vpe_inst.added_vocab_start_index,
        vpe_inst.added_vocab_end_index,
    )
    assert masked_in_i32.dtype == torch.int32
    assert torch.equal(masked_in_i32, masked_in.to(torch.int32))
    assert torch.equal(mask_i32, mask)

    # 8. "proxy" in envs.JAX_PLATFORMS top-level block
    import vllm.platforms

    monkeypatch.setattr(vllm_torchtpu.envs, "JAX_PLATFORMS", "proxy")
    monkeypatch.setattr(
        vllm.platforms, "_current_platform", vllm.platforms._current_platform
    )
    monkeypatch.setattr(
        vllm.platforms, "_init_trace", getattr(vllm.platforms, "_init_trace", "")
    )
    fake_pw = ModuleType("pathwaysutils")
    fake_pw.initialize = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pathwaysutils", fake_pw)
    monkeypatch.setattr(
        "vllm.platforms.resolve_current_platform_cls_qualname", lambda: "qual"
    )
    monkeypatch.setattr(
        "vllm.platforms.resolve_obj_by_qualname", lambda q: lambda: SimpleNamespace()
    )
    runpy.run_path(vllm_torchtpu.__file__)

    def _boom():
        raise RuntimeError("pw init failed")

    fake_pw.initialize = _boom  # type: ignore[attr-defined]
    runpy.run_path(vllm_torchtpu.__file__)
