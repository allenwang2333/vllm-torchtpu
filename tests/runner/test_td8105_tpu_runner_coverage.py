# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests achieving 100% statement and branch coverage on tpu_runner.py."""

from __future__ import annotations

import contextlib
import inspect
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

import vllm_torchtpu.runner.tpu_runner as tr_mod
from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutKind
from vllm_torchtpu.runner.tpu_runner import (
    ExecuteModelState,
    TPUModelRunner,
)
from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer


def _make_bare_runner() -> TPUModelRunner:
    runner = TPUModelRunner.__new__(TPUModelRunner)
    runner.device = torch.device("cpu")
    runner.vllm_config = MagicMock()
    runner.model_config = MagicMock()
    runner.cache_config = MagicMock()
    runner.scheduler_config = MagicMock()
    runner.scheduler_config.scheduler_cls = "default"
    runner.parallel_config = MagicMock()
    runner.parallel_config.data_parallel_size = 1
    runner.parallel_config.pipeline_parallel_size = 1
    runner.parallel_config.tensor_parallel_size = 1
    runner.parallel_config.world_size = 1
    runner.parallel_config.enable_expert_parallel = False
    runner.parallel_config.use_sequence_parallel_moe = False
    runner.parallel_config.is_moe_model = False
    runner.vllm_config.parallel_config = runner.parallel_config
    runner.vllm_config.scheduler_config = runner.scheduler_config
    runner.speculative_config = None
    runner.lora_config = None
    runner.load_config = MagicMock()
    runner.device_config = MagicMock()
    runner.device_config.device = torch.device("cpu")
    runner.kv_cache_config = None
    runner.model = MagicMock()
    runner.is_pooling_model = False
    runner.enforce_eager = False
    runner.check_recompilation = False
    runner.num_xla_graphs = 0
    runner.max_model_len = 128
    runner.most_model_len = None
    runner.max_num_reqs = 8
    runner.num_reqs_max_model_len = 8
    runner.num_reqs_most_model_len = 4
    runner.max_num_blocks_per_req = 4
    runner.num_blocks_per_most_len_req = 2
    runner.max_num_tokens = 16
    runner.num_tokens_paddings = [8, 16]
    runner.num_reqs_paddings = [8]
    runner.vocab_size = 32
    runner.block_size = 16
    runner.uses_mrope = False
    runner.supports_mm_inputs = False
    runner._pp_is_first = True
    runner._pp_is_last = True
    runner._pp_wave = None
    runner._pp_topk_buffer = None
    runner._pp_intermediate_template = None
    runner._pp_pending_scheduler_output = None
    runner._pp_chunk_granularity = None
    runner._pcp_enabled = False
    runner._mtp_enabled = False
    runner._has_mamba_state = False
    runner.mamba_slot_read_offsets = None
    runner._mamba_ckpt_window = 1
    runner._mamba_copy_plan = []
    runner._pending_mamba_state_copies = []
    runner._mamba_offset_seeded = set()
    runner._read_offset_scratch = {}
    runner._pool_block_split = 1
    runner.kv_cache_raw_tensors = []
    runner.bm_pool_layout = None
    runner._unified_kv_layout = False
    runner._hidden_states_dtype = torch.float32
    runner._profile_inputs = {}
    runner._attention_runs_batched_kernel = False
    runner._attention_capacity = {}
    runner._attention_kernel_block_size = 16
    runner._attention_kv_cache_group_id = None
    runner._attention_kernels_initialized = False
    runner._attn_layer_names = ["layer.0"]
    runner._attn_metadata_builder_ctx = None
    runner.empty_slot_mappings = {}
    runner._kv_cache_bundle = None
    runner.mesh = MagicMock()
    runner.drafter = None
    runner.rejection_sampler = None
    runner.spec_decode_manager = MagicMock()
    runner.structured_decoding_manager = MagicMock()
    runner.kv_cache_manager = MagicMock()
    runner.sequence_layout_planner = MagicMock()
    runner.execute_model_state = None
    runner._pre_async_results = None
    runner._sampling_generator = None
    runner._tp_rank0_mask = None
    runner._input_staging_fence = None
    runner._last_sequence_layout_plan = None
    runner._decode_device_cache_key = None
    runner.reorder_batch_threshold = 1
    runner.phase_based_profiler = None
    runner.batch_counter = 0
    runner._stage_timing = MagicMock()
    runner._stage_timing.stage.return_value = contextlib.nullcontext()
    runner._stage_timing.host_stage.return_value = contextlib.nullcontext()
    runner._stage_timing.take_samples.return_value = ({}, {})
    runner.input_ids_cpu = torch.zeros(16, dtype=torch.int32)
    runner.positions_cpu = torch.zeros(16, dtype=torch.int32)
    runner.positions_np = runner.positions_cpu.numpy()
    runner.query_start_loc_cpu = torch.zeros(17, dtype=torch.int32)
    runner.query_start_loc_np = runner.query_start_loc_cpu.numpy()
    runner.seq_lens_cpu = torch.zeros(16, dtype=torch.int32)
    runner.seq_lens_np = runner.seq_lens_cpu.numpy()
    runner.block_table_cpu = torch.zeros((8, 4), dtype=torch.int32)
    runner.mamba_state_indices_cpu = torch.zeros(8, dtype=torch.int32)
    runner._request_distribution_cpu = torch.zeros(3, dtype=torch.int32)
    runner._combined_request_distribution_cpu = torch.zeros(6, dtype=torch.int32)
    runner.arange_np = np.arange(16, dtype=np.int64)
    runner.requests = {}
    return runner


def test_module_helpers_and_compiled_unwrapped():
    # _spec_logprob_row_tokens & TPUModelRunner.spec_logprob_row_tokens
    fn = inspect.unwrap(tr_mod._spec_logprob_row_tokens)
    next_tokens = torch.tensor([[10, 11, -1], [20, -1, -1]], dtype=torch.int32)
    bonus_idx = torch.tensor([2, 0], dtype=torch.int32)
    draft_lens = torch.tensor([2, 0], dtype=torch.int32)
    final_idx = torch.zeros(4, dtype=torch.int32)
    out = fn(next_tokens, bonus_idx, draft_lens, final_idx)
    assert out.shape == (4, 1)

    runner = _make_bare_runner()
    method_fn = inspect.unwrap(TPUModelRunner.spec_logprob_row_tokens)
    out2 = method_fn(runner, next_tokens, bonus_idx, draft_lens, final_idx)
    assert torch.equal(out, out2)

    # _spec_warmup_request_block_ids
    runner.kv_cache_config = SimpleNamespace(
        num_blocks=2,
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16)),
            SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=8)),
        ],
    )
    runner.parallel_config.tensor_parallel_size = 1
    with patch.object(tr_mod.envs, "TPU_DEEPSEEK_V41_KV_SHARD", False):
        assert tr_mod._spec_warmup_request_block_ids(runner, [2, 1], 0, 1) is None

    # Sharded branch
    runner.parallel_config.tensor_parallel_size = 2
    fake_layout_small = SimpleNamespace(
        num_owners=2, num_replicated=2, blocks_per_owner=2
    )
    fake_layout_ok = SimpleNamespace(
        num_owners=2, num_replicated=10, blocks_per_owner=10
    )
    with (
        patch.object(tr_mod.envs, "TPU_DEEPSEEK_V41_KV_SHARD", True),
        patch(
            "vllm_torchtpu.core.owner_affine_block_pool.worker_owner_layout",
            return_value=fake_layout_small,
        ),
        patch(
            "vllm_torchtpu.core.owner_affine_block_pool.is_replicated_spec",
            side_effect=lambda spec: spec.block_size == 8,
        ),
    ):
        assert tr_mod._spec_warmup_request_block_ids(runner, [2, 2], 0, 2) is None
        runner._kv_shard_owner_layout = fake_layout_ok
        bids = tr_mod._spec_warmup_request_block_ids(runner, [2, 2], 1, 2)
        assert bids is not None and len(bids) == 2

    # _build_kv_connector_output when _KV_CONNECTOR_OUTPUT_SUPPORTS_INVALID_BLOCK_GROUP
    # is False
    with patch.object(
        tr_mod, "_KV_CONNECTOR_OUTPUT_SUPPORTS_INVALID_BLOCK_GROUP", False
    ):
        kv_out = tr_mod._build_kv_connector_output(
            finished_sending=None,
            finished_recving=None,
            kv_connector_worker_meta=None,
            invalid_block_ids=set(),
            invalid_block_group_index=None,
        )
        assert kv_out is not None

    # _validate_libtpu_version
    with (
        patch.object(tr_mod.importlib_metadata, "version", return_value="0.0.34"),
        pytest.raises(RuntimeError, match="Argmax is having accuracy issue"),
    ):
        tr_mod._validate_libtpu_version()
    with (
        patch.object(tr_mod.importlib_metadata, "version", return_value="0.0.35"),
        patch.object(tr_mod.logger, "warning_once") as mock_warn,
    ):
        tr_mod._validate_libtpu_version()
        mock_warn.assert_called_once()
    with patch.object(tr_mod.importlib_metadata, "version", return_value="0.0.37"):
        tr_mod._validate_libtpu_version()

    # _torch_tpu_wrapper
    orig_stream = getattr(torch.cuda, "Stream", None)
    if hasattr(torch.cuda, "mem_get_info"):
        delattr(torch.cuda, "mem_get_info")
    with (
        patch.object(torch, "tpu", MagicMock(), create=True),
        patch.object(torch.accelerator, "get_memory_info", return_value=(1, 2)),
        tr_mod._torch_tpu_wrapper(),
    ):
        assert torch.cuda.mem_get_info() == (1, 2)
        assert torch.cuda.empty_cache() is None
        delattr(torch.cuda, "empty_cache")
    assert getattr(torch.cuda, "Stream", None) is orig_stream
    # Also test when mem_get_info IS already set on torch.cuda so v is not None on
    # restore
    torch.cuda.mem_get_info = lambda: (3, 4)
    try:
        with (
            patch.object(torch, "tpu", MagicMock(), create=True),
            tr_mod._torch_tpu_wrapper(),
        ):
            pass
        assert torch.cuda.mem_get_info() == (3, 4)
    finally:
        delattr(torch.cuda, "mem_get_info")

    # _find_topk_indices_buffer
    m_empty = torch.nn.Linear(2, 2)
    assert tr_mod._find_topk_indices_buffer(m_empty) is None

    buf1 = torch.zeros((4, 2), dtype=torch.int32)
    m_one = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    m_one[0].topk_indices_buffer = buf1  # type: ignore[attr-defined]
    m_one[1].topk_indices_buffer = buf1  # type: ignore[attr-defined]
    assert tr_mod._find_topk_indices_buffer(m_one) is buf1

    buf2 = torch.zeros((8, 2), dtype=torch.int32)
    m_two = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    m_two[0].topk_indices_buffer = buf1  # type: ignore[attr-defined]
    m_two[1].topk_indices_buffer = buf2  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="several distinct top-k index tables"):
        tr_mod._find_topk_indices_buffer(m_two)

    # Bottom utility functions
    assert tr_mod._get_padded_num_kv_cache_update_slices(32, 4, 16) == 10
    assert tr_mod._prev_power_of_2(0) == 0
    assert tr_mod._prev_power_of_2(-5) == 0
    assert tr_mod._prev_power_of_2(12) == 8
    assert tr_mod._get_num_slices_per_kv_cache_update_block(1024) == 64
    assert tr_mod._get_num_slices_per_kv_cache_update_block(4 * 1024 * 1024) == 4


def test_tpu_model_runner_init_and_properties():
    runner = _make_bare_runner()
    runner.drafter = MagicMock(spec=Eagle3Proposer)
    assert runner._async_drafter is runner.drafter

    runner.speculative_config = None
    assert runner._is_async_drafter is False
    runner.speculative_config = SimpleNamespace(
        use_eagle=lambda: False, method="dflash"
    )
    assert runner._is_async_drafter is True

    # Test __init__ with preinit, dflash, pooling, cache_dtype != "auto", block_table,
    # mrope, mm
    def fake_super_init(self, vllm_config, device):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.scheduler_config = vllm_config.scheduler_config
        self.parallel_config = vllm_config.parallel_config
        self.speculative_config = vllm_config.speculative_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.device_config = vllm_config.device_config
        self.device = device
        self.dtype = "bfloat16"
        self.max_model_len = 64
        self.max_num_reqs = 8
        self.max_num_tokens = 16
        self.uses_mrope = getattr(vllm_config, "_test_mrope", False)
        self.supports_mm_inputs = getattr(vllm_config, "_test_mm", False)
        self.requests = {}
        self.input_batch = getattr(vllm_config, "_test_input_batch", SimpleNamespace())

    vllm_cfg1 = MagicMock()
    vllm_cfg1._test_mrope = True
    vllm_cfg1._test_mm = True
    bt0 = MagicMock()
    bt0.get_cpu_tensor.return_value = torch.zeros((8, 6), dtype=torch.int32)
    vllm_cfg1._test_input_batch = SimpleNamespace(block_table=[bt0])
    vllm_cfg1.speculative_config.method = "dflash"
    vllm_cfg1.speculative_config.num_speculative_tokens = 2
    vllm_cfg1.model_config.runner_type = "pooling"
    vllm_cfg1.model_config.enforce_eager = True
    vllm_cfg1.model_config.get_sliding_window.return_value = None
    vllm_cfg1.model_config.get_num_layers_by_block_type.return_value = 2
    vllm_cfg1.model_config.get_num_kv_heads.return_value = 4
    vllm_cfg1.model_config.get_head_size.return_value = 64
    vllm_cfg1.model_config.get_vocab_size.return_value = 128
    vllm_cfg1.cache_config.cache_dtype = "fp8"
    vllm_cfg1.cache_config.block_size = 16
    vllm_cfg1.compilation_config.compile_sizes = [8, 16]
    vllm_cfg1.parallel_config.pipeline_parallel_size = 1
    vllm_cfg1.profiler_config.profiler = None

    planner1 = MagicMock()
    planner1.requires_backend_preinit = True
    planner1.backend_preinit_world_size = 4

    with (
        patch.object(torch, "tpu", MagicMock(), create=True),
        patch.object(tr_mod.GPUModelRunner, "__init__", fake_super_init),
        patch.object(
            tr_mod.PcpStaticSupportValidator,
            "from_vllm_config",
            return_value=SimpleNamespace(enabled=False, speculative_method=None),
        ),
        patch.object(tr_mod, "create_sequence_layout_planner", return_value=planner1),
        patch.object(tr_mod.jax, "local_devices", return_value=[MagicMock()]),
        patch.object(tr_mod, "normalize_draft_config") as mock_norm,
        patch.object(tr_mod, "_validate_libtpu_version"),
        patch.object(tr_mod, "unified_kv_layout_enabled", return_value=True),
        patch.object(tr_mod.envs, "VLLM_TPU_MOST_MODEL_LEN", 32),
        patch.object(tr_mod, "RunnerStageTiming", MagicMock()),
        patch.object(
            TPUModelRunner, "_create_mesh_for_parallelism", return_value="mesh"
        ),
        patch.object(TPUModelRunner, "_init_speculative_decoding"),
        patch.object(TPUModelRunner, "_init_phased_profiling"),
        patch.object(TPUModelRunner, "_make_buffer", return_value=MagicMock()),
        patch(
            "vllm_torchtpu.runner.tpu_pooling.install_tpu_pooling_patches"
        ) as mock_pool_patch,
        patch(
            "vllm_torchtpu.runner.tpu_pooling.build_pooling_metadata",
            return_value="pooled_meta",
        ),
        patch.dict(
            sys.modules,
            {"vllm.v1.worker.tpu_input_batch": SimpleNamespace(InputBatch=MagicMock())},
        ),
    ):
        r1 = TPUModelRunner(
            vllm_cfg1,
            torch.device("cpu"),
            profiler_rank=0,
            profiler_world_size=1,
        )
        mock_norm.assert_called_once()
        mock_pool_patch.assert_called_once()
        assert r1.input_batch.get_pooling_metadata() == "pooled_meta"
        assert r1.max_num_blocks_per_req == 6

    # Variant 2: cache_dtype == "auto" with str self.dtype, block_table raising
    # IndexError, uses_mrope=False, supports_mm_inputs=False
    vllm_cfg2 = MagicMock()
    vllm_cfg2._test_mrope = False
    vllm_cfg2._test_mm = False
    vllm_cfg2._test_input_batch = SimpleNamespace(block_table=[])
    vllm_cfg2.speculative_config = None
    vllm_cfg2.model_config.runner_type = "generate"
    vllm_cfg2.model_config.enforce_eager = True
    vllm_cfg2.model_config.get_sliding_window.return_value = None
    vllm_cfg2.model_config.get_num_layers_by_block_type.return_value = 2
    vllm_cfg2.model_config.get_num_kv_heads.return_value = 4
    vllm_cfg2.model_config.get_head_size.return_value = 64
    vllm_cfg2.model_config.get_vocab_size.return_value = 128
    vllm_cfg2.cache_config.cache_dtype = "auto"
    vllm_cfg2.cache_config.block_size = 16
    vllm_cfg2.compilation_config.compile_sizes = [8, 16]
    vllm_cfg2.parallel_config.pipeline_parallel_size = 1
    vllm_cfg2.profiler_config.profiler = None

    planner2 = MagicMock()
    planner2.requires_backend_preinit = False
    with (
        patch.object(torch, "tpu", MagicMock(), create=True),
        patch.object(tr_mod.GPUModelRunner, "__init__", fake_super_init),
        patch.object(
            tr_mod.PcpStaticSupportValidator,
            "from_vllm_config",
            return_value=SimpleNamespace(enabled=False, speculative_method=None),
        ),
        patch.object(tr_mod, "create_sequence_layout_planner", return_value=planner2),
        patch.object(tr_mod, "_validate_libtpu_version"),
        patch.object(tr_mod, "unified_kv_layout_enabled", return_value=False),
        patch.object(tr_mod.envs, "VLLM_TPU_MOST_MODEL_LEN", None),
        patch.object(tr_mod, "RunnerStageTiming", MagicMock()),
        patch.object(
            TPUModelRunner, "_create_mesh_for_parallelism", return_value="mesh"
        ),
        patch.object(TPUModelRunner, "_init_speculative_decoding"),
        patch.object(TPUModelRunner, "_init_phased_profiling"),
    ):
        r2 = TPUModelRunner(
            vllm_cfg2,
            torch.device("cpu"),
            profiler_rank=0,
            profiler_world_size=1,
        )
        assert r2.kv_cache_dtype == torch.bfloat16
        assert r2.max_num_blocks_per_req == 4
        vllm_cfg2._test_input_batch = None
        r3 = TPUModelRunner(
            vllm_cfg2,
            torch.device("cpu"),
            profiler_rank=0,
            profiler_world_size=1,
        )
        assert r3.max_num_blocks_per_req == 4


def test_hooks_spec_init_rope_mesh_xla_graphs():
    runner = _make_bare_runner()
    runner._init_device_properties()
    with patch.object(tr_mod, "synchronize_device") as mock_sync:
        runner._sync_device()
        mock_sync.assert_called_once()

    # maybe_setup_kv_connector & kv_connector_no_forward
    so = SimpleNamespace(kv_connector_metadata=MagicMock(), finished_req_ids=set())
    with patch.object(tr_mod, "has_kv_transfer_group", return_value=False):
        runner.maybe_setup_kv_connector(so)

    fake_conn = MagicMock()
    fake_conn.get_finished.return_value = (None, None)
    fake_conn.get_block_ids_with_load_errors.return_value = set()
    fake_conn.build_connector_worker_meta.return_value = None
    fake_conn.get_kv_connector_stats.return_value = None
    with (
        patch.object(tr_mod, "has_kv_transfer_group", return_value=True),
        patch.object(tr_mod, "get_kv_transfer_group", return_value=fake_conn),
        patch.object(tr_mod.dist_utils, "get_raiden_inline_load", return_value=False),
    ):
        runner.maybe_setup_kv_connector(so)
        fake_conn.start_load_kv.assert_called_with(None)
    with (
        patch.object(tr_mod, "has_kv_transfer_group", return_value=True),
        patch.object(tr_mod, "get_kv_transfer_group", return_value=fake_conn),
        patch.object(tr_mod.dist_utils, "get_raiden_inline_load", return_value=True),
    ):
        out_empty = runner.kv_connector_no_forward(so, runner.vllm_config)
        assert out_empty is EMPTY_MODEL_RUNNER_OUTPUT
        fake_conn.start_load_kv.assert_called_with(
            None, wait_for_completion=True, report_completion=True
        )

    # _init_speculative_decoding
    runner.speculative_config = None
    runner._init_speculative_decoding()
    assert runner.drafter is None

    for method, cls_name in [
        ("ngram", "NgramProposer"),
        ("dflash", "DFlashProposer"),
        ("dspark", "DSparkProposer"),
        ("eagle3", "Eagle3Proposer"),
    ]:
        runner.speculative_config = SimpleNamespace(
            method=method, use_eagle=lambda m=method: m == "eagle3"
        )
        with (
            patch.object(tr_mod, "RejectionSampler", MagicMock()),
            patch.object(tr_mod, cls_name, MagicMock()) as mock_cls,
        ):
            runner._init_speculative_decoding()
            mock_cls.assert_called_once()

    runner.speculative_config = SimpleNamespace(
        method="unsupported", use_eagle=lambda: False
    )
    with (
        patch.object(tr_mod, "RejectionSampler", MagicMock()),
        pytest.raises(NotImplementedError, match="Unsupported speculative decoding"),
    ):
        runner._init_speculative_decoding()

    # _truncate_rope_caches
    runner.model_config.is_multimodal_model = True
    runner.model_config.multimodal_config = None
    runner._truncate_rope_caches()

    runner.model_config.is_multimodal_model = False
    runner.model_config.max_model_len = 8
    runner.model = torch.nn.Linear(2, 2)
    runner._truncate_rope_caches()  # num_eligible == 0

    rope1 = tr_mod.RotaryEmbedding.__new__(tr_mod.RotaryEmbedding)
    torch.nn.Module.__init__(rope1)
    rope1.cos_sin_cache = torch.zeros((16, 4))
    rope1.cos_sin_cache_bf16 = None
    rope2 = tr_mod.RotaryEmbedding.__new__(tr_mod.RotaryEmbedding)
    torch.nn.Module.__init__(rope2)
    rope2.cos_sin_cache = torch.zeros((4, 4))
    runner.model = torch.nn.Sequential(rope1, rope2, torch.nn.Linear(2, 2))
    runner._truncate_rope_caches()
    assert rope1.cos_sin_cache.shape[0] == 8

    # _create_mesh_for_parallelism
    with (
        patch.object(tr_mod.jax, "local_devices", return_value=[]),
        pytest.raises(ValueError, match="No TPU devices are visible"),
    ):
        runner._create_mesh_for_parallelism()

    runner.parallel_config.world_size = 1
    runner.parallel_config.tensor_parallel_size = 2
    with (
        patch.object(tr_mod.jax, "local_devices", return_value=[MagicMock()]),
        pytest.raises(
            ValueError, match="Single-process TPU mesh TP>1 is not supported"
        ),
    ):
        runner._create_mesh_for_parallelism()

    runner.parallel_config.tensor_parallel_size = 1
    with (
        patch.object(tr_mod.jax, "local_devices", return_value=[SimpleNamespace(id=0)]),
        patch.object(tr_mod.jax, "device_count", return_value=1),
        patch.object(tr_mod.jax, "devices", return_value=[SimpleNamespace(id=0)]),
        patch.object(tr_mod, "Mesh", return_value="fake_mesh") as mock_mesh,
    ):
        assert runner._create_mesh_for_parallelism() == "fake_mesh"
        mock_mesh.assert_called_once()

    # _describe_xla_graph & _fresh_xla_graphs
    class _FakeEntry:
        def method(self):
            return 1

        @property
        def broken(self):
            raise RuntimeError("boom")

        long_val = "x" * 130

    desc = TPUModelRunner._describe_xla_graph(_FakeEntry())
    assert "long_val=" in desc and "..." in desc

    runner._xla_graphs_checked_at = 10
    entries = [
        SimpleNamespace(last_read=None, read_count=1),
        SimpleNamespace(last_read=5, read_count=1),
        SimpleNamespace(last_read=15, read_count=1),
    ]
    fresh = runner._fresh_xla_graphs(entries)
    assert len(fresh) == 1


def test_metadata_mamba_and_cow_copies():
    runner = _make_bare_runner()
    builder = MagicMock()
    builder.build.return_value = "built_meta"
    runner.attn_groups = [
        [SimpleNamespace(metadata_builders=[builder], layer_names=["l0"])]
    ]
    runner._attn_metadata_builder_ctx = SimpleNamespace(staged_block_tables=None)
    with (
        patch.object(tr_mod, "stage_block_table_uploads") as mock_stage,
        patch.object(tr_mod.envs, "TPU_DEEPSEEK_V41_KV_SHARD", True),
        patch.object(tr_mod, "attach_step_owner_token_layout") as mock_attach,
    ):
        meta, _ = runner._build_attention_metadata(
            num_tokens=8, num_reqs=4, num_tokens_padded=None, num_reqs_padded=None
        )
        assert meta == {"l0": "built_meta"}
        mock_stage.assert_called_once()
        mock_attach.assert_called_once()

    # Also test when staged_block_tables is already set and TPU_DEEPSEEK_V41_KV_SHARD is
    # False
    runner._attn_metadata_builder_ctx = SimpleNamespace(staged_block_tables="staged")
    with patch.object(tr_mod.envs, "TPU_DEEPSEEK_V41_KV_SHARD", False):
        meta2, _ = runner._build_attention_metadata(
            num_tokens=8, num_reqs=4, num_tokens_padded=8, num_reqs_padded=4
        )
        assert meta2 == {"l0": "built_meta"}

    # get_kv_prewarm_shapes & prewarm_kv_offload_shape
    runner.kv_cache_manager.get_kv_prewarm_shapes.return_value = [16, 32]
    assert runner.get_kv_prewarm_shapes() == [16, 32]
    runner.prewarm_kv_offload_shape(16)
    runner.kv_cache_manager.prewarm_kv_offload_shape.assert_called_once_with(16)

    # _build_mamba_copy_plan branches
    runner.cache_config.mamba_cache_mode = "none"
    runner._build_mamba_copy_plan(MagicMock(), {}, None)
    assert runner._mamba_copy_plan == []

    runner.cache_config.mamba_cache_mode = "align"
    mamba_spec = tr_mod.MambaSpec.__new__(tr_mod.MambaSpec)
    kv_cfg = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=mamba_spec, layer_names=["m0"])]
    )
    # Empty raws & 1D raw tensor in layer_to_raw
    runner._build_mamba_copy_plan(kv_cfg, {"other": torch.zeros(4)}, None)
    assert runner._mamba_copy_plan == []

    # _pad_dev_to_bucket when padded == n
    t8 = torch.zeros(8, dtype=torch.int32)
    assert runner._pad_dev_to_bucket(t8) is t8

    # _flush_mamba_state_seed_copies when empty
    runner._pending_mamba_state_copies = []
    runner._flush_mamba_state_seed_copies()

    # _dense_kv_copy_supported cache hit & _precompile_dense_kv_cache_block_copies
    runner.kv_cache_config = None
    runner._precompile_dense_kv_cache_block_copies()

    runner.kv_cache_config = SimpleNamespace(has_mamba_layers=False, num_blocks=4)
    runner.kv_caches = [torch.zeros((8, 4))]
    with (
        patch("vllm_torchtpu._dcp_fine_prefix_match_unit", return_value=16),
        patch.object(tr_mod, "copy_pool_blocks") as mock_copy_pool,
        patch.object(tr_mod, "synchronize_tensors"),
    ):
        assert runner._dense_kv_copy_supported() is True
        # Second call hits _dense_kv_copy_cache
        assert runner._dense_kv_copy_supported() is True
        runner._precompile_dense_kv_cache_block_copies()
        assert mock_copy_pool.called


def test_async_substitution_and_room_guard():
    runner = _make_bare_runner()
    inp = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    # Empty indices
    assert (
        runner._apply_async_token_substitution(
            inp, np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        )
        is inp
    )
    # _pre_async_results is None
    runner._pre_async_results = None
    idx = np.array([0, 1], dtype=np.int32)
    assert runner._apply_async_token_substitution(inp, idx, idx) is inp

    # _fast_token_substitution when n_subst != len(input_ids) and when len(next_tpu) <
    # len(input_ids)
    runner._fast_token_substitution = True
    runner._pre_async_results = SimpleNamespace(
        next_tokens_tpu=torch.tensor([10, 20], dtype=torch.int32)
    )
    with patch.object(
        tr_mod, "_substitute_placeholder_token", side_effect=lambda a, b, c, d: a
    ):
        # n_subst (2) != len(inp) (4)
        runner._apply_async_token_substitution(inp, idx, idx)
        # n_subst == 4 == len(inp), but len(next_tpu) == 2 < 4
        idx4 = np.arange(4, dtype=np.int32)
        runner._apply_async_token_substitution(inp, idx4, idx4)

    # _flush_disjoint_async_results
    runner._pre_async_results = None
    runner._flush_disjoint_async_results()

    runner.input_batch = SimpleNamespace(req_ids=["r1"], num_reqs=1)
    runner._pre_async_results = SimpleNamespace(req_id_to_index_copy={"r1": 0})
    runner._flush_disjoint_async_results()
    assert runner._pre_async_results is not None

    runner._pre_async_results = SimpleNamespace(req_id_to_index_copy={"r_old": 0})
    runner._modify_prev_results = MagicMock()
    runner._flush_disjoint_async_results()
    runner._modify_prev_results.assert_called_once()
    assert runner._pre_async_results is None

    # _install_spec_token_room_guard edge cases
    orig_update = MagicMock(return_value="orig_ret")
    runner.input_batch = SimpleNamespace(
        update_req_spec_token_ids=orig_update,
        req_id_to_index={"r1": 0},
        token_ids_cpu=np.zeros((2, 4), dtype=np.int32),
        num_tokens_no_spec=np.array([3, 0], dtype=np.int32),
        spec_token_ids=None,
        block_table=None,
    )
    runner._install_spec_token_room_guard()
    req = SimpleNamespace(req_id="r1", prev_num_draft_len=0)
    # Empty ids -> line 2541
    assert runner.input_batch.update_req_spec_token_ids(req, {}) == "orig_ret"
    # Clamped write with spec_token_ids=None -> 2571->2574
    runner.input_batch.update_req_spec_token_ids(req, {"r1": [10, 11, 12]})
    assert req.prev_num_draft_len == 3


def test_sync_replicated_drafts_across_tp():
    runner = _make_bare_runner()
    assert runner._sync_replicated_drafts_across_tp(None) is None
    drafts = torch.tensor([[1, 2]], dtype=torch.int32)
    runner.speculative_config = None
    assert runner._sync_replicated_drafts_across_tp(drafts) is drafts

    runner.speculative_config = SimpleNamespace(draft_tensor_parallel_size=1)
    with patch("vllm.distributed.get_tensor_model_parallel_world_size", return_value=1):
        assert runner._sync_replicated_drafts_across_tp(drafts) is drafts

    unwrapped_pin = inspect.unwrap(TPUModelRunner._tpu_pin_drafts_to_rank0)
    with (
        patch("vllm.distributed.get_tensor_model_parallel_world_size", return_value=2),
        patch("vllm.distributed.get_tensor_model_parallel_rank", return_value=0),
        patch(
            "vllm.distributed.tensor_model_parallel_all_reduce",
            side_effect=lambda x: x,
        ),
        patch.object(
            runner,
            "_tpu_pin_drafts_to_rank0",
            side_effect=lambda d, m: unwrapped_pin(runner, d, m),
        ),
    ):
        res1 = runner._sync_replicated_drafts_across_tp(drafts)
        assert torch.equal(res1, drafts)
        # Second call reuses _tp_rank0_mask
        res2 = runner._sync_replicated_drafts_across_tp(drafts)
        assert torch.equal(res2, drafts)


def test_prepare_inputs_branches():
    runner = _make_bare_runner()
    runner.most_model_len = 4
    runner.num_reqs_most_model_len = 1
    runner.num_reqs_max_model_len = 2
    runner.uses_mrope = True
    runner.mrope_positions = SimpleNamespace(
        cpu=torch.zeros((3, 17), dtype=torch.int32)
    )
    runner._calc_mrope_positions = MagicMock()
    runner.mamba_slot_read_offsets = torch.zeros(8, dtype=torch.int32)
    runner._fast_token_substitution = True
    runner.lora_config = MagicMock()
    runner.set_active_loras = MagicMock()
    runner.attn_groups = []
    runner.input_batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["r0", "r1"],
        req_id_to_index={"r0": 0, "r1": 1},
        num_computed_tokens_cpu=np.array([1, 1], dtype=np.int32),
        token_ids_cpu=np.zeros((2, 16), dtype=np.int32),
        token_ids_cpu_tensor=torch.zeros((2, 16), dtype=torch.int32),
        num_prompt_tokens=np.array([2, 2], dtype=np.int32),
    )

    plan_all = SimpleNamespace(
        kind=SequenceLayoutKind.ALL,
        global_padded_num_tokens=8,
        global_num_tokens=2,
        local_num_tokens=2,
        local_padded_num_tokens=8,
        token_slice=slice(0, 8),
        logits_indices_cpu=None,
        descriptor=SimpleNamespace(
            cache_key="k", kind=SequenceLayoutKind.ALL, protocol="p", version=1
        ),
    )
    runner.sequence_layout_planner.prepare_real.return_value = plan_all

    # 1) num_tokens <= most_model_len (use_max_model_len=False), 2 reqs >
    # num_reqs_most_model_len (1)
    so1 = SimpleNamespace(
        total_num_scheduled_tokens=2,
        num_scheduled_tokens={"r0": 1, "r1": 1},
    )
    with patch.object(tr_mod, "stage_block_table_uploads"):
        runner._prepare_inputs(so1, start_index=0, num_decode_reqs=2)
        # Call again with start_index=1 so len <= num_reqs_most_model_len and
        # _cached_comb_key hits
        runner._prepare_inputs(so1, start_index=1, num_decode_reqs=2)

    # 2) num_tokens > most_model_len -> sets use_max_model_len = True, and non-ALL
    # layout for uses_mrope
    other_kind = SimpleNamespace(value="other")
    plan_partial = SimpleNamespace(
        kind=other_kind,
        global_padded_num_tokens=8,
        global_num_tokens=6,
        local_num_tokens=6,
        local_padded_num_tokens=8,
        token_slice=slice(0, 8),
        logits_indices_cpu=None,
        descriptor=SimpleNamespace(
            cache_key="k2", kind=other_kind, protocol="p", version=1
        ),
    )
    runner.sequence_layout_planner.prepare_real.return_value = plan_partial
    so2 = SimpleNamespace(
        total_num_scheduled_tokens=6,
        num_scheduled_tokens={"r0": 5, "r1": 1},
    )
    with patch.object(tr_mod, "stage_block_table_uploads"):
        runner._prepare_inputs(so2, start_index=0, num_decode_reqs=0)
    assert runner.set_active_loras.called


def test_mm_and_dp_helpers():
    runner = _make_bare_runner()
    runner.model = MagicMock()
    with (
        patch.object(
            tr_mod, "chunked_video_encoding", return_value=contextlib.nullcontext()
        ),
        patch.object(
            tr_mod.GPUModelRunner, "_execute_mm_encoder", return_value=["enc"]
        ),
    ):
        assert runner._execute_mm_encoder(MagicMock()) == ["enc"]

    # _get_model_inputs
    inp = torch.zeros(8, dtype=torch.int32)
    runner.supports_mm_inputs = False
    assert runner._get_model_inputs(inp, None) == (inp, None)

    runner.supports_mm_inputs = True
    runner.model.embed_input_ids.return_value = "embeds"
    assert runner._get_model_inputs(inp, None) == (None, "embeds")
    assert runner._get_model_inputs(inp, ([], torch.ones(4, dtype=torch.bool))) == (
        None,
        "embeds",
    )
    assert runner._get_model_inputs(inp, ([], torch.ones(8, dtype=torch.bool))) == (
        None,
        "embeds",
    )

    # _count_input_chunks
    so_zero = SimpleNamespace(total_num_scheduled_tokens=0)
    assert runner._count_input_chunks(so_zero) == (0, 0)

    runner.input_batch = SimpleNamespace(num_reqs=2, req_ids=["r0", "r1"])
    runner.most_model_len = 4
    runner.num_reqs_most_model_len = 1
    runner.num_reqs_max_model_len = 2
    so_short = SimpleNamespace(
        total_num_scheduled_tokens=2, num_scheduled_tokens={"r0": 1, "r1": 1}
    )
    assert runner._count_input_chunks(so_short) == (2, 1)
    so_long = SimpleNamespace(
        total_num_scheduled_tokens=6, num_scheduled_tokens={"r0": 5, "r1": 1}
    )
    assert runner._count_input_chunks(so_long) == (1, 2)

    # _dp_lockstep_enabled, _dp_coordinated_step, _dp_num_tokens_across_dp
    runner.parallel_config.enable_expert_parallel = True
    with patch.object(tr_mod.utils, "get_dp_size", return_value=1):
        assert runner._dp_lockstep_enabled() is False
        assert runner._dp_coordinated_step(8, 1) == (None, 1)
        assert runner._dp_num_tokens_across_dp(8) is None

    with (
        patch.object(tr_mod.utils, "get_dp_size", return_value=2),
        patch("vllm.distributed.parallel_state.get_dp_group") as mock_dp_grp,
        patch.object(tr_mod.torch.distributed, "all_reduce"),
    ):
        mock_dp_grp.return_value = SimpleNamespace(cpu_group="grp")
        assert runner._dp_lockstep_enabled() is True
        bucket, chunks = runner._dp_coordinated_step(
            5, 1, local_num_reqs=2, has_encoder_inputs=True
        )
        assert bucket == 8 and chunks == 1
        across = runner._dp_num_tokens_across_dp(8)
        assert across is not None and across.shape == (2,)

    # _execute_dp_mm_encoder
    runner._dp_step_has_encoder_inputs = False
    runner._execute_dp_mm_encoder()

    runner._dp_step_has_encoder_inputs = True
    runner.encoder_cudagraph_manager = None
    with pytest.raises(ValueError, match="requires encoder graph replay"):
        runner._execute_dp_mm_encoder()

    mgr = MagicMock()
    mgr.execute_dp_coordinated.return_value = ["out0"]
    runner.encoder_cudagraph_manager = mgr
    runner._cache_encoder_output = MagicMock()
    so_mm = SimpleNamespace(ec_manager_metadata="ec", free_encoder_mm_hashes=["f"])
    with patch.object(tr_mod, "synchronize_tensors"):
        # scheduler_output is None
        runner._execute_dp_mm_encoder(None)
        # lora_config error captured in input_error
        runner.lora_config = MagicMock()
        runner._batch_mm_inputs_from_scheduler = MagicMock(
            return_value=(["h0"], [("image", "kw")], None)
        )
        runner._execute_dp_mm_encoder(so_mm)
        assert (
            "supports images without LoRA"
            in mgr.execute_dp_coordinated.call_args.kwargs["input_error"]
        )

        # len(batches) != 1 error
        runner.lora_config = None
        with patch.object(tr_mod, "group_and_batch_mm_kwargs", return_value=[]):
            runner._execute_dp_mm_encoder(so_mm)
            assert (
                "requires one image batch"
                in mgr.execute_dp_coordinated.call_args.kwargs["input_error"]
            )

        # Normal 1-batch path + empty mm_kwargs path
        runner._cache_encoder_output.reset_mock()
        with patch.object(
            tr_mod,
            "group_and_batch_mm_kwargs",
            return_value=[("image", 1, {"pixel_values": 1})],
        ):
            runner._execute_dp_mm_encoder(so_mm)
            runner._cache_encoder_output.assert_called_once()
        runner._batch_mm_inputs_from_scheduler = MagicMock(return_value=([], [], None))
        mgr.execute_dp_coordinated.return_value = []
        runner._execute_dp_mm_encoder(so_mm)

    # _run_dp_dummy_chunk & _run_dp_idle_pairing (spec=None)
    runner._dummy_run = MagicMock()
    runner.speculative_config = None
    runner._dp_step_has_encoder_inputs = False
    runner._run_dp_idle_pairing(8, 1)
    runner._dummy_run.assert_called_once()


def test_execute_model_branches():
    runner = _make_bare_runner()
    runner.execute_model_state = MagicMock()
    with pytest.raises(RuntimeError, match="State error"):
        runner.execute_model(MagicMock())
    runner.execute_model_state = None

    runner._update_states = MagicMock()
    runner.scheduler_config.async_scheduling = True
    runner._flush_disjoint_async_results = MagicMock()
    so_empty = SimpleNamespace(total_num_scheduled_tokens=0)
    with patch.object(tr_mod, "has_kv_transfer_group", return_value=False):
        assert runner.execute_model(so_empty) is EMPTY_MODEL_RUNNER_OUTPUT
    with (
        patch.object(tr_mod, "has_kv_transfer_group", return_value=True),
        patch.object(runner, "kv_connector_no_forward", return_value="kv_no_fwd"),
    ):
        assert runner.execute_model(so_empty) == "kv_no_fwd"

    # PCP + MTP guard
    runner._pcp_enabled = True
    runner._mtp_enabled = True
    runner.input_batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["r0"],
        req_id_to_index={"r0": 0},
        num_computed_tokens_cpu=np.array([4], dtype=np.int32),
        num_prompt_tokens=np.array([4], dtype=np.int32),
        num_computed_tokens_cpu_tensor=torch.tensor([4], dtype=torch.int32),
    )
    so_bad = SimpleNamespace(
        total_num_scheduled_tokens=1,
        num_scheduled_tokens={"r0": 1},
    )
    with pytest.raises(RuntimeError, match="PCP \\+ MTP only supports prefill"):
        runner.execute_model(so_bad)
    runner._pcp_enabled = False
    runner._mtp_enabled = False

    # Multimodal + phase_based_profiler + TraceAnnotation + DP dummy chunk + pooling +
    # PP non-last
    runner.supports_mm_inputs = True
    runner._pp_is_first = True
    runner.phase_based_profiler = MagicMock()
    runner.input_batch.num_computed_tokens_cpu = np.array([0], dtype=np.int32)
    runner.input_batch.get_pooling_metadata = MagicMock(return_value=MagicMock())
    so_run = SimpleNamespace(
        total_num_scheduled_tokens=2,
        num_scheduled_tokens={"r0": 2},
        scheduled_encoder_inputs={"r0": "enc"},
        scheduled_spec_decode_tokens={},
    )
    runner._reorder_batch_for_rpa = MagicMock(return_value=(0, 0))
    runner._count_input_chunks = MagicMock(return_value=(1, 1))
    # Return target_num_chunks=2 while num_reqs=1 so chunk 1 runs _run_dp_dummy_chunk!
    runner._dp_coordinated_step = MagicMock(return_value=(8, 2))
    runner._execute_mm_encoder = MagicMock()
    runner._gather_mm_embeddings = MagicMock(
        return_value=([torch.zeros((1, 4))], torch.tensor([True, False]))
    )
    runner._run_dp_dummy_chunk = MagicMock()

    plan = SimpleNamespace(
        kind=SequenceLayoutKind.ALL,
        local_num_tokens=2,
        descriptor=SimpleNamespace(cache_key="k"),
    )
    spec_md = SimpleNamespace(
        draft_token_ids=torch.zeros(2, dtype=torch.int32),
        final_logits_indices=torch.zeros(1, dtype=torch.int32),
        target_logits_indices=torch.zeros(2, dtype=torch.int32),
        draft_lengths=torch.ones(1, dtype=torch.int32),
    )

    def fake_prepare(*args, **kwargs):
        runner.input_ids = torch.zeros(2, dtype=torch.int32)
        runner.position_ids = torch.zeros(2, dtype=torch.int32)
        runner._last_sequence_layout_plan = plan
        runner._attn_metadata_builder_ctx = SimpleNamespace(
            mamba_state_indices=torch.zeros(1, dtype=torch.int32),
            unified_mamba_state_indices=None,
        )
        return (
            {},
            torch.zeros(1, dtype=torch.int32),
            8,
            1,
            1,
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            spec_md,
            None,
            plan,
        )

    runner._prepare_inputs = fake_prepare
    runner._apply_async_token_substitution = MagicMock(side_effect=lambda x, *a: x)
    runner._record_input_staging_fence = MagicMock()
    runner._get_model_inputs = MagicMock(return_value=(None, torch.zeros((2, 4))))
    runner.forward_model = MagicMock(
        return_value=(torch.zeros((2, 4)), [torch.zeros((2, 4))])
    )
    runner.sequence_layout_planner.finalize_hidden_states.side_effect = lambda h, p: h
    runner.model = MagicMock()
    runner.model.pooler.return_value = [torch.zeros(4)]
    runner.is_pooling_model = True
    runner.mamba_slot_read_offsets = torch.zeros(4, dtype=torch.int32)
    runner.drafter = MagicMock(spec=Eagle3Proposer)
    runner.speculative_config = SimpleNamespace(
        use_eagle=lambda: True, method="eagle3", num_speculative_tokens=1
    )

    with (
        patch.object(tr_mod, "synchronize_tensors"),
        patch.object(
            tr_mod.runner_utils,
            "get_batch_composition_stats",
            return_value={
                "num_prefill_tokens": 2,
                "num_decode_tokens": 0,
                "phase": "prefill",
                "batch_id": 1,
                "total_num_scheduled_tokens": 2,
                "padded_total_num_scheduled_tokens": 8,
                "min_kv_len": 0,
            },
        ),
        patch.object(tr_mod.TraceAnnotation, "is_enabled", return_value=True),
        patch(
            "vllm_torchtpu.runner.tpu_runner_async_output.extract_draft_token_ids",
            return_value=torch.zeros(2, dtype=torch.int32),
        ),
        patch.object(runner, "_reset_read_offsets_for_new_requests"),
        patch.object(runner, "sample_tokens", return_value="pooled_sample_out"),
    ):
        res = runner.execute_model(so_run)
        assert res == "pooled_sample_out"
        runner._run_dp_dummy_chunk.assert_called_once_with(8)
    runner.execute_model_state = None

    # Also test execute_model when not _pp_is_first, _pp_wave is not None, not
    # _pp_is_last, and _dp_lockstep_enabled
    runner._pp_is_first = False
    runner._pp_is_last = False
    runner._pp_wave = MagicMock()
    runner._pp_wave.start_forward.return_value = tr_mod.IntermediateTensors(
        {"h": torch.zeros((2, 4))}
    )
    runner.forward_model = MagicMock(
        return_value=(tr_mod.IntermediateTensors({"h": torch.zeros((2, 4))}), None)
    )
    runner._dp_coordinated_step = MagicMock(return_value=(8, 1))
    with (
        patch(
            "vllm_torchtpu.runner.tpu_runner_async_output.extract_draft_token_ids",
            return_value=torch.zeros(2, dtype=torch.int32),
        ),
        patch.object(runner, "_reset_read_offsets_for_new_requests"),
    ):
        assert runner.execute_model(so_run) is None
    assert runner._pp_pending_scheduler_output is so_run
    runner.execute_model_state = None

    # And when _pp_is_first=True with _dp_lockstep_enabled()=True, _pp_is_last=True,
    # _pp_wave is not None, mm_embeds_list=[], unified_mamba_state_indices
    runner._pp_is_first = True
    runner._pp_is_last = True
    runner.is_pooling_model = False
    runner._gather_mm_embeddings = MagicMock(
        return_value=([], torch.tensor([False, False]))
    )
    runner._execute_dp_mm_encoder = MagicMock()

    def fake_prepare_unified(*args, **kwargs):
        runner.input_ids = torch.zeros(2, dtype=torch.int32)
        runner.position_ids = torch.zeros(2, dtype=torch.int32)
        runner._last_sequence_layout_plan = plan
        runner._attn_metadata_builder_ctx = SimpleNamespace(
            mamba_state_indices=None,
            unified_mamba_state_indices=[torch.zeros(1, dtype=torch.int32)],
        )
        return (
            {},
            torch.zeros(1, dtype=torch.int32),
            8,
            1,
            1,
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            None,
            torch.zeros(2, dtype=torch.int32),
            plan,
        )

    runner._prepare_inputs = fake_prepare_unified
    runner.forward_model = MagicMock(return_value=(torch.zeros((2, 4)), None))
    runner.sequence_layout_planner.maybe_select_logits_hidden_states.return_value = None
    runner.compute_selected_logits = MagicMock(return_value=torch.zeros((1, 32)))
    with (
        patch.object(runner, "_dp_lockstep_enabled", return_value=True),
        patch.object(runner, "_can_use_greedy_top_tokens", return_value=False),
        patch.object(runner, "_reset_read_offsets_for_new_requests"),
    ):
        assert runner.execute_model(so_run) is None
        runner._execute_dp_mm_encoder.assert_called_once()


def test_sample_tokens_and_spec_verify_chunk():
    runner = _make_bare_runner()
    runner.spec_decode_manager = None
    assert runner.take_draft_token_ids() is None
    runner.spec_decode_manager = MagicMock()
    runner.spec_decode_manager.take_draft_token_ids.return_value = "drafts"
    assert runner.take_draft_token_ids() == "drafts"

    # _get_sampling_generator
    runner.model_config.seed = 42
    gen = runner._get_sampling_generator()
    assert runner._get_sampling_generator() is gen

    # _mamba_state_index_groups & _reset_read_offsets_for_new_requests
    runner.mamba_slot_read_offsets = None
    assert runner._mamba_state_index_groups() is None
    runner._reset_read_offsets_for_new_requests(0, 1)

    runner.mamba_slot_read_offsets = torch.zeros(8, dtype=torch.int32)
    runner._attn_metadata_builder_ctx = SimpleNamespace(
        mamba_state_indices=None, unified_mamba_state_indices=[]
    )
    assert runner._mamba_state_index_groups() == []
    runner._reset_read_offsets_for_new_requests(0, 1)

    runner._attn_metadata_builder_ctx = SimpleNamespace(
        mamba_state_indices=torch.zeros(2, dtype=torch.int32),
        unified_mamba_state_indices=None,
    )
    runner.input_batch = SimpleNamespace(
        req_ids=[None, "r1", "r2"],
        num_reqs=3,
        max_num_logprobs=None,
        all_greedy=True,
        temperature_cpu_tensor=torch.ones(4, dtype=torch.float32),
        top_k_cpu_tensor=torch.tensor([10, 100, 5, 5], dtype=torch.int32),
        top_p_cpu_tensor=torch.ones(4, dtype=torch.float32),
        req_id_to_index={"r1": 0, "r2": 1},
        generators={0: MagicMock(get_offset=lambda: 8)},
        token_ids_cpu=np.zeros((4, 16), dtype=np.int32),
        num_tokens_no_spec=np.zeros(4, dtype=np.int32),
    )
    runner.requests = {
        "r1": SimpleNamespace(num_computed_tokens=0, num_tokens=5, output_token_ids=[])
    }
    with patch.object(tr_mod, "_reset_read_offsets"):
        runner._reset_read_offsets_for_new_requests(0, 3)

    # _sample_spec_verify_chunk: 1) greedy no grammar, 2) greedy with grammar, 3)
    # non-greedy
    runner.drafter = MagicMock(spec=Eagle3Proposer)
    runner.speculative_config = SimpleNamespace(
        num_speculative_tokens=2,
        use_eagle=lambda: True,
        method="eagle3",
        draft_tensor_parallel_size=2,
    )
    runner.rejection_sampler = MagicMock(
        return_value=torch.tensor([[1, 2, -1]], dtype=torch.int32)
    )
    runner.rejection_sampler.synthetic_mode = False
    runner.spec_bonus_and_target_logits = inspect.unwrap(
        TPUModelRunner.spec_bonus_and_target_logits
    ).__get__(runner, TPUModelRunner)
    runner.spec_gather_bonus_and_target_logits = inspect.unwrap(
        TPUModelRunner.spec_gather_bonus_and_target_logits
    ).__get__(runner, TPUModelRunner)
    runner.sample_from_logits_func = inspect.unwrap(
        TPUModelRunner.sample_from_logits
    ).__get__(runner, TPUModelRunner)
    runner.structured_decoding_manager.mask_spec_logits.side_effect = lambda t, b, *a: (
        t,
        b,
    )

    md = SimpleNamespace(
        bonus_logits_indices=torch.tensor([2], dtype=torch.int32),
        target_logits_indices=torch.tensor([0, 1], dtype=torch.int32),
        draft_token_ids=torch.tensor([3, 4], dtype=torch.int32),
        draft_lengths=torch.tensor([2], dtype=torch.int32),
        draft_lengths_cpu=np.array([2], dtype=np.int32),
        segment_ids=torch.tensor([0, 0], dtype=torch.int32),
        group_indices=torch.tensor([0, 1], dtype=torch.int32),
        final_logits_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
    )
    logits = torch.randn((3, 32))
    so = SimpleNamespace(
        scheduled_spec_decode_tokens={"r1": [3, 4]},
        num_scheduled_tokens={"r1": 1, "r2": 1},
        has_structured_output_requests=True,
    )
    out1 = runner._sample_spec_verify_chunk(logits, md, None, so, 0, 1, True, gen)
    assert out1.shape == (1, 3)

    runner.rejection_sampler.synthetic_mode = True
    out2 = runner._sample_spec_verify_chunk(
        logits, md, MagicMock(), so, 0, 1, True, gen
    )
    assert out2.shape == (1, 3)

    out3 = runner._sample_spec_verify_chunk(logits, md, None, so, 0, 1, False, gen)
    assert out3.shape == (1, 3)

    # sample_tokens: max_num_logprobs == -1 raises NotImplementedError
    runner.execute_model_state = ExecuteModelState(
        scheduler_output=so,
        logits_list=[logits],
        pooler_output_list=[],
        num_reqs_list=[1],
        spec_decode_metadata_list=[None],
        draft_chunks=[],
        mamba_state_indices_list=[None],
    )
    runner.input_batch.max_num_logprobs = -1
    with pytest.raises(NotImplementedError, match="full logprobs"):
        runner.sample_tokens(None)

    # sample_tokens: pooling model with tensor, list of [tensor, None], and tuple
    runner.input_batch.max_num_logprobs = None
    runner.input_batch.req_ids = ["r1", "r2"]
    runner.input_batch.num_reqs = 2
    runner.requests = {
        "r1": SimpleNamespace(num_computed_tokens=0, num_tokens=5, output_token_ids=[]),
        "r2": SimpleNamespace(num_computed_tokens=1, num_tokens=1, output_token_ids=[]),
    }
    runner.is_pooling_model = True
    runner.scheduler_config.async_scheduling = False
    runner.execute_model_state = ExecuteModelState(
        scheduler_output=so,
        logits_list=[],
        pooler_output_list=[
            torch.zeros((1, 4)),
            [torch.zeros(4), None],
            (torch.zeros(4),),
            None,
        ],
        num_reqs_list=[2],
        spec_decode_metadata_list=[None],
        draft_chunks=[],
        mamba_state_indices_list=[None],
    )
    with patch.object(tr_mod, "has_kv_transfer_group", return_value=False):
        pool_res = runner.sample_tokens(None)
        assert len(pool_res.pooler_output) == 4

    # sample_tokens: spec step with 2 chunks (chunk 0: md is not None, chunk 1: md is
    # None), logprobs, grammar, non-greedy, async
    runner.is_pooling_model = False
    runner.input_batch.max_num_logprobs = 2
    runner.input_batch.all_greedy = False
    runner.model_config.max_logprobs = 2
    runner.scheduler_config.async_scheduling = True
    runner.drafter = MagicMock(spec=Eagle3Proposer)
    runner.spec_logprob_row_tokens = inspect.unwrap(
        TPUModelRunner.spec_logprob_row_tokens
    ).__get__(runner, TPUModelRunner)
    runner.gather_logprobs = inspect.unwrap(TPUModelRunner.gather_logprobs).__get__(
        runner, TPUModelRunner
    )
    runner.spec_decode_manager.propose_draft_token_ids.return_value = torch.zeros(
        (2, 2), dtype=torch.int32
    )
    runner._assemble_async_spec_substitution = MagicMock(
        return_value=(
            [torch.zeros(6, dtype=torch.int32)],
            {0: 0, 1: 1},
            torch.zeros(2, dtype=torch.int32),
            {0: 2, 1: 0},
        )
    )
    runner._modify_prev_results = MagicMock()
    runner._update_placeholder = MagicMock(return_value={"r2": 1})

    runner.execute_model_state = ExecuteModelState(
        scheduler_output=so,
        logits_list=[logits, torch.randn((1, 32))],
        pooler_output_list=[],
        num_reqs_list=[1, 1],
        spec_decode_metadata_list=[md, None],
        draft_chunks=[],
        mamba_state_indices_list=[None, None],
    )
    with (
        patch.object(torch, "tpu", MagicMock(), create=True),
        patch.object(tr_mod, "has_kv_transfer_group", return_value=False),
    ):
        runner.sample_tokens(MagicMock())
        runner.spec_decode_manager.stage_draft_token_ids_for_host.assert_called_once()

    # Also test spec step chunk 1 (md is None) with all_greedy=True and
    # needs_logprobs=False
    runner.input_batch.all_greedy = True
    runner.input_batch.max_num_logprobs = None
    runner.execute_model_state = ExecuteModelState(
        scheduler_output=so,
        logits_list=[logits, torch.randn((1, 32))],
        pooler_output_list=[],
        num_reqs_list=[1, 1],
        spec_decode_metadata_list=[md, None],
        draft_chunks=[],
        mamba_state_indices_list=[None, None],
    )
    with (
        patch.object(torch, "tpu", MagicMock(), create=True),
        patch.object(tr_mod, "has_kv_transfer_group", return_value=False),
    ):
        runner.sample_tokens(None)

    # Non-spec step with grammar_output, all_greedy=False, sync mode, partial request
    # (r1) + empty valid_tokens (r2)
    runner.scheduler_config.async_scheduling = False
    runner.input_batch.all_greedy = False
    runner.structured_decoding_manager.mask_logits.side_effect = (
        lambda logits, *a: logits
    )
    runner.execute_model_state = ExecuteModelState(
        scheduler_output=so,
        logits_list=[torch.randn((2, 32))],
        pooler_output_list=[],
        num_reqs_list=[2],
        spec_decode_metadata_list=[None],
        draft_chunks=[],
        mamba_state_indices_list=[None],
    )
    fake_async_out = MagicMock()
    fake_async_out.get_output.return_value = SimpleNamespace(sampled_token_ids=[[], []])
    with (
        patch.object(torch, "tpu", MagicMock(), create=True),
        patch.object(tr_mod, "has_kv_transfer_group", return_value=False),
        patch.object(tr_mod, "AsyncTPUModelRunnerOutput", return_value=fake_async_out),
    ):
        runner.sample_tokens(MagicMock())


def test_load_model_and_initialize_kernels():
    runner = _make_bare_runner()
    runner.model_config.hf_config.architectures = ["DeepseekV41ForCausalLM"]
    runner.parallel_config.pipeline_parallel_size = 2
    runner.speculative_config = SimpleNamespace(
        use_eagle=lambda: True,
        method="eagle3",
        draft_tensor_parallel_size=1,
    )
    runner.drafter = MagicMock(spec=Eagle3Proposer)
    runner.drafter.draft_model = torch.nn.Linear(2, 2)
    runner.drafter._draft_attn_layer_names = {"draft_l0"}

    fake_model = torch.nn.Linear(2, 2)
    fake_model.token_lookback_depth = 4  # type: ignore[assignment]
    fake_model.topk_indices_buffer = torch.zeros((4, 2), dtype=torch.int32)  # type: ignore[assignment]
    runner._make_buffer = MagicMock(return_value="lookback_buf")

    fake_loader = MagicMock()
    fake_loader.load_model.return_value = fake_model

    with (
        patch.object(tr_mod, "get_tpu_quantization_config", return_value="qcfg"),
        patch.object(tr_mod, "get_model_loader", return_value=fake_loader),
        patch.object(tr_mod, "validate_online_fp8"),
        patch.object(tr_mod.envs, "TPU_ROPE_CACHE_TRUNCATE", True),
        patch.object(tr_mod.envs, "TPU_ROPE_CACHE_ROW_MAJOR", True),
        patch.object(tr_mod.envs, "TPU_MOE_HASH_TABLE_ROW_MAJOR", True),
        patch.object(tr_mod.envs, "TPU_MOE_SKIP_PADDED_TOKENS", True),
        patch.object(runner, "_truncate_rope_caches") as mock_trunc,
        patch.object(tr_mod.runner_utils, "relayout_rope_caches") as mock_rc,
        patch.object(tr_mod.runner_utils, "relayout_rope_tables") as mock_rt,
        patch.object(tr_mod.runner_utils, "relayout_hash_tables") as mock_ht,
        patch("vllm_torchtpu.model_loader_patches.evict_loaded_weights_page_cache"),
        patch.object(runner, "_initialize_pallas_kernels") as mock_init_pk,
        patch.object(
            tr_mod.token_padding.TokenPaddingState, "create", return_value="pad_state"
        ),
        patch.object(tr_mod.token_padding, "set_padding_state") as mock_set_pad,
    ):
        runner.load_model()
        assert runner.lookback_token_ids == "lookback_buf"
        assert (runner._pp_topk_buffer == -1).all()
        mock_trunc.assert_called_once()
        mock_rc.assert_called_once()
        assert mock_rt.call_count == 2
        mock_ht.assert_called_once()
        mock_init_pk.assert_called_once()
        mock_set_pad.assert_called_once_with("pad_state")

    # _initialize_pallas_kernels, _initialize_quantization_kernels,
    # _initialize_attention_kernels
    class _CustomSubMod(torch.nn.Module):
        def _build_attn_op(self):
            pass

        attn_op = "a"

        def _build_compressor_op(self):
            pass

        compressor_op = "c"

        def _build_indexer_op(self):
            pass

        indexer_op = "i"
        mhc_ops = "m1"
        mhc_post_op = "m2"
        qnorm_rope_op = "q"
        kv_rope_op = "k"
        o_proj_op = "o"

    runner.model = _CustomSubMod()
    runner.drafter.draft_model = _CustomSubMod()

    impl_target = MagicMock(spec=tr_mod.PallasAttentionBackendImpl)
    impl_target.decode_query_size = 1
    impl_draft = MagicMock(spec=tr_mod.PallasAttentionBackendImpl)
    impl_draft.decode_query_size = 4
    layers_dict = {
        "target_l0": SimpleNamespace(impl=impl_target),
        "draft_l0": SimpleNamespace(impl=impl_draft),
        "other_l0": SimpleNamespace(impl="not_pallas"),
    }
    with (
        patch.object(tr_mod, "get_layers_from_vllm_config", return_value=layers_dict),
        patch("vllm_torchtpu.layers.adapter.linear_common._get_quantized_matmul_op"),
        patch(
            "vllm_torchtpu.layers.adapter.linear_common._get_quantized_matmul_fp4_op"
        ),
    ):
        runner._initialize_pallas_kernels()
        assert runner._attention_kernels_initialized is True
        # Early return when already initialized and force=False
        runner._initialize_attention_kernels(force=False)


def test_dummy_run_and_pipeline_chunk_helpers():
    runner = _make_bare_runner()
    runner.uses_mrope = True
    runner.kv_cache_config = SimpleNamespace()
    runner._has_mamba_state = True
    runner.mamba_slot_read_offsets = torch.zeros(8, dtype=torch.int32)
    runner._token_padding_state = MagicMock()
    runner.supports_mm_inputs = True
    runner._pp_is_first = True
    runner._pp_topk_buffer = torch.zeros((16, 2), dtype=torch.int32)
    runner.sequence_layout_planner.prepare_dummy.return_value = SimpleNamespace(
        descriptor=SimpleNamespace(kind=SequenceLayoutKind.ALL, protocol="p", version=1)
    )
    runner._build_attention_metadata = MagicMock(return_value=({}, None))
    runner.maybe_select_dummy_loras = MagicMock(return_value=contextlib.nullcontext())
    runner._get_model_inputs = MagicMock(
        return_value=(torch.zeros(8, dtype=torch.int32), None)
    )

    # 1) through_handoff=True returning IntermediateTensors
    runner._pp_wave = MagicMock()
    runner._pp_wave.start_forward.return_value = tr_mod.IntermediateTensors(
        {"topk_indices": torch.zeros((8, 2), dtype=torch.int32)}
    )
    runner.forward_model = MagicMock(
        return_value=(
            tr_mod.IntermediateTensors({"h": torch.zeros((8, 4))}),
            None,
        )
    )
    with patch.object(tr_mod, "synchronize_tensors"):
        assert runner._dummy_run(8, 8, 4, through_handoff=True) is None

    # 2) through_handoff=True returning Tensor + dp_lockstep=True
    runner._pp_wave.start_forward.return_value = tr_mod.IntermediateTensors(
        {"topk_indices": torch.zeros((8, 2), dtype=torch.int32)}
    )
    runner.forward_model = MagicMock(return_value=(torch.zeros((8, 4)), None))
    runner.compute_selected_logits = MagicMock(return_value=torch.zeros((8, 32)))
    with patch.object(tr_mod, "synchronize_tensors"):
        assert (
            runner._dummy_run(8, 8, 4, dp_lockstep=True, through_handoff=True) is None
        )
        runner.compute_selected_logits.assert_called_once()

    # 3) kv_cache_config is None, _attn_layer_names is None, not _pp_is_first,
    # profile_prefix=4, returning IntermediateTensors
    runner.kv_cache_config = None
    runner._attn_layer_names = None
    runner._pp_is_first = False
    runner._pp_intermediate_tensors = MagicMock(
        return_value=tr_mod.IntermediateTensors(
            {"topk_indices": torch.zeros((8, 2), dtype=torch.int32)}
        )
    )
    runner.forward_model = MagicMock(
        return_value=(
            tr_mod.IntermediateTensors({"h": torch.zeros((8, 4))}),
            None,
        )
    )
    with (
        patch.object(tr_mod, "get_layers_from_vllm_config", return_value={"l0": 1}),
        patch.object(tr_mod, "synchronize_tensors"),
    ):
        ms = runner._dummy_run(8, 8, 4, profile_prefix=4)
        assert ms is not None
        # Call _profile_input again with same shape to hit cache
        cached = runner._profile_input("ids", (runner.max_num_tokens,), torch.int32)
        assert cached.shape == (runner.max_num_tokens,)

    # _attention_kernel_shapes when _attention_kv_cache_group_id is None
    runner._attention_kv_cache_group_id = None
    assert runner._attention_kernel_shapes(True) == (8, 8, 16)

    # _add_pipeline_chunk_buckets when extra is empty
    runner._pp_chunk_granularity = None
    runner.num_tokens_paddings = [8, 16]
    with (
        patch(
            "vllm_torchtpu.core.pp_chunk_scheduler.uses_dynamic_chunks",
            return_value=True,
        ),
        patch("vllm_torchtpu.core.pp_chunks.chunk_granularity", return_value=8),
        patch("vllm_torchtpu.core.pp_chunks.chunk_buckets", return_value=[8, 16]),
    ):
        runner._add_pipeline_chunk_buckets()
        assert runner._pp_chunk_granularity == 8


def test_precompile_and_capture_model():
    runner = _make_bare_runner()
    runner.num_tokens_paddings = [4, 8, 32]
    runner.num_reqs_paddings = [4, 8]
    runner.max_num_reqs = 8
    runner.model_config.get_hidden_size.return_value = 4
    runner.compute_selected_logits = MagicMock(return_value=torch.zeros((4, 32)))
    runner.compute_selected_top_tokens = MagicMock(return_value=torch.zeros(4))
    runner.compute_logits_from_hidden_states = MagicMock(
        return_value=torch.zeros((4, 32))
    )
    runner.compute_top_tokens_from_hidden_states = MagicMock(
        return_value=torch.zeros(4)
    )
    runner.sample_from_logits_func = MagicMock(return_value=torch.zeros((4, 1)))
    runner.gather_logprobs = MagicMock(
        return_value=SimpleNamespace(logprobs=torch.zeros((4, 2)))
    )
    runner.spec_logprob_row_tokens = MagicMock(
        return_value=torch.zeros((4, 1), dtype=torch.int64)
    )
    runner.spec_bonus_and_target_logits = MagicMock(
        return_value=(torch.zeros(4), torch.zeros((4, 32)))
    )
    runner.spec_gather_bonus_and_target_logits = MagicMock(
        return_value=(torch.zeros((4, 32)), torch.zeros((4, 32)))
    )
    runner.structured_decoding_manager = SimpleNamespace(
        structured_decode_arange=torch.arange(32),
        require_structured_out_cpu=torch.zeros((8, 1), dtype=torch.bool),
        grammar_bitmask_cpu=torch.zeros((8, 1), dtype=torch.int32),
        require_structured_out_target_cpu=torch.zeros((8, 1), dtype=torch.bool),
        target_grammar_bitmask_cpu=torch.zeros((8, 1), dtype=torch.int32),
        structured_decode=MagicMock(return_value=torch.zeros((4, 32))),
    )

    with (
        patch.object(tr_mod, "synchronize_tensors"),
        patch.object(runner, "_model_supports_top_tokens", return_value=True),
    ):
        runner._precompile_compute_selected_logits()
        runner._precompile_compute_logits_from_hidden_states()
    with (
        patch.object(tr_mod, "synchronize_tensors"),
        patch.object(runner, "_model_supports_top_tokens", return_value=False),
    ):
        runner._precompile_compute_logits_from_hidden_states()

    with patch.object(tr_mod, "synchronize_tensors"):
        # _precompile_structured_decoding: spec=None vs spec!=None
        runner.speculative_config = None
        runner._precompile_structured_decoding()
        runner.speculative_config = SimpleNamespace(
            num_speculative_tokens=1, use_eagle=lambda: True, method="eagle3"
        )
        runner._precompile_structured_decoding()
        runner._precompile_sample_from_logits()
        runner._precompile_gather_logprobs()

        # _precompile_spec_logprobs: include an unreachable bucket (num_tokens=2,
        # num_reqs=8 with prev_num_reqs=4 >= num_tokens)
        runner.num_tokens_paddings = [2, 4, 8, 32]
        runner.num_reqs_paddings = [4, 8]
        runner._precompile_spec_logprobs()

        # _precompile_substitute_placeholder_token
        runner.scheduler_config.async_scheduling = False
        runner._precompile_substitute_placeholder_token()
        runner.scheduler_config.async_scheduling = True
        with patch.object(
            tr_mod, "_substitute_placeholder_token", return_value=torch.zeros(4)
        ):
            runner._precompile_substitute_placeholder_token()

        # _precompile_rejection_sampler
        runner.speculative_config = None
        runner._precompile_rejection_sampler()
        runner.speculative_config = SimpleNamespace(
            num_speculative_tokens=1, use_eagle=lambda: True, method="eagle3"
        )
        runner.rejection_sampler = MagicMock(return_value=torch.zeros((4, 2)))
        runner.rejection_sampler.synthetic_mode = True
        with patch.object(runner, "_model_supports_top_tokens", return_value=True):
            runner._precompile_rejection_sampler()

    # _precompile_sampling_subgraphs
    runner._pp_is_last = False
    runner._precompile_sampling_subgraphs()
    runner._pp_is_last = True
    runner.sequence_layout_planner.uses_selected_logits_hidden_states = True
    with (
        patch.object(runner, "_precompile_compute_selected_logits"),
        patch.object(
            runner, "_precompile_compute_logits_from_hidden_states"
        ) as mock_cl,
        patch.object(runner, "_precompile_structured_decoding"),
        patch.object(runner, "_precompile_sample_from_logits"),
        patch.object(runner, "_precompile_gather_logprobs"),
    ):
        runner._precompile_sampling_subgraphs()
        mock_cl.assert_called_once()
        runner.sequence_layout_planner.uses_selected_logits_hidden_states = False
        runner._precompile_sampling_subgraphs()

    # capture_model
    runner.parallel_config.pipeline_parallel_size = 2
    runner._pp_wave = None
    runner._pp_intermediate_tensors = MagicMock()
    runner._pp_intermediate_template = {"h": ((4,), torch.float32)}
    runner._pp_topk_buffer = torch.zeros((2, 2), dtype=torch.int32)
    runner.max_num_tokens = 16
    with pytest.raises(RuntimeError, match="fewer than the 16 tokens"):
        runner.capture_model()

    runner._pp_topk_buffer = torch.zeros((16, 2), dtype=torch.int32)
    runner.enforce_eager = True
    with patch.object(tr_mod, "PPWave") as mock_ppwave:
        runner.capture_model()
        mock_ppwave.return_value.warmup.assert_called_once()

    runner.enforce_eager = False
    runner.maybe_setup_dummy_loras = MagicMock(return_value=contextlib.nullcontext())
    runner.drafter = MagicMock(spec=Eagle3Proposer)
    enc_mgr = MagicMock()
    with (
        patch.object(runner, "_precompile_backbone"),
        patch.object(runner, "_precompile_sampling_subgraphs"),
        patch.object(runner, "_precompile_mamba_rollback_helpers"),
        patch.object(tr_mod, "maybe_create_mm_encoder_manager", return_value=enc_mgr),
        patch.object(runner, "_warmup_pcp_mtp_prefill") as mock_pcp_warm,
        patch.object(runner, "_precompile_rejection_sampler") as mock_rej,
        patch.object(runner, "_precompile_spec_logprobs"),
        patch.object(runner, "_warmup_spec_decode") as mock_spec_warm,
    ):
        # 1) PCP + MTP enabled
        runner._pcp_enabled = True
        runner._mtp_enabled = True
        runner.capture_model()
        enc_mgr.precompile_vision_encoder.assert_called_once()
        mock_pcp_warm.assert_called_once()

        # 2) Regular async drafter, no encoder manager
        runner._pcp_enabled = False
        with patch.object(tr_mod, "maybe_create_mm_encoder_manager", return_value=None):
            runner.capture_model()
            mock_rej.assert_called_once()
            mock_spec_warm.assert_called_once()

        # 3) Non-async drafter
        runner.speculative_config = None
        with patch.object(tr_mod, "maybe_create_mm_encoder_manager", return_value=None):
            runner.capture_model()


def test_backbone_mamba_and_spec_warmups():
    runner = _make_bare_runner()
    runner.num_tokens_paddings = [8]
    runner.most_model_len = 32
    runner.num_reqs_most_model_len = 4
    runner.num_blocks_per_most_len_req = 2
    runner._dummy_run = MagicMock()
    runner._pp_wave = MagicMock()

    # _precompile_backbone & _warmup_handoff_forwards
    fake_buckets = MagicMock()
    fake_buckets.needs_retrace.return_value = True
    fake_buckets.refused = set()
    with (
        patch.object(
            tr_mod.shape_variants,
            "warmup",
            return_value=contextlib.nullcontext(fake_buckets),
        ),
        patch.object(tr_mod.shape_variants, "retrace") as mock_retrace,
        patch.object(tr_mod, "synchronize_device"),
    ):
        runner._precompile_backbone()
        mock_retrace.assert_called_once()
        assert runner._dummy_run.call_count == 4

        # Also test _warmup_handoff_forwards with most_model_len = None
        runner.most_model_len = None
        runner._warmup_handoff_forwards()

        # And refused buckets raising ShapeSpecializationError
        fake_buckets.refused = {8}
        with pytest.raises(tr_mod.shape_variants.ShapeSpecializationError):
            runner._precompile_backbone()

    # _precompile_mamba_state_seed_copies & _precompile_mamba_rollback_helpers
    runner._mamba_copy_plan = []
    runner.kv_cache_raw_tensors = []
    runner._precompile_mamba_state_seed_copies()

    t_raw = torch.zeros((4, 4))
    runner._mamba_copy_plan = [(0, [t_raw])]
    runner.kv_cache_raw_tensors = [t_raw]
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[1])
    runner._copy_mamba_state_blocks = MagicMock()
    with patch.object(tr_mod, "synchronize_tensors"):
        runner._precompile_mamba_state_seed_copies()
        assert runner._copy_mamba_state_blocks.called

    runner.mamba_slot_read_offsets = None
    runner._precompile_mamba_rollback_helpers()
    runner.mamba_slot_read_offsets = torch.zeros(16, dtype=torch.int32)
    with (
        patch.object(tr_mod, "_rollback_offsets_seed_compiled") as mock_seed,
        patch.object(tr_mod, "_rollback_offsets_migrate_compiled") as mock_mig,
    ):
        runner._precompile_mamba_rollback_helpers()
        assert mock_seed.called and mock_mig.called

    # _warmup_pcp_mtp_prefill & _warmup_one_pcp_mtp_prefill
    runner.speculative_config = SimpleNamespace(
        num_speculative_tokens=1, use_eagle=lambda: True, method="mtp"
    )
    with patch.object(tr_mod.envs, "SPEC_WARMUP", False):
        runner._warmup_pcp_mtp_prefill()
    runner.enforce_eager = True
    with patch.object(tr_mod.envs, "SPEC_WARMUP", True):
        runner._warmup_pcp_mtp_prefill()
    runner.enforce_eager = False

    with (
        patch.object(tr_mod.envs, "SPEC_WARMUP", True),
        patch.object(runner, "_warmup_one_pcp_mtp_prefill", side_effect=[False, True]),
    ):
        runner._warmup_pcp_mtp_prefill()
    with (
        patch.object(tr_mod.envs, "SPEC_WARMUP", True),
        patch.object(runner, "_warmup_one_pcp_mtp_prefill", side_effect=[False, False]),
        pytest.raises(RuntimeError, match="PCP MTP prefill warmup failed after retry"),
    ):
        runner._warmup_pcp_mtp_prefill()

    # _warmup_one_pcp_mtp_prefill internals
    runner.kv_cache_config = SimpleNamespace(
        num_blocks=2,
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16))],
    )
    with pytest.raises(RuntimeError, match="not enough KV blocks"):
        runner._warmup_one_pcp_mtp_prefill(8)

    runner.kv_cache_config.num_blocks = 10
    runner.execute_model = MagicMock(return_value=None)
    runner.sample_tokens = MagicMock()
    runner.take_draft_token_ids = MagicMock()
    runner._warmup_spec_decode_cleanup = MagicMock(return_value=True)
    assert runner._warmup_one_pcp_mtp_prefill(8) is True

    runner.execute_model = MagicMock(side_effect=RuntimeError("compile fail"))
    assert runner._warmup_one_pcp_mtp_prefill(8, quiet=True) is False
    assert runner._warmup_one_pcp_mtp_prefill(8, quiet=False) is False

    runner._warmup_spec_decode_cleanup = MagicMock(return_value=False)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        runner._warmup_one_pcp_mtp_prefill(8, quiet=True)


def test_warmup_spec_decode_and_cleanup():
    runner = _make_bare_runner()
    with patch.object(tr_mod.envs, "SPEC_WARMUP", False):
        runner._warmup_spec_decode()
    with patch.object(tr_mod.envs, "SPEC_WARMUP", True):
        runner.enforce_eager = True
        runner._warmup_spec_decode()
        runner.enforce_eager = False
        runner.speculative_config = None
        runner._warmup_spec_decode()

        runner.speculative_config = SimpleNamespace(
            num_speculative_tokens=1, use_eagle=lambda: True, method="eagle3"
        )
        runner.input_batch = SimpleNamespace(num_reqs=1, req_id_to_index={})
        runner._warmup_spec_decode()

        runner.input_batch.num_reqs = 0
        runner.max_model_len = 16
        runner.num_tokens_paddings = [8, 32]
        runner.num_reqs_paddings = [1, 2]
        runner.scheduler_config.max_num_seqs = 2
        with (
            patch.object(
                runner, "_warmup_one_spec_request", side_effect=[False, True, True]
            ) as mock_one,
            patch.object(runner, "_warmup_spec_batch") as mock_batch,
        ):
            runner._warmup_spec_decode()
            assert mock_one.call_count == 3
            assert mock_batch.call_count == 2

    # _warmup_spec_batch
    runner.kv_cache_config = SimpleNamespace(
        num_blocks=2,
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16))],
    )
    runner.parallel_config.tensor_parallel_size = 1
    with patch.object(tr_mod.envs, "TPU_DEEPSEEK_V41_KV_SHARD", False):
        runner._warmup_spec_batch(2, 8, greedy=True)  # not enough kv blocks

        runner.kv_cache_config.num_blocks = 20
        runner.execute_model = MagicMock(return_value=None)
        runner.sample_tokens = MagicMock()
        runner.take_draft_token_ids = MagicMock()
        runner._warmup_spec_decode_cleanup = MagicMock(return_value=True)
        runner._warmup_spec_batch(2, 8, greedy=True)

        # kv_shard=True + exception in try
        runner.parallel_config.tensor_parallel_size = 2
        with (
            patch.object(tr_mod.envs, "TPU_DEEPSEEK_V41_KV_SHARD", True),
            patch.object(tr_mod, "_spec_warmup_request_block_ids", return_value=([1],)),
        ):
            runner.execute_model = MagicMock(side_effect=RuntimeError("batch fail"))
            runner._warmup_spec_batch(2, 8, greedy=False)

    # _warmup_one_spec_request
    runner.parallel_config.tensor_parallel_size = 1
    with patch.object(tr_mod.envs, "TPU_DEEPSEEK_V41_KV_SHARD", False):
        runner.kv_cache_config.num_blocks = 1
        assert runner._warmup_one_spec_request(8, 0) is False

        runner.kv_cache_config.num_blocks = 20
        runner.execute_model = MagicMock(return_value=None)
        runner.take_draft_token_ids = MagicMock(
            side_effect=[
                SimpleNamespace(draft_token_ids=[[5]]),
                None,
            ]
            + [SimpleNamespace(draft_token_ids=[[6]])] * 10
        )
        assert runner._warmup_one_spec_request(8, 0, greedy=True) is True

        runner.execute_model = MagicMock(side_effect=RuntimeError("req fail"))
        assert runner._warmup_one_spec_request(8, 0, quiet=True) is False
        assert runner._warmup_one_spec_request(8, 0, quiet=False) is False

    # _warmup_spec_decode_cleanup (restore real method on instance)
    del runner._warmup_spec_decode_cleanup
    pre_res = MagicMock()
    pre_res.wait_for_copy.side_effect = RuntimeError("copy fail")
    runner._pre_async_results = pre_res
    runner.input_batch = SimpleNamespace(req_id_to_index={"r0": 0}, num_reqs=1)
    runner.execute_model = MagicMock(side_effect=RuntimeError("drain fail"))

    class _BadDict(dict):
        def pop(self, *a, **kw):
            raise RuntimeError("pop fail")

    runner.requests = _BadDict({"r0": 1})

    class _BadMgr:
        @property
        def _draft_token_ids(self):
            return 1

        @_draft_token_ids.setter
        def _draft_token_ids(self, val):
            raise RuntimeError("attr fail")

    runner.spec_decode_manager = _BadMgr()

    class _BadDrafter(Eagle3Proposer):
        def __init__(self):
            pass

        @property
        def draft_chunks(self):
            return None

        @draft_chunks.setter
        def draft_chunks(self, val):
            raise RuntimeError("chunk fail")

    runner.drafter = _BadDrafter()
    assert runner._warmup_spec_decode_cleanup("r0") is False


def test_profile_run_and_bottom_methods(tmp_path):
    runner = _make_bare_runner()
    runner.vllm_config.compilation_config.cache_dir = str(tmp_path)
    runner.vllm_config.compilation_config.local_cache_dir = str(tmp_path)
    runner.vllm_config.compilation_config.compile_sizes = [8, 16]
    runner._initialize_attention_kernels = MagicMock()
    runner._dummy_run = MagicMock()
    runner.maybe_setup_dummy_loras = MagicMock(return_value=contextlib.nullcontext())
    runner._precompile_sampling_subgraphs = MagicMock()
    runner.model = torch.nn.Linear(2, 2)

    with (
        patch("vllm.compilation.caching.aot_compile_hash_factors", return_value=["f"]),
        patch(
            "vllm_torchtpu.compilation.tpu_compiler.compute_tpu_compilation_hash",
            return_value="h",
        ),
        patch("vllm.compilation.wrapper.reset_compile_wrapper"),
        patch("vllm_torchtpu.layers.adapter.linear_common._get_quantized_matmul_op"),
        patch(
            "vllm_torchtpu.layers.adapter.linear_common._get_quantized_matmul_fp4_op"
        ),
        patch.object(tr_mod.torch._dynamo, "reset"),
        patch.object(tr_mod, "synchronize_device"),
    ):
        runner.enforce_eager = False
        runner.profile_run(16)
        runner._precompile_sampling_subgraphs.assert_called_once()
        runner.enforce_eager = True
        runner.profile_run(16)

    # _resolve_tpu_group_backend
    layer_mock = MagicMock()
    layer_mock.get_attn_backend.return_value = "custom_backend"
    with patch.object(
        tr_mod, "get_layers_from_vllm_config", return_value={"l0": layer_mock}
    ):
        assert (
            runner._resolve_tpu_group_backend(["l0"], MagicMock()) == "custom_backend"
        )
    with patch.object(tr_mod, "get_layers_from_vllm_config", return_value={}):
        spec_attn = tr_mod.AttentionSpec.__new__(tr_mod.AttentionSpec)
        assert (
            runner._resolve_tpu_group_backend(["l0"], spec_attn)
            is tr_mod.PallasAttentionBackend
        )
        with pytest.raises(NotImplementedError, match="Unsupported KV cache spec"):
            runner._resolve_tpu_group_backend(["l0"], MagicMock())

    # may_reinitialize_input_batch
    bt0 = MagicMock()
    bt0.get_cpu_tensor.return_value = torch.zeros((8, 6), dtype=torch.int32)
    runner.input_batch = SimpleNamespace(block_table=[bt0])
    with patch.object(tr_mod.GPUModelRunner, "may_reinitialize_input_batch"):
        runner.may_reinitialize_input_batch(MagicMock(), [16])
        assert runner.block_table_cpu.shape == (8, 6)
        # Call again when shape already matches
        runner.may_reinitialize_input_batch(MagicMock(), [16])
        # Empty block_table -> catches IndexError
        runner.input_batch = SimpleNamespace(block_table=[])
        runner.may_reinitialize_input_batch(MagicMock(), [16])
        # input_batch is None
        runner.input_batch = None
        runner.may_reinitialize_input_batch(MagicMock(), [16])

    # _token_padding_update
    runner._token_padding_state = None
    runner._token_padding_update(8)
    runner._token_padding_state = MagicMock()
    runner._last_sequence_layout_plan = SimpleNamespace(
        kind=SequenceLayoutKind.ALL, local_num_tokens=4
    )
    runner._token_padding_update(8)
    runner._token_padding_state.update.assert_called_with(4, 8)
    runner._last_sequence_layout_plan = None
    runner._token_padding_update(8)
    runner._token_padding_state.update.assert_called_with(8, 8)

    # forward_model: DeepSeek V4.1 lookback, 1-element list, 2-element tuple
    runner.model_config.hf_config.architectures = ["DeepseekV41ForCausalLM"]
    runner.lookback_token_ids = MagicMock()
    runner._prepare_lookback_token_ids = MagicMock(return_value="lb")
    runner.input_batch = SimpleNamespace(num_reqs=2)
    runner.model = MagicMock(return_value=[torch.zeros((2, 4))])
    out_single, aux_single = runner.forward_model(torch.zeros(2), torch.zeros(2))
    assert out_single.shape == (2, 4) and aux_single is None

    runner.model = MagicMock(return_value=(torch.zeros((2, 4)), [torch.zeros((2, 4))]))
    out_pair, aux_pair = runner.forward_model(torch.zeros(2), torch.zeros(2))
    assert out_pair.shape == (2, 4) and len(aux_pair) == 1

    # _pp_intermediate_tensors with _pp_topk_buffer, _pp_outgoing_tensors, pp_push,
    # pp_settle
    runner._pp_intermediate_template = None
    runner._pp_topk_buffer = torch.zeros((8, 2), dtype=torch.int32)
    runner.model.make_empty_intermediate_tensors.return_value = {
        "h": torch.zeros((1, 4))
    }
    runner.model_config.dtype = torch.float32
    it = runner._pp_intermediate_tensors(2, zeros=True)
    assert (it["topk_indices"] == -1).all()

    out_with_topk = runner._pp_outgoing_tensors({"h": torch.zeros((2, 4))}, 2)
    assert "topk_indices" in out_with_topk
    runner._pp_topk_buffer = None
    assert "topk_indices" not in runner._pp_outgoing_tensors(
        {"h": torch.zeros((2, 4))}, 2
    )

    runner._pp_wave = None
    runner.pp_push()
    runner.pp_settle()
    runner._pp_wave = MagicMock()
    runner.pp_push()
    runner.pp_settle()
    runner._pp_wave.push.assert_called_once()
    runner._pp_wave.settle.assert_called_once()

    # compute_selected_logits when is_pooling_model=True, and gather_logprobs when
    # max_logprobs == 0
    runner.is_pooling_model = True
    csl_fn = inspect.unwrap(TPUModelRunner.compute_selected_logits)
    assert (
        csl_fn(runner, torch.zeros((4, 4)), torch.zeros(2, dtype=torch.int32)).numel()
        == 0
    )

    runner.model_config.max_logprobs = 0
    glp_fn = inspect.unwrap(TPUModelRunner.gather_logprobs)
    glp_out = glp_fn(
        runner, torch.randn((2, 8)), torch.zeros((2, 1), dtype=torch.int64)
    )
    assert glp_out.logprobs.shape == (2, 1)


def test_remaining_helpers_and_xla_graph_branches():
    # Line 494: _build_kv_connector_output when
    # _KV_CONNECTOR_OUTPUT_SUPPORTS_INVALID_BLOCK_GROUP is True
    with (
        patch.object(tr_mod, "_KV_CONNECTOR_OUTPUT_SUPPORTS_INVALID_BLOCK_GROUP", True),
        patch.object(tr_mod, "KVConnectorOutput") as mock_kvc,
    ):
        tr_mod._build_kv_connector_output(
            finished_sending={"a"},
            finished_recving={"b"},
            kv_connector_worker_meta=None,
            invalid_block_ids={1},
            invalid_block_group_index=2,
            kv_connector_stats=None,
        )
        mock_kvc.assert_called_once()

    # Line 565->563: _torch_tpu_wrapper finally when v is None and attribute was already
    # deleted inside body
    fake_tpu = MagicMock()
    with (
        patch.object(torch, "tpu", fake_tpu, create=True),
        patch.object(torch.cuda, "Stream", None),
    ):
        delattr(torch.cuda, "Stream")
        with tr_mod._torch_tpu_wrapper():
            delattr(torch.cuda, "Stream")

    # Line 620: _uniform_placeholder_count returning None
    assert tr_mod._uniform_placeholder_count({1: 2}, 1) is None
    assert tr_mod._uniform_placeholder_count({0: 2}, 2) is None
    assert tr_mod._uniform_placeholder_count({0: 1, 1: 2}, 2) is None

    # Line 1226: maybe_wait_for_kv_save when has_kv_transfer_group() is True
    runner = _make_bare_runner()
    kvt = MagicMock()
    with (
        patch.object(tr_mod, "has_kv_transfer_group", return_value=True),
        patch.object(tr_mod, "get_kv_transfer_group", return_value=kvt),
    ):
        runner.maybe_wait_for_kv_save()
        kvt.wait_for_save.assert_called_once()

    # Line 1437->1439 & 1441: _describe_xla_graph when len(text) <= 120 and when fields
    # is empty
    e_short = SimpleNamespace(frame_id="f0", frame_compile_id=None, spec="short_spec")
    assert "spec=short_spec" in TPUModelRunner._describe_xla_graph(e_short)
    e_empty = SimpleNamespace(frame_id=None, frame_compile_id=None, spec=None)
    assert TPUModelRunner._describe_xla_graph(e_empty)

    # Line 1466: _fresh_xla_graphs when read_count != 1
    runner._xla_graphs_checked_at = None
    entries_list = [
        SimpleNamespace(
            last_read=10,
            read_count=1,
            compile_duration_us=1000,
            frame_id="f1",
            frame_compile_id=0,
            spec="s1",
        ),
        SimpleNamespace(
            last_read=20,
            read_count=2,
            compile_duration_us=2000,
            frame_id="f2",
            frame_compile_id=1,
            spec="s2",
        ),
    ]
    fresh = runner._fresh_xla_graphs(entries_list)
    assert len(fresh) == 1

    # Lines 1482-1534: _update_num_xla_graphs all branches
    runner.check_recompilation = True
    runner.enforce_eager = False
    fake_tpu_mod = MagicMock()
    with patch.object(torch, "tpu", fake_tpu_mod, create=True):
        # 1) Cache shrank: total_graphs < self.num_xla_graphs
        runner.num_xla_graphs = 10
        fake_tpu_mod._get_cache_stats.return_value = SimpleNamespace(
            per_entry_stats=[1, 2, 3], num_cache_reqs=3, num_cache_hits=0
        )
        runner._update_num_xla_graphs("shrink_case")
        assert runner.num_xla_graphs == 3

        # 2) new_compiled_graphs == 0 and not fresh
        runner._xla_graphs_checked_at = 100
        fake_tpu_mod._get_cache_stats.return_value = SimpleNamespace(
            per_entry_stats=[
                SimpleNamespace(last_read=10, read_count=2),
                SimpleNamespace(last_read=20, read_count=2),
                SimpleNamespace(last_read=30, read_count=2),
            ],
            num_cache_reqs=3,
            num_cache_hits=3,
        )
        runner._update_num_xla_graphs("no_change_case")

        # 3) fresh non-empty with len(fresh) > _MAX_DESCRIBED_GRAPHS
        runner._xla_graphs_checked_at = None
        fake_tpu_mod._get_cache_stats.return_value = SimpleNamespace(
            per_entry_stats=[
                SimpleNamespace(
                    last_read=10,
                    read_count=1,
                    compile_duration_us=1000,
                    frame_id="f1",
                    frame_compile_id=0,
                    spec="s1",
                ),
                SimpleNamespace(
                    last_read=20,
                    read_count=1,
                    compile_duration_us=2000,
                    frame_id="f2",
                    frame_compile_id=1,
                    spec="s2",
                ),
                SimpleNamespace(last_read=5, read_count=2),
                SimpleNamespace(last_read=5, read_count=2),
                SimpleNamespace(last_read=5, read_count=2),
            ],
            num_cache_reqs=5,
            num_cache_hits=3,
        )
        with patch.object(TPUModelRunner, "_MAX_DESCRIBED_GRAPHS", 1):
            runner._update_num_xla_graphs("fresh_overflow_case")
        assert runner.num_xla_graphs == 5

        # 4) fresh empty (e.g. read_count=2) but new_compiled_graphs > 0 (1518->1534)
        fake_tpu_mod._get_cache_stats.return_value = SimpleNamespace(
            per_entry_stats=[SimpleNamespace(last_read=1, read_count=2)] * 7,
            num_cache_reqs=7,
            num_cache_hits=7,
        )
        runner._update_num_xla_graphs("no_fresh_but_new_case")
        assert runner.num_xla_graphs == 7


def test_remaining_kv_and_mamba_branches():
    runner = _make_bare_runner()
    # Lines 1669->1672, 1674, 1690: build_attention_metadata_for_layers cache hit and
    # staged_block_tables is None
    b0 = MagicMock()
    b0.kv_cache_group_id = 0
    b0.build.return_value = "built_meta"
    g0 = SimpleNamespace(layer_names=["l0"], metadata_builders=[b0])
    runner.attn_groups = [[g0]]
    runner._attn_metadata_builder_ctx = SimpleNamespace(
        staged_block_tables=None,
    )
    with patch.object(tr_mod, "stage_block_table_uploads") as mock_stage_bt:
        # 1st call: populates _layer_group_selection_cache and calls
        # stage_block_table_uploads
        per_layer = runner.build_attention_metadata_for_layers(
            layer_names=["l0"],
            num_reqs_padded=2,
        )
        assert per_layer["l0"] == "built_meta"
        mock_stage_bt.assert_called_once()
        # 2nd call: hits cached_selection is not None (1669->1672, 1674)
        per_layer2 = runner.build_attention_metadata_for_layers(
            layer_names=["l0"],
            num_reqs_padded=2,
        )
        assert per_layer2["l0"] == "built_meta"

    # Lines 1710-1711, 1714-1715: get_kv_cache_spec & initialize_kv_cache
    with patch.object(
        tr_mod, "set_current_vllm_config", return_value=contextlib.nullcontext()
    ):
        runner.get_kv_cache_spec()
        runner.kv_cache_manager.get_kv_cache_spec.assert_called_once()
        runner.initialize_kv_cache("cfg")
        runner.kv_cache_manager.initialize_kv_cache.assert_called_once_with("cfg")

    # Lines 1823, 1829-1834, 1841-1842, 1856-1863: _build_mamba_copy_plan
    runner.cache_config.mamba_cache_mode = "align"
    mamba_spec = MagicMock(spec=tr_mod.MambaSpec)
    mamba_spec.block_size = 4
    mamba_spec.page_size_bytes = 64
    other_spec = MagicMock()
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=other_spec, layer_names=["attn0"]),
            SimpleNamespace(kv_cache_spec=mamba_spec, layer_names=[]),
            SimpleNamespace(
                kv_cache_spec=mamba_spec, layer_names=["m0", "m0_dup", "m_missing"]
            ),
        ]
    )
    t_1d = torch.zeros(16, dtype=torch.float32)
    t_2d = torch.zeros((4, 4), dtype=torch.float32)
    layer_to_raw = {"m0": t_1d, "m0_dup": t_1d, "m_2d": t_2d}

    # 1) bm_pool_layout is not None (lines 1841-1842)
    bm_layout = SimpleNamespace(num_pools=2, split=3)
    runner._build_mamba_copy_plan(runner.kv_cache_config, layer_to_raw, bm_layout)
    assert len(runner._mamba_copy_plan) == 1 and runner._pool_block_split == 1

    # 2) bm_pool_layout is None (lines 1844->1843, 1856-1863)
    runner._pool_block_split = 1
    runner._build_mamba_copy_plan(runner.kv_cache_config, layer_to_raw, None)
    assert len(runner._mamba_copy_plan) == 1
    # Also test when all raw tensors in layer_to_raw have dim() == 1 (1843->exit)
    runner._build_mamba_copy_plan(runner.kv_cache_config, {"m0": t_1d}, None)

    # Lines 1884, 1973, 1991, 2009, 2026->2040: _collect_mamba_state_seed_copies
    runner._mamba_copy_plan = [(0, [t_2d])]
    runner._mamba_state_block_size = 4
    runner.cache_config.block_size = 4
    runner._mamba_state_pos = {"dead_req": 0, "r0": 0, "r1": 0}
    bt_col = MagicMock()
    bt_col.get_cpu_tensor.return_value = torch.tensor(
        [[5, 5], [0, 6]], dtype=torch.int32
    )
    runner.input_batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["r0", "r1"],
        req_id_to_index={"r0": 0, "r1": 1},
        num_computed_tokens_cpu=np.array([3, 3], dtype=np.int32),
        block_table=[bt_col],
    )
    so = SimpleNamespace(num_scheduled_tokens={"r0": 2, "r1": 2})
    # When src == dst (5 == 5) and src == 0 (0 -> 6), pairs is empty -> hits 1884, 1973,
    # 1991
    runner._collect_mamba_state_seed_copies(so, 0, 2)
    assert "dead_req" not in runner._mamba_state_pos

    # Now valid crossing (5 -> 6) with _pool_block_split = 2 and mamba_slot_read_offsets
    # = None (hits 2009, 2026->2040)
    runner._mamba_state_pos = {"r0": 0}
    bt_col.get_cpu_tensor.return_value = torch.tensor([[5, 6]], dtype=torch.int32)
    runner.input_batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["r0"],
        req_id_to_index={"r0": 0},
        num_computed_tokens_cpu=np.array([3], dtype=np.int32),
        block_table=[bt_col],
    )
    runner._pool_block_split = 2
    runner.mamba_slot_read_offsets = None
    runner._pending_mamba_state_copies = []
    runner._collect_mamba_state_seed_copies(
        SimpleNamespace(num_scheduled_tokens={"r0": 2}), 0, 1
    )
    assert len(runner._pending_mamba_state_copies) == 1

    # Lines 2095-2096: _expand_pool_split when _pool_block_split > 1
    exp_s, exp_d = runner._expand_pool_split(
        torch.tensor([3], dtype=torch.int32),
        torch.tensor([4], dtype=torch.int32),
    )
    assert exp_s.tolist() == [6, 7] and exp_d.tolist() == [8, 9]

    # Lines 2137-2139, 2144-2149: _flush_mamba_state_seed_copies &
    # _copy_mamba_state_blocks
    src_t = torch.tensor([1, 0], dtype=torch.int32)
    dst_t = torch.tensor([2, 0], dtype=torch.int32)
    raw0 = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    raw1 = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    runner._pending_mamba_state_copies = [([raw0, raw1], src_t, dst_t)]
    runner.bm_pool_layout = None
    with (
        patch.object(tr_mod, "copy_mamba_state_blocks") as mock_cmsb,
        patch.object(tr_mod, "copy_mamba_state_rows") as mock_cmsr,
    ):
        runner._flush_mamba_state_seed_copies()
        mock_cmsb.assert_called_once()

        # Also test _copy_mamba_state_blocks when bm_pool_layout is not None
        runner.bm_pool_layout = SimpleNamespace(num_pools=1, split=1)
        runner._copy_mamba_state_blocks([raw0], src_t, dst_t)
        mock_cmsr.assert_called_once()
    runner.bm_pool_layout = None
    runner._pool_block_split = 1

    # Lines 2163-2196: _apply_kv_cache_block_copies (both with
    # mamba_slot_read_offsets=None and Tensor)
    runner.kv_cache_raw_tensors = [raw0]
    copy_item = SimpleNamespace(src_block_id=1, dst_block_id=3)
    with (
        patch.object(tr_mod, "copy_mamba_state_blocks") as mock_cmsb2,
        patch.object(tr_mod, "_rollback_offsets_migrate") as mock_rom,
    ):
        runner.mamba_slot_read_offsets = None
        runner._apply_kv_cache_block_copies([copy_item])
        mock_cmsb2.assert_called_once()

        runner.mamba_slot_read_offsets = torch.zeros(8, dtype=torch.int32)
        runner._apply_kv_cache_block_copies([copy_item])
        mock_rom.assert_called_once()

    # Lines 2240, 2256-2270: _dense_kv_copy_pool_sets duplicate tensor &
    # _apply_dense_kv_cache_block_copies
    t_pool = torch.zeros((4, 4))
    runner.kv_cache_config = SimpleNamespace(num_blocks=2)
    runner.kv_caches = [t_pool, t_pool]
    pools = runner._dense_kv_copy_pool_sets()
    assert len(pools) == 1

    with patch.object(tr_mod, "copy_pool_blocks") as mock_cpb:
        runner._apply_dense_kv_cache_block_copies([copy_item])
        mock_cpb.assert_called_once()


def test_remaining_async_sub_and_batch_state_branches():
    runner = _make_bare_runner()
    # Lines 2322, 2388-2393, 2443: _prepare_async_token_substitution_indices &
    # _apply_async_token_substitution
    runner.input_batch = SimpleNamespace(
        req_id_to_index={"r0": 0, "r1": 1},
        req_ids=["r0", "r1"],
    )
    runner._pre_async_results = SimpleNamespace(
        req_ids=["r0", "r1"],
        req_id_to_index_copy={"r0": 0, "r1": 1},
        next_token_indices=tr_mod._cached_identity_map(2),
        num_draft_per_req=None,
        discard_req_indices=set(),
        spec_decode_num_rejected_tokens=None,
    )
    runner._last_async_seq_idx_prefix = np.array([0, 1], dtype=np.int32)
    runner._last_sequence_layout_plan = SimpleNamespace(
        kind=SequenceLayoutKind.ALL,
        local_index_for_request_major_token=lambda x: x,
    )
    cur_idx, pre_idx = runner._prepare_async_token_substitution_indices(
        0, 2, np.array([1, 1], dtype=np.int32)
    )
    assert list(cur_idx) == [0, 1] and list(pre_idx) == [0, 1]

    # Non-ALL layout where local_index_for_request_major_token returns None for 0 and 0
    # for 1 (lines 2388-2393)
    runner._last_sequence_layout_plan = SimpleNamespace(
        kind=MagicMock(),
        local_index_for_request_major_token=lambda x: None if x == 0 else 0,
    )
    cur_pcp, pre_pcp = runner._prepare_async_token_substitution_indices(
        0, 2, np.array([1, 1], dtype=np.int32)
    )
    assert list(cur_pcp) == [0] and list(pre_pcp) == [1]

    # _apply_async_token_substitution with _fast_token_substitution = True (line 2443)
    # and fallthrough
    runner._fast_token_substitution = True
    runner.arange_np = np.arange(16, dtype=np.int64)
    input_ids_tpu = torch.tensor([10, 20], dtype=torch.int32)
    next_tokens_tpu = torch.tensor([99, 88, 77], dtype=torch.int64)
    runner._pre_async_results = SimpleNamespace(next_tokens_tpu=next_tokens_tpu)
    fast_out = runner._apply_async_token_substitution(
        input_ids_tpu,
        np.array([0, 1], dtype=np.int32),
        np.array([0, 1], dtype=np.int32),
    )
    assert torch.equal(fast_out, torch.tensor([99, 88], dtype=torch.int32))

    # Fallthrough when arange_np is None or indices don't match identity
    runner.arange_np = None
    with patch.object(
        tr_mod,
        "_substitute_placeholder_token",
        side_effect=inspect.unwrap(tr_mod._substitute_placeholder_token),
    ):
        slow_out = runner._apply_async_token_substitution(
            input_ids_tpu,
            np.array([1], dtype=np.int32),
            np.array([0], dtype=np.int32),
        )
        assert slow_out.shape == (2,)
    runner.arange_np = np.arange(16, dtype=np.int64)

    # Lines 2533, 2580-2608: _install_spec_token_room_guard with list block_tables
    runner.input_batch = SimpleNamespace(_tpu_spec_room_guard=True)
    runner._install_spec_token_room_guard()  # early return line 2533

    rewrote_calls = []

    class _FakeSubBT:
        def __init__(self, with_cpu_np: bool, with_rewrote: bool = True):
            self.block_size = 4
            self.max_num_blocks_per_req = 4
            self.num_blocks_per_row = np.zeros(4, dtype=np.int32)
            arr = np.zeros((4, 4), dtype=np.int32)
            if with_cpu_np:
                self._bt_cpu_np = arr
                self.block_table = SimpleNamespace(np=None)
            else:
                self.block_table = SimpleNamespace(np=arr)
            if with_rewrote:
                self._rewrote = lambda idx: rewrote_calls.append(idx)

        def append_row(self, bids, row_idx):
            self.num_blocks_per_row[row_idx] += len(bids)

        def swap_row(self, i1, i2):
            pass

    bt_with_np = _FakeSubBT(with_cpu_np=True, with_rewrote=True)
    bt_no_np = _FakeSubBT(with_cpu_np=False, with_rewrote=False)

    class _FakeMultiBlockTable:
        def __init__(self):
            self.block_tables = [bt_with_np, bt_no_np]

    mbt = _FakeMultiBlockTable()
    runner.input_batch = SimpleNamespace(
        token_ids_cpu=np.zeros((4, 16), dtype=np.int32),
        num_tokens=np.zeros(4, dtype=np.int32),
        num_tokens_no_spec=np.zeros(4, dtype=np.int32),
        num_computed_tokens_cpu=np.zeros(4, dtype=np.int32),
        update_req_spec_token_ids=lambda *a: None,
        block_table=mbt,
    )
    runner.speculative_config = SimpleNamespace(num_speculative_tokens=1)
    runner._install_spec_token_room_guard()
    mbt.append_row([[], []], 0)
    mbt.append_row([[10, 11], [12]], 0)
    assert bt_with_np.num_blocks_per_row[0] == 2
    mbt.swap_row(0, 0)
    mbt.swap_row(0, 1)
    assert bt_with_np.num_blocks_per_row[1] == 2
    assert len(rewrote_calls) >= 2

    # Line 2612: _may_reorder_batch
    assert runner._may_reorder_batch(MagicMock()) is None

    # Lines 2615-2662: _update_states
    import dataclasses

    @dataclasses.dataclass
    class _FakeSOUpdate:
        finished_req_ids: set
        preempted_req_ids: set
        scheduled_cached_reqs: Any
        kv_cache_block_copies: Any

    runner._mamba_state_pos = {"fin_r": 1, "pre_r": 2, "res_r": 3}
    runner._mamba_offset_seeded = {"fin_r", "pre_r", "res_r"}
    runner._pre_async_results = None
    runner.requests = {"r0": SimpleNamespace(prev_num_draft_len=2)}
    so_up = _FakeSOUpdate(
        finished_req_ids={"fin_r"},
        preempted_req_ids={"pre_r"},
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids={"res_r"}),
        kv_cache_block_copies=[SimpleNamespace(src_block_id=1, dst_block_id=2)],
    )
    runner.kv_cache_raw_tensors = [torch.zeros((4, 4))]
    runner._apply_kv_cache_block_copies = MagicMock()
    runner._apply_dense_kv_cache_block_copies = MagicMock()
    with patch.object(tr_mod.GPUModelRunner, "_update_states", return_value=True):
        # 1) use_page_local_copy=True
        assert runner._update_states(so_up) is True
        runner._apply_kv_cache_block_copies.assert_called_once()

        # 2) use_dense_copy=True and _pre_async_results not None
        # withfinished/preempted/resumed
        runner.kv_cache_raw_tensors = []
        runner._dense_kv_copy_supported = MagicMock(return_value=True)
        runner._pre_async_results = SimpleNamespace(
            req_id_to_index_copy={"fin_r": 0, "keep_r": 1},
            request_seq_lens=[(0, None, 2, "fin_r"), (1, None, 2, "keep_r")],
        )
        so_up2 = _FakeSOUpdate(
            finished_req_ids={"fin_r"},
            preempted_req_ids=set(),
            scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
            kv_cache_block_copies=[SimpleNamespace(src_block_id=1, dst_block_id=2)],
        )
        runner._update_states(so_up2)
        assert "fin_r" not in runner._pre_async_results.req_id_to_index_copy
        assert len(runner._pre_async_results.request_seq_lens) == 1
        runner._apply_dense_kv_cache_block_copies.assert_called_once()

        # 3) neither page_local nor dense copy, and empty finished/preempted/resumed
        runner._dense_kv_copy_supported = MagicMock(return_value=False)
        so_up3 = _FakeSOUpdate(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
            kv_cache_block_copies=[],
        )
        runner._update_states(so_up3)

    # Lines 2750-2763, 2775-2776: _modify_prev_results vectorized non-uniform
    # pre_num_draft_per_req & arange_np
    r0_state = SimpleNamespace(req_id="r0", output_token_ids=[1, 2])
    r1_state = SimpleNamespace(req_id="r1", output_token_ids=[3, 4, 5])
    runner.requests = {"r0": r0_state, "r1": r1_state}
    runner._pre_async_results = SimpleNamespace(
        req_ids=["r0", "r1"],
        request_seq_lens=[(0, r0_state, 5, "r0"), (1, r1_state, 6, "r1")],
        discard_sampled_tokens_req_indices=set(),
        num_draft_per_req={0: 1, 1: 2},
        copy_state=None,
        wait_for_copy=lambda: torch.tensor(
            [[10, 11, -1], [20, 21, 22]], dtype=torch.int32
        ),
    )
    runner.input_batch = SimpleNamespace(
        req_ids=["r0", "r1"],
        req_id_to_index={"r0": 0, "r1": 1},
        token_ids_cpu=np.zeros((4, 16), dtype=np.int32),
        num_tokens=np.array([5, 6, 0, 0], dtype=np.int32),
        num_tokens_no_spec=np.array([5, 6, 0, 0], dtype=np.int32),
        num_accepted_tokens_cpu=np.zeros(4, dtype=np.int64),
    )
    runner.arange_np = np.arange(16, dtype=np.int64)
    runner._modify_prev_results()
    assert r0_state.output_token_ids == [10, 11]
    assert r1_state.output_token_ids == [20, 21, 22]

    # Also test when arange_np is None on uniform vectorized path (lines 2778-2779)
    runner.arange_np = None
    r0_state.output_token_ids = [1]
    r1_state.output_token_ids = [2]
    runner._pre_async_results.num_draft_per_req = None
    runner._pre_async_results.wait_for_copy = lambda: torch.tensor(
        [[10], [20]], dtype=torch.int32
    )
    runner._modify_prev_results()
    runner.arange_np = np.arange(16, dtype=np.int64)

    # Lines 2839, 2862-2865, 2873-2890, 2894: _update_placeholder
    assert runner._update_placeholder([], [], {}) == {}
    # 1) Uniform fast-path with _cached_identity_map (2862-2865)
    runner.input_batch = SimpleNamespace(
        req_id_to_index={"r0": 0, "r1": 1},
        token_ids_cpu=np.zeros((4, 16), dtype=np.int32),
        num_tokens=np.array([3, 4, 0, 0], dtype=np.int32),
        num_tokens_no_spec=np.array([3, 4, 0, 0], dtype=np.int32),
    )
    r0_up = SimpleNamespace(req_id="r0", output_token_ids=[])
    r1_up = SimpleNamespace(req_id="r1", output_token_ids=[])
    seq_lens_4 = [(0, r0_up, 3, "r0"), (1, r1_up, 4, "r1")]
    res_id = runner._update_placeholder(
        discard_sampled_tokens_req_indices=[],
        request_seq_lens=seq_lens_4,
        next_token_indices=tr_mod._cached_identity_map(2),
        num_draft_per_req=None,
    )
    assert res_id == {"r0": 0, "r1": 1}

    # 2) Non-uniform fast-path (2873-2890)
    res_nonunif = runner._update_placeholder(
        discard_sampled_tokens_req_indices=[],
        request_seq_lens=seq_lens_4,
        next_token_indices={0: 0, 1: 1},
        num_draft_per_req={0: 1, 1: 2},
    )
    assert res_nonunif == {"r0": 0, "r1": 1}

    # 3) Slow path with discard_sampled_tokens_req_indices (2894)
    res_disc = runner._update_placeholder(
        discard_sampled_tokens_req_indices=[0],
        request_seq_lens=seq_lens_4,
        next_token_indices={0: 0, 1: 1},
        num_draft_per_req={0: 1, 1: 2},
    )
    assert res_disc == {"r1": 1}


def test_remaining_prepare_execute_sample_load_dummy_and_bottom():
    runner = _make_bare_runner()
    runner._record_input_staging_fence = MagicMock()
    # _prepare_inputs remaining branches:
    # 1) num_windowed_reqs > 0 (3088->3090), len > num_reqs_max_model_len (3120-3123),
    #    _pcp_enabled & _mtp_enabled (3269, 3297-3305),
    #    speculative_config with 0 drafts and >0 drafts (3423-3435, 3497),
    #    layout_plan.logits_indices_cpu not None (3499),
    #    mamba_slot_read_offsets not None with _fast_token_substitution=False (3565)
    runner.num_reqs_max_model_len = 1
    runner.most_model_len = None
    runner._pcp_enabled = True
    runner._mtp_enabled = True
    runner.speculative_config = SimpleNamespace(
        num_speculative_tokens=1, use_eagle=lambda: True, method="mtp"
    )
    runner.mamba_slot_read_offsets = torch.zeros(8, dtype=torch.int32)
    runner._fast_token_substitution = False
    bt0 = MagicMock()
    bt0.max_num_blocks_per_req = 4
    bt0.get_cpu_tensor.return_value = torch.zeros((8, 4), dtype=torch.int32)
    runner.input_batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["r0", "r1"],
        req_id_to_index={"r0": 0, "r1": 1},
        num_computed_tokens_cpu=np.array([2, 0], dtype=np.int32),
        num_tokens=np.array([4, 2], dtype=np.int32),
        num_tokens_no_spec=np.array([4, 2], dtype=np.int32),
        num_prompt_tokens=np.array([4, 2], dtype=np.int32),
        token_ids_cpu=np.zeros((8, 16), dtype=np.int32),
        token_ids_cpu_tensor=torch.zeros((8, 16), dtype=torch.int32),
        block_table=[bt0],
        mamba_state_idx=np.array([1, 2], dtype=np.int32),
    )
    runner.requests = {
        "r0": SimpleNamespace(num_computed_tokens=2, num_tokens=4, output_token_ids=[]),
        "r1": SimpleNamespace(num_computed_tokens=0, num_tokens=2, output_token_ids=[]),
    }
    non_all_kind = MagicMock()
    fake_plan = SimpleNamespace(
        kind=non_all_kind,
        descriptor=SimpleNamespace(
            kind=non_all_kind, protocol="p", version=1, cache_key="k"
        ),
        global_num_tokens=2,
        global_padded_num_tokens=8,
        local_num_tokens=2,
        local_padded_num_tokens=8,
        token_slice=slice(0, 8),
        input_ids_cpu=torch.zeros(8, dtype=torch.int32),
        positions_cpu=torch.zeros(8, dtype=torch.int32),
        query_start_loc_cpu=(0, 2),
        seq_lens_cpu=(4,),
        max_context_len_cpu=4,
        logits_indices_cpu=torch.tensor([1], dtype=torch.int32),
        num_spec_tokens_override=None,
    )
    runner.sequence_layout_planner.prepare_real.return_value = fake_plan
    runner._build_attention_metadata = MagicMock(return_value=({"l0": "m"}, "common"))
    runner.spec_decode_manager.get_spec_decode_metadata.return_value = SimpleNamespace(
        final_logits_indices=torch.tensor([0, 1], dtype=torch.int32)
    )
    so_prep_spec = SimpleNamespace(
        total_num_scheduled_tokens=4,
        num_scheduled_tokens={"r0": 2, "r1": 2},
        scheduled_spec_decode_tokens={"r0": [5]},
    )
    runner._prepare_inputs(
        so_prep_spec, start_index=0, num_decode_reqs=0, num_windowed_reqs=1
    )

    so_prep_no_spec = SimpleNamespace(
        total_num_scheduled_tokens=4,
        num_scheduled_tokens={"r0": 2, "r1": 2},
        scheduled_spec_decode_tokens={},
    )
    runner._prepare_inputs(
        so_prep_no_spec, start_index=0, num_decode_reqs=0, num_windowed_reqs=1
    )

    # execute_model remaining branches:
    runner._pcp_enabled = True
    runner._mtp_enabled = True
    runner.scheduler_config.async_scheduling = False
    runner.phase_based_profiler = None
    runner.supports_mm_inputs = True
    runner._pp_is_first = True
    runner._pp_is_last = True
    runner.is_pooling_model = False
    runner.mamba_slot_read_offsets = None
    runner.requests = {
        "r0": SimpleNamespace(num_computed_tokens=0, num_tokens=4, output_token_ids=[])
    }
    runner.input_batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["r0"],
        req_id_to_index={"r0": 0},
        num_computed_tokens_cpu=np.array([0], dtype=np.int32),
        num_prompt_tokens=np.array([4], dtype=np.int32),
        all_greedy=True,
        max_num_logprobs=None,
    )
    so_exec = SimpleNamespace(
        total_num_scheduled_tokens=2,
        num_scheduled_tokens={"r0": 2},
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_cached_reqs=None,
        kv_cache_block_to_copy=[],
        blocks_to_reset_read_offset=[],
        has_structured_output_requests=False,
    )
    runner._update_states = MagicMock(return_value=False)
    runner._reorder_batch_for_rpa = MagicMock(return_value=(0, 0))
    runner._count_input_chunks = MagicMock(return_value=(1, 1))
    runner._dp_coordinated_step = MagicMock(return_value=(8, 1))

    def fake_prep_exec(*args, **kwargs):
        runner.input_ids = torch.zeros(8, dtype=torch.int32)
        runner.position_ids = torch.zeros(8, dtype=torch.int32)
        runner._last_sequence_layout_plan = fake_plan
        runner._attn_metadata_builder_ctx = SimpleNamespace(
            mamba_state_indices=None,
            unified_mamba_state_indices=None,
        )
        return (
            {"l0": "m"},
            torch.zeros(1, dtype=torch.int32),
            8,
            1,
            1,
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            None,
            None,
            fake_plan,
        )

    runner._prepare_inputs = fake_prep_exec
    runner._execute_mm_encoder = MagicMock()
    runner._gather_mm_embeddings = MagicMock(
        return_value=([torch.zeros((0, 4))], torch.zeros(8, dtype=torch.bool))
    )
    runner._get_model_inputs = MagicMock(return_value=(None, torch.zeros((8, 4))))
    runner.forward_model = MagicMock(return_value=(torch.zeros((8, 4)), None))
    runner.sequence_layout_planner.uses_selected_logits_hidden_states = False
    runner.sequence_layout_planner.maybe_select_logits_hidden_states.return_value = None
    runner.sequence_layout_planner.finalize_hidden_states.side_effect = lambda h, p: h
    runner._can_use_greedy_top_tokens = MagicMock(return_value=True)
    runner.compute_selected_top_tokens = MagicMock(
        return_value=torch.zeros(1, dtype=torch.int32)
    )
    with (
        patch.object(tr_mod, "synchronize_tensors"),
        patch.object(
            tr_mod.runner_utils,
            "get_batch_composition_stats",
            return_value={
                "num_prefill_tokens": 2,
                "num_decode_tokens": 0,
                "phase": "prefill",
                "batch_id": 1,
                "total_num_scheduled_tokens": 2,
                "padded_total_num_scheduled_tokens": 8,
            },
        ),
        patch.object(tr_mod.TraceAnnotation, "is_enabled", return_value=True),
        patch.object(tr_mod, "TraceAnnotation", return_value=contextlib.nullcontext()),
        patch.object(
            tr_mod, "set_forward_context", return_value=contextlib.nullcontext()
        ),
        patch.object(
            tr_mod,
            "set_vllm_model_wrapper_context",
            return_value=contextlib.nullcontext(),
        ),
        patch.object(runner, "sample_tokens", return_value="sync_out"),
    ):
        runner.execute_model(so_exec)
        runner.compute_selected_top_tokens.assert_called_once()

    # Also test sample_hidden_states is not None with use_top_tokens=True and False
    runner.execute_model_state = None
    runner.sequence_layout_planner.uses_selected_logits_hidden_states = True
    runner.sequence_layout_planner.maybe_select_logits_hidden_states.return_value = (
        torch.zeros((1, 4))
    )

    def fake_prep_exec_shs(*args, **kwargs):
        res = list(fake_prep_exec(*args, **kwargs))
        res[8] = torch.zeros((1, 4))
        return tuple(res)

    runner._prepare_inputs = fake_prep_exec_shs
    runner._gather_mm_embeddings = MagicMock(return_value=([], None))
    runner.compute_top_tokens_from_hidden_states = MagicMock(
        return_value=torch.zeros(1, dtype=torch.int32)
    )
    runner.compute_logits_from_hidden_states = MagicMock(
        return_value=torch.zeros((1, 32))
    )
    with (
        patch.object(tr_mod, "synchronize_tensors"),
        patch.object(tr_mod.TraceAnnotation, "is_enabled", return_value=False),
        patch.object(
            tr_mod, "set_forward_context", return_value=contextlib.nullcontext()
        ),
        patch.object(
            tr_mod,
            "set_vllm_model_wrapper_context",
            return_value=contextlib.nullcontext(),
        ),
        patch.object(runner, "sample_tokens", return_value="sync_out"),
    ):
        runner.execute_model(so_exec)
        runner.compute_top_tokens_from_hidden_states.assert_called_once()

        runner.execute_model_state = None
        runner.speculative_config = None
        runner._can_use_greedy_top_tokens = MagicMock(return_value=False)
        runner.execute_model(so_exec)
        runner.compute_logits_from_hidden_states.assert_called_once()
    runner.sequence_layout_planner.uses_selected_logits_hidden_states = False

    # Line 4561: _greedy_sample with 1D tensor
    assert torch.equal(
        runner._greedy_sample(torch.tensor([3, 4], dtype=torch.int32)),
        torch.tensor([[3], [4]], dtype=torch.int32),
    )

    # sample_tokens remaining branches:
    # non-spec step with all_greedy=True, async_scheduling=True, needs_logprobs=True,
    # 2 chunks (len(next_tokens_tpu_chunks) > 1 -> line 5113)
    runner.scheduler_config.async_scheduling = True
    runner.speculative_config = None
    runner.drafter = None
    runner.input_batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["r0", "r1"],
        req_id_to_index={"r0": 0, "r1": 1},
        all_greedy=True,
        max_num_logprobs=2,
        vocab_size=32,
        generators={},
        num_accepted_tokens_cpu=np.zeros(4, dtype=np.int64),
        token_ids_cpu=np.zeros((4, 16), dtype=np.int32),
        num_tokens=np.array([2, 2, 0, 0], dtype=np.int32),
        num_tokens_no_spec=np.array([2, 2, 0, 0], dtype=np.int32),
    )
    runner.requests = {
        "r0": SimpleNamespace(num_computed_tokens=0, num_tokens=2, output_token_ids=[]),
        "r1": SimpleNamespace(num_computed_tokens=0, num_tokens=2, output_token_ids=[]),
    }
    runner.gather_logprobs = inspect.unwrap(TPUModelRunner.gather_logprobs).__get__(
        runner, TPUModelRunner
    )
    runner.sample_from_logits_func = inspect.unwrap(
        TPUModelRunner.sample_from_logits
    ).__get__(runner, TPUModelRunner)
    runner.model_config.max_logprobs = 2
    runner._modify_prev_results = MagicMock()
    runner._update_placeholder = MagicMock(return_value={})
    so_samp = SimpleNamespace(
        total_num_scheduled_tokens=4,
        num_scheduled_tokens={"r0": 2, "r1": 2},
        scheduled_spec_decode_tokens={},
        finished_sending=None,
        finished_recving=None,
    )
    runner.execute_model_state = ExecuteModelState(
        scheduler_output=so_samp,
        logits_list=[torch.randn((1, 32)), torch.randn((1, 32))],
        pooler_output_list=[],
        num_reqs_list=[1, 1],
        spec_decode_metadata_list=[None, None],
        draft_chunks=[],
        mamba_state_indices_list=[None, None],
    )
    with (
        patch.object(torch, "tpu", MagicMock(), create=True),
        patch.object(tr_mod, "has_kv_transfer_group", return_value=False),
    ):
        runner.sample_tokens(None)
        assert runner._pre_async_results is not None
        assert runner._pre_async_results.next_tokens_tpu.shape[0] == 2

    # sample_tokens with num_reqs_list=[] -> next_tokens_tpu_chunks empty ->
    # _pre_async_results = None (line 5147)
    runner.execute_model_state = ExecuteModelState(
        scheduler_output=so_samp,
        logits_list=[],
        pooler_output_list=[],
        num_reqs_list=[],
        spec_decode_metadata_list=[],
        draft_chunks=[],
        mamba_state_indices_list=[],
    )
    with (
        patch.object(torch, "tpu", MagicMock(), create=True),
        patch.object(tr_mod, "has_kv_transfer_group", return_value=False),
    ):
        runner.sample_tokens(None)
        assert runner._pre_async_results is None

    # sample_tokens sync mode with valid_tokens non-empty, partial req with
    # generator=None, and ngram propose (lines 4921->4926, 4935-4951, 5191-5201, 5207)
    runner.scheduler_config.async_scheduling = False
    runner.speculative_config = SimpleNamespace(
        num_speculative_tokens=1, use_eagle=lambda: False, method="ngram"
    )
    runner.requests = {
        "r0": SimpleNamespace(num_computed_tokens=0, num_tokens=5, output_token_ids=[]),
        "r1": SimpleNamespace(num_computed_tokens=0, num_tokens=2, output_token_ids=[]),
    }
    runner.execute_model_state = ExecuteModelState(
        scheduler_output=so_samp,
        logits_list=[torch.randn((2, 32))],
        pooler_output_list=[],
        num_reqs_list=[2],
        spec_decode_metadata_list=[None],
        draft_chunks=[],
        mamba_state_indices_list=[None],
    )
    fake_sync_out = MagicMock()
    fake_sync_out.get_output.return_value = SimpleNamespace(
        sampled_token_ids=[[], [7, 8]]
    )
    with (
        patch.object(torch, "tpu", MagicMock(), create=True),
        patch.object(tr_mod, "has_kv_transfer_group", return_value=False),
        patch.object(tr_mod, "AsyncTPUModelRunnerOutput", return_value=fake_sync_out),
    ):
        runner.sample_tokens(None)
        assert runner.requests["r1"].output_token_ids == [7, 8]
        runner.spec_decode_manager.propose_draft_token_ids.assert_called_once()


def test_remaining_load_dummy_attention_and_top_tokens():
    runner = _make_bare_runner()
    runner.device_config.device = torch.device("cpu")
    # load_model with non-deepseek, depth=0, no topk, env flags False, no drafter
    runner.model_config.hf_config.architectures = ["LlamaForCausalLM"]
    runner.parallel_config.pipeline_parallel_size = 1
    runner.speculative_config = None
    runner.drafter = None
    fake_model = torch.nn.Linear(2, 2)
    fake_loader = MagicMock()
    fake_loader.load_model.return_value = fake_model
    with (
        patch.object(tr_mod, "get_tpu_quantization_config", return_value="qcfg"),
        patch.object(tr_mod, "get_model_loader", return_value=fake_loader),
        patch.object(tr_mod, "validate_online_fp8"),
        patch.object(tr_mod.envs, "TPU_ROPE_CACHE_TRUNCATE", False),
        patch.object(tr_mod.envs, "TPU_ROPE_CACHE_ROW_MAJOR", False),
        patch.object(tr_mod.envs, "TPU_MOE_HASH_TABLE_ROW_MAJOR", False),
        patch.object(tr_mod.envs, "TPU_MOE_SKIP_PADDED_TOKENS", False),
        patch("vllm_torchtpu.model_loader_patches.evict_loaded_weights_page_cache"),
        patch.object(runner, "_initialize_pallas_kernels"),
    ):
        runner.load_model()
        assert getattr(runner, "_token_padding_state", None) is None
    runner.device = torch.device("cpu")

    # _dummy_run with through_handoff=True returning IntermediateTensors vs Tensor,
    # uses_mrope=False, etc.
    runner.uses_mrope = False
    runner.kv_cache_config = None
    runner._attn_layer_names = ["l0"]
    runner._token_padding_state = None
    runner._pp_wave = MagicMock()
    runner._pp_wave.start_forward.return_value = tr_mod.IntermediateTensors(
        {"h": torch.zeros((4, 4))}
    )
    runner.maybe_select_dummy_loras = MagicMock(return_value=contextlib.nullcontext())
    runner.forward_model = MagicMock(
        return_value=(tr_mod.IntermediateTensors({"h": torch.zeros((4, 4))}), None)
    )
    with (
        patch.object(
            tr_mod, "set_forward_context", return_value=contextlib.nullcontext()
        ),
        patch.object(
            tr_mod,
            "set_vllm_model_wrapper_context",
            return_value=contextlib.nullcontext(),
        ),
        patch.object(tr_mod, "synchronize_tensors"),
    ):
        assert runner._dummy_run(4, 2, 2, through_handoff=True) is None
        runner._pp_wave.end_forward.assert_called_once()

        # Now through_handoff=True returning regular Tensor
        runner.forward_model = MagicMock(return_value=(torch.zeros((4, 4)), None))
        assert runner._dummy_run(4, 2, 2, through_handoff=True) is None

    # _attention_schedule_capacity, _attention_kernel_shapes, _check_attention_schedule,
    # _add_pipeline_chunk_buckets, profile_pipeline_chunks
    runner._attention_runs_batched_kernel = None
    impl_batched = MagicMock(spec=tr_mod.PallasAttentionBackendImpl)
    impl_batched.runs_batched_rpa_schedule.return_value = True
    runner.cache_config.get_resolved_kv_cache_layout.return_value = next(
        iter(tr_mod.KV_LAYOUT_BY_VLLM_LAYOUT)
    )
    runner.model_config.get_num_attention_heads.return_value = 4
    runner.num_kv_heads = 2
    runner.head_size = 64
    runner.model_config.dtype = torch.bfloat16
    runner.kv_cache_dtype = torch.bfloat16
    runner._attention_kernel_block_size = 16
    runner._attention_kv_cache_group_id = 0
    runner.input_batch = SimpleNamespace(
        block_table=[SimpleNamespace(max_num_blocks_per_req=4)]
    )
    with (
        patch.object(
            tr_mod,
            "get_layers_from_vllm_config",
            return_value={"l0": SimpleNamespace(impl=impl_batched)},
        ),
        patch.object(
            tr_mod.rpa_batched,
            "schedule_capacity",
            return_value=(10, 8, 16),
            create=True,
        ),
    ):
        cap = runner._attention_schedule_capacity(use_max_model_len=True)
        assert cap == {"decode": (10, 8, 16), "mixed": (10, 8, 16)}
        # Cached hit
        assert runner._attention_schedule_capacity(use_max_model_len=True) == cap

    # _attention_kernel_shapes with group_id=None and use_max_model_len=False
    runner._attention_kv_cache_group_id = None
    runner.most_model_len = 32
    runner.num_reqs_most_model_len = 2
    assert runner._attention_kernel_shapes(use_max_model_len=True) == (
        runner.num_reqs_max_model_len,
        tr_mod.cdiv(runner.max_model_len, 16),
        16,
    )
    assert runner._attention_kernel_shapes(use_max_model_len=False) == (2, 2, 16)

    # _check_attention_schedule (both within capacity and exceeding capacity, plus hi <=
    # lo on line 5815)
    runner.seq_lens_np = np.array([16, 16], dtype=np.int32)
    with (
        patch.object(
            tr_mod.rpa_batched,
            "schedule_capacity",
            return_value=(10, 8, 16),
            create=True,
        ),
        patch("vllm_torchtpu.core.pp_chunks.schedule_pairs", return_value=5),
    ):
        runner._check_attention_schedule(
            2, np.array([1, 4], dtype=np.int32), num_decode=1
        )
        runner._check_attention_schedule(
            2, np.array([1, 4], dtype=np.int32), num_decode=0
        )
    with (
        patch("vllm_torchtpu.core.pp_chunks.schedule_pairs", return_value=20),
        pytest.raises(RuntimeError, match="SMEM schedule holds"),
    ):
        runner._check_attention_schedule(
            2, np.array([1, 4], dtype=np.int32), num_decode=1
        )
    runner._attention_runs_batched_kernel = False
    runner._check_attention_schedule(0, np.array([], dtype=np.int32), 0)

    # _add_pipeline_chunk_buckets when extra is non-empty and when extra is empty
    runner._pp_chunk_granularity = None
    runner._has_mamba_state = True
    runner.cache_config.mamba_cache_mode = "align"
    runner.cache_config.block_size = 4
    runner.max_num_tokens = 16
    runner.num_tokens_paddings = [8, 16]
    with (
        patch(
            "vllm_torchtpu.core.pp_chunk_scheduler.uses_dynamic_chunks",
            return_value=True,
        ),
        patch("vllm_torchtpu.core.pp_chunks.chunk_granularity", return_value=4),
        patch(
            "vllm_torchtpu.core.pp_chunks.chunk_buckets",
            return_value=[4, 8, 12, 16],
        ),
    ):
        runner._add_pipeline_chunk_buckets()
        assert 4 in runner.num_tokens_paddings and 12 in runner.num_tokens_paddings
        # Call again when _pp_chunk_granularity is already set
        runner._add_pipeline_chunk_buckets()

    # profile_pipeline_chunks (both schedule is None and schedule is not None)
    runner._dummy_run = MagicMock(return_value=1.5)
    with patch(
        "vllm_torchtpu.core.pp_chunks.profile_points",
        return_value=[(8, 0)],
    ):
        prof_no_sched = runner.profile_pipeline_chunks()
        assert prof_no_sched["granularity"] == 4

    runner._attention_runs_batched_kernel = True
    runner._attention_capacity[
        (runner.num_reqs_max_model_len, tr_mod.cdiv(runner.max_model_len, 16))
    ] = {
        "decode": (10, 8, 16),
        "mixed": (10, 8, 16),
    }
    with (
        patch("vllm_torchtpu.core.pp_chunks.chunk_pairs", return_value=5),
        patch(
            "vllm_torchtpu.core.pp_chunks.profile_points",
            return_value=[(8, 0)],
        ),
    ):
        prof = runner.profile_pipeline_chunks()
        assert prof["granularity"] == 4 and len(prof["samples"]) == 1

    with (
        patch("vllm_torchtpu.core.pp_chunks.chunk_pairs", return_value=20),
        pytest.raises(RuntimeError, match="SMEM schedule holds"),
    ):
        runner.profile_pipeline_chunks()

    # _pp_intermediate_tensors with random=True & _pp_take_topk_indices
    runner._pp_intermediate_template = {"h": ((4,), torch.float32)}
    runner._pp_topk_buffer = torch.zeros((8, 2), dtype=torch.int32)
    it_rand = runner._pp_intermediate_tensors(2, zeros=False, random=True)
    assert it_rand["h"].shape == (2, 4)

    with pytest.raises(RuntimeError, match="handed no top-k indices"):
        runner._pp_take_topk_indices(it_rand, 2)
    it_rand.tensors[tr_mod._TOPK_INDICES] = torch.ones((2, 2), dtype=torch.int32)
    runner._pp_take_topk_indices(it_rand, 2)
    assert (runner._pp_topk_buffer[:2] == 1).all()
    runner._pp_take_topk_indices(None, 2)

    # _model_supports_top_tokens & _can_use_greedy_top_tokens all branches
    runner.is_pooling_model = True
    assert runner._model_supports_top_tokens() is False
    runner.is_pooling_model = False
    runner._dsv41_smoke_logits_hooked = True
    assert runner._model_supports_top_tokens() is False
    del runner._dsv41_smoke_logits_hooked
    runner.compute_selected_logits = lambda *a: None  # type: ignore[method-assign]
    assert runner._model_supports_top_tokens() is False
    del runner.compute_selected_logits
    runner.model = None
    assert runner._model_supports_top_tokens() is False

    class _FakeModel:
        pass

    fm = _FakeModel()
    runner.model = fm
    assert runner._model_supports_top_tokens() is False
    fm.logits_processor = SimpleNamespace(logits_as_input=True)
    fm.lm_head = SimpleNamespace()
    assert runner._model_supports_top_tokens() is False
    fm.logits_processor.logits_as_input = False
    fm.logits_processor.scale = True
    assert runner._model_supports_top_tokens() is False
    fm.logits_processor.scale = 1.0
    fm.logits_processor.get_top_tokens = lambda lm, h: torch.zeros(
        h.shape[0], dtype=torch.int32
    )
    fm.lm_head.tp_size = 1
    fm.lm_head.shard_indices = SimpleNamespace(num_org_vocab_padding=0)
    assert runner._model_supports_top_tokens() is True

    runner.input_batch = SimpleNamespace(all_greedy=True, max_num_logprobs=None)
    assert (
        runner._can_use_greedy_top_tokens(
            SimpleNamespace(has_structured_output_requests=False)
        )
        is True
    )

    # Unwrapped compiled methods at bottom of TPUModelRunner (lines 7376-7461)
    fm.compute_logits = lambda h: h
    # Note: setting fm.compute_logits on instance puts it in fm.__dict__; test unwrapped
    # methods directly
    csl_fn = inspect.unwrap(TPUModelRunner.compute_selected_logits)
    assert csl_fn(
        runner, torch.zeros((4, 2)), torch.tensor([0, 2], dtype=torch.int32)
    ).shape == (2, 2)
    cst_fn = inspect.unwrap(TPUModelRunner.compute_selected_top_tokens)
    assert cst_fn(
        runner, torch.zeros((4, 2)), torch.tensor([0, 2], dtype=torch.int32)
    ).shape == (2,)
    clh_fn = inspect.unwrap(TPUModelRunner.compute_logits_from_hidden_states)
    assert clh_fn(runner, torch.zeros((3, 2))).shape == (3, 2)
    cth_fn = inspect.unwrap(TPUModelRunner.compute_top_tokens_from_hidden_states)
    assert cth_fn(runner, torch.zeros((3, 2))).shape == (3,)

    sfl_fn = inspect.unwrap(TPUModelRunner.sample_from_logits)
    assert torch.equal(
        sfl_fn(
            runner,
            torch.tensor([4, 5], dtype=torch.int32),
            torch.ones((2, 1)),
            torch.ones((2, 1)),
            torch.zeros((2, 1), dtype=torch.int32),
            torch.ones((2, 1)),
            all_greedy=True,
        ),
        torch.tensor([[4], [5]], dtype=torch.int32),
    )
    assert sfl_fn(
        runner,
        torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
        torch.ones((2, 1)),
        torch.ones((2, 2)),
        torch.zeros((2, 1), dtype=torch.int32),
        torch.ones((2, 1)),
        all_greedy=True,
    ).shape == (2, 1)


def test_remaining_branch_completions():
    runner = _make_bare_runner()
    runner.sample_from_logits_func = inspect.unwrap(
        TPUModelRunner.sample_from_logits
    ).__get__(runner, TPUModelRunner)
    # 1) 1529->1534: _update_num_xla_graphs with len(fresh) <= _MAX_DESCRIBED_GRAPHS
    runner.check_recompilation = True
    runner._xla_graphs_checked_at = None
    runner.num_xla_graphs = 0
    fake_tpu = SimpleNamespace(
        _get_cache_stats=lambda: SimpleNamespace(
            per_entry_stats=[
                SimpleNamespace(
                    last_read=100,
                    read_count=1,
                    compile_duration_ns=1000,
                    program_name="f",
                    input_signature="s",
                )
            ],
            num_cache_reqs=1,
            num_cache_hits=0,
        )
    )
    with patch.object(torch, "tpu", fake_tpu, create=True):
        runner._update_num_xla_graphs("step")
        assert runner.num_xla_graphs == 1
    runner.check_recompilation = False

    # 2) 3434->3448: _prepare_inputs with sched_spec_tokens non-empty for other reqs so
    # num_draft_tokens.any() is False
    runner.speculative_config = SimpleNamespace(
        num_speculative_tokens=1, use_eagle=lambda: False, method="ngram"
    )
    runner._pcp_enabled = False
    runner._mtp_enabled = False
    runner.mamba_slot_read_offsets = None
    bt0 = MagicMock()
    bt0.max_num_blocks_per_req = 4
    bt0.get_cpu_tensor.return_value = torch.zeros((8, 4), dtype=torch.int32)
    runner.input_batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["r0"],
        req_id_to_index={"r0": 0},
        num_computed_tokens_cpu=np.array([0], dtype=np.int32),
        num_tokens=np.array([2], dtype=np.int32),
        num_tokens_no_spec=np.array([2], dtype=np.int32),
        num_prompt_tokens=np.array([2], dtype=np.int32),
        token_ids_cpu=np.zeros((8, 16), dtype=np.int32),
        token_ids_cpu_tensor=torch.zeros((8, 16), dtype=torch.int32),
        block_table=[bt0],
        mamba_state_idx=np.array([1], dtype=np.int32),
    )
    runner.requests = {
        "r0": SimpleNamespace(num_computed_tokens=0, num_tokens=2, output_token_ids=[])
    }
    fake_plan = SimpleNamespace(
        kind=SequenceLayoutKind.ALL,
        descriptor=SimpleNamespace(
            kind=SequenceLayoutKind.ALL, protocol="p", version=1, cache_key="k"
        ),
        global_num_tokens=2,
        global_padded_num_tokens=8,
        local_num_tokens=2,
        local_padded_num_tokens=8,
        token_slice=slice(0, 8),
        input_ids_cpu=None,
        positions_cpu=None,
        query_start_loc_cpu=None,
        seq_lens_cpu=None,
        max_context_len_cpu=None,
        logits_indices_cpu=None,
        num_spec_tokens_override=None,
    )
    runner.sequence_layout_planner.prepare_real.return_value = fake_plan
    runner._build_attention_metadata = MagicMock(return_value=({"l0": "m"}, "common"))
    so_zero_drafts = SimpleNamespace(
        total_num_scheduled_tokens=2,
        num_scheduled_tokens={"r0": 2},
        scheduled_spec_decode_tokens={"other_req": [5]},
    )
    runner._prepare_inputs(
        so_zero_drafts, start_index=0, num_decode_reqs=0, num_windowed_reqs=0
    )

    # 3) 5083-5089 & 5100->5107: sample_tokens with _is_async_drafter=True on non-spec
    # step + pooling model async
    runner.scheduler_config.async_scheduling = True
    runner.speculative_config = SimpleNamespace(
        num_speculative_tokens=1,
        use_eagle=lambda: True,
        method="eagle3",
        draft_tensor_parallel_size=2,
    )
    runner.drafter = tr_mod.Eagle3Proposer.__new__(tr_mod.Eagle3Proposer)
    runner.drafter.draft_chunks = []
    runner.spec_decode_manager.propose_draft_token_ids = MagicMock(
        return_value=torch.tensor([[9]], dtype=torch.int32)
    )
    runner._assemble_async_prefill_bootstrap = MagicMock(
        return_value=(
            [torch.tensor([5, 9], dtype=torch.int32)],
            {0: 0},
            None,
            {0: 1},
        )
    )
    runner._modify_prev_results = MagicMock()
    runner._update_placeholder = MagicMock(return_value={"r0": 0})
    runner.input_batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["r0"],
        req_id_to_index={"r0": 0},
        all_greedy=True,
        max_num_logprobs=None,
        vocab_size=32,
        generators={},
        num_accepted_tokens_cpu=np.zeros(4, dtype=np.int64),
        token_ids_cpu=np.zeros((4, 16), dtype=np.int32),
        num_tokens=np.array([2, 0, 0, 0], dtype=np.int32),
        num_tokens_no_spec=np.array([2, 0, 0, 0], dtype=np.int32),
    )
    so_async_boot = SimpleNamespace(
        total_num_scheduled_tokens=2,
        num_scheduled_tokens={"r0": 2},
        scheduled_spec_decode_tokens={},
        has_structured_output_requests=False,
        finished_sending=None,
        finished_recving=None,
    )
    runner.execute_model_state = ExecuteModelState(
        scheduler_output=so_async_boot,
        logits_list=[torch.randn((1, 32))],
        pooler_output_list=[],
        num_reqs_list=[1],
        spec_decode_metadata_list=[None],
        draft_chunks=[],
        mamba_state_indices_list=[None],
    )
    with (
        patch.object(torch, "tpu", MagicMock(), create=True),
        patch.object(tr_mod, "has_kv_transfer_group", return_value=False),
    ):
        runner.sample_tokens(None)
        runner._assemble_async_prefill_bootstrap.assert_called_once()

    # Pooling model with async_scheduling=True (5100->5107)
    runner.is_pooling_model = True
    runner.drafter = None
    runner.speculative_config = None
    runner.execute_model_state = ExecuteModelState(
        scheduler_output=so_async_boot,
        logits_list=[],
        pooler_output_list=[torch.zeros((1, 4))],
        num_reqs_list=[],
        spec_decode_metadata_list=[],
        draft_chunks=[],
        mamba_state_indices_list=[],
    )
    with (
        patch.object(torch, "tpu", MagicMock(), create=True),
        patch.object(tr_mod, "has_kv_transfer_group", return_value=False),
    ):
        runner.sample_tokens(None)
    runner.is_pooling_model = False

    # 4) 5261->5264, 5277->5284: load_model with PP>1 and _pp_topk_buffer=None, plus
    # _is_async_drafter with draft_model=None
    runner.parallel_config.pipeline_parallel_size = 2
    runner.speculative_config = SimpleNamespace(
        num_speculative_tokens=1, use_eagle=lambda: True, method="eagle3"
    )
    runner.drafter = tr_mod.Eagle3Proposer.__new__(tr_mod.Eagle3Proposer)
    runner.drafter.load_model = MagicMock()
    runner.drafter.draft_model = None
    fake_loader = MagicMock()
    fake_loader.load_model.return_value = torch.nn.Linear(2, 2)
    with (
        patch.object(tr_mod, "get_tpu_quantization_config", return_value="qcfg"),
        patch.object(tr_mod, "get_model_loader", return_value=fake_loader),
        patch.object(tr_mod, "validate_online_fp8"),
        patch.object(tr_mod.envs, "TPU_ROPE_CACHE_TRUNCATE", False),
        patch.object(tr_mod.envs, "TPU_ROPE_CACHE_ROW_MAJOR", False),
        patch.object(tr_mod.envs, "TPU_MOE_HASH_TABLE_ROW_MAJOR", False),
        patch.object(tr_mod.envs, "TPU_MOE_SKIP_PADDED_TOKENS", False),
        patch("vllm_torchtpu.model_loader_patches.evict_loaded_weights_page_cache"),
        patch.object(runner, "_initialize_pallas_kernels"),
    ):
        runner.load_model()
    runner.device = torch.device("cpu")
    runner.drafter = None
    runner.speculative_config = None
    runner.parallel_config.pipeline_parallel_size = 1

    # 5) 5408->5412, 5412->5418, 5418->5423, 5430->5423: _initialize_attention_kernels
    # with plain module
    runner._attention_kernels_initialized = False
    runner.model = torch.nn.Linear(2, 2)
    with (
        patch.object(tr_mod, "get_layers_from_vllm_config", return_value={}),
        patch.object(
            tr_mod,
            "set_vllm_model_wrapper_context",
            return_value=contextlib.nullcontext(),
        ),
    ):
        runner._initialize_attention_kernels(force=True)

    # 6) 5558->5564, 5626->5631, 5654->5656, 5656->5658, 5679->5681: _dummy_run branches
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    runner.mamba_slot_read_offsets = None
    runner._has_mamba_state = False
    runner._pp_is_first = True
    runner._pp_topk_buffer = None
    runner.supports_mm_inputs = False
    runner.maybe_select_dummy_loras = MagicMock(return_value=contextlib.nullcontext())
    runner._build_attention_metadata = MagicMock(return_value=({"l0": "m"}, None))
    runner.forward_model = MagicMock(return_value=(torch.zeros((4, 4)), None))
    with (
        patch.object(
            tr_mod, "set_forward_context", return_value=contextlib.nullcontext()
        ),
        patch.object(
            tr_mod,
            "set_vllm_model_wrapper_context",
            return_value=contextlib.nullcontext(),
        ),
        patch.object(tr_mod, "synchronize_tensors"),
    ):
        ms = runner._dummy_run(
            4, 2, 2, profile_prefix=0, through_handoff=False, dp_lockstep=False
        )
        assert ms is not None

    # 7) 5965->5960, 5969->5974, 6022->exit, 6091, 6108, 6121->6126, 6191, 6209->6218,
    # 6218->6189, 6232->6239
    runner.num_tokens_paddings = [4]
    runner.num_reqs_paddings = []
    runner._model_supports_top_tokens = MagicMock(return_value=False)
    runner.compute_selected_logits = MagicMock(return_value=torch.zeros((1, 4)))
    with patch.object(tr_mod, "synchronize_tensors"):
        runner._precompile_compute_selected_logits()  # 5965->5960
        runner.num_reqs_paddings = [1, 2]
        runner.max_num_reqs = 4
        runner._precompile_compute_selected_logits()  # 5969->5974

    # 6022->exit: _precompile_structured_decoding where all num_tokens <=
    # max_target_rows
    runner.speculative_config = SimpleNamespace(
        num_speculative_tokens=1, use_eagle=lambda: False, method="ngram"
    )
    runner._pcp_enabled = False
    runner._mtp_enabled = False
    runner.structured_decoding_manager = SimpleNamespace(
        structured_decode_arange=torch.arange(4),
        require_structured_out_cpu=torch.zeros((8, 1), dtype=torch.bool),
        grammar_bitmask_cpu=torch.zeros((8, 4), dtype=torch.int32),
        require_structured_out_target_cpu=torch.zeros((8, 1), dtype=torch.bool),
        target_grammar_bitmask_cpu=torch.zeros((8, 4), dtype=torch.int32),
        structured_decode=MagicMock(return_value=torch.zeros((1, 4))),
    )
    runner.num_tokens_paddings = [4]
    with patch.object(tr_mod, "synchronize_tensors"):
        runner._precompile_structured_decoding()

    # 6091, 6108: _precompile_spec_logprobs with unreachable bucket (6108) and >
    # max_logits_len (6091)
    runner.num_tokens_paddings = [4, 64]
    runner.num_reqs_paddings = [1]
    runner.max_num_reqs = 1
    runner._spec_verify_bucket_reachable = MagicMock(return_value=False)
    with patch.object(tr_mod, "synchronize_tensors"):
        runner._precompile_spec_logprobs()

    # 6121->6126: _precompile_substitute_placeholder_token with _is_async_drafter=False
    runner.scheduler_config.async_scheduling = True
    runner.drafter = None
    runner.num_tokens_paddings = [4]
    runner.num_reqs_paddings = [1]
    with (
        patch.object(
            tr_mod, "_substitute_placeholder_token", return_value=torch.zeros(4)
        ),
        patch.object(tr_mod, "synchronize_tensors"),
    ):
        runner._precompile_substitute_placeholder_token()

    # 6191, 6209->6218, 6218->6189, 6232->6239: _precompile_rejection_sampler
    runner.speculative_config = SimpleNamespace(
        num_speculative_tokens=1, use_eagle=lambda: True, method="eagle3"
    )
    runner.drafter = tr_mod.Eagle3Proposer.__new__(tr_mod.Eagle3Proposer)
    runner.rejection_sampler = MagicMock()
    runner.rejection_sampler.synthetic_mode = False
    runner.rejection_sampler.return_value = torch.zeros((1, 2), dtype=torch.int32)
    runner.spec_bonus_and_target_logits = MagicMock(
        return_value=(torch.zeros(1, dtype=torch.int32), torch.zeros((4, 32)))
    )
    runner.spec_gather_bonus_and_target_logits = MagicMock(
        return_value=(torch.zeros((1, 32)), torch.zeros((4, 32)))
    )
    runner._model_supports_top_tokens = MagicMock(return_value=False)
    runner.num_tokens_paddings = [4, 64]
    runner.num_reqs_paddings = [1]
    runner.max_num_reqs = 1
    # First with reachable=False -> hits 6218->6189 and 6191
    runner._spec_verify_bucket_reachable = MagicMock(return_value=False)
    with patch.object(tr_mod, "synchronize_tensors"):
        runner._precompile_rejection_sampler()
    # Second with reachable=True and supports_top_tokens=False -> hits 6209->6218 and
    # 6232->6239
    runner._spec_verify_bucket_reachable = MagicMock(return_value=True)
    with patch.object(tr_mod, "synchronize_tensors"):
        runner._precompile_rejection_sampler()

    # 8) 6392->6399: _warmup_pcp_mtp_prefill when first attempt succeeds
    runner.enforce_eager = False
    runner._warmup_one_pcp_mtp_prefill = MagicMock(return_value=True)
    with (
        patch.object(tr_mod.envs, "SPEC_WARMUP", True),
        patch.object(
            tr_mod, "_suspend_kv_transfer_group", return_value=contextlib.nullcontext()
        ),
    ):
        runner._warmup_pcp_mtp_prefill()
        runner._warmup_one_pcp_mtp_prefill.assert_called_once()

    # 9) 6475->6477, 6499->exit: _precompile_backbone with needs_retrace=False and
    # _pp_wave=None
    runner.num_tokens_paddings = [4]
    runner.most_model_len = None
    runner._pp_wave = None
    runner._dummy_run = MagicMock()
    fake_buckets = SimpleNamespace(needs_retrace=lambda n: False, refused=set())
    with patch.object(
        tr_mod.shape_variants,
        "warmup",
        return_value=contextlib.nullcontext(fake_buckets),
    ):
        runner._precompile_backbone()

    # 10) 6553->6560: _precompile_mamba_state_seed_copies with _mamba_copy_plan
    # non-empty and kv_cache_raw_tensors=[]
    runner._mamba_copy_plan = [(0, [torch.zeros((4, 4))])]
    runner.kv_cache_raw_tensors = []
    runner.kv_cache_config = None
    runner._copy_mamba_state_blocks = MagicMock()
    with patch.object(tr_mod, "synchronize_tensors"):
        runner._precompile_mamba_state_seed_copies()

    # 11) 6686->6692: _warmup_spec_decode with max_num_seqs=1
    runner.input_batch.num_reqs = 0
    runner.scheduler_config.max_num_seqs = 1
    runner.num_tokens_paddings = [4]
    runner._warmup_one_spec_request = MagicMock(return_value=True)
    with (
        patch.object(tr_mod.envs, "SPEC_WARMUP", True),
        patch.object(
            tr_mod, "_suspend_kv_transfer_group", return_value=contextlib.nullcontext()
        ),
    ):
        runner._warmup_spec_decode()

    # 12) 6927->6931, 6939->6946, 6949->6960, 6955, 6968->6966, 6974->6979, 6981->6987,
    # 6992->6997: _warmup_spec_decode_cleanup
    runner._pre_async_results = None
    runner.requests = {"w0": SimpleNamespace()}
    runner.input_batch = SimpleNamespace(req_id_to_index={"w0": 0}, num_reqs=0)
    runner.spec_decode_manager = SimpleNamespace()  # no _draft_token_ids attr
    runner.drafter = None

    def _clear_reqs(so):
        runner.requests.clear()
        runner.input_batch.req_id_to_index.clear()

    runner.execute_model = _clear_reqs  # type: ignore[method-assign]
    assert runner._warmup_spec_decode_cleanup(["w0"]) is True
    # Also call when present is empty (6949->6960)
    assert runner._warmup_spec_decode_cleanup(["w0"]) is True
