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
"""Tests for the KV-cache spec, sizing and allocation paths.

Moved verbatim out of `tests/runner/test_tpu_runner.py` alongside the code
they cover; see vllm-project/vllm-torchtpu#713.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    get_current_vllm_config_or_none,
)
from vllm.model_executor.layers.attention import Attention, MLAAttention
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.worker.utils import AttentionGroup

from vllm_torchtpu.layers.adapter.attention import (
    PallasAttentionBackend,
    PallasBatchedRPAAttentionBackend,
    PallasMLAttentionBackend,
)
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadataBuilder
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform
from vllm_torchtpu.runner.kv_cache_dsv4 import DsV4KVCacheAllocator
from vllm_torchtpu.runner.kv_cache_manager import (
    KVCacheManager,
    check_kv_caches_cover_block_ids,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.worker.tpu_worker import TPUWorker


@pytest.fixture(autouse=True)
def _cpu_tpu_device_fallback(monkeypatch):
    from vllm_torchtpu.tpu_info import get_num_chips

    if get_num_chips() == 0:
        import vllm.config.device as vllm_device_mod

        orig_device = torch.device
        monkeypatch.setattr(
            vllm_device_mod.torch,
            "device",
            lambda dev, *args, **kwargs: (
                orig_device("cpu")
                if dev == "tpu"
                else orig_device(dev, *args, **kwargs)
            ),
        )


def _attention_layer_mock():
    """An `Attention` mock whose `get_attn_backend()` returns a real backend.

    `get_kv_cache_spec` measures page padding through the backend the layer
    reports. A bare `MagicMock` returns a mock shape, which iterates as empty
    and silently makes that arithmetic meaningless.
    """
    layer = MagicMock(spec=Attention)
    layer.get_attn_backend.return_value = PallasAttentionBackend
    return layer


class DummyMamba(MambaBase):
    def __init__(self):
        super().__init__()

    def get_kv_cache_spec(self, vllm_config):
        return MambaSpec(
            block_size=16,
            shapes=[(4, 128), (8, 64, 32)],
            dtypes=[torch.bfloat16, torch.float32],
            page_size_padded=vllm_config.cache_config.mamba_page_size_padded,
        )

    def get_state_dtype(self):
        return (torch.bfloat16, torch.float32)

    def get_state_shape(self):
        return ((4, 128), (8, 64, 32))

    @property
    def mamba_type(self):
        return "dummy"


class _FakeReserveSpec:
    """Offloading-spec stand-in resolved via kv_connector_extra_config by
    utils.estimate_kv_connector_hbm_reserve (same lookup as vLLM's
    OffloadingSpecFactory)."""

    RESERVE_BYTES = 1024 * 1024 * 1024  # 1 GiB

    @classmethod
    def estimate_hbm_reserve_bytes(cls, vllm_config):
        return cls.RESERVE_BYTES


@pytest.mark.parametrize("allocated_blocks", [0, 2, 3])
def test_check_kv_caches_cover_block_ids_rejects_short_cache(allocated_blocks):
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.bfloat16,
        page_size_padded=16384,
    )
    tensor = KVCacheTensor(
        size=16384 * 8,
        layers=["attn.0", "attn.1"],
        layer_stride=16384 * 4,
        block_stride=16384,
    )

    with pytest.raises(ValueError, match="scheduler can issue block IDs") as exc:
        check_kv_caches_cover_block_ids(
            {name: torch.empty((allocated_blocks, 1)) for name in tensor.layers},
            [tensor],
            4,
            lambda _: spec,
        )
    assert f"attn.0 holds {allocated_blocks} blocks" in str(exc.value)
    assert "2 layer(s) in this 131072-byte backing allocation" in str(exc.value)


def test_check_kv_caches_cover_block_ids_rejects_missing_attention_cache():
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=128, dtype=torch.bfloat16
    )
    tensor = KVCacheTensor(
        size=spec.page_size_bytes * 8,
        layers=["attn.0", "attn.1"],
        layer_stride=spec.page_size_bytes * 4,
        block_stride=spec.page_size_bytes,
    )
    with pytest.raises(ValueError, match="Missing KV cache.*attn.1"):
        check_kv_caches_cover_block_ids(
            {"attn.0": torch.empty((4, 1))}, [tensor], 4, lambda _: spec
        )


class TestKVCacheManager:
    def setup_method(self):
        self.mock_device = torch.device("cpu")

        model_config = ModelConfig(
            tokenizer_mode="auto",
            trust_remote_code=False,
            seed=0,
            dtype=torch.bfloat16,
        )
        cache_config = CacheConfig(
            block_size=16,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
        )
        scheduler_config = SchedulerConfig(
            max_num_seqs=16,
            max_model_len=1024,
            is_encoder_decoder=False,
        )
        parallel_config = ParallelConfig(
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
        )
        vllm_config = VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            scheduler_config=scheduler_config,
            parallel_config=parallel_config,
            observability_config={},
            additional_config={},
        )

        self.runner = MagicMock(spec=TPUModelRunner)
        self.runner.device = self.mock_device
        self.runner.block_size = 16
        self.runner.kv_cache_dtype = torch.bfloat16
        self.runner.use_spmd = False
        self.runner.vllm_config = vllm_config
        self.runner.cache_config = vllm_config.cache_config
        self.runner.scheduler_config = vllm_config.scheduler_config
        self.runner._hybrid_uniform_page_size_bytes = None
        self.runner.max_num_reqs = 16
        self.runner.max_model_len = 1024
        self.runner.max_num_tokens = 2048
        self.runner.pin_memory = False
        self.runner.model_config = vllm_config.model_config
        self.runner.shared_kv_cache_layers = {}
        self.runner.runner_only_attn_layers = set()
        self.runner.enforce_eager = False
        self.runner.speculative_config = None
        # Slots per request in the mamba pool (1 without spec decode).
        self.runner._mamba_slot_stride = 1

        # Bind the actual methods to our mock. The KV-cache spec/sizing bodies
        # live on KVCacheManager, which reads all of its state back off the
        # runner, so a real manager over the mock exercises the real code.
        self.runner.kv_cache_manager = KVCacheManager(self.runner)
        # Compact-mamba state starts unset (matches real __init__).
        self.runner._mamba_num_blocks = None
        self.runner._uniform_mamba_layout = vllm_config.kv_transfer_config is not None
        self.runner._unified_kv_layout = False
        self.runner.kv_cache_raw_tensors = []
        self.runner.get_kv_cache_spec = TPUModelRunner.get_kv_cache_spec.__get__(
            self.runner
        )
        self.runner.initialize_kv_cache = TPUModelRunner.initialize_kv_cache.__get__(
            self.runner
        )

        self._find_non_ssm_backend_patcher = patch.object(
            TpuPlatform, "_find_non_ssm_backend", return_value=PallasAttentionBackend
        )
        self._find_non_ssm_backend_patcher.start()

    def teardown_method(self):
        self._find_non_ssm_backend_patcher.stop()

    @patch(
        "vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_page_size_bytes",
        return_value=4096,
    )
    @patch(
        "vllm_torchtpu.utils.torch.accelerator.get_memory_info",
        return_value=(10 * 1024 * 1024 * 1024, 10 * 1024 * 1024 * 1024),
    )
    def test_update_mamba_page_size_padded(self, mock_mem_info, mock_get_page_size):
        layers = {}
        for i in range(1):
            mock_attn = _attention_layer_mock()
            mock_attn.num_kv_heads = 2
            mock_attn.head_size = 128
            layers[f"attn_{i}"] = mock_attn

        for i in range(3):
            layers[f"mamba_{i}"] = DummyMamba()

        self.runner.kv_cache_manager._update_mamba_page_size_padded(layers)

        # Mamba unpadded size:
        # (4 * 128 * 2) + (8 * 64 * 32 * 4) = 1024 + 65536 = 66560
        # uniform size = (1 * 4096) + (3 * 66560) = 4096 + 199680 = 203776
        assert self.runner._hybrid_uniform_page_size_bytes == 203776
        assert self.runner.cache_config.mamba_page_size_padded == 203776

        # Compact-mamba sizing runs first and wins (it can read HBM here).
        # It caps mamba at max_num_reqs + 1 = 17 recurrent slots and gives the
        # rest of the budget to the attention pool.
        #   1 attn + 3 mamba layers -> group_size=min=1 (3 < 1*1.5 is False),
        #   num_attn_groups=1, num_mamba_groups=3.
        #   avail = 10GB * 0.9 = 9,663,676,416 (headroom pinned to 0)
        #   avail_per_tensor = avail // 1 = 9,663,676,416
        #   mamba_per_tensor = 3 * 17 * 66560 = 3,394,560
        #   attn_num_blocks =
        #       (9,663,676,416 - 3,394,560) // (1 * 4096) = 2,358,467
        mock_mem_info.assert_called_once()
        assert self.runner._mamba_num_blocks == 17
        assert self.runner.cache_config.num_gpu_blocks_override == 2358467

    @patch(
        "vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_page_size_bytes",
        return_value=4096,
    )
    @patch(
        "vllm_torchtpu.utils.torch.accelerator.get_memory_info",
        return_value=(3 * 1024 * 1024, 3 * 1024 * 1024),
    )
    def test_compact_mamba_sizing_raises_when_mamba_exceeds_budget(
        self, mock_mem_info, mock_get_page_size
    ):
        """Mamba slots alone (3 * 17 * 66560 = 3,394,560 B) exceed the 3 MiB
        KV budget, so compact sizing cannot fit. Raise instead of falling back
        to the uniform layout, which pads every block to the mamba page and
        silently shrinks the block pool ~50x."""
        layers = {}
        mock_attn = _attention_layer_mock()
        mock_attn.num_kv_heads = 2
        mock_attn.head_size = 128
        layers["attn_0"] = mock_attn
        for i in range(3):
            layers[f"mamba_{i}"] = DummyMamba()

        with pytest.raises(ValueError, match="does not fit"):
            self.runner.kv_cache_manager._update_mamba_page_size_padded(layers)

    @patch(
        "vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_page_size_bytes",
        return_value=4096,
    )
    @patch(
        "vllm_torchtpu.utils.torch.accelerator.get_memory_info",
        return_value=(10 * 1024 * 1024 * 1024, 10 * 1024 * 1024 * 1024),
    )
    def test_kv_connector_reserve_shrinks_num_blocks_override(
        self, mock_mem_info, mock_get_page_size
    ):
        """The KV-connector HBM reserve (e.g. the offload H2D staging pool)
        must come out of the budget the block-count overrides size against,
        or the pinned blocks fill the reserve back up and defeat the
        worker-side subtraction. End-to-end through the same
        kv_connector_extra_config spec lookup the worker uses."""
        kv_tc = MagicMock()
        kv_tc.kv_connector_extra_config = {
            "spec_name": _FakeReserveSpec.__name__,
            "spec_module_path": _FakeReserveSpec.__module__,
        }
        self.runner.vllm_config.kv_transfer_config = kv_tc

        layers = {}
        mock_attn = _attention_layer_mock()
        mock_attn.num_kv_heads = 2
        mock_attn.head_size = 128
        layers["attn_0"] = mock_attn
        for i in range(3):
            layers[f"mamba_{i}"] = DummyMamba()

        self.runner.kv_cache_manager._update_mamba_page_size_padded(layers)

        # Same sizing as test_update_mamba_page_size_padded, but avail is
        # reduced by the 1 GiB reserve:
        #   avail = 10 GiB * 0.9 - 1 GiB = 8,589,934,592
        #   attn_num_blocks = (8,589,934,592 - 3 * 17 * 66560) // 4096
        #                   = 2,096,323  (vs 2,358,467 with no reserve)
        assert self.runner.cache_config.num_gpu_blocks_override == 2096323

    @patch(
        "vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_shape",
        return_value=(100, 16, 2, 1, 128),
    )
    @patch(
        "vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_page_size_bytes",
        return_value=4096,
    )
    @patch("vllm_torchtpu.utils.tpu_bind_kv_cache")
    @patch(
        "vllm_torchtpu.runner.kv_cache_manager.has_kv_transfer_group",
        return_value=False,
    )
    @patch("vllm.v1.worker.gpu_input_batch.InputBatch")
    def test_initialize_kv_cache_hybrid_duplication(
        self,
        mock_input_batch,
        mock_has_kv_transfer,
        mock_bind_kv_cache,
        mock_get_page_size,
        mock_get_shape,
    ):
        num_blocks = 100
        uniform_size = 203776
        tensor_size = uniform_size * num_blocks

        attn_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=128,
            dtype=torch.bfloat16,
            page_size_padded=uniform_size,
        )
        mamba_spec = MambaSpec(
            block_size=16,
            shapes=[(4, 128), (8, 64, 32)],
            dtypes=[torch.bfloat16, torch.float32],
            page_size_padded=uniform_size,
        )

        layer_names = ["attn.0", "mamba.0", "mamba.1", "mamba.2"]

        kv_cache_groups = [
            KVCacheGroupSpec(layer_names=["attn.0"], kv_cache_spec=attn_spec),
            KVCacheGroupSpec(layer_names=["mamba.0"], kv_cache_spec=mamba_spec),
            KVCacheGroupSpec(layer_names=["mamba.1"], kv_cache_spec=mamba_spec),
            KVCacheGroupSpec(layer_names=["mamba.2"], kv_cache_spec=mamba_spec),
        ]

        kv_cache_tensors = [
            KVCacheTensor(
                size=tensor_size,
                layers=[name],
                layer_stride=tensor_size,
                block_stride=uniform_size,
            )
            for name in layer_names
        ]
        kv_cache_config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_groups=kv_cache_groups,
        )

        self.runner.vllm_config.compilation_config.static_forward_context = {}
        self.runner.block_table_cpu = torch.zeros((1, 1), dtype=torch.int32)

        # One block table per kv_cache_group (the per-group dtype check
        # introduced with multi-group support iterates over every group).
        mock_block_tables = []
        for _ in kv_cache_groups:
            mock_bt = MagicMock()
            mock_bt.max_num_blocks_per_req = 1
            mock_bt.get_cpu_tensor.return_value = torch.zeros((1, 1), dtype=torch.int32)
            mock_block_tables.append(mock_bt)
        mock_input_batch.return_value.block_table = mock_block_tables

        self.runner.input_batch = mock_input_batch.return_value
        self.runner.kv_caches = []
        # Compact-mamba sizing not exercised here (initialize_kv_cache is
        # called directly with a fixed config): mamba shares the uniform
        # num_blocks. _mamba_num_blocks=None selects that fallback path.
        self.runner._mamba_num_blocks = None
        self.runner.initialize_kv_cache(kv_cache_config)

        mock_bind_kv_cache.assert_called_once()
        created_caches = mock_bind_kv_cache.call_args[0][0]

        assert len(created_caches) == 4

        # Mamba shares the uniform num_blocks (compact sizing not applied).
        for name in ["mamba.0", "mamba.1", "mamba.2"]:
            assert isinstance(created_caches[name], tuple)
            assert len(created_caches[name]) == 2
            assert created_caches[name][0].shape == (num_blocks, 4, 128)
            assert created_caches[name][1].shape == (num_blocks, 8, 64, 32)

        # Slot pool initialized to the (uniform) mamba block count.
        self.runner._init_mamba_slot_pool.assert_called_once_with(num_blocks)

        assert isinstance(created_caches["attn.0"], torch.Tensor)
        assert created_caches["attn.0"].shape == (100, 16, 2, 1, 128)

    @patch(
        "vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_shape",
        return_value=(100, 16, 2, 1, 128),
    )
    @patch(
        "vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_page_size_bytes",
        return_value=4096,
    )
    @patch("vllm_torchtpu.utils.tpu_bind_kv_cache")
    @patch(
        "vllm_torchtpu.runner.kv_cache_manager.has_kv_transfer_group",
        return_value=False,
    )
    @patch("vllm.v1.worker.gpu_input_batch.InputBatch")
    def test_initialize_kv_cache_compact_mamba(
        self,
        mock_input_batch,
        mock_has_kv_transfer,
        mock_bind_kv_cache,
        mock_get_page_size,
        mock_get_shape,
    ):
        """Compact-mamba: when _mamba_num_blocks is set, mamba layers allocate
        exactly that many recurrent slots while attention keeps num_blocks, and
        the slot pool is initialized to _mamba_num_blocks."""
        num_blocks = 100
        mamba_num_blocks = 17  # max_num_reqs(16) + 1
        uniform_size = 203776
        tensor_size = uniform_size * num_blocks

        attn_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=128,
            dtype=torch.bfloat16,
            page_size_padded=uniform_size,
        )
        mamba_spec = MambaSpec(
            block_size=16,
            shapes=[(4, 128), (8, 64, 32)],
            dtypes=[torch.bfloat16, torch.float32],
            page_size_padded=uniform_size,
        )
        layer_names = ["attn.0", "mamba.0", "mamba.1", "mamba.2"]
        kv_cache_groups = [
            KVCacheGroupSpec(layer_names=["attn.0"], kv_cache_spec=attn_spec),
            KVCacheGroupSpec(layer_names=["mamba.0"], kv_cache_spec=mamba_spec),
            KVCacheGroupSpec(layer_names=["mamba.1"], kv_cache_spec=mamba_spec),
            KVCacheGroupSpec(layer_names=["mamba.2"], kv_cache_spec=mamba_spec),
        ]
        kv_cache_config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=tensor_size,
                    layers=[name],
                    layer_stride=tensor_size,
                    block_stride=uniform_size,
                )
                for name in layer_names
            ],
            kv_cache_groups=kv_cache_groups,
        )

        self.runner.vllm_config.compilation_config.static_forward_context = {}
        self.runner.block_table_cpu = torch.zeros((1, 1), dtype=torch.int32)
        mock_block_tables = []
        for _ in kv_cache_groups:
            mock_bt = MagicMock()
            mock_bt.max_num_blocks_per_req = 1
            mock_bt.get_cpu_tensor.return_value = torch.zeros((1, 1), dtype=torch.int32)
            mock_block_tables.append(mock_bt)
        mock_input_batch.return_value.block_table = mock_block_tables
        self.runner.input_batch = mock_input_batch.return_value
        self.runner.kv_caches = []
        self.runner._mamba_num_blocks = mamba_num_blocks
        self.runner.initialize_kv_cache(kv_cache_config)

        created_caches = mock_bind_kv_cache.call_args[0][0]
        # Mamba: only mamba_num_blocks slots; attention: full num_blocks.
        for name in ["mamba.0", "mamba.1", "mamba.2"]:
            assert created_caches[name][0].shape == (mamba_num_blocks, 4, 128)
            assert created_caches[name][1].shape == (mamba_num_blocks, 8, 64, 32)
        assert created_caches["attn.0"].shape == (100, 16, 2, 1, 128)
        # Slot pool sized to the compact mamba block count.
        self.runner._init_mamba_slot_pool.assert_called_once_with(mamba_num_blocks)

    @patch("vllm_torchtpu.runner.kv_cache_manager.get_layers_from_vllm_config")
    def test_batched_kv_spec_rpc_establishes_config_context(self, mock_get_layers):
        self.runner.cache_config.kv_cache_layout = "LBNHC"
        layer = _attention_layer_mock()
        layer.get_attn_backend.return_value = PallasBatchedRPAAttentionBackend
        layer.attn_type = AttentionType.DECODER
        layer.num_kv_heads = 2
        layer.head_size = 128
        layer.kv_sharing_target_layer_name = None
        layer.sliding_window = None
        mock_get_layers.return_value = {"attn.0": layer}
        worker = SimpleNamespace(model_runner=self.runner)

        assert get_current_vllm_config_or_none() is None
        with patch.object(
            TpuPlatform,
            "_find_non_ssm_backend",
            return_value=PallasBatchedRPAAttentionBackend,
        ):
            specs = TPUWorker.get_kv_cache_spec(worker)
        assert specs["attn.0"].page_size_bytes == 16384
        assert get_current_vllm_config_or_none() is None

    @patch("vllm_torchtpu.runner.kv_cache_manager.get_layers_from_vllm_config")
    @patch(
        "vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_page_size_bytes",
        return_value=4096,
    )
    @patch(
        "vllm_torchtpu.utils.torch.accelerator.get_memory_info",
        return_value=(10 * 1024 * 1024 * 1024, 10 * 1024 * 1024 * 1024),
    )
    def test_get_kv_cache_spec_hybrid_padded_size(
        self, mock_mem_info, mock_get_page_size, mock_get_layers
    ):
        layers = {}

        mock_attn = _attention_layer_mock()
        mock_attn.attn_type = AttentionType.DECODER
        mock_attn.num_kv_heads = 2
        mock_attn.head_size = 128
        mock_attn.kv_sharing_target_layer_name = None
        mock_attn.sliding_window = None
        layers["attn_0"] = mock_attn

        for i in range(3):
            layers[f"mamba_{i}"] = DummyMamba()

        mock_get_layers.return_value = layers

        # Call the method
        kv_cache_specs = self.runner.get_kv_cache_spec()

        # Mamba unpadded size: (4 * 128 * 2) + (8 * 64 * 32 * 4) = 66560
        # uniform size = (1 * 4096) + (3 * 66560) = 203776
        expected_padded_size = 203776

        # Verify specs
        assert "attn_0" in kv_cache_specs
        attn_spec = kv_cache_specs["attn_0"]
        assert isinstance(attn_spec, FullAttentionSpec)
        assert attn_spec.page_size_padded == expected_padded_size

        for i in range(3):
            layer_name = f"mamba_{i}"
            assert layer_name in kv_cache_specs
            mamba_spec = kv_cache_specs[layer_name]
            assert isinstance(mamba_spec, MambaSpec)
            assert mamba_spec.page_size_padded == expected_padded_size

    @patch.object(
        TpuPlatform, "_find_non_ssm_backend", return_value=PallasMLAttentionBackend
    )
    @patch("vllm_torchtpu.runner.kv_cache_manager.get_layers_from_vllm_config")
    @patch(
        "vllm_torchtpu.runner.kv_cache_manager.PallasMLAttentionBackend.get_kv_cache_page_size_bytes",
        return_value=8192,
    )
    @patch(
        "vllm_torchtpu.utils.torch.accelerator.get_memory_info",
        return_value=(10 * 1024 * 1024 * 1024, 10 * 1024 * 1024 * 1024),
    )
    def test_get_kv_cache_spec_mla_mamba_hybrid(
        self,
        mock_mem_info,
        mock_get_page_size,
        mock_get_layers,
        mock_find_non_ssm_backend,
    ):
        mock_mla = MagicMock(spec=MLAAttention)
        mock_mla.num_kv_heads = 1
        mock_mla.head_size = 576
        layers = {"mla_0": mock_mla}
        for i in range(3):
            layers[f"mamba_{i}"] = DummyMamba()
        mock_get_layers.return_value = layers

        kv_cache_specs = self.runner.get_kv_cache_spec()

        # uniform size = one 8192-byte MLA page + three 66560-byte states
        expected_padded_size = 207872
        assert isinstance(kv_cache_specs["mla_0"], MLAAttentionSpec)
        assert kv_cache_specs["mla_0"].page_size_padded == expected_padded_size
        for i in range(3):
            assert kv_cache_specs[f"mamba_{i}"].page_size_padded == expected_padded_size
        mock_get_page_size.assert_called()

    @patch("vllm_torchtpu.runner.kv_cache_manager.get_layers_from_vllm_config")
    @patch(
        "vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_page_size_bytes",
        return_value=4096,
    )
    def test_get_kv_cache_spec_pure_attention_no_cache_config_updates(
        self, mock_get_page_size, mock_get_layers
    ):
        mock_attn = _attention_layer_mock()
        mock_attn.attn_type = AttentionType.DECODER
        mock_attn.num_kv_heads = 2
        mock_attn.head_size = 128
        mock_attn.kv_sharing_target_layer_name = None
        mock_attn.sliding_window = None

        layers = {"layer.0": mock_attn}
        mock_get_layers.return_value = layers

        with patch.object(
            self.runner.kv_cache_manager, "_update_mamba_page_size_padded"
        ) as mock_update:
            self.runner.get_kv_cache_spec()
            mock_update.assert_not_called()

    @patch("vllm_torchtpu.runner.kv_cache_manager.get_layers_from_vllm_config")
    @patch(
        "vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_page_size_bytes",
        return_value=4096,
    )
    def test_get_kv_cache_spec_records_shared_layers(
        self, mock_get_page_size, mock_get_layers
    ):
        owner = _attention_layer_mock()
        owner.attn_type = AttentionType.DECODER
        owner.num_kv_heads = 2
        owner.head_size = 128
        owner.kv_sharing_target_layer_name = None
        owner.sliding_window = None
        owner.impl = SimpleNamespace(kv_cache_quantized_dtype=None)

        shared = _attention_layer_mock()
        shared.attn_type = AttentionType.DECODER
        shared.num_kv_heads = 2
        shared.head_size = 128
        shared.kv_sharing_target_layer_name = "layer.0"
        shared.sliding_window = None
        shared.impl = SimpleNamespace(kv_cache_quantized_dtype=None)

        mock_get_layers.return_value = {
            "layer.0": owner,
            "layer.1": shared,
        }

        kv_cache_specs = self.runner.get_kv_cache_spec()

        assert set(kv_cache_specs) == {"layer.0"}
        assert self.runner.shared_kv_cache_layers == {"layer.1": "layer.0"}

    def _make_shared_pair(self, attn_window, target_window):
        attn = _attention_layer_mock()
        attn.attn_type = AttentionType.DECODER
        attn.num_kv_heads = 2
        attn.head_size = 128
        attn.sliding_window = attn_window
        attn.impl = SimpleNamespace(kv_cache_quantized_dtype=None)

        target = _attention_layer_mock()
        target.attn_type = AttentionType.DECODER
        target.num_kv_heads = 2
        target.head_size = 128
        target.sliding_window = target_window
        target.impl = SimpleNamespace(kv_cache_quantized_dtype=None)
        return attn, target

    def test_validate_shared_layout_skips_sliding_window_when_hma_disabled(self):
        # HMA disabled: specs are unified to full attention, so a window
        # mismatch is harmless and must not raise.
        attn, target = self._make_shared_pair(attn_window=512, target_window=None)
        KVCacheManager._validate_shared_kv_cache_layout(
            "layer.1", attn, "layer.0", target, hma_enabled=False
        )

    def test_validate_shared_layout_enforces_sliding_window_when_hma_enabled(self):
        # HMA enabled: sliding-window layers keep a smaller window-sized cache,
        # so a window mismatch means the layers disagree on storage -> raise.
        attn, target = self._make_shared_pair(attn_window=512, target_window=None)
        with pytest.raises(ValueError, match="sliding_window"):
            KVCacheManager._validate_shared_kv_cache_layout(
                "layer.1", attn, "layer.0", target, hma_enabled=True
            )

    def test_validate_shared_layout_matching_window_ok_when_hma_enabled(self):
        # HMA enabled but windows match: no mismatch, no raise.
        attn, target = self._make_shared_pair(attn_window=512, target_window=512)
        KVCacheManager._validate_shared_kv_cache_layout(
            "layer.1", attn, "layer.0", target, hma_enabled=True
        )

    def _make_fp8_shared_pair(
        self, attn_quant, target_quant, attn_scale=1.0, target_scale=1.0
    ):
        attn = _attention_layer_mock()
        attn.attn_type = AttentionType.DECODER
        attn.num_kv_heads = 2
        attn.head_size = 128
        attn.sliding_window = None
        attn._k_scale_float = attn_scale
        attn._v_scale_float = attn_scale
        attn.impl = MagicMock(kv_cache_quantized_dtype=attn_quant)

        target = _attention_layer_mock()
        target.attn_type = AttentionType.DECODER
        target.num_kv_heads = 2
        target.head_size = 128
        target.sliding_window = None
        target._k_scale_float = target_scale
        target._v_scale_float = target_scale
        target.impl = MagicMock(kv_cache_quantized_dtype=target_quant)
        return attn, target

    def test_validate_shared_layout_fp8_matching_ok(self):
        # Both sides fp8 with matching scales/dtype: no raise.
        attn, target = self._make_fp8_shared_pair("fp8_e4m3", "fp8_e4m3")
        KVCacheManager._validate_shared_kv_cache_layout(
            "layer.1", attn, "layer.0", target, hma_enabled=False
        )

    def test_validate_shared_layout_fp8_mismatched_scales_raises(self):
        # Both fp8 but scales differ: dequantizing with the wrong scales
        # corrupts attention -> raise.
        attn, target = self._make_fp8_shared_pair(
            "fp8_e4m3", "fp8_e4m3", attn_scale=1.0, target_scale=2.0
        )
        with pytest.raises(ValueError, match="k/v scales"):
            KVCacheManager._validate_shared_kv_cache_layout(
                "layer.1", attn, "layer.0", target, hma_enabled=False
            )

    def test_validate_shared_layout_fp8_target_only_raises(self):
        # Asymmetric: target is fp8, shared layer is not. The shared layer would
        # read the target's packed fp8 bytes as bf16 -> silent garbage. Must
        # raise instead of being skipped.
        attn, target = self._make_fp8_shared_pair(None, "fp8_e4m3")
        with pytest.raises(ValueError, match="kv cache"):
            KVCacheManager._validate_shared_kv_cache_layout(
                "layer.1", attn, "layer.0", target, hma_enabled=False
            )

    def test_validate_shared_layout_fp8_attn_only_raises(self):
        # Asymmetric the other way: shared layer is fp8, target is not.
        attn, target = self._make_fp8_shared_pair("fp8_e4m3", None)
        with pytest.raises(ValueError, match="kv cache"):
            KVCacheManager._validate_shared_kv_cache_layout(
                "layer.1", attn, "layer.0", target, hma_enabled=False
            )

    @patch(
        "vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_shape",
        return_value=(100, 16, 2, 1, 128),
    )
    @patch("vllm_torchtpu.utils.tpu_bind_kv_cache")
    @patch(
        "vllm_torchtpu.runner.kv_cache_manager.has_kv_transfer_group",
        return_value=False,
    )
    @patch("vllm.v1.worker.gpu_input_batch.InputBatch")
    def test_initialize_kv_cache_binds_shared_layer_to_target_cache(
        self, mock_input_batch, mock_has_kv_transfer, mock_bind_kv_cache, mock_get_shape
    ):
        attn_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=128,
            dtype=torch.bfloat16,
            page_size_padded=16384,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=100,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=16384 * 100,
                    layers=["layer.0"],
                    layer_stride=16384 * 100,
                    block_stride=16384,
                )
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=attn_spec)
            ],
        )

        self.runner.shared_kv_cache_layers = {"layer.1": "layer.0"}
        self.runner.vllm_config.compilation_config.static_forward_context = {}
        self.runner.block_table_cpu = torch.zeros((1, 1), dtype=torch.int32)
        self.runner.kv_caches = []
        mock_bt = MagicMock()
        mock_bt.max_num_blocks_per_req = 1
        mock_bt.get_cpu_tensor.return_value = torch.zeros((1, 1), dtype=torch.int32)
        mock_input_batch.return_value.block_table = [mock_bt]
        self.runner.input_batch = mock_input_batch.return_value

        self.runner.initialize_kv_cache(kv_cache_config)

        created_caches = mock_bind_kv_cache.call_args[0][0]
        assert set(created_caches) == {"layer.0", "layer.1"}
        assert created_caches["layer.1"] is created_caches["layer.0"]
        group_names = self.runner.kv_cache_config.kv_cache_groups[0].layer_names
        assert group_names == ["layer.0", "layer.1"]
        assert self.runner.attn_groups[0][0].layer_names == ["layer.0", "layer.1"]
        assert kv_cache_config.kv_cache_groups[0].layer_names == ["layer.0"]

    @patch(
        "vllm_torchtpu.runner.kv_cache_manager.has_kv_transfer_group",
        return_value=False,
    )
    @patch("vllm.v1.worker.gpu_input_batch.InputBatch")
    def test_initialize_kv_cache_rejects_missing_shared_target(
        self, mock_input_batch, mock_has_kv_transfer
    ):
        attn_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=128,
            dtype=torch.bfloat16,
            page_size_padded=16384,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=16384,
                    layers=["layer.0"],
                    layer_stride=16384,
                    block_stride=16384,
                )
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=attn_spec)
            ],
        )
        self.runner.shared_kv_cache_layers = {"layer.1": "missing.layer"}

        with pytest.raises(ValueError, match="target layer is missing"):
            self.runner.initialize_kv_cache(kv_cache_config)

    @patch(
        "vllm_torchtpu.runner.kv_cache_manager.has_kv_transfer_group",
        return_value=False,
    )
    @patch("vllm.v1.worker.gpu_input_batch.InputBatch")
    def test_initialize_kv_cache_rejects_duplicate_shared_allocation(
        self, mock_input_batch, mock_has_kv_transfer
    ):
        attn_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=128,
            dtype=torch.bfloat16,
            page_size_padded=16384,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=16384,
                    layers=["layer.0"],
                    layer_stride=16384,
                    block_stride=16384,
                ),
                KVCacheTensor(
                    size=16384,
                    layers=["layer.1"],
                    layer_stride=16384,
                    block_stride=16384,
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=attn_spec),
                KVCacheGroupSpec(layer_names=["layer.1"], kv_cache_spec=attn_spec),
            ],
        )
        self.runner.shared_kv_cache_layers = {"layer.1": "layer.0"}

        with pytest.raises(ValueError, match="independent KV cache"):
            self.runner.initialize_kv_cache(kv_cache_config)

    def test_initialize_kv_cache_block_major(self):
        """Verifies that block-major initialization allocates a single shared bundle
        backing all layer views."""
        num_blocks = 4
        # Byte-consistent page size: 16*4*1*128 bf16 elements = 16384 B to ensure exact
        # uniform tiling.
        per_layer_shape = (16, 4, 1, 128)
        # FullAttentionSpec.real_page_size_bytes for (16, 2, 128, bf16) is
        # block_size * (k+v) * head_size * itemsize = 16 * 4 * 128 * 2 = 16384.
        per_layer_bytes = 16384
        per_layer_size = per_layer_bytes * num_blocks

        attn_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=128,
            dtype=torch.bfloat16,
            page_size_padded=per_layer_bytes,
        )
        layer_names = ["attn.0", "attn.1", "attn.2"]
        # Uniform KV cache group representing all layers of a dense model.
        kv_cache_groups = [
            KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=attn_spec)
        ]
        kv_cache_config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=per_layer_size * len(layer_names),
                    layers=layer_names,
                    layer_stride=per_layer_bytes,
                    block_stride=per_layer_bytes * len(layer_names),
                )
            ],
            kv_cache_groups=kv_cache_groups,
        )

        self.runner.vllm_config.compilation_config.static_forward_context = {}
        self.runner.block_table_cpu = torch.zeros((1, 1), dtype=torch.int32)
        self.runner.kv_caches = []
        # The block-major topology gates read parallel_config (PCP/PP).
        self.runner.parallel_config = self.runner.vllm_config.parallel_config

        mock_input_batch = MagicMock()
        mock_block_tables = []
        for _ in kv_cache_groups:
            mock_bt = MagicMock()
            mock_bt.max_num_blocks_per_req = 1
            mock_bt.get_cpu_tensor.return_value = torch.zeros((1, 1), dtype=torch.int32)
            mock_block_tables.append(mock_bt)
        mock_input_batch.block_table = mock_block_tables
        self.runner.input_batch = mock_input_batch

        cross_layer_connector = MagicMock()
        cross_layer_connector.prefer_cross_layer_blocks = True
        with (
            patch(
                "vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_shape",
                return_value=(num_blocks,) + per_layer_shape,
            ),
            patch("vllm_torchtpu.utils.tpu_bind_kv_cache") as mock_bind,
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.has_kv_transfer_group",
                return_value=False,
            ),
            patch(
                "vllm.v1.worker.kv_connector_model_runner_mixin.has_kv_transfer_group",
                return_value=True,
            ),
            patch(
                "vllm.v1.worker.kv_connector_model_runner_mixin.get_kv_transfer_group",
                return_value=cross_layer_connector,
            ),
            patch("vllm_torchtpu.envs.VLLM_TPU_BLOCK_MAJOR_KV", True),
        ):
            self.runner.initialize_kv_cache(kv_cache_config)

        # Verify bundle tensor shape (num_blocks, num_layers, *per_layer_shape) and
        # layer index mappings.
        assert isinstance(self.runner._kv_cache_bundle, torch.Tensor)
        expected_bundle_shape = (num_blocks, len(layer_names)) + per_layer_shape
        assert tuple(self.runner._kv_cache_bundle.shape) == expected_bundle_shape
        assert self.runner._kv_cache_bundle.dtype == torch.bfloat16
        assert self.runner._kv_cache_bundle_layer_index == {
            "attn.0": 0,
            "attn.1": 1,
            "attn.2": 2,
        }

        # Verify that bound per-layer views alias the underlying bundle storage via
        # strided slices.
        created = mock_bind.call_args[0][0]
        for idx, name in enumerate(layer_names):
            view = created[name]
            assert isinstance(view, torch.Tensor)
            assert tuple(view.shape) == (num_blocks,) + per_layer_shape
            # Storage aliasing: same underlying buffer as the bundle.
            assert (
                view.untyped_storage().data_ptr()
                == self.runner._kv_cache_bundle.untyped_storage().data_ptr()
            )

        # Verify that mutating a per-layer strided view modifies the bundle in-place
        # while isolating other layers.
        target = 1
        created[layer_names[target]].fill_(7)
        assert torch.all(self.runner._kv_cache_bundle[:, target] == 7), (
            "write through per-layer view did not mutate bundle storage"
        )
        # Other layers untouched.
        for idx, name in enumerate(layer_names):
            if idx == target:
                continue
            assert torch.all(self.runner._kv_cache_bundle[:, idx] == 0), (
                f"layer {idx} disturbed by write to layer {target}"
            )

    def test_initialize_kv_cache_block_major_rejects_mamba(self):
        """Verifies that standard dense block-major initialization fails closed on
        unsupported multi-group configurations."""
        attn_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=128,
            dtype=torch.bfloat16,
            page_size_padded=16384,
        )
        mamba_spec = MambaSpec(
            block_size=16,
            shapes=[(4, 128)],
            dtypes=[torch.bfloat16],
            page_size_padded=16384,
        )
        kv_cache_groups = [
            KVCacheGroupSpec(layer_names=["attn.0"], kv_cache_spec=attn_spec),
            KVCacheGroupSpec(layer_names=["mamba.0"], kv_cache_spec=mamba_spec),
        ]
        kv_cache_config = KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=16384,
                    layers=["attn.0"],
                    layer_stride=16384,
                    block_stride=16384,
                ),
                KVCacheTensor(
                    size=16384,
                    layers=["mamba.0"],
                    layer_stride=16384,
                    block_stride=16384,
                ),
            ],
            kv_cache_groups=kv_cache_groups,
        )

        self.runner.vllm_config.compilation_config.static_forward_context = {}
        self.runner.block_table_cpu = torch.zeros((1, 1), dtype=torch.int32)
        self.runner.kv_caches = []

        mock_input_batch = MagicMock()
        mock_block_tables = []
        for _ in kv_cache_groups:
            mock_bt = MagicMock()
            mock_bt.max_num_blocks_per_req = 1
            mock_bt.get_cpu_tensor.return_value = torch.zeros((1, 1), dtype=torch.int32)
            mock_block_tables.append(mock_bt)
        mock_input_batch.block_table = mock_block_tables
        self.runner.input_batch = mock_input_batch
        self.runner._mamba_num_blocks = None

        with (
            patch(
                "vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_shape",
                return_value=(1, 16, 2, 1, 128),
            ),
            patch("vllm_torchtpu.utils.tpu_bind_kv_cache"),
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.has_kv_transfer_group",
                return_value=False,
            ),
            patch("vllm_torchtpu.envs.VLLM_TPU_BLOCK_MAJOR_KV", True),
            pytest.raises(NotImplementedError, match="hybrid attention"),
        ):
            self.runner.initialize_kv_cache(kv_cache_config)

        assert self.runner._kv_cache_bundle is None
        assert self.runner._kv_cache_bundle_layer_index == {}

    @pytest.mark.parametrize("alias_head_size", [128, 256])
    def test_default_attention_hma_preserves_native_pool_budget(self, alias_head_size):
        """Full/SWA overlays share storage while retaining the last block ID."""
        from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups
        from vllm.v1.kv_cache_interface import SlidingWindowSpec

        def spec(cls, head_size=128, **kwargs):
            return PallasAttentionBackend.customize_spec(
                cls(
                    block_size=16,
                    num_kv_heads=256 // head_size,
                    head_size=head_size,
                    dtype=torch.bfloat16,
                    **kwargs,
                )
            )

        full = spec(FullAttentionSpec)
        groups = [
            KVCacheGroupSpec(layer_names=["full.0", "full.1"], kv_cache_spec=full),
            KVCacheGroupSpec(
                layer_names=["swa.0", "swa.1"],
                kv_cache_spec=spec(
                    SlidingWindowSpec, head_size=alias_head_size, sliding_window=32
                ),
            ),
        ]
        num_blocks = 4
        budget = 2 * num_blocks * full.page_size_bytes
        config = get_kv_cache_config_from_groups(
            self.runner.vllm_config, groups, budget
        )
        assert config.num_blocks == num_blocks
        assert all(tensor.size == budget for tensor in config.kv_cache_tensors)
        self.runner.vllm_config.compilation_config.static_forward_context = {}
        self.runner.block_table_cpu = torch.zeros((1, 1), dtype=torch.int32)
        tables = []
        for _ in groups:
            table = MagicMock()
            table.get_cpu_tensor.return_value = self.runner.block_table_cpu
            tables.append(table)
        self.runner.input_batch = SimpleNamespace(block_table=tables)

        with (
            patch("vllm_torchtpu.utils.tpu_bind_kv_cache") as bind,
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.has_kv_transfer_group",
                return_value=False,
            ),
        ):
            self.runner.initialize_kv_cache(config)
        caches = bind.call_args.args[0]
        unique = {id(cache): cache for cache in caches.values()}
        assert sum(cache.nbytes for cache in unique.values()) == budget
        assert len(unique) == 2
        for index in range(2):
            assert caches[f"full.{index}"] is caches[f"swa.{index}"]
            assert caches[f"full.{index}"].shape[0] == num_blocks
        caches["full.0"][num_blocks - 1].fill_(7)
        assert torch.all(caches["swa.0"][num_blocks - 1] == 7)
        assert torch.count_nonzero(caches["full.0"][: num_blocks - 1]) == 0
        assert torch.count_nonzero(caches["full.1"]) == 0

    @pytest.mark.parametrize(
        "sliding_heads, sliding_page_bytes, shared",
        [
            (4, 524288, True),
            (8, 1048576, True),
            (6, 786432, False),
        ],
    )
    def test_gemma_hma_native_geometries_fit_scheduler_budget(
        self, sliding_heads, sliding_page_bytes, shared
    ):
        """Compatible shapes share regions; other shapes retain native arrays."""
        from vllm.v1.core.kv_cache_utils import (
            get_kv_cache_config_from_groups,
            get_kv_cache_groups,
        )
        from vllm.v1.kv_cache_interface import SlidingWindowSpec

        self.runner.block_size = 256
        self.runner.cache_config.block_size = 256
        self.runner.cache_config.kv_cache_layout = "LBNHC"
        self.runner.kv_cache_dtype = torch.float8_e4m3fn
        self.runner.scheduler_config.disable_hybrid_kv_cache_manager = False
        layers = {}
        for prefix, count, heads, head_size, window in [
            ("full", 2, 1, 512, None),
            ("swa", 6, sliding_heads, 256, 1024),
        ]:
            for index in range(count):
                layer = _attention_layer_mock()
                layer.attn_type = AttentionType.DECODER
                layer.num_kv_heads = heads
                layer.head_size = head_size
                layer.sliding_window = window
                layer.kv_sharing_target_layer_name = None
                layers[f"{prefix}.{index}"] = layer
        with patch(
            "vllm_torchtpu.runner.kv_cache_manager.get_layers_from_vllm_config",
            return_value=layers,
        ):
            specs = self.runner.get_kv_cache_spec()
        groups = get_kv_cache_groups(self.runner.vllm_config, specs)
        assert len(groups) == 4
        assert (
            sum(isinstance(group.kv_cache_spec, SlidingWindowSpec) for group in groups)
            == 3
        )
        num_blocks = 4
        page_bytes = (
            max(524288, sliding_page_bytes) if shared else 524288 + sliding_page_bytes
        )
        budget = num_blocks * 2 * page_bytes
        config = get_kv_cache_config_from_groups(
            self.runner.vllm_config, groups, budget
        )
        self.runner.vllm_config.compilation_config.static_forward_context = {}
        self.runner.block_table_cpu = torch.zeros((1, 1), dtype=torch.int32)
        self.runner.input_batch = SimpleNamespace(
            block_table=[
                SimpleNamespace(
                    max_num_blocks_per_req=4,
                    get_cpu_tensor=lambda: self.runner.block_table_cpu,
                )
                for _ in groups
            ]
        )
        with (
            patch("vllm_torchtpu.utils.tpu_bind_kv_cache") as bind,
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.has_kv_transfer_group",
                return_value=False,
            ),
        ):
            self.runner.initialize_kv_cache(config)
        caches = bind.call_args.args[0]
        unique = {id(cache): cache for cache in caches.values()}
        assert config.num_blocks == num_blocks
        assert len(unique) == (2 if shared else 4)
        assert sum(cache.nbytes for cache in unique.values()) == budget
        if shared:
            assert caches["full.0"] is caches["swa.0"]
            assert caches["full.0"].shape == (4, 256, 4, page_bytes // 1024)
        else:
            assert caches["full.0"].shape == (4, 256, 1, 4, 512)
            assert caches["swa.0"].shape == (4, 256, sliding_heads // 2, 4, 256)
            assert caches["full.0"] is not caches["swa.0"]
        assert caches["swa.0"] is caches["swa.1"] is caches["swa.2"]
        assert caches["swa.3"] is caches["swa.4"] is caches["swa.5"]
        caches["swa.0"][-1].fill_(7)
        assert torch.all(caches["swa.2"][-1].float() == 7)
        assert torch.count_nonzero(caches["swa.0"][:-1].float()) == 0
        assert torch.count_nonzero(caches["swa.3"].float()) == 0
        if not shared:
            assert torch.count_nonzero(caches["full.0"].float()) == 0

    @pytest.mark.usefixtures("vllm_config_context")
    @pytest.mark.parametrize(
        "full_heads,full_head_size,dtype,backend",
        [
            (2, 128, torch.bfloat16, PallasAttentionBackend),
            (1, 256, torch.bfloat16, PallasBatchedRPAAttentionBackend),
            (1, 256, torch.float8_e4m3fn, PallasBatchedRPAAttentionBackend),
        ],
    )
    def test_initialize_kv_cache_multi_group(
        self, full_heads, full_head_size, dtype, backend
    ):
        """Different attention geometries share the full scheduler block pool."""
        page_bytes = max(
            backend.get_kv_cache_page_size_bytes(16, heads, head_size, dtype)
            for heads, head_size in [(full_heads, full_head_size), (2, 128)]
        )
        full_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=full_heads,
            head_size=full_head_size,
            dtype=dtype,
            page_size_padded=page_bytes,
        )
        sliding_spec = SlidingWindowSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=128,
            dtype=dtype,
            page_size_padded=page_bytes,
            sliding_window=32,
        )
        kv_cache_groups = [
            KVCacheGroupSpec(layer_names=["attn.0"], kv_cache_spec=full_spec),
            KVCacheGroupSpec(layer_names=["attn.1"], kv_cache_spec=sliding_spec),
        ]
        kv_cache_config = KVCacheConfig(
            num_blocks=4,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=page_bytes * 4,
                    layers=["attn.0"],
                    layer_stride=page_bytes * 4,
                    block_stride=page_bytes,
                ),
                KVCacheTensor(
                    size=page_bytes * 4,
                    layers=["attn.1"],
                    layer_stride=page_bytes * 4,
                    block_stride=page_bytes,
                ),
            ],
            kv_cache_groups=kv_cache_groups,
        )

        self.runner.vllm_config.compilation_config.static_forward_context = {}
        self.runner.block_table_cpu = torch.zeros((1, 4), dtype=torch.int32)
        self.runner.kv_caches = []

        mock_input_batch = MagicMock()
        mock_block_tables = []
        for _ in kv_cache_groups:
            mock_bt = MagicMock()
            mock_bt.max_num_blocks_per_req = 4
            mock_bt.get_cpu_tensor.return_value = torch.zeros((1, 4), dtype=torch.int32)
            mock_block_tables.append(mock_bt)
        mock_input_batch.block_table = mock_block_tables
        self.runner.input_batch = mock_input_batch

        with (
            patch("vllm_torchtpu.utils.tpu_bind_kv_cache") as mock_bind,
            patch.object(TpuPlatform, "_find_non_ssm_backend", return_value=backend),
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.has_kv_transfer_group",
                return_value=False,
            ),
        ):
            self.runner.initialize_kv_cache(kv_cache_config)

        # (1) One shared TPU builder per group, in the parent's
        # list-of-list-of-AttentionGroup layout.
        assert len(self.runner.attn_groups) == len(kv_cache_groups)
        for gid, group_list in enumerate(self.runner.attn_groups):
            assert len(group_list) == 1
            grp = group_list[0]
            assert isinstance(grp, AttentionGroup)
            assert grp.backend is None
            assert grp.kv_cache_group_id == gid
            assert len(grp.metadata_builders) == 1
            assert isinstance(grp.metadata_builders[0], AttentionMetadataBuilder)
            assert grp.metadata_builders[0].kv_cache_group_id == gid

        # (2) may_reinitialize_input_batch saw the per-group block_sizes
        # list (replacing the old explicit single-group InputBatch reinit).
        self.runner.may_reinitialize_input_batch.assert_called_once_with(
            kv_cache_config, [16, 16]
        )

        # (3) empty_slot_mappings: one zero-element tensor per kv_cache_group.
        assert set(self.runner.empty_slot_mappings.keys()) == {0, 1}
        for tensor in self.runner.empty_slot_mappings.values():
            assert tensor.numel() == 0

        caches = mock_bind.call_args.args[0]
        first, second = caches["attn.0"], caches["attn.1"]
        assert first is second
        assert first.shape[0] == kv_cache_config.num_blocks
        assert first.untyped_storage().nbytes() == page_bytes * 4
        assert first.stride(0) * dtype.itemsize == page_bytes
        caches["attn.0"][kv_cache_config.num_blocks - 1].fill_(1)
        assert torch.all(caches["attn.1"][-1] == 1)

    def test_initialize_kv_cache_composite_specs(self):
        """Verify composite kv_cache_specs unpacking when a group's spec wraps
        a dictionary of individual per-layer specs."""
        layer_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=128,
            dtype=torch.bfloat16,
            page_size_padded=16384,
        )
        composite_spec = MagicMock()
        composite_spec.kv_cache_specs = {"attn.0": layer_spec}

        kv_cache_groups = [
            KVCacheGroupSpec(layer_names=["attn.0"], kv_cache_spec=composite_spec),
        ]
        kv_cache_config = KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=16384,
                    layers=["attn.0"],
                    layer_stride=16384,
                    block_stride=16384,
                ),
            ],
            kv_cache_groups=kv_cache_groups,
        )

        self.runner.vllm_config.compilation_config.static_forward_context = {}
        self.runner.block_table_cpu = torch.zeros((1, 1), dtype=torch.int32)
        self.runner.kv_caches = []

        mock_input_batch = MagicMock()
        mock_bt = MagicMock()
        mock_bt.max_num_blocks_per_req = 4
        mock_bt.get_cpu_tensor.return_value = torch.zeros((1, 1), dtype=torch.int32)
        mock_input_batch.block_table = [mock_bt]
        self.runner.input_batch = mock_input_batch

        with (
            patch(
                "vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_shape",
                return_value=(1, 16, 2, 1, 128),
            ),
            patch("vllm_torchtpu.utils.tpu_bind_kv_cache") as mock_bind,
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.has_kv_transfer_group",
                return_value=False,
            ),
        ):
            self.runner.initialize_kv_cache(kv_cache_config)

        mock_bind.assert_called_once()
        created_caches = mock_bind.call_args[0][0]
        assert "attn.0" in created_caches
        assert created_caches["attn.0"].shape == (1, 16, 2, 1, 128)

    @pytest.mark.parametrize("allocated_blocks", [9, 10, 11])
    def test_initialize_kv_cache_num_blocks_override(self, allocated_blocks):
        """Validate the scheduler's explicit block count before binding."""
        attn_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=128,
            dtype=torch.bfloat16,
            page_size_padded=16384,
        )
        kv_cache_groups = [
            KVCacheGroupSpec(layer_names=["attn.0"], kv_cache_spec=attn_spec),
        ]

        # num_blocks explicitly provided on kv_cache_config
        kv_cache_config_explicit = KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=16384 * 10 + 123,
                    layers=["attn.0"],
                    layer_stride=16384 * 10 + 123,
                    block_stride=16384,
                ),
            ],
            kv_cache_groups=kv_cache_groups,
        )

        self.runner.vllm_config.compilation_config.static_forward_context = {}
        self.runner.block_table_cpu = torch.zeros((1, 1), dtype=torch.int32)
        self.runner.kv_caches = []

        mock_input_batch = MagicMock()
        mock_bt = MagicMock()
        mock_bt.max_num_blocks_per_req = 4
        mock_bt.get_cpu_tensor.return_value = torch.zeros((1, 1), dtype=torch.int32)
        mock_input_batch.block_table = [mock_bt]
        self.runner.input_batch = mock_input_batch

        with (
            patch(
                "vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_shape",
                return_value=(allocated_blocks, 16, 2, 1, 128),
            ) as mock_get_shape,
            patch("vllm_torchtpu.utils.tpu_bind_kv_cache") as mock_bind,
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.has_kv_transfer_group",
                return_value=False,
            ),
        ):
            if allocated_blocks < kv_cache_config_explicit.num_blocks:
                with pytest.raises(ValueError, match="scheduler can issue block IDs"):
                    self.runner.initialize_kv_cache(kv_cache_config_explicit)
                mock_bind.assert_not_called()
                return
            self.runner.initialize_kv_cache(kv_cache_config_explicit)
            mock_bind.assert_called_once()
            mock_get_shape.assert_called_once_with(
                10,
                attn_spec.block_size,
                attn_spec.num_kv_heads,
                attn_spec.head_size,
                attn_spec.dtype,
            )

    def test_initialize_kv_cache_rejects_unresolved_num_blocks(self):
        # num_blocks is resolved by the scheduler before reaching the runner;
        # an unresolved config must fail loudly instead of being recomputed.
        attn_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=128,
            dtype=torch.bfloat16,
            page_size_padded=16384,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=None,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=16384 * 5,
                    layers=["attn.0"],
                    layer_stride=16384 * 5,
                    block_stride=16384,
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=["attn.0"], kv_cache_spec=attn_spec),
            ],
        )

        with pytest.raises(AssertionError, match="num_blocks"):
            self.runner.initialize_kv_cache(kv_cache_config)


def test_dsv4_layer_classification():
    """Every DSv4 layer is sorted by the kernel that reads its array.

    Mirrors the reference (`tpu_inference/runner/kv_cache_manager.py`):
    MLAAttentionSpec layers anchor the overlays, `*.compressor.state_cache`
    layers are placed afterwards, and SWA layers stay grouped because layers
    of one cache group share a block table. A layer matching none of the
    three is a layout change the overlay plan cannot absorb, so it raises
    rather than being silently left without an array.
    """
    from vllm.v1.kv_cache_interface import SlidingWindowMLASpec

    def mla(compress_ratio, head_size=640):
        return MLAAttentionSpec(
            block_size=1024,
            num_kv_heads=1,
            head_size=head_size,
            dtype=torch.uint8,
            tokens_per_state=compress_ratio,
            alignment=None,
        )

    def swa():
        return SlidingWindowMLASpec(
            block_size=128,
            num_kv_heads=1,
            head_size=640,
            dtype=torch.uint8,
            sliding_window=128,
            alignment=None,
        )

    specs = {
        "m.layers.0.attn": mla(4),
        "m.layers.1.attn": mla(128),
        "m.layers.0.attn.indexer.k_cache": mla(4, head_size=256),
        "m.layers.0.attn.compressor.state_cache": swa(),
        "m.layers.0.attn.swa_cache": swa(),
        "m.layers.1.attn.swa_cache": swa(),
    }
    groups = [
        SimpleNamespace(
            layer_names=[
                "m.layers.0.attn",
                "m.layers.1.attn",
                "m.layers.0.attn.indexer.k_cache",
                "m.layers.0.attn.compressor.state_cache",
            ]
        ),
        SimpleNamespace(
            layer_names=["m.layers.0.attn.swa_cache", "m.layers.1.attn.swa_cache"]
        ),
    ]

    runner = MagicMock()
    runner.shared_kv_cache_layers = {}

    mla_names, swa_groups, state_names = DsV4KVCacheAllocator(
        runner
    )._classify_ds_v4_layers(SimpleNamespace(kv_cache_groups=groups), specs.__getitem__)

    assert mla_names == [
        "m.layers.0.attn",
        "m.layers.1.attn",
        "m.layers.0.attn.indexer.k_cache",
    ]
    assert state_names == ["m.layers.0.attn.compressor.state_cache"]
    # Grouped, not flattened: position within the group picks the host array.
    assert swa_groups == [["m.layers.0.attn.swa_cache", "m.layers.1.attn.swa_cache"]]

    stray = SimpleNamespace(layer_names=["m.layers.0.mystery"])
    with pytest.raises(ValueError, match="no known role"):
        DsV4KVCacheAllocator(runner)._classify_ds_v4_layers(
            SimpleNamespace(kv_cache_groups=groups + [stray]),
            {**specs, "m.layers.0.mystery": swa()}.__getitem__,
        )


def test_dsv4_swa_overlay_shares_hosts_across_groups():
    """SWA caches take a host by position *within* their own cache group.

    Mirrors the reference: `swa_host_indices[position]` with `position`
    restarting at 0 for every group, so the one-SWA-layer-per-group layout
    vLLM actually produces puts every SWA cache on the *same* CSA NoPE array.
    That is safe -- distinct cache groups never own the same block ID at the
    same time -- and it is why `_validate_ds_v4_overlay` checks for collisions
    within a group only. An earlier design forced globally distinct hosts,
    which allocated arrays the reference does not.
    """
    csa_a = torch.zeros(4, 8, 4, 128, dtype=torch.uint8)
    csa_b = torch.zeros(4, 8, 4, 128, dtype=torch.uint8)

    # The MLA layers form one cache group; each SWA cache is its own group,
    # which is the shape vLLM actually produces here. A CSA layer and the SWA
    # cache overlaying its NoPE array therefore never share a block table.
    groups = [
        SimpleNamespace(layer_names=[f"m.layers.{i}.attn" for i in range(3)]),
    ] + [
        SimpleNamespace(layer_names=[f"m.layers.{i}.attn.swa_cache"]) for i in range(3)
    ]
    # Positional-within-group assignment: every group's sole SWA layer is at
    # position 0, so all three share the first CSA NoPE array.
    kv_caches = {
        "m.layers.0.attn": (csa_a, torch.zeros(4, 2, 4, 128, dtype=torch.uint8)),
        "m.layers.1.attn": (csa_b, torch.zeros(4, 2, 4, 128, dtype=torch.uint8)),
        "m.layers.2.attn": torch.zeros(4, 16, 4, 128, dtype=torch.uint8),
        "m.layers.0.attn.swa_cache": csa_a,
        "m.layers.1.attn.swa_cache": csa_a,
        "m.layers.2.attn.swa_cache": csa_a,
    }
    DsV4KVCacheAllocator._validate_ds_v4_overlay(
        SimpleNamespace(kv_cache_groups=groups), kv_caches
    )

    # Two layers of ONE group on one array is the case that does corrupt:
    # they share a block table, so they would write each other's pages.
    one_group = [
        SimpleNamespace(
            layer_names=["m.layers.0.attn.swa_cache", "m.layers.1.attn.swa_cache"]
        )
    ]
    with pytest.raises(ValueError, match="same array"):
        DsV4KVCacheAllocator._validate_ds_v4_overlay(
            SimpleNamespace(kv_cache_groups=one_group), kv_caches
        )


def test_dsv4_specs_are_exempt_from_tpu_normalization():
    """DSv4 specs must reach vLLM verbatim.

    Without the exemption these specs get a `page_size_padded` derived from the
    logical 1024-token `block_size` rather than `storage_block_size`. (The
    packed `uint8` dtype is separately preserved by `_normalize_one_spec`'s
    integer-MLA carve-out, so dtype alone does not distinguish the two paths.)
    """
    from vllm.v1.kv_cache_interface import SlidingWindowMLASpec

    from vllm_torchtpu.kv_cache_spec_normalizer import normalize_kv_cache_specs_for_tpu

    # A DSv4-shaped compressed main-latent spec: 1024-token logical block
    # compressed 4:1, packed as uint8.
    ds_v4_spec = MLAAttentionSpec(
        block_size=1024,
        num_kv_heads=1,
        head_size=640,
        dtype=torch.uint8,
        tokens_per_state=4,
        alignment=None,
    )
    ds_v4_swa_spec = SlidingWindowMLASpec(
        block_size=256,
        num_kv_heads=1,
        head_size=640,
        dtype=torch.uint8,
        sliding_window=128,
        alignment=None,
    )
    specs = {
        "model.layers.0.attn": ds_v4_spec,
        "model.layers.0.attn.swa_cache": ds_v4_swa_spec,
    }

    out = normalize_kv_cache_specs_for_tpu(
        specs,
        torch.float8_e4m3fn,
        attention_backend=PallasAttentionBackend,
        exempt_layers=set(specs),
    )

    for name, original in specs.items():
        assert out[name] is original, f"{name} was normalized"
        assert out[name].dtype == torch.uint8
        assert out[name].page_size_padded is None

    # Without the exemption the same specs are rewritten -- guards against the
    # exemption silently becoming a no-op.
    unexempt = normalize_kv_cache_specs_for_tpu(
        specs, torch.float8_e4m3fn, attention_backend=PallasAttentionBackend
    )
    assert unexempt["model.layers.0.attn"] is not ds_v4_spec
    assert unexempt["model.layers.0.attn"].page_size_padded is not None
    # A page sized from the logical block, not from the compressed rows: this
    # is exactly the over-allocation the exemption avoids.
    assert (
        unexempt["model.layers.0.attn"].page_size_padded
        > ds_v4_spec.block_size // ds_v4_spec.tokens_per_state * 640
    )


def test_dsv4_groups_are_uniform_type_with_mixed_block_sizes():
    """vLLM hands DSv4 its groups wrapped in `UniformTypeKVCacheSpecs`.

    Two consequences for `initialize_kv_cache`, both exercised here: the group
    spec must be unwrapped before any `isinstance` type test, and block size is
    not uniform across groups, so it cannot be asserted equal.
    """
    from vllm.v1.core.kv_cache_utils import get_kv_cache_groups
    from vllm.v1.kv_cache_interface import (
        AttentionSpec,
        MambaSpec,
        SlidingWindowMLASpec,
        UniformTypeKVCacheSpecs,
    )

    # DSv4's real shape: a 1024-token main latent block compressed 4:1, and
    # SWA caches whose block size is that compressed row count (256), capped
    # by the window. Two window sizes give two distinct SWA groups.
    specs: dict = {}
    for i in range(2):
        specs[f"m.layers.{i}.attn"] = MLAAttentionSpec(
            block_size=1024,
            num_kv_heads=1,
            head_size=640,
            dtype=torch.uint8,
            tokens_per_state=4,
            cache_dtype_str="fp8_ds_mla",
            model_version="deepseek_v4",
            alignment=None,
        )
        for window, block in ((256, 256), (128, 128)):
            specs[f"m.layers.{i}.attn.swa_cache_w{window}"] = SlidingWindowMLASpec(
                block_size=block,
                num_kv_heads=1,
                head_size=640,
                dtype=torch.uint8,
                sliding_window=window,
                cache_dtype_str="fp8_ds_mla",
                model_version="deepseek_v4",
                alignment=None,
            )

    config = SimpleNamespace(
        cache_config=CacheConfig(),
        speculative_config=None,
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
    )
    config.cache_config.kv_cache_layout = "BLHNC"
    grouped = [group.kv_cache_spec for group in get_kv_cache_groups(config, specs)]
    assert grouped is not None, (
        "vLLM did not take its DeepseekV4 grouping path for DSv4-shaped specs"
    )
    assert all(isinstance(g, UniformTypeKVCacheSpecs) for g in grouped), (
        f"expected UniformTypeKVCacheSpecs groups, got "
        f"{[type(g).__name__ for g in grouped]}"
    )

    # A bare isinstance test against the group spec sees neither AttentionSpec
    # nor MambaSpec, which is why the representative spec is peeked at.
    assert not any(isinstance(g, (AttentionSpec, MambaSpec)) for g in grouped)

    block_sizes = {g.block_size for g in grouped}
    assert len(block_sizes) > 1, (
        f"expected mixed block sizes across DSv4 groups, got {block_sizes}"
    )


def test_dsv4_is_cache_for_ds_v4_predicate():
    """The exemption must cover all four DSv4 cache-owning module types."""
    from vllm.models.deepseek_v4.attention import (
        DeepseekV4Attention,
        DeepseekV4IndexerCache,
    )
    from vllm.models.deepseek_v4.compressor import CompressorStateCache
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache

    from vllm_torchtpu.runner.kv_cache_manager import is_cache_for_ds_v4

    for cls in (
        DeepseekV4Attention,
        DeepseekV4SWACache,
        DeepseekV4IndexerCache,
        CompressorStateCache,
    ):
        assert is_cache_for_ds_v4(MagicMock(spec=cls)), cls.__name__

    assert not is_cache_for_ds_v4(MagicMock(spec=Attention))
    assert not is_cache_for_ds_v4(MagicMock(spec=MLAAttention))


def test_dsv4_compressor_state_cache_overlay_target():
    """A compressor state cache is named after the layer whose records it joins.

    Mirrors how vLLM wires `DeepseekCompressor.k_cache_prefix`: the main
    compressor's state rides on its own attention layer's array, the indexer
    compressor's on the indexer's `k_cache`. The mapping is derived from the
    layer NAME before anything is allocated -- an overlaid cache that gets its
    own array still costs its HBM at peak, even if aliased away afterwards.
    """
    assert (
        DsV4KVCacheAllocator._ds_v4_compressed_kv_layer_name(
            "model.layers.0.self_attn.compressor.state_cache"
        )
        == "model.layers.0.self_attn"
    )

    assert (
        DsV4KVCacheAllocator._ds_v4_compressed_kv_layer_name(
            "model.layers.0.attn.indexer.compressor.state_cache"
        )
        == "model.layers.0.attn.indexer.k_cache"
    )

    # A renamed state cache would otherwise derive a host that does not exist
    # and overlay onto whatever happened to be there.
    with pytest.raises(ValueError, match="unexpected layer"):
        DsV4KVCacheAllocator._ds_v4_compressed_kv_layer_name(
            "model.layers.0.self_attn.compressor.state"
        )


def test_dsv4_state_cache_overlays_its_own_compressed_kv_array():
    """CSA / indexer states share their own layer's array; HCA states do not.

    An HCA page holds two token states, far too small to host the f32 state
    rows, so those overlay a CSA NoPE array instead. Both cases must survive
    the group collision check: the state cache and its compressed-KV layer are
    in *different* cache groups, which is what makes sharing one array safe.
    """
    csa_nope = torch.zeros(4, 8, 4, 128, dtype=torch.uint8)
    csa_rope = torch.zeros(4, 2, 4, 128, dtype=torch.uint8)
    indexer = torch.zeros(4, 2, 4, 256, dtype=torch.uint8)
    hca = torch.zeros(4, 16, 4, 128, dtype=torch.uint8)

    kv_caches = {
        "m.layers.0.attn": (csa_nope, csa_rope),
        "m.layers.0.attn.indexer.k_cache": indexer,
        "m.layers.1.attn": hca,
        # CSA and indexer states: their own layer's array.
        "m.layers.0.attn.compressor.state_cache": csa_nope,
        "m.layers.0.attn.indexer.compressor.state_cache": indexer,
        # HCA state: a CSA NoPE array, its own page being too small.
        "m.layers.1.attn.compressor.state_cache": csa_nope,
    }
    groups = [
        SimpleNamespace(
            layer_names=[
                "m.layers.0.attn",
                "m.layers.0.attn.indexer.k_cache",
                "m.layers.1.attn",
            ]
        ),
        SimpleNamespace(
            layer_names=[
                "m.layers.0.attn.compressor.state_cache",
                "m.layers.0.attn.indexer.compressor.state_cache",
            ]
        ),
        SimpleNamespace(layer_names=["m.layers.1.attn.compressor.state_cache"]),
    ]
    DsV4KVCacheAllocator._validate_ds_v4_overlay(
        SimpleNamespace(kv_cache_groups=groups), kv_caches
    )

    # The CSA layer is matched on its NoPE array, not its RoPE companion, so a
    # same-group layer landing on that NoPE array is still caught.
    clash = [
        SimpleNamespace(layer_names=["m.layers.0.attn", "m.layers.0.attn.swa_cache"])
    ]
    with pytest.raises(ValueError, match="same array"):
        DsV4KVCacheAllocator._validate_ds_v4_overlay(
            SimpleNamespace(kv_cache_groups=clash),
            {**kv_caches, "m.layers.0.attn.swa_cache": csa_nope},
        )


def test_dsv4_csa_allocates_int32_nope_and_rope_arrays():
    """A CSA layer gets int32 NoPE and RoPE arrays (see `csa_cache_layout`).

    NoPE keeps one 128-word row per token, RoPE packs four tokens' 32 words
    per row, so the sparse gather can fetch a token's own 128 RoPE bytes. The
    bytes are the uint8 pair's; overflow overlay hosts copy the NoPE host.
    """
    from vllm.v1.kv_cache_interface import SlidingWindowMLASpec

    def mla(compress_ratio, head_size=640):
        return MLAAttentionSpec(
            block_size=256,
            num_kv_heads=1,
            head_size=head_size,
            dtype=torch.uint8,
            tokens_per_state=compress_ratio,
            alignment=None,
        )

    def swa():
        return SlidingWindowMLASpec(
            block_size=64,
            num_kv_heads=1,
            head_size=640,
            dtype=torch.uint8,
            sliding_window=128,
            alignment=None,
        )

    specs = {
        "m.layers.0.attn": mla(4),
        "m.layers.0.attn.indexer.k_cache": mla(4, head_size=256),
        "m.layers.1.attn": mla(128),
        "m.layers.0.attn.compressor.state_cache": swa(),
        "m.layers.1.attn.compressor.state_cache": swa(),
        "m.layers.0.attn.swa_cache": swa(),
        "m.layers.1.attn.swa_cache": swa(),
    }
    groups = [
        SimpleNamespace(
            layer_names=[
                "m.layers.0.attn",
                "m.layers.0.attn.indexer.k_cache",
                "m.layers.1.attn",
            ]
        ),
        SimpleNamespace(layer_names=["m.layers.0.attn.compressor.state_cache"]),
        SimpleNamespace(layer_names=["m.layers.1.attn.compressor.state_cache"]),
        SimpleNamespace(
            layer_names=["m.layers.0.attn.swa_cache", "m.layers.1.attn.swa_cache"]
        ),
    ]
    runner = MagicMock()
    runner.shared_kv_cache_layers = {}
    runner.device = "cpu"
    kv_caches: dict = {}
    num_blocks = 3

    DsV4KVCacheAllocator(runner)._initialize_ds_v4_kv_cache(
        SimpleNamespace(kv_cache_groups=groups),
        specs.__getitem__,
        kv_caches,
        num_blocks,
    )

    # T = 256 / 4 = 64 compressed tokens per page.
    nope, rope = kv_caches["m.layers.0.attn"]
    assert nope.dtype == rope.dtype == torch.int32
    assert tuple(nope.shape) == (num_blocks, 64, 128)
    assert tuple(rope.shape) == (num_blocks, 16, 128)
    assert nope.numel() * nope.element_size() == num_blocks * 64 * 512
    assert rope.numel() * rope.element_size() == num_blocks * 64 * 128
    # The CSA state rides its own layer's entry; the HCA state overlays the
    # CSA NoPE array.
    assert (
        kv_caches["m.layers.0.attn.compressor.state_cache"]
        is (kv_caches["m.layers.0.attn"])
    )
    assert kv_caches["m.layers.1.attn.compressor.state_cache"] is nope
    # Two SWA layers of one group: the first overlays the CSA NoPE array, the
    # second gets a standalone host of the same shape and dtype.
    assert kv_caches["m.layers.0.attn.swa_cache"] is nope
    overflow = kv_caches["m.layers.1.attn.swa_cache"]
    assert overflow is not nope
    assert overflow.dtype == torch.int32
    assert tuple(overflow.shape) == tuple(nope.shape)


class TestUpdateAttentionPageSizePadded:
    """_update_attention_page_size_padded must size pages with the caller's
    block_size instead of self.block_size, which is initialized at __init__
    time. This is because for hybrid models the update is skipped during that
    time. It only gets updated after the model is loaded, so using
    self.block_size will cause the unify_kv_cache_spec_page_size
    `page_size_padded >= real_page_size_bytes` assertion.
    """

    @pytest.fixture(autouse=True)
    def mock_non_ssm_backend(self):
        with patch.object(
            TpuPlatform, "_find_non_ssm_backend", return_value=PallasAttentionBackend
        ):
            yield

    @staticmethod
    def _attn(num_kv_heads, head_size):
        layer = MagicMock(spec=Attention)
        layer.num_kv_heads = num_kv_heads
        layer.head_size = head_size
        return layer

    @staticmethod
    def _stub(stale_block_size=16):
        return SimpleNamespace(
            block_size=stale_block_size,
            kv_cache_dtype=torch.bfloat16,
            cache_config=SimpleNamespace(mamba_page_size_padded=None),
            _hybrid_uniform_page_size_bytes=None,
            vllm_config=MagicMock(),
        )

    def test_uses_caller_block_size_not_stale_snapshot(self):
        # gemma-4 E4B geometry: sliding head_dim 256 vs global head_dim 512.
        layers = {
            "sliding": self._attn(num_kv_heads=2, head_size=256),
            "full": self._attn(num_kv_heads=2, head_size=512),
        }
        stub = self._stub(stale_block_size=16)

        KVCacheManager(stub)._update_attention_page_size_padded(layers, 256)

        expected = PallasAttentionBackend.get_kv_cache_page_size_bytes(
            256, 2, 512, torch.bfloat16
        )
        stale = PallasAttentionBackend.get_kv_cache_page_size_bytes(
            16, 2, 512, torch.bfloat16
        )
        assert stub._hybrid_uniform_page_size_bytes == expected
        assert stub._hybrid_uniform_page_size_bytes != stale
        assert stub.cache_config.mamba_page_size_padded == expected

    def test_uniform_layers_leave_pinning_unset(self):
        layers = {
            "a": self._attn(num_kv_heads=2, head_size=256),
            "b": self._attn(num_kv_heads=2, head_size=256),
        }
        stub = self._stub()

        KVCacheManager(stub)._update_attention_page_size_padded(layers, 256)

        assert stub._hybrid_uniform_page_size_bytes is None
        assert (
            stub.cache_config.mamba_page_size_padded
            == PallasAttentionBackend.get_kv_cache_page_size_bytes(
                256, 2, 256, torch.bfloat16
            )
        )


class TestKVCacheManagerFullCoverage(TestKVCacheManager):
    def test_is_cache_for_ds_v4_and_cover_block_ids_skips(self):
        from vllm_torchtpu.runner.kv_cache_manager import is_cache_for_ds_v4

        native_mod = SimpleNamespace(tpu_native_kv=True)
        assert not is_cache_for_ds_v4(native_mod)

        mamba_spec = MambaSpec(
            block_size=16,
            shapes=[(4, 128)],
            dtypes=[torch.bfloat16],
        )
        mla_spec = MLAAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
        )
        tensor = KVCacheTensor(
            size=4096,
            layers=["mamba.0", "mla.0"],
            layer_stride=2048,
            block_stride=256,
        )
        check_kv_caches_cover_block_ids(
            {},
            [tensor],
            4,
            {"mamba.0": mamba_spec, "mla.0": mla_spec}.__getitem__,
        )

    @patch("vllm_torchtpu.runner.kv_cache_manager.get_layers_from_vllm_config")
    def test_get_kv_cache_spec_all_layer_branches_and_drafter(self, mock_get_layers):
        from vllm.model_executor.layers.attention import ChunkedLocalAttention
        from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
        from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache

        # 1. No attention layers (has_attention=False) and Mamba returning None
        mamba_none = MagicMock(spec=MambaBase)
        mamba_none.get_kv_cache_spec.return_value = None
        dsv4_none = SimpleNamespace(
            tpu_native_kv=True, get_kv_cache_spec=lambda cfg: None
        )
        dsv4_some = SimpleNamespace(
            tpu_native_kv=True,
            get_kv_cache_spec=lambda cfg: MLAAttentionSpec(
                block_size=16, num_kv_heads=1, head_size=128, dtype=torch.uint8
            ),
        )
        indexer = MagicMock(spec=DeepseekV32IndexerCache)
        indexer.tpu_native_kv = False
        indexer.get_kv_cache_spec.return_value = MLAAttentionSpec(
            block_size=16, num_kv_heads=1, head_size=128, dtype=torch.uint8
        )
        other_layer = MagicMock(spec=AttentionLayerBase)
        other_layer.tpu_native_kv = False

        mock_get_layers.return_value = {
            "mamba.none": mamba_none,
            "dsv4.none": dsv4_none,
            "dsv4.some": dsv4_some,
            "indexer.0": indexer,
            "other.0": other_layer,
        }
        specs = self.runner.get_kv_cache_spec()
        assert "dsv4.some" in specs
        assert "indexer.0" in specs
        assert "mamba.none" not in specs
        assert "dsv4.none" not in specs

        # 2. Shared KV target error branches
        bad_missing = _attention_layer_mock()
        bad_missing.num_kv_heads = 2
        bad_missing.head_size = 128
        bad_missing.kv_sharing_target_layer_name = "nonexistent"
        mock_get_layers.return_value = {"attn.0": bad_missing}
        with pytest.raises(ValueError, match="missing target layer"):
            self.runner.get_kv_cache_spec()

        bad_non_attn = _attention_layer_mock()
        bad_non_attn.num_kv_heads = 2
        bad_non_attn.head_size = 128
        bad_non_attn.kv_sharing_target_layer_name = "mamba.0"
        mock_get_layers.return_value = {
            "mamba.0": DummyMamba(),
            "attn.0": bad_non_attn,
        }
        with (
            patch.object(
                self.runner.kv_cache_manager, "_update_mamba_page_size_padded"
            ),
            pytest.raises(ValueError, match="non-attention target layer"),
        ):
            self.runner.get_kv_cache_spec()

        chained_target = _attention_layer_mock()
        chained_target.num_kv_heads = 2
        chained_target.head_size = 128
        chained_target.kv_sharing_target_layer_name = "attn.root"
        chained_leaf = _attention_layer_mock()
        chained_leaf.num_kv_heads = 2
        chained_leaf.head_size = 128
        chained_leaf.kv_sharing_target_layer_name = "attn.mid"
        mock_get_layers.return_value = {
            "attn.leaf": chained_leaf,
            "attn.mid": chained_target,
        }
        with pytest.raises(ValueError, match="itself a shared KV layer"):
            self.runner.get_kv_cache_spec()

        # 3. ChunkedLocalAttention, SlidingWindow, ENCODER, ENCODER_DECODER,
        # unknown attn_type, duplicate MLA
        chunked = MagicMock(spec=ChunkedLocalAttention)
        chunked.tpu_native_kv = False
        chunked.kv_sharing_target_layer_name = None
        chunked.attn_type = AttentionType.DECODER
        chunked.num_kv_heads = 2
        chunked.head_size = 128
        chunked.sliding_window = 256

        enc_attn = _attention_layer_mock()
        enc_attn.kv_sharing_target_layer_name = None
        enc_attn.attn_type = AttentionType.ENCODER
        enc_attn.num_kv_heads = 2
        enc_attn.head_size = 128

        mla_mod = MagicMock(spec=MLAAttention)
        mla_mod.tpu_native_kv = False
        mla_mod.head_size = 576

        class _DupItemsDict(dict):
            def items(self):
                base = list(super().items())
                return base + [("mla.0", mla_mod)]

        dup_layers = _DupItemsDict(
            {
                "chunked.0": chunked,
                "enc.0": enc_attn,
                "mla.0": mla_mod,
            }
        )
        self.runner._unified_kv_layout = True
        self.runner._is_async_drafter = True
        draft_layer_0 = SimpleNamespace(self_attn=None)
        draft_layer_1 = SimpleNamespace(
            self_attn=SimpleNamespace(
                attn=SimpleNamespace(kv_sharing_target_layer_name="chunked.0")
            )
        )
        self.runner.drafter = SimpleNamespace(
            draft_model=SimpleNamespace(
                model=SimpleNamespace(layers=[draft_layer_0, draft_layer_1])
            )
        )
        self.runner.shared_kv_cache_layers = {
            "draft_model.layers.1.self_attn.attn": "chunked.0"
        }
        mock_get_layers.return_value = dup_layers
        with patch(
            "vllm_torchtpu.runner.kv_cache_manager.normalize_kv_cache_specs_for_tpu",
            side_effect=lambda s, *a, **kw: s,
        ):
            specs = self.runner.get_kv_cache_spec()
        assert isinstance(specs["chunked.0"], SlidingWindowSpec)
        assert isinstance(specs["mla.0"], MLAAttentionSpec)
        assert "enc.0" not in specs

        # Async drafter error: unregistered shared layer
        self.runner.shared_kv_cache_layers = {}
        with pytest.raises(RuntimeError, match="was not registered as a shared layer"):
            self.runner.get_kv_cache_spec()

        # Async drafter error: draft layer still has its own KVCacheSpec
        self.runner.shared_kv_cache_layers = {
            "draft_model.layers.1.self_attn.attn": "chunked.0"
        }
        dup_layers["draft_model.layers.1.self_attn.attn"] = dsv4_some
        with pytest.raises(RuntimeError, match="still has its own KVCacheSpec"):
            self.runner.get_kv_cache_spec()

        # ENCODER_DECODER and unknown attn_type
        self.runner._is_async_drafter = False
        enc_dec = _attention_layer_mock()
        enc_dec.kv_sharing_target_layer_name = None
        enc_dec.attn_type = AttentionType.ENCODER_DECODER
        enc_dec.num_kv_heads = 2
        enc_dec.head_size = 128
        mock_get_layers.return_value = {"ed.0": enc_dec}
        with pytest.raises(NotImplementedError):
            self.runner.get_kv_cache_spec()

        unknown_attn = _attention_layer_mock()
        unknown_attn.kv_sharing_target_layer_name = None
        unknown_attn.attn_type = "invalid_type"
        unknown_attn.num_kv_heads = 2
        unknown_attn.head_size = 128
        mock_get_layers.return_value = {"unk.0": unknown_attn}
        with pytest.raises(ValueError, match="Unknown attention type"):
            self.runner.get_kv_cache_spec()

    def test_pad_native_attention_regions_and_shared_aliases_and_sizing(self):
        mgr = self.runner.kv_cache_manager

        # _pad_native_attention_regions with len(groups) <= 1 and len(native_pages) <= 1
        spec_a = FullAttentionSpec(
            block_size=16, num_kv_heads=2, head_size=128, dtype=torch.bfloat16
        )
        spec_mla = MLAAttentionSpec(
            block_size=16, num_kv_heads=1, head_size=128, dtype=torch.bfloat16
        )
        with (
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.can_share_attention_cache",
                return_value=False,
            ),
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.get_kv_cache_groups",
                return_value=[SimpleNamespace(kv_cache_spec=spec_a)],
            ),
        ):
            assert mgr._pad_native_attention_regions(
                {"a": spec_a}, PallasAttentionBackend
            ) == {"a": spec_a}

        with (
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.can_share_attention_cache",
                return_value=False,
            ),
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.get_kv_cache_groups",
                return_value=[
                    SimpleNamespace(kv_cache_spec=spec_mla),
                    SimpleNamespace(kv_cache_spec=spec_mla),
                ],
            ),
        ):
            assert mgr._pad_native_attention_regions(
                {"m0": spec_mla, "m1": spec_mla}, PallasAttentionBackend
            ) == {"m0": spec_mla, "m1": spec_mla}

        # _add_shared_kv_cache_aliases errors
        self.runner.shared_kv_cache_layers = {"shared.0": "target.0"}
        with pytest.raises(ValueError, match="was allocated its own KV cache"):
            mgr._add_shared_kv_cache_aliases({"shared.0": torch.empty(1)})
        with pytest.raises(ValueError, match="target cache was not allocated"):
            mgr._add_shared_kv_cache_aliases({})

        # _update_attention_page_size_padded empty and MLAAttention
        mgr._update_attention_page_size_padded({}, 16)
        mla_mod = MagicMock(spec=MLAAttention)
        mla_mod.head_size = 576
        mgr._update_attention_page_size_padded(
            {"mla.0": mla_mod, "mamba.0": DummyMamba()}, 16
        )

        # _update_mamba_page_size_padded with no attn, and with attn but no mamba
        mgr._update_mamba_page_size_padded({"mamba.0": DummyMamba()})
        attn_mod = _attention_layer_mock()
        attn_mod.num_kv_heads = 2
        attn_mod.head_size = 128
        mgr._update_mamba_page_size_padded({"attn.0": attn_mod})

        # _available_kv_cache_hbm with explicit kv_cache_memory_bytes
        self.runner.cache_config.kv_cache_memory_bytes = 65536
        assert mgr._available_kv_cache_hbm() == 65536

        # _maybe_set_compact_mamba_num_blocks_override early returns & error
        self.runner._uniform_mamba_layout = True
        mgr._maybe_set_compact_mamba_num_blocks_override(4096, 1024, 1, 1, 1)
        self.runner._uniform_mamba_layout = False
        self.runner.cache_config.num_gpu_blocks_override = 10
        mgr._maybe_set_compact_mamba_num_blocks_override(4096, 1024, 1, 1, 1)
        self.runner.cache_config.num_gpu_blocks_override = None
        mgr._maybe_set_compact_mamba_num_blocks_override(4096, 1024, 1, 1, 0)
        self.runner.cache_config.kv_cache_memory_bytes = -1
        mgr._maybe_set_compact_mamba_num_blocks_override(4096, 1024, 1, 1, 1)
        self.runner.cache_config.kv_cache_memory_bytes = 17 * 1024 + 100
        with pytest.raises(ValueError, match="no attention blocks remain"):
            mgr._maybe_set_compact_mamba_num_blocks_override(4096, 1024, 1, 1, 1)

        # _maybe_set_num_blocks_override branches and fallback
        self.runner.cache_config.num_gpu_blocks_override = 10
        mgr._maybe_set_num_blocks_override(4096, 8192, 1)
        self.runner.cache_config.num_gpu_blocks_override = None
        self.runner.cache_config.kv_cache_memory_bytes = -1
        mgr._maybe_set_num_blocks_override(4096, 8192, 1)
        self.runner.cache_config.kv_cache_memory_bytes = 100
        mgr._maybe_set_num_blocks_override(4096, 8192, 1)
        self.runner.cache_config.kv_cache_memory_bytes = 4096
        mgr._maybe_set_num_blocks_override(4096, 8192, 1)
        self.runner.cache_config.kv_cache_memory_bytes = 100 * 1024 * 1024
        self.runner._uniform_mamba_layout = True
        self.runner._mamba_num_blocks = None
        mgr._update_mamba_page_size_padded(
            {"attn.0": attn_mod, "mamba.0": DummyMamba()}
        )
        assert self.runner.cache_config.num_gpu_blocks_override is not None

    def test_prewarm_and_owner_token_layout_helpers(self):
        mgr = self.runner.kv_cache_manager
        with patch(
            "vllm_torchtpu.runner.kv_cache_manager.has_kv_transfer_group",
            return_value=False,
        ):
            assert mgr.get_kv_prewarm_shapes() == []
            mgr.prewarm_kv_offload_shape(16)

        prewarmed = []
        fake_conn_empty = SimpleNamespace(connector_worker=None)
        fake_conn_full = SimpleNamespace(
            connector_worker=SimpleNamespace(
                spec=SimpleNamespace(
                    prewarm_shapes=[16, 32],
                    prewarm_shape=lambda p: prewarmed.append(p),
                )
            )
        )
        with (
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.has_kv_transfer_group",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.get_kv_transfer_group",
                return_value=fake_conn_empty,
            ),
        ):
            assert mgr.get_kv_prewarm_shapes() == []
            mgr.prewarm_kv_offload_shape(16)

        with (
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.has_kv_transfer_group",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.get_kv_transfer_group",
                return_value=fake_conn_full,
            ),
        ):
            assert mgr.get_kv_prewarm_shapes() == [16, 32]
            mgr.prewarm_kv_offload_shape(64)
            assert prewarmed == [64]

        self.runner._kv_shard_owner_layout = None
        assert mgr.prepare_owner_token_layout({}, [4]) is None

        self.runner._kv_shard_owner_layout = SimpleNamespace()
        self.runner.num_tokens_paddings = (8,)
        with patch(
            "vllm_torchtpu.runner.owner_token_layout.prepare_kv_shard_owner_token_layout",
            return_value="layout_ok",
        ) as mock_prep:
            assert mgr.prepare_owner_token_layout({}, [4]) == "layout_ok"
            assert (
                mgr.prepare_owner_token_layout({}, [4], token_paddings=(16,))
                == "layout_ok"
            )
            assert mock_prep.call_count == 2

    def test_unified_kv_cache_and_block_major_pool_methods(self):
        from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackendImpl
        from vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op import (
            VllmGatedDeltaNetAttention,
        )

        mgr = self.runner.kv_cache_manager

        # _check_block_major_pool_supported branches
        self.runner.speculative_config = SimpleNamespace()
        with pytest.raises(NotImplementedError, match="speculative decoding"):
            mgr._check_block_major_pool_supported()
        self.runner.speculative_config = None

        self.runner.parallel_config = SimpleNamespace(
            decode_context_parallel_size=2,
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
        )
        with pytest.raises(NotImplementedError, match="decode context parallelism"):
            mgr._check_block_major_pool_supported()

        self.runner.parallel_config.decode_context_parallel_size = 1
        self.runner.parallel_config.pipeline_parallel_size = 2
        with pytest.raises(NotImplementedError, match="pipeline parallelism"):
            mgr._check_block_major_pool_supported()

        self.runner.parallel_config.pipeline_parallel_size = 1
        self.runner.use_spmd = True
        with pytest.raises(NotImplementedError, match="SPMD"):
            mgr._check_block_major_pool_supported()
        self.runner.use_spmd = False
        mgr._check_block_major_pool_supported()

        # _bind_block_major_pool_layers branches
        gdn_layer = MagicMock(spec=VllmGatedDeltaNetAttention)
        pallas_impl = MagicMock(spec=PallasAttentionBackendImpl)
        attn_layer = SimpleNamespace(impl=pallas_impl)
        bad_layer = SimpleNamespace(impl=SimpleNamespace())
        self.runner.shared_kv_cache_layers = {"attn.shared": "attn.0"}
        bm_layout = SimpleNamespace(pool_index_by_layer={"gdn.0": 0, "attn.0": 1})
        with patch(
            "vllm_torchtpu.runner.kv_cache_manager.get_layers_from_vllm_config",
            return_value={
                "gdn.0": gdn_layer,
                "attn.0": attn_layer,
                "attn.shared": attn_layer,
            },
        ):
            mgr._bind_block_major_pool_layers(bm_layout)
        gdn_layer.set_block_major_pool.assert_called_once_with(0, bm_layout)
        assert pallas_impl.set_block_major_pool.call_count == 2

        self.runner.shared_kv_cache_layers = {}
        with (
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.get_layers_from_vllm_config",
                return_value={"bad.0": bad_layer},
            ),
            pytest.raises(
                NotImplementedError, match="cannot address the block-major unified pool"
            ),
        ):
            mgr._bind_block_major_pool_layers(
                SimpleNamespace(pool_index_by_layer={"bad.0": 0})
            )

        # _initialize_unified_kv_cache via initialize_kv_cache
        full_spec = FullAttentionSpec(
            block_size=16, num_kv_heads=2, head_size=128, dtype=torch.bfloat16
        )
        sw_spec = SlidingWindowSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=128,
            dtype=torch.bfloat16,
            sliding_window=64,
        )
        mamba_spec = MambaSpec(
            block_size=16,
            shapes=[(4, 128)],
            dtypes=[torch.bfloat16],
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=4096,
                    layers=["attn.0", "sw.0", "mamba.0"],
                    layer_stride=1024,
                    block_stride=512,
                )
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=["attn.0"], kv_cache_spec=full_spec),
                KVCacheGroupSpec(layer_names=["sw.0"], kv_cache_spec=sw_spec),
                KVCacheGroupSpec(layer_names=["mamba.0"], kv_cache_spec=mamba_spec),
            ],
        )
        self.runner._unified_kv_layout = True
        self.runner.use_spmd = True
        self.runner.shared_kv_cache_layers = {"attn.shared": "attn.0"}
        self.runner.vllm_config.compilation_config.static_forward_context = {}
        self.runner.block_table_cpu = torch.zeros((1, 1), dtype=torch.int32)
        mock_bt = MagicMock()
        mock_bt.get_cpu_tensor.return_value = torch.zeros((1, 1), dtype=torch.int32)
        self.runner.input_batch = SimpleNamespace(block_table=[mock_bt] * 3)
        self.runner._resolve_tpu_group_backend.return_value = PallasAttentionBackend

        raw_tensor = torch.zeros((8, 16, 4, 128), dtype=torch.bfloat16)
        bm_pool = SimpleNamespace(
            num_pools=2,
            split=1,
            pool_index_by_layer={"attn.0": 0},
        )
        mat_with_bm = SimpleNamespace(
            kv_caches={
                "attn.0": raw_tensor,
                "sw.0": raw_tensor,
                "mamba.0": raw_tensor,
            },
            raw_tensors=[raw_tensor],
            bm_pool_layout=bm_pool,
            layer_to_raw={"attn.0": 0, "sw.0": 0, "mamba.0": 0},
        )
        fake_conn_full = SimpleNamespace(
            register_kv_caches=MagicMock(),
            set_host_xfer_buffer_ops=MagicMock(),
            register_runner=MagicMock(),
        )
        fake_conn_min = SimpleNamespace(register_kv_caches=MagicMock())

        with (
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.prepare_kernel_block_sizes",
                return_value=[16, 16, None],
            ),
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.build_kernel_block_size_by_group_id",
                return_value={0: 16, 1: 16, 2: None},
            ),
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.unified_block_major_enabled",
                return_value=True,
            ),
            patch.object(mgr, "_check_block_major_pool_supported"),
            patch.object(mgr, "_bind_block_major_pool_layers"),
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.materialize_kv_cache_tensors",
                return_value=mat_with_bm,
            ),
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.format_kv_cache_layout_summary",
                return_value="summary",
            ),
            patch("vllm_torchtpu.utils.tpu_bind_kv_cache"),
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.has_kv_transfer_group",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.get_kv_transfer_group",
                return_value=fake_conn_full,
            ),
            patch.object(
                mgr,
                "_maybe_add_kv_sharing_layers_to_kv_cache_groups",
            ),
        ):
            self.runner.speculative_config = SimpleNamespace()
            self.runner.enforce_eager = False
            self.runner.initialize_kv_cache(kv_cache_config)
            assert self.runner._attention_kernel_block_size == 16
            fake_conn_full.register_kv_caches.assert_called_once()
            fake_conn_full.set_host_xfer_buffer_ops.assert_called_once()
            fake_conn_full.register_runner.assert_called_once()

        # Second pass: block_major=False, bm_pool_layout=None, empty raw_tensors,
        # minimal connector, enforce_eager=True
        mat_no_bm = SimpleNamespace(
            kv_caches={
                "attn.0": raw_tensor,
                "sw.0": raw_tensor,
                "mamba.0": raw_tensor,
            },
            raw_tensors=[],
            bm_pool_layout=None,
            layer_to_raw={},
        )
        with (
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.prepare_kernel_block_sizes",
                return_value=[None, None, None],
            ),
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.build_kernel_block_size_by_group_id",
                return_value={},
            ),
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.unified_block_major_enabled",
                return_value=False,
            ),
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.materialize_kv_cache_tensors",
                return_value=mat_no_bm,
            ),
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.format_kv_cache_layout_summary",
                return_value="summary",
            ),
            patch("vllm_torchtpu.utils.tpu_bind_kv_cache"),
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.has_kv_transfer_group",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.get_kv_transfer_group",
                return_value=fake_conn_min,
            ),
            patch.object(
                mgr,
                "_maybe_add_kv_sharing_layers_to_kv_cache_groups",
            ),
        ):
            self.runner.use_spmd = False
            self.runner.speculative_config = None
            self.runner.enforce_eager = True
            self.runner.initialize_kv_cache(kv_cache_config)
            fake_conn_min.register_kv_caches.assert_called_once()

        with (
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.prepare_kernel_block_sizes",
                return_value=[None, None, None],
            ),
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.build_kernel_block_size_by_group_id",
                return_value={},
            ),
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.unified_block_major_enabled",
                return_value=False,
            ),
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.materialize_kv_cache_tensors",
                return_value=mat_no_bm,
            ),
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.format_kv_cache_layout_summary",
                return_value="summary",
            ),
            patch("vllm_torchtpu.utils.tpu_bind_kv_cache"),
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.has_kv_transfer_group",
                return_value=False,
            ),
            patch.object(
                mgr,
                "_maybe_add_kv_sharing_layers_to_kv_cache_groups",
            ),
        ):
            self.runner.initialize_kv_cache(kv_cache_config)

    def test_initialize_kv_cache_remaining_branches(self, monkeypatch):
        from vllm.v1.kv_cache_interface import (
            KVCacheSpec,
            UniformTypeKVCacheSpecs,
        )

        from vllm_torchtpu import envs
        from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackendImpl

        mgr = self.runner.kv_cache_manager
        self.runner._unified_kv_layout = False
        self.runner.block_table_cpu = torch.zeros((1, 1), dtype=torch.int32)
        mock_bt = MagicMock()
        mock_bt.get_cpu_tensor.return_value = torch.zeros((1, 1), dtype=torch.int32)

        # 1. Unsupported spec in >1 kv_cache_groups
        circ_spec = MagicMock(spec=KVCacheSpec, block_size=16)
        full_spec = FullAttentionSpec(
            block_size=16, num_kv_heads=2, head_size=128, dtype=torch.bfloat16
        )
        bad_multi_cfg = KVCacheConfig(
            num_blocks=4,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=["c.0"], kv_cache_spec=circ_spec),
                KVCacheGroupSpec(layer_names=["a.0"], kv_cache_spec=full_spec),
            ],
        )
        with pytest.raises(
            NotImplementedError,
            match="Only AttentionSpec and MambaSpec are supported",
        ):
            mgr.initialize_kv_cache(bad_multi_cfg)

        # 2. Unsupported spec in single group (line 1734)
        self.runner.input_batch = SimpleNamespace(block_table=[mock_bt])
        self.runner.vllm_config.compilation_config.static_forward_context = {}
        bad_single_cfg = KVCacheConfig(
            num_blocks=4,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=4096,
                    layers=["c.0"],
                    layer_stride=4096,
                    block_stride=1024,
                )
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=["c.0"], kv_cache_spec=circ_spec),
            ],
        )
        with pytest.raises(NotImplementedError):
            mgr.initialize_kv_cache(bad_single_cfg)

        # 3. Upstream allocator + UniformTypeKVCacheSpecs
        uni_spec = UniformTypeKVCacheSpecs(
            block_size=16,
            kv_cache_specs={"a.0": full_spec},
        )
        up_cfg = KVCacheConfig(
            num_blocks=4,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=4096,
                    layers=["a.0"],
                    layer_stride=4096,
                    block_stride=1024,
                )
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=["a.0"], kv_cache_spec=uni_spec),
            ],
        )
        self.runner.vllm_config.compilation_config.static_forward_context = {
            "a.0": SimpleNamespace(tpu_native_kv=True)
        }
        fake_mat = SimpleNamespace(
            kv_caches={"a.0": torch.zeros((4, 16, 2, 1, 128))},
            raw_tensors=[torch.zeros((4, 16), dtype=torch.uint8)],
            rope_pools="rope_pools",
            native_raw_cache="raw_cache",
            native_cache_layouts="layouts",
        )
        for impl_val, shard_val in (("v2", True), ("v2", False), ("v1", False)):
            monkeypatch.setitem(vars(envs), "TPU_DEEPSEEK_V41_IMPL", impl_val)
            monkeypatch.setitem(vars(envs), "TPU_DEEPSEEK_V41_KV_SHARD", shard_val)
            with (
                patch(
                    "vllm_torchtpu.models.vllm.deepseek_v41.v2.cache.resolve_worker_owner_layout",
                    return_value=SimpleNamespace(num_replicated=1, num_blocks=4),
                ),
                patch(
                    "vllm_torchtpu.models.vllm.deepseek_v41.v2.cache.materialize_upstream_kv_cache",
                    return_value=fake_mat,
                ),
                patch.object(mgr.ds_v4, "initialize_v41", return_value=fake_mat),
                patch("vllm_torchtpu.utils.tpu_bind_kv_cache"),
                patch(
                    "vllm_torchtpu.runner.kv_cache_manager.has_kv_transfer_group",
                    return_value=False,
                ),
            ):
                mgr.initialize_kv_cache(up_cfg)

        # 4. Block-major KV cache unsupported topology checks (lines 1453-1502)
        monkeypatch.setitem(vars(envs), "VLLM_TPU_BLOCK_MAJOR_KV", True)
        self.runner.vllm_config.compilation_config.static_forward_context = {}
        self.runner.parallel_config = SimpleNamespace(
            prefill_context_parallel_size=1,
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
        )

        for attr, val, match_str in (
            ("use_spmd", True, "SPMD"),
            ("speculative_config", SimpleNamespace(), "speculative decoding"),
        ):
            setattr(self.runner, attr, val)
            with pytest.raises(NotImplementedError, match=match_str):
                mgr.initialize_kv_cache(up_cfg)
            setattr(
                self.runner,
                attr,
                False if attr == "use_spmd" else None,
            )

        from vllm.models.deepseek_v4.attention import DeepseekV4Attention

        self.runner.vllm_config.compilation_config.static_forward_context = {
            "a.0": MagicMock(spec=DeepseekV4Attention, tpu_native_kv=False)
        }
        with pytest.raises(NotImplementedError, match="DSv4 packed KV layout"):
            mgr.initialize_kv_cache(up_cfg)
        self.runner.vllm_config.compilation_config.static_forward_context = {}

        self.runner.parallel_config.prefill_context_parallel_size = 2
        with pytest.raises(NotImplementedError, match="prefill context parallelism"):
            mgr.initialize_kv_cache(up_cfg)
        self.runner.parallel_config.prefill_context_parallel_size = 1

        self.runner.parallel_config.pipeline_parallel_size = 2
        with pytest.raises(NotImplementedError, match="pipeline parallelism"):
            mgr.initialize_kv_cache(up_cfg)
        self.runner.parallel_config.pipeline_parallel_size = 1

        with (
            patch.object(self.runner.model_config, "get_head_size", return_value=64),
            pytest.raises(NotImplementedError, match="head_size=64"),
        ):
            mgr.initialize_kv_cache(up_cfg)

        # Heterogeneous attention specs in block-major
        full_spec_256 = FullAttentionSpec(
            block_size=16, num_kv_heads=2, head_size=256, dtype=torch.bfloat16
        )
        het_cfg = KVCacheConfig(
            num_blocks=4,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=8192,
                    layers=["a.0", "a.1"],
                    layer_stride=4096,
                    block_stride=1024,
                )
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=["a.0", "a.1"],
                    kv_cache_spec=UniformTypeKVCacheSpecs(
                        block_size=16,
                        kv_cache_specs={"a.0": full_spec, "a.1": full_spec_256},
                    ),
                )
            ],
        )
        with pytest.raises(NotImplementedError, match="heterogeneous attention specs"):
            mgr.initialize_kv_cache(het_cfg)

        # Non-attention layer in kv_cache_tensor under block-major
        mamba_spec = MambaSpec(
            block_size=16, shapes=[(4, 128)], dtypes=[torch.bfloat16]
        )
        non_attn_spec = MagicMock(spec=KVCacheSpec, block_size=16, page_size_bytes=1024)
        non_attn_tensor_cfg = KVCacheConfig(
            num_blocks=4,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=8192,
                    layers=["a.0", "m.0"],
                    layer_stride=4096,
                    block_stride=1024,
                )
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=["a.0", "m.0"],
                    kv_cache_spec=UniformTypeKVCacheSpecs(
                        block_size=16,
                        kv_cache_specs={"a.0": full_spec, "m.0": non_attn_spec},
                    ),
                )
            ],
        )
        with pytest.raises(
            NotImplementedError, match="containing non-attention layers"
        ):
            mgr.initialize_kv_cache(non_attn_tensor_cfg)

        # Multiple groups under block-major
        multi_grp_bm_cfg = KVCacheConfig(
            num_blocks=4,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=8192,
                    layers=["a.0", "a.1"],
                    layer_stride=4096,
                    block_stride=1024,
                )
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=["a.0"], kv_cache_spec=full_spec),
                KVCacheGroupSpec(layer_names=["a.1"], kv_cache_spec=full_spec),
            ],
        )
        self.runner.input_batch = SimpleNamespace(block_table=[mock_bt, mock_bt])
        with pytest.raises(NotImplementedError, match="single uniform attention group"):
            mgr.initialize_kv_cache(multi_grp_bm_cfg)

        # Block-major with eligible PallasAttentionBackendImpl layer (lines 1782-1791)
        self.runner.input_batch = SimpleNamespace(block_table=[mock_bt])
        self.runner.mesh = None
        pallas_impl = MagicMock(spec=PallasAttentionBackendImpl)
        eligible_attn = SimpleNamespace(impl=pallas_impl)
        with (
            patch(
                "vllm_torchtpu.offload.block_major_layout.block_major_layer_indices",
                return_value={"a.0": 0},
            ),
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.get_layers_from_vllm_config",
                return_value={"a.0": eligible_attn},
            ),
            patch("vllm_torchtpu.utils.tpu_bind_kv_cache"),
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.has_kv_transfer_group",
                return_value=False,
            ),
        ):
            mgr.initialize_kv_cache(up_cfg)
            pallas_impl.setup_bundled.assert_called_once()
        monkeypatch.setitem(vars(envs), "VLLM_TPU_BLOCK_MAJOR_KV", False)

        # 5. _is_ds_v4=True in initialize_kv_cache (line 1536)
        self.runner.vllm_config.compilation_config.static_forward_context = {
            "a.0": MagicMock(spec=DeepseekV4Attention, tpu_native_kv=False)
        }
        with (
            patch.object(mgr.ds_v4, "_initialize_ds_v4_kv_cache") as mock_dsv4_init,
            patch("vllm_torchtpu.utils.tpu_bind_kv_cache"),
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.has_kv_transfer_group",
                return_value=False,
            ),
        ):
            mgr.initialize_kv_cache(up_cfg)
            mock_dsv4_init.assert_called_once()

        # 6. Sparse/dense MLAAttentionSpec + SPMD + KV transfer + mamba spec decode
        mla_spec = MLAAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            page_size_padded=65536,
        )
        sparse_mod = SimpleNamespace(use_sparse=True, tpu_native_kv=False)
        dense_mod = SimpleNamespace(use_sparse=False, tpu_native_kv=False)
        self.runner.vllm_config.compilation_config.static_forward_context = {
            "mla.sparse": sparse_mod,
            "mla.dense0": dense_mod,
            "mla.dense1": dense_mod,
        }
        mla_cfg = KVCacheConfig(
            num_blocks=4,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=65536 * 4,
                    layers=["mla.sparse", "mla.dense0"],
                    layer_stride=0,
                    block_stride=65536,
                ),
                KVCacheTensor(
                    size=65536 * 4,
                    layers=["mla.dense1"],
                    layer_stride=0,
                    block_stride=65536,
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=["mla.sparse", "mla.dense0", "mla.dense1"],
                    kv_cache_spec=mla_spec,
                ),
            ],
        )
        fake_sparse_spec = SimpleNamespace(shape=(4, 16, 128), torch_dtype=torch.int32)
        fake_conn_full = SimpleNamespace(
            register_kv_caches=MagicMock(),
            set_host_xfer_buffer_ops=MagicMock(),
            register_runner=MagicMock(),
        )
        with (
            patch.object(
                PallasMLAttentionBackend,
                "get_sparse_kv_cache_specs",
                return_value=(fake_sparse_spec, fake_sparse_spec),
            ),
            patch.object(
                PallasMLAttentionBackend,
                "get_kv_cache_shape",
                return_value=(4, 16, 1, 128),
            ),
            patch("vllm_torchtpu.runner.kv_cache_manager.mla_dispatch.admit_layer"),
            patch("vllm_torchtpu.utils.tpu_bind_kv_cache"),
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.has_kv_transfer_group",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.get_kv_transfer_group",
                return_value=fake_conn_full,
            ),
        ):
            mgr.initialize_kv_cache(mla_cfg)
            fake_conn_full.register_kv_caches.assert_called_once()
            fake_conn_full.set_host_xfer_buffer_ops.assert_called_once()
            fake_conn_full.register_runner.assert_called_once()

        # SPMD AttentionSpec + minimal connector + mamba speculative decoding
        self.runner.use_spmd = True
        self.runner.speculative_config = SimpleNamespace()
        self.runner._uniform_mamba_layout = True
        self.runner.input_batch = SimpleNamespace(block_table=[mock_bt, mock_bt])
        hybrid_cfg = KVCacheConfig(
            num_blocks=4,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=65536,
                    layers=["a.0", "m.0"],
                    layer_stride=32768,
                    block_stride=8192,
                )
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=["a.0"], kv_cache_spec=full_spec),
                KVCacheGroupSpec(layer_names=["m.0"], kv_cache_spec=mamba_spec),
            ],
        )
        fake_conn_min = SimpleNamespace(register_kv_caches=MagicMock())
        with (
            patch("vllm_torchtpu.utils.tpu_bind_kv_cache"),
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.has_kv_transfer_group",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.runner.kv_cache_manager.get_kv_transfer_group",
                return_value=fake_conn_min,
            ),
        ):
            with pytest.raises(
                NotImplementedError, match="requires the compact mamba slot layout"
            ):
                mgr.initialize_kv_cache(hybrid_cfg)

            self.runner._uniform_mamba_layout = False
            self.runner.enforce_eager = True
            mgr.initialize_kv_cache(hybrid_cfg)
            assert self.runner.mamba_slot_read_offsets.shape == (4,)
