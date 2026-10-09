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

import sys
from types import SimpleNamespace

import pytest
import requests
import torch
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs, tpu_info
from vllm_torchtpu.distributed import dcp, intra_chip, tpu_communicator
from vllm_torchtpu.executors import kv_block_override
from vllm_torchtpu.platforms import pcp_validation


def test_intra_chip_group_and_chip_rank_groups(monkeypatch):
    # Reset module globals
    monkeypatch.setattr(intra_chip, "_group", None)
    monkeypatch.setattr(intra_chip, "_initialized", False)

    # 1. topology is None -> returns None; second call hits _initialized
    monkeypatch.setattr(intra_chip, "get_chip_topology", lambda: None)
    assert intra_chip.get_intra_chip_group() is None
    assert intra_chip.get_intra_chip_group() is None

    # 2. cores_per_chip < 2
    monkeypatch.setattr(
        intra_chip,
        "get_chip_topology",
        lambda: SimpleNamespace(cores_per_chip=1, device_ids=(0, 1)),
    )
    assert intra_chip._chip_rank_groups() is None

    # 3. dist not available or not initialized
    monkeypatch.setattr(
        intra_chip,
        "get_chip_topology",
        lambda: SimpleNamespace(
            cores_per_chip=2,
            device_ids=(0, 1, 2, 3),
            chip_of=lambda r: (r // 2, r % 2),
        ),
    )
    monkeypatch.setattr(intra_chip.dist, "is_available", lambda: True)
    monkeypatch.setattr(intra_chip.dist, "is_initialized", lambda: False)
    assert intra_chip._chip_rank_groups() is None

    # 4. world_size != num_devices
    monkeypatch.setattr(intra_chip.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(intra_chip.dist, "get_world_size", lambda: 2)
    assert intra_chip._chip_rank_groups() is None

    # 5. Ragged chip groups (3 ranks on chip 0, 1 rank on chip 1)
    monkeypatch.setattr(intra_chip.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(
        intra_chip,
        "get_chip_topology",
        lambda: SimpleNamespace(
            cores_per_chip=2,
            device_ids=(0, 1, 2, 3),
            chip_of=lambda r: (0 if r < 3 else 1, r),
        ),
    )
    assert intra_chip._chip_rank_groups() is None

    # 6. Valid 2x2 chip groups -> get_intra_chip_group builds group
    monkeypatch.setattr(intra_chip, "_initialized", False)
    monkeypatch.setattr(
        intra_chip,
        "get_chip_topology",
        lambda: SimpleNamespace(
            cores_per_chip=2,
            device_ids=(0, 1, 2, 3),
            chip_of=lambda r: (r // 2, r % 2),
        ),
    )
    monkeypatch.setattr(intra_chip.dist, "get_backend", lambda grp: "gloo")
    import vllm.distributed.parallel_state as ps

    fake_world = SimpleNamespace(device_group="world_dg", local_rank=1)
    fake_group = SimpleNamespace(rank_in_group=1, world_size=2)
    monkeypatch.setattr(ps, "get_world_group", lambda: fake_world)
    monkeypatch.setattr(
        ps,
        "init_model_parallel_group",
        lambda groups, local_rank, backend, group_name: fake_group,
    )
    grp = intra_chip.get_intra_chip_group()
    assert grp is fake_group
    assert intra_chip.get_intra_chip_group() is fake_group


def test_tpu_communicator_all_paths(monkeypatch):
    def _fake_base_init(self, *args, **kwargs):
        self.use_all2all = kwargs.get("use_all2all", False)
        self.cpu_group = "fake_cpu_group"

    monkeypatch.setattr(
        tpu_communicator.DeviceCommunicatorBase, "__init__", _fake_base_init
    )
    monkeypatch.setattr(
        tpu_communicator.TpuNullAll2AllManager,
        "__init__",
        lambda self, cpu_group: setattr(self, "cpu_group", cpu_group),
    )

    comm_a2a = tpu_communicator.TpuDeviceCommunicator(use_all2all=True)
    assert isinstance(comm_a2a.all2all_manager, tpu_communicator.TpuNullAll2AllManager)
    comm = tpu_communicator.TpuDeviceCommunicator(use_all2all=False)

    class _FakeGroup:
        def __init__(self, world_size: int):
            self.world_size = world_size

        def all_gather(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
            return torch.cat([x, x], dim=dim)

        def reduce_scatter(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
            return x.chunk(2, dim=dim)[0]

    monkeypatch.setattr(tpu_communicator, "get_dp_group", lambda: _FakeGroup(1))
    monkeypatch.setattr(tpu_communicator, "get_ep_group", lambda: _FakeGroup(2))

    h = torch.ones((4, 8))
    logits = torch.ones((4, 4))
    w = torch.ones((4, 2))
    ids = torch.zeros((4, 2), dtype=torch.int32)

    with pytest.raises(NotImplementedError):
        comm.dispatch_router_logits(h, logits, extra_tensors=[h])
    with pytest.raises(NotImplementedError):
        comm.dispatch(h, w, ids, extra_tensors=[h])

    # world_size == 1 (DP group)
    assert comm.dispatch_router_logits(h, logits, is_sequence_parallel=False) == (
        h,
        logits,
    )
    assert comm.dispatch(h, w, ids, is_sequence_parallel=False) == (h, w, ids)
    assert comm.combine(h, is_sequence_parallel=False) is h

    # world_size == 2 (EP group when SP=True, or DP group when patched)
    monkeypatch.setattr(tpu_communicator, "get_dp_group", lambda: _FakeGroup(2))
    h_same, l_same = comm.dispatch_router_logits(h, h, is_sequence_parallel=False)
    assert h_same is l_same and h_same.shape == (8, 8)

    # Non-SP with TP-sliced logits (shape[0] < hidden_states.shape[0])
    sliced_logits = torch.ones((2, 4))
    h_out, l_out = comm.dispatch_router_logits(
        h, sliced_logits, is_sequence_parallel=False
    )
    assert h_out.shape == (8, 8) and l_out is sliced_logits

    # Non-SP with full logits (shape[0] == hidden_states.shape[0])
    h_full, l_full = comm.dispatch_router_logits(h, logits, is_sequence_parallel=False)
    assert h_full.shape == (8, 8) and l_full.shape == (8, 4)

    # SP dispatch & combine
    h_d, w_d, ids_d = comm.dispatch(h, w, ids, is_sequence_parallel=True)
    assert h_d.shape == (8, 8) and w_d.shape == (8, 2) and ids_d.shape == (8, 2)
    c_out = comm.combine(h_d, is_sequence_parallel=True)
    assert c_out.shape == (4, 8)


def test_dcp_module_all_paths(monkeypatch):
    import vllm.distributed.parallel_state as ps

    fake_grp = SimpleNamespace(rank_in_group=3, world_size=4)
    monkeypatch.setattr(ps, "get_dcp_group", lambda: fake_grp, raising=False)
    assert dcp.get_dcp_group() is fake_grp
    assert dcp.get_dcp_rank() == 3
    assert dcp.get_dcp_world_size() == 4

    monkeypatch.setattr(dcp, "get_cp_group_layout", lambda grp: ("layout", grp))
    assert dcp.get_dcp_group_layout() == ("layout", fake_grp)

    monkeypatch.setattr(
        dcp, "get_or_create_cp_mesh", lambda axis_name, group: (axis_name, group)
    )
    assert dcp.get_or_create_dcp_mesh() == (dcp.DCP_AXIS_NAME, fake_grp)
    assert dcp.get_or_create_dcp_mesh("custom_dcp") == ("custom_dcp", fake_grp)

    def _raise_assert():
        raise AssertionError("not initialized")

    monkeypatch.setattr(ps, "get_dcp_group", _raise_assert, raising=False)
    assert dcp.get_dcp_group() is None
    assert dcp.get_dcp_rank() == 0
    assert dcp.get_dcp_world_size() == 1


def test_tpu_info_all_paths(monkeypatch):
    tpu_info.get_chip_version.cache_clear()

    # 1. get_tpu_metadata: 200 non-empty, 404/empty, and RequestException
    monkeypatch.setattr(
        requests,
        "get",
        lambda url, headers: SimpleNamespace(status_code=200, text="v6e-8"),
    )
    assert tpu_info.get_tpu_metadata("accelerator-type") == "v6e-8"

    monkeypatch.setattr(
        requests,
        "get",
        lambda url, headers: SimpleNamespace(status_code=404, text=""),
    )
    assert tpu_info.get_tpu_metadata("accelerator-type") is None

    def _raise_req(url, headers):
        raise requests.RequestException("network error")

    monkeypatch.setattr(requests, "get", _raise_req)
    assert tpu_info.get_tpu_metadata("accelerator-type") is None

    # 2. get_tpu_type & get_num_cores_per_chip
    monkeypatch.setattr(envs, "TPU_ACCELERATOR_TYPE", "v6e-8", raising=False)
    assert tpu_info.get_tpu_type() == "v6e-8"
    assert tpu_info.get_num_cores_per_chip() == 1

    monkeypatch.setattr(envs, "TPU_ACCELERATOR_TYPE", None, raising=False)
    monkeypatch.setattr(tpu_info, "get_tpu_metadata", lambda key="": "tpu7x-16")
    assert tpu_info.get_tpu_type() == "tpu7x-16"
    assert tpu_info.get_num_cores_per_chip() == 2

    # 3. get_chip_version: recognized & unrecognized
    tpu_info.get_chip_version.cache_clear()
    assert tpu_info.get_chip_version("v5litepod-4") == pltpu.ChipVersion.TPU_V5E
    assert tpu_info.get_chip_version("tpu7x-8") == pltpu.ChipVersion.TPU_7X
    assert tpu_info.get_chip_version("unknown-chip") == pltpu.ChipVersion.TPU_7X
    tpu_info.get_chip_version.cache_clear()

    # 4. get_node_name & get_node_worker_id
    monkeypatch.setattr(envs, "TPU_NAME", "node-env", raising=False)
    assert tpu_info.get_node_name() == "node-env"
    monkeypatch.setattr(envs, "TPU_NAME", "", raising=False)
    monkeypatch.setattr(tpu_info, "get_tpu_metadata", lambda key="": "node-meta")
    assert tpu_info.get_node_name() == "node-meta"

    monkeypatch.setattr(envs, "TPU_WORKER_ID", "3", raising=False)
    assert tpu_info.get_node_worker_id() == 3
    monkeypatch.setattr(envs, "TPU_WORKER_ID", None, raising=False)
    monkeypatch.setattr(tpu_info, "get_tpu_metadata", lambda key="": "2")
    assert tpu_info.get_node_worker_id() == 2
    monkeypatch.setattr(tpu_info, "get_tpu_metadata", lambda key="": None)
    assert tpu_info.get_node_worker_id() == 0

    # 5. get_num_chips: /dev/accel*, /dev/vfio, and FileNotFoundError
    monkeypatch.setattr(
        tpu_info.glob, "glob", lambda pat: ["/dev/accel0", "/dev/accel1"]
    )
    assert tpu_info.get_num_chips() == 2

    monkeypatch.setattr(tpu_info.glob, "glob", lambda pat: [])
    monkeypatch.setattr(tpu_info.os, "listdir", lambda path: ["0", "1", "vfio"])
    assert tpu_info.get_num_chips() == 2

    def _raise_fnf(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(tpu_info.os, "listdir", _raise_fnf)
    assert tpu_info.get_num_chips() == 0


def test_pcp_validation_all_paths():
    validator = pcp_validation.PcpStaticSupportValidator

    # 1. None config -> disabled default singleton
    cfg_none = validator.validate_platform_config(None, kv_cache_layout="HND")
    assert not cfg_none.enabled

    def _make_vllm_cfg(
        *,
        pcp_size: int = 2,
        interleave_size: int = 128,
        dcp_size: int = 1,
        pp_size: int = 1,
        is_moe: bool = False,
        ep: bool = False,
        spec_method: str | None = None,
        spec_tokens: int = 1,
    ):
        spec_cfg = (
            None
            if spec_method is None
            else SimpleNamespace(method=spec_method, num_speculative_tokens=spec_tokens)
        )
        return SimpleNamespace(
            parallel_config=SimpleNamespace(
                prefill_context_parallel_size=pcp_size,
                cp_kv_cache_interleave_size=interleave_size,
                decode_context_parallel_size=dcp_size,
                pipeline_parallel_size=pp_size,
                is_moe_model=is_moe,
                enable_expert_parallel=ep,
            ),
            scheduler_config=SimpleNamespace(async_scheduling=True),
            speculative_config=spec_cfg,
        )

    # 2. HND with interleave_size % 128 != 0
    with pytest.raises(ValueError, match="multiple of 128"):
        validator.validate_platform_config(
            _make_vllm_cfg(interleave_size=64), kv_cache_layout="HND"
        )

    # 3. dcp_size > 1
    with pytest.raises(NotImplementedError, match="does not support DCP"):
        validator.validate_platform_config(
            _make_vllm_cfg(dcp_size=2), kv_cache_layout="HND"
        )

    # 4. pipeline_parallel_size != 1
    with pytest.raises(NotImplementedError, match="pipeline parallelism"):
        validator.validate_platform_config(
            _make_vllm_cfg(pp_size=2), kv_cache_layout="HND"
        )

    # 5. is_moe and not expert_parallel
    with pytest.raises(NotImplementedError, match="enable-expert-parallel"):
        validator.validate_platform_config(
            _make_vllm_cfg(is_moe=True, ep=False), kv_cache_layout="HND"
        )

    # 6. speculative_method != "mtp" and num_speculative_tokens != 1
    with pytest.raises(NotImplementedError, match="method=mtp"):
        validator.validate_platform_config(
            _make_vllm_cfg(spec_method="eagle", spec_tokens=1),
            kv_cache_layout="HND",
        )
    with pytest.raises(NotImplementedError, match="num_speculative_tokens=1"):
        validator.validate_platform_config(
            _make_vllm_cfg(spec_method="mtp", spec_tokens=2),
            kv_cache_layout="HND",
        )

    # 7. interleave_size <= 0 (with NHD layout)
    with pytest.raises(ValueError, match="cp_kv_cache_interleave_size > 0"):
        validator.validate_platform_config(
            _make_vllm_cfg(interleave_size=0), kv_cache_layout="NHD"
        )

    # 8. Valid PCP + MoE + EP + MTP=1 config
    valid_cfg = validator.validate_platform_config(
        _make_vllm_cfg(is_moe=True, ep=True, spec_method="mtp", spec_tokens=1),
        kv_cache_layout="HND",
    )
    assert valid_cfg.enabled and valid_cfg.pcp_size == 2


def test_kv_block_override_all_paths():
    reconcile = kv_block_override.reconcile_num_gpu_blocks_override

    # 1. Single-stage: all None, within tolerance, and exceeding tolerance
    assert reconcile([None, None]) is None
    assert reconcile([1000, 1004, None]) == 1000
    with pytest.raises(ValueError, match="workers disagree"):
        reconcile([1000, 1020])

    # 2. Multi-stage: all stages None, missing stage, and valid multi-stage min
    assert reconcile([None, None, None, None], workers_per_stage=2) is None
    with pytest.raises(ValueError, match="pipeline stage"):
        reconcile([1000, 1002, None, None], workers_per_stage=2)
    assert reconcile([1000, 1002, 900, 901], workers_per_stage=2) == 900
    assert sys is not None
