# SPDX-License-Identifier: Apache-2.0
"""The planner must include native head-packing padding in its allocation."""

import pytest
import torch
from qwen4_exp_test_utils import q4exp_text_config

from vllm_torchtpu.models.vllm.qwen4_exp.qsa import Qwen4ExpQSA

pytestmark = pytest.mark.usefixtures("vllm_config_context")


@pytest.mark.parametrize("dtype", ["bfloat16", "fp8_e4m3"])
@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_planner_covers_native_owner_and_binding(dtype, tp, monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: tp
    )
    model = Qwen4ExpQSA(
        config=q4exp_text_config(hidden_size=16),
        tp_rank=0,
        tp_size=tp,
        max_model_len=128,
        device="cpu",
        cache_dtype=dtype,
    )
    layouts = model.get_cache_owner_layouts(3, 2)
    owners = [
        torch.full(shape, fill, dtype=dt, device="cpu")
        for _, shape, dt, fill in layouts
    ]
    model.bind_kv_cache(owners)
    spec = model.get_kv_cache_spec(None)
    expected_dtype = torch.float8_e4m3fn if dtype == "fp8_e4m3" else torch.bfloat16
    # RPA must consume the persistent owner directly, without a full-cache
    # uint8/FP8 conversion before and after each attention call.
    assert owners[0].dtype == spec.dtype == expected_dtype
    assert owners[0].numel() * owners[0].element_size() == 3 * spec.page_size_bytes
    assert spec.num_kv_heads == max(2 // tp, 1)
    assert len(model.kv_cache) == 4
    with pytest.raises(ValueError):
        model.bind_kv_cache([owners[0][..., :-1], *owners[1:]])
