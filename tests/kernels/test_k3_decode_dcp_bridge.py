# SPDX-License-Identifier: Apache-2.0
"""TPU smoke for the Torch/JAX bridge and in-place striped cache writes."""

from types import SimpleNamespace

import pytest
import torch
import torch_tpu  # noqa: F401

from vllm_torchtpu.layers.adapter.attention import PallasMLAttentionBackendImpl


def test_dense_decode_bridge():
    from vllm.distributed.parallel_state import (get_dcp_group,
                                                 model_parallel_is_initialized)

    if not model_parallel_is_initialized():
        pytest.skip(
            "Requires native model-parallel workers initialized with DCP2")
    if get_dcp_group().world_size != 2:
        pytest.skip("Requires DCP2")
    rank = get_dcp_group().rank_in_group
    layer = SimpleNamespace(kv_cache_quantized_dtype=None,
                            scale=0.125,
                            layer_name=f"k3_dcp_bridge_{rank}")
    op = PallasMLAttentionBackendImpl._build_dense_decode_dcp_op(
        SimpleNamespace(dcp_size=2,
                        native_dense_decode_dcp=True,
                        dcp_interleave_size=16), layer)
    heads = 12
    dtype = torch.bfloat16
    cache = torch.ones((8, 8, 2, 640), dtype=dtype, device="tpu")
    q = torch.ones((2, heads, 512), dtype=dtype, device="tpu")
    q_pe = torch.ones((2, heads, 64), dtype=dtype, device="tpu")
    kv = torch.full((2, 512), 2.0, dtype=dtype, device="tpu")
    pe = torch.full((2, 64), 2.0, dtype=dtype, device="tpu")
    lengths = torch.tensor([17, 16], dtype=torch.int32, device="tpu")
    tables = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]],
                          dtype=torch.int32,
                          device="tpu")
    starts = torch.tensor([0, 1, 2], dtype=torch.int32, device="tpu")
    output = op(cache, q, q_pe, kv, pe, lengths, tables, starts)
    assert output.shape == (2, heads, 512)
    assert torch.isfinite(output.cpu()).all()
    actual_cache = cache.cpu()
    # Token 16 belongs to rank 1; token 15 belongs to rank 0.
    assert actual_cache[0, 0, 0, 0].item() == (2 if rank == 1 else 1)
    assert actual_cache[4, 7, 1, 0].item() == (2 if rank == 0 else 1)
