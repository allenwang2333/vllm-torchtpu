# SPDX-License-Identifier: Apache-2.0
"""TPU smoke for the Torch/JAX bridge and in-place striped cache writes."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_tpu  # noqa: F401

from vllm_torchtpu.layers.adapter.attention import PallasMLAttentionBackendImpl


@pytest.mark.parametrize("rank", [0, 1])
def test_dense_decode_bridge(rank):
    layer = SimpleNamespace(kv_cache_quantized_dtype=None,
                            scale=0.125,
                            layer_name=f"k3_dcp_bridge_{rank}")
    with patch("vllm_torchtpu.distributed.dcp.get_dcp_rank",
               return_value=rank):
        op = PallasMLAttentionBackendImpl._build_dense_decode_dcp_op(
            SimpleNamespace(dcp_size=2,
                            native_dense_decode_dcp=True,
                            dcp_interleave_size=16), layer)
    heads = 24
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
    output, lse = op(cache, q, q_pe, kv, pe, lengths, tables, starts)
    assert output.shape == (2, heads, 512)
    assert lse.shape == (2, heads)
    assert torch.isfinite(output.cpu()).all()
    # Rank 1 has no context for the second request; its LSE is -inf.
    assert torch.isfinite(lse.cpu()[0]).all()
    actual_cache = cache.cpu()
    # Token 16 belongs to rank 1; token 15 belongs to rank 0.
    assert actual_cache[0, 0, 0, 0].item() == (2 if rank == 1 else 1)
    assert actual_cache[4, 7, 1, 0].item() == (2 if rank == 0 else 1)
