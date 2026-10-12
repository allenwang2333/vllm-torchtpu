"""Check raw-row metadata inside a compiled attention custom op."""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import (
    _resolve_fp8_pages,
)


def test_raw_row_starts_compile_inside_custom_op(device):
    if device.type != "tpu":
        pytest.skip("requires the TorchTPU compiler")
    from torch_tpu._internal.compile._backend import TpuBackend

    @torch.library.custom_op("dsv41_test::raw_row_starts", mutates_args=())
    def resolve(records: torch.Tensor) -> torch.Tensor:
        main = SimpleNamespace(shape=(1, 32), raw=records)
        _, _, starts = _resolve_fp8_pages(main, records, records.shape[0])
        return records[0, :2] + starts

    @resolve.register_fake
    def fake(records):
        return records.new_empty((2,))

    compiled = torch.compile(
        resolve, backend=TpuBackend(), fullgraph=True, dynamic=False
    )
    records = torch.ones((8, 2), dtype=torch.int32, device=device)
    torch.testing.assert_close(
        compiled(records).cpu(), torch.tensor([1, 9], dtype=torch.int32)
    )
