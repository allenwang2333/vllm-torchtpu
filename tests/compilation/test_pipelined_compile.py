# SPDX-License-Identifier: Apache-2.0
"""End-to-end unit test for the pipelined bucket-compilation path.

Drives the adaptor's module-level TpuBackend — the exact object serving
uses — through aot_autograd on a tiny graph, with the compile pool active.
This exercises the full contract stack (aot stage2 artifact checks, lazy
future resolution on first execution), which the compile_mlir thread smoke
bypasses. Requires a TPU.
"""
import os

import pytest
import torch
import torch._guards

os.environ.setdefault("TPU_COMPILE_THREADS", "4")


@pytest.fixture(scope="module")
def tpu_compiler_mod():
    torch_tpu = pytest.importorskip("torch_tpu")  # noqa: F841
    from vllm_torchtpu.compilation import tpu_compiler
    if tpu_compiler._compile_pool is None:
        pytest.skip("pipelined compilation disabled (TPU_COMPILE_THREADS=0)")
    return tpu_compiler


def test_pipelined_backend_compiles_and_runs(tpu_compiler_mod):
    """The shipped pipelined backend must produce runnable, correct
    artifacts through the full aot_autograd path (regression guard for the
    OutputCode/serialization contract and lazy-future resolution)."""

    class Net(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(64, 128, bias=False)
            self.b = torch.nn.Linear(128, 64, bias=False)

        def forward(self, x):
            return (self.b(torch.nn.functional.gelu(self.a(x))), )

    with torch.no_grad():
        model = Net().to(dtype=torch.bfloat16, device="tpu")
        for p in model.parameters():
            p.requires_grad_(False)
        traced = torch.fx.symbolic_trace(model)
        x = torch.randn(8, 64, dtype=torch.bfloat16).to("tpu")

        with torch._guards.tracing(None):
            compiled_fn = tpu_compiler_mod._tpu_backend(traced, [x])

        out = compiled_fn(x)
        out = out[0] if isinstance(out, (tuple, list)) else out
        ref = traced(x)[0]
        diff = (out.float() - ref.float()).abs().max().item()
        assert diff < 1e-1, f"pipelined artifact output mismatch: {diff}"
