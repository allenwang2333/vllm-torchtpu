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
"""TPU_MOE_FP8_WIRE_ALLGATHER: the wire changes, the value the kernel sees does not.

The claim under test is narrow and checkable without a TPU: the GMM kernel
quantizes its own lhs to fp8 e4m3 per row over contiguous 512-element blocks of
the contracting dimension, so quantizing with that same recipe *before* the
all-gather and dequantizing after it hands the kernel a tensor it re-quantizes
to the same codes and the same scales. These tests re-implement the kernel's
recipe (`kernels/megablox/gmm_v2.py`, lhs `quant_block_size=512`) and assert the
codes match bit for bit, not merely to a tolerance -- a tolerance would pass on
the one failure mode that matters, a change of scale granularity.
"""

import os
from unittest.mock import patch

import pytest
import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.vllm import pipelined_fused_moe as mod
from vllm_torchtpu.layers.vllm.pipelined_fused_moe import (
    _all_gather_activations, fp8_wire_allgather_applies,
    pipelined_fused_moe_gmm)

_BLOCK = mod._FP8_WIRE_QUANT_BLOCK
_E4M3_MAX = mod._FP8_WIRE_E4M3_MAX


class _MockCollectiveGroup:
    """Replicates along dim 0, the way a real all-gather concatenates ranks."""

    def __init__(self, world_size=4):
        self.world_size = world_size

    def all_gather(self, tensor, dim=-1):
        return torch.cat([tensor] * self.world_size, dim=dim)

    def reduce_scatter(self, tensor, dim=-1):
        return tensor.chunk(self.world_size, dim=dim)[0]


def _kernel_quantize(x: torch.Tensor):
    """What gmm_v2 does to its lhs.

    `block_abs_max = jnp.max(jnp.abs(block_lhs), axis=1)`;
    `block_scale = block_abs_max / dtype_max`;
    `block_lhs_q = (block_lhs * block_scale_inv).astype(lhs_q_dtype)`.
    """
    rows, hidden = x.shape
    blocks = hidden // _BLOCK
    blocked = x.reshape(rows, blocks, _BLOCK)
    scale = blocked.abs().amax(dim=-1, keepdim=True) / _E4M3_MAX
    scale_inv = torch.where(scale == 0, torch.zeros_like(scale),
                            scale.reciprocal())
    codes = (blocked * scale_inv).to(torch.float8_e4m3fn)
    return codes, scale


def _knob(on: bool):
    return patch.dict(os.environ,
                      {"TPU_MOE_FP8_WIRE_ALLGATHER": "1" if on else "0"})


def test_knob_defaults_off():
    """An unset knob must leave the wire format alone."""
    with patch.dict(os.environ, {}, clear=True):
        assert envs.TPU_MOE_FP8_WIRE_ALLGATHER is False
    with _knob(True):
        assert envs.TPU_MOE_FP8_WIRE_ALLGATHER is True
    with _knob(False):
        assert envs.TPU_MOE_FP8_WIRE_ALLGATHER is False


def test_off_path_is_the_untouched_collective():
    group = _MockCollectiveGroup(world_size=4)
    x = torch.randn(6, 2 * _BLOCK, dtype=torch.bfloat16)
    with _knob(False):
        got = _all_gather_activations(group, x)
    assert torch.equal(got, group.all_gather(x, dim=0))
    assert got.dtype == x.dtype


@pytest.mark.parametrize("hidden", [_BLOCK, 3 * _BLOCK, 12 * _BLOCK])
def test_kernel_sees_identical_fp8_codes_and_scales(hidden):
    """The whole justification: the round trip is invisible to the kernel."""
    torch.manual_seed(0)
    group = _MockCollectiveGroup(world_size=4)
    x = torch.randn(8, hidden, dtype=torch.bfloat16) * 3.0

    with _knob(False):
        baseline = _all_gather_activations(group, x)
    with _knob(True):
        wired = _all_gather_activations(group, x)

    assert wired.shape == baseline.shape
    assert wired.dtype == baseline.dtype

    base_codes, base_scale = _kernel_quantize(baseline)
    wire_codes, wire_scale = _kernel_quantize(wired)

    assert torch.equal(base_codes.view(torch.uint8),
                       wire_codes.view(torch.uint8)), "fp8 codes diverged"
    assert torch.equal(base_scale, wire_scale), "block scales diverged"


def test_all_zero_block_survives():
    """scale == 0 must not produce inf/NaN, the way the kernel's where does."""
    group = _MockCollectiveGroup(world_size=2)
    x = torch.zeros(4, 2 * _BLOCK, dtype=torch.bfloat16)
    x[:, _BLOCK:] = 1.5
    with _knob(True):
        got = _all_gather_activations(group, x)
    assert torch.isfinite(got).all()
    assert torch.equal(got, group.all_gather(x, dim=0))


def test_scale_is_row_local_so_gathering_more_ranks_changes_nothing():
    """Q1, as an assertion.

    A scale computed over the gathered tensor would move when the number of
    ranks moves. This one must not: the reduction runs along the hidden
    dimension, inside a row.
    """
    torch.manual_seed(1)
    x = torch.randn(5, 4 * _BLOCK, dtype=torch.bfloat16)
    with _knob(True):
        small = _all_gather_activations(_MockCollectiveGroup(2), x)
        large = _all_gather_activations(_MockCollectiveGroup(8), x)
    assert torch.equal(small[:5], large[:5])


def test_shapes_the_knob_must_decline():
    """A hidden dim that is not a multiple of the kernel's block is left alone."""
    x = torch.randn(4, _BLOCK + 128, dtype=torch.bfloat16)
    with _knob(True):
        assert not fp8_wire_allgather_applies(x)
    ints = torch.zeros(4, _BLOCK, dtype=torch.int32)
    with _knob(True):
        assert not fp8_wire_allgather_applies(ints)


_TOPK = 2


@pytest.mark.parametrize("wire", ["0", "1"])
@pytest.mark.parametrize("chunked", [False, True])
def test_every_knob_arm_survives_dynamo(wire, chunked):
    """Every arm must TRACE, not merely run eagerly.

    Dynamo resolves a function's global and closure references when it traces,
    not when the guarded branch that reads one runs, so a name bound only on
    the arm that happens to execute eagerly still takes the compile down -- on
    whichever arm nobody tested. Both wire arms and both chunking arms are
    compiled here for that reason.

    Patch with plain functions, never MagicMock: Dynamo refuses to inline a
    mock ("Attempted to inline function marked as skipped"), which fails every
    arm identically and makes this test prove nothing.
    """
    parallel_size, seq_len, hidden = 2, 4, _BLOCK
    hs = torch.randn(seq_len, hidden, dtype=torch.bfloat16)
    weights = torch.randn(seq_len, _TOPK)
    ids = torch.randint(0, 32, (seq_len, _TOPK), dtype=torch.int32)
    group = _MockCollectiveGroup(world_size=parallel_size)

    def call(h, w, i):
        return pipelined_fused_moe_gmm(hidden_states=h,
                                       w1=None,
                                       w2=None,
                                       w1_scale=None,
                                       w2_scale=None,
                                       w1_bias=None,
                                       w2_bias=None,
                                       topk_weights=w,
                                       topk_ids=i,
                                       experts_start=None,
                                       topk=_TOPK,
                                       activation="silu")

    env = {
        "TPU_MOE_FP8_WIRE_ALLGATHER": wire,
        # num_chunks = 2 selects the pipelined branch; 0 disables chunking and
        # selects the single-shot one. Both reach _all_gather_activations.
        "TPU_MOE_COLLECTION_CHUNK_SIZE":
        str(parallel_size * seq_len // 2) if chunked else "0",
    }

    torch._dynamo.reset()
    compiled = torch.compile(call, backend="eager", fullgraph=True)

    with patch.dict(os.environ, env), \
         patch.object(mod, "_get_moe_collective_group",
                      lambda: (group, "dp")), \
         patch.object(mod, "fused_moe_gmm",
                      lambda **kw: torch.zeros(kw["hidden_states"].shape[0],
                                               hidden,
                                               dtype=torch.bfloat16)):
        out = compiled(hs, weights, ids)

    assert out.shape == (seq_len, hidden)


def _libtpu_flag_in_subprocess(env_extra):
    """env_override builds LIBTPU_INIT_ARGS at import, so read it back fresh."""
    import subprocess
    import sys
    env = dict(os.environ)
    env.pop("LIBTPU_INIT_ARGS", None)
    env.pop("SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES", None)
    env.pop("TPU_MOE_FP8_WIRE_ALLGATHER", None)
    env["TPU_ACCELERATOR_TYPE"] = "tpu7x-16"
    env.update(env_extra)
    out = subprocess.run(
        [sys.executable, "-c", "import os;import vllm_torchtpu.env_override;"
         "print(os.environ.get('LIBTPU_INIT_ARGS',''))"],
        env=env, capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()[-1]


def _sc_allgather_bytes(flags: str) -> int:
    for token in flags.split():
        key = "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes="
        if token.startswith(key):
            return int(token[len(key):])
    raise AssertionError(f"flag absent from {flags!r}")


def test_knob_moves_the_sparsecore_offload_threshold():
    """Halving the payload must not silently move the collective off SparseCore.

    The threshold is the per-tensor-core VMEM capacity, so a gathered tensor
    that sat above it in bf16 sits below it in fp8, and the collective moves
    back to the TensorCore -- a measured regression, and one that would be
    read as "fp8 on the wire does not help". The knob has to move both.
    """
    off = _sc_allgather_bytes(_libtpu_flag_in_subprocess({}))
    on = _sc_allgather_bytes(
        _libtpu_flag_in_subprocess({"TPU_MOE_FP8_WIRE_ALLGATHER": "1"}))
    assert off == 67108864, off
    assert on == off // 2
