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
"""Check HC kernels against upstream FP32 arithmetic between BF16 outputs."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_torchtpu.kernels.qwen4_exp.hc_full import _full_mix
from vllm_torchtpu.kernels.qwen4_exp.vllm_adapter import (
    build_full_mix_op,
    build_gated_residual_op,
)


def inputs(tokens=256, branches=4, hidden=128, lowrank=128):
    rng = np.random.default_rng(41)
    width = branches * hidden
    shapes = ((tokens, width), (width,), (lowrank, width), (width, lowrank))
    return tuple((rng.normal(size=s) * 0.1).astype(np.float32) for s in shapes)


def reference(x, weight, down, up, branches=4):
    # NVIDIA grouped Gemma RMSNorm stores only the final affine result in BF16.
    # hc_silu also promotes the BF16 down projection before division and SiLU.
    hidden = x.shape[1] // branches
    values = x.astype(jnp.float32).reshape(-1, branches, hidden)
    norm = values * jax.lax.rsqrt(jnp.mean(values**2, -1, keepdims=True) + 1e-6)
    norm = norm.reshape(x.shape)
    norm = (norm + norm * weight.astype(jnp.float32)).astype(jnp.bfloat16)
    mid = (norm.astype(jnp.float32) @ down.astype(jnp.float32).T).astype(jnp.bfloat16)
    mid = jax.nn.silu(mid.astype(jnp.float32) / branches).astype(jnp.bfloat16)
    logits = (mid.astype(jnp.float32) @ up.astype(jnp.float32).T).astype(jnp.bfloat16)
    weighted = jax.nn.sigmoid(logits.astype(jnp.float32)) * norm.astype(jnp.float32)
    mixed = jnp.mean(weighted.reshape(-1, branches, hidden), axis=1).astype(
        jnp.bfloat16
    )
    return mixed, norm, mid


@pytest.mark.parametrize(
    "tokens,branches,hidden,tile,lowrank",
    [
        (127, 4, 128, 128, 128),
        (0, 4, 128, 128, 128),
        (256, 4, 128, 0, 128),
        (256, 4, 128, -128, 128),
        (256, 4, 128, 96, 128),
        (256, 4, 129, 128, 128),
        (256, 4, 0, 128, 128),
        (256, 0, 128, 128, 128),
    ],
)
def test_hc_rejects_invalid_geometry(tokens, branches, hidden, tile, lowrank):
    shape = jax.ShapeDtypeStruct
    width = branches * hidden
    x = shape((tokens, width), jnp.bfloat16)
    weight = shape((width,), jnp.bfloat16)
    down = shape((lowrank, width), jnp.bfloat16)
    up = shape((width, lowrank), jnp.bfloat16)
    fn = functools.partial(
        _full_mix, branches=branches, hidden=hidden, eps=1e-6, token_tile=tile
    )
    with pytest.raises(ValueError, match="HC .* requires"):
        jax.eval_shape(fn, x, weight, down, up)


@pytest.mark.parametrize("kernel", ["residual", "full"])
def test_hc_torch_fake_shapes(kernel):
    if kernel == "residual":
        op = build_gated_residual_op(4, 128, 128)
    else:
        op = build_full_mix_op(4, 128, 1e-6, 128)
    with FakeTensorMode():
        x = torch.empty(256, 512, dtype=torch.bfloat16)
        weight = torch.empty(512, dtype=torch.bfloat16)
        down = torch.empty(128, 512, dtype=torch.bfloat16)
        up = torch.empty(512, 128, dtype=torch.bfloat16)
        mid = torch.empty(256, 128, dtype=torch.bfloat16)
        if kernel == "residual":
            result = (op(x, mid, torch.empty(256, 4, dtype=torch.float32)),)
            shapes = ((256, 512),)
        else:
            result, shapes = op(x, weight, down, up), ((256, 128), (256, 512))
        for tensor, expected_shape in zip(result, shapes, strict=True):
            assert tensor.shape == expected_shape
            assert tensor.dtype == torch.bfloat16
            assert tensor.device == x.device


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "tokens,branches,hidden,lowrank,tile",
    [
        (256, 4, 128, 128, 128),
        (512, 4, 128, 128, 128),
        (256, 3, 128, 128, 128),
        (512, 4, 2560, 320, 256),
    ],
)
def test_hc_full_mix_matches_cpu_reference(tokens, branches, hidden, lowrank, tile):
    cpu = jax.devices("cpu")[0]
    tpu = jax.devices("tpu")[0]
    with jax.default_device(cpu):
        arrays = tuple(
            jnp.asarray(a, jnp.bfloat16)
            for a in inputs(tokens, branches, hidden, lowrank)
        )
        expected_mix, expected_norm, _ = reference(*arrays, branches=branches)
    x, weight, down, up = tuple(jax.device_put(a, tpu) for a in arrays)
    full_mix, full_norm = _full_mix(
        x, weight, down, up, branches=branches, hidden=hidden, eps=1e-6, token_tile=tile
    )
    # Use the existing HC layer's NRMSE/cosine numerical acceptance contract.
    for actual, expected in [
        (full_norm, expected_norm),
        (full_mix, expected_mix),
    ]:
        a, e = (np.asarray(v, np.float32) for v in (actual, expected))
        assert np.isfinite(a).all()
        assert np.linalg.norm(a - e) / np.linalg.norm(e) <= 0.01
        assert (
            np.vdot(a.ravel(), e.ravel()) / (np.linalg.norm(a) * np.linalg.norm(e))
            >= 0.999
        )


@pytest.mark.tpu_test
@pytest.mark.parametrize("tile", [128, 256])
def test_hc_full_mix_preserves_affine_cancellation(tile):
    cpu = jax.devices("cpu")[0]
    tpu = jax.devices("tpu")[0]
    with jax.default_device(cpu):
        arrays = list(jnp.asarray(a, jnp.bfloat16) for a in inputs(tokens=2 * tile))
        # Premature BF16 rounding of norm * weight loses this small residual.
        arrays[1] = jnp.full((512,), -1 + 2**-8, dtype=jnp.bfloat16)
        _, expected, _ = reference(*arrays)
    x, weight, down, up = tuple(jax.device_put(a, tpu) for a in arrays)
    _, full_norm = _full_mix(
        x, weight, down, up, branches=4, hidden=128, eps=1e-6, token_tile=tile
    )
    a, e = (np.asarray(v, np.float32) for v in (full_norm, expected))
    assert np.isfinite(a).all()
    assert np.linalg.norm(a - e) / np.linalg.norm(e) <= 0.01


@pytest.mark.tpu_test
def test_hc_full_mix_torch_compile():
    from torch_tpu._internal import compile as tpu_compile
    from torch_tpu._internal import sync

    # Three branches exercises FP32 division by a non-power-of-two value.
    host = tuple(torch.from_numpy(a).bfloat16() for a in inputs(branches=3))
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        arrays = tuple(jnp.asarray(a.float().numpy(), jnp.bfloat16) for a in host)
        expected_mix, expected_norm, _ = reference(*arrays, branches=3)
    op = build_full_mix_op(3, 128, 1e-6, 128)
    run = torch.compile(
        op, backend=tpu_compile.TpuBackend(), fullgraph=True, dynamic=False
    )
    actual = run(*(a.to("tpu") for a in host))
    sync.synchronize(list(actual), wait=True)
    for result, expected in zip(actual, (expected_mix, expected_norm), strict=True):
        a = result.cpu().float().numpy()
        e = np.asarray(expected, np.float32)
        assert np.isfinite(a).all()
        assert np.linalg.norm(a - e) / np.linalg.norm(e) <= 0.01
        assert (
            np.vdot(a.ravel(), e.ravel()) / (np.linalg.norm(a) * np.linalg.norm(e))
            >= 0.999
        )
