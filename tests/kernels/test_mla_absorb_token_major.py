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
"""TPU_MLA_ABSORB_TOKEN_MAJOR: the token-major W_UK absorb.

The claim is bit-identity with the incumbent `transpose -> bmm -> transpose`
on the q_b head layout GLM-5.2 serves (192 nope + 64 rope, 64 heads under
DCP query replication, lora 512), at mixed bucket sizes, with and without a
per-channel weight scale. Checked

* the JAX kernel against the head-major `jnp.matmul` reference;
* the torch entry `mla_absorb_w_uk` ("pallas" and "einsum") against its own
  reference path (mode None), eager and under `torch.compile(backend="tpu")`;
* the bucket gate: decode-sized buckets keep the bmm.

with a positive control the comparison must catch: the kernel fed the weight
in a wrong head order (heads rolled by one).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.mla.absorb import mla_absorb_token_major, pick_blocks

HEADS = 64
NOPE = 192
QK = 256
LORA = 512


def _require_tpu():
    if jax.devices()[0].platform != "tpu":
        pytest.skip("needs a TPU")


def _inputs(num_tokens, heads=HEADS, scaled=False, seed=0):
    k = jax.random.split(jax.random.key(seed + num_tokens), 3)
    q = jax.random.normal(k[0], (num_tokens, heads, QK), jnp.float32).astype(
        jnp.bfloat16
    )
    w = (jax.random.normal(k[1], (heads, NOPE, LORA), jnp.float32) * 0.05).astype(
        jnp.bfloat16
    )
    s = None
    if scaled:
        s = jax.random.uniform(k[2], (heads, 1, LORA), jnp.float32, 0.5, 2.0).astype(
            jnp.bfloat16
        )
    return q, w, s


@jax.jit
def _reference(q, w):
    # The incumbent: head-major bmm between two transposes.
    out = jnp.matmul(
        jnp.transpose(q[..., :NOPE], (1, 0, 2)), w, preferred_element_type=None
    )
    return jnp.transpose(out, (1, 0, 2))


@jax.jit
def _reference_scaled(q, w, s):
    out = jnp.matmul(jnp.transpose(q[..., :NOPE], (1, 0, 2)), w)
    return jnp.transpose(out * s, (1, 0, 2))


def _max_ulp(a, b):
    a = np.asarray(a).view(np.uint16).astype(np.int32)
    b = np.asarray(b).view(np.uint16).astype(np.int32)
    # bf16 bit patterns -> ordered integers
    a = np.where(a & 0x8000, 0x8000 - a, a)
    b = np.where(b & 0x8000, 0x8000 - b, b)
    return int(np.max(np.abs(a - b)))


@pytest.mark.tpu_test
@pytest.mark.parametrize("scaled", [False, True], ids=["bf16", "scaled"])
@pytest.mark.parametrize("num_tokens", [1024, 2048, 4096])
def test_kernel_matches_head_major_bmm(num_tokens, scaled):
    _require_tpu()
    q, w, s = _inputs(num_tokens, scaled=scaled)
    got = jax.jit(
        lambda q, w, s: mla_absorb_token_major(q, w, s, nope_dim=NOPE)
    )(q, w, s)
    want = _reference_scaled(q, w, s) if scaled else _reference(q, w)
    assert got.shape == (num_tokens, HEADS, LORA) and got.dtype == jnp.bfloat16
    assert _max_ulp(got, want) == 0


@pytest.mark.tpu_test
def test_positive_control_wrong_head_order_fails():
    """A weight whose heads are rolled by one must not pass the comparison."""
    _require_tpu()
    q, w, _ = _inputs(1024)
    want = _reference(q, w)
    bad = jax.jit(lambda q, w: mla_absorb_token_major(q, w, nope_dim=NOPE))(
        q, jnp.roll(w, 1, axis=0)
    )
    assert _max_ulp(bad, want) > 0
    assert not np.array_equal(np.asarray(bad), np.asarray(want))


def test_pick_blocks():
    assert pick_blocks(4096, 64) == (512, 16)
    assert pick_blocks(64, 64) == (64, 16)
    assert pick_blocks(4, 64) is None  # decode-sized: not tiled, bmm stays
    assert pick_blocks(1024, 4) is None  # fewer heads than one bf16 tile


def test_mode_gate(monkeypatch):
    from vllm_torchtpu.layers.adapter import attention

    monkeypatch.delenv("TPU_MLA_ABSORB_TOKEN_MAJOR", raising=False)
    assert attention.mla_absorb_mode(4096) is None
    monkeypatch.setenv("TPU_MLA_ABSORB_TOKEN_MAJOR", "1")
    assert attention.mla_absorb_mode(4096) == "pallas"
    assert attention.mla_absorb_mode(64) == "pallas"
    for decode_bucket in (4, 8, 16, 32):
        assert attention.mla_absorb_mode(decode_bucket) is None
    monkeypatch.setenv("TPU_MLA_ABSORB_TOKEN_MAJOR", "einsum")
    monkeypatch.setenv("TPU_MLA_ABSORB_TOKEN_MAJOR_MIN_TOKENS", "2048")
    assert attention.mla_absorb_mode(1024) is None
    assert attention.mla_absorb_mode(2048) == "einsum"


@pytest.mark.tpu_test
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
@pytest.mark.parametrize("mode", ["pallas", "einsum"])
@pytest.mark.parametrize("num_tokens", [1024, 2048, 4096])
def test_torch_entry_matches_reference(num_tokens, mode, compiled):
    _require_tpu()
    import torch

    from vllm_torchtpu.layers.adapter import attention

    attention.get_mla_absorb_op(False)  # outside Dynamo
    g = torch.Generator().manual_seed(num_tokens)
    q = torch.randn(num_tokens, HEADS, QK, generator=g).to(torch.bfloat16)
    w = (torch.randn(HEADS, NOPE, LORA, generator=g) * 0.05).to(torch.bfloat16)
    q, w = q.to("tpu"), w.to("tpu")

    def run(q, w, m):
        src = q if m == "pallas" else q[..., :NOPE]
        return attention.mla_absorb_w_uk(src, NOPE, w, None, torch.bfloat16, m)

    def ref(q, w):
        return attention.mla_absorb_w_uk(
            q[..., :NOPE], NOPE, w, None, torch.bfloat16, None
        )

    if compiled:
        torch._dynamo.reset()
        run = torch.compile(run, backend="tpu", fullgraph=True, dynamic=False)
        ref = torch.compile(ref, backend="tpu", fullgraph=True, dynamic=False)
    got = run(q, w, mode).cpu()
    want = ref(q, w).cpu()
    assert got.shape == (num_tokens, HEADS, LORA)
    torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.tpu_test
@pytest.mark.parametrize("num_tokens", [4, 16])
def test_decode_buckets_keep_the_bmm(num_tokens, monkeypatch):
    """With the flag on, a decode bucket still takes the reference bmm path."""
    _require_tpu()
    import torch

    from vllm_torchtpu.layers.adapter import attention

    monkeypatch.setenv("TPU_MLA_ABSORB_TOKEN_MAJOR", "1")
    mode = attention.mla_absorb_mode(num_tokens)
    assert mode is None
    g = torch.Generator().manual_seed(num_tokens)
    q = torch.randn(num_tokens, HEADS, QK, generator=g).to(torch.bfloat16).to("tpu")
    w = (torch.randn(HEADS, NOPE, LORA, generator=g) * 0.05).to(torch.bfloat16)
    w = w.to("tpu")
    got = attention.mla_absorb_w_uk(q[..., :NOPE], NOPE, w, None, torch.bfloat16, mode)
    want = torch.bmm(q[..., :NOPE].transpose(0, 1), w).transpose(0, 1)
    torch.testing.assert_close(got.cpu(), want.cpu(), rtol=0, atol=0)
