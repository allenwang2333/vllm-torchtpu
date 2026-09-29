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
"""The vocabulary-sharded LM head: the JAX body on a one-device CPU mesh, the
head method's padding, and what `install` does to the model."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)

from vllm_torchtpu.layers.adapter import sharded_lm_head

pytestmark = pytest.mark.cpu_test

VOCAB = 256
HIDDEN = 32


@pytest.fixture(autouse=True)
def _cpu_backend():
    """Keep every array in this file off the accelerator."""
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _model(tied: bool = False) -> torch.nn.Module:
    model = torch.nn.Module()
    model.lm_head = ParallelLMHead(VOCAB, HIDDEN, disable_tp=True)
    torch.nn.init.normal_(model.lm_head.weight)
    if tied:
        model.embed_tokens = VocabParallelEmbedding(VOCAB, HIDDEN, disable_tp=True)
        model.lm_head.weight = model.embed_tokens.weight
    return model


def _fake_ep(monkeypatch, ep_size: int, index: int) -> None:
    """An EP group of `ep_size` with this rank at `index`; the op is the local
    matmul, which is what one rank of the real op contributes."""
    mesh = SimpleNamespace(shape={sharded_lm_head.EP_AXIS_NAME: ep_size})
    monkeypatch.setattr(sharded_lm_head, "build_ep_mesh", lambda: mesh)
    monkeypatch.setattr(sharded_lm_head, "ep_mesh_index", lambda: index)
    monkeypatch.setattr(sharded_lm_head, "build_op", lambda mesh: lambda x, w: x @ w.T)


def test_body_matches_the_matmul_on_a_one_device_mesh():
    mesh = Mesh(np.array(jax.devices("cpu")[:1]), (sharded_lm_head.EP_AXIS_NAME,))
    x = jax.random.normal(jax.random.PRNGKey(0), (16, HIDDEN), jnp.float32)
    w = jax.random.normal(jax.random.PRNGKey(1), (VOCAB, HIDDEN), jnp.float32)
    out = sharded_lm_head.head_fn(mesh)(x, w)
    np.testing.assert_allclose(np.asarray(out), np.asarray(x @ w.T), rtol=1e-4)


def test_method_pads_to_the_fixed_rows_and_trims():
    calls = []

    def op(x, w):
        calls.append(tuple(x.shape))
        return x @ w.T

    layer = SimpleNamespace(weight_slice=torch.randn(24, 8))
    method = sharded_lm_head.ShardedLMHeadMethod(op, rows=16)
    x = torch.randn(5, 8)
    out = method.apply(layer, x)
    assert calls == [(16, 8)]
    torch.testing.assert_close(out, x @ layer.weight_slice.T)
    with pytest.raises(ValueError, match="exceed"):
        method.apply(layer, torch.randn(17, 8))


def test_install_keeps_this_ranks_vocabulary_rows(monkeypatch):
    _fake_ep(monkeypatch, ep_size=4, index=2)
    model = _model()
    weight = model.lm_head.weight.detach().clone()
    sharded_lm_head.install(model, [16])
    head = model.lm_head
    part = VOCAB // 4
    torch.testing.assert_close(head.weight_slice, weight[2 * part : 3 * part])
    assert head.weight.shape == (1, HIDDEN)
    x = torch.randn(5, HIDDEN)
    out = head.quant_method.apply(head, x)
    torch.testing.assert_close(out, x @ weight[2 * part : 3 * part].T)


def test_one_request_bucket_and_no_speculative_decoding():
    config = SimpleNamespace(speculative_config=None)
    assert sharded_lm_head.request_buckets(config, 40) == [40]
    config.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    with pytest.raises(NotImplementedError, match="speculative decoding"):
        sharded_lm_head.request_buckets(config, 40)


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("no_head", "no ParallelLMHead"),
        ("tied", "tied"),
        ("tensor_parallel", "tensor-parallel"),
        ("quantized", "unquantized"),
        ("no_ep", "expert-parallel"),
        ("vocab_split", "does not split"),
    ],
)
def test_install_refuses_an_unsupported_head(monkeypatch, case, match):
    _fake_ep(monkeypatch, ep_size=3 if case == "vocab_split" else 4, index=0)
    if case == "no_ep":
        monkeypatch.setattr(sharded_lm_head, "build_ep_mesh", lambda: None)
    model = torch.nn.Module() if case == "no_head" else _model(tied=case == "tied")
    if case == "tensor_parallel":
        model.lm_head.tp_size = 2
    if case == "quantized":
        model.lm_head.weight.data = model.lm_head.weight.data.to(torch.float8_e4m3fn)
    with pytest.raises(NotImplementedError, match=match):
        sharded_lm_head.install(model, [16])
