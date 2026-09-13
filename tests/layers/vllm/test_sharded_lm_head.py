"""The sharded vocabulary projection (layers/vllm/sharded_lm_head.py): the JAX body on a one-device CPU mesh
against a plain matmul, and the head method's padding and trimming with a stub op."""
import os
import types

os.environ.setdefault(
    "JAX_PLATFORMS",
    "cpu")  # the body test needs no chip; the chips may be busy
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh

from vllm_torchtpu.layers.vllm import sharded_lm_head as sh


@pytest.fixture(autouse=True)
def _on_the_host():
    """Every JAX array on the CPU device: the mesh below is a CPU mesh, and the
    reference matmul must run on the same device with the same (full f32)
    precision, whatever backend and matmul precision an earlier test in the
    session left as the default."""
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def test_body_matches_the_matmul_on_a_one_device_mesh():
    mesh = Mesh(np.array(jax.devices("cpu")[:1]), (sh.EP_AXIS_NAME, ))
    fn = sh.head_fn(mesh)
    x = jax.random.normal(jax.random.PRNGKey(0), (16, 32), jnp.float32)
    w = jax.random.normal(jax.random.PRNGKey(1), (64, 32), jnp.float32)
    with mesh:
        out = fn(x, w)
    np.testing.assert_allclose(np.asarray(out),
                               np.asarray(x @ w.T),
                               rtol=1e-4,
                               atol=1e-4)


def test_method_pads_and_trims(monkeypatch):
    calls = []

    def stub(x, w):
        calls.append(tuple(x.shape))
        return x @ w.T

    monkeypatch.setattr(sh, "_OP", stub)
    layer = types.SimpleNamespace(weight_slice=torch.randn(24, 8))
    m = sh.ShardedHeadMethod(rows=16)
    x = torch.randn(5, 8)
    out = m.apply(layer, x)
    assert calls == [(16, 8)] and out.shape == (5, 24)
    torch.testing.assert_close(out, x @ layer.weight_slice.T)
    try:
        m.apply(layer, torch.randn(17, 8))
        raise AssertionError("17 rows must be refused")
    except ValueError:
        pass


def test_unsupported_reason_follows_the_known_limit():
    assert sh.unsupported_reason(16, 8192) is None
    assert sh.unsupported_reason(20, 768) is None
    assert "20 rows" in sh.unsupported_reason(20, 1024)
    assert "32 rows" in sh.unsupported_reason(32, 768)


def test_install_leaves_the_head_alone_outside_the_limit():
    head = torch.nn.Linear(256, 1024, bias=False)
    model = torch.nn.Module()
    model.lm_head = head
    before = head.quant_method if hasattr(head, "quant_method") else None
    assert sh.install(model, 32, 768) is False
    assert getattr(head, "quant_method", None) is before
