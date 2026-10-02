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
"""One-shot collectives against numpy, in Pallas TPU interpret mode with race detection on 8
emulated CPU devices (run in its own process: XLA_FLAGS must be set before jax initializes)."""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.kernels.collectives import oneshot  # noqa: E402


def _expected(op, parts, n, rows):
    if op == "all_gather":
        return np.concatenate([np.concatenate(list(parts))] * n)
    if op == "all_reduce":
        return np.concatenate([parts.sum(0)] * n)
    if op == "reduce_scatter":
        return parts.reshape(n, n, rows // n, -1).sum(0).reshape(rows, -1)
    return np.concatenate(  # all_to_all: rank j gets slice j of every rank, by source
        [np.concatenate([parts[i].reshape(n, rows // n, -1)[j] for i in range(n)])
         for j in range(n)])


@pytest.mark.parametrize("op", ["all_gather", "all_to_all", "reduce_scatter", "all_reduce"])
def test_oneshot_matches_numpy(op, monkeypatch):
    if len(jax.devices()) < 8 or jax.devices()[0].platform != "cpu":
        pytest.skip("needs 8 emulated CPU devices (JAX_PLATFORMS=cpu)")
    n = len(jax.devices())
    mesh = Mesh(np.array(jax.devices()), (oneshot.AXIS,))
    monkeypatch.setattr(oneshot, "INTERPRET", pltpu.InterpretParams(detect_races=True))
    rows = 2 * n if op in ("all_to_all", "reduce_scatter") else 2
    x = np.random.default_rng(0).integers(0, 5, size=(n * rows, 128)).astype(np.float32)
    xs = jax.device_put(jnp.asarray(x), NamedSharding(mesh, P(oneshot.AXIS, None)))
    got = np.asarray(jax.jit(oneshot.jax_fn(op, mesh, out_sharded=True))(xs))
    np.testing.assert_array_equal(got, _expected(op, x.reshape(n, rows, 128), n, rows))


def test_twoshot_all_reduce_matches_numpy(monkeypatch):
    if len(jax.devices()) < 8 or jax.devices()[0].platform != "cpu":
        pytest.skip("needs 8 emulated CPU devices (JAX_PLATFORMS=cpu)")
    n = len(jax.devices())
    mesh = Mesh(np.array(jax.devices()), (oneshot.AXIS,))
    monkeypatch.setattr(oneshot, "INTERPRET", pltpu.InterpretParams(detect_races=True))
    x = np.random.default_rng(1).integers(0, 5, size=(n * n, 128)).astype(np.float32)
    xs = jax.device_put(jnp.asarray(x), NamedSharding(mesh, P(oneshot.AXIS, None)))
    fn = oneshot.jax_fn("all_reduce", mesh, out_sharded=True, variant="twoshot")
    got = np.asarray(jax.jit(fn)(xs))
    parts = x.reshape(n, n, 128)
    np.testing.assert_array_equal(got, np.concatenate([parts.sum(0)] * n))
