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
"""Vocab-sharded lm_head + cross-rank argmax vs the full-vocab head (CPU,
4 virtual devices)."""

import jax

jax.config.update("jax_num_cpu_devices", 4)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4_megakernel import vocab_head as VH  # noqa: E402

V, H, N = 1000, 256, 16  # V not a multiple of R * 128: the last shard is padded


def _mesh(r):
    devs = jax.devices("cpu")
    if len(devs) < r:
        pytest.skip(f"needs {r} cpu devices")
    return jax.sharding.Mesh(np.array(devs[:r]), ("d",))


def _sharded(fn, r, x, w, *extra):
    shards = jnp.concatenate([VH.shard_lm_head(w, k, r) for k in range(r)])

    def body(ws, *rest):
        me = jax.lax.axis_index("d")
        return fn(x, ws, me, "d", V, *rest)[None]

    specs = (P("d"),) + (P(),) * len(extra)
    f = jax.jit(jax.shard_map(body, mesh=_mesh(r), in_specs=specs,
                              out_specs=P("d"), check_vma=False))  # fmt: skip
    out = np.asarray(f(shards, *extra))
    assert (out == out[0]).all(), "every rank gets the same tokens"
    return out[0]


def _data(seed=0):
    k1, k2 = jax.random.split(jax.random.key(seed))
    x = jax.random.normal(k1, (N, H), jnp.float32).astype(jnp.bfloat16)
    w = (jax.random.normal(k2, (V, H), jnp.float32) * 0.05).astype(jnp.bfloat16)
    return x, w


def _full_logits(x, w):
    return jax.lax.dot_general(
        x, w, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
    ).astype(jnp.bfloat16)


@pytest.mark.parametrize("r", [1, 2, 4])
def test_greedy_matches_full_vocab_argmax(r):
    x, w = _data()
    ref = np.asarray(jnp.argmax(_full_logits(x, w).astype(jnp.float32), axis=1))
    np.testing.assert_array_equal(_sharded(VH.greedy_tokens, r, x, w), ref)


def test_shard_logits_are_the_full_logits_columns():
    x, w = _data(1)
    full = np.asarray(_full_logits(x, w).astype(jnp.float32))
    vs = VH.vocab_shard(V, 4)
    for k in range(4):
        got = np.asarray(
            VH.shard_logits(x, VH.shard_lm_head(w, k, 4), k, V), np.float32
        )
        lo, hi = k * vs, min(V, (k + 1) * vs)
        np.testing.assert_array_equal(got[:, : hi - lo], full[:, lo:hi])
        assert np.isneginf(got[:, hi - lo :]).all()  # padding never wins


def test_ties_go_to_the_lowest_vocab_index():
    x, w = _data(2)
    w = np.asarray(w).copy()
    vs = VH.vocab_shard(V, 4)
    # one row's argmax column duplicated in a later shard and later in its own
    ref = np.asarray(jnp.argmax(_full_logits(x, jnp.asarray(w)).astype(jnp.float32), 1))
    v0 = int(ref[0])
    for dup in (v0 + 1 if (v0 + 1) % vs else v0 - 1, (v0 + 2 * vs) % V, V - 1):
        w[dup] = w[v0]
    w = jnp.asarray(w)
    full = _full_logits(x, w).astype(jnp.float32)
    want = np.asarray(jnp.argmax(full, axis=1))  # first occurrence
    got = _sharded(VH.greedy_tokens, 4, x, w)
    np.testing.assert_array_equal(got, want)
    assert got[0] == min(
        v0, *[d for d in range(V) if (np.asarray(w[d]) == np.asarray(w[v0])).all()]
    )


def test_sampling_is_independent_of_the_shard_count_and_greedy_at_zero():
    x, w = _data(3)
    temps = jnp.asarray([0.0, 0.7, 1.0, 1.3] * (N // 4), jnp.float32)
    key = jax.random.key(42)
    s1 = _sharded(VH.sample_tokens, 1, x, w, temps, key)
    s2 = _sharded(VH.sample_tokens, 2, x, w, temps, key)
    s4 = _sharded(VH.sample_tokens, 4, x, w, temps, key)
    np.testing.assert_array_equal(s1, s2)
    np.testing.assert_array_equal(s1, s4)
    greedy = _sharded(VH.greedy_tokens, 4, x, w)
    np.testing.assert_array_equal(s4[::4], greedy[::4])
    assert (s4 != greedy).any()  # T > 0 rows actually sample


def test_sampling_follows_softmax():
    """Gumbel-max frequencies vs softmax(logits / T) on one row: the total
    variation of 4000 draws is within 1.3x that of an exact multinomial
    sample of the same size."""
    x, w = _data(4)
    x = x[:1].repeat(N, axis=0)
    t = 1.0
    temps = jnp.full((N,), t, jnp.float32)
    logits = _full_logits(x[:1], w).astype(jnp.float32)[0]
    p = np.asarray(jax.nn.softmax(logits / t))
    counts = np.zeros(V)
    for s in range(250):
        toks = _sharded(VH.sample_tokens, 4, x, w, temps, jax.random.key(1000 + s))
        np.add.at(counts, toks, 1)
    tv = 0.5 * np.abs(counts / counts.sum() - p).sum()
    rng = np.random.default_rng(0)
    exact = [
        0.5 * np.abs(rng.multinomial(int(counts.sum()), p) / counts.sum() - p).sum()
        for _ in range(20)
    ]
    assert tv < 1.3 * np.mean(exact), (tv, np.mean(exact))


def test_step_own_logits_equal_the_full_head_rows():
    """step_impl._own_logits: every rank computes its shard on all T rows,
    all-gathers, and keeps its own rows (ownt: global row of own row k)."""
    from types import SimpleNamespace

    from vllm_torchtpu.kernels.deepseek_v4_megakernel import kernel_interface as KI
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import step_impl as SI

    r_, m = 4, 4
    x, w = _data(5)  # x: the T = 16 global rows
    shards = jnp.concatenate([VH.shard_lm_head(w, k, r_) for k in range(r_)])
    # rank k owns global rows 4k .. 4k+2 as its own rows 0..2; own row 3 empty
    ownt = np.array([[4 * k, 4 * k + 1, 4 * k + 2, -1] for k in range(r_)], np.int32)
    spec = SimpleNamespace(vocab_size=V, axis_name="d")

    def body(ws, ot):
        meta = (
            jnp.zeros((KI.META_LEN,), jnp.int32)
            .at[KI.META_RANK]
            .set(jax.lax.axis_index("d"))
        )
        return SI._own_logits(x, ws, ot[0], meta, spec)[None]

    f = jax.jit(jax.shard_map(body, mesh=_mesh(r_), in_specs=(P("d"), P("d")),
                              out_specs=P("d"), check_vma=False))  # fmt: skip
    got = np.asarray(f(shards, jnp.asarray(ownt)), np.float32).reshape(r_, m, V)
    full = np.asarray(_full_logits(x, w).astype(jnp.float32))
    for k in range(r_):
        want = full[4 * k : 4 * k + 3]
        if jax.default_backend() == "cpu":
            np.testing.assert_array_equal(got[k, :3], want)
        else:  # the TPU dot of a vocab slice may round 1 bf16 ulp differently
            np.testing.assert_allclose(got[k, :3], want, rtol=2**-7, atol=0)
        assert (got[k, 3] == 0).all()
