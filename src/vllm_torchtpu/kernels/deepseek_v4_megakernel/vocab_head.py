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
"""Vocab-sharded lm_head with a cross-rank argmax (the megakernel step's
token head, B9 D4).

Each of the R ranks of the step's mesh axis holds rows [r * Vs, (r + 1) * Vs)
of the lm_head (Vs = vocab_shard(V, R); the last shard is zero-padded) and
computes those logits for every row; the token is then the argmax over all
ranks' (max, first index) pairs.

Exactness rules (tested in test_vocab_head.py):
- logits are the stock dtype (bf16 out of a bf16 x bf16 dot accumulating in
  f32). On CPU a shard's logits equal the full-vocab logits' columns bitwise;
  on TPU XLA may reduce K differently for the narrower dot: at DSv4 size
  (V=129280, H=7168, R=16) 0.014% of logits differ by an ulp, so the argmax
  can differ from a full-vocab head's only where the top two are that close
  (80/80 rows agreed, j-9d430213).
- ties: the lowest vocab index wins (torch.argmax / jnp.argmax): the first
  index within a shard, then the lowest rank among equal maxima.
- padded columns are -inf and never win.
- sampling at temperature T > 0 is Gumbel-max over logits / T with uniform
  noise drawn per (row, vocab id) from one key: the same draw whatever R, so
  it is a distributionally exact sample, but not the stock sampler's draw.
"""

import jax
import jax.numpy as jnp
from jax import lax

LANE = 128


def vocab_shard(vocab: int, ranks: int) -> int:
    """Rows of the lm_head per rank: ceil(V / R), rounded up to 128 lanes."""
    per = -(-vocab // ranks)
    return -(-per // LANE) * LANE


def shard_lm_head(w, rank: int, ranks: int):
    """[V, H] lm_head -> this rank's [Vs, H] rows (zero rows past V)."""
    vocab, hidden = w.shape
    vs = vocab_shard(vocab, ranks)
    lo = rank * vs
    rows = max(0, min(vs, vocab - lo))
    out = jnp.zeros((vs, hidden), w.dtype)
    if rows:
        out = out.at[:rows].set(w[lo : lo + rows])
    return out


def shard_logits(x, w_shard, me, vocab: int):
    """[N, H] bf16 x this rank's [Vs, H] rows -> [N, Vs] bf16 logits, padded
    columns -inf (`me`: this rank's index on the mesh axis, traced)."""
    vs = w_shard.shape[0]
    logits = lax.dot_general(
        x, w_shard, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
    ).astype(x.dtype)
    col = me * vs + lax.broadcasted_iota(jnp.int32, logits.shape, 1)
    return jnp.where(col < vocab, logits, -jnp.inf)


def _global_argmax(score, me, axis):
    """[N, Vs] f32 scores -> global [N] int32 argmax over the mesh axis
    (lowest vocab index on ties)."""
    vs = score.shape[1]
    mx = jnp.max(score, axis=1)
    idx = jnp.argmax(score, axis=1).astype(jnp.int32) + me * vs
    all_mx = lax.all_gather(mx, axis)  # [R, N]
    all_idx = lax.all_gather(idx, axis)
    best = jnp.max(all_mx, axis=0)
    # first rank holding the max = lowest vocab index (shards ascend with rank)
    r = jnp.argmax(all_mx == best[None, :], axis=0)
    return jnp.take_along_axis(all_idx, r[None, :], axis=0)[0]


def greedy_tokens(x, w_shard, me, axis, vocab: int):
    """Greedy token ids [N] of rows x [N, H] (every rank passes the same rows)."""
    return _global_argmax(shard_logits(x, w_shard, me, vocab).astype(jnp.float32),
                          me, axis)  # fmt: skip


def sample_tokens(x, w_shard, me, axis, vocab: int, temps, key):
    """Token ids [N]: greedy where temps <= 0, else Gumbel-max over
    logits / T. Noise for vocab id v of row n is uniform(fold_in(key, n), v),
    drawn over this rank's columns only (the same values whatever R)."""
    logits = shard_logits(x, w_shard, me, vocab).astype(jnp.float32)
    n, vs = logits.shape
    cols = me * vs + jnp.arange(vs, dtype=jnp.int32)

    def row_noise(i):
        k = jax.random.fold_in(key, i)
        return jax.vmap(lambda c: jax.random.uniform(jax.random.fold_in(k, c)))(cols)

    u = jax.vmap(row_noise)(jnp.arange(n, dtype=jnp.int32))
    u = jnp.clip(u, jnp.finfo(jnp.float32).tiny, 1.0 - 1e-7)
    greedy = temps <= 0.0
    scaled = logits / jnp.where(greedy, 1.0, temps)[:, None] - jnp.log(-jnp.log(u))
    score = jnp.where(greedy[:, None], logits, scaled)
    return _global_argmax(score, me, axis)
