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
"""Preflight for TPU_TOKEN_BUCKET_EXTRA=48 on the GLM-5.2-FP8 decode step.

The steady-state decode batch on GLM-5.2-FP8 at 2x2x2 schedules 46 real
tokens per DP rank and pads up to the 64 token bucket. Adding a 48 bucket
(``TPU_TOKEN_BUCKET_EXTRA=48``) makes it pad to 48 instead. 48 is not a power
of two, and the decode MoE token movement runs on SparseCore
(``ragged_gather_v2`` / ``ragged_gather_reduce_v2``), so this file answers:

1. does ``_get_token_paddings`` actually grow the 48 bucket (pure Python), and
2. does the decode MoE produce the same answer for the 46 real rows when the
   per-rank bucket is 48 as it does when the per-rank bucket is 64.

Shapes are taken from the production config, not from a neighbouring test:
``hidden_size=6144``, ``moe_intermediate_size=2048``, 256 routed experts,
top-8, EP=16 -> 16 local experts (config.json for zai-org/GLM-5.2-FP8, and
the ``bf16[64, 6144]`` MoE dispatch all-gather operand in the baseline
xprof). The MoE runs on the *gathered* token count, 16 ranks x bucket, so the
two arms are 1024 and 768 local tokens.
"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
# Import vllm (and with it the torchtpu platform plugin) before anything
# reaches into vllm_torchtpu.platforms. Importing that package first leaves it
# half-built when vllm's own plugin loader gets to it, and the plugin load
# fails with "partially initialized module".
import vllm.envs  # noqa: F401  isort: skip

# --- production shape constants -------------------------------------------
NUM_RANKS = 16  # DP=16 / EP=16
REAL_PER_RANK = 46  # measured total_num_scheduled_tokens in the decode phase
HIDDEN = 6144  # config.json hidden_size
INTERMEDIATE = 2048  # config.json moe_intermediate_size
GLOBAL_EXPERTS = 256  # config.json n_routed_experts
LOCAL_EXPERTS = GLOBAL_EXPERTS // NUM_RANKS  # 16
TOPK = 8  # config.json num_experts_per_tok
EXPERTS_START = 0  # this shard's first global expert id


def _require_tpu():
    if jax.default_backend() != "tpu":
        pytest.skip("requires a TPU")


# --------------------------------------------------------------------------
# 1. the bucket list itself (pure Python, no TPU)
# --------------------------------------------------------------------------
def test_token_bucket_extra_parses_and_inserts_48(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.platforms.tpu_platform import _get_token_paddings

    getter = envs.environment_variables["TPU_TOKEN_BUCKET_EXTRA"]
    monkeypatch.delenv("TPU_TOKEN_BUCKET_EXTRA", raising=False)
    assert getter() == []
    monkeypatch.setenv("TPU_TOKEN_BUCKET_EXTRA", "48")
    assert getter() == [48]

    # control: the default list has nothing between 32 and 64
    assert _get_token_paddings(16, 1024, []) == [
        16, 32, 64, 128, 256, 512, 1024
    ]
    # the change
    assert _get_token_paddings(16, 1024, getter()) == [
        16, 32, 48, 64, 128, 256, 512, 1024
    ]


# --------------------------------------------------------------------------
# 2. the decode MoE at a 48-row bucket vs a 64-row bucket
# --------------------------------------------------------------------------
def _make_weights(seed=0):
    key = jax.random.key(seed)
    w1k, w2k = jax.random.split(key, 2)
    w1 = (jax.random.normal(w1k,
                            (LOCAL_EXPERTS, HIDDEN, INTERMEDIATE * 2),
                            dtype=jnp.float32) / 10).astype(jnp.bfloat16)
    w2 = (jax.random.normal(w2k, (LOCAL_EXPERTS, INTERMEDIATE, HIDDEN),
                            dtype=jnp.float32) / 10).astype(jnp.bfloat16)
    return w1, w2


def _make_real_rows(seed=1):
    """The 46 real tokens each DP rank contributes, identical across arms."""
    key = jax.random.key(seed)
    hk, ik, wk = jax.random.split(key, 3)
    h = (jax.random.normal(hk, (NUM_RANKS, REAL_PER_RANK, HIDDEN),
                           dtype=jnp.float32) / 10).astype(jnp.bfloat16)
    ids = jax.random.randint(ik, (NUM_RANKS, REAL_PER_RANK, TOPK), 0,
                             GLOBAL_EXPERTS, jnp.int32)
    # With 256 global experts and 16 local ones, a uniformly routed token has
    # a 60% chance of no local expert at all and would produce an all-zero
    # output row in *both* arms -- a row that compares equal no matter what
    # the kernel does. Pin slot 0 of every token to a local expert so all 736
    # compared rows carry real work. The remaining 7 slots stay uniform over
    # the 256 global experts, so the shard still sees the production mix.
    local_slot0 = (jnp.arange(NUM_RANKS * REAL_PER_RANK, dtype=jnp.int32) %
                   LOCAL_EXPERTS).reshape(NUM_RANKS, REAL_PER_RANK)
    ids = ids.at[..., 0].set(EXPERTS_START + local_slot0)
    w = jax.nn.softmax(jax.random.normal(
        wk, (NUM_RANKS, REAL_PER_RANK, TOPK), dtype=jnp.float32),
                       axis=-1)
    return h, ids, w


def _lay_out(bucket, real_h, real_ids, real_w, junk_seed):
    """Build the post-all-gather [NUM_RANKS * bucket, HIDDEN] MoE input.

    Each rank contributes REAL_PER_RANK real rows followed by
    ``bucket - REAL_PER_RANK`` padded rows. Padded rows keep plausible expert
    ids but carry zero routing weight, which is what
    ``token_padding.zero_routing_weights_for_padding`` does in production.
    The padded rows are filled with *different* junk in each arm on purpose:
    if padding could contaminate a real row, the comparison must see it.
    """
    n = NUM_RANKS * bucket
    key = jax.random.key(junk_seed)
    jk, jik = jax.random.split(key, 2)
    h = (jax.random.normal(jk, (n, HIDDEN), dtype=jnp.float32) * 3.0).astype(
        jnp.bfloat16)
    ids = jax.random.randint(jik, (n, TOPK), 0, GLOBAL_EXPERTS, jnp.int32)
    w = jnp.zeros((n, TOPK), dtype=jnp.float32)
    for r in range(NUM_RANKS):
        lo = r * bucket
        hi = lo + REAL_PER_RANK
        h = h.at[lo:hi].set(real_h[r])
        ids = ids.at[lo:hi].set(real_ids[r])
        w = w.at[lo:hi].set(real_w[r])
    return h, ids, w


def _run_moe(h, ids, w, w1, w2, threshold):
    from vllm_torchtpu.layers.common.fused_moe_gmm import fused_moe_func
    return fused_moe_func(
        hidden_states=h,
        w1=w1,
        w2=w2,
        w1_scale=None,
        w2_scale=None,
        w1_bias=None,
        w2_bias=None,
        topk_weights=w,
        topk_ids=ids,
        experts_start=jnp.array(EXPERTS_START, dtype=jnp.int32),
        topk=TOPK,
        activation="silu",
        use_ep=True,
        use_sparse_core=True,
        onehot_moe_permute_threshold=threshold,
    )


def _real_rows(out, bucket):
    return jnp.concatenate(
        [out[r * bucket:r * bucket + REAL_PER_RANK] for r in range(NUM_RANKS)],
        axis=0)


def _time(fn, n=20):
    fn().block_until_ready()
    t0 = time.perf_counter()
    for _ in range(n):
        r = fn()
    r.block_until_ready()
    return (time.perf_counter() - t0) / n


def test_decode_moe_48_bucket_matches_64_bucket():
    _require_tpu()
    from vllm_torchtpu.layers.common.fused_moe_gmm import \
        resolve_onehot_permute_threshold

    threshold = resolve_onehot_permute_threshold()
    print(f"\nonehot_moe_permute_threshold (auto) = {threshold}")
    print(f"flat routed rows: 64-arm={NUM_RANKS * 64 * TOPK} "
          f"48-arm={NUM_RANKS * 48 * TOPK}  "
          f"(<= threshold selects the one-hot path instead of SparseCore)")

    w1, w2 = _make_weights()
    real_h, real_ids, real_w = _make_real_rows()

    h64, i64, w64 = _lay_out(64, real_h, real_ids, real_w, junk_seed=11)
    h48, i48, w48 = _lay_out(48, real_h, real_ids, real_w, junk_seed=22)

    out64 = _run_moe(h64, i64, w64, w1, w2, threshold)
    out48 = _run_moe(h48, i48, w48, w1, w2, threshold)
    out64.block_until_ready()
    out48.block_until_ready()

    a = np.asarray(_real_rows(out64, 64).astype(jnp.float32))
    b = np.asarray(_real_rows(out48, 48).astype(jnp.float32))
    assert a.shape == b.shape == (NUM_RANKS * REAL_PER_RANK, HIDDEN)

    denom = np.maximum(np.abs(a).max(), 1e-6)
    max_abs = float(np.abs(a - b).max())
    max_rel = max_abs / float(denom)
    nan64 = int(np.isnan(a).sum())
    nan48 = int(np.isnan(b).sum())
    nonzero_rows = int((np.abs(a).max(axis=1) > 0).sum())
    print(f"real rows compared: {a.shape[0]} x {a.shape[1]}")
    print(f"rows with non-zero output (i.e. rows the check can see): "
          f"{nonzero_rows}/{a.shape[0]}")
    print(f"|out64| max = {float(np.abs(a).max()):.6f}")
    print(f"max_abs_diff(48 vs 64, real rows) = {max_abs:.6e}")
    print(f"max_rel_diff(48 vs 64, real rows) = {max_rel:.6e}")
    print(f"NaNs: 64-arm={nan64} 48-arm={nan48}")

    # --- negative control: the same comparison must go red on a real change.
    # Perturb every real row, not one row: a single row may route to no local
    # expert and then contributes nothing to the comparison either way.
    h48b, i48b, w48b = _lay_out(48, real_h + jnp.bfloat16(0.5), real_ids,
                                real_w, junk_seed=22)
    out_bad = _run_moe(h48b, i48b, w48b, w1, w2, threshold)
    c = np.asarray(_real_rows(out_bad, 48).astype(jnp.float32))
    ctrl = float(np.abs(a - c).max())
    print(f"NEGATIVE CONTROL max_abs_diff (all real rows perturbed) "
          f"= {ctrl:.6e}")

    # --- timing, at the MoE-layer level only
    t64 = _time(lambda: _run_moe(h64, i64, w64, w1, w2, threshold))
    t48 = _time(lambda: _run_moe(h48, i48, w48, w1, w2, threshold))
    print(f"moe layer time: 64-arm={t64 * 1e3:.4f} ms  "
          f"48-arm={t48 * 1e3:.4f} ms  ratio={t48 / t64:.4f}")

    # --- executable size, the only resident-memory proxy reachable here
    try:
        for bucket, (hh, ii, ww) in ((64, (h64, i64, w64)), (48, (h48, i48,
                                                                  w48))):
            from vllm_torchtpu.layers.common.fused_moe_gmm import \
                fused_moe_func
            comp = jax.jit(
                fused_moe_func,
                static_argnames=("topk", "activation", "use_ep",
                                 "use_sparse_core",
                                 "onehot_moe_permute_threshold"),
            ).lower(hidden_states=hh,
                    w1=w1,
                    w2=w2,
                    w1_scale=None,
                    w2_scale=None,
                    w1_bias=None,
                    w2_bias=None,
                    topk_weights=ww,
                    topk_ids=ii,
                    experts_start=jnp.array(EXPERTS_START, dtype=jnp.int32),
                    topk=TOPK,
                    activation="silu",
                    use_ep=True,
                    use_sparse_core=True,
                    onehot_moe_permute_threshold=threshold).compile()
            ma = comp.memory_analysis()
            print(f"bucket={bucket} generated_code_bytes="
                  f"{getattr(ma, 'generated_code_size_in_bytes', None)} "
                  f"temp_bytes={getattr(ma, 'temp_size_in_bytes', None)} "
                  f"output_bytes={getattr(ma, 'output_size_in_bytes', None)}")
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"memory_analysis unavailable: {exc!r}")

    try:
        print("device memory_stats:", jax.local_devices()[0].memory_stats())
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"memory_stats unavailable: {exc!r}")

    assert nan48 == 0 and nan64 == 0, "NaN in the MoE output"
    assert nonzero_rows == a.shape[0], (
        f"only {nonzero_rows}/{a.shape[0]} compared rows carry work")
    assert ctrl > 100 * max(max_abs, 1e-6), (
        "negative control did not move the comparison; the check is hollow")
    assert max_rel < 2e-2, f"48-row MoE disagrees with 64-row MoE: {max_rel}"


if __name__ == "__main__":
    os.environ.setdefault("TPU_TOKEN_BUCKET_EXTRA", "48")
    raise SystemExit(pytest.main([__file__, "-s", "-q"]))
