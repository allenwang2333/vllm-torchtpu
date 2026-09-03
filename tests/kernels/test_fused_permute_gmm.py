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
"""Tests for fused_permute_gmm."""

import dataclasses
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.megablox import gmm_v2 as gmm_v2_mod
from vllm_torchtpu.kernels.megablox.fused_permute_gmm.config import (
    Tuning, estimate_vmem_bytes, gather_tiling, ring_chunks)
from vllm_torchtpu.kernels.megablox.fused_permute_gmm.wrapper import \
    fused_permute_gmm
from vllm_torchtpu.kernels.megablox.gmm_v2 import gmm_v2
from vllm_torchtpu.kernels.sparse_core.ragged_gather_v2 import ragged_gather_v2

# Qwen3.5-397B GMM1: K, gate+up N, top-k and the local expert count under EP 8.
HIDDEN = 4096
GMM1_N = 2048
TOPK = 10
NUM_GROUPS = 64


def _require_tpu() -> None:
    try:
        backend = jax.default_backend()
    except Exception as exc:
        pytest.fail(f"JAX TPU backend failed to initialize: {exc}")
    if backend != "tpu":
        pytest.fail(f"Expected JAX TPU backend, got {backend}.")


def _bits(x: jax.Array) -> np.ndarray:
    """bf16 bit pattern with negative zero canonicalized.

    A revisited gm tile accumulates `+= 0.0`, which turns -0.0 into +0.0; both
    kernels do it, just not on the same rows. That is a sign bit, not a value.
    """
    bits = np.asarray(x).view(np.uint16).copy()
    bits[bits == 0x8000] = 0
    return bits


def _gmm1(lhs: jax.Array, rhs: jax.Array, scale: jax.Array,
          group_sizes: jax.Array) -> jax.Array:
    return gmm_v2(lhs=lhs,
                  rhs=rhs,
                  rhs_scale=scale,
                  group_sizes=group_sizes,
                  group_offset=jnp.zeros((1, ), jnp.int32)[0],
                  zero_initialize=False,
                  fuse_act="silu",
                  preferred_element_type=jnp.bfloat16,
                  maybe_quantize_lhs=True)


@dataclasses.dataclass(frozen=True)
class Case:
    """One routed GMM1 problem: a pool, its expert weights and a routing."""
    pool: jax.Array  # [num_tokens, HIDDEN] bf16
    rhs: jax.Array  # [num_groups, HIDDEN, GMM1_N] fp8
    scale: jax.Array  # [num_groups, 1, 1, GMM1_N] f32
    group_sizes: jax.Array  # int32[num_groups]
    gather_indices: jax.Array  # int32[num_tokens * topk]

    @classmethod
    def routed(cls,
               num_tokens: int,
               live_groups: int = NUM_GROUPS,
               num_groups: int = NUM_GROUPS,
               seed: int = 0) -> "Case":
        """Every token picks TOPK experts among `live_groups` of `num_groups`.

        `live_groups < num_groups` is the routing decode actually produces: a
        handful of experts carry nearly every row, which is the regime a tile
        sized from the mean row count per group gets wrong.
        """
        rng = np.random.default_rng(seed)
        live = rng.choice(num_groups, size=live_groups, replace=False)
        ids = rng.choice(live, size=num_tokens * TOPK).astype(np.int32)
        order = np.argsort(ids, kind="stable")
        rows = np.arange(num_tokens * TOPK, dtype=np.int32)[order] // TOPK
        group_sizes = np.bincount(ids, minlength=num_groups).astype(np.int32)
        return cls.build(num_tokens, num_groups, jnp.asarray(group_sizes),
                         jnp.asarray(rows), seed)

    @classmethod
    def round_robin(cls, num_tokens: int, topk: int, num_groups: int,
                    seed: int) -> "Case":
        """`num_tokens * topk` rows dealt evenly over the groups."""
        size_m = num_tokens * topk
        rows = jnp.arange(size_m, dtype=jnp.int32) % num_tokens
        group_sizes = jnp.full((num_groups, ), size_m // num_groups, jnp.int32)
        return cls.build(num_tokens, num_groups, group_sizes, rows, seed)

    @classmethod
    def build(cls, num_tokens: int, num_groups: int, group_sizes: jax.Array,
              gather_indices: jax.Array, seed: int) -> "Case":
        k_w, k_x = jax.random.split(jax.random.key(seed))
        rhs = (jax.random.normal(k_w,
                                 (num_groups, HIDDEN, GMM1_N), jnp.bfloat16) *
               0.1).astype(jnp.float8_e4m3fn)
        scale = jnp.full((num_groups, 1, 1, GMM1_N), 0.1, jnp.float32)
        pool = jax.random.normal(k_x, (num_tokens, HIDDEN), jnp.bfloat16) * 0.5
        return cls(pool, rhs, scale, group_sizes, gather_indices)

    @property
    def valid_rows(self) -> int:
        return int(jnp.sum(self.group_sizes))

    def materialized(self) -> jax.Array:
        """The reference: GMM1 fed the permuted pool."""
        return _gmm1(self.pool[self.gather_indices], self.rhs, self.scale,
                     self.group_sizes)

    def fused(self,
              pool: jax.Array | None = None,
              zero_initialize: bool = False,
              **kwargs) -> jax.Array:
        return fused_permute_gmm(self.pool if pool is None else pool,
                                 self.rhs,
                                 self.group_sizes,
                                 self.gather_indices,
                                 self.scale,
                                 group_offset=jnp.zeros((1, ), jnp.int32),
                                 zero_initialize=zero_initialize,
                                 fuse_act="silu",
                                 preferred_element_type=jnp.bfloat16,
                                 maybe_quantize_lhs=True,
                                 **kwargs)

    def assert_bitwise(self, out: jax.Array, reference: jax.Array) -> None:
        """Equal over the valid prefix; rows past it are never written."""
        n = self.valid_rows
        np.testing.assert_array_equal(_bits(out[:n]), _bits(reference[:n]))


def _gather_tiles(num_tokens: int, num_groups: int = NUM_GROUPS) -> int:
    """Gm tiles of a routing with `num_tokens * TOPK` rows, by the kernel's law."""
    size_m = num_tokens * TOPK
    return -(-size_m // gather_tiling(size_m, num_groups))


def _tokens_for_tiles(tiles: int) -> int:
    """Smallest token count whose gather grid is exactly `tiles` tiles."""
    for n in range(6, 512):
        if _gather_tiles(n) == tiles:
            return n
    raise AssertionError(f"no token count in range gives {tiles} tiles")


# Every grid length the warm-up guard has to cover, derived from the depth
# rather than fixed: at depth 3 the exposed set is {1, 2}, at 7 it is {1..6}.
DEGENERATE_TOKENS = sorted(
    {6, 8, 12} | {_tokens_for_tiles(t)
                  for t in range(3,
                                 Tuning().lookahead)})


@pytest.mark.parametrize("live_groups", [64, 24])
@pytest.mark.parametrize("num_tokens", [512, 2048])
def test_fused_permute_gmm_matches_materialized(num_tokens, live_groups):
    """The fused permute is bit-for-bit the materialized one.

    Concentrated routing (`live_groups=24`) is a separate case, not a variant:
    a gather tile sized from the mean rows per group splits the few fat groups
    it produces, and the split tiles' dots are fully exposed.
    """
    _require_tpu()
    case = Case.routed(num_tokens, live_groups=live_groups)
    case.assert_bitwise(case.fused(), case.materialized())


@pytest.mark.parametrize("num_tokens", DEGENERATE_TOKENS)
def test_fused_permute_gmm_survives_a_grid_shorter_than_the_lookahead(
        num_tokens):
    """A grid with fewer steps than the prefetch depth must leave no live DMA.

    The warm-up issues the first few tiles, but tile t is waited by step t and
    by nothing else, and the step count is a runtime value. A tile issued with
    no step to wait it leaves a nonzero DMA semaphore at Mosaic exit, which
    halts the core and fails the whole slice rather than the result, so an
    output comparison cannot catch it. Called twice because a leaked semaphore
    is consumed by the next call's wait and surfaces only on the one after.
    """
    _require_tpu()
    tiles = _gather_tiles(num_tokens)
    assert tiles < Tuning().lookahead, (
        f"{num_tokens} tokens gives {tiles} tiles, which is not below the "
        f"prefetch depth {Tuning().lookahead} -- this shape no longer "
        "exercises the warm-up guard")
    case = Case.routed(num_tokens, live_groups=1, seed=num_tokens)

    for _ in range(2):
        out = jax.block_until_ready(case.fused())
    case.assert_bitwise(out, case.materialized())


@pytest.mark.parametrize("num_tokens,topk", [(8, 5), (13, 4), (64, 10)])
def test_fused_permute_gmm_handles_unaligned_row_counts(num_tokens, topk):
    """size_m = num_tokens * topk need not be a multiple of the sublane block.

    The kernel reshapes its output by that block, so the wrapper pads the row
    count up to it the way gmm_v2 pads its lhs; without the pad this raises
    from inside the kernel rather than falling back.
    """
    _require_tpu()
    case = Case.round_robin(num_tokens, topk, num_groups=4, seed=11)
    out = case.fused()
    assert out.shape[0] == num_tokens * topk
    case.assert_bitwise(out, case.materialized())


def test_fused_permute_gmm_accepts_a_blocked_pool():
    """The pool may arrive already blocked as [t, K // 128, 128].

    That is the layout the per-row DMA reads either way; handing it in saves
    the wrapper's reshape, and both forms must give the same result.
    """
    _require_tpu()
    case = Case.routed(512, num_groups=32, live_groups=32, seed=21)
    lanes = pltpu.get_tpu_info().num_lanes
    blocked = case.pool.reshape(case.pool.shape[0], HIDDEN // lanes, lanes)
    case.assert_bitwise(case.fused(pool=blocked), case.fused())


def test_fused_permute_gmm_zero_initializes_the_tail():
    """zero_initialize covers the rows past the last group.

    The zero-init pair has to stay last in the kernel body, after the gather
    bundle spliced in ahead of it, so the rows it writes are not re-gathered.
    """
    _require_tpu()
    case = Case.routed(512, num_groups=32, live_groups=32, seed=31)
    # leave a tail of rows past every group for zero_initialize to cover
    case = dataclasses.replace(case,
                               group_sizes=case.group_sizes.at[-1].add(-64),
                               pool=case.pool + 1.0)
    out = case.fused(zero_initialize=True)
    tail = np.asarray(out[case.valid_rows:], np.float32)
    np.testing.assert_array_equal(tail, np.zeros_like(tail))
    case.assert_bitwise(out, case.materialized())


def test_dispatch_forms_agree():
    """Every dispatch form feeding GMM1 must produce the same GMM1.

    The five forms are the ones a MoE layer can choose between: hand GMM1 an
    already-permuted activation (`ideal`, the floor -- the permute is free),
    materialize the permute with a one-hot matmul, with an XLA gather, or with
    the SparseCore ragged gather, or fold it into GMM1 (`fused`).
    """
    _require_tpu()
    case = Case.routed(2048, seed=3)
    pool, idx, gs = case.pool, case.gather_indices, case.group_sizes
    num_tokens = pool.shape[0]
    valid = jnp.sum(gs, dtype=jnp.int32)
    onehot = jax.nn.one_hot(idx, num_tokens, dtype=pool.dtype)
    # materialised off-graph, so `ideal` is a different program from `xla_gather`
    permuted = jnp.asarray(np.asarray(pool)[np.asarray(idx)])
    forms = {
        "ideal":
        lambda: _gmm1(permuted, case.rhs, case.scale, gs),
        "onehot":
        lambda: _gmm1(onehot @ pool, case.rhs, case.scale, gs),
        "xla_gather":
        lambda: _gmm1(pool[idx], case.rhs, case.scale, gs),
        "ragged_sc":
        lambda: _gmm1(
            ragged_gather_v2(pool, idx, jnp.array([0], dtype=jnp.int32), valid[
                None]), case.rhs, case.scale, gs),
        "fused":
        case.fused,
    }

    reference = None
    for name, fn in forms.items():
        out = jax.block_until_ready(jax.jit(fn)())
        prefix = np.asarray(out[:case.valid_rows], np.float32)
        if reference is None:
            reference = prefix
            continue
        # one-hot and the SparseCore gather reduce in a different order, so they
        # match to tolerance; the XLA gather and the fused path are exact.
        np.testing.assert_allclose(prefix,
                                   reference,
                                   atol=2e-2,
                                   rtol=2e-2,
                                   err_msg=f"{name} disagrees with ideal")


@pytest.mark.parametrize("rows,size_group,expected", [
    (10240, 64, 192),
    (81920, 64, 512),
    (640, 64, 64),
    (20480, 512, 64),
    (128, 1, 160),
])
def test_gather_tiling_is_live_rows_per_group(rows, size_group, expected):
    """tile_m tracks the live rows a gm tile actually carries.

    `rows` is what the caller passes: the routed rows divided by ep_size, since
    only that fraction is local. The result is clamped to [64, 512] and aligned
    to 32.
    """
    assert gather_tiling(rows, size_group) == expected


def _gmm_cfgs(rhs_quant_block: int) -> "gmm_v2_mod.GmmConfigs":
    return gmm_v2_mod.GmmConfigs(
        tiles=gmm_v2_mod.TileSizes(tile_m=192,
                                   tile_k=4096,
                                   tile_n=1024,
                                   bucket_base=32),
        dims=gmm_v2_mod.Dimensions(size_m=81920,
                                   size_k=4096,
                                   size_n=2048,
                                   size_group=64,
                                   size_lhs_group=64,
                                   size_lhs_sublane=16),
        lhs_cfgs=gmm_v2_mod.InputConfigs(quant_dtype=None,
                                         quant_block_size=None,
                                         dtype=jnp.bfloat16),
        rhs_cfgs=gmm_v2_mod.InputConfigs(quant_dtype=jnp.float8_e4m3fn,
                                         quant_block_size=rhs_quant_block,
                                         dtype=jnp.float8_e4m3fn,
                                         has_scale=True),
        out_dtype=jnp.bfloat16,
        acc_dtype=jnp.float32,
        zero_init=False,
        fuse_act="silu")


def test_blocks_per_tile_follows_the_collapsed_k_loop():
    """A dequantized-before-matmul rhs collapses the k loop, and the count must follow.

    `inner_kernel` rebinds `rhs_qbs = tile_k` in that case, so its (n, k) loop runs
    one k block. The issue hook slices its rows by this count, so a count derived
    from the scale block instead would issue a fraction of the tile's rows while
    `wait_tile` still waits for all of them -- a nonzero DMA semaphore at exit,
    which halts the core rather than returning a wrong answer.
    """
    _require_tpu(
    )  # `should_dequantize_before_matmul` reads the MXU column size
    mxu = pltpu.get_tpu_info().mxu_column_size
    n_blocks = -(-2 * 1024 // mxu)

    collapsed = _gmm_cfgs(rhs_quant_block=mxu // 4)
    assert collapsed.rhs_cfgs.should_dequantize_before_matmul
    assert collapsed.blocks_per_tile(mxu) == n_blocks

    wide = _gmm_cfgs(rhs_quant_block=4096)
    assert not wide.rhs_cfgs.should_dequantize_before_matmul
    assert wide.blocks_per_tile(mxu) == n_blocks


def test_ring_pad_is_priced_into_the_vmem_estimate():
    """The VMEM estimate must price the ring the shape actually allocates.

    `gather_tile_sizes` shrinks the tile until `estimate_vmem_bytes` fits, so an
    estimate narrower than `ring_chunks` hands back a tile whose ring overflows.
    They agree trivially when `size_k` is a multiple of 1024; the case that
    separates them is one where the pad has to round up, so this uses
    `size_k=3584` (28 lane-blocks + 8 = 36, rounded to 40).

    The estimate is read through `num_slots`, which scales the gather term and
    nothing else. Differencing a padded estimate against a zero-pad one would
    not work: `ring_chunks(28, 0)` rounds to 40 as well, so both sides read zero
    and the assertion would hold for any ring width.
    """
    size_k, lanes, tile_m, tile_n, fuse = 3584, 128, 192, 1024, 2
    tuning = Tuning()
    chunks = ring_chunks(size_k // lanes, tuning.ring_pad_slots)
    assert chunks == 40 and chunks * lanes != size_k + tuning.ring_pad_slots * lanes

    dims = gmm_v2_mod.Dimensions(size_m=81920,
                                 size_k=size_k,
                                 size_n=2048,
                                 size_group=64,
                                 size_lhs_group=64,
                                 size_lhs_sublane=16)
    lhs_cfgs = gmm_v2_mod.InputConfigs(quant_dtype=jnp.float8_e4m3fn,
                                       quant_block_size=size_k,
                                       dtype=jnp.bfloat16)
    rhs_cfgs = gmm_v2_mod.InputConfigs(quant_dtype=jnp.float8_e4m3fn,
                                       quant_block_size=128,
                                       dtype=jnp.float8_e4m3fn,
                                       has_scale=True)

    # One extra slot costs exactly one ring row-block: the only term it scales.
    def est(num_slots):
        return estimate_vmem_bytes(dims,
                                   lhs_cfgs,
                                   rhs_cfgs,
                                   tile_m,
                                   tile_n,
                                   fuse,
                                   tuning=dataclasses.replace(
                                       tuning, num_slots=num_slots),
                                   num_lanes=lanes)

    per_slot = est(tuning.num_slots + 1) - est(tuning.num_slots)
    itemsize = jnp.dtype(jnp.bfloat16).itemsize
    assert per_slot == tile_m * chunks * lanes * itemsize


def test_tuning_rejects_a_ring_that_aliases_before_its_wait():
    with pytest.raises(ValueError, match="num_slots must exceed lookahead"):
        Tuning(num_slots=3, lookahead=3)


def test_smem_index_limit_follows_the_hardware(monkeypatch):
    """The index budget is a fraction of the device's SMEM, not a per-generation literal."""
    monkeypatch.setattr(
        "jax.experimental.pallas.tpu.get_tpu_info",
        lambda: types.SimpleNamespace(smem_capacity_bytes=2 << 20))
    assert Tuning.from_device().smem_index_limit_bytes == 960_000


def test_smem_index_limit_falls_back_without_a_device(monkeypatch):
    """Op construction reaches this before there is a device; the v7x measurement stands."""

    def _boom():
        raise ValueError("Unsupported TPU device kind: cpu")

    monkeypatch.setattr("jax.experimental.pallas.tpu.get_tpu_info", _boom)
    assert Tuning.from_device().smem_index_limit_bytes == 480_000
