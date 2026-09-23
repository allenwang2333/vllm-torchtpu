"""Parity of the few-token MoE kernel with its plain-JAX reference, at the
served shape (64 experts per shard, hidden 4096, intermediate 1024, top-10)
and a small one, including partial row tiles and the 128/256-row buckets."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.moe_lowtok import moe_lowtok, moe_lowtok_reference

FP8 = jnp.float8_e4m3fn


@pytest.fixture(autouse=True)
def _needs_a_tpu():
    import jax
    if jax.default_backend() != "tpu":
        pytest.skip("needs a TPU")


def _operands(seed, tokens, experts, hidden, inter, topk, local_fraction=0.5):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(
        rng.standard_normal((tokens, hidden)).astype(np.float32), jnp.bfloat16)
    # global ids over experts / local_fraction, so about local_fraction of the
    # routed rows belong to this shard
    n_global = int(experts / local_fraction)
    ids = np.stack([
        rng.choice(n_global, size=topk, replace=False) for _ in range(tokens)
    ])
    local = np.where(ids < experts, ids, -1).astype(np.int32)
    wts = rng.random((tokens, topk)).astype(np.float32)
    wts = wts / wts.sum(axis=1, keepdims=True)
    w13 = jnp.asarray(
        np.clip(
            rng.standard_normal((experts, hidden, 2 * inter)) * 40, -448, 448),
        FP8)
    w2 = jnp.asarray(
        np.clip(rng.standard_normal((experts, inter, hidden)) * 40, -448, 448),
        FP8)
    s13 = jnp.asarray(
        rng.random((experts, 1, 1, 2 * inter)).astype(np.float32) /
        (40 * np.sqrt(hidden)))
    s2 = jnp.asarray(
        rng.random((experts, 1, 1, hidden)).astype(np.float32) /
        (40 * np.sqrt(inter)))
    return x, jnp.asarray(local), jnp.asarray(wts,
                                              jnp.bfloat16), w13, s13, w2, s2


@pytest.mark.parametrize("tokens", [1, 4, 8, 13, 32, 64])
@pytest.mark.parametrize("shape", [(8, 512, 256, 4), (64, 4096, 1024, 10)])
def test_parity(tokens, shape):
    experts, hidden, inter, topk = shape
    ops = _operands(tokens * 7 + experts, tokens, experts, hidden, inter, topk)
    got = np.asarray(jax.block_until_ready(moe_lowtok(*ops)), np.float32)
    want = np.asarray(moe_lowtok_reference(*ops), np.float32)
    assert got.shape == (tokens, hidden)
    scale = np.max(np.abs(want)) + 1e-6
    # The same fp8 arithmetic in a different accumulation order: an activation
    # near an fp8 rounding boundary can land one fp8 step away, which moves an
    # output element by a few percent of the peak, so a per-element bound is
    # loose and the whole-output bound is tight.
    np.testing.assert_allclose(got / scale, want / scale, atol=6e-2, rtol=0)
    rel_l2 = np.linalg.norm(got - want) / (np.linalg.norm(want) + 1e-9)
    # Larger batches use a tighter aggregate-error bound.
    assert rel_l2 < (2.5e-2 if tokens < 4 else 1e-2), rel_l2


@pytest.mark.parametrize("tokens", [128, 256])
def test_parity_large_buckets(tokens):
    """The grouped path can select this kernel for larger gathered batches."""
    test_parity(tokens, (64, 4096, 1024, 10))


def test_no_local_expert_is_zero():
    experts, hidden, inter, topk = 8, 512, 256, 4
    x, ids, wts, w13, s13, w2, s2 = _operands(3, 4, experts, hidden, inter,
                                              topk)
    ids = jnp.full_like(ids, -1)
    got = np.asarray(
        jax.block_until_ready(moe_lowtok(x, ids, wts, w13, s13, w2, s2)),
        np.float32)
    assert np.all(got == 0)


@pytest.mark.parametrize("tokens", [1, 4, 8, 13, 32])
@pytest.mark.parametrize("shape", [(8, 512, 256, 4, 16),
                                   (64, 4096, 1024, 10, 128)])
def test_scores_path_matches_the_router_and_the_kernel(tokens, shape):
    """The scores entry (router folded in) equals the served router's selection
    fed to the ids entry: same experts, same renormalised bf16 weights."""
    from vllm_torchtpu.kernels.moe_lowtok import (contiguous_tables,
                                                  moe_lowtok_scores,
                                                  router_select_reference)
    experts, hidden, inter, topk, start = shape
    rng = np.random.default_rng(tokens * 11 + experts)
    x = jnp.asarray(
        rng.standard_normal((tokens, hidden)).astype(np.float32), jnp.bfloat16)
    logits = rng.standard_normal((tokens, 512)).astype(np.float32) * 2
    logits[:, start:start +
           experts] += 2.0  # some of the routed pairs land on this shard
    scores = jax.nn.softmax(jnp.asarray(logits), axis=-1)
    w13 = jnp.asarray(
        np.clip(
            rng.standard_normal((experts, hidden, 2 * inter)) * 40, -448, 448),
        FP8)
    w2 = jnp.asarray(
        np.clip(rng.standard_normal((experts, inter, hidden)) * 40, -448, 448),
        FP8)
    s13 = jnp.asarray(
        rng.random((experts, 1, 1, 2 * inter)).astype(np.float32) /
        (40 * np.sqrt(hidden)))
    s2 = jnp.asarray(
        rng.random((experts, 1, 1, hidden)).astype(np.float32) /
        (40 * np.sqrt(inter)))
    w, ids = router_select_reference(scores, topk)
    local = jnp.where((ids >= start) & (ids < start + experts), ids - start,
                      -1)
    want = np.asarray(
        jax.block_until_ready(moe_lowtok(x, local, w, w13, s13, w2, s2)),
        np.float32)
    gid, prim, copy = contiguous_tables(jnp.int32(start), experts,
                                        512 // experts)
    got = np.asarray(
        jax.block_until_ready(
            moe_lowtok_scores(x,
                              scores,
                              jnp.int32(start // experts),
                              gid,
                              prim,
                              copy,
                              w13,
                              s13,
                              w2,
                              s2,
                              topk=topk,
                              ranks=512 // experts)), np.float32)
    scale = np.max(np.abs(want)) + 1e-6
    np.testing.assert_allclose(got / scale, want / scale, atol=6e-2, rtol=0)
    assert np.linalg.norm(got - want) / (np.linalg.norm(want) + 1e-9) < 1e-2


def test_copies_placement_sums_to_the_full_model():
    """Eight ranks of 8 experts (64 global) plus 2 copy slots each: the sum of
    the ranks' outputs under the in-kernel holder assignment equals the plain
    reference over all experts, so every hit expert is served exactly once."""
    from vllm_torchtpu.kernels.moe_lowtok import (LANES, moe_lowtok_scores,
                                                  router_select_reference)
    R, LOCAL, COPIES, hidden, inter, topk, tokens = 8, 8, 2, 512, 256, 4, 5
    E = R * LOCAL
    rng = np.random.default_rng(7)
    x = jnp.asarray(
        rng.standard_normal((tokens, hidden)).astype(np.float32), jnp.bfloat16)
    W13 = np.clip(rng.standard_normal((E, hidden, 2 * inter)) * 40, -448,
                  448).astype(np.float32)
    W2 = np.clip(rng.standard_normal((E, inter, hidden)) * 40, -448,
                 448).astype(np.float32)
    S13 = rng.random(
        (E, 1, 1, 2 * inter)).astype(np.float32) / (40 * np.sqrt(hidden))
    S2 = rng.random(
        (E, 1, 1, hidden)).astype(np.float32) / (40 * np.sqrt(inter))
    logits = np.full((tokens, LANES), -30.0, np.float32)
    logits[:, :E] = rng.standard_normal(
        (tokens, E)) * 3  # only the first E table lanes are real experts
    scores = jax.nn.softmax(jnp.asarray(logits), axis=-1)
    w, ids = router_select_reference(scores, topk)
    want = np.zeros((tokens, hidden), np.float32)
    for r in range(R):
        local = jnp.where((ids >= r * LOCAL) & (ids < (r + 1) * LOCAL),
                          ids - r * LOCAL, -1)
        want += np.asarray(
            moe_lowtok_reference(
                x, local, w, jnp.asarray(W13[r * LOCAL:(r + 1) * LOCAL], FP8),
                jnp.asarray(S13[r * LOCAL:(r + 1) * LOCAL]),
                jnp.asarray(W2[r * LOCAL:(r + 1) * LOCAL], FP8),
                jnp.asarray(S2[r * LOCAL:(r + 1) * LOCAL])), np.float32)
    prim = np.full(LANES, -1, np.int32)
    prim[:E] = np.arange(E) // LOCAL
    copy = np.full(LANES, -1, np.int32)
    slots = {r: list(range(r * LOCAL, (r + 1) * LOCAL)) for r in range(R)}
    hit = np.unique(np.asarray(ids))
    for e in hit:  # every hit expert gets a copy on another rank while slots last
        r2 = (prim[e] + 1 + int(e) % (R - 1)) % R
        if len(slots[r2]) < LOCAL + COPIES and copy[e] < 0:
            copy[e] = r2
            slots[r2].append(int(e))
    got = np.zeros((tokens, hidden), np.float32)
    for r in range(R):
        gid = np.asarray(slots[r], np.int32)
        y = moe_lowtok_scores(x,
                              scores,
                              jnp.int32(r),
                              jnp.asarray(gid),
                              jnp.asarray(prim),
                              jnp.asarray(copy),
                              jnp.asarray(W13[gid], FP8),
                              jnp.asarray(S13[gid]),
                              jnp.asarray(W2[gid], FP8),
                              jnp.asarray(S2[gid]),
                              topk=topk,
                              ranks=R)
        got += np.asarray(jax.block_until_ready(y), np.float32)
    scale = np.max(np.abs(want)) + 1e-6
    np.testing.assert_allclose(got / scale, want / scale, atol=6e-2, rtol=0)
    assert np.linalg.norm(got - want) / (np.linalg.norm(want) + 1e-9) < 1e-2


def test_scores_path_skips_rows_with_zero_scores():
    """A padded row arrives with all-zero scores (token_padding.zero_scores_for_padding): the in-kernel router
    gives it zero weights, it adds no hits and its output row is zero, while the other rows match the ids entry."""
    from vllm_torchtpu.kernels.moe_lowtok import (contiguous_tables,
                                                  moe_lowtok_scores,
                                                  router_select_reference)
    experts, hidden, inter, topk, start, tokens = 8, 512, 256, 4, 16, 5
    rng = np.random.default_rng(23)
    x = jnp.asarray(
        rng.standard_normal((tokens, hidden)).astype(np.float32), jnp.bfloat16)
    logits = rng.standard_normal((tokens, 512)).astype(np.float32) * 2
    logits[:, start:start + experts] += 2.0
    scores = jax.nn.softmax(jnp.asarray(logits), axis=-1)
    scores = scores.at[2].set(0.0)  # row 2 is padding
    w13 = jnp.asarray(
        np.clip(
            rng.standard_normal((experts, hidden, 2 * inter)) * 40, -448, 448),
        FP8)
    w2 = jnp.asarray(
        np.clip(rng.standard_normal((experts, inter, hidden)) * 40, -448, 448),
        FP8)
    s13 = jnp.asarray(
        rng.random((experts, 1, 1, 2 * inter)).astype(np.float32) /
        (40 * np.sqrt(hidden)))
    s2 = jnp.asarray(
        rng.random((experts, 1, 1, hidden)).astype(np.float32) /
        (40 * np.sqrt(inter)))
    w, ids = router_select_reference(scores, topk)
    local = jnp.where((ids >= start) & (ids < start + experts), ids - start,
                      -1)
    local = local.at[2].set(-1)  # the reference: the padded row routes nowhere
    want = np.asarray(
        jax.block_until_ready(moe_lowtok(x, local, w, w13, s13, w2, s2)),
        np.float32)
    gid, prim, copy = contiguous_tables(jnp.int32(start), experts,
                                        512 // experts)
    got = np.asarray(
        jax.block_until_ready(
            moe_lowtok_scores(x,
                              scores,
                              jnp.int32(start // experts),
                              gid,
                              prim,
                              copy,
                              w13,
                              s13,
                              w2,
                              s2,
                              topk=topk,
                              ranks=512 // experts)), np.float32)
    assert np.all(got[2] == 0)
    scale = np.max(np.abs(want)) + 1e-6
    np.testing.assert_allclose(got / scale, want / scale, atol=6e-2, rtol=0)


def test_scores_entry_takes_experts_start_with_owned():
    """The served call passes the rank's first owned expert id and the owned count; the kernel divides."""
    from vllm_torchtpu.kernels.moe_lowtok import (contiguous_tables,
                                                  moe_lowtok_scores)
    experts, hidden, inter, topk, start, tokens = 8, 512, 256, 4, 24, 6
    rng = np.random.default_rng(31)
    x = jnp.asarray(
        rng.standard_normal((tokens, hidden)).astype(np.float32), jnp.bfloat16)
    logits = rng.standard_normal((tokens, 512)).astype(np.float32) * 2
    logits[:, start:start + experts] += 2.0
    scores = jax.nn.softmax(jnp.asarray(logits), axis=-1)
    w13 = jnp.asarray(
        np.clip(
            rng.standard_normal((experts, hidden, 2 * inter)) * 40, -448, 448),
        FP8)
    w2 = jnp.asarray(
        np.clip(rng.standard_normal((experts, inter, hidden)) * 40, -448, 448),
        FP8)
    s13 = jnp.asarray(
        rng.random((experts, 1, 1, 2 * inter)).astype(np.float32) /
        (40 * np.sqrt(hidden)))
    s2 = jnp.asarray(
        rng.random((experts, 1, 1, hidden)).astype(np.float32) /
        (40 * np.sqrt(inter)))
    gid, prim, copy = contiguous_tables(jnp.int32(start), experts,
                                        512 // experts)
    a = moe_lowtok_scores(x,
                          scores,
                          jnp.int32(start // experts),
                          gid,
                          prim,
                          copy,
                          w13,
                          s13,
                          w2,
                          s2,
                          topk=topk,
                          ranks=512 // experts)
    b = moe_lowtok_scores(x,
                          scores,
                          jnp.int32(start),
                          gid,
                          prim,
                          copy,
                          w13,
                          s13,
                          w2,
                          s2,
                          topk=topk,
                          ranks=512 // experts,
                          owned=experts)
    np.testing.assert_array_equal(np.asarray(jax.block_until_ready(a)),
                                  np.asarray(jax.block_until_ready(b)))


def test_sequential_assignment_balances_the_copies():
    """With three fixed hits on rank 0, three of four copied hits go to
    rank 1; the last ties and stays on rank 0, leaving loads of four and three.
    The total output must preserve every selected expert contribution.
    """
    from vllm_torchtpu.kernels.moe_lowtok import (LANES, hot_tables,
                                                  moe_lowtok_scores)
    R, LOCAL, hidden, inter, topk, tokens = 3, 8, 512, 256, 7, 1
    E = R * LOCAL
    rng = np.random.default_rng(3)
    x = jnp.asarray(
        rng.standard_normal((tokens, hidden)).astype(np.float32), jnp.bfloat16)
    W13 = np.clip(rng.standard_normal((E, hidden, 2 * inter)) * 40, -448,
                  448).astype(np.float32)
    W2 = np.clip(rng.standard_normal((E, inter, hidden)) * 40, -448,
                 448).astype(np.float32)
    S13 = rng.random(
        (E, 1, 1, 2 * inter)).astype(np.float32) / (40 * np.sqrt(hidden))
    S2 = rng.random(
        (E, 1, 1, hidden)).astype(np.float32) / (40 * np.sqrt(inter))
    hit = [0, 1, 2, 3, 4, 5,
           6]  # all owned by rank 0; 3..6 get copies on rank 1
    logits = np.full((tokens, LANES), -30.0, np.float32)
    logits[0, hit] = np.linspace(3.0, 2.0, len(hit))
    scores = jax.nn.softmax(jnp.asarray(logits), axis=-1)
    prim = np.full(LANES, -1, np.int32)
    prim[:E] = np.arange(E) // LOCAL
    copy = np.full(LANES, -1, np.int32)
    copy[3:7] = 1
    slots = {
        0: list(range(0, 8)),
        1: list(range(8, 16)) + [3, 4, 5, 6],
        2: list(range(16, 24))
    }
    hot, hp, hc = hot_tables(prim, copy, order=[3, 4, 5, 6])
    served = {}
    for r in range(R):
        gid = np.asarray(slots[r], np.int32)
        y = moe_lowtok_scores(x,
                              scores,
                              jnp.int32(r),
                              jnp.asarray(gid),
                              jnp.asarray(prim),
                              jnp.asarray(copy),
                              jnp.asarray(W13[gid], FP8),
                              jnp.asarray(S13[gid]),
                              jnp.asarray(W2[gid], FP8),
                              jnp.asarray(S2[gid]),
                              topk=topk,
                              ranks=R,
                              hot=hot,
                              hprim=hp,
                              hcopy=hc)
        served[r] = np.asarray(jax.block_until_ready(y), np.float32)
    # rank 2 serves nothing; rank 1 serves three copied experts (its output is not zero and not the whole model)
    assert np.abs(served[2]).max() == 0
    assert np.abs(served[1]).max() > 0 and np.abs(served[0]).max() > 0
    total = served[0] + served[1]
    from vllm_torchtpu.kernels.moe_lowtok import (moe_lowtok_reference,
                                                  router_select_reference)
    w, ids = router_select_reference(scores, topk)
    want = np.asarray(
        moe_lowtok_reference(x, ids, w, jnp.asarray(W13, FP8),
                             jnp.asarray(S13), jnp.asarray(W2, FP8),
                             jnp.asarray(S2)), np.float32)
    scale = np.max(np.abs(want)) + 1e-6
    np.testing.assert_allclose(total / scale, want / scale, atol=6e-2, rtol=0)
    # Verify the assignment, not just the summed output: duplicates or an
    # unchanged primary-only assignment can otherwise hide an EPLB regression.
    for rank, assigned in ((0, [0, 1, 2, 6]), (1, [3, 4, 5])):
        rank_ids = jnp.where(jnp.isin(ids, jnp.asarray(assigned)), ids, -1)
        rank_want = np.asarray(
            moe_lowtok_reference(x, rank_ids, w, jnp.asarray(W13, FP8),
                                 jnp.asarray(S13), jnp.asarray(W2, FP8),
                                 jnp.asarray(S2)), np.float32)
        np.testing.assert_allclose(served[rank] / scale,
                                   rank_want / scale,
                                   atol=6e-2,
                                   rtol=0)
