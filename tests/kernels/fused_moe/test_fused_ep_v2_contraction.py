# SPDX-License-Identifier: Apache-2.0
"""v2's fp8 expert FFN contracts in ONE dot per matmul. Pinned, because a
sibling investigation has v1's unexplained 14% pointing at split contractions.

A bisection of the fused-EP **v1** kernel's 14% rel-L2 (measured on device,
0.0077 under a patched Pallas interpreter and 0.0090 in an independent numpy
model -- so the program is right and the *lowering* is not) indicts the
combination "quantized operands **and** a contraction split into several
blocks": v1's failing arm is fp8 with num_bd1/num_bf/num_bd2 = 2/2/2, and
upstream's own v1 tests never make that combination.

v2 at ``WeightFormat.FP8`` is structurally outside that class and has no knob
that could put it inside: ``expert_ffn_fp8`` issues exactly two
``lax.dot_general`` calls, each over its full contraction, on whole expert
weight slabs resident in VMEM. The chunked forms exist only for the weight
formats that have to widen an operand first -- four-bit to fp8 per scale block,
integer to bf16 per 512-row chunk -- and neither is reachable from the fp8 path.

That is worth a test rather than a comment: "the contraction is not split" is
exactly the property a tiling change would take away silently, and if the
bisection lands on split contractions it is also the reason v2 would still be
serviceable when v1 is not.
"""

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from vllm_torchtpu.kernels.fused_moe.v2 import kernel as v2kernel  # noqa: E402

# Qwen3.5-397B-A17B-FP8: both contractions are split at every block size the
# general GMM path uses (4096/1024 = 4, 1024/512 = 2), which is what makes this
# a statement about v2 rather than about the shapes.
HIDDEN, INTER, TILE_M = 4096, 1024, 128


def _dots(jaxpr):
    return [e for e in jaxpr.eqns if e.primitive.name == "dot_general"]


def test_the_fp8_expert_ffn_is_two_unsplit_dots():
    fp8 = jnp.float8_e4m3fn
    sds = jax.ShapeDtypeStruct
    jaxpr = jax.make_jaxpr(v2kernel.expert_ffn_fp8)(
        sds((TILE_M, HIDDEN), fp8),  # act_q
        sds((TILE_M, 1), jnp.float32),  # act_scale
        sds((HIDDEN, 2 * INTER), fp8),  # w1
        sds((INTER, HIDDEN), fp8),  # w2
        sds((1, 2 * INTER), jnp.float32),  # w1s
    ).jaxpr
    dots = _dots(jaxpr)
    assert len(dots) == 2, (
        f"the fp8 FFN issued {len(dots)} matmuls; two means gate/up and down "
        "each contract in one pass, and more means the contraction was split "
        "-- the configuration a sibling bisection is currently indicting in v1"
    )

    contractions = []
    for eq in dots:
        (lhs_c, _rhs_c), _batch = eq.params["dimension_numbers"]
        contractions.append(eq.invars[0].aval.shape[lhs_c[0]])
    assert contractions == [
        HIDDEN, INTER
    ], (f"contracted over {contractions}, not the full {[HIDDEN, INTER]}")


def test_only_the_widening_weight_formats_chunk_their_contraction():
    """The chunked bodies exist, and the fp8 path cannot reach them."""
    import inspect
    src = inspect.getsource(v2kernel)
    assert "_chunked_dot" in src
    fp8_body = inspect.getsource(v2kernel.expert_ffn_fp8)
    assert "_chunked_dot" not in fp8_body
    for name in ("expert_ffn_int8", "expert_ffn_blockscale"):
        assert "_chunked_dot" in inspect.getsource(getattr(v2kernel, name)) or \
            "for b in range" in inspect.getsource(getattr(v2kernel, name)), name
