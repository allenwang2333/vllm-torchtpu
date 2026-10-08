"""Bit-exactness of the replicated-qr `fp8_attention_qo` body on 8 CPU devices.

Run by tests/models/vllm/test_dsv41_v2_decode_local_q.py in a subprocess with
``XLA_FLAGS=--xla_force_host_platform_device_count=8`` (the flag must be set
before JAX initializes its backend, which a pytest session has already done).
Compares, per rank, the new path against the production token-parallel
sequence it replaces (`fp8_attention`'s ``_token_parallel_body`` followed by
the torch-side ``_o_proj``) with real ``shard_map`` / ``all_to_all``.
"""

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.layers.adapter.linear_common import _quantized_matmul_jax
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention_kernels as ak
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)

TP = 8
HEADS_PER_GROUP = 8
HEAD_DIM = 16
Q_LORA = 16
R = 8
PAGE = 4
assert len(jax.devices()) == TP, jax.devices()
mesh = Mesh(np.array(jax.devices()), ("tp",))
rng = np.random.default_rng(0)


# Pure-jnp, per-token stand-ins for the TPU-only Pallas kernels. Both paths
# call the same stand-ins; they depend on every operand so a wrong slice,
# pad or rank offset changes the result.
def rope_ref(x, pos, cs):
    return x + cs[pos][:, None, : x.shape[-1]].astype(x.dtype)


def attn_ref(q, s_p, m_p, r_p, s_r, m_r, pgs, q_loc, snk, **kwargs):
    rec = (s_r.sum(1) + m_r.sum(1)).astype(q.dtype)[:, None, None]
    return (
        q
        + rec
        + q_loc.sum().astype(q.dtype)
        + snk[: q.shape[1]].astype(q.dtype)[None, :, None]
    )


def o_proj_ref(x, pos, cs, w, ws, head_dim):
    t, heads, d = x.shape
    x = (x + cs[pos][:, None, :d].astype(x.dtype)).astype(jnp.float32)
    groups = heads // HEADS_PER_GROUP
    y = jnp.einsum(
        "tgd,dgr->tgr",
        x.reshape(t, groups, HEADS_PER_GROUP * d),
        w.astype(jnp.float32).reshape(HEADS_PER_GROUP * d, groups, -1),
    )
    y = y * ws.reshape(groups, -1)[None]
    return y.reshape(t, -1).astype(jnp.bfloat16)


def make_inputs(tokens, *, tp=TP, local_groups=1):
    total_groups = tp * local_groups
    heads = total_groups * HEADS_PER_GROUP

    def b(*s):
        return jnp.asarray(rng.integers(0, 2, size=s), dtype=jnp.bfloat16)

    def f8(*s):
        return jnp.asarray(rng.integers(0, 2, size=s), dtype=jnp.float8_e4m3fn)

    return dict(
        qr=b(tokens, Q_LORA),
        wq_b_full=f8(Q_LORA, heads * HEAD_DIM),
        wq_b_full_scale=jnp.ones((heads * HEAD_DIM,), jnp.float32),
        positions=jnp.asarray(rng.integers(0, 32, size=tokens), jnp.int32),
        cos_sin_cache=jnp.asarray(rng.integers(0, 2, size=(32, HEAD_DIM)), jnp.float32),
        swa_pool=jnp.zeros((4, PAGE, 8), jnp.bfloat16),
        main_pool=jnp.zeros((4, PAGE, 8), jnp.bfloat16),
        rope_pool=jnp.zeros((4, PAGE, 8), jnp.bfloat16),
        swa_records=jnp.asarray(rng.integers(0, 2, size=(tokens, 4)), jnp.int32),
        main_records=jnp.asarray(rng.integers(0, 2, size=(tokens, 2)), jnp.int32),
        pages=jnp.zeros((2, 2), jnp.int32),
        query_start_loc=jnp.asarray([0, tokens // 3, tokens], jnp.int32),
        full_sink=jnp.asarray(rng.integers(0, 2, size=heads), jnp.float32),
        wo_a_full=f8(HEADS_PER_GROUP * HEAD_DIM, total_groups * R),
        wo_a_full_scale=jnp.asarray(
            rng.choice([0.5, 1.0, 2.0], size=total_groups * R), jnp.float32
        ),
        valid=jnp.asarray(rng.integers(0, 2, size=tokens), jnp.int32),
    )


def reference_per_rank(tokens, a, *, tp=TP, local_groups=1, tp_mesh=mesh):
    """Production decode sequence: `_token_parallel_body` + torch `_o_proj`."""
    tokens_local = tokens // tp
    pad_rows = -tokens_local % 8
    local_r = local_groups * R

    def body(
        qr, wq, wqs, pos, cs, s_p, m_p, r_p, s_r, m_r, pgs, q_loc, f_snk, w, ws, ok
    ):
        r = jax.lax.axis_index("tp")
        cols = local_groups * HEADS_PER_GROUP * HEAD_DIM
        wq_r = jax.lax.dynamic_slice_in_dim(wq, r * cols, cols, axis=1)
        wqs_r = jax.lax.dynamic_slice_in_dim(wqs, r * cols, cols, axis=0)
        q = _quantized_matmul_jax(qr, wq_r, wqs_r).reshape(tokens, -1, HEAD_DIM)
        q = rope_ref(q, pos, cs)
        q_tok = jax.lax.all_to_all(
            q, axis_name="tp", split_axis=0, concat_axis=1, tiled=True
        )
        s_r_tok, m_r_tok, q_loc_tok = ak.slice_token_parallel_attention_inputs(
            s_r, m_r, q_loc, r.reshape(1), tp, tokens
        )
        if pad_rows:
            q_tok = jnp.pad(q_tok, ((0, pad_rows), (0, 0), (0, 0)))
            s_r_tok = jnp.pad(s_r_tok, ((0, pad_rows), (0, 0)), constant_values=-1)
            m_r_tok = jnp.pad(m_r_tok, ((0, pad_rows), (0, 0)), constant_values=-1)
        out_tok = attn_ref(
            q_tok, s_p, m_p, r_p, s_r_tok, m_r_tok, pgs, q_loc_tok, f_snk
        )
        if pad_rows:
            out_tok = out_tok[:-pad_rows]
        out = jax.lax.all_to_all(
            out_tok, axis_name="tp", split_axis=1, concat_axis=0, tiled=True
        )
        out = jnp.where(ok[:, None, None] != 0, out, jnp.zeros((), out.dtype))
        w_r = jax.lax.dynamic_slice_in_dim(w, r * local_r, local_r, axis=1)
        ws_r = jax.lax.dynamic_slice_in_dim(ws, r * local_r, local_r, axis=0)
        return o_proj_ref(out, pos, cs, w_r, ws_r, HEAD_DIM)[None]

    fn = jax.shard_map(
        body, mesh=tp_mesh, in_specs=(P(),) * 16, out_specs=P("tp"), check_vma=False
    )
    return np.asarray(
        fn(
            a["qr"],
            a["wq_b_full"],
            a["wq_b_full_scale"],
            a["positions"],
            a["cos_sin_cache"],
            a["swa_pool"],
            a["main_pool"],
            a["rope_pool"],
            a["swa_records"],
            a["main_records"],
            a["pages"],
            a["query_start_loc"],
            a["full_sink"],
            a["wo_a_full"],
            a["wo_a_full_scale"],
            a["valid"],
        )
    )


def candidate_per_rank(fn, tokens, a, *, tp=TP, local_groups=1):
    real_shard_map = jax.shard_map

    def per_rank_shard_map(body, *, mesh, in_specs, out_specs, check_vma):
        # Expose every rank's output and feed each rank its own tp_rank
        # (the production op passes it as a per-rank "replicated" array).
        assert len(in_specs) == 17, len(in_specs)
        in_specs = tuple(P("tp") if i == 13 else s for i, s in enumerate(in_specs))
        return real_shard_map(
            lambda *args: body(*args)[None],
            mesh=mesh,
            in_specs=in_specs,
            out_specs=P("tp"),
            check_vma=check_vma,
        )

    unused_local = jnp.zeros((1,))
    with (
        mock.patch("jax.shard_map", side_effect=per_rank_shard_map),
        mock.patch(
            "vllm_torchtpu.models.vllm.deepseek_v4.attention._rope_jax",
            side_effect=rope_ref,
        ),
        mock.patch(
            "vllm_torchtpu.models.vllm.deepseek_v4.attention._o_proj_jax",
            side_effect=o_proj_ref,
        ),
        mock.patch.object(ak, "sparse_fp8_attention_jax", side_effect=attn_ref),
    ):
        out = fn(
            a["qr"],
            a["qr"],
            unused_local,
            unused_local,
            a["wq_b_full"],
            a["wq_b_full_scale"],
            a["positions"],
            a["cos_sin_cache"],
            a["swa_pool"],
            a["main_pool"],
            a["rope_pool"],
            a["swa_records"],
            a["main_records"],
            a["pages"],
            a["query_start_loc"],
            jnp.zeros((local_groups * HEADS_PER_GROUP,)),
            a["full_sink"],
            jnp.arange(tp, dtype=jnp.int32),
            unused_local,
            unused_local,
            a["wo_a_full"],
            a["wo_a_full_scale"],
            a["valid"],
            0.125,
            PAGE,
            None,
            False,
            HEAD_DIM,
            True,
            False,
        )
    return np.asarray(out)


captured = {}
real_jax_op = None
mock_tp = mock.MagicMock()
mock_tp.device_group.size.return_value = TP
from torch_tpu._internal import pallas  # noqa: E402

real_jax_op = pallas.jax_op


def recorder(name, fn, **kwargs):
    captured[name] = fn
    return real_jax_op(name, fn, **kwargs)


with (
    set_vllm_model_wrapper_context(mesh=mesh),
    mock.patch("vllm.distributed.get_tp_group", return_value=mock_tp),
    mock.patch("torch_tpu._internal.pallas.get_pallas_mesh", return_value=mesh),
    mock.patch("torch_tpu._internal.pallas.jax_op", side_effect=recorder),
):
    ak._attention_op = None
    ak.initialize_paged_attention()
fn = captured["pallas::dsv41v2_sparse_fp8_attention_qo"]

for tokens in (96, 192, 56):
    a = make_inputs(tokens)
    ref = reference_per_rank(tokens, a)
    got = candidate_per_rank(fn, tokens, a)
    assert ref.shape == got.shape == (TP, tokens, R), (ref.shape, got.shape)
    assert ref.dtype == got.dtype, (ref.dtype, got.dtype)
    assert np.isfinite(ref.astype(jnp.float32)).all()
    assert ref.astype(np.float32).any(), "degenerate reference"
    # Ranks differ from one another (each owns one o_group).
    assert not np.array_equal(ref[0], ref[1])
    mism = np.argwhere(ref != got)
    assert mism.size == 0, (
        f"tokens={tokens}: {mism.shape[0]} mismatches, first {mism[:5]}"
    )
    print(f"tokens={tokens}: bit-exact on {TP} ranks", flush=True)

# Verify TP=4 with 2 o_groups per rank (16 local heads/rank, 8 total groups).
mesh_tp4 = Mesh(np.array(jax.devices()[:4]), ("tp",))
mock_tp.device_group.size.return_value = 4
with (
    set_vllm_model_wrapper_context(mesh=mesh_tp4),
    mock.patch("vllm.distributed.get_tp_group", return_value=mock_tp),
    mock.patch("torch_tpu._internal.pallas.get_pallas_mesh", return_value=mesh_tp4),
    mock.patch("torch_tpu._internal.pallas.jax_op", side_effect=recorder),
):
    ak._attention_op = None
    ak.initialize_paged_attention()
fn_tp4 = captured["pallas::dsv41v2_sparse_fp8_attention_qo"]

for tokens in (24, 96, 144, 28):
    a = make_inputs(tokens, tp=4, local_groups=2)
    ref = reference_per_rank(tokens, a, tp=4, local_groups=2, tp_mesh=mesh_tp4)
    got = candidate_per_rank(fn_tp4, tokens, a, tp=4, local_groups=2)
    assert ref.shape == got.shape == (4, tokens, 2 * R), (ref.shape, got.shape)
    assert ref.dtype == got.dtype, (ref.dtype, got.dtype)
    assert np.isfinite(ref.astype(jnp.float32)).all()
    assert ref.astype(np.float32).any(), "degenerate reference"
    assert not np.array_equal(ref[0], ref[1])
    mism = np.argwhere(ref != got)
    assert mism.size == 0, (
        f"tp=4 tokens={tokens}: {mism.shape[0]} mismatches, first {mism[:5]}"
    )
    print(f"tokens={tokens}: bit-exact on 4 ranks (2 groups/rank)", flush=True)
print("EXACTNESS_OK")
