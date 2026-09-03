"""Stage-A-only launcher, parametrised by kernel body and q layout.

`run_scores_kernel` is a closure nested inside `streamindex_topk`
(`streamindex_topk.py:607-739` / `streamindex_topk_c.py`'s copy of it), so it
cannot be imported.  The launch below is a **verbatim transcription** of that
closure with the enclosing scope's values passed in explicitly -- the same
device `probes/_stage_a.py` uses, generalised over which module's `kernel` is
launched and which `q_layout` it is launched with.

The transcription is a place a divergence could hide, so it is checked rather
than asserted: `bench_headmajor.py` runs an `incumbent` arm that launches the
*unmodified* `src/.../streamindex_topk.py::kernel` through this same wrapper,
and requires it to be bit-identical to the `head_minor` arm.  If the copy or
the transcription drifted, that check fails first.

Geometry is the production chunked-prefill row `prod_prefill_q1024_ctx9216`
from `instruments/bench_streamidx_topk.py`: 1 sequence, q_len 1024, ctx 9216,
max_model_len 9216, page_size 1024, kv_packing 4, lkv_dim 256, H 32, D 128,
bkv_p 2, bq_sz 128, seq_batch_size 1, case MIXED.  That is the program P7b and
R3 dumped as `StreamIdxTC-m-bq_128-bkvp_2.1`.
"""

from __future__ import annotations

import functools
import hashlib
import os
import pathlib
import sys

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_HERE = pathlib.Path(__file__).resolve().parent
_INSTRUMENTS = _HERE.parent.parent / "instruments"
if str(_INSTRUMENTS) not in sys.path:
    sys.path.insert(0, str(_INSTRUMENTS))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import streamindex_topk_c as sxc  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as sx  # noqa: E402

DEFAULT_VMEM_LIMIT_BYTES = sxc.DEFAULT_VMEM_LIMIT_BYTES
MlaCase = sxc.MlaCase
align_to = sxc.align_to
cdiv = sxc.cdiv

# arm name -> (module supplying `kernel` + `prepare_*`, q_layout)
ARMS = {
    "incumbent": (sx, None),
    "head_minor": (sxc, "head_minor"),
    "head_major": (sxc, "head_major"),
    "head_major_hbm": (sxc, "head_major_hbm"),
}


def source_md5(mod) -> str:
    return hashlib.md5(pathlib.Path(mod.__file__).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# The production prefill operating point.
# ---------------------------------------------------------------------------

PREFILL = dict(
    num_seqs=1,
    q_len=1024,
    context_len=9216,
    max_model_len=9216,
    page_size=1024,
    kv_packing=4,
    lkv_dim=256,
    h_idx=32,
    d_idx=128,
    k=2048,
    compression_ratio=1,
    bkv_p=2,
    bq_per_block=128,
)


def prefill_geometry(q_len=None, context_len=None):
    g = dict(PREFILL)
    if q_len is not None:
        g["q_len"] = q_len
    if context_len is not None:
        g["context_len"] = context_len
        g["max_model_len"] = max(g["max_model_len"], context_len)
    g["pages_per_seq"] = cdiv(g["max_model_len"], g["page_size"])
    g["num_sublanes_total"] = (align_to(g["pages_per_seq"], g["bkv_p"]) *
                               g["page_size"] // 128)
    g["num_positions"] = g["num_sublanes_total"] * 128
    g["num_tokens"] = g["num_seqs"] * g["q_len"]
    g["total_pages"] = g["num_seqs"] * g["pages_per_seq"]
    g["bq_sz"] = g["bq_per_block"]
    g["bkv_sz"] = g["bkv_p"] * g["page_size"]
    g["num_bq"] = max(1, cdiv(g["q_len"], g["bq_sz"]))
    g["num_bkv"] = max(
        1, cdiv(g["context_len"] // g["compression_ratio"], g["bkv_sz"]))
    return g


def _make_cache(key, total_pages, page_size, kv_packing, lkv_dim, d_idx):
    """The production indexer K record: [fp8 x D | e8m0 scale | pad]."""
    shape = (total_pages, page_size // kv_packing, kv_packing, lkv_dim)
    raw = jax.random.bits(key, shape, dtype=jnp.uint8)
    exp = jnp.uint8(5) + ((raw >> jnp.uint8(3)) & jnp.uint8(3))
    vals = (raw & jnp.uint8(0x87)) | (exp << jnp.uint8(3))
    lane = jnp.arange(lkv_dim, dtype=jnp.int32)
    return jnp.where(lane < d_idx, vals,
                     jnp.where(lane == d_idx, jnp.uint8(127), jnp.uint8(0)))


def build_prefill_inputs(g, seed: int = 0):
    """Device-resident inputs at the prefill operating point.

    Constructed the same way `instruments/bench_streamidx_topk.py::build_inputs`
    does (that file is read-only and is not edited; the construction is
    reproduced so this launcher has no import-time dependency on it).
    """
    keys = jax.random.split(jax.random.key(seed), 4)
    q = jax.random.normal(keys[0], (g["num_tokens"], g["h_idx"], g["d_idx"]),
                          jnp.float32).astype(jnp.float8_e4m3fn)
    weights = jax.random.uniform(keys[1], (g["num_tokens"], g["h_idx"]),
                                 jnp.float32, 0.25, 1.75).astype(jnp.bfloat16)
    cache_kv = _make_cache(keys[2], g["total_pages"], g["page_size"],
                           g["kv_packing"], g["lkv_dim"], g["d_idx"])
    block_table = jax.random.permutation(
        keys[3], jnp.arange(g["total_pages"], dtype=jnp.int32)).reshape(
            g["num_seqs"], g["pages_per_seq"])
    seq_lens = jnp.full((g["num_seqs"], ), g["context_len"], jnp.int32)
    cu_q_lens = jnp.arange(g["num_seqs"] + 1, dtype=jnp.int32) * g["q_len"]
    return dict(
        q_raw=q,
        weights_raw=weights,
        cache_kv=cache_kv,
        seq_lens=seq_lens,
        page_indices=block_table.reshape(-1),
        cu_q_lens=cu_q_lens,
        distribution=jnp.array([0, 0, g["num_seqs"]], jnp.int32),
    )


def prepared_operands(arm, g, raw):
    """`q` and `indexer_weights` as the op hands them to the pallas call.

    For `head_major_hbm` this is where the (T, H, D) -> (H, T, D) transpose
    happens; it is XLA work outside the kernel and is timed separately.
    """
    mod, layout = ARMS[arm]
    if layout is None or layout == "head_minor":
        q = sxc.prepare_q_inputs(raw["q_raw"], "head_minor")
    else:
        q = sxc.prepare_q_inputs(raw["q_raw"], layout)
    w = sxc.prepare_index_weights(raw["weights_raw"], raw["q_raw"].dtype)
    return q, w


# ---------------------------------------------------------------------------
# The launcher.  Verbatim transcription of `run_scores_kernel`'s MIXED call.
# ---------------------------------------------------------------------------


def make_stage_a_fn(
    arm: str,
    *,
    num_q_heads: int,
    head_dim: int,
    page_size_per_kv_packing: int,
    kv_packing: int,
    lkv_dim: int,
    num_sublanes_total: int,
    max_num_tokens: int,
    num_kv_pages_per_block: int,
    num_queries_per_block: int,
    static_q_len,
    seq_batch_size: int,
    case=MlaCase.MIXED,
    compression_ratio: int = 1,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    out_dtype=jnp.float32,
):
    mod, q_layout = ARMS[arm]
    page_size = page_size_per_kv_packing * kv_packing

    def f(q, prepared_indexer_weights, cache_kv, scores_init, seq_lens,
          page_indices, cu_q_lens, start_seq_idx, end_seq_idx):
        # ---- verbatim from run_scores_kernel ----
        bkv_p = num_kv_pages_per_block
        if static_q_len is not None:
            bq_sz = min(num_queries_per_block, static_q_len)
        else:
            bq_sz = num_queries_per_block
        bkv_sz_per_kv_packing = bkv_p * page_size_per_kv_packing
        bkv_buf_sz_per_kv_packing = bkv_sz_per_kv_packing
        bkv_sz = bkv_p * page_size
        num_sublanes_bkv = bkv_sz // 128

        grid = ((end_seq_idx - start_seq_idx) // seq_batch_size, )

        in_specs = [
            pl.BlockSpec(memory_space=pltpu.HBM),  # q
            pl.BlockSpec(memory_space=pltpu.HBM),  # prepared_indexer_weights
            pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv
            pl.BlockSpec(memory_space=pltpu.HBM),  # scores_init
        ]
        out_specs = pl.BlockSpec(memory_space=pltpu.HBM)

        bkv_double_buf = pltpu.VMEM(
            (2, seq_batch_size, bkv_buf_sz_per_kv_packing, kv_packing,
             lkv_dim), cache_kv.dtype)
        if q_layout == "head_major_hbm":
            bq_buf_shape = (2, seq_batch_size, num_q_heads, bq_sz, head_dim)
        else:
            bq_buf_shape = (2, seq_batch_size, bq_sz, num_q_heads, head_dim)
        bq_double_bufq = pltpu.VMEM(bq_buf_shape, q.dtype)
        bq_weights_double_buf = pltpu.VMEM(
            (2, seq_batch_size, bq_sz, num_q_heads),
            prepared_indexer_weights.dtype)
        bo_scores_double_buf = pltpu.VMEM(
            (2, seq_batch_size * bq_sz, num_sublanes_bkv, 128), out_dtype)

        scratch_shapes = [
            bkv_double_buf,
            bq_double_bufq,
            bq_weights_double_buf,
            bo_scores_double_buf,
            pltpu.SemaphoreType.DMA((4, 2, seq_batch_size)),
        ]

        scalar_prefetches = (
            seq_lens,
            page_indices,
            cu_q_lens,
            jnp.array([start_seq_idx, end_seq_idx], jnp.int32),
            jnp.zeros((3, ), jnp.int32),
            jnp.full((2, ), -1, jnp.int32),
        )

        kw = dict(
            compression_ratio=compression_ratio,
            static_q_len=static_q_len,
            bq_sz=bq_sz,
            bkv_p=bkv_p,
            seq_batch_size=seq_batch_size,
        )
        if q_layout is not None:
            kw["q_layout"] = q_layout
        suffix = {
            None: "",
            "head_minor": "",
            "head_major": "-hm",
            "head_major_hbm": "-hmhbm",
        }[q_layout]
        scope_name = (
            f"StreamIdxTC{suffix}-{case.symbol}-bq_{bq_sz}-bkvp_{bkv_p}")
        pallas_kernel = jax.named_scope(scope_name)(pl.pallas_call(
            functools.partial(mod.kernel, **kw),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=len(scalar_prefetches),
                in_specs=in_specs,
                out_specs=out_specs,
                grid=grid,
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary", ),
                vmem_limit_bytes=vmem_limit_bytes,
                disable_bounds_checks=True,
            ),
            out_shape=jax.ShapeDtypeStruct(
                shape=(max_num_tokens, num_sublanes_total, 128),
                dtype=out_dtype,
            ),
            input_output_aliases={len(scalar_prefetches) + 3: 0},
            name=scope_name,
        ))
        return pallas_kernel(
            *scalar_prefetches,
            q,
            prepared_indexer_weights,
            cache_kv,
            scores_init,
        )
        # ---------------------------- end verbatim ---------------------------

    return f


def prefill_stage_a(arm, g):
    """The MIXED (chunked-prefill) stage-A call at the prefill shape."""
    return make_stage_a_fn(
        arm,
        num_q_heads=g["h_idx"],
        head_dim=g["d_idx"],
        page_size_per_kv_packing=g["page_size"] // g["kv_packing"],
        kv_packing=g["kv_packing"],
        lkv_dim=g["lkv_dim"],
        num_sublanes_total=g["num_sublanes_total"],
        max_num_tokens=g["num_tokens"],
        num_kv_pages_per_block=g["bkv_p"],
        num_queries_per_block=g["bq_per_block"],
        static_q_len=None,
        seq_batch_size=1,
        case=MlaCase.MIXED,
        compression_ratio=g["compression_ratio"],
    )


def set_llo_dump_env(outdir):
    """LIBTPU_INIT_ARGS dump flags.  MUST be called before `import jax`."""
    jf = os.path.join(outdir, "jf")
    mosaic = os.path.join(outdir, "mosaic")
    os.makedirs(jf, exist_ok=True)
    os.makedirs(mosaic, exist_ok=True)
    flags = (f"--xla_jf_dump_to={jf} "
             "--xla_jf_dump_llo_text=true "
             f"--xla_mosaic_dump_to={mosaic}")
    prev = os.environ.get("LIBTPU_INIT_ARGS", "")
    os.environ["LIBTPU_INIT_ARGS"] = (prev + " " + flags).strip()
    return jf, mosaic
