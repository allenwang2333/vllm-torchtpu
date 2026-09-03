"""B x C -- Option B's chunked pipeline built on Option C's stage-A launcher.

This module is deliberately thin.  Everything that defines Option B's
experiment -- the geometry, the chunk plan, the pseudo-sequence presentation,
the two hard guards, the stage-B wrapper, the unfoldable chain carriers and
the shape of the four timing arms -- is **imported from**
`variants/option-b-pipeline/pipeline.py`, not re-typed.  The only thing that
changes is *which stage-A launcher* the arms call:

    Option B alone   probes/_stage_a.py::make_stage_a_fn
                       -> src/.../streamindex_topk.py::kernel

    B x C            variants/option-c-head-major/stage_a_c.py::make_stage_a_fn(arm)
                       -> streamindex_topk_c.py::kernel with q_layout=<arm>

`stage_a_c.make_stage_a_fn` is Option C's own transcription of the same
`run_scores_kernel` closure that `_stage_a.py` transcribes.  The two differ in
exactly two places, both of which are neutral here:

  * it launches `mod.kernel` with an extra static `q_layout=` kwarg (suppressed
    entirely for the `incumbent` arm, which launches `src`'s kernel);
  * the output shape uses an explicit `max_num_tokens=` instead of `q.shape[0]`.
    Every chunk at a given (phase, G) has the same token count, so this module
    passes `plan[0]["num_tokens"]` and the two are the same number.

For `q_layout="head_major"` the transpose happens **in registers inside the
kernel**; `prepare_q_inputs` is byte-for-byte the incumbent's (only
`head_major_hbm`, which does not compile and is not used here, transposes in
HBM).  So `pipeline.build_inputs` / `pipeline.chunk_arrays` are reused as-is
and both q layouts read the same HBM bytes.

Nothing under `src/vllm_torchtpu/` or `instruments/` is modified or copied.
"""

from __future__ import annotations

import hashlib
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent  # kernel_work/streamidx-topk
for _p in (_ROOT / "probes", _ROOT / "variants" / "option-b-pipeline",
           _ROOT / "variants" / "option-c-head-major"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pipeline as P  # noqa: E402  (variants/option-b-pipeline/pipeline.py)
import stage_a_c  # noqa: E402  (variants/option-c-head-major/stage_a_c.py)
import streamindex_topk_c as sxc  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4 import \
    streamindex_topk as sx  # noqa: E402

# Re-export the pieces that are Option B's, so the probe reads like flip3's.
geometry = P.geometry
chunk_plan = P.chunk_plan
chunk_arrays = P.chunk_arrays
build_inputs = P.build_inputs
flat_args = P.flat_args
stage_b = P.stage_b
waves_guard = P.waves_guard
p_tripwire = P.p_tripwire
PickPartitionCalled = P.PickPartitionCalled
ARMS = P.ARMS  # ("a", "b", "b_chained", "pipe")

# The q-layout arms of the 2x3 grid, plus the `incumbent` control that launches
# the unmodified src kernel through the same wrapper.
Q_ARMS = ("incumbent", "head_minor", "head_major")


def module_md5s() -> dict:
    out = dict(P.module_md5s())
    for name, path in (("streamindex_topk_c.py", sxc.__file__),
                       ("stage_a_c.py", stage_a_c.__file__),
                       ("pipeline_c.py", str(_HERE / "pipeline_c.py"))):
        out[name] = {
            "path": str(path),
            "md5": hashlib.md5(pathlib.Path(path).read_bytes()).hexdigest(),
        }
    return out


def stage_a_fn(phase: str, q_arm: str, num_tokens: int):
    """Option C's launcher, wired to Option B's geometry."""
    g = geometry(phase)
    return stage_a_c.make_stage_a_fn(
        q_arm,
        num_q_heads=g["h_idx"],
        head_dim=g["d_idx"],
        page_size_per_kv_packing=g["page_size"] // g["kv_packing"],
        kv_packing=g["kv_packing"],
        lkv_dim=g["lkv_dim"],
        num_sublanes_total=g["num_sublanes_total"],
        max_num_tokens=num_tokens,
        num_kv_pages_per_block=g["bkv_p"],
        num_queries_per_block=g["bq_per_block"],
        static_q_len=g["static_q_len"],
        seq_batch_size=g["seq_batch_size"],
        case=g["case"],
        compression_ratio=g["compression_ratio"],
    )


def build_body(phase, G, arm, R, *, q_arm, force_p1, return_outputs=False):
    """Structurally identical to `pipeline.build_body`; the one edit is that
    `sa` comes from `stage_a_fn(phase, q_arm, nt)` instead of
    `pipeline.stage_a_fn(phase)`.  Everything else -- the -inf init, the two
    carriers, the ordering of the G chunks, the `tail` bookkeeping -- is the
    same code path, so `head_minor` must reproduce Option B's own numbers.
    """
    assert arm in ARMS, arm
    assert q_arm in Q_ARMS, q_arm
    g = geometry(phase)
    k = g["k"]
    n = g["num_positions"]
    sub = g["num_sublanes_total"]
    plan = chunk_plan(phase, G)
    nts = {c["num_tokens"] for c in plan}
    assert len(nts) == 1, nts
    sa = stage_a_fn(phase, q_arm, plan[0]["num_tokens"])

    def body(chunks, cache_kv, premade):
        eps = jnp.int32(0)
        tail = []
        for r in range(R):
            acc = jnp.int32(0)
            tail = []
            for i, c in enumerate(chunks):
                (q, w, seq_lens, page_indices, cu_q_lens, eff, start, end) = c
                nt = plan[i]["num_tokens"]
                if arm in ("a", "pipe"):
                    init = jnp.full((nt, sub, 128), -jnp.inf, jnp.float32)
                    s = sa(q, w, cache_kv, init, seq_lens, page_indices,
                           cu_q_lens, start + eps, end + eps)
                    acc = acc + P._minA(s)
                    if arm == "a":
                        tail.append(s)
                        continue
                    tk = stage_b(s.reshape(nt, n), k, eff, force_p1=force_p1)
                else:
                    rl = eff + eps
                    tk = stage_b(premade[i], k, rl, force_p1=force_p1)
                    if arm == "b_chained":
                        eps = P._minB(tk)
                acc = acc + P._minB(tk)
                tail.append(tk)
            eps = acc
        if return_outputs:
            return (eps, ) + tuple(tail)
        return (eps, )

    return body


# ---------------------------------------------------------------------------
# The unchunked incumbent, called end to end -- the correctness reference.
# ---------------------------------------------------------------------------


def raw_inputs(phase: str, seed: int = 0):
    """The *unprepared* q / weights, plus the whole-op metadata, built exactly
    the way `pipeline.build_inputs` builds its prepared copies (same keys, same
    order, same construction), so the reference and the harness see the same
    numbers."""
    g = geometry(phase)
    keys = jax.random.split(jax.random.key(seed), 4)

    q = jax.random.normal(keys[0], (g["num_tokens"], g["h_idx"], g["d_idx"]),
                          jnp.float32).astype(jnp.float8_e4m3fn)
    weights = jax.random.uniform(keys[1], (g["num_tokens"], g["h_idx"]),
                                 jnp.float32, 0.25, 1.75).astype(jnp.bfloat16)

    shape = (g["total_pages"], g["page_size"] // g["kv_packing"],
             g["kv_packing"], g["lkv_dim"])
    raw = jax.random.bits(keys[2], shape, dtype=jnp.uint8)
    exp = jnp.uint8(5) + ((raw >> jnp.uint8(3)) & jnp.uint8(3))
    vals = (raw & jnp.uint8(0x87)) | (exp << jnp.uint8(3))
    lane = jnp.arange(g["lkv_dim"], dtype=jnp.int32)
    cache_kv = jnp.where(
        lane < g["d_idx"], vals,
        jnp.where(lane == g["d_idx"], jnp.uint8(127), jnp.uint8(0)))

    perm = np.asarray(
        jax.random.permutation(keys[3],
                               jnp.arange(g["total_pages"], dtype=jnp.int32)))
    return g, dict(q_raw=q, weights_raw=weights, cache_kv=cache_kv,
                   page_perm=perm)


def reference_topk(phase: str, seed: int = 0):
    """`src/.../streamindex_topk.py::streamindex_topk` on the *whole,
    unchunked* op -- one sequence of `q_len` tokens at prefill.  This is the
    incumbent as production calls it: three stage-A launches (two empty here)
    followed by one `sparsecore_topk`."""
    g, raw = raw_inputs(phase, seed)
    pps = g["pages_per_seq"]
    if phase == "prefill":
        seq_lens = jnp.array([g["context_len"]], jnp.int32)
        page_indices = jnp.asarray(raw["page_perm"][:pps], jnp.int32)
        cu_q_lens = jnp.array([0, g["q_len"]], jnp.int32)
        distribution = jnp.array([0, 0, 1], jnp.int32)
    else:
        ns = g["num_seqs"]
        seq_lens = jnp.full((ns, ), g["context_len"], jnp.int32)
        page_indices = jnp.asarray(raw["page_perm"][:ns * pps], jnp.int32)
        cu_q_lens = jnp.arange(ns + 1, dtype=jnp.int32)
        distribution = jnp.array([ns, ns, ns], jnp.int32)

    tk = sx.streamindex_topk(
        raw["q_raw"],
        raw["weights_raw"],
        raw["cache_kv"],
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
        k=g["k"],
        compression_ratio=g["compression_ratio"],
        num_kv_pages_per_block=(g["bkv_p"], ) * 3,
        num_queries_per_block=(1, g["bq_per_block"], g["bq_per_block"]),
        decode_req_batch_size=g["seq_batch_size"] if phase == "decode" else 4,
        enable_early_exit=False,
    )
    return g, np.asarray(jax.block_until_ready(tk))
