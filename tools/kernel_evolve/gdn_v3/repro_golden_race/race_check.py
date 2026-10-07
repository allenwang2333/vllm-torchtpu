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
"""Repeated BATCHED-decode state check of the pooled GDN v3 kernel on a TPU.

Runs the real Mosaic `fused_conv1d_gdn` (no interpret mode) of the tree given
as ROOT over the unified KV pool, at one DP rank of Qwen3.5-397B-A17B, and
compares every output and every written SSM state against a float64
reference. Unlike a pass/fail check it does not stop at the first bad state:
for every state outside tolerance it classifies the wrong elements by which
other value they hold -- the sequence's own pre-call state (`stale`), or the
pre-call / new state of the sequence `d` positions away in the batch
(`pre[+d]`, `new[+d]`) -- and reports where in the 4 MiB state the wrong
bytes sit.

    python3 race_check.py ROOT [--calls 3] [--suite full|quick] [--out x.jsonl]

Each call rebuilds the inputs and the pool from the case name's seed, so the
calls of one case are identical and their digests should agree.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import time
import types
from pathlib import Path

#: The versions pyproject.toml pins at the commit this was written against.
PINNED = {"jax": "0.11.2", "jaxlib": "0.11.2", "libtpu": "0.0.47"}

# One DP rank of Qwen3.5-397B-A17B: attention is not tensor-parallel.
N_KQ, N_V, D_K, D_V, KERNEL_SIZE = 16, 64, 128, 128, 4
CONV_DIM = 2 * N_KQ * D_K + N_V * D_V

# The served unified KV pool: fp8 pages of 128 one-KiB token rows; one GDN
# state (4 MiB fp32 SSM + bf16 conv) takes 4224 rows of one manager block.
POOL_KBS, POOL_PAYLOAD, POOL_LANES = 128, (1, 4), 256
POOL_DTYPE = "float8_e4m3fn"
MANAGER_TOKENS = 4224
ROW_BYTES = math.prod(POOL_PAYLOAD) * POOL_LANES

#: Output bound, relative to the sequence's max |reference| (bf16 output).
OUT_RTOL = 2e-2
#: A state is reported bad above this. fp32 rounding alone gives ~5e-4 for
#: decodes and ~5e-3 for prefill on device, so 2e-2 only flags gross errors.
STATE_FAIL_RTOL = 2e-2
#: An element agrees with a candidate value within this, relative to the
#: sequence's max |reference|.
MATCH_RTOL = 2e-3


def _decodes(n, ctx0=100):
    return [(1, ctx0 + i) for i in range(n)]


# name -> (seed name, params). `tile` is decode_tile_size (the kernel's
# default is 4); BATCHED runs ceil(decodes / tile) pipeline steps.
def _suite(kind):
    base = {
        # Same layouts (and seeds) as the earlier one-shot device check.
        "dec64_t64_r64": dict(tokens=64, reqs=64, decode=_decodes(64), prefill=[]),
        "dec70_t128_mixed": dict(
            tokens=128,
            reqs=128,
            decode=[(1, 3 + (i % 7)) if i % 9 else (1, 0) for i in range(70)],
            prefill=[(20, 4), (9, 0)],
        ),
        "dec37_t64": dict(
            tokens=64, reqs=64, decode=[(1, 5 + i) for i in range(36)] + [(1, 0)],
            prefill=[],
        ),  # fmt: skip
        "dec7_t16": dict(
            tokens=16, reqs=8, decode=[(1, 2 + i) for i in range(7)], prefill=[]
        ),
    }
    cases = {name: (name, dict(p, tile=4)) for name, p in base.items()}
    if kind == "quick":
        return {k: cases[k] for k in ("dec64_t64_r64", "dec7_t16")}
    if kind == "boundary":
        # Where between 10 and 16 steps the tear starts, and long pipelines
        # at tile 2 and 4. (Tile 8 at this geometry exceeds VMEM.)
        out = {}
        for n, tile, tokens in ((48, 4, 64), (56, 4, 64), (64, 4, 64),
                                (128, 2, 128), (128, 4, 128)):  # fmt: skip
            name = f"b_dec{n}_tile{tile}"
            out[name] = (name, dict(tokens=tokens, reqs=max(64, n), decode=_decodes(n, 7),
                                    prefill=[], tile=tile))  # fmt: skip
        return out
    # decode_tile_size sweep at 64 decodes: 64, 32 and 16 steps.
    for tile in (1, 2):
        cases[f"dec64_t64_r64_tile{tile}"] = (
            "dec64_t64_r64",
            dict(base["dec64_t64_r64"], tile=tile),
        )
    # Pipeline-step sweep at the default tile of 4: 2, 3, 4 and 8 steps.
    for n in (8, 12, 16, 32):
        tokens = max(16, n)
        cases[f"steps{n // 4}_dec{n}"] = (
            f"steps{n // 4}_dec{n}",
            dict(tokens=tokens, reqs=64, decode=_decodes(n), prefill=[], tile=4),
        )
    return cases


# --------------------------------------------------------------------------
# Setup: import the kernel from ROOT/src and nothing else of vllm_torchtpu.
# --------------------------------------------------------------------------


def setup(root, *, platform="tpu"):
    found = {}
    for name in PINNED:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
    os.environ["JAX_PLATFORMS"] = platform
    # Same environment as the check that first saw the failure.
    os.environ.pop("LIBTPU_INIT_ARGS", None)
    sys.dont_write_bytecode = True
    src = Path(root).resolve() / "src"
    if any(m == "vllm_torchtpu" or m.startswith("vllm_torchtpu.") for m in sys.modules):
        raise RuntimeError("vllm_torchtpu was imported before ROOT was put on the path")
    # pool_adapters reads one vLLM setting; the served layout is the default.
    vllm = types.ModuleType("vllm")
    vllm.envs = types.SimpleNamespace(VLLM_KV_CACHE_LAYOUT=None)
    sys.modules["vllm"], sys.modules["vllm.envs"] = vllm, vllm.envs
    sys.path.insert(0, str(src))

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax._src import tpu_custom_call

    tpu_custom_call.ir_version_override = lambda: tpu_custom_call._FWD_COMPAT_VERSION

    from vllm_torchtpu import gdn_pool_layout
    from vllm_torchtpu.kernels import pool_adapters
    from vllm_torchtpu.kernels.gdn.v3 import wrapper

    if not Path(wrapper.__file__).resolve().is_relative_to(src):
        raise RuntimeError(f"kernel imported from {wrapper.__file__}, not {src}")
    return types.SimpleNamespace(
        jax=jax, jnp=jnp, np=np, wrapper=wrapper, pool_adapters=pool_adapters,
        layout=gdn_pool_layout, versions=found,
    )  # fmt: skip


# --------------------------------------------------------------------------
# The pooled call, as the serving path makes it
# --------------------------------------------------------------------------


def _pool_shape(num_slots):
    return (num_slots * (MANAGER_TOKENS // POOL_KBS), POOL_KBS, *POOL_PAYLOAD, POOL_LANES)


def _plan(h, pool):
    layout = h.layout
    tok_bytes = ROW_BYTES * h.jnp.dtype(pool.dtype).itemsize
    state = layout.derive_pooled_gdn_state_layout(
        ssm_bytes=layout.pooled_gdn_ssm_state_bytes(
            num_v_heads=N_V, head_k_dim=D_K, head_v_dim=D_V
        ),
        conv_bytes=layout.pooled_gdn_conv_state_bytes(
            kernel_size=KERNEL_SIZE, conv_dim=CONV_DIM
        ),
        token_bytes=tok_bytes,
    )
    assert state.required_tokens <= MANAGER_TOKENS, state
    return h.pool_adapters.v3_state_source(
        pool,
        split=MANAGER_TOKENS // POOL_KBS,
        ssm_ntok=state.ssm_tokens,
        conv_tok0=state.ssm_tokens,
        conv_ntok=state.conv_tokens,
        conv_dim=CONV_DIM,
        n_v=N_V,
        d_k=D_K,
        d_v=D_V,
        kernel_size=KERNEL_SIZE,
    )


def _pooled_call(h, tile):
    def call(qkv, b, a, pool, cw, cb, a_log, dt_bias, qsl, idx, dist, seq_lens):
        return h.wrapper.fused_conv1d_gdn(
            qkv, b, a, None, None, cw, cb, a_log, dt_bias, qsl, idx, dist, seq_lens,
            n_kq=N_KQ, n_v=N_V, d_k=D_K, d_v=D_V, kernel_size=KERNEL_SIZE,
            state_source=pool, state_plan=_plan(h, pool), decode_tile_size=tile,
        )  # fmt: skip

    return call


# --------------------------------------------------------------------------
# Pool bytes, packed independently of the kernel
# --------------------------------------------------------------------------


def _np_view(np, raw, view_dtype):
    """uint8 rows (..., S, L) -> view_dtype rows, packed along S as on TPU."""
    r = np.dtype(view_dtype).itemsize
    *lead, s, lanes = raw.shape
    u = np.ascontiguousarray(np.swapaxes(raw.reshape(*lead, s // r, r, lanes), -1, -2))
    return u.view(view_dtype).reshape(*lead, s // r, lanes)


def _np_unview(np, view):
    r = view.dtype.itemsize
    *lead, s, lanes = view.shape
    u = np.ascontiguousarray(view).view(np.uint8).reshape(*lead, s, lanes, r)
    return np.swapaxes(u, -1, -2).reshape(*lead, s * r, lanes)


def _region_blocks(region, stride, block):
    first = block * stride + region.kb0
    return slice(first, first + region.nblocks), slice(
        region.row0, region.row0 + region.nrows
    )


def _np_read(np, pool_u8, region, stride, block, shape, view_dtype):
    blocks, rows = _region_blocks(region, stride, block)
    view = _np_view(np, pool_u8[blocks, rows], view_dtype)
    flat = view.reshape(-1, view.shape[-1] // region.lane_split)
    return flat[: region.rows_used].reshape(shape)


def _np_write(np, pool_u8, region, stride, block, values, view_dtype):
    blocks, rows = _region_blocks(region, stride, block)
    view = _np_view(np, pool_u8[blocks, rows], view_dtype).copy()
    flat = view.reshape(-1, view.shape[-1] // region.lane_split)
    flat[: region.rows_used] = values.astype(view_dtype).reshape(region.rows_used, -1)
    pool_u8[blocks, rows] = _np_unview(np, flat.reshape(view.shape))


# --------------------------------------------------------------------------
# Reference
# --------------------------------------------------------------------------


def _reference_seq(np, qkv, b, a, conv0, rec0, w):
    """float64 outputs and final (conv, SSM) state of one sequence."""
    q_len = qkv.shape[0]
    x = np.concatenate([conv0, qkv], axis=0).astype(np.float64)
    conv_w = w["conv_weight"][:, 0, :].astype(np.float64)
    acc = sum(x[k : k + q_len] * conv_w[:, k] for k in range(KERNEL_SIZE))
    acc = acc + w["conv_bias"].astype(np.float64)
    act = acc / (1.0 + np.exp(-acc))
    key = N_KQ * D_K
    rep = N_V // N_KQ
    q = np.repeat(act[:, :key].reshape(q_len, N_KQ, D_K), rep, axis=1)
    k = np.repeat(act[:, key : 2 * key].reshape(q_len, N_KQ, D_K), rep, axis=1)
    v = act[:, 2 * key :].reshape(q_len, N_V, D_V)
    q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6) * D_K**-0.5
    k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
    beta = 1.0 / (1.0 + np.exp(-b.astype(np.float64)))
    g = -np.exp(w["a_log"].astype(np.float64)) * np.logaddexp(
        0.0, a.astype(np.float64) + w["dt_bias"].astype(np.float64)
    )
    state = rec0.astype(np.float64)
    outs = []
    for t in range(q_len):
        state = state * np.exp(g[t])[:, None, None]
        v_new = beta[t][:, None] * (v[t] - np.einsum("hk,hkv->hv", k[t], state))
        state = state + k[t][:, :, None] * v_new[:, None, :]
        outs.append(np.einsum("hk,hkv->hv", q[t], state))
    return np.stack(outs), state, x[q_len : q_len + KERNEL_SIZE - 1]


# --------------------------------------------------------------------------
# Element-level classification of one bad state
# --------------------------------------------------------------------------


def _row_map(np, label_of, max_runs=24):
    """Run-length map of the state by 1 KiB pool row: [label, start KiB, end KiB].

    A row takes the most common label of its wrong elements, `ok` if none.
    """
    per_row = ROW_BYTES // 4
    names = list(label_of)
    ids = np.zeros(next(iter(label_of.values())).size, np.int32)
    for k, name in enumerate(names, start=1):
        ids[label_of[name]] = k
    rows = ids.reshape(-1, per_row)
    counts = np.stack([(rows == k).sum(1) for k in range(1, len(names) + 1)], 1)
    row_id = np.where(counts.max(1) > 0, counts.argmax(1) + 1, 0)
    edges = np.flatnonzero(np.diff(row_id)) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [row_id.size]])
    runs = [
        ["ok" if row_id[a] == 0 else names[row_id[a] - 1], int(a), int(b)]
        for a, b in zip(starts, ends)
    ]
    return runs[:max_runs], len(runs)


def classify(np, i, got, new, pre, has_pre, untouched, rng, sample=20000):
    """Which value each wrong element of sequence i's state holds.

    `new[j]` / `pre[j]` are sequence j's reference post-call state and the
    state it read (`has_pre[j]` False for a fresh sequence, which reads none).
    `untouched` is what sequence i's block held before the call. A `pre` or
    `untouched` match is bit-exact: those bytes would be a copy of the pool.
    """
    got = got.reshape(-1).astype(np.float64)
    scale = max(float(np.abs(new[i]).max()), 1e-6)
    tol = MATCH_RTOL * scale
    wrong = np.abs(got - new[i].reshape(-1)) > tol
    n_wrong = int(wrong.sum())
    result = dict(n_elem=got.size, n_wrong=n_wrong, frac_wrong=round(n_wrong / got.size, 4))
    if not n_wrong:
        return result
    w_idx = np.flatnonzero(wrong)
    probe = w_idx if w_idx.size <= sample else rng.choice(w_idx, sample, replace=False)
    cands = {} if has_pre[i] else {"untouched": (untouched.reshape(-1), 0.0)}
    for j in range(len(new)):
        label = "stale" if j == i else f"pre[{j - i:+d}]"
        if has_pre[j]:
            cands[label] = (pre[j].reshape(-1), 0.0)
        if j != i:
            cands[f"new[{j - i:+d}]"] = (new[j].reshape(-1), tol)
    hits = {
        label: int((np.abs(got[probe] - ref[probe]) <= t).sum())
        for label, (ref, t) in cands.items()
    }
    top = [k for k, v in sorted(hits.items(), key=lambda kv: -kv[1]) if v][:3]
    explained = np.zeros_like(wrong)
    masks = {}
    for label in top:
        ref, t = cands[label]
        m = wrong & ~explained & (np.abs(got - ref) <= t)
        if m.any():
            explained |= m
            masks[label] = m
    masks["other"] = wrong & ~explained
    row_map, n_runs = _row_map(np, masks)
    result.update(
        matches={k: int(m.sum()) for k, m in masks.items()},
        row_map_kib=row_map,
        n_row_runs=n_runs,
    )
    return result


# --------------------------------------------------------------------------
# One case, one call
# --------------------------------------------------------------------------


@dataclasses.dataclass
class _Seq:
    q_len: int
    ctx: int
    block: int


def _weights(np, rng, jnp):
    bf16 = jnp.bfloat16
    return dict(
        conv_weight=np.asarray(rng.normal(0, 0.3, (CONV_DIM, 1, KERNEL_SIZE)), bf16),
        conv_bias=np.asarray(rng.normal(0, 0.1, (CONV_DIM,)), bf16),
        a_log=rng.normal(0, 0.5, (N_V,)).astype(np.float32),
        dt_bias=rng.normal(0, 0.5, (N_V,)).astype(np.float32),
    )


def run_call(h, seed_name, params, compiled):
    """Build the case from its seed, run the kernel once, check everything."""
    np, jnp, jax = h.np, h.jnp, h.jax
    rng = np.random.default_rng(int(hashlib.sha256(seed_name.encode()).hexdigest()[:8], 16))
    w = _weights(np, rng, jnp)
    layout = params["decode"] + params["prefill"]
    tokens, reqs, tile = params["tokens"], params["reqs"], params["tile"]
    num_decode = len(params["decode"])
    nslots = 1 + len(layout)
    blocks = [int(x) for x in rng.permutation(np.arange(1, nslots))]
    pool_shape = _pool_shape(nslots)
    pool_u8 = rng.integers(0, 256, pool_shape, dtype=np.uint8)
    plan = _plan(h, jax.ShapeDtypeStruct(pool_shape, jnp.dtype(POOL_DTYPE)))
    stride = plan.stride
    conv_shape, rec_shape = (KERNEL_SIZE - 1, CONV_DIM), (N_V, D_K, D_V)

    def read(block, pool):
        conv = _np_read(np, pool, plan.conv, stride, block, conv_shape, jnp.bfloat16)
        rec = _np_read(np, pool, plan.recurrent, stride, block, rec_shape, np.float32)
        return conv, rec

    seqs = [_Seq(q, ctx, blocks[i]) for i, (q, ctx) in enumerate(layout)]
    for s in seqs:
        if s.ctx:
            _np_write(np, pool_u8, plan.conv, stride, s.block,
                      np.asarray(rng.normal(0, 1, conv_shape), jnp.bfloat16), jnp.bfloat16)  # fmt: skip
            _np_write(np, pool_u8, plan.recurrent, stride, s.block,
                      rng.normal(0, 0.1, rec_shape).astype(np.float32), np.float32)  # fmt: skip

    total = sum(s.q_len for s in seqs)
    assert total <= tokens and len(seqs) <= reqs
    qsl = np.full(reqs + 1, total, np.int32)
    qsl[: len(seqs) + 1] = np.concatenate([[0], np.cumsum([s.q_len for s in seqs])])
    idx = np.zeros(reqs, np.int32)
    seq_lens = np.zeros(reqs, np.int32)
    for i, s in enumerate(seqs):
        idx[i], seq_lens[i] = s.block, s.q_len + s.ctx
    bf16 = jnp.bfloat16
    acts = {k: np.asarray(rng.normal(0, 1, (tokens, d)), bf16)
            for k, d in (("qkv", CONV_DIM), ("b", N_V), ("a", N_V))}  # fmt: skip
    dist = np.array([num_decode, num_decode, len(seqs)], np.int32)

    before = pool_u8.copy()
    want = []
    for i, s in enumerate(seqs):
        lo, hi = qsl[i], qsl[i + 1]
        conv0, rec0 = read(s.block, before) if s.ctx else (
            np.zeros(conv_shape, np.float32), np.zeros(rec_shape, np.float32))  # fmt: skip
        out, rec, conv = _reference_seq(
            np, np.asarray(acts["qkv"][lo:hi], np.float32), acts["b"][lo:hi],
            acts["a"][lo:hi], np.asarray(conv0, np.float32), rec0, w,
        )  # fmt: skip
        want.append((lo, hi, out, rec, conv, rec0))

    key = (tokens, reqs, nslots, tile)
    if key not in compiled:
        compiled[key] = jax.jit(_pooled_call(h, tile), donate_argnums=(3,))
    args = [
        jnp.asarray(acts["qkv"]), jnp.asarray(acts["b"]), jnp.asarray(acts["a"]),
        jnp.asarray(pool_u8).view(jnp.dtype(POOL_DTYPE)),
        jnp.asarray(w["conv_weight"]), jnp.asarray(w["conv_bias"]),
        jnp.asarray(w["a_log"]), jnp.asarray(w["dt_bias"]),
        jnp.asarray(qsl), jnp.asarray(idx), jnp.asarray(dist), jnp.asarray(seq_lens),
    ]  # fmt: skip
    new_pool, out = jax.block_until_ready(compiled[key](*args))
    after = np.asarray(new_pool.view(jnp.uint8))
    out = np.asarray(out.astype(jnp.float32)).reshape(tokens, N_V, D_V)

    # Bytes the call must not touch: everything but the sequences' own regions.
    allowed = np.zeros(pool_shape[:2], bool)
    for s in seqs:
        for region in (plan.conv, plan.recurrent):
            bl, rows = _region_blocks(region, stride, s.block)
            assert rows == slice(0, POOL_KBS), "partial-block regions"
            allowed[bl] = True
    changed = np.any((after != before).reshape(*pool_shape[:2], -1), axis=-1)
    stray = sorted({int(x) // stride for x in np.nonzero(changed & ~allowed)[0]})

    new_states = [x[3] for x in want]
    pre_states = [x[5] for x in want]
    has_pre = [bool(s.ctx) for s in seqs]
    cls_rng = np.random.default_rng(0)
    seq_rows, bad = [], []
    for i, (s, (lo, hi, ref_out, ref_rec, ref_conv, _)) in enumerate(zip(seqs, want)):
        out_err = float(np.abs(out[lo:hi] - ref_out).max()) / max(float(np.abs(ref_out).max()), 1e-6)
        conv, rec = read(s.block, after)
        conv_ok = bool(np.array_equal(np.asarray(conv, np.float32),
                                      np.asarray(ref_conv.astype(bf16), np.float32)))  # fmt: skip
        rec64 = np.asarray(rec, np.float64)
        finite = bool(np.all(np.isfinite(rec64)))
        state_err = float(np.abs(rec64 - ref_rec).max()) / max(float(np.abs(ref_rec).max()), 1e-6) if finite else float("inf")
        seq_rows.append((out_err, state_err))
        if state_err > STATE_FAIL_RTOL or not conv_ok or out_err > OUT_RTOL:
            entry = dict(
                seq=i, tile_pos=(i // tile, i % tile) if i < num_decode else None,
                decode=i < num_decode, fresh=not s.ctx, ckpt=0,
                state_err=round(state_err, 4), out_err=round(out_err, 5), conv_exact=conv_ok,
            )  # fmt: skip
            if state_err > STATE_FAIL_RTOL and finite:
                entry["cls"] = classify(np, i, rec64, new_states, pre_states, has_pre,
                                        read(s.block, before)[1], cls_rng)  # fmt: skip
            bad.append(entry)
    digest = hashlib.sha256(after.tobytes() + out.tobytes()).hexdigest()[:12]
    return dict(
        worst_out_err=round(max(r[0] for r in seq_rows), 6),
        worst_state_err=round(max(r[1] for r in seq_rows), 6),
        digest=digest,
        n_bad=len(bad),
        stray_blocks=stray,
        bad=bad,
    )


def summarize(bad):
    """Dominant classification over a call's bad states, e.g. {'pre[+8]': 12}."""
    counts = collections.Counter()
    for b in bad:
        m = b.get("cls", {}).get("matches", {})
        if m:
            counts[max(m, key=m.get)] += 1
        else:
            counts["conv/output only" if "cls" not in b else "none"] += 1
    return dict(counts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", help="checkout whose src/vllm_torchtpu is tested")
    ap.add_argument("--calls", type=int, default=3)
    ap.add_argument("--suite", choices=("full", "quick", "boundary"), default="full")
    ap.add_argument("--cases", default="", help="comma-separated subset of the suite")
    ap.add_argument("--label", default="", help="arm name written into every record")
    ap.add_argument("--out", type=Path, help="append one JSON record per call here")
    args = ap.parse_args(argv)

    h = setup(args.root)
    print("versions", json.dumps(h.versions), flush=True)
    if h.versions != PINNED:
        print(f"WARNING: versions differ from the pins {PINNED}", flush=True)
    print("devices", h.jax.devices(), "kernel", h.wrapper.__file__, flush=True)
    cases = _suite(args.suite)
    if args.cases:
        cases = {k: cases[k] for k in args.cases.split(",")}
    compiled = {}
    for name, (seed_name, params) in cases.items():
        steps = -(-len(params["decode"]) // min(params["tile"], params["tokens"]))
        for call in range(args.calls):
            t0 = time.time()
            rec = dict(arm=args.label, case=name, tile=params["tile"],
                       decodes=len(params["decode"]), batched_steps=steps, call=call)  # fmt: skip
            try:
                rec.update(run_call(h, seed_name, params, compiled))
                rec["verdict"] = "FAIL" if rec["n_bad"] or rec["stray_blocks"] else "PASS"
            except Exception as exc:  # noqa: BLE001 - one broken case must not hide the rest
                rec.update(verdict="ERROR", error=f"{type(exc).__name__}: {str(exc)[:400]}")
            rec["seconds"] = round(time.time() - t0, 1)
            short = {k: rec.get(k) for k in ("arm", "case", "tile", "batched_steps", "call",
                     "verdict", "worst_state_err", "worst_out_err", "digest", "n_bad", "error")}  # fmt: skip
            short["bad_seqs"] = [b["seq"] for b in rec.get("bad", [])]
            short["classes"] = summarize(rec.get("bad", []))
            print("CALL", json.dumps(short), flush=True)
            if args.out:
                with args.out.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
