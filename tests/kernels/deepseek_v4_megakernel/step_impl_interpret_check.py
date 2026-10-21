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
"""CPU-interpret check of `step_impl.build_step` (the adapter's contract around
the whole-step kernel) on R simulated ranks.

    JAX_PLATFORMS=cpu python -m \\
        tests.kernels.deepseek_v4_megakernel.step_impl_interpret_check --R 4
        ... --R 4 --attn-tp 2   # vLLM TP x DP: groups of 2, vocab-shard embed

Reduced widths (dense_config.small_config), 3 layers (HCA, HCA, CSA), stub
phases. The contract fn gets its operands exactly as the adapter passes them
(meta, own ids / positions / seq_lens, block tables, weights in manifest
order, caches) and is checked against `step_kernel.build` called directly plus
an independent numpy reading of the stock tail (`mhc_collapse_head`, vLLM
RMSNorm, aux mean over the hc streams). Prints one RESULT JSON line.
"""

import argparse
import dataclasses
import json
import os

import jax

R_DEFAULT = int(os.environ.get("STEP_R", "4"))
jax.config.update("jax_num_cpu_devices", R_DEFAULT)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4_megakernel import dense_config as dc  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import dense_weights as dw  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import kernel_interface as KI  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import step_impl as SI  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import step_kernel as SK  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import step_reference as SR  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4_megakernel.step_phases import (  # noqa: E402
    pack_cos_sin,
)

from .test_step_impl import np_tail  # noqa: E402


def _rel(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--R", type=int, default=R_DEFAULT)
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--q", type=int, default=7)
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--attn-tp", type=int, default=1)
    a = p.parse_args()
    R, T, q, B, A = a.R, a.T, a.q, a.B, a.attn_tp
    G = R // A
    M = B * q
    ratios = (128, 128, 4)
    full = dc.small_config(
        n_heads=8 * R, tp=1, rows=T, moe_inter=128 * R, ratios=ratios
    )
    tp_cfg = dataclasses.replace(full, tp=R)
    aux_layers = (2, 3)  # 1-based ids as DeepseekV4Model tags them
    spec = KI.ModelSpec(
        hidden_size=full.hidden, num_layers=3, compress_ratios=ratios,
        dtype="bfloat16", q=q, mode="dp", axis_name="d", num_ranks=R,
        aux_layers=aux_layers, vocab_size=512, num_hash_layers=1,
        n_routed_experts=full.n_experts, attn_tp=A,
    )  # fmt: skip
    # three KV groups, no layout detail needed by the stub attention
    lc = KI.LayerCaches
    cr = KI.CacheRef
    layers = (
        lc("hca", 128, swa=cr(0, 0), kv=cr(1, 1), state=cr(0, 2)),
        lc("hca", 128, swa=cr(2, 0), kv=cr(3, 1), state=cr(2, 2)),
        lc("csa", 4, swa=cr(0, 0), kv=cr(4, 1), state=cr(4, 2), kv_rope=cr(5, 1),
           indexer_k=cr(6, 1), indexer_state=cr(6, 2)),
    )  # fmt: skip
    widths = (3, 5, 4)
    n_arr = 7
    cfg = KI.StepConfig(
        model=spec, layers=layers,
        cache_shapes=tuple(((4, 8, 128), "int32") for _ in range(n_arr)),
        group_widths=widths, swa_page=128,
    )  # fmt: skip
    bucket = KI.Bucket(q=q, rows=T, reqs=B)
    fn = SI.build_step(cfg, bucket, attn_impl="stub", moe_impl="stub",
                       interpret=True, dense=tp_cfg)  # fmt: skip

    # weights: the step's real inputs; everything a stub phase never reads
    # is a small placeholder under its manifest name
    w = dw.make_weights(full)
    packs = [
        dw.pack_weights(SR.rank_weights(w, full, SK.head_block(r, R, A), R), tp_cfg)
        for r in range(R)
    ]
    V = 512
    VS = V // A  # rank r's embed: vocab shard r % A (attn_tp > 1)
    key = jax.random.split(jax.random.key(7), 5)
    embed = jax.random.normal(key[0], (V, full.hidden), jnp.float32).astype(
        jnp.bfloat16
    )
    ang = jnp.arange(4096)[:, None] * 10000.0 ** (-jnp.arange(32) / 32.0)[None]
    cos_sin = jnp.concatenate([jnp.cos(ang), jnp.sin(ang)], axis=1).astype(jnp.float32)
    cos_sin = pack_cos_sin(cos_sin)  # as the served manifest operand
    hc_fn = jax.random.normal(key[1], (4, 4 * full.hidden), jnp.float32) * 0.02
    hc_base = jax.random.normal(key[2], (4,), jnp.float32)
    hc_scale = jnp.asarray([0.7], jnp.float32)
    norm_w = (1 + 0.1 * jax.random.normal(key[3], (full.hidden,))).astype(jnp.bfloat16)
    glob = dict(embed=embed, final_norm=norm_w, hc_head_fn=hc_fn,
                hc_head_base=hc_base, hc_head_scale=hc_scale, cos_sin=cos_sin,
                comp_cos_sin=cos_sin, idx_cos_sin=cos_sin)  # fmt: skip
    names = [e.name for e in KI.weight_manifest(spec)]
    per_rank_w = []  # [R][name]
    for r in range(R):
        d = {}
        for n in names:
            if n == "embed" and A > 1:
                d[n] = embed[(r % A) * VS : (r % A + 1) * VS]
            elif n in glob:
                d[n] = glob[n]
            elif n == "dense_smalls":
                d[n] = packs[r]["smalls"]
            elif n.startswith("dense_"):
                d[n] = packs[r]["tiles"][n.removeprefix("dense_")]
            else:
                d[n] = jnp.zeros((8, 128), jnp.float32)
        per_rank_w.append(d)

    # group 0 and group G-1 one request each (the same batch on every rank of
    # a group); the rest idle
    rng = np.random.default_rng(0)
    ids = np.zeros((R, M), np.int32)
    pos = np.zeros((R, M), np.int32)
    seq = np.zeros((R, B), np.int32)
    for grp in (0, G - 1):
        p0 = int(rng.integers(100, 3000))
        rr = slice(grp * A, (grp + 1) * A)
        ids[rr, :q] = rng.integers(0, V, q)
        pos[rr, :q] = np.arange(p0, p0 + q)
        seq[rr, 0] = p0 + q
    bts = [
        np.repeat(rng.integers(1, 50, (G, B, wd)).astype(np.int32), A, axis=0)
        for wd in widths
    ]
    caches = [
        rng.integers(0, 1000, (R, 4, 8, 128)).astype(np.int32) for _ in range(n_arr)
    ]
    meta = np.zeros((R, KI.META_LEN), np.int32)
    meta[:, KI.META_RANK] = np.arange(R)
    meta[:, KI.META_EBLK] = np.arange(R)
    meta[:, KI.META_Q] = q
    meta[:, KI.META_VSTART] = (np.arange(R) % A) * VS

    def stack(x):
        return jnp.concatenate([jnp.asarray(v)[None] for v in x])

    operands = [stack(meta), jnp.asarray(ids), jnp.asarray(pos), jnp.asarray(seq)]
    operands += [jnp.asarray(b) for b in bts]
    operands += [stack([per_rank_w[r][n] for r in range(R)]) for n in names]
    operands += [jnp.asarray(c) for c in caches]
    n_out = 1 + len(aux_layers) + n_arr

    def per_rank(*xs):
        return tuple(o[None] for o in fn(*[x[0] for x in xs]))

    mesh = jax.make_mesh((R,), ("d",))
    specs = (P("d"),) * len(operands)
    out_specs = (P("d"),) * n_out
    f = jax.jit(jax.shard_map(per_rank, mesh=mesh, in_specs=specs,
                              out_specs=out_specs, check_vma=False))  # fmt: skip
    shard = jax.sharding.NamedSharding(mesh, P("d"))
    outs = jax.block_until_ready(f(*[jax.device_put(x, shard) for x in operands]))
    hidden, auxs, cache_out = (
        outs[0],
        outs[1 : 1 + len(aux_layers)],
        outs[1 + len(aux_layers) :],
    )

    # the same step called directly, then the tail in numpy
    scfg = SI.kernel_config(cfg, bucket, tp_cfg)
    step = SK.build(scfg, axis="d", interpret=True)
    tiles_by_rank = [p_["tiles"] for p_ in packs]

    def direct(ids_, pos_, seq_, smalls, *tl):
        tiles = dict(zip(tp_cfg.stream_names, [t[0] for t in tl]))
        res, aux, *_ = step(ids_[0], pos_[0], seq_[0], embed, cos_sin, smalls[0],
                            tiles, None)  # fmt: skip
        return res[None], aux[None]

    dspecs = (P("d"),) * (4 + len(tp_cfg.stream_names))
    g = jax.jit(jax.shard_map(direct, mesh=mesh, in_specs=dspecs,
                              out_specs=(P("d"), P("d")), check_vma=False))  # fmt: skip
    dargs = [
        jnp.asarray(ids),
        jnp.asarray(pos),
        jnp.asarray(seq),
        stack([p_["smalls"] for p_ in packs]),
    ]
    dargs += [stack([t[n] for t in tiles_by_rank]) for n in tp_cfg.stream_names]
    res, aux = jax.block_until_ready(g(*[jax.device_put(x, shard) for x in dargs]))
    res, aux = np.asarray(res.astype(jnp.float32)), np.asarray(aux.astype(jnp.float32))
    hidden = np.asarray(hidden.astype(jnp.float32))
    auxs = [np.asarray(x.astype(jnp.float32)) for x in auxs]
    cache_out = [np.asarray(c) for c in cache_out]
    worst_h = worst_a = 0.0
    for r in range(R):
        h_ref, a_ref = np_tail(res[r], aux[r], hc_fn, hc_scale, hc_base, norm_w)
        worst_h = max(worst_h, _rel(hidden[r], h_ref))
        worst_a = max(
            worst_a, max(_rel(auxs[i][r], a_ref[i]) for i in range(len(aux_layers)))
        )
    out = {
        "hidden_shape": list(hidden.shape[1:]),
        "n_aux": len(auxs),
        "hidden_rel": worst_h,
        "aux_rel": worst_a,
        "attn_tp": A,
        "live_rows_nonzero": all(np.abs(hidden[r][:q]).sum() > 0 for r in (0, R - 1)),
        "idle_rank_zero": bool(np.all(hidden[A] == 0)) if G > 2 else True,
        # every rank of a group returns the group's rows
        "group_rows_equal": all(
            np.array_equal(hidden[r], hidden[r - r % A]) for r in range(R)
        ),
        "caches_unchanged": all(
            np.array_equal(c, o) for c, o in zip(caches, cache_out)
        ),
    }
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
