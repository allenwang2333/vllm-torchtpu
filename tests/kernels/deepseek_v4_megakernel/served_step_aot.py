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
"""The served step program, built the way warmup builds it, compiled
deviceless for tpu7x.

    python -m tests.kernels.deepseek_v4_megakernel.served_step_aot [--buckets 16,32]

DSv4-Pro (61 layers, 3 hash layers), R = 16 DP ranks. Caches come from
`runner_binding.runner_bound_model` at the served per-page dims (block
size 1024) bound as the runner binds them, plus 3 draft SWA layers; weight
operands are the avals of `weight_map.plan_operands` over the served tensor
specs (fp4 pairs as the torch_tpu bridge presents them). The program is
`step_impl.build_step` under the adapter's shard_map (every operand rank-
concatenated on axis 0), cache_inplace as served. This is the program the
first warmup step exports. One RESULT line per bucket (OK: compiled, and no
copy / transpose of >= 8 MiB in the optimized HLO, i.e. no cache copied
around the kernel and no operand relaid out per step), a VMEM headroom line
(one more ring bank still fits the largest bucket); exit 0 iff every bucket
is OK.
"""

import argparse
import dataclasses
import functools
import os
import re
import time
import types

os.environ.setdefault("TPU_SKIP_MDS_QUERY", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.experimental import topologies  # noqa: E402
from jax.sharding import NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu import envs  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import (
    dense_config as dc,  # noqa: E402
)
from vllm_torchtpu.kernels.deepseek_v4_megakernel import (
    kernel_interface as KI,  # noqa: E402
)
from vllm_torchtpu.kernels.deepseek_v4_megakernel import step_impl as SI  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import (
    vllm_adapter as VA,  # noqa: E402
)
from vllm_torchtpu.kernels.deepseek_v4_megakernel import weight_map as WM  # noqa: E402

from .runner_binding import runner_bound_model  # noqa: E402

RATIOS = tuple(dc.DSV4_PRO_RATIOS)
R = 16
HF = types.SimpleNamespace(
    num_hidden_layers=len(RATIOS), hidden_size=7168, q_lora_rank=1536, head_dim=512,
    num_attention_heads=128, o_groups=16, o_lora_rank=1024,
    moe_intermediate_size=3072, n_routed_experts=384, index_n_heads=64,
    index_head_dim=128, hc_mult=4, vocab_size=129280,
    compress_ratios=list(RATIOS) + [0], num_hash_layers=3, num_experts_per_tok=6,
    qk_rope_head_dim=64,
)  # fmt: skip
# per-page dims at block size 1024 (kv_cache_dsv4._initialize_ds_v4_kv_cache)
SERVED_CACHE = dict(nope=(256, 128), rope=(64, 128), idx=(64, 4, 256), hca=(16, 4, 128))
ROPE_LEN = 65536
BIG_COPY_MIB = 8.0


def _aval(served: WM.Served) -> jax.ShapeDtypeStruct:
    shape = tuple(ROPE_LEN if d == -1 else d for d in served.shape)
    if served.dtype == WM.FP4X2:  # the bridge: two fp4 per torch element
        return jax.ShapeDtypeStruct(shape[:-1] + (2 * shape[-1],), jnp.float4_e2m1fn)
    return jax.ShapeDtypeStruct(shape, jnp.dtype(served.dtype))


def weight_avals(
    spec: KI.ModelSpec, dense, logits: bool = False
) -> list[jax.ShapeDtypeStruct]:
    g = WM.Geometry.from_hf(HF, ratios=spec.compress_ratios)
    if spec.attn_tp > 1:  # the served tensors are vLLM TP shards
        g = WM.tp_local(g, spec.attn_tp)
    e_local = spec.n_routed_experts // spec.num_ranks
    layers = [WM.served_layer_spec(g, i, e_local) for i in range(spec.num_layers)]
    glob = WM.served_global_spec(g, spec.attn_tp > 1)
    if spec.attn_tp > 1:
        glob["embed"] = dataclasses.replace(
            glob["embed"], shape=(-(-HF.vocab_size // spec.attn_tp), HF.hidden_size)
        )
    flags = [{k: False for k in lyr} for lyr in layers]
    out = []
    manifest = functools.partial(KI.weight_manifest, logits=logits)
    shard = WM.head_shard(0, spec.num_ranks, spec.attn_tp)
    for b in WM.plan_operands(spec, flags, g, shard, dense, manifest=manifest):
        if b.entry.rule == "placement":  # the adapter's placement_tables rows
            out.append(jax.ShapeDtypeStruct(b.entry.shape, jnp.int32))
            continue
        src = [_aval(glob[x.key] if x.layer is None else layers[x.layer][x.key])
               for x in b.sources]  # fmt: skip
        out.append(src[0] if b.fn is None else jax.eval_shape(b.fn, *src))
    return out


def compile_bucket(
    spec, bound, widths, bucket, dense, max_ctx, mesh, logits=False
) -> str:
    """Lower and compile one bucket's served step; a RESULT line."""
    cfg = _step_config(spec, bound, widths)
    fn = SI.build_step(cfg, bucket, cache_inplace=True, dense=dense, max_ctx=max_ctx,
                       logits=logits)  # fmt: skip
    M, B, i32 = bucket.own_rows, bucket.reqs, jnp.int32
    head = [jax.ShapeDtypeStruct(sh, i32) for sh in ((KI.META_LEN,), (M,), (M,), (B,))]
    head += [jax.ShapeDtypeStruct((B, wd), i32) for wd in widths]
    tag = (f"T={bucket.rows} q={bucket.q} reqs={B} banks={dense.n_banks} "
           f"max_ctx={max_ctx} logits={int(logits)}")  # fmt: skip
    return _compile(fn, head + weight_avals(spec, dense, logits) + _cache_avals(cfg),
                    1 + len(spec.aux_layers), mesh, tag)  # fmt: skip


def compile_draft_bucket(
    spec, tspec, bound, widths, bucket, dense, max_ctx, mesh, logits=False
):
    """Lower and compile the DSpark draft step of one target bucket, from the
    draft layers' caches bound as the runner binds them."""
    cfg = _step_config(spec, bound, widths)
    fn = SI.build_draft_step(cfg, bucket, tspec.q, cache_inplace=True, dense=dense,
                             max_ctx=max_ctx, logits=logits)  # fmt: skip
    M, B, i32 = bucket.own_rows, bucket.reqs, jnp.int32
    Mc, n_aux = B * tspec.q, len(tspec.aux_layers)
    head = [jax.ShapeDtypeStruct(sh, i32) for sh in ((KI.META_LEN,), (M,), (M,), (B,))]
    head += [
        jax.ShapeDtypeStruct((Mc, n_aux * spec.hidden_size), jnp.bfloat16),
        jax.ShapeDtypeStruct((Mc,), i32),
        jax.ShapeDtypeStruct((B,), i32),
    ]
    head += [jax.ShapeDtypeStruct((B, wd), i32) for wd in widths]
    tag = (f"draft T={bucket.rows} q={bucket.q} ctx_q={tspec.q} reqs={B} "
           f"banks={dense.n_banks} max_ctx={max_ctx} logits={int(logits)}")  # fmt: skip
    operands = head + draft_weight_avals(spec, dense, n_aux, logits) + _cache_avals(cfg)
    return _compile(fn, operands, 1, mesh, tag)


def draft_weight_avals(spec, dense, n_aux, logits=False) -> list[jax.ShapeDtypeStruct]:
    g = WM.Geometry.from_hf(HF, ratios=tuple(RATIOS))
    e_local = spec.n_routed_experts // spec.num_ranks
    glob, layers = WM.served_draft_spec(g, spec.num_layers, n_aux, e_local)
    glob["lm_head"] = WM.served_global_spec(g)["lm_head"]  # the target's
    flags = [{k: False for k in lyr} for lyr in layers]
    gd = WM.draft_geometry(g, spec.num_layers)
    out = []
    for b in WM.plan_operands(spec, flags, gd, WM.Shard(0, spec.num_ranks), dense,
                              manifest=functools.partial(KI.draft_weight_manifest,
                                                         logits=logits)):  # fmt: skip
        src = [_aval(glob[x.key] if x.layer is None else layers[x.layer][x.key])
               for x in b.sources]  # fmt: skip
        out.append(src[0] if b.fn is None else jax.eval_shape(b.fn, *src))
    return out


def _step_config(spec, bound, widths) -> KI.StepConfig:
    return KI.StepConfig(
        model=spec, layers=bound.layers,
        cache_shapes=tuple((tuple(t.shape), str(t.dtype).removeprefix("torch."))
                           for t in bound.arrays),
        group_widths=tuple(widths), swa_page=128,
    )  # fmt: skip


def _cache_avals(cfg) -> list[jax.ShapeDtypeStruct]:
    return [jax.ShapeDtypeStruct(s, jnp.dtype(d)) for s, d in cfg.cache_shapes]


def _compile(fn, per_rank, n_out, mesh, tag) -> str:
    sh = NamedSharding(mesh, P("ep"))
    glob = [jax.ShapeDtypeStruct((R * x.shape[0],) + x.shape[1:], x.dtype, sharding=sh)
            for x in per_rank]  # fmt: skip
    specs, out_specs = (P("ep"),) * len(glob), (P("ep"),) * n_out
    f = jax.jit(jax.shard_map(lambda *xs: tuple(fn(*xs)), mesh=mesh, in_specs=specs,
                              out_specs=out_specs, check_vma=False))  # fmt: skip
    tag = f"{tag} operands={len(glob)}"
    t0 = time.time()
    try:
        compiled = f.lower(*glob).compile()
    except Exception as exc:  # noqa: BLE001 - the verdict is the output
        why = re.search(r"Ran out of memory.*?capacity by [0-9.]+[KMG]", str(exc))
        return f"FAIL {tag} {why.group(0) if why else repr(exc)[:300]}"
    big = big_copies(compiled.as_text())
    ma = compiled.memory_analysis()
    verdict = "OK" if not big else "FAIL"
    top = " ".join(f"{op}:{shape}:{mib:.0f}MiB" for op, shape, mib in big[:4])
    return (
        f"{verdict} {tag} compile_s={time.time() - t0:.0f} big_copies={len(big)} "
        f"hbm_temp_mib={ma.temp_size_in_bytes / 2**20:.0f} {top}"
    ).rstrip()


_BYTES = {"f32": 4, "s32": 4, "u32": 4, "bf16": 2, "s16": 2, "u8": 1, "s8": 1,
          "f8e4m3fn": 1, "f4e2m1fn": 0.5, "pred": 1}  # fmt: skip


def big_copies(hlo: str, min_mib: float = BIG_COPY_MIB) -> list[tuple[str, str, float]]:
    """copy / copy-start / transpose instructions of >= min_mib in the optimized
    HLO: a cache copied around the kernel, or an operand relaid out each step."""
    out = []
    for ln in hlo.splitlines():
        m = re.search(r"= (\w+)\[([0-9,]*)\]\S* (copy|copy-start|transpose)\(", ln)
        if not m or m.group(1) not in _BYTES:
            continue
        n = 1
        for d in filter(None, m.group(2).split(",")):
            n *= int(d)
        mib = n * _BYTES[m.group(1)] / 2**20
        if mib >= min_mib:
            out.append((m.group(3), f"{m.group(1)}[{m.group(2)}]", mib))
    return sorted(out, key=lambda x: -x[2])


def main() -> int:
    p = argparse.ArgumentParser()
    # defaults: the served env defaults
    p.add_argument("--buckets", default=envs.VLLM_TPU_DSV4_MK_BUCKETS)
    p.add_argument("--q", type=int, default=7)
    p.add_argument("--reqs", type=int, default=envs.VLLM_TPU_DSV4_MK_OWN_REQS)
    p.add_argument("--max-ctx", type=int, default=envs.VLLM_TPU_DSV4_MK_MAX_CTX)
    p.add_argument("--num-blocks", type=int, default=64)
    p.add_argument("--width", type=int, default=0, help="0: max-ctx / block 1024")
    p.add_argument("--topology", default="tpu7x:2x2x2")
    p.add_argument("--banks", type=int, default=0, help="0: the served count")
    p.add_argument("--no-headroom", action="store_true")
    draft_q = 6 if envs.VLLM_TPU_DSV4_MK_DRAFT else 0  # the DSpark block (dense: q-1)
    p.add_argument("--draft-q", type=int, default=draft_q, help="0: no draft step")
    p.add_argument("--logits", type=int, default=int(envs.VLLM_TPU_DSV4_MK_LOGITS),
                   help="1: the steps apply the vocab-sharded lm_head")  # fmt: skip
    p.add_argument("--attn-tp", type=int, default=1,
                   help="vLLM TP per DP engine (TP8 x DP2: 8)")  # fmt: skip
    p.add_argument("--placement", choices=("", "perm"), default="",
                   help="static expert placement tables, 24 slots")  # fmt: skip
    a = p.parse_args()
    topo = topologies.get_topology_desc(platform="tpu", topology_name=a.topology)
    assert len(topo.devices) == R, topo.devices
    mesh = jax.sharding.Mesh(topo.devices, ("ep",))
    spec = KI.ModelSpec(
        hidden_size=7168, num_layers=len(RATIOS), compress_ratios=RATIOS,
        dtype="bfloat16", q=a.q, mode="dp", axis_name="ep", num_ranks=R,
        aux_layers=(59, 60, 61), vocab_size=129280, num_hash_layers=3,
        n_routed_experts=384, attn_tp=a.attn_tp, moe_placement=a.placement,
        expert_slots=24 if a.placement else 0,
    )  # fmt: skip
    targets, drafts, groups, _ = runner_bound_model(
        RATIOS, n_draft=3, num_blocks=a.num_blocks, shapes=SERVED_CACHE
    )
    group_of = {n: gi for gi, grp in enumerate(groups) for n in grp.layer_names}
    bound = VA.bind_layer_caches(targets, group_of)
    widths = [a.width or -(-a.max_ctx // 1024)] * len(bound.group_ids)
    dense = WM.dense_config(spec)
    if a.banks:
        dense = dataclasses.replace(dense, n_banks=a.banks)
    buckets = VA.parse_row_buckets(a.buckets, a.q, a.reqs, a.max_ctx)
    ok = True
    for b in buckets:
        line = compile_bucket(spec, bound, widths, b, dense, b.max_ctx, mesh,
                              bool(a.logits))  # fmt: skip
        ok &= line.startswith("OK")
        print("RESULT", line, flush=True)
    if a.draft_q:  # the DSpark draft step of each bucket
        from vllm_torchtpu.kernels.deepseek_v4_megakernel.draft import (  # noqa: E402
            vllm_adapter as DA,
        )

        dspec = dataclasses.replace(
            spec, num_layers=3, compress_ratios=(1, 1, 1), q=a.draft_q,
            aux_layers=(), num_hash_layers=0,
        )  # fmt: skip
        dbound = VA.bind_layer_caches(drafts, group_of)
        dwidths = [widths[0]] * len(dbound.group_ids)
        ddense = WM.dense_config(dspec)
        for b in buckets:
            db = DA.draft_bucket(b, a.draft_q)
            line = compile_draft_bucket(dspec, spec, dbound, dwidths, db, ddense,
                                        b.max_ctx, mesh, bool(a.logits))  # fmt: skip
            ok &= line.startswith("OK")
            print("RESULT", line, flush=True)
    if not a.no_headroom:  # one more 5 MiB bank must still fit the largest bucket
        more = dataclasses.replace(dense, n_banks=dense.n_banks + 1)
        last = buckets[-1]
        line = compile_bucket(spec, bound, widths, last, more, last.max_ctx, mesh,
                              bool(a.logits))  # fmt: skip
        mib = dense.bank_bytes / 2**20
        cmp = ">=" if line.startswith("OK") else "<"
        print(f"RESULT headroom {cmp}{mib:.0f}MiB VMEM "
              f"({line.split()[0]} with banks={more.n_banks})", flush=True)  # fmt: skip
    print(f"RESULT served_step_aot ok={ok}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
