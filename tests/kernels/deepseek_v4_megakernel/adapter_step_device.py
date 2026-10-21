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
"""The serving adapter driving the real whole-step kernel on 8 TPU ranks.

    python -m tests.kernels.deepseek_v4_megakernel.adapter_step_device [--fault]

R = 8 DP ranks on one host (8 processes, torch.distributed.run, tpu_dist),
reduced widths (dense_config.small_config, 64 heads = 8 per rank, 3 layers
HCA/HCA/CSA), stub attention and MoE phases. Ranks 0 and 7 each serve one
7-row verify window, the rest are idle. Every rank runs the adapter path end
to end -- DP decision from the reduced entries (T = 16 from 7 + 7 rows),
per-rank prep of runner-shaped metadata, the torch_tpu export of the
shard_map'd step (sharded_jax_callable, list[Tensor] custom op),
torch.compile(backend="tpu"), cache donation, output fit. Then one plain JAX
process runs the same `step_impl` contract fn on the same 8 devices with the
operands the prep should have produced: the adapter path must match it bit
for bit (that isolates the adapter from the kernel). Both are also compared
with `step_reference` (plain JAX, kernel and stock summation orders) + the
stock tail, reported, not gated: that is the kernel's numerics, not the
adapter's. --fault changes one input token of the twin (positive control:
the bit-exact comparison must fail).
"""

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

WORLD = 8
Q, B, T = 7, 2, 16
M = B * Q
RATIOS = (128, 128, 4)
BUSY = (0, WORLD - 1)
SEQ0 = 100  # fake_metadata: request 0's seq_len
# --logits: the steps apply the vocab-sharded lm_head (VLLM_TPU_DSV4_MK_LOGITS)
LOGITS = os.environ.get("DSV4_ADAPTER_LOGITS") == "1"


def _inputs(rank):
    """The runner's padded ids / positions on `rank` (zeros when idle)."""
    if rank not in BUSY:
        return np.zeros(T, np.int32), np.zeros(T, np.int32)
    ids = (np.arange(T, dtype=np.int32) * 13 + 31 * rank) % 512
    return ids, np.arange(T, dtype=np.int32) + 900 + rank


def _setup(jax, jnp):
    """Model, per-rank manifest-ordered weights (JAX, host), globals."""
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import dense_config as dc
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import dense_weights as dw
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import kernel_interface as KI
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import step_phases as SP
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import step_reference as SR
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import vocab_head as VH

    full = dc.small_config(n_heads=8 * WORLD, tp=1, rows=T, moe_inter=128 * WORLD,
                           ratios=RATIOS)  # fmt: skip
    tp_cfg = dataclasses.replace(full, tp=WORLD)
    spec = KI.ModelSpec(
        hidden_size=full.hidden, num_layers=3, compress_ratios=RATIOS,
        dtype="bfloat16", q=Q, mode="dp", axis_name="ep", num_ranks=WORLD,
        aux_layers=(2, 3), vocab_size=512, num_hash_layers=1,
        n_routed_experts=full.n_experts,
    )  # fmt: skip
    with jax.default_device(jax.devices("cpu")[0]):
        w = dw.make_weights(full)
        key = jax.random.split(jax.random.key(3), 4)
        embed = jax.random.normal(key[0], (512, full.hidden)).astype(jnp.bfloat16)
        norm = 1 + 0.1 * jax.random.normal(key[1], (full.hidden,))
        glob = dict(
            embed=embed,
            final_norm=norm.astype(jnp.bfloat16),
            hc_head_fn=jax.random.normal(key[2], (4, 4 * full.hidden)) * 0.02,
            hc_head_base=jax.random.normal(key[3], (4,)),
            hc_head_scale=jnp.asarray([0.7], jnp.float32),
        )
        lm_head = (0.05 * jax.random.normal(key[1], (512, full.hidden))).astype(
            jnp.bfloat16
        )
        ang = jnp.arange(4096)[:, None] * 10000.0 ** (-jnp.arange(32) / 32.0)[None]
        cs = jnp.concatenate([jnp.cos(ang), jnp.sin(ang)], 1).astype(jnp.float32)
        packed = SP.pack_cos_sin(cs)  # the served manifest operands
        glob.update(cos_sin=packed, comp_cos_sin=packed, idx_cos_sin=packed)

        def weights(rank):
            pack = dw.pack_weights(SR.rank_weights(w, full, rank, WORLD), tp_cfg)
            out = []
            for n in [e.name for e in KI.weight_manifest(spec, logits=LOGITS)]:
                if n == "lm_head":
                    out.append(VH.shard_lm_head(lm_head, rank, WORLD))
                elif n in glob:
                    out.append(glob[n])
                elif n == "dense_smalls":
                    out.append(pack["smalls"])
                elif n.startswith("dense_"):
                    out.append(pack["tiles"][n.removeprefix("dense_")])
                else:  # read only by the real phases
                    out.append(jnp.zeros((8, 128), jnp.float32))
            return [jax.device_get(a) for a in out]

    return dict(full=full, tp_cfg=tp_cfg, spec=spec, w=w, glob=glob, weights=weights,
                cos_sin=cs, lm_head=lm_head)  # fmt: skip


def _fake_tree(device):
    from tests.kernels.deepseek_v4_megakernel.test_vllm_adapter import make_fake_layers
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import vllm_adapter as VA

    layers, groups, _ = make_fake_layers(device)
    bound = VA.bind_layer_caches(layers, groups)
    widths = [3 if g % 2 else 5 for g in bound.group_ids]
    return layers, bound, widths


def _worker(result_dir: Path) -> None:
    import jax
    import jax.numpy as jnp
    import torch
    import torch_tpu  # noqa: F401
    from torch import distributed as dist

    rank = int(os.environ["RANK"])
    out: dict = {"rank": rank}
    try:
        dist.init_process_group(backend="tpu_dist")
        torch.empty((1,), device="tpu").cpu()
        from torch_tpu._internal.pallas import get_pallas_mesh

        from tests.kernels.deepseek_v4_megakernel.test_vllm_adapter import fake_metadata
        from tests.kernels.deepseek_v4_megakernel.test_weight_map import j2t
        from vllm_torchtpu.kernels.deepseek_v4_megakernel import kernel_interface as KI
        from vllm_torchtpu.kernels.deepseek_v4_megakernel import vllm_adapter as VA

        env = _setup(jax, jnp)
        layers, bound, widths = _fake_tree("tpu")
        mk = VA.DSv4Megakernel(
            spec=env["spec"],
            buckets=VA.parse_row_buckets(str(T), Q, B),
            kernel=VA.ServingKernel(
                rank,
                logits=LOGITS,
                attn_impl="stub",
                moe_impl="stub",
                dense=env["tp_cfg"],
            ),  # fmt: skip
            mesh=get_pallas_mesh(axis_names=("ep",)),
            rank=rank,
            expert_block=rank,
            weights=[j2t(a).to("tpu") for a in env["weights"](rank)],
            decoder_layers=layers,
            device="tpu",
            op_namespace="dsv4_mk_adapter_dev",
        )
        mk.bind(bound, widths, swa_page=128)
        before = [t.cpu() for t in mk.caches.arrays]
        busy = rank in BUSY
        rows = KI.request_block_rows(Q)  # one request: one global block
        local = VA.LocalStep(VA.ELIGIBLE, VA.ELIGIBLE, rows) if busy else None
        entries = [0, 0] + [rows if r in BUSY else 0 for r in range(WORLD)]  # reduced
        d = mk.decide_dp(local, max_tokens=Q, max_chunks=1, any_prefill=False,
                         entries=entries)  # fmt: skip
        out["decision"] = [d.run, d.key, d.idle]
        if busy:
            md = fake_metadata(mk, num_reqs=1, padded_reqs=4, bt_cols=widths,
                               device="tpu")  # fmt: skip
            VA._current_attn_metadata = lambda: md
        ids, pos = _inputs(rank)
        ids_t, pos_t = torch.from_numpy(ids).to("tpu"), torch.from_numpy(pos).to("tpu")
        hidden, aux = mk.forward(None, ids_t, pos_t, None, None)
        h = hidden.float().cpu().numpy()
        a = np.stack([x.float().cpu().numpy() for x in aux])
        np.savez(result_dir / f"adapter_{rank}.npz", hidden=h, aux=a)
        out.update(
            hidden_finite=bool(np.isfinite(h).all()),
            padding_rows_zero=bool(not h[M:].any()),
            caches_unchanged=all(
                torch.equal(t.cpu(), b) for t, b in zip(mk.caches.arrays, before)
            ),
        )
    except BaseException as exc:  # noqa: BLE001 - reported to the parent
        traceback.print_exc()
        out["error"] = repr(exc)
    finally:
        (result_dir / f"rank_{rank}.json").write_text(json.dumps(out))
    if "error" in out:
        raise SystemExit(1)


def _jax_twin(result_dir: Path) -> None:
    """The contract fn under a plain JAX shard_map on the 8 local devices,
    with the operands the adapter's prep should have built; plus references."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P

    from tests.kernels.deepseek_v4_megakernel.test_step_impl import np_tail
    from tests.kernels.deepseek_v4_megakernel.test_weight_map import t2j
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import kernel_interface as KI
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import step_impl as SI
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import step_reference as SR

    env = _setup(jax, jnp)
    layers, bound, widths = _fake_tree("cpu")
    cfg = KI.StepConfig(
        model=env["spec"], layers=bound.layers,
        cache_shapes=tuple((tuple(t.shape), str(t.dtype)[6:]) for t in bound.arrays),
        group_widths=tuple(widths), swa_page=128,
    )  # fmt: skip
    bucket = KI.Bucket(q=Q, rows=T, reqs=B)
    fn = SI.build_step(cfg, bucket, attn_impl="stub", moe_impl="stub",
                       dense=env["tp_cfg"], logits=LOGITS)  # fmt: skip
    per_rank = []
    for r in range(WORLD):
        ids, pos = _inputs(r)
        busy = r in BUSY
        meta = np.zeros(KI.META_LEN, np.int32)
        meta[KI.META_RANK], meta[KI.META_Q], meta[KI.META_EBLK] = r, Q, r
        ids_m, pos_m = np.zeros(M, np.int32), np.zeros(M, np.int32)
        seq = np.zeros(B, np.int32)
        bts = [np.zeros((B, wd), np.int32) for wd in widths]
        if busy:  # request 0 active (fake_metadata: table row 0 = 1..w)
            ids_m[:Q], pos_m[:Q], seq[0] = ids[:Q], pos[:Q], SEQ0
            if os.environ.get("DSV4_ADAPTER_FAULT") and r == 0:
                ids_m[0] = (ids_m[0] + 1) % 512  # positive control: one token off
            for bt, wd in zip(bts, widths):
                bt[0] = np.arange(1, wd + 1)
        caches = [np.asarray(t2j(t)) for t in bound.arrays]
        per_rank.append([meta, ids_m, pos_m, seq, *bts, *env["weights"](r), *caches])
    ops = [np.stack([pr[i] for pr in per_rank]) for i in range(len(per_rank[0]))]
    n_out = 1 + 2 + len(bound.arrays)

    def body(*xs):
        return tuple(o[None] for o in fn(*[x[0] for x in xs]))

    mesh = jax.make_mesh((WORLD,), ("ep",))
    specs, out_specs = (P("ep"),) * len(ops), (P("ep"),) * n_out
    f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=specs, out_specs=out_specs,
                              check_vma=False))  # fmt: skip
    shard = jax.sharding.NamedSharding(mesh, P("ep"))
    outs = f(*[jax.device_put(jnp.asarray(o), shard) for o in ops])
    hid = np.asarray(outs[0].astype(jnp.float32))
    aux = np.stack([np.asarray(outs[1 + i].astype(jnp.float32)) for i in range(2)], 1)
    lm_ref = np.zeros((1,), np.float32)
    if LOGITS:  # the same step without the head's lm_head, then hidden @ lm_head.T
        plain = SI.build_step(cfg, bucket, attn_impl="stub", moe_impl="stub",
                              dense=env["tp_cfg"])  # fmt: skip
        n_w = len(KI.weight_manifest(env["spec"], logits=True))
        G0 = len(widths)
        # drop the lm_head operand (last weight) for the plain step
        cut = 4 + G0 + n_w - 1
        ops_plain = ops[:cut] + ops[cut + 1 :]

        def body_plain(*xs):
            return tuple(o[None] for o in plain(*[x[0] for x in xs]))

        fp = jax.jit(jax.shard_map(body_plain, mesh=mesh,
                                   in_specs=(P("ep"),) * len(ops_plain),
                                   out_specs=out_specs, check_vma=False))  # fmt: skip
        h_plain = fp(*[jax.device_put(jnp.asarray(o), shard) for o in ops_plain])[0]
        lm = np.asarray(env["lm_head"].astype(jnp.float32))
        lm_ref = np.asarray(h_plain.astype(jnp.float32)) @ lm.T
        # rounded to bf16 as the stock LogitsProcessor's output
        lm_ref = np.asarray(
            jnp.asarray(lm_ref).astype(jnp.bfloat16).astype(jnp.float32)
        )

    # the kernel's own residual streams (before the tail), for attribution
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import step_kernel as SK

    step = SK.build(SI.kernel_config(cfg, bucket, env["tp_cfg"]), axis="ep")
    names = [e.name for e in KI.weight_manifest(env["spec"])]
    G = len(widths)

    def raw(*xs):
        xs = [x[0] for x in xs]
        w_ = dict(zip(names, xs[4 + G : 4 + G + len(names)]))
        tiles = {n: w_["dense_" + n] for n in env["tp_cfg"].stream_names}
        me = xs[0][KI.META_RANK]
        res, ax, *_ = step(xs[1], xs[2], xs[3], w_["embed"], w_["cos_sin"],
                           w_["dense_smalls"], tiles, None, me=me)  # fmt: skip
        return res[None], ax[None]

    two = (P("ep"), P("ep"))
    rf = jax.jit(jax.shard_map(raw, mesh=mesh, in_specs=specs, out_specs=two,
                               check_vma=False))  # fmt: skip
    kres, _ = rf(*[jax.device_put(jnp.asarray(o), shard) for o in ops])
    kres = np.asarray(kres.astype(jnp.float32))

    all_ids = np.stack([pr[1] for pr in per_rank])
    all_pos = np.stack([pr[2] for pr in per_rank])
    all_seq = np.stack([pr[3] for pr in per_rank])
    rc = SR.RefConfig(full=env["full"], R=WORLD, T=T, q=Q, aux_layers=(1, 2),
                      ar_mode="hrsag", ar_group=8)  # fmt: skip
    g = env["glob"]
    tail = (g["hc_head_fn"], g["hc_head_scale"], g["hc_head_base"], g["final_norm"])
    refs, ref_res = {}, {}
    for order in ("kernel", "stock"):
        with jax.default_device(jax.devices("cpu")[0]):
            res, ax = SR.step_reference(rc, env["w"], g["embed"], env["cos_sin"],
                                        all_ids, all_pos, all_seq, order)  # fmt: skip
        ref_res[order] = np.asarray(jnp.asarray(res, jnp.float32))
        refs[order] = [np_tail(res[r], ax[r], *tail) for r in range(WORLD)]
    tail_np = np.stack([np_tail(kres[r], np.zeros((2,) + kres[r].shape), *tail)[0]
                        for r in range(WORLD)])  # fmt: skip
    np.savez(result_dir / "twin.npz", hidden=hid, aux=aux, lm_ref=lm_ref,
             ref_k=np.stack([h for h, _ in refs["kernel"]]),
             ref_s=np.stack([h for h, _ in refs["stock"]]),
             kres=kres, ref_res_k=ref_res["kernel"], ref_res_s=ref_res["stock"],
             tail_np=tail_np)  # fmt: skip


def _rel(x, y):
    return float(np.linalg.norm(x - y) / (np.linalg.norm(y) + 1e-30))


def _launch(result_dir: Path) -> int:
    from torch_tpu._internal.distributed.launchers.singlehost_wrapper import (
        prepare_tpu_environment,
    )

    saved = dict(os.environ)
    prepare_tpu_environment(world_size=WORLD)
    env = os.environ.copy()
    os.environ.clear()
    os.environ.update(saved)
    env.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false")
    env.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")
    me = "tests.kernels.deepseek_v4_megakernel.adapter_step_device"
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
           f"--nproc-per-node={WORLD}", "-m", me, "--worker",
           "--result-dir", str(result_dir)]  # fmt: skip
    return subprocess.run(cmd, env=env, check=False).returncode


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--twin", action="store_true")
    p.add_argument("--fault", action="store_true", help="positive control")
    p.add_argument("--logits", action="store_true", help="vocab-sharded lm_head")
    p.add_argument("--result-dir", type=Path, default=Path("/tmp/dsv4_adapter_step"))
    args = p.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    if args.logits:  # read by the workers and the twin (inherited)
        os.environ["DSV4_ADAPTER_LOGITS"] = "1"
    if args.worker:
        _worker(args.result_dir)
        return 0
    if args.twin:
        _jax_twin(args.result_dir)
        return 0
    code = _launch(args.result_dir)
    if args.fault:
        os.environ["DSV4_ADAPTER_FAULT"] = "1"  # read by the twin
    me = "tests.kernels.deepseek_v4_megakernel.adapter_step_device"
    twin = subprocess.run(
        [sys.executable, "-m", me, "--twin", "--result-dir", str(args.result_dir)],
        check=False,
    ).returncode
    bad = code != 0 or twin != 0
    tw = np.load(args.result_dir / "twin.npz") if twin == 0 else None
    for r in range(WORLD):
        f = args.result_dir / f"rank_{r}.json"
        res = (
            json.loads(f.read_text()) if f.exists() else {"rank": r, "error": "missing"}
        )
        if "error" not in res and tw is not None:
            ad = np.load(args.result_dir / f"adapter_{r}.npz")
            res["vs_jax_twin_bitexact"] = bool(
                np.array_equal(ad["hidden"][:M], tw["hidden"][r])
                and np.array_equal(ad["aux"][:, :M], tw["aux"][r])
            )
            res["vs_twin_rel"] = _rel(ad["hidden"][:M], tw["hidden"][r])
            res["twin_vs_ref_kernel"] = _rel(tw["hidden"][r], tw["ref_k"][r])
            res["kernel_res_vs_ref"] = _rel(tw["kres"][r], tw["ref_res_k"][r])
            res["res_band"] = _rel(tw["ref_res_s"][r], tw["ref_res_k"][r])
            res["tail_tpu_vs_numpy"] = _rel(tw["hidden"][r], tw["tail_np"][r])
            res["band_stock_vs_kernel"] = _rel(tw["ref_s"][r], tw["ref_k"][r])
            if LOGITS or args.logits:  # the logits vs the hidden through the lm_head
                got, ref = tw["hidden"][r][:Q], tw["lm_ref"][r][:Q]  # request 0's rows
                res["logits_vs_hidden_lm_head"] = _rel(got, ref)
                res["logits_bitwise_equal"] = float(np.mean(got == ref))
                res["argmax_agree"] = (
                    f"{int(np.sum(got.argmax(-1) == ref.argmax(-1)))}/{Q}"
                )
                bad |= not res["logits_vs_hidden_lm_head"] < 1e-2
            bad |= not res["vs_jax_twin_bitexact"] or not res["caches_unchanged"]
            bad |= not res["decision"][0] or not res["padding_rows_zero"]
        else:
            bad = True
        print("RESULT", json.dumps(res), flush=True)
    print(f"RESULT ok={not bad}", flush=True)
    return int(bad)


if __name__ == "__main__":
    sys.exit(main())
