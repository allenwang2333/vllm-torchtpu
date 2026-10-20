# SPDX-License-Identifier: Apache-2.0
"""Byte-compare the served fused-EP MoE layer with and without an expert
relabelling (TPU_MOE_EXPERT_PERM_PATH), on the device, through the isolated
harness in ../moe_attr.

    python check_relabel.py --perm placement.json --routing-npz routing_round0.npz \
        --layers 22,10 [--step 10] [--kernel-rev d186106c4] [--out res.jsonl]

For each layer it relabels the experts the way weight load does (expert n holds
checkpoint expert argsort(perm)[n]; gate columns and score bias follow) and
runs four comparisons of the layer output [2304, 5120] bf16:
  identity  the relabelling plumbing with perm = arange (must be bytewise equal)
  scores    router path on harness logits, permuted vs not
  replay    the served step's real expert ids (hash path), permuted vs not
  gate      x @ W_gate^T with W_gate's rows relabelled vs the columns relabelled
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "moe_attr"))

import ablate  # noqa: E402
import bench_moe_attr as bench  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402


def same_bytes(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    if a.shape != b.shape:
        return False
    return bool(np.array_equal(a.view(np.uint8), b.view(np.uint8)))


def rel_l2(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--perm", required=True)
    ap.add_argument("--routing-npz", required=True)
    ap.add_argument("--layers", default="22,10")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--skew", type=float, default=0.7)
    ap.add_argument("--kernel-rev", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.kernel_rev:
        ablate.pin_source_rev(a.kernel_rev)
    _, bench.moe_layer = ablate.install("base")
    mesh = bench.mesh8()
    shard = NamedSharding(mesh, P(bench.AXIS))
    geo = bench.GEO
    e = geo["e_total"]
    x, gates, bias = bench.make_inputs(
        mesh,
        1,
        tokens=geo["tokens"],
        hidden=geo["hidden"],
        e_total=e,
        n_layers=1,
        skew=a.skew,
    )
    gate = gates[0]
    w3d = bench.make_fp4_weights(
        mesh, 100, hidden=geo["hidden"], inter=geo["inter"], e_total=e, qb=geo["qb"]
    )
    with open(a.perm) as f:
        perms = json.load(f)["perm"]
    table = np.load(a.routing_npz)
    layers = table["layers"].tolist()
    wg = jax.random.normal(jax.random.key(5), (e, geo["hidden"]), jnp.float32).astype(
        jnp.bfloat16
    )

    @jax.jit
    def gate_logits(x, w):
        return jnp.dot(x, w.T, preferred_element_type=jnp.float32)

    def relabel_weights(inv):
        idx = jnp.asarray(inv)
        moved = tuple(jax.device_put(jnp.take(t, idx, axis=0), shard) for t in w3d)
        return bench.to_scale_form(mesh, moved, "flat")

    w_off = bench.to_scale_form(mesh, w3d, "flat")
    results = []

    def record(layer, check, equal, ref, new):
        rec = dict(
            layer=layer,
            check=check,
            bitexact=equal,
            rel_l2=0.0 if equal else rel_l2(new, ref),
            kernel_rev=a.kernel_rev,
        )
        print("RESULT", json.dumps(rec), flush=True)
        results.append(rec)

    ref_scores = np.asarray(bench.moe_call(x, w_off, gate, bias, mesh))
    for layer in (int(v) for v in a.layers.split(",")):
        perm = np.asarray(perms[str(layer)])
        inv = np.argsort(perm)
        # identity: the same plumbing with perm = arange.
        ident = relabel_weights(np.arange(e))
        out = bench.moe_call(x, ident, gate[:, np.arange(e)], bias[np.arange(e)], mesh)
        record(layer, "identity", same_bytes(out, ref_scores), ref_scores, out)
        w_new = relabel_weights(inv)
        out = bench.moe_call(
            x, w_new, jax.device_put(gate[:, inv], shard), bias[inv], mesh
        )
        record(layer, "scores", same_bytes(out, ref_scores), ref_scores, out)
        ids = table["topk_ids"][layers.index(layer), a.step].astype(np.int32)
        ref = np.asarray(
            bench.moe_call(
                x, w_off, gate, bias, mesh, hash_ids=jax.device_put(ids, shard)
            )
        )
        out = bench.moe_call(
            x,
            w_new,
            jax.device_put(gate[:, inv], shard),
            bias[inv],
            mesh,
            hash_ids=jax.device_put(perm[ids].astype(np.int32), shard),
        )
        record(layer, "replay", same_bytes(out, ref), ref, out)
        ref_g = np.asarray(gate_logits(x, wg))[:, inv]
        out_g = gate_logits(x, wg[inv])
        record(layer, "gate", same_bytes(out_g, ref_g), ref_g, out_g)
    if a.out:
        with open(a.out, "a") as f:
            for rec in results:
                f.write(json.dumps(rec) + "\n")
    if not all(r["bitexact"] for r in results):
        raise SystemExit("relabelled output differs")


if __name__ == "__main__":
    main()
