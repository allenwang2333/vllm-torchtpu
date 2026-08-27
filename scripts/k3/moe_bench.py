"""Stage-by-stage bench for the Kimi-K3 decode MoE block: fused vs unfused.

Purpose: iterate on the mega-MoE kernel in SECONDS instead of the ~20 minutes an
end-to-end server bench costs. Each stage of MEGA_MOE_KERNEL_PLAN.md has an
"unfused" reference built from the same ops the model runs, and a "fused" slot
that is filled in as kernels land. Run any subset; unimplemented fused stages
fall back to the reference and are reported as such, so the harness stays useful
from S1 through S5.

WHAT THIS HARNESS IS AND IS NOT
-------------------------------
It measures the block in a SMALL program. The model runs it inside a 92-layer
unrolled graph, and that difference is real and unexplained: identical
instructions that reach 1.9-2.7 TB/s here run at 0.60 TB/s in the model for
19 of 24 instances (see MEGA_MOE_KERNEL_PLAN.md sec 9). The model also has MSA
promoting some weights to VMEM, which this harness deliberately defeats.

So: use ABSOLUTE numbers only to compare fused vs unfused *within a run*, and
treat the model column as a calibration check, not a target. A win here is
necessary but NOT sufficient for a win in the model -- both kernels built so far
won here and have not yet been shown to win end to end.

METHODOLOGY (deviating from any of this has produced wrong answers before)
-------------------------------------------------------------------------
* Slope timing. Timing a chain of DEPTH blocks and dividing by DEPTH charges
  every block a share of host dispatch (~100 us). Measure at two depths and take
  (T_hi - T_lo) / (DEPTH_hi - DEPTH_lo), which cancels any depth-independent cost.
* Distinct weights per chain step, sized past VMEM, or XLA hoists the load out
  of the chain and MSA promotes the single copy -- both report impossible rates.
* --check gates numerics against the unfused reference before any timing.

USAGE
    PYTHONPATH=src .venv/bin/python scripts/k3/moe_bench.py --check
    PYTHONPATH=src .venv/bin/python scripts/k3/moe_bench.py --stages sharedmlp
    PYTHONPATH=src .venv/bin/python scripts/k3/moe_bench.py --tokens 8 --depths 4,16
    PYTHONPATH=src .venv/bin/python scripts/k3/moe_bench.py --stages mega --check \
        --bn1 384 --ahead1 1 --bn2 3584 --ahead2 2   # tuned mega config (VMEM-fit)
    PYTHONPATH=src .venv/bin/python scripts/k3/moe_bench.py --stages full --check \
        --bnd 256 --aheadd 2 --bn1 384 --ahead1 1 --aheade 2 --bnup 512 --aheadup 2

The `mega` stage (gate + topk + down_proj + shared MLP + routed experts in one
Pallas call, src/vllm_torchtpu/kernels/kimi_k3/mega_moe.py) is opt-in;
--bnd/--aheadd tune its down_proj stream, --aheade the expert rings, and
--bn1/--bn2/--ahead1/--ahead2 the shared-MLP streams (shared with the
sharedmlp stage, which is tuned for 256/512/5/4). Setting --aheadd 0 disables
ring priming, which is the A/B knob for how much the stream-during-topk
overlap is worth. The `full` stage adds the all-reduce + norm + up_proj tail
inside the same kernel (one-shot AR; needs --devices > 1 and runs meshed);
--bnup/--aheadup tune its up_proj ring.
"""
from __future__ import annotations

import argparse
import dataclasses
import time

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------- shapes ----


@dataclasses.dataclass(frozen=True)
class Shapes:
    """Kimi-K3 decode shapes, per core, at TP=8 with intra-chip latent TP."""
    tokens: int = 8
    hidden: int = 7168
    latent: int = 3584  # routed_expert_hidden_size
    latent_shard: int = 1792  # latent / 2, intra-chip TP halves down_proj
    up_shard: int = 3584  # up_proj is [3584, 3584] per core
    n_experts_global: int = 224
    top_k: int = 16
    shared_inter: int = 768  # moe_intermediate 3072 * 2 shared / TP8
    n_experts_local: int = 8  # mega stage: local experts per core (proxy)
    expert_inter: int = 384  # mega stage: per-core routed intermediate shard
    eps: float = 1e-5
    beta: float = 4.0
    linear_beta: float = 25.0


# Measured per-layer cost in the model (rank-0 control profile, median step,
# 92 layers). Calibration reference only -- see the caveat at the top.
MODEL_US = {
    "gate": 9.68,  # router gate + grouped topk (the `sort` ops)
    "downproj": 11.76,  # latent down_proj [1792,7168]
    "expertgmm": 44.30,  # both mxfp4 GMMs, k=3584 and k=3072
    "permute": 6.81,  # token gather / permute
    "sharedmlp": 21.13,  # shared gate_up 12.64 + shared down_proj 8.49
    "tail": 47.40,  # LL all-reduce 35.32 + norm + up_proj 12.08
}
# The mega stage fuses gate + downproj + sharedmlp + the routed experts (which
# also removes the permute). NOT calibration-comparable: the harness expert
# stage is DENSE over n_experts_local and the unfused arm pays XLA's mxfp4
# dequant materialization, where the model is ragged and dequants inside
# gmm_v2 -- the expert/permute share of this column (~51 us) prices different
# work than the harness does.
MODEL_US["mega"] = (MODEL_US["gate"] + MODEL_US["downproj"] +
                    MODEL_US["sharedmlp"] + MODEL_US["expertgmm"] +
                    MODEL_US["permute"])
# The full stage is mega + the collective tail: the whole MoE block.
MODEL_US["full"] = MODEL_US["mega"] + MODEL_US["tail"]
STAGE_ORDER = [
    "gate", "downproj", "permute", "expertgmm", "sharedmlp", "tail", "mega",
    "full"
]

# Stages whose unfused reference is knowingly incomplete, so their model column
# is not comparable and they must be excluded from the calibration ratio.
# `tail` is complete only with --devices > 1; without a mesh it measures
# norm+up_proj and misses the collective, which is 35.3 of its 47.4 us.
INCOMPLETE_REF_1DEV = {
    "tail": "collective needs --devices > 1",
    "mega": "dense local experts vs model's ragged dispatch",
    "full": "dense experts + collective skew not reproduced"
}

# ------------------------------------------------------- unfused reference ----


def ref_gate(h, w_gate, s: Shapes):
    """Router gate + grouped topk (noaux_tc, sigmoid scoring)."""
    logits = jax.lax.dot_general(h.astype(jnp.float32),
                                 w_gate.astype(jnp.float32),
                                 (((1, ), (1, )), ((), ())))
    scores = jax.nn.sigmoid(logits)
    weights, ids = jax.lax.top_k(scores, s.top_k)
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    return weights, ids


def ref_downproj(h, w_down, s: Shapes):
    """hidden -> latent shard. Norm is upstream and shared, so not included."""
    return jax.lax.dot_general(h,
                               w_down, (((1, ), (1, )), ((), ())),
                               preferred_element_type=jnp.float32).astype(
                                   h.dtype)


def ref_sharedmlp(h, w_gu, w_dn, s: Shapes):
    """gate_up -> situ -> down_proj, this rank's partial."""
    gu = jax.lax.dot_general(h,
                             w_gu, (((1, ), (1, )), ((), ())),
                             preferred_element_type=jnp.float32).astype(
                                 h.dtype)
    g = gu[:, :s.shared_inter].astype(jnp.float32)
    u = gu[:, s.shared_inter:].astype(jnp.float32)
    g = s.beta * jnp.tanh(g / s.beta) * jax.nn.sigmoid(g)
    u = s.linear_beta * jnp.tanh(u / s.linear_beta)
    a = (g * u).astype(h.dtype)
    return jax.lax.dot_general(a,
                               w_dn, (((1, ), (1, )), ((), ())),
                               preferred_element_type=jnp.float32).astype(
                                   h.dtype)


def ref_tail(lat, gamma, w_up, s: Shapes):
    """RMS norm on the reduced latent, then up_proj. Single-device path only.

    The collective is added by `tail_arms` when a mesh is available; see the
    skew caveat there before comparing this stage's numbers to the model.
    """
    f = lat.astype(jnp.float32)
    inv = jax.lax.rsqrt(jnp.mean(f * f, axis=-1, keepdims=True) + s.eps)
    n = (f * inv * gamma.astype(jnp.float32)).astype(lat.dtype)
    return jax.lax.dot_general(n,
                               w_up, (((1, ), (1, )), ((), ())),
                               preferred_element_type=jnp.float32).astype(
                                   lat.dtype)


def ref_experts(lat, wt, ids, w13q, w13s, w2q, w2s, s: Shapes):
    """Routed experts, dense over local experts: out[t] = sum_e m[e,t] ffn_e(lat[t]).

    mxfp4 weights: ``w13q``/``w2q`` are packed uint8 ``[E_local, K, N//2]``
    with f32 scales ``[E_local, K//32, N]`` (quantize_fp4). Experts
    consume this core's latent shard but produce a FULL-latent partial (the
    model's TP semantics: partials across cores are summed by the tail's
    all-reduce). Global expert id g maps to local expert ``g % E_local``
    (single-device harness: all routing is local). Dense on purpose: same
    streamed bytes as the fused arm.
    """
    from vllm_torchtpu.kernels.kimi_k3.mega_moe import dequant_fp4
    ids_local = ids % s.n_experts_local
    acc = jnp.zeros((lat.shape[0], s.latent), jnp.float32)
    for e in range(s.n_experts_local):
        m = jnp.sum(jnp.where(ids_local == e, wt, 0.0), axis=1)  # [T]
        w13 = dequant_fp4(w13q[e], w13s[e]).astype(lat.dtype)
        w2 = dequant_fp4(w2q[e], w2s[e]).astype(lat.dtype)
        gu = jax.lax.dot_general(lat,
                                 w13, (((1, ), (0, )), ((), ())),
                                 preferred_element_type=jnp.float32)
        g = gu[:, :s.expert_inter]
        u = gu[:, s.expert_inter:]
        g = s.beta * jnp.tanh(g / s.beta) * jax.nn.sigmoid(g)
        u = s.linear_beta * jnp.tanh(u / s.linear_beta)
        y = jax.lax.dot_general((g * u).astype(lat.dtype),
                                w2, (((1, ), (0, )), ((), ())),
                                preferred_element_type=jnp.float32)
        acc = acc + y * m[:, None].astype(jnp.float32)
    return acc.astype(lat.dtype)


# ---------------------------------------------------------- fused variants ----


def fused_sharedmlp(h, w_gu_blocked, w_dn_blocked, gamma_ones, s: Shapes,
                    bn1: int, bn2: int, a1: int, a2: int):
    from vllm_torchtpu.kernels.kimi_k3.shared_mlp import shared_mlp
    return shared_mlp(h,
                      w_gu_blocked,
                      w_dn_blocked,
                      gamma_ones,
                      bn1=bn1,
                      bn2=bn2,
                      ahead1=a1,
                      ahead2=a2,
                      do_norm=False,
                      beta=s.beta,
                      linear_beta=s.linear_beta)


def fused_mega(x, w_gate_p, w_down_b, w_gu_b, w_dn_b, w13q, w13s, w2q, w2s,
               gamma_ones, s: Shapes, bn_d: int, bn1: int, bn2: int, a_d: int,
               a1: int, a2: int, a_e: int):
    """gate + topk + downproj + sharedmlp + routed experts, one Pallas boundary.

    Returns (topk_weights, topk_ids, latent, shared_out, expert_out); the
    caller folds them back into the chain.
    """
    from vllm_torchtpu.kernels.kimi_k3.mega_moe import mega_moe
    return mega_moe(x,
                    w_gate_p,
                    w_down_b,
                    w_gu_b,
                    w_dn_b,
                    w13q,
                    w13s,
                    w2q,
                    w2s,
                    gamma_ones,
                    n_experts=s.n_experts_global,
                    top_k=s.top_k,
                    bn_d=bn_d,
                    bn1=bn1,
                    bn2=bn2,
                    ahead_d=a_d,
                    ahead1=a1,
                    ahead2=a2,
                    ahead_e=a_e,
                    do_norm=False,
                    beta=s.beta,
                    linear_beta=s.linear_beta,
                    vmem_limit_bytes=110 * 1024 * 1024)


def tail_arms(nd: int, s: Shapes, bn: int, n_ahead: int):
    """Unfused (LL all-reduce -> norm -> up_proj) and fused (v4) tail arms.

    IMPORTANT -- how to read the collective here. In the model the LL all-reduce
    costs 35.3 us/layer but ranges 8-68 us per occurrence, and most of that
    spread is waiting for the slowest peer, which depends on upstream imbalance
    across the 92 layers. This harness runs all devices in lockstep in a tight
    loop, so it measures the collective's WORK and almost none of its SKEW, and
    will read far below the model. Use it to compare fused vs unfused tails
    against each other, never to predict the tail's absolute cost in situ.
    """
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.collectives.all_reduce_norm_matmul_v2 import \
        padded_latent_width
    from vllm_torchtpu.kernels.collectives.all_reduce_norm_matmul_v4 import (
        all_reduce_norm_matmul_v4, blockify_weight)
    from vllm_torchtpu.kernels.collectives.low_latency_all_reduce import (
        make_two_shot_workspace, two_shot_all_reduce_ws)

    AXIS = "dev"
    devs = jax.devices()[:nd]
    mesh = Mesh(devs, (AXIS, ))

    def rep(t):
        return jax.device_put(t, NamedSharding(mesh, P()))

    k_pad = padded_latent_width(s.latent, nd)
    cols = k_pad // nd

    def rms(v, g):
        f = v.astype(jnp.float32)
        inv = jax.lax.rsqrt(jnp.mean(f * f, axis=-1, keepdims=True) + s.eps)
        return (f * inv * g.astype(jnp.float32)).astype(v.dtype)

    # Chain steps must stay data-dependent or XLA reorders/hoists them.
    def tie(x, prev):
        return (x if prev is None else x +
                (prev.reshape(-1)[0] * 1e-30).astype(x.dtype))

    def unfused(x, g, rs, ag, st, *ws):
        acc = None
        for w in ws:
            red, rs, ag, st = two_shot_all_reduce_ws(tie(x, acc),
                                                     rs,
                                                     ag,
                                                     st,
                                                     axis_name=AXIS,
                                                     num_devices=nd,
                                                     collective_id=3,
                                                     has_side_effects=False)
            v = jax.lax.dot_general(rms(red.reshape(s.tokens, s.latent), g),
                                    w, (((1, ), (1, )), ((), ())),
                                    preferred_element_type=jnp.float32)
            acc = v if acc is None else acc + v
        return acc

    def fused(x, g, rs, ag, st, *ws):
        acc = None
        for w in ws:
            v, rs, ag, st = all_reduce_norm_matmul_v4(tie(x, acc),
                                                      w,
                                                      g,
                                                      rs,
                                                      ag,
                                                      st,
                                                      axis_name=AXIS,
                                                      num_devices=nd,
                                                      latent=s.latent,
                                                      bn=bn,
                                                      n_ahead=n_ahead,
                                                      collective_id=9,
                                                      has_side_effects=False)
            acc = v if acc is None else acc + v
        return acc

    return mesh, rep, AXIS, k_pad, cols, unfused, fused, blockify_weight, \
        make_two_shot_workspace


# Stages with no kernel yet. Keep the keys so --stages accepts them and the
# report shows an explicit "not implemented" rather than silently omitting them.
FUSED_AVAILABLE = {"sharedmlp", "mega"}

# ------------------------------------------------------------------ timing ----


def repeated(fn, trials: int):
    """Median and half-range of `trials` independent slope measurements.

    A single slope reading of the shared-MLP delta varied 8x (-0.15 to -1.19 us)
    across runs of an identical config. Report the spread or the harness cannot
    distinguish a real kernel win from run-to-run noise.
    """
    vals = sorted(fn() for _ in range(trials))
    med = vals[len(vals) // 2]
    return med, (vals[-1] - vals[0]) / 2


def slope_time_pairs(make, lo: int, hi: int, reps: int) -> float:
    """Slope over depth where the callable itself depends on depth.

    shard_map's in_specs must match the argument count, so the meshed arms need
    a fresh jit per chain depth rather than one jit reused.
    """

    def once(depth):
        fn, a = make(depth)
        jax.block_until_ready(fn(*a))
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(*a))
            best = min(best, time.perf_counter() - t0)
        return best

    return (once(hi) - once(lo)) / (hi - lo)


def slope_time(fn, args_for_depth, lo: int, hi: int, reps: int) -> float:
    """Per-block time with depth-independent cost cancelled."""

    def once(depth):
        a = args_for_depth(depth)
        fn(*a).block_until_ready()
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            fn(*a).block_until_ready()
            best = min(best, time.perf_counter() - t0)
        return best

    return (once(hi) - once(lo)) / (hi - lo)


def make_weights(key,
                 s: Shapes,
                 depth: int,
                 stage: str,
                 bn1: int,
                 bn2: int,
                 bn_d: int = 256):
    """Distinct weights per chain step so nothing is hoisted or promoted."""
    from vllm_torchtpu.kernels.kimi_k3.shared_mlp import blockify

    def mk(i, shape):
        return (jax.random.normal(jax.random.fold_in(key, i), shape,
                                  jnp.bfloat16))

    out = []
    for i in range(depth):
        if stage == "gate":
            out.append((mk(i, (s.n_experts_global, s.hidden)), ))
        elif stage == "downproj":
            out.append((mk(i, (s.latent_shard, s.hidden)), ))
        elif stage == "sharedmlp":
            gu = mk(100 + i, (2 * s.shared_inter, s.hidden))
            dn = mk(200 + i, (s.hidden, s.shared_inter))
            out.append((gu, dn, blockify(gu.T, bn1), blockify(dn.T, bn2)))
        elif stage == "mega":
            from vllm_torchtpu.kernels.kimi_k3.mega_moe import (prepare_gate,
                                                                quantize_fp4)
            wg = mk(i, (s.n_experts_global, s.hidden))
            wd = mk(300 + i, (s.latent_shard, s.hidden))
            gu = mk(100 + i, (2 * s.shared_inter, s.hidden))
            dn = mk(200 + i, (s.hidden, s.shared_inter))
            w13q, w13s = quantize_fp4(
                mk(500 + i,
                   (s.n_experts_local, s.latent_shard, 2 * s.expert_inter)))
            w2q, w2s = quantize_fp4(
                mk(600 + i, (s.n_experts_local, s.expert_inter, s.latent)))
            out.append((wg, wd, gu, dn, w13q, w13s, w2q, w2s, prepare_gate(wg),
                        blockify(wd.T,
                                 bn_d), blockify(gu.T,
                                                 bn1), blockify(dn.T, bn2)))
        elif stage == "tail":
            out.append((mk(300 + i,
                           (1, s.latent)), mk(400 + i,
                                              (s.up_shard, s.latent))))
        else:
            raise NotImplementedError(f"no weights for stage {stage!r}")
    return out


def stage_chain(stage: str,
                s: Shapes,
                fused: bool,
                bn1,
                bn2,
                a1,
                a2,
                bn_d=1792,
                a_d=1,
                a_e=3):
    """Return a jitted chain over DEPTH independent blocks of one stage."""

    def body(x, ws, gamma_ones):
        for w in ws:
            if stage == "gate":
                wt, ids = ref_gate(x, w[0], s)
                # keep the result live and shaped like x so the chain continues
                x = x + jnp.sum(wt).astype(x.dtype)
            elif stage == "downproj":
                y = ref_downproj(x, w[0], s)
                x = jnp.pad(y, ((0, 0), (0, s.hidden - s.latent_shard)))
            elif stage == "sharedmlp":
                x = (fused_sharedmlp(x, w[2], w[3], gamma_ones, s, bn1, bn2,
                                     a1, a2) if fused else ref_sharedmlp(
                                         x, w[0], w[1], s))
            elif stage == "mega":
                if fused:
                    wt, ids, lat, sh, exp = fused_mega(x, w[8], w[9], w[10],
                                                       w[11], w[4], w[5], w[6],
                                                       w[7], gamma_ones, s,
                                                       bn_d, bn1, bn2, a_d, a1,
                                                       a2, a_e)
                else:
                    wt, ids = ref_gate(x, w[0], s)
                    lat = ref_downproj(x, w[1], s)
                    sh = ref_sharedmlp(x, w[2], w[3], s)
                    exp = ref_experts(lat, wt, ids, w[4], w[5], w[6], w[7], s)

                def pad(v):
                    return (jnp.pad(v, ((0, 0), (0, s.hidden - v.shape[1]))))

                x = sh + pad(lat) + pad(exp)
                # keep topk outputs live without perturbing the chain
                tie = (jnp.sum(wt) + jnp.sum(ids).astype(jnp.float32)) * 1e-30
                x = x + tie.astype(x.dtype)
            elif stage == "tail":
                lat = x[:, :s.latent]
                y = ref_tail(lat, w[0], w[1], s)
                x = jnp.pad(y, ((0, 0), (0, s.hidden - s.up_shard)))
            else:
                raise NotImplementedError(stage)
        return x

    return jax.jit(body)


# ------------------------------------------------------------------- main ----


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="gate,downproj,sharedmlp,tail")
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--depths", default="4,16", help="lo,hi for the slope fit")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--trials",
                    type=int,
                    default=5,
                    help="independent repeats of each slope measurement")
    ap.add_argument("--bn1", type=int, default=256)
    ap.add_argument("--bn2", type=int, default=512)
    ap.add_argument("--ahead1", type=int, default=5)
    ap.add_argument("--ahead2", type=int, default=4)
    ap.add_argument("--bnd",
                    type=int,
                    default=1792,
                    help="mega stage: down_proj block width")
    ap.add_argument("--aheadd",
                    type=int,
                    default=1,
                    help="mega stage: down_proj ring prefetch depth")
    ap.add_argument("--aheade",
                    type=int,
                    default=2,
                    help="mega stage: expert ring prefetch depth")
    ap.add_argument("--bnup",
                    type=int,
                    default=512,
                    help="full stage: up_proj block width")
    ap.add_argument("--aheadup",
                    type=int,
                    default=1,
                    help="full stage: up_proj ring prefetch depth")
    ap.add_argument("--union",
                    action="store_true",
                    help="full stage: enable the down-ring/up-ring union "
                    "(measured neutral-to-worse in lockstep; a "
                    "skew-regime play)")
    ap.add_argument("--devices",
                    type=int,
                    default=0,
                    help="devices for the meshed tail stage; "
                    "0 = all. 1 disables the collective.")
    ap.add_argument("--tail-bn", type=int, default=512)
    ap.add_argument("--tail-ahead", type=int, default=4)
    ap.add_argument("--check",
                    action="store_true",
                    help="gate fused vs unfused numerics before timing")
    args = ap.parse_args()

    s = Shapes(tokens=args.tokens)
    lo, hi = (int(v) for v in args.depths.split(","))
    stages = [st for st in STAGE_ORDER if st in args.stages.split(",")]
    nd_pre = args.devices or len(jax.devices())
    do_tail_mesh = "tail" in stages and nd_pre > 1
    if do_tail_mesh:
        # The meshed tail below supersedes the single-device one, which would
        # otherwise be double-counted and drag the calibration down.
        stages = [st for st in stages if st != "tail"]
    # The full stage only exists meshed (its tail is a collective).
    do_full_mesh = "full" in stages and nd_pre > 1
    if "full" in stages:
        stages = [st for st in stages if st != "full"]
        if not do_full_mesh:
            print("  full        needs --devices > 1 (collective); skipped")
    key = jax.random.key(0)
    x = jax.random.normal(key, (s.tokens, s.hidden), jnp.bfloat16)
    ones = jnp.ones((1, s.hidden), jnp.bfloat16)

    print(f"device {jax.devices()[0].device_kind}  tokens={s.tokens}  "
          f"depths {lo}->{hi}  reps={args.reps}")

    if args.check:
        print("\n--- numerics (fused vs unfused reference) ---")

        def rel(g, r):
            return (float(
                jnp.max(
                    jnp.abs(g.astype(jnp.float32) - r.astype(jnp.float32)) /
                    jnp.maximum(jnp.abs(r.astype(jnp.float32)), 1e-2))))

        for st in stages:
            if st not in FUSED_AVAILABLE:
                print(f"  {st:11} no fused impl yet")
                continue
            w = make_weights(key, s, 1, st, args.bn1, args.bn2, args.bnd)[0]
            if st == "sharedmlp":
                r = rel(
                    fused_sharedmlp(x, w[2], w[3], ones, s, args.bn1, args.bn2,
                                    args.ahead1, args.ahead2),
                    ref_sharedmlp(x, w[0], w[1], s))
                print(f"  {st:11} max_rel={r:.3e}  "
                      f"{'OK' if r < 5e-3 else 'FAIL'}")
            elif st == "mega":
                rwt, rids = ref_gate(x, w[0], s)
                rlat = ref_downproj(x, w[1], s)
                rsh = ref_sharedmlp(x, w[2], w[3], s)
                rexp = ref_experts(rlat, rwt, rids, w[4], w[5], w[6], w[7], s)
                fwt, fids, flat, fsh, fexp = fused_mega(
                    x, w[8], w[9], w[10], w[11], w[4], w[5], w[6], w[7], ones,
                    s, args.bnd, args.bn1, args.bn2, args.aheadd, args.ahead1,
                    args.ahead2, args.aheade)
                idm = float(jnp.mean(fids == rids))

                # topk_w and shared are bit-exact in every config. latent is
                # bit-exact only at bn_d=256: wider down_proj blocks make
                # Mosaic lower the [T,7168]x[7168,1792] dot with a different
                # K-accumulation order than XLA's (5.8e-3 elementwise at
                # bn_d=1792, concentrated in one token row -- a Mosaic quirk).
                # expert_out inherits that through its input, and the situ
                # nonlinearity inflates near-zero elements, so latent and
                # experts are gated NORM-relative (max|err| / max|ref|);
                # measured: 7e-4 / 7e-3 at the tuned config (the expert ref
                # norm is a masked sum over few active experts, so it is
                # smaller and the ratio inflates ~10x vs latent's).
                def reln(g, r):
                    return (float(
                        jnp.max(
                            jnp.abs(
                                g.astype(jnp.float32) - r.astype(jnp.float32)))
                        / jnp.max(jnp.abs(r.astype(jnp.float32)))))

                r_wt, r_sh = rel(fwt, rwt), rel(fsh, rsh)
                r_lat, r_exp = reln(flat, rlat), reln(fexp, rexp)
                ok = (r_wt < 5e-3 and r_sh < 5e-3 and r_lat < 5e-3
                      and r_exp < 1e-2 and idm == 1.0)
                print(
                    f"  {st:11} ids_match={idm:.4f}  {'OK' if ok else 'FAIL'}")
                print(
                    f"             topk_w={r_wt:.3e} latent={r_lat:.3e}(norm) "
                    f"shared={r_sh:.3e} experts={r_exp:.3e}(norm)")

    print(f"\n{'stage':11} {'unfused us':>13} {'fused us':>13} {'delta':>9} "
          f"{'model us':>9}  note")
    tot_u = tot_f = 0.0
    measured = {}
    measured_tail = {}
    for st in stages:
        try:
            ws_lo = make_weights(key, s, lo, st, args.bn1, args.bn2, args.bnd)
            ws_hi = make_weights(key, s, hi, st, args.bn1, args.bn2, args.bnd)
        except NotImplementedError:
            print(f"{st:11} {'-':>11} {'-':>10} {'-':>9} "
                  f"{MODEL_US.get(st, 0):9.2f}  no reference yet")
            continue

        def pick(d):
            return (x, ws_hi[:d] if d == hi else ws_lo[:d], ones)

        cu = stage_chain(st, s, False, args.bn1, args.bn2, args.ahead1,
                         args.ahead2, args.bnd, args.aheadd, args.aheade)
        u, ue = repeated(lambda: slope_time(cu, pick, lo, hi, args.reps) * 1e6,
                         args.trials)
        tot_u += u
        measured[st] = u
        if st in FUSED_AVAILABLE:
            cf = stage_chain(st, s, True, args.bn1, args.bn2, args.ahead1,
                             args.ahead2, args.bnd, args.aheadd, args.aheade)
            f, fe = repeated(
                lambda: slope_time(cf, pick, lo, hi, args.reps) * 1e6,
                args.trials)
            tot_f += f
            # Only call a win when it clears the combined spread.
            note = ("fused" if f + fe < u - ue else "FUSED SLOWER" if f -
                    fe > u + ue else "WITHIN NOISE")
            print(f"{st:11} {u:8.2f}+-{ue:.2f} {f:7.2f}+-{fe:.2f} "
                  f"{f-u:+9.2f} {MODEL_US.get(st, 0):9.2f}  {note}")
        else:
            tot_f += u
            print(f"{st:11} {u:8.2f}+-{ue:.2f} {'-':>13} {'-':>9} "
                  f"{MODEL_US.get(st, 0):9.2f}  no kernel yet")
    # --- meshed tail: real collective, both arms -----------------------------
    nd = nd_pre
    if do_tail_mesh:
        import jax.sharding as jsh
        (mesh, rep, AXIS, k_pad, cols, unfused, fused, blockify_w,
         mk_ws) = tail_arms(nd, s, args.tail_bn, args.tail_ahead)
        pspec = (jsh.PartitionSpec(), )
        flat = jax.random.normal(key, (s.tokens, k_pad), jnp.bfloat16)
        flat = flat.at[:, s.latent:].set(0)
        x3 = rep(flat.reshape(s.tokens, nd, cols).transpose(1, 0, 2))
        g = rep(
            jax.random.normal(jax.random.fold_in(key, 7), (1, s.latent),
                              jnp.bfloat16))
        raw = [
            jax.random.normal(jax.random.fold_in(key, 900 + i),
                              (s.up_shard, s.latent), jnp.bfloat16)
            for i in range(hi)
        ]
        w_nt = [rep(w) for w in raw]  # [n_out, latent]
        w_bl = [rep(blockify_w(w.T, args.tail_bn)) for w in raw]
        # The unfused arm's collective takes a 2-D payload with rows divisible
        # by the device count; [tokens*nd, latent/nd] carries the same
        # tokens x latent elements. The fused v4 arm takes [nd, tokens, cols].
        ks_rows, ks_cols = s.tokens * nd, s.latent // nd
        ks_ws = tuple(
            rep(t) for t in mk_ws((ks_rows, ks_cols), jnp.bfloat16, nd))

        def v4_ws():
            return (rep(jnp.zeros((2, nd - 1, s.tokens, cols), jnp.bfloat16)),
                    rep(jnp.zeros((2, s.tokens, k_pad), jnp.bfloat16)),
                    rep(jnp.zeros((1, ), jnp.int32)))

        x_u = rep(flat[:, :s.latent].reshape(ks_rows, ks_cols))

        def make_arm(fn, xx, ws_all, extra):

            def make(depth):
                e = extra if extra is not None else v4_ws()
                a = (xx, g, *e, *ws_all[:depth])
                jf = jax.jit(
                    jax.shard_map(fn,
                                  mesh=mesh,
                                  in_specs=pspec * len(a),
                                  out_specs=jsh.PartitionSpec(),
                                  check_vma=False))
                return jf, a

            return make

        for label, fn, xx, ws_all, extra in (("tail-unfused", unfused, x_u,
                                              w_nt, ks_ws),
                                             ("tail-fused-v4", fused, x3, w_bl,
                                              None)):
            m = make_arm(fn, xx, ws_all, extra)
            measured_tail[label] = repeated(
                lambda _m=m: slope_time_pairs(_m, lo, hi, args.reps) * 1e6,
                args.trials)
        (u, ue), (fu, fe) = (measured_tail["tail-unfused"],
                             measured_tail["tail-fused-v4"])
        print(
            f"{'tail(mesh)':11} {u:8.2f}+-{ue:.2f} {fu:7.2f}+-{fe:.2f} "
            f"{fu-u:+9.2f} {MODEL_US['tail']:9.2f}  {nd} dev, SKEW NOT REPRODUCED"
        )
        measured["tail"] = u
        tot_u += u
        tot_f += fu

    # --- meshed full: the whole MoE block in one kernel, both arms ----------
    if do_full_mesh:
        import jax.sharding as jsh
        from jax.sharding import Mesh, NamedSharding
        from jax.sharding import PartitionSpec as P

        from vllm_torchtpu.kernels.collectives.low_latency_all_reduce import (
            make_two_shot_workspace, two_shot_all_reduce_ws)
        from vllm_torchtpu.kernels.kimi_k3.mega_moe import \
            blockify as mega_blockify
        from vllm_torchtpu.kernels.kimi_k3.mega_moe import mega_moe

        AXIS = "dev"
        mesh = Mesh(jax.devices()[:nd], (AXIS, ))

        def rep(t):
            return jax.device_put(t, NamedSharding(mesh, P()))

        pspec = (P(), )
        xr = rep(x)
        g1r = rep(ones)
        g2h = jax.random.normal(jax.random.fold_in(key, 77), (1, s.latent),
                                jnp.bfloat16)
        g2 = rep(g2h)
        ks_rows, ks_cols = s.tokens * nd, s.latent // nd

        NWF = 14  # arrays per chain step; indices used by both arms
        WS = {}  # raw host copies for the numerics reference
        ws_full = []
        for i in range(hi):
            w = make_weights(jax.random.fold_in(key, 1000 + i), s, 1, "mega",
                             args.bn1, args.bn2, args.bnd)[0]
            wup = jax.random.normal(jax.random.fold_in(key, 700 + i),
                                    (s.up_shard, s.latent), jnp.bfloat16)
            if i == 0:
                WS = dict(w=w, wup=wup)
            ws_full.append(
                [rep(t) for t in w] +
                [rep(wup), rep(mega_blockify(wup.T, args.bnup))])

        ks_ws = tuple(
            rep(t)
            for t in make_two_shot_workspace((ks_rows,
                                              ks_cols), jnp.bfloat16, nd))

        def full_ws():
            return (rep(jnp.zeros((2, nd, s.tokens, s.latent), jnp.bfloat16)),
                    rep(jnp.zeros((1, ), jnp.int32)))

        def pad(v):
            return jnp.pad(v, ((0, 0), (0, s.hidden - v.shape[1])))

        def full_unfused(x, gg1, gg2, rs, ag, st, *ws):
            del gg1
            for i in range(len(ws) // NWF):
                wg, wd, gu, dn = ws[NWF * i:NWF * i + 4]
                w13q, w13s, w2q, w2s = ws[NWF * i + 4:NWF * i + 8]
                wup = ws[NWF * i + 12]
                wt, ids = ref_gate(x, wg, s)
                lat = ref_downproj(x, wd, s)
                sh = ref_sharedmlp(x, gu, dn, s)
                exp = ref_experts(lat, wt, ids, w13q, w13s, w2q, w2s, s)
                red, rs, ag, st = two_shot_all_reduce_ws(
                    exp.reshape(ks_rows, ks_cols),
                    rs,
                    ag,
                    st,
                    axis_name=AXIS,
                    num_devices=nd,
                    collective_id=3,
                    has_side_effects=False)
                y = ref_tail(red.reshape(s.tokens, s.latent), gg2, wup, s)
                tie = (jnp.sum(wt) + jnp.sum(ids).astype(jnp.float32) +
                       jnp.sum(lat)) * 1e-30
                x = sh + pad(y) + tie.astype(x.dtype)
            return x, rs, ag, st

        def full_fused(x, gg1, gg2, redws, st, *ws):
            for i in range(len(ws) // NWF):
                w13q, w13s, w2q, w2s = ws[NWF * i + 4:NWF * i + 8]
                wgp, wdb, gub, dnb = ws[NWF * i + 8:NWF * i + 12]
                wupb = ws[NWF * i + 13]
                tw, ti, lat, sh, exp, tl, redws, st = mega_moe(
                    x,
                    wgp,
                    wdb,
                    gub,
                    dnb,
                    w13q,
                    w13s,
                    w2q,
                    w2s,
                    gg1,
                    n_experts=s.n_experts_global,
                    top_k=s.top_k,
                    bn_d=args.bnd,
                    bn1=args.bn1,
                    bn2=args.bn2,
                    ahead_d=args.aheadd,
                    ahead1=args.ahead1,
                    ahead2=args.ahead2,
                    ahead_e=args.aheade,
                    do_norm=False,
                    beta=s.beta,
                    linear_beta=s.linear_beta,
                    tail=dict(red_ws=redws,
                              step=st,
                              w_up=wupb,
                              gamma2=gg2,
                              axis_name=AXIS,
                              num_devices=nd,
                              bn_up=args.bnup,
                              ahead_up=args.aheadup,
                              collective_id=11,
                              union_up=args.union),
                    vmem_limit_bytes=110 * 1024 * 1024)
                tie = (jnp.sum(tw) + jnp.sum(ti).astype(jnp.float32) +
                       jnp.sum(lat) + jnp.sum(exp)) * 1e-30
                x = sh + pad(tl) + tie.astype(x.dtype)
            return x, redws, st

        if args.check:
            w, wup = WS["w"], WS["wup"]
            rwt, rids = ref_gate(x, w[0], s)
            rlat = ref_downproj(x, w[1], s)
            rexp = ref_experts(rlat, rwt, rids, w[4], w[5], w[6], w[7], s)
            # replicated inputs -> the all-reduce sums nd identical partials
            rtl = ref_tail((nd * rexp.astype(jnp.float32)).astype(x.dtype),
                           g2h, wup, s)

            def check_call(xx, rw, stp):
                return mega_moe(xx,
                                w[8],
                                w[9],
                                w[10],
                                w[11],
                                w[4],
                                w[5],
                                w[6],
                                w[7],
                                g1r,
                                n_experts=s.n_experts_global,
                                top_k=s.top_k,
                                bn_d=args.bnd,
                                bn1=args.bn1,
                                bn2=args.bn2,
                                ahead_d=args.aheadd,
                                ahead1=args.ahead1,
                                ahead2=args.ahead2,
                                ahead_e=args.aheade,
                                do_norm=False,
                                beta=s.beta,
                                linear_beta=s.linear_beta,
                                tail=dict(red_ws=rw,
                                          step=stp,
                                          w_up=rep(
                                              mega_blockify(wup.T, args.bnup)),
                                          gamma2=g2,
                                          axis_name=AXIS,
                                          num_devices=nd,
                                          bn_up=args.bnup,
                                          ahead_up=args.aheadup,
                                          collective_id=11,
                                          union_up=args.union),
                                vmem_limit_bytes=110 * 1024 * 1024)

            jf = jax.jit(
                jax.shard_map(check_call,
                              mesh=mesh,
                              in_specs=pspec * 3,
                              out_specs=(P(), ) * 8,
                              check_vma=False))
            out = jf(xr, *full_ws())

            def reln(g, r):
                return (float(
                    jnp.max(
                        jnp.abs(
                            jax.device_get(g).astype(jnp.float32) -
                            r.astype(jnp.float32))) /
                    jnp.max(jnp.abs(r.astype(jnp.float32)))))

            r_exp, r_tl = reln(out[4], rexp), reln(out[5], rtl)
            ok = r_exp < 1e-2 and r_tl < 1e-2
            print(f"  {'full':11} experts={r_exp:.3e}(norm) "
                  f"tail={r_tl:.3e}(norm)  {'OK' if ok else 'FAIL'}")

        def make_full(fn, n_extra, extra_fn):

            def make(depth):
                flat = [a for i in range(depth) for a in ws_full[i]]
                a = (xr, g1r, g2, *extra_fn(), *flat)
                n_out = 1 + n_extra
                jf = jax.jit(
                    jax.shard_map(fn,
                                  mesh=mesh,
                                  in_specs=pspec * len(a),
                                  out_specs=(P(), ) * n_out,
                                  check_vma=False))
                return jf, a

            return make

        measured_full = {}
        for label, fn, n_extra, extra_fn in (("full-unfused", full_unfused, 3,
                                              lambda: ks_ws),
                                             ("full-fused", full_fused, 2,
                                              full_ws)):
            m = make_full(fn, n_extra, extra_fn)
            measured_full[label] = repeated(
                lambda _m=m: slope_time_pairs(_m, lo, hi, args.reps) * 1e6,
                args.trials)
        (u, ue), (fu, fe) = (measured_full["full-unfused"],
                             measured_full["full-fused"])
        print(
            f"{'full(mesh)':11} {u:8.2f}+-{ue:.2f} {fu:7.2f}+-{fe:.2f} "
            f"{fu-u:+9.2f} {MODEL_US['full']:9.2f}  {nd} dev, SKEW NOT REPRODUCED"
        )
        measured["full"] = u
        tot_u += u
        tot_f += fu

    # include the meshed tail/full, which are filtered out of `stages`
    model_tot = sum(MODEL_US[st] for st in stages if st in MODEL_US)
    if do_tail_mesh:
        model_tot += MODEL_US['tail']
    if do_full_mesh:
        model_tot += MODEL_US['full']
    print(f"{'TOTAL':11} {tot_u:11.2f} {tot_f:10.2f} {tot_f-tot_u:+9.2f} "
          f"{model_tot:9.2f}")

    # Calibration. Per-stage, because an aggregate ratio hides exactly the
    # stage that is misbehaving -- and that stage is the interesting one.
    print("\n--- calibration: harness / model, unfused path ---")
    cal_h = cal_m = 0.0
    for st, u in measured.items():
        if st in INCOMPLETE_REF_1DEV and (args.devices == 1
                                          or st in ("mega", "full")):
            print(f"  {st:11} {u:7.2f} / {MODEL_US[st]:6.2f}   excluded: "
                  f"{INCOMPLETE_REF_1DEV[st]}")
            continue
        cal_h += u
        cal_m += MODEL_US[st]
        r = u / MODEL_US[st]
        flag = "" if 0.85 <= r <= 1.15 else "   <-- harness does not match model here"
        print(f"  {st:11} {u:7.2f} / {MODEL_US[st]:6.2f} = {r:4.2f}x{flag}")
    if cal_m:
        print(f"  {'comparable':11} {cal_h:7.2f} / {cal_m:6.2f} = "
              f"{cal_h/cal_m:4.2f}x")
    print(
        "\nA ratio near 1.0 means the harness reproduces the model for that "
        "stage.\nA ratio well below 1.0 means the model is SLOWER than a clean "
        "run of the\nsame ops -- the fused kernel's apparent win there is "
        "against the model's\ndegradation, not against XLA's capability. "
        "Check that distinction before\nbelieving any speedup.")


if __name__ == "__main__":
    main()
