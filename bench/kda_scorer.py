#!/usr/bin/env python3
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
"""Multi-arm microbenchmark for gdn/v3 KDA at Kimi-Linear-48B shapes.

Measures several *source trees* of the same kernel side by side in one job --
one of them the golden reference, marked `control` -- and prints one JSON
object on stdout. Only the ratio to the control means anything: absolute TPU
microseconds move with clocks, thermals and node neighbours, so every number
here is reported beside a golden arm measured on the same clock.

What it measures
----------------
`vllm_torchtpu.kernels.gdn.v3.wrapper.fused_conv1d_gdn` with
`attention_mode=KDA`, called the way `layers/adapter/custom_ops/
kda_attention_op.py:320-345` calls it for Kimi-Linear-48B-A3B-Instruct at
tensor-parallel size 8: four heads a rank, head dim 128, a bias-free conv of
width 4, the unbounded softplus gate (`gate_lower_bound=None`), bf16
activations, fp32 recurrent state, and the kernel's own default tiling. The
tiling is deliberately NOT passed in: `mixed_tile_size` / `decode_tile_size`
are part of what a candidate may change, and passing them here would hide it.

The serving stack stays out. Importing `vllm_torchtpu` pulls in neither vllm
nor torch; what it does pull in is the Mosaic lowering patch in
`env_override.py`, which is active in production and therefore has to be active
here too.

Correctness oracle
------------------
A float64 token-at-a-time recurrence, preceded by the conv1d and silu the
kernel folds in. Copied -- not imported -- from the kernel's own test suite so
that the oracle does not live in the tree the candidates edit; the source
revision is pinned in REFERENCE_FROM and drift from it is checkable by reading
the two side by side.

Usage
-----
    python3 bench/kda_scorer.py --job-id JOB \
        --arm golden=/arms/golden:control --arm golden-aa=/arms/aa \
        --arm cand=/arms/cand \
        --config prefill1024 --config decode1 --rounds 3

Each arm is measured in its own process -- tracing two Pallas kernels in one
process trips XLA's memory_space_assignment -- and each process imports the
kernel from its own tree and refuses to run if it resolved to another one. The
whole set is measured `--rounds` times in a rotated order, and each arm's time
is the median of its rounds.

stdout is exactly one JSON value. Everything human-readable goes to stderr.
There is no exit-code channel: an arm, a config or a whole job this file cannot
vouch for is one it does not print.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np

#: Where the oracle was copied from. A path and a revision, so that drift from
#: the arithmetic the kernel's owners use is checkable rather than invisible.
REFERENCE_FROM = ("tests/kernels/test_gdn_attention_v3_kda.py @ ec47eea7 "
                  "(vllm-project/vllm-torchtpu)")

#: Kimi-Linear-48B-A3B-Instruct, one rank of a tensor-parallel-8 deployment.
#: 32 KDA heads over 8 ranks is 4 a rank; head_dim and the conv width are not
#: sharded. From the HF config (`linear_attn_config`) crossed with the TP split
#: in `models/vllm/kimi_k3/attention.py:372-382`.
N_HEADS, HEAD_DIM, CONV_WIDTH = 4, 128, 4
DIM = 3 * N_HEADS * HEAD_DIM  # q, k and v share one fused activation row.

#: The shapes the named workload actually runs, and nothing else. A 1024-token
#: prompt at max_num_batched_tokens=4096 is one whole-prompt prefill chunk; at
#: max_concurrency=1 every decode step is a single sequence of one token.
CONFIGS = {
    "prefill1024": {"tokens": 1024, "seqs": 1, "decode": False, "ctx": 0},
    "decode1": {"tokens": 1, "seqs": 1, "decode": True, "ctx": 1024},
}

#: Chained calls per burst, per config: enough that one burst is milliseconds
#: of device work and the host round trip behind it rounds away. The calls are
#: chained inside one jit -- each iteration's state feeds the next, exactly as
#: the decode loop does.
CHAIN = {"prefill1024": 32, "decode1": 128}
WARMUP, BURSTS = 2, 7

#: Rounds of the whole arm set. Every arm is measured once per round, in a
#: rotated order, and its score is the median across rounds.
#:
#: One process per arm is forced (two Pallas kernels traced in one process trip
#: XLA's memory_space_assignment), and the first version of this file measured
#: each arm in exactly one process, back to back. That reads whatever the
#: machine was doing during that process as a property of the arm: two copies
#: of golden came out 24% apart and an arm whose kernel is bit-identical to
#: golden's on a config came out 2.2x faster there. Interleaving and taking the
#: median is what makes the ratio a fact about the code.
ROUNDS = 3

#: One seed for every arm. Two arms on different inputs are measuring different
#: work, and the difference reads as a speedup.
SEED = 20260915


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def scorer_hash() -> str:
    with open(os.path.abspath(__file__), "rb") as fh:
        return _sha(fh.read())


# ---------------------------------------------------------------------------
# the arm's own source
# ---------------------------------------------------------------------------

def load_kernel(src: str):
    """Import the kernel from THIS arm's tree, or refuse to measure anything.

    The image carries an editable install of `vllm_torchtpu`, which resolves to
    one fixed tree whatever path you meant -- so without the check below every
    arm can silently measure golden, and a whole campaign of ratios is 1.0 with
    nothing to say so.
    """
    root = os.path.join(os.path.abspath(src), "src")
    if not os.path.isdir(os.path.join(root, "vllm_torchtpu")):
        raise SystemExit(f"arm source {root} has no vllm_torchtpu package")
    # An editable install installs a meta-path finder that outranks sys.path.
    sys.meta_path[:] = [
        f for f in sys.meta_path
        if not type(f).__module__.startswith("__editable__")
    ]
    for name in [m for m in sys.modules if m.split(".")[0] == "vllm_torchtpu"]:
        del sys.modules[name]
    sys.path.insert(0, root)

    from vllm_torchtpu.kernels.gdn.v3 import compute_kda, config, wrapper

    for mod in (wrapper, config, compute_kda):
        got = os.path.realpath(mod.__file__)
        if not got.startswith(os.path.realpath(root) + os.sep):
            raise SystemExit(
                f"arm was told to measure {root} but imported {mod.__name__} "
                f"from {got}")
    import jax
    _log(f"arm imports {os.path.realpath(compute_kda.__file__)} "
         f"(jax {jax.__version__}, {jax.devices()[0].device_kind})")
    return wrapper, config


def tree_hash(src: str) -> str:
    """The git tree hash of the arm's source, computed here, inside the job.

    Not a claim the caller makes about which code this is: `git add -A` over
    the arm's own directory, with a scratch index so nothing else is disturbed.
    Two arms built from the same source get the same value, which is what an
    A/A pair is; an arm that did not get the patch it was supposed to get
    reports golden's tree and says so.
    """
    src = os.path.abspath(src)
    # Both the index and the object store live outside the arm, or `git add -A`
    # would hash the scratch repository into the tree it is measuring.
    scratch = os.path.join(
        "/tmp", "kda-tree-" + hashlib.sha256(src.encode()).hexdigest()[:16])
    if os.path.isdir(scratch):
        shutil.rmtree(scratch)
    env = dict(os.environ, GIT_INDEX_FILE=os.path.join(scratch, "index"),
               GIT_DIR=os.path.join(scratch, "git"), GIT_WORK_TREE=src)
    try:
        subprocess.run(["git", "init", "-q", "--bare", env["GIT_DIR"]],
                       check=True, capture_output=True)
        subprocess.run(["git", "add", "-A", "--", "."], cwd=src, env=env,
                       check=True, capture_output=True)
        out = subprocess.run(["git", "write-tree"], cwd=src, env=env,
                             check=True, capture_output=True, text=True)
        return out.stdout.strip()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# ---------------------------------------------------------------------------
# inputs -- identical across arms by construction
# ---------------------------------------------------------------------------

def make_inputs(name: str):
    """Seeded numpy draws for one frozen config, in the production dtypes."""
    import jax.numpy as jnp

    cfg = CONFIGS[name]
    tokens, seqs, ctx = cfg["tokens"], cfg["seqs"], cfg["ctx"]
    rng = np.random.default_rng(SEED + sum(map(ord, name)))
    per_seq = tokens // seqs

    def randn(*shape):
        return rng.standard_normal(shape, dtype=np.float32)

    slots = seqs + 1  # slot 0 is the reserved null block for padded tokens.
    raw = {
        "qkv": randn(tokens, DIM),
        "b": randn(tokens, N_HEADS),
        # The KDA gate is per (head, channel), so it is n_v * d_k wide.
        "a": randn(tokens, N_HEADS * HEAD_DIM),
        "conv_weight": randn(DIM, 1, CONV_WIDTH) * 0.1,
        # The gate weights are drawn in the LONG-MEMORY regime -- `dt_bias`
        # offset so that `softplus(a + dt_bias)` is small and the per-token
        # decay sits near 1. This is a deliberate property of the fitness
        # function, not an accident of the seed: under a standard normal draw
        # the gate forgets in about two tokens, every cross-block term of the
        # chunked solve decays to zero before it is used, and the benchmark
        # then cannot tell a kernel that computes them from one that drops
        # them. The wrong-kernel arm is what proves the difference is visible.
        "a_log": randn(N_HEADS) * 0.2,
        "dt_bias": randn(N_HEADS, HEAD_DIM) * 0.3 - 4.0,
    }
    # A decode step reads a state a thousand tokens deep; a prefill chunk that
    # starts a sequence does not read one at all (`seq_lens == query_len`).
    scale = 0.05 if cfg["decode"] else 0.0
    raw["conv_state"] = randn(slots, CONV_WIDTH - 1, DIM) * scale
    raw["recurrent_state"] = randn(slots, N_HEADS, HEAD_DIM, HEAD_DIM) * scale

    q_loc = np.arange(seqs + 1, dtype=np.int32) * per_seq
    meta = {
        "query_start_loc": q_loc,
        "state_indices": np.arange(1, seqs + 1, dtype=np.int32),
        # [decode_end, prefill_end, mixed_end]
        "distribution": np.array([seqs if cfg["decode"] else 0, seqs, seqs],
                                 np.int32),
        "seq_lens": np.full((seqs, ), per_seq + ctx, np.int32),
    }
    # bf16 activations and conv state, fp32 recurrent state and gate weights:
    # the dtypes `kda_attention_op` hands the kernel on the pooled path.
    dt = {
        "qkv": jnp.bfloat16, "b": jnp.bfloat16, "a": jnp.bfloat16,
        "conv_weight": jnp.bfloat16, "conv_state": jnp.bfloat16,
        "recurrent_state": jnp.float32, "a_log": jnp.float32,
        "dt_bias": jnp.float32,
    }
    return raw, meta, dt


def device_args(raw, meta, dt):
    import jax.numpy as jnp
    args = {k: jnp.asarray(v, dt[k]) for k, v in raw.items()}
    args.update({k: jnp.asarray(v) for k, v in meta.items()})
    return args


# ---------------------------------------------------------------------------
# the correctness oracle, copied -- see REFERENCE_FROM
# ---------------------------------------------------------------------------

def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """Mirror the kernel's `_l2_norm_f32`, which accumulates in float32."""
    x32 = x.astype(np.float32)
    inv = 1.0 / np.sqrt(np.sum(x32 * x32, axis=-1, keepdims=True) + 1e-6)
    return (x32 * inv).astype(np.float64)


def reference(raw, meta) -> tuple[np.ndarray, np.ndarray]:
    """conv1d + silu + the KDA recurrence, one token at a time, in float64.

    Returns the per-token output `[T, H*V]` and the final recurrent state
    `[slots, H, K, V]`.
    """
    qkv = raw["qkv"].astype(np.float64)
    conv_w = raw["conv_weight"].astype(np.float64)
    conv_state = raw["conv_state"].astype(np.float64).copy()
    state = raw["recurrent_state"].astype(np.float64).copy()
    a_log = raw["a_log"].astype(np.float64).reshape(N_HEADS, 1)
    dt_bias = raw["dt_bias"].astype(np.float64).reshape(N_HEADS, HEAD_DIM)
    q_loc, seq_lens = meta["query_start_loc"], meta["seq_lens"]
    slot_of = meta["state_indices"]
    num_seqs = int(meta["distribution"][2])
    scale = float(HEAD_DIM**-0.5)

    out = np.zeros((qkv.shape[0], N_HEADS * HEAD_DIM), np.float64)
    for s in range(num_seqs):
        slot = int(slot_of[s])
        start, end = int(q_loc[s]), int(q_loc[s + 1])
        qlen = end - start
        if qlen <= 0:
            continue
        fresh = (int(seq_lens[s]) - qlen) <= 0
        c_state = (np.zeros_like(conv_state[slot]) if fresh
                   else conv_state[slot])
        if fresh:
            state[slot] = 0.0

        # conv1d over [state tail | this chunk], then silu. Bias-free, as
        # Kimi's short convolution is.
        x = np.concatenate([c_state, qkv[start:end]], axis=0)
        acc = np.zeros((qlen, DIM), np.float64)
        for k in range(CONV_WIDTH):
            acc += x[k:k + qlen] * conv_w[:, 0, k][None, :]
        acc = acc / (1.0 + np.exp(-acc))
        conv_state[slot] = x[-(CONV_WIDTH - 1):]

        key = N_HEADS * HEAD_DIM
        q = _l2_normalize(acc[:, :key].reshape(qlen, N_HEADS, HEAD_DIM))
        k_ = _l2_normalize(
            acc[:, key:2 * key].reshape(qlen, N_HEADS, HEAD_DIM))
        v = acc[:, 2 * key:].reshape(qlen, N_HEADS, HEAD_DIM)
        beta = 1.0 / (1.0 + np.exp(-raw["b"][start:end].astype(np.float64)))
        gate_in = raw["a"][start:end].astype(np.float64).reshape(
            qlen, N_HEADS, HEAD_DIM)
        # Unbounded gate: -exp(A_log) * softplus(a + dt_bias). Kimi-Linear-48B
        # sets no gate_lower_bound.
        gate = -np.exp(a_log) * np.logaddexp(0.0, gate_in + dt_bias)

        running = state[slot]
        for t in range(qlen):
            running = running * np.exp(gate[t])[:, :, None]
            pred = np.einsum("hk,hkv->hv", k_[t], running)
            delta = beta[t][:, None] * (v[t] - pred)
            running = running + k_[t][:, :, None] * delta[:, None, :]
            out[start + t] = (scale * np.einsum("hk,hkv->hv", q[t],
                                                running)).reshape(-1)
        state[slot] = running
    return out, state


def worst_sequence_error(got_out, got_state, ref_out, ref_state,
                         meta) -> float:
    """The worst SEQUENCE's relative L2, over the output and the state it left.

    Not the batch mean: a mean divides the one sequence that is wrong by every
    other sequence's size. The state counts because a decode step's whole
    contribution to the next step is the state it wrote.
    """
    def rel(a, b):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        denom = np.linalg.norm(b)
        return float(np.linalg.norm(a - b) / (denom if denom else 1.0))

    q_loc, slot_of = meta["query_start_loc"], meta["state_indices"]
    worst = 0.0
    for s in range(int(meta["distribution"][2])):
        start, end, slot = int(q_loc[s]), int(q_loc[s + 1]), int(slot_of[s])
        worst = max(worst, rel(got_out[start:end], ref_out[start:end]),
                    rel(got_state[slot], ref_state[slot]))
    return worst


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------

def _call_once(wrapper, cfgmod, args):
    return wrapper.fused_conv1d_gdn(
        qkv=args["qkv"], b=args["b"], a=args["a"],
        conv_state=args["conv_state"],
        recurrent_state=args["recurrent_state"],
        conv_weight=args["conv_weight"], conv_bias=None,
        a_log=args["a_log"], dt_bias=args["dt_bias"],
        query_start_loc=args["query_start_loc"],
        state_indices=args["state_indices"],
        distribution=args["distribution"], seq_lens=args["seq_lens"],
        n_kq=N_HEADS, n_v=N_HEADS, d_k=HEAD_DIM, d_v=HEAD_DIM,
        kernel_size=CONV_WIDTH,
        attention_mode=cfgmod.AttentionMode.KDA,
        gate_lower_bound=None)


def _persistent_compile_cache() -> None:
    """Keep compiled programs across the job's processes.

    Every arm is a process and every round re-runs every arm, so without this
    the loop spends its time in Mosaic rather than on the device. Golden's
    program is identical every round by construction, which is what a cache is
    for; two arms with different source have different keys and cannot share an
    entry.
    """
    import jax
    jax.config.update("jax_compilation_cache_dir", "/tmp/kda-jaxcache")
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)


def _chained(wrapper, cfgmod, chain_len: int):
    """chain_len calls under one jit, each fed the state the last one wrote.

    The shape of the real loop, and the reason the number below is device time
    rather than host dispatch: one round trip covers chain_len kernel launches.
    """
    import jax

    def chain(qkv, b, a, conv_state, recurrent_state, conv_weight, a_log,
              dt_bias, query_start_loc, state_indices, distribution, seq_lens):
        args = dict(qkv=qkv, b=b, a=a, conv_weight=conv_weight, a_log=a_log,
                    dt_bias=dt_bias, query_start_loc=query_start_loc,
                    state_indices=state_indices, distribution=distribution,
                    seq_lens=seq_lens)

        def body(_, carry):
            cs, rs = carry
            (cs2, rs2), _out = _call_once(
                wrapper, cfgmod,
                dict(args, conv_state=cs, recurrent_state=rs))
            return (cs2.astype(cs.dtype), rs2.astype(rs.dtype))

        return jax.lax.fori_loop(0, chain_len, body,
                                 (conv_state, recurrent_state))

    return jax.jit(chain)


def _dispatch_identity(hlo_text: str) -> str:
    """A digest of the custom-call payloads in the executable, in launch order.

    The Pallas kernels reach the executable as `tpu_custom_call` targets
    carrying their lowered Mosaic module, so this is the runtime's own answer
    to which kernels the program launched -- not a label the arm chose. Two
    arms that lower to the same set ran the same code whatever their source
    said, which is the claim this has to support.
    """
    lines = [ln.strip() for ln in hlo_text.splitlines()
             if "custom_call_target" in ln or "tpu_custom_call" in ln]
    if not lines:
        raise RuntimeError("no custom call in the compiled program: the "
                           "Pallas kernel did not reach the executable")
    digest = hashlib.sha256("\n".join(lines).encode()).hexdigest()[:32]
    return f"mosaic:{digest}"


def measure(wrapper, cfgmod, name: str) -> dict:
    """One frozen config on this arm: time, correctness, and what ran."""
    import jax

    raw, meta, dt = make_inputs(name)

    # Correctness first, from a single call on its own arrays: the kernel
    # donates its state buffers, so nothing here may be reused afterwards.
    args = device_args(raw, meta, dt)
    (_, new_state), out = _call_once(wrapper, cfgmod, args)
    got_out = np.asarray(jax.device_get(out), np.float64)
    got_state = np.asarray(jax.device_get(new_state), np.float64)
    ref_out, ref_state = reference(raw, meta)
    worst = worst_sequence_error(got_out, got_state, ref_out, ref_state, meta)

    chain_len = CHAIN[name]
    chain = _chained(wrapper, cfgmod, chain_len)
    order = ("qkv", "b", "a", "conv_state", "recurrent_state", "conv_weight",
             "a_log", "dt_bias", "query_start_loc", "state_indices",
             "distribution", "seq_lens")
    compiled = chain.lower(*[device_args(raw, meta, dt)[k]
                             for k in order]).compile()
    text = compiled.as_text()

    def burst() -> float:
        # Built AND LANDED before the clock starts. jnp.asarray is async, and
        # with several megabytes of activations a transfer that is still in
        # flight at t0 finishes inside the timed region -- which is most of
        # what the first version of this file was measuring.
        fresh = device_args(raw, meta, dt)
        jax.block_until_ready(list(fresh.values()))
        t0 = time.perf_counter()
        jax.block_until_ready(chain(*[fresh[k] for k in order]))
        return (time.perf_counter() - t0) * 1e6 / chain_len

    for _ in range(WARMUP):
        burst()
    samples = [burst() for _ in range(BURSTS)]
    latency = statistics.median(samples)
    _log(f"  {name}: {latency:.1f} us/call  (min {min(samples):.1f}, "
         f"max {max(samples):.1f}, n={BURSTS}x{chain_len})  "
         f"worst_error {worst:.3e}")

    return {
        "latency_us": float(latency),
        # The scorer's own verdict is deliberately generous: the campaign
        # re-checks worst_error against its own frozen tolerance, and a scorer
        # with a looser bar must not be able to pass what the campaign fails.
        "correct": bool(worst < 0.25),
        "worst_error": float(worst),
        "checksum": _sha(
            np.ascontiguousarray(got_out.astype(np.float32)).tobytes() +
            np.ascontiguousarray(got_state.astype(np.float32)).tobytes()),
        "dispatched": _dispatch_identity(text),
        "declined": False,
        "_binary": _sha(text.encode()),
    }


def run_worker(src: str, configs: list[str]) -> dict:
    # Hashed before anything imports from it: an import writes __pycache__ into
    # the tree it is hashing, and the answer has to be a fact about the source.
    arm = {"tree": tree_hash(src), "configs": {}}
    _persistent_compile_cache()
    wrapper, cfgmod = load_kernel(src)
    binaries = []
    for name in configs:
        try:
            got = measure(wrapper, cfgmod, name)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            # An omission is the only honest report of a config this process
            # could not measure: `declined` means the KERNEL refused the shape,
            # and saying that about a harness failure would be a clean,
            # confident, wrong answer.
            _log(f"  {name}: NOT MEASURED -- {type(exc).__name__}: {exc}")
            continue
        binaries.append(got.pop("_binary"))
        arm["configs"][name] = got
    if not binaries:
        raise SystemExit("no config produced a measurement")
    arm["binary_hash"] = _sha("|".join(binaries).encode())
    return arm


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------

def _measure_arm(name: str, src: str, configs: list[str]) -> dict | None:
    """One process, one arm, every config. None if it produced nothing."""
    argv = [sys.executable, "-u", os.path.abspath(__file__), "--worker",
            "--src", src]
    for cfg in configs:
        argv += ["--config", cfg]
    proc = subprocess.run(
        argv, capture_output=True, text=True,
        env=dict(os.environ, PYTHONPATH=os.path.join(src, "src"),
                 PYTHONDONTWRITEBYTECODE="1"))
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        _log(f"--- arm {name} produced nothing (exit {proc.returncode})")
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        _log(f"--- arm {name} printed no readable result ({exc})")
        return None


def _merge_rounds(name: str, rounds: list[dict], configs: list[str]):
    """One arm's rounds into one arm, timing by median and the rest by consent.

    Everything except the latency is a claim about the code -- the tree it was
    built from, the artifact that was loaded, the kernel that ran, the bytes it
    produced, its error against the reference. Those must be the same in every
    round, and an arm whose rounds disagree about one of them is an arm nobody
    can describe: it is dropped rather than averaged into a number that
    describes none of the rounds.
    """
    first = rounds[0]
    identity = ("tree", "binary_hash")
    for r in rounds[1:]:
        for key in identity:
            if r.get(key) != first.get(key):
                _log(f"--- arm {name} changed its {key} between rounds; "
                     f"it is left out of the payload")
                return None
    arm = {"tree": first["tree"], "binary_hash": first["binary_hash"],
           "configs": {}}
    for cfg in configs:
        seen = [r["configs"][cfg] for r in rounds if cfg in r["configs"]]
        if len(seen) != len(rounds):
            _log(f"--- arm {name}: config {cfg} was measured in "
                 f"{len(seen)}/{len(rounds)} rounds, so it is not reported")
            continue
        for key in ("checksum", "dispatched", "correct"):
            if len({json.dumps(s[key]) for s in seen}) != 1:
                _log(f"--- arm {name}: config {cfg} disagrees on {key} "
                     f"between rounds; it is not reported")
                break
        else:
            times = [s["latency_us"] for s in seen]
            arm["configs"][cfg] = dict(
                seen[0], latency_us=statistics.median(times),
                worst_error=max(s["worst_error"] for s in seen))
            _log(f"    {name} {cfg}: median {statistics.median(times):.1f} us "
                 f"of {[round(t, 1) for t in times]}")
    return arm if arm["configs"] else None


def run_driver(job_id: str, arms: list[tuple[str, str, bool]],
               configs: list[str], rounds: int) -> dict:
    collected: dict[str, list[dict]] = {name: [] for name, _, _ in arms}
    for r in range(rounds):
        # Rotated, so a machine that drifts over the job does not spend the
        # whole drift on whichever arm is always measured last.
        order = arms[r % len(arms):] + arms[:r % len(arms)]
        for name, src, control in order:
            _log(f"--- round {r + 1}/{rounds} arm {name} "
                 f"({'control' if control else 'candidate'}) from {src}")
            got = _measure_arm(name, src, configs)
            if got is not None:
                collected[name].append(got)

    out = {"scorer_hash": scorer_hash(), "job_id": job_id, "arms": []}
    for name, _, control in arms:
        if len(collected[name]) != rounds:
            _log(f"--- arm {name} produced {len(collected[name])}/{rounds} "
                 f"rounds; it is left out of the payload")
            continue
        arm = _merge_rounds(name, collected[name], configs)
        if arm is None:
            continue
        arm["name"], arm["control"] = name, control
        out["arms"].append(arm)
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--src")
    ap.add_argument("--job-id", default="")
    ap.add_argument("--arm", action="append", default=[],
                    help="NAME=PATH[:control]")
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    args = ap.parse_args(argv)
    configs = args.config or list(CONFIGS)
    for name in configs:
        if name not in CONFIGS:
            raise SystemExit(f"unknown config {name!r}; have {list(CONFIGS)}")

    if args.worker:
        json.dump(run_worker(args.src, configs), sys.stdout)
        return 0

    arms = []
    for spec in args.arm:
        name, _, rest = spec.partition("=")
        control = rest.endswith(":control")
        arms.append((name, rest[:-len(":control")] if control else rest,
                     control))
    if not arms:
        raise SystemExit("nothing to measure: pass --arm NAME=PATH[:control]")
    json.dump(run_driver(args.job_id, arms, configs, args.rounds),
              sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
