#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the phase 3 AgentX harness benches.

Used by p3_p_leg_bench.py and p3_d_leg_bench.py (offline vLLM engine benches)
and by the kernel benches under tests/kernels/test_p3_*.py. It provides:

  * environment_id(): image, libtpu, torch_tpu, jax, vLLM, source sha and
    dirty-tree hash, topology, job id. Every bench prints it and writes it to
    env.json in its run directory.
  * Timeline: absolute marks for the queue, init, load, compile, warmup and
    timed windows, reported separately.
  * run_dir(): one output directory per run, never shared between pods.
  * SITECUSTOMIZE: source of a sitecustomize.py that the engine processes load
    at interpreter start (they are spawned and inherit PYTHONPATH). It patches
    the dummy loader (synthetic weights that survive the fp8 cast), the fused
    EP MoE routing (hash or trace replay), the scheduler (one JSONL record per
    engine step with computed tokens, running count, spec tokens and sampled
    tokens) and the TPU model runner (load, compile and warmup phase markers).

Nothing here touches the TPU unless environment_id(devices=True) is asked for,
so the engine benches can call it in the parent process before the workers
spawn.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PLACEMENT_DIR = REPO_ROOT / "scripts/vllm/benchmark/inferencex/qwen3.5"
DEFAULT_OUT_ROOT = "/workspace/tmp/p3-harness/jobs"
MODEL_CONFIG_DIR = "/workspace/tmp/agentx-p2/qwen35-397b-config"


def job_id() -> str:
    return (
        os.environ.get("CDK_JOB_ID")
        or os.environ.get("JS_NAME")
        or f"local-{socket.gethostname()}"
    )


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True, text=True, timeout=60, check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def source_identity() -> dict:
    """Source sha and a hash of the uncommitted diff.

    A --map-dir snapshot or an export copy has no .git; the export step writes
    P3_SNAPSHOT.json next to pyproject.toml with the same keys, and P3_SOURCE_SHA
    in the environment overrides both.
    """
    out = {"source_sha": os.environ.get("P3_SOURCE_SHA", ""), "source_dirty_sha": ""}
    snap = REPO_ROOT / "P3_SNAPSHOT.json"
    if snap.exists():
        try:
            data = json.loads(snap.read_text())
            out["source_sha"] = out["source_sha"] or data.get("source_sha", "")
            out["source_dirty_sha"] = data.get("source_dirty_sha", "")
            out["snapshot"] = data.get("snapshot", "")
        except (OSError, ValueError):
            pass
    if not out["source_sha"] and (REPO_ROOT / ".git").exists():
        out["source_sha"] = _git("rev-parse", "HEAD")
        diff = _git("diff", "HEAD", "--", "src", "scripts", "tests")
        out["source_dirty_sha"] = hashlib.sha256(diff.encode()).hexdigest()[:12] if diff else ""
    out["source_sha"] = out["source_sha"] or "unknown"
    return out


def package_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for dist in ("torch-tpu", "libtpu", "jax", "jaxlib", "vllm", "vllm-torchtpu",
                 "torch", "tpu-sync-torch", "tpu-raiden-torch"):
        try:
            out[dist] = version(dist)
        except PackageNotFoundError:
            out[dist] = ""
    return out


def topology() -> dict:
    env = os.environ
    return {
        "tpu_topology": env.get("TPU_TOPOLOGY", ""),
        "chips_per_host_bounds": env.get("TPU_CHIPS_PER_HOST_BOUNDS", ""),
        "host_bounds": env.get("TPU_HOST_BOUNDS", ""),
        "process_bounds": env.get("TPU_PROCESS_BOUNDS", ""),
        "worker_id": env.get("TPU_WORKER_ID", ""),
        "num_hosts": env.get("P3_NUM_HOSTS", ""),
        "accelerator": env.get("P3_ACCELERATOR", "tpu7x"),
    }


def environment_id(devices: bool = False) -> dict:
    """The environment id every bench prints and records.

    devices=True adds jax.devices() (id, coords, core); it opens the TPU, so the
    offline engine benches never ask for it in the parent process.
    """
    env = {
        "image": os.environ.get("P3_IMAGE") or os.environ.get("SERVER_IMAGE")
        or os.environ.get("IMAGE") or "",
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "job": job_id(),
        "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    env.update(package_versions())
    env.update(source_identity())
    env["topology"] = topology()
    if devices:
        import jax

        env["devices"] = [
            {"id": d.id, "coords": list(getattr(d, "coords", ())),
             "core": getattr(d, "core_on_chip", None), "kind": d.device_kind,
             "process": d.process_index}
            for d in jax.devices()
        ]
        env["jax_process_count"] = jax.process_count()
    return env


def environment_id_string(env: dict) -> str:
    """One line: the fields that decide whether two numbers are comparable."""
    t = env.get("topology", {})
    image = env.get("image", "")
    if "@sha256:" in image:
        image = image.split("@sha256:")[0].rsplit("/", 1)[-1] + "@" + image.split("@sha256:")[1][:12]
    elif "/" in image:
        image = image.rsplit("/", 1)[-1]
    return (
        f"image={image or 'unknown'};libtpu={env.get('libtpu', '')};"
        f"torch_tpu={env.get('torch-tpu', '')};jax={env.get('jax', '')};"
        f"vllm={env.get('vllm', '')};sha={env.get('source_sha', '')[:9]}"
        f"{'+' + env['source_dirty_sha'] if env.get('source_dirty_sha') else ''};"
        f"topo={t.get('tpu_topology') or t.get('chips_per_host_bounds') or '?'}"
        f"x{t.get('num_hosts') or '1'}host"
    )


def stack_tag() -> str:
    """Names the installed runtime, so compile caches never mix stacks."""
    v = package_versions()
    return "_".join(
        f"{k}-{v.get(k) or 'none'}" for k in ("torch-tpu", "libtpu", "jax", "vllm")
    ).replace("+", ".")


def cache_key(parts: list[str]) -> str:
    """Short hash for the compile-cache directory of one configuration.

    Routing, source identity, layout, quantization and every env override go
    in, because not all of them are in vllm_torchtpu's own cache key, and a
    cache hit skips tracing, so a patch that is not in the key would silently
    not apply.
    """
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:10]


class Timeline:
    """Absolute UTC timestamps of the phase boundaries of one run.

    Marks: created (job create time, from P3_JOB_CREATED_TS or CDK), start
    (process start), engine_init_start, load_start, load_end, compile_start,
    compile_end, warmup_end, timed_start, timed_end. windows() turns them
    into queue, init, load, compile, warmup and timed durations; phases that
    were not observed come back as null, never as zero.
    """

    def __init__(self):
        self.marks: dict[str, float] = {"start": time.time()}
        created = os.environ.get("P3_JOB_CREATED_TS") or os.environ.get("CDK_JOB_CREATED_TS")
        if created:
            try:
                self.marks["created"] = float(created)
            except ValueError:
                try:
                    self.marks["created"] = datetime.datetime.fromisoformat(
                        created.replace("Z", "+00:00")
                    ).timestamp()
                except ValueError:
                    pass
        pod_start = os.environ.get("P3_POD_START_TS")
        if pod_start:
            try:
                self.marks["pod_start"] = float(pod_start)
            except ValueError:
                pass

    def mark(self, name: str, t: float | None = None) -> None:
        self.marks[name] = time.time() if t is None else t

    def absorb_phases(self, records: list[dict]) -> None:
        """Takes load/compile/warmup boundaries from the runner phase records
        (min start, max end over ranks)."""
        for rec in records:
            if rec.get("kind") != "phase":
                continue
            key = f"{rec['phase']}_{rec['ev']}"
            t = rec["t"]
            if rec["ev"] == "start":
                self.marks[key] = min(self.marks.get(key, t), t)
            else:
                self.marks[key] = max(self.marks.get(key, t), t)

    def windows(self) -> dict:
        m = self.marks

        def span(a, b):
            return round(m[b] - m[a], 3) if a in m and b in m else None

        return {
            "queue_s": span("created", "pod_start") if "pod_start" in m else span("created", "start"),
            "pod_setup_s": span("pod_start", "start"),
            "init_s": span("engine_init_start", "load_model_start"),
            "load_s": span("load_model_start", "load_model_end"),
            "kv_init_s": span("load_model_end", "capture_model_start"),
            "compile_s": span("capture_model_start", "capture_model_end"),
            "spec_warmup_s": span("_warmup_spec_decode_start", "_warmup_spec_decode_end"),
            "engine_ready_s": span("engine_init_start", "engine_ready"),
            "bench_warmup_s": span("engine_ready", "timed_start"),
            "timed_s": span("timed_start", "timed_end"),
            "marks_utc": {k: datetime.datetime.fromtimestamp(v, datetime.timezone.utc).isoformat()
                          for k, v in sorted(m.items(), key=lambda kv: kv[1])},
        }


def run_dir(bench: str, tag: str = "", root: str | None = None) -> str:
    """<root>/<job-id>/<bench>[-tag]-<utc>/ created fresh; never shared."""
    root = root or os.environ.get("P3_OUT_ROOT") or DEFAULT_OUT_ROOT
    if not os.path.isdir(os.path.dirname(root.rstrip("/")) or "/"):
        root = os.path.join(os.getcwd(), "p3-harness-out")
    name = bench + (f"-{tag}" if tag else "") + "-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = os.path.join(root, job_id(), name)
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def append_jsonl(path: str, row: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def read_jsonl(path: str) -> list[dict]:
    """Tolerates NUL gaps left by appends from another pod over NFSv3."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip("\x00 \n")
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def write_env(out_dir: str, extra: dict | None = None, devices: bool = False) -> dict:
    env = environment_id(devices=devices)
    if extra:
        env.update(extra)
    env["id"] = environment_id_string(env)
    write_json(os.path.join(out_dir, "env.json"), env)
    print(f"[p3] ENVIRONMENT_ID {env['id']}", flush=True)
    print(f"[p3] out_dir {out_dir}", flush=True)
    return env


def parse_env_overrides(items: list[str], base: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """KEY=VALUE sets, KEY= (empty value) unsets. Returns (env, unset keys)."""
    env = dict(base)
    unset = []
    for kv in items:
        key, _, val = kv.partition("=")
        if val:
            env[key] = val
        else:
            env.pop(key, None)
            unset.append(key)
    return env, unset


def cache_dirs(cache_root: str, key: str) -> dict[str, str]:
    cache = os.path.join(cache_root, stack_tag(), "key-" + key)
    return {
        "VLLM_CACHE_ROOT": os.path.join(cache, "vllm"),
        "JAX_COMPILATION_CACHE_DIR": os.path.join(cache, "jax_cache"),
        "JAX_COMPILER_CACHE_DIR": os.path.join(cache, "jax_cache"),
        "TORCH_TPU_CACHE_DIR": os.path.join(cache, "torch_tpu_cache"),
        "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE": os.path.join(cache, "torch_tpu_cache"),
        "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT": os.path.join(cache, "torch_tpu_tier3"),
        "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": os.path.join(cache, "torch_tpu_tier3"),
        "VLLM_XLA_CACHE_PATH": os.path.join(cache, "xla_cache"),
    }


def topology_env(num_hosts: int) -> dict[str, str]:
    """TPU_* bounds for a tpu7x 2x2xN slice (one host per 2x2x1)."""
    return {
        "TPU_CHIPS_PER_HOST_BOUNDS": "2,2,1",
        "TPU_HOST_BOUNDS": f"1,1,{num_hosts}",
        "TPU_TOPOLOGY": f"2x2x{num_hosts}",
    }


SHAPE_QUANTILES = (0.10, 0.50, 0.90, 0.99, 1.0)


def load_shapes(path: str, n: int, seed: int = 0, select: str = "sample",
                subset: str = "all", template_tokens: int = 130) -> list[dict]:
    """Samples n (context, uncached suffix, output) triples from the frozen input.

    path is the input line's per_request.csv(.gz) (columns in, out, new_tokens,
    first_turn, kind, ...). Every row is one served request, so uniform row
    sampling weights shapes by request count. select="quantiles" instead takes
    the rows nearest the context quantiles p10/p50/p90/p99/max, cycled to n.
    subset: all, first_turn, later_turn, cold (fully uncached). The server
    prompt is the recorded prompt plus the chat template (about 130 tokens);
    ctx here is that server prompt, new the uncached suffix including the
    template on first turns, out the recorded output length.
    """
    import csv
    import gzip
    import random

    opener = gzip.open if path.endswith(".gz") else open
    rows = []
    with opener(path, "rt", newline="") as f:
        for r in csv.DictReader(f):
            try:
                ctx = int(float(r["in"])) + template_tokens
                new = min(int(float(r["new_tokens"])) + (template_tokens if r.get("first_turn") == "True" else 0), ctx)
                out = int(float(r["out"]))
            except (KeyError, ValueError):
                continue
            first = r.get("first_turn") == "True"
            cold = new >= ctx
            if subset == "first_turn" and not first:
                continue
            if subset == "later_turn" and first:
                continue
            if subset == "cold" and not cold:
                continue
            rows.append({"ctx": ctx, "new": new, "out": max(out, 1), "first_turn": first,
                         "cold": cold, "kind": r.get("kind", ""), "trace": r.get("trace", "")[:8]})
    if not rows:
        raise SystemExit(f"{path}: no rows for subset {subset}")
    rng = random.Random(seed)
    if select == "quantiles":
        by_ctx = sorted(rows, key=lambda r: r["ctx"])
        picks = [by_ctx[min(len(by_ctx) - 1, int(q * (len(by_ctx) - 1)))] for q in SHAPE_QUANTILES]
        return [dict(picks[i % len(picks)], quantile=SHAPE_QUANTILES[i % len(picks)]) for i in range(n)]
    return [dict(rng.choice(rows)) for _ in range(n)]


# Loaded at interpreter start in the engine-core and worker processes. The
# patches apply as the target modules load, so nothing here imports vllm.
SITECUSTOMIZE = r'''
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import sys
import time

_LOG = os.environ.get("P3_STEP_LOG")
_PREFIX = os.environ.get("P3_LOG_PREFIX", "p3")
_POOL_N = 16_777_259  # prime: tiled rows never repeat at power-of-two widths
_pools = {}


def _emit(rec):
    if not _LOG:
        return
    rec.setdefault("t", time.time())
    rec.setdefault("pid", os.getpid())
    with open(_LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _pool(torch, dtype, lo, hi):
    key = (dtype, lo, hi)
    if key not in _pools:
        gen = torch.Generator().manual_seed(1234)
        vals = torch.rand(_POOL_N, generator=gen, dtype=torch.float32)
        _pools[key] = (vals * (hi - lo) + lo).to(dtype)
    return _pools[key]


def _range(name, t):
    # fp8 weights pair with per-block float scales; [-1, 1] keeps most values
    # above the e4m3 subnormal floor, so the dequantized weight is nonzero.
    if t.element_size() == 1:
        return -1.0, 1.0
    if name.endswith(("scale", "scale_inv")):
        return 1e-3, 2e-3
    return -1e-3, 1e-3


def _fill_model(model):
    import torch

    seen = set()
    start = time.time()
    nbytes = 0
    # The state dict holds what a checkpoint load writes: parameters and
    # persistent buffers. Non-persistent buffers (the RoPE cos/sin cache) are
    # computed on device at init in the layout the compiled programs expect;
    # a host copy into them changes that layout.
    named = list(model.state_dict(keep_vars=True).items())
    for i, (name, t) in enumerate(named):
        if id(t) in seen or t.device.type == "meta":
            continue
        seen.add(id(t))
        # Scalars are KV/activation scales; their defaults keep fp8 KV finite.
        if not torch.is_floating_point(t) or t.numel() <= 1:
            continue
        lo, hi = _range(name, t)
        pool = _pool(torch, t.dtype, lo, hi)
        n = t.numel()
        off = (i * 7919) % _POOL_N
        reps = -(-(off + n) // _POOL_N)
        src = pool[off:off + n] if reps == 1 else pool.repeat(reps)[off:off + n]
        with torch.no_grad():
            t.data.copy_(src.view(t.shape))
        nbytes += n * t.element_size()
    print(f"[{_PREFIX}] synthetic weights: {nbytes / 2**30:.1f} GiB in "
          f"{time.time() - start:.0f} s", flush=True)
    _emit({"kind": "phase", "phase": "synthetic_fill", "ev": "end", "gib": nbytes / 2**30})


def _patch_dummy_loader(module):
    if os.environ.get("P3_WEIGHTS", "synthetic") != "synthetic":
        return
    cls = module.DummyModelLoader
    original = cls.load_weights

    def load_weights(self, model, model_config):
        if any(module.get_layerwise_info(m).can_load() for m in model.modules()):
            return original(self, model, model_config)
        _fill_model(model)

    cls.load_weights = load_weights


def _patch_scheduler(module):
    if not _LOG:
        return
    cls = module.Scheduler
    original = cls.update_from_output
    sizes = [int(x) for x in os.environ.get("P3_COMPILE_SIZES", "").split(",") if x]

    def update_from_output(self, scheduler_output, model_runner_output):
        tok = scheduler_output.total_num_scheduled_tokens
        spec = scheduler_output.scheduled_spec_decode_tokens or {}
        rec = {
            "kind": "step",
            "tok": tok,
            "reqs": len(scheduler_output.num_scheduled_tokens),
            "new": len(scheduler_output.scheduled_new_reqs),
            "running": len(self.running),
            "waiting": len(self.waiting),
            "spec": sum(len(v) for v in spec.values()),
            "spec_reqs": len(spec),
        }
        if sizes:
            rec["padded"] = next((s for s in sizes if s >= tok), tok) if tok else 0
        sampled = getattr(model_runner_output, "sampled_token_ids", None)
        if sampled:
            lens = [len(s) for s in sampled]
            rec["sampled"] = sum(lens)
            rec["sampled_reqs"] = len(lens)
        out = original(self, scheduler_output, model_runner_output)
        rec["running_after"] = len(self.running)
        rec["t_end"] = time.time()
        _emit(rec)
        return out

    cls.update_from_output = update_from_output


def _patch_runner(module):
    if not _LOG:
        return
    cls = module.TPUModelRunner
    for name in ("load_model", "capture_model", "_warmup_spec_decode", "initialize_kv_cache",
                 "profile_run"):
        fn = getattr(cls, name, None)
        if fn is None:
            continue

        def wrap(fn=fn, name=name):
            def wrapped(self, *a, **k):
                _emit({"kind": "phase", "phase": name, "ev": "start"})
                try:
                    return fn(self, *a, **k)
                finally:
                    _emit({"kind": "phase", "phase": name, "ev": "end"})
            return wrapped

        setattr(cls, name, wrap())


_routing_calls = [0]
_routing_table = [None]


def _patch_routing(module):
    # Synthetic weights route tokens unevenly (the hidden states of different
    # tokens end up alike), which inflates the fused EP time. "hash" replaces
    # the router logits with a hash of the logits and of the row's sum, so each
    # token draws its experts close to uniformly. "trace" replays a recorded
    # routing table: P3_ROUTING_FILE is an npz with picks [L, N, K] (expert
    # ids per layer, row and slot); row i of a call takes table row i mod N and
    # the layer is the call index mod L at trace time (every layer traces once
    # per compiled size, in order). NaN rows (padding) stay NaN and are still
    # dropped by the kernel. "model" leaves the router alone.
    mode = os.environ.get("P3_ROUTING", "hash")
    if mode == "model":
        return
    import jax.numpy as jnp
    import numpy as np

    original = module.fused_ep_moe
    if mode == "trace":
        d = np.load(os.environ["P3_ROUTING_FILE"])
        picks = np.asarray(d["picks"])
        L = int(os.environ.get("P3_NUM_LAYERS", picks.shape[0]))
        _routing_table[0] = (picks, L)

    def fused_ep_moe(x, w1, w2, w1_scale, w2_scale, gating, *args, **kwargs):
        if mode == "hash":
            row = jnp.sum(x.astype(jnp.float32), axis=-1, keepdims=True)
            u = jnp.sin((gating.astype(jnp.float32) + row) * 7919.0) * 43758.5453
            g = u - jnp.floor(u)
        else:
            picks, L = _routing_table[0]
            layer = _routing_calls[0] % L
            _routing_calls[0] += 1
            n, e = gating.shape
            tab = picks[layer % picks.shape[0]]
            rows = np.arange(n) % tab.shape[0]
            logits = np.full((n, e), 1e-3, np.float32)
            logits += np.arange(e, dtype=np.float32)[None, :] * 1e-6
            for k in range(tab.shape[1]):
                np.put_along_axis(logits, tab[rows, k:k + 1].astype(np.int64), 2.0 - 0.01 * k, axis=-1)
            keep = jnp.isnan(jnp.sum(gating.astype(jnp.float32), axis=-1, keepdims=True))
            g = jnp.where(keep, jnp.nan, jnp.asarray(logits))
        return original(x, w1, w2, w1_scale, w2_scale, g, *args, **kwargs)

    module.fused_ep_moe = fused_ep_moe


_PATCHES = {
    "vllm.model_executor.model_loader.dummy_loader": _patch_dummy_loader,
    "vllm.v1.core.sched.scheduler": _patch_scheduler,
    "vllm_torchtpu.runner.tpu_runner": _patch_runner,
    "vllm_torchtpu.layers.adapter.fused_moe_ep_v3": _patch_routing,
}


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        patch = _PATCHES.pop(name, None)
        if patch is None:
            return None
        if not _PATCHES:
            sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        exec_module = spec.loader.exec_module

        def patched_exec_module(module):
            exec_module(module)
            patch(module)

        spec.loader.exec_module = patched_exec_module
        return spec


sys.meta_path.insert(0, _PatchFinder())

# This file shadows any sitecustomize further down sys.path; run that one too.
_self = sys.modules.get(__name__)
sys.modules.pop(__name__, None)
_here = __file__
for _entry in sys.path:
    _spec = (
        importlib.machinery.PathFinder.find_spec(__name__, [_entry]) if _entry else None
    )
    if _spec and _spec.origin and _spec.origin != _here:
        _module = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_module)
        break
if _self is not None:
    sys.modules[__name__] = _self
'''


def install_sitecustomize(out_dir: str) -> str:
    site_dir = os.path.join(out_dir, "sitecustomize")
    os.makedirs(site_dir, exist_ok=True)
    with open(os.path.join(site_dir, "sitecustomize.py"), "w") as f:
        f.write(SITECUSTOMIZE)
    return site_dir


def phase_records(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("kind") == "phase"]


def step_records(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("kind") == "step"]


def summarize_phases(records: list[dict]) -> dict:
    """Per phase: earliest start, latest end, duration, number of ranks seen."""
    out: dict[str, dict] = {}
    for r in phase_records(records):
        p = out.setdefault(r["phase"], {"start": None, "end": None, "pids": set()})
        p["pids"].add(r.get("pid"))
        if r["ev"] == "start":
            p["start"] = r["t"] if p["start"] is None else min(p["start"], r["t"])
        else:
            p["end"] = r["t"] if p["end"] is None else max(p["end"], r["t"])
    for p in out.values():
        p["pids"] = len(p["pids"])
        p["s"] = round(p["end"] - p["start"], 3) if p["start"] and p["end"] else None
    return out
