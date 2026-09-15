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
"""Preflight for the ``torch.empty`` -> ``torch.zeros`` KV-pool change.

``kv_cache_materializer.py`` has two allocation routes and only one of them was
touched by that change:

* ``make_attention_cache_tensor()``   -- the *direct* route, taken when
  ``_can_allocate_attention_cache_directly()`` is true.  Already ``torch.zeros``
  before the change and after it.
* ``allocate_raw_kv_cache_tensors()`` -- the *raw* route, taken otherwise.  This
  is the one that used to take a ``torch.empty`` fast path for the fp8 dtypes.

So "does the change affect model M" reduces to "which route does M take", and
that is a property of the ``KVCacheConfig`` vLLM builds, not of the kernel.
``route`` answers it by building the real config for a real checkpoint and
asking the production predicate.

``alloc`` answers the other half: zero-filling a multi-GB pool is not free, and
the direct route has always paid it.  It times both primitives at a pool size
given in GiB so the cost can be compared against a server's start-up budget.

Usage::

    python scripts/preflight_kv_pool_alloc.py route --model <hf-id> --tp 8
    python scripts/preflight_kv_pool_alloc.py alloc --gib 1.93 --device tpu
"""
import argparse
import statistics
import sys
import time

import torch

FP8_DTYPES = {
    "e4m3": torch.float8_e4m3fn,
    "e5m2": torch.float8_e5m2,
    "bf16": torch.bfloat16,
}


def _sync(device: torch.device) -> None:
    """Force any lazily-queued device work to land before we read the clock."""
    if device.type == "cpu":
        return
    try:
        import torch_xla.core.xla_model as xm

        xm.mark_step()
        xm.wait_device_ops()
        return
    except Exception:
        pass
    try:
        torch.tpu.synchronize()
    except Exception:
        pass


def cmd_route(args: argparse.Namespace) -> int:
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

    from vllm_torchtpu.kv_cache_materializer import \
        _can_allocate_attention_cache_directly

    engine_args = EngineArgs(
        model=args.model,
        tensor_parallel_size=args.tp,
        kv_cache_dtype=args.kv_cache_dtype,
        max_model_len=args.max_model_len,
        load_format="dummy",
        trust_remote_code=True,
        enforce_eager=True,
    )
    vllm_config = engine_args.create_engine_config()

    hf_text = vllm_config.model_config.hf_text_config
    n_layers = hf_text.num_hidden_layers
    n_kv = vllm_config.model_config.get_num_kv_heads(
        vllm_config.parallel_config)
    head_size = vllm_config.model_config.get_head_size()
    block_size = vllm_config.cache_config.block_size

    print(f"model            = {args.model}")
    print(f"tp               = {args.tp}")
    print(f"num_layers       = {n_layers}")
    print(f"per_rank_kv_heads= {n_kv}")
    print(f"head_size        = {head_size}")
    print(f"block_size       = {block_size}")
    print(f"cache_dtype      = {vllm_config.cache_config.cache_dtype}")

    # The runner enumerates one spec per attention layer.  Qwen3-MoE and every
    # other uniform full-attention model gives N identical FullAttentionSpecs;
    # a hybrid model would mix in a MambaSpec and change the answer, which is
    # exactly what this probe is here to detect rather than assume.
    kwargs = dict(block_size=block_size,
                  num_kv_heads=n_kv,
                  head_size=head_size,
                  dtype=FP8_DTYPES[args.spec_dtype])
    try:
        specs: dict[str, KVCacheSpec] = {
            f"model.layers.{i}.self_attn": FullAttentionSpec(
                **kwargs, sliding_window=None)
            for i in range(n_layers)
        }
    except TypeError:
        specs = {
            f"model.layers.{i}.self_attn": FullAttentionSpec(**kwargs)
            for i in range(n_layers)
        }

    available = int(args.gib * 1024**3)
    cfg = get_kv_cache_configs(vllm_config, [specs], [available])[0]

    group_types = [
        type(g.kv_cache_spec).__name__ for g in cfg.kv_cache_groups
    ]
    shared_lens = sorted({len(t.shared_by) for t in cfg.kv_cache_tensors})
    direct = _can_allocate_attention_cache_directly(cfg)

    print(f"num_groups       = {len(cfg.kv_cache_groups)}")
    print(f"group_spec_types = {sorted(set(group_types))}")
    print(f"num_tensors      = {len(cfg.kv_cache_tensors)}")
    print(f"shared_by_lens   = {shared_lens}")
    print(f"num_blocks       = {cfg.num_blocks}")
    print(f"ROUTE            = "
          f"{'DIRECT make_attention_cache_tensor' if direct else 'RAW allocate_raw_kv_cache_tensors'}")
    print(f"AFFECTED_BY_CHANGE = {not direct}")
    return 0


def cmd_alloc(args: argparse.Namespace) -> int:
    if args.device.startswith("tpu"):
        # Registers the "tpu" device string, exactly as tests/conftest.py's
        # `device` fixture does under --use-tpu.
        import torch_tpu  # noqa: F401
    device = torch.device(args.device)
    dtype = FP8_DTYPES[args.dtype]
    elem = torch.empty(0, dtype=dtype).element_size()
    n_tensors = max(1, args.num_tensors)
    numel = int(args.gib * 1024**3) // elem // n_tensors
    # Shape it like a pool row-major rather than as one flat vector.
    rows = 4096
    numel = (numel // rows) * rows
    shape = (rows, numel // rows)
    total_gib = numel * elem * n_tensors / 1024**3
    print(f"device={device} dtype={dtype} tensors={n_tensors} shape={shape} "
          f"bytes={numel * elem * n_tensors} ({total_gib:.3f} GiB total)")

    results: dict[str, list[float]] = {"zeros": [], "empty": []}
    for trial in range(args.trials):
        for name, fn in (("empty", torch.empty), ("zeros", torch.zeros)):
            _sync(device)
            t0 = time.perf_counter()
            held = [
                fn(shape, dtype=dtype, device=device) for _ in range(n_tensors)
            ]
            _sync(device)
            dt = time.perf_counter() - t0
            results[name].append(dt)
            print(f"trial {trial} {name:5s} {dt * 1e3:9.3f} ms")
            del held

    print()
    for name in ("empty", "zeros"):
        xs = sorted(results[name])
        print(f"{name:5s} median {statistics.median(xs) * 1e3:9.3f} ms  "
              f"min {xs[0] * 1e3:9.3f} ms  max {xs[-1] * 1e3:9.3f} ms")
    z = statistics.median(results["zeros"])
    e = statistics.median(results["empty"])
    print(f"\nZERO_FILL_COST_MS_PER_{total_gib:.3f}GiB = {(z - e) * 1e3:.3f}")
    if args.pool_gib:
        scale = args.pool_gib / total_gib
        print(f"EXTRAPOLATED_TO_{args.pool_gib}GiB_MS = {(z - e) * 1e3 * scale:.3f}")
    return 0


def cmd_provenance(args: argparse.Namespace) -> int:
    """Say which copy of the materializer is on ``sys.path``.

    A shard that means to exercise the baseline and silently imports the
    branch is the most expensive mistake available here, and ``PYTHONPATH``
    ordering is exactly the kind of thing that looks right and is not.  The
    ``_FP8_DTYPES`` constant existed before the change and was removed by it,
    so its presence is an unambiguous arm label.
    """
    import vllm_torchtpu.kv_cache_materializer as m

    print("materializer_file =", m.__file__)
    print("has_FP8_DTYPES    =", hasattr(m, "_FP8_DTYPES"))
    print("ARM               =",
          "BASELINE" if hasattr(m, "_FP8_DTYPES") else "CHANGED")
    return 0


def cmd_backend(args: argparse.Namespace) -> int:
    """Report the attention backend a given ``--attention-backend`` resolves to.

    The RPA v3 kernel is reached only through ``FLASH_ATTN``; ``CUSTOM`` is the
    separate batched-RPA kernel under ``kernels/experimental/batched_rpa/``.
    ``run_benchmarks.sh`` defaults ``ATTENTION_BACKEND`` to ``CUSTOM``, so which
    one a benchmark actually selects decides whether a v3 change is live at all.
    """
    from vllm.engine.arg_utils import EngineArgs

    engine_args = EngineArgs(
        model=args.model,
        tensor_parallel_size=args.tp,
        kv_cache_dtype=args.kv_cache_dtype,
        max_model_len=args.max_model_len,
        load_format="dummy",
        trust_remote_code=True,
        enforce_eager=True,
        attention_backend=args.attention_backend,
    )
    vllm_config = engine_args.create_engine_config()
    from vllm.attention.selector import get_attn_backend

    backend = get_attn_backend(
        vllm_config.model_config.get_head_size(),
        vllm_config.model_config.dtype,
        vllm_config.cache_config.cache_dtype,
        vllm_config.cache_config.block_size,
    )
    name = getattr(backend, "get_name", lambda: type(backend).__name__)()
    print("requested_backend =", args.attention_backend)
    print("resolved_class    =", f"{backend.__module__}.{backend.__qualname__}")
    print("resolved_name     =", name)
    print("USES_RPA_V3       =", name == "FLASH_ATTN")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("route", help="which materializer route a model takes")
    r.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8")
    r.add_argument("--tp", type=int, default=8)
    r.add_argument("--kv-cache-dtype", default="fp8")
    r.add_argument("--max-model-len", type=int, default=16384)
    r.add_argument("--spec-dtype", default="e4m3", choices=sorted(FP8_DTYPES))
    r.add_argument("--gib", type=float, default=1.93)
    r.set_defaults(func=cmd_route)

    a = sub.add_parser("alloc", help="time torch.zeros vs torch.empty")
    a.add_argument("--gib", type=float, default=1.93)
    a.add_argument("--device", default="tpu")
    a.add_argument("--dtype", default="e4m3", choices=sorted(FP8_DTYPES))
    a.add_argument("--trials", type=int, default=5)
    a.add_argument("--pool-gib", type=float, default=0.0,
                   help="extrapolate the zero-fill cost to this pool size")
    a.add_argument("--num-tensors", type=int, default=1,
                   help="split the pool across N tensors, as the runner does "
                        "(one per attention layer) -- 48 for Qwen3-Coder-30B")
    a.set_defaults(func=cmd_alloc)

    p_prov = sub.add_parser("provenance", help="which materializer is imported")
    p_prov.set_defaults(func=cmd_provenance)

    b = sub.add_parser("backend", help="which attention backend resolves")
    b.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8")
    b.add_argument("--tp", type=int, default=8)
    b.add_argument("--kv-cache-dtype", default="fp8")
    b.add_argument("--max-model-len", type=int, default=16384)
    b.add_argument("--attention-backend", default="CUSTOM")
    b.set_defaults(func=cmd_backend)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
