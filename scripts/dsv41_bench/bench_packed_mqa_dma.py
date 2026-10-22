# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DMA ablation of the iteration42 packed-page FP8 control, matched fixtures."""

import argparse
import functools
import gzip
import importlib.util
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v41 import index_key_layout
from vllm_torchtpu.kernels.deepseek_v41.index_selection.tuned_block_sizes import (
    PagedMQAConfig,
)


def load(name: str, path: Path, source: str | None = None):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if source is None:
        spec.loader.exec_module(module)
    else:
        exec(compile(source, str(path), "exec"), module.__dict__)
    return module


@jax.jit
def quantize_rows(values):
    values = values.astype(jnp.float32)
    scale = jnp.exp2(jnp.ceil(jnp.log2(jnp.maximum(abs(values).max(-1), 1e-4) / 448)))
    return (values / scale[..., None]).astype(jnp.float8_e4m3fn), scale


@jax.jit
def reference_page(q, keys, weights, q_scale, k_scale):
    logits = jnp.einsum("hd,kd->hk", q, keys, preferred_element_type=jnp.float32)
    logits = jnp.maximum(logits, 0).astype(jnp.bfloat16).astype(jnp.float32)
    return (logits * (weights * q_scale)[:, None]).sum(0) * k_scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sweep-inflight", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    prefix = "vllm_torchtpu.kernels.deepseek_v41.index_selection."
    pipeline_path = args.implementation_dir / "pipeline_control.py"
    pipeline_module = load(prefix + "paged_mqa_pipeline", pipeline_path)
    mqa = load(
        prefix + "paged_mqa_logits", args.implementation_dir / "logits_control.py"
    )
    packed = load("packed_cache", args.implementation_dir / "packed_cache.py")

    class ReadOnlyOps(mqa.MQAOps):
        def issue(self, tile, fragment, group):
            pass

        def transform(self, fragment, group):
            pass

        def reduce(self, fragment, group):
            pass

        def consume(self, tile, fragment, group):
            return None

        def store(self, tile, fragment, parts):
            pass

        def publish(self, tile, wait=False, *, slot=None):
            pass

        def fill_padding(self):
            slot = self.key_slot(self.live_tiles - 1)
            # Return raw word bits, preserving all bits in the correctness probe.
            self.scores[0, 0, pl.ds(0, 128)] = jax.lax.bitcast_convert_type(
                self.keys[slot, 0, :], jnp.float32
            )
            pltpu.sync_copy(
                self.scores.at[0, 0, pl.ds(0, 128)],
                self.out.at[self.state.query.start, pl.ds(0, 128)],
            )

    class SweepReadOnlyOps(ReadOnlyOps):
        def wait_keys(self, tile):
            super().wait_keys(tile)
            slot = self.key_slot(tile)
            rows = self.keys.shape[1]
            probe = jnp.concatenate(
                (
                    self.keys[slot, 0],
                    self.keys[slot, rows - 33],
                    self.keys[slot, rows - 1],
                )
            )
            self.scores[0, 0, pl.ds(tile * 384, 384)] = jax.lax.bitcast_convert_type(
                probe, jnp.float32
            )

        def fill_padding(self):
            count = (
                self.plan.page_count * self.page_records // self.plan.config.key_tile
            )
            pltpu.sync_copy(
                self.scores.at[0, 0, pl.ds(0, count * 384)],
                self.out.at[self.state.query.start, pl.ds(0, count * 384)],
            )

    cases = [(4, 3, "full"), (4, 3, "dma_only")]
    if args.sweep_inflight:
        cases += [(n, n - 1, "dma_only") for n in (2, 3, 5, 6, 7, 8, 12)]
        cases += [(8, ahead, "dma_only") for ahead in (1, 2, 3, 4, 5)]
    schedules = {}
    for buffers, ahead, _ in cases:
        if ahead == buffers - 1:
            schedules[buffers, ahead] = pipeline_module.Pipeline
            continue
        # Change only the lookahead distance; MQAOps still uses the real ring size.
        source = pipeline_path.read_text()
        assert source.count("buffers = self.plan.config.key_buffers") == 1
        assert source.count("self.plan.config.key_buffers - 1") == 2
        source = source.replace(
            "buffers = self.plan.config.key_buffers", f"buffers = {ahead + 1}"
        )
        source = source.replace("self.plan.config.key_buffers - 1", str(ahead))
        (args.out / f"pipeline-ahead{ahead}.py").write_text(source)
        schedules[buffers, ahead] = load(
            prefix + f"pipeline_ahead{ahead}", pipeline_path, source
        ).Pipeline
    for seed in (2026, 2027):
        rng = np.random.default_rng(seed)
        r, qn, width = 48, 6, 98304
        # Identical generation/order to iteration42/run_fast.py.
        words = rng.integers(0, 2**32, (r * width // 2, 128), dtype=np.uint32)
        words = (words & np.uint32(0x807F807F)) | np.uint32(0x3E003E00)
        native = tuple(jnp.asarray(x, jnp.int32) for x in np.split(words, 2))
        del words
        page_ids = rng.permutation(r * width // 128).reshape(r, -1).astype(np.int32)
        page_ids[0, 3] = -1
        pages = jnp.asarray(page_ids)
        q = jnp.asarray(rng.normal(size=(r * qn, 32, 128)), jnp.bfloat16)
        weights = jnp.asarray(rng.normal(size=(r * qn, 32)), jnp.float32)
        lengths = jnp.full((r * qn,), width - 17, jnp.int32)
        offsets = jnp.arange(r + 1, dtype=jnp.int32) * qn
        q, q_scale = quantize_rows(q)
        quantized = [
            quantize_rows(index_key_layout.unpack_rows(bank)) for bank in native
        ]
        del native
        pool = jax.block_until_ready(
            tuple(packed.pack_pages(k, s) for k, s in quantized)
        )
        inputs = q, weights, pool, pages, lengths, offsets
        for buffers, ahead, variant in cases:
            config = PagedMQAConfig(6, 32768, 8, 6, buffers)
            read_ops = SweepReadOnlyOps if args.sweep_inflight else ReadOnlyOps
            ops = mqa.MQAOps if variant == "full" else read_ops
            jax.clear_caches()
            print("START", seed, variant, buffers, ahead, flush=True)
            with (
                patch.object(mqa, "MQAOps", ops),
                patch.object(mqa, "Pipeline", schedules[buffers, ahead]),
            ):
                executable = mqa.paged_mqa_logits.lower(
                    *inputs, page_records=128, q_scale=q_scale, config=config
                ).compile()
            run = functools.partial(executable, *inputs, q_scale=q_scale)
            actual = np.asarray(jax.block_until_ready(run()))
            reference_errors = []
            reference_norms = []
            reference_pass = True
            for group in range(r):
                row = group * qn
                logical_page = (
                    (width - config.key_tile) // 128 if variant == "dma_only" else 0
                )
                physical_page = int(page_ids[group, logical_page])
                bank, local = divmod(physical_page, (r * width // 128) // 2)
                if variant == "dma_only":
                    if args.sweep_inflight:
                        probes = []
                        for tile in range(width // config.key_tile):
                            first = tile * config.key_tile // 128
                            last = first + config.key_tile // 128 - 1
                            for logical, word in ((first, 0), (last, 0), (last, 32)):
                                physical = max(0, int(page_ids[group, logical]))
                                owner, index = divmod(physical, r * width // 128 // 2)
                                probes.append(
                                    np.asarray(pool[owner][index * 33 + word])
                                )
                        expected = np.concatenate(probes)
                    else:
                        expected = np.asarray(pool[bank][local * 33])
                    np.testing.assert_array_equal(
                        actual[row, : expected.size].view(np.int32), expected
                    )
                else:
                    keys, scales = quantized[bank]
                    expected = reference_page(
                        q[row],
                        keys[local * 128 : (local + 1) * 128],
                        weights[row],
                        q_scale[row],
                        scales[local * 128 : (local + 1) * 128],
                    )
                    expected = np.asarray(expected)
                    reference_pass &= bool(
                        np.allclose(actual[row, :128], expected, rtol=2e-5, atol=1e-4)
                    )
                    reference_errors.append(actual[row, :128] - expected)
                    reference_norms.append(expected)
            if reference_errors:
                np.savez(
                    args.out / f"reference-s{seed}.npz",
                    errors=np.stack(reference_errors),
                    reference=np.stack(reference_norms),
                )
            for _ in range(3):
                jax.block_until_ready(run())
            name = f"packed_{variant}_b{buffers}_a{ahead}_s{seed}"
            trace_dir = args.out / "xprof" / name
            # Separate captures prevent verbose DMA traces overflowing the
            # device trace buffer when several invocations share one capture.
            for sample_index in range(5):
                with jax.profiler.trace(str(trace_dir / str(sample_index))):
                    time.sleep(0.1)
                    jax.block_until_ready(run())
                    time.sleep(0.1)
            samples, modules = [], []
            for path in trace_dir.rglob("*.trace.json.gz"):
                with gzip.open(path, "rt") as stream:
                    events = json.load(stream)["traceEvents"]
                devices = {
                    e["pid"]
                    for e in events
                    if e.get("name") == "process_name"
                    and e["args"].get("name") == "/device:TPU:0"
                }
                lanes = {
                    (e["pid"], e["tid"]): e["args"].get("name")
                    for e in events
                    if e.get("name") == "thread_name"
                }
                kernel_events, module_events = [], []
                for event in events:
                    if event.get("pid") not in devices or "dur" not in event:
                        continue
                    if "paged_mqa_logits" not in event["name"]:
                        continue
                    lane = lanes.get((event["pid"], event["tid"]))
                    if lane == "XLA Ops":
                        kernel_events.append(event)
                    if lane == "XLA Modules":
                        module_events.append(event)
                for kernel_event in kernel_events:
                    for module_event in module_events:
                        if (
                            module_event["ts"] <= kernel_event["ts"]
                            and module_event["ts"] + module_event["dur"]
                            >= kernel_event["ts"] + kernel_event["dur"]
                            and module_event["dur"] - kernel_event["dur"] < 20
                        ):
                            samples.append(kernel_event["dur"])
                            modules.append(module_event["dur"])
                            break
            if not samples:
                print(f"NO_COMPLETE_TRACE_SAMPLES {name}", flush=True)
                continue
            assert len(samples) == len(modules)
            collected_samples = len(samples)
            samples, modules = samples[-5:], modules[-5:]
            us = float(np.median(samples))
            record = dict(
                variant=name,
                mode=variant,
                seed=seed,
                requests=r,
                queries=qn,
                width=width,
                config=vars(config),
                prefetch_tiles=ahead,
                max_visible_prefetch_tiles=min(ahead, 2 * width // config.key_tile),
                correctness="PASS" if reference_pass else "REFERENCE_MISMATCH",
                reference_max_abs=(
                    float(np.max(np.abs(reference_errors))) if reference_errors else 0.0
                ),
                reference_relative_l2=(
                    float(
                        np.linalg.norm(reference_errors)
                        / np.linalg.norm(reference_norms)
                    )
                    if reference_errors
                    else 0.0
                ),
                latency_ms=us / 1000,
                module_us=float(np.median(modules)),
                device_samples_us=samples,
                collected_samples=collected_samples,
                sample_quality="OK" if collected_samples >= 3 else "LOW_SAMPLE_COUNT",
                module_samples_us=modules,
                key_read_bytes=r * width * 132,
                key_read_tb_s=r * width * 132 / us / 1e6,
                timing="XProf TPU XLA Ops",
                source="iteration42/control",
                page_records=128,
                dmas_per_page=1,
            )
            with (args.out / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
