# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched packed-MQA experiment: precompute per-page DMA source metadata."""

import argparse
import functools
import gzip
import json
import sys
import time
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Run the benchmark without importing the unrelated vLLM model runtime.
ROOT = Path(__file__).resolve().parents[2]
package = types.ModuleType("vllm_torchtpu")
package.__path__ = [str(ROOT / "src/vllm_torchtpu")]
sys.modules.setdefault("vllm_torchtpu", package)

from bench_packed_mqa_dma import load, quantize_rows  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v41 import index_key_layout  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v41.index_selection.tuned_block_sizes import (  # noqa: E402
    PagedMQAConfig,
)

BANK_MASK = 1
VALID_MASK = 2
OFFSET_SHIFT = 2


@functools.partial(
    jax.jit, static_argnames=("total_pages", "first_bank_pages", "page_rows")
)
def prepare_metadata(
    page_ids: jax.Array, *, total_pages: int, first_bank_pages: int, page_rows: int
) -> jax.Array:
    """Encode a bank-local row offset and two flags in one nonnegative int32."""
    max_rows = max(first_bank_pages, total_pages - first_bank_pages) * page_rows
    if max_rows > np.iinfo(np.int32).max >> OFFSET_SHIFT:
        raise ValueError("Bank-local row offsets do not fit the metadata encoding")
    valid = (page_ids >= 0) & (page_ids < total_pages)
    safe = jnp.clip(page_ids, 0, total_pages - 1)
    bank = (safe >= first_bank_pages).astype(jnp.int32)
    local = (safe - bank * first_bank_pages) * page_rows
    return (local << OFFSET_SHIFT) | bank | (valid.astype(jnp.int32) * VALID_MASK)


def check_metadata() -> None:
    """Check all boundary semantics against independently decoded host records."""
    for first, second in ((4, 0), (3, 5), (18432, 18432)):
        total = first + second
        pages = np.array(
            [
                np.iinfo(np.int32).min,
                -1,
                0,
                first - 1,
                first,
                total - 1,
                total,
                np.iinfo(np.int32).max,
            ],
            dtype=np.int32,
        )
        encoded = np.asarray(
            prepare_metadata(
                jnp.asarray(pages),
                total_pages=total,
                first_bank_pages=first,
                page_rows=33,
            )
        )
        for original, word in zip(pages, encoded, strict=True):
            safe = min(max(int(original), 0), total - 1)
            bank = int(safe >= first)
            assert int(word) >> OFFSET_SHIFT == (safe - bank * first) * 33
            assert int(word) & BANK_MASK == bank
            assert bool(int(word) & VALID_MASK) == (0 <= original < total)
    print("PASS metadata: one/two banks, both boundaries, holes, int32 extremes")


def make_module(
    directory: Path, mode: str, prepared: bool, *, source: str | None = None
) -> types.ModuleType:
    prefix = "vllm_torchtpu.kernels.deepseek_v41.index_selection."
    variant = "metadata" if prepared else "baseline"
    if source is None:
        source = (directory / "logits_control.py").read_text()
    source = source.replace(
        'name="paged_mqa_logits"', f'name="paged_mqa_{mode}_{variant}"'
    )
    if prepared:
        old = "valid = ((page >= 0) & (page < self.total_pages)).astype(jnp.int32)"
        assert source.count(old) == 1
        source = source.replace(old, "valid = ((page & 2) != 0).astype(jnp.int32)")
    module = load(
        prefix + f"metadata_probe_{mode}_{variant}",
        directory / "logits_control.py",
        source,
    )
    if prepared:

        class PreparedOps(module.MQAOps):
            def load_page(self, tile, i, metadata_slot, slot):
                record = self.pages[
                    metadata_slot,
                    tile * (self.plan.config.key_tile // self.page_records) + i,
                ]
                address = record >> OFFSET_SHIFT
                rows = self.page_records // self.plan.key_slots + 1

                def copy(bank):
                    pltpu.make_async_copy(
                        self.pool[bank].at[pl.ds(address, rows)],
                        self.keys.at[slot, pl.ds(i * rows, rows)],
                        self.read_sem.at[slot],
                    ).start()

                if len(self.pool) == 1:
                    copy(0)
                else:
                    jax.lax.cond(
                        (record & BANK_MASK) == 0, lambda: copy(0), lambda: copy(1)
                    )

        module.MQAOps = PreparedOps
    if mode == "dma":

        class ReadOnlyOps(module.MQAOps):
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
                self.scores[0, 0, pl.ds(tile * 384, 384)] = (
                    jax.lax.bitcast_convert_type(probe, jnp.float32)
                )

            def fill_padding(self):
                count = (
                    self.plan.page_count
                    * self.page_records
                    // self.plan.config.key_tile
                )
                pltpu.sync_copy(
                    self.scores.at[0, 0, pl.ds(0, count * 384)],
                    self.out.at[self.state.query.start, pl.ds(0, count * 384)],
                )

        module.MQAOps = ReadOnlyOps
    return module


def make_inputs(seed: int, packed: types.ModuleType) -> tuple:
    rng = np.random.default_rng(seed)
    groups, queries, width = 48, 6, 98304
    words = rng.integers(0, 2**32, (groups * width // 2, 128), dtype=np.uint32)
    words = (words & np.uint32(0x807F807F)) | np.uint32(0x3E003E00)
    native = tuple(jnp.asarray(x, jnp.int32) for x in np.split(words, 2))
    del words
    page_ids = rng.permutation(groups * width // 128).reshape(groups, -1)
    page_ids = page_ids.astype(np.int32)
    page_ids[0, 3] = -1
    q = jnp.asarray(rng.normal(size=(groups * queries, 32, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.normal(size=(groups * queries, 32)), jnp.float32)
    q, q_scale = quantize_rows(q)
    quantized = [quantize_rows(index_key_layout.unpack_rows(bank)) for bank in native]
    pool = jax.block_until_ready(tuple(packed.pack_pages(k, s) for k, s in quantized))
    return (
        q,
        weights,
        pool,
        jnp.asarray(page_ids),
        jnp.full((groups * queries,), width - 17, jnp.int32),
        jnp.arange(groups + 1, dtype=jnp.int32) * queries,
        q_scale,
    )


def metadata_for(inputs: tuple) -> jax.Array:
    pool, pages = inputs[2:4]
    page_rows = 33
    return prepare_metadata(
        pages,
        total_pages=sum(bank.shape[0] for bank in pool) // page_rows,
        first_bank_pages=pool[0].shape[0] // page_rows,
        page_rows=page_rows,
    )


def compile_variant(
    module: types.ModuleType, inputs: tuple, mode: str, variant: str, out: Path
) -> jax.stages.Compiled:
    config = PagedMQAConfig(6, 32768, 8, 6, 4)

    def score(q, weights, pool, pages, lengths, offsets, q_scale):
        if variant == "e2e":
            pages = prepare_metadata(
                pages,
                total_pages=sum(bank.shape[0] for bank in pool) // 33,
                first_bank_pages=pool[0].shape[0] // 33,
                page_rows=33,
            )
        return module.paged_mqa_logits(
            q,
            weights,
            pool,
            pages,
            lengths,
            offsets,
            page_records=128,
            q_scale=q_scale,
            config=config,
        )

    score.__name__ = f"metadata_trial_{mode}_{variant}"
    print("COMPILE", score.__name__, flush=True)
    executable = jax.jit(score).lower(*inputs).compile()
    (out / f"{mode}_{variant}.hlo.txt").write_text(executable.as_text())
    return executable


@functools.partial(jax.jit, static_argnames=("mode",))
def same_bits(left: jax.Array, right: jax.Array, *, mode: str) -> jax.Array:
    if mode == "dma":
        # Only the per-tile probe locations are defined in the DMA ablation.
        left, right = left[::6, :1152], right[::6, :1152]
    return jnp.all(
        jax.lax.bitcast_convert_type(left, jnp.int32)
        == jax.lax.bitcast_convert_type(right, jnp.int32)
    )


def validate(runs: dict, inputs: tuple, mode: str) -> None:
    total = sum(bank.shape[0] for bank in inputs[2]) // 33
    boundary = inputs[2][0].shape[0] // 33
    for edge in (False, True):
        pages = inputs[3]
        if edge:
            pages = pages.at[0, :8].set(
                jnp.array(
                    [
                        np.iinfo(np.int32).min,
                        -1,
                        0,
                        boundary - 1,
                        boundary,
                        total - 1,
                        total,
                        np.iinfo(np.int32).max,
                    ],
                    jnp.int32,
                )
            )
        data = (*inputs[:3], pages, *inputs[4:])
        metadata = metadata_for(data)
        cached = (*data[:3], metadata, *data[4:])
        baseline = runs["baseline"](*data)
        for name, args in (("cached", cached), ("e2e", data)):
            candidate = runs[name](*args)
            assert bool(same_bits(baseline, candidate, mode=mode)), (mode, name, edge)
        if mode == "dma":
            observed = np.asarray(baseline[::6, :1152]).view(np.int32)
            host_pages = np.asarray(pages)
            for group in range(host_pages.shape[0]):
                expected = []
                for tile in range(3):
                    for logical, row in (
                        (tile * 256, 0),
                        ((tile + 1) * 256 - 1, 0),
                        ((tile + 1) * 256 - 1, 32),
                    ):
                        physical = int(
                            np.clip(host_pages[group, logical], 0, total - 1)
                        )
                        bank = int(physical >= boundary)
                        local = physical - bank * boundary
                        expected.append(np.asarray(inputs[2][bank][local * 33 + row]))
                np.testing.assert_array_equal(observed[group], np.concatenate(expected))
    print(
        "PASS",
        mode,
        "baseline bitwise parity, both banks and invalid-page edges",
        flush=True,
    )


def raw_span(event: dict) -> tuple[int, int]:
    args = event["args"]
    return int(args["device_offset_ps"]), int(args["device_duration_ps"])


def parse_capture(path: Path, mode: str, expected: list[str]) -> list[dict]:
    paths = list(path.rglob("*.trace.json.gz"))
    if len(paths) != 1:
        raise ValueError(f"Expected one host trace, found {len(paths)} in {path}")
    with gzip.open(paths[0], "rt") as stream:
        events = json.load(stream)["traceEvents"]
    devices = {
        e["pid"]
        for e in events
        if e.get("name") == "process_name" and e["args"].get("name") == "/device:TPU:0"
    }
    lanes = {
        (e["pid"], e["tid"]): e["args"].get("name")
        for e in events
        if e.get("name") == "thread_name"
    }
    modules, kernels = [], []
    for event in events:
        if event.get("pid") not in devices or "dur" not in event:
            continue
        lane = lanes.get((event["pid"], event["tid"]))
        if lane == "XLA Modules" and f"metadata_trial_{mode}_" in event["name"]:
            modules.append(event)
        if lane == "XLA Ops" and "paged_mqa" in event["name"]:
            kernels.append(event)
    modules.sort(key=lambda e: e["ts"])
    if len(modules) != len(expected):
        raise ValueError(
            f"Incomplete trace: {len(modules)} modules, need {len(expected)}"
        )
    samples = []
    for module, variant in zip(modules, expected, strict=True):
        assert f"metadata_trial_{mode}_{variant}" in module["name"], module["name"]
        _, duration = raw_span(module)
        # Pair lanes on XProf's common timeline, retaining raw device durations
        # for timing rather than comparing raw offsets across different lanes.
        enclosed = [
            e
            for e in kernels
            if module["ts"] <= e["ts"]
            and e["ts"] + e["dur"] <= module["ts"] + module["dur"]
        ]
        if len(enclosed) != 1:
            print(
                "TRACE_PAIR_FAILURE",
                json.dumps(
                    {
                        "module": module,
                        "kernel_candidates": kernels[:8],
                        "op_names": sorted(
                            {
                                e["name"]
                                for e in events
                                if "dur" in e
                                and lanes.get((e["pid"], e["tid"])) == "XLA Ops"
                            }
                        ),
                    }
                ),
                flush=True,
            )
            raise ValueError(
                f"Expected one scorer in {module['name']}; got {len(enclosed)}"
            )
        samples.append(
            dict(
                variant=variant,
                module_us=duration / 1e6,
                kernel_us=raw_span(enclosed[0])[1] / 1e6,
            )
        )
    return samples


def measure(runs: dict, inputs: tuple, mode: str, seed: int, out: Path) -> None:
    cached = (*inputs[:3], metadata_for(inputs), *inputs[4:])
    calls = {
        name: functools.partial(run, *(cached if name == "cached" else inputs))
        for name, run in runs.items()
    }
    order = ["baseline", "cached", "e2e", "e2e", "cached", "baseline"]
    for _ in range(5):
        for name in order:
            jax.block_until_ready(calls[name]())
    samples = []
    for capture in range(4):
        path = out / "xprof" / f"{mode}_s{seed}_{capture}"
        with jax.profiler.trace(str(path)):
            time.sleep(0.1)
            for name in order * 5:
                jax.block_until_ready(calls[name]())
            time.sleep(0.1)
        samples.extend(parse_capture(path, mode, order * 5))
    (out / f"samples_{mode}_s{seed}.json").write_text(json.dumps(samples, indent=2))
    for variant in order[:3]:
        selected = [r for r in samples if r["variant"] == variant]
        module_us = float(np.median([r["module_us"] for r in selected]))
        kernel_us = float(np.median([r["kernel_us"] for r in selected]))
        deltas = []
        for block in range(0, len(samples), len(order)):
            records = samples[block : block + len(order)]
            base = np.mean(
                [r["module_us"] for r in records if r["variant"] == "baseline"]
            )
            candidate = np.mean(
                [r["module_us"] for r in records if r["variant"] == variant]
            )
            deltas.append(float(candidate - base))
        record = dict(
            variant=f"{mode}_{variant}_s{seed}",
            mode=mode,
            seed=seed,
            latency_ms=module_us / 1000,
            kernel_us=kernel_us,
            key_read_bytes=48 * 98304 * 132,
            key_read_tb_s=48 * 98304 * 132 / kernel_us / 1e6,
            correctness="PASS",
            correctness_contract="baseline bitwise parity",
            samples=len(selected),
            paired_module_deltas_us=deltas,
            metadata_bytes=48 * 768 * 4,
            config=dict(
                query_tile=6,
                key_tile=32768,
                issue_depth=8,
                query_group=6,
                key_buffers=4,
            ),
            timing="raw XProf device_duration_ps; module includes preprocessing",
        )
        with (out / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cpu-check", action="store_true")
    args = parser.parse_args()
    check_metadata()
    if args.cpu_check:
        return
    if args.implementation_dir is None or args.out is None:
        parser.error("--implementation-dir and --out are required for TPU measurement")
    args.out.mkdir(parents=True, exist_ok=False)
    prefix = "vllm_torchtpu.kernels.deepseek_v41.index_selection."
    load(prefix + "paged_mqa_pipeline", args.implementation_dir / "pipeline_control.py")
    packed = load("metadata_packed_cache", args.implementation_dir / "packed_cache.py")
    inputs = make_inputs(2026, packed)
    executables = {}
    for mode in ("dma", "full"):
        baseline = make_module(args.implementation_dir, mode, False)
        prepared = make_module(args.implementation_dir, mode, True)
        runs = {}
        for name in ("baseline", "cached", "e2e"):
            runs[name] = compile_variant(
                baseline if name == "baseline" else prepared,
                inputs,
                mode,
                name,
                args.out,
            )
        executables[mode] = runs
        validate(runs, inputs, mode)
        measure(runs, inputs, mode, 2026, args.out)
    inputs = make_inputs(2027, packed)
    for mode, runs in executables.items():
        validate(runs, inputs, mode)
        measure(runs, inputs, mode, 2027, args.out)
    print(
        "COMPLETE: metadata checks, bitwise parity, and matched device timing",
        flush=True,
    )


if __name__ == "__main__":
    main()
