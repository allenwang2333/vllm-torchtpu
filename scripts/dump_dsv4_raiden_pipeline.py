# SPDX-License-Identifier: Apache-2.0
"""Dump every intermediate object on the DSv4 Raiden admission path.

Admitting DeepSeek-V4 into Raiden is a chain of transformations: the runner's
per-layer arrays become classified role entries, which become a pool manifest,
which becomes a per-tag geometry table, a transfer plan, a layout fingerprint,
and finally the three flat arrays one transfer puts on the wire. Each stage is
a plain data structure, but only the ends of the chain appear in any one test,
so it is hard to see what the middle looks like.

This script runs the whole chain deviceless on the test fixture's DSv4
materialization and prints each stage as JSON.

    python scripts/dump_dsv4_raiden_pipeline.py
    python scripts/dump_dsv4_raiden_pipeline.py --stage 4
    python scripts/dump_dsv4_raiden_pipeline.py --json > pipeline.json

Run it from the repo root with `PYTHONPATH=$PWD/src:$PWD` so it picks up the
worktree's sources rather than the installed package.
"""

import argparse
import contextlib
import json
import sys
import types

# Importing the connector boots the vLLM platform plugin, which logs to
# stdout. Send that to stderr so `--json` output stays parseable.
with contextlib.redirect_stdout(sys.stderr):
    from vllm_torchtpu.distributed.kv_transfer import tpu_connector as tc
    from vllm_torchtpu.distributed.kv_transfer.raiden import byte_spans as bs
    from vllm_torchtpu.distributed.kv_transfer.raiden import (
        layout_fingerprint as lf)
    from vllm_torchtpu.distributed.kv_transfer.raiden import pool_manifest as pm

    try:
        from tests.distributed.raiden_test_utils import dsv4_materialization
    except ImportError:  # run from inside tests/, or repo root not on path
        from raiden_test_utils import dsv4_materialization  # type: ignore

# A DeepSeek-V4-Flash-ish page: 256 compressed tokens, so the CSA group pages
# at 1024 logical tokens. Same numbers the connector plan test pins.
TOKENS_PER_PAGE = 256
NUM_BLOCKS = 16

# Deviceless stand-ins for the TPU layout probe: the natural
# [blocks, rows, packing, width] order and the packed uint8 tile shape.
LAYOUT_GETTER = lambda tensor: ([3, 2, 1, 0], [[4, 128], [4, 1]], 8)  # noqa: E731
PACKAGE_VERSION = lambda package: f"{package}-fake"  # noqa: E731


def array_labeller():
    """Names each distinct physical array A0, A1, ... in first-seen order."""
    seen = {}

    def label(array):
        key = array.untyped_storage().data_ptr()
        if key not in seen:
            seen[key] = f"A{len(seen)}"
        return seen[key]

    return label, seen


def shape_str(array):
    """`(16, 256, 4, 128)` -> `"16x256x4x128"`, so a dump stays readable."""
    return "x".join(str(dim) for dim in array.shape)


def arrays_of(binding):
    """A layer's bound arrays: CSA binds a (nope, rope) pair, others one."""
    return list(binding) if isinstance(binding, tuple) else [binding]


# ---------------------------------------------------------------------------
# Stage 0. What the model runner produced.
# ---------------------------------------------------------------------------


def stage_0_allocation(named, groups, label):
    """The runner's output: cache groups, their layers, and the arrays.

    This is the input to everything else. The thing to look for is that the
    array labels repeat across groups: an SWA layer and a CSA layer name the
    same A-number, because the runner overlaid one on the other at byte
    offset 0.
    """
    out = []
    for ordinal, group in enumerate(groups):
        spec = next(iter(group.kv_cache_spec.kv_cache_specs.values()))
        layers = []
        for name in group.layer_names:
            layers.append({
                "layer":
                name,
                "arrays": [{
                    "array": label(array),
                    "shape": shape_str(array),
                    "bytes": array.nbytes,
                } for array in arrays_of(named[name])],
            })
        out.append({
            "group_ordinal": ordinal,
            "wrapper": type(group.kv_cache_spec).__name__,
            "wrapper_block_size": group.kv_cache_spec.block_size,
            "layer_spec": {
                "block_size": spec.block_size,
                "storage_block_size": spec.storage_block_size,
                "compress_ratio": spec.compress_ratio,
                "sliding_window": spec.sliding_window,
            },
            "layers": layers,
        })
    return out


def stage_0b_physical(named, label):
    """The same allocation seen from the arrays: who lives on each one.

    Any array listed with more than one layer is aliased. Nothing in the
    bytes says those layers are separate; only the allocator's promise never
    to hand one block ID to two cache groups keeps them apart.
    """
    residents = {}
    for name, binding in named.items():
        for position, array in enumerate(arrays_of(binding)):
            entry = residents.setdefault(label(array), {
                "shape": shape_str(array),
                "bytes": array.nbytes,
                "layers": [],
            })
            entry["layers"].append({
                "layer": name,
                "slot": position,
            })
    return residents


# ---------------------------------------------------------------------------
# Stage 1. Classification: layers -> (role tag, group ordinal, array).
# ---------------------------------------------------------------------------


def stage_1_classify(named, groups, label):
    entries = pm._dsv4_classify(groups, named)
    return [{
        "tag_base": tag_base,
        "layer": layer_name,
        "group_ordinal": ordinal,
        "array": label(array),
        "shape": shape_str(array),
    } for tag_base, layer_name, ordinal, array in entries]


# ---------------------------------------------------------------------------
# Stage 2. The pool manifest: one pool per (role, group, array).
# ---------------------------------------------------------------------------


def stage_2_manifest(manifest, label):
    """The manifest is what Raiden actually registers.

    A pool is not an array. It is a named, block-strided window into one
    storage: `(storage_index, base_offset_bytes, block_stride_bytes,
    num_blocks, regions)`. Two pools with different tags can name the exact
    same window, which is how the aliased caches are described.
    """
    return {
        "binding":
        manifest.binding,
        "num_storages":
        len(manifest.storages),
        "storages": [{
            "storage_index": index,
            "array": label(storage),
            "shape": shape_str(storage),
            "bytes": storage.nbytes,
        } for index, storage in enumerate(manifest.storages)],
        "pools": [
            dict(pool.to_pool_dict(),
                 layer_name=pool.layer_name,
                 live_bytes_per_block=pool.live_bytes_per_block)
            for pool in manifest.pools
        ],
    }


# ---------------------------------------------------------------------------
# Stage 3. Per-tag geometry: the manifest collapsed to what lowering needs.
# ---------------------------------------------------------------------------


def stage_3_geometry(manifest):
    return manifest.geometry_by_tag()


# ---------------------------------------------------------------------------
# Stage 4. The transfer plan.
# ---------------------------------------------------------------------------


def stage_4_plan(manifest, groups):
    """Binds each tag to the block table it will be addressed through.

    `plan_position` indexes the per-group block tables that travel on the
    wire, so two tags sharing a plan position (CSA NoPE and RoPE) are
    addressed through one block table.
    """
    worker = types.SimpleNamespace(_stage3_dsv4_group_count=0)
    plan = tc.TPURaidenConnectorWorker._measure_dsv4_pool_plan(
        worker, manifest, groups)
    return {
        "group_count":
        worker._stage3_dsv4_group_count,
        "pools": [{
            "tag": pool.tag,
            "plan_position": pool.plan_position,
            "page_tokens": pool.page_tokens,
            "live_bytes_per_block": pool.live_bytes_per_block,
            "row_bytes": pool.row_bytes,
            "window_tokens": pool.window_tokens,
            "rows_per_page": pool.live_bytes_per_block // pool.row_bytes,
        } for pool in plan],
    }, plan


# ---------------------------------------------------------------------------
# Stage 5. The layout fingerprint (the cross-replica handshake).
# ---------------------------------------------------------------------------


def stage_5_fingerprint(manifest):
    identity, payload = lf.measured_dsv4_layout_fingerprint(
        manifest,
        layout_getter=LAYOUT_GETTER,
        package_version=PACKAGE_VERSION,
    )
    return {"identity": identity, "payload": payload}


# ---------------------------------------------------------------------------
# Stage 6. One request, lowered to the wire.
# ---------------------------------------------------------------------------


def fake_block_tables(plan, num_tokens, groups):
    """Plausible per-group block tables for a request of `num_tokens`.

    Group ordinals get disjoint block-ID ranges, 1000 apart, because that is
    the property the whole design rests on: no ID is ever shared between
    groups.
    """
    tables = {}
    for pool in plan:
        if pool.plan_position in tables:
            continue
        pages = -(-num_tokens // pool.page_tokens)
        base = 1000 * (pool.plan_position + 1)
        tables[pool.plan_position] = [base + i for i in range(pages)]
    return [tables[position] for position in sorted(tables)]


def stage_6_wire(plan, num_tokens, groups):
    group_blocks = fake_block_tables(plan, num_tokens, groups)
    transfer_tags, dst_blocks, dst_counts, per_tag = [], [], [], []
    for pool in plan:
        raw = group_blocks[pool.plan_position]
        pages, pool_tokens = tc._stage3_dsv4_pool_pages(pool, raw, num_tokens)
        transfer_tags.append(pool.tag)
        dst_blocks.extend(pages)
        dst_counts.append(len(pages))
        registration = bs.lower_dsv4_row_spans(
            tag=pool.tag,
            num_tokens=pool_tokens,
            transfer_rank=0,
            parallelism=1,
            page_tokens=pool.page_tokens,
            live_bytes_per_block=pool.live_bytes_per_block,
            row_bytes=pool.row_bytes,
            block_ids=pages,
        )
        per_tag.append({
            "tag":
            pool.tag,
            "plan_position":
            pool.plan_position,
            "group_block_table":
            raw,
            "pages_transferred":
            pages,
            "tokens_after_window_trim":
            pool_tokens,
            "declared_bytes":
            registration.declared_bytes,
            "registered_block_ids":
            list(registration.block_ids),
            "spans": [{
                "src_block_ordinal": span.src_block_ordinal,
                "src_offset_bytes": span.src_offset_bytes,
                "dst_block_index": span.dst_block_index,
                "dst_offset_bytes": span.dst_offset_bytes,
                "size_bytes": span.size_bytes,
            } for span in registration.spans],
        })
    return {
        "num_tokens": num_tokens,
        "group_block_ids": group_blocks,
        "wire": {
            "transfer_tags": transfer_tags,
            "dst_counts": dst_counts,
            "dst_device_block_ids": dst_blocks,
        },
        "per_tag": per_tag,
    }


# ---------------------------------------------------------------------------


STAGE_TITLES = {
    0: "runner allocation: cache groups, layers, arrays",
    1: "physical view: which layers share each array",
    2: "classify: (role, group, array) entries",
    3: "pool manifest: what Raiden registers",
    4: "per-tag geometry",
    5: "transfer plan",
    6: "layout fingerprint",
    7: "one request lowered to the wire",
}


def build(num_tokens):
    named, groups = dsv4_materialization(num_blocks=NUM_BLOCKS,
                                         tokens_per_page=TOKENS_PER_PAGE)
    label, _ = array_labeller()
    # Label in group order first so the A-numbers read left to right.
    for group in groups:
        for name in group.layer_names:
            for array in arrays_of(named[name]):
                label(array)

    # No raw tensor pool: DSv4 admission requires private typed caches.
    manifest = pm.build_dsv4_pool_manifest(named_kv_caches=named,
                                           kv_cache_groups=groups,
                                           raw_tensors=())
    plan_dump, plan = stage_4_plan(manifest, groups)
    return {
        0: stage_0_allocation(named, groups, label),
        1: stage_0b_physical(named, label),
        2: stage_1_classify(named, groups, label),
        3: stage_2_manifest(manifest, label),
        4: stage_3_geometry(manifest),
        5: plan_dump,
        6: stage_5_fingerprint(manifest),
        7: stage_6_wire(plan, num_tokens, groups),
    }


def print_summary(stages):
    """The whole chain on one page, for reading rather than parsing."""
    print("\nPHYSICAL ARRAYS  (a row with two layers is an aliased array: "
          "same bytes,\ndifferent cache groups, kept apart only by block-ID "
          "disjointness)\n")
    for name, entry in stages[1].items():
        layers = entry["layers"]
        head = f"  {name}  {entry['shape']:>14}  {entry['bytes']:>9} B"
        print(f"{head}  {layers[0]['layer']}")
        for extra in layers[1:]:
            print(f"{' ' * len(head)}  {extra['layer']}   <- aliased")

    print("\nPOOLS  (tag -> storage window; two tags on one storage_index at "
          "offset 0\nare the same bytes under two names)\n")
    print(f"  {'tag':<20} {'stor':>4} {'off':>4} {'stride':>7} "
          f"{'live':>7} {'rows':>5} {'row B':>6}  layer")
    for pool in stages[3]["pools"]:
        region = pool["regions"][0]
        print(f"  {pool['tag']:<20} {pool['storage_index']:>4} "
              f"{pool['base_offset_bytes']:>4} "
              f"{pool['block_stride_bytes']:>7} "
              f"{pool['live_bytes_per_block']:>7} "
              f"{region['num_units']:>5} {region['unit_bytes']:>6}  "
              f"{pool['layer_name']}")

    print("\nTRANSFER PLAN  (plan_position is the index into the per-group "
          "block tables\non the wire; two tags sharing one are addressed "
          "through one table)\n")
    print(f"  {'tag':<20} {'pos':>3} {'page_tok':>8} {'rows/pg':>7} "
          f"{'row B':>6} {'window':>7}")
    for pool in stages[5]["pools"]:
        window = pool["window_tokens"] or "-"
        print(f"  {pool['tag']:<20} {pool['plan_position']:>3} "
              f"{pool['page_tokens']:>8} {pool['rows_per_page']:>7} "
              f"{pool['row_bytes']:>6} {str(window):>7}")

    wire = stages[7]
    print(f"\nONE REQUEST, {wire['num_tokens']} tokens\n")
    for position, table in enumerate(wire["group_block_ids"]):
        print(f"  group_block_ids[{position}] = {table}")
    print()
    print(f"  {'tag':<20} {'pages sent':<28} {'bytes':>9}")
    for entry in wire["per_tag"]:
        pages = str(entry["pages_transferred"])
        if len(pages) > 27:
            pages = pages[:24] + "...]"
        print(f"  {entry['tag']:<20} {pages:<28} "
              f"{entry['declared_bytes']:>9}")
    print(f"\n  dst_counts           = {wire['wire']['dst_counts']}")
    print(f"  total dst block ids  = "
          f"{len(wire['wire']['dst_device_block_ids'])}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",
                        type=int,
                        action="append",
                        help="print only this stage (repeatable)")
    parser.add_argument("--tokens",
                        type=int,
                        default=1534,
                        help="request length for the stage-7 lowering")
    parser.add_argument("--json",
                        action="store_true",
                        help="emit one JSON document, no headers")
    parser.add_argument("--summary",
                        action="store_true",
                        help="the whole chain as one page of tables")
    args = parser.parse_args(argv)

    stages = build(args.tokens)
    wanted = sorted(args.stage) if args.stage else sorted(stages)

    if args.summary:
        print_summary(stages)
        return 0

    if args.json:
        json.dump({str(index): stages[index] for index in wanted},
                  sys.stdout,
                  indent=2)
        print()
        return 0

    for index in wanted:
        title = STAGE_TITLES[index]
        print(f"\n{'=' * 72}\nSTAGE {index}. {title}\n{'=' * 72}")
        print(json.dumps(stages[index], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
