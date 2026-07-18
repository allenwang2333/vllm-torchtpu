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
"""TPU ↔ CPU KV cache offloading handler.

This module plugs into vllm's `OffloadingConnector` framework to give TPU
inference a large CPU-resident host pool for KV-cache blocks evicted from
HBM. Cache hits on the host pool are streamed back into HBM via host-side
gather + DMA copy, then scattered into the live `kv_cache` via torch
in-place indexing.

Architecture
------------
- TPU allocates one private dense array per attention layer and one per
  mamba/GDN state tensor (see `kv_cache_materializer`, alias fallback), so
  a module-level hook replaces `OffloadingConnectorWorker.
  register_kv_caches` with a TPU canonicalization: every array becomes its
  own int8 `(num_blocks, row_bytes)` `CanonicalKVCacheTensor`, and each
  KV-cache group's `CanonicalKVCacheRef`s point at that group's arrays.
- Transfers are KV-cache-group aware: block ids arrive ordered by group
  (`GPULoadStoreSpec.group_sizes`), and each group's rows are copied
  between only that group's device arrays and cpu-pool tensors. For hybrid
  models (e.g. Qwen3.5) this moves both attention KV and mamba/GDN state
  pages, matching the scheduler-side `OffloadingConnectorScheduler`
  contract that treats mamba groups as window-1 sliding-window groups.
- `TPUCPUOffloadingSpec` subclasses `vllm.v1.kv_offload.cpu.spec.
  CPUOffloadingSpec`; it overrides `get_handlers()` and exposes
  `estimate_hbm_reserve_bytes(vllm_config) -> int` so the worker can
  reserve HBM for the H2D staging buffer before profile_run.
- `CpuTpuOffloadingHandlers` holds one D2H and one H2D
  `SingleDirectionOffloadingHandler` instance; the spec yields them per
  the OffloadingConnector contract.
- Each handler owns a single background DMA worker thread plus (H2D only)
  a depth-1 device-side staging buffer; transfers serialize through them.
- Oversized H2D loads (more than `KV_H2D_POOL_MAX_BLOCKS` blocks) are
  split into max-buffer-sized chunks; each chunk consumes one engine
  step's worth of latency, the depth-1 invariant is preserved.
- Scatter dispatch is **scheduler-gated**: per-rank `get_finished` only
  defers the scatter (adds the chunk to `_pending_scatters` and reports
  `finished_recving` based on `dma_done`). The actual scatter HLO is
  enqueued from a monkey-patched `OffloadingConnectorWorker.
  start_kv_transfers`, filtered by `scatter_now_req_ids` set on the
  metadata by a monkey-patched `OffloadingConnectorScheduler.
  build_connector_meta` from `scheduler_output.num_scheduled_tokens`.
  A request only enters that set after `KVOutputAggregator` released
  `finished_recving` from every rank, so the scatter HLOs dispatch in
  lockstep across ranks at the same broadcast-sync entry point.

Usage
-----
Pass `--kv-transfer-config` to `vllm serve`:

    vllm serve <model> \\
        --tensor-parallel-size=8 \\
        --kv-offloading-size 400 \\
        --kv-transfer-config '{
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "spec_name": "TPUCPUOffloadingSpec",
                "spec_module_path": "vllm_torchtpu.offload.cpu_tpu"
            }
        }'

`--kv-offloading-size N` reserves N GiB of host RAM for the pool. Sizing
guidance: pool large enough to fit the working set of repeated prefixes,
but small enough that GPU_MEM_UTIL still leaves usable HBM after the H2D
staging buffer reserve (see `estimate_hbm_reserve_bytes`).

A working launcher script lives at
`examples/kv-offload/run_kv_offload_single_host.sh`.

Environment knobs (all optional)
--------------------------------
- `KV_H2D_POOL_MAX_BLOCKS` (default 2048)
    Caps the H2D staging buffer's `(max_padded, …)` first dim. Rounded
    to the next power of 2. Larger = fewer chunked loads on long-prefix
    workloads but more HBM reserved up front.
- `TPU_PREMAPPED_BUFFER_SIZE` (libtpu env, **per device**)
    Sized so libtpu's premap pool can absorb one H2D transfer worth of
    pinned-host staging. 16 GiB per chip works for the standard 480B
    overflow workload (see recipe). Bigger is fine on host RAM budget;
    too small forces lazy `copy_` to fall back to slower per-call pinning.
"""
from __future__ import annotations

import functools
import math
import os
import queue as _queue
import threading
import time
import weakref
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch
from torch_tpu._internal.sync import synchronize as _tpu_sync
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import (BlockIDsLoadStoreSpec,
                                     CanonicalKVCacheRef, CanonicalKVCaches,
                                     CanonicalKVCacheTensor, GPULoadStoreSpec,
                                     LoadStoreSpec)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.worker.worker import (OffloadingHandler,
                                              TransferResult, TransferSpec)

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Module-level patches: route scatter dispatch through a scheduler-driven
# "ready set" so all ranks scatter the same transfers at the same step
# boundary.
#
# Design
# ------
# Two hooks, both idempotent:
#
# 1. SCHEDULER side — `OffloadingConnectorScheduler.build_connector_meta`
#    is wrapped to annotate req_ids in THIS step's active batch.
#
# 2. WORKER side — `OffloadingConnectorWorker.start_kv_transfers` is
#    reimplemented to:
#      a. Pass `scatter_now_req_ids` into each handler's
#         `flush_pending_scatters`, which filters the deferred scatter
#         list and only dispatches HLOs for transfers whose `req_id` is
#         in the set.
#      b. Stash `req_id` on the handler keyed by `job_id` *before*
#         calling `worker.transfer_async`, so the handler can tag the
#         resulting `Transfer` for later lookup.
#    All ranks reach this from the same `execute_model` broadcast
#    (~µs jitter), so the scatter HLOs go onto libtpu in lockstep.
#
# ---------------------------------------------------------------------------
def _install_scheduler_scatter_now_hook() -> None:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import \
        OffloadingConnectorScheduler  # noqa: E501

    orig = OffloadingConnectorScheduler.build_connector_meta
    if getattr(orig, "_tpu_scatter_now_patched", False):
        return
    orig_update = OffloadingConnectorScheduler.update_connector_output

    def update_connector_output_with_recving(
            self, connector_output):  # type: ignore[no-untyped-def]
        # finished_recving reaches the scheduler only after the
        # KVOutputAggregator merged it from ALL ranks, so it is an
        # all-ranks-synced signal in its own right. Remember the req_ids
        # so the next step's metadata releases their deferred scatters.
        finished_recving = getattr(connector_output, "finished_recving", None)
        if finished_recving:
            pending = getattr(self, "_tpu_scatter_now_pending", None)
            if pending is None:
                pending = set()
                self._tpu_scatter_now_pending = pending
            pending.update(finished_recving)
        return orig_update(self, connector_output)

    def build_connector_meta_with_scatter_now(
            self, scheduler_output):  # type: ignore[no-untyped-def]
        meta = orig(self, scheduler_output)
        # Release scatters for req_ids in this step's scheduled batch,
        # plus req_ids whose finished_recving the scheduler has consumed.
        # The latter matters under block starvation: a restored request
        # may not win scheduling capacity for many steps (or ever, if
        # every block is held by other requests stuck behind the same
        # staging buffer) — gating its scatter on being scheduled
        # deadlocks the whole H2D pipeline. Both signals are aggregator-
        # synced, so every rank dispatches the same scatters at the same
        # step boundary.
        scatter_now = set(scheduler_output.num_scheduled_tokens.keys())
        pending = getattr(self, "_tpu_scatter_now_pending", None)
        if pending:
            scatter_now |= pending
            pending.clear()
        meta.scatter_now_req_ids = scatter_now
        return meta

    build_connector_meta_with_scatter_now._tpu_scatter_now_patched = True  # type: ignore[attr-defined]
    OffloadingConnectorScheduler.build_connector_meta = (
        build_connector_meta_with_scatter_now)
    OffloadingConnectorScheduler.update_connector_output = (
        update_connector_output_with_recving)


def _install_flush_pending_scatters_hook() -> None:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import \
        OffloadingConnectorWorker

    orig = OffloadingConnectorWorker.start_kv_transfers
    if getattr(orig, "_tpu_flush_patched", False):
        return

    def start_kv_transfers_with_flush(
            self, metadata):  # type: ignore[no-untyped-def]
        # Step 0: block until stores whose source blocks the scheduler is
        # about to hand to new writers have finished reading them. Upstream
        # runs this fence in handle_preemptions, which the TPU runner never
        # calls; it matters here because raiden stores read the live KV
        # arrays asynchronously after dispatch.
        if getattr(metadata, "jobs_to_flush", None):
            self.worker.wait(metadata.jobs_to_flush)

        # Step 1: drain deferred scatter, filtered by scheduler signal.
        # All ranks reach this point from the same `execute_model`
        # broadcast, so the scatter HLOs land on libtpu in lockstep.
        # `scatter_now_req_ids` is unconditionally annotated by the
        # scheduler-side hook above — AttributeError here means the
        # scheduler hook failed to install (loud failure, by design).
        scatter_now = metadata.scatter_now_req_ids
        for handler in self.worker.handlers:
            if hasattr(handler, "flush_pending_scatters"):
                handler.flush_pending_scatters(scatter_now)

        # Step 2: dispatch pending stores (deferred from prior step's
        # `prepare_store_kv`). Stores don't carry the req_id stash —
        # they're not subject to the scatter-now gate.
        for job_id, transfer_spec in self._unsubmitted_store_jobs:
            success = self.worker.transfer_async(job_id, transfer_spec)
            assert success
        self._unsubmitted_store_jobs.clear()

        # Step 3: dispatch new loads. Stash req_id on the appropriate
        # H2D handler BEFORE calling `worker.transfer_async` so the
        # handler can copy it onto the resulting Transfer.
        for job_id, entry in metadata.load_jobs.items():
            self._load_jobs[job_id] = entry.req_id
            # Find the handler that will receive this transfer.
            src, dst = entry.transfer_spec
            ttype = (src.medium(), dst.medium())
            handler = self.worker.transfer_type_to_handler.get(ttype)
            if handler is not None and hasattr(handler,
                                               "_stash_req_id_for_job"):
                handler._stash_req_id_for_job(job_id, entry.req_id)
            success = self.worker.transfer_async(job_id, entry.transfer_spec)
            assert success

    start_kv_transfers_with_flush._tpu_flush_patched = True  # type: ignore[attr-defined]
    OffloadingConnectorWorker.start_kv_transfers = (
        start_kv_transfers_with_flush)


def _install_drop_aborted_scatters_hook() -> None:
    """Wrap `OffloadingConnectorWorker.get_finished` so handlers discard
    deferred scatters of finished/aborted requests.

    A request aborted while WAITING_FOR_REMOTE_KVS never enters the
    scheduled batch, so its gated scatter would hold the depth-1 staging
    buffer forever and wedge every later H2D load.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import \
        OffloadingConnectorWorker

    orig = OffloadingConnectorWorker.get_finished
    if getattr(orig, "_tpu_drop_aborted_patched", False):
        return

    def get_finished_with_drop(
            self, finished_req_ids):  # type: ignore[no-untyped-def]
        if finished_req_ids:
            for handler in self.worker.handlers:
                if hasattr(handler, "drop_pending_scatters_for"):
                    handler.drop_pending_scatters_for(finished_req_ids)
        return orig(self, finished_req_ids)

    get_finished_with_drop._tpu_drop_aborted_patched = True  # type: ignore[attr-defined]
    OffloadingConnectorWorker.get_finished = get_finished_with_drop


def tpu_per_rank_block_row_bytes(kv_cache_config: KVCacheConfig) -> int:
    """Bytes one block row occupies across all TPU per-layer arrays.

    TPU materializes hybrid KV as private per-layer (per-state) arrays, so
    a block costs every attention layer's physical page plus every
    mamba/GDN layer's unpadded state page — more than vLLM's KVCacheTensor
    accounting, which assumes layers share pages.
    """
    from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

    from vllm_torchtpu.layers.vllm.attention import PallasAttentionBackend

    total = 0
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        num_layers = len(group.layer_names)
        if isinstance(spec, AttentionSpec):
            page = PallasAttentionBackend.get_kv_cache_page_size_bytes(
                spec.block_size, spec.num_kv_heads, spec.head_size, spec.dtype)
        elif isinstance(spec, MambaSpec):
            from vllm.utils.torch_utils import get_dtype_size

            from vllm_torchtpu.kv_cache_materializer import \
                _maybe_pad_mamba_state_shape

            # Physical per-block state bytes: the materializer may pad a
            # state's leading dim for DMA row addressability, so size from
            # the padded shapes rather than the spec's logical page.
            page = sum(
                math.prod(_maybe_pad_mamba_state_shape(shape)) *
                get_dtype_size(dtype)
                for shape, dtype in zip(spec.shapes, spec.dtypes))
        else:
            raise NotImplementedError(
                f"Unsupported KV cache spec for TPU offloading: {spec!r}")
        total += num_layers * page
    return total


# Leading (major) dim of each source KV array, keyed by storage data_ptr.
# Attention arrays are materialized at kernel-block granularity (shape[0] =
# num_blocks x scheduler_block/kernel_block); the raiden path needs that
# granularity to register buffers, while the canonical views stay at
# scheduler-block granularity. Populated during canonicalization.
_SOURCE_MAJOR_DIM_BY_STORAGE: dict[int, int] = {}


def build_tpu_canonical_kv_caches(
    kv_cache_config: KVCacheConfig,
    kv_caches: dict[str, torch.Tensor | list[torch.Tensor]],
) -> CanonicalKVCaches:
    """Canonicalize TPU per-layer KV-cache arrays for the offload handlers.

    TPU cannot overlay differently-typed layers on shared bytes, so each
    attention layer owns a private dense array and each mamba/GDN layer owns
    one private dense array per state tensor (`kv_cache_materializer`, alias
    fallback). Each array is viewed as int8 `(num_blocks, row_bytes)` and
    becomes its own `CanonicalKVCacheTensor`; rows are copied whole, so
    `page_size_bytes` equals the physical row size (no padding).
    """
    num_blocks = kv_cache_config.num_blocks
    tensors: list[CanonicalKVCacheTensor] = []
    tensor_idx_by_storage: dict[int, int] = {}
    _SOURCE_MAJOR_DIM_BY_STORAGE.clear()
    group_data_refs: list[list[CanonicalKVCacheRef]] = []
    for group in kv_cache_config.kv_cache_groups:
        group_refs: list[CanonicalKVCacheRef] = []
        for layer_name in group.layer_names:
            layer_cache = kv_caches[layer_name]
            is_mamba_states = not isinstance(layer_cache, torch.Tensor)
            arrays = list(layer_cache) if is_mamba_states else [layer_cache]
            for arr in arrays:
                storage = arr.untyped_storage()
                if arr.storage_offset() != 0 or storage.size() != arr.nbytes:
                    raise NotImplementedError(
                        f"KV cache for layer {layer_name} is a view into "
                        "shared storage; TPU offloading requires private "
                        "per-layer arrays "
                        "(TPU_VLLM_KV_CACHE_ALIAS_FALLBACK=1).")
                # Mamba state arrays must have exactly one slot per block;
                # attention arrays may be at kernel-block granularity (a
                # whole number of rows per scheduler block).
                if ((is_mamba_states and arr.shape[0] != num_blocks)
                        or arr.shape[0] % num_blocks != 0
                        or arr.nbytes % num_blocks != 0):
                    raise NotImplementedError(
                        f"KV cache for layer {layer_name} has "
                        f"{arr.shape[0]} slots / {arr.nbytes} bytes, "
                        f"incompatible with num_blocks={num_blocks}. "
                        "Compact-mamba sizing shrinks mamba arrays below "
                        "num_blocks slots; TPU offloading requires the "
                        "uniform layout (pin --num-gpu-blocks-override).")
                # KV-sharing layers alias another layer's storage; reuse
                # its canonical tensor so shared bytes move exactly once.
                existing = tensor_idx_by_storage.get(storage.data_ptr())
                if existing is not None:
                    group_refs.append(
                        CanonicalKVCacheRef(
                            tensor_idx=existing,
                            page_size_bytes=tensors[existing].page_size_bytes))
                    continue
                row_bytes = arr.nbytes // num_blocks
                int8_view = (torch.tensor(
                    [], dtype=torch.int8,
                    device=arr.device).set_(storage).view(
                        num_blocks, row_bytes))
                tensors.append(
                    CanonicalKVCacheTensor(tensor=int8_view,
                                           page_size_bytes=row_bytes))
                tensor_idx_by_storage[storage.data_ptr()] = len(tensors) - 1
                _SOURCE_MAJOR_DIM_BY_STORAGE[storage.data_ptr()] = int(
                    arr.shape[0])
                group_refs.append(
                    CanonicalKVCacheRef(tensor_idx=len(tensors) - 1,
                                        page_size_bytes=row_bytes))
        group_data_refs.append(group_refs)
    return CanonicalKVCaches(tensors=tensors, group_data_refs=group_data_refs)


def _install_register_kv_caches_hook() -> None:
    """Replace upstream `register_kv_caches` with TPU canonicalization.

    The upstream implementation views each layer's storage as
    `(num_blocks, page_size_padded)` and asserts that layers sharing a
    `KVCacheTensor` share one physical allocation. Neither holds on TPU:
    layers get private per-layer (per-state) arrays whose rows are the
    unpadded physical page, so the padded view overflows the storage and
    the shared-storage asserts fail.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import \
        OffloadingConnectorWorker

    orig = OffloadingConnectorWorker.register_kv_caches
    if getattr(orig, "_tpu_register_patched", False):
        return

    def register_kv_caches_tpu(self,
                               kv_caches):  # type: ignore[no-untyped-def]
        if not isinstance(self.spec, TPUCPUOffloadingSpec):
            return orig(self, kv_caches)
        canonical = build_tpu_canonical_kv_caches(self.spec.kv_cache_config,
                                                  kv_caches)
        self._register_handlers(canonical)

    register_kv_caches_tpu._tpu_register_patched = True  # type: ignore[attr-defined]
    OffloadingConnectorWorker.register_kv_caches = register_kv_caches_tpu


# Raiden D2H store handlers whose queued dispatches the completion hook
# below drains as engine steps complete on device.
_deferred_store_handlers: weakref.WeakSet = weakref.WeakSet()

# monotonic timestamp of the last engine step seen by the dispatch hook;
# read by the hang-dump watchdog.
_last_step_ts = [0.0]
_hang_dump_started = False

# Engine-step bookkeeping for store dispatch. `_current_step` stamps store
# jobs as they are queued (a job's source rows were written by strictly
# earlier steps); `_last_completed_step` is the newest step whose
# sampled-token copy has resolved on the host — TensorCore programs
# execute serially, so at that moment every device write through that
# step has finished. Written by the runner main thread and the async
# output-resolution thread; reads/writes are single-value and guarded by
# _dispatch_lock where ordering matters.
_current_step = [0]
_last_completed_step = [-1]
_dispatch_lock = threading.Lock()


def _dispatch_deferred_stores() -> None:
    """Dispatch eligible queued transfers. Runner main thread only —
    raiden managers are also driven from get_finished/wait on that
    thread, and concurrent calls into one manager are not known to be
    safe."""
    _last_step_ts[0] = time.monotonic()
    for handler in list(_deferred_store_handlers):
        handler.dispatch_queued_stores(_last_completed_step[0])


def _record_step_completed(step: int) -> None:
    with _dispatch_lock:
        if step > _last_completed_step[0]:
            _last_completed_step[0] = step


def _start_hang_dump_watchdog() -> None:
    """Dump all thread stacks to stderr when the engine stops stepping.

    Gated by KV_OFFLOAD_HANG_DUMP_S (seconds of step silence before a
    dump; 0 disables). A worker wedged inside a step is otherwise
    invisible: the engine's RPC timeout kills the server minutes later
    with no worker-side stack trace.
    """
    global _hang_dump_started
    interval_s = float(os.environ.get("KV_OFFLOAD_HANG_DUMP_S", "0"))
    if interval_s <= 0 or _hang_dump_started:
        return
    _hang_dump_started = True
    import faulthandler

    def _watch() -> None:
        while True:
            time.sleep(interval_s / 2)
            last = _last_step_ts[0]
            if last and time.monotonic() - last > interval_s:
                logger.warning(
                    "[kv-offload] no engine step for %.0fs; dumping all "
                    "thread stacks",
                    time.monotonic() - last)
                faulthandler.dump_traceback(all_threads=True)
                _last_step_ts[0] = time.monotonic()

    threading.Thread(target=_watch, name="kv-offload-hang-dump",
                     daemon=True).start()


def _install_store_dispatch_hook() -> None:
    """Dispatch queued raiden D2H stores as their source writes complete.

    Raiden's d2h reads device HBM directly, outside the lazy graph, so a
    store must not start until every enqueued program that writes the
    source rows has finished. There is no ordering primitive that can
    express this without referencing the KV arrays themselves — and any
    program materialized against a KV array races the per-step buffer
    donation swap (`Buffer has been deleted or donated`). Instead, each
    queued transfer is stamped with an engine step and becomes eligible
    once that step's sampled-token copy resolves on the host
    (`AsyncTPUCopyState.wait`): TensorCore programs execute serially, so
    the resolved copy means every device program through that step has
    finished. The copy may resolve on the worker's async-output thread;
    that hook only RECORDS the completed step — dispatch itself always
    runs on the runner main thread (end of `sample_tokens`,
    `kv_connector_no_forward` steps, and the jobs_to_flush wait loop),
    because raiden managers are also driven from get_finished/wait there
    and concurrent calls into one manager are not known to be safe.

    The stored rows stay pinned by the offloading scheduler until the
    job's futures complete, so later steps cannot overwrite them while
    the DMA reads.
    """
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
    from vllm_torchtpu.runner.tpu_runner_async_output import AsyncTPUCopyState
    if getattr(TPUModelRunner.sample_tokens, "_tpu_store_dispatch_patched",
               False):
        return

    orig_sample_tokens = TPUModelRunner.sample_tokens
    orig_no_forward = TPUModelRunner.kv_connector_no_forward
    orig_from_chunks = AsyncTPUCopyState.from_device_chunks.__func__
    orig_wait = AsyncTPUCopyState.wait

    @functools.wraps(orig_sample_tokens)
    def sample_tokens_with_step_counter(self, *args, **kwargs):
        try:
            return orig_sample_tokens(self, *args, **kwargs)
        finally:
            _current_step[0] += 1
            # Non-async scheduling resolved THIS step's copy inline;
            # async resolved the previous step's (in _modify_prev_results
            # or on the output thread, which only records). Either way
            # the newest recorded completion is dispatchable now, on the
            # main thread.
            _dispatch_deferred_stores()

    @functools.wraps(orig_no_forward)
    def no_forward_with_store_dispatch(self, *args, **kwargs):
        _dispatch_deferred_stores()
        output = orig_no_forward(self, *args, **kwargs)
        # A no-forward step can itself queue stores (its connector
        # metadata carries store jobs); dispatch any whose stamped step
        # has already completed. Later-stamped jobs drain when the
        # in-flight step's copy resolves.
        _dispatch_deferred_stores()
        return output

    @classmethod
    @functools.wraps(orig_from_chunks)
    def from_chunks_with_step(cls, *args, **kwargs):
        state = orig_from_chunks(cls, *args, **kwargs)
        state._kv_offload_step = _current_step[0]  # noqa: SLF001
        return state

    @functools.wraps(orig_wait)
    def wait_with_store_dispatch(self):
        was_completed = self.completed
        orig_wait(self)
        step = getattr(self, "_kv_offload_step", None)
        if not was_completed and step is not None:
            # Record only: this may run on the worker's async-output
            # thread, and raiden calls must stay on the runner main
            # thread (the next step's dispatch picks it up).
            _record_step_completed(step)

    sample_tokens_with_step_counter._tpu_store_dispatch_patched = True  # type: ignore[attr-defined]
    TPUModelRunner.sample_tokens = sample_tokens_with_step_counter
    TPUModelRunner.kv_connector_no_forward = no_forward_with_store_dispatch
    AsyncTPUCopyState.from_device_chunks = from_chunks_with_step
    AsyncTPUCopyState.wait = wait_with_store_dispatch


# These hooks reshape the OffloadingConnector's store/load dispatch into the
# torch handler's deferred-scatter + lockstep flow (flush_pending_scatters,
# _unsubmitted_store_jobs). The raiden handler does direct D2h/H2d + poll-based
# get_finished and uses the STOCK connector dispatch/completion path, so
# installing these would replace start_kv_transfers with a flow it doesn't
# participate in -- leaving the scheduler's store-completion/block-free gate
# unsatisfied and the engine busy-spinning. Only install for the torch path.
# The raiden path needs these hooks too: non-row-addressable tensors (GDN
# conv state) ride the host-staged sidecar, whose deferred scatters use the
# scatter-now gate. The patched start_kv_transfers is a superset of the
# stock dispatch, so pure-raiden transfers behave identically under it.
_install_scheduler_scatter_now_hook()
_install_flush_pending_scatters_hook()
_install_drop_aborted_scatters_hook()
# The registration hook only canonicalizes tensor views; it applies to both
# the torch and raiden paths.
_install_register_kv_caches_hook()
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Maximum D2H stores whose gather buffers may be in flight (queued or
# draining) at once; transfer_async blocks when saturated. Each in-flight
# store holds one gathered copy of its blocks in HBM.
_D2H_MAX_INFLIGHT = int(os.environ.get("KV_D2H_MAX_INFLIGHT", "4"))


def _pad_to_power_of_2(ids: np.ndarray) -> tuple[np.ndarray, int]:
    """Return (padded_ids, original_len) padded to the next power of two.

    Repeated final element fills the padding so scattered writes are
    idempotent.  The caller must use `original_len` to trim results when
    needed.
    """
    n = len(ids)
    if n == 0:
        return ids, 0
    p = 1
    while p < n:
        p <<= 1
    if p == n:
        return ids, n
    padded = np.empty(p, dtype=ids.dtype)
    padded[:n] = ids
    padded[n:] = ids[n - 1]
    return padded, n


# ---------------------------------------------------------------------------
# Background DMA worker
# ---------------------------------------------------------------------------


class _DmaWorker:
    """Single background thread draining a FIFO work queue.

    One instance per direction (D2H / H2D). Each `submit(fn, *args)` enqueues
    work and returns a `threading.Event` that is set when `fn` returns. The
    worker blocks on `queue.get()` when idle, and the FIFO queue guarantees
    serial execution — combined with `event.synchronize()` inside the task,
    this guarantees only one DMA per direction is in flight at any moment.
    """

    def __init__(self, name: str) -> None:
        self._queue: _queue.SimpleQueue = _queue.SimpleQueue()
        self._thread = threading.Thread(target=self._run,
                                        name=name,
                                        daemon=True)
        self._thread.start()

    def submit(self, fn, *args) -> threading.Event:
        done = threading.Event()
        self._queue.put((fn, args, done))
        return done

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            fn, args, done = item
            try:
                fn(*args)
            except Exception:
                logger.exception("[kv-offload] DMA worker task failed")
            finally:
                done.set()
                # Unbind the task's closure and buffers before blocking on
                # the next get(); otherwise the completed task's staging
                # buffers stay reachable through these locals for as long
                # as the queue is idle.
                del item, fn, args, done

    def shutdown(self) -> None:
        self._queue.put(None)


# ---------------------------------------------------------------------------
# Transfer record
# ---------------------------------------------------------------------------


@dataclass
class Transfer:
    job_id: int
    num_bytes: int
    # threading.Event set by _DmaWorker when the DMA task returns. Because
    # the task ends with `_tpu_sync(wait=True)` (D2H) or a blocking
    # `d.copy_(h)` (H2D), dma_done.is_set() implies the DMA is physically
    # complete on-device. For H2D, dma_done starts as None until
    # `_start_h2d_dma` kicks the worker — deferred submission keeps only one
    # H2D DMA in flight at any time, so the single _h2d_device_buffer is
    # never overwritten while the previous transfer's scatter is still
    # reading it.
    dma_done: threading.Event | None
    # D2H: the device gather output, kept alive until the worker's D2H
    # copy_ drains it so XLA's gather IR isn't dropped early. Unused by
    # H2D (staging views live in `current_scatters`).
    device_buffer: list | None = None
    # H2D: per-chunk work units; each chunk is a list of `_H2DPart` (at
    # most one per KV-cache group, each capped at the staging buffer's
    # max_padded rows) whose parts share one buffer acquisition and one
    # deferred scatter. Chunks stream through the depth-1 device buffer
    # one per get_finished() round-trip.
    chunk_descs: list | None = None
    # H2D: total number of chunks the transfer is split into. Set at
    # transfer_async time, never mutated thereafter.
    chunks_total: int = 0
    # H2D: number of chunks already dispatched-and-scattered. Incremented
    # in get_finished() after each chunk's DMA completes.
    chunks_done: int = 0
    # H2D: current chunk's scatter list — per part (tensor_idxs,
    # dst_ids_i32 device tensor, staging views). Rebuilt for each chunk
    # in `_start_h2d_dma`.
    current_scatters: list | None = None
    # H2D: set by the worker task once it has taken `_h2d_buffer_free`;
    # the error path in get_finished() releases the buffer only when the
    # failed task actually held it.
    buffer_acquired: bool = False
    # Set by the DMA worker task if it raises. get_finished() reads this to
    # report success=False.
    error: BaseException | None = None
    # Request ID owning this transfer. Stashed at transfer_async time from
    # the connector_worker's _jobs mapping. Used by flush_pending_scatters
    # to filter against scheduler-side `scatter_now_req_ids` so scatter
    # dispatch happens in lockstep across ranks (gated by the aggregator,
    # never by per-rank coordination).
    req_id: str | None = None


@dataclass
class _PendingScatter:
    """A chunk whose H2D completed; scatter HLO awaits the next
    broadcast-synchronized `start_kv_transfers` for dispatch.

    Snapshotting the chunk's scatter list here lets the underlying
    Transfer be re-submitted for its next chunk (which overwrites the
    Transfer's per-chunk fields) before this snapshot is consumed.
    """
    transfer: Transfer
    # Per part: (tensor_idxs, dst_ids_i32 device tensor, staging views).
    scatters: list[tuple[list[int], object, list]]
    is_last_chunk: bool


@dataclass
class _H2DPart:
    """One KV-cache group's share of an H2D chunk: rows gathered at
    `src_ids` in the cpu pool and scattered to `dst_ids` in the device KV
    cache, touching only `tensor_idxs`."""
    tensor_idxs: list[int]
    src_ids: np.ndarray
    dst_ids: np.ndarray


# ---------------------------------------------------------------------------
# Single-direction handler
# ---------------------------------------------------------------------------


class SingleDirectionOffloadingHandler(OffloadingHandler):
    """Handles KV-cache transfers for one direction (D2H or H2D) on TPU.

    Operates on canonical int8 `(num_blocks, row_bytes)` views — one per
    physical KV-cache array (attention KV or mamba/GDN state). Block ids
    arrive ordered by KV-cache group (`GPULoadStoreSpec.group_sizes`);
    each group's rows are gathered from / scattered to only that group's
    tensors, so hybrid models transfer every group's bytes.
    """

    def __init__(
        self,
        src_tensors: list[torch.Tensor],
        dst_tensors: list[torch.Tensor],
        group_tensor_idxs: list[list[int]],
        tpu_to_cpu: bool,
    ):
        """
        Args:
          src_tensors / dst_tensors: per-canonical-tensor int8 2D views,
              index-aligned across the two lists. One side is the device
              KV cache, the other the CPU pool; rows correspond 1:1
              (offloaded block size == gpu block size).
          group_tensor_idxs: per KV-cache group, the canonical tensor
              indices holding that group's layers' data.
          tpu_to_cpu: True for the D2H (store) direction, False for H2D
              (load).
        """
        assert len(src_tensors) == len(dst_tensors)
        for s, d in zip(src_tensors, dst_tensors):
            assert s.ndim == 2 and d.ndim == 2
            assert s.shape[1] == d.shape[1]

        self.src_tensors = src_tensors
        self.dst_tensors = dst_tensors
        self.group_tensor_idxs = group_tensor_idxs
        self.row_bytes = [
            int(t.shape[1] * t.element_size()) for t in src_tensors
        ]

        self.tpu_to_cpu: bool = tpu_to_cpu
        self.transfer_type = ("GPU", "CPU") if self.tpu_to_cpu else ("CPU",
                                                                     "GPU")
        self._device: torch.device = (src_tensors[0].device if self.tpu_to_cpu
                                      else dst_tensors[0].device)

        self._transfer_map: dict[int, Transfer] = {}
        self._transfers: deque[Transfer] = deque()
        # Bridge from connector_worker (which knows req_id at start_kv
        # _transfers time) to handler.transfer_async (which builds the
        # Transfer). Used by H2D loads only — D2H stores don't get tagged
        # because they're not subject to the scatter-now gate — but
        # initialized here so the unconditional `Transfer.req_id` lookup
        # in `transfer_async` works on both directions' instances.
        self._req_id_by_job_id: dict[int, str] = {}

        # Lightweight diagnostics — printed sparingly to spot stalls.

        # Single background DMA worker for this direction. The FIFO queue
        # plus the worker task's terminating `_tpu_sync(..., wait=True)`
        # together guarantee that only one DMA per direction is in flight
        # at any instant. Thread isolation replaces what a dedicated TPU
        # stream + event.synchronize() would do on CUDA — no explicit
        # stream context is needed.
        worker_name = ("kv-offload-d2h-worker"
                       if self.tpu_to_cpu else "kv-offload-h2d-worker")
        self._dma_worker = _DmaWorker(name=worker_name)

        # D2H store flow control (see transfer_async). H2D loads are
        # already bounded by the depth-1 staging buffer.
        if self.tpu_to_cpu:
            self._d2h_inflight = threading.Semaphore(_D2H_MAX_INFLIGHT)

        # H2D device-staging buffer. Single pre-allocated buffer at the max
        # padded shape; each H2D transfer takes a view [:n_padded].
        self._h2d_device_buffer: list[torch.Tensor] | None = None
        self._h2d_max_padded: int = 0
        if not self.tpu_to_cpu:
            ref = self.dst_tensors
            max_pool_blocks = int(
                os.environ.get("KV_H2D_POOL_MAX_BLOCKS", "2048"))
            max_n = min(ref[0].shape[0], max_pool_blocks)
            max_padded = 1
            while max_padded < max_n:
                max_padded <<= 1
            self._h2d_max_padded = max_padded
            self._h2d_device_buffer = [
                torch.zeros(
                    (max_padded, ) + t.shape[1:],
                    dtype=t.dtype,
                    device=self._device,
                ) for t in ref
            ]
            # Force materialization so subsequent copy_ calls don't go
            # through a lazy-allocation path that could stall PJRT.
            _tpu_sync(self._h2d_device_buffer, wait=True)
            # Buffer-reuse gate. SET = device buffer is logically free for
            # the worker to write the next H2D into. CLEARED = a transfer
            # is mid-flight (worker is writing) OR pending scatter (main
            # thread hasn't yet enqueued the TC HLO that reads it).
            #
            # Main thread `set()`s after enqueuing scatter on libtpu's
            # command stream (NOT after the scatter completes on device —
            # we rely on XLA's HBM-range dependency tracking to serialize
            # the next H2D DMA after the scatter on the device). This
            # keeps main thread non-blocking; the worker is the only one
            # that ever waits.
            self._h2d_buffer_free = threading.Event()
            self._h2d_buffer_free.set()
            # H2D: chunks whose DMA completed; scatter HLO deferred to the
            # next broadcast-synchronized `start_kv_transfers`.
            self._pending_scatters: list[_PendingScatter] = []
            logger.info(
                "[kv-offload] H2D device buffer: %d blocks (max_padded)",
                max_padded,
            )

    # -- H2D deferred-submission management ----------------------------------

    def _stash_req_id_for_job(self, job_id: int, req_id: str) -> None:
        """Called from monkey-patched `OffloadingConnectorWorker.
        start_kv_transfers` immediately before `worker.transfer_async`,
        so the downstream `self.transfer_async` can tag the Transfer
        with its owning req_id without a framework signature change.
        """
        self._req_id_by_job_id[job_id] = req_id

    def _start_h2d_dma(self, t: Transfer) -> None:
        """Submit the worker closure that will host-gather, gate on the
        device buffer being free, and DMA the chunk into device_buffer.

        Submission is eager (called from transfer_async and from
        get_finished for the next chunk of a multi-chunk transfer). The
        worker queue serializes execution; the `_h2d_buffer_free` Event
        gates the actual DMA enqueue so libtpu sees:
            scatter(prev) → H2D(this)
        in that command-stream order. XLA's HBM-range dependency on
        `device_buffer` then serializes them on-device — no host-side
        `_tpu_sync` is needed and the main thread never blocks.

        Dispatches just the chunk indexed by `t.chunks_done` — one part
        per KV-cache group, all sharing a single buffer acquisition and a
        single deferred scatter. get_finished() re-submits so the worker
        picks up the next chunk.
        """
        chunk: list[_H2DPart] = t.chunk_descs[t.chunks_done]

        scatters: list[tuple[list[int], object, list]] = []
        gather_specs: list[tuple[list, list, np.ndarray]] = []
        for part in chunk:
            src_padded, _ = _pad_to_power_of_2(part.src_ids)
            dst_padded, _ = _pad_to_power_of_2(part.dst_ids)
            n_padded = len(src_padded)
            dst_ids_i32 = (torch.from_numpy(dst_padded.astype(np.int32)).to(
                torch.int32).to(self._device))
            views = [
                self._h2d_device_buffer[i][:n_padded] for i in part.tensor_idxs
            ]
            pools = [self.src_tensors[i] for i in part.tensor_idxs]
            scatters.append((part.tensor_idxs, dst_ids_i32, views))
            gather_specs.append((pools, views, src_padded))

        t.current_scatters = scatters
        t.buffer_acquired = False
        buffer_free = self._h2d_buffer_free  # captured for the closure

        def _h2d_task(specs: list):
            try:
                # Host-side gathers first. Safe at any time — they touch
                # only cpu_pool (read-only here) and freshly allocated host
                # buffers; no shared TPU state.
                staged = []
                for pools, views, src_ids_padded in specs:
                    src_ids_torch = torch.from_numpy(
                        src_ids_padded.astype(np.int64))
                    n_padded = len(src_ids_padded)
                    hosts = []
                    for cpu_pool in pools:
                        h = torch.empty(
                            (n_padded, ) + cpu_pool.shape[1:],
                            dtype=cpu_pool.dtype,
                        )
                        torch.index_select(cpu_pool, 0, src_ids_torch, out=h)
                        hosts.append(h)
                    staged.append((views, hosts))
                # Gate on the device buffer being free. Blocks the WORKER
                # thread, never the main thread. Main `set()`s after
                # enqueuing the scatter HLO for the prior transfer, which
                # guarantees: prev scatter is on libtpu's command stream
                # before this H2D's copy_ enqueue lands on it. Parts touch
                # disjoint staging tensors, so one acquisition covers the
                # whole chunk.
                buffer_free.wait()
                buffer_free.clear()
                t.buffer_acquired = True
                for views, hosts in staged:
                    for d, h in zip(views, hosts):
                        d.copy_(h)
            except BaseException as e:
                t.error = e
                raise

        t.dma_done = self._dma_worker.submit(_h2d_task, gather_specs)

    @property
    def prewarm_shapes(self) -> list[int]:
        return []

    def prewarm_shape(self, p: int) -> None:  # noqa: ARG002
        return None

    # -- OffloadingHandler interface -----------------------------------------

    def _split_by_group(
        self,
        gpu_spec: GPULoadStoreSpec,
        cpu_spec: BlockIDsLoadStoreSpec,
    ) -> list[tuple[list[int], np.ndarray, np.ndarray]]:
        """Split a transfer's flat id lists into per-group segments.

        Both id lists are ordered by KV-cache group; `group_sizes` gives
        each group's length. With offloaded block size == gpu block size,
        the cpu ids map 1:1 onto the gpu ids, so both lists share the
        group boundaries. Returns (tensor_idxs, gpu_ids, cpu_ids) per
        non-empty group.
        """
        gpu_ids = gpu_spec.block_ids
        cpu_ids = cpu_spec.block_ids
        assert gpu_ids.ndim == 1 and cpu_ids.ndim == 1
        group_sizes = gpu_spec.group_sizes
        assert len(group_sizes) == len(self.group_tensor_idxs)
        assert sum(group_sizes) == len(gpu_ids)
        assert len(cpu_ids) == len(gpu_ids)

        segments = []
        offset = 0
        for tensor_idxs, size in zip(self.group_tensor_idxs, group_sizes):
            # A group with no tensors in this handler's subset still
            # advances the id offset — its blocks are carried by another
            # handler (composite raiden + host-staged sidecar split).
            if size and tensor_idxs:
                segments.append((tensor_idxs, gpu_ids[offset:offset + size],
                                 cpu_ids[offset:offset + size]))
            offset += size
        return segments

    def transfer_async(self, job_id: int, transfer_spec: TransferSpec) -> bool:
        src_spec, dst_spec = transfer_spec
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)
        gpu_spec = src_spec if self.tpu_to_cpu else dst_spec
        cpu_spec = dst_spec if self.tpu_to_cpu else src_spec
        assert isinstance(gpu_spec, GPULoadStoreSpec)
        segments = self._split_by_group(gpu_spec, cpu_spec)
        assert segments, "transfer with no blocks in any KV-cache group"

        n = sum(len(seg[1]) for seg in segments)
        num_bytes = sum(
            len(gpu_ids) * sum(self.row_bytes[i] for i in tensor_idxs)
            for tensor_idxs, gpu_ids, _ in segments)

        if self.tpu_to_cpu:
            # ── D2H: per-group device row-gather (main) → lazy copy_ + sync
            #         (worker thread) → host scatter into that group's
            #         cpu_pool tensors, all inside the worker task. ──
            #
            # Backpressure: each queued store holds its gather output in
            # HBM until the worker drains it. Under eviction storms the
            # store rate outruns the DMA engine, so cap the number of
            # in-flight stores; when saturated, block here until one
            # drains. All ranks execute the same store schedule, so the
            # stall is symmetric across ranks — local flow control, not a
            # cross-rank rendezvous.
            if not self._d2h_inflight.acquire(blocking=False):
                logger.info_once(
                    "[kv-offload] D2H store backpressure engaged "
                    "(KV_D2H_MAX_INFLIGHT=%d); generation throttles to the "
                    "D2H drain rate under eviction bursts.", _D2H_MAX_INFLIGHT)
                self._d2h_inflight.acquire()

            seg_work = []
            all_device_buffers = []
            for tensor_idxs, gpu_ids, cpu_ids in segments:
                padded_ids, _ = _pad_to_power_of_2(gpu_ids)
                ids_i32 = (torch.from_numpy(padded_ids.astype(np.int32)).to(
                    torch.int32).to(self._device))

                # Compute stream (main thread): device gather. Stays on
                # the main thread because moving it would force XLA to
                # coordinate kv_l (live KV cache) reads across threads.
                ids_i64 = ids_i32.to(torch.int64)
                device_buffer = [
                    self.src_tensors[i][ids_i64] for i in tensor_idxs
                ]
                all_device_buffers.extend(device_buffer)
                # Per-group IR flush: each gather program is keyed by
                # (group tensors, padded n) — a population bounded by
                # num_groups x the power-of-2 ladder. One flush across all
                # groups would key one program per COMBINATION of
                # per-group sizes, and that unbounded program-cache churn
                # exhausts the on-device program arena under store bursts.
                _tpu_sync(device_buffer, wait=False)

                # Unpinned host staging matching the gathered device
                # shape. The lazy D2H copy_ below stages through libtpu's
                # premap pool (TPU_PREMAPPED_BUFFER_SIZE), so the dst
                # doesn't need to be pinned ourselves — saves the per-call
                # pin-allocation cost.
                host_buffer = [
                    torch.empty(db.shape, dtype=db.dtype)
                    for db in device_buffer
                ]
                seg_work.append(
                    (tensor_idxs, device_buffer, host_buffer, cpu_ids,
                     len(gpu_ids)))

            # Build the Transfer first so the worker task closure can stash
            # any exception on it; success=False is then reported from
            # get_finished() instead of silently masking a failed DMA.
            t = Transfer(
                job_id=job_id,
                num_bytes=num_bytes,
                dma_done=None,
                device_buffer=all_device_buffers,
            )
            cpu_pool_tensors = self.dst_tensors
            inflight = self._d2h_inflight

            # Submit the actual DMA work to the per-direction worker.
            # The task does device -> host copies AND the cpu_pool host
            # scatters. dma_done is signaled only after all are complete.
            def _d2h_task():
                try:
                    for tensor_idxs, devs, hosts, cpu_ids, n_g in seg_work:
                        for d, h in zip(devs, hosts):
                            h.copy_(d)
                        for i, h in zip(tensor_idxs, hosts):
                            cpu_pool_tensors[i][cpu_ids] = h[:n_g]
                except BaseException as e:
                    t.error = e
                    raise
                finally:
                    seg_work.clear()
                    inflight.release()

            t.dma_done = self._dma_worker.submit(_d2h_task)

        else:
            # H2D: host index_select (main) → lazy copy_ + sync (worker)
            #      → deferred scatter in flush_pending_scatters.
            # Chunk k carries the k-th max_padded-sized id range of every
            # group, so a typical load (every group within the staging
            # cap) is a single chunk with one DMA and one deferred
            # scatter. Multi-chunk transfers stream one chunk per
            # get_finished() round-trip.
            chunk_cap = self._h2d_max_padded
            chunks: list[list[_H2DPart]] = []
            for tensor_idxs, gpu_ids, cpu_ids in segments:
                for k, s in enumerate(range(0, len(gpu_ids), chunk_cap)):
                    if k == len(chunks):
                        chunks.append([])
                    chunks[k].append(
                        _H2DPart(
                            tensor_idxs=tensor_idxs,
                            src_ids=cpu_ids[s:s + chunk_cap],
                            dst_ids=gpu_ids[s:s + chunk_cap],
                        ))
            if len(chunks) > 1:
                logger.info_once(
                    "[kv-offload] H2D transfer of %d blocks exceeds device "
                    "buffer (%d); will be loaded as %d chunks across that "
                    "many engine steps.", n, chunk_cap, len(chunks))

            t = Transfer(
                job_id=job_id,
                num_bytes=num_bytes,
                dma_done=None,
                chunk_descs=chunks,
                chunks_total=len(chunks),
                chunks_done=0,
            )

        # Tag the Transfer with its owning req_id so flush_pending_scatters
        # can later filter against scheduler-provided `scatter_now_req_ids`.
        # The req_id was stashed by the monkey-patched start_kv_transfers
        # immediately before this call.
        t.req_id = self._req_id_by_job_id.pop(job_id, None)
        self._transfer_map[job_id] = t
        self._transfers.append(t)
        if not self.tpu_to_cpu:
            # Eager submit. Worker queue serializes; the buffer_free Event
            # gates the actual DMA enqueue against the prior scatter.
            self._start_h2d_dma(t)
        return True

    def flush_pending_scatters(self, scatter_now_req_ids: set[str]) -> None:
        """Dispatch H2D scatter HLOs for deferred chunks whose owning
        request has been promoted to the active batch on ALL ranks.

        `scatter_now_req_ids` is the set of req_ids in the current step's
        scheduled batch, populated by the scheduler-side hook from
        `scheduler_output.num_scheduled_tokens`. A request only enters
        that set after `KVOutputAggregator` released its `finished_recving`
        from every rank — set membership is the "all ranks done" signal
        that lets each rank dispatch its scatter in lockstep with the
        others. Transfers whose req_id isn't in the set stay in
        `_pending_scatters` for a later step's flush.

        For chunked transfers, re-submits the next chunk to the worker
        after this chunk's scatter is enqueued — preserving the libtpu
        command-stream order `scatter(prev) → H2D(next)` that XLA needs
        to serialize same-buffer reads/writes correctly.
        """
        # Check tpu_to_cpu FIRST — `_pending_scatters` is only initialized
        # on the H2D handler (see __init__'s `if not self.tpu_to_cpu` block).
        # The D2H handler shares this class but has no deferred-scatter
        # concept; bail before touching the missing attribute.
        if self.tpu_to_cpu or not self._pending_scatters:
            return

        # Intermediate (non-last) chunks of a chunked transfer flush
        # eagerly — they write to KV-cache slots already reserved for the
        # request and the request hasn't been promoted yet (its
        # `finished_recving` only fires on the LAST chunk). Gating
        # intermediate chunks on `scatter_now_req_ids` would deadlock:
        # the next chunk can't dispatch until the prior chunk's scatter is
        # enqueued (buffer-reuse invariant), but the req_id won't enter
        # `scatter_now_req_ids` until `finished_recving` is reported,
        # which only happens on the last chunk's `get_finished`.
        to_flush = [
            ps for ps in self._pending_scatters if
            (not ps.is_last_chunk or ps.transfer.req_id in scatter_now_req_ids)
        ]
        to_keep = [
            ps for ps in self._pending_scatters
            if (ps.is_last_chunk
                and ps.transfer.req_id not in scatter_now_req_ids)
        ]

        if not to_flush:
            # Nothing to dispatch on this rank this step. Return early
            # so we don't accidentally freed _h2d_buffer_free below.
            return

        try:
            for ps in to_flush:
                if ps.transfer.error is not None:
                    continue
                for tensor_idxs, dst_ids_i32, views in ps.scatters:
                    dst_ids_i64 = dst_ids_i32.to(torch.int64)
                    for i, dev_buf in zip(tensor_idxs, views):
                        self.dst_tensors[i][dst_ids_i64] = dev_buf
                        # Flush per tensor: each scatter program then
                        # in-place-aliases a single KV array. Left in one
                        # lazy region, the chunk's scatters fuse into one
                        # program whose aliased I/O spans every array it
                        # touches (tens of GB), and loading it demands a
                        # same-sized contiguous reservation that a busy
                        # HBM cannot provide (RuntimeProgramAllocation-
                        # Failure on 'tt_jit_index_put_').
                        _tpu_sync([self.dst_tensors[i]], wait=False)
        finally:
            # Scatter HLOs are now enqueued on libtpu's command stream;
            # the worker may proceed with the next H2D (XLA serializes
            # via device_buffer's HBM-range event chain).
            self._h2d_buffer_free.set()
        # Re-submit the next chunk of any flushed transfer whose chunk
        # was not its last. The newly-submitted DMA will block on the
        # worker thread's `buffer_free.wait()` if the buffer is still
        # held by another deferred chunk (depth=1 invariant).
        for ps in to_flush:
            if ps.transfer.error is None and not ps.is_last_chunk:
                self._start_h2d_dma(ps.transfer)
        self._pending_scatters = to_keep

    def drop_pending_scatters_for(self, req_ids: set[str]) -> None:
        """Discard deferred scatters owned by requests that finished or
        were aborted before their scatter was dispatched.

        An aborted request is freed without ever entering the scheduled
        batch, so its gated last-chunk scatter would otherwise sit in
        `_pending_scatters` forever with the staging buffer held. Its
        destination blocks are already freed (and possibly reallocated),
        so the scatter must not run; dropping it and releasing the buffer
        is the correct teardown.
        """
        if self.tpu_to_cpu or not self._pending_scatters or not req_ids:
            return
        keep = [
            ps for ps in self._pending_scatters
            if ps.transfer.req_id not in req_ids
        ]
        if len(keep) != len(self._pending_scatters):
            self._pending_scatters = keep
            self._h2d_buffer_free.set()

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        # Scan the whole queue, not just the head: with the depth-1
        # staging buffer, a later transfer's chunk can complete while an
        # earlier transfer's next chunk is still blocked on
        # `buffer_free.wait()` — its scatter must be registered (and later
        # flushed, releasing the buffer) regardless of queue position, or
        # the pipeline wedges.
        for t in list(self._transfers):
            # dma_done is set by _DmaWorker when the task returns.
            #  - D2H: the task's blocking h.copy_(d) waits for the DMA to
            #    drain into host, then runs the cpu_pool host scatter, so
            #    dma_done.is_set() ⇒ both copy AND scatter are done.
            #  - H2D: the task's d.copy_(h) returns once the op is enqueued
            #    on libtpu's command stream (the DMA itself may still be
            #    in flight on the DMA engine).
            #
            # `dma_done is None` is the chunked re-entry marker: we set it
            # to None after queueing a chunk for deferred scatter, and the
            # next chunk's re-submission (from flush_pending_scatters) will
            # set it to a fresh Event.
            if t.dma_done is None or not t.dma_done.is_set():
                continue

            if self.tpu_to_cpu:
                # D2H: the worker task ran both the device→host copy AND
                # the cpu_pool host scatter. The main thread only has to
                # drop buffer refs.
                self._transfers.remove(t)
                t.device_buffer = None
            else:
                # H2D: defer the scatter HLO dispatch to
                # `flush_pending_scatters`, which runs at the next
                # broadcast-synchronized `start_kv_transfers` on all ranks
                # at once. Reporting `finished_recving` here (without
                # scattering yet) is safe because the scheduler's
                # KVOutputAggregator waits for ALL ranks to report before
                # promoting the request; the request's forward at step k+1
                # is preceded by start_kv_transfers's flush on the same
                # step, so XLA sees `scatter → forward` on the kv_cache
                # buffer and serializes correctly.
                t.chunks_done += 1
                is_last = (t.error is not None
                           or t.chunks_done >= t.chunks_total)
                if t.error is None:
                    self._pending_scatters.append(
                        _PendingScatter(
                            transfer=t,
                            scatters=t.current_scatters,
                            is_last_chunk=is_last,
                        ))
                elif t.buffer_acquired:
                    # Error path: no scatter will run for this chunk, so
                    # release the buffer the failed task acquired. A task
                    # that failed before acquisition never held it — the
                    # buffer still belongs to another chunk's pending
                    # scatter and must stay held.
                    self._h2d_buffer_free.set()
                # Detach so the next chunk's `_start_h2d_dma` (called from
                # flush_pending_scatters) can replace these fields without
                # clobbering this chunk's snapshot in `_pending_scatters`.
                t.current_scatters = None
                t.dma_done = None  # re-entry marker for chunked transfers
                if not is_last:
                    # Non-last chunk: keep the Transfer queued; do NOT
                    # report finished_recving yet. flush_pending_scatters
                    # will re-submit the next chunk after enqueuing this
                    # chunk's scatter.
                    continue
                self._transfers.remove(t)

            results.append(
                TransferResult(
                    job_id=t.job_id,
                    success=(t.error is None),
                    transfer_size=t.num_bytes,
                    # TODO: elapsed_time() not yet implemented on TpuEvent.
                    transfer_time=1e-9,
                    transfer_type=self.transfer_type,
                ))
            del self._transfer_map[t.job_id]
        return results

    def wait(self, job_ids: set[int]) -> None:
        """Block until the named transfers have completed.

        Mirrors the CUDA reference impl in
        vllm/v1/kv_offload/cpu/gpu_worker.py:wait — purely a flush
        primitive. Internal-state cleanup (popping from `_transfers`,
        deleting from `_transfer_map`, building TransferResults) is left
        to the next `get_finished()` call. Removing transfers here would
        prevent `get_finished()` from emitting their TransferResults,
        which the scheduler relies on to mark jobs complete.
        """
        for jid in job_ids:
            t = self._transfer_map.get(jid)
            if t is None:
                continue
            if self.tpu_to_cpu:
                # Worker task ends with the cpu_pool host scatter, so
                # dma_done.is_set() implies the D2H is fully landed.
                t.dma_done.wait()
            else:
                # H2D skips non-ready transfers rather than waiting; the
                # OffloadingConnector only ever calls wait() for D2H flush.
                raise AssertionError(
                    f"wait() called on H2D handler for job={jid}; "
                    "OffloadingConnector should never invoke this path.")

    def shutdown(self) -> None:
        """Tear down the worker thread cleanly.

        For H2D, the worker may be blocked inside `_h2d_task` on
        `buffer_free.wait()` — `_dma_worker.shutdown()` alone would
        deadlock because the worker can't pull the shutdown sentinel
        until the current task returns. Set the buffer-free Event first
        to unblock the wait; the task then runs its (possibly futile)
        copy_ to completion and the worker thread exits on the sentinel.
        """
        if not self.tpu_to_cpu:
            self._h2d_buffer_free.set()
        self._dma_worker.shutdown()


# ---------------------------------------------------------------------------
# Raiden-backed offloading handler
# ---------------------------------------------------------------------------

# When set, KV offload uses raiden's KVCacheManager.D2h/H2d — direct PJRT
# raw-buffer DMA between the device KV cache and raiden's DmaMapped host pool —
# instead of the torch Pallas-gather + copy_ + scatter path. No device staging
# buffer, no cpu_pool tensor, no scatter-now gate (the DMA writes kv_cache
# directly; there is no scatter HLO to dispatch in lockstep).
_USE_RAIDEN_OFFLOAD = os.environ.get("VLLM_TPU_OFFLOAD_RAIDEN", "0") == "1"
# Upper bound on the raiden offload flush wait (seconds). Dedicated to the
# offload path -- intentionally NOT the disagg p2p-pull timeout -- so it can be
# tuned independently.
_RAIDEN_OFFLOAD_WAIT_TIMEOUT_S = float(
    os.environ.get("VLLM_TPU_OFFLOAD_WAIT_TIMEOUT_S", "30"))


class _RaidenOffloadingHandler(OffloadingHandler):
    """One-direction (D2H or H2D) KV offload via raiden's direct DMA.

    Each KV-cache group carries a list of `(KVCacheManager, factor)`
    sub-managers, one per row class — a raiden manager sizes and
    addresses every registered layer with layer 0's slice, so all
    tensors it holds must share one slice size and one kernel-row
    factor (a d2h/h2d call copies the addressed rows of every tensor
    its manager registered, so ids must also be group-scoped).
    `transfer_async` splits the flat id lists by
    `GPULoadStoreSpec.group_sizes` into one segment per sub-manager of
    each non-empty group; H2D loads issue immediately, D2H stores queue
    until the runner's next device-quiescent point (see
    `_install_store_dispatch_hook`). `get_finished` polls the returned
    `RaidenFuture`s. The host pool lives inside raiden (DmaMapped), so
    there is no `cpu_pool` tensor here.

    Tensors whose on-device layout is not row-addressable (raiden's
    per-row slice differs from nbytes/num_blocks — e.g. GDN conv state
    without TPU_VLLM_PAD_MAMBA_STATE_FOR_DMA, whose block dim lands
    inside the tile) are carried by a host-staged
    `SingleDirectionOffloadingHandler` sidecar instead; a job completes
    when its raiden futures are ready AND the sidecar reported its part.
    """

    def __init__(self,
                 group_mgrs: list[list[tuple]],
                 tpu_to_cpu: bool,
                 group_row_bytes: list[int],
                 sidecar: SingleDirectionOffloadingHandler | None = None,
                 sidecar_group_has_work: list[bool] | None = None):
        assert len(group_mgrs) == len(group_row_bytes)
        self._group_mgrs = group_mgrs
        self.tpu_to_cpu = tpu_to_cpu
        self._group_row_bytes = group_row_bytes
        self._sidecar = sidecar
        self._sidecar_group_has_work = sidecar_group_has_work or []
        self.transfer_type = ("GPU", "CPU") if tpu_to_cpu else ("CPU", "GPU")
        # job_id -> [list[RaidenFuture], num_bytes, sidecar_pending,
        #            dispatched]
        self._pending: dict[int, list] = {}
        # Transfers queue here at transfer_async time, stamped with an
        # engine step, and dispatch once that step's device programs have
        # completed (see _install_store_dispatch_hook). Appended by the
        # runner main thread, drained by whichever thread resolves a
        # step's sampled-token copy.
        self._queued_transfers: list[tuple] = []
        self._queue_lock = threading.Lock()
        _install_store_dispatch_hook()
        _deferred_store_handlers.add(self)
        if tpu_to_cpu:
            _start_hang_dump_watchdog()

    # -- sidecar delegation (H2D deferred-scatter machinery) ----------------

    def _stash_req_id_for_job(self, job_id: int, req_id: str) -> None:
        if self._sidecar is not None:
            self._sidecar._stash_req_id_for_job(job_id, req_id)

    def flush_pending_scatters(self, scatter_now_req_ids: set[str]) -> None:
        if self._sidecar is not None:
            self._sidecar.flush_pending_scatters(scatter_now_req_ids)

    def drop_pending_scatters_for(self, req_ids: set[str]) -> None:
        if self._sidecar is not None:
            self._sidecar.drop_pending_scatters_for(req_ids)

    def _sidecar_involved(self, group_sizes) -> bool:
        if self._sidecar is None:
            return False
        return any(size and has_work for size, has_work in zip(
            group_sizes, self._sidecar_group_has_work))

    def transfer_async(self, job_id: int, transfer_spec: TransferSpec) -> bool:
        src_spec, dst_spec = transfer_spec
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)
        gpu_spec = src_spec if self.tpu_to_cpu else dst_spec
        cpu_spec = dst_spec if self.tpu_to_cpu else src_spec
        assert isinstance(gpu_spec, GPULoadStoreSpec)
        group_sizes = gpu_spec.group_sizes
        assert len(group_sizes) == len(self._group_mgrs)
        assert sum(group_sizes) == len(gpu_spec.block_ids)
        assert len(cpu_spec.block_ids) == len(gpu_spec.block_ids)

        segments: list[tuple[object, list[int], list[int], list[int]]] = []
        num_bytes = 0
        offset = 0
        for group, size in enumerate(group_sizes):
            if size and self._group_mgrs[group]:
                # Buffers are registered at the source arrays' kernel-block
                # granularity; a scheduler block spans `factor` contiguous
                # rows there, expressed via copy_sizes per segment.
                for mgr, factor in self._group_mgrs[group]:
                    gpu_ids = [
                        int(b) * factor
                        for b in gpu_spec.block_ids[offset:offset + size]
                    ]
                    cpu_ids = [
                        int(b) * factor
                        for b in cpu_spec.block_ids[offset:offset + size]
                    ]
                    segments.append((mgr, gpu_ids, cpu_ids, [factor] * size))
                num_bytes += size * self._group_row_bytes[group]
            offset += size

        sidecar_pending = self._sidecar_involved(group_sizes)
        assert segments or sidecar_pending, (
            "transfer with no blocks in any KV-cache group")

        futures: list = []
        entry = [futures, num_bytes, sidecar_pending, False]
        self._pending[job_id] = entry

        if self.tpu_to_cpu:
            # A store must read (raiden d2h) or gather (sidecar) its
            # source rows only after every program that writes them has
            # finished; the job is stamped with the current engine step
            # and dispatched once that step completes on device (see
            # _install_store_dispatch_hook).
            with self._queue_lock:
                self._queued_transfers.append(
                    (_current_step[0], job_id, transfer_spec, segments,
                     futures, entry, sidecar_pending))
        else:
            # A restore's target blocks may have been freed by the
            # previous step's optimistic (async-scheduling) bookkeeping
            # while that step still executes and writes them; the DMA has
            # no ordering against TensorCore programs, so defer the h2d
            # until the step in flight at queue time has completed. Steps
            # scheduled after the blocks were reserved never touch them.
            with self._queue_lock:
                self._queued_transfers.append(
                    (_current_step[0] - 1, job_id, transfer_spec, segments,
                     futures, entry, sidecar_pending))
        return True

    def dispatch_queued_stores(self, completed_step: int) -> None:
        """Issue queued transfers whose stamped step has completed on
        device: the sidecar's part, raiden d2h store DMAs, and raiden h2d
        restore DMAs.

        The DMA starts immediately and moves bytes unsynchronized against
        device programs, so a job may only dispatch once every program
        through its stamped step has finished (d2h: the writes it reads;
        h2d: writes into blocks its targets may have been recycled from).
        """
        with self._queue_lock:
            ready = [
                job for job in self._queued_transfers
                if job[0] <= completed_step
            ]
            if not ready:
                return
            self._queued_transfers = [
                job for job in self._queued_transfers
                if job[0] > completed_step
            ]
        for (_, job_id, transfer_spec, segments, futures, entry,
             sidecar_pending) in ready:
            if sidecar_pending:
                assert self._sidecar.transfer_async(job_id, transfer_spec)
            for mgr, gpu_ids, cpu_ids, sizes in segments:
                if self.tpu_to_cpu:
                    futures.append(mgr.d2h(gpu_ids, cpu_ids, sizes))
                else:
                    futures.append(mgr.h2d(cpu_ids, gpu_ids, sizes))
            entry[3] = True

    def get_finished(self) -> list[TransferResult]:
        # Drain the sidecar first; its per-job results gate composite
        # completion.
        if self._sidecar is not None:
            for r in self._sidecar.get_finished():
                entry = self._pending.get(r.job_id)
                if entry is None:
                    continue
                entry[2] = False
                entry[1] += r.transfer_size or 0
                if not r.success:
                    logger.error(
                        "[kv-offload] sidecar transfer failed for job=%s",
                        r.job_id)

        results: list[TransferResult] = []
        for job_id in list(self._pending.keys()):
            futures, num_bytes, sidecar_pending, dispatched = (
                self._pending[job_id])
            if sidecar_pending or not dispatched:
                continue
            # Poll-only completion: is_ready() True means the DMA finished.
            # Do NOT call fut.wait()/Await here -- PJRT_Event_Await (and the
            # xla::Future BlockUntilReady it also drains) can deadlock
            # against the concurrently-executing model. The disagg connector
            # likewise reports completion by polling (complete_read) and
            # never blocks.
            if not all(fut.is_ready() for fut in futures):
                continue
            # ok() is a non-blocking error probe (no Await), valid once
            # is_ready() is True; it tells a successful transfer from one
            # that completed with an error.
            ok = all(fut.ok() for fut in futures)
            if not ok:
                errs = [fut.error_message() for fut in futures if not fut.ok()]
                logger.error("[kv-offload] Raiden transfer job=%s failed: %s",
                             job_id, errs)
            results.append(
                TransferResult(
                    job_id=job_id,
                    success=ok,
                    transfer_size=num_bytes,
                    transfer_time=1e-9,
                    transfer_type=self.transfer_type,
                ))
            # Completed: drop the futures (their keep-alives release the
            # buffers).
            del self._pending[job_id]
        return results

    def wait(self, job_ids: set[int]) -> None:
        """Block until the named transfers complete (flush primitive).

        Mirrors SingleDirectionOffloadingHandler.wait: cleanup (popping
        `_pending`, emitting TransferResults) is left to the next
        `get_finished()`, which the scheduler relies on to mark jobs done.
        """
        # Dispatch whatever has become eligible before polling (before
        # sidecar.wait, which polls only already-submitted sidecar
        # transfers). Jobs stamped with a still-in-flight step become
        # eligible inside the poll loop below, as the step's copy
        # resolves concurrently.
        self.dispatch_queued_stores(_last_completed_step[0])
        if self._sidecar is not None:
            self._sidecar.wait(job_ids)
        # Flush by POLLING is_ready (never the blocking Await, which can
        # deadlock against the live model -- see get_finished). Bound the wait
        # with a dedicated offload timeout so a stalled raiden DMA can't
        # busy-poll forever; on timeout the jobs stay in _pending and are
        # reported by a later get_finished().
        remaining = {jid for jid in job_ids if jid in self._pending}
        deadline = time.perf_counter() + _RAIDEN_OFFLOAD_WAIT_TIMEOUT_S
        while remaining:
            self.dispatch_queued_stores(_last_completed_step[0])
            for jid in list(remaining):
                entry = self._pending.get(jid)
                if entry is None or (entry[3] and all(f.is_ready()
                                                      for f in entry[0])):
                    remaining.discard(jid)
            if not remaining:
                break
            if time.perf_counter() >= deadline:
                logger.warning(
                    "[kv-offload] Raiden wait timed out after %.1fs for "
                    "jobs=%s", _RAIDEN_OFFLOAD_WAIT_TIMEOUT_S,
                    sorted(remaining))
                return
            time.sleep(0.0005)

    def shutdown(self) -> None:
        _deferred_store_handlers.discard(self)
        if self._sidecar is not None:
            self._sidecar.shutdown()


# ---------------------------------------------------------------------------
# Factory: allocates CPU pinned tensors, creates bidirectional handlers
# ---------------------------------------------------------------------------


class CpuTpuOffloadingHandlers:
    """Mirrors CpuGpuOffloadingHandlers for TPU.

    Every canonical KV-cache tensor — an attention KV array or a mamba/GDN
    state array, viewed as int8 `(num_blocks, row_bytes)` — gets a matching
    CPU pool tensor; per-group tensor indices route each KV-cache group's
    block ids to only that group's arrays.
    """

    def __init__(
        self,
        *,
        kv_caches: CanonicalKVCaches,
        num_cpu_blocks: int,
    ):
        """
        Args:
          kv_caches: TPU-canonicalized caches (see
              `build_tpu_canonical_kv_caches`) — each .tensor is a 2D int8
              view (num_blocks, row_bytes) sharing storage with the
              model's kv_cache.
          num_cpu_blocks: number of CPU-side blocks; CPU block ids share
              one id space across KV-cache groups, mirroring the device
              block pool.
        """
        assert kv_caches.tensors

        tpu_tensors: list[torch.Tensor] = []
        for kv_cache_tensor in kv_caches.tensors:
            int8_view = kv_cache_tensor.tensor
            assert int8_view.dtype == torch.int8 and int8_view.ndim == 2
            tpu_tensors.append(int8_view)

        group_tensor_idxs: list[list[int]] = []
        for refs in kv_caches.group_data_refs:
            idxs: list[int] = []
            for ref in refs:
                if ref.tensor_idx not in idxs:
                    idxs.append(ref.tensor_idx)
            group_tensor_idxs.append(idxs)

        if _USE_RAIDEN_OFFLOAD:
            # raiden owns a DmaMapped host pool (no cpu_pool tensor). A
            # d2h/h2d call copies the addressed rows of every tensor its
            # manager registered, so build one KVCacheManager per KV-cache
            # group over that group's device views; the two direct-DMA
            # handlers share the manager list. host_blocks_to_allocate
            # matches the scheduler's CPU pool in every group, so CPU
            # block IDs map 1:1 to raiden host slots (one id space,
            # mirroring the device block pool).
            #
            # raiden addresses rows by a per-buffer major-dim slice
            # computed from the on-device layout; when a tensor's block
            # dim lands inside the tile (GDN conv state), that slice
            # degenerates to the whole buffer and both host sizing and
            # transfers are wrong. Classify each distinct (shape, dtype)
            # class by probing a one-block manager and comparing raiden's
            # reported slice with nbytes/num_blocks; misfit tensors ride a
            # host-staged sidecar instead.
            from tpu_raiden.api.torch import kv_cache_manager as _kcm

            def _source_major_view(tensor: torch.Tensor) -> torch.Tensor:
                """Reshape the canonical view to the source array's major
                (kernel-block) granularity — raiden validates the leading
                dim against the underlying buffer."""
                major = _SOURCE_MAJOR_DIM_BY_STORAGE.get(
                    tensor.untyped_storage().data_ptr(), tensor.shape[0])
                return tensor.view(major, -1)

            def _rows_per_block(tensor: torch.Tensor) -> int:
                major = _SOURCE_MAJOR_DIM_BY_STORAGE.get(
                    tensor.untyped_storage().data_ptr(), tensor.shape[0])
                assert major % tensor.shape[0] == 0
                return major // tensor.shape[0]

            def _row_addressable(tensor: torch.Tensor) -> bool:
                view = _source_major_view(tensor)
                try:
                    probe = _kcm.KVCacheManager(
                        [[view]],
                        local_control_port=0,
                        host_blocks_to_allocate=1,
                        unsafe_skip_buffer_lock=True,
                        parallelism=1,
                    )
                except ValueError as e:
                    # raiden rejects layouts its raw row transfer cannot
                    # address; those tensors ride the sidecar.
                    logger.info(
                        "[kv-offload] raiden rejected tensor registration "
                        "(%s); routing via host-staged sidecar.", e)
                    return False
                slice_bytes = int(probe._impl.slice_byte_size)
                del probe
                return slice_bytes == view.nbytes // view.shape[0]

            addressable_by_class: dict[tuple, bool] = {}
            raiden_side: list[bool] = []
            for t in tpu_tensors:
                key = (tuple(t.shape), t.dtype, _rows_per_block(t))
                if key not in addressable_by_class:
                    addressable_by_class[key] = _row_addressable(t)
                    if not addressable_by_class[key]:
                        logger.info(
                            "[kv-offload] tensor class %s not raiden "
                            "row-addressable; routing via host-staged "
                            "sidecar.", key)
                raiden_side.append(addressable_by_class[key])

            # raiden holds raw pointers / PJRT aliases to these device
            # buffers for its lifetime. Keep the tensors (and thus the
            # underlying buffers) alive on the factory, or they get GC'd
            # after __init__ and raiden's d2h/h2d dereferences freed
            # buffers (segfault).
            self._raiden_device_tensors = []
            self._raiden_group_mgrs = []
            group_row_bytes: list[int] = []
            host_bytes = 0
            sidecar_tensor_ids: list[int] = []
            sidecar_group_idxs: list[list[int]] = []
            sidecar_group_has_work: list[bool] = []
            for tensor_idxs in group_tensor_idxs:
                raiden_idxs = [i for i in tensor_idxs if raiden_side[i]]
                torch_idxs = [i for i in tensor_idxs if not raiden_side[i]]
                remapped: list[int] = []
                for i in torch_idxs:
                    if i not in sidecar_tensor_ids:
                        sidecar_tensor_ids.append(i)
                    remapped.append(sidecar_tensor_ids.index(i))
                sidecar_group_idxs.append(remapped)
                sidecar_group_has_work.append(bool(remapped))

                # A manager sizes and addresses every registered layer
                # with layer 0's slice, so partition this group's tensors
                # into row classes of identical (kernel-row factor,
                # per-row slice bytes) — one manager per class.
                row_classes: dict[tuple[int, int], list[int]] = {}
                for i in raiden_idxs:
                    view = _source_major_view(tpu_tensors[i])
                    key = (_rows_per_block(tpu_tensors[i]),
                           int(view.nbytes // view.shape[0]))
                    row_classes.setdefault(key, []).append(i)
                submgrs: list[tuple] = []
                for (factor, slice_bytes), idxs in sorted(row_classes.items()):
                    device_tensors = [[_source_major_view(tpu_tensors[i])]
                                      for i in idxs]
                    self._raiden_device_tensors.append(device_tensors)
                    submgrs.append((
                        _kcm.KVCacheManager(
                            device_tensors,
                            local_control_port=0,
                            host_blocks_to_allocate=num_cpu_blocks * factor,
                            unsafe_skip_buffer_lock=True,
                            # Local offload copies through the DmaMapped
                            # pool; the wrapper's parallelism default of 4
                            # multiplies per-layer host buffering for
                            # parallel network streams that local D2h/H2d
                            # never uses.
                            parallelism=1,
                        ),
                        factor))
                    host_bytes += (num_cpu_blocks * factor * slice_bytes *
                                   len(idxs))
                self._raiden_group_mgrs.append(submgrs)
                row_bytes = sum(
                    int(tpu_tensors[i].shape[1] *
                        tpu_tensors[i].element_size()) for i in raiden_idxs)
                group_row_bytes.append(row_bytes)

            d2h_sidecar = h2d_sidecar = None
            if sidecar_tensor_ids:
                sidecar_device = [tpu_tensors[i] for i in sidecar_tensor_ids]
                sidecar_cpu = [
                    torch.zeros((num_cpu_blocks, t.shape[1]),
                                dtype=torch.int8,
                                device="cpu") for t in sidecar_device
                ]
                sidecar_bytes = sum(c.numel() for c in sidecar_cpu)
                logger.info(
                    "[kv-offload] sidecar host pool: %d tensors x %d "
                    "blocks (%.2f GiB unpinned)", len(sidecar_cpu),
                    num_cpu_blocks, sidecar_bytes / (1 << 30))
                d2h_sidecar = SingleDirectionOffloadingHandler(
                    src_tensors=sidecar_device,
                    dst_tensors=sidecar_cpu,
                    group_tensor_idxs=sidecar_group_idxs,
                    tpu_to_cpu=True,
                )
                h2d_sidecar = SingleDirectionOffloadingHandler(
                    src_tensors=sidecar_cpu,
                    dst_tensors=sidecar_device,
                    group_tensor_idxs=sidecar_group_idxs,
                    tpu_to_cpu=False,
                )

            self.gpu_to_cpu_handler = _RaidenOffloadingHandler(
                self._raiden_group_mgrs,
                tpu_to_cpu=True,
                group_row_bytes=group_row_bytes,
                sidecar=d2h_sidecar,
                sidecar_group_has_work=sidecar_group_has_work,
            )
            self.cpu_to_gpu_handler = _RaidenOffloadingHandler(
                self._raiden_group_mgrs,
                tpu_to_cpu=False,
                group_row_bytes=group_row_bytes,
                sidecar=h2d_sidecar,
                sidecar_group_has_work=sidecar_group_has_work,
            )
            logger.info(
                "[kv-offload] RAIDEN direct-DMA offload: %d host blocks x "
                "%d KV-cache groups (%.2f GiB DmaMapped host pool, %d "
                "sidecar tensors)",
                num_cpu_blocks,
                len(group_tensor_idxs),
                host_bytes / (1 << 30),
                len(sidecar_tensor_ids),
            )
            return

        # cpu_pool is intentionally NOT pinned. The lazy copy_ in the DMA
        # worker stages through libtpu's premap pool
        # (TPU_PREMAPPED_BUFFER_SIZE); pinning the full cpu_pool would lock
        # hundreds of GB and starve the libtpu premap region.
        cpu_tensors: list[torch.Tensor] = []
        total_bytes = 0
        for t in tpu_tensors:
            cpu_tensor = torch.zeros((num_cpu_blocks, t.shape[1]),
                                     dtype=torch.int8,
                                     device="cpu")
            total_bytes += cpu_tensor.numel()
            cpu_tensors.append(cpu_tensor)

        logger.info(
            "[kv-offload] Allocated %.2f GiB unpinned CPU pool "
            "(%d tensors x %d blocks, %d KV-cache groups)",
            total_bytes / (1 << 30),
            len(cpu_tensors),
            num_cpu_blocks,
            len(group_tensor_idxs),
        )

        self.gpu_to_cpu_handler = SingleDirectionOffloadingHandler(
            src_tensors=tpu_tensors,
            dst_tensors=cpu_tensors,
            group_tensor_idxs=group_tensor_idxs,
            tpu_to_cpu=True,
        )
        self.cpu_to_gpu_handler = SingleDirectionOffloadingHandler(
            src_tensors=cpu_tensors,
            dst_tensors=tpu_tensors,
            group_tensor_idxs=group_tensor_idxs,
            tpu_to_cpu=False,
        )


# ---------------------------------------------------------------------------
# Spec: subclasses CPUOffloadingSpec, overrides get_handlers() for TPU
# ---------------------------------------------------------------------------


class TPUCPUOffloadingSpec(CPUOffloadingSpec):
    """CPU offloading spec for TPU.

    Reuses CPUOffloadingSpec.__init__ (block-size calculation, eviction-policy
    config) and get_manager() (LRU/ARC setup).  Only get_handlers() is
    overridden to create TPU-specific transfer handlers.
    """

    def __init__(self, vllm_config: VllmConfig,
                 kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)
        self._tpu_handlers: CpuTpuOffloadingHandlers | None = None

        # CPUOffloadingSpec sizes num_blocks from vLLM's KVCacheTensor
        # byte accounting, which assumes layers sharing a tensor share its
        # pages. TPU materializes hybrid KV as private per-layer
        # (per-state) arrays whose combined per-block footprint is larger
        # (Qwen3.5: 4x), so that formula over-allocates host RAM by the
        # same factor. Recompute from the bytes the TPU pools actually
        # hold per block row, times the world size (tensor x pipeline,
        # stable across scheduler and worker processes).
        parallel_config = vllm_config.parallel_config
        world_size = (parallel_config.tensor_parallel_size *
                      parallel_config.pipeline_parallel_size)
        row_bytes = tpu_per_rank_block_row_bytes(kv_cache_config)
        cpu_bytes_to_use = int(self.extra_config.get("cpu_bytes_to_use", 0))
        if row_bytes > 0 and cpu_bytes_to_use > 0:
            self.num_blocks = cpu_bytes_to_use // (
                row_bytes * self.block_size_factor * world_size)
            logger.info(
                "[kv-offload] CPU pool: %d blocks "
                "(%.2f MiB/block-row/rank, world_size=%d, %.2f GiB/rank).",
                self.num_blocks, row_bytes / (1 << 20), world_size,
                self.num_blocks * row_bytes * self.block_size_factor /
                (1 << 30))

    @property
    def prewarm_shapes(self) -> list[int]:
        """H2D Pallas-scatter shapes the runner should pre-compile.

        Delegates to the H2D handler, which is the actual owner of the
        Pallas kernels. Returns [] if get_handlers() hasn't run yet —
        TpuMultiprocExecutor calls this AFTER register_kv_caches, so the
        handler is set by then; the guard is defensive.
        """
        if self._tpu_handlers is None:
            return []
        return getattr(self._tpu_handlers.cpu_to_gpu_handler, "prewarm_shapes",
                       [])

    def prewarm_shape(self, p: int) -> None:
        """Pre-compile H2D Pallas scatter for block-count `p`. Delegates
        to the H2D handler; no-op before get_handlers() has run."""
        if self._tpu_handlers is None:
            return
        handler = self._tpu_handlers.cpu_to_gpu_handler
        if hasattr(handler, "prewarm_shape"):
            handler.prewarm_shape(p)

    @classmethod
    def estimate_hbm_reserve_bytes(cls, vllm_config: VllmConfig) -> int:
        """Upper bound on HBM the H2D staging buffer will consume.

        Called by tpu_worker before profile_run to shrink the KV-cache
        budget so vllm leaves room for the SingleDirectionOffloadingHandler's
        _h2d_device_buffer, which is allocated lazily after
        determine_available_memory() returns.

        Only TPUCPUOffloadingSpec needs this reserve; other connectors
        (e.g., TPUConnector for P/D disagg) don't allocate an HBM staging
        buffer, so the worker must NOT shrink their budgets. Living on the
        spec class makes the gating explicit at the call site.
        """
        if _USE_RAIDEN_OFFLOAD:
            # raiden DMAs between the KV cache and its host pool directly;
            # there is no HBM staging buffer to reserve for.
            return 0
        max_pool_blocks = int(os.environ.get("KV_H2D_POOL_MAX_BLOCKS", "2048"))
        max_padded = 1
        while max_padded < max_pool_blocks:
            max_padded <<= 1

        from vllm_torchtpu.layers.vllm.attention import (
            TPU_STR_DTYPE_TO_TORCH_DTYPE, PallasAttentionBackend)

        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config

        # _resolve_kv_cache_dtype is strict on "auto", so resolve it here
        # the same way tpu_worker / tpu_runner already do.
        if cache_config.cache_dtype == "auto":
            model_dtype = model_config.dtype
            kv_dtype = (TPU_STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
                        if isinstance(model_dtype, str) else model_dtype)
        else:
            kv_dtype = TPU_STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        per_block_bytes = PallasAttentionBackend.get_kv_cache_page_size_bytes(
            cache_config.block_size,
            model_config.get_num_kv_heads(parallel_config),
            model_config.get_head_size(),
            kv_dtype,
        )
        # The H2D staging buffer holds `max_padded` rows for every
        # canonical tensor: one per attention layer, plus (on hybrid
        # models) the per-layer mamba/GDN state arrays. Budget attention
        # layers at the attention page and the remaining layers at the
        # per-layer mamba state footprint (config-derived, per rank).
        num_layers = model_config.get_num_layers(parallel_config)
        num_attn_layers = num_layers
        mamba_bytes = 0
        if getattr(model_config, "is_hybrid", False):
            from vllm_torchtpu.platforms.tpu_block_size_utils import \
                _hybrid_mamba_page_size_bytes
            mamba_page = _hybrid_mamba_page_size_bytes(vllm_config)
            if mamba_page is not None:
                num_attn_layers = model_config.get_num_layers_by_block_type(
                    parallel_config, "attention")
                mamba_bytes = (num_layers - num_attn_layers) * mamba_page
        return max_padded * (num_attn_layers * per_block_bytes + mamba_bytes)

    def get_handlers(
        self,
        kv_caches: CanonicalKVCaches,
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec],
                        OffloadingHandler]]:
        if self._tpu_handlers is None:
            gpu_block_sizes = set(self.gpu_block_size)
            assert len(gpu_block_sizes) == 1, (
                "TPU offloading requires a uniform block_size across KV "
                f"cache groups, got {self.gpu_block_size}")
            assert self.block_size_factor == 1, (
                "TPU offloading requires offloaded_block_size == gpu block "
                "size; do not set 'block_size' in kv_connector_extra_config")
            assert len(kv_caches.group_data_refs) == len(
                self.kv_cache_config.kv_cache_groups)

            self._tpu_handlers = CpuTpuOffloadingHandlers(
                kv_caches=kv_caches,
                num_cpu_blocks=self.num_blocks,
            )

        yield (
            GPULoadStoreSpec,
            CPULoadStoreSpec,
            self._tpu_handlers.gpu_to_cpu_handler,
        )
        yield (
            CPULoadStoreSpec,
            GPULoadStoreSpec,
            self._tpu_handlers.cpu_to_gpu_handler,
        )
