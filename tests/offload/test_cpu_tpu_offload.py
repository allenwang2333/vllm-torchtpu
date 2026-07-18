# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the TPU↔CPU KV cache offloading stack.

Covers the pieces that don't require a real TPU:
- `_resolve_kv_cache_dtype` strictness (raise on "auto") and
  `get_kv_cache_shape` probe-path placeholder semantics in
  `vllm_torchtpu.layers.vllm.attention`.
- `TPUCPUOffloadingSpec.estimate_hbm_reserve_bytes` size formula scales
  proportionally with `KV_H2D_POOL_MAX_BLOCKS`.
- `TPUCPUOffloadingSpec.prewarm_shapes` / `prewarm_shape` delegate to the
  H2D handler when `_tpu_handlers` is populated; no-op before.
- `Transfer` dataclass defaults and the error-capture wrapping pattern
  used by `_h2d_task` / `_d2h_task`.
- `transfer_async` H2D chunking math: `chunks_total = ceil(len(src_ids)
  / chunk_size)` at the boundary cases.
- `tpu_worker._estimate_kv_connector_hbm_reserve` dynamic spec resolution
  (no kv_connector / missing metadata / unimportable module → 0).

Heavy dependencies (real Pallas kernels, torch_tpu DMA, vllm config
plumbing) are mocked at construction time so the tests run on a host
without a TPU.
"""
import os
import unittest
from unittest.mock import MagicMock, patch

import torch


class TestResolveKvCacheDtype(unittest.TestCase):
    """attention._resolve_kv_cache_dtype contract."""

    def test_passes_through_torch_dtype(self):
        from vllm_torchtpu.layers.vllm.attention import _resolve_kv_cache_dtype
        self.assertEqual(_resolve_kv_cache_dtype(torch.bfloat16),
                         torch.bfloat16)

    def test_resolves_known_strings(self):
        from vllm_torchtpu.layers.vllm.attention import _resolve_kv_cache_dtype
        self.assertEqual(_resolve_kv_cache_dtype("bfloat16"), torch.bfloat16)
        self.assertEqual(_resolve_kv_cache_dtype("fp8_e4m3"),
                         torch.float8_e4m3fn)
        self.assertEqual(_resolve_kv_cache_dtype("half"), torch.half)

    def test_auto_raises(self):
        from vllm_torchtpu.layers.vllm.attention import _resolve_kv_cache_dtype
        with self.assertRaisesRegex(ValueError, "must be resolved"):
            _resolve_kv_cache_dtype("auto")

    def test_unknown_raises(self):
        from vllm_torchtpu.layers.vllm.attention import _resolve_kv_cache_dtype
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            _resolve_kv_cache_dtype("not-a-dtype")


class TestGetKvCacheShapeProbe(unittest.TestCase):
    """OffloadingConnectorWorker probes get_kv_cache_shape without a dtype
    to discover the num_blocks dimension. The placeholder return must
    preserve num_blocks at position 0 — see
    vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py
    register_kv_caches."""

    def test_auto_returns_placeholder_with_num_blocks_at_dim_0(self):
        from vllm_torchtpu.layers.vllm.attention import PallasAttentionBackend
        shape = PallasAttentionBackend.get_kv_cache_shape(num_blocks=1234,
                                                          block_size=16,
                                                          num_kv_heads=1,
                                                          head_size=256)
        self.assertEqual(shape[0], 1234)
        self.assertEqual(shape.index(1234), 0)

    def test_concrete_dtype_returns_real_shape(self):
        from vllm_torchtpu.layers.vllm.attention import PallasAttentionBackend
        shape = PallasAttentionBackend.get_kv_cache_shape(
            num_blocks=2048,
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            cache_dtype_str="bfloat16")
        self.assertEqual(shape[0], 2048)
        self.assertEqual(shape[1], 16)  # block_size


class TestEstimateHbmReserveBytes(unittest.TestCase):
    """TPUCPUOffloadingSpec.estimate_hbm_reserve_bytes scales linearly
    with both num_layers and KV_H2D_POOL_MAX_BLOCKS."""

    @staticmethod
    def _make_vllm_config(num_layers=32,
                          num_kv_heads=8,
                          head_size=128,
                          block_size=16):
        cfg = MagicMock()
        cfg.model_config.get_num_layers.return_value = num_layers
        cfg.model_config.get_num_kv_heads.return_value = num_kv_heads
        cfg.model_config.get_head_size.return_value = head_size
        cfg.model_config.dtype = "bfloat16"
        cfg.model_config.is_hybrid = False
        cfg.cache_config.block_size = block_size
        cfg.cache_config.cache_dtype = "auto"  # resolved via model_dtype
        return cfg

    def test_doubling_pool_doubles_reserve(self):
        from vllm_torchtpu.offload.cpu_tpu import TPUCPUOffloadingSpec
        cfg = self._make_vllm_config()
        with patch.dict(os.environ, {"KV_H2D_POOL_MAX_BLOCKS": "1024"}):
            half = TPUCPUOffloadingSpec.estimate_hbm_reserve_bytes(cfg)
        with patch.dict(os.environ, {"KV_H2D_POOL_MAX_BLOCKS": "2048"}):
            full = TPUCPUOffloadingSpec.estimate_hbm_reserve_bytes(cfg)
        self.assertEqual(full, 2 * half)

    def test_doubling_layers_doubles_reserve(self):
        from vllm_torchtpu.offload.cpu_tpu import TPUCPUOffloadingSpec
        with patch.dict(os.environ, {"KV_H2D_POOL_MAX_BLOCKS": "2048"}):
            small = TPUCPUOffloadingSpec.estimate_hbm_reserve_bytes(
                self._make_vllm_config(num_layers=32))
            large = TPUCPUOffloadingSpec.estimate_hbm_reserve_bytes(
                self._make_vllm_config(num_layers=64))
        self.assertEqual(large, 2 * small)

    def test_pool_blocks_rounds_to_next_power_of_2(self):
        from vllm_torchtpu.offload.cpu_tpu import TPUCPUOffloadingSpec
        cfg = self._make_vllm_config()
        with patch.dict(os.environ, {"KV_H2D_POOL_MAX_BLOCKS": "1500"}):
            reserve_1500 = TPUCPUOffloadingSpec.estimate_hbm_reserve_bytes(cfg)
        with patch.dict(os.environ, {"KV_H2D_POOL_MAX_BLOCKS": "2048"}):
            reserve_2048 = TPUCPUOffloadingSpec.estimate_hbm_reserve_bytes(cfg)
        # 1500 rounds up to 2048
        self.assertEqual(reserve_1500, reserve_2048)

    def test_raiden_defaults_to_zero_reserve(self):
        from vllm_torchtpu.offload.cpu_tpu import TPUCPUOffloadingSpec
        cfg = self._make_vllm_config()
        # Temporarily pop global env variable so we can test the fallback path
        # when it is absent. We use try/finally to ensure it is restored even
        # if the test assertions fail (avoiding environment leakage to other tests).
        orig_max_blocks = os.environ.pop("KV_H2D_POOL_MAX_BLOCKS", None)
        try:
            with patch("vllm_torchtpu.offload.cpu_tpu._USE_RAIDEN_OFFLOAD",
                       True):
                raiden_reserve = TPUCPUOffloadingSpec.estimate_hbm_reserve_bytes(
                    cfg)
            self.assertEqual(raiden_reserve, 0)
        finally:
            if orig_max_blocks is not None:
                os.environ["KV_H2D_POOL_MAX_BLOCKS"] = orig_max_blocks


class TestPrewarmDelegation(unittest.TestCase):
    """TPUCPUOffloadingSpec.prewarm_shapes / prewarm_shape forward to the
    H2D handler. Before get_handlers() runs (_tpu_handlers is None) they
    are no-ops so the runner's collective_rpc doesn't crash."""

    def test_prewarm_shapes_none_when_handlers_not_built(self):
        from vllm_torchtpu.offload.cpu_tpu import TPUCPUOffloadingSpec
        spec = TPUCPUOffloadingSpec.__new__(TPUCPUOffloadingSpec)
        spec._tpu_handlers = None
        self.assertEqual(spec.prewarm_shapes, [])
        spec.prewarm_shape(2048)  # must not raise

    def test_prewarm_shapes_delegates_to_h2d_handler(self):
        from vllm_torchtpu.offload.cpu_tpu import TPUCPUOffloadingSpec
        spec = TPUCPUOffloadingSpec.__new__(TPUCPUOffloadingSpec)
        spec._tpu_handlers = MagicMock()
        spec._tpu_handlers.cpu_to_gpu_handler.prewarm_shapes = [1, 2, 4, 8]
        self.assertEqual(spec.prewarm_shapes, [1, 2, 4, 8])

    def test_prewarm_shape_forwards_argument(self):
        from vllm_torchtpu.offload.cpu_tpu import TPUCPUOffloadingSpec
        spec = TPUCPUOffloadingSpec.__new__(TPUCPUOffloadingSpec)
        spec._tpu_handlers = MagicMock()
        spec.prewarm_shape(512)
        spec._tpu_handlers.cpu_to_gpu_handler.prewarm_shape.\
            assert_called_once_with(512)


class TestTransferDataclass(unittest.TestCase):
    """Transfer's chunk-state fields and error-capture pattern."""

    def test_defaults(self):
        from vllm_torchtpu.offload.cpu_tpu import Transfer
        t = Transfer(job_id=1, num_bytes=100, dma_done=None)
        self.assertIsNone(t.device_buffer)
        self.assertIsNone(t.chunk_descs)
        self.assertIsNone(t.current_scatters)
        self.assertFalse(t.buffer_acquired)
        self.assertEqual(t.chunks_total, 0)
        self.assertEqual(t.chunks_done, 0)
        self.assertIsNone(t.error)

    def test_error_capture_pattern(self):
        """The wrapper pattern in _h2d_task / _d2h_task sets t.error
        before re-raising so get_finished can report success=False."""
        from vllm_torchtpu.offload.cpu_tpu import Transfer
        t = Transfer(job_id=1, num_bytes=100, dma_done=None)
        try:
            try:
                raise RuntimeError("simulated DMA failure")
            except BaseException as e:
                t.error = e
                raise
        except RuntimeError:
            pass
        self.assertIsNotNone(t.error)
        self.assertIsInstance(t.error, RuntimeError)
        self.assertEqual(str(t.error), "simulated DMA failure")


class TestChunkingMath(unittest.TestCase):
    """transfer_async H2D branch emits `ceil(group_len / chunk_cap)`
    chunks per non-empty KV-cache group. Exercise boundary cases at the
    arithmetic level without spinning up a real handler (which would need
    a TPU device for the staging buffer)."""

    @staticmethod
    def _chunks(n_ids: int, chunk_cap: int) -> int:
        return len(range(0, n_ids, chunk_cap)) if n_ids else 0

    def test_single_chunk_when_fits(self):
        for n in (1, 100, 2047, 2048):
            self.assertEqual(self._chunks(n, 2048), 1, f"n={n}")

    def test_multi_chunk_when_oversized(self):
        self.assertEqual(self._chunks(2049, 2048), 2)
        self.assertEqual(self._chunks(4096, 2048), 2)
        self.assertEqual(self._chunks(4097, 2048), 3)
        self.assertEqual(self._chunks(5120, 2048), 3)

    def test_smaller_pool_more_chunks(self):
        self.assertEqual(self._chunks(2048, 512), 4)
        self.assertEqual(self._chunks(2049, 512), 5)


class TestMultiChunkH2D(unittest.TestCase):
    """Chunked H2D loads must progress chunk-by-chunk through the
    `get_finished` / `flush_pending_scatters` cycle even before the
    owning request enters `scatter_now_req_ids`.

    The owning request is in WAITING_FOR_REMOTE_KVS until the LAST
    chunk's `finished_recving` is reported. That means the req_id is
    *not* in `scheduler_output.num_scheduled_tokens` (the source of
    `scatter_now_req_ids`) for any intermediate chunk. So
    `flush_pending_scatters` MUST treat intermediate chunks as eligible
    for dispatch independently of the scheduler-gate set — otherwise no
    chunked transfer can ever complete.

    The test drives the handler's chunk FSM directly without TPU
    dependencies: `__init__` is bypassed (it would allocate the
    device-staging buffer via `torch.zeros`) and `_start_h2d_dma` is
    stubbed to immediately signal `dma_done`.
    """

    def _make_handler(self):
        """Construct a barely-initialized H2D handler — just enough state
        for `get_finished` and `flush_pending_scatters` to run."""
        import threading
        from collections import deque

        import vllm_torchtpu.offload.cpu_tpu as cpu_tpu
        from vllm_torchtpu.offload.cpu_tpu import \
            SingleDirectionOffloadingHandler

        sync_patch = patch.object(cpu_tpu, "_tpu_sync", lambda *a, **k: None)
        sync_patch.start()
        self.addCleanup(sync_patch.stop)

        h = SingleDirectionOffloadingHandler.__new__(
            SingleDirectionOffloadingHandler)
        h.tpu_to_cpu = False
        h.src_tensors = [MagicMock()]
        h.dst_tensors = [MagicMock()]
        h._transfer_map = {}
        h._transfers = deque()
        h._req_id_by_job_id = {}
        h._h2d_buffer_free = threading.Event()
        h._h2d_buffer_free.set()
        h._pending_scatters = []
        h._dma_worker = MagicMock()
        h.transfer_type = ("CPU", "GPU")
        # Record chunks the handler dispatches.
        h._dispatched_chunks: list[int] = []

        def fake_start(t):
            h._dispatched_chunks.append(t.chunks_done)
            ev = threading.Event()
            ev.set()
            t.dma_done = ev
            dst_ids = MagicMock()
            dst_ids.to = lambda dtype: MagicMock()
            t.current_scatters = [(part.tensor_idxs, dst_ids, [MagicMock()])
                                  for part in t.chunk_descs[t.chunks_done]]

        h._start_h2d_dma = fake_start
        return h

    def _build_chunked_transfer(self, n_ids: int, req_id: str = "req-A"):
        import numpy as np

        from vllm_torchtpu.offload.cpu_tpu import Transfer, _H2DPart
        chunk_descs = [[
            _H2DPart(
                tensor_idxs=[0],
                src_ids=np.arange(s, min(s + 2, n_ids), dtype=np.int64),
                dst_ids=np.arange(s, min(s + 2, n_ids), dtype=np.int64),
            )
        ] for s in range(0, n_ids, 2)]
        return Transfer(
            job_id=1,
            num_bytes=0,
            dma_done=None,
            chunk_descs=chunk_descs,
            chunks_total=len(chunk_descs),
            chunks_done=0,
            req_id=req_id,
        )

    def test_intermediate_chunks_dispatch_without_scheduler_gate(self):
        """Each intermediate chunk must dispatch before `finished_recving`
        is reported (otherwise the owning request never gets promoted and
        the LAST chunk never lands). Only the LAST chunk is subject to
        `scatter_now_req_ids`.

        Contract: starting from chunk 0 in flight, looping
        `get_finished` + `flush_pending_scatters(empty_set)` must
        eventually dispatch every chunk and report `finished_recving` on
        the last one."""
        h = self._make_handler()
        t = self._build_chunked_transfer(n_ids=8)
        h._transfer_map[t.job_id] = t
        h._transfers.append(t)
        h._start_h2d_dma(t)  # chunk 0 in flight

        empty_scatter_now: set[str] = set()
        for _ in range(t.chunks_total + 2):
            results = h.get_finished()
            h.flush_pending_scatters(empty_scatter_now)
            if results:
                self.assertEqual(results[0].job_id, t.job_id)
                self.assertEqual(t.chunks_done, t.chunks_total)
                self.assertEqual(h._dispatched_chunks,
                                 list(range(t.chunks_total)))
                return

        self.fail(
            "Chunked transfer stalled: dispatched only chunks "
            f"{h._dispatched_chunks} of {t.chunks_total} total; "
            f"{len(h._pending_scatters)} chunks left in `_pending_scatters`. "
            "Intermediate chunks were not eligible for flush despite the "
            "owning req_id correctly being absent from `scatter_now_req_ids` "
            "(the request is still in WAITING_FOR_REMOTE_KVS).")


class TestBuildTpuCanonicalKvCaches(unittest.TestCase):
    """build_tpu_canonical_kv_caches turns per-layer private arrays
    (attention: one tensor; mamba: one tensor per state) into int8
    (num_blocks, row_bytes) canonical tensors with per-group refs."""

    @staticmethod
    def _make_kv_cache_config(num_blocks, groups):
        cfg = MagicMock()
        cfg.num_blocks = num_blocks
        group_mocks = []
        for layer_names in groups:
            g = MagicMock()
            g.layer_names = layer_names
            group_mocks.append(g)
        cfg.kv_cache_groups = group_mocks
        return cfg

    def test_hybrid_grouping_and_row_bytes(self):
        from vllm_torchtpu.offload.cpu_tpu import build_tpu_canonical_kv_caches
        num_blocks = 4
        # 1 attention group (2 layers) + 1 mamba group (2 layers x 2 states).
        kv_caches = {
            "attn.0":
            torch.zeros((num_blocks, 8, 2, 4), dtype=torch.bfloat16),
            "attn.1":
            torch.zeros((num_blocks, 8, 2, 4), dtype=torch.bfloat16),
            "mamba.0": [
                torch.zeros((num_blocks, 16, 3), dtype=torch.bfloat16),
                torch.zeros((num_blocks, 2, 8, 8), dtype=torch.float32),
            ],
            "mamba.1": [
                torch.zeros((num_blocks, 16, 3), dtype=torch.bfloat16),
                torch.zeros((num_blocks, 2, 8, 8), dtype=torch.float32),
            ],
        }
        cfg = self._make_kv_cache_config(
            num_blocks, [["attn.0", "attn.1"], ["mamba.0", "mamba.1"]])
        canonical = build_tpu_canonical_kv_caches(cfg, kv_caches)

        self.assertEqual(len(canonical.tensors), 6)  # 2 attn + 2x2 states
        self.assertEqual(len(canonical.group_data_refs), 2)
        attn_row = 8 * 2 * 4 * 2  # bf16
        conv_row = 16 * 3 * 2  # bf16
        ssm_row = 2 * 8 * 8 * 4  # fp32
        self.assertEqual(
            [t.page_size_bytes for t in canonical.tensors],
            [attn_row, attn_row, conv_row, ssm_row, conv_row, ssm_row])
        for t in canonical.tensors:
            self.assertEqual(t.tensor.dtype, torch.int8)
            self.assertEqual(tuple(t.tensor.shape),
                             (num_blocks, t.page_size_bytes))
        self.assertEqual([r.tensor_idx for r in canonical.group_data_refs[0]],
                         [0, 1])
        self.assertEqual([r.tensor_idx for r in canonical.group_data_refs[1]],
                         [2, 3, 4, 5])

    def test_rejects_shared_storage_view(self):
        from vllm_torchtpu.offload.cpu_tpu import build_tpu_canonical_kv_caches
        raw = torch.zeros(1024, dtype=torch.int8)
        view = raw.view(torch.bfloat16)[16:].view(4, -1)
        cfg = self._make_kv_cache_config(4, [["attn.0"]])
        with self.assertRaisesRegex(NotImplementedError, "shared storage"):
            build_tpu_canonical_kv_caches(cfg, {"attn.0": view})

    def test_rejects_compact_mamba_layout(self):
        from vllm_torchtpu.offload.cpu_tpu import build_tpu_canonical_kv_caches
        cfg = self._make_kv_cache_config(4, [["mamba.0"]])
        # 3 slots against num_blocks=4: nbytes not divisible by num_blocks.
        state = torch.zeros((3, 5), dtype=torch.int8)
        with self.assertRaisesRegex(NotImplementedError, "Compact-mamba"):
            build_tpu_canonical_kv_caches(cfg, {"mamba.0": [state]})


class TestGroupAwareRoundtrip(unittest.TestCase):
    """CPU-only store→clobber→reload roundtrip through both handlers with
    two KV-cache groups, driving the real transfer_async / get_finished /
    flush_pending_scatters machinery (device tensors simulated on CPU,
    `_tpu_sync` stubbed)."""

    NUM_BLOCKS = 16
    NUM_CPU_BLOCKS = 8

    def _make_handlers(self):
        import vllm_torchtpu.offload.cpu_tpu as cpu_tpu

        # Group 0: two "attention" tensors; group 1: one "state" tensor.
        torch.manual_seed(0)
        device_tensors = [
            torch.randint(-128, 127, (self.NUM_BLOCKS, 32), dtype=torch.int8)
            for _ in range(2)
        ] + [
            torch.randint(-128, 127, (self.NUM_BLOCKS, 48), dtype=torch.int8)
        ]
        cpu_tensors = [
            torch.zeros((self.NUM_CPU_BLOCKS, t.shape[1]), dtype=torch.int8)
            for t in device_tensors
        ]
        group_tensor_idxs = [[0, 1], [2]]

        sync_patch = patch.object(cpu_tpu, "_tpu_sync", lambda *a, **k: None)
        sync_patch.start()
        self.addCleanup(sync_patch.stop)

        d2h = cpu_tpu.SingleDirectionOffloadingHandler(
            src_tensors=device_tensors,
            dst_tensors=cpu_tensors,
            group_tensor_idxs=group_tensor_idxs,
            tpu_to_cpu=True,
        )
        h2d = cpu_tpu.SingleDirectionOffloadingHandler(
            src_tensors=cpu_tensors,
            dst_tensors=device_tensors,
            group_tensor_idxs=group_tensor_idxs,
            tpu_to_cpu=False,
        )
        self.addCleanup(d2h.shutdown)
        self.addCleanup(h2d.shutdown)
        return device_tensors, cpu_tensors, d2h, h2d

    @staticmethod
    def _specs(gpu_ids, cpu_ids, group_sizes, block_indices):
        from vllm.v1.kv_offload.base import GPULoadStoreSpec
        from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
        gpu_spec = GPULoadStoreSpec(gpu_ids,
                                    group_sizes=group_sizes,
                                    block_indices=block_indices)
        cpu_spec = CPULoadStoreSpec(cpu_ids)
        return gpu_spec, cpu_spec

    def _drain(self, handler, job_id, max_iters=20, flush_req_ids=None):
        """Run the get_finished / flush cycle until job_id completes."""
        if flush_req_ids is None:
            flush_req_ids = {"req-A"}
        for _ in range(max_iters):
            results = handler.get_finished()
            if not handler.tpu_to_cpu:
                handler.flush_pending_scatters(flush_req_ids)
            for r in results:
                if r.job_id == job_id:
                    return r
            import time
            time.sleep(0.01)
        self.fail(f"job {job_id} did not complete")

    def test_two_group_store_reload_preserves_rows(self):
        device_tensors, cpu_tensors, d2h, h2d = self._make_handlers()

        # Store: group 0 gpu blocks [3, 7] -> cpu [0, 1];
        #        group 1 gpu block [5]     -> cpu [2].
        gpu_ids = [3, 7, 5]
        cpu_ids = [0, 1, 2]
        gpu_spec, cpu_spec = self._specs(gpu_ids,
                                         cpu_ids,
                                         group_sizes=[2, 1],
                                         block_indices=[0, 0])
        originals = [t.clone() for t in device_tensors]
        self.assertTrue(d2h.transfer_async(1, (gpu_spec, cpu_spec)))
        r = self._drain(d2h, 1)
        self.assertTrue(r.success)

        # Group-0 rows land in tensors 0/1 only; group-1 row in tensor 2.
        for i in (0, 1):
            torch.testing.assert_close(cpu_tensors[i][0], originals[i][3])
            torch.testing.assert_close(cpu_tensors[i][1], originals[i][7])
        torch.testing.assert_close(cpu_tensors[2][2], originals[2][5])

        # Clobber the stored device rows, then reload into fresh slots:
        # group 0 cpu [0, 1] -> gpu [10, 12]; group 1 cpu [2] -> gpu [9].
        for i in (0, 1):
            device_tensors[i][3] = -1
            device_tensors[i][7] = -1
        device_tensors[2][5] = -1

        h2d._req_id_by_job_id[2] = "req-A"
        gpu_spec, cpu_spec = self._specs([10, 12, 9], [0, 1, 2],
                                         group_sizes=[2, 1],
                                         block_indices=[0, 0])
        self.assertTrue(h2d.transfer_async(2, (cpu_spec, gpu_spec)))
        r = self._drain(h2d, 2)
        self.assertTrue(r.success)

        for i in (0, 1):
            torch.testing.assert_close(device_tensors[i][10], originals[i][3])
            torch.testing.assert_close(device_tensors[i][12], originals[i][7])
        torch.testing.assert_close(device_tensors[2][9], originals[2][5])
        # Untouched rows keep their original contents.
        torch.testing.assert_close(device_tensors[2][10], originals[2][10])

    def test_concurrent_multichunk_loads_complete(self):
        """Two multi-chunk loads in flight at once must both complete:
        the depth-1 staging buffer interleaves their chunks, so
        get_finished must register completed chunks anywhere in the
        queue, not just at its head."""
        with patch.dict(os.environ, {"KV_H2D_POOL_MAX_BLOCKS": "2"}):
            device_tensors, cpu_tensors, d2h, h2d = self._make_handlers()

        # Seed the cpu pool: group-0 rows 0-4 and group-1 row 5.
        originals = [t.clone() for t in device_tensors]
        gpu_spec, cpu_spec = self._specs([0, 1, 2, 3, 4, 5],
                                         [0, 1, 2, 3, 4, 5],
                                         group_sizes=[5, 1],
                                         block_indices=[0, 0])
        self.assertTrue(d2h.transfer_async(1, (gpu_spec, cpu_spec)))
        self._drain(d2h, 1)

        # Two loads, each 5 group-0 blocks (3 chunks at cap 2) + 1
        # group-1 block, into disjoint fresh slots.
        h2d._req_id_by_job_id[2] = "req-A"
        spec_a = self._specs([8, 9, 10, 11, 12, 13], [0, 1, 2, 3, 4, 5],
                             group_sizes=[5, 1],
                             block_indices=[0, 0])
        h2d._req_id_by_job_id[3] = "req-B"
        spec_b = self._specs([14, 15, 8, 9, 10, 11], [0, 1, 2, 3, 4, 5],
                             group_sizes=[5, 1],
                             block_indices=[0, 0])
        self.assertTrue(h2d.transfer_async(2, (spec_a[1], spec_a[0])))
        self.assertTrue(h2d.transfer_async(3, (spec_b[1], spec_b[0])))

        import time
        done: set[int] = set()
        for _ in range(100):
            for r in h2d.get_finished():
                self.assertTrue(r.success)
                done.add(r.job_id)
            h2d.flush_pending_scatters({"req-A", "req-B"})
            if done == {2, 3}:
                break
            time.sleep(0.01)
        self.assertEqual(
            done, {2, 3}, f"stuck: {len(h2d._transfers)} transfers queued, "
            f"{len(h2d._pending_scatters)} pending scatters")
        # Job 3 wrote group-1 cpu row 5 into device slot 11 last.
        torch.testing.assert_close(device_tensors[2][11], originals[2][5])
        torch.testing.assert_close(device_tensors[0][14], originals[0][0])

    def test_aborted_request_releases_staging_buffer(self):
        """A gated last-chunk scatter of an aborted request must be
        dropped and the staging buffer released, or every later load
        wedges on buffer_free.wait()."""
        device_tensors, cpu_tensors, d2h, h2d = self._make_handlers()
        gpu_spec, cpu_spec = self._specs([4], [3],
                                         group_sizes=[1, 0],
                                         block_indices=[0, 0])
        self.assertTrue(d2h.transfer_async(1, (gpu_spec, cpu_spec)))
        self._drain(d2h, 1)

        h2d._req_id_by_job_id[2] = "req-dead"
        gpu_spec, cpu_spec = self._specs([10], [3],
                                         group_sizes=[1, 0],
                                         block_indices=[0, 0])
        self.assertTrue(h2d.transfer_async(2, (cpu_spec, gpu_spec)))
        r = self._drain(h2d, 2, flush_req_ids=set())
        self.assertTrue(r.success)
        # DMA done, scatter still gated; the buffer is held.
        self.assertEqual(len(h2d._pending_scatters), 1)
        self.assertFalse(h2d._h2d_buffer_free.is_set())

        h2d.drop_pending_scatters_for({"req-dead"})
        self.assertEqual(h2d._pending_scatters, [])
        self.assertTrue(h2d._h2d_buffer_free.is_set())

    def test_empty_group_is_skipped(self):
        device_tensors, cpu_tensors, d2h, _ = self._make_handlers()
        originals = [t.clone() for t in device_tensors]
        gpu_spec, cpu_spec = self._specs([4], [3],
                                         group_sizes=[1, 0],
                                         block_indices=[0, 0])
        self.assertTrue(d2h.transfer_async(3, (gpu_spec, cpu_spec)))
        r = self._drain(d2h, 3)
        self.assertTrue(r.success)
        torch.testing.assert_close(cpu_tensors[0][3], originals[0][4])
        torch.testing.assert_close(cpu_tensors[1][3], originals[1][4])
        # Group 1's tensor never written.
        self.assertTrue(bool((cpu_tensors[2] == 0).all()))


class TestEstimateKvConnectorHbmReserve(unittest.TestCase):
    """tpu_worker._estimate_kv_connector_hbm_reserve dynamically resolves
    the configured spec via kv_connector_extra_config without hardcoding
    connector or spec class names."""

    @staticmethod
    def _make_worker(kv_transfer_config):
        from vllm_torchtpu.worker.tpu_worker import TPUWorker
        worker = TPUWorker.__new__(TPUWorker)
        worker.vllm_config = MagicMock()
        worker.vllm_config.kv_transfer_config = kv_transfer_config
        return worker

    def test_no_kv_connector_returns_zero(self):
        worker = self._make_worker(kv_transfer_config=None)
        self.assertEqual(worker._estimate_kv_connector_hbm_reserve(), 0)

    def test_missing_spec_metadata_returns_zero(self):
        """A kv_connector without spec_name / spec_module_path keys
        (e.g., a P/D disagg connector that doesn't use OffloadingConnector's
        factory mechanism) gets a zero reserve — its budget must not be
        shrunk."""
        kv_tc = MagicMock()
        kv_tc.kv_connector_extra_config = {}
        worker = self._make_worker(kv_transfer_config=kv_tc)
        self.assertEqual(worker._estimate_kv_connector_hbm_reserve(), 0)

    def test_unimportable_spec_module_returns_zero(self):
        kv_tc = MagicMock()
        kv_tc.kv_connector_extra_config = {
            "spec_name": "Nonexistent",
            "spec_module_path": "does.not.exist.module",
        }
        worker = self._make_worker(kv_transfer_config=kv_tc)
        self.assertEqual(worker._estimate_kv_connector_hbm_reserve(), 0)

    def test_spec_without_estimate_method_returns_zero(self):
        """Even if the spec class resolves, returning 0 when it doesn't
        advertise an HBM reserve preserves the duck-typing contract."""
        kv_tc = MagicMock()
        # Point at this very test module — has no estimate_hbm_reserve_bytes
        # classmethod, so duck-typing falls through to 0.
        kv_tc.kv_connector_extra_config = {
            "spec_name": "TestEstimateKvConnectorHbmReserve",
            "spec_module_path": "tests.offload.test_cpu_tpu_offload",
        }
        worker = self._make_worker(kv_transfer_config=kv_tc)
        self.assertEqual(worker._estimate_kv_connector_hbm_reserve(), 0)


class TestPadMambaStateForDma(unittest.TestCase):
    """TPU_VLLM_PAD_MAMBA_STATE_FOR_DMA padding of 2-D mamba states."""

    def _pad(self, shape, enabled=True):
        from vllm_torchtpu.kv_cache_materializer import \
            _maybe_pad_mamba_state_shape
        env = {
            "TPU_VLLM_PAD_MAMBA_STATE_FOR_DMA": "1" if enabled else "0",
            "TPU_VLLM_KV_CACHE_ALIAS_FALLBACK": "1",
        }
        with patch.dict(os.environ, env):
            return _maybe_pad_mamba_state_shape(shape)

    def test_conv_state_rounds_to_multiple_of_4(self):
        self.assertEqual(self._pad((3, 2048)), (4, 2048))
        self.assertEqual(self._pad((5, 2048)), (8, 2048))

    def test_aligned_and_non_2d_shapes_unchanged(self):
        self.assertEqual(self._pad((4, 2048)), (4, 2048))
        self.assertEqual(self._pad((8, 128, 128)), (8, 128, 128))

    def test_disabled_returns_original(self):
        self.assertEqual(self._pad((3, 2048), enabled=False), (3, 2048))

    @staticmethod
    def _bare_handler(tpu_to_cpu):
        import threading

        from vllm_torchtpu.offload.cpu_tpu import _RaidenOffloadingHandler
        h = _RaidenOffloadingHandler.__new__(_RaidenOffloadingHandler)
        h.tpu_to_cpu = tpu_to_cpu
        h._queued_transfers = []
        h._queue_lock = threading.Lock()
        h._sidecar = None
        h._pending = {}
        return h

    def test_deferred_store_dispatch_gates_on_completed_step(self):
        h = self._bare_handler(tpu_to_cpu=True)
        mgr = MagicMock()
        mgr.d2h.return_value = "fut"
        futures: list = []
        entry = [futures, 0, False, False]
        h._queued_transfers.append(
            (5, 1, None, [(mgr, [0], [0], [1])], futures, entry, False))

        h.dispatch_queued_stores(4)
        self.assertFalse(entry[3])
        mgr.d2h.assert_not_called()

        h.dispatch_queued_stores(5)
        self.assertTrue(entry[3])
        mgr.d2h.assert_called_once_with([0], [0], [1])
        self.assertEqual(futures, ["fut"])
        self.assertEqual(h._queued_transfers, [])

    def test_deferred_restore_dispatch_gates_and_uses_h2d(self):
        h = self._bare_handler(tpu_to_cpu=False)
        mgr = MagicMock()
        mgr.h2d.return_value = "fut"
        futures: list = []
        entry = [futures, 0, False, False]
        h._queued_transfers.append(
            (3, 7, None, [(mgr, [8], [2], [1])], futures, entry, False))

        h.dispatch_queued_stores(2)
        self.assertFalse(entry[3])
        mgr.h2d.assert_not_called()
        mgr.d2h.assert_not_called()

        h.dispatch_queued_stores(3)
        self.assertTrue(entry[3])
        mgr.h2d.assert_called_once_with([2], [8], [1])
        mgr.d2h.assert_not_called()
        self.assertEqual(futures, ["fut"])

    def test_jax_split_and_repad_roundtrip(self):
        import jax.numpy as jnp

        from vllm_torchtpu.layers.common.gdn_attention import (
            _repad_conv_state, _split_conv_state_pad)
        kernel_size = 4
        padded = jnp.arange(2 * 4 * 8, dtype=jnp.float32).reshape(2, 4, 8)
        real, pad = _split_conv_state_pad(padded, kernel_size)
        self.assertEqual(pad, 1)
        self.assertEqual(real.shape, (2, 3, 8))
        out = _repad_conv_state(real, pad)
        self.assertEqual(out.shape, (2, 4, 8))
        self.assertTrue(bool((out[:, 3, :] == 0).all()))
        # Unpadded storage passes through untouched.
        real2, pad2 = _split_conv_state_pad(real, kernel_size)
        self.assertEqual(pad2, 0)
        self.assertIs(_repad_conv_state(real2, pad2), real2)


if __name__ == "__main__":
    unittest.main()
