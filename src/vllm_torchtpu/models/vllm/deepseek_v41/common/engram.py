"""Upstream Engram hashing, request history and TPU embedding lookup.

Shared by the v1 and v2 DeepSeek V4.1 implementations. Each implementation
subclasses ``NgramHashState`` to bind its own slot-cache write op and, when
its SWA cache module differs, the cache lifetime in ``ensure_cache``.
"""

import dataclasses
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from vllm.distributed import get_engram_dp_group, get_engram_dp_size
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.models.deepseek_v41.common import engram as upstream

try:
    from vllm.models.deepseek_v41.common.engram import (
        engram_gathered_num_tokens,
        engram_head_shard_rank,
    )
except ImportError:
    from vllm.models.deepseek_v41.nvidia.engram import (
        engram_gathered_num_tokens,
        engram_head_shard_rank,
    )

if not hasattr(upstream, "gather_engram_hashes"):
    try:
        from vllm.models.deepseek_v41.nvidia import engram as _nvidia_engram

        upstream.gather_engram_hashes = getattr(
            _nvidia_engram, "gather_engram_hashes", None
        )
    except ImportError:
        pass

if not hasattr(upstream, "engram_gathered_num_tokens"):
    upstream.engram_gathered_num_tokens = engram_gathered_num_tokens

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.upstream_bindings import (
    bind_device_ops,
    device_hooks,
)
from vllm_torchtpu.model_loader_patches import register_row_slice
from vllm_torchtpu.models.vllm.deepseek_v41.common.host_row_table import (
    HostRowTable,
)


class NgramHashState(upstream.NgramHashState):
    """Keep upstream tokenizer/layout/cache lifetime; replace Triton hashing."""

    def write_cache(
        self,
        cache: torch.Tensor,
        slots: torch.Tensor,
        values: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        """Write mapped token ids into the slot cache; bound per implementation."""
        raise NotImplementedError

    def ensure_cache(self) -> bool:
        if torch.compiler.is_compiling():
            # The outer CausalLM initializes the cache before tracing.
            # Layer views inside the graph are temporary Python tensor objects;
            # comparing their weakrefs must not reset persistent token history.
            return (
                self._cache is not None
                if self.use_slot_cache
                else self.swa_cache_module.kv_cache.numel() > 0
            )
        kv_cache = self.swa_cache_module.kv_cache
        if self.use_slot_cache and kv_cache.numel():
            # Hybrid allocation can grow the physical SWA page beyond the
            # block size copied at model construction. Match the bound view
            # used by slot_mapping and block_table before upstream sizes its
            # generated-token history cache.
            block_size = kv_cache.shape[1]
            if self.block_size != block_size:
                self.block_size = block_size
                self._kv_cache_ref = None
        return super().ensure_cache()

    def forward(
        self,
        input_ids,
        positions,
        query_start_loc,
        dead_mask,
        lookback_token_ids,
        lookback_dead_mask,
        slot_mapping,
        block_table,
    ):
        count = input_ids.numel()
        layers, ngram = self.multipliers.shape
        if count == 0:
            return input_ids.new_empty(
                (0, layers, (ngram - 1) * self.primes.shape[-1]), dtype=torch.int32
            )
        if slot_mapping is not None:
            slot_mapping = slot_mapping[:count]
        mapped: torch.Tensor | None = None
        if self.use_slot_cache:
            assert self._cache is not None and slot_mapping is not None
            mapped = self.token_map.index_select(0, input_ids.int()).masked_fill(
                dead_mask, upstream.DEAD_ID
            )
            valid = slot_mapping >= 0
            self.write_cache(self._cache, slot_mapping, mapped, valid)

        def fallback(past, reqs, window):
            if not self.use_slot_cache:
                return torch.full_like(window, self.pad_id)
            assert self._cache is not None
            # Positions, block ids and slots are < 2**31: gather in int32 from
            # the flattened table instead of 2D int64 advanced indexing.
            logical = past.clamp(
                min=0, max=block_table.shape[1] * self.block_size - 1
            ).int()
            block_idx = (
                reqs.int() * block_table.shape[1] + logical // self.block_size
            ).reshape(-1)
            blocks = (
                block_table.reshape(-1).index_select(0, block_idx).view_as(logical)
            ).int()
            slots = (blocks * self.block_size + logical % self.block_size).clamp(
                min=0, max=self._cache.numel() - 1
            )
            return self._cache.index_select(0, slots.reshape(-1)).view_as(slots)

        tables = (self.token_map, self.multipliers, self.primes, self.offsets)
        return self._hash(
            tables,
            input_ids,
            positions,
            query_start_loc,
            dead_mask,
            lookback_token_ids,
            lookback_dead_mask,
            fallback,
            mapped_batch=mapped,
        )

    def hash_on_host(
        self,
        input_ids,
        positions,
        query_start_loc,
        dead_mask,
        lookback_token_ids,
        lookback_dead_mask,
    ):
        """``forward`` on CPU tensors, for steps whose tokens are all prompt.

        Every n-gram source of a real prompt row is then either in the chunk
        or in ``lookback_token_ids`` (the runner fills it from the prompt), so
        the generated-token slot cache is never consulted and is not written:
        prompt history is always re-supplied by lookback, and decode steps
        (which do need the cache) keep the device path. Padded rows read
        ``pad_id`` here instead of an arbitrary cache slot; their outputs are
        never consumed. Integer math, so hashes equal the device path's.

        This sits on the host critical path of ``OFFLOAD=1`` decode steps, so
        it is a fixed number of whole-chunk NumPy operations laid out
        token-last (``[ngram, T]``, ``[layers, ngram, T]``, ``[layers, ngram -
        1, heads, T]``) so every inner loop runs over the tokens rather than
        over the 4 shifts or 8 heads; the lookback window is resolved once per
        request and the in-chunk sources are a strided view of the mapped ids.
        """
        count = input_ids.numel()
        layers, ngram = self.multipliers.shape
        if count == 0:
            return input_ids.new_empty(
                (0, layers, (ngram - 1) * self.primes.shape[-1]), dtype=torch.int32
            )
        tables = getattr(self, "_host_tables", None)
        if tables is None:
            tables = _HostHashTables.from_state(self)
            self._host_tables = tables
        depth = int(self.lookback_depth)
        pad_id = int(self.pad_id)
        ids_np = np.asarray(input_ids)
        pos_np = np.asarray(positions, dtype=np.int64)
        qsl_np = np.asarray(query_start_loc)
        dead_np = np.asarray(dead_mask, dtype=np.bool_)
        max_req_rows = max(1, qsl_np.shape[0] - (1 if int(qsl_np[-1]) >= count else 0))
        active_reqs = min(int(lookback_token_ids.shape[0]), max_req_rows)
        lb_np = np.asarray(lookback_token_ids)[:active_reqs, :depth]
        lb_dead_np = np.asarray(
            lookback_dead_mask[:active_reqs, :depth], dtype=np.bool_
        )
        hashed, rows, padded, batch, terms = tables._ensure_scratch(count)
        _, starts, req_base = tables.request_rows(
            qsl_np, rows, count, active_reqs, depth
        )
        # Offset of each token from its request's first chunk position; the
        # n-gram source `shift` back is in the chunk iff shift <= offset.
        offset = pos_np - pos_np[starts]
        shifts = tables.shifts
        # In-chunk sources: batch[s, t] = mapped[max(t - s, 0)], as a cached
        # strided window over `padded` instead of a fresh allocation/gather.
        mapped = padded[ngram - 1 :]
        np.take(tables.token_map, ids_np, out=mapped)
        mapped[dead_np] = upstream.DEAD_ID
        padded[: ngram - 1] = mapped[0]
        # Lookback sources, resolved once per active request: column `c` of
        # `window_table` is the mapped id `c + 1` positions before the chunk
        # (pad_id when unknown), and column `depth` is pad_id for sources
        # beyond the window. Source `shift` of a token at chunk offset `o`
        # reads column shift - o - 1.
        window_table = tables.window_table(active_reqs, depth)
        wt_cols = window_table[:, :depth]
        np.take(tables.token_map, lb_np, out=wt_cols, mode="clip")
        wt_cols[lb_dead_np] = upstream.DEAD_ID
        wt_cols[lb_np < 0] = pad_id
        window_table[:, depth] = pad_id
        # The [ngram, T] steps below write into reused buffers (`work`): same
        # values as `cols = shifts - (offset + 1)`, `np.where(in_chunk, batch,
        # window)` and `np.where(blocked, pad_id, source)`, without a fresh
        # array per step.
        cols, in_chunk, source, blocked, scratch_mask = tables.work(count)
        np.subtract(shifts, offset[None, :], out=cols)
        cols -= 1
        np.less(cols, 0, out=in_chunk)
        np.clip(cols, 0, depth, out=cols)
        cols += req_base
        np.take(window_table.reshape(-1), cols, out=source)
        np.copyto(source, batch, where=in_chunk)
        # A dead or pre-sequence source blocks every longer n-gram.
        np.greater(shifts, pos_np[None, :], out=blocked)
        np.equal(source, upstream.DEAD_ID, out=scratch_mask)
        blocked |= scratch_mask
        for shift in range(1, ngram):
            blocked[shift] |= blocked[shift - 1]
        np.copyto(source, pad_id, where=blocked)
        np.multiply(tables.multipliers[:, :, None], source[None, :, :], out=terms)
        rolled = terms[:, 1:]
        np.bitwise_xor(terms[:, 0], rolled[:, 0], out=rolled[:, 0])
        for shift in range(2, ngram):
            np.bitwise_xor(
                rolled[:, shift - 2], rolled[:, shift - 1], out=rolled[:, shift - 1]
            )
        # The int64 bucket ids reuse one scratch buffer: a fresh array of this
        # size is page-faulted in on every call otherwise.
        np.remainder(rolled[:, :, None, :], tables.primes, out=hashed)
        # Add the table offsets while narrowing straight into the device
        # path's [T, layers, heads] layout: one pass instead of an in-place
        # int64 add plus a transposing astype copy (95 + 211 us at T=1152).
        # The result is a fresh array each call: callers keep the tensor.
        heads = hashed.shape[2]
        out = np.empty((count, layers, (ngram - 1) * heads), dtype=np.int32)
        np.add(
            hashed,
            tables.offsets,
            out=out.reshape(count, layers, ngram - 1, heads).transpose(1, 2, 3, 0),
            casting="unsafe",
        )
        return torch.from_numpy(out)

    def _hash(
        self,
        tables,
        input_ids,
        positions,
        query_start_loc,
        dead_mask,
        lookback_token_ids,
        lookback_dead_mask,
        fallback,
        *,
        mapped_batch: torch.Tensor | None = None,
    ):
        token_map, multipliers, primes, offsets = tables
        count = input_ids.numel()
        _, ngram = multipliers.shape
        dev = input_ids.device
        # Rows, positions, request ids and lookback columns are < 2**31: keep
        # them int32 and gather with flat 1D ``index_select``; ``source`` is
        # widened to int64 only for the hash multiply below.
        rows = torch.arange(count, dtype=torch.int32, device=dev)
        qsl_i32 = query_start_loc.int()
        pos_i32 = positions.int()
        reqs = (
            (rows.unsqueeze(1) >= qsl_i32[1:].unsqueeze(0))
            .sum(-1, dtype=torch.int32)
            .clamp(max=lookback_token_ids.shape[0] - 1)
        )
        starts = qsl_i32.index_select(0, reqs).clamp(max=count - 1)
        chunk_positions = pos_i32.index_select(0, starts)
        shifts = torch.arange(ngram, dtype=torch.int32, device=dev)
        past = pos_i32[:, None] - shifts
        in_chunk = past >= chunk_positions[:, None]
        indices = (rows[:, None] - shifts).clamp(min=0)
        if mapped_batch is None:
            mapped_batch = token_map.index_select(0, input_ids.int()).masked_fill(
                dead_mask, upstream.DEAD_ID
            )
        batch = mapped_batch.index_select(0, indices.reshape(-1)).view(count, ngram)
        cols = chunk_positions[:, None] - 1 - past
        in_window = ~in_chunk & (cols >= 0) & (cols < self.lookback_depth)
        cols = cols.clamp(min=0, max=self.lookback_depth - 1)
        reqs_col = reqs[:, None]
        lb_idx = (reqs_col * lookback_token_ids.shape[1] + cols).reshape(-1)
        window_ids = (
            lookback_token_ids.reshape(-1).index_select(0, lb_idx).view(count, ngram)
        )
        known = in_window & (window_ids >= 0)
        mapped_lookback = (
            token_map.index_select(0, lookback_token_ids.clamp(min=0).reshape(-1).int())
            .view_as(lookback_token_ids)
            .masked_fill(lookback_dead_mask, upstream.DEAD_ID)
        )
        window = mapped_lookback.reshape(-1).index_select(0, lb_idx).view(count, ngram)
        source = torch.where(
            in_chunk,
            batch,
            torch.where(known, window, fallback(past, reqs_col, window)),
        ).long()
        blocked = ((past < 0) | (source == upstream.DEAD_ID)).cumsum(dim=1) > 0
        source = source.masked_fill(blocked, self.pad_id)
        terms = source[:, None, :] * multipliers[None, :, :]
        rolling = terms[:, :, 0]
        rolling_steps = []
        for shift in range(1, ngram):
            rolling = rolling ^ terms[:, :, shift]
            rolling_steps.append(rolling)
        rolled = torch.stack(rolling_steps, dim=2)
        hashed = (rolled[:, :, :, None] % primes[None, :, :, :]).flatten(2) + offsets[
            None, :, :
        ]
        return hashed.to(torch.int32)


@dataclasses.dataclass(frozen=True)
class _HostHashTables:
    """Host (NumPy) copies of ``NgramHashState``'s tables, token-last shaped.

    Built once per state on the first ``hash_on_host`` call. ``token_map`` is
    int32 (ids are far below 2**31) so the random per-token gathers touch
    half the cache lines of the int64 original.
    """

    token_map: np.ndarray  # [vocab] int32
    multipliers: np.ndarray  # [layers, ngram] int64
    primes: np.ndarray  # [layers, ngram - 1, heads, 1] int64
    offsets: np.ndarray  # [layers, ngram - 1, heads, 1] int64
    shifts: np.ndarray  # [ngram, 1] int64
    _scratch: list[np.ndarray] = dataclasses.field(default_factory=list)
    _memo: dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_state(cls, state) -> "_HostHashTables":
        token_map, multipliers, primes, offsets = (
            np.asarray(t.to("cpu"), dtype=np.int64)
            for t in (state.token_map, state.multipliers, state.primes, state.offsets)
        )
        primes_4d = np.ascontiguousarray(primes[..., None])
        return cls(
            token_map=token_map.astype(np.int32),
            multipliers=multipliers,
            primes=primes_4d,
            offsets=np.ascontiguousarray(offsets.reshape(primes_4d.shape)),
            shifts=np.arange(multipliers.shape[1], dtype=np.int64)[:, None],
        )

    def _ensure_scratch(
        self, count: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self._scratch or self._scratch[0].shape[-1] != count:
            layers, ngram = self.multipliers.shape
            hashed = np.empty((*self.primes.shape[:3], count), dtype=np.int64)
            rows = np.arange(count, dtype=np.int64)
            padded = np.empty(count + ngram - 1, dtype=self.token_map.dtype)
            batch = np.lib.stride_tricks.sliding_window_view(padded, count)[::-1]
            terms = np.empty((layers, ngram, count), dtype=np.int64)
            self._scratch[:] = [hashed, rows, padded, batch, terms]
        return (
            self._scratch[0],
            self._scratch[1],
            self._scratch[2],
            self._scratch[3],
            self._scratch[4],
        )

    def scratch(self, count: int) -> np.ndarray:
        """``[layers, ngram - 1, heads, count]`` int64 work buffer, reused."""
        return self._ensure_scratch(count)[0]

    def request_rows(
        self,
        qsl_np: np.ndarray,
        rows: np.ndarray,
        count: int,
        active_reqs: int,
        depth: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Each token row's request, that request's first chunk row, and the
        request's flat ``window_table`` row base ``[1, T]`` (``req * (depth +
        1)``).

        Memoized on the step layout: steady decode repeats the same
        ``query_start_loc`` (e.g. ``0, 6, ..., 1152``), so the per-row
        ``searchsorted`` runs only when it changes. Callers only read them.
        """
        key = (count, active_reqs, depth, qsl_np.dtype.str, qsl_np.tobytes())
        cached = self._memo.get("request_rows")
        if cached is not None and cached[0] == key:
            return cached[1], cached[2], cached[3]
        reqs = np.minimum(
            np.searchsorted(qsl_np[1:], rows, side="right"),
            active_reqs - 1,
        )
        starts = np.minimum(qsl_np[reqs], count - 1)
        req_base = reqs[None, :] * (depth + 1)
        self._memo["request_rows"] = (key, reqs, starts, req_base)
        return reqs, starts, req_base

    def work(
        self, count: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Reused ``[ngram, count]`` buffers for ``hash_on_host``'s source
        selection: (cols int64, in_chunk bool, source token-map dtype,
        blocked bool, scratch bool). Overwritten on every call."""
        bufs = self._memo.get("work")
        if bufs is None or bufs[0].shape[1] != count:
            shape = (self.multipliers.shape[1], count)
            bufs = (
                np.empty(shape, dtype=np.int64),
                np.empty(shape, dtype=np.bool_),
                np.empty(shape, dtype=self.token_map.dtype),
                np.empty(shape, dtype=np.bool_),
                np.empty(shape, dtype=np.bool_),
            )
            self._memo["work"] = bufs
        return bufs

    def window_table(self, active_reqs: int, depth: int) -> np.ndarray:
        """``[active_reqs, depth + 1]`` lookback window buffer, reused.

        ``hash_on_host`` overwrites every entry before reading it.
        """
        shape = (active_reqs, depth + 1)
        table = self._memo.get("window_table")
        if table is None or table.shape != shape:
            table = np.empty(shape, dtype=self.token_map.dtype)
            self._memo["window_table"] = table
        return table


def _load_local_rows(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Upstream's head-shard loader for a tensor holding only this rank's rows."""
    if param.dtype == torch.uint8 or loaded_weight.dtype == torch.float8_e8m0fnu:
        loaded_weight = loaded_weight.view(torch.uint8)
    else:
        loaded_weight = loaded_weight.to(param.dtype)
    assert loaded_weight.shape == param.shape, (
        f"engram shard {tuple(loaded_weight.shape)} does not fit param "
        f"{tuple(param.shape)}"
    )
    param.data.copy_(loaded_weight)


class ParallelEngramEmbedding(upstream.ParallelEngramEmbedding):
    """Reuse upstream head sharding and checkpoint loaders; support Host DRAM."""

    host_dma: bool = False
    _host_rows: HostRowTable | None = None

    @staticmethod
    def _weight_loader(param, loaded_weight):
        loader = upstream.ParallelEngramEmbedding._weight_loader
        if loaded_weight.shape[0] == param.shape[0]:
            # Already this rank's rows: the sharded Run:AI stream fetches only
            # them (see Engram._create_embedding). A full table has as many
            # rows only with a single shard, where both loaders agree.
            loader = _load_local_rows
        if param.device.type != "tpu":
            return loader(param, loaded_weight)
        from vllm_torchtpu.model_loader_patches import ExpertParamStager

        # Reuse upstream's shard selection and scale-byte interpretation.
        # A host stand-in detaches the shard from the giant mapped table;
        # the existing TPU stager bounds uploads and synchronizes publication.
        stager = ExpertParamStager()
        loader(stager.host_param(param), loaded_weight)
        stager.flush(param)

    def __init__(
        self,
        num_embeddings: int,
        dim: int,
        head_sizes: tuple[int, ...],
        *args,
        cpu_offload: bool | None = None,
        **kwargs,
    ):
        # Upstream's own `cpu_offload` means a pinned table read through UVA
        # (CUDA only) and it resets the attribute in `__init__`; keep it False
        # and track the TPU host table separately.
        self.host_offload = (
            bool(envs.TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD)
            if cpu_offload is None
            else bool(cpu_offload)
        )
        self.cpu_offload = self.host_offload
        self.dp_shared_memory = bool(kwargs.pop("dp_shared_memory", False))
        self.dp_size = 1 if self.dp_shared_memory else get_engram_dp_size()
        # Rows load into Host DRAM as with cpu_offload, then
        # `move_rows_to_pinned_host` hands them to the TPU's pinned_host
        # memory, after which `lookup` runs on device like an HBM table.
        self.host_dma = self.host_offload and bool(
            envs.TPU_DEEPSEEK_V41_ENGRAM_HOST_DMA
        )
        self._host_rows: HostRowTable | None = None
        self._head_sizes = tuple(head_sizes)
        # Upstream asks for the SM count solely to size its Triton lookup grid.
        # The TPU lookup below does not use that launch property.
        with device_hooks(
            torch.cuda,
            get_device_properties=lambda _: SimpleNamespace(multi_processor_count=0),
        ):
            try:
                super().__init__(
                    num_embeddings, dim, head_sizes, *args, cpu_offload=False, **kwargs
                )
            except TypeError:
                super().__init__(num_embeddings, dim, head_sizes, *args, **kwargs)
        self.cpu_offload = self.host_offload
        upstream.set_weight_attrs(self.weight, {"dummy_weight_value": 1.0})
        if self.weight_scale_inv is not None:
            upstream.set_weight_attrs(
                self.weight_scale_inv, {"dummy_weight_value": 127}
            )

    def _get_shard_info(self) -> tuple[int, int]:
        if getattr(self, "dp_size", 1) == 1:
            return super()._get_shard_info()
        return self.tp_size * self.dp_size, engram_head_shard_rank()

    def _allocate_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.host_offload:
            return super()._allocate_weights()
        return (
            torch.empty(
                self.part_num_embeddings,
                self.dim,
                dtype=torch.float8_e4m3fn,
                device="cpu",
            ),
            torch.empty(
                self.part_num_embeddings,
                self.dim // self.block_size,
                dtype=torch.uint8,
                device="cpu",
            ),
        )

    def max_shard_rows(self) -> int:
        """Rows of the largest head shard on any rank (upstream's split)."""
        num_shards, _ = self._get_shard_info()
        per = self.part_n_hash_cols
        sizes = self._head_sizes
        return max(sum(sizes[r * per : (r + 1) * per]) for r in range(num_shards))

    def move_rows_to_pinned_host(self, device: torch.device) -> None:
        """Loaded Host-DRAM rows -> TPU pinned_host, scales -> HBM (host_dma).

        Runs once, outside compilation, after weights are loaded. Frees the
        CPU tables. Chunks are sized for the largest shard so that every
        rank compiles the same program.
        """
        if not self.host_dma or self._host_rows is not None:
            return
        weight, scales = self._storage()
        self._host_rows = HostRowTable(
            weight, device, capacity_rows=self.max_shard_rows()
        )
        # `.data =` cannot move a CPU parameter to the TPU; replace it.
        self.weight_scale_inv = torch.nn.Parameter(
            scales.to(device), requires_grad=False
        )
        self.weight.data = weight.new_empty((0, self.dim))

    def _select_local_heads(
        self, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start = min(self.head_start, self.n_hash_cols)
        end = min(self.head_start + self.part_n_hash_cols, self.n_hash_cols)
        # Hash ids are int32 (``_hash`` / host hash); keep the rows int32.
        selected = indices[:, start:end].int()
        valid = (selected >= self.vocab_start_idx) & (selected < self.vocab_end_idx)
        pad = self.part_n_hash_cols - (end - start)
        if pad:
            selected = torch.nn.functional.pad(selected, (0, pad))
            valid = torch.nn.functional.pad(valid, (0, pad), value=False)
        rows = (selected - self.vocab_start_idx).clamp(
            min=0, max=self.part_num_embeddings - 1
        )
        return rows, valid

    def _host_dma_lookup(self, indices, out):
        assert self._host_rows is not None
        rows, valid = self._select_local_heads(indices)
        dequant = self._host_rows.lookup_and_dequant(
            rows.flatten(),
            valid.flatten(),
            self.weight_scale_inv.data,
            self.block_size,
        )
        out.copy_(dequant.view(out.shape))

    def lookup(self, indices, out, background=False):
        if self._host_rows is not None:
            return self._host_dma_lookup(indices, out)
        weight, scales = self._storage()
        work_device = weight.device
        if indices.device != work_device:
            indices = indices.to(work_device)
        rows, valid = self._select_local_heads(indices)
        flat_rows = rows.reshape(-1)
        values = (
            weight.index_select(0, flat_rows)
            .view(*rows.shape, *weight.shape[1:])
            .unflatten(-1, (-1, self.block_size))
            .float()
        )
        scale_bits = scales.index_select(0, flat_rows).view(
            *rows.shape, *scales.shape[1:]
        )
        scale = (scale_bits.to(torch.int32) << 23).view(torch.float32).unsqueeze(-1)
        out.copy_(
            (values * scale)
            .flatten(-2)
            .masked_fill(~valid[:, :, None], 0)
            .to(out.dtype)
        )

    def _select_local_heads_np(
        self, indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """``_select_local_heads`` for host (NumPy) hash ids."""
        start = min(self.head_start, self.n_hash_cols)
        end = min(self.head_start + self.part_n_hash_cols, self.n_hash_cols)
        selected = indices[:, start:end] - np.int64(self.vocab_start_idx)
        valid = selected.view(np.uint64) < (self.vocab_end_idx - self.vocab_start_idx)
        pad = self.part_n_hash_cols - (end - start)
        if pad:
            selected = np.pad(selected, ((0, 0), (0, pad)))
            valid = np.pad(valid, ((0, 0), (0, pad)))
        # `selected` is a fresh array (subtraction / pad) and `valid` was read
        # from it above, so clamp it in place instead of allocating `rows`.
        rows = np.clip(selected, 0, self.part_num_embeddings - 1, out=selected)
        return rows, valid

    def gather_raw(self, indices, raw_out, scale_out):
        """Gather local heads' raw FP8 bytes and E8M0 scale bytes (no dequant).

        Host-DRAM tables only. ``raw_out`` is ``[T, local_heads, dim]`` uint8
        and ``scale_out`` is ``[T, local_heads, dim // block_size]`` uint8,
        both on the host. Rows outside this shard (or padded heads) get the
        ``_INVALID_SCALE`` sentinel in every scale block; their raw bytes are
        left as gathered (masking them too cost ~11 ms per 8192-token step on
        the host). ``dequant_fp8_rows`` maps sentinel blocks to +0,
        exactly the zero rows that ``lookup`` writes for them.

        Everything but the wide raw-row gather is NumPy: at decode sizes the
        per-op cost of the torch CPU ops (head select, ``valid.all()``, the
        8-byte scale-row ``index_select``) exceeded the work itself. The raw
        rows stay on ``torch.index_select``, which parallelizes the
        memory-latency-bound 256-byte row copies across intra-op threads.
        """
        weight, scales = self._storage()
        work_device = weight.device
        if indices.device != work_device:
            indices = indices.to(work_device)
        rows, valid = self._select_local_heads_np(np.asarray(indices))
        flat = rows.reshape(-1)
        # The uint8 / NumPy views of the storage, rebuilt only when it changes
        # (every step of all 4 Engram layers otherwise re-creates both).
        views = getattr(self, "_raw_views", None)
        if views is None or views[0] is not weight or views[1] is not scales:
            views = (weight, scales, weight.view(torch.uint8), np.asarray(scales))
            self._raw_views = views
        torch.index_select(
            views[2],
            0,
            torch.from_numpy(flat),
            out=raw_out.view(-1, self.dim),
        )
        blocks = self.dim // self.block_size
        scale_dst = np.asarray(scale_out)
        np.take(views[3], flat, axis=0, out=scale_dst.reshape(-1, blocks))
        if not valid.all():
            scale_dst[~valid] = _INVALID_SCALE


# E8M0 scale byte for 2**0. Scale bytes encode 2**(byte - 127).
_UNIT_SCALE = 127
# E8M0 byte 255 is NaN and never appears in a real table; staging uses it to
# mark rows outside the shard, padded heads and the unused tail.
_INVALID_SCALE = 255


def dequant_fp8_rows(raw, scale, block_size):
    """``[..., dim]`` FP8 bytes x ``[..., dim // block_size]`` E8M0 bytes -> bf16.

    Matches ``ParallelEngramEmbedding.lookup``'s ``values.float() *
    exp2(scale - 127)`` rounded once to bf16. The power of two is built from
    its float32 exponent bits (``byte << 23``), which is exact for bytes
    1..254 and does not depend on the backend's ``exp2`` accuracy; the product
    of a float32 and a power of two is then exact, so only the final bf16
    rounding remains. Byte 0 (2**-127, a float32 subnormal) maps to 0.
    Byte 255 (``_INVALID_SCALE``) yields +0 whatever the raw bytes are,
    including FP8 NaN.
    """
    values = raw.view(torch.float8_e4m3fn).unflatten(-1, (-1, block_size)).float()
    power = (scale.to(torch.int32) << 23).view(torch.float32).unsqueeze(-1)
    invalid = (scale == _INVALID_SCALE).unsqueeze(-1)
    dequant = torch.where(invalid, 0.0, values * power).flatten(-2)
    return dequant.to(torch.bfloat16)


def _unpack_bytes(words, num_bytes):
    """Little-endian bytes of int32 ``words`` (1-D) -> first ``num_bytes`` uint8.

    Inverse of the host's ``uint8_buffer.view(torch.int32)`` on a
    little-endian host. Staging crosses H2D as 1-D int32 because TorchTPU only
    takes the fast linearizer path for 1-D 32-bit tensors.
    """
    shifts = torch.arange(0, 32, 8, dtype=torch.int32, device=words.device)
    unpacked = (words.unsqueeze(-1) >> shifts) & 0xFF
    return unpacked.to(torch.uint8).reshape(-1)[:num_bytes]


def _gather_engram_rows(rows, dim):
    # Retain the existing 128-lane width alignment for narrow fixtures.
    # This does not fix the native Engram NaN: it also reproduces at width
    # 128, while an isolated 3-D BF16 gather passes. Released heads are 256.
    width = rows.shape[-1]
    padding = (-width) % 128
    if padding:
        rows = torch.nn.functional.pad(rows, (0, padding))
    result = upstream.tensor_model_parallel_all_gather(rows, dim=dim)
    return result[..., :width].contiguous() if padding else result


def _engram_dp_slot(default_tokens: int) -> int:
    if (
        is_forward_context_available()
        and getattr(get_forward_context(), "dp_metadata", None) is not None
    ):
        slot = engram_gathered_num_tokens()
        if slot > 0:
            return slot
    return default_tokens


def _dp_gather_engram_rows(staged: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Exchange DP tokens for heads, retaining only this replica's tokens."""
    dp_group = get_engram_dp_group()
    if dp_group is None or dp_group.world_size <= 1:
        return staged[:num_tokens]
    dp_size = dp_group.world_size
    slot, remainder = divmod(staged.shape[0], dp_size)
    assert remainder == 0 and 0 <= num_tokens <= slot
    local_heads, width = staged.shape[1:]
    padding = (-width) % 128
    if padding:
        staged = torch.nn.functional.pad(staged, (0, padding))
    gathered = dp_group.all_gather(staged, dim=0)
    if padding:
        gathered = gathered[..., :width]
    start = dp_group.rank_in_group * slot
    selected = gathered.view(dp_size, dp_size * slot, local_heads, width)[
        :, start : start + num_tokens
    ]
    return (
        selected.transpose(0, 1)
        .reshape(num_tokens, dp_size * local_heads, width)
        .contiguous()
    )


def _dp_gather_packed_rows(
    active_raw: torch.Tensor,
    num_tokens: int,
    *,
    cols: int,
    head_dim: int,
    blocks: int,
    block_size: int,
) -> torch.Tensor | None:
    """DP token-for-heads exchange on the packed int32 staging words.

    Same result as dequantizing every staged token and calling
    ``_dp_gather_engram_rows`` on the bf16 rows, but the collective moves the
    packed FP8 + E8M0 bytes (``cols * head_dim * 9 / 8`` B/token instead of
    ``cols * head_dim * 2`` B/token, 1.9x less), and only this replica's
    ``num_tokens`` of the ``dp * slot`` staged tokens are unpacked and
    dequantized (``dp_size``x fewer bf16/fp32 intermediates in HBM).
    ``dequant_fp8_rows`` is elementwise per (token, head, block), so slicing
    before it is bit-identical to slicing after it.

    ``active_raw`` is the ``[staged_tokens, words_per_tok]`` view of a 1-D
    buffer laid out as all raw words then all scale words (see
    ``_init_staging``). Returns ``None`` without a DP group, in which case the
    caller keeps the single-replica path.
    """
    dp_group = get_engram_dp_group()
    if dp_group is None or dp_group.world_size <= 1:
        return None
    dp_size = dp_group.world_size
    staged_tokens = active_raw.shape[0]
    slot, remainder = divmod(staged_tokens, dp_size)
    assert remainder == 0 and 0 <= num_tokens <= slot
    raw_words_per_tok = (cols * head_dim) // 4
    raw_words = staged_tokens * raw_words_per_tok
    start = dp_group.rank_in_group * slot
    gathered = dp_group.all_gather(active_raw, dim=0).view(dp_size, -1)
    # Raw FP8 bytes: token-major words, so this replica's window is a row
    # slice taken before unpacking.
    raw_sel = gathered[:, :raw_words].reshape(dp_size, staged_tokens, -1)[
        :, start : start + num_tokens
    ]
    raw = _unpack_bytes(raw_sel, dp_size * num_tokens * cols * head_dim).reshape(
        dp_size, num_tokens, cols, head_dim
    )
    # Scale bytes (~1/32 of the raw bytes): a token's bytes may straddle a
    # word when ``cols * blocks % 4 != 0``, so unpack every replica's small
    # scale section and slice bytes.
    scale_words = gathered[:, raw_words:]
    scale = (
        _unpack_bytes(scale_words, scale_words.numel() * 4)
        .reshape(dp_size, -1)[:, : staged_tokens * cols * blocks]
        .reshape(dp_size, staged_tokens, cols, blocks)[:, start : start + num_tokens]
    )
    # `.contiguous()` like `_dp_gather_engram_rows`: with `cols == 1` the
    # transposed reshape is a view, and so was the dequantized result.
    return dequant_fp8_rows(
        raw.transpose(0, 1).reshape(num_tokens, dp_size * cols, head_dim),
        scale.transpose(0, 1).reshape(num_tokens, dp_size * cols, blocks),
        block_size,
    ).contiguous()


def _embed_dp_size(engram: object) -> int:
    embed = getattr(engram, "embed_tokens", None)
    dp_size = getattr(embed, "dp_size", 1)
    return dp_size if type(dp_size) is int else 1


class Engram(upstream.Engram):
    """Inherit projections, staging and collective head gather from upstream."""

    embed = bind_device_ops(
        upstream.Engram.embed, tensor_model_parallel_all_gather=_gather_engram_rows
    )

    def _create_embedding(self, layout, layer_hash_index):
        embedding = ParallelEngramEmbedding(
            layout.num_embeddings[layer_hash_index],
            layout.head_dim,
            tuple(size for order in layout.primes[layer_hash_index] for size in order),
        )
        # Each table is ~98 GB in one tensor, over the Run:AI streamer's
        # per-tensor memory limit, and a rank keeps only its heads' rows.
        layer_id = layout.layer_ids[layer_hash_index]
        for leaf in ("weight", "scale"):
            register_row_slice(
                f"layers.{layer_id}.engram.embed.{leaf}",
                embedding.vocab_start_idx,
                embedding.part_num_embeddings,
            )
        return embedding

    def _init_staging(self, max_tokens: int, head_dim: int) -> None:
        dp_tokens = max_tokens * _embed_dp_size(self)
        super()._init_staging(dp_tokens, head_dim)
        self._staged_raw: torch.Tensor | None = None
        self._active_raw: torch.Tensor | None = None
        self._active_rows: torch.Tensor | None = None
        self._dp_staged: torch.Tensor | None = None
        self._staged_by_tokens: dict[int, torch.Tensor] = {}
        self._staged_views_by_tokens: dict[int, torch.Tensor] = {}
        self._staged_rows_by_tokens: dict[int, torch.Tensor] = {}
        self._host_pack_by_tokens: dict[
            int,
            tuple[
                torch.Tensor,
                torch.Tensor,
                np.ndarray,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                tuple[int, int],
            ],
        ] = {}
        if (
            getattr(
                self.embed_tokens,
                "host_offload",
                getattr(self.embed_tokens, "cpu_offload", False),
            )
        ) and (getattr(self.embed_tokens, "host_dma", False) is not True):
            # Host DRAM tables: the host only gathers raw FP8 bytes and scale
            # bytes (a memcpy-bound index_select); dequantization runs inside
            # the compiled graph (`_ready_rows`). Dequantizing on the host cost
            # ~86 ms/layer per 8192-token chunk single-threaded vs ~16 ms for the
            # raw gather.
            # Both buffers cross H2D packed into a single 1-D int32 tensor per
            # layer (`_unpack_bytes`): a 3-D uint8 `copy_()` takes TorchTPU's
            # slow XlaLinearize path (~9.75 vs ~0.59 ms/layer),
            # and packing raw + scale halves PJRT H2D transfers per step.
            cols = self.embed_tokens.part_n_hash_cols
            blocks = head_dim // self.embed_tokens.block_size
            assert head_dim % 4 == 0, head_dim
            raw_words_per_tok = (cols * head_dim) // 4
            scale_words_per_tok = -(-(cols * blocks) // 4)
            words_per_tok = raw_words_per_tok + scale_words_per_tok
            self._scale_blocks = blocks
            self._host_packed_buf = torch.zeros(
                dp_tokens * words_per_tok * 4, dtype=torch.uint8, device="cpu"
            )
            self._host_raw = self._host_packed_buf[
                : dp_tokens * raw_words_per_tok * 4
            ].view(dp_tokens, cols, head_dim)
            scale_bytes = dp_tokens * cols * blocks
            self._raw_scale_buf = torch.full(
                (dp_tokens * scale_words_per_tok * 4,),
                _INVALID_SCALE,
                dtype=torch.uint8,
                device="cpu",
            )
            self._raw_scale_view = self._raw_scale_buf[:scale_bytes].view(
                dp_tokens, cols, blocks
            )
            self._pending_scale_sync: tuple[int, int] | None = None
            self._staged_raw = torch.zeros(dp_tokens * words_per_tok, dtype=torch.int32)
            # -1 is four `_INVALID_SCALE` bytes, matching `_host_scale_buf`.
            self._staged_raw[dp_tokens * raw_words_per_tok :].fill_(-1)
            self._staged_by_tokens[dp_tokens] = self._staged_raw
            self._active_raw = self._staged_views(dp_tokens)

    def _sync_host_scale_buf(self) -> torch.Tensor:
        pending = getattr(self, "_pending_scale_sync", None)
        if pending is not None:
            raw_bytes, scale_word_bytes = pending
            np.asarray(self._raw_scale_buf)[:scale_word_bytes] = np.asarray(
                self._host_packed_buf
            )[raw_bytes : raw_bytes + scale_word_bytes]
            self._pending_scale_sync = None
        return self._raw_scale_buf

    @property
    def _host_scale_buf(self) -> torch.Tensor:
        return self._sync_host_scale_buf()

    @property
    def _host_scale(self) -> torch.Tensor:
        self._sync_host_scale_buf()
        return self._raw_scale_view

    def _staged_tensors(self, num_tokens: int) -> torch.Tensor:
        assert self._staged_raw is not None
        staged = self._staged_by_tokens.get(num_tokens)
        if staged is None:
            _, cols, head_dim = self._host_raw.shape
            blocks = self._scale_blocks
            raw_words = num_tokens * ((cols * head_dim) // 4)
            scale_words = num_tokens * (-(-(cols * blocks) // 4))
            dev = self._staged_raw.device
            staged = torch.zeros(raw_words + scale_words, dtype=torch.int32, device=dev)
            staged[raw_words:].fill_(-1)
            self._staged_by_tokens[num_tokens] = staged
        return staged

    def _staged_views(self, num_tokens: int) -> torch.Tensor:
        view = self._staged_views_by_tokens.get(num_tokens)
        if view is None:
            _, cols, head_dim = self._host_raw.shape
            blocks = self._scale_blocks
            raw_words_per_tok = (cols * head_dim) // 4
            scale_words_per_tok = -(-(cols * blocks) // 4)
            staged = self._staged_tensors(num_tokens)
            view = staged.view(num_tokens, raw_words_per_tok + scale_words_per_tok)
            if num_tokens >= 2:
                torch._dynamo.mark_dynamic(view, 0)
            self._staged_views_by_tokens[num_tokens] = view
        return view

    def _host_dma_rows(self, num_tokens: int) -> torch.Tensor:
        rows = self._staged_rows_by_tokens.get(num_tokens)
        if rows is None:
            _, cols, head_dim = self.staged_rows.shape
            rows = self.staged_rows.new_empty((num_tokens, cols, head_dim))
            if num_tokens >= 2:
                torch._dynamo.mark_dynamic(rows, 0)
            self._staged_rows_by_tokens[num_tokens] = rows
        return rows

    def _host_pack_bundle(
        self, num_tokens: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        np.ndarray,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[int, int],
    ]:
        bundle = self._host_pack_by_tokens.get(num_tokens)
        if bundle is None:
            _, cols, head_dim = self._host_raw.shape
            blocks = self._scale_blocks
            raw_words = num_tokens * ((cols * head_dim) // 4)
            raw_bytes = raw_words * 4
            scale_bytes = num_tokens * cols * blocks
            scale_words = num_tokens * (-(-(cols * blocks) // 4))
            scale_word_bytes = scale_words * 4
            raw_slice = self._host_raw[:num_tokens]
            scale_dst = self._host_packed_buf[raw_bytes : raw_bytes + scale_bytes].view(
                num_tokens, cols, blocks
            )
            pad_slice = np.asarray(self._host_packed_buf)[
                raw_bytes + scale_bytes : raw_bytes + scale_word_bytes
            ]
            host_words_slice = self._host_packed_buf.view(torch.int32)[
                : raw_words + scale_words
            ]
            staged = self._staged_tensors(num_tokens)
            active_raw = self._staged_views(num_tokens)
            bundle = (
                raw_slice,
                scale_dst,
                pad_slice,
                host_words_slice,
                staged,
                active_raw,
                (raw_bytes, scale_word_bytes),
            )
            self._host_pack_by_tokens[num_tokens] = bundle
        return bundle

    def prepare_embeddings(self, hash_ids: torch.Tensor) -> None:
        embed = self.embed_tokens
        if (
            getattr(embed, "host_dma", False) is True
            and getattr(embed, "_host_rows", None) is not None
        ):
            rows = self._host_dma_rows(hash_ids.shape[0])
            embed.lookup(hash_ids, rows)
            self._active_rows = rows
            return
        if (
            self._staged_raw is not None
            and hash_ids.shape[0] <= self._host_raw.shape[0]
        ):
            (
                raw_slice,
                scale_dst,
                pad_slice,
                host_words_slice,
                staged,
                active_raw,
                pending_sync,
            ) = self._host_pack_bundle(hash_ids.shape[0])
            # Gather scale bytes directly behind the active raw slice in
            # `_host_packed_buf`, avoiding an intermediate `_host_scale_buf`
            # memmove across all 4 Engram layers on every step.
            embed.gather_raw(hash_ids, raw_slice, scale_dst)
            pad_slice[:] = _INVALID_SCALE
            self._pending_scale_sync = pending_sync
            # On TorchTPU, `copy_()` into a slice/view of a materialized device
            # tensor allocates a fresh buffer per call instead of overwriting in
            # place, whereas a whole-tensor `copy_()` overwrites in place (see
            # `deepseek_v4/model.py:_HostStagedParams`). Allocate and reuse one
            # exact-size whole 1-D packed device tensor per `num_tokens` bucket
            # so each Engram layer issues a single fast 1-D H2D transfer, while
            # exposing a 2-D `[num_tokens, words_per_tok]` dynamic-dim0 view to
            # `torch.compile` (`create_concrete_args`).
            staged.copy_(host_words_slice, non_blocking=True)
            self._active_raw = active_raw
            return
        super().prepare_embeddings(hash_ids)
        if _embed_dp_size(self) > 1:
            # HBM table under DP: the gathered ids are `dp * slot` rows; keep
            # the slice so `_ready_rows` reads the group's slot from it instead
            # of re-deriving it from the forward context inside the stage.
            self._dp_staged = self.staged_rows[: hash_ids.shape[0]]

    def _ready_rows(self, num_tokens: int) -> torch.Tensor:
        dp_size = _embed_dp_size(self)
        active_rows = getattr(self, "_active_rows", None)
        if active_rows is not None:
            return (
                _dp_gather_engram_rows(active_rows, num_tokens)
                if dp_size > 1
                else active_rows
            )
        if self._staged_raw is None:
            if dp_size > 1:
                staged = getattr(self, "_dp_staged", None)
                if staged is None:
                    slot = _engram_dp_slot(num_tokens)
                    staged = self.staged_rows[: slot * dp_size]
                return _dp_gather_engram_rows(staged, num_tokens)
            return super()._ready_rows(num_tokens)
        _, cols, head_dim = self._host_raw.shape
        blocks = self._scale_blocks
        raw_words_per_tok = (cols * head_dim) // 4
        scale_words_per_tok = -(-(cols * blocks) // 4)
        assert self._active_raw is not None
        if dp_size > 1:
            # Exchange the packed words and dequantize only this replica's
            # tokens, instead of dequantizing all `dp * slot` staged tokens
            # and gathering bf16 rows.
            rows = _dp_gather_packed_rows(
                self._active_raw,
                num_tokens,
                cols=cols,
                head_dim=head_dim,
                blocks=blocks,
                block_size=self.embed_tokens.block_size,
            )
            if rows is not None:
                return rows
        staged_tokens = num_tokens if dp_size == 1 else self._active_raw.shape[0]
        raw_bytes = staged_tokens * cols * head_dim
        scale_bytes = staged_tokens * cols * blocks
        by_word = self._active_raw.view(
            raw_words_per_tok + scale_words_per_tok, staged_tokens
        )
        raw = _unpack_bytes(by_word[:raw_words_per_tok].view(-1), raw_bytes)
        scale = _unpack_bytes(by_word[raw_words_per_tok:].view(-1), scale_bytes)
        rows = dequant_fp8_rows(
            raw.reshape(staged_tokens, cols, head_dim),
            scale.reshape(staged_tokens, cols, blocks),
            self.embed_tokens.block_size,
        )
        return _dp_gather_engram_rows(rows, num_tokens) if dp_size > 1 else rows

    def _sp_embed(self, hash_ids):
        """Rows for this rank's SP token chunk with all hash heads.

        Each rank stages ``[T, local_heads, dim]`` rows for every token. A
        token-axis all-gather yields ``[tp, T_pad, local_heads, dim]`` whose
        rank-major head order matches upstream's non-SP ``dim=1`` gather.
        """
        rows = self._ready_rows(hash_ids.shape[0])
        cols = self.embed_tokens.n_hash_cols
        tp = self.embed_tokens.tp_size
        num_tokens = rows.shape[0]
        chunk = -(-num_tokens // tp)
        rank = upstream.get_tensor_model_parallel_rank()
        if tp == 1:
            return rows[:, :cols]
        pad = chunk * tp - num_tokens
        if pad:
            rows = torch.nn.functional.pad(rows, (0, 0, 0, 0, 0, pad))
        gathered = _gather_engram_rows(rows, dim=0).unflatten(0, (tp, chunk * tp))
        local = gathered[:, rank * chunk : (rank + 1) * chunk]
        return local.transpose(0, 1).flatten(1, 2)[:, :cols]

    def forward(self, hidden_states, hash_ids, token_mask=None):
        if self.use_sequence_parallel:
            tp = self.embed_tokens.tp_size
            chunk = -(-hash_ids.shape[0] // tp)
            assert hidden_states.shape[0] == chunk, (hidden_states.shape, chunk)
            if token_mask is not None:
                # Padded SP rows keep the gate shut.
                pad = chunk * tp - token_mask.shape[0]
                token_mask = torch.cat([token_mask, token_mask.new_zeros(pad)])
                rank = upstream.get_tensor_model_parallel_rank()
                token_mask = token_mask[rank * chunk : (rank + 1) * chunk]
            embedded = self._sp_embed(hash_ids)
        else:
            embedded = self.embed(hash_ids)
        kv = self.wkv(embedded.flatten(-2))
        keys = (
            kv[:, : self.hc_mult * self.dim]
            .unflatten(-1, (self.hc_mult, self.dim))
            .float()
        )
        values = kv[:, self.hc_mult * self.dim :].float()
        hidden = hidden_states.float()
        dot = (hidden * self.q_weight * self.k_weight * keys).sum(-1)
        dot *= torch.rsqrt(hidden.square().mean(-1) + self.eps)
        dot *= torch.rsqrt(keys.square().mean(-1) + self.eps) * self.dim**-0.5
        gate_input = dot.abs().clamp(min=self.clamp_value).sqrt()
        gate = torch.where(dot < 0, -gate_input, gate_input).sigmoid()
        if token_mask is not None:
            gate = gate * token_mask[:, None]
        return (hidden + gate[:, :, None] * values[:, None, :]).to(hidden_states.dtype)
