# SPDX-License-Identifier: Apache-2.0
"""Request-affine KV block allocation for owner-sharded KV caches.

With ``TPU_DEEPSEEK_V41_KV_SHARD=1``, the KV cache groups that grow with the
context (full-attention specs, such as the DeepSeek-V4.1 main compressed KV and
Indexer K) keep each request on one rank of its tensor-parallel group: the
request's owner. Window-bounded groups (sliding-window and circular-buffer
specs) stay replicated on every rank, so the layers that read them for every
token of the batch need no change. The scheduler keeps one block-id space of
``N = R + tp * P`` ids::

    [0, R)                          replicated region, includes the null block
    [R + o * P, R + (o + 1) * P)    region of owner o

Every rank stores ``R + P`` blocks: the replicated region at the same local ids,
then its own region at local ids ``[R, R + P)``. ``OwnerLayout.local_block_id``
gives the mapping, and ``owner_layout`` derives the layout from the configs, so
the scheduler and the workers agree on it.

A request gets its owner on its first successful allocation and keeps it until
it is freed or preempted. Its blocks in owner-sharded groups come from its
owner's region, and its blocks in replicated groups from the replicated region.
When a running request cannot grow, only blocks of its owner help it, so FCFS
preemption evicts the latest request of the same owner (or the request itself)
instead of the latest request overall (``_queue_same_owner_victim``).

The engine core sizes the KV cache per rank: ``M`` blocks. Workers keep that
config and store ``M = R + P`` blocks, while the scheduler allocates from
``N = R + tp * (M - R)`` ids (``globalize_kv_cache_config``). Workers get the
same layout from ``worker_owner_layout``. Block translation and owner-parallel
attention on the workers are separate changes. With data parallelism, each
replica has its own scheduler and pool, and owners are the ranks of that
replica's TP group.
"""

import contextlib
import contextvars
import dataclasses
import functools
from collections.abc import Callable, Iterator
from typing import Any

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    get_max_concurrency_for_kv_cache_config,
)
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request

from vllm_torchtpu import envs

logger = init_logger(__name__)

# The owner whose region serves allocations and free-block counts.
_CURRENT_OWNER: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "vllm_torchtpu_kv_owner", default=None
)
# Set while a replicated KV cache group allocates.
_IN_REPLICATED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "vllm_torchtpu_kv_replicated", default=False
)
# The layout of pools built in this context. Set while a scheduler is built.
_LAYOUT: contextvars.ContextVar["OwnerLayout | None"] = contextvars.ContextVar(
    "vllm_torchtpu_kv_layout", default=None
)


@dataclasses.dataclass(frozen=True)
class OwnerLayout:
    """How the scheduler's KV block ids map onto the ranks of a TP group."""

    num_replicated: int
    num_owners: int
    blocks_per_owner: int
    num_blocks: int = dataclasses.field(init=False, repr=False, compare=False)
    blocks_per_rank: int = dataclasses.field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.num_replicated < 1 or self.num_owners < 2 or self.blocks_per_owner < 1:
            raise ValueError(f"Invalid owner-sharded KV layout: {self}")
        object.__setattr__(
            self,
            "num_blocks",
            self.num_replicated + self.num_owners * self.blocks_per_owner,
        )
        object.__setattr__(
            self,
            "blocks_per_rank",
            self.num_replicated + self.blocks_per_owner,
        )

    @classmethod
    def split(
        cls, num_blocks: int, num_owners: int, min_replicated: int
    ) -> "OwnerLayout":
        """Split ``num_blocks`` ids into equal owner regions and a replicated
        region of at least ``min_replicated`` ids, which also takes the
        remainder of the division."""
        if num_owners < 2:
            raise ValueError(
                f"Owner-sharded KV needs at least 2 owners, got {num_owners}"
            )
        per_owner = (num_blocks - min_replicated) // num_owners
        return cls(num_blocks - num_owners * per_owner, num_owners, per_owner)

    def region_of(self, block_id: int) -> int:
        """The owner of ``block_id``, or ``num_owners`` if it is replicated."""
        if not 0 <= block_id < self.num_blocks:
            raise ValueError(
                f"KV block {block_id} is outside the {self.num_blocks}-block layout"
            )
        if block_id < self.num_replicated:
            return self.num_owners
        return (block_id - self.num_replicated) // self.blocks_per_owner

    def local_block_id(self, block_id: int, rank: int) -> int:
        """Where ``rank`` stores ``block_id``, or -1 if it does not store it."""
        region = self.region_of(block_id)
        if region == self.num_owners:
            return block_id
        if region == rank:
            return block_id - rank * self.blocks_per_owner
        return -1


def is_replicated_spec(spec: KVCacheSpec) -> bool:
    """Whether every rank stores this group's blocks.

    Window-bounded groups cost a few blocks per request, and the layers that
    read them attend over every token of the batch.
    """
    return isinstance(spec, (SlidingWindowSpec, CircularBufferSpec))


def replicated_blocks_needed(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> int:
    """An upper bound on the blocks the replicated groups hold at once.

    Counts the null block, one ring block per request for each circular-buffer
    group, and the window plus the in-flight tokens for each sliding-window
    group. One request holds ``cdiv(window - 1 + in_flight, block_size) + 1``
    blocks (``SlidingWindowSpec.max_admission_blocks_per_request``). Summed
    over requests, the in-flight tokens are bounded by the step budget instead
    of once per request.
    """
    max_seqs = vllm_config.scheduler_config.max_num_seqs
    spec_config = vllm_config.speculative_config
    lookahead = spec_config.num_speculative_tokens if spec_config is not None else 0
    in_flight = vllm_config.max_in_flight_tokens + max_seqs * lookahead
    total = 1
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, CircularBufferSpec):
            total += max_seqs
        elif isinstance(spec, SlidingWindowSpec):
            window = spec.sliding_window - 1 + spec.extra_retained_tokens
            total += max_seqs * (cdiv(window, spec.block_size) + 2)
            total += cdiv(in_flight, spec.block_size)
    return total


def owner_layout(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> OwnerLayout:
    """The block layout that the scheduler and every worker must agree on."""
    return OwnerLayout.split(
        kv_cache_config.num_blocks,
        vllm_config.parallel_config.tensor_parallel_size,
        replicated_blocks_needed(vllm_config, kv_cache_config),
    )


def _owned_groups(kv_cache_config: KVCacheConfig) -> list[KVCacheGroupSpec]:
    return [
        group
        for group in kv_cache_config.kv_cache_groups
        if not is_replicated_spec(group.kv_cache_spec)
    ]


def globalize_kv_cache_config(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> KVCacheConfig:
    """The scheduler's copy of one rank's ``kv_cache_config``.

    Every rank stores ``M = kv_cache_config.num_blocks`` blocks: the ``R``
    replicated ones and ``P = M - R`` of its own region. The scheduler hands
    out ``N = R + tp * P`` ids. Only ``num_blocks`` changes: the workers keep
    the per-rank config, so the KV cache tensors keep their size.
    """
    if not _owned_groups(kv_cache_config):
        raise ValueError(
            "TPU_DEEPSEEK_V41_KV_SHARD=1 but no KV cache group is owner-sharded"
        )
    num_replicated = replicated_blocks_needed(vllm_config, kv_cache_config)
    per_owner = kv_cache_config.num_blocks - num_replicated
    if per_owner < 1:
        raise ValueError(
            f"TPU_DEEPSEEK_V41_KV_SHARD=1 needs more than {num_replicated} KV "
            "blocks per rank for the replicated cache groups, but each rank "
            f"holds {kv_cache_config.num_blocks}; raise the KV cache memory or "
            "lower max_num_seqs"
        )
    layout = OwnerLayout(
        num_replicated, vllm_config.parallel_config.tensor_parallel_size, per_owner
    )
    return dataclasses.replace(kv_cache_config, num_blocks=layout.num_blocks)


def worker_owner_layout(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> OwnerLayout:
    """The scheduler's layout, derived from a worker's per-rank config.

    Workers store ``layout.blocks_per_rank`` blocks, which equals their
    ``kv_cache_config.num_blocks``, and translate the scheduler's block ids
    with ``layout.local_block_id``.
    """
    return owner_layout(
        vllm_config, globalize_kv_cache_config(vllm_config, kv_cache_config)
    )


def max_concurrency(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig, layout: OwnerLayout
) -> float:
    """How many ``max_model_len`` requests the owner regions hold.

    Only the owner-sharded groups count: ``replicated_blocks_needed`` sizes
    the replicated region for ``max_num_seqs`` requests. Like vLLM's own
    estimate, the result is fractional; it also ignores that a request cannot
    span owners.
    """
    owned = dataclasses.replace(
        kv_cache_config,
        num_blocks=layout.num_owners * layout.blocks_per_owner,
        kv_cache_groups=_owned_groups(kv_cache_config),
    )
    return get_max_concurrency_for_kv_cache_config(vllm_config, owned)


@contextlib.contextmanager
def owner_scope(owner: int) -> Iterator[None]:
    """Route block allocation and free-block counts to ``owner``'s region."""
    token = _CURRENT_OWNER.set(owner)
    try:
        yield
    finally:
        _CURRENT_OWNER.reset(token)


@contextlib.contextmanager
def replicated_scope() -> Iterator[None]:
    """Route block allocation and free-block counts to the replicated region."""
    token = _IN_REPLICATED.set(True)
    try:
        yield
    finally:
        _IN_REPLICATED.reset(token)


@contextlib.contextmanager
def layout_scope(layout: OwnerLayout) -> Iterator[None]:
    """Give every ``OwnerAffineBlockPool`` built inside this ``layout``."""
    token = _LAYOUT.set(layout)
    try:
        yield
    finally:
        _LAYOUT.reset(token)


class RegionFreeQueue(FreeKVCacheBlockQueue):
    """A ``FreeKVCacheBlockQueue`` that keeps one independent queue per region.

    The regions are the owners ``0 .. num_owners - 1`` and the replicated
    region at index ``num_owners``. ``BlockPool`` reaches its free queue only
    through these methods, so the inherited allocation and free paths keep
    working unchanged. That includes the plugin's LIFO free order
    (``_patch_vllm_block_pool_lifo_free``). Pops come from the region in scope,
    pushes are routed by block id, and every region keeps its own order.
    ``num_free_blocks`` is the total over all regions.
    """

    def __init__(
        self, layout: OwnerLayout, region_blocks: list[list[KVCacheBlock]]
    ) -> None:
        super().__init__([])
        self.layout = layout
        self._num_owners = layout.num_owners
        self.queues = [FreeKVCacheBlockQueue(blocks) for blocks in region_blocks]
        self.num_free_blocks = sum(len(blocks) for blocks in region_blocks)

    def scoped_queue(self) -> FreeKVCacheBlockQueue:
        region = self._num_owners if _IN_REPLICATED.get() else _CURRENT_OWNER.get()
        if region is None:
            raise RuntimeError(
                "KV blocks were requested outside an owner scope. With "
                "TPU_DEEPSEEK_V41_KV_SHARD=1, every allocation must run inside "
                "owner_scope() or replicated_scope()."
            )
        return self.queues[region]

    def _by_region(self, blocks: list[KVCacheBlock]) -> dict[int, list[KVCacheBlock]]:
        # Group before touching any queue, so that a bad id changes nothing.
        region_of = self.layout.region_of
        groups: dict[int, list[KVCacheBlock]] = {}
        for block in blocks:
            region = region_of(block.block_id)
            group = groups.get(region)
            if group is None:
                groups[region] = [block]
            else:
                group.append(block)
        return groups

    def popleft(self) -> KVCacheBlock:
        block = self.scoped_queue().popleft()
        self.num_free_blocks -= 1
        return block

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        blocks = self.scoped_queue().popleft_n(n)
        self.num_free_blocks -= n
        return blocks

    def remove(self, block: KVCacheBlock) -> None:
        self.queues[self.layout.region_of(block.block_id)].remove(block)
        self.num_free_blocks -= 1

    def append(self, block: KVCacheBlock) -> None:
        self.queues[self.layout.region_of(block.block_id)].append(block)
        self.num_free_blocks += 1

    def prepend_n(self, blocks: list[KVCacheBlock]) -> None:
        for region, group in self._by_region(blocks).items():
            self.queues[region].prepend_n(group)
        self.num_free_blocks += len(blocks)

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        for region, group in self._by_region(blocks).items():
            self.queues[region].append_n(group)
        self.num_free_blocks += len(blocks)

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        return [b for queue in self.queues for b in queue.get_all_free_blocks()]

    def iter_blocks_after(self, cursor: KVCacheBlock | None) -> Iterator[KVCacheBlock]:
        # Only prefix-cache eviction walks the free list in order.
        raise NotImplementedError("Owner-sharded KV runs without prefix caching")


class OwnerAffineBlockPool(BlockPool):
    """A ``BlockPool`` whose block ids are split into an ``OwnerLayout``.

    Allocation draws from the region in scope (``owner_scope`` or
    ``replicated_scope``). Outside a scope, ``get_num_free_blocks`` reports the
    total over all regions (for metrics) and allocation raises. A freed block
    returns to the region its id belongs to, so deferred frees need no scope.
    """

    def __init__(self, *args: Any, layout: OwnerLayout | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if layout is None:
            layout = _LAYOUT.get()
        if layout is None:
            raise RuntimeError(
                "OwnerAffineBlockPool needs a layout: build it inside a "
                "scheduler or layout_scope()"
            )
        if self.enable_caching:
            raise ValueError("Owner-sharded KV requires prefix caching to be off")
        if layout.num_blocks != self.num_gpu_blocks:
            raise ValueError(
                f"{layout} does not cover the pool's {self.num_gpu_blocks} blocks"
            )
        regions: list[list[KVCacheBlock]] = [[] for _ in range(layout.num_owners + 1)]
        for block in self.blocks:
            if not block.is_null:
                regions[layout.region_of(block.block_id)].append(block)
        self.layout = layout
        self.region_free_queue = RegionFreeQueue(layout, regions)
        self.free_block_queue = self.region_free_queue
        self._all_owners = tuple(range(layout.num_owners))
        self._owner_queues = tuple(self.region_free_queue.queues[: layout.num_owners])
        self._replicated_queue = self.region_free_queue.queues[layout.num_owners]
        # Replicated-group blocks counted by the admission check in progress.
        self.pending_replicated_blocks = 0
        self._request_owner: dict[str, int] = {}
        self._owner_num_requests = [0] * layout.num_owners
        self._step_owner_budget: list[int] | None = None
        self._step_req_tokens: dict[str, int] = {}
        logger.info(
            "Owner-sharded KV block pool: %d replicated blocks + %d owners x %d",
            layout.num_replicated,
            layout.num_owners,
            layout.blocks_per_owner,
        )

    def begin_step(self, per_owner_budget: int) -> None:
        self._step_owner_budget = [per_owner_budget] * self.layout.num_owners
        self._step_req_tokens.clear()

    def end_step(self) -> None:
        self._step_owner_budget = None
        self._step_req_tokens.clear()

    def step_owner_budget(self, owner: int) -> int | None:
        if self._step_owner_budget is None:
            return None
        return self._step_owner_budget[owner]

    def record_step_tokens(self, request_id: str, owner: int, num_tokens: int) -> None:
        step_budget = self._step_owner_budget
        if step_budget is not None and num_tokens > 0:
            step_budget[owner] -= num_tokens
            step_req = self._step_req_tokens
            step_req[request_id] = step_req.get(request_id, 0) + num_tokens

    def get_num_free_blocks(self) -> int:
        if _IN_REPLICATED.get():
            return self._replicated_queue.num_free_blocks
        owner = _CURRENT_OWNER.get()
        if owner is None:
            return self.region_free_queue.num_free_blocks
        return self._owner_queues[owner].num_free_blocks

    def get_usage(self) -> float:
        # The total over all regions: the base version would report one
        # region's share whenever it runs inside a scope.
        return 1.0 - self.region_free_queue.num_free_blocks / (self.num_gpu_blocks - 1)

    @property
    def num_free_replicated_blocks(self) -> int:
        return self._replicated_queue.num_free_blocks

    def owner_free_blocks(self) -> list[int]:
        return [queue.num_free_blocks for queue in self._owner_queues]

    def owner_of(self, request_id: str) -> int | None:
        return self._request_owner.get(request_id)

    def best_owner_candidate(self) -> int | None:
        """Single best candidate owner for a new request, or ``None`` if every
        owner's step token budget is exhausted."""
        req_counts = self._owner_num_requests
        owner_queues = self._owner_queues
        budget = self._step_owner_budget
        best_owner: int | None = None
        best_reqs = 0
        best_free = -1
        for o in self._all_owners:
            if budget is not None and budget[o] <= 0:
                continue
            reqs = req_counts[o]
            free = owner_queues[o].num_free_blocks
            if (
                best_owner is None
                or reqs < best_reqs
                or (reqs == best_reqs and free > best_free)
            ):
                best_owner = o
                best_reqs = reqs
                best_free = free
        return best_owner

    def owner_candidates(self) -> list[int]:
        """Owners to try for a new request, best first.

        Fewest live requests comes first: it keeps decode tokens, and with
        them the per-rank attention segments, balanced. Most free blocks
        breaks ties.
        """
        req_counts = self._owner_num_requests
        owner_queues = self._owner_queues
        budget = self._step_owner_budget
        owners = (
            self._all_owners
            if budget is None
            else [o for o in self._all_owners if budget[o] > 0]
        )
        return sorted(
            owners,
            key=lambda o: (req_counts[o], -owner_queues[o].num_free_blocks),
        )

    def bind(self, request_id: str, owner: int) -> None:
        if request_id in self._request_owner:
            raise ValueError(f"Request {request_id} already has a KV owner")
        self._request_owner[request_id] = owner
        self._owner_num_requests[owner] += 1

    def release(self, request_id: str) -> None:
        owner = self._request_owner.pop(request_id, None)
        if owner is not None:
            self._owner_num_requests[owner] -= 1
            restored = self._step_req_tokens.pop(request_id, 0)
            if restored and self._step_owner_budget is not None:
                self._step_owner_budget[owner] += restored


def validate_config(vllm_config: VllmConfig) -> None:
    """Reject configurations that owner-sharded KV does not support."""
    parallel = vllm_config.parallel_config
    problems = []
    if parallel.tensor_parallel_size < 2:
        problems.append("tensor_parallel_size must be at least 2")
    if parallel.pipeline_parallel_size != 1:
        problems.append("pipeline parallelism is not supported")
    if parallel.decode_context_parallel_size != 1:
        problems.append("decode context parallelism is not supported")
    if parallel.prefill_context_parallel_size != 1:
        problems.append("prefill context parallelism is not supported")
    if vllm_config.cache_config.enable_prefix_caching:
        problems.append("prefix caching must be disabled")
    if vllm_config.kv_transfer_config is not None:
        problems.append("KV transfer connectors are not supported")
    if envs.TPU_DEEPSEEK_V41_IMPL != "v2":
        problems.append("TPU_DEEPSEEK_V41_IMPL must be v2")
    if envs.TPU_DEEPSEEK_V41_SP:
        problems.append("TPU_DEEPSEEK_V41_SP must be off")
    if envs.TPU_DEEPSEEK_V41_LOCAL_Q:
        problems.append("TPU_DEEPSEEK_V41_LOCAL_Q must be off")
    if problems:
        raise ValueError("TPU_DEEPSEEK_V41_KV_SHARD=1 but " + "; ".join(problems))


def _in_replicated_scope(method: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(method)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        token = _IN_REPLICATED.set(True)
        try:
            return method(*args, **kwargs)
        finally:
            _IN_REPLICATED.reset(token)

    return wrapper


def _count_as_replicated(
    method: Callable[..., int], pool: OwnerAffineBlockPool
) -> Callable[..., int]:
    @functools.wraps(method)
    def get_num_blocks_to_allocate(*args: Any, **kwargs: Any) -> int:
        pool.pending_replicated_blocks += method(*args, **kwargs)
        return 0

    return get_num_blocks_to_allocate


def _check_replicated(
    method: Callable[..., int], pool: OwnerAffineBlockPool
) -> Callable[..., int]:
    replicated_queue = pool._replicated_queue

    @functools.wraps(method)
    def get_num_blocks_to_allocate(*args: Any, **kwargs: Any) -> int:
        pool.pending_replicated_blocks = 0
        num_blocks = method(*args, **kwargs)
        if pool.pending_replicated_blocks > replicated_queue.num_free_blocks:
            # More than any free count, so the request does not fit.
            return pool.num_gpu_blocks
        return num_blocks

    return get_num_blocks_to_allocate


def _route_replicated_groups(coordinator: Any, pool: OwnerAffineBlockPool) -> None:
    """Allocate the replicated groups' blocks from the replicated region.

    Admission then compares the owner-sharded groups' blocks with the owner's
    free blocks, and the replicated groups' blocks with the replicated
    region's free blocks.
    """
    for manager in coordinator.single_type_managers:
        if not is_replicated_spec(manager.kv_cache_spec):
            continue
        manager.allocate_new_blocks = _in_replicated_scope(manager.allocate_new_blocks)
        manager.allocate_external_computed_blocks = _in_replicated_scope(
            manager.allocate_external_computed_blocks
        )
        manager.get_num_blocks_to_allocate = _count_as_replicated(
            manager.get_num_blocks_to_allocate, pool
        )
    coordinator.get_num_blocks_to_allocate = _check_replicated(
        coordinator.get_num_blocks_to_allocate, pool
    )


def _arg(args: tuple[Any, ...], kwargs: dict[str, Any], name: str, index: int) -> Any:
    return kwargs[name] if name in kwargs else args[index]


def _with_arg(
    args: tuple[Any, ...], kwargs: dict[str, Any], name: str, index: int, value: Any
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if name in kwargs:
        return args, {**kwargs, name: value}
    return (*args[:index], value, *args[index + 1 :]), kwargs


def _publish_capacity(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig, layout: OwnerLayout
) -> None:
    """Report the scheduler's block count and capacity in place of one rank's.

    The engine core set them from the per-rank config before building the
    scheduler; the frontend reads them from ``cache_config``.
    """
    concurrency = max_concurrency(vllm_config, kv_cache_config, layout)
    max_model_len = vllm_config.model_config.max_model_len
    cache_config = vllm_config.cache_config
    cache_config.num_gpu_blocks = layout.num_blocks
    cache_config.kv_cache_max_concurrency = concurrency
    cache_config.kv_cache_size_tokens = int(concurrency * max_model_len)
    logger.info(
        "Owner-sharded KV: %d blocks per rank (%d replicated + %d owned) give "
        "the scheduler %d blocks over %d owners. Maximum concurrency for %s "
        "tokens per request: %.2fx",
        layout.blocks_per_rank,
        layout.num_replicated,
        layout.blocks_per_owner,
        layout.num_blocks,
        layout.num_owners,
        f"{max_model_len:,}",
        concurrency,
    )


def _queue_same_owner_victim(
    running: list[Request], request: Request, pool: OwnerAffineBlockPool
) -> None:
    """Put the request that FCFS preemption should evict at ``running[-1]``.

    ``request`` could not grow on its owner, and only that owner's blocks
    help it. Evicting a request of another owner would discard its KV and
    free nothing ``request`` can use. So the last request of the same owner
    behind ``request`` goes to the end, or ``request`` itself if there is
    none. The requests ahead of ``request`` are already scheduled this step
    and stay where they are.
    """
    owner = pool.owner_of(request.request_id)
    if owner is None or request not in running:
        return
    index = running.index(request)
    for i in range(len(running) - 1, index, -1):
        if pool.owner_of(running[i].request_id) == owner:
            running.append(running.pop(i))
            return
    running.append(running.pop(index))


def _steer_preemption(scheduler: Any) -> None:
    """Make ``scheduler``'s FCFS preemption evict requests of the right owner
    and cap each owner's scheduled tokens per step to ``1 / num_owners`` of the
    step token budget so ``owner_perm`` fits within the step's compile bucket.
    """
    manager = scheduler.kv_cache_manager
    pool = manager.block_pool
    if not isinstance(pool, OwnerAffineBlockPool):
        return
    if (
        hasattr(scheduler, "schedule")
        and hasattr(scheduler, "_reserve_prefill_lookahead")
        and isinstance(getattr(scheduler, "max_num_scheduled_tokens", None), int)
    ):
        orig_schedule = scheduler.schedule
        orig_reserve = scheduler._reserve_prefill_lookahead
        num_owners = pool.layout.num_owners
        request_owner = pool._request_owner

        @functools.wraps(orig_schedule)
        def budgeted_schedule(*args: Any, **kwargs: Any) -> Any:
            per_owner = max(1, scheduler.max_num_scheduled_tokens // num_owners)
            pool.begin_step(per_owner)
            try:
                return orig_schedule(*args, **kwargs)
            finally:
                pool.end_step()

        @functools.wraps(orig_reserve)
        def budgeted_reserve_prefill_lookahead(
            request: Request, num_computed_tokens: int, num_new_tokens: int
        ) -> int:
            step_budget = pool._step_owner_budget
            owner = request_owner.get(request.request_id)
            if owner is not None:
                owner_budget = step_budget[owner] if step_budget is not None else None
            else:
                best = pool.best_owner_candidate()
                owner_budget = (
                    (step_budget[best] if step_budget is not None else None)
                    if best is not None
                    else 0
                )
            if owner_budget is not None and num_new_tokens > owner_budget:
                if (
                    num_new_tokens > 1
                    and request.num_tokens - num_computed_tokens == 1
                    and request not in scheduler.running
                ):
                    return 0
                num_new_tokens = max(0, owner_budget)
            return orig_reserve(request, num_computed_tokens, num_new_tokens)

        scheduler.schedule = budgeted_schedule
        scheduler._reserve_prefill_lookahead = budgeted_reserve_prefill_lookahead

    if scheduler.policy != SchedulingPolicy.FCFS:
        return
    allocate_slots = manager.allocate_slots

    @functools.wraps(allocate_slots)
    def steered_allocate_slots(request: Request, *args: Any, **kwargs: Any) -> Any:
        blocks = allocate_slots(request, *args, **kwargs)
        if blocks is None:
            _queue_same_owner_victim(scheduler.running, request, pool)
        return blocks

    manager.allocate_slots = steered_allocate_slots


def install() -> None:
    """Make schedulers allocate KV blocks from ``OwnerAffineBlockPool``.

    Rebinds the ``BlockPool`` name that ``KVCacheCoordinator`` resolves at
    construction. Wraps ``Scheduler.__init__`` so that the config is
    validated, the scheduler allocates from every owner's region, the pool
    gets its layout, and FCFS preemption evicts requests of the right owner.
    Wraps ``KVCacheManager`` so that each request's allocations run inside
    its owner's scope and the replicated groups allocate from the replicated
    region.
    """
    from vllm.v1.core import kv_cache_coordinator
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.scheduler import Scheduler

    if KVCacheManager.__dict__.get("_tpu_owner_affine_kv", False):
        return
    kv_cache_coordinator.BlockPool = OwnerAffineBlockPool

    orig_scheduler_init = Scheduler.__init__
    orig_init = KVCacheManager.__init__
    orig_allocate_slots = KVCacheManager.allocate_slots
    orig_free = KVCacheManager.free
    orig_pop_blocks_for_free = KVCacheManager.pop_blocks_for_free

    @functools.wraps(orig_scheduler_init)
    def scheduler_init(self, *args, **kwargs):
        vllm_config = _arg(args, kwargs, "vllm_config", 0)
        kv_cache_config = _arg(args, kwargs, "kv_cache_config", 1)
        validate_config(vllm_config)
        # The engine core passes one rank's config; the workers keep it.
        kv_cache_config = globalize_kv_cache_config(vllm_config, kv_cache_config)
        args, kwargs = _with_arg(args, kwargs, "kv_cache_config", 1, kv_cache_config)
        layout = owner_layout(vllm_config, kv_cache_config)
        _publish_capacity(vllm_config, kv_cache_config, layout)
        with layout_scope(layout):
            orig_scheduler_init(self, *args, **kwargs)
        _steer_preemption(self)

    @functools.wraps(orig_init)
    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        pool = self.block_pool
        if isinstance(pool, OwnerAffineBlockPool):
            # Admission compares against one owner's free blocks, so the
            # watermark must be one owner's share of it.
            self.watermark_blocks = cdiv(self.watermark_blocks, pool.layout.num_owners)
            _route_replicated_groups(self.coordinator, pool)

    @functools.wraps(orig_allocate_slots)
    def allocate_slots(self, request, *args, **kwargs):
        pool = self.block_pool
        if type(pool) is not OwnerAffineBlockPool and not isinstance(
            pool, OwnerAffineBlockPool
        ):
            return orig_allocate_slots(self, request, *args, **kwargs)
        num_new_tokens = int(args[0]) if args else int(kwargs.get("num_new_tokens", 0))
        req_id = request.request_id
        owner = pool._request_owner.get(req_id)
        if owner is not None:
            token = _CURRENT_OWNER.set(owner)
            try:
                blocks = orig_allocate_slots(self, request, *args, **kwargs)
            finally:
                _CURRENT_OWNER.reset(token)
            if blocks is not None:
                pool.record_step_tokens(req_id, owner, num_new_tokens)
            return blocks
        # A request without an owner holds no blocks, and a failed attempt
        # returns before allocating anything, so trying owners in turn has
        # no side effects. Try the single best candidate before sorting all.
        best = pool.best_owner_candidate()
        if best is None:
            return None
        best_budget = pool.step_owner_budget(best)
        if best_budget is None or best_budget >= num_new_tokens:
            token = _CURRENT_OWNER.set(best)
            try:
                blocks = orig_allocate_slots(self, request, *args, **kwargs)
            finally:
                _CURRENT_OWNER.reset(token)
            if blocks is not None:
                pool.bind(req_id, best)
                pool.record_step_tokens(req_id, best, num_new_tokens)
                return blocks
        for owner in pool.owner_candidates():
            if owner == best:
                continue
            owner_budget = pool.step_owner_budget(owner)
            if owner_budget is not None and owner_budget < num_new_tokens:
                continue
            token = _CURRENT_OWNER.set(owner)
            try:
                blocks = orig_allocate_slots(self, request, *args, **kwargs)
            finally:
                _CURRENT_OWNER.reset(token)
            if blocks is not None:
                pool.bind(req_id, owner)
                pool.record_step_tokens(req_id, owner, num_new_tokens)
                return blocks
        return None

    @functools.wraps(orig_free)
    def free(self, request):
        orig_free(self, request)
        pool = self.block_pool
        if type(pool) is OwnerAffineBlockPool or isinstance(pool, OwnerAffineBlockPool):
            pool.release(request.request_id)

    @functools.wraps(orig_pop_blocks_for_free)
    def pop_blocks_for_free(self, request):
        blocks = orig_pop_blocks_for_free(self, request)
        pool = self.block_pool
        if type(pool) is OwnerAffineBlockPool or isinstance(pool, OwnerAffineBlockPool):
            pool.release(request.request_id)
        return blocks

    Scheduler.__init__ = scheduler_init
    KVCacheManager.__init__ = __init__
    KVCacheManager.allocate_slots = allocate_slots
    KVCacheManager.free = free
    KVCacheManager.pop_blocks_for_free = pop_blocks_for_free
    KVCacheManager._tpu_owner_affine_kv = True
    logger.info("Applied TPU patch: owner-affine KV block allocation")
