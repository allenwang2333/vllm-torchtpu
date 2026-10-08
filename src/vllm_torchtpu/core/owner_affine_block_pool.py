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
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    SlidingWindowSpec,
)

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

    def __post_init__(self) -> None:
        if self.num_replicated < 1 or self.num_owners < 2 or self.blocks_per_owner < 1:
            raise ValueError(f"Invalid owner-sharded KV layout: {self}")

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

    @property
    def num_blocks(self) -> int:
        return self.num_replicated + self.num_owners * self.blocks_per_owner

    @property
    def blocks_per_rank(self) -> int:
        return self.num_replicated + self.blocks_per_owner

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
        self.queues = [FreeKVCacheBlockQueue(blocks) for blocks in region_blocks]
        self.num_free_blocks = sum(len(blocks) for blocks in region_blocks)

    def scoped_region(self) -> int | None:
        if _IN_REPLICATED.get():
            return self.layout.num_owners
        return _CURRENT_OWNER.get()

    def scoped_queue(self) -> FreeKVCacheBlockQueue:
        region = self.scoped_region()
        if region is None:
            raise RuntimeError(
                "KV blocks were requested outside an owner scope. With "
                "TPU_DEEPSEEK_V41_KV_SHARD=1, every allocation must run inside "
                "owner_scope() or replicated_scope()."
            )
        return self.queues[region]

    def _by_region(self, blocks: list[KVCacheBlock]) -> dict[int, list[KVCacheBlock]]:
        # Group before touching any queue, so that a bad id changes nothing.
        groups: dict[int, list[KVCacheBlock]] = {}
        for block in blocks:
            groups.setdefault(self.layout.region_of(block.block_id), []).append(block)
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
        # Replicated-group blocks counted by the admission check in progress.
        self.pending_replicated_blocks = 0
        self._request_owner: dict[str, int] = {}
        self._owner_num_requests = [0] * layout.num_owners
        logger.info(
            "Owner-sharded KV block pool: %d replicated blocks + %d owners x %d",
            layout.num_replicated,
            layout.num_owners,
            layout.blocks_per_owner,
        )

    def get_num_free_blocks(self) -> int:
        region = self.region_free_queue.scoped_region()
        if region is None:
            return self.region_free_queue.num_free_blocks
        return self.region_free_queue.queues[region].num_free_blocks

    def get_usage(self) -> float:
        # The total over all regions: the base version would report one
        # region's share whenever it runs inside a scope.
        return 1.0 - self.region_free_queue.num_free_blocks / (self.num_gpu_blocks - 1)

    @property
    def num_free_replicated_blocks(self) -> int:
        return self.region_free_queue.queues[self.layout.num_owners].num_free_blocks

    def owner_free_blocks(self) -> list[int]:
        owners = self.region_free_queue.queues[: self.layout.num_owners]
        return [queue.num_free_blocks for queue in owners]

    def owner_of(self, request_id: str) -> int | None:
        return self._request_owner.get(request_id)

    def owner_candidates(self) -> list[int]:
        """Owners to try for a new request, best first.

        Fewest live requests comes first: it keeps decode tokens, and with
        them the per-rank attention segments, balanced. Most free blocks
        breaks ties.
        """
        free = self.owner_free_blocks()
        return sorted(
            range(self.layout.num_owners),
            key=lambda o: (self._owner_num_requests[o], -free[o], o),
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
        with replicated_scope():
            return method(*args, **kwargs)

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
    @functools.wraps(method)
    def get_num_blocks_to_allocate(*args: Any, **kwargs: Any) -> int:
        pool.pending_replicated_blocks = 0
        num_blocks = method(*args, **kwargs)
        if pool.pending_replicated_blocks > pool.num_free_replicated_blocks:
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


def install() -> None:
    """Make schedulers allocate KV blocks from ``OwnerAffineBlockPool``.

    Rebinds the ``BlockPool`` name that ``KVCacheCoordinator`` resolves at
    construction. Wraps ``Scheduler.__init__`` so that the config is
    validated, the scheduler allocates from every owner's region, and the
    pool gets its layout. Wraps ``KVCacheManager`` so that each request's
    allocations run inside its owner's scope and the replicated groups
    allocate from the replicated region.
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
        if not isinstance(pool, OwnerAffineBlockPool):
            return orig_allocate_slots(self, request, *args, **kwargs)
        owner = pool.owner_of(request.request_id)
        if owner is not None:
            with owner_scope(owner):
                return orig_allocate_slots(self, request, *args, **kwargs)
        # A request without an owner holds no blocks, and a failed attempt
        # returns before allocating anything, so trying owners in turn has
        # no side effects.
        for owner in pool.owner_candidates():
            with owner_scope(owner):
                blocks = orig_allocate_slots(self, request, *args, **kwargs)
            if blocks is not None:
                pool.bind(request.request_id, owner)
                return blocks
        return None

    @functools.wraps(orig_free)
    def free(self, request):
        orig_free(self, request)
        if isinstance(self.block_pool, OwnerAffineBlockPool):
            self.block_pool.release(request.request_id)

    @functools.wraps(orig_pop_blocks_for_free)
    def pop_blocks_for_free(self, request):
        blocks = orig_pop_blocks_for_free(self, request)
        if isinstance(self.block_pool, OwnerAffineBlockPool):
            self.block_pool.release(request.request_id)
        return blocks

    Scheduler.__init__ = scheduler_init
    KVCacheManager.__init__ = __init__
    KVCacheManager.allocate_slots = allocate_slots
    KVCacheManager.free = free
    KVCacheManager.pop_blocks_for_free = pop_blocks_for_free
    KVCacheManager._tpu_owner_affine_kv = True
    logger.info("Applied TPU patch: owner-affine KV block allocation")
