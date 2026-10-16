"""Intra-chip TP for DSv4 attention weights, with KV and attention kept DP.

Under DP=16/TP=1 every core holds all of `wq_b`, `wo_a` and `wo_b`, roughly
285 MB per layer. `VllmDeepseekV4MLAAttentionIntraChip` splits those three
across the two cores of a chip: each core keeps half the heads of `wq_b` and
half the o-groups of `wo_a`/`wo_b`. vLLM still sees DP=16/TP=1, so the
scheduler, router, KV pool and the sparse-attention kernels are unchanged and
run over all 128 heads for this rank's own sequences.

Per layer, inside the chip pair (tokens are padded to the same bucket on every
DP rank, so shapes match):
  1. all-gather `qr` -> q for my 64 heads over both cores' tokens
  2. swap halves -> q for all 128 heads over my tokens; attention runs locally
  3. swap halves back -> o for my 64 heads over both cores' tokens
  4. fused inverse-RoPE + `wo_a` on my 8 groups, `wo_b` on my 8192 rows
  5. reduce-scatter -> my tokens' full output

Every collective is issued here, over the chip-pair group; none are left to
the compiler. Experimental: `DeepseekV4DecoderLayer` uses this class when
TPU_DSV4_ATTN_INTRA_CHIP_TP=1.
"""

from __future__ import annotations

import functools

import jax
import torch
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas
from vllm.distributed import parallel_state
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear

from vllm_torchtpu.kernels.deepseek_v4.o_projection import (
    fused_reverse_rope_wo_a_projection_pair,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.deepseek_v4.attention import (
    VllmDeepseekV4MLAAttention,
    _pallas_op_cache,
)
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import get_pallas_op_mesh

logger = init_logger(__name__)

_group = None
_initialized = False


def _pair_groups() -> list[list[int]]:
    """Global ranks grouped by physical chip, from each rank's device id."""
    from vllm_torchtpu.distributed.chip_topology import get_chip_topology
    from vllm_torchtpu.distributed.mesh_utils import (
        _collect_rank_to_device_id,
        _get_current_global_rank,
        _get_tpu_global_device_id,
    )

    world = parallel_state.get_world_group()
    rank_to_device = _collect_rank_to_device_id(
        world, _get_current_global_rank(), _get_tpu_global_device_id()
    )
    # tpu7x numbers a chip's two TensorCores 2k and 2k+1. Where this process
    # can see device coordinates, they must agree with that numbering.
    chip_of_device = {d: d // 2 for d in rank_to_device.values()}
    topology = get_chip_topology()
    if topology is not None:
        for chip in topology.chips:
            if len({d // 2 for d in chip}) != 1:
                raise RuntimeError(
                    f"ATTN_INTRA_CHIP_TP: chip {chip} does not pair as device_id // 2"
                )
    by_chip: dict[int, list[int]] = {}
    for rank, device in rank_to_device.items():
        by_chip.setdefault(chip_of_device[device], []).append(rank)
    groups = [sorted(by_chip[c]) for c in sorted(by_chip)]
    logger.info(
        "ATTN_INTRA_CHIP_TP: chip pairs %s (rank->device %s)",
        groups,
        dict(sorted(rank_to_device.items())),
    )
    if any(len(g) != 2 for g in groups):
        raise RuntimeError(f"ATTN_INTRA_CHIP_TP needs 2 cores per chip, got {groups}")
    return groups


def get_chip_pair_group():
    """The GroupCoordinator over this rank's chip pair. Collective on first call."""
    global _group, _initialized
    if _initialized:
        return _group
    _initialized = True
    import torch.distributed as dist

    world = parallel_state.get_world_group()
    _group = parallel_state.init_model_parallel_group(
        _pair_groups(),
        world.local_rank,
        dist.get_backend(world.device_group),
        group_name="attn_intra_chip",
    )
    return _group


def _use_my_slice(layer: torch.nn.Module, rank: int, size: int) -> None:
    """Make `layer` load slice `rank` of `size` from the full checkpoint tensors.

    vLLM's loaders slice by `tp_rank`/`tp_size`, kept on the layer (unquantized
    weights) and on each quantized parameter (block-fp8 weights and scales).
    Both are read from the global TP group at construction, which is size 1
    here, so they are set to this core's place in the chip pair instead.
    """
    layer.tp_rank, layer.tp_size = rank, size
    for param in layer.parameters():
        if hasattr(param, "tp_rank"):
            param.tp_rank, param.tp_size = rank, size


def _o_proj_pair_jax(
    first: jax.Array,
    second: jax.Array,
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    wo_a: jax.Array,
    wo_a_scale: jax.Array,
    *,
    rank: int,
    half: int,
    head_dim: int,
) -> jax.Array:
    """wo_a over [core 0 tokens; core 1 tokens]; see `o_proj_pair_op`."""
    num_tokens = positions.shape[0] // 2
    # Rank 0: first = my o (heads [0, H/2)), second = gathered rows [T, 2T).
    # Rank 1: first = gathered rows [0, T), second = my o (heads [H/2, H)).
    second_row0, second_head0 = (num_tokens, 0) if rank == 0 else (0, half)
    return fused_reverse_rope_wo_a_projection_pair(
        first,
        second,
        positions,
        cos_sin_cache,
        wo_a,
        wo_a_scale.reshape(-1),
        first_row0=0,
        first_head0=0,
        second_row0=second_row0,
        second_head0=second_head0,
        num_heads=half,
        head_dim=head_dim,
    )


def _fake_o_proj_pair(first, second, positions, cos_sin_cache, wo_a, *args, **kw):
    """Abstract implementation for PyTorch Dynamo graph tracing."""
    return torch.empty(
        (positions.shape[0], wo_a.shape[-1]), dtype=first.dtype, device=first.device
    )


def _swap_with_peer(group, mine: torch.Tensor, theirs: torch.Tensor):
    """Send `theirs` to the peer core; return what the peer sent us.

    A 2-rank all-gather that keeps the peer's half.
    """
    rank = group.rank_in_group
    gathered = group.all_gather(theirs.contiguous(), dim=0)
    return gathered.view(2, *theirs.shape)[1 - rank]


class VllmDeepseekV4MLAAttentionIntraChip(VllmDeepseekV4MLAAttention):
    """DSv4 attention with `wq_b`/`wo_a`/`wo_b` split across a chip's two cores."""

    def __init__(self, vllm_config, prefix: str = "", *args, **kwargs) -> None:
        super().__init__(vllm_config, prefix, *args, **kwargs)
        group = get_chip_pair_group()
        rank, size = group.rank_in_group, group.world_size
        quant_config = vllm_config.quant_config
        # Rebuilt half-size; the replicated ones from the base are dropped.
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wq_b",
            tp_rank=rank,
            tp_size=size,
        )
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_a",
            tp_rank=rank,
            tp_size=size,
        )
        self.wo_a.is_bmm = True
        self.wo_a.bmm_batch_size = self.n_groups // size
        # RowParallelLinear takes no tp_rank/tp_size, so it is built at my
        # half of the input rows with TP off and pointed at its slice below.
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank // size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            return_bias=False,
            prefix=f"{prefix}.wo_b",
            disable_tp=True,
        )
        for layer in (self.wq_b, self.wo_a, self.wo_b):
            _use_my_slice(layer, rank, size)
        object.__setattr__(self, "_chip_pair", group)
        logger.info_once(
            "ATTN_INTRA_CHIP_TP: wq_b/wo_a/wo_b sharded across the chip pair"
        )

    @property
    def o_proj_pair_op(self):
        """The fused inverse-RoPE + wo_a op over [core 0 tokens; core 1
        tokens] with my heads, read from `o` and the gathered peer rows.

        Core 0's tokens are its own `o` rows (heads from 0) or the peer's
        gathered rows [0, T); core 1's tokens are the gathered rows [T, 2T) or
        its own `o` rows (heads from H/2). Built by the runner's op pre-build.
        """
        op = self.__dict__.get("_o_proj_pair_op_instance")
        if op is not None:
            return op
        rank = self._chip_pair.rank_in_group
        head_dim = int(self.head_dim)
        half = self.n_local_heads // 2
        op_name = f"pallas::deepseek_v4_o_proj_pair_d{head_dim}_r{rank}"
        op = _pallas_op_cache.get(op_name)
        if op is None:
            op = pallas.jax_op(
                op_name,
                functools.partial(
                    _o_proj_pair_jax, rank=rank, half=half, head_dim=head_dim
                ),
                mesh=get_pallas_op_mesh(),
                input_partition_specs=tuple(P() for _ in range(6)),
            )
            op.register_fake(_fake_o_proj_pair)
            _pallas_op_cache[op_name] = op
        object.__setattr__(self, "_o_proj_pair_op_instance", op)
        return op

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        # The base reshapes wo_a by the local head and group counts, which are
        # halved for these weights but not for attention, so swap them in here.
        size = self._chip_pair.world_size
        heads, groups = self.n_local_heads, self.n_local_groups
        self.n_local_heads, self.n_local_groups = heads // size, groups // size
        try:
            super().process_weights_after_loading(act_dtype)
        finally:
            self.n_local_heads, self.n_local_groups = heads, groups

    def attention_impl(self, hidden_states, qr, kv, indexer_weights, positions, out):
        group = self._chip_pair
        rank = group.rank_in_group
        num_tokens = qr.shape[0]
        # q for my heads over [core0 tokens; core1 tokens].
        q_half = self.wq_b(group.all_gather(qr.contiguous(), dim=0))
        q_half = q_half.view(2, num_tokens, -1)
        # My tokens with my heads stay; my tokens with the peer's heads arrive.
        peer = _swap_with_peer(group, q_half[rank], q_half[1 - rank])
        pieces = [q_half[rank], peer] if rank == 0 else [peer, q_half[rank]]
        q = torch.cat(pieces, dim=-1).view(
            num_tokens, self.mla_attn.n_local_heads, self.mla_attn.head_dim
        )
        q = self.qnorm_rope(q, positions)
        kv = self.kv_rope(kv, positions)

        topk_indices = None
        compressor = getattr(self.mla_attn, "compressor", None)
        if self.indexer is not None:
            indexer_emb = getattr(
                self.mla_attn,
                "indexer_rotary_emb",
                getattr(self.mla_attn, "rotary_emb", None),
            )
            topk_indices = self.indexer(
                hidden_states, qr, indexer_weights, positions, indexer_emb
            )
        if compressor is not None:
            compressor(hidden_states, positions, self.mla_attn.rotary_emb)

        return self.forward_mqa(q, kv, positions, None, topk_indices=topk_indices)

    def _o_proj(self, o, positions):
        group = self._chip_pair
        rank = group.rank_in_group
        half = self.n_local_heads // 2
        # The peer's tokens with my heads arrive as its rows of the gathered
        # buffer; my tokens with my heads stay in `o`. wo_a reads both in
        # place, core 0's tokens first (see `o_proj_pair_op`).
        theirs = o[:, (1 - rank) * half : (2 - rank) * half]
        gathered = group.all_gather(theirs.contiguous(), dim=0)
        first, second = (o, gathered) if rank == 0 else (gathered, o)
        positions_all = group.all_gather(
            positions.to(torch.int32).contiguous(), dim=0
        ).to(positions.dtype)
        z = self.o_proj_pair_op(
            first,
            second,
            positions_all,
            self.rotary_emb.cos_sin_cache,
            self.wo_a.weight,
            self.wo_a.weight_scale,
        )
        partial = self.wo_b(z)
        if isinstance(partial, tuple):
            partial = partial[0]
        return group.reduce_scatter(partial.contiguous(), dim=0)
