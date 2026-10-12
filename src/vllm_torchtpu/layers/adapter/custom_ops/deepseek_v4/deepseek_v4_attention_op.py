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

import functools

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas
from vllm.config import VllmConfig
from vllm.model_executor.layers.attention_layer_base import (
    AttentionBackend,  # isort: skip
)
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWABackend,
    DeepseekV4SWACache,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, SlidingWindowMLASpec

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.deepseek_v4.core_attention.mla import (
    mla_ragged_paged_attention,
)
from vllm_torchtpu.kernels.deepseek_v4.core_attention.mla_swa import (
    mla_sliding_window_ragged_paged_attention,
)
from vllm_torchtpu.kernels.deepseek_v4.core_attention.sparse_mla import (
    prepare_outputs,
    prepare_sparse_attention,
    run_sparse_mla_chunk,
    sparse_attention_chunk_end,
    sparse_attention_chunks,
    sparse_ragged_paged_attention,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import align_to

logger = init_logger(__name__)

# tokamax's PyTorch CSA gather, imported here at module import -- never inside
# a compiled region, where Dynamo would trace the import -- or None when
# tokamax is absent or broken, in which case the in-tree CSA op runs.
try:
    from tokamax.experimental.torch_tpu.ops.csa_gather import (
        torch_pallas_mosaic_tpu as tokamax_csa_gather,
    )
except (ImportError, AttributeError) as _e:
    tokamax_csa_gather = None
    logger.info("tokamax's CSA gather is unavailable: %s", _e)

_tokamax_csa_op_cache: dict[str, object] = {}

BATCH_AXIS = None


def _run_swa(
    q: jax.Array,
    new_kv: jax.Array,
    sw_cache: jax.Array,
    swa_kv_lens: jax.Array,
    swa_page_indices: jax.Array,
    swa_cu_q_lens: jax.Array,
    swa_distribution: jax.Array,
    attention_sinks: jax.Array,
    *,
    sm_scale: float,
    sliding_window: int,
    logical_page_size: int,
    swa_only: bool,
    non_causal_block: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Sliding-window pass: updates the SWA cache and returns its partials.

    When a compressed-KV pass follows, the output is left unnormalized so
    that pass can fold `swa_l` / `swa_m` into its own softmax accumulation.
    """
    if sw_cache.shape[0] == 0:
        # Profiling-shape trace: no cache to attend over.
        return (
            jnp.zeros_like(q),
            sw_cache,
            jnp.zeros((q.shape[0], q.shape[1]), dtype=jnp.float32),
            jnp.full((q.shape[0], q.shape[1]), -1e9, dtype=jnp.float32),
        )
    return mla_sliding_window_ragged_paged_attention(
        q=q,
        new_kv=new_kv,
        cache_kv=sw_cache,
        kv_lens=swa_kv_lens,
        page_indices=swa_page_indices,
        cu_q_lens=swa_cu_q_lens,
        distribution=swa_distribution,
        attention_sinks=attention_sinks,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        logical_page_size=logical_page_size,
        # The following parameters are tuned based on microbenchmark results.
        num_kv_pages_per_block=(2, 2, 2),
        num_queries_per_block=(1, 32, 32),
        q_compute_block_size=4,
        unnormalized_output=not swa_only,
        non_causal_block=non_causal_block,
    )


# Module-level so `pallas.jax_op` can trace and register it as a torch op.
def _attention_hca(
    q: jax.Array,
    new_kv: jax.Array,
    sw_cache: jax.Array,
    swa_kv_lens: jax.Array,
    swa_page_indices: jax.Array,
    swa_cu_q_lens: jax.Array,
    swa_distribution: jax.Array,
    main_cache_kv: jax.Array,
    main_kv_lens: jax.Array,
    extra: jax.Array,  # pass kv_lens_to_attend for HCA
    main_page_indices: jax.Array,
    main_cu_q_lens: jax.Array,
    main_distribution: jax.Array,
    attention_sinks: jax.Array,
    *,
    sm_scale: float,
    sliding_window: int,
    logical_page_size: int,
    swa_only: bool,
    two_caches_same_buffer: bool,
    non_causal_block: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """SWA-only layers, and HCA layers over a single raw-bf16 latent array."""
    if main_cache_kv.shape[0] == 0 or (swa_only and sw_cache.shape[0] == 0):
        # Profiling-shape trace: the kernel is skipped. Unused inputs stay in
        # the compiled signature because torch_tpu jits with keep_unused=True.
        return jnp.zeros(q.shape, dtype=q.dtype), sw_cache

    swa_page_indices = swa_page_indices.flatten()
    main_page_indices = main_page_indices.flatten()

    swa_output, updated_sw_cache, swa_l, swa_m = _run_swa(
        q=q,
        new_kv=new_kv,
        sw_cache=sw_cache,
        swa_kv_lens=swa_kv_lens,
        swa_page_indices=swa_page_indices,
        swa_cu_q_lens=swa_cu_q_lens,
        swa_distribution=swa_distribution,
        attention_sinks=attention_sinks,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        logical_page_size=logical_page_size,
        swa_only=swa_only,
        non_causal_block=non_causal_block,
    )

    if swa_only:
        return swa_output, updated_sw_cache

    # When both caches are one buffer, the compressed-KV read must see the
    # SWA kernel's write from this pass, not the pre-write snapshot.
    if two_caches_same_buffer:
        main_cache_kv = updated_sw_cache

    output = mla_ragged_paged_attention(
        q=q,
        cache_kv=main_cache_kv,
        kv_lens=main_kv_lens,
        kv_lens_to_attend=extra,
        page_indices=main_page_indices,
        cu_q_lens=main_cu_q_lens,
        distribution=main_distribution,
        attention_sinks=attention_sinks,
        swa_accumution=swa_output,
        swa_l=swa_l,
        swa_m=swa_m,
        sm_scale=sm_scale,
        # The following parameters are tuned based on microbenchmark results.
        num_kv_pages_per_block=(128, 32, 32),
        num_queries_per_block=(1, 32, 32),
    )
    return output, updated_sw_cache


def _attention_csa(
    q: jax.Array,
    new_kv: jax.Array,
    sw_cache: jax.Array,
    swa_kv_lens: jax.Array,
    swa_page_indices: jax.Array,
    swa_cu_q_lens: jax.Array,
    swa_distribution: jax.Array,
    main_cache_kv: jax.Array,
    main_kv_lens: jax.Array,
    extra: jax.Array,  # pass topk_indices for CSA
    main_page_indices: jax.Array,
    main_cu_q_lens: jax.Array,
    main_distribution: jax.Array,
    attention_sinks: jax.Array,
    main_cache_rope: jax.Array,
    *,
    sm_scale: float,
    sliding_window: int,
    logical_page_size: int,
    swa_only: bool,
    two_caches_same_buffer: bool,
    non_causal_block: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """CSA: sparse gather over top-k rows, NoPE and RoPE in two arrays."""
    if main_cache_kv.shape[0] == 0 or main_cache_rope.shape[0] == 0:
        return jnp.zeros(q.shape, dtype=q.dtype), sw_cache

    swa_page_indices = swa_page_indices.flatten()
    main_page_indices = main_page_indices.flatten()

    swa_output, updated_sw_cache, swa_l, swa_m = _run_swa(
        q=q,
        new_kv=new_kv,
        sw_cache=sw_cache,
        swa_kv_lens=swa_kv_lens,
        swa_page_indices=swa_page_indices,
        swa_cu_q_lens=swa_cu_q_lens,
        swa_distribution=swa_distribution,
        attention_sinks=attention_sinks,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        logical_page_size=logical_page_size,
        swa_only=False,
        non_causal_block=non_causal_block,
    )

    if two_caches_same_buffer:
        main_cache_kv = updated_sw_cache

    output = sparse_ragged_paged_attention(
        q=q,
        cache_kv_nope=main_cache_kv,
        cache_kv_rope=main_cache_rope,
        topk_indices=extra,
        page_indices=main_page_indices,
        cu_q_lens=main_cu_q_lens,
        distribution=main_distribution,
        attention_sinks=attention_sinks,
        swa_accumution=swa_output,
        swa_l=swa_l,
        swa_m=swa_m,
        sm_scale=sm_scale,
    )
    return output, updated_sw_cache


# Module-level so `pallas.jax_op` can trace and register it as a torch op.
def _csa_prologue(
    q: jax.Array,
    new_kv: jax.Array,
    sw_cache: jax.Array,
    swa_kv_lens: jax.Array,
    swa_page_indices: jax.Array,
    swa_cu_q_lens: jax.Array,
    swa_distribution: jax.Array,
    main_cache_kv: jax.Array,
    topk_indices: jax.Array,
    main_page_indices: jax.Array,
    main_cu_q_lens: jax.Array,
    main_distribution: jax.Array,
    attention_sinks: jax.Array,
    main_cache_rope: jax.Array,
    *,
    sm_scale: float,
    sliding_window: int,
    logical_page_size: int,
    two_caches_same_buffer: bool,
    non_causal_block: bool = False,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """`_attention_csa` up to its gather: the SWA pass and the sparse prep.

    Returns the updated SWA cache; then `prepare_sparse_attention`'s outputs,
    with the valid token count as a `(1,)` array; then, per query chunk of
    `sparse_attention_chunks`, where its kernel grid ends and how many indices
    it gathers. All the scalars are int32, computed here so torch does no
    arithmetic on them between the ops.
    """
    swa_output, updated_sw_cache, swa_l, swa_m = _run_swa(
        q=q,
        new_kv=new_kv,
        sw_cache=sw_cache,
        swa_kv_lens=swa_kv_lens,
        swa_page_indices=swa_page_indices.flatten(),
        swa_cu_q_lens=swa_cu_q_lens,
        swa_distribution=swa_distribution,
        attention_sinks=attention_sinks,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        logical_page_size=logical_page_size,
        swa_only=False,
        non_causal_block=non_causal_block,
    )
    if two_caches_same_buffer:
        main_cache_kv = updated_sw_cache
    *prepared, kv_lens, gather_indices, num_valid_tokens = prepare_sparse_attention(
        q,
        main_cache_kv,
        main_cache_rope,
        topk_indices,
        main_page_indices.flatten(),
        main_cu_q_lens,
        main_distribution,
        attention_sinks,
        swa_output,
        swa_l,
        swa_m,
    )
    topk = topk_indices.shape[-1]
    batch_ends, num_valid_indices = [], []
    for start, end, batch in sparse_attention_chunks(q.shape[0]):
        batch_end, valid_rows = sparse_attention_chunk_end(
            start, end, batch, num_valid_tokens
        )
        batch_ends.append(batch_end)
        num_valid_indices.append(valid_rows * topk)
    return (
        updated_sw_cache,
        *prepared,
        kv_lens.astype(jnp.int32),
        gather_indices.astype(jnp.int32),
        num_valid_tokens.reshape(1).astype(jnp.int32),
        jnp.stack(batch_ends).astype(jnp.int32),
        jnp.stack(num_valid_indices).astype(jnp.int32),
    )


def _sparse_mla_chunk(
    q: jax.Array,
    gathered_nope: jax.Array,
    gathered_rope: jax.Array,
    kv_lens: jax.Array,
    attention_sinks: jax.Array,
    swa_accumution: jax.Array,
    swa_l: jax.Array,
    swa_m: jax.Array,
    end_seq_idx: jax.Array,
    num_valid_tokens: jax.Array,
    *,
    start_seq_idx: int,
    kernel_batch_size: int,
    sm_scale: float,
) -> jax.Array:
    """`run_sparse_mla_chunk`, with its two device scalars as `(1,)` arrays."""
    return run_sparse_mla_chunk(
        q,
        gathered_nope,
        gathered_rope,
        kv_lens,
        attention_sinks,
        swa_accumution,
        swa_l,
        swa_m,
        start_seq_idx=start_seq_idx,
        end_seq_idx=end_seq_idx[0],
        num_valid_tokens=num_valid_tokens[0],
        kernel_batch_size=kernel_batch_size,
        sm_scale=sm_scale,
    )


def tokamax_csa_unsupported_reason(mesh) -> str | None:
    """Why CSA attention cannot gather with tokamax's PyTorch op, or None."""
    if not envs.VLLM_TPU_USE_TOKAMAX:
        return "VLLM_TPU_USE_TOKAMAX=0"
    if tokamax_csa_gather is None:
        return "tokamax is not importable"
    if mesh is not None:
        # tokamax's op registers without a mesh, so it cannot take tensors
        # sharded over a multi-device mesh in this process.
        return "a multi-device mesh in this process"
    return None


def build_tokamax_csa_attention(
    op_name: str,
    in_tree_op,
    *,
    sm_scale: float,
    sliding_window: int,
    logical_page_size: int,
    two_caches_same_buffer: bool,
    non_causal_block: bool = False,
):
    """CSA attention that gathers with tokamax's `PallasMosaicTpuCsaGather`.

    `_attention_csa` runs as one `jax_op`; this splits it around the gather so
    tokamax's PyTorch op can do it. A prologue `jax_op` runs the SWA pass and
    the sparse-attention prep, then per query chunk tokamax gathers the top-k
    cache rows and a `jax_op` runs the sparse MLA kernel over them, writing
    the chunk's rows of the output in place. Returns a callable with the
    in-tree op's signature and outputs, `(output, updated_sw_cache)`.
    `in_tree_op` still serves the profiling pass, whose caches are empty.
    """
    ops = _tokamax_csa_op_cache.get(op_name)
    if ops is None:
        prologue_op = pallas.jax_op(
            f"{op_name}_prologue",
            functools.partial(
                _csa_prologue,
                sm_scale=sm_scale,
                sliding_window=sliding_window,
                logical_page_size=logical_page_size,
                two_caches_same_buffer=two_caches_same_buffer,
                non_causal_block=non_causal_block,
            ),
            # The SWA kernel updates the cache in place.
            donate_argnums=(2,),
        )

        def _fake_prologue(q, new_kv, sw_cache, *args):
            topk_indices = args[5]
            tokens, heads, dim = q.shape
            heads = align_to(heads, 4 // q.element_size())
            num_l_heads = align_to(heads, 128)
            dim = align_to(dim, 128)
            num_chunks = (tokens + 127) // 128  # see sparse_attention_chunks
            return (
                torch.empty_like(sw_cache),
                q.new_empty((tokens, heads, dim)),
                q.new_empty((heads,), dtype=torch.float32),
                q.new_empty((tokens, heads, dim)),
                q.new_empty((tokens, num_l_heads), dtype=torch.float32),
                q.new_empty((tokens, num_l_heads), dtype=torch.float32),
                q.new_empty((tokens,), dtype=torch.int32),
                topk_indices.new_empty((topk_indices.numel(),), dtype=torch.int32),
                q.new_empty((1,), dtype=torch.int32),
                q.new_empty((num_chunks,), dtype=torch.int32),
                q.new_empty((num_chunks,), dtype=torch.int32),
            )

        prologue_op.register_fake(_fake_prologue)

        chunk_op = pallas.jax_op(
            f"{op_name}_sparse_mla_chunk",
            functools.partial(_sparse_mla_chunk, sm_scale=sm_scale),
            # The kernel writes the chunk's rows into `q` and returns it.
            donate_argnums=(0,),
        )
        chunk_op.register_fake(lambda q, *args, **kwargs: torch.empty_like(q))
        ops = _tokamax_csa_op_cache[op_name] = (prologue_op, chunk_op)
    prologue_op, chunk_op = ops

    gather = tokamax_csa_gather.PallasMosaicTpuCsaGather
    # Built here, outside any compiled region: it is a pydantic dataclass.
    gather_configs = (tokamax_csa_gather.Config(), None)

    def tokamax_csa_attention(
        q,
        new_kv,
        sw_cache,
        swa_kv_lens,
        swa_page_indices,
        swa_cu_q_lens,
        swa_distribution,
        main_cache_kv,
        main_kv_lens,
        topk_indices,
        main_page_indices,
        main_cu_q_lens,
        main_distribution,
        attention_sinks,
        main_cache_rope,
    ):
        if main_cache_kv.shape[0] == 0 or main_cache_rope.shape[0] == 0:
            return in_tree_op(
                q,
                new_kv,
                sw_cache,
                swa_kv_lens,
                swa_page_indices,
                swa_cu_q_lens,
                swa_distribution,
                main_cache_kv,
                main_kv_lens,
                topk_indices,
                main_page_indices,
                main_cu_q_lens,
                main_distribution,
                attention_sinks,
                main_cache_rope,
            )
        (
            updated_sw_cache,
            out,
            sinks,
            swa_acc,
            swa_l,
            swa_m,
            kv_lens,
            gather_indices,
            num_valid_tokens,
            batch_ends,
            num_valid_indices,
        ) = prologue_op(
            q,
            new_kv,
            sw_cache,
            swa_kv_lens,
            swa_page_indices,
            swa_cu_q_lens,
            swa_distribution,
            main_cache_kv,
            topk_indices,
            main_page_indices,
            main_cu_q_lens,
            main_distribution,
            attention_sinks,
            main_cache_rope,
        )
        nope_cache = updated_sw_cache if two_caches_same_buffer else main_cache_kv
        topk = topk_indices.shape[-1]
        for i, (start, end, batch) in enumerate(sparse_attention_chunks(out.shape[0])):
            nope, rope = gather(
                nope_cache,
                main_cache_rope,
                gather_indices[start * topk : end * topk],
                num_valid_indices[i : i + 1],
                top_k=topk,
                configs=gather_configs,
            )
            out = chunk_op(
                out,
                nope.view(end - start, topk, -1),
                rope.view(end - start, topk // 4, -1),
                kv_lens,
                sinks,
                swa_acc,
                swa_l,
                swa_m,
                batch_ends[i : i + 1],
                num_valid_tokens,
                start_seq_idx=start,
                kernel_batch_size=batch,
            )
        return prepare_outputs(out, q.shape[1], q.shape[2]), updated_sw_cache

    return tokamax_csa_attention


def get_packed_mla_head_size(hf_config) -> int:
    """Calculate TPU 128-byte aligned packed width for DSv4 FP8 cache entries."""
    kv_bytes = getattr(hf_config, "kv_lora_rank", 448)
    rope_bytes = getattr(hf_config, "qk_rope_head_dim", 64) * 2
    scale_bytes = kv_bytes // 64
    return align_to(kv_bytes + rope_bytes + scale_bytes, 128)


class VllmDeepseekSparseSWABackend(DeepseekSparseSWABackend):
    """TPU attention backend for DeepSeek-V4 SWA cache."""

    @classmethod
    def get_min_page_size(cls, vllm_config: VllmConfig) -> int:
        return 1


class VllmDeepseekV4SWACache(DeepseekV4SWACache):
    """Sliding-window KV cache, paged to fit inside a CSA main page."""

    def __init__(
        self,
        head_dim: int,
        window_size: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config,
        backend_cls: type[AttentionBackend] | None = None,
    ) -> None:
        super().__init__(
            head_dim,
            window_size,
            dtype,
            prefix,
            cache_config,
            backend_cls=backend_cls or VllmDeepseekSparseSWABackend,
        )
        # Initialize with default block size at construction time;
        # get_kv_cache_spec recomputes self.block_size from finalized runtime config.
        self.block_size = self._swa_block_size(cache_config.block_size, window_size)

    @staticmethod
    def _swa_block_size(compressed_kv_cache_bz: int, window_size: int | None) -> int:
        # We would like to overlay the SWA cache with CSA's main NOPE cache
        # on the same KV-Tensor, whose shape is [num_pages, page_size, 4, 128]
        # u8.
        # Thus set swa cache's block size accordingly.
        # In SWA, we store kv cache in bf16 to avoid expensive quantization
        # and dequantization. The extra storage overhead is small since kvs
        # outside the sliding window can be reclaimed as needed.
        # 2 because bf16 occupies 2 bytes per element.
        csa_compression_ratio = 4
        block_size = compressed_kv_cache_bz // csa_compression_ratio // 2
        if window_size is None:
            return block_size
        return min(block_size, window_size)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # Update self.block_size so the returned spec and the Pallas kernel's
        # logical_page_size in attention.py agree on the final runtime block size.
        self.block_size = self._swa_block_size(
            vllm_config.cache_config.block_size, self.window_size
        )
        # `mla_swa` keeps the SWA entries as raw bf16, packed as uint8.
        return SlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=1024,
            dtype=torch.uint8,
            sliding_window=self.window_size,
            cache_dtype_str=self.cache_config.cache_dtype,
            alignment=None,
        )
