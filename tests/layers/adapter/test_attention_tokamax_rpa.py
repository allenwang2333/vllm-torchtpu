# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone tests for the tokamax-only batched-RPA path in `attention.py`.

Batched RPA has exactly one implementation now: tokamax's
`PallasMosaicTpuBatchedRpa` singleton, called at the torch level by
`PallasAttentionBackendImpl._build_rpa_kernel`. The in-tree batched kernel, the
`USE_BATCHED_RPA_LONGCTX` fork and the PCP streaming kernels are gone. What is
left to protect is (a) the adapter seam between this repo's calling convention
and tokamax's, and (b) the guards that make every configuration tokamax cannot
serve fail loudly instead of silently attending over the wrong thing.

Standalone by construction: every fixture this module needs is defined here, so
it runs on its own (`pytest tests/layers/adapter/test_attention_tokamax_rpa.py`)
without `tests/conftest.py`. All but the last test run on CPU against a stubbed
tokamax operator -- they check wiring, not numerics. `test_real_tokamax_op_*`
is the one that needs a TPU and skips without one.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from vllm import envs as vllm_envs

from vllm_torchtpu.kernels.experimental.batched_rpa.configs import KVLayout
from vllm_torchtpu.layers.adapter import attention as attn
from vllm_torchtpu.layers.adapter.attention import (
    PallasAttentionBackend, PallasAttentionBackendImpl,
    PallasBatchedRPAAttentionBackend, PallasBatchedRPAAttentionBackendImpl,
    _pallas_rpa_kernel_default, _pallas_rpa_kernel_local)
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import \
    set_vllm_model_wrapper_context

_TOKAMAX_OP_MODULE = ("tokamax.experimental.torch_tpu.ops.batched_rpa."
                      "torch_pallas_mosaic_tpu")
_TOKAMAX_OP_ATTR = "PallasMosaicTpuBatchedRpa"

# =========================================================================
# Representative workload
# =========================================================================
#
# One continuous-batching step of a GQA decoder, which is what this kernel
# actually sees in production: a handful of 1-token decodes at the front of the
# persistent batch, then a chunked prefill resuming mid-sequence, then a fresh
# prefill. Shapes follow Llama-3-8B's attention config (32 query heads over 8 KV
# heads, head_dim 128) at the only page size the CUSTOM backend still offers.
#
# The pool is deliberately small (24 pages, ~25 MiB in bf16) so the suite stays
# fast; the *structure* -- 5D packed layout, page count covering every active
# request, padded batch rows -- is the part that has to be faithful.


@dataclass(frozen=True)
class Workload:
    """A mixed decode + chunked-prefill step, sized like a real one."""

    num_heads: int = 32
    num_kv_heads: int = 8
    head_size: int = 128
    # `PallasBatchedRPAAttentionBackend.get_supported_kernel_block_sizes()`.
    page_size: int = 256
    max_model_len: int = 1024
    # Persistent batch is padded to a fixed width; only `num_reqs` rows are live.
    max_num_seqs: int = 8
    num_reqs: int = 6
    num_decodes: int = 4
    num_blocks: int = 24
    # Scheduled tokens padded up to a compile bucket, as the runner does.
    num_tokens: int = 772
    padded_num_tokens: int = 1024
    dtype: torch.dtype = torch.bfloat16

    @property
    def pages_per_req(self) -> int:
        return self.max_model_len // self.page_size

    @property
    def kv_packing(self) -> int:
        return attn.get_dtype_packing(self.dtype)

    @property
    def kv_cache_shape(self) -> tuple[int, ...]:
        return PallasAttentionBackend.get_kv_cache_shape(
            self.num_blocks,
            self.page_size,
            self.num_kv_heads,
            self.head_size,
            self.dtype,
        )

    def kv_cache(self) -> torch.Tensor:
        return torch.zeros(self.kv_cache_shape, dtype=self.dtype)

    def qkv(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """q/k/v for the whole padded step, in vLLM's 3D [tokens, heads, dim]."""
        gen = torch.Generator().manual_seed(0)

        def _rand(heads: int) -> torch.Tensor:
            return torch.randn(self.padded_num_tokens,
                               heads,
                               self.head_size,
                               generator=gen).to(self.dtype)

        return (_rand(self.num_heads), _rand(self.num_kv_heads),
                _rand(self.num_kv_heads))

    def metadata(self) -> AttentionMetadata:
        # 4 decodes (1 token each), then a 512-token prefill chunk on a request
        # with 512 tokens already cached, then a fresh 256-token prefill.
        # 4 + 512 + 256 = 772 scheduled tokens.
        starts = [0, 1, 2, 3, 4, 516, 772]
        starts += [self.num_tokens] * (self.max_num_seqs + 1 - len(starts))
        # Total context each request will have attended after this step.
        seq_lens = [1024, 768, 512, 257, 1024, 256]
        seq_lens += [0] * (self.max_num_seqs - len(seq_lens))

        # Distinct pages per live request; padded rows park on page 0.
        # Flat, not [max_num_seqs, pages_per_req]: `AttentionMetadata` documents
        # `(max_num_seqs * max_num_blocks_per_req,)` and tokamax's
        # `RpaConfigs` asserts `page_indices.ndim == 1`.
        block_tables = torch.zeros(self.max_num_seqs * self.pages_per_req,
                                   dtype=torch.int32)
        for req in range(self.num_reqs):
            base = req * self.pages_per_req
            block_tables[base:base + self.pages_per_req] = torch.arange(
                base, base + self.pages_per_req, dtype=torch.int32)

        return AttentionMetadata(
            input_positions=torch.arange(self.padded_num_tokens,
                                         dtype=torch.int32),
            block_tables=block_tables,
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
            query_start_loc=torch.tensor(starts, dtype=torch.int32),
            # V3 convention: [decode_end, prefill_end, mixed_end].
            request_distribution=torch.tensor(
                [self.num_decodes, self.num_decodes, self.num_reqs],
                dtype=torch.int32),
        )


WORKLOAD = Workload()


def _impl(cls=PallasBatchedRPAAttentionBackendImpl,
          workload: Workload = WORKLOAD,
          **kwargs) -> PallasAttentionBackendImpl:
    head_size = kwargs.pop("head_size", workload.head_size)
    kwargs.setdefault("kv_cache_dtype", "bfloat16")
    kwargs.setdefault("sliding_window", None)
    return cls(
        num_heads=kwargs.pop("num_heads", workload.num_heads),
        head_size=head_size,
        scale=kwargs.pop("scale", head_size**-0.5),
        num_kv_heads=kwargs.pop("num_kv_heads", workload.num_kv_heads),
        alibi_slopes=None,
        **kwargs,
    )


def _layer(k_scale: float = 0.0, v_scale: float = 0.0) -> Any:
    layer = MagicMock()
    layer._k_scale_float = k_scale
    layer._v_scale_float = v_scale
    return layer


def _mesh(model: int = 1) -> Any:
    return SimpleNamespace(shape={"attn_dp": 1, "expert": 1, "model": model})


def _vllm_config(pcp: int = 1, dcp: int = 1) -> Any:
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=WORKLOAD.max_model_len),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp,
                                        decode_context_parallel_size=dcp),
    )


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture(autouse=True)
def kv_cache_layout(monkeypatch):
    """Pin the layout to NHD (HEAD_ALONG_SUBLANE), the production default.

    `PallasAttentionBackendImpl.__init__` resolves the layout through
    `_resolve_kv_layout`, which prefers the one the live vLLM config negotiated
    against the backend. There is no live config here -- building a
    `VllmConfig` under the torchtpu platform wants a registered `tpu` torch
    device -- so these tests drive the other source it reads,
    `VLLM_KV_CACHE_LAYOUT`, which is the override an operator would set anyway.

    Autouse, and through `monkeypatch` so the HND test below cannot leak its
    setting into whatever runs after it.
    """
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "NHD")
    assert vllm_envs.VLLM_KV_CACHE_LAYOUT == "NHD"


@pytest.fixture(autouse=True)
def clean_kernel_registry():
    """Drop registered ops between tests.

    `_kernel_registry` is class-level: it is how two layers with the same
    config share one torch custom op. That sharing is the production behaviour
    and is asserted below, but across tests it would hand a later test a
    closure holding an earlier test's stubbed operator.
    """
    PallasAttentionBackendImpl._kernel_registry.clear()
    yield
    PallasAttentionBackendImpl._kernel_registry.clear()


@pytest.fixture
def no_synchronize(monkeypatch):
    """`forward` syncs the pool through torch_tpu when not compiling."""
    monkeypatch.setattr(attn, "synchronize_tensors",
                        lambda *args, **kwargs: None)


@dataclass
class TokamaxCall:
    """One recorded invocation of the stubbed operator."""

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    kv_cache: torch.Tensor
    seq_lens: torch.Tensor
    block_tables: torch.Tensor
    query_start_loc: torch.Tensor
    request_distribution: torch.Tensor
    config: dict


# A value no real attention output would produce, so "the stub ran" is provable.
SENTINEL_OUT = 7.0
# Written by the stub at one 4D cache coordinate to prove the fold inverts.
SENTINEL_CACHE = 42.0
SENTINEL_CACHE_IDX_4D = (3, 5, 2, 7)
# Same element addressed through the pool's own 5D shape, with kv_packing=2:
# merged head index 2 == (packed_head 1, packing lane 0).
SENTINEL_CACHE_IDX_5D = (3, 5, 1, 0, 7)


def _fold(kv_cache: torch.Tensor) -> torch.Tensor:
    """The 5D -> 4D pool fold tokamax does on its own side of the boundary.

    `[pages, page_size, kv_heads_x2 // packing, packing, head_dim]` ->
    `[pages, page_size, kv_heads_x2, head_dim]`. Reproduced here only so the
    stub operator can behave like the real one; `attention.py` deliberately
    does not fold, see `test_pool_crosses_the_boundary_unfolded`.
    """
    pages, page_size, packed_heads, packing, head_dim = kv_cache.shape
    return kv_cache.view(pages, page_size, packed_heads * packing, head_dim)


@pytest.fixture
def fake_tokamax_op(monkeypatch):
    """Swap tokamax's singleton for a recorder with the same call convention.

    Patched as an attribute of the real tokamax module rather than by faking
    the module, so the lazy import inside `_tokamax_rpa_kernel` still has to
    resolve: a wrong import path is a test failure, not a silently stubbed
    success. The real singleton lowers through Pallas onto a TPU, which is why
    only the convention -- positional tensors, the config by keyword, and
    `(outputs, new_kv_cache)` back -- is exercised here.

    Like the real operator, the stub takes the pool in whatever shape it is
    handed, folds it onto the 4D cache the kernel models, and gives back the
    caller's shape.
    """
    module = pytest.importorskip(_TOKAMAX_OP_MODULE)
    calls: list[TokamaxCall] = []

    def _op(query, key, value, kv_cache, seq_lens, block_tables,
            query_start_loc, request_distribution, **config):
        calls.append(
            TokamaxCall(query=query,
                        key=key,
                        value=value,
                        kv_cache=kv_cache,
                        seq_lens=seq_lens,
                        block_tables=block_tables,
                        query_start_loc=query_start_loc,
                        request_distribution=request_distribution,
                        config=config))
        outputs = torch.full_like(query, SENTINEL_OUT)
        new_kv_cache = kv_cache.clone()
        # Written through the fold, so the writeback in the closure is only
        # correct if the two index into the same element.
        _fold(new_kv_cache)[SENTINEL_CACHE_IDX_4D] = SENTINEL_CACHE
        return outputs, new_kv_cache

    monkeypatch.setattr(module, _TOKAMAX_OP_ATTR, _op)
    return calls


@pytest.fixture
def tpu():
    """A TPU, or a skip.

    Resolved in a fixture, never at module scope: building a JAX backend during
    collection claims the accelerator for the whole pytest process and breaks
    every test that spawns a worker.
    """
    import jax
    if jax.device_count() == 0 or jax.devices()[0].platform != "tpu":
        pytest.skip("needs a TPU")
    return jax.devices()


# =========================================================================
# Backend surface
# =========================================================================


def test_batched_backend_offers_only_the_page_size_tokamax_tiles():
    """HEAD_ALONG_SUBLANE puts one page per 128-lane tile. The other multiples
    of 128 belonged to the HND and PCP-streaming kernels, which are gone."""
    assert PallasBatchedRPAAttentionBackend.get_supported_kernel_block_sizes(
    ) == [256]


def test_batched_backend_allocates_the_same_pool_as_the_default_one():
    """The SEQ_ALONG_LANE override is deliberately not there any more: tokamax
    models only HEAD_ALONG_SUBLANE, for which the inherited shape is correct."""
    args = (WORKLOAD.num_blocks, WORKLOAD.page_size, WORKLOAD.num_kv_heads,
            WORKLOAD.head_size, WORKLOAD.dtype)
    assert (PallasBatchedRPAAttentionBackend.get_kv_cache_shape(
        *args) == PallasAttentionBackend.get_kv_cache_shape(*args))


def test_workload_pool_shape_is_the_packed_5d_layout():
    """Pins the shape the rest of this file reasons about: bf16 packs two KV
    heads per 32-bit lane, and K/V doubles the head axis."""
    assert WORKLOAD.kv_packing == 2
    assert WORKLOAD.kv_cache_shape == (
        WORKLOAD.num_blocks,
        WORKLOAD.page_size,
        WORKLOAD.num_kv_heads * 2 // 2,
        2,
        WORKLOAD.head_size,
    )


def test_batched_backend_resolves_to_the_tokamax_impl():
    assert (PallasBatchedRPAAttentionBackend.get_impl_cls()
            is PallasBatchedRPAAttentionBackendImpl)
    assert PallasBatchedRPAAttentionBackend.get_name() == "CUSTOM"


# =========================================================================
# The 5D -> 4D fold, and why it does not happen here
# =========================================================================


def test_the_packed_pool_folds_freely_onto_tokamaxs_cache():
    """The premise that lets tokamax take the pool as it is allocated.

    `[pages, page_size, kv_heads_x2 // packing, packing, head_dim]` and
    tokamax's `[pages, page_size, kv_heads_x2, head_dim]` are the same buffer:
    the packing lane is the minor of the merged head axis, so the reshape is
    free and exactly invertible in both directions. Tokamax does it inside its
    own JAX trace; this pins the layout fact it depends on.
    """
    kv_cache = WORKLOAD.kv_cache()
    kv_cache.normal_()
    folded = _fold(kv_cache)

    pages, page_size, packed_heads, packing, head_dim = kv_cache.shape
    assert folded.shape == (pages, page_size, packed_heads * packing, head_dim)
    # A view, not a copy: same storage, no allocation.
    assert folded.data_ptr() == kv_cache.data_ptr()
    assert folded.untyped_storage().data_ptr() == kv_cache.untyped_storage(
    ).data_ptr()
    assert torch.equal(folded.view(kv_cache.shape), kv_cache)
    # Merged head index == packed_head * kv_packing + lane.
    folded[SENTINEL_CACHE_IDX_4D] = SENTINEL_CACHE
    assert kv_cache[SENTINEL_CACHE_IDX_5D] == SENTINEL_CACHE


def test_pool_crosses_the_boundary_unfolded(fake_tokamax_op, no_synchronize):
    """The operator is handed the runner's pool itself, not a reshape of it.

    Free as the fold is, a reshape on this side of the custom-op boundary is a
    tensor in its own right, and the pool is donated: the operator would then
    reuse *that* buffer and leave the runner's pool live alongside it. On a
    397B-class model that is an extra 40 GiB of HBM temporaries per worker,
    which fails the backbone compile outright. Folding jax-side costs nothing.
    """
    impl = _impl()
    q, k, v = WORKLOAD.qkv()
    kv_cache = WORKLOAD.kv_cache()
    with set_vllm_model_wrapper_context(mesh=_mesh(),
                                        vllm_config=_vllm_config()):
        impl.forward(_layer(), q, k, v, kv_cache, WORKLOAD.metadata())

    assert fake_tokamax_op[0].kv_cache is kv_cache


# =========================================================================
# `_build_rpa_kernel` -- what tokamax can and cannot be asked for
# =========================================================================


def test_build_rpa_kernel_caches_per_config(fake_tokamax_op):
    impl = _impl()
    with set_vllm_model_wrapper_context(mesh=_mesh(),
                                        vllm_config=_vllm_config()):
        first = impl._build_rpa_kernel(None, None, None)
        again = impl._build_rpa_kernel(None, None, None)
        # Scales are part of the cache key.
        other = impl._build_rpa_kernel(None, 0.5, 0.25)
    assert first is again
    assert first is not other


def test_build_rpa_kernel_is_shared_across_layers(fake_tokamax_op):
    """Two layers with the same config share one closure.

    Sharing is really tokamax's -- its singleton keys compilations on the
    shapes and the keyword config by value, so identical layers land on one
    kernel either way. The class-level `_kernel_registry` makes that explicit
    and saves rebuilding the closure per layer, exactly as on the in-tree
    path; the per-instance `_kernel_config_cache` only saves the lookup."""
    with set_vllm_model_wrapper_context(mesh=_mesh(),
                                        vllm_config=_vllm_config()):
        a = _impl()._build_rpa_kernel(None, None, None)
        b = _impl()._build_rpa_kernel(None, None, None)
    assert a is b


def test_build_rpa_kernel_rejects_in_process_tensor_parallelism(
        fake_tokamax_op):
    """tokamax's singleton registers itself once, at import, with no mesh and
    no `input_partition_specs`, so there is no `shard_map` to split the heads
    over and every rank would attend over every rank's heads. Multi-process TP
    -- one chip per worker, `model=1` -- is the configuration this serves."""
    with set_vllm_model_wrapper_context(mesh=_mesh(model=4),
                                        vllm_config=_vllm_config()):
        with pytest.raises(NotImplementedError, match="model-axis devices"):
            _impl()._build_rpa_kernel(None, None, None)


def test_build_rpa_kernel_rejects_head_dim_64(fake_tokamax_op):
    """head_dim 64 selects the hd64 kernel and its differently-packed pool."""
    impl = _impl(head_size=64)
    with set_vllm_model_wrapper_context(mesh=_mesh(),
                                        vllm_config=_vllm_config()):
        with pytest.raises(NotImplementedError, match="hd64"):
            impl._build_rpa_kernel(None, None, None)


def test_build_rpa_kernel_rejects_hnd_layout(fake_tokamax_op, monkeypatch):
    """The layout tokamax's `_fwd` defaults to is HEAD_ALONG_SUBLANE, and the
    closure leaves it there rather than forwarding a layout it cannot honour."""
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "HND")
    impl = _impl()
    assert impl.kv_layout is KVLayout.SEQ_ALONG_LANE
    with set_vllm_model_wrapper_context(mesh=_mesh(),
                                        vllm_config=_vllm_config()):
        with pytest.raises(NotImplementedError, match="VLLM_KV_CACHE_LAYOUT"):
            impl._build_rpa_kernel(None, None, None)


def test_build_rpa_kernel_rejects_kv_sharing(fake_tokamax_op):
    """tokamax still lists `skip_kv_update` in `static_argnames` but never
    reads it, so a KV-sharing layer would clobber its target's slots."""
    impl = _impl(kv_sharing_target_layer_name="model.layers.0.self_attn")
    with set_vllm_model_wrapper_context(mesh=_mesh(),
                                        vllm_config=_vllm_config()):
        with pytest.raises(NotImplementedError,
                           match="Cross-layer KV sharing"):
            impl._build_rpa_kernel(None, None, None, skip_kv_update=True)


def test_build_rpa_kernel_rejects_multi_token_decode(fake_tokamax_op):
    """`_fwd` pins the decode blocks to bq_sz=1, so speculative verify windows
    would silently attend one token at a time."""
    impl = _impl()
    impl.decode_query_size = 4
    with set_vllm_model_wrapper_context(mesh=_mesh(),
                                        vllm_config=_vllm_config()):
        with pytest.raises(NotImplementedError, match="decode_query_size=4"):
            impl._build_rpa_kernel(None, None, None)


def test_batched_impl_keeps_decode_query_size_at_one():
    """The longctx fork used to raise this to num_speculative_tokens + 1."""
    assert PallasBatchedRPAAttentionBackendImpl.decode_query_size == 1
    assert _impl().decode_query_size == 1


def test_build_rpa_kernel_rejects_sinks(fake_tokamax_op):
    """Sinks are not in tokamax's signature; the closure refuses rather than
    dropping them."""
    impl = _impl(sinks=torch.zeros(WORKLOAD.num_heads))
    kv_cache = WORKLOAD.kv_cache()
    query, key, value = WORKLOAD.qkv()
    md = WORKLOAD.metadata()
    with set_vllm_model_wrapper_context(mesh=_mesh(),
                                        vllm_config=_vllm_config()):
        kernel = impl._build_rpa_kernel(None, None, None)
        with pytest.raises(NotImplementedError, match="Attention sinks"):
            kernel(kv_cache, query, key, value, md.seq_lens, md.block_tables,
                   md.query_start_loc, md.request_distribution, impl.sinks)


def test_build_rpa_kernel_forwards_attention_config(fake_tokamax_op):
    """Everything tokamax takes by keyword, including the fp8 scales, has to
    arrive by value -- that tuple is its compilation cache key."""
    impl = _impl(sliding_window=512, logits_soft_cap=30.0)
    kv_cache = WORKLOAD.kv_cache()
    query, key, value = WORKLOAD.qkv()
    md = WORKLOAD.metadata()
    with set_vllm_model_wrapper_context(mesh=_mesh(),
                                        vllm_config=_vllm_config()):
        kernel = impl._build_rpa_kernel(None, 0.5, 0.25)
        kernel(kv_cache, query, key, value, md.seq_lens, md.block_tables,
               md.query_start_loc, md.request_distribution, None)

    assert len(fake_tokamax_op) == 1
    assert fake_tokamax_op[0].config == {
        "use_causal_mask": True,
        "sm_scale": WORKLOAD.head_size**-0.5,
        "sliding_window": 512,
        "soft_cap": 30.0,
        "q_scale": None,
        "k_scale": 0.5,
        "v_scale": 0.25,
    }


def test_build_rpa_kernel_rejects_a_resized_cache(monkeypatch):
    """The writeback copies the returned cache into the pool, so a shape
    change has to be caught before it becomes a confusing `copy_` error that
    says nothing about where the cache came from."""
    module = pytest.importorskip(_TOKAMAX_OP_MODULE)
    monkeypatch.setattr(
        module, _TOKAMAX_OP_ATTR, lambda query, *args, **kwargs:
        (torch.zeros_like(query), torch.zeros(1, dtype=query.dtype)))

    impl = _impl()
    kv_cache = WORKLOAD.kv_cache()
    query, key, value = WORKLOAD.qkv()
    md = WORKLOAD.metadata()
    with set_vllm_model_wrapper_context(mesh=_mesh(),
                                        vllm_config=_vllm_config()):
        kernel = impl._build_rpa_kernel(None, None, None)
        with pytest.raises(RuntimeError, match="incompatible KV cache"):
            kernel(kv_cache, query, key, value, md.seq_lens, md.block_tables,
                   md.query_start_loc, md.request_distribution, None)


# =========================================================================
# `initialize_kernel`
# =========================================================================


def test_initialize_kernel_builds_the_tokamax_kernel(fake_tokamax_op):
    impl = _impl()
    with set_vllm_model_wrapper_context(mesh=_mesh(),
                                        vllm_config=_vllm_config()):
        impl.initialize_kernel(_layer())
    assert impl.rpa_kernel is not None
    assert impl.rpa_kernel.__name__ == "tokamax_rpa_kernel_impl"


def test_initialize_kernel_passes_fp8_scales_through(fake_tokamax_op):
    impl = _impl(kv_cache_dtype="fp8_e4m3")
    assert impl.kv_cache_quantized_dtype is torch.float8_e4m3fn
    with set_vllm_model_wrapper_context(mesh=_mesh(),
                                        vllm_config=_vllm_config()):
        impl.initialize_kernel(_layer(k_scale=0.5, v_scale=0.25))
        # Same scales -> the cached closure; so the build did use them as key.
        assert impl.rpa_kernel is impl._build_rpa_kernel(None, 0.5, 0.25)


def test_initialize_kernel_refuses_prefill_context_parallelism(
        fake_tokamax_op):
    """PCP needed the streaming RPA kernel. tokamax has no streaming variant,
    so this raises instead of quietly attending over the wrong layout."""
    impl = _impl()
    with set_vllm_model_wrapper_context(mesh=_mesh(),
                                        vllm_config=_vllm_config(pcp=2)):
        with pytest.raises(NotImplementedError,
                           match="Prefill context parallelism"):
            impl.initialize_kernel(_layer())


def test_initialize_kernel_still_short_circuits_for_dcp(fake_tokamax_op):
    """Decode context parallelism was kept. It returns before the tokamax
    builder, and with no DCP group initialized it builds nothing at all."""
    impl = _impl()
    with set_vllm_model_wrapper_context(mesh=_mesh(),
                                        vllm_config=_vllm_config(dcp=2)):
        impl.initialize_kernel(_layer())
    assert impl.rpa_kernel is None
    assert impl.dcp_world_size == 1
    assert not fake_tokamax_op


def test_runs_batched_rpa_schedule_is_false_everywhere():
    """tokamax spills its (query block, KV block) schedule to HBM, so the
    runner's SMEM bound no longer applies to any remaining path."""
    assert _impl().runs_batched_rpa_schedule() is False
    assert _impl(
        cls=PallasAttentionBackendImpl).runs_batched_rpa_schedule() is False


# =========================================================================
# `forward`
# =========================================================================


def test_forward_drives_the_tokamax_op_over_a_mixed_batch(
        fake_tokamax_op, no_synchronize):
    impl = _impl()
    q, k, v = WORKLOAD.qkv()
    kv_cache = WORKLOAD.kv_cache()
    md = WORKLOAD.metadata()
    with set_vllm_model_wrapper_context(mesh=_mesh(),
                                        vllm_config=_vllm_config()):
        outputs = impl.forward(_layer(), q, k, v, kv_cache, md)

    assert len(fake_tokamax_op) == 1
    call = fake_tokamax_op[0]
    # Argument order is tokamax's, not `_kernel_entry`'s: q/k/v first, cache
    # fourth, no `sinks`. q/k/v are distinct random tensors, so this also
    # catches a k/v transposition, which the shapes alone would not.
    assert torch.equal(call.query, q)
    assert torch.equal(call.key, k)
    assert torch.equal(call.value, v)
    # The cache arrives in the packed 5D shape the runner allocated; tokamax
    # folds it onto its own 4D cache jax-side.
    assert call.kv_cache.shape == WORKLOAD.kv_cache_shape
    # Ragged metadata passes through untouched.
    assert torch.equal(call.seq_lens, md.seq_lens)
    assert torch.equal(call.block_tables, md.block_tables)
    assert torch.equal(call.query_start_loc, md.query_start_loc)
    assert torch.equal(call.request_distribution,
                       torch.tensor([4, 4, 6], dtype=torch.int32))

    assert outputs.shape == (WORKLOAD.padded_num_tokens, WORKLOAD.num_heads,
                             WORKLOAD.head_size)
    assert torch.all(outputs == SENTINEL_OUT)
    # And the op's cache came back into the pool the runner owns.
    assert kv_cache[SENTINEL_CACHE_IDX_5D] == SENTINEL_CACHE


def test_forward_accepts_the_flattened_2d_query_layout(fake_tokamax_op,
                                                       no_synchronize):
    """Some call sites hand attention [tokens, heads * head_dim]; the result
    has to come back in the same rank."""
    impl = _impl()
    q, k, v = WORKLOAD.qkv()
    flat = q.reshape(WORKLOAD.padded_num_tokens,
                     WORKLOAD.num_heads * WORKLOAD.head_size)
    md = WORKLOAD.metadata()
    kv_cache = WORKLOAD.kv_cache()
    with set_vllm_model_wrapper_context(mesh=_mesh(),
                                        vllm_config=_vllm_config()):
        outputs = impl.forward(_layer(), flat,
                               k.reshape(WORKLOAD.padded_num_tokens, -1),
                               v.reshape(WORKLOAD.padded_num_tokens, -1),
                               kv_cache, md)
    assert outputs.shape == flat.shape
    assert fake_tokamax_op[0].query.shape == q.shape


def test_forward_writes_into_the_caller_supplied_output(
        fake_tokamax_op, no_synchronize):
    impl = _impl()
    q, k, v = WORKLOAD.qkv()
    out = torch.empty_like(q)
    md = WORKLOAD.metadata()
    with set_vllm_model_wrapper_context(mesh=_mesh(),
                                        vllm_config=_vllm_config()):
        returned = impl.forward(_layer(),
                                q,
                                k,
                                v,
                                WORKLOAD.kv_cache(),
                                md,
                                output=out)
    assert torch.all(out == SENTINEL_OUT)
    assert torch.equal(returned, out)


def test_forward_pads_a_narrow_head_to_the_pool_width(fake_tokamax_op,
                                                      no_synchronize):
    """A draft layer sharing a wider pool: q/k/v get zero-padded out to the
    page width and the result sliced back, so the caller never sees it."""
    narrow = 96
    impl = _impl(head_size=narrow)
    gen = torch.Generator().manual_seed(1)
    q = torch.randn(WORKLOAD.padded_num_tokens,
                    WORKLOAD.num_heads,
                    narrow,
                    generator=gen).to(WORKLOAD.dtype)
    k = torch.randn(WORKLOAD.padded_num_tokens,
                    WORKLOAD.num_kv_heads,
                    narrow,
                    generator=gen).to(WORKLOAD.dtype)
    md = WORKLOAD.metadata()
    with set_vllm_model_wrapper_context(mesh=_mesh(),
                                        vllm_config=_vllm_config()):
        outputs = impl.forward(_layer(), q, k, k.clone(), WORKLOAD.kv_cache(),
                               md)

    # The kernel saw the full 128-wide page...
    assert fake_tokamax_op[0].query.shape[-1] == WORKLOAD.head_size
    assert torch.all(fake_tokamax_op[0].query[..., narrow:] == 0)
    # ...and the layer got its own width back.
    assert outputs.shape[-1] == narrow


def test_forward_is_a_noop_during_memory_profiling(fake_tokamax_op):
    """`determine_available_memory` calls forward with an empty pool."""
    impl = _impl()
    q, k, v = WORKLOAD.qkv()
    with set_vllm_model_wrapper_context(mesh=_mesh(),
                                        vllm_config=_vllm_config()):
        outputs = impl.forward(_layer(), q, k, v,
                               torch.zeros(0, dtype=WORKLOAD.dtype),
                               WORKLOAD.metadata())
    assert outputs.shape == q.shape
    assert not fake_tokamax_op


def test_forward_rejects_fused_output_quantization(fake_tokamax_op):
    impl = _impl()
    q, k, v = WORKLOAD.qkv()
    with set_vllm_model_wrapper_context(mesh=_mesh(),
                                        vllm_config=_vllm_config()):
        with pytest.raises(NotImplementedError, match="output quantization"):
            impl.forward(_layer(),
                         q,
                         k,
                         v,
                         WORKLOAD.kv_cache(),
                         WORKLOAD.metadata(),
                         output_scale=torch.ones(1))


# =========================================================================
# Fail-closed: block-major KV has no tokamax variant
# =========================================================================


def test_bundled_kernel_refuses_the_batched_impl():
    """With no `_kernel_entry` override left on the batched impl, the identity
    check below it would let block-major KV silently run the *default* bundled
    kernel. The isinstance guard is what stops that."""
    with pytest.raises(NotImplementedError, match="no bundled variant"):
        _impl()._build_bundled_kernel(None, None, None)


def test_bundled_kernel_still_refuses_a_rebound_layer_major_kernel():
    """The runner rebinds `_kernel_entry` per instance for the eagle3 draft."""
    impl = _impl(cls=PallasAttentionBackendImpl)
    impl._kernel_entry = _pallas_rpa_kernel_local
    with pytest.raises(NotImplementedError, match="overrides the layer-major"):
        impl._build_bundled_kernel(None, None, None)


# =========================================================================
# Regression: the other implementations stay deleted
# =========================================================================


@pytest.mark.parametrize("name", [
    "_pallas_rpa_kernel_batched",
    "_build_tokamax_rpa_kernel",
    "_validate_pcp_streaming_support",
    "_select_kernel_mesh",
])
def test_removed_symbols_stay_removed(name):
    """`_build_rpa_kernel` is the single builder; re-adding a second one should
    be a deliberate act, not a quiet re-import."""
    assert not hasattr(attn, name)
    assert not hasattr(PallasAttentionBackendImpl, name)


def test_batched_impl_adds_nothing_but_a_marker():
    """It exists so the CUSTOM backend and `_build_bundled_kernel` have
    something to dispatch on; tokamax's op is not an in-tree kernel function."""
    # `_abc_impl` is ABCMeta's, stamped onto every AttentionImpl subclass.
    # `_kernel_instance_counter` is the base class's op-name counter. The
    # tokamax path never allocates one -- tokamax owns the op name -- but the
    # in-tree path rebinds it onto whichever class builds a kernel, so leaving
    # it out keeps this test independent of test order.
    inherited_machinery = {"_abc_impl", "_kernel_instance_counter"}
    own = {
        k
        for k in vars(PallasBatchedRPAAttentionBackendImpl)
        if not k.startswith("__") and k not in inherited_machinery
    }
    assert own == {"_kernel_op_prefix"}
    assert (PallasBatchedRPAAttentionBackendImpl._kernel_op_prefix ==
            "pallas::rpa_kernel_batched")


def test_default_backend_still_uses_the_in_tree_kernel():
    """The FLASH_ATTN path was explicitly out of scope of the tokamax-only
    change; only batched RPA collapsed onto tokamax."""
    assert (PallasAttentionBackendImpl._kernel_entry
            is _pallas_rpa_kernel_default)
    assert PallasAttentionBackend.get_name() == "FLASH_ATTN"


def test_attention_module_no_longer_imports_streaming_rpa_entry_points():
    """Only the generic `jax_op` helper survives, for sparse-MLA DCP."""
    module = sys.modules[attn.__name__]
    assert hasattr(module, "pcp_streaming_jax_op")
    for name in ("pcp_streaming_ragged_paged_attention",
                 "is_pcp_streaming_attention_metadata",
                 "ragged_paged_attention_batched"):
        assert not hasattr(module, name)


# =========================================================================
# The real operator (TPU only)
# =========================================================================


def test_real_tokamax_op_runs_the_mixed_batch(tpu):
    """End-to-end smoke test against the operator itself, not a stub.

    The only check that the packed pool, the ragged metadata and the keyword
    config are actually shapes tokamax accepts.
    """
    pytest.importorskip(_TOKAMAX_OP_MODULE)
    device = torch.device("tpu")

    impl = _impl()
    q, k, v = WORKLOAD.qkv()
    q, k, v = q.to(device), k.to(device), v.to(device)
    kv_cache = WORKLOAD.kv_cache().to(device)
    md = WORKLOAD.metadata()
    md = AttentionMetadata(
        input_positions=md.input_positions.to(device),
        block_tables=md.block_tables.to(device),
        seq_lens=md.seq_lens.to(device),
        query_start_loc=md.query_start_loc.to(device),
        request_distribution=md.request_distribution.to(device),
    )

    import jax
    mesh = jax.sharding.Mesh(jax.devices()[:1], ("model", ))
    with set_vllm_model_wrapper_context(mesh=mesh, vllm_config=_vllm_config()):
        impl.initialize_kernel(_layer())
        outputs = impl.forward(_layer(), q, k, v, kv_cache, md)

    assert outputs.shape == q.shape
    assert torch.isfinite(outputs.float()).all()
    # The decode requests each wrote one token into their first page.
    assert kv_cache.float().abs().sum().item() > 0
