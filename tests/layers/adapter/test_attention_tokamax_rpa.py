# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone tests for the tokamax-only `attention.py`.

Ragged paged attention has exactly one implementation now: tokamax's
`PallasMosaicTpuBatchedRpa` singleton, called at the torch level by
`PallasAttentionBackendImpl._build_rpa_kernel`. The in-tree flash kernel
entries, the batched kernel, the `USE_BATCHED_RPA_LONGCTX` fork, the bundled
(block-major) variant and the PCP streaming path are all gone, and with them the
`_kernel_entry` indirection that used to choose between them. What is left to
protect is (a) the adapter seam between this repo's calling convention and
tokamax's, and (b) the guards that make every configuration tokamax cannot serve
fail loudly instead of silently attending over the wrong thing.

Standalone by construction: every fixture this module needs is defined here, so
it runs on its own (`pytest tests/layers/adapter/test_attention_tokamax_rpa.py`)
without `tests/conftest.py`. All but the last test run on CPU against a stubbed
tokamax operator -- they check wiring, not numerics. `test_real_tokamax_op_*`
is the one that needs a TPU and skips without one.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
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
from vllm_torchtpu.layers.adapter import mla_attention as mla_attn
from vllm_torchtpu.layers.adapter.attention import (
    PallasAttentionBackend, PallasAttentionBackendImpl,
    PallasBatchedRPAAttentionBackend)
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
# heads, head_dim 128) at the only page size the backend still offers.
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
    # `PallasAttentionBackend.get_supported_kernel_block_sizes()`.
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


def _impl(workload: Workload = WORKLOAD,
          **kwargs) -> PallasAttentionBackendImpl:
    head_size = kwargs.pop("head_size", workload.head_size)
    kwargs.setdefault("kv_cache_dtype", "bfloat16")
    kwargs.setdefault("sliding_window", None)
    return PallasAttentionBackendImpl(
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
    """Drop built closures between tests.

    `_kernel_registry` is class-level: it is how two layers with the same
    config share one closure over tokamax's operator. That sharing is the
    production behaviour and is asserted below, but across tests it would hand
    a later test a closure holding an earlier test's stubbed operator.
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


def test_backend_offers_only_the_page_size_tokamax_tiles():
    """HEAD_ALONG_SUBLANE puts one page per 128-lane tile. The other multiples
    of 128 belonged to the HND and PCP-streaming kernels, which are gone."""
    assert PallasAttentionBackend.get_supported_kernel_block_sizes() == [256]


def test_backend_offers_only_the_layout_tokamax_models():
    """The pool is physically NHD. Advertising BLNHC as well used to let vLLM
    negotiate the block-major layout the bundled kernel wanted."""
    from vllm.v1.kv_cache_layout import KVCacheLayout as VllmKVCacheLayout
    assert PallasAttentionBackend.supported_kv_cache_layouts() == (
        VllmKVCacheLayout.LBNHC, )


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


def test_pool_shape_has_no_head_dim_64_branch():
    """The hd64 kernel packed K and V differently. It is gone, so head_dim 64
    gets the same `num_kv_heads * 2` head axis as every other width -- and
    `_tokamax_rpa_closure` refuses it outright rather than attend over it."""
    shape = PallasAttentionBackend.get_kv_cache_shape(WORKLOAD.num_blocks, 256,
                                                      WORKLOAD.num_kv_heads,
                                                      64, WORKLOAD.dtype)
    assert shape[2:4] == (WORKLOAD.num_kv_heads * 2 // 2, 2)
    # Padded up to the TPU's 128-lane minimum, as every head width is.
    assert shape[-1] == 128


def test_custom_backend_is_the_same_backend_under_another_name():
    """vLLM's registry is keyed by `AttentionBackendEnum`, so
    `--attention-backend CUSTOM` has to resolve to a class of its own. It
    overrides nothing but the name it reports."""
    # `_abc_impl` is ABCMeta's, stamped onto every AttentionBackend subclass.
    own = {
        k
        for k in vars(PallasBatchedRPAAttentionBackend)
        if not k.startswith("__") and k != "_abc_impl"
    }
    assert own == {"get_name"}
    assert PallasBatchedRPAAttentionBackend.get_name() == "CUSTOM"
    assert PallasAttentionBackend.get_name() == "FLASH_ATTN"
    assert (PallasBatchedRPAAttentionBackend.get_impl_cls() is
            PallasAttentionBackend.get_impl_cls() is
            PallasAttentionBackendImpl)


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
    and saves rebuilding the closure per layer; the per-instance
    `_kernel_config_cache` only saves the lookup."""
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
    """head_dim 64 selected the hd64 kernel and its differently-packed pool."""
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


def test_build_rpa_kernel_rejects_block_major_kv(fake_tokamax_op, monkeypatch):
    """Block-major KV was served by the bundled kernel, which took the whole
    pool plus a dynamic layer index. tokamax's operator takes one layer's."""
    monkeypatch.setenv("VLLM_TPU_BLOCK_MAJOR_KV", "1")
    with set_vllm_model_wrapper_context(mesh=_mesh(),
                                        vllm_config=_vllm_config()):
        with pytest.raises(NotImplementedError,
                           match="VLLM_TPU_BLOCK_MAJOR_KV"):
            _impl()._build_rpa_kernel(None, None, None)


def test_build_rpa_kernel_rejects_kernel_iter_mode(fake_tokamax_op,
                                                   monkeypatch):
    """The timing loop re-runs one kernel in place; tokamax donates the pool,
    so the second iteration would read a donated buffer."""
    monkeypatch.setenv("TPU_KERNEL_ITER_MODE", "1")
    with set_vllm_model_wrapper_context(mesh=_mesh(),
                                        vllm_config=_vllm_config()):
        with pytest.raises(NotImplementedError, match="TPU_KERNEL_ITER_MODE"):
            _impl()._build_rpa_kernel(None, None, None)


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


def test_decode_query_size_stays_at_one():
    """The longctx fork used to raise this to num_speculative_tokens + 1."""
    assert PallasAttentionBackendImpl.decode_query_size == 1
    assert _impl().decode_query_size == 1


def test_build_rpa_kernel_rejects_a_non_causal_mask(fake_tokamax_op):
    """DFlash asks for this; tokamax's wrapper raises for it mid-compile, so
    it is hoisted to a build-time guard with a message that names the knob."""
    impl = _impl(use_causal_mask=False)
    with set_vllm_model_wrapper_context(mesh=_mesh(),
                                        vllm_config=_vllm_config()):
        with pytest.raises(NotImplementedError, match="use_causal_mask=False"):
            impl._build_rpa_kernel(None, None, None)


def test_sinks_are_refused_at_construction():
    """Sinks are not in tokamax's signature at all, so there is nowhere to
    forward them -- and nothing downstream that could notice they were
    dropped. Refuse while the layer is still being built."""
    with pytest.raises(NotImplementedError, match="Attention sinks"):
        _impl(sinks=torch.zeros(WORKLOAD.num_heads))


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
               md.query_start_loc, md.request_distribution)

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
                   md.query_start_loc, md.request_distribution)


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


@pytest.mark.parametrize(("kwargs", "match"), [
    ({
        "pcp": 2
    }, "Prefill context parallelism"),
    ({
        "dcp": 2
    }, "Decode context parallelism"),
])
def test_initialize_kernel_refuses_context_parallelism(fake_tokamax_op, kwargs,
                                                       match):
    """Both modes split the KV axis across ranks and recombine the partial
    softmaxes, which needs a kernel that hands back its log-sum-exp. tokamax's
    operator returns the attention output only, so neither can be faked by
    running it per rank -- hence a raise rather than a silent wrong answer."""
    impl = _impl()
    with set_vllm_model_wrapper_context(mesh=_mesh(),
                                        vllm_config=_vllm_config(**kwargs)):
        with pytest.raises(NotImplementedError, match=match):
            impl.initialize_kernel(_layer())
    assert not fake_tokamax_op


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
    # Argument order is tokamax's: q/k/v first, cache fourth, no `sinks`. q/k/v
    # are distinct random tensors, so this also catches a k/v transposition,
    # which the shapes alone would not.
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
# Regression: the other implementations stay deleted
# =========================================================================


@pytest.mark.parametrize(
    "name",
    [
        # The in-tree kernel entries and the indirection that chose between them.
        "_kernel_entry",
        "_kernel_entry_bundled",
        "_kernel_op_prefix",
        "_pallas_rpa_kernel_impl",
        "_pallas_rpa_kernel_default",
        "_pallas_rpa_kernel_local",
        "_pallas_rpa_kernel_batched",
        "_DRAFT_KV_BLOCK_CAP",
        # The bundled (block-major) variant.
        "_build_bundled_kernel",
        "setup_bundled",
        "_bundled_kernel_registry",
        # The batched schedule the runner used to have to respect.
        "runs_batched_rpa_schedule",
        # Context parallelism.
        "_validate_pcp_streaming_support",
        "_select_kernel_mesh",
        "_run_dcp_forward",
        # A second builder: `_build_rpa_kernel` is the only entry point.
        "_build_tokamax_rpa_kernel",
        # The second impl class. One kernel, one impl.
        "PallasBatchedRPAAttentionBackendImpl",
    ])
def test_removed_symbols_stay_removed(name):
    assert not hasattr(attn, name)
    assert not hasattr(PallasAttentionBackendImpl, name)


@pytest.mark.parametrize("module_name", [
    "vllm_torchtpu.layers.adapter.rpa_kernels",
    "vllm_torchtpu.layers.adapter.cp_attention",
    "vllm_torchtpu.kernels.experimental.batched_rpa.kernel",
    "vllm_torchtpu.kernels.experimental.batched_rpa.flash_attention",
    "vllm_torchtpu.kernels.experimental.batched_rpa.schedule",
    "vllm_torchtpu.kernels.experimental.batched_rpa_longctx",
])
def test_removed_modules_stay_removed(module_name):
    """`find_spec` rather than an import, so a module left on disk but broken
    still fails this."""
    assert importlib.util.find_spec(module_name) is None


def test_batched_rpa_package_is_down_to_what_pcp_streaming_needs():
    """`kernels/experimental/batched_rpa` outlives its kernel: the PCP
    streaming RPA kernel -- the substrate for GDN attention and MLA -- still
    reaches for its `KVLayout` enum, its utils and `wrapper.prepare_inputs`."""
    from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper
    assert hasattr(wrapper, "prepare_inputs")
    for name in ("ragged_paged_attention", "prepare_outputs",
                 "get_kv_cache_shape", "calculate_block_sizes",
                 "schedule_capacity", "schedule_pairs"):
        assert not hasattr(wrapper, name)


def test_attention_module_no_longer_imports_streaming_rpa_entry_points():
    """Nothing from the PCP streaming package is left in `attention`.

    The generic `jax_op` helper does survive, but in `mla_attention`: its one
    remaining caller is the sparse-MLA DCP op, which moved there with the rest
    of MLA.
    """
    module = sys.modules[attn.__name__]
    for name in ("pcp_streaming_jax_op",
                 "pcp_streaming_ragged_paged_attention",
                 "is_pcp_streaming_attention_metadata",
                 "ragged_paged_attention_batched"):
        assert not hasattr(module, name)
    assert hasattr(mla_attn, "pcp_streaming_jax_op")


def _imported_modules(module) -> set[str]:
    """Every module name `module` imports, from its source rather than its
    globals -- a `from x import y` binds `y`, not `x`."""
    names = set()
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_mla_lives_in_its_own_module():
    """`attention` is the RPA half only -- MLA shares no kernel with it."""
    module = sys.modules[attn.__name__]
    for name in ("PallasMLAttentionBackend", "PallasMLAttentionBackendImpl",
                 "VllmTPUDeepseekV32IndexerBackend", "mla_attention",
                 "mla_v2_kernel", "SparseMLAKVCacheSpec"):
        assert not hasattr(module, name)

    # The dependency runs one way only: MLA reuses the KV dtype/page helpers,
    # `attention` reaches for nothing in return. A cycle here would break the
    # import that `layers/__init__` does for the registration side effects.
    assert mla_attn.get_dtype_packing is attn.get_dtype_packing
    assert mla_attn.get_tpu_min_page_size is attn.get_tpu_min_page_size
    assert not any(
        m.startswith("vllm_torchtpu.layers.adapter.mla_attention")
        for m in _imported_modules(module))


def test_attention_reaches_no_in_tree_attention_kernel():
    """The whole point of the collapse: `attention` imports the metadata, the
    quantization helper and the layout enum, and gets its kernel from tokamax.
    Nothing under `kernels/` that computes attention is left in its imports."""
    imported = _imported_modules(sys.modules[attn.__name__])
    kernels = {
        m
        for m in imported if ".kernels." in m or m.endswith(".kernels")
    }
    # Only the layout enum, which is a plain `enum.Enum` of pool layouts.
    assert kernels == {"vllm_torchtpu.kernels.experimental.batched_rpa"}
    assert not any("attention_interface" in m for m in imported)


def test_both_backends_register_under_their_own_module():
    """vLLM instantiates a backend from the path `register_backend` recorded,
    so moving a class rewrites a string the platform hands back."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    assert (AttentionBackendEnum.FLASH_ATTN.get_path() ==
            "vllm_torchtpu.layers.adapter.attention.PallasAttentionBackend")
    assert (AttentionBackendEnum.CUSTOM.get_path() ==
            "vllm_torchtpu.layers.adapter.attention."
            "PallasBatchedRPAAttentionBackend")
    assert (AttentionBackendEnum.FLASH_ATTN_MLA.get_path() ==
            "vllm_torchtpu.layers.adapter.mla_attention."
            "PallasMLAttentionBackend")


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
