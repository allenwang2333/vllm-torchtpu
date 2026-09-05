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
"""The pooled indexer K cache -- spec, geometry, exemption.

What this file can and cannot reach on the CPU rung, stated up front because
the natural check is "construct the engine on the tiny fixture", and that
is **not possible here**:

* ``vllm_torchtpu.runner.tpu_runner`` does not import on this rung at all. It
  pulls ``ModelOptNvFp4LinearMethod`` through
  ``layers/vllm/quantization/nvfp4.py:50``, and the developer checkout of vLLM
  on ``PYTHONPATH`` no longer has that name. Pre-existing rebase drift, not
  ours, and not ours to fix -- but it means the runner branch
  has no CPU test and every claim about it is a *device* claim.
* ``MLAAttentionSpec(compress_ratio=...)`` is a ``TypeError`` against that same
  checkout, because ``main`` replaced ``compress_ratio`` with
  ``tokens_per_state``. The pod's ``v0.27.0`` has it. See
  ``docs/vllm-kv-cache-api-pin.md``.

So everything the spec class does *around* the vLLM dataclass -- the derived
block geometry, the divisibility refusals, the packed shape, the pool-to-page
mapping -- lives in pure functions in
``models/vllm/glm5_next/indexer.py`` and is tested here for real. The spec
construction itself is tested against the real class when the imported vLLM is
the pinned one, and against a *documented emulation* of ``v0.27.0``'s two
compress-ratio members otherwise, so the normalizer's behaviour is still
covered rather than skipped into silence.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import MambaSpec, MLAAttentionSpec

from vllm_torchtpu.kv_cache_spec_normalizer import \
    normalize_kv_cache_specs_for_tpu
from vllm_torchtpu.models.vllm.glm5_next.indexer import (
    POOLED_CACHE_KV_PACKING, Glm5NextIndexerKCache,
    derive_pooled_indexer_geometry, is_pooled_indexer_cache,
    pooled_indexer_kv_cache_spec, pooled_record_bytes)

# The production geometry. `block_size` is vLLM's default 16 because GLM is
# hybrid, so `tpu_platform.py:1159-1163` leaves it alone -- unlike DeepSeek-V4,
# which takes the MLA page size (1024) at `:1150-1157`. That single difference
# is what makes DSv4's tuned `num_kv_pages_per_block` illegal here.
GLM_BLOCK_SIZE = 16
GLM_INDEX_KPOOL = 4
GLM_INDEX_HEAD_DIM = 128
GLM_MAX_MODEL_LEN = 131072

DSV4_BLOCK_SIZE = 1024
DSV4_NUM_KV_PAGES_PER_BLOCK = (3, 2, 2)


def _glm(**overrides):
    kwargs = dict(
        block_size=GLM_BLOCK_SIZE,
        index_kpool=GLM_INDEX_KPOOL,
        index_head_dim=GLM_INDEX_HEAD_DIM,
        max_model_len=GLM_MAX_MODEL_LEN,
    )
    kwargs.update(overrides)
    return derive_pooled_indexer_geometry(**kwargs)


# --- the derived block geometry -------------------------------------------


def test_glm_block_geometry_is_derived_and_prints_every_input():
    """The geometry numbers, *computed* rather than copied.

    Printed rather than assumed, because they are not obvious:
    the tiny fixture pads 12 real pages up to 32 and is therefore not
    representative of the block-utilisation regime.
    """
    geom = _glm()
    print(geom.describe())

    # `page_size` is compressed *records* per page -- block_size //
    # compress_ratio -- and is the quantity `streamindex_topk` calls
    # `page_size` at :746. Not the 16 raw tokens in a block.
    assert geom.page_size == 4
    assert geom.pages_per_seq == GLM_MAX_MODEL_LEN // GLM_BLOCK_SIZE == 8192
    # 4 * 32 = 128: the smallest tuple `streamindex_topk.py:749-754` admits.
    assert geom.num_kv_pages_per_block == (32, 32, 32)
    assert geom.num_sublanes_total == 8192 * 4 // 128 == 256
    # The acceptance number: per-page bytes == block_size * 64.
    assert geom.page_bytes == GLM_BLOCK_SIZE * 64 == 1024
    assert geom.record_bytes == 256


def test_every_derived_tiling_entry_satisfies_both_divisibility_rules():
    """The loud rule and the one section 4(a) proves is implied by it."""
    for max_model_len in (192, 2048, 4096, 32768, 131072, 1_048_576):
        geom = _glm(max_model_len=max_model_len)
        for bkv_p in geom.num_kv_pages_per_block:
            assert geom.page_size * bkv_p % 128 == 0
            spanned = -(-geom.pages_per_seq // bkv_p) * bkv_p * geom.page_size
            assert spanned % 128 == 0
        print(f"max_model_len={max_model_len}: {geom.describe()}")


def test_dsv4_tuned_tiling_cannot_build_for_glm():
    """A refusal at construction rather than at trace time.

    `(3, 2, 2)` is copied into every new `streamindex_topk` call site by
    reflex (`deepseek_v4_indexer.py:94`). At GLM's page_size 4 it is a
    `ValueError` from inside a Pallas trace, minutes into a TPU queue.
    """
    with pytest.raises(ValueError) as excinfo:
        _glm(num_kv_pages_per_block=DSV4_NUM_KV_PAGES_PER_BLOCK)
    message = str(excinfo.value)
    # The two numbers a reader needs in order to fix it.
    assert f"block_size={GLM_BLOCK_SIZE}" in message
    assert f"max_model_len={GLM_MAX_MODEL_LEN}" in message
    # And the repair: every entry must be a multiple of 32.
    assert "multiple of 32" in message
    print(message)


def test_dsv4_own_geometry_still_admits_its_own_tuning():
    """The refusal above must be about GLM's page, not about `(3, 2, 2)`.

    A rule that banned the tuple outright would be indistinguishable from a
    correct one in every GLM test and wrong for DeepSeek-V4, whose
    `block_size` is 1024 and whose `page_size` is therefore 256.
    """
    geom = derive_pooled_indexer_geometry(
        block_size=DSV4_BLOCK_SIZE,
        index_kpool=4,
        index_head_dim=GLM_INDEX_HEAD_DIM,
        max_model_len=GLM_MAX_MODEL_LEN,
        num_kv_pages_per_block=DSV4_NUM_KV_PAGES_PER_BLOCK,
    )
    assert geom.page_size == 256
    assert geom.num_kv_pages_per_block == DSV4_NUM_KV_PAGES_PER_BLOCK
    print(geom.describe())


def test_block_size_not_a_multiple_of_index_kpool_is_refused():
    """A pool must not straddle two KV blocks."""
    with pytest.raises(ValueError, match="straddle"):
        _glm(block_size=18)


def test_block_size_that_floors_the_packing_axis_is_refused():
    """The failure that allocates a cache with a missing dimension.

    At `block_size=8, index_kpool=4` the page holds 2 records and the packed
    shape's middle axis is `2 // 4 == 0`. `torch.zeros` accepts a zero-sized
    dimension without complaint, every scatter into it is a no-op, and every
    gather returns nothing -- there is no index error because there are no
    indices. The assertion below is the only thing between that and a model
    that generates fluent text with an all-zero indexer.
    """
    assert 8 // GLM_INDEX_KPOOL // POOLED_CACHE_KV_PACKING == 0, (
        "premise of this test: page_size // kv_packing floors to zero here")
    with pytest.raises(ValueError) as excinfo:
        _glm(block_size=8)
    assert "floors to 0" in str(excinfo.value)
    print(str(excinfo.value))


def test_score_columns_cover_every_candidate_pool():
    """`num_sublanes_total * 128` must not be short of the real pool count.

    Implied by `align_to(pages_per_seq, bkv_p) >= pages_per_seq`, but it is
    the property that actually matters -- a short score buffer drops the
    highest-numbered (most recent) pools, which is invisible at short lengths.
    """
    for max_model_len in (192, 2048, 131072):
        geom = _glm(max_model_len=max_model_len)
        assert geom.score_columns >= geom.real_pools_per_seq, geom.describe()


def test_the_tiny_fixture_geometry_is_not_representative():
    """Recorded, not asserted away: section 4(b)'s point, with numbers.

    A 192-token fixture at block_size 16 has 12 real pages and pads to 32, so
    it exercises 37.5% block utilisation and 128 score columns against 48
    real pools. Nothing about block utilisation generalises from it.
    """
    geom = _glm(max_model_len=192)
    assert geom.pages_per_seq == 12
    assert geom.num_sublanes_total == 1
    assert geom.score_columns == 128
    assert geom.real_pools_per_seq == 48
    assert geom.block_utilisation == pytest.approx(12 / 32)
    print(geom.describe())


# --- the packed shape and the pool-to-page mapping ------------------------


def test_pooled_cache_shape_matches_the_dsv4_packed_layout():
    """`(num_blocks, page_size // packing, packing, record_bytes)`.

    The same 4-D layout `tpu_runner.py:5912` allocates for DeepSeek-V4's
    indexer cache and `streamindex_topk.py:745` unpacks.
    """
    geom = _glm()
    shape = geom.cache_shape(num_blocks=97)
    assert shape == (97, 1, 4, 256)
    # Bytes per block must equal the page the spec declares, or vLLM's
    # num_blocks was computed against a different footprint than we allocate.
    assert shape[1] * shape[2] * shape[3] == geom.page_bytes


def test_pool_i_lands_in_the_page_of_raw_token_kpool_times_i():
    """Plan item 9(d), as arithmetic over the whole first block-table row.

    Two things at once: every raw token of pool `i` is in the same KV block
    (so the pool never needs two pages), and pool `i` occupies slot
    `i % (block_size // index_kpool)` of that block's page.
    """
    geom = _glm()
    pools_per_page = geom.page_size
    for pool in range(4 * pools_per_page):
        first_raw = pool * GLM_INDEX_KPOOL
        last_raw = first_raw + GLM_INDEX_KPOOL - 1
        block = first_raw // GLM_BLOCK_SIZE
        assert last_raw // GLM_BLOCK_SIZE == block, (
            f"pool {pool} straddles blocks {block} and "
            f"{last_raw // GLM_BLOCK_SIZE}")
        assert block == pool // pools_per_page
        slot = pool % pools_per_page
        # ... and the slot's address inside the packed page.
        assert (slot // geom.kv_packing,
                slot % geom.kv_packing) == (slot // POOLED_CACHE_KV_PACKING,
                                            slot % POOLED_CACHE_KV_PACKING)


def test_record_bytes_reserve_the_scale_byte():
    """128 fp8 values + 1 e8m0 scale, aligned to 128 -> 256.

    Dropping the ``+ 1`` gives 128, which is still 128-aligned and still
    allocates -- the scale byte would just overlap the next record's first
    value and decode as a finite fp8 number.
    """
    assert pooled_record_bytes(128) == 256
    assert pooled_record_bytes(127) == 128
    assert pooled_record_bytes(128) > 128


# --- the spec, against whichever vLLM is imported -------------------------


def _vllm_has_compress_ratio() -> bool:
    return "compress_ratio" in {
        f.name
        for f in dataclasses.fields(MLAAttentionSpec)
    }


@dataclasses.dataclass(frozen=True, kw_only=True)
class _EmulatedV027MLAAttentionSpec(MLAAttentionSpec):
    """``v0.27.0``'s two compress-ratio members, on top of whatever is here.

    A transcription of ``vllm/v1/kv_cache_interface.py`` at the pin --
    ``compress_ratio`` (``:386``), ``storage_block_size`` (``:396-397``),
    ``real_page_size_bytes`` (``:413-418``) and ``unpadded_page_size_bytes``
    (``:185-194``) -- used **only** when the imported vLLM is the developer
    checkout. It exists so the normalizer test below covers real behaviour
    instead of skipping: without ``storage_block_size`` the 4x padding this
    exemption prevents cannot be demonstrated at all on the CPU rung.

    ``unpadded_page_size_bytes`` is the one that is easy to forget and the
    one that decides ``num_blocks``. At the pin it *delegates* to
    ``real_page_size_bytes``, so the compression reaches vLLM's budget; on
    ``main`` it is computed independently from ``num_states``. Omitting the
    override here made this emulation report a 4096-byte pooled page for a
    1024-byte allocation -- a 4x under-count of usable blocks, with no error.
    """

    compress_ratio: int = 1

    @property
    def storage_block_size(self) -> int:
        return self.block_size // self.compress_ratio

    @property
    def real_page_size_bytes(self) -> int:
        return (self.storage_block_size * self.num_kv_heads * self.head_size *
                self.dtype.itemsize)

    @property
    def unpadded_page_size_bytes(self) -> int:
        return self.real_page_size_bytes


def _pooled_spec_for_this_vllm():
    """The pooled spec, real if possible and emulated otherwise."""
    if _vllm_has_compress_ratio():
        return pooled_indexer_kv_cache_spec(
            block_size=GLM_BLOCK_SIZE,
            index_kpool=GLM_INDEX_KPOOL,
            index_head_dim=GLM_INDEX_HEAD_DIM,
        ), "real"
    return _EmulatedV027MLAAttentionSpec(
        block_size=GLM_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=pooled_record_bytes(GLM_INDEX_HEAD_DIM),
        dtype=torch.uint8,
        compress_ratio=GLM_INDEX_KPOOL,
        alignment=None,
    ), "emulated v0.27.0"


def test_spec_uses_the_v0_27_0_spelling_when_that_vllm_is_imported():
    """Against the pinned vLLM, the spec's own arithmetic must agree with us."""
    if not _vllm_has_compress_ratio():
        pytest.skip(
            "imported vLLM has no MLAAttentionSpec.compress_ratio, so this is "
            "the developer checkout, not the pod's v0.27.0. Expected on the "
            "CPU rung; see docs/vllm-kv-cache-api-pin.md.")
    spec = pooled_indexer_kv_cache_spec(
        block_size=GLM_BLOCK_SIZE,
        index_kpool=GLM_INDEX_KPOOL,
        index_head_dim=GLM_INDEX_HEAD_DIM,
    )
    geom = _glm()
    assert spec.compress_ratio == GLM_INDEX_KPOOL
    assert spec.storage_block_size == geom.page_size
    assert spec.head_size == geom.record_bytes
    assert spec.dtype is torch.uint8
    assert spec.real_page_size_bytes == geom.page_bytes


def test_spec_construction_names_the_pin_document_on_the_wrong_vllm():
    """The bare TypeError sends the reader hunting for a typo."""
    if _vllm_has_compress_ratio():
        pytest.skip("imported vLLM accepts compress_ratio; nothing to refuse")
    with pytest.raises(TypeError) as excinfo:
        pooled_indexer_kv_cache_spec(
            block_size=GLM_BLOCK_SIZE,
            index_kpool=GLM_INDEX_KPOOL,
            index_head_dim=GLM_INDEX_HEAD_DIM,
        )
    assert "docs/vllm-kv-cache-api-pin.md" in str(excinfo.value)


# --- the normalizer exemption, and the witness that it is load-bearing ----


def _mamba_spec(page_bytes: int) -> MambaSpec:
    """A stand-in KDA/mamba spec, sized to dominate the hybrid max page."""
    return MambaSpec(
        shapes=((page_bytes // 4, ), ),
        dtypes=(torch.float32, ),
        block_size=GLM_BLOCK_SIZE,
        num_speculative_blocks=0,
        mamba_type="linear_attention",
        page_size_padded=None,
    )


def test_the_normalizer_exemption_is_load_bearing():
    """Same spec, exempt and not: the not-exempt one is silently inflated.

    Two inflations stack, and the test separates them:

    * `_normalize_one_spec` sizes an MLA page from `spec.block_size` rather
      than `spec.storage_block_size` (`kv_cache_spec_normalizer.py:72-78`),
      taking a 1 KiB pooled page to 4 KiB;
    * `_has_hybrid_attention_and_mamba` then pads every non-exempt spec up to
      the model's maximum page size.

    Neither raises. Both cost HBM that `uniform_page_size_bytes` never
    budgeted for.
    """
    pooled, provenance = _pooled_spec_for_this_vllm()
    print(f"pooled spec provenance: {provenance}")
    geom = _glm()
    mla = MLAAttentionSpec(
        block_size=GLM_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
    )
    specs = {
        "layers.3.self_attn.indexer.k_cache": pooled,
        "layers.3.self_attn.mla_attn": mla,
        "layers.0.linear_attn": _mamba_spec(4096),
    }

    exempt = normalize_kv_cache_specs_for_tpu(
        dict(specs),
        torch.bfloat16,
        enable_unified_kv_layout=True,
        exempt_layers={"layers.3.self_attn.indexer.k_cache"},
    )["layers.3.self_attn.indexer.k_cache"]
    not_exempt = normalize_kv_cache_specs_for_tpu(
        dict(specs),
        torch.bfloat16,
        enable_unified_kv_layout=True,
        exempt_layers=set(),
    )["layers.3.self_attn.indexer.k_cache"]

    assert exempt is pooled, (
        "an exempt spec must pass through untouched; anything else means the "
        "pooled page was resized by a formula that does not model it")
    assert exempt.page_size_bytes == geom.page_bytes, (
        f"exempt pooled page is {exempt.page_size_bytes} B, expected "
        f"{geom.page_bytes} B")

    inflated = not_exempt.page_size_bytes
    print(f"pooled page: exempt={geom.page_bytes} B, "
          f"not exempt={inflated} B "
          f"({inflated / geom.page_bytes:.1f}x)")
    assert inflated > geom.page_bytes, (
        "the exemption would be dead code if the normalizer left this spec "
        "alone -- this witness is what keeps the exemption honest")
    # The block_size-instead-of-storage_block_size half, on its own.
    assert inflated >= GLM_BLOCK_SIZE * geom.record_bytes == 4096


def test_exemption_does_not_leak_to_the_other_specs():
    """Exempting the pooled cache must not exempt anything else."""
    pooled, _ = _pooled_spec_for_this_vllm()
    mla = MLAAttentionSpec(block_size=GLM_BLOCK_SIZE,
                           num_kv_heads=1,
                           head_size=576,
                           dtype=torch.bfloat16)
    specs = {
        "indexer": pooled,
        "mla": mla,
        "kda": _mamba_spec(4096),
    }
    out = normalize_kv_cache_specs_for_tpu(
        dict(specs),
        torch.bfloat16,
        enable_unified_kv_layout=True,
        exempt_layers={"indexer"},
    )
    assert out["mla"].page_size_bytes == out["kda"].page_size_bytes, (
        "the hybrid uniform page must still cover MLA and mamba")
    assert out["indexer"].page_size_bytes != out["mla"].page_size_bytes


# --- the class, and what it must not be -----------------------------------


def test_pooled_cache_is_not_a_deepseek_cache_subclass():
    """The no-inheritance rule, checked at the type level.

    `tpu_runner.is_cache_for_ds_v4` is an `isinstance` over these classes and
    any `True` sets `_is_ds_v4`, rerouting the whole model through
    `_initialize_ds_v4_kv_cache` -- which raises on GLM's first KDA layer.
    Re-implemented here rather than imported because `tpu_runner` does not
    import on the CPU rung (see this module's docstring).
    """
    from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
    from vllm.models.deepseek_v4.attention import DeepseekV4IndexerCache
    from vllm.models.deepseek_v4.compressor import CompressorStateCache
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache

    ds_v4_cache_types = (DeepseekV4SWACache, DeepseekV4IndexerCache,
                         CompressorStateCache, DeepseekV32IndexerCache)
    for cache_type in ds_v4_cache_types:
        assert not issubclass(Glm5NextIndexerKCache, cache_type), (
            f"Glm5NextIndexerKCache inherits {cache_type.__name__}, so "
            "is_cache_for_ds_v4 returns True for it and the runner reroutes "
            "GLM through _initialize_ds_v4_kv_cache")


def _register_fake_vllm_config(monkeypatch) -> dict:
    """A `get_current_vllm_config()` with just a static forward context.

    `Glm5NextIndexerKCache.__init__` registers itself there the way
    `DeepseekV32IndexerCache` does; that dict is the only reason the runner
    ever sees the cache, so construction is not testable without it.
    """
    import vllm.config as vllm_config_module

    context: dict = {}
    monkeypatch.setattr(
        vllm_config_module,
        "get_current_vllm_config",
        lambda: SimpleNamespace(compilation_config=SimpleNamespace(
            static_forward_context=context)),
    )
    return context


def _make_cache(prefix: str,
                *,
                block_size: int = GLM_BLOCK_SIZE,
                max_model_len: int = GLM_MAX_MODEL_LEN):
    return Glm5NextIndexerKCache(
        index_kpool=GLM_INDEX_KPOOL,
        index_head_dim=GLM_INDEX_HEAD_DIM,
        cache_config=SimpleNamespace(block_size=block_size),
        prefix=prefix,
        max_model_len=max_model_len,
    )


def test_construction_registers_and_derives(monkeypatch):
    context = _register_fake_vllm_config(monkeypatch)
    cache = _make_cache("layers.3.self_attn.indexer.k_cache")
    assert context["layers.3.self_attn.indexer.k_cache"] is cache
    assert is_pooled_indexer_cache(cache)
    assert cache.geometry.page_bytes == 1024
    print(cache.geometry.describe())


def test_construction_refuses_an_illegal_block_size(monkeypatch):
    """The refusal must land at construction, not at the first Pallas trace."""
    _register_fake_vllm_config(monkeypatch)
    with pytest.raises(ValueError, match="floors to 0"):
        _make_cache("l3.k_cache", block_size=8)


def test_duplicate_prefix_is_refused(monkeypatch):
    """Two caches at one prefix would silently share a KV tensor."""
    _register_fake_vllm_config(monkeypatch)
    _make_cache("l3.k_cache")
    with pytest.raises(ValueError, match="Duplicate layer name"):
        _make_cache("l3.k_cache")


def test_get_kv_cache_spec_rederives_when_the_block_size_moved(monkeypatch):
    """`update_block_size_for_backend` can raise `block_size` after `__init__`.

    A stale geometry would size the page for a block layout the allocator no
    longer uses -- and since both numbers are legal, nothing would complain.
    """
    _register_fake_vllm_config(monkeypatch)
    cache = _make_cache("l3.k_cache")
    assert cache.geometry.page_size == 4
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=64),
        model_config=SimpleNamespace(max_model_len=4096),
    )
    try:
        cache.get_kv_cache_spec(vllm_config)
    except TypeError:
        # The developer checkout cannot build the spec; the re-derivation
        # happens before that call and is what this test is about.
        pass
    assert cache.geometry.block_size == 64
    assert cache.geometry.page_size == 16
    assert cache.geometry.max_model_len == 4096
    assert cache.geometry.page_bytes == 64 * 64


def test_bind_kv_cache_keeps_the_packing_axis(monkeypatch):
    """`DeepseekV32IndexerCache` squeezes dim 1; this one must not.

    At GLM's geometry `page_size // kv_packing == 1`, so a `squeeze(1)`
    inherited by reflex would drop the packing axis and every later view
    would read at the wrong stride -- in range, no error.
    """
    _register_fake_vllm_config(monkeypatch)
    cache = _make_cache("l3.k_cache")
    tensor = torch.zeros(cache.geometry.cache_shape(5), dtype=torch.uint8)
    assert tensor.shape == (5, 1, 4, 256)
    cache.bind_kv_cache(tensor)
    assert cache.kv_cache.shape == (5, 1, 4, 256)


def test_get_attn_backend_matches_the_mla_sibling(monkeypatch):
    """Order-independence of `_find_non_ssm_backend`; see the runner assert."""
    from vllm_torchtpu.layers.vllm.attention import PallasMLAttentionBackend

    _register_fake_vllm_config(monkeypatch)
    cache = _make_cache("l3.k_cache")
    assert cache.get_attn_backend() is PallasMLAttentionBackend


def test_is_pooled_indexer_cache_is_narrow():
    """The predicate must not answer True for anything else."""
    assert not is_pooled_indexer_cache(None)
    assert not is_pooled_indexer_cache(torch.nn.Linear(2, 2))
    assert not is_pooled_indexer_cache(object())


# --- source-level tripwire for the runner wiring --------------------------


def test_runner_still_wires_the_pooled_cache_branch():
    """A *source* check, and only because the runner will not import here.

    This is the weakest test in the file and it is here for one reason: the
    critique's finding is that `get_kv_cache_spec`'s module loop ends in a
    bare `else: continue`, so a cache module with no branch is dropped in
    silence. The runtime tripwire for that is
    `TPUModelRunner._assert_pooled_indexer_caches_have_specs`, which cannot
    be called from the CPU rung (`vllm_torchtpu.runner.tpu_runner` fails to
    import -- see this module's docstring). Until the CPU stub is deep enough
    to import it, this at least fails if someone deletes the branch, the
    exemption or the assertion.

    Substrings, not an AST walk, deliberately: the point is only "these three
    edits are still present", and a structural matcher here would encode the
    loop's shape and break on any refactor of it.
    """
    import pathlib

    source = (pathlib.Path(__file__).resolve().parents[4] / "src" /
              "vllm_torchtpu" / "runner" / "tpu_runner.py").read_text()

    required = {
        "the spec-producing branch":
        "elif is_compressed_indexer_cache(attn_module):",
        "the normalizer exemption":
        "exempt_layers=ds_v4_layers | compressed_indexer_layers,",
        "the was-it-really-allocated tripwire":
        "self._assert_pooled_indexer_caches_have_specs(",
        "the packed allocation branch": "attn_module.geometry.cache_shape(",
        # The pooled cache must be sized by the BLOCK TABLE's block count, not
        # by the shared-tensor loop's `tensor_size // sum(per-block bytes)`.
        # Those differ once the pooled spec declares the model's uniform page:
        # the loop divides by the TPU-actual 1024 B while vLLM sized the
        # tensor from the declared 2255872 B. Measured, that asked for
        # 19.09 GiB on one layer against 13.65 GiB free -- ~53x the
        # 3.91 GiB the runner's own planner had budgeted for all eleven.
        "the block-table block count for the pooled cache":
        "pooled_num_blocks = int(kv_cache_config.num_blocks)",
        # Host staging is what runs the container out of memory: eleven
        # pooled caches x eight workers is ~31 GiB of RAM before the copy,
        # and the worker is SIGKILLed with no Python traceback.
        "the pooled cache allocated straight on device":
        "kv_caches[layer_name] = torch.zeros(shape,",
        "the planned-vs-allocated check": "self._report_kv_cache_allocation(",
    }
    missing = [what for what, text in required.items() if text not in source]
    assert not missing, (
        f"tpu_runner.py no longer contains {missing}. A pooled indexer cache "
        "with no spec branch is never allocated and nothing reports it: its "
        "records read back byte-zero, score exactly 0 through the ReLU at "
        "streamindex_topk.py:333, always rank last, and the DSA layers "
        "degrade to a recency window with fluent output.")
