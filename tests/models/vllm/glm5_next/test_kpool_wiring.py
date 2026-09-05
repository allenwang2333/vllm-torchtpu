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
"""Wiring the kpool indexer in, and the guards around it.

Every test here pins something whose wrong answer is *silent* -- shapes stay
valid, nothing raises, and the only symptom is worse retrieval or an OOM
attributed to the wrong thing:

* the rope slab is 128 on the dense path and 64 on the sparse one, and the two
  are chosen by a selector that is **not** ``is_sparse``, because the
  configuration a dense baseline needs -- 576 layout, indexer off --
  does not exist under the naive rule;
* ``VllmTPUIndexer.rebind`` is a ``__class__`` reassignment and will happily
  turn the kpool indexer into the DeepSeek-V2 lightning indexer;
* the pending ring's depth is derived from the spec-decode config, and a
  hard-coded 4 is green at exactly the ``num_speculative_tokens`` the probe
  defaults to;
* the fp32 score buffer is sized by the compressed *capacity*, so
  ``max_num_batched_tokens`` -- not ``max_model_len`` -- is the binding knob;
* a prefix-cache hit leaves the pooled records unwritten, and a byte-zero
  pooled key scores exactly 0 through the ReLU: finite, legal, always last.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip(
    "torch_tpu",
    reason="the attention wrapper imports torch_tpu._internal.pallas; run "
    "with VLLM_TORCHTPU_CPU_STUB=1 or on a TPU host")

from vllm_torchtpu.kernels.glm5_kpool import PENDING_TAG_EMPTY  # noqa: E402
from vllm_torchtpu.kernels.glm5_kpool import (derive_out_width,
                                              derive_pending_state_geometry)
from vllm_torchtpu.models.vllm.glm5_next import attention as attn  # noqa: E402
from vllm_torchtpu.models.vllm.glm5_next import indexer as idx  # noqa: E402
from vllm_torchtpu.models.vllm.glm5_next.model import (  # noqa: E402
    _ALLOW_UNSOUND_CONTEXT, _check_context_within_dense_mla_bound,
    assert_indexer_enablement_uniform)


@pytest.fixture(autouse=True)
def _fake_tp_group(monkeypatch):
    """A single-rank TP group, so ``BasevLLMParameter`` can be constructed.

    ``ReplicatedLinear(disable_tp=True)`` never reads the group, but
    ``BasevLLMParameter.__init__`` calls ``get_tensor_model_parallel_rank()``
    unconditionally. Same fixture as
    ``tests/models/vllm/glm5_next/test_kpool_indexer_weights.py``.
    """
    import types

    from vllm.distributed import parallel_state
    monkeypatch.setattr(parallel_state,
                        "_TP",
                        types.SimpleNamespace(rank_in_group=0, world_size=1),
                        raising=False)


# The real checkpoint's numbers, so a message quoting them quotes reality.
REAL_INDEX_TOPK = 2048
REAL_INDEX_KPOOL = 4
REAL_INDEX_HEAD_DIM = 128
REAL_KV_LORA_RANK = 512
GLM_BLOCK_SIZE = 16

# --------------------------------------------------------------------------
# --- the geometry selector
# --------------------------------------------------------------------------


def _geometry_for_slab(slab: int) -> str:
    for name, width in attn._ROPE_SLAB_BY_GEOMETRY.items():
        if width == slab:
            return name
    raise AssertionError(slab)


def test_auto_follows_the_indexer_but_is_not_the_indexer():
    assert attn.resolve_mla_geometry(
        indexer_enabled=True, requested="auto") == attn.MLA_GEOMETRY_SPARSE_576
    assert attn.resolve_mla_geometry(
        indexer_enabled=False, requested="auto") == attn.MLA_GEOMETRY_DENSE_640


def test_the_slab_is_128_dense_and_64_sparse():
    assert attn.rope_slab_width(attn.MLA_GEOMETRY_DENSE_640) == 128
    assert attn.rope_slab_width(attn.MLA_GEOMETRY_SPARSE_576) == 64
    assert attn.mla_head_size(REAL_KV_LORA_RANK,
                              attn.MLA_GEOMETRY_DENSE_640) == 640
    assert attn.mla_head_size(REAL_KV_LORA_RANK,
                              attn.MLA_GEOMETRY_SPARSE_576) == 576


def test_the_baseline_configuration_exists_dense_kernel_576_layout():
    """The dense baseline at the sparse layout: why the selector is not
    ``is_sparse``.

    Under a naive "64 iff sparse" rule this configuration is unreachable, so
    job A would measure ``head_size`` 640 and job B 576 -- reintroducing the
    KV-layout confound the A/B exists to remove.
    """
    geometry = attn.resolve_mla_geometry(
        indexer_enabled=False, requested=attn.MLA_GEOMETRY_SPARSE_576)
    assert geometry == attn.MLA_GEOMETRY_SPARSE_576
    assert attn.rope_slab_width(geometry) == 64
    assert attn.mla_head_size(REAL_KV_LORA_RANK, geometry) == 576


def test_the_baseline_configuration_is_expressible_through_the_env(
        monkeypatch):
    """Exactly the two variables a cdk recipe would set, and nothing else."""
    monkeypatch.setenv(attn.KPOOL_INDEXER_ENV, "0")
    monkeypatch.setenv(attn.MLA_GEOMETRY_ENV, attn.MLA_GEOMETRY_SPARSE_576)

    enabled = attn.kpool_indexer_enabled()
    assert enabled is False
    assert attn.resolve_mla_geometry(
        indexer_enabled=enabled) == attn.MLA_GEOMETRY_SPARSE_576


def test_indexer_on_at_the_640_layout_is_refused():
    """Both consumers of the 64-wide slab assert; neither adapts."""
    with pytest.raises(ValueError, match="attention_interface.py"):
        attn.resolve_mla_geometry(indexer_enabled=True,
                                  requested=attn.MLA_GEOMETRY_DENSE_640)


def test_an_unknown_geometry_is_refused_rather_than_defaulted():
    with pytest.raises(ValueError, match="unknown MLA geometry"):
        attn.resolve_mla_geometry(indexer_enabled=False, requested="576")


def test_a_typo_in_the_kill_switch_raises(monkeypatch):
    """The dangerous default is "whatever the default happens to be"."""
    monkeypatch.setenv(attn.KPOOL_INDEXER_ENV, "flase")
    with pytest.raises(ValueError, match="neither on nor off"):
        attn.kpool_indexer_enabled()


def test_the_indexer_is_on_by_default_and_the_env_overrides_it(monkeypatch):
    """The indexer is the supported configuration, so the switch is opt-out.

    Its pending ring is addressed by a runner-owned ``kpool_state_indices``
    slot pool, so the default is only defensible once that field exists --
    checked here, because an enabled indexer without it builds fine and then
    raises on its first real forward, minutes into a run and after the whole
    checkpoint has loaded.
    """
    import dataclasses

    from vllm_torchtpu.layers.common.attention_metadata import \
        AttentionMetadata

    assert any(f.name == "kpool_state_indices"
               for f in dataclasses.fields(AttentionMetadata))

    monkeypatch.delenv(attn.KPOOL_INDEXER_ENV, raising=False)
    assert attn.kpool_indexer_enabled() is True

    monkeypatch.setenv(attn.KPOOL_INDEXER_ENV, "1")
    assert attn.kpool_indexer_enabled() is True
    monkeypatch.setenv(attn.KPOOL_INDEXER_ENV, "0")
    assert attn.kpool_indexer_enabled() is False


def test_the_64_slab_satisfies_the_dsa_gather_assertion_and_128_does_not():
    """The arithmetic of ``attention_interface.py:805-807``, checked directly.

    That assertion lives inside a function that needs a device, so it is
    re-derived here from ``dsa_gather``'s own tile constants rather than
    copied as literals: if the tile geometry ever moves, this test moves with
    it and the geometry table does not, which is the disagreement worth
    catching.
    """
    from vllm_torchtpu.kernels.mla.sparse import dsa_gather

    tile_subrows = dsa_gather.TILE_SUBROWS
    lane_bytes = dsa_gather.TILE_LANE_BYTES

    def holds(slab: int) -> bool:
        head_size = attn.mla_head_size(REAL_KV_LORA_RANK,
                                       _geometry_for_slab(slab))
        lkv_dim = head_size - slab
        return lkv_dim == tile_subrows * lane_bytes and slab * 2 == lane_bytes

    assert holds(64), (tile_subrows, lane_bytes)
    assert not holds(128)


def test_the_sparse_specs_split_512_plus_64_at_head_size_576():
    """``get_sparse_kv_cache_specs`` hard-codes the 64; 640 would give 576/64.

    Checked through the backend rather than restated: the quantity that
    matters is ``head_size - _DS_MLA_ROPE_HEAD_DIM`` and the bug shape is a
    nope width that is not 512.
    """
    from vllm_torchtpu.layers.vllm.attention import PallasMLAttentionBackend

    nope, rope = PallasMLAttentionBackend.get_sparse_kv_cache_specs(
        num_blocks=4,
        block_size=GLM_BLOCK_SIZE,
        head_size=attn.mla_head_size(REAL_KV_LORA_RANK,
                                     attn.MLA_GEOMETRY_SPARSE_576),
        cache_dtype_str="fp8",
    )
    # The nope width is the load-bearing one: it is `head_size - 64`, so a
    # 640 head_size would make it 576 and `dsa_gather`'s
    # `lkv_dim == TILE_SUBROWS * TILE_LANE_BYTES` assert would fire.
    assert nope.head_dim == 512
    # The rope side is padded up to the lane width by the kernel's own
    # `get_kv_cache_shape`; what matters is that the spec was *derived* from
    # a 64-wide slab, which the nope width above already pins.
    assert rope.head_dim >= 64


# --------------------------------------------------------------------------
# --- the rebind hazard
# --------------------------------------------------------------------------


class _StandInIndexerCache:
    """Enough of ``DeepseekV32IndexerCache`` for ``rebind`` to mutate."""

    def __init__(self) -> None:
        self.head_dim = 128


def _upstream_indexer():
    """A minimal object that really is an upstream ``Indexer`` instance.

    Built with ``__new__`` because ``Indexer.__init__`` wants a full vLLM
    config and a distributed group; ``rebind`` only touches four attributes.
    """
    from vllm.model_executor.models.deepseek_v2 import Indexer

    obj = Indexer.__new__(Indexer)
    obj.head_dim = 128
    obj.quant_block_size = 128
    obj.k_cache = _StandInIndexerCache()
    return obj


def test_rebind_retypes_a_real_indexer():
    """The guard must stay a no-op for GLM-5.2 / DeepSeek, which need it."""
    from vllm_torchtpu.layers.vllm.custom_ops.mla_attention_op import (
        VllmTPUIndexer, VllmTPUIndexerCache, maybe_rebind_indexer)

    indexer = _upstream_indexer()
    out = maybe_rebind_indexer(indexer)
    assert out is indexer
    assert isinstance(indexer, VllmTPUIndexer)
    assert isinstance(indexer.k_cache, VllmTPUIndexerCache)
    # +1 e8m0 scale byte per quant block.
    assert indexer.k_cache.head_dim == 129


def test_rebind_is_not_applied_to_the_kpool_indexer(monkeypatch):
    """The regression test for the rebind guard. Delete the ``isinstance`` in
    ``maybe_rebind_indexer`` and this fails.

    It fails *loudly* here and only here: ``__class__`` reassignment raises
    nothing, so in production the same event is a working model that retrieves
    against the wrong scores.
    """
    from vllm_torchtpu.layers.vllm.custom_ops.mla_attention_op import \
        maybe_rebind_indexer

    indexer = _build_indexer(monkeypatch)
    k_cache_class = type(indexer.k_cache)
    pooled_record_bytes = indexer.k_cache.geometry.record_bytes

    out = maybe_rebind_indexer(indexer)

    assert out is indexer
    assert type(indexer) is idx.Glm5NextKpoolIndexer
    assert type(indexer.k_cache) is k_cache_class
    # `rebind` mutates `k_cache.head_dim` in place; the pooled cache has no
    # such attribute to begin with, and its geometry must be untouched.
    assert indexer.k_cache.geometry.record_bytes == pooled_record_bytes


def test_the_hazard_the_guard_prevents_is_real(monkeypatch):
    """Not a restatement: this shows ``rebind`` really would swallow it.

    Without this, "the guard is an isinstance" is a claim about style. With
    it, removing the guard is a demonstrated class swap.
    """
    from vllm_torchtpu.layers.vllm.custom_ops.mla_attention_op import \
        VllmTPUIndexer

    indexer = _build_indexer(monkeypatch, prefix="model.layers.7.self_attn")
    # `rebind`'s own assert compares quant_block_size to head_dim; the kpool
    # indexer has no quant_block_size at all, so give it the value that makes
    # the assert pass and the class swap therefore reachable.
    indexer.quant_block_size = indexer.head_dim
    indexer.k_cache.head_dim = indexer.head_dim

    VllmTPUIndexer.rebind(indexer)

    assert type(indexer) is VllmTPUIndexer, (
        "rebind refused this object, so the guard would be unnecessary -- if "
        "that is now true, simplify the guard instead of keeping a test that "
        "no longer describes a hazard")


# --------------------------------------------------------------------------
# fixtures for a real Glm5NextKpoolIndexer
# --------------------------------------------------------------------------


def _text_config(**overrides):
    base = dict(
        hidden_size=256,
        q_lora_rank=64,
        index_n_heads=4,
        index_head_dim=REAL_INDEX_HEAD_DIM,
        index_topk=64,
        index_kpool=REAL_INDEX_KPOOL,
        index_kpool_always_select_tail=True,
        qk_rope_head_dim=0,
        num_hidden_layers=8,
        layer_types=["linear_attention"] * 8,
        indexer_types=["full"] * 8,
        index_share_for_mtp_iteration=True,
    )
    base.update(overrides)
    # Layers 3 and 7 are DSA, mirroring the real 3:1 schedule.
    for i in (3, 7):
        base["layer_types"][i] = "full_attention"
    return SimpleNamespace(**base)


def _vllm_config(**overrides):
    fields = dict(
        block_size=GLM_BLOCK_SIZE,
        enable_prefix_caching=False,
        max_model_len=2048,
        max_num_batched_tokens=1024,
        max_num_seqs=8,
        max_concurrent_batches=1,
        num_speculative_tokens=0,
    )
    fields.update(overrides)
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=fields["block_size"],
            enable_prefix_caching=fields["enable_prefix_caching"]),
        model_config=SimpleNamespace(max_model_len=fields["max_model_len"]),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=fields["max_num_batched_tokens"],
            max_num_seqs=fields["max_num_seqs"]),
        compilation_config=SimpleNamespace(static_forward_context={}),
        max_concurrent_batches=fields["max_concurrent_batches"],
        num_speculative_tokens=fields["num_speculative_tokens"],
    )


def _build_indexer(monkeypatch,
                   *,
                   prefix: str = "model.layers.3.self_attn",
                   text_config=None,
                   vllm_config=None,
                   **kwargs):
    """A fully wired ``Glm5NextKpoolIndexer``, pooled cache and ring included."""
    import vllm.config

    vllm_config = vllm_config if vllm_config is not None else _vllm_config()
    monkeypatch.setattr(vllm.config, "get_current_vllm_config",
                        lambda: vllm_config)
    return idx.Glm5NextKpoolIndexer(
        text_config if text_config is not None else _text_config(),
        None,
        prefix=f"{prefix}.indexer",
        layer_idx=3,
        vllm_config=vllm_config,
        **kwargs,
    )


# --------------------------------------------------------------------------
# --- the two ring tensors
# --------------------------------------------------------------------------


def test_the_ring_is_allocated_at_construction(monkeypatch):
    """Construction, not ``initialize_kv_cache``.

    No ``KVCacheSpec`` accounts for these two tensors -- that is the whole
    point of them, their size is flat in ``max_model_len`` -- so the only
    thing that keeps them out of the block pool's budget is being resident
    before ``determine_available_memory()`` profiles.
    """
    indexer = _build_indexer(monkeypatch)
    geo = indexer.pending_geometry
    assert tuple(indexer.pending_kv.shape) == geo.kv_shape
    assert tuple(indexer.pending_tag.shape) == geo.tag_shape
    assert indexer.pending_kv.dtype is torch.float32
    assert indexer.pending_tag.dtype is torch.int32
    assert torch.all(indexer.pending_tag == PENDING_TAG_EMPTY)


def test_the_ring_is_state_not_weights(monkeypatch):
    """A persistent buffer would appear in every loader report as a tensor
    the checkpoint does not have."""
    indexer = _build_indexer(monkeypatch)
    names = set(indexer.state_dict())
    assert not [n for n in names if "pending" in n], sorted(names)
    # ...but it still moves with the module, which a plain attribute would
    # not: that is the reason for `register_buffer` over `self.pending_kv =`.
    assert "pending_kv" in dict(indexer.named_buffers())
    assert "pending_tag" in dict(indexer.named_buffers())


@pytest.mark.parametrize("num_spec", [0, 1, 2, 3, 4, 7])
def test_the_ring_depth_is_derived_from_the_spec_config(monkeypatch, num_spec):
    """Never a constant. A depth of ``index_kpool`` is green at exactly the
    ``num_speculative_tokens=1`` the MTP probe defaults to and wrong from 2
    up (step-11 spike: 228 wrong pooled records over 270 schedules at K=2).
    """
    config = _vllm_config(num_speculative_tokens=num_spec)
    indexer = _build_indexer(monkeypatch, vllm_config=config)
    expected = derive_pending_state_geometry(
        index_kpool=REAL_INDEX_KPOOL,
        index_head_dim=REAL_INDEX_HEAD_DIM,
        max_num_reqs=config.scheduler_config.max_num_seqs,
        max_concurrent_batches=config.max_concurrent_batches,
        num_speculative_tokens=num_spec,
    )
    assert indexer.pending_geometry == expected
    assert indexer.pending_geometry.pending_depth == REAL_INDEX_KPOOL + num_spec
    assert tuple(indexer.pending_kv.shape)[1] == REAL_INDEX_KPOOL + num_spec


def test_async_scheduling_widens_the_ring(monkeypatch):
    """``max_concurrent_batches`` is 2 under async scheduling, and it
    multiplies the rollback window rather than adding to it."""
    indexer = _build_indexer(monkeypatch,
                             vllm_config=_vllm_config(
                                 max_concurrent_batches=2,
                                 num_speculative_tokens=3))
    assert indexer.pending_geometry.pending_depth == REAL_INDEX_KPOOL + 6


def test_the_ring_size_does_not_move_with_max_model_len(monkeypatch):
    """The property that makes it worth living outside the KV allocator."""
    short = _build_indexer(monkeypatch, vllm_config=_vllm_config())
    long = _build_indexer(monkeypatch,
                          prefix="model.layers.7.self_attn",
                          vllm_config=_vllm_config(
                              max_model_len=131072,
                              max_num_batched_tokens=1024))
    assert (short.pending_geometry.total_bytes ==
            long.pending_geometry.total_bytes)


def test_the_mtp_ring_is_gated_by_the_index_reuse_interlock(monkeypatch):
    """One constant gates both halves; today the answer is always "allocate".

    Reading ``index_share_for_mtp_iteration`` directly instead would give the
    draft head's layer no ring while its indexer still ran.
    """
    text_config = _text_config()
    assert idx.KPOOL_MTP_INDEX_REUSE is False
    assert idx.mtp_indexer_needs_pending_state(text_config) is True

    monkeypatch.setattr(idx, "KPOOL_MTP_INDEX_REUSE", True)
    # `index_share_for_mtp_iteration` is True in the real checkpoint, so once
    # reuse exists layer 45's indexer never runs and its ring is dead weight.
    assert idx.mtp_indexer_needs_pending_state(text_config) is False
    assert idx.mtp_indexer_needs_pending_state(
        _text_config(index_share_for_mtp_iteration=False)) is True


def test_no_ring_means_forward_refuses_rather_than_reading_none(monkeypatch):
    indexer = _build_indexer(monkeypatch, allocate_pending_state=False)
    assert indexer.pending_kv is None
    assert indexer.pending_geometry is None
    with pytest.raises(ValueError, match="allocate_pending_state=False"):
        indexer(torch.zeros(2, 256), torch.zeros(2, 64), torch.zeros(2))


def test_forward_refuses_without_the_runner_handoff(monkeypatch):
    """`kpool_state_indices` has no defensible fallback.

    ``mamba_state_indices`` is the tempting one and is an out-of-bounds read
    under unified-align mode, where it holds an attention-pool block id rather
    than a slot in this module's ``[max_num_reqs + 1, ...]`` ring.
    """
    import vllm.forward_context

    from vllm_torchtpu.layers.common.attention_metadata import \
        AttentionMetadata

    indexer = _build_indexer(monkeypatch)
    metadata = AttentionMetadata.__new__(AttentionMetadata)
    # The field now EXISTS (the runner half landed), so the interesting case
    # is no longer "the dataclass has no such attribute" -- it is "the
    # attribute is there and the runner did not fill it", which is what a
    # model with a kpool indexer and `_has_kpool_state` false would produce.
    # `None` is the value both paths present, and it must still refuse.
    assert getattr(metadata, "kpool_state_indices", "absent") is None
    monkeypatch.setattr(
        vllm.forward_context, "get_forward_context", lambda: SimpleNamespace(
            attn_metadata={indexer.k_cache.prefix: metadata}))
    indexer.k_cache.kv_cache = torch.zeros(4, 1, 4, 256, dtype=torch.uint8)
    with pytest.raises(ValueError, match="kpool_state_indices"):
        indexer(torch.zeros(2, 256), torch.zeros(2, 64),
                torch.zeros(2, dtype=torch.int32))


def test_the_weights_only_indexer_still_builds_and_refuses_to_run():
    """The step-6/step-7 construction, kept working and kept unusable."""
    indexer = idx.Glm5NextKpoolIndexer(_text_config(),
                                       None,
                                       prefix="p",
                                       layer_idx=3)
    assert indexer.k_cache is None
    assert indexer.pending_geometry is None
    with pytest.raises(ValueError, match="weights-only"):
        indexer(torch.zeros(2, 256), torch.zeros(2, 64), torch.zeros(2))


def test_the_indexer_refuses_a_rotary_embedding(monkeypatch):
    indexer = _build_indexer(monkeypatch)
    with pytest.raises(ValueError, match="rotary"):
        indexer(torch.zeros(2, 256), torch.zeros(2, 64), torch.zeros(2),
                object())


# --------------------------------------------------------------------------
# --- the shared topk_indices buffer
# --------------------------------------------------------------------------


def test_one_topk_buffer_for_the_whole_engine(monkeypatch):
    """Twelve private copies at 8192 tokens is ~850 MB nothing accounts for."""
    config = _vllm_config()
    a = _build_indexer(monkeypatch, vllm_config=config)
    b = _build_indexer(monkeypatch,
                       prefix="model.layers.7.self_attn",
                       vllm_config=config)
    assert a.topk_indices_buffer is b.topk_indices_buffer


def test_the_topk_buffer_is_out_width_not_index_topk(monkeypatch):
    """``index_topk`` is short by ``index_kpool - 1`` -- the tail columns."""
    indexer = _build_indexer(monkeypatch)
    width = indexer.topk_indices_buffer.shape[-1]
    assert width == derive_out_width(64, REAL_INDEX_KPOOL)
    assert width > 64
    assert indexer.topk_tokens == width
    assert torch.all(indexer.topk_indices_buffer == -1)


def test_two_widths_in_one_engine_are_refused():
    config = _vllm_config()
    idx.shared_topk_indices_buffer(config, out_width=128)
    with pytest.raises(ValueError, match="two different topk_indices buffer"):
        idx.shared_topk_indices_buffer(config, out_width=2176)


# --------------------------------------------------------------------------
# the indexer-aware context bound
# --------------------------------------------------------------------------


def _bound_cfg(max_len):
    return SimpleNamespace(model_config=SimpleNamespace(max_model_len=max_len))


def test_the_bound_still_refuses_a_dense_build_past_index_topk():
    with pytest.raises(ValueError, match="kpool indexer is NOT built"):
        _check_context_within_dense_mla_bound(
            _bound_cfg(8192),
            SimpleNamespace(index_topk=REAL_INDEX_TOPK),
            indexer_enabled=False)


def test_the_bound_allows_long_context_once_the_indexer_is_built():
    _check_context_within_dense_mla_bound(
        _bound_cfg(1_048_576),
        SimpleNamespace(index_topk=REAL_INDEX_TOPK),
        indexer_enabled=True)


def test_the_bound_is_not_simply_deleted():
    """The kill switch and an unsupported config both land in the same place.

    A deleted guard would make `VLLM_TORCHTPU_GLM5_KPOOL=0` at 128 K a run
    that produces fluent, wrong text with nothing to say so.
    """
    with pytest.raises(ValueError, match="fluent-but-wrong"):
        _check_context_within_dense_mla_bound(
            _bound_cfg(131072),
            SimpleNamespace(index_topk=REAL_INDEX_TOPK),
            indexer_enabled=False)


def test_the_bound_reads_the_env_when_not_told(monkeypatch):
    """The guard and the layers must consult the same resolver."""
    monkeypatch.setenv(attn.KPOOL_INDEXER_ENV, "0")
    with pytest.raises(ValueError, match="kpool indexer is NOT built"):
        _check_context_within_dense_mla_bound(
            _bound_cfg(4096), SimpleNamespace(index_topk=REAL_INDEX_TOPK))
    monkeypatch.setenv(attn.KPOOL_INDEXER_ENV, "1")
    _check_context_within_dense_mla_bound(
        _bound_cfg(4096), SimpleNamespace(index_topk=REAL_INDEX_TOPK))


def test_the_escape_hatch_still_works_with_the_indexer_off(monkeypatch):
    monkeypatch.setenv(_ALLOW_UNSOUND_CONTEXT, "1")
    _check_context_within_dense_mla_bound(
        _bound_cfg(1_048_576),
        SimpleNamespace(index_topk=REAL_INDEX_TOPK),
        indexer_enabled=False)


def test_a_mixed_stack_is_refused():
    """Six sparse layers and five dense ones is a legal-looking model that
    retrieves on half its full-attention layers."""
    with pytest.raises(ValueError, match="stack is mixed"):
        assert_indexer_enablement_uniform(
            {
                "layers.3": True,
                "layers.7": True,
                "layers.11": False
            }, True)


def test_the_guard_and_the_layers_must_agree():
    with pytest.raises(ValueError, match="disagree"):
        assert_indexer_enablement_uniform(
            {
                "layers.3": False,
                "layers.7": False
            }, True)
    assert_indexer_enablement_uniform({
        "layers.3": True,
        "layers.7": True
    }, True)
    assert_indexer_enablement_uniform({}, False)


# --------------------------------------------------------------------------
# the score-buffer budget
# --------------------------------------------------------------------------


def _geometry(max_model_len, block_size=GLM_BLOCK_SIZE):
    return idx.derive_pooled_indexer_geometry(
        block_size=block_size,
        index_kpool=REAL_INDEX_KPOOL,
        index_head_dim=REAL_INDEX_HEAD_DIM,
        max_model_len=max_model_len,
    )


@pytest.mark.parametrize("max_model_len", [2048, 131072, 1048576])
def test_the_buffer_is_one_fp32_column_per_pool_of_capacity(max_model_len):
    """``max_model_len`` bytes per query token at ratio 4, and the width does
    not shrink for a short request."""
    geometry = _geometry(max_model_len)
    assert geometry.score_columns == max_model_len // REAL_INDEX_KPOOL
    assert idx.score_buffer_bytes(
        max_num_batched_tokens=1,
        score_columns=geometry.score_columns) == max_model_len


def test_128k_at_8192_tokens_fits_and_1m_at_8192_does_not():
    """The two interesting configurations, as the bound's two ends."""
    idx.assert_score_buffer_within_budget(max_model_len=131072,
                                          max_num_batched_tokens=8192,
                                          geometry=_geometry(131072),
                                          num_indexer_layers=12)
    with pytest.raises(ValueError) as excinfo:
        idx.assert_score_buffer_within_budget(max_model_len=1048576,
                                              max_num_batched_tokens=8192,
                                              geometry=_geometry(1048576),
                                              num_indexer_layers=12)
    message = str(excinfo.value)
    # The three numbers the message has to carry, plus the knob.
    assert "max_model_len=1048576" in message
    assert "max_num_batched_tokens=8192" in message
    assert str(8192 * (1048576 // 4) * 4) in message
    assert "Lower max_num_batched_tokens" in message


def test_1m_context_is_reachable_at_a_smaller_chunk():
    """The guard must name a knob that actually works, not just refuse."""
    description = idx.assert_score_buffer_within_budget(
        max_model_len=1048576,
        max_num_batched_tokens=1024,
        geometry=_geometry(1048576),
        num_indexer_layers=12)
    assert "(1.000 GiB) per indexer call" in description
    # And the headline figure, so the log line a
    # human reads carries the number they will compare against HBM.
    assert "12.000 GiB summed over 12 indexer layers" in description


def test_the_sparsecore_envelope_is_named_at_1m():
    """``1M / 4 == 262144`` is exactly the documented ceiling, and lowering
    the chunk size does not move it."""
    with pytest.raises(ValueError, match="sparsecore_topk"):
        idx.assert_score_buffer_within_budget(max_model_len=1048576,
                                              max_num_batched_tokens=8192,
                                              geometry=_geometry(1048576),
                                              num_indexer_layers=12)
    assert (
        _geometry(1048576).score_columns == idx.SPARSECORE_TOPK_MAX_COLUMNS)


def test_the_budget_is_overridable_and_disableable(monkeypatch):
    monkeypatch.setenv(idx.SCORE_BUFFER_BUDGET_ENV, "16384")
    idx.assert_score_buffer_within_budget(max_model_len=1048576,
                                          max_num_batched_tokens=8192,
                                          geometry=_geometry(1048576),
                                          num_indexer_layers=12)
    monkeypatch.setenv(idx.SCORE_BUFFER_BUDGET_ENV, "0")
    assert idx.score_buffer_budget_bytes() is None
    monkeypatch.setenv(idx.SCORE_BUFFER_BUDGET_ENV, "lots")
    with pytest.raises(ValueError, match="not an integer"):
        idx.score_buffer_budget_bytes()


def test_the_guard_runs_at_indexer_construction(monkeypatch):
    """Not merely available -- wired, and before anything is allocated."""
    with pytest.raises(ValueError, match="score buffer does not fit"):
        _build_indexer(monkeypatch,
                       vllm_config=_vllm_config(max_model_len=1048576,
                                                max_num_batched_tokens=8192))


# --------------------------------------------------------------------------
# prefix caching
# --------------------------------------------------------------------------


def test_prefix_caching_is_refused_and_the_message_names_the_pooled_cache():
    with pytest.raises(ValueError) as excinfo:
        idx.assert_prefix_caching_disabled(
            SimpleNamespace(enable_prefix_caching=True), prefix="layers.3")
    message = str(excinfo.value)
    assert "POOLED INDEXER K CACHE" in message
    # The mechanism, so a reader does not go looking for a NaN.
    assert "scores exactly 0" in message


def test_prefix_caching_off_is_allowed():
    idx.assert_prefix_caching_disabled(
        SimpleNamespace(enable_prefix_caching=False), prefix="layers.3")
    idx.assert_prefix_caching_disabled(None, prefix="layers.3")


def test_prefix_caching_is_refused_at_indexer_construction(monkeypatch):
    """tpu_platform gates on ``mamba_cache_mode``, not on the indexer, so the
    configuration is reachable and nothing else refuses it."""
    with pytest.raises(ValueError, match="prefix caching is enabled"):
        _build_indexer(monkeypatch,
                       vllm_config=_vllm_config(enable_prefix_caching=True))


def test_nonzero_diagnostics_raise_when_checked():
    from vllm_torchtpu.kernels.glm5_kpool import KPOOL_WRITE_DIAG_SIZE

    clean = torch.zeros(KPOOL_WRITE_DIAG_SIZE, dtype=torch.int32)
    idx.assert_kpool_diagnostics_clean(clean, prefix="layers.3")

    dirty = clean.clone()
    dirty[0] = 7
    with pytest.raises(ValueError, match="tag_violations=7"):
        idx.assert_kpool_diagnostics_clean(dirty, prefix="layers.3")


def test_the_diagnostics_gate_is_fixed_at_construction(monkeypatch):
    """It must not be re-decided per step; see KPOOL_CHECK_DIAGNOSTICS_ENV.

    The flag gates a host sync inside `forward`. A value that can change
    between two steps changes the program traced for those steps, across
    every DSA layer at once -- and the bounded-budget version this replaced
    spent its whole budget inside warmup, so it only ever checked
    compilation shapes.
    """
    monkeypatch.setenv(idx.KPOOL_CHECK_DIAGNOSTICS_ENV, "1")
    indexer = _build_indexer(monkeypatch)
    assert indexer._check_diagnostics is True

    monkeypatch.setenv(idx.KPOOL_CHECK_DIAGNOSTICS_ENV, "0")
    assert indexer._check_diagnostics is True, (
        "the gate was re-read after construction")


def test_the_diagnostics_gate_is_off_by_default(monkeypatch):
    monkeypatch.delenv(idx.KPOOL_CHECK_DIAGNOSTICS_ENV, raising=False)
    assert _build_indexer(monkeypatch)._check_diagnostics is False


def test_the_weights_only_indexer_has_the_gate_attribute():
    """`forward` reads it, so it must exist even on the path that refuses."""
    indexer = idx.Glm5NextKpoolIndexer(_text_config(),
                                       None,
                                       prefix="model.layers.3.self_attn"
                                       ".indexer",
                                       layer_idx=3,
                                       vllm_config=None)
    assert indexer._check_diagnostics is False
