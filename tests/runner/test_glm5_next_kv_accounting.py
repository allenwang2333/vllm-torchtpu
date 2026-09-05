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
"""HBM and pagination accounting for GLM-5.3-Flash.

**What this file cannot do, and why.** The natural place to check KV-cache
accounting is after ``initialize_kv_cache``. Nothing can call that on the
CPU rung:
``vllm_torchtpu.runner.tpu_runner`` does not import here at all. It reaches
``layers/vllm/quantization/nvfp4.py:50``, which imports
``ModelOptNvFp4LinearMethod`` from ``vllm.model_executor.layers.quantization
.modelopt`` -- a name the developer checkout of vLLM on ``PYTHONPATH`` no
longer defines. That is pre-existing rebase drift (the same drift that makes
``tests/runner/test_tpu_runner.py`` uncollectable and ``test_kimi_k3.py`` red),
it is not this change's, and it is not this change's to fix.

So the two checks are split:

* the **arithmetic** -- which is all of (a) and, as shown below, all of (b)
  that is not a monotone function of it -- runs here for real, against the
  production functions (``PallasMLAttentionBackend.get_kv_cache_page_size_bytes``,
  ``get_sparse_kv_cache_specs``,
  ``Glm5NextForCausalLM.get_mamba_state_shape_from_config``) and the
  production accounting helpers the runner calls
  (``models/vllm/glm5_next/indexer.py``);
* the **wiring** -- that ``initialize_kv_cache`` really calls the check, and
  what vLLM's grouping does with a pooled spec whose page size differs from
  the uniform page -- remains a device claim, unreachable from here.

Two suspected overruns are recorded here as *numbers*, because measuring them
turned both into non-findings:

1. A suspected sparse-split shortfall (``10240`` allocated against a
   budgeted ``16 * 576 = 9216``) **does not exist**. The TPU backend does not
   budget ``block_size * head_size``: it pads the minor axis to a 128 multiple
   and accounts for dtype packing, so at ``head_size=576`` / fp8 the budget is
   ``10240`` and the split allocation is also ``10240``. Exactly equal.
2. A suspected three-way divergence is **half right**. ``head_size``
   640 -> 576 does **not** move ``attn_page_size_bytes`` at all, for the same
   padding reason (both round to a 640-wide minor axis). Only bf16 -> fp8
   moves it, and it moves it by exactly 2x.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.layers.vllm.attention import PallasMLAttentionBackend
from vllm_torchtpu.models.vllm.glm5_next.indexer import (
    assert_kv_cache_allocation_within_plan, derive_pooled_indexer_geometry,
    kv_cache_allocation_report, measure_allocated_cache_bytes)

# --- the deployment's numbers, recorded ------------------------------------
#
# Read from `zai-org/GLM-5.3-Flash` snapshot `03eb5366`'s `config.json`, not
# derived: 45 text layers, 11 DSA (`layer_types[i] != "linear_attention"` at
# 3, 7, ..., 43) and 34 KDA.
GLM_NUM_DSA_LAYERS = 11
GLM_NUM_KDA_LAYERS = 34
GLM_LINEAR_ATTN_CONFIG = {
    "num_heads": 64,
    "head_dim": 128,
    "short_conv_kernel_size": 4,
}
GLM_KV_LORA_RANK = 512
#: `models/vllm/glm5_next/attention.py:56`. The dense workaround for
#: `qk_rope_head_dim == 0`; the sparse path takes it to 64.
ZERO_ROPE_SLAB_TODAY = 128
ZERO_ROPE_SLAB_SPARSE = 64
GLM_BLOCK_SIZE = 16
GLM_INDEX_KPOOL = 4
GLM_INDEX_HEAD_DIM = 128
TP_SIZE = 8


def _mla_head_size(rope_slab: int) -> int:
    """`MLAAttention.head_size == kv_lora_rank + qk_rope_head_dim`."""
    return GLM_KV_LORA_RANK + rope_slab


def _fake_vllm_config(tp_size: int = TP_SIZE):
    """Just enough for the two mamba-state classmethods, which read little."""
    text = SimpleNamespace(linear_attn_config=dict(GLM_LINEAR_ATTN_CONFIG))
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(text_config=text), dtype=torch.bfloat16),
        cache_config=SimpleNamespace(mamba_cache_dtype="auto"),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp_size),
    )


def _mamba_unpadded_bytes(tp_size: int = TP_SIZE) -> int:
    """`unpadded_mamba_page_size` as `tpu_runner.py:1798-1801` computes it."""
    from vllm_torchtpu.models.vllm.glm5_next.model import Glm5NextForCausalLM
    config = _fake_vllm_config(tp_size)
    shapes = Glm5NextForCausalLM.get_mamba_state_shape_from_config(config)
    dtypes = Glm5NextForCausalLM.get_mamba_state_dtype_from_config(config)
    return sum(
        math.prod(shape) * dtype.itemsize
        for shape, dtype in zip(shapes, dtypes))


def _group_counts(num_attn: int,
                  num_mamba: int,
                  num_pooled: int = 0) -> tuple[int, int, int, int]:
    """vLLM's group-size heuristic, as `_update_mamba_page_size_padded`
    duplicates it.

    Transcribed a third time on purpose and *pinned* by
    `test_group_counts_depend_only_on_layer_counts`: the claim this
    turns on is that these numbers do not move with `head_size` or KV dtype,
    and that is only visible if the inputs are visible.

    The pooled indexer caches are a **third bucket**. vLLM's own
    `_get_kv_cache_groups_uniform_page_size` buckets by *spec*, so 11 MLA + 34
    mamba + 11 pooled is min=11 / max=34 and six groups, not five -- and every
    `KVCacheTensor` is then `shared_by` one layer from each. Passing
    `num_pooled=0` reproduces the old two-bucket arithmetic exactly, which is
    what every model without a pooled cache still gets.

    Returns `(num_attn_groups, num_mamba_groups, num_pooled_groups,
    group_size)`.
    """
    counts = [num_attn, num_mamba] + ([num_pooled] if num_pooled else [])
    min_count = min(counts)
    max_count = max(counts)
    group_size = max_count if max_count < min_count * 1.5 else min_count
    return (-(-num_attn // group_size), -(-num_mamba // group_size),
            -(-num_pooled // group_size), group_size)


def _attn_page_bytes(head_size: int, dtype: torch.dtype) -> int:
    return PallasMLAttentionBackend.get_kv_cache_page_size_bytes(
        GLM_BLOCK_SIZE, 1, head_size, dtype)


def _sparse_split_bytes_per_block(head_size: int, dtype: torch.dtype) -> int:
    """What `tpu_runner.py:6517-6535` really allocates for one sparse layer."""
    specs = PallasMLAttentionBackend.get_sparse_kv_cache_specs(
        1, GLM_BLOCK_SIZE, head_size, dtype)
    # The runner allocates `torch.zeros(spec.shape, dtype=torch.uint8)`, so
    # bytes are `prod(shape)` with `num_blocks == 1` here.
    return sum(math.prod(spec.shape) for spec in specs)


# --- (a) planned vs allocated ---------------------------------------------


def _block_tensor(bytes_per_block: int, num_blocks: int) -> torch.Tensor:
    return torch.zeros((num_blocks, bytes_per_block), dtype=torch.uint8)


def test_measure_allocated_cache_bytes_handles_all_three_cache_shapes():
    """Single tensor, tuple of tensors, and the 4-D packed pooled layout."""
    single = measure_allocated_cache_bytes("mla", _block_tensor(20480, 7))
    assert (single.num_blocks, single.bytes_per_block) == (7, 20480)

    pair = measure_allocated_cache_bytes("sparse", (torch.zeros(
        (7, 16, 4, 128),
        dtype=torch.uint8), torch.zeros((7, 4, 4, 128), dtype=torch.uint8)))
    assert pair.bytes_per_block == 8192 + 2048 == 10240

    geom = derive_pooled_indexer_geometry(
        block_size=GLM_BLOCK_SIZE,
        index_kpool=GLM_INDEX_KPOOL,
        index_head_dim=GLM_INDEX_HEAD_DIM,
        max_model_len=131072,
    )
    pooled = measure_allocated_cache_bytes(
        "pooled", torch.zeros(geom.cache_shape(7), dtype=torch.uint8))
    assert pooled.bytes_per_block == geom.page_bytes == 1024


def test_measure_refuses_caches_that_disagree_about_the_block_count():
    """A shorter tensor in the tuple means clipped block ids, not a crash."""
    with pytest.raises(ValueError, match="disagree about the block count"):
        measure_allocated_cache_bytes("bad", (torch.zeros(
            (7, 16), dtype=torch.uint8), torch.zeros(
                (5, 16), dtype=torch.uint8)))


def test_pooled_cache_outside_every_kv_cache_tensor_is_named_in_the_report():
    """The failure mode the report exists for: bytes nobody budgeted.

    `uniform_page_size_bytes` counts attention and mamba groups only, so a
    pooled cache that ends up in no `KVCacheTensor` is real HBM that vLLM's
    `num_blocks` arithmetic never saw. It must at minimum be *visible*.
    """
    kv_caches = {
        "l3.mla_attn": _block_tensor(20480, 100),
        "l3.indexer.k_cache": _block_tensor(1024, 100),
    }
    report, overruns = kv_cache_allocation_report(
        kv_caches=kv_caches,
        tensor_groups=[["l3.mla_attn"]],
        planned_page_size_bytes=20480,
    )
    print(report)
    assert not overruns
    assert "NOT IN ANY KVCacheTensor: l3.indexer.k_cache" in report


def test_pooled_cache_inside_a_full_group_is_an_overrun_and_raises():
    """A budget without the pooled term breaks; the widened one does not.

    Both halves matter. The first is the bug: `uniform_page_size_bytes` used to
    be two terms, so a pooled cache inside the group was a *guaranteed*
    overrun, and this check is fail-closed -- GLM-5.3 would have refused to
    boot inside `initialize_kv_cache`. The second is the fix: with the third
    term, the same allocation is exactly on budget rather than over it.
    """
    mamba_unpadded = _mamba_unpadded_bytes()
    attn_page = _attn_page_bytes(_mla_head_size(ZERO_ROPE_SLAB_TODAY),
                                 torch.bfloat16)
    geom = derive_pooled_indexer_geometry(
        block_size=GLM_BLOCK_SIZE,
        index_kpool=GLM_INDEX_KPOOL,
        index_head_dim=GLM_INDEX_HEAD_DIM,
        max_model_len=131072,
    )
    (num_attn_groups, num_mamba_groups, num_pooled_groups,
     _) = _group_counts(GLM_NUM_DSA_LAYERS, GLM_NUM_KDA_LAYERS,
                        GLM_NUM_DSA_LAYERS)
    two_term = num_attn_groups * attn_page + num_mamba_groups * mamba_unpadded
    three_term = two_term + num_pooled_groups * geom.page_bytes

    shared_by = ["l3.mla_attn"] + [f"l{i}.linear_attn" for i in range(4)]
    kv_caches = {"l3.mla_attn": _block_tensor(attn_page, 100)}
    for name in shared_by[1:]:
        kv_caches[name] = _block_tensor(mamba_unpadded, 100)
    # Exactly on budget before the pooled cache joins.
    assert_kv_cache_allocation_within_plan(
        kv_caches=kv_caches,
        tensor_groups=[shared_by],
        planned_page_size_bytes=two_term,
    )

    kv_caches["l3.indexer.k_cache"] = _block_tensor(geom.page_bytes, 100)
    with pytest.raises(ValueError) as excinfo:
        assert_kv_cache_allocation_within_plan(
            kv_caches=kv_caches,
            tensor_groups=[shared_by + ["l3.indexer.k_cache"]],
            planned_page_size_bytes=two_term,
        )
    assert "exceeds the page size vLLM budgeted" in str(excinfo.value)
    print(str(excinfo.value))

    # ... and the widened budget accommodates exactly the same allocation.
    print(f"two-term budget={two_term} three-term budget={three_term} "
          f"(+{three_term - two_term} = num_pooled_groups="
          f"{num_pooled_groups} x pooled_page={geom.page_bytes})")
    assert_kv_cache_allocation_within_plan(
        kv_caches=kv_caches,
        tensor_groups=[shared_by + ["l3.indexer.k_cache"]],
        planned_page_size_bytes=three_term,
    )


def test_the_whole_allocation_comparison_sees_what_the_per_block_one_cannot():
    """(c): total bytes on device vs total bytes vLLM handed out.

    The per-group comparison is per *block* against the uniform page, and in
    the layout GLM actually gets the uniform page **is** the per-group sum by
    construction -- so it can only ever flag a group that is essentially the
    whole model. A wrong `num_blocks` moves no per-block number at all. The
    absolute comparison is the independent one.
    """
    shared_by = ["l3.mla_attn", "l0.linear_attn", "l3.indexer.k_cache"]
    per_block = {
        "l3.mla_attn": 20480,
        "l0.linear_attn": 561152,
        "l3.indexer.k_cache": 1024
    }
    planned_page = sum(per_block.values())

    # On budget per block, and on budget in total.
    kv_caches = {n: _block_tensor(b, 100) for n, b in per_block.items()}
    budget = [100 * planned_page]
    report = assert_kv_cache_allocation_within_plan(
        kv_caches=kv_caches,
        tensor_groups=[shared_by],
        planned_page_size_bytes=planned_page,
        tensor_sizes=budget,
    )
    print(report)
    assert "WHOLE ALLOCATION" in report
    assert "delta=+0 B" in report

    # Now allocate 110 blocks per layer instead of 100. Every per-block number
    # is unchanged, so the per-group check stays silent -- and 10% more HBM is
    # on the device than vLLM handed out.
    over = {n: _block_tensor(b, 110) for n, b in per_block.items()}
    _, overruns = kv_cache_allocation_report(
        kv_caches=over,
        tensor_groups=[shared_by],
        planned_page_size_bytes=planned_page,
        tensor_sizes=budget,
    )
    assert not overruns, (
        "the per-block view must be blind to this, or the test is not showing "
        "what it claims to show")
    with pytest.raises(ValueError, match="more than the .* B vLLM"):
        assert_kv_cache_allocation_within_plan(
            kv_caches=over,
            tensor_groups=[shared_by],
            planned_page_size_bytes=planned_page,
            tensor_sizes=budget,
        )

    # Allocating *less* is legitimate: compact-mamba caps mamba at
    # `max_num_reqs + 1` slots, so the total is routinely under budget.
    under = dict(kv_caches)
    under["l0.linear_attn"] = _block_tensor(per_block["l0.linear_attn"], 33)
    report = assert_kv_cache_allocation_within_plan(
        kv_caches=under,
        tensor_groups=[shared_by],
        planned_page_size_bytes=planned_page,
        tensor_sizes=budget,
    )
    assert "delta=-" in report


def test_the_sparse_split_cache_is_not_over_budget_after_all():
    """The suspected `10240 vs 9216` shortfall does not exist.

    The suspicion was that the sparse split cache's real per-block bytes
    (`nope 16x512 + rope 16x128 = 10240`) exceed "a budgeted `16 x 576 =
    9216`". They do not, because nothing budgets `block_size * head_size`.
    `PallasMLAttentionBackend.get_kv_cache_page_size_bytes` goes through
    `get_kv_cache_shape`, which pads the minor axis to a 128 multiple
    (576 -> 640) and packs by dtype. The two numbers come out equal.

    Asserting the equality rather than the suspected inequality, with both
    numbers printed, so a future change to either side is caught.
    """
    for slab, dtype in ((ZERO_ROPE_SLAB_SPARSE, torch.float8_e4m3fn),
                        (ZERO_ROPE_SLAB_SPARSE, torch.bfloat16)):
        head_size = _mla_head_size(slab)
        budget = _attn_page_bytes(head_size, dtype)
        allocated = _sparse_split_bytes_per_block(head_size, dtype)
        print(f"sparse split, head_size={head_size} dtype={dtype}: "
              f"budgeted={budget} allocated={allocated} "
              f"delta={allocated - budget:+d}")
        assert allocated <= budget, (
            "the sparse (nope, rope) split now costs more per block than the "
            "single-tensor page vLLM budgets from; the suspected OOM "
            f"scenario has become real (budget {budget}, allocated "
            f"{allocated})")
    # The concrete production case, pinned.
    assert _sparse_split_bytes_per_block(576, torch.float8_e4m3fn) == 10240
    assert _attn_page_bytes(576, torch.float8_e4m3fn) == 10240


# --- (b) three-way KDA invariance -----------------------------------------

#: (label, mla head_size, kv cache dtype) for the three configurations step
#: 10(b) names: today, fp8 KV only, fp8 + the 64-wide rope slab.
THREE_WAY = (
    ("a: today (dense slab 128, bf16 KV)",
     _mla_head_size(ZERO_ROPE_SLAB_TODAY), torch.bfloat16),
    ("b: fp8 KV only", _mla_head_size(ZERO_ROPE_SLAB_TODAY),
     torch.float8_e4m3fn),
    ("c: fp8 KV + rope slab 64", _mla_head_size(ZERO_ROPE_SLAB_SPARSE),
     torch.float8_e4m3fn),
)


def test_group_counts_depend_only_on_layer_counts():
    """The half of `_hybrid_uniform_page_size_bytes` that provably cannot move.

    `uniform = num_attn_groups * attn_page + num_mamba_groups *
    mamba_unpadded`. The two multipliers are a function of the *layer counts*
    alone (`tpu_runner.py:1828-1841`), so neither the rope slab nor the KV
    dtype can touch them -- which is what makes the single-number comparison
    below sufficient rather than a proxy.
    """
    num_attn_groups, num_mamba_groups, _, group_size = _group_counts(
        GLM_NUM_DSA_LAYERS, GLM_NUM_KDA_LAYERS)
    assert (group_size, num_attn_groups, num_mamba_groups) == (11, 1, 4)

    # And adding the third (pooled) bucket does not move them at GLM's shape:
    # the pooled count is 11, which is already the minimum, so min/max and
    # therefore `group_size` are unchanged. It contributes one more group.
    (pooled_attn, pooled_mamba, pooled_groups,
     pooled_size) = _group_counts(GLM_NUM_DSA_LAYERS, GLM_NUM_KDA_LAYERS,
                                  GLM_NUM_DSA_LAYERS)
    assert (pooled_size, pooled_attn, pooled_mamba, pooled_groups) == (11, 1,
                                                                       4, 1)


def test_kda_state_shape_and_dtype_are_independent_of_both_knobs():
    """The "KDA layers untouched" claim, at the only place it is true.

    `Glm5NextForCausalLM.get_mamba_state_shape_from_config` reads
    `linear_attn_config` and `tensor_parallel_size`; the dtype hook reads
    `model_config.dtype` and `cache_config.mamba_cache_dtype`. Neither reads
    the MLA `head_size` or `cache_config.cache_dtype`, so the KDA cache
    *shape* really is invariant across the three configurations -- it is the
    page **padding** written onto it that is not.
    """
    from vllm_torchtpu.models.vllm.glm5_next.model import Glm5NextForCausalLM
    config = _fake_vllm_config()
    shapes = Glm5NextForCausalLM.get_mamba_state_shape_from_config(config)
    dtypes = Glm5NextForCausalLM.get_mamba_state_dtype_from_config(config)
    assert shapes == ((3, 3, 8, 128), (8, 128, 128))
    assert dtypes == (torch.float32, torch.float32)
    assert _mamba_unpadded_bytes() == 36864 + 524288 == 561152


def test_three_way_page_size_table_is_recorded_not_wished_away():
    """FINDING: only fp8 moves the page. The rope slab does not.

    The second half of that suspicion was that *both* numerics changes move
    `attn_page_size_bytes`. Measured against the real backend at
    `block_size=16`:

    ==========================  =========  ==========
    configuration               attn_page  uniform
    ==========================  =========  ==========
    a: slab 128, bf16 KV            20480     2265088
    b: slab 128, fp8 KV             10240     2254848
    c: slab  64, fp8 KV             10240     2254848
    ==========================  =========  ==========

    (b) and (c) are byte-identical, because
    `PallasMLAttentionBackend.get_kv_cache_shape` pads the minor axis to a
    128 multiple and `cdiv(576, 128) * 128 == cdiv(640, 128) * 128 == 640`.
    So the rope slab change is *not* a repagination at this block size, and
    an indexer-off/indexer-on A/B differs only by the KV dtype.

    (a) -> (b) is exactly 2x on the attention term, which is 10240 B/block on
    the uniform page: 0.452% of it, because 4 mamba groups at 561152 B
    dominate. `num_gpu_blocks_override` and `_mamba_num_blocks` are monotone
    in that number and therefore move by about the same fraction -- that part
    cannot be measured without a device, since it needs the HBM budget.

    **Unchanged by the pooled term.** `uniform` here is the attention+mamba
    part on purpose: the pooled term is a *constant* 1024 B/block across all
    three configurations (`geometry.page_bytes` reads `block_size`,
    `index_kpool` and `index_head_dim`, none of which is a knob in this
    table), so it shifts every row by the same amount and cancels out of every
    comparison below. See
    `test_the_pooled_term_is_now_the_third_term_of_the_uniform_page` for the
    three-term numbers.
    """
    mamba_unpadded = _mamba_unpadded_bytes()
    num_attn_groups, num_mamba_groups, _, _ = _group_counts(
        GLM_NUM_DSA_LAYERS, GLM_NUM_KDA_LAYERS)

    table = {}
    for label, head_size, dtype in THREE_WAY:
        attn_page = _attn_page_bytes(head_size, dtype)
        uniform = (num_attn_groups * attn_page +
                   num_mamba_groups * mamba_unpadded)
        table[label] = (attn_page, uniform)
        print(f"{label}: head_size={head_size} dtype={dtype} "
              f"attn_page={attn_page} uniform={uniform}")

    today = table["a: today (dense slab 128, bf16 KV)"]
    fp8_only = table["b: fp8 KV only"]
    fp8_slab64 = table["c: fp8 KV + rope slab 64"]

    assert today == (20480, 2265088)
    assert fp8_only == (10240, 2254848)
    # The finding: the rope slab is a no-op for pagination at block_size 16.
    assert fp8_slab64 == fp8_only, (
        "the 128 -> 64 rope slab now changes attn_page_size_bytes, which it "
        "did not when this was written. Re-derive the table in this "
        "docstring: an indexer-off/indexer-on A/B assumes the "
        f"two differ only by KV dtype. today={today} fp8={fp8_only} "
        f"fp8+slab64={fp8_slab64}")
    # ... and the half of that prediction which does hold.
    assert today != fp8_only, (
        "bf16 -> fp8 KV no longer changes the page size; the 'KDA layers "
        "untouched' claim would then be true and a dense baseline needs "
        "no re-measurement")
    assert today[0] == 2 * fp8_only[0]
    delta = today[1] - fp8_only[1]
    print(f"uniform page moves by {delta} B/block "
          f"({delta / today[1]:.3%} of the page); mamba_unpadded="
          f"{mamba_unpadded} B x {num_mamba_groups} groups dominates it")
    assert delta == 10240


def test_the_pooled_term_is_now_the_third_term_of_the_uniform_page():
    """The first half of the suspicion, and the numbers before and after.

    This test used to pin the *bug*: `uniform_page_size_bytes` counted
    attention and mamba groups only, so whatever the pooled caches cost was
    HBM no budget knew about. That was fail-closed against
    `assert_kv_cache_allocation_within_plan`, i.e. GLM-5.3 could not have
    booted with a pooled cache instantiated.

    The two-term budget was **not** short by the 11264 B/block this test used
    to print. That figure is 11 pooled caches x 1024, i.e. all eleven layers
    -- but vLLM puts one layer from each group into each `KVCacheTensor`, so
    the per-tensor shortfall is `num_pooled_groups x pooled_page = 1 x 1024`.
    Both numbers are printed below; the budget now carries the second.
    """
    geom = derive_pooled_indexer_geometry(
        block_size=GLM_BLOCK_SIZE,
        index_kpool=GLM_INDEX_KPOOL,
        index_head_dim=GLM_INDEX_HEAD_DIM,
        max_model_len=131072,
    )
    mamba_unpadded = _mamba_unpadded_bytes()
    (num_attn_groups, num_mamba_groups, num_pooled_groups,
     group_size) = _group_counts(GLM_NUM_DSA_LAYERS, GLM_NUM_KDA_LAYERS,
                                 GLM_NUM_DSA_LAYERS)
    attn_page = _attn_page_bytes(_mla_head_size(ZERO_ROPE_SLAB_SPARSE),
                                 torch.float8_e4m3fn)
    two_term = num_attn_groups * attn_page + num_mamba_groups * mamba_unpadded
    pooled_term = num_pooled_groups * geom.page_bytes
    three_term = two_term + pooled_term
    # All eleven caches, which is what the whole model costs per block --
    # *not* what one KVCacheTensor is short by.
    all_layers = GLM_NUM_DSA_LAYERS * geom.page_bytes
    print(f"uniform page (fp8 + slab 64): two-term={two_term} B/block, "
          f"pooled term = num_pooled_groups={num_pooled_groups} x "
          f"pooled_page={geom.page_bytes} = {pooled_term} B/block "
          f"({pooled_term / two_term:.4%}), three-term={three_term}; "
          f"all {GLM_NUM_DSA_LAYERS} pooled caches together are "
          f"{all_layers} B/block across {group_size} KVCacheTensors")
    assert (two_term, pooled_term, three_term) == (2254848, 1024, 2255872)
    assert all_layers == 11264 == group_size * pooled_term
