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
"""Torch-side unit tests for the fused expert-parallel MoE bridge.

**Everything the kernel decides, it decides here, at weight load.** vLLM
compiles the backbone once with every Dynamo guard dropped
(``TorchCompileWithNoGuardsWrapper``), so a branch inside the traced forward is
answered once and that answer is the program for every token bucket. A layer
therefore either arms -- and then serves every step from the smallest decode
bucket to the largest prefill one -- or it declines, with a reason. These tests
pin the arming decision and its inputs; ``test_fused_moe_ep_engagement.py`` pins
what the compiler then receives.
"""

from types import SimpleNamespace

import pytest
import torch
import vllm.config as vllm_config_mod

from vllm_torchtpu import envs
from vllm_torchtpu.layers.vllm import fused_moe_ep

# Qwen3.5-397B's MoE geometry, which every measured number in this file is at.
TOP_K = 10
NUM_EXPERTS = 512
EP_SIZE = 8


def _layer(**overrides):
    moe_config = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=True),
        ep_size=EP_SIZE,
        dp_size=8,
        num_experts=NUM_EXPERTS,
        experts_per_token=TOP_K,
        is_sequence_parallel=False,
        pcp_size=1,
        skip_final_all_reduce=False,
        is_lora_enabled=False,
    )
    layer = SimpleNamespace(
        moe_config=moe_config,
        renormalize=True,
        scoring_func="softmax",
        _tpu_activation_str="silu",
        has_bias=False,
    )
    for key, value in overrides.items():
        target = moe_config if hasattr(moe_config, key) else layer
        setattr(target, key, value)
    return layer


@pytest.fixture
def tp1(monkeypatch):
    """The DP8_EP layout: tokens are not replicated across a TP group."""
    monkeypatch.setattr(fused_moe_ep, "_replica_group", lambda: (1, 0))


@pytest.fixture
def engine_cap(monkeypatch):
    """Set the scheduler cap the band reads, the way a booted engine would.

    ``max_engine_node_tokens`` goes through ``get_current_vllm_config_or_none``,
    so this patches the source rather than the answer: the multiplication by
    ``dp_size`` -- the step from a per-engine cap to the node-wide count every
    measurement is indexed by -- stays under test.
    """

    def _set(max_num_batched_tokens):
        config = (None if max_num_batched_tokens is None else SimpleNamespace(
            scheduler_config=SimpleNamespace(
                max_num_batched_tokens=max_num_batched_tokens)))
        monkeypatch.setattr(vllm_config_mod, "get_current_vllm_config_or_none",
                            lambda: config)

    return _set


# Everything this file drives through the environment. `vllm_torchtpu.envs`
# serves settings through module `__getattr__`, so a sibling test's
# `monkeypatch.setattr(envs, ...)` leaves a real attribute behind that makes
# every later `setenv` of that name silently invisible -- and the shared
# conftest fixture that used to sweep those up is not on this branch. Each file
# that uses `setenv` therefore clears its own names.
_ENV_NAMES = (
    "USE_MOE_FUSED_EP_V2",
    "USE_MOE_EP_KERNEL",
    "MOE_FUSED_EP_KERNEL_MIN_TOKENS",
    "MOE_FUSED_EP_KERNEL_MAX_TOKENS",
    "MOE_FUSED_EP_V2_CAPACITY_SKEW",
)


@pytest.fixture(autouse=True)
def clean_arming_state():
    for name in _ENV_NAMES:
        if name in vars(envs):
            delattr(envs, name)
    fused_moe_ep.reset_engagement_state()
    yield
    fused_moe_ep.reset_engagement_state()


@pytest.fixture
def armable(monkeypatch, tp1, engine_cap):
    """Flag on, mesh accepted, and an engine whose largest batch is in band."""
    monkeypatch.setenv("USE_MOE_FUSED_EP_V2", "1")
    monkeypatch.setattr(fused_moe_ep, "validate_ep_mesh",
                        lambda layer, **kwargs: None)
    engine_cap(1024)  # x dp_size=8 -> 8192 node-wide tokens
    return engine_cap


# --------------------------------------------------------------------------
# The band, which is now one question about the engine rather than one per step
# --------------------------------------------------------------------------


def test_the_band_defaults_are_the_measured_ones(monkeypatch):
    """The floor is the crossover, and it is not a recipe's number.

    Whole-MoE-layer microseconds at this geometry, EP=8, one clock, against the
    production GMM path: the fused kernel is **610 us against 509** at 1024
    node-wide tokens and 471 against 430 at 256 -- it *loses* through that
    band -- and only crosses over at 2048 (607 against 692), reaching 1.71x by
    8192. The old default of 1024 armed a deployment precisely where the
    kernel is slowest, which nothing about a per-step threshold could have
    rescued: an armed layer takes every step.
    """
    monkeypatch.delenv("MOE_FUSED_EP_KERNEL_MIN_TOKENS", raising=False)
    monkeypatch.delenv("MOE_FUSED_EP_KERNEL_MAX_TOKENS", raising=False)
    assert envs.MOE_FUSED_EP_KERNEL_MIN_TOKENS == 2048
    # And a cap by default, because the default slab is the drop-free one: the
    # sizing that removes the cliff needs a drop path that does not work yet
    # (`test_fused_ep_v2_slab_overflow.py`), so the cap it justified stays.
    assert fused_moe_ep.max_tokens() == 8192


@pytest.mark.parametrize(
    "cap,dp_size,armed",
    [
        (2048, 8, False),  # 16384 node-wide: above the cap the drop-free slab
        #                    keeps; armable only with a sized slab, which is
        #                    opt-in until its drop path drops
        (1024, 8, True),  # 8192 node-wide: the top of the band as shipped
        (256, 8, True),  # 2048 node-wide: exactly the crossover
        (248, 8, False),  # 1984 node-wide: below it
        (255, 8, True),  # 2048 too: the rows are padded to 256 before the
        #                  kernel sees them, and the bound is on what the
        #                  kernel is handed -- see the SMEM section
        (2048, 1, False),  # dp=1 is refused on its own terms, whatever the cap
        (16, 8, False),  # a decode-only engine: 128 node-wide
    ])
def test_the_band_is_answered_from_the_engine_cap_times_dp(
        tp1, engine_cap, monkeypatch, cap, dp_size, armed):
    """``max_num_batched_tokens * dp_size``, asked once, of the whole engine.

    The scheduler's cap is the largest ``x.shape[0]`` an MoE layer can ever see
    -- the runner pads every step to a bucket no larger than it -- and
    ``dp_size`` turns that into the node-wide count the kernel's cost is a
    function of. Both factors are config, identical on every rank, and readable
    before the first forward: that is what makes the decision static.
    """
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_MIN_TOKENS", "2048")
    engine_cap(cap)
    assert (fused_moe_ep.out_of_band_reason(_layer(dp_size=dp_size))
            is None) is armed


def test_the_refusal_names_the_engine_batch_and_the_band(
        tp1, engine_cap, monkeypatch):
    """A decline has to be actionable from a boot log, without the source.

    It names the number that was computed, the two factors it came from, the
    band edge it missed, and -- because this is the part an operator cannot
    guess -- that there is no per-step choice to fall back on.
    """
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_MIN_TOKENS", "2048")
    engine_cap(64)
    reason = fused_moe_ep.out_of_band_reason(_layer(dp_size=8))
    assert "512 node-wide tokens" in reason  # 64 x 8, the computed batch
    assert "dp_size=8" in reason
    assert "max_num_batched_tokens" in reason  # the knob that moves it
    assert "2048" in reason  # the edge it missed
    assert "guards" in reason  # why there is no per-step fallback


@pytest.mark.parametrize(
    "node_tokens,in_band",
    [
        (512, False),  # below the floor
        (2048, True),  # v2's win band opens
        (8192, True),  # 1.71x of the GMM path, measured
        (16384, False),  # 4.79x of it on the drop-free slab -- the cliff
    ])
def test_the_band_is_a_band_not_a_floor(monkeypatch, node_tokens, in_band):
    """v2 is slow at both ends, so one threshold cannot express the policy."""
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_MIN_TOKENS", "2048")
    monkeypatch.delenv("MOE_FUSED_EP_KERNEL_MAX_TOKENS", raising=False)
    monkeypatch.setenv("MOE_FUSED_EP_V2_CAPACITY_SKEW", "0")  # the cliff is on
    assert fused_moe_ep.in_token_band(node_tokens) is in_band


def test_the_token_cap_can_be_lifted(monkeypatch):
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_MAX_TOKENS", "0")
    monkeypatch.setenv("MOE_FUSED_EP_V2_CAPACITY_SKEW", "0")
    assert fused_moe_ep.max_tokens() == 0
    assert fused_moe_ep.in_token_band(16384)


def test_sizing_the_slab_lifts_the_token_cap(monkeypatch):
    """The cap and the slab are the same fact seen twice.

    The 8192-token cap goes with the drop-free slab, which is the default. A
    deployment that opts into a sized slab has taken on the cliff's cause and
    should not also have to lift the cap by hand -- but it has also taken on a
    drop path that writes out of bounds when a shard overflows
    (`test_fused_ep_v2_slab_overflow.py`), which is why it is opt-in.
    """
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_MIN_TOKENS", "2048")
    monkeypatch.delenv("MOE_FUSED_EP_KERNEL_MAX_TOKENS", raising=False)
    monkeypatch.delenv("MOE_FUSED_EP_V2_CAPACITY_SKEW", raising=False)

    assert fused_moe_ep.max_tokens() == 8192
    assert not fused_moe_ep.in_token_band(16384)

    monkeypatch.setenv("MOE_FUSED_EP_V2_CAPACITY_SKEW", "3")
    assert fused_moe_ep.capacity_skew() == 3.0
    assert fused_moe_ep.max_tokens() == 0
    assert fused_moe_ep.in_token_band(16384)

    # ...and an explicit cap still wins over both.
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_MAX_TOKENS", "4096")
    assert fused_moe_ep.max_tokens() == 4096


def test_the_slab_is_drop_free_by_default(monkeypatch):
    """The default is OFF, because the drop path a sized slab needs is broken.

    It writes past the end of the buffer when a shard overflows -- an
    out-of-bounds DMA, which on a chip is a core halt. The sizing is kept
    because it is what makes 16384 affordable, but it is opt-in until the drop
    drops.
    """
    monkeypatch.delenv("MOE_FUSED_EP_V2_CAPACITY_SKEW", raising=False)
    assert fused_moe_ep.capacity_skew() == 0.0
    monkeypatch.setenv("MOE_FUSED_EP_V2_CAPACITY_SKEW", "3")
    assert fused_moe_ep.capacity_skew() == 3.0


def test_an_unreadable_engine_cap_arms_on_the_flag_alone(
        tp1, engine_cap, monkeypatch):
    """No config to read is not a reason to refuse; being wrong costs speed."""
    warned = []
    monkeypatch.setattr(fused_moe_ep.logger, "warning_once",
                        lambda *a, **k: warned.append(a))
    engine_cap(None)
    assert fused_moe_ep.out_of_band_reason(_layer(dp_size=8)) is None
    assert warned and "unreadable" in warned[0][0]


# --------------------------------------------------------------------------
# The layouts, which are refused for reasons the band cannot express
# --------------------------------------------------------------------------


def test_supported_layout_is_accepted(tp1):
    assert fused_moe_ep.unsupported_reason(_layer()) is None


@pytest.mark.parametrize("overrides,expected", [
    ({
        "moe_parallel_config": SimpleNamespace(use_ep=False)
    }, "expert parallelism is off"),
    ({
        "is_sequence_parallel": True
    }, "sequence parallelism"),
    ({
        "pcp_size": 2
    }, "PCP"),
    ({
        "use_grouped_topk": True
    }, "grouped top-k"),
    ({
        "e_score_correction_bias": torch.zeros(4)
    }, "bias-corrected"),
    ({
        "routed_scaling_factor": 2.5
    }, "routed_scaling_factor"),
    ({
        "scoring_func": "sqrtsoftplus"
    }, "scoring_func"),
    ({
        "has_bias": True
    }, "expert biases"),
    ({
        "num_experts": 500
    }, "divisible"),
    ({
        "dp_size": 4
    }, "not dp_size"),
    ({
        "skip_final_all_reduce": True
    }, "skip_final_all_reduce"),
    ({
        "is_lora_enabled": True
    }, "LoRA"),
])
def test_unsupported_layouts_are_refused_with_a_reason(tp1, overrides,
                                                       expected):
    reason = fused_moe_ep.unsupported_reason(_layer(**overrides))
    assert reason is not None and expected in reason


def test_the_kernel_refuses_what_it_cannot_route(tp1):
    """One program now, so scoring and activation are properties of it.

    The kernel softmaxes internally and fuses two activations; there is no
    second version to fall through to since the token-stationary v1 was cut.
    """
    layer = _layer()
    layer.scoring_func = "sigmoid"
    assert "scoring_func" in fused_moe_ep.unsupported_reason(layer)
    layer.scoring_func = "softmax"
    layer._tpu_activation_str = "gelu"
    assert "activation" in fused_moe_ep.unsupported_reason(layer)
    for supported in ("silu", "swigluoai"):
        layer._tpu_activation_str = supported
        assert fused_moe_ep.unsupported_reason(layer) is None


def test_a_tp_group_is_refused_until_it_is_measured(monkeypatch):
    """DP4TP2 has coherent token ownership and is still refused, deliberately.

    ``ep_size == dp_size * tp_size`` makes the row slice/scatter well defined,
    which is why the ownership check below passes -- but the sliced path has
    never been measured for this kernel, and an armed layer now serves every
    step rather than the large ones only. The narrower refusal comes first.
    """
    monkeypatch.setattr(fused_moe_ep, "_replica_group", lambda: (2, 1))
    reason = fused_moe_ep.unsupported_reason(_layer(dp_size=4))
    assert reason is not None and "tp_size=2" in reason
    # The ownership check is still the one that catches an incoherent layout,
    # and it must fire before the tp refusal so the message names the cause.
    assert "not dp_size" in fused_moe_ep.unsupported_reason(_layer(dp_size=8))


def test_dp1_is_refused_because_there_is_no_transport_to_remove(
        tp1, engine_cap, monkeypatch):
    """TP8_EP: the kernel's whole win is deleting the DP dispatch/combine pair.

    With ``dp_size == 1`` there is no such pair, and the comparison changes
    sides: production's kernel alone measures 290 us against v2's 299 at 64
    node-wide tokens and 1747 against 1907 at 16384. Under a per-step
    threshold this layout looked merely small; under arming it is a deployment
    we must turn away, because there is no token count at which it wins.
    """
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_MIN_TOKENS", "2048")
    engine_cap(16384)  # far inside the band -- the size is not the objection
    reason = fused_moe_ep.out_of_band_reason(_layer(dp_size=1))
    assert reason is not None and "dp_size=1" in reason


# --------------------------------------------------------------------------
# The SMEM ceiling: a hard limit, checked against the same engine batch
# --------------------------------------------------------------------------


def _kernel_takes(node_tokens, skew, moe):
    """Would the kernel itself accept this many node-wide tokens?

    An oracle for the gate, composed the other way round: the slab formula and
    the budget come from the kernel's own module, and the **padding is applied
    by the caller**. The gate has to reach the same answer with the padding
    applied inside it, so a gate that forgets to pad disagrees with this and
    the test says which shape it would have armed.

    ``fused_moe_ep_v2._check_smem`` is the shipping spelling of this, but it
    reads ``pltpu.get_tpu_info()`` and raises "Unsupported TPU device kind:
    cpu" on this host, so the CPU gate composes the same two facts by hand.
    """
    from vllm_torchtpu.layers.common.fused_moe_ep_v2 import (
        _SMEM_OVERHEAD_BYTES, _slab_rows)

    budget = fused_moe_ep._smem_capacity_bytes() - _SMEM_OVERHEAD_BYTES
    rows = _slab_rows(node_tokens, moe.experts_per_token, moe.num_experts,
                      moe.ep_size, skew)
    return rows * 4 <= budget


@pytest.mark.parametrize("skew,last_ok", [(0, 19801), (3.0, 52322)])
def test_the_smem_ceiling_is_refused_from_both_sides(tp1, monkeypatch, skew,
                                                     last_ok):
    """The kernel's plan is scalar-prefetch, so SMEM caps the batch, not VMEM.

    One int32 per drop-free slab row lands in a 1 MiB scalar memory with a
    ~236 kB overhead, which at top-10 over 512 experts puts the ceiling at
    **19,801** node-wide tokens on the drop-free slab and **52,322** once the
    slab is sized (skew 3.0, ep 8). Both edges are asserted because the value
    is a formula, not a constant: a change to ROWBLK, the tile height or the
    overhead moves it, and the failure it replaces is Mosaic's "Ran out of
    memory in memory space smem" at the first compile of a serving run --
    which names neither the token count nor the setting that controls it.
    """
    monkeypatch.setenv("MOE_FUSED_EP_V2_CAPACITY_SKEW", str(skew))
    moe = _layer(dp_size=8).moe_config

    monkeypatch.setattr(fused_moe_ep,
                        "max_engine_node_tokens",
                        lambda dp, tp=1: last_ok)
    assert fused_moe_ep._smem_reason(moe) is None
    assert _kernel_takes(last_ok, skew,
                         moe), "the oracle disagrees at the edge"

    monkeypatch.setattr(fused_moe_ep,
                        "max_engine_node_tokens",
                        lambda dp, tp=1: last_ok + 1)
    reason = fused_moe_ep._smem_reason(moe)
    assert reason is not None
    assert f"{last_ok + 1} node-wide tokens" in reason
    assert "SMEM" in reason and "max-num-batched-tokens" in reason
    assert not _kernel_takes(last_ok + 1, skew, moe)


# The largest `--max-num-batched-tokens` that still fits, and the first that
# does not, per slab setting -- at dp=8, tp=1, top-10 over 512 experts. These
# are *caps*, not node counts: `_padded_rows` rounds each replica's rows up to a
# multiple of 8 first, so 2473 and 6537 are the first caps whose PADDED batch
# (19840 and 52352) crosses a ceiling their unpadded product (19784, 52296)
# is still under.
SMEM_CAP_EDGES = [(0, 2472, 2473), (3.0, 6536, 6537)]


@pytest.mark.parametrize("skew,last_ok,first_bad", SMEM_CAP_EDGES)
def test_the_bound_is_on_the_rows_the_kernel_is_handed(tp1, engine_cap,
                                                       monkeypatch, skew,
                                                       last_ok, first_bad):
    """The quantity bounded must be the one the kernel receives.

    The scheduler's cap is not that quantity: this bridge pads each replica's
    slice up to ``_MIN_LOCAL_ROWS`` before the kernel sees it, so the kernel is
    handed ``dp * _padded_rows(cap, tp)``. Bounding ``cap * dp`` instead admits
    a deployment whose largest bucket the kernel then refuses to compile --
    arming, and dying at the first compile of a serving run rather than at the
    refusal that exists to prevent exactly that.
    """
    monkeypatch.setenv("MOE_FUSED_EP_V2_CAPACITY_SKEW", str(skew))
    layer = _layer(dp_size=8)

    engine_cap(last_ok)
    assert fused_moe_ep.unsupported_reason(layer) is None

    engine_cap(first_bad)
    reason = fused_moe_ep.unsupported_reason(layer)
    assert reason is not None and "SMEM" in reason
    # ...and the number it names is the padded one, not the product.
    padded = fused_moe_ep._padded_rows(first_bad, 1) * 8
    assert f"{padded} node-wide tokens" in reason
    assert first_bad * 8 < padded, "pick a cap that the padding actually moves"


@pytest.mark.parametrize("skew,window", [(0, range(2473, 2476)),
                                         (3.0, range(6537, 6541))])
def test_the_caps_that_armed_and_then_died_are_refused(tp1, engine_cap,
                                                       monkeypatch, skew,
                                                       window):
    """The exact deployments the unpadded bound let through.

    Every cap in this window has ``cap * dp`` inside the ceiling and
    ``dp * _padded_rows(cap, tp)`` outside it, so the old bound armed the layer
    and the kernel then raised ``ValueError`` at the first compile of the
    largest bucket -- after weight load, on a chip, with the boot lost. At the
    shipped slab that is ``--max-num-batched-tokens`` 6537 to 6540.
    """
    monkeypatch.setenv("MOE_FUSED_EP_V2_CAPACITY_SKEW", str(skew))
    layer = _layer(dp_size=8)
    for cap in window:
        assert cap * 8 <= (19801 if skew == 0 else 52322), (
            f"{cap} is outside the window the unpadded bound admitted")
        engine_cap(cap)
        reason = fused_moe_ep.unsupported_reason(layer)
        assert reason is not None and "SMEM" in reason, (
            f"cap {cap} armed; the kernel is handed "
            f"{fused_moe_ep._padded_rows(cap, 1) * 8} node-wide tokens and "
            "refuses to compile them")


@pytest.mark.parametrize("skew", [0, 3.0])
def test_arming_implies_the_kernel_can_take_the_largest_bucket(
        tp1, engine_cap, monkeypatch, skew):
    """The invariant the gate exists for, checked against the kernel's own rule.

    Swept across both edges rather than asserted at one point, because the two
    sides compose the padding differently: the gate pads inside
    ``max_engine_node_tokens`` and the oracle pads in the test. A gate that
    drops the padding, or applies it to the wrong factor, disagrees here.
    """
    monkeypatch.setenv("MOE_FUSED_EP_V2_CAPACITY_SKEW", str(skew))
    layer = _layer(dp_size=8)
    edges = {0: (2464, 2480), 3.0: (6528, 6544)}[skew]
    for cap in range(edges[0], edges[1] + 1):
        engine_cap(cap)
        armed = fused_moe_ep.unsupported_reason(layer) is None
        handed = fused_moe_ep._padded_rows(cap, 1) * 8
        assert armed == _kernel_takes(handed, skew, layer.moe_config), (
            f"cap {cap}: gate says armed={armed}, but the kernel is handed "
            f"{handed} node-wide tokens")


# --------------------------------------------------------------------------
# Arming
# --------------------------------------------------------------------------


def test_flag_defaults_off(tp1, monkeypatch):
    monkeypatch.delenv("USE_MOE_FUSED_EP_V2", raising=False)
    assert envs.USE_MOE_FUSED_EP_V2 is False
    assert fused_moe_ep.is_enabled(_layer()) is False
    assert fused_moe_ep.fused_path_is_live() is False


def test_both_fused_kernel_flags_at_once_is_refused(armable, monkeypatch):
    """Two kernels, one layer, and the second one to look would win silently.

    They disagree about the weight layout, the collective id and the output
    contract, and each would claim this layer's. Which one actually served
    would depend on import order, so it is refused by name instead.
    """
    monkeypatch.setenv("USE_MOE_EP_KERNEL", "1")
    with pytest.raises(ValueError, match="both set"):
        fused_moe_ep.is_enabled(_layer())


def test_an_armed_layer_is_recorded_once(armable):
    """Arming is the process-wide record the boot report and the ladder read."""
    assert fused_moe_ep.is_enabled(_layer()) is True
    assert fused_moe_ep.fused_path_is_live() is True


def test_a_declined_layer_is_not_recorded(armable, engine_cap):
    """A refusal must not leave the run looking armed: the rendezvous and the
    report both key on this, and a barrier entered by one rank alone hangs."""
    engine_cap(16)  # 128 node-wide tokens, below the floor
    assert fused_moe_ep.is_enabled(_layer()) is False
    assert fused_moe_ep.fused_path_is_live() is False


def test_routing_simulation_disables_the_kernel(tp1, monkeypatch):
    monkeypatch.setenv("VLLM_MOE_ROUTING_SIMULATION_STRATEGY",
                       "uniform_random")
    assert "SIMULATION" in fused_moe_ep.unsupported_reason(_layer())


def test_refusal_checks_read_attributes_that_actually_exist():
    """The refusal path degrades silently, so pin the names it reads.

    Every check is a ``getattr`` with a default: if upstream renames one, the
    kernel would quietly stop arming -- or quietly arm on a configuration it
    cannot serve -- instead of failing. The scheduler field is on the list
    because the band now reads it, and it is the one input that comes from
    outside the MoE config.
    """
    from vllm.config import SchedulerConfig
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig, FusedMoEParallelConfig)

    for name in ("num_experts", "experts_per_token", "has_bias", "ep_size",
                 "dp_size", "pcp_size", "is_sequence_parallel", "ep_rank",
                 "moe_parallel_config", "skip_final_all_reduce",
                 "is_lora_enabled"):
        assert (name in FusedMoEConfig.__annotations__ or isinstance(
            getattr(FusedMoEConfig, name, None), property)), name
    assert "use_ep" in FusedMoEParallelConfig.__annotations__
    assert "max_num_batched_tokens" in SchedulerConfig.__annotations__


# --------------------------------------------------------------------------
# The row slice/scatter around the kernel, which arming does not change
# --------------------------------------------------------------------------


@pytest.mark.parametrize("replica_size", [2, 4, 8])
def test_row_slices_partition_the_batch(replica_size):
    """Every row is owned by exactly one replica, and summing rebuilds it.

    That sum is what vLLM's TP all-reduce performs. Kept although
    ``unsupported_reason`` refuses ``tp_size > 1`` today: the refusal is a
    measurement gap, and this is the arithmetic that will have to hold when it
    is closed.
    """
    rows, hidden = 4 * replica_size, 3
    x = torch.arange(rows * hidden, dtype=torch.float32).reshape(rows, hidden)

    total = torch.zeros_like(x)
    for rank in range(replica_size):
        rank_t = torch.tensor(rank, dtype=torch.int32)
        index = fused_moe_ep._row_slice_index(rows, replica_size, rank_t)
        local = torch.index_select(x, 0, index)
        assert local.shape == (rows // replica_size, hidden)
        scattered = fused_moe_ep._scatter_replica_rows(local, replica_size,
                                                       rank_t)
        assert scattered.shape == x.shape
        total += scattered

    torch.testing.assert_close(total, x)


def test_replica_rank_reaches_the_graph_as_data(tp1):
    """The row slice must be indexed by a tensor, not a Python int.

    A Python int would be folded into the traced program as a constant, so
    every rank would compile a structurally different program under one op
    name -- the 2026-07-30 EP desync (see moe_routing.register_experts_start_buffer).
    """
    rank = torch.tensor(1, dtype=torch.int32)
    index = fused_moe_ep._row_slice_index(8, 2, rank)
    assert isinstance(index, torch.Tensor)
    assert index.dtype == torch.int32
    torch.testing.assert_close(index,
                               torch.tensor([4, 5, 6, 7], dtype=torch.int32))


def test_register_replica_rank_buffer_is_none_without_tp(tp1):
    layer = torch.nn.Module()
    fused_moe_ep.register_replica_rank_buffer(layer,
                                              device=torch.device("cpu"))
    assert layer._ep_replica_rank is None


def test_register_replica_rank_buffer_holds_the_rank(monkeypatch):
    monkeypatch.setattr(fused_moe_ep, "_replica_group", lambda: (2, 1))
    layer = torch.nn.Module()
    fused_moe_ep.register_replica_rank_buffer(layer,
                                              device=torch.device("cpu"))
    assert layer._ep_replica_rank.item() == 1
    assert layer._ep_replica_rank.dtype == torch.int32
    assert "_ep_replica_rank" not in layer.state_dict()


# `block_size_override` and its `MOE_FUSED_EP_KERNEL_BLOCKS` env are gone from
# the branch -- the value was parsed and then swallowed by `_v2_partial`, so it
# tuned nothing -- and the four cases that pinned the parser went with them.
# Nothing else covered that function; the coverage is deliberately dropped, not
# relocated.


def test_an_engine_above_the_cap_declines_and_names_both_numbers(
        tp1, engine_cap, monkeypatch):
    """The shipped behaviour at our own benchmark shape, pinned.

    With the drop-free slab the token cap comes back, so a DP8 engine at
    2048 tokens/rank -- 16,384 node-wide -- does not arm. That is correct: the
    sized slab that would make it profitable overflows into an out-of-bounds
    DMA (`test_fused_ep_v2_slab_overflow.py`). A reviewer meets this decline
    before anything else, so it has to say what happened and which knob moves
    it, not merely that something was refused.
    """
    monkeypatch.delenv("MOE_FUSED_EP_V2_CAPACITY_SKEW", raising=False)
    monkeypatch.delenv("MOE_FUSED_EP_KERNEL_MAX_TOKENS", raising=False)
    engine_cap(2048)  # x dp_size=8 -> 16,384 node-wide tokens
    reason = fused_moe_ep.out_of_band_reason(_layer())

    assert reason is not None
    assert "16384" in reason and "8192" in reason, reason
    assert "max-num-batched-tokens" in reason, reason
