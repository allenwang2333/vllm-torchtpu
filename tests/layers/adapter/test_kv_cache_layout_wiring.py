"""Wiring of `VLLM_KV_CACHE_LAYOUT` onto the batched-RPA KV cache layout.

    unset / NHD -> HEAD_ALONG_SUBLANE
    HND         -> SEQ_ALONG_LANE

Only the mapping and its resolution seams are left here. The CUSTOM backend no
longer *allocates* a SEQ_ALONG_LANE pool: tokamax's operator models only
HEAD_ALONG_SUBLANE, so `PallasBatchedRPAAttentionBackend.get_kv_cache_shape`
went back to the inherited one and HND is refused when the kernel is built (see
`test_attention_tokamax_rpa.py`). The enum is still read by the DCP kernels and
by the runner, which is what these tests protect.
"""
import contextlib

import pytest
import torch
from vllm.config import get_current_vllm_config

from vllm_torchtpu.kernels.experimental.batched_rpa import \
    configs as batched_rpa_configs
from vllm_torchtpu.layers.adapter.attention import (
    KV_LAYOUT_BY_VLLM_LAYOUT, PallasAttentionBackend,
    PallasBatchedRPAAttentionBackend, _resolve_kv_layout)

BF16 = torch.bfloat16
FP8 = torch.float8_e4m3fn
KVLayout = batched_rpa_configs.KVLayout

BASE = PallasAttentionBackend
BATCHED = PallasBatchedRPAAttentionBackend

# (num_kv_heads, head_size); 1 KV head is the b/510425663 case (a TP=4 rank of
# a 4-KV-head model), 128 and 256 are the head widths in use on this backend.
HEAD_CASES = [(1, 128), (1, 256), (2, 128), (4, 128), (8, 128), (8, 256)]
HEAD_CASES_CORE = [(1, 128), (2, 128), (8, 256)]


@contextlib.contextmanager
def _layout_env(monkeypatch, value):
    """Resolve legacy env aliases through the native config API."""
    from vllm import envs as vllm_envs
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.attention import AttentionConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

    was_cached = vllm_envs._is_envs_cache_enabled()
    vllm_envs.disable_envs_cache()
    if value is None:
        monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    else:
        monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", value)
    config = VllmConfig(attention_config=AttentionConfig(
        backend=AttentionBackendEnum.CUSTOM))
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    try:
        with set_current_vllm_config(config):
            yield
    finally:
        monkeypatch.undo()
        if was_cached:
            vllm_envs.enable_envs_cache()


@pytest.fixture
def nhd(monkeypatch):
    """Default selection. Explicitly unset so a dirty environment can't pass."""
    with _layout_env(monkeypatch, None):
        yield


@pytest.fixture
def hnd(monkeypatch):
    with _layout_env(monkeypatch, "HND"):
        yield


def page_bytes(backend, num_kv_heads, head_size, dtype, block_size=128):
    return backend.get_kv_cache_page_size_bytes(block_size, num_kv_heads,
                                                head_size, dtype)


# Selection


@pytest.mark.parametrize("value,expected", [
    (None, KVLayout.HEAD_ALONG_SUBLANE),
    ("NHD", KVLayout.HEAD_ALONG_SUBLANE),
    ("HND", KVLayout.SEQ_ALONG_LANE),
])
def test_env_selects_layout(monkeypatch, value, expected):
    with _layout_env(monkeypatch, value):
        assert KV_LAYOUT_BY_VLLM_LAYOUT[get_current_vllm_config(
        ).cache_config.get_resolved_kv_cache_layout()] is expected


@pytest.mark.parametrize("value,expected", [
    (None, KVLayout.HEAD_ALONG_SUBLANE),
    ("NHD", KVLayout.HEAD_ALONG_SUBLANE),
    ("HND", KVLayout.SEQ_ALONG_LANE),
])
def test_resolve_reads_the_negotiated_config_layout(monkeypatch, value,
                                                    expected):
    """With a config in scope, `__init__` takes the per-backend layout the
    config negotiated rather than the raw process-wide one."""
    with _layout_env(monkeypatch, value):
        assert _resolve_kv_layout() is expected
        assert _resolve_kv_layout() is KV_LAYOUT_BY_VLLM_LAYOUT[
            get_current_vllm_config(
            ).cache_config.get_resolved_kv_cache_layout()]


@pytest.mark.parametrize("value,expected", [
    (None, KVLayout.HEAD_ALONG_SUBLANE),
    ("NHD", KVLayout.HEAD_ALONG_SUBLANE),
    ("HND", KVLayout.SEQ_ALONG_LANE),
])
def test_resolve_falls_back_to_the_env_override(monkeypatch, value, expected):
    """An impl constructed outside a `set_current_vllm_config()` scope has no
    backend to negotiate against, so it reads the same `VLLM_KV_CACHE_LAYOUT`
    the resolver reads -- legacy spelling included -- and otherwise takes the
    NHD default the pool is allocated in. There is no process-wide layout to
    fall back on any more: `get_kv_cache_layout()` went away with the refactor
    that introduced `vllm.v1.kv_cache_layout`."""
    from vllm import envs as vllm_envs

    was_cached = vllm_envs._is_envs_cache_enabled()
    vllm_envs.disable_envs_cache()
    if value is None:
        monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    else:
        monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", value)
    try:
        assert _resolve_kv_layout() is expected
    finally:
        monkeypatch.undo()
        if was_cached:
            vllm_envs.enable_envs_cache()


# The CUSTOM backend's pool is the inherited one, in every layout


@pytest.mark.parametrize("num_kv_heads,head_size", HEAD_CASES)
@pytest.mark.parametrize("dtype", [BF16, FP8])
@pytest.mark.parametrize("block_size", [16, 256])
def test_shape_matches_inherited(nhd, num_kv_heads, head_size, dtype,
                                 block_size):
    assert BATCHED.get_kv_cache_shape(7, block_size, num_kv_heads, head_size,
                                      dtype) == BASE.get_kv_cache_shape(
                                          7, block_size, num_kv_heads,
                                          head_size, dtype)


@pytest.mark.parametrize("num_kv_heads,head_size", HEAD_CASES)
@pytest.mark.parametrize("dtype", [BF16, FP8])
def test_hnd_no_longer_reshapes_the_pool(hnd, num_kv_heads, head_size, dtype):
    """The SEQ_ALONG_LANE override is gone. Setting HND cannot quietly change
    the allocation any more -- it is refused at kernel build time instead."""
    assert BATCHED.get_kv_cache_shape(7, 128, num_kv_heads, head_size,
                                      dtype) == BASE.get_kv_cache_shape(
                                          7, 128, num_kv_heads, head_size,
                                          dtype)


def test_block_size_is_the_single_tiled_page(nhd):
    assert BATCHED.get_supported_kernel_block_sizes() == [256]


def test_hnd_does_not_reopen_the_other_page_sizes(hnd):
    """`[128]` and the PCP-streaming list belonged to the in-tree kernels."""
    assert BATCHED.get_supported_kernel_block_sizes() == [256]


def test_supported_layouts_no_longer_offer_hnd():
    assert "LBHNC" not in {
        layout.name
        for layout in BATCHED.supported_kv_cache_layouts()
    }


# HND's own refusal lives with the rest of the tokamax guards, in
# test_attention_tokamax_rpa.py::test_build_rpa_kernel_rejects_hnd_layout.
# (There is no DCP counterpart any more: RPA has no context-parallel path left,
# so `cp_attention.py` and its tests went with the in-tree kernels.)

# Shape contracts the rest of the stack depends on


@pytest.mark.parametrize("dtype", [BF16, FP8])
def test_page_size_bytes_follows_the_allocated_shape(nhd, dtype):
    shape = BATCHED.get_kv_cache_shape(1, 128, 2, 128, dtype)
    numel = 1
    for dim in shape:
        numel *= dim
    itemsize = torch.empty((), dtype=dtype).element_size()
    assert page_bytes(BATCHED, 2, 128, dtype) == numel * itemsize


@pytest.mark.parametrize("dtype", [BF16, FP8])
def test_num_blocks_stays_at_dim_zero(nhd, dtype):
    """vLLM's offloading connector locates the block axis with
    `shape.index(num_blocks)`."""
    assert BATCHED.get_kv_cache_shape(4242, 128, 2, 128, dtype)[0] == 4242
    assert BATCHED.get_kv_cache_shape(4242, 128, 2, 128, "auto")[0] == 4242


def test_page_size_bytes_is_not_overridden():
    """Inherited on purpose: the base classmethod sizes itself from
    `cls.get_kv_cache_shape`."""
    assert "get_kv_cache_page_size_bytes" not in vars(BATCHED)


def test_shape_is_not_overridden_either():
    """Pins the removal: an override would have to re-derive the layout, which
    is what needed a live vLLM config in the scheduler process."""
    assert "get_kv_cache_shape" not in vars(BATCHED)


# Unified block pool


def test_unified_pool_accepts_seq_along_lane():
    """SEQ_ALONG_LANE can back the unified pool: the GDN op reshapes at its
    own boundary rather than needing token-contiguous pages. Pins the removal
    of `validate_kv_layout_supports_unified_pool`."""
    from vllm_torchtpu.platforms import tpu_block_size_utils
    assert not hasattr(tpu_block_size_utils,
                       "validate_kv_layout_supports_unified_pool")


# The page size the memory planner actually sees


def _normalized_page_bytes(backend, num_kv_heads, head_size, dtype):
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    from vllm_torchtpu.kv_cache_spec_normalizer import \
        normalize_kv_cache_specs_for_tpu
    spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        page_size_padded=page_bytes(backend, num_kv_heads, head_size, dtype),
    )
    normalized = normalize_kv_cache_specs_for_tpu({"l": spec},
                                                  dtype,
                                                  attention_backend=backend)
    return normalized["l"].page_size_bytes


@pytest.mark.parametrize("num_kv_heads,head_size", HEAD_CASES_CORE)
@pytest.mark.parametrize("dtype", [BF16, FP8])
def test_normalizer_agrees_with_the_allocated_page(nhd, num_kv_heads,
                                                   head_size, dtype):
    """The planner's per-block budget must equal what gets allocated, or blocks
    are counted against a page nobody builds."""
    assert (_normalized_page_bytes(BATCHED, num_kv_heads, head_size,
                                   dtype) == page_bytes(
                                       BATCHED, num_kv_heads, head_size,
                                       dtype))


@pytest.mark.parametrize("num_kv_heads,head_size", HEAD_CASES_CORE)
@pytest.mark.parametrize("dtype", [BF16, FP8])
def test_normalizer_matches_the_base_backend(nhd, num_kv_heads, head_size,
                                             dtype):
    assert (_normalized_page_bytes(BATCHED, num_kv_heads, head_size,
                                   dtype) == _normalized_page_bytes(
                                       BASE, num_kv_heads, head_size, dtype))


# Layout lookups on the forward path


def test_no_layout_lookup_on_the_compiled_forward_path():
    """Nothing on the forward path may resolve the KV layout.

    Resolving it needs a current vLLM config; a compiled forward has none, and
    Dynamo traces into the accessor rather than folding it away. Callers there
    must use the value `__init__` resolved.

    A source check, because reproducing it needs a real engine -- it surfaced
    only in CI, as `ObservedAssertionErrorError` raised inside the Dynamo
    region.
    """
    import inspect

    from vllm_torchtpu.layers.adapter import attention as attn

    on_forward_path = [
        attn.PallasAttentionBackendImpl.forward,
        attn.PallasAttentionBackendImpl._build_rpa_kernel,
        attn.PallasAttentionBackendImpl._tokamax_rpa_closure,
        attn.PallasAttentionBackendImpl._tokamax_rpa_kernel,
    ]
    offenders = [
        fn.__qualname__ for fn in on_forward_path
        if "get_resolved_kv_cache_layout" in inspect.getsource(fn)
        or "get_kv_cache_layout(" in inspect.getsource(fn)
    ]
    assert not offenders, f"{offenders} resolve the KV layout at forward time"


def test_raiden_geometry_resolves_the_layout_under_a_config():
    """`resolve_kernel_geometry` runs in the EngineCore, where no config is set.

    Both calls resolve the KV cache layout, which falls through to the KV
    connector and asserts without one.
    """
    import ast
    import inspect
    import textwrap

    from vllm_torchtpu.offload import raiden_store

    tree = ast.parse(
        textwrap.dedent(inspect.getsource(
            raiden_store.resolve_kernel_geometry)))
    guarded = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        if not any(
                isinstance(i.context_expr, ast.Call)
                and getattr(i.context_expr.func, "id",
                            None) == "set_current_vllm_config"
                for i in node.items):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call):
                f = inner.func
                guarded.add(f.attr if isinstance(f, ast.Attribute
                                                 ) else getattr(f, "id", None))

    assert {"select_common_block_size", "get_kv_cache_shape"} <= guarded


@pytest.mark.parametrize("num_kv_heads,head_size", HEAD_CASES_CORE)
@pytest.mark.parametrize("dtype", [BF16, FP8])
def test_shape_needs_no_tpu_client(nhd, monkeypatch, num_kv_heads, head_size,
                                   dtype):
    """The shape must be arithmetic, not a device query.

    `resolve_kernel_geometry` calls this from the scheduler process, which owns
    no chip -- the workers hold them all. The batched wrapper reads lane and
    sublane geometry via `pltpu.get_tpu_info()`, so routing any layout through
    it made engine startup die with
    `TPU initialization failed: open(/dev/vfio/N): Device or resource busy`.
    """
    from jax.experimental.pallas import tpu as pltpu

    def no_device(*_args, **_kwargs):
        raise AssertionError("get_kv_cache_shape queried the TPU")

    monkeypatch.setattr(pltpu, "get_tpu_info", no_device)
    assert BATCHED.get_kv_cache_shape(4, 128, num_kv_heads, head_size,
                                      dtype) == BASE.get_kv_cache_shape(
                                          4, 128, num_kv_heads, head_size,
                                          dtype)


def test_shape_needs_no_config_at_all():
    """No fixture on purpose: with the layout branch gone, neither the dtype-
    less connector probe nor a concrete dtype may read the current vLLM config.
    Resolving the layout before those early returns used to raise
    `Current vLLM config is not set`."""
    assert (BATCHED.get_kv_cache_shape(4, 128, 1, 256,
                                       "auto") == BASE.get_kv_cache_shape(
                                           4, 128, 1, 256, "auto"))
    assert (BATCHED.get_kv_cache_shape(4, 128, 1, 64,
                                       BF16) == BASE.get_kv_cache_shape(
                                           4, 128, 1, 64, BF16))
