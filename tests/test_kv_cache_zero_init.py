"""The KV cache pool must come up finite, fp8 included.

RPA v3's default ``RPA_V_MASK_MODE=none`` drops the explicit zeroing of the V
rows past ``effective_kv_len`` and relies on ``p == 0`` to annihilate them,
which is worth -8.5% on the decode kernel (``j-30133ae0``). ``p == 0``
annihilates a *finite* V and nothing else. So the padding rows -- which no
sequence ever writes, but which every kernel invocation reads -- have to be
finite from the moment the pool exists.

``kv_cache_materializer.py`` used to take a ``torch.empty`` fast path for the
fp8 dtypes specifically, on the reasonable-sounding grounds that nobody reads
uninitialized cache. RPA v3 does. Measured on a v7x pod (``j-761bc2df``),
``torch.empty`` there is not merely unspecified, it is maximally hostile:

    torch.float8_e4m3fn trial 0 nonzero_bytes 524288 of 524288 nonfinite 524288
    torch.float8_e5m2   trial 0 nonzero_bytes 524288 of 524288 nonfinite 524288

Every byte non-zero, every value non-finite, on all three trials of both
dtypes -- because ``0xFF`` is a NaN in ``f8e4m3fn`` and uninitialized HBM is
full of high bytes. That combination shipped as the default for two commits.

Scope it honestly: the ``torch.empty`` fast path existed only in
``allocate_raw_kv_cache_tensors``, which is reached only when
``_can_allocate_attention_cache_directly()`` is False -- hybrid
attention+mamba. Every non-hybrid model, Qwen3-Coder-30B included, already
went through ``make_attention_cache_tensor`` and its ``torch.zeros``. So the
window was "NaN on every decode of a hybrid fp8 model", not of every model.
The guard below still covers both routes, because the cheap thing to defend
is the invariant rather than the list of models that currently satisfy it.

Paired with ``tests/kernels/test_rpa_v_mask_padding.py``, which pins what the
kernel needs; this pins that the allocator supplies it.
"""
import pytest
import torch

FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]
FP8_IDS = ["e4m3fn", "e5m2"]

# Big enough that a page-granular allocator cannot hand back a coincidentally
# clean page, and shaped like a real pool rather than a toy.
POOL_SHAPE = (4096, 128)


def _as_bytes(t: torch.Tensor) -> torch.Tensor:
    # fp8 has no comparison ops on most backends; compare the raw bytes.
    return t.cpu().view(torch.uint8)


@pytest.mark.parametrize("dtype", FP8_DTYPES + [torch.bfloat16],
                         ids=FP8_IDS + ["bf16"])
def test_zeros_is_actually_zero_on_the_target_device(device, dtype):
    """`torch.zeros` is the mechanism the fix relies on. Verify it directly.

    Not as tautological as it looks: the fp8 dtypes are new enough that a
    backend can plausibly lack a fill kernel for them and fall back to
    something that is not zero. That is the failure this catches.
    """
    t = torch.zeros(POOL_SHAPE, dtype=dtype, device=device)
    nonzero = int((_as_bytes(t) != 0).sum())
    assert nonzero == 0, (
        f"torch.zeros({dtype}) left {nonzero} non-zero bytes on {device}; "
        f"the KV pool cannot rely on it and RPA_V_MASK_MODE=none is unsafe")


def _torch_calls(fn) -> set[str]:
    """Names of `torch.<name>(...)` calls actually made in `fn`'s body.

    An AST walk rather than a substring scan, because the first version of
    this test searched the source text and failed on the *comment* explaining
    why torch.empty was removed (j-f636426b). A guard that cannot tell code
    from prose about the code is worse than none: it fires on the fix.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    return {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
    }


def test_the_materializer_has_no_uninitialized_kv_pool_path():
    """No allocation route into the KV pool may use `torch.empty`.

    A behavioural test cannot reach every branch here -- the pooled path is
    gated on `_can_allocate_attention_cache_directly`, which depends on a
    KVCacheConfig that varies by model -- so this inspects the two allocators
    instead. It is the only check that covers both routes at once.
    """
    from vllm_torchtpu import kv_cache_materializer

    for fn in (kv_cache_materializer.allocate_raw_kv_cache_tensors,
               kv_cache_materializer.make_attention_cache_tensor):
        calls = _torch_calls(fn)
        assert "empty" not in calls, (
            f"{fn.__name__} allocates the KV pool with torch.empty. On TPU "
            f"that is 100% non-finite for fp8 (j-761bc2df, j-f636426b), and "
            f"RPA v3's default V-mask elision requires finite padding. Use "
            f"torch.zeros, or set RPA_V_MASK_MODE=full and re-measure.")
        assert "zeros" in calls, (
            f"{fn.__name__} no longer allocates with torch.zeros (saw "
            f"{sorted(calls)}); if the allocation moved, move this guard "
            f"with it rather than deleting it")


@pytest.mark.parametrize("dtype", FP8_DTYPES, ids=FP8_IDS)
def test_uninitialized_fp8_really_is_hostile(device, dtype):
    """Documents *why* the guard above exists, rather than asserting it.

    This is the measurement that turned the hazard from theoretical into
    observed. It is deliberately not an assertion about `torch.empty` being
    dirty -- that would be a test of undefined behaviour, and a future
    allocator that happens to hand back zeroed pages should not fail the
    suite. It reports, so the next reader sees the numbers on their hardware.
    """
    t = torch.empty(POOL_SHAPE, dtype=dtype, device=device)
    b = _as_bytes(t)
    f = t.cpu().float()
    nonzero, nonfinite = int((b != 0).sum()), int((~torch.isfinite(f)).sum())
    print(f"\ntorch.empty({dtype}) on {device}: "
          f"{nonzero}/{b.numel()} non-zero bytes, "
          f"{nonfinite}/{f.numel()} non-finite values")
    if nonfinite == 0:
        pytest.skip(
            f"torch.empty({dtype}) happened to be finite on {device} this "
            f"run; the guard above does not depend on it being dirty")
