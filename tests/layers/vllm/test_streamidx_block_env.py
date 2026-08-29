"""Coverage for the `TPU_STREAMIDX_*` block-size overrides.

`streamindex_topk`'s `num_kv_pages_per_block` / `num_queries_per_block` were
hardcoded first-fit constants. They are now read from the environment so a sweep
can run without a rebuild, and a sweep on GLM-5.2 shapes found that decode and
prefill want *opposite* values -- which is why the env accepts the kernel's
`(decode, prefill, mixed)` triple form and not just a scalar.

Two things need holding down:

- the parse, including that an unset env reproduces the original constants
  exactly, so an untouched environment cannot perturb the compile cache key;
- the wiring, i.e. that the module constant actually tracks the env. A sweep
  that passes block sizes to the kernel as kwargs -- which is how the
  microbench measured them -- proves nothing about this path.

Pure CPU. No TPU, no kernel invocation.
"""

import os
import subprocess
import sys

import pytest

from vllm_torchtpu.layers.vllm.custom_ops.mla_attention_op import (
    STREAMIDX_NUM_KV_PAGES_PER_BLOCK, STREAMIDX_NUM_QUERIES_PER_BLOCK,
    _streamidx_block_size)

# The values the kernel shipped with, before the env override existed.
UNTUNED_KV_PAGES = 8
UNTUNED_QUERIES = 32


@pytest.mark.parametrize("raw, expected", [
    ("", 99),
    ("   ", 99),
    ("4", 4),
    ("  4  ", 4),
    ("8,2,2", (8, 2, 2)),
    (" 8 , 2 , 2 ", (8, 2, 2)),
    ("1,1,1", (1, 1, 1)),
])
def test_parse_accepts_scalar_triple_and_empty(raw, expected):
    assert _streamidx_block_size(raw, 99) == expected


@pytest.mark.parametrize("raw", ["8,2", "8,2,2,2", "8,,2", "eight", "8;2;2"])
def test_parse_rejects_malformed(raw):
    """A typo must fail loudly rather than silently benchmarking the default."""
    with pytest.raises(ValueError):
        _streamidx_block_size(raw, 99)


def test_unset_env_reproduces_the_untuned_constants():
    """The override must be inert unless explicitly set.

    Skipped rather than failed if the caller exported the vars, since that is a
    legitimate way to run the suite -- see the sweep-config case below.
    """
    if os.getenv("TPU_STREAMIDX_KV_PAGES_PER_BLOCK") or os.getenv(
            "TPU_STREAMIDX_QUERIES_PER_BLOCK"):
        pytest.skip("TPU_STREAMIDX_* is set in this environment")
    assert STREAMIDX_NUM_KV_PAGES_PER_BLOCK == UNTUNED_KV_PAGES
    assert STREAMIDX_NUM_QUERIES_PER_BLOCK == UNTUNED_QUERIES


# Import in a fresh interpreter rather than `importlib.reload`, which would
# re-run this module's custom-op registration.
_PROBE = ("from vllm_torchtpu.layers.vllm.custom_ops.mla_attention_op import ("
          "STREAMIDX_NUM_KV_PAGES_PER_BLOCK as kv, "
          "STREAMIDX_NUM_QUERIES_PER_BLOCK as q); print(repr((kv, q)))")


def _probe(env_overrides: dict[str, str]) -> str:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("TPU_STREAMIDX_")
    }
    env.update(env_overrides)
    out = subprocess.run([sys.executable, "-c", _PROBE],
                         capture_output=True,
                         text=True,
                         env=env,
                         timeout=600)
    assert out.returncode == 0, out.stderr[-4000:]
    return out.stdout.strip().splitlines()[-1]


_TRIM_PROBE = (
    "from vllm_torchtpu.layers.vllm.custom_ops.mla_attention_op import "
    "STREAMIDX_TRIM_TOPK as t; print(repr(t))")


def _trim_probe(value: str | None) -> str:
    env = {
        k: v
        for k, v in os.environ.items() if k != "TPU_STREAMIDX_TRIM_TOPK"
    }
    if value is not None:
        env["TPU_STREAMIDX_TRIM_TOPK"] = value
    out = subprocess.run([sys.executable, "-c", _TRIM_PROBE],
                         capture_output=True,
                         text=True,
                         env=env,
                         timeout=600)
    assert out.returncode == 0, out.stderr[-4000:]
    return out.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize("value, expected", [
    (None, "False"),
    ("0", "False"),
    ("1", "True"),
])
def test_trim_topk_env_reaches_the_module_constant(value, expected):
    """`TPU_STREAMIDX_TRIM_TOPK` must default off and be readable at import.

    Off by default is the load-bearing half: the trim is an A/B against the
    padded-sort baseline, and every other arm in flight shares this source
    tree via --map-dir. An unset environment must reproduce the old behaviour
    exactly or those arms stop being comparable to their baseline.
    """
    assert _trim_probe(value) == expected


@pytest.mark.parametrize("overrides, expected", [
    ({}, f"({UNTUNED_KV_PAGES}, {UNTUNED_QUERIES})"),
    ({
        "TPU_STREAMIDX_KV_PAGES_PER_BLOCK": "8,2,2"
    }, f"((8, 2, 2), {UNTUNED_QUERIES})"),
    ({
        "TPU_STREAMIDX_KV_PAGES_PER_BLOCK": "2",
        "TPU_STREAMIDX_QUERIES_PER_BLOCK": "16",
    }, "(2, 16)"),
])
def test_env_reaches_the_module_constant(overrides, expected):
    """End-to-end: exported variable -> imported constant, in a clean process.

    The middle case is the shipping candidate: prefill wants bkv_p=2 while
    decode is fastest at 8, and `num_queries_per_block` measured inert, so it
    is left at its default.
    """
    assert _probe(overrides) == expected
