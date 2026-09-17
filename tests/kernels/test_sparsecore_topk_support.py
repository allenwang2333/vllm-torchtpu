# SPDX-License-Identifier: Apache-2.0
"""The SparseCore top-k kernel is admitted only on the SparseCore geometry it
is written for (16-lane subcores, 32 of them); other generations fall back to
the TensorCore reduction, which must keep the same contract."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from vllm_torchtpu.kernels.deepseek_v4 import \
    sparsecore_topk as sct  # noqa: E402


@pytest.fixture
def cpu_backend():
    """Run the fallback's arrays on CPU.

    It is a handful of index operations, and the chips on a serving host are
    usually busy; selecting the platform here keeps the test off them rather
    than failing when one is held.
    """
    jax.config.update("jax_platforms", "cpu")
    try:
        yield
    finally:
        jax.config.update("jax_platforms", "")


_GET_INFO = ("vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk."
             "pltpu.get_tpu_info")


def _info(num_cores, num_subcores, num_lanes):
    return SimpleNamespace(sparse_core=SimpleNamespace(
        num_cores=num_cores, num_subcores=num_subcores, num_lanes=num_lanes))


@pytest.mark.parametrize(
    ("geometry", "expected"),
    [
        ((2, 16, 16), True),  # what the kernel is written for
        ((2, 16, 8), False),  # 8-lane subcores: the iota does not lower
        ((4, 16, 8), False),
        ((1, 4, 16), False),  # 16 lanes but too few subcores
    ])
def test_supported_by_sparsecore_geometry(geometry, expected):
    sct.sparsecore_topk_supported.cache_clear()
    with patch(_GET_INFO, return_value=_info(*geometry)):
        assert sct.sparsecore_topk_supported() is expected
    sct.sparsecore_topk_supported.cache_clear()


def test_unsupported_without_a_sparsecore_or_a_tpu():
    sct.sparsecore_topk_supported.cache_clear()
    with patch(_GET_INFO, return_value=SimpleNamespace(sparse_core=None)):
        assert sct.sparsecore_topk_supported() is False
    sct.sparsecore_topk_supported.cache_clear()
    with patch(_GET_INFO, side_effect=RuntimeError("no TPU backend")):
        assert sct.sparsecore_topk_supported() is False
    sct.sparsecore_topk_supported.cache_clear()


def test_the_fallback_keeps_the_kernel_contract(cpu_backend):
    """Exact top-k column indices, and -1 where a row runs out of columns."""
    rng = np.random.default_rng(0)
    scores = jnp.asarray(rng.standard_normal((4, 32)), dtype=jnp.float32)
    row_lengths = jnp.asarray([32, 8, 3, 0], dtype=jnp.int32)

    out = np.asarray(sct._tensorcore_topk(scores, 4, row_lengths))

    reference = np.asarray(scores)
    for row, length in enumerate(np.asarray(row_lengths)):
        live = reference[row, :length]
        want = sorted(np.argsort(-live)[:min(4, length)].tolist())
        got = sorted(i for i in out[row].tolist() if i >= 0)
        assert got == want, row
        # every column past the row's length is excluded, and short rows
        # keep the kernel's -1 padding
        assert all(i < length for i in got)
        assert out[row].tolist().count(-1) == 4 - len(want)
