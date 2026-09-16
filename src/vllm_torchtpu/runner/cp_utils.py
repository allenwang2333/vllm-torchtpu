# SPDX-License-Identifier: Apache-2.0
"""PCP+DCP compatibility gate for TPU, mirroring upstream vLLM's
``check_attention_cp_compatibility`` 
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_SUPPORTED_BACKENDS = frozenset({"sparse_mla"})


def is_pcp_dcp_compatible(pcp_size: int, dcp_size: int) -> bool:
    """Both CP axes are live, so the two token layouts have to be reconciled.

    PCP chunks the token axis; DCP shards the KV positions. They need not
    cover the same ranks: `dcp == pcp` makes them coincide, `dcp == tp * pcp`
    makes DCP span both axes, and the sparse-MLA path serves either. Every
    token-major tensor it hands across the boundary is replicated over the DCP
    group -- so the PCP gather leaves it alone -- except the top-k, which is
    kept out of that remap entirely (see
    `_SPARSE_MLA_DCP_INPUT_PARTITION_SPECS`).

    Callers use this to decide whether the PCP token remap is needed at all,
    not just to validate.
    """
    return pcp_size > 1 and dcp_size > 1


def validate_pcp_dcp_ratio(pcp_size: int, dcp_size: int,
                           tp_size: int) -> None:
    """With both CP axes on, DCP must span the PCP axis or all of TP x PCP.

    The same rule vLLM's own `ParallelConfig` applies, restated here so a
    layout the TPU kernels have never been run against cannot reach them if
    that validator ever widens. `dcp == pcp` makes the two groups coincide;
    `dcp == tp * pcp` makes DCP span both axes. Anything else leaves a rank
    holding a KV shard no coherent set of ranks completes.
    """
    if dcp_size > 1 and pcp_size > 1 and dcp_size not in (pcp_size,
                                                          tp_size * pcp_size):
        raise NotImplementedError(
            f"pcp_size={pcp_size} and dcp_size={dcp_size} are both >1, but "
            f"dcp_size is neither pcp_size ({pcp_size}) nor "
            f"tp_size * pcp_size ({tp_size * pcp_size}). See "
            "runner/cp_utils.py.")


def validate_pcp_dcp_backend_support(pcp_size: int, dcp_size: int,
                                     backend: str) -> None:
    """Call after validate_pcp_dcp_ratio: only backends in
    _SUPPORTED_BACKENDS may run with both CP axes enabled."""
    if is_pcp_dcp_compatible(pcp_size,
                             dcp_size) and backend not in _SUPPORTED_BACKENDS:
        raise NotImplementedError(
            f"{backend!r} does not support PCP and DCP together "
            f"(pcp_size={pcp_size}, dcp_size={dcp_size}) yet; only "
            f"{sorted(_SUPPORTED_BACKENDS)} does. See runner/cp_utils.py.")


def check_pcp_dcp_backend_compatibility(vllm_config: VllmConfig) -> None:
    """Call once, after the model is built -- layers register in
    static_forward_context at construction, so this walk needs them to
    already exist. No-op unless both CP axes are enabled.

    vllm.config/vllm.model_executor are imported here, not at module scope:
    this module loads early via platforms/pcp_validation.py during vLLM's
    plugin discovery, before vllm.config finishes initializing -- a
    module-scope import here would be circular.
    """
    from vllm.config import get_layers_from_vllm_config
    from vllm.model_executor.layers.attention_layer_base import \
        AttentionLayerBase

    parallel_config = vllm_config.parallel_config
    pcp_size = parallel_config.prefill_context_parallel_size
    dcp_size = parallel_config.decode_context_parallel_size
    if not is_pcp_dcp_compatible(pcp_size, dcp_size):
        return

    layers = get_layers_from_vllm_config(vllm_config, AttentionLayerBase)
    for layer in layers.values():
        # Skip cache-only companions (e.g. the sparse-MLA indexer's cache),
        # which have no `.impl` of their own -- upstream skips these too.
        if getattr(layer, "impl", None) is None:
            continue
        use_sparse = getattr(layer, "use_sparse", None)
        if use_sparse is None:
            backend = "dense_gqa"
        elif use_sparse:
            backend = "sparse_mla"
        else:
            backend = "dense_mla"
        validate_pcp_dcp_backend_support(pcp_size, dcp_size, backend)
