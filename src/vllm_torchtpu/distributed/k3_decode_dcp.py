# SPDX-License-Identifier: Apache-2.0
"""Admission and page-striping configuration for native K3 decode DCP."""

from __future__ import annotations


def native_size(config) -> int:
    """Native TP-preserving K3 DCP, including the nested language model."""
    if config is None:
        return 1
    model = getattr(config, "model_config", None)
    architectures = getattr(model, "architectures", ()) or ()
    if not any(
            name in ("KimiK3ForConditionalGeneration", "KimiLinearForCausalLM")
            for name in architectures):
        return 1
    return int(
        getattr(config.parallel_config, "decode_context_parallel_size", 1))


def validate_native_config(config) -> None:
    """Admit the decode benchmark without modifying native process groups.

    Live PD needs destination-aware MLA placement; the existing connector
    cannot yet be assumed to produce the decoder's striped pages.
    """
    cp = native_size(config)
    if cp == 1:
        return
    pc = config.parallel_config
    if cp != 2 or pc.tensor_parallel_size % cp:
        raise ValueError("Native K3 DCP requires DCP2 and TP divisible by 2")
    if pc.pipeline_parallel_size != 1 or pc.prefill_context_parallel_size != 1:
        raise ValueError("Native K3 decode DCP requires PP1 and PCP1")
    if config.speculative_config is not None:
        raise ValueError("Native K3 DCP does not support speculative decode")
    transfer = config.kv_transfer_config
    if transfer is None or transfer.kv_connector != "DecodeBenchConnector":
        raise ValueError(
            "Native K3 DCP currently requires DecodeBenchConnector; "
            "live PD KV placement is not implemented")
    if config.cache_config.cache_dtype != "auto":
        raise ValueError(
            "Native K3 DCP currently requires BF16 KV (cache_dtype=auto)")


def configure_native_interleave(config) -> None:
    """Resolve page striping before model loading builds attention ops.

    vLLM performs the same adjustment for KV connectors later, after cache
    specs are known. Both its deprecated and current fields must agree.
    """
    if native_size(config) > 1:
        pc = config.parallel_config
        pc.cp_kv_cache_interleave_size = config.cache_config.block_size
        pc.dcp_kv_cache_interleave_size = config.cache_config.block_size


def validate_decode_step(config, scheduler_output) -> None:
    if native_size(config) > 1 and any(
            n != 1 for n in scheduler_output.num_scheduled_tokens.values()):
        raise ValueError("K3 DCP supports one decode token per request; "
                         "prefill/extend must run on the P side")
