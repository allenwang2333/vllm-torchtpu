# SPDX-License-Identifier: Apache-2.0
"""Offload the trailing chunk of every KV group under EAGLE/MTP.

vLLM's KV offloading connector applies its own copy of the EAGLE block drop:
each EAGLE group's trailing chunk is skipped on store while decoding and
popped from every host lookup, and partial-tail offload is disabled whenever
an EAGLE group exists. The drop only protects draft-model KV (acceptance),
not the target's KV or Mamba state, and with sparse prefix-cache retention it
leaves the host cache almost never hitting. This builds the offloading
scheduler config without EAGLE groups.
"""

from functools import wraps

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def patch_offload_without_eagle_groups() -> None:
    """Build the KV offloading scheduler config without EAGLE groups.

    The offloading connector applies its own copy of the drop: each EAGLE
    group's trailing chunk is skipped on store while decoding and popped from
    every host lookup. With no annotated group it flags *every* group, the
    Mamba groups included, so a host hit must end one chunk before the
    retained Mamba boundary. Under sparse retention
    (``--prefix-cache-retention-interval 0``) no state exists there and the
    host cache almost never hits. Like the device-side drop this only
    protects draft-model KV, so it goes with it. Recomputes
    ``supports_partial_tail`` now that ``is_eagle_group`` is cleared so
    128-token fine-grained partial-tail chunks are offloaded and reloaded.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading import (
        scheduler as offload_scheduler,
    )

    config_cls = offload_scheduler.SchedulerOffloadConfig
    original_from_spec = config_cls.from_spec
    if getattr(original_from_spec, "_tpu_disable_eagle_block_drop_patch", False):
        return

    @wraps(original_from_spec)
    def from_spec(spec, vllm_config, kv_cache_config):
        config = original_from_spec(spec, vllm_config, kv_cache_config)
        if not any(group.is_eagle_group for group in config.kv_group_configs):
            return config
        kv_group_configs = tuple(
            group._replace(is_eagle_group=False)
            for group in config.kv_group_configs
        )
        group_block_sizes = {g.tokens_per_block for g in kv_group_configs}
        has_partial_recurrent_group = any(
            g.requires_cow_source and g.tokens_per_block > spec.tokens_per_hash
            for g in kv_group_configs
        )
        dcp_size = getattr(
            getattr(vllm_config, "parallel_config", None),
            "decode_context_parallel_size",
            1,
        )
        supports_partial_tail = (
            spec.blocks_per_chunk == 1
            and len(group_block_sizes) == 1
            and has_partial_recurrent_group
            and all(
                g.sliding_window_size_in_chunks is None or g.requires_cow_source
                for g in kv_group_configs
            )
            and dcp_size == 1
        )
        logger.info(
            "KV offloading keeps the trailing chunk of every group "
            "(supports_partial_tail=%s).",
            supports_partial_tail,
        )
        return config._replace(
            kv_group_configs=kv_group_configs,
            supports_partial_tail=supports_partial_tail,
        )

    from_spec._tpu_disable_eagle_block_drop_patch = True
    config_cls.from_spec = staticmethod(from_spec)
