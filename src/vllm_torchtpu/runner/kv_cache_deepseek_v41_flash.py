# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1-Flash KV-cache allocation: CSA2/SWA/indexer cache arrays."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner.kv_cache_dsv4 import DsV4KVCacheAllocator
from vllm_torchtpu.utils import synchronize_tensors

logger = init_logger(__name__)


class DeepseekV41FlashKVCacheAllocator(DsV4KVCacheAllocator):
    """DeepSeek-V4.1-Flash KV-cache allocator (CSA2 + SEQ_ALONG_LANE indexer)."""

    @staticmethod
    def aux_partition_sizes(
        num_blocks: int, max_num_reqs: int | None = None
    ) -> tuple[int, int]:
        """Tail-block counts reserved for compressor state and SWA caches."""
        if num_blocks >= 4096:
            reqs = (
                max_num_reqs if max_num_reqs is not None else max(64, num_blocks // 107)
            )
            num_state = max(
                128,
                min((9 * reqs) // 4 if reqs > 64 else 2 * reqs, num_blocks // 32),
            )
            num_swa = max(768, min((11 * reqs) // 2, num_blocks // 11))
            return num_state, num_swa
        if num_blocks >= 64:
            return max(4, min(64, num_blocks // 32)), max(
                8, min(128, (num_blocks * 3) // 16)
            )
        return 0, 0

    def _initialize_ds_v4_kv_cache(
        self,
        kv_cache_config: KVCacheConfig,
        per_layer_spec: Callable[[str], KVCacheSpec],
        kv_caches: dict[str, torch.Tensor],
        num_blocks: int,
    ) -> None:
        runner = self.runner
        bs = runner.block_size = runner.cache_config.block_size
        cap = min(runner.max_num_reqs, runner.max_num_tokens)
        most_len = runner.most_model_len
        runner.num_blocks_per_most_len_req = runner.num_reqs_most_model_len = None
        if most_len is not None:
            runner.num_blocks_per_most_len_req = cdiv(most_len, bs)
            runner.num_reqs_most_model_len = cap
        runner.num_reqs_max_model_len = cap

        mla_layer_names, swa_layer_groups, state_layer_names = (
            self._classify_ds_v4_layers(kv_cache_config, per_layer_spec)
        )
        packing = self._DS_V4_KV_PACKING

        max_reqs = getattr(runner, "max_num_reqs", None)
        if not isinstance(max_reqs, int):
            max_reqs = getattr(
                getattr(runner, "scheduler_config", None), "max_num_seqs", None
            )
        if not isinstance(max_reqs, int):
            max_reqs = None

        num_state_aux, num_swa_aux = self.aux_partition_sizes(num_blocks, max_reqs)
        main_end = max(1, num_blocks - num_state_aux - num_swa_aux)

        def _create_cache(
            shape: tuple[int, ...],
            tag: str,
            dtype: torch.dtype = torch.uint8,
            scale_idx: tuple | None = None,
        ) -> torch.Tensor:
            if scale_idx is not None and main_end > 1:
                import gc

                cpu_buf = torch.zeros(shape, dtype=dtype)
                cpu_buf[(slice(1, main_end), *scale_idx)] = (
                    (0x7F << 24) if dtype == torch.int32 else 0x7F
                )  # e8m0 = 1.0
                cache = cpu_buf.to(runner.device)
                del cpu_buf
                gc.collect()
            else:
                cache = torch.zeros(shape, dtype=dtype, device=runner.device)
            synchronize_tensors([cache])
            logger.debug(
                "DeepSeek-V4.1 KV array for %s: shape=%s dtype=%s", tag, shape, dtype
            )
            return cache

        csa_nope_hosts: list[torch.Tensor] = []
        for layer_name in mla_layer_names:
            spec = per_layer_spec(layer_name)
            page_size = spec.num_states
            if layer_name.endswith(self._DS_V4_INDEXER_CACHE_SUFFIX):
                shape = (num_blocks, spec.head_size // packing, packing, page_size)
                kv_caches[layer_name] = _create_cache(
                    shape, layer_name, scale_idx=(32, 0, slice(None))
                )
            else:
                nope = _create_cache(
                    (num_blocks, page_size, csa_cache_layout.ROW_WORDS),
                    layer_name,
                    torch.int32,
                    (slice(None), slice(64, 128)),
                )
                rope = _create_cache(
                    (
                        num_blocks,
                        page_size // csa_cache_layout.ROPE_TOKENS_PER_ROW,
                        csa_cache_layout.ROW_WORDS,
                    ),
                    f"{layer_name}{self._DS_V4_ROPE_CACHE_SUFFIX}",
                    torch.int32,
                )
                kv_caches[layer_name] = (nope, rope)
                csa_nope_hosts.append(nope)

        if not csa_nope_hosts:
            raise ValueError(f"No CSA layer in MLA layers: {mla_layer_names}")
        mla_ptrs: dict[int, str] = {}
        for layer_name in mla_layer_names:
            entry = kv_caches[layer_name]
            for idx_t, t in enumerate(entry if isinstance(entry, tuple) else (entry,)):
                if t.numel() > 0:
                    ptr = int(t.data_ptr())
                    tag_t = f"{layer_name}[{idx_t}]"
                    assert ptr not in mla_ptrs, (
                        f"KV cache aliasing detected: {tag_t} aliases {mla_ptrs[ptr]}"
                    )
                    mla_ptrs[ptr] = tag_t

        swa_rows = (
            per_layer_spec(swa_layer_groups[0][0]).block_size * 2
            if (swa_layer_groups and swa_layer_groups[0])
            else min(h.shape[1] for h in csa_nope_hosts)
        )
        csa_nope_hosts = sorted(
            [h for h in csa_nope_hosts if h.shape[1] >= swa_rows],
            key=lambda h: int(h.shape[1]),
        )
        if not csa_nope_hosts:
            raise ValueError(
                f"No CSA NoPE host with rows >= {swa_rows} in MLA layers: {mla_layer_names}"
            )

        swa_overflow_blocks = min(num_blocks, max(64, num_swa_aux + 1))
        swa_overflow_shape = (
            swa_overflow_blocks,
            swa_rows,
            csa_cache_layout.ROW_WORDS,
        )

        def _host_at(
            hosts: list[torch.Tensor],
            position: int,
            alloc_shape: tuple[int, ...],
        ) -> torch.Tensor:
            while position >= len(hosts):
                hosts.append(
                    _create_cache(
                        alloc_shape,
                        f"deepseek_v41_flash_overflow.{len(hosts)}",
                        csa_nope_hosts[0].dtype,
                    )
                )
            return hosts[position]

        overlay_hosts = list(csa_nope_hosts)
        for group_swa_layers in swa_layer_groups:
            for position, layer_name in enumerate(group_swa_layers):
                kv_caches[layer_name] = _host_at(
                    overlay_hosts, position, swa_overflow_shape
                )

        for position, layer_name in enumerate(state_layer_names):
            kv_layer_name = self._ds_v4_compressed_kv_layer_name(layer_name)
            if kv_layer_name not in kv_caches:
                raise ValueError(f"State cache {layer_name} missing KV {kv_layer_name}")
            kv_caches[layer_name] = kv_caches[kv_layer_name]

        self._validate_ds_v4_overlay(kv_cache_config, kv_caches)

        model = getattr(runner, "model", None)
        draft_model = getattr(getattr(runner, "drafter", None), "draft_model", None)
        roots = [m for m in (model, draft_model) if m is not None]
        if roots:
            init_caches: list[torch.Tensor] = []
            for root in roots:
                for mod in root.modules():
                    if hasattr(mod, "ensure_cache"):
                        swa = getattr(mod, "swa_cache_module", None)
                        if swa is not None and swa.prefix in kv_caches:
                            swa.kv_cache = kv_caches[swa.prefix]
                        if mod.ensure_cache():
                            init_caches.append(mod._cache)
                    if hasattr(mod, "state_cache") and getattr(mod, "state_cache", None) is not None and hasattr(mod, "k_cache_prefix"):
                        _st_e = kv_caches.get(mod.state_cache.prefix)
                        _k_e = kv_caches.get(mod.k_cache_prefix)
                        _st_t = _st_e[0] if isinstance(_st_e, tuple) else _st_e
                        _no_t = _k_e[0] if isinstance(_k_e, tuple) else _k_e
                        if _st_t is not None and _no_t is not None:
                            _al = bool(_st_t.data_ptr() == _no_t.data_ptr())
                            object.__setattr__(mod, "_aliased_state_to_nope", _al)
                            assert _al == (_st_t.data_ptr() == _no_t.data_ptr())
                    if hasattr(mod, "_get_dummy_main_cache"):
                        init_caches.append(mod._get_dummy_main_cache(runner.device))
            if init_caches:
                synchronize_tensors(init_caches)

        arrays = [t for e in kv_caches.values() for t in self._ds_v4_cache_arrays(e)]
        logger.info(
            "DeepSeek-V4.1-Flash KV cache: %d arrays for %d layers (mla=%d of which "
            "csa=%d, swa=%d in %d group(s) sized %s, state=%d), num_blocks=%d, "
            "largest array=%s",
            len({id(t) for t in arrays}),
            len(kv_caches),
            len(mla_layer_names),
            len(csa_nope_hosts),
            sum(len(g) for g in swa_layer_groups),
            len(swa_layer_groups),
            [len(g) for g in swa_layer_groups],
            len(state_layer_names),
            num_blocks,
            max((tuple(t.shape) for t in arrays), key=math.prod, default=()),
        )
