# SPDX-License-Identifier: Apache-2.0
"""TpuMultiprocExecutor: MultiprocExecutor with per-shape KV offload prewarm.

Splits H2D Pallas kernel prewarm into one collective_rpc call per power-of-2
shape so each shape gets its own 60-second shm_broadcast window (~4s per shape)
instead of all 14 shapes sharing a single window and timing out.
"""

from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Ranks are expected to agree on the compact-mamba block count to within
# rounding. Beyond this they still get `min()` (safe -- the most constrained rank
# bounds everyone) but the divergence is reported, because it means one rank is
# carrying materially more memory than its peers.
_NUM_BLOCKS_OVERRIDE_TOL = 4


class TpuMultiprocExecutor(MultiprocExecutor):

    def _init_executor(self) -> None:
        # EngineCore does not run platform patch setup before spawning workers.
        from vllm_torchtpu import _patch_multiproc_worker_global_rank_env
        _patch_multiproc_worker_global_rank_env()
        super()._init_executor()

    def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]:
        specs = super().get_kv_cache_specs()

        if self.vllm_config.cache_config.num_gpu_blocks_override is None:
            # Compact-mamba sizing sets `num_gpu_blocks_override` on the worker's
            # cache_config during the RPC above; workers are separate processes, so
            # copy it to the engine-side config here.
            overrides = [
                ovr
                for ovr in self.collective_rpc("get_num_gpu_blocks_override")
                if ovr is not None
            ]
            if overrides:
                spread = max(overrides) - min(overrides)
                if spread > _NUM_BLOCKS_OVERRIDE_TOL:
                    # Was an assert. Taking min() is safe regardless -- the
                    # smallest rank bounds the pool, so nobody over-allocates --
                    # so a divergence should not block startup. But it does mean
                    # some rank holds materially more memory than its peers, and
                    # the surplus on every other rank goes unused.
                    #
                    # Seen on Kimi-K3 (9L, TP=4): rank 3 reported 0.90 GiB less
                    # available than ranks 0-2, a 15847-block spread. Kimi-Linear
                    # at TP=8 agrees exactly across all 8 ranks, so this is not
                    # measurement noise -- suspicion is the mxfp4 expert path,
                    # which is what K3 has and Kimi-Linear does not.
                    logger.warning(
                        "Ranks disagree on the compact-mamba KV block count by "
                        "%d blocks (tolerance %d): %s. Using the minimum (%d), "
                        "so the surplus on the other ranks is left unused. One "
                        "rank is holding more memory than its peers; check the "
                        "per-rank 'Compact-mamba KV cache' lines for which.",
                        spread, _NUM_BLOCKS_OVERRIDE_TOL, overrides,
                        min(overrides))
                self.vllm_config.cache_config.num_gpu_blocks_override = min(
                    overrides)

        return specs

    def initialize_from_config(self,
                               kv_cache_configs: list[KVCacheConfig]) -> None:
        super().initialize_from_config(kv_cache_configs)
        # Per-shape H2D prewarm: each shape is its own collective_rpc call with
        # its own 60-second shm_broadcast window.  Workers that have no KV
        # offload spec return [] from get_kv_prewarm_shapes() — a no-op.
        shapes = self.collective_rpc("get_kv_prewarm_shapes")
        if not shapes or not shapes[0]:
            return
        for p in shapes[0]:
            self.collective_rpc("prewarm_kv_offload_shape", args=(p, ))
