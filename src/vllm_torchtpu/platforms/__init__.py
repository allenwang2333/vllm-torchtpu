# ruff: noqa
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

_tpu_info_logged = False


def _log_tpu_info() -> None:
    global _tpu_info_logged
    if _tpu_info_logged:
        return
    _tpu_info_logged = True

    from vllm_torchtpu import tpu_info as ti
    from vllm_torchtpu.logger import init_logger

    logger = init_logger(__name__)
    try:
        logger.info(
            "TPU info: node_name=%s | tpu_type=%s | worker_id=%s | "
            "num_chips=%s | num_cores_per_chip=%s", ti.get_node_name(),
            ti.get_tpu_type(), ti.get_node_worker_id(), ti.get_num_chips(),
            ti.get_num_cores_per_chip())
    except Exception as error:
        logger.error(
            "Error occurred while logging TPU info: %s. "
            "Are you running on CPU?", error)




def _patch_block_pool_free_order() -> None:
    """Restore pre-#48017 LIFO block reuse when prefix caching is off.

    vllm a82f1b38 ("Skip LRU hash-split in free_blocks when prefix caching
    is off") switched hashless freed blocks from prepend (LIFO reuse) to
    append (FIFO) for GPU cache locality. On TPU v7x the FIFO order cycles
    the entire KV block-id space under sustained load, and an indexed
    SparseCore-offloaded op on the paged-KV path traps on the resulting
    block indices (E0200 RuntimeUnexpectedCoreHalt, UserFatal on SC2;
    deterministic on gemma-4-26b dp4-tp2-ep at c1024, also seen on
    Qwen3.5-397B GKE disagg decode). Bisect evidence:
    docs/sc2-sparsecore-halt-2026-08.md on branch
    mhhua/sc2-sparsecore-halt-evidence and vllm-torchtpu-dev builds
    136-145. Prepending restores the old, safe reuse order; behavior with
    prefix caching enabled is unchanged (hash-split identical either way).
    Remove once the SparseCore-side index handling is fixed in torch-tpu.
    """
    try:
        from vllm.v1.core.block_pool import BlockPool
    except Exception:
        return

    def free_blocks(self, ordered_blocks) -> None:
        blocks_with_hash = []
        blocks_without_hash = []
        for block in ordered_blocks:
            block.ref_cnt -= 1
            if block.ref_cnt == 0 and not block.is_null:
                if block.block_hash is None:
                    blocks_without_hash.append(block)
                else:
                    blocks_with_hash.append(block)
        self.free_block_queue.prepend_n(blocks_without_hash)
        self.free_block_queue.append_n(blocks_with_hash)

    BlockPool.free_blocks = free_blocks


def register_tpu_platform() -> str:
    """vLLM out-of-tree platform plugin entry point."""
    _log_tpu_info()
    _patch_block_pool_free_order()
    return f"{TpuPlatform.__module__}.{TpuPlatform.__qualname__}"
