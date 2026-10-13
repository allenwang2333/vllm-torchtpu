"""V2 binding of the shared DeepSeek V4.1 Engram."""

from typing import override

import torch

from ..common import engram as common
from ..common.engram import (
    _INVALID_SCALE,
    _UNIT_SCALE,
    DPSharedEngramStorage,
    Engram,
    ParallelEngramEmbedding,
    _gather_engram_rows,
    _load_local_rows,
    _unpack_bytes,
    can_share_engram_tables,
    dequant_fp8_rows,
    register_row_slice,
    upstream,
)
from .cache import write_cache

__all__ = [
    "DPSharedEngramStorage",
    "Engram",
    "NgramHashState",
    "ParallelEngramEmbedding",
    "_INVALID_SCALE",
    "_UNIT_SCALE",
    "_gather_engram_rows",
    "_load_local_rows",
    "_unpack_bytes",
    "can_share_engram_tables",
    "dequant_fp8_rows",
    "register_row_slice",
    "upstream",
]


class NgramHashState(common.NgramHashState):
    write_cache = staticmethod(write_cache)

    @override
    def ensure_cache(self) -> bool:
        if torch.compiler.is_compiling():
            return super().ensure_cache()
        kv_cache = self.swa_cache_module.kv_cache
        ref = self._kv_cache_ref
        if (
            ref is not None
            and ref() is kv_cache
            and self.block_size == kv_cache.shape[1]
        ):
            return True
        self.write_cache = write_cache
        return super().ensure_cache()
