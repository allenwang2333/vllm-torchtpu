# SPDX-License-Identifier: Apache-2.0
"""Reuse prefix-cache block hashes across prompts that share a prefix
(``TPU_BLOCK_HASH_REUSE``).

EngineCore hashes every new prompt from its first token: one chained
``sha256(pickle((parent, block_tokens, extra_keys)))`` per hash block. Hybrid
models hash at the GCD of their group block sizes (16 tokens for DeepSeek-V4
on TPU), so a 500k-token agent turn costs ~31k hashes, ~0.1 s of Python on
the request's path and under the GIL, although the previous turn of the same
conversation already hashed all but its last few thousand tokens.

A chained hash depends only on the tokens up to its block end and on the
extra keys. For prompts without extra keys (no multimodal features, LoRA,
cache salt or prompt embeddings) the hashes of a shared token prefix are
therefore identical, so they are copied from the stored request that shares
the longest prefix and only the rest is hashed, by vLLM's own hasher.
Requests with extra keys, and hashing of later (decode) blocks, take vLLM's
hasher unchanged.
"""

from __future__ import annotations

import array
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Stored prompts sharing a first hash block that are tried per lookup, newest
# first. Sessions with a common system prompt share it.
_MAX_CANDIDATES = 16
_LOG_EVERY = 500


def _common_blocks(new: bytes, old: bytes, block_bytes: int) -> int:
    """Number of leading whole blocks on which ``new`` and ``old`` agree."""
    hi = min(len(new), len(old)) // block_bytes
    if new[: hi * block_bytes] == old[: hi * block_bytes]:
        return hi
    lo = 0  # invariant: lo blocks agree, hi blocks do not
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if new[: mid * block_bytes] == old[: mid * block_bytes]:
            lo = mid
        else:
            hi = mid
    return lo


class PrefixHashStore:
    """Prompt token bytes and their block hashes, LRU-bounded in tokens."""

    def __init__(self, block_size: int, max_tokens: int):
        self._block_bytes = block_size * 4  # int32 token ids
        self._max_bytes = max_tokens * 4
        self._entries: OrderedDict[int, tuple[bytes, list[Any]]] = OrderedDict()
        self._by_head: dict[bytes, list[int]] = {}
        self._bytes = 0
        self._next_id = 0
        self._lock = threading.Lock()
        self.prompts = 0
        self.reused_blocks = 0
        self.hashed_blocks = 0

    def lookup(self, tokens: bytes) -> tuple[list[Any], int | None]:
        """Hashes of the longest stored prefix of ``tokens`` and its entry id."""
        with self._lock:
            best, best_id = 0, None
            for eid in reversed(self._by_head.get(tokens[: self._block_bytes], ())):
                old, hashes = self._entries[eid]
                n = min(_common_blocks(tokens, old, self._block_bytes), len(hashes))
                if n > best:
                    best, best_id = n, eid
            if best_id is None:
                return [], None
            self._entries.move_to_end(best_id)
            return self._entries[best_id][1][:best], best_id

    def insert(self, tokens: bytes, hashes: list[Any], covered: int | None) -> None:
        """Store a prompt; drop ``covered``, the entry it reused, if a prefix."""
        if len(tokens) < self._block_bytes or len(tokens) > self._max_bytes:
            return
        with self._lock:
            if covered is not None and covered in self._entries:
                old = self._entries[covered][0]
                if tokens.startswith(old):
                    self._remove(covered)
            eid = self._next_id
            self._next_id += 1
            self._entries[eid] = (tokens, hashes)
            head = self._by_head.setdefault(tokens[: self._block_bytes], [])
            head.append(eid)
            if len(head) > _MAX_CANDIDATES:
                self._remove(head[0])
            self._bytes += len(tokens)
            while self._bytes > self._max_bytes:
                self._remove(next(iter(self._entries)))

    def _remove(self, eid: int) -> None:
        tokens, _ = self._entries.pop(eid)
        self._bytes -= len(tokens)
        head = tokens[: self._block_bytes]
        ids = self._by_head[head]
        ids.remove(eid)
        if not ids:
            del self._by_head[head]


def _has_extra_keys(request: Any) -> bool:
    return bool(
        request.mm_features
        or request.lora_request is not None
        or request.cache_salt
        or request.prompt_embeds is not None
    )


def reusing_block_hasher(
    block_hasher: Callable[[Any], list[Any]],
    hash_block_size: int,
    max_tokens: int,
) -> Callable[[Any], list[Any]]:
    """Wrap vLLM's request block hasher with a ``PrefixHashStore``."""
    store = PrefixHashStore(hash_block_size, max_tokens)

    def request_block_hasher(request: Any) -> list[Any]:
        if request.block_hashes or _has_extra_keys(request):
            return block_hasher(request)
        tokens = array.array("i", request.prompt_token_ids).tobytes()
        reused, eid = store.lookup(tokens)
        if reused:
            # vLLM's hasher continues the chain after request.block_hashes;
            # the caller extends the original list object with our result.
            own = request.block_hashes
            request.block_hashes = reused
            try:
                tail = block_hasher(request)
            finally:
                request.block_hashes = own
            hashes = reused + tail
        else:
            hashes = block_hasher(request)
        store.insert(tokens, hashes, eid)
        store.prompts += 1
        store.reused_blocks += len(reused)
        store.hashed_blocks += len(hashes) - len(reused)
        if store.prompts % _LOG_EVERY == 0:
            logger.info(
                "TPU_BLOCK_HASH_REUSE: %d prompts, %d block hashes reused, "
                "%d computed.",
                store.prompts,
                store.reused_blocks,
                store.hashed_blocks,
            )
        return hashes

    request_block_hasher.store = store  # type: ignore[attr-defined]
    return request_block_hasher


def patch_block_hash_reuse() -> None:
    """Make EngineCore build its request block hasher with reuse."""
    from vllm_torchtpu import envs

    if not envs.TPU_BLOCK_HASH_REUSE:
        return
    from vllm.v1.engine import core

    if getattr(core, "_tpu_block_hash_reuse", False):
        return
    orig = core.get_request_block_hasher
    max_tokens = envs.TPU_BLOCK_HASH_REUSE_MAX_TOKENS

    def get_request_block_hasher(hash_block_size, caching_hash_fn):
        return reusing_block_hasher(
            orig(hash_block_size, caching_hash_fn), hash_block_size, max_tokens
        )

    core.get_request_block_hasher = get_request_block_hasher
    core._tpu_block_hash_reuse = True
    logger.info(
        "TPU_BLOCK_HASH_REUSE: prompt block hashes reuse the longest stored "
        "prefix (store bound %d tokens).",
        max_tokens,
    )
