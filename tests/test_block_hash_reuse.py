# SPDX-License-Identifier: Apache-2.0
"""TPU_BLOCK_HASH_REUSE gives every request exactly the block hashes vLLM's own
hasher gives it, for prompts that extend, diverge from or repeat earlier ones."""

import random

import pytest
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.request import Request

from vllm_torchtpu.core import block_hash_reuse as bhr

BS = 16


def _request(tokens, hasher, cache_salt=None):
    return Request(
        request_id=str(random.random()),
        prompt_token_ids=list(tokens),
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        cache_salt=cache_salt,
        block_hasher=hasher,
    )


@pytest.fixture
def hashers():
    init_none_hash(sha256)
    plain = get_request_block_hasher(BS, sha256)
    return plain, bhr.reusing_block_hasher(
        get_request_block_hasher(BS, sha256), BS, max_tokens=20_000
    )


def _same(tokens, plain, reuse, **kw):
    want, got = _request(tokens, plain, **kw), _request(tokens, reuse, **kw)
    assert got.block_hashes == want.block_hashes
    for t in (7, 8, 9):  # decode blocks keep chaining from the reused list
        want.append_output_token_ids([t] * 11)
        got.append_output_token_ids([t] * 11)
    assert got.block_hashes == want.block_hashes


def test_conversations_match_plain_hasher(hashers):
    plain, reuse = hashers
    rng = random.Random(0)
    system = [rng.randrange(1000) for _ in range(300)]
    sessions = [list(system) for _ in range(4)]
    for _ in range(60):
        s = rng.randrange(len(sessions))
        r = rng.random()
        if r < 0.7:  # next turn: the history grows
            sessions[s] += [rng.randrange(1000) for _ in range(rng.randrange(1, 400))]
        elif r < 0.85:  # history rewritten past some point
            cut = rng.randrange(len(sessions[s]))
            sessions[s] = sessions[s][:cut] + [rng.randrange(1000) for _ in range(50)]
        # else: the same prompt again
        _same(sessions[s], plain, reuse)
    store = reuse.store
    assert store.reused_blocks > store.hashed_blocks > 0
    assert store._bytes <= 20_000 * 4


def test_edges(hashers):
    plain, reuse = hashers
    base = list(range(100, 100 + 8 * BS))
    for tokens in [
        [],
        base[:5],  # shorter than a block: nothing to hash or store
        base,
        base,  # identical prompt
        base[: 3 * BS],  # a prefix of a stored prompt
        base[: 3 * BS + 7],
        base + [1],
        base[:-1] + [0],  # diverges inside the last block
        [0] + base,  # diverges in the first block
    ]:
        _same(tokens, plain, reuse)


def test_extra_keys_bypass_store(hashers):
    plain, reuse = hashers
    tokens = list(range(10 * BS))
    _same(tokens, plain, reuse)
    before = reuse.store.prompts
    _same(tokens, plain, reuse, cache_salt="salt")
    assert reuse.store.prompts == before


def test_store_bound_and_covered_entries():
    store = bhr.PrefixHashStore(BS, max_tokens=10 * BS)
    tokens = bytes(range(4 * BS)) * 4  # 4 blocks of int32 "tokens"
    store.insert(tokens, [b"h"] * 4, None)
    longer = tokens + bytes(4 * BS) * 2
    hashes, eid = store.lookup(longer)
    assert hashes == [b"h"] * 4
    store.insert(longer, [b"h"] * 6, eid)  # the prefix entry is dropped
    assert len(store._entries) == 1
    store.insert(bytes(4 * BS) * 8, [b"x"] * 8, None)  # over the bound: evict
    assert store._bytes <= 10 * BS * 4


def test_patch_wraps_engine_core_hasher(monkeypatch):
    from vllm.v1.engine import core

    monkeypatch.setenv("TPU_BLOCK_HASH_REUSE", "1")
    monkeypatch.setattr(core, "get_request_block_hasher", get_request_block_hasher)
    monkeypatch.setattr(core, "_tpu_block_hash_reuse", False, raising=False)
    bhr.patch_block_hash_reuse()
    init_none_hash(sha256)
    hasher = core.get_request_block_hasher(BS, sha256)
    assert isinstance(hasher.store, bhr.PrefixHashStore)
