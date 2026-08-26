# SPDX-License-Identifier: Apache-2.0
"""Deviceless duck-typed tensor fakes for the Raiden pool-manifest tests.

Moved out of tpu_connector_v2_test_utils.py so that collecting the live
Raiden tests no longer imports the TPUConnectorV2 cluster at module scope.
FakeStorage and FakeTensor move as a pair: FakeTensor's default constructor
builds a FakeStorage, and FakeStorage._next_ptr is mutable class state that
must exist exactly once (two copies would hand out colliding data_ptr()
values and silently weaken the dead-storage check).
"""


class FakeStorage:
    """Duck-typed untyped_storage() stand-in with a unique data pointer."""

    _next_ptr = 0x1000

    def __init__(self):
        FakeStorage._next_ptr += 0x100000000
        self._ptr = FakeStorage._next_ptr

    def data_ptr(self):
        return self._ptr


class FakeTensor:
    """Duck-typed tensor for pool-manifest tests (no torch required)."""

    def __init__(self,
                 shape,
                 element_size,
                 dtype="torch.fake",
                 storage=None,
                 storage_offset_elems=0):
        self.shape = tuple(shape)
        self._element_size = int(element_size)
        self.dtype = dtype
        self._storage = storage if storage is not None else FakeStorage()
        self._storage_offset = int(storage_offset_elems)
        numel = 1
        for dim in self.shape:
            numel *= dim
        self.nbytes = numel * self._element_size

    def element_size(self):
        return self._element_size

    def untyped_storage(self):
        return self._storage

    def storage_offset(self):
        return self._storage_offset


def glm_named_kv_caches(*,
                        num_blocks=16,
                        fa_shape=None,
                        fa_esz=1,
                        fa_dtype="torch.float8_e5m2",
                        idx_shape=None):
    """Named per-layer caches shaped like a GLM-5.2 materialization.

    Two MLA latent caches plus one uint8 DSA indexer cache, all packed
    [blocks, rows, packing, width].
    """
    fa_shape = fa_shape or (num_blocks, 256, 4, 640)
    idx_shape = idx_shape or (num_blocks, 256, 4, 256)
    named = {
        f"model.layers.{idx}.self_attn.mla_attn":
        FakeTensor(fa_shape, fa_esz, dtype=fa_dtype)
        for idx in range(2)
    }
    named["model.layers.0.self_attn.indexer"] = FakeTensor(idx_shape,
                                                           1,
                                                           dtype="torch.uint8")
    return named


class FakeKVCacheGroup:
    """Duck-typed KVCacheGroupSpec (layer names plus one shared spec)."""

    def __init__(self, layer_names, spec):
        self.layer_names = list(layer_names)
        self.kv_cache_spec = spec


class FakeDsv4Spec:
    """Duck-typed DSv4 KV cache spec: only the fields the manifest reads."""

    def __init__(self, *, storage_block_size, compress_ratio=1):
        self.storage_block_size = int(storage_block_size)
        self.compress_ratio = int(compress_ratio)
        # Logical page size, what the scheduler pages this group at. The
        # compressed roles store one row per `compress_ratio` tokens, so their
        # storage block is that much smaller than their block_size.
        self.block_size = self.storage_block_size * self.compress_ratio


def dsv4_materialization(*, num_blocks=16, tokens_per_page=256):
    """A DSv4 materialization with the runner's overlay plan applied.

    Two CSA layers, two indexer layers, one HCA layer, two SWA layers and the
    four compressor state caches. Every SWA cache and every state cache
    aliases an already-allocated array, so the returned caches exercise the
    block-ID-disjoint aliasing that DSv4 admission has to describe.

    Returns `(named_kv_caches, kv_cache_groups)`.
    """
    tokens = int(tokens_per_page)

    def cache(rows, width):
        return FakeTensor((num_blocks, rows, 4, width), 1, dtype="torch.uint8")

    csa_nope = [cache(tokens, 128) for _ in range(2)]
    csa_rope = [cache(tokens // 4, 128) for _ in range(2)]
    indexer = [cache(tokens // 4, 256) for _ in range(2)]
    hca = cache(tokens * 2, 128)

    csa_names = [f"model.layers.{i}.self_attn.attn" for i in range(2)]
    idx_names = [
        f"model.layers.{i}.self_attn.attn.indexer.k_cache" for i in range(2)
    ]
    hca_name = "model.layers.2.self_attn.attn"
    swa_names = [f"model.layers.{i}.self_attn.swa_cache" for i in range(2)]

    named = {}
    for name, nope, rope in zip(csa_names, csa_nope, csa_rope):
        named[name] = (nope, rope)
    for name, array in zip(idx_names, indexer):
        named[name] = array
    named[hca_name] = hca
    # SWA overlays the CSA NoPE arrays by position within its cache group.
    for name, host in zip(swa_names, csa_nope):
        named[name] = host

    csa_state = [f"{name}.compressor.state_cache" for name in csa_names]
    idx_state = [
        f"model.layers.{i}.self_attn.attn.indexer.compressor.state_cache"
        for i in range(2)
    ]
    hca_state = f"{hca_name}.compressor.state_cache"
    # CSA/indexer states share their own compressed-KV array; the HCA state is
    # far too big for the HCA page, so it overlays a CSA NoPE array instead.
    for name, host in zip(csa_state, csa_nope):
        named[name] = (host, csa_rope[csa_nope.index(host)])
    for name, host in zip(idx_state, indexer):
        named[name] = host
    named[hca_state] = csa_nope[0]

    csa_spec = FakeDsv4Spec(storage_block_size=tokens, compress_ratio=4)
    idx_spec = FakeDsv4Spec(storage_block_size=tokens, compress_ratio=4)
    hca_spec = FakeDsv4Spec(storage_block_size=tokens, compress_ratio=1)
    # SWA keeps raw latents, so its page holds an eighth of the CSA group's
    # tokens at the same byte size, and it pages at that smaller size.
    swa_spec = FakeDsv4Spec(storage_block_size=tokens // 2, compress_ratio=1)
    # Each state role has its own window, so vLLM puts it in its own group.
    groups = [
        FakeKVCacheGroup(csa_names, csa_spec),
        FakeKVCacheGroup(idx_names, idx_spec),
        FakeKVCacheGroup([hca_name], hca_spec),
        FakeKVCacheGroup(swa_names, swa_spec),
        FakeKVCacheGroup(csa_state, csa_spec),
        FakeKVCacheGroup(idx_state, idx_spec),
        FakeKVCacheGroup([hca_state], hca_spec),
    ]
    return named, groups
