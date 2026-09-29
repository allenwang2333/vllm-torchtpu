"""Host-side layouts shared by the megakernel, its wrapper and its tests.

- `smalls_layout`: the per-layer f32 vector pack (norm weights, per-channel scales, biases), one
  row per layer, every field 128-lane aligned so the kernel slices it statically.
- served paged-cache packing (kernels/mla/kv_cache_utils.py SPARSECORE, and the indexer's
  HEAD_ALONG_SUBLANE uint8 record), used by tests to build caches and read them back.
"""

import jax
import jax.numpy as jnp
import numpy as np

FP8 = jnp.float8_e4m3fn


def rup(x, a=128):
    return (x + a - 1) // a * a


def smalls_layout(cfg):
    """name -> (offset, length); returns (dict, total width)."""
    fields = [
        ("in_ln", cfg.H), ("post_ln", cfg.H), ("q_a_ln", cfg.QL), ("kv_a_ln", cfg.KL),
        ("qkv_a_s", cfg.QKVA), ("q_b_s", cfg.NH * cfg.DQ), ("o_proj_s", cfg.H),
        ("idx_wq_s", cfg.IH * cfg.IDIM), ("idx_kn_w", cfg.IDIM), ("idx_kn_b", cfg.IDIM),
        ("gate_bias", cfg.E), ("sw13_s", 2 * cfg.IS), ("sw2_s", cfg.H),
        ("w2_s", cfg.H),
    ]
    off, lay = 0, {}
    for name, n in fields:
        lay[name] = (off, n)
        off += rup(n)
    return lay, off


def build_smalls(cfg, layers):
    """layers: list of per-layer weight dicts (reference.make_weights format) -> f32 [L, 1, S]."""
    lay, S = smalls_layout(cfg)
    out = np.zeros((cfg.L, 1, S), np.float32)
    for l, lw in enumerate(layers):
        for name, (o, n) in lay.items():
            if name in lw:
                out[l, 0, o:o + n] = np.asarray(lw[name], np.float32).reshape(-1)
    return jnp.asarray(out)


# --- MLA caches (SPARSECORE layout) ---------------------------------------------------------

def pack_bands(u8, token_bytes):
    """[T, n] uint8 -> [T, token_bytes // 4] uint32: word j = bytes (j, B+j, 2B+j, 3B+j)."""
    u8 = np.pad(u8, ((0, 0), (0, token_bytes - u8.shape[1]))).astype(np.uint32)
    b = u8.reshape(u8.shape[0], 4, token_bytes // 4)
    return b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16) | (b[:, 3] << 24)


def unpack_bands(words, n):
    words = np.asarray(words, np.uint32)
    b = np.stack([(words >> (8 * k)) & 0xFF for k in range(4)], 1).astype(np.uint8)
    return b.reshape(words.shape[0], -1)[:, :n]


def fp8_bytes(x):
    return np.asarray(jax.lax.bitcast_convert_type(jnp.asarray(x, FP8), jnp.uint8))


def bytes_fp8(b):
    return jax.lax.bitcast_convert_type(jnp.asarray(b, jnp.uint8), FP8)


def rope_words(rope_packed):
    """u32 words per token of the served ROPE cache: 32 padded (128 B), 16 packed (64 B)."""
    return 16 if rope_packed else 32


def pack_caches(cfg, logical, block_tables, num_pages, rope_packed=False):
    """logical: reference cache dict for one layer, one rank (nope8 [C,512], rope8 [C,64],
    optional idx_k [C,128] fp8 / idx_e [C] int32). block_tables [P] int32 (logical page -> physical).
    Returns dict of served arrays for that rank: nope u32 [nb, page, 128], rope u32 [nb, page/4, 128]
    (padded) or [nb, page/8, 128] (rope_packed), idx u8 [nb, page/4, 4, 256]."""
    pg = cfg.page
    rw = rope_words(rope_packed)
    C = logical["nope8"].shape[0]
    bt = np.asarray(block_tables)
    nope = np.zeros((num_pages * pg, 128), np.uint32)
    rope = np.zeros((num_pages * pg, rw), np.uint32)
    pos = np.arange(C)
    phys = bt[pos // pg] * pg + pos % pg
    nope[phys] = pack_bands(fp8_bytes(logical["nope8"]), 512)
    rope[phys] = pack_bands(fp8_bytes(logical["rope8"]), 4 * rw)
    out = {"nope": jnp.asarray(nope.reshape(num_pages, pg, 128)),
           "rope": jnp.asarray(rope.reshape(num_pages, pg * rw // 128, 128))}
    if "idx_k" in logical:
        rec = np.zeros((num_pages * pg, 256), np.uint8)
        rec[phys, :128] = fp8_bytes(logical["idx_k"])
        rec[phys, 128] = (np.asarray(logical["idx_e"]) + 127).astype(np.uint8)
        out["idx"] = jnp.asarray(rec.reshape(num_pages, pg // 4, 4, 256))
    return out


def unpack_caches(cfg, served, block_tables, C, rope_packed=False):
    pg = cfg.page
    bt = np.asarray(block_tables)
    pos = np.arange(C)
    phys = bt[pos // pg] * pg + pos % pg
    nope = np.asarray(served["nope"]).reshape(-1, 128)[phys]
    rope = np.asarray(served["rope"]).reshape(-1, rope_words(rope_packed))[phys]
    out = {"nope8": bytes_fp8(unpack_bands(nope, 512)),
           "rope8": bytes_fp8(unpack_bands(rope, 64))}
    if "idx" in served:
        rec = np.asarray(served["idx"]).reshape(-1, 256)[phys]
        out["idx_k"] = bytes_fp8(rec[:, :128])
        out["idx_e"] = jnp.asarray(rec[:, 128].astype(np.int32) - 127)
    return out
