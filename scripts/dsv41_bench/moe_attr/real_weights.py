"""DeepSeek-V4.1 routed experts: checkpoint -> fused-EP runtime form.

Mirrors the serving loader (`VllmDeepseekV4Mxfp4MoEMethod` in
layers/adapter/quantization/mxfp4.py): MXFP4 (e2m1 nibbles, low nibble first,
e8m0 scale per 32) dequantized exactly to bf16, w1|w3 concatenated and laid out
K-major, the intermediate padded 2304 -> 2560 per half, requantized to fp4 e2m1
with one f32 scale per 512 contraction rows by the tree's own
`quantize_to_native_fp4_kmajor`, scales handed over row-padded flat. The
checkpoint is read straight from its safetensors shard (no torch).
"""

import json
import struct
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.layers.core.fused_moe_gmm import quantize_to_native_fp4_kmajor

E8M0_MIN_EXP = -127  # layers/core/quantization.py
BLOCK = 512  # REQUANTIZED_BLOCK_SIZE, the serving default


def shard_for_layer(index_json: Path, layer: int) -> str:
    weight_map = json.loads(Path(index_json).read_text())["weight_map"]
    files = {
        f for k, f in weight_map.items() if k.startswith(f"layers.{layer}.ffn.experts.")
    }
    if len(files) != 1:
        raise ValueError(f"layer {layer} experts span {sorted(files)}")
    return files.pop()


def read_layer_raw(shard: Path, layer: int, n_experts: int) -> dict:
    """{name: uint8 [E, ...]} for w1, w3, w2 (packed) and s1, s3, s2 (e8m0)."""
    with open(shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    data = np.memmap(shard, dtype=np.uint8, mode="r", offset=8 + n)
    out = {}
    for name, (w, kind) in {
        "w1": ("w1", "weight"),
        "w3": ("w3", "weight"),
        "w2": ("w2", "weight"),
        "s1": ("w1", "scale"),
        "s3": ("w3", "scale"),
        "s2": ("w2", "scale"),
    }.items():
        parts = []
        for e in range(n_experts):
            meta = header[f"layers.{layer}.ffn.experts.{e}.{w}.{kind}"]
            if meta["dtype"] not in ("I8", "U8", "F8_E8M0"):
                raise ValueError(f"unexpected dtype {meta['dtype']} for {w}.{kind}")
            lo, hi = meta["data_offsets"]
            parts.append(np.asarray(data[lo:hi]).reshape(meta["shape"]))
        out[name] = np.stack(parts)
    return out


def _decode_e2m1(nibble):
    """The loader's on-device decode (_dequantize_mxfp4_packed_on_device)."""
    exponent = (nibble >> 1) & 3
    mantissa = (nibble & 1).astype(jnp.float32)
    magnitude = jnp.where(
        exponent == 0,
        mantissa * 0.5,
        jnp.where(
            exponent == 1,
            1.0 + mantissa * 0.5,
            jnp.where(exponent == 2, 2.0 + mantissa, 4.0 + mantissa * 2.0),
        ),
    )
    return jnp.where((nibble & 8) != 0, -magnitude, magnitude)


def dequant_mxfp4(packed, scale_u8):
    """[O, In/2] packed, [O, In/32] e8m0 -> bf16 [O, In] (exact)."""
    p = packed.astype(jnp.int32) & 0xFF
    v = jnp.stack([_decode_e2m1(p & 0x0F), _decode_e2m1(p >> 4)], axis=-1)
    v = v.reshape(packed.shape[0], -1)
    s = jnp.ldexp(
        jnp.ones(scale_u8.shape, jnp.float32), scale_u8.astype(jnp.int32) + E8M0_MIN_EXP
    )
    o, k = v.shape
    nb = scale_u8.shape[-1]
    return (v.reshape(o, nb, k // nb) * s[..., None]).reshape(o, k).astype(jnp.bfloat16)


def expert_runtime(w1, w3, w2, s1, s3, s2, *, inter_padded, block=BLOCK):
    """One expert -> w13 [H, 2Ip], w2 [Ip, H] (fp4), s13 [H/b, 2Ip], s2 [Ip/b, H]."""
    w13 = jnp.concatenate([dequant_mxfp4(w1, s1), dequant_mxfp4(w3, s3)], axis=0).T
    inter = w1.shape[0]
    pad = inter_padded - inter
    w13 = jnp.concatenate(
        [
            jnp.pad(w13[:, :inter], ((0, 0), (0, pad))),
            jnp.pad(w13[:, inter:], ((0, 0), (0, pad))),
        ],
        axis=1,
    )
    w2k = jnp.pad(dequant_mxfp4(w2, s2).T, ((0, pad), (0, 0)))
    q13, sc13 = quantize_to_native_fp4_kmajor(w13, block, pack=False)
    q2, sc2 = quantize_to_native_fp4_kmajor(w2k, min(block, inter), pack=False)
    return q13, q2, sc13[:, 0, :], sc2[:, 0, :]


def runtime_weights(mesh, axis, raw: dict, *, inter_padded=2560):
    """(w1, w2, s1, s2) sharded over the expert axis, scales [E, K/b, N]."""

    def local(w1, w3, w2, s1, s3, s2):
        return lax.map(
            lambda a: expert_runtime(*a, inter_padded=inter_padded),
            (w1, w3, w2, s1, s3, s2),
        )

    names = ("w1", "w3", "w2", "s1", "s3", "s2")
    sharding = jax.sharding.NamedSharding(mesh, P(axis))
    args = [jax.device_put(raw[n], sharding) for n in names]
    return jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P(axis),) * 6,
            out_specs=(P(axis),) * 4,
            check_vma=False,
        )
    )(*args)
