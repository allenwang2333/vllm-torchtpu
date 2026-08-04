"""Fetch individual safetensors tensors over HTTP range requests.

A 433 GiB checkpoint is not worth downloading to read a few 512-element
LayerNorm vectors. safetensors puts a JSON header at the front of each shard
giving every tensor's byte range, and the HF resolve endpoint honours Range,
so each tensor costs one small GET.
"""
import json, subprocess, sys, struct
import numpy as np, ml_dtypes

def _get(url, start=None, end=None):
    cmd = ["curl", "-sL", "--fail"]
    if start is not None:
        cmd += ["-r", f"{start}-{end}"]
    cmd += [url]
    r = subprocess.run(cmd, capture_output=True, timeout=300)
    if r.returncode:
        raise RuntimeError(f"curl {r.returncode}: {r.stderr[:200]!r}")
    return r.stdout

_HDR = {}
def header(repo, shard):
    key = (repo, shard)
    if key not in _HDR:
        url = f"https://huggingface.co/{repo}/resolve/main/{shard}"
        n = struct.unpack("<Q", _get(url, 0, 7))[0]
        _HDR[key] = (json.loads(_get(url, 8, 8 + n - 1)), 8 + n, url)
    return _HDR[key]

DT = {"BF16": ml_dtypes.bfloat16, "F16": np.float16, "F32": np.float32,
      "F8_E4M3": ml_dtypes.float8_e4m3fn, "U8": np.uint8, "I8": np.int8}

def tensor(repo, shard, name):
    hdr, base, url = header(repo, shard)
    meta = hdr[name]
    lo, hi = meta["data_offsets"]
    raw = _get(url, base + lo, base + hi - 1)
    arr = np.frombuffer(raw, dtype=DT[meta["dtype"]]).reshape(meta["shape"])
    return arr.astype(np.float32)


def kv_a_layernorm_profile(repo, index_path, layers):
    """rms(gamma) per layer for an MLA model's `kv_a_layernorm`.

    RMSNorm emits a unit-RMS direction scaled by gamma, so `rms(gamma)` is the
    RMS of `kv_c_normed` -- the tensor the TPU path quantizes to fp8. Measured
    against DeepSeek-V2-Lite it over-predicts the on-device value by 1.16-1.36x
    (gamma is largest where the normalized activation is smallest), so it is a
    slightly optimistic proxy: the real latent is a little *smaller* still.

    This matters because the fp8 e4m3 min-normal is 2^-6 = 0.0156 in absolute
    terms. With the default per-tensor `k_scale = 1.0` any model whose latent
    RMS sits near or below that quantizes inside the subnormal range, where
    e4m3 has no implicit leading mantissa bit and precision collapses.
    """
    import json
    wm = json.load(open(index_path))["weight_map"]
    out = []
    for layer in layers:
        name = f"model.layers.{layer}.self_attn.kv_a_layernorm.weight"
        if name not in wm:
            continue
        g = tensor(repo, wm[name], name)
        out.append((layer, float((g**2).mean() ** 0.5)))
    return out
