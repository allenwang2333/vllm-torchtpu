"""Fixed-input MLA KV dtype/packing audit; no model checkpoint required."""
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.mla.v2.kernel import mla_ragged_paged_attention


def run_case(tokens, prefix, dtype, scale):
    heads, latent, rope, page = 3, 512, 64, 256
    packing = 4 if dtype == jnp.float8_e4m3fn else 2
    rng = np.random.default_rng(980406)
    length = prefix + tokens
    # Exact binary fractions in both FP8 and BF16: no quantization error.
    keys = rng.integers(-4, 5, (length, latent)).astype(np.float32) / 4
    ropes = rng.integers(-4, 5, (length, rope)).astype(np.float32) / 4
    queries = rng.integers(-4, 5,
                           (heads, tokens, latent)).astype(np.float32) / 4
    qr = rng.integers(-4, 5, (tokens, heads, rope)).astype(np.float32) / 4
    pages = (length + page - 1) // page
    initial = np.zeros((pages * page, 640), np.float32)
    initial[:prefix, :latent] = keys[:prefix]
    initial[:prefix, latent:latent + rope] = ropes[:prefix]
    cache = jnp.asarray(initial / scale,
                        dtype).reshape(pages, page // packing, packing, 640)
    dist = [1, 1, 1] if tokens == 1 else [0, 0, 1]
    args = (jnp.asarray(queries, jnp.bfloat16), jnp.asarray(qr, jnp.bfloat16),
            jnp.asarray(keys[prefix:] / scale,
                        dtype), jnp.asarray(ropes[prefix:] / scale,
                                            dtype), cache,
            jnp.asarray([length], jnp.int32), jnp.arange(pages,
                                                         dtype=jnp.int32),
            jnp.asarray([0, tokens], jnp.int32), jnp.asarray(dist, jnp.int32))
    outputs = []
    cache_errors = []
    for _ in range(3):
        out, updated = mla_ragged_paged_attention(
            *args,
            sm_scale=0.125,
            k_scale=scale,
            v_scale=scale,
            num_kv_pages_per_block=1,
            num_queries_per_block=16,
        )
        outputs.append(np.asarray(out, dtype=np.float32))
        actual = np.asarray(updated, dtype=np.float32).reshape(-1, 640) * scale
        expected = np.concatenate([keys, ropes], axis=1)
        cache_errors.append(
            float(np.max(np.abs(actual[:length, :576] - expected))))
    logits = np.einsum('htd,sd->hts', queries, keys)
    logits += np.einsum('thd,sd->hts', qr, ropes)
    logits *= 0.125
    mask = np.arange(length)[None, :] <= (prefix + np.arange(tokens))[:, None]
    logits = np.where(mask[None], logits, -np.inf)
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    reference = np.einsum('hts,sd->htd', probs, keys)
    error = float(np.max(np.abs(outputs[0] - reference)))
    repeat_error = max(float(np.max(np.abs(x - outputs[0]))) for x in outputs)
    row = dict(tokens=tokens,
               prefix=prefix,
               dtype=jnp.dtype(dtype).name,
               scale=scale,
               output_max_abs_error=error,
               cache_max_abs_error=max(cache_errors),
               repeat_max_abs_error=repeat_error,
               finite=bool(np.isfinite(outputs).all()))
    print('MLA_KV_AUDIT ' + json.dumps(row), flush=True)
    return row


def main():
    root = Path(os.environ.get('ARTIFACT_LOCAL_DIR', '/tmp/mla-kv-audit'))
    out = root / 'rank-0/benchmark'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for tokens, prefix in [(1, 3), (6, 0), (6, 1), (6, 2), (6, 3), (6, 253),
                           (8, 255)]:
        for dtype in (jnp.bfloat16, jnp.float8_e4m3fn):
            for scale in (1.0, 0.5):
                row = run_case(tokens, prefix, dtype, scale)
                rows.append(row)
                with (out / 'metrics.jsonl').open('a') as f:
                    f.write(json.dumps(row) + '\n')
    summary = dict(devices=str(jax.devices()), cases=len(rows), results=rows)
    (root / 'summary.json').write_text(json.dumps(summary, indent=2))
    assert all(
        r['finite'] and r['cache_max_abs_error'] == 0
        and r['repeat_max_abs_error'] == 0 and r['output_max_abs_error'] < 0.02
        for r in rows), summary
    print('MLA_KV_AUDIT_PASSED', flush=True)


if __name__ == '__main__':
    main()
