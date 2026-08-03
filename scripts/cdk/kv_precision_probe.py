# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Capture the real MLA KV tensors a TPU layer is about to fp8-quantize.

`PallasMLAttentionBackendImpl.forward` quantizes both `kv_c_normed` (the
512-d compressed latent) and `k_pe` (the 64-d RoPE tail) with a single
per-tensor `k_scale` before handing them to the kernel. The GPU reference
(`fp8_ds_mla`) instead uses a per-128-block fp32 scale on the latent and
keeps the RoPE tail in bf16. This probe grabs the unquantized tensors from a
real forward pass so `kv_precision_analysis.py` can price that difference on
actual activations instead of a synthetic stand-in.

The hook sits at the top of `forward`, before the `quantize_kv` calls, so it
sees exactly what the cache write would consume. Keys are captured in full
(they are only 576 floats per token); queries are subsampled across tokens
but keep every head, since they are [T, H, 576] and would otherwise dominate
the dump.

    cdk job create ttpu-dev SCRIPT=kv_precision_probe \
      PROBE_MODEL=deepseek-ai/DeepSeek-V2-Lite
"""
import os
import sys
import traceback

import numpy as np

# Which decoder layers to capture. Early layers have atypical activation
# statistics, so the default spans the stack rather than trusting layer 0.
CAPTURE_LAYERS = os.environ.get("CAPTURE_LAYERS", "1,13,25")
# Query rows kept per layer. All heads are retained for each.
MAX_Q_ROWS = int(os.environ.get("MAX_Q_ROWS", "64"))

CAPTURED = {}
# Only capture once the real prefill is in flight. `kv_cache.numel() == 0`
# rules out the memory-profiling pass, but vLLM also runs warmup forwards
# *after* the cache is allocated, and those carry dummy activations whose
# statistics are not the ones the quantizer will see in production.
ARMED = [False]


def _to_numpy(t):
    """Materialize a TPU tensor as float32 numpy.

    torch_tpu is eager, so `.cpu()` is a real transfer; the float() cast
    happens host-side to avoid a second device kernel.
    """
    import torch
    with torch.no_grad():
        return t.detach().to("cpu").to(torch.float32).numpy()


def install_hook(target_indices):
    """Patch the MLA impl to record its inputs for the selected layers."""
    import torch

    from vllm_torchtpu.layers.vllm.attention import \
        PallasMLAttentionBackendImpl

    original = PallasMLAttentionBackendImpl.forward

    def patched(self, layer, q, kv_c_normed, k_pe, kv_cache, attn_metadata,
                output=None, output_scale=None, output_block_scale=None,
                **kwargs):
        try:
            name = getattr(layer, "layer_name", "")
            idx = _layer_index(name)
            if (ARMED[0] and idx in target_indices and name not in CAPTURED
                    and kv_cache.numel() > 0 and isinstance(q, tuple)):
                _capture(self, layer, name, q, kv_c_normed, k_pe)
        except Exception:
            print(f"### capture failed for {getattr(layer, 'layer_name', '?')}",
                  flush=True)
            traceback.print_exc()
        return original(self, layer, q, kv_c_normed, k_pe, kv_cache,
                        attn_metadata, output=output, output_scale=output_scale,
                        output_block_scale=output_block_scale, **kwargs)

    PallasMLAttentionBackendImpl.forward = patched
    print(f"### hook installed for layer indices {sorted(target_indices)}",
          flush=True)


def _has_tpu():
    """True when a real device is attached.

    torch_tpu only registers the `tpu` device type when chips are present, so
    on the CPU recipe this raises rather than returning an empty device list.
    """
    import torch
    try:
        torch.zeros(1, device="tpu")
        return True
    except Exception:
        return False


def _layer_index(name):
    """`model.layers.13.self_attn.attn` -> 13."""
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return -1


def _capture(impl, layer, name, q, kv_c_normed, k_pe):
    import torch

    q_nope, q_pe = q
    # Callers reach here with assorted leading dims (rope can add one under
    # compilation), so normalize against the layer's declared widths rather
    # than trusting shape[0].
    kv_c_normed = kv_c_normed.reshape(-1, layer.kv_lora_rank)
    k_pe = k_pe.reshape(-1, layer.qk_rope_head_dim)
    num_tokens = kv_c_normed.shape[0]
    q_nope = q_nope.reshape(num_tokens, layer.num_heads, -1)
    q_pe = q_pe.reshape(num_tokens, layer.num_heads, layer.qk_rope_head_dim)

    # ql_nope is what the kernel actually dots against the cached latent:
    # q_nope projected through W_UK_T. Recomputed here exactly as
    # `PallasMLAttentionBackendImpl.forward` does it a few lines later.
    with torch.no_grad():
        q_nope_t = q_nope.transpose(0, 1)
        w_uk_t = layer.W_UK_T.to(q_nope_t.dtype)
        ql_nope = torch.bmm(q_nope_t, w_uk_t).transpose(0, 1)

    stride = max(1, num_tokens // MAX_Q_ROWS)
    sel = slice(None, None, stride)

    entry = {
        "kv_c_normed": _to_numpy(kv_c_normed).astype(np.float32),
        "k_pe": _to_numpy(k_pe).astype(np.float32),
        "ql_nope": _to_numpy(ql_nope[sel]).astype(np.float32),
        "q_pe": _to_numpy(q_pe[sel]).astype(np.float32),
        # Which token each sampled query row is, so the analysis can apply the
        # causal mask. Without it every query would appear to attend to the
        # whole sequence and the softmax -- hence the output error -- would be
        # wrong for early tokens.
        "q_token_idx": np.arange(num_tokens, dtype=np.int32)[sel],
        "k_scale": np.float32(getattr(layer, "_k_scale_float", 1.0) or 1.0),
        "sm_scale": np.float32(layer.scale),
    }
    CAPTURED[name] = entry
    print(f"### captured {name}: latent{entry['kv_c_normed'].shape} "
          f"rope{entry['k_pe'].shape} q{entry['ql_nope'].shape} "
          f"k_scale={float(entry['k_scale'])} sm_scale={float(entry['sm_scale'])}",
          flush=True)


def build_prompt(target_tokens):
    """A prompt long enough to produce a realistic prefill.

    Natural text, not a repeated token: activation statistics on a degenerate
    prompt are not the ones the quantizer will see in production.
    """
    para = (
        "The compressed latent cache used by multi-head latent attention "
        "stores a single low-rank vector per token instead of separate key "
        "and value projections for every head. Because the same latent is "
        "shared across all query heads, any error introduced when the latent "
        "is quantized is amplified by the number of heads that read it. "
        "The decoupled rotary component is stored alongside the latent and "
        "carries the positional signal that lets the model distinguish "
        "otherwise identical tokens at different offsets. "
    )
    # ~4 chars/token is close enough; the exact length does not matter, only
    # that it is long enough to exercise a real prefill.
    reps = max(1, (target_tokens * 4) // len(para))
    return para * reps


def main():
    model = os.environ.get("PROBE_MODEL", "deepseek-ai/DeepSeek-V2-Lite")
    max_len = int(os.environ.get("MAX_MODEL_LEN", "4096"))
    prompt_len = int(os.environ.get("PROMPT_LEN", "2048"))
    tp = int(os.environ.get("TP_SIZE", "1"))
    mem_util = float(os.environ.get("GPU_MEM_UTIL", "0.7"))
    # With --mount-gcs=false there is no CDK_OUTPUT_DIR. Fall back to the
    # weights PVC, which persists across jobs, rather than the container's
    # ephemeral /tmp -- the printed table is the primary result, but the npz
    # is what lets the analysis be re-run without another device job.
    out_dir = os.environ.get("CDK_OUTPUT_DIR") or "/hf-cache/kv_probe"
    kv_dtype = os.environ.get("KV_CACHE_DTYPE", "fp8")

    targets = {int(x) for x in CAPTURE_LAYERS.split(",") if x.strip()}

    import torch  # noqa: F401
    import torch_tpu  # noqa: F401
    from vllm import LLM, SamplingParams

    install_hook(targets)

    print(f"### model={model} tp={tp} max_len={max_len} "
          f"prompt_len={prompt_len} kv_dtype={kv_dtype}", flush=True)

    if os.environ.get("PROBE_DRY_RUN", "0") == "1" or not _has_tpu():
        # Import/wiring check, so a typo costs a CPU pod instead of four TPU
        # chips. Triggered automatically on ttpu-dev-cpu, which has no device
        # -- the recipe cannot pass an undeclared PROBE_DRY_RUN through anyway.
        # Exercises the analysis path too, on synthetic tensors.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import kv_precision_analysis as ana
        data = ana.synthetic(num_tokens=128)
        print(ana._fmt(ana.analyze(data, 576**-0.5, topk=64)), flush=True)
        print("### DRY RUN OK", flush=True)
        return 0

    llm = LLM(
        model=model,
        tensor_parallel_size=tp,
        max_model_len=max_len,
        max_num_batched_tokens=max_len,
        max_num_seqs=4,
        enforce_eager=True,
        trust_remote_code=True,
        gpu_memory_utilization=mem_util,
        kv_cache_dtype=kv_dtype,
    )

    prompt = build_prompt(prompt_len)
    # Arm only now: everything before this point (memory profiling, post-cache
    # warmup) runs on dummy activations.
    ARMED[0] = True
    # One token out: only the prefill activations matter here.
    llm.generate([prompt], SamplingParams(max_tokens=1, temperature=0.0))
    ARMED[0] = False

    if not CAPTURED:
        print("### FATAL: nothing captured -- did the hook match any layer?")
        return 4

    os.makedirs(out_dir, exist_ok=True)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import kv_precision_analysis as ana

    for name, entry in CAPTURED.items():
        tag = name.replace(".", "_")
        path = os.path.join(out_dir, f"kv_dump_{tag}.npz")
        np.savez_compressed(path, **entry)
        print(f"\n### wrote {path}", flush=True)

        print(f"\n{'=' * 78}\n### {name}\n{'=' * 78}", flush=True)
        try:
            report = ana.analyze(entry, float(entry["sm_scale"]),
                                 topk=int(os.environ.get("ANALYSIS_TOPK",
                                                         "2048")))
            print(ana._fmt(report), flush=True)
        except Exception:
            traceback.print_exc()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
