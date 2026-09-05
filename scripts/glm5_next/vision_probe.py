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
"""On-device check for GLM-5.3-Flash's vision tower, stage by stage.

Nothing in the vision workstream had ever run on a TPU when this was written,
so this probe deliberately does *not* start an engine: it builds
``Glm5NextVisionTransformer`` alone, loads the fixture's ``model.visual.*``
weights through the production loader, and diffs every stage boundary against
the fp32 CPU oracle from ``dump_vision_reference.py``.

Standalone rather than end-to-end on purpose. The multimodal wrapper needs a
processor and a tokenizer, which the random fixture does not ship, and an
engine run would fold the tower's error into the text stack's. What is under
test here is only the four things that could plausibly be wrong on this
backend:

  1. **RoPE convention.** transformers builds a ``head_dim``-wide table as
     ``cat(emb, emb)`` and rotates halves; vLLM builds a ``head_dim/2``-wide
     one with ``get_rope(partial_rotary_factor=0.5)`` plus a gather and lets
     ``ApplyRotaryEmb(is_neox_style=True)`` do the halves. Those *should*
     agree, and ``STAGE 2`` checks the tables against each other before any
     matmul runs -- if they disagree, every later number is noise.
  2. **Conv3d / Conv2d lowering** on torch_tpu, which nothing else in this
     repo exercises: ``patch_embed`` and ``downsample``.
  3. **The clamped SwiGLU** and the merger's ``context_dim``.
  4. **The stacked ``gate_proj``/``up_proj`` -> ``gate_up_proj`` mapping**,
     which is a load-time error if wrong in shape and silent if wrong in
     order.

Usage:
    python scripts/glm5_next/vision_probe.py --model <fixture> \\
        --vision-oracle a.npz,b.npz --dtypes bfloat16,float32

``--vision-oracle`` takes a comma-separated list so one job can cover several
packed-sequence shapes. Two are worth having: one under the flash kernel's
128-wide q block (so most of the padded sequence is mask) and one over it (so
the kernel spans more than one block), both with two images so ``cu_seqlens``
segmentation is doing real work.
"""

import argparse
import os
import sys

import numpy as np

#: Per-stage median relative error a correct bf16 port should stay under.
#: On the tiny fixture's two-block tower there is little room for drift to
#: accumulate and this is loose by orders of magnitude. On the real
#: checkpoint's 24 blocks it is not: the worst stage measured is 3.97e-2
#: (``blocks.19.mlp.out``, TPU bf16), against 3.82e-2 for the *same* stage in
#: CPU bf16 and 3.74e-6 in CPU fp32. So the bar is doing real work at depth 24
#: and the margin is 20%, not 1000x -- if a future tower is deeper, measure
#: the fp32 and CPU-bf16 rows before trusting a failure here.
_STAGE_REL_MED_MAX = 5e-2
#: Cosine between device and reference merger output, **averaged** over merged
#: patches. The mean and not the minimum, which is the bar this originally
#: carried, because on the real 24-block tower the minimum measures bf16
#: conditioning rather than the port. Measured, all against the same fp32
#: oracle, all on the *real* checkpoint's tower:
#:
#:   run                        cosine min   cosine mean   pooler rel-med
#:   TPU  bf16, randn pixels      0.9223       0.99542         2.14e-2
#:   CPU  bf16, randn pixels      0.9244       0.99535         2.28e-2
#:   CPU  fp32, randn pixels      1.000000     1.000000        2.39e-6
#:   CPU  bf16, a real image      0.7868       0.97869         2.76e-2
#:
#: The fp32 row is what says the port is right: same code, same weights,
#: agreement to 2e-6. The first two rows differ by 2e-4 in the mean, i.e. the
#: device reproduces the CPU's bf16 result; and the last row says the tail is
#: worse on real images than on noise, so a min-cosine bar tuned on either
#: input is measuring the input. The merger's output is small (|max| ~ 0.18)
#: and arrived at by cancellation, which is why a 2^-8 mantissa moves
#: individual patches by whole percent while the ensemble barely moves.
_COSINE_MEAN_MIN = 0.99
#: Kept as a *reported* number, not a gate: a collapse of the tail below this
#: alongside a healthy mean would be worth looking at.
_COSINE_MIN_REPORT = 0.5
#: How far the rotary table may sit from the reference, per dtype. ``get_rope``
#: caches ``cos``/``sin`` in the *default* dtype, so the bf16 tower's table is
#: genuinely a bf16 table and 2^-9 rounding on values in [-1, 1] is expected.
#: The bar is still decisive: getting the convention wrong -- interleaved
#: instead of halves, or h/w swapped -- moves the table by O(1), three orders
#: of magnitude above this, which is why the wrong-convention distance is
#: printed alongside.
_ROPE_ABS_MAX = {"torch.float32": 1e-5, "torch.bfloat16": 8e-3}


def _stat_row(name: str, got: np.ndarray, want: np.ndarray) -> float:
    """Print one comparison row and return the per-token median relative error.

    The median over tokens rather than the max: a single patch whose value is
    near a clamp boundary can flip by a lot under bf16 while saying nothing
    about whether the port is right, and it would dominate a max metric.
    """
    if got.shape != want.shape:
        print(
            f"{name:26s} SHAPE MISMATCH device={got.shape} "
            f"oracle={want.shape}",
            flush=True)
        return float("inf")
    got = got.astype(np.float64)
    want = want.astype(np.float64)
    diff = np.abs(got - want)
    scale = max(float(np.abs(want).max()), 1e-9)
    per_token = diff.reshape(diff.shape[0], -1).max(-1)
    rel_med = float(np.median(per_token)) / scale
    print(
        f"{name:26s} {float(diff.max()):12.4e} "
        f"{float(diff.max()) / scale:10.3e} {rel_med:10.3e} "
        f"{float(np.percentile(per_token, 95)) / scale:10.3e}",
        flush=True)
    return rel_med


def _load_by_path(path: str):
    """Import a module from an absolute file path."""
    import importlib.util

    name = "_glm5_next_probe_" + os.path.basename(path).removesuffix(".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_cpu_silu_and_mul() -> None:
    """Provide ``_C::silu_and_mul`` so the tower can be *constructed* on CPU.

    ``SiluAndMul.__init__`` grabs ``torch.ops._C.silu_and_mul`` eagerly on any
    CPU platform, and the dev venv has no compiled ``vllm._C``. The tower never
    calls it -- GLM's clamped SwiGLU replaces every use -- but the attribute
    lookup happens at construction time, so without this the ViT cannot be
    built off-device at all. On TPU this function is never reached.
    """
    import torch

    if hasattr(torch.ops._C, "silu_and_mul"):
        return
    lib = torch.library.Library("_C", "FRAGMENT")
    lib.define("silu_and_mul(Tensor! out, Tensor x) -> ()")

    def _impl(out: torch.Tensor, x: torch.Tensor) -> None:
        d = x.shape[-1] // 2
        out.copy_(torch.nn.functional.silu(x[..., :d]) * x[..., d:])

    lib.impl("silu_and_mul", _impl, "CPU")
    # Keep the Library alive; it deregisters the op when collected.
    _install_cpu_silu_and_mul._lib = lib


def _build_config(model: str, max_model_len: int):
    from vllm.engine.arg_utils import EngineArgs

    from vllm_torchtpu.models.vllm import register_models
    register_models()

    engine_args = EngineArgs(
        model=model,
        max_model_len=max_model_len,
        dtype="bfloat16",
        enforce_eager=True,
        trust_remote_code=False,
        load_format="dummy",
        # The tower is built by hand below, so vLLM never needs to resolve the
        # multimodal architecture -- which would want a processor and a
        # tokenizer that the random fixture does not ship.
        hf_overrides={"architectures": ["Glm5NextForCausalLM"]},
        skip_tokenizer_init=True,
    )
    return engine_args.create_engine_config()


def _resolve_model_dir(model: str) -> str:
    """A local directory for ``model``, which may be a HF repo id.

    The real checkpoint is served to the probe as ``zai-org/GLM-5.3-Flash``
    with the 328 GB snapshot already on the weights PVC, so this is a cache
    lookup rather than a download -- but it must not *become* a download of
    328 GB either, hence the metadata-only pattern list.
    """
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download
    return snapshot_download(model, allow_patterns=["*.json"])


def _tower_shards(model_dir: str, prefix: str) -> list[str]:
    """Only the shards that actually hold ``prefix``\\*.

    Reading every shard would work for the 2-block fixture and would mean
    streaming 328 GB off the PVC for the real checkpoint, to keep 1.1 GB of
    it. All 347 ``model.visual.*`` tensors live in one shard there
    (``model-00062-of-00062``), so the index turns a ten-minute read into a
    two-second one.
    """
    import glob
    import json

    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as fh:
            weight_map = json.load(fh)["weight_map"]
        shards = sorted({
            shard
            for name, shard in weight_map.items() if name.startswith(prefix)
        })
        if shards:
            return [os.path.join(model_dir, s) for s in shards]
    return sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))


def _load_tower_weights(tower, model_dir: str) -> None:
    """Load ``model.visual.*`` through the production stacked-param loader."""
    from safetensors import safe_open

    from vllm_torchtpu.models.vllm.glm5_next.multimodal import \
        Glm5NextForConditionalGeneration as MM

    params = dict(tower.named_parameters())
    loaded: set[str] = set()
    prefix = MM._TOWER_PREFIX
    seen = 0
    shards = _tower_shards(model_dir, prefix)
    print(
        f"tower shards                {len(shards)}: "
        f"{[os.path.basename(s) for s in shards]}",
        flush=True)
    for shard in shards:
        with safe_open(shard, framework="pt") as fh:
            for name in fh.keys():
                if not name.startswith(prefix):
                    continue
                seen += 1
                MM._load_tower_weight(name[len(prefix):], fh.get_tensor(name),
                                      params, loaded)

    missing = sorted(set(params) - {n[len("visual."):] for n in loaded})
    print(f"tower tensors in checkpoint {seen}", flush=True)
    print(f"tower params filled         {len(loaded)}/{len(params)}",
          flush=True)
    if missing:
        raise RuntimeError(f"unfilled tower params: {missing}")


def _register_taps(tower, captured: dict) -> list:
    """Hook the same stage boundaries ``dump_vision_reference.py`` records."""
    import torch

    handles = []

    def tap(name: str, squeeze_batch: bool = False):

        def hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            t = t.detach().to(torch.float32).cpu()
            if squeeze_batch and t.dim() == 3:
                # The blocks run on [s, b=1, c]; the oracle is [s, c].
                t = t.squeeze(1)
            captured[name] = t.numpy()

        return hook

    handles.append(
        tower.patch_embed.register_forward_hook(tap("patch_embed.out")))
    for i, blk in enumerate(tower.blocks):
        handles.append(
            blk.register_forward_hook(
                tap(f"blocks.{i}.out", squeeze_batch=True)))
        handles.append(
            blk.norm1.register_forward_hook(
                tap(f"blocks.{i}.norm1.out", squeeze_batch=True)))
        handles.append(
            blk.attn.register_forward_hook(
                tap(f"blocks.{i}.attn.out", squeeze_batch=True)))
        handles.append(
            blk.norm2.register_forward_hook(
                tap(f"blocks.{i}.norm2.out", squeeze_batch=True)))
        handles.append(
            blk.mlp.register_forward_hook(
                tap(f"blocks.{i}.mlp.out", squeeze_batch=True)))
    handles.append(
        tower.post_layernorm.register_forward_hook(
            tap("post_layernorm.out", squeeze_batch=True)))
    handles.append(
        tower.merger.proj.register_forward_hook(tap("merger.proj.out")))
    handles.append(
        tower.merger.extra_activation_func.register_forward_hook(
            tap("merger.act1.out")))
    handles.append(
        tower.merger.down_proj.register_forward_hook(tap("merger.out")))
    return handles


def _check_rope(tower, grid_thw, oracle, bar: float) -> bool:
    """Compare the rotary tables before anything multiplies by them.

    This is the flagged risk: two completely different constructions that must
    produce the same numbers. The oracle stores the full ``head_dim``-wide
    ``cat(emb, emb)`` table; vLLM's is the ``head_dim/2``-wide half of it that
    ``ApplyRotaryEmb`` expects, so the reference is sliced to match.
    """
    import torch

    cos, sin, pos_ids = tower.rot_pos_emb(grid_thw)
    cos = cos.detach().to(torch.float32).cpu().numpy()
    sin = sin.detach().to(torch.float32).cpu().numpy()
    pos_ids = pos_ids.detach().cpu().numpy().astype(np.int64)

    ref_pos = oracle["position_ids"]
    pos_ok = (pos_ids.shape == ref_pos.shape and bool(
        (pos_ids == ref_pos).all()))
    print(
        f"position_ids               device={pos_ids.shape} "
        f"oracle={ref_pos.shape} identical={pos_ok}",
        flush=True)
    if not pos_ok:
        n = min(len(pos_ids), len(ref_pos))
        bad = np.flatnonzero((pos_ids[:n] != ref_pos[:n]).any(-1))
        if len(bad):
            i = int(bad[0])
            print(
                f"  first mismatch at token {i}: device={pos_ids[i]} "
                f"oracle={ref_pos[i]}",
                flush=True)

    half = cos.shape[-1]
    ref_cos = oracle["rope.cos"][:, :half]
    ref_sin = oracle["rope.sin"][:, :half]
    dcos = float(np.abs(cos - ref_cos).max())
    dsin = float(np.abs(sin - ref_sin).max())
    print(
        f"rope cos/sin               max|diff| {dcos:.3e} / {dsin:.3e} "
        f"(bar {bar:.0e})",
        flush=True)

    # The contrast that makes the bar meaningful, printed pass or fail: how
    # far off a *wrong* frequency order would be. If the two numbers are the
    # same order of magnitude, the bar is not testing anything.
    scrambled = float(np.abs(cos - ref_cos[:, ::-1]).max())
    print(f"  wrong-freq-order would be {scrambled:.3e}", flush=True)

    ok = pos_ok and dcos <= bar and dsin <= bar
    print(f"  -> {'ok' if ok else 'FAIL'}", flush=True)
    return ok


def run_one(vllm_config, args, oracle, device, dtype, label: str) -> bool:
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.model_loader.utils import \
        process_weights_after_loading
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_torchtpu.models.vllm.glm5_next.vision import \
        Glm5NextVisionTransformer

    hf_config = vllm_config.model_config.hf_config
    print("=" * 60, flush=True)
    print(
        f"RUN dtype={dtype} device={device} oracle={label} "
        f"({oracle['pixel_values'].shape[0]} patches)",
        flush=True)

    # The engine builds the tower with the checkpoint's FP8 config, not with
    # `None`, and relies on every `visual.*` module being in
    # `modules_to_not_convert` for that to be a no-op. Passing `None` here --
    # which is what this probe did originally -- tests the arithmetic but
    # skips that entire question, and an FP8 method that quietly claims a
    # tower layer produces NaN in the engine while the probe stays green.
    quant_config = vllm_config.quant_config if args.quant else None
    if quant_config is not None:
        # The engine does this before building anything, and without it
        # `is_layer_skipped` cannot expand ``mlp.gate_up_proj`` into the
        # ``mlp.gate_proj``/``mlp.up_proj`` pair the ignore list actually
        # names -- so the fused layers would be quantized here and not in the
        # engine, which is the opposite of what this run is for.
        from vllm.model_executor.model_loader.utils import \
            configure_quant_config

        from vllm_torchtpu.models.vllm.glm5_next.multimodal import \
            Glm5NextForConditionalGeneration as MM
        configure_quant_config(quant_config, MM)
    print(f"quant_config                {type(quant_config).__name__}",
          flush=True)
    with set_current_vllm_config(vllm_config), set_default_torch_dtype(dtype):
        with torch.device(device):
            tower = Glm5NextVisionTransformer(
                hf_config.text_config,
                hf_config.vision_config,
                quant_config=quant_config,
                prefix="visual",
            )
    tower.eval()
    print(
        f"tower built: {sum(p.numel() for p in tower.parameters())} params "
        f"in {len(list(tower.named_parameters()))} tensors",
        flush=True)
    # Which quant method each tower linear actually got. Anything other than
    # ``UnquantizedLinearMethod`` means an FP8 path is live on a tower the
    # checkpoint stores in bf16.
    methods: dict[str, int] = {}
    for module in tower.modules():
        method = getattr(module, "quant_method", None)
        if method is not None:
            methods[type(method).__name__] = methods.get(
                type(method).__name__, 0) + 1
    print(f"linear quant methods        {methods}", flush=True)
    # Which attention actually runs. `CustomOp.register_oot` swaps in
    # `TpuMMEncoderAttention` (the Pallas flash kernel) only if
    # `vllm_torchtpu.layers.vllm.vision_attention` got imported -- which
    # happens inside `TpuPlatform.check_and_update_config`. If that import is
    # ever lost, the tower silently falls back to SDPA and this probe would
    # pass while testing nothing about the kernel. So print it.
    # The class alone is not proof: `TpuMMEncoderAttention.forward_oot` falls
    # back to SDPA for GQA or ragged q/kv, so print the *bound method*
    # `CustomOp` actually dispatched to. `forward_oot` here means the Pallas
    # flash kernel; `forward_native` means this probe verified nothing about
    # the kernel.
    attn = tower.blocks[0].attn
    print(
        f"encoder attention impl      {type(attn.attn).__name__}"
        f".{attn.attn._forward_method.__name__} "
        f"(backend {tower.attn_backend})",
        flush=True)
    print(
        f"conv layers                 patch_embed.proj="
        f"{type(tower.patch_embed.proj).__name__} downsample="
        f"{type(tower.downsample).__name__}",
        flush=True)

    _load_tower_weights(tower, _resolve_model_dir(args.model))
    # The same post-load pass the real loader runs. Not optional even for an
    # unquantized bf16 tower: it is where `MMEncoderAttention` finishes its own
    # setup, and on the CPU platform where the linears get their GEMM impl.
    with set_current_vllm_config(vllm_config), set_default_torch_dtype(dtype):
        process_weights_after_loading(tower, vllm_config.model_config, device)

    grid_thw = torch.tensor(oracle["grid_thw"], dtype=torch.int32)
    pixel_values = torch.as_tensor(oracle["pixel_values"]).to(dtype)

    print("-" * 60, flush=True)
    print("STAGE 2: rotary tables (no matmul involved yet)", flush=True)
    rope_ok = _check_rope(tower, grid_thw, oracle, _ROPE_ABS_MAX[str(dtype)])

    captured: dict[str, np.ndarray] = {}
    handles = _register_taps(tower, captured)
    with torch.inference_mode():
        out = tower(pixel_values.to(device), grid_thw)
        out = out.detach().to(torch.float32).cpu().numpy()
    for handle in handles:
        handle.remove()

    print("-" * 60, flush=True)
    print("STAGE 3: per-stage divergence vs the fp32 CPU oracle", flush=True)
    print(
        f"{'stage':26s} {'max|diff|':>12s} {'rel-max':>10s} "
        f"{'rel-med':>10s} {'rel-p95':>10s}",
        flush=True)

    order = ["patch_embed.out"]
    for i in range(len(tower.blocks)):
        order += [
            f"blocks.{i}.norm1.out", f"blocks.{i}.attn.out",
            f"blocks.{i}.norm2.out", f"blocks.{i}.mlp.out", f"blocks.{i}.out"
        ]
    order += [
        "post_layernorm.out", "merger.proj.out", "merger.act1.out",
        "merger.out"
    ]

    worst = 0.0
    worst_stage = ""
    for key in order:
        if key not in captured or key not in oracle.files:
            print(f"{key:26s} (not captured)", flush=True)
            continue
        rel_med = _stat_row(key, captured[key], oracle[key])
        if rel_med > worst:
            worst, worst_stage = rel_med, key

    print("-" * 60, flush=True)
    print("STAGE 4: tower output (what gets scattered into the text stream)",
          flush=True)
    ref = oracle["pooler_output"].astype(np.float64)
    dev = out.astype(np.float64)
    if dev.shape != ref.shape:
        print(f"output SHAPE MISMATCH device={dev.shape} oracle={ref.shape}",
              flush=True)
        return False
    rel_med = _stat_row("pooler_output", out, oracle["pooler_output"])
    if rel_med > worst:
        worst, worst_stage = rel_med, "pooler_output"
    denom = np.linalg.norm(dev, axis=-1) * np.linalg.norm(ref, axis=-1)
    cos = (dev * ref).sum(-1) / np.maximum(denom, 1e-30)
    print(
        f"per-patch cosine  min/p5/mean  {float(cos.min()):.6f} / "
        f"{float(np.percentile(cos, 5)):.6f} / {float(cos.mean()):.6f}  "
        f"(mean bar {_COSINE_MEAN_MIN}, min reported only)",
        flush=True)
    print(
        f"output |max|      dev/ref   {float(np.abs(dev).max()):.4f} / "
        f"{float(np.abs(ref).max()):.4f}",
        flush=True)

    stage_ok = worst <= _STAGE_REL_MED_MAX
    print(
        f"worst stage rel-med {worst:.3e} at {worst_stage or '-'} "
        f"(bar {_STAGE_REL_MED_MAX:.0e}) -> {'ok' if stage_ok else 'FAIL'}",
        flush=True)
    cos_ok = float(cos.mean()) >= _COSINE_MEAN_MIN
    if float(cos.min()) < _COSINE_MIN_REPORT:
        print(
            f"NOTE: {int((cos < _COSINE_MIN_REPORT).sum())} patch(es) below "
            f"cosine {_COSINE_MIN_REPORT}; the mean still gates, but a tail "
            "this long has not been seen in bf16 and is worth a look",
            flush=True)
    ok = bool(rope_ok and stage_ok and cos_ok)
    print(f"dtype={dtype} oracle={label}: {'PASS' if ok else 'FAIL'}",
          flush=True)

    if args.dump:
        # One file per (dtype, oracle): the recipe ships these back, and a
        # later hypothesis can then be tested against the real device tensors
        # instead of costing another job.
        root, ext = os.path.splitext(args.dump)
        tag = f"{str(dtype).split('.')[-1]}_{os.path.splitext(label)[0]}"
        np.savez(f"{root}_{tag}{ext}", **captured, pooler_output=out)
        print(f"dumped device activations to {root}_{tag}{ext}", flush=True)
        # Also copy into --artifact-dir, if given, under a name a batch
        # harness can predict: the per-(dtype, oracle) names above cannot be
        # guessed by whatever collects the run's outputs.
        out_dir = getattr(args, "artifact_dir", None)
        if out_dir and os.path.isdir(out_dir):
            np.savez(os.path.join(out_dir, f"activations_{tag}{ext}"),
                     **captured,
                     pooler_output=out)
            print(f"  and to {out_dir}/activations_{tag}{ext}", flush=True)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vision-oracle",
                        default=None,
                        help="npz from dump_vision_reference.py; defaults to "
                        "vision_oracle.npz next to --model")
    # Accepted for recipe compatibility; the tower has no KV cache and this
    # probe never starts an engine.
    parser.add_argument("--oracle", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--device", default="tpu", choices=["tpu", "cpu"])
    parser.add_argument("--dtypes",
                        default="bfloat16",
                        help="comma-separated; float32 as a second pass "
                        "separates bf16 rounding from a wiring bug")
    parser.add_argument("--dump",
                        default="/tmp/glm5_next_activations.npz",
                        help="where to write the device activations")
    parser.add_argument("--artifact-dir",
                        default=None,
                        help="also copy each dump here under a predictable "
                        "name, for a harness that collects outputs")
    parser.add_argument("--quant",
                        action="store_true",
                        help="build the tower with the checkpoint's own "
                        "quant config, the way the engine does, instead "
                        "of with None")
    parser.add_argument("--cpu-stub",
                        action="store_true",
                        help="fake torch_tpu (and torchvision) so the whole "
                        "probe can be rehearsed off-device")
    args = parser.parse_args()

    if args.cpu_stub:
        # The test suite's stubs, loaded by path: `tests` is not a package,
        # so there is no import path to them that does not depend on how the
        # caller seeded sys.path.
        _repo = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        _load_by_path(os.path.join(_repo, "tests", "stubs",
                                   "torch_tpu.py")).install()
        _load_by_path(os.path.join(_repo, "tests", "stubs",
                                   "torchvision.py")).install()
        _install_cpu_silu_and_mul()

    import torch

    if args.tensor_parallel_size != 1:
        # The tower's linears would shard, and this probe compares against a
        # single-rank oracle. Refuse rather than report a wrong-looking diff.
        raise SystemExit("vision_probe only supports --tensor-parallel-size 1")

    oracle_paths = (args.vision_oracle.split(",") if args.vision_oracle else
                    [os.path.join(args.model, "vision_oracle.npz")])
    oracles = []
    for path in oracle_paths:
        path = path.strip()
        if not os.path.exists(path):
            raise SystemExit(
                f"no vision oracle at {path}; run "
                "scripts/glm5_next/dump_vision_reference.py first")
        oracle = np.load(path)
        oracles.append((os.path.basename(path), oracle))
        print(
            f"oracle {path}: grid {oracle['grid_thw'].tolist()}, "
            f"{oracle['pixel_values'].shape[0]} patches -> "
            f"{oracle['pooler_output'].shape[0]} merged tokens",
            flush=True)

    print("=" * 60, flush=True)
    print("STAGE 1: config + distributed init", flush=True)
    vllm_config = _build_config(args.model, args.max_model_len)
    vision_config = vllm_config.model_config.hf_config.vision_config
    print(
        f"vision_config depth={vision_config.depth} "
        f"hidden={vision_config.hidden_size} "
        f"heads={vision_config.num_heads} "
        f"out_hidden={vision_config.out_hidden_size} "
        f"proj_inter={vision_config.projection_intermediate_size} "
        f"swiglu_limit={vision_config.swiglu_limit}",
        flush=True)

    from vllm.config import set_current_vllm_config
    from vllm.distributed import (init_distributed_environment,
                                  initialize_model_parallel)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "12357")
    # Inside the config context: `initialize_model_parallel` reads the ambient
    # vLLM config and asserts if there is none.
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(world_size=1,
                                     rank=0,
                                     distributed_init_method="env://",
                                     local_rank=0,
                                     backend="gloo")
        initialize_model_parallel(1, 1)

    device = torch.device(args.device)
    results: dict[tuple[str, str], bool] = {}
    for name in args.dtypes.split(","):
        name = name.strip()
        dtype = getattr(torch, name)
        for label, oracle in oracles:
            results[(name, label)] = run_one(vllm_config, args, oracle, device,
                                             dtype, label)

    print("=" * 60, flush=True)
    for (name, label), ok in results.items():
        print(f"  {name:10s} {label:28s} {'PASS' if ok else 'FAIL'}",
              flush=True)
    # bfloat16 is the shipping configuration, so it alone decides the verdict;
    # a float32 pass is diagnostic (it separates rounding from wiring) and on
    # TPU is not even a true fp32 matmul.
    bf16 = [ok for (name, _), ok in results.items() if name == "bfloat16"]
    verdict = all(bf16) if bf16 else all(results.values())
    print(f"RESULT: {'PASS' if verdict else 'FAIL'}", flush=True)
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
    main()
