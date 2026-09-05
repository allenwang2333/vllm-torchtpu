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
"""Dump an fp32 CPU oracle for GLM-5.3-Flash's *vision tower*.

The text-side counterpart is ``dump_reference.py``; this is the same idea for
``model.visual``. It runs transformers' ``Glm5NextVisionModel`` on a fixed
synthetic ``pixel_values`` / ``image_grid_thw`` and records every stage
boundary, so a device run can say "block 0 already diverges" instead of "the
output is wrong".

The stage list is chosen around the things most likely to break in the port,
in the order the forward hits them:

  * ``rope.cos`` / ``rope.sin`` -- the *whole* head_dim-wide table, i.e. after
    transformers' ``cat(emb, emb)``. vLLM builds the same table a completely
    different way (``get_rope(partial_rotary_factor=0.5)`` plus a gather), so
    this is the one tensor that can be compared without running a single
    matmul.
  * ``patch_embed.out`` -- Conv3d, which is untested on torch_tpu.
  * ``blocks.N.{norm1,attn,norm2,mlp,out}`` -- localises a divergence to the
    attention or the clamped SwiGLU.
  * ``post_layernorm.out`` and ``downsample.out`` -- Conv2d, likewise untested.
  * ``merger.*`` -- the bias-free projection stack whose ``context_dim`` comes
    from ``projection_intermediate_size``, not the text config.

``last_hidden_state`` is the *downsample* output and ``pooler_output`` is the
merger output; vLLM's tower returns only the latter, which is what actually
gets scattered into the text embeddings.

Usage:
    PYTHONPATH=$GLM5_NEXT_TRANSFORMERS_SRC python scripts/glm5_next/dump_vision_reference.py \
        --model /path/to/glm5_next_tiny \
        --out /path/to/glm5_next_tiny/vision_oracle.npz
"""

import argparse

import numpy as np
import torch
from transformers.models.glm5_next.configuration_glm5_next import \
    Glm5NextConfig
from transformers.models.glm5_next.modeling_glm5_next import \
    Glm5NextVisionModel


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).cpu().numpy()


def parse_grid(spec: str) -> list[list[int]]:
    """``"1,4,4;1,2,6"`` -> ``[[1, 4, 4], [1, 2, 6]]``."""
    grid = []
    for item in spec.split(";"):
        t, h, w = (int(x) for x in item.split(","))
        grid.append([t, h, w])
    return grid


def validate_grid(grid: list[list[int]], merge_size: int) -> None:
    """Reject a grid the tower would silently mis-shape.

    ``downsample`` reshapes the token axis to ``(-1, m, m, C)``, which is only
    the intended spatial regrouping when every image's ``h`` and ``w`` are
    multiples of ``m``. A grid that violates it still *runs* -- the view
    succeeds as long as ``t*h*w`` is divisible by ``m*m`` -- and merges
    patches from the wrong places.
    """
    for t, h, w in grid:
        if h % merge_size or w % merge_size:
            raise ValueError(
                f"grid ({t},{h},{w}) is not a multiple of spatial_merge_size "
                f"{merge_size} in h and w; the downsample regrouping would be "
                "silently wrong")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--grid",
        default="1,4,4;1,2,6",
        help="semicolon-separated t,h,w per image. Two images by default so "
        "the packed-attention cu_seqlens split is actually exercised: "
        "with one image a broken segment mask still passes.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = Glm5NextConfig.from_pretrained(args.model)
    vision_config = config.vision_config
    grid = parse_grid(args.grid)
    validate_grid(grid, vision_config.spatial_merge_size)

    torch.manual_seed(args.seed)

    # Eager attention: the reference must be the plain math, not a fused
    # kernel whose own numerics are what we would then be comparing against.
    vision_config._attn_implementation = "eager"
    model = Glm5NextVisionModel._from_config(vision_config,
                                             dtype=torch.float32)
    model.eval()

    state = _load_tower_state(args.model)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"tower state_dict mismatch: missing={missing} "
                           f"unexpected={unexpected}")

    grid_thw = torch.tensor(grid, dtype=torch.long)
    num_patches = int(grid_thw.prod(-1).sum())
    patch_features = (vision_config.in_channels *
                      vision_config.temporal_patch_size *
                      vision_config.patch_size**2)
    # Unit-scale normal, i.e. roughly what a normalised image looks like after
    # the processor's mean/std. Feeding raw 0..255 would put the SwiGLU clamp
    # far into saturation and hide a wrong limit.
    pixel_values = torch.randn(num_patches,
                               patch_features,
                               dtype=torch.float32)

    captured: dict[str, np.ndarray] = {}
    handles = []

    def tap(name: str):

        def hook(_mod, _inp, out):
            captured[name] = _np(out[0] if isinstance(out, tuple) else out)

        return hook

    handles.append(
        model.patch_embed.register_forward_hook(tap("patch_embed.out")))
    handles.append(
        model.post_layernorm.register_forward_hook(tap("post_layernorm.out")))
    handles.append(
        model.downsample.register_forward_hook(tap("downsample.raw")))
    for i, blk in enumerate(model.blocks):
        handles.append(blk.register_forward_hook(tap(f"blocks.{i}.out")))
        handles.append(
            blk.norm1.register_forward_hook(tap(f"blocks.{i}.norm1.out")))
        handles.append(
            blk.attn.register_forward_hook(tap(f"blocks.{i}.attn.out")))
        handles.append(
            blk.norm2.register_forward_hook(tap(f"blocks.{i}.norm2.out")))
        handles.append(
            blk.mlp.register_forward_hook(tap(f"blocks.{i}.mlp.out")))
    handles.append(
        model.merger.proj.register_forward_hook(tap("merger.proj.out")))
    handles.append(
        model.merger.act1.register_forward_hook(tap("merger.act1.out")))
    handles.append(
        model.merger.down_proj.register_forward_hook(tap("merger.out")))

    # The rotary table is a local, not a module output, so recompute it the
    # same way `forward` does rather than hooking anything.
    from transformers.vision_utils import get_vision_position_ids
    position_ids = get_vision_position_ids(grid_thw,
                                           vision_config.spatial_merge_size)
    rotary_emb = model.rotary_pos_emb(position_ids)
    emb = torch.cat((rotary_emb, rotary_emb), dim=-1)

    with torch.inference_mode():
        out = model(hidden_states=pixel_values, grid_thw=grid_thw)

    for handle in handles:
        handle.remove()

    captured["pixel_values"] = _np(pixel_values)
    captured["grid_thw"] = grid_thw.numpy().astype(np.int64)
    captured["position_ids"] = position_ids.numpy().astype(np.int64)
    captured["rope.cos"] = _np(emb.cos())
    captured["rope.sin"] = _np(emb.sin())
    captured["last_hidden_state"] = _np(out.last_hidden_state)
    captured["pooler_output"] = _np(out.pooler_output)
    captured.pop("downsample.raw", None)  # NCHW; last_hidden_state is the view

    np.savez(args.out, **captured)
    print(f"wrote {args.out} with {len(captured)} arrays")
    for key in sorted(captured):
        print(f"  {key:26s} {captured[key].shape}")


def _load_tower_state(model_dir: str) -> dict[str, torch.Tensor]:
    """Pull ``model.visual.*`` out of the checkpoint, prefix stripped."""
    import glob
    import os

    from safetensors.torch import load_file

    prefix = "model.visual."
    state: dict[str, torch.Tensor] = {}
    for shard in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        for key, value in load_file(shard).items():
            if key.startswith(prefix):
                state[key[len(prefix):]] = value.to(torch.float32)
    if not state:
        raise RuntimeError(f"no {prefix}* tensors in {model_dir}")
    return state


if __name__ == "__main__":
    main()
