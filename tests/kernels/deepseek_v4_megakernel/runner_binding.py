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
"""DeepSeek-V4 decoder layers with KV caches bound the way the runner binds
them, for megakernel adapter tests (no vLLM needed)."""

from types import SimpleNamespace as NS

import torch

SMALL_CACHE_SHAPES = dict(
    nope=(8, 128), rope=(2, 128), idx=(2, 4, 256), hca=(4, 4, 128)
)


def runner_bound_model(
    ratios=(128, 4), n_draft=1, num_blocks=4, device="cpu", shapes=None
):
    """Decoder layers whose caches are bound the way the real runner binds
    them: `kv_cache_dsv4._initialize_ds_v4_kv_cache`'s overlay rules (ported
    structurally, small shapes) and vLLM's per-layer `kv_cache` binding by
    static-forward-context name. Returns (target layers, draft layers,
    kv_cache_groups, kv_caches by layer name).

    - CSA `*.attn`: a (nope int32, rope int32) pair
    - its `*.compressor.state_cache`: the SAME pair object
    - `*.indexer.k_cache`: uint8; its compressor state: the same tensor
    - HCA `*.attn`: uint8; its state: the i-th CSA NoPE array (by position)
    - SWA caches (target and draft, one group): the CSA NoPE array at their
      position in the group, overflow arrays shaped like NoPE past the end

    `shapes`: per-page dims of each array kind (`SMALL_CACHE_SHAPES` keys).
    """
    shapes = shapes or SMALL_CACHE_SHAPES

    def mk(shape, dtype):
        return torch.zeros(shape, dtype=dtype).to(device)

    names = []  # (layer name, module) for every cache-holding module
    targets, drafts = [], []
    for i, ratio in enumerate(ratios):
        p = f"model.layers.{i}.attn"
        swa = NS(prefix=f"{p}.swa_cache", kv_cache=None, block_size=128)
        state = NS(prefix=f"{p}.compressor.state_cache", kv_cache=None)
        attn = NS(compress_ratio=ratio, custom_prefix=p, prefix=p, kv_cache=None,
                  swa_cache_layer=swa, compressor=NS(state_cache=state),
                  indexer=None)  # fmt: skip
        names += [(p, attn), (swa.prefix, swa), (state.prefix, state)]
        if ratio == 4:
            k = NS(prefix=f"{p}.indexer.k_cache", kv_cache=None)
            ist = NS(prefix=f"{p}.indexer.compressor.state_cache", kv_cache=None)
            attn.indexer = NS(k_cache=k, compressor=NS(state_cache=ist))
            names += [(k.prefix, k), (ist.prefix, ist)]
        targets.append(NS(attn=attn))
    for j in range(n_draft):
        p = f"drafter.layers.{j}.attn"
        swa = NS(prefix=f"{p}.swa_cache", kv_cache=None, block_size=128)
        attn = NS(compress_ratio=1, custom_prefix=p, swa_cache_layer=swa)
        names.append((swa.prefix, swa))
        drafts.append(NS(attn=attn))

    by_role: dict[str, list[str]] = {}
    for n, _ in names:
        if n.endswith(".swa_cache"):
            role = "swa"
        elif n.endswith(".indexer.compressor.state_cache"):
            role = "idx_state"
        elif n.endswith(".compressor.state_cache"):
            role = "state"
        elif n.endswith(".indexer.k_cache"):
            role = "idx"
        else:
            role = "csa" if ratios[int(n.split(".")[2])] == 4 else "hca"
        by_role.setdefault(role, []).append(n)
    order = ("csa", "hca", "idx", "swa", "state", "idx_state")
    groups = [NS(layer_names=by_role[r]) for r in order if r in by_role]

    kv: dict[str, object] = {}
    nope_hosts = []
    for n in by_role.get("csa", []):
        nope = mk((num_blocks, *shapes["nope"]), torch.int32)
        kv[n] = (nope, mk((num_blocks, *shapes["rope"]), torch.int32))
        nope_hosts.append(nope)
    for n in by_role.get("idx", []):
        kv[n] = mk((num_blocks, *shapes["idx"]), torch.uint8)
    for n in by_role.get("hca", []):
        kv[n] = mk((num_blocks, *shapes["hca"]), torch.uint8)
    hosts = list(nope_hosts)

    def host_at(pos):
        while pos >= len(hosts):
            hosts.append(mk(tuple(nope_hosts[0].shape), nope_hosts[0].dtype))
        return hosts[pos]

    for pos, n in enumerate(by_role.get("swa", [])):
        kv[n] = host_at(pos)
    hca_states = []
    for n in by_role.get("state", []) + by_role.get("idx_state", []):
        base = n.removesuffix(".compressor.state_cache")
        kv_layer = base + ".k_cache" if base.endswith(".indexer") else base
        if kv_layer in by_role.get("hca", []):
            hca_states.append(n)
        else:
            kv[n] = kv[kv_layer]
    for pos, n in enumerate(hca_states):
        kv[n] = host_at(pos)
    for n, module in names:
        module.kv_cache = kv[n]
    return targets, drafts, groups, kv
