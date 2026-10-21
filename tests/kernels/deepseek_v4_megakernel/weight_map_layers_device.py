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
"""`build_weight_operands` at the real layer count, every manifest operand.

    python -m tests.kernels.deepseek_v4_megakernel.weight_map_layers_device [cpu]

A fake served tree with the real attribute paths, the real 61-layer
HCA/CSA pattern, 3 hash layers and R = 16, at reduced widths (hidden 512,
q_lora 256, o_lora 256, 2 local experts; head_dim 512 / 128 heads / indexer
128 as served, which the manifest's attention stacks fix). On device every
operand goes through `torch_tpu_run_jax` (the packed streams take up to
884 tensors) and is compared bit for bit with the same JAX fn on host
copies; `cpu` runs the host bridge only. Prints RESULT lines.
"""

import sys
import time
import types

import torch

from vllm_torchtpu.kernels.deepseek_v4_megakernel import dense_config as dc
from vllm_torchtpu.kernels.deepseek_v4_megakernel import kernel_interface as KI
from vllm_torchtpu.kernels.deepseek_v4_megakernel import vllm_adapter as VA

from .test_weight_map import host_run_jax, make_tree

RATIOS = tuple(dc.DSV4_PRO_RATIOS)
R = 16
HF = types.SimpleNamespace(
    num_hidden_layers=len(RATIOS), hidden_size=512, q_lora_rank=256, head_dim=512,
    num_attention_heads=128, o_groups=16, o_lora_rank=256,
    moe_intermediate_size=1024, n_routed_experts=128, index_n_heads=8,
    index_head_dim=128, hc_mult=4, vocab_size=1024,
    compress_ratios=list(RATIOS) + [0], num_hash_layers=3, num_experts_per_tok=6,
    qk_rope_head_dim=64,
)  # fmt: skip
DENSE = dc.DenseChainConfig(
    hidden=512, q_lora=256, head_dim=512, n_heads=128, o_lora=256, moe_inter=1024,
    n_experts=128, idx_heads=8, idx_head_dim=128, ratios=RATIOS, tp=R,
)  # fmt: skip


def spec():
    return KI.ModelSpec(
        hidden_size=512, num_layers=len(RATIOS), compress_ratios=RATIOS,
        dtype="bfloat16", q=7, mode="dp", axis_name="ep", num_ranks=R,
        aux_layers=(59, 60, 61), vocab_size=1024, num_hash_layers=3,
        n_routed_experts=128,
    )  # fmt: skip


def main(device: str) -> int:
    tree = make_tree(hf=HF, e_local=2, device=device)
    man = KI.weight_manifest(spec(), DENSE)
    calls = []

    def run(name, fn, *tensors):
        calls.append((name, len(tensors)))
        if device == "cpu":
            return host_run_jax(name, fn, *tensors)
        got = VA.torch_tpu_run_jax(name, fn, *tensors)
        want = host_run_jax(name, fn, *[t.cpu() for t in tensors])
        g = got.cpu()
        same = g.shape == want.shape and torch.equal(
            g.view(torch.uint8).flatten(), want.view(torch.uint8).flatten()
        )
        print(f"RESULT op {name} inputs={len(tensors)} {tuple(g.shape)} "
              f"{'bitexact' if same else 'DIFF'}", flush=True)  # fmt: skip
        bad.append(not same)
        return got

    bad: list[bool] = []
    t0 = time.perf_counter()
    ops = VA.build_weight_operands(tree, spec(), 5, run, dense=DENSE)
    dt = time.perf_counter() - t0
    n_served = sum(1 for e in man if e.rule == "served")
    print(f"RESULT operands={len(ops)} manifest={len(man)} served={n_served} "
          f"built={len(calls)} max_inputs={max(n for _, n in calls)} "
          f"seconds={dt:.1f}", flush=True)  # fmt: skip
    ok = len(ops) == len(man) and len(calls) == len(man) - n_served and not any(bad)
    print(f"RESULT ok={ok}", flush=True)
    return int(not ok)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "tpu"))
