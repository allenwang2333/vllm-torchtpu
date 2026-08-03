#!/bin/bash
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
#
# Does `VllmTPUMLAAttention.__init__`'s positional super() call still line up
# with this vLLM's `MLAAttention.__init__`?
#
# The subclass passes cache_config..indexer positionally. Upstream has since
# inserted `dcp_q_replicate` after `kv_b_proj`, which shifts every later
# argument by one. Run with SHADOW_VLLM=1 (the NFS checkout) and 0 (the
# image's pinned copy) to see which vLLM revisions the branch actually works
# against.
#
#   cdk job create ttpu-dev-cpu SCRIPT=<path>/mla_sig_check SHADOW_VLLM=0
set -uo pipefail

python3 - <<'PY'
import inspect
import vllm
from vllm.model_executor.layers.attention.mla_attention import MLAAttention

print(f"vllm {getattr(vllm, '__version__', '?')} from {vllm.__file__}")
params = list(inspect.signature(MLAAttention.__init__).parameters)
print(f"MLAAttention.__init__ positional order ({len(params)} params):")
for i, p in enumerate(params):
    print(f"  [{i:2}] {p}")

# The order the TPU subclass supplies positionally, after `self`.
supplied = ["num_heads", "scale", "qk_nope_head_dim", "qk_rope_head_dim",
            "v_head_dim", "q_lora_rank", "kv_lora_rank", "kv_b_proj",
            "cache_config", "quant_config", "prefix", "attn_backend",
            "use_sparse", "indexer"]
print("\npositional call alignment (subclass value -> parameter it binds to):")
ok = True
for i, name in enumerate(supplied):
    target = params[i + 1] if i + 1 < len(params) else "<overflow>"
    flag = "" if target == name else "   <-- MISMATCH"
    if target != name:
        ok = False
    print(f"  {name:<18} -> {target}{flag}")
print("\nRESULT:", "aligned" if ok else "SHIFTED - positional super() is broken")
PY
