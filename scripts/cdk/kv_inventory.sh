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
# Inventory the weights PVC and NFS model mirrors, so the KV-precision probe
# knows which checkpoint it can actually load. CPU-only: run via ttpu-dev-cpu.
#
# Answers:
#   1. What is already cached on /hf-cache (the persistent weights PVC)?
#   2. How much room is left on it?
#   3. Do the NFS model mirrors hold anything MLA/DSA shaped?
#   4. For any cached model, what does its config say about MLA/DSA
#      (index_topk, kv_lora_rank, qk_rope_head_dim, quantization)?

set -uo pipefail

hr() { printf '\n=== %s ===\n' "$1"; }
try() { "$@" 2>&1 | sed 's/^/  /'; }

hr "weights PVC free space"
try df -h /hf-cache

hr "/hf-cache top level"
try ls -la /hf-cache

hr "cached HF repos"
try find /hf-cache -maxdepth 3 -name "models--*" -printf '%p\n'

hr "per-repo size (may be slow on a cold PVC)"
try bash -c 'for d in /hf-cache/hub/models--*; do [ -d "$d" ] && du -sh "$d" 2>/dev/null; done'

hr "NFS model mirrors"
try ls -la /nfs/models
try ls -la /nfs/hf-models

hr "configs of everything cached"
"${PYTHON:-python3}" - <<'PY' 2>&1 | sed 's/^/  /'
import json, pathlib

# The fields that decide whether a checkpoint exercises the MLA / DSA path
# at all, plus the ones the TPU sparse kernel is specialized for (512+64).
KEYS = [
    "architectures", "model_type", "num_hidden_layers", "hidden_size",
    "kv_lora_rank", "q_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim",
    "v_head_dim", "num_attention_heads",
    "index_topk", "index_n_heads", "index_head_dim",
    "index_topk_freq", "index_topk_pattern", "index_skip_topk_offset",
    "indexer_rope_interleave",
]

roots = [pathlib.Path("/hf-cache"), pathlib.Path("/nfs/models"),
         pathlib.Path("/nfs/hf-models")]
found = 0
for root in roots:
    if not root.exists():
        continue
    for cfg in sorted(root.rglob("config.json")):
        # Skip nested component configs (vision towers, etc.)
        if cfg.parent.name in {"vision_config", "text_config"}:
            continue
        try:
            data = json.loads(cfg.read_text())
        except Exception as exc:
            print(f"{cfg}: UNREADABLE {exc}")
            continue
        found += 1
        print(f"\n{cfg}")
        for k in KEYS:
            if k in data:
                print(f"    {k} = {data[k]}")
        q = data.get("quantization_config")
        if q:
            print(f"    quantization_config.quant_method = {q.get('quant_method')}")
            print(f"    quantization_config.fmt = {q.get('fmt')} weight_block_size = {q.get('weight_block_size')}")
        if "index_topk" in data:
            print("    --> DSA / sparse-MLA model (is_v32 path)")
        elif "kv_lora_rank" in data:
            print("    --> dense MLA model")
if not found:
    print("no config.json found under any root")
PY

hr "done"
