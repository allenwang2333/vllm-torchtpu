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
# Can lm_eval load GLM-5.2's tokenizer in this image?
#
# The full-set eval died in AutoTokenizer.from_pretrained: transformers 5.8.0
# validates `layer_types` against a fixed allow-list and GLM-5.2 declares
# `deepseek_sparse_attention`, which is not on it. AutoTokenizer pulls
# AutoConfig, so the model config poisons a tokenizer-only load.
#
# vLLM itself is unaffected -- it loaded and served the model. This is purely
# an lm_eval-side problem, so it can be settled on a CPU pod instead of
# spending another 55-minute TPU admission to find out.
set -uo pipefail
REPO="${MODEL_ID:-nvidia/GLM-5.2-NVFP4}"
BASE="https://huggingface.co/${REPO}/resolve/main"
D=/tmp/glm_tokenizer
mkdir -p "$D"
for f in tokenizer.json tokenizer_config.json chat_template.jinja special_tokens_map.json; do
  curl -sfL "$BASE/$f" -o "$D/$f" && echo "### got $f" || echo "### absent $f"
done
python3 - <<'PY'
import traceback, transformers
from transformers import AutoTokenizer
print(f"### transformers {transformers.__version__}")

def attempt(label, src, **kw):
    try:
        t = AutoTokenizer.from_pretrained(src, **kw)
        print(f"### {label}: OK  (class={type(t).__name__}, vocab={len(t)})")
        return t
    except Exception as e:
        print(f"### {label}: FAIL {type(e).__name__}: {str(e).splitlines()[0][:160]}")
        return None

# 1. What the eval does today.
attempt("repo id (current behaviour)", "nvidia/GLM-5.2-NVFP4")
# 2. Same files, minus config.json, so AutoConfig is never consulted.
tok = attempt("stripped local dir", "/tmp/glm_tokenizer")

if tok is not None:
    try:
        s = tok.apply_chat_template([{"role": "user", "content": "hi"}],
                                    tokenize=False, add_generation_prompt=True)
        print(f"### chat template OK: {s!r}")
    except Exception as e:
        print(f"### chat template FAIL {type(e).__name__}: {str(e)[:160]}")
        traceback.print_exc()
PY
