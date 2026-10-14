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

# Point 4 with the doc's 1.5 TiB host KV store per role (~384 GiB per host),
# four times the default in recipe-base.sh. Run without the fleet's gcsfuse
# caches (common.sh), which freed ~60 GiB a host; with them, a 1.5 TiB store
# got the prefill head evicted (vllm-torchtpu-dev build 562).

# shellcheck disable=SC2034  # read by recipe-base.sh
RECIPE_OFFLOAD_BYTES=1649267441664
# shellcheck source=scripts/vllm/benchmarking/configs/kimi-k3-megakernel/p4-c32.sh
source "$(dirname "${BASH_SOURCE[0]}")/p4-c32.sh"
