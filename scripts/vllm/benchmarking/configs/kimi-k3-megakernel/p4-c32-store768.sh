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

# Point 4 with a 768 GiB host KV store per role (~192 GiB per host), twice the
# default in recipe-base.sh and half the doc's 1.5 TiB. Tests whether point 4's
# prefill recompute and throughput gap to the reference numbers come from the
# smaller store.

# shellcheck disable=SC2034  # read by recipe-base.sh
RECIPE_OFFLOAD_BYTES=824633720832
# shellcheck source=scripts/vllm/benchmarking/configs/kimi-k3-megakernel/p4-c32.sh
source "$(dirname "${BASH_SOURCE[0]}")/p4-c32.sh"
