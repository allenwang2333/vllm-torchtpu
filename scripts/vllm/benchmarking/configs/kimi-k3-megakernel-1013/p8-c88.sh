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

# Kimi-K3 megakernel recipe point 8 of the Oct 13 set: concurrency 88; decode
# admits 128 sequences, no speculation, attention DP2 x DCP16. See
# recipe-1013.sh.

# shellcheck disable=SC2034  # read by recipe-base.sh
RECIPE_CONCURRENCY=88
RECIPE_DECODE=dp2_dcp16
RECIPE_DECODE_MAX_NUM_SEQS=128
# shellcheck source=scripts/vllm/benchmarking/configs/kimi-k3-megakernel-1013/recipe-1013.sh
source "$(dirname "${BASH_SOURCE[0]}")/recipe-1013.sh"
