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

# The Oct 13 set of Kimi-K3 megakernel recipe points (the "Recipes 1013" tab of
# the "Kimi-K3 on TPU7x: 1P1D prefill/decode recipes" doc) on the AgentX 1P1D
# pair. A point is one file beside this one that sets the RECIPE_* values
# recipe-base.sh reads and sources this file, and
# K3_PD_CONFIG=kimi-k3-megakernel-1013/<point> serves it.
#
# What a point carries from the doc: the AgentX concurrency, decode's
# --max-num-seqs, and either DSpark with its drafts and synthetic acceptance
# (points 1-6) or the attention DP2 x DCP16 decode without speculation (points
# 7-9). The serve is otherwise recipe-base.sh's, as for kimi-k3-megakernel/,
# with its departures for GKE.
#
# Not carried, as each needs more than a config: TPU P-state 7 and
# RAIDEN_NUMA_POLICY=bind on both roles; the prefill without
# --prefix-cache-retention-interval; the global registry and decode-host store
# servers of points 3-9, and the decode without a host offload connector that
# goes with them; the --long-prefill-token-threshold, --numa-bind and 2 TiB
# host store of the points 7-9 prefill; and the one-hour benchmark with the
# doc's aiperf options (the client takes AGENTX_DURATION from its environment,
# not from a config).

# shellcheck source=scripts/vllm/benchmarking/configs/kimi-k3-megakernel/recipe-base.sh
source "$(dirname "${BASH_SOURCE[0]}")/../kimi-k3-megakernel/recipe-base.sh"
