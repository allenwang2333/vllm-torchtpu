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
"""Golden check: the cases, and how today's calls and a plan are compared.

Each case runs a pipeline step of today's scripts with stub programs
(golden_capture.py) for a scenario under .buildkite/kubernetes/scenarios/;
golden_compare.py normalizes both sides to the same representation before it
compares them.
"""

import pathlib

from golden_capture import SCENARIO_ROOT, Capture
from golden_cases import qwen_point, sharding_table
from golden_compare import Normalizer, agentx_flags
from scenario.resolve import Resolver


def test_qwen_points_are_the_sharding_table() -> None:
    resolver = Resolver(SCENARIO_ROOT)
    points = resolver.resolve(resolver.file_of("qwen3.5-397b/inferencex-sweep"))
    names = [qwen_point(conc, sharding) for _, _, conc, sharding in sharding_table()]
    assert [p.point for p in points] == names


def test_normalization_is_representation_only() -> None:
    root, state = pathlib.Path("/scratch/repo"), pathlib.Path("/scratch/state")
    norm = Normalizer(Capture(root, state, [], [], {}), ["perf_eval_results/x"])

    def flags(*argv: str) -> dict[str, object]:
        return dict(norm.call(argv, {}).flags)

    assert flags("vllm", "serve", "m", "--temperature=0") == flags(
        "vllm", "serve", "--model", "m", "--temperature", "0.0"
    )
    assert flags("x", "--c", '{"b":1,"a":[2]}') == flags(
        "x", "--c", '{"a": [2], "b": 1}'
    )
    assert flags("x", "--gpu-memory-utilization", "0.90") == {
        "gpu-memory-utilization": 0.9
    }
    assert flags("x", "--flag")["flag"] is True
    assert flags(
        "lm_eval", "--model_args", "base_url=http://localhost:8000,n=128,n=16"
    ) == {"model_args": {"base_url": "http://127.0.0.1:8000", "n": 16.0}}
    assert flags("vllm", "bench", "serve", "--result-file=/scratch/repo/r.json") == {
        "result-filename": "r.json"
    }
    assert flags("vllm", "serve", "m", "--tokenizer", "m") == {"model": "m"}
    assert norm.text("perf_eval_results/x/a.json") == "<results>/a.json"
    toy = "/scratch/repo/examples/disagg/toy_proxy_server.py"
    assert flags("python3", toy, "--prefiller-host", "h", "--decoder-port", "1") == {
        "prefiller-hosts": "h",
        "decoder-ports": 1.0,
    }


def test_agentx_env_as_client_flags() -> None:
    pair = {
        "AIPERF_SERVER_URL": "http://127.0.0.1:8100",
        "CONC": "4",
        "MODEL_PREFIX": "kimik3",
        "IS_MULTINODE": "true",
        "PREFILL_NUM_WORKERS": "1",
        "PREFILL_TP": "16",
        "DECODE_NUM_WORKERS": "1",
        "DECODE_TP": "16",
        "RESULT_FILENAME": "agentx_c4",
    }
    assert agentx_flags(pair) == [
        "--server-url",
        "http://127.0.0.1:8100",
        "--conc-list",
        "4",
        "--model-prefix",
        "kimik3",
        "--chips",
        "32",
        "--label",
        "agentx",
    ]
    one = {"CONC": "1", "IS_MULTINODE": "false", "TP": "16", "RESULT_FILENAME": "x_c1"}
    assert agentx_flags(one) == ["--conc-list", "1", "--chips", "16", "--label", "x"]
