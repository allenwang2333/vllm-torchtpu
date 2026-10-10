# SPDX-License-Identifier: Apache-2.0
"""TD8.230: KV_SHARD=1 defaults the 100k decode config to no prefix caching."""

import pathlib
import shlex
import subprocess

import pytest

CONFIG = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "vllm"
    / "benchmarking"
    / "configs"
    / "deepseek-v41-flash-tp8-ep-100k-decode.sh"
)


def _prefix_caching(**env: str) -> str:
    lines = ["unset ENABLE_PREFIX_CACHING TPU_DEEPSEEK_V41_KV_SHARD"]
    lines += [f"export {k}={shlex.quote(v)}" for k, v in env.items()]
    lines += [
        f"source {shlex.quote(str(CONFIG))}",
        'printf %s "$ENABLE_PREFIX_CACHING"',
    ]
    result = subprocess.run(
        ["bash", "-c", "\n".join(lines)], capture_output=True, text=True, check=True
    )
    return result.stdout


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, "true"),
        ({"TPU_DEEPSEEK_V41_KV_SHARD": "0"}, "true"),
        # OwnerAffineBlockPool.validate_config rejects prefix caching here.
        ({"TPU_DEEPSEEK_V41_KV_SHARD": "1"}, "false"),
        # An explicit caller choice still wins.
        ({"TPU_DEEPSEEK_V41_KV_SHARD": "1", "ENABLE_PREFIX_CACHING": "true"}, "true"),
        ({"ENABLE_PREFIX_CACHING": "false"}, "false"),
    ],
)
def test_prefix_caching_default_follows_kv_shard(env, expected):
    assert _prefix_caching(**env) == expected
