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
"""The Plan's JSON and encoded forms (R-PLAN-7, R-NF-3, R-NF-7)."""

import base64
import gzip
import json

import pytest
from scenario.plan import codec
from scenario.plan.types import Concat, Ref


def test_refs_are_tagged() -> None:
    value = Concat(("http://", Ref("role_head", role="prefill"), ":8000"))
    assert codec._dump(value) == {
        "concat": ["http://", {"ref": "role_head", "role": "prefill"}, ":8000"]
    }
    assert codec._dump(Ref("pulled_model_dir", fallback="m")) == {
        "ref": "pulled_model_dir",
        "fallback": "m",
    }


def _encode_raw(data: object) -> str:
    raw = json.dumps(data).encode()
    return base64.b64encode(gzip.compress(raw, mtime=0)).decode()


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("not base64!", "not an encoded Plan"),
        (base64.b64encode(b"plain").decode(), "not an encoded Plan"),
        (_encode_raw({"a": 1}), "no format field"),
        (_encode_raw({"format": 1, "roles": []}), "not a Plan of format 1"),
    ],
)
def test_garbage_is_refused(text: str, message: str) -> None:
    with pytest.raises(codec.PlanFormatError, match=message):
        codec.decode(text)
