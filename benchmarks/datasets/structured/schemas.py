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
"""Standard structured decoding benchmark schemas."""

from typing import Any

FLAT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"type": "string"},
        "is_active": {"type": "boolean"},
        "role": {"type": "string", "enum": ["admin", "user", "guest"]},
    },
    "required": ["name", "age", "email", "is_active", "role"],
}

COMPLEX_AST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "$defs": {
        "Expression": {
            "type": "object",
            "properties": {
                "op": {
                    "type": "string",
                    "enum": ["+", "-", "*", "/", "==", "<", ">"],
                },
                "left": {"$ref": "#/$defs/Node"},
                "right": {"$ref": "#/$defs/Node"},
            },
            "required": ["op", "left", "right"],
        },
        "Node": {
            "anyOf": [
                {"type": "number"},
                {"type": "string"},
                {"$ref": "#/$defs/Expression"},
            ]
        },
    },
    "properties": {
        "ast": {"$ref": "#/$defs/Expression"},
        "description": {"type": "string"},
    },
    "required": ["ast", "description"],
}
