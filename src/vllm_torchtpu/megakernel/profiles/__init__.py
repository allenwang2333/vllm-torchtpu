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
"""Checked-in named execution profiles.

`default` holds the `default` profile's tile rule (imported eagerly; it depends
only on `spec`). Named profiles register themselves when their module is
imported; `profile.get_profile` / `list_profiles` import every module in
`BUILTIN_PROFILE_MODULES` first, and attribute access imports one lazily.
"""

from __future__ import annotations

from typing import Any

from . import default

BUILTIN_PROFILE_MODULES = ("qwen35_candidate30",)


def load_builtin_profiles() -> None:
    """Import (and thereby register) every checked-in named profile."""
    from . import qwen35_candidate30  # noqa: F401


def __getattr__(name: str) -> Any:
    if name == "qwen35_candidate30":
        from . import qwen35_candidate30

        return qwen35_candidate30
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["BUILTIN_PROFILE_MODULES", "default", "load_builtin_profiles"]
