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
"""Find set environment variables that nothing reads.

``envs.py`` lists every knob this package reads, so a set variable that is
neither there nor in vLLM's registry does nothing. Two cases are reported:

- ``VLLM_*`` names, which only vLLM and this package use, when neither
  registry has them (for example a removed ``VLLM_TPU_*`` knob).
- Names that share a prefix with this package's knobs (``TPU_``, ``MOE_``,
  ...) and closely match a registered name without being one, which is
  almost always a typo. Other ``TPU_*`` names are left alone: libtpu, JAX
  and GKE read and inject many of them, and no list of those stays current.
"""

import difflib
from collections.abc import Collection, Iterable
from dataclasses import dataclass

# High enough that a one- or two-letter slip matches and unrelated names
# sharing a prefix do not.
_TYPO_CUTOFF = 0.9


@dataclass(frozen=True)
class UnreadEnvVar:
    """A set environment variable that no registry reads."""

    name: str
    suggestion: str | None = None

    def __str__(self) -> str:
        if self.suggestion is None:
            return self.name
        return f"{self.name} (did you mean {self.suggestion}?)"


def _prefix(name: str) -> str:
    return name.split("_", 1)[0]


def _closest(name: str, candidates: Collection[str]) -> str | None:
    matches = difflib.get_close_matches(name, candidates, n=1, cutoff=_TYPO_CUTOFF)
    return matches[0] if matches else None


def find_unread_env_vars(
    set_names: Iterable[str],
    torchtpu_names: Collection[str],
    vllm_names: Collection[str],
) -> list[UnreadEnvVar]:
    """Returns the set names that neither registry reads, sorted by name.

    Args:
        set_names: Names of the variables set in the environment.
        torchtpu_names: Names this package reads.
        vllm_names: Names vLLM reads.
    """
    known = set(torchtpu_names) | set(vllm_names)
    known_vllm = sorted(n for n in known if n.startswith("VLLM_"))
    by_prefix: dict[str, list[str]] = {}
    for registered in torchtpu_names:
        if not registered.startswith("VLLM_"):
            by_prefix.setdefault(_prefix(registered), []).append(registered)

    unread = []
    for name in sorted(set(set_names) - known):
        if name.startswith("VLLM_"):
            unread.append(UnreadEnvVar(name, _closest(name, known_vllm)))
            continue
        candidates = by_prefix.get(_prefix(name))
        if candidates:
            suggestion = _closest(name, candidates)
            if suggestion is not None:
                unread.append(UnreadEnvVar(name, suggestion))
    return unread
