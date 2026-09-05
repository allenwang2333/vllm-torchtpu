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

import gc
import weakref
from types import SimpleNamespace

import torch
from torch import nn

from vllm_torchtpu.spec_decode import utils


def test_own_vocab_weights_are_kept_when_tp_layout_matches():
    target_embed = object()
    target_head = object()
    draft_embed = object()
    draft_head = object()
    target = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=target_embed),
        lm_head=target_head,
    )
    draft = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=draft_embed),
        lm_head=draft_head,
        has_own_embed_tokens=True,
        has_own_lm_head=True,
    )

    utils.maybe_share_embeddings(draft, target, tp_layout_matches=True)
    utils.maybe_share_lm_head(draft,
                              target,
                              draft_replicated=False,
                              tp_layout_matches=True)

    assert draft.model.embed_tokens is draft_embed
    assert draft.lm_head is draft_head


class _VocabLikeModule(nn.Module):
    """A module shaped like the ones vLLM's vocab-parallel layers produce.

    The only property that matters is the one that causes the leak:
    ``set_weight_attrs`` puts the module's own bound ``weight_loader`` on its
    ``Parameter``, so the module is reachable from its own parameter.
    """

    def __init__(self, rows: int = 8, cols: int = 4) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(rows, cols),
                                   requires_grad=False)
        # This is the cycle: module -> _parameters -> weight -> bound method
        # -> module.
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param, loaded_weight):  # pragma: no cover
        param.data.copy_(loaded_weight)


def test_the_orphaned_draft_module_is_a_reference_cycle():
    """The premise. If this stops holding, the release below is unnecessary.

    Dropping the last named reference must *not* be enough to free the module,
    because the bound weight_loader on its own Parameter keeps it alive until
    the cyclic collector runs.
    """
    gc.disable()
    try:
        orphan = _VocabLikeModule()
        ref = weakref.ref(orphan)
        del orphan
        assert ref() is not None, (
            "the orphaned module was reclaimed by refcounting alone, so the "
            "weight_loader cycle this guards against no longer exists")
    finally:
        gc.enable()
    gc.collect()
    assert ref() is None


def test_release_detaches_the_orphans_parameters():
    """Detaching is what makes the release deterministic.

    gc.collect() alone leaves the timing to the collector's thresholds, and
    HBM is measured moments later by a live device query -- so ranks that
    happened to collect would size a bigger KV pool than ranks that did not.
    """
    orphan = _VocabLikeModule()
    weight_ref = weakref.ref(orphan.weight)
    replacement = _VocabLikeModule()

    utils.release_replaced_draft_module(orphan, replacement)

    assert orphan._parameters["weight"] is None
    gc.collect()
    assert weight_ref() is None


def test_release_is_a_no_op_when_the_orphan_is_the_replacement():
    """The tp=1 aliasing case.

    When the draft already holds the target's module, the "orphan" is the
    tensor still in use. Clearing it would delete the target's embedding, so
    the guard is identity, not equality.
    """
    shared = _VocabLikeModule()

    utils.release_replaced_draft_module(shared, shared)

    assert shared._parameters["weight"] is not None
    assert isinstance(shared.weight, nn.Parameter)


def test_release_tolerates_no_orphan():
    utils.release_replaced_draft_module(None, _VocabLikeModule())
