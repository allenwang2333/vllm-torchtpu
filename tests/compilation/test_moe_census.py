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
"""The MoE graph census: which MoE program is in the graph being compiled.

The census exists because every other engagement observable tried on this
kernel measured something adjacent to it. It is stringly typed by necessity --
the op names are minted at runtime with a counter -- so the *names* are pinned
here against the two builders that create them.
"""

import inspect
from types import SimpleNamespace

import pytest

from vllm_torchtpu.compilation import moe_census


def _node(op, target):
    return SimpleNamespace(op=op, target=target)


def test_the_census_separates_the_two_moe_programs():
    counts = moe_census.census([
        _node("call_function", "pallas.fused_ep_moe_kernel_0.default"),
        _node("call_function", "pallas.fused_ep_moe_kernel_0.default"),
        _node("call_function", "pallas.fused_moe_kernel_3.default"),
        _node("call_function", "torch.ops._c10d_functional.all_reduce.default"),
        _node("call_function", "torch.ops.vllm.all_gather.default"),
        _node("call_function", "torch.ops.vllm.reduce_scatter.default"),
        _node("placeholder", "x"),
        _node("call_function", "torch.ops.aten.mul.Tensor"),
    ])
    assert counts["fused_ep"] == 2
    assert counts["gmm"] == 1
    # By kind, not as one number: the dispatch all-gather is fully exposed in a
    # production capture while the combine reduce-scatter is async, so "which
    # collective disappeared" is the mechanism claim (D1).
    assert (counts["ar"], counts["ag"], counts["rs"], counts["a2a"]) == (1, 1, 1, 0)


def test_the_fused_name_is_not_swallowed_by_the_gmm_label():
    """``fused_moe_kernel`` must not match ``fused_ep_moe_kernel``.

    They differ by three characters in the middle and one is nearly a prefix of
    the other; if the labels ever collide the census reports the fused arm as
    the control, which is exactly the confusion it exists to prevent.
    """
    counts = moe_census.census(
        [_node("call_function", "pallas.fused_ep_moe_kernel_7.default")])
    assert counts["gmm"] == 0 and counts["fused_ep"] == 1


def test_every_field_is_printed_even_at_zero():
    """A blank field reads as "not measured"; a zero is a measurement."""
    line = moe_census.format_counts(moe_census.census([]))
    assert line == "fused_ep=0 gmm=0 ag=0 rs=0 ar=0 a2a=0"


@pytest.mark.parametrize("module_path,builder,needle", [
    ("vllm_torchtpu.layers.vllm.fused_moe_ep", "_get_op",
     "fused_ep_moe_kernel"),
    ("vllm_torchtpu.layers.vllm.fused_moe", "_build_fused_moe_custom_op",
     "fused_moe_kernel"),
])
def test_the_labels_match_the_names_the_builders_mint(module_path, builder,
                                                      needle):
    """Rot detector for a stringly-typed contract.

    The census matches on a substring of the op name. Renaming an op would make
    it silently report zero -- indistinguishable from "the kernel did not
    engage", which is the one confusion this whole module exists to remove. So
    assert the substring is still in the source that builds the name, and that
    the census knows about it.
    """
    import importlib

    source = inspect.getsource(
        getattr(importlib.import_module(module_path), builder))
    assert needle in source, (
        f"{module_path}.{builder} no longer mints a name containing "
        f"{needle!r}; moe_census.MOE_OP_LABELS must be updated or the census "
        "will report zero for a kernel that is present")
    assert needle in {name for _, name in moe_census.MOE_OP_LABELS}


def test_the_recorder_tallies_by_range_and_totals():
    recorder = moe_census._Recorder()
    recorder.observe(
        [_node("call_function", "pallas.fused_ep_moe_kernel_0.default")],
        "Range(16384, 16384)")
    recorder.observe(
        [_node("call_function", "pallas.fused_ep_moe_kernel_0.default")],
        "Range(16384, 16384)")
    recorder.observe([_node("call_function", "pallas.fused_moe_kernel_1.default")],
                     "Range(128, 128)")
    assert recorder.total("fused_ep") == 2
    assert recorder.total("gmm") == 1
    assert recorder.graphs == 3
    table = recorder.table()
    # Ordered by bucket, so a reader scans the ladder top to bottom.
    assert table.index("Range(128") < table.index("Range(16384")
    assert "TOTAL: fused_ep=2 gmm=1 ag=0 rs=0 ar=0 a2a=0" in table


def test_an_empty_recorder_says_so_loudly():
    """"No graphs" and "no MoE ops" are different facts and must read that way."""
    assert "NOTHING WAS COMPILED" in moe_census._Recorder().table()
