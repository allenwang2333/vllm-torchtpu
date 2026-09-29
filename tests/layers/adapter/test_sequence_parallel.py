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
"""Row-gated sequence parallelism on the CPU.

The decoder, model and MTP forwards run on two simulated TP ranks (threads
exchanging through fake collectives) against a one-device reference; the
MoE block goes through the real communicator.
"""

import threading
from types import SimpleNamespace

import pytest
import torch
import vllm.distributed as vllm_distributed

from vllm_torchtpu.distributed import tpu_communicator as comm
from vllm_torchtpu.layers.adapter import sequence_parallel as sp
from vllm_torchtpu.models.vllm import qwen3_next_sp_patch as patch

pytestmark = pytest.mark.cpu_test

_TP, _H, _MIN_ROWS = 2, 8, 8


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("TPU_SEQUENCE_PARALLEL_PREFILL", "1")
    monkeypatch.setattr(sp, "MIN_ROWS", _MIN_ROWS)
    monkeypatch.setattr(sp, "_active", False)
    monkeypatch.setattr(sp, "_layers_built", 0)


def test_the_step_form_follows_the_row_count(enabled):
    assert sp.enabled()
    assert not sp.step_is_sequence_parallel(_MIN_ROWS - 1)
    assert sp.step_is_sequence_parallel(_MIN_ROWS)
    sp.set_active(True)
    assert sp.rows_are_sliced(True) and not sp.rows_are_replicated(True)
    sp.set_active(False)
    assert sp.rows_are_replicated(True) and not sp.rows_are_sliced(True)
    # A layer not built sequence parallel takes neither branch.
    assert not sp.rows_are_sliced(False) and not sp.rows_are_replicated(False)


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("TPU_SEQUENCE_PARALLEL_PREFILL", raising=False)
    assert not sp.enabled()
    assert not sp.step_is_sequence_parallel(1 << 20)


@pytest.mark.parametrize("flag", ["0", "1"])
def test_only_the_layers_that_implement_it_see_sequence_parallel_moe(monkeypatch, flag):
    from vllm.config.parallel import ParallelConfig

    import vllm_torchtpu

    monkeypatch.setenv("TPU_SEQUENCE_PARALLEL_PREFILL", flag)
    monkeypatch.setattr(ParallelConfig, "_tpu_no_sp_moe_patch", False, raising=False)
    monkeypatch.setattr(
        ParallelConfig,
        "use_sequence_parallel_moe",
        ParallelConfig.use_sequence_parallel_moe,
    )
    vllm_torchtpu._patch_disable_sequence_parallel_moe()
    tp8 = ParallelConfig(tensor_parallel_size=8, enable_expert_parallel=True)
    others = [
        ParallelConfig(data_parallel_size=8, enable_expert_parallel=True),
        ParallelConfig(tensor_parallel_size=8, enable_expert_parallel=False),
        ParallelConfig(
            tensor_parallel_size=4, data_parallel_size=2, enable_expert_parallel=True
        ),
    ]
    assert not tp8.use_sequence_parallel_moe
    with sp.building_layer():
        assert tp8.use_sequence_parallel_moe is (flag == "1")
        assert not any(config.use_sequence_parallel_moe for config in others)
    assert not tp8.use_sequence_parallel_moe


def test_uneven_buckets_are_refused_at_boot(enabled):
    sp.check_buckets([16, 32, 48, 64], _TP)
    with pytest.raises(NotImplementedError, match=r"buckets \[9\]"):
        sp.check_buckets([4, 9, 16], _TP)


def test_a_model_without_row_gated_layers_is_refused(enabled):
    with pytest.raises(NotImplementedError, match="TPU_SEQUENCE_PARALLEL_PREFILL"):
        sp.check_installed(8)
    sp.record_layer()
    sp.check_installed(8)


def test_the_runner_records_the_step_form_before_the_model(enabled):
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    seen = []
    runner = SimpleNamespace(model=lambda **kw: seen.append(sp.is_active()) or kw)
    for rows in (_MIN_ROWS - 1, _MIN_ROWS):
        TPUModelRunner.forward_model(runner, torch.zeros(rows), torch.arange(rows))
    assert seen == [False, True]


class _FakeTP:
    """TP collectives between threads, one thread per rank."""

    def __init__(self, world_size):
        self.world_size = world_size
        self.rank_in_group = None
        self._slots = [None] * world_size
        self._barrier = threading.Barrier(world_size)
        self._local = threading.local()

    @property
    def rank(self):
        return self._local.rank

    def _exchange(self, x):
        self._slots[self.rank] = x
        self._barrier.wait()
        out = list(self._slots)
        self._barrier.wait()
        return out

    def all_gather(self, x, dim=0):
        return torch.cat(self._exchange(x), dim=dim)

    def reduce_scatter(self, x, dim=0):
        return sum(self._exchange(x)).chunk(self.world_size, dim=dim)[self.rank]

    def all_reduce(self, x):
        return sum(self._exchange(x))

    def run(self, fn):
        results, errors = [None] * self.world_size, []

        def worker(rank):
            self._local.rank = rank
            try:
                results[rank] = fn(rank)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)
                self._barrier.abort()

        threads = [
            threading.Thread(target=worker, args=(r,)) for r in range(self.world_size)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
        return results


class _Norm(torch.nn.Module):
    def forward(self, x, residual=None):
        y = x if residual is None else x + residual
        out = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-6)
        return out if residual is None else (out, y)


class _Attention(torch.nn.Module):
    """Mixes rows (a running sum), so it needs the step's rows whole; `w`
    is this rank's row-parallel share of the output projection."""

    def __init__(self, w):
        super().__init__()
        self.w = w

    def forward(self, hidden_states, positions=None):
        return torch.cumsum(hidden_states, 0) @ self.w


class _Moe(torch.nn.Module):
    """Experts split over the ranks, exchanged by the real communicator."""

    def __init__(self, scale, communicator):
        super().__init__()
        self.scale = scale
        self.communicator = communicator

    def forward(self, hidden_states, already_sequence_parallel=False):
        if self.communicator is None:
            return hidden_states * self.scale
        x, _ = self.communicator.dispatch_router_logits(
            hidden_states, hidden_states[:, :1], is_sequence_parallel=True
        )
        return self.communicator.combine(x * self.scale, is_sequence_parallel=True)


def _layer(w, scale, communicator, rank, layer_type="linear_attention"):
    layer = torch.nn.Module()
    layer.layer_type = layer_type
    layer.layer_scale = False
    layer.input_layernorm = _Norm()
    layer.post_attention_layernorm = _Norm()
    if layer_type == "linear_attention":
        layer.linear_attn = _Attention(w)
    else:
        layer.self_attn = _Attention(w)
    layer.mlp = _Moe(scale, communicator)
    setattr(layer, patch.SP_LAYER_ATTR, communicator is not None)
    setattr(layer, patch.SP_RANK_ATTR, torch.tensor([rank], dtype=torch.int32))
    return layer


def _reference_layer(layer, hidden_states, residual, positions):
    if residual is None:
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
    else:
        hidden_states, residual = layer.input_layernorm(hidden_states, residual)
    if layer.layer_type == "linear_attention":
        hidden_states = layer.linear_attn(hidden_states)
    else:
        hidden_states = layer.self_attn(hidden_states)
    hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
    return layer.mlp(hidden_states), residual


class _Model(torch.nn.Module):
    """vLLM 0.29's model forward with the chunk and gather it skips once the
    layers clear `use_attn_reduce_scatter_for_moe`."""

    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.start_layer = 0
        self.norm = _Norm()

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None
    ):
        hidden_states, residual = inputs_embeds, None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual, positions)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class _Predictor(torch.nn.Module):
    """vLLM 0.29's MTP predictor forward, likewise."""

    def __init__(self, layer):
        super().__init__()
        self.layers = torch.nn.ModuleList([layer])
        self.num_mtp_layers = 1
        self.norm = _Norm()

    def forward(
        self,
        input_ids,
        positions,
        hidden_states,
        intermediate_tensors=None,
        inputs_embeds=None,
        spec_step_idx=0,
    ):
        hidden_states = hidden_states + inputs_embeds
        layer = self.layers[spec_step_idx % self.num_mtp_layers]
        hidden_states, residual = layer(hidden_states, None, positions)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@pytest.fixture
def tp(monkeypatch, enabled):
    """Two TP ranks whose TP and EP groups are the fake."""
    group = _FakeTP(_TP)
    for name, fn in dict(
        tensor_model_parallel_all_gather=group.all_gather,
        tensor_model_parallel_reduce_scatter=group.reduce_scatter,
        tensor_model_parallel_all_reduce=group.all_reduce,
        get_tensor_model_parallel_world_size=lambda: _TP,
    ).items():
        monkeypatch.setattr(vllm_distributed, name, fn)
    monkeypatch.setattr(comm, "get_ep_group", lambda: group)

    def init(instance, *args, **kwargs):
        instance.use_all2all = False

    monkeypatch.setattr(comm.DeviceCommunicatorBase, "__init__", init)
    forward = patch._row_gated_forward(lambda *a, **k: pytest.fail("not row-gated"))
    return SimpleNamespace(group=group, forward=forward)


def _shares(total, rank):
    """Rank `rank`'s share of `total` over _TP ranks: unequal, summing back."""
    weights = torch.linspace(0.25, 1.75, _TP)
    return total * weights[rank] / weights.sum()


def _weights(num_layers, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [
        (torch.randn(_H, _H, generator=g) / _H, torch.randn(_H, generator=g))
        for _ in range(num_layers)
    ]


def _bind(module, forward):
    module.forward = forward.__get__(module)
    return module


@pytest.mark.parametrize(
    "rows", [4, _MIN_ROWS, 2 * _MIN_ROWS], ids=lambda r: f"{r}rows"
)
def test_the_target_model_matches_one_device(tp, rows):
    weights = _weights(3)
    layer_types = ["linear_attention", "linear_attention", "full_attention"]
    embeds = torch.randn(rows, _H, generator=torch.Generator().manual_seed(1))
    positions = torch.arange(rows)

    reference = _Model(
        [_layer(w, s, None, 0, t) for (w, s), t in zip(weights, layer_types)]
    )
    for layer in reference.layers:
        _bind(layer, _reference_layer)
    expected = reference(None, positions, inputs_embeds=embeds)

    model_forward = patch._gather_model_rows(_Model.forward)
    sp.set_active(sp.step_is_sequence_parallel(rows))

    def rank_forward(rank):
        communicator = comm.TpuDeviceCommunicator()
        layers = [
            _bind(
                _layer(_shares(w, rank), _shares(s, rank), communicator, rank, t),
                tp.forward,
            )
            for (w, s), t in zip(weights, layer_types)
        ]
        model = _Model(layers)
        return model_forward(model, None, positions, inputs_embeds=embeds)

    for out in tp.group.run(rank_forward):
        assert out.shape == (rows, _H)
        torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("rows", [4, 2 * _MIN_ROWS], ids=lambda r: f"{r}rows")
def test_the_mtp_draft_matches_one_device(tp, rows):
    ((w, s),) = _weights(1, seed=2)
    g = torch.Generator().manual_seed(3)
    target_hidden = torch.randn(rows, _H, generator=g)
    embeds = torch.randn(rows, _H, generator=g)
    positions = torch.arange(rows)

    reference = _Predictor(
        _bind(_layer(w, s, None, 0, "full_attention"), _reference_layer)
    )
    expected = reference(None, positions, target_hidden, inputs_embeds=embeds)

    predictor_forward = patch._gather_mtp_rows(_Predictor.forward)
    sp.set_active(sp.step_is_sequence_parallel(rows))

    def rank_forward(rank):
        communicator = comm.TpuDeviceCommunicator()
        layer = _layer(
            _shares(w, rank), _shares(s, rank), communicator, rank, "full_attention"
        )
        predictor = _Predictor(_bind(layer, tp.forward))
        return predictor_forward(
            predictor, None, positions, target_hidden, inputs_embeds=embeds
        )

    for out in tp.group.run(rank_forward):
        torch.testing.assert_close(out, expected)


def test_a_stale_step_form_breaks_the_draft(tp):
    """What the draft's own `set_active` prevents: a sequence-parallel draft
    step run under a smaller target step's record combines other ranks' rows."""
    ((w, s),) = _weights(1, seed=2)
    rows = 2 * _MIN_ROWS
    g = torch.Generator().manual_seed(3)
    target_hidden, embeds = torch.randn(rows, _H, generator=g), torch.randn(rows, _H)
    positions = torch.arange(rows)
    reference = _Predictor(
        _bind(_layer(w, s, None, 0, "full_attention"), _reference_layer)
    )
    expected = reference(None, positions, target_hidden, inputs_embeds=embeds)
    predictor_forward = patch._gather_mtp_rows(_Predictor.forward)
    sp.set_active(False)

    def rank_forward(rank):
        communicator = comm.TpuDeviceCommunicator()
        layer = _layer(
            _shares(w, rank), _shares(s, rank), communicator, rank, "full_attention"
        )
        return predictor_forward(
            _Predictor(_bind(layer, tp.forward)),
            None,
            positions,
            target_hidden,
            inputs_embeds=embeds,
        )

    for out in tp.group.run(rank_forward):
        assert not torch.allclose(out, expected)


def test_both_forms_lift_the_same_inputs(enabled, monkeypatch):
    """A bucket compiled by one form's trace is handed to the other's
    (compilation/shape_variants): their graphs must take the same inputs."""
    from vllm_torchtpu.compilation import shape_variants

    identity = dict(
        tensor_model_parallel_all_gather=lambda x, dim=0: x,
        tensor_model_parallel_reduce_scatter=lambda x, dim=0: x,
        tensor_model_parallel_all_reduce=lambda x: x,
        get_tensor_model_parallel_world_size=lambda: 1,
    )
    for name, fn in identity.items():
        monkeypatch.setattr(vllm_distributed, name, fn)
    forward = patch._row_gated_forward(None)
    layers = [
        _bind(_layer(w, s, None, 0, t), forward)
        for (w, s), t in zip(_weights(2), ["linear_attention", "full_attention"])
    ]
    for layer in layers:
        setattr(layer, patch.SP_LAYER_ATTR, True)
    model = _Model(layers)
    traces = []

    def backend(graph, example_inputs):
        traces.append(shape_variants._handover_inputs(graph))
        return graph.forward

    torch._dynamo.reset()
    compiled = torch.compile(
        lambda x, p: model(None, p, inputs_embeds=x), backend=backend, fullgraph=True
    )
    for rows in (4, 2 * _MIN_ROWS):
        x, p = torch.randn(rows, _H), torch.arange(rows)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(p, 0)
        compiled(x, p)
    torch._dynamo.reset()
    assert len(traces) == 2
    below, above = traces
    assert any(patch.SP_RANK_ATTR in name for name, _ in below)
    assert sorted(below) == sorted(above)


def test_install_patches_the_vllm_classes_once(enabled, monkeypatch):
    import inspect

    from vllm.model_executor.models import (
        qwen3_5,
        qwen3_5_mtp,
        qwen3_next,
        qwen3_next_mtp,
    )

    layer_cls = qwen3_next.Qwen3NextDecoderLayer
    targets = [
        (layer_cls, "__init__"),
        (layer_cls, "forward"),
        (qwen3_5.Qwen3_5DecoderLayer, "__init__"),
        (qwen3_next.Qwen3NextModel, "forward"),
        (qwen3_next_mtp.Qwen3NextMultiTokenPredictor, "forward"),
        (qwen3_5_mtp.Qwen3_5MultiTokenPredictor, "forward"),
    ]
    originals = [vars(cls)[name] for cls, name in targets]
    for (cls, name), original in zip(targets, originals):
        monkeypatch.setattr(cls, name, original)
    monkeypatch.setattr(layer_cls, "_tpu_sp_by_rows_patch", False, raising=False)
    patch.install()
    patched = [vars(cls)[name] for cls, name in targets]
    for original, new in zip(originals, patched):
        assert new is not original
        # support_torch_compile binds the forward's arguments by name.
        assert inspect.signature(new) == inspect.signature(original)
    assert qwen3_5.Qwen3_5DecoderLayer.forward is layer_cls.forward
    patch.install()
    assert [vars(cls)[name] for cls, name in targets] == patched


def test_install_is_a_no_op_when_not_selected(monkeypatch):
    from vllm.model_executor.models import qwen3_next

    monkeypatch.delenv("TPU_SEQUENCE_PARALLEL_PREFILL", raising=False)
    forward = qwen3_next.Qwen3NextDecoderLayer.forward
    patch.install()
    assert qwen3_next.Qwen3NextDecoderLayer.forward is forward


@pytest.mark.parametrize("tp_size,dp_size", [(8, 1), (1, 8)])
def test_a_layer_is_built_for_the_row_gated_form(
    enabled, monkeypatch, tp_size, dp_size
):
    """vLLM's layer reads `use_sequence_parallel_moe` while it is built; the
    wrapper then takes the form over from it."""
    monkeypatch.setattr(vllm_distributed, "get_tensor_model_parallel_rank", lambda: 3)
    parallel_config = SimpleNamespace(
        tensor_parallel_size=tp_size,
        enable_expert_parallel=True,
        data_parallel_size=dp_size,
        prefill_context_parallel_size=1,
        pipeline_parallel_size=1,
    )
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        compilation_config=SimpleNamespace(compile_sizes=[16, 512, 1024]),
    )

    class Layer(torch.nn.Module):
        def __init__(self, vllm_config, layer_type, prefix=""):
            super().__init__()
            self.use_attn_reduce_scatter_for_moe = sp.use_sequence_parallel_moe(
                vllm_config.parallel_config
            )
            self.input_layernorm = torch.nn.Linear(1, 1)

    Layer.__init__ = patch._build_row_gated(Layer.__init__)
    layer = Layer(vllm_config, "full_attention", prefix="model.layers.0")
    row_gated = tp_size > 1
    assert getattr(layer, patch.SP_LAYER_ATTR) is row_gated
    assert layer.use_attn_reduce_scatter_for_moe is False
    assert sp._layers_built == int(row_gated)
    if row_gated:
        assert getattr(layer, patch.SP_RANK_ATTR).tolist() == [3]
        vllm_config.compilation_config.compile_sizes = [16, 516]
        with pytest.raises(NotImplementedError, match="516"):
            Layer(vllm_config, "full_attention")
    else:
        assert not hasattr(layer, patch.SP_RANK_ATTR)
