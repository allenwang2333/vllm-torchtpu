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
"""Unit tests for the env-gated per-(layer, expert) router load histograms.

The load-bearing test is `test_disabled_dispatches_no_aten_ops`: the whole
design rests on the claim that with the dump off the traced graph is unchanged,
and a CPU test that counts dispatched aten ops is the cheapest place to make
that a check rather than an assurance.
"""

import importlib

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import vllm_torchtpu.envs as envs


class _OpCounter(TorchDispatchMode):
    """Counts every aten op that reaches the dispatcher inside the block."""

    def __init__(self):
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **kwargs or {})


def _reload(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    envs_mod = importlib.reload(envs)
    mod = importlib.import_module("vllm_torchtpu.layers.vllm.expert_load")
    return importlib.reload(mod), envs_mod


@pytest.fixture(autouse=True)
def _restore_modules():
    """Every test reloads the module under a different env; put it back."""
    yield
    importlib.reload(envs)
    importlib.reload(
        importlib.import_module("vllm_torchtpu.layers.vllm.expert_load"))
    importlib.reload(
        importlib.import_module("vllm_torchtpu.layers.vllm.moe_routing"))


class _Layer(torch.nn.Module):

    def __init__(self, global_num_experts=16, ep_size=4, use_ep=True):
        super().__init__()
        self.global_num_experts = global_num_experts
        self.moe_config = type(
            "C", (), {
                "ep_size": ep_size,
                "ep_rank": 0,
                "moe_parallel_config": type("P", (), {"use_ep": use_ep})(),
            })()


class _Model(torch.nn.Module):

    def __init__(self, n_layers=3, **kw):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer(**kw) for _ in range(n_layers)])


def _arm(monkeypatch, tmp_path, slots=4, bucket=2, **extra):
    return _reload(monkeypatch,
                   TPU_MOE_EXPERT_LOAD_DIR=str(tmp_path / "eld"),
                   TPU_MOE_EXPERT_LOAD_SLOTS=slots,
                   TPU_MOE_EXPERT_LOAD_BUCKET=bucket,
                   **extra)[0]


def test_disabled_registers_nothing_and_dispatches_no_aten_ops(monkeypatch):
    el = _reload(monkeypatch, TPU_MOE_EXPERT_LOAD_DIR="")[0]
    assert not el.enabled()
    layer = _Layer()
    el.register_buffers(layer, device=torch.device("cpu"))
    assert not any(n.startswith("_eld") for n, _ in layer.named_buffers())

    ids = torch.randint(0, 16, (16, 8))
    weights = torch.ones(16, 8)
    with _OpCounter() as counter:
        el.record(layer, ids, weights)
    assert counter.ops == [], f"record() is not free when off: {counter.ops}"
    assert el.dump(_Model(), rank=0) is None


def test_counts_match_a_numpy_bincount(monkeypatch, tmp_path):
    el = _arm(monkeypatch, tmp_path)
    layer = _Layer(global_num_experts=16)
    el.register_buffers(layer, device=torch.device("cpu"))

    torch.manual_seed(0)
    ids = torch.randint(0, 16, (5, 3))
    weights = torch.ones(5, 3)
    el.record(layer, ids, weights)

    want = np.bincount(ids.numpy().reshape(-1), minlength=16)
    np.testing.assert_array_equal(layer._eld_hist_all[0].numpy(), want)
    np.testing.assert_array_equal(layer._eld_hist_real[0].numpy(), want)
    assert layer._eld_rows[0].tolist() == [15, 15]
    assert layer._eld_calls[0].item() == 1
    assert layer._eld_step.item() == 1


def test_padding_mask_marks_rows_pad_even_with_non_zero_weights(
        monkeypatch, tmp_path):
    """The mxfp4 path zeroes padded weights AFTER routing, so at this hook the
    padding mask is the only signal. Missing it would make hist_real == hist_all
    on the one model this exists for."""
    el = _arm(monkeypatch, tmp_path)
    from vllm_torchtpu.layers.vllm import token_padding
    state = token_padding.TokenPaddingState.create(3, torch.device("cpu"))
    state.update(local_valid_tokens=1, local_padded_tokens=3)
    token_padding.set_padding_state(state)
    try:
        layer = _Layer(global_num_experts=8)
        el.register_buffers(layer, device=torch.device("cpu"))
        ids = torch.tensor([[1, 2], [3, 4], [5, 6]])
        el.record(layer, ids, torch.ones(3, 2))  # every weight non-zero
        real = layer._eld_hist_real.numpy()[0]
        np.testing.assert_array_equal(real, [0, 1, 1, 0, 0, 0, 0, 0])
        assert layer._eld_rows[0].tolist() == [6, 2]
    finally:
        token_padding.set_padding_state(None)


def test_zero_weight_rows_are_counted_separately(monkeypatch, tmp_path):
    """hist_all - hist_real is exactly where the padded rows went."""
    el = _arm(monkeypatch, tmp_path)
    layer = _Layer(global_num_experts=8)
    el.register_buffers(layer, device=torch.device("cpu"))

    ids = torch.tensor([[1, 2], [3, 4], [5, 6]])
    weights = torch.tensor([[0.5, 0.5], [0.0, 0.0], [0.0, 0.0]])  # 2 pad rows
    el.record(layer, ids, weights)

    diff = (layer._eld_hist_all - layer._eld_hist_real).numpy()
    np.testing.assert_array_equal(diff[0], [0, 0, 0, 1, 1, 1, 1, 0])
    assert diff[1:].sum() == 0
    assert layer._eld_rows[0].tolist() == [6, 2]


def test_slot_advances_with_bucket_and_last_slot_absorbs_overflow(
        monkeypatch, tmp_path):
    el = _arm(monkeypatch, tmp_path, slots=3, bucket=2)
    layer = _Layer(global_num_experts=4)
    el.register_buffers(layer, device=torch.device("cpu"))

    ids = torch.zeros(1, 1, dtype=torch.long)
    for _ in range(10):
        el.record(layer, ids, torch.ones(1, 1))

    # 2 calls per slot for slots 0 and 1; the remaining 6 pile into slot 2.
    assert layer._eld_calls.tolist() == [2, 2, 6]
    assert layer._eld_hist_all[:, 0].tolist() == [2, 2, 6]
    assert layer._eld_step.item() == 10


def test_every_token_count_is_instrumented(monkeypatch, tmp_path):
    """No branch on the token count. Skipping large buckets to leave the
    prefill graph alone is a branch on a SYMBOLIC dimension -- a guard, not a
    compile-time constant -- and it makes the module's placeholder list depend
    on the specialization, which breaks the bucket hand-over (BK #322)."""
    el = _arm(monkeypatch, tmp_path)
    layer = _Layer(global_num_experts=4)
    el.register_buffers(layer, device=torch.device("cpu"))

    for tokens in (1, 16, 8192):
        el.record(layer, torch.zeros(tokens, 2, dtype=torch.long),
                  torch.ones(tokens, 2))
    assert layer._eld_calls.sum().item() == 3
    assert layer._eld_hist_all[:, 0].sum().item() == 2 * (1 + 16 + 8192)


def test_reset_zeroes_everything(monkeypatch, tmp_path):
    el = _arm(monkeypatch, tmp_path)
    model = _Model(n_layers=2, global_num_experts=4)
    for layer in model.layers:
        el.register_buffers(layer, device=torch.device("cpu"))
        el.record(layer, torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 1))
    el.reset(model)
    for layer in model.layers:
        assert layer._eld_hist_all.sum().item() == 0
        assert layer._eld_step.item() == 0


def test_dump_is_well_formed_and_self_describing(monkeypatch, tmp_path):
    el = _arm(monkeypatch, tmp_path, slots=4, bucket=2)
    model = _Model(n_layers=3, global_num_experts=16, ep_size=4)
    for layer in model.layers:
        el.register_buffers(layer, device=torch.device("cpu"))
        el.record(layer, torch.tensor([[0, 5]]), torch.ones(1, 2))

    path = el.dump(model, rank=0)
    assert path is not None and path.endswith("expert_load_rank0.npz")
    assert not list((tmp_path / "eld").glob("*.tmp")), "left a partial file"

    z = np.load(path, allow_pickle=False)
    assert z["hist_all"].shape == (3, 4, 16)
    assert z["hist_real"].shape == (3, 4, 16)
    assert z["rows"].shape == (3, 4, 2)
    assert z["calls"].shape == (3, 4)
    assert z["bucket"] == 2 and z["rank"] == 0
    assert [n.split(".")[-1] for n in z["layer_names"]] == ["0", "1", "2"]
    # 16 experts over 4 EP ranks, linear placement, plus the closing bound.
    np.testing.assert_array_equal(z["experts_start"], [0, 4, 8, 12, 16])
    assert z["hist_all"][:, 0, 0].tolist() == [1, 1, 1]


def test_only_listed_ranks_write(monkeypatch, tmp_path):
    el = _arm(monkeypatch, tmp_path, TPU_MOE_EXPERT_LOAD_RANKS="0,3")
    model = _Model(n_layers=1, global_num_experts=4)
    el.register_buffers(model.layers[0], device=torch.device("cpu"))
    assert el.dump(model, rank=3) is not None
    assert el.dump(model, rank=1) is None


def test_experts_start_table_matches_get_experts_start(monkeypatch, tmp_path):
    """The table is what turns per-expert counts into per-rank counts, so it
    has to agree with the placement the kernel actually uses."""
    el = _arm(monkeypatch, tmp_path)
    from vllm_torchtpu.layers.vllm import moe_routing
    for ep_size, n_experts in [(4, 16), (4, 18), (32, 896)]:
        layer = _Layer(global_num_experts=n_experts, ep_size=ep_size)
        table = el._experts_start_table(layer)
        assert table[-1] == n_experts
        for rank in range(ep_size):
            layer.moe_config.ep_rank = rank
            assert table[rank] == moe_routing.get_experts_start(layer)


def test_driver_resets_on_the_first_step_then_dumps(monkeypatch, tmp_path):
    el = _arm(monkeypatch, tmp_path, TPU_MOE_EXPERT_LOAD_EVERY=4)
    model = _Model(n_layers=1, global_num_experts=4)
    layer = model.layers[0]
    el.register_buffers(layer, device=torch.device("cpu"))
    # Compile warm-up: work no client asked for, must not occupy slot 0.
    el.record(layer, torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 1))

    driver = el.DumpDriver(rank=0)
    runner = type("R", (), {"get_model": lambda self: model})()
    driver.step(runner)
    assert layer._eld_step.item() == 0
    assert not (tmp_path / "eld" / "expert_load_rank0.npz").exists()

    for _ in range(3):
        el.record(layer, torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 1))
        driver.step(runner)
    assert (tmp_path / "eld" / "expert_load_rank0.npz").exists()


def test_route_records_through_the_real_entry_point(monkeypatch, tmp_path):
    """moe_routing.route() is the single hook; prove the wiring, not the stub."""
    el = _arm(monkeypatch, tmp_path)
    moe_routing = importlib.reload(
        importlib.import_module("vllm_torchtpu.layers.vllm.moe_routing"))

    layer = _Layer(global_num_experts=8)
    layer.moe_config.experts_per_token = 2
    layer.renormalize = False
    layer.use_grouped_topk = False
    el.register_buffers(layer, device=torch.device("cpu"))

    logits = torch.zeros(4, 8)
    logits[:, 3] = 10.0  # every token's top-1 is expert 3
    logits[:, 6] = 5.0
    _, ids = moe_routing.route(layer, torch.zeros(4, 8), logits)
    assert set(ids.reshape(-1).tolist()) == {3, 6}
    assert layer._eld_hist_all[0, 3].item() == 4
    assert layer._eld_hist_all[0, 6].item() == 4
    assert layer._eld_hist_all.sum().item() == 8
