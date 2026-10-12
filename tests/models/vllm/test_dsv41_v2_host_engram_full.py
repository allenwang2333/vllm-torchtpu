# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""100% statement + branch coverage tests for DeepSeek-V4.1 v2/host_engram.py."""

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import logging
import os
import sys
import types
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

import numpy as np
import pytest
import torch
import torch._dynamo

if importlib.util.find_spec("vllm") is None:
    if "tpu" not in torch._dynamo.list_backends():
        torch._dynamo.register_backend(
            name="tpu", compiler_fn=lambda gm, *a, **kw: gm.forward
        )

    class _StubModule(types.ModuleType):
        def __init__(self, name: str):
            super().__init__(name)
            self.__path__ = []  # type: ignore[attr-defined]
            self.__file__ = f"<stub:{name}>"
            self.__qualname__ = name
            self.__globals__ = {}
            self.__code__ = (lambda *a, **kw: None).__code__
            self.__defaults__ = None
            self.__kwdefaults__ = None
            self.__closure__ = None

        def __mro_entries__(self, bases):
            if not any(
                isinstance(b, type) and issubclass(b, torch.nn.Module) for b in bases
            ):
                return (type(self.__name__.rsplit(".", 1)[-1], (torch.nn.Module,), {}),)
            return (type(self.__name__.rsplit(".", 1)[-1], (), {}),)

        def __or__(self, other):
            return object

        def __ror__(self, other):
            return object

        def __getattr__(self, attr: str):
            if attr.startswith("__"):
                raise AttributeError(attr)
            full = f"{self.__name__}.{attr}"
            if full in sys.modules:
                return sys.modules[full]
            sub = _StubModule(full)
            sys.modules[full] = sub
            setattr(self, attr, sub)
            return sub

        def __call__(self, *args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return type(
                "_CtxStub",
                (SimpleNamespace,),
                {
                    "__enter__": lambda s: s,
                    "__exit__": lambda s, *a: False,
                    "__call__": lambda s, *a, **kw: (
                        a[0] if len(a) == 1 and callable(a[0]) else s
                    ),
                },
            )()

    class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        PREFIXES = (
            "portpicker",
            "absl",
            "requests",
            "vllm",
            "torch_tpu",
            "tpu_info",
            "vllm_torchtpu.layers.adapter",
            "vllm_torchtpu.model_loader_patches",
        )

        def find_spec(self, fullname, path, target=None):
            if getattr(
                self, "block_common_engram_attrs", False
            ) and fullname.startswith("vllm.models.deepseek_v41.common.engram."):
                return None
            if any(
                fullname == p or fullname.startswith(p + ".") for p in self.PREFIXES
            ):
                return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
            return None

        def create_module(self, spec):
            if spec.name in sys.modules:
                return sys.modules[spec.name]
            mod = _StubModule(spec.name)
            sys.modules[spec.name] = mod
            if "." in spec.name:
                parent_name, attr_name = spec.name.rsplit(".", 1)
                parent = sys.modules.get(parent_name)
                if parent is not None:
                    setattr(parent, attr_name, mod)
            return mod

        def exec_module(self, module):
            pass

    sys.meta_path.insert(0, _StubFinder())
    import vllm.logger

    def _init_logger(name: str):
        log = logging.getLogger(name)
        log.info_once = log.info  # type: ignore[attr-defined]
        return log

    vllm.logger.init_logger = _init_logger  # type: ignore[attr-defined]

from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutKind
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import host_engram as he_mod

he_mod.DEAD_ID = -1


def test_import_fallbacks_and_can_hash_on_host(monkeypatch) -> None:
    inp_none = he_mod.HostEngramInputs(
        input_ids=torch.zeros(4, dtype=torch.int32),
        positions=torch.zeros(4, dtype=torch.int32),
        query_start_loc=torch.zeros(2, dtype=torch.int32),
        lookback_token_ids=torch.zeros((2, 2), dtype=torch.int32),
        all_prompt=True,
        host_ready=None,
    )
    assert inp_none.can_hash_on_host is True
    inp_ready = he_mod.HostEngramInputs(
        input_ids=torch.zeros(4, dtype=torch.int32),
        positions=torch.zeros(4, dtype=torch.int32),
        query_start_loc=torch.zeros(2, dtype=torch.int32),
        lookback_token_ids=torch.zeros((2, 2), dtype=torch.int32),
        all_prompt=True,
        host_ready=False,
    )
    assert inp_ready.can_hash_on_host is False

    # Exercise top-level and function-level ImportError fallbacks directly on he_mod via
    # reload
    stub_finder = sys.meta_path[0]
    common_mod = sys.modules["vllm.models.deepseek_v41.common.engram"]
    common_mod.DEAD_ID = -1  # type: ignore[attr-defined]
    nv_mod = types.ModuleType("vllm.models.deepseek_v41.nvidia.engram")
    nv_mod.engram_gathered_num_tokens = lambda: 0  # type: ignore[attr-defined]
    nv_mod.gather_engram_hashes = lambda x, **kw: x  # type: ignore[attr-defined]
    sys.modules["vllm.models.deepseek_v41.nvidia.engram"] = nv_mod
    common_parent = sys.modules["vllm.models.deepseek_v41.common"]
    plain_common = types.ModuleType("vllm.models.deepseek_v41.common.engram")
    plain_common.DEAD_ID = -1  # type: ignore[attr-defined]
    sys.modules["vllm.models.deepseek_v41.common.engram"] = plain_common
    common_parent.engram = plain_common  # type: ignore[attr-defined]
    for k in list(sys.modules):
        if k.startswith("vllm.models.deepseek_v41.common.engram."):
            sys.modules.pop(k, None)
    stub_finder.block_common_engram_attrs = True  # type: ignore[attr-defined]
    try:
        importlib.reload(he_mod)
        he_mod.get_engram_dp_group = lambda: None  # type: ignore[assignment]
        t_hash = torch.tensor([[1, 2]], dtype=torch.int32)
        assert (
            he_mod.gather_host_engram_hashes(t_hash, dp_shared_memory=False) is t_hash
        )
    finally:
        stub_finder.block_common_engram_attrs = False  # type: ignore[attr-defined]
        sys.modules["vllm.models.deepseek_v41.common.engram"] = common_mod
        common_parent.engram = common_mod  # type: ignore[attr-defined]
        common_mod.engram_gathered_num_tokens = lambda: 0  # type: ignore[attr-defined]
        common_mod.gather_engram_hashes = lambda x, **kw: x  # type: ignore[attr-defined]
        importlib.reload(he_mod)
        he_mod.DEAD_ID = -1


def test_stage_and_resolve_next_tokens_and_async_source(monkeypatch) -> None:
    runner = SimpleNamespace()
    # Non-tensor / CPU tensor early exit
    he_mod._stage_async_next_tokens_cpu(runner, SimpleNamespace(next_tokens_tpu=None))
    assert runner._host_engram_staged_next is None
    he_mod._stage_async_next_tokens_cpu(
        runner, SimpleNamespace(next_tokens_tpu=torch.tensor([1, 2], dtype=torch.int32))
    )
    assert runner._host_engram_staged_next is None

    # Fake TPU-like tensor object (ndim=2 and ndim=1)
    class _FakeTpuTensor(torch.Tensor):
        @property
        def device(self):  # type: ignore[override]
            return SimpleNamespace(type="xla")

        def reshape(self, *shape):  # type: ignore[override]
            t = super().reshape(*shape)
            return t.as_subclass(_FakeTpuTensor)

        def to(self, *args, **kwargs):  # type: ignore[override]
            return torch.tensor(self.detach().tolist(), dtype=torch.int32)

    events_recorded = []

    class _FakeEvent:
        def record(self):
            events_recorded.append("record")

        def synchronize(self):
            events_recorded.append("sync")

    monkeypatch.setattr(torch, "tpu", SimpleNamespace(Event=_FakeEvent), raising=False)
    pre2d = SimpleNamespace(
        next_tokens_tpu=torch.tensor([[10, 11]], dtype=torch.int32).as_subclass(
            _FakeTpuTensor
        )
    )
    he_mod._stage_async_next_tokens_cpu(runner, pre2d)
    resolved = he_mod._resolve_next_tokens_cpu(pre2d, runner=runner)
    assert resolved is not None and resolved.tolist() == [10, 11]
    # Second call sees ready_event=None branch
    resolved2 = he_mod._resolve_next_tokens_cpu(pre2d, runner=runner)
    assert resolved2 is not None and resolved2.tolist() == [10, 11]

    pre1d = SimpleNamespace(
        next_tokens_tpu=torch.tensor([7], dtype=torch.int32).as_subclass(_FakeTpuTensor)
    )
    he_mod._stage_async_next_tokens_cpu(runner, pre1d)
    # Direct un-staged _resolve_next_tokens_cpu with xla tensor, cpu tensor, and None
    assert he_mod._resolve_next_tokens_cpu(
        pre2d, runner=SimpleNamespace()
    ).tolist() == [
        10,
        11,
    ]
    assert he_mod._resolve_next_tokens_cpu(
        SimpleNamespace(next_tokens_tpu=torch.tensor([5], dtype=torch.int32)),
        runner=SimpleNamespace(),
    ).tolist() == [5]
    assert (
        he_mod._resolve_next_tokens_cpu(SimpleNamespace(), runner=SimpleNamespace())
        is None
    )

    # _async_source_tokens_cpu branches
    assert he_mod._async_source_tokens_cpu(
        pre1d, prefer_next_tokens_tpu=True, runner=runner
    ).tolist() == [7]
    # prefer_next_tokens_tpu=True with None next_tokens falls through to copy_state
    pre_chunks1 = SimpleNamespace(
        next_tokens_tpu=None,
        copy_state=SimpleNamespace(
            chunks_cpu=[torch.tensor([[3, 4]], dtype=torch.int32)],
            wait=lambda: events_recorded.append("copy_wait"),
        ),
    )
    assert he_mod._async_source_tokens_cpu(
        pre_chunks1, prefer_next_tokens_tpu=True, waited=None
    ).tolist() == [3, 4]
    pre_chunks1_1d = SimpleNamespace(
        next_tokens_tpu=None,
        copy_state=SimpleNamespace(
            chunks_cpu=[torch.tensor([8, 9], dtype=torch.int32)],
            wait=None,
        ),
        wait_for_copy=lambda: None,
    )
    assert he_mod._async_source_tokens_cpu(pre_chunks1_1d).tolist() == [8, 9]
    pre_chunks2 = SimpleNamespace(
        next_tokens_tpu=None,
        copy_state=SimpleNamespace(
            chunks_cpu=[
                torch.tensor([1], dtype=torch.int32),
                torch.tensor([2], dtype=torch.int32),
            ],
            wait=None,
        ),
        wait_for_copy=None,
    )
    assert he_mod._async_source_tokens_cpu(pre_chunks2).tolist() == [1, 2]
    pre_waited = SimpleNamespace(next_tokens_tpu=None, copy_state=None)
    assert he_mod._async_source_tokens_cpu(
        pre_waited, waited=torch.tensor([[6, 7]], dtype=torch.int32)
    ).tolist() == [6, 7]
    assert he_mod._async_source_tokens_cpu(pre_waited, waited=None) is None


def test_resolve_async_spec_state_and_lookback_and_gather(monkeypatch) -> None:
    runner = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=0)
    )
    assert he_mod._resolve_async_spec_state(runner, SimpleNamespace()) is None

    runner.speculative_config = SimpleNamespace(num_speculative_tokens=2)
    pre_no_tensor = SimpleNamespace(
        req_id_to_index_copy={"r0": 0}, wait_for_copy=lambda: None
    )
    assert he_mod._resolve_async_spec_state(runner, pre_no_tensor) is None

    pre_bad_idx = SimpleNamespace(
        req_id_to_index_copy={"r0": 5},
        wait_for_copy=lambda: torch.tensor([[1, 2]], dtype=torch.int32),
    )
    assert he_mod._resolve_async_spec_state(runner, pre_bad_idx) is None

    # All-valid uniform drafts + spec_num_rejected not None + discard valid & OOB
    pre_ok = SimpleNamespace(
        req_id_to_index_copy={"r0": 0, "r1": 1},
        wait_for_copy=lambda: torch.tensor([[10, 11], [12, 13]], dtype=torch.int32),
        spec_decode_num_rejected_tokens=torch.tensor([1, 1], dtype=torch.int32),
        num_draft_per_req={0: 2, 1: 2},
        discard_sampled_tokens_req_indices=[0, 99],
    )
    st1 = he_mod._resolve_async_spec_state(runner, pre_ok, identity_num_reqs=2)
    assert st1 is not None and st1.lengths.tolist() == [0, 2]

    # Non-monotonic -1 tokens + non-uniform drafts + all OOB discard
    pre_ragged = SimpleNamespace(
        req_id_to_index_copy={"r0": 0, "r1": 1},
        wait_for_copy=lambda: torch.tensor(
            [[10, -1, 11], [-1, 12, -1]], dtype=torch.int32
        ),
        spec_decode_num_rejected_tokens=torch.tensor([1, 2], dtype=torch.int32),
        num_draft_per_req={0: 2, 1: 0},
        discard_sampled_tokens_req_indices=[99],
    )
    st2 = he_mod._resolve_async_spec_state(runner, pre_ragged)
    assert st2 is not None and st2.tokens[0, :2].tolist() == [10, 11]

    # Monotonic -1 tokens + empty num_draft_per_req + spec_num_rejected=None
    pre_mono = SimpleNamespace(
        req_id_to_index_copy={"r0": 0},
        wait_for_copy=lambda: torch.tensor([10, -1], dtype=torch.int32),
        spec_decode_num_rejected_tokens=None,
        num_draft_per_req={},
        discard_sampled_tokens_req_indices=[],
    )
    st3 = he_mod._resolve_async_spec_state(runner, pre_mono)
    assert st3 is not None and st3.stride == 1

    # _acquire_host_scratch initial clone, partial sync (sync_start < sync_len), and
    # full_overwrite
    t_src = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    s1 = he_mod._acquire_host_scratch(
        runner, "_test_scr", t_src, 3, full_overwrite=False
    )
    assert s1.tolist() == [1, 2, 3, 4]
    s1.numpy()[:] = 0
    s_sync = he_mod._acquire_host_scratch(
        runner, "_test_scr", t_src, 2, full_overwrite=False
    )
    assert s_sync[:3].tolist() == [1, 2, 3]
    s2 = he_mod._acquire_host_scratch(
        runner, "_test_scr", t_src, 4, full_overwrite=True
    )
    assert s2 is s1

    # _ragged_arange uniform and ragged
    assert he_mod._ragged_arange(
        np.array([2, 6], dtype=np.int32), np.array([2, 2], dtype=np.int32)
    ).tolist() == [2, 3, 6, 7]
    assert he_mod._ragged_arange(
        np.array([2, 6], dtype=np.int32), np.array([1, 2], dtype=np.int32)
    ).tolist() == [2, 6, 7]

    # _dummy_host_inputs cache miss & hit (both with qsl tensor and qsl=None)
    d_runner = SimpleNamespace(
        lookback_token_ids=SimpleNamespace(cpu=torch.zeros((2, 3), dtype=torch.int32)),
        max_num_tokens=8,
        max_num_reqs=2,
    )
    d_ctx = SimpleNamespace(
        position_ids_override=torch.zeros((1, 4), dtype=torch.int32),
        query_start_loc=None,
        num_reqs=2,
    )
    d1 = he_mod._dummy_host_inputs(d_runner, d_ctx)
    d2 = he_mod._dummy_host_inputs(d_runner, d_ctx)
    assert d1 is d2

    # _resolve_decode_lookback_cpu: num_reqs<=0, in-bounds, partial valid, all-invalid,
    # overlay fast & slow
    lb_runner = SimpleNamespace(
        lookback_token_ids=SimpleNamespace(cpu=torch.zeros((4, 2), dtype=torch.int32))
    )
    assert he_mod._resolve_decode_lookback_cpu(
        lb_runner, SimpleNamespace(), 0, 0
    ).shape == (4, 2)
    batch_tensor = SimpleNamespace(
        num_computed_tokens_cpu=np.array([3, 3, 1, 0], dtype=np.int32),
        token_ids_cpu=torch.tensor(
            [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33], [40, 41, 42, 43]],
            dtype=torch.int32,
        ),
    )
    # Fast in-bounds slice
    out_fast = he_mod._resolve_decode_lookback_cpu(lb_runner, batch_tensor, 2, 0)
    assert out_fast[:2].tolist() == [[12, 11], [22, 21]]
    # Partial out-of-bounds with numpy array token_ids_cpu + sampled_overlay partial hit
    batch_np = SimpleNamespace(
        num_computed_tokens_cpu=np.array([1, 0], dtype=np.int32),
        token_ids_cpu=batch_tensor.token_ids_cpu.numpy(),
    )
    ov_partial = he_mod._SampledOverlay(
        tokens=np.array([[91, 92], [93, 94]], dtype=np.int32),
        starts=np.array([0.0, 0.0], dtype=np.float32),
        lengths=np.array([1, 0], dtype=np.int32),
    )
    out_part = he_mod._resolve_decode_lookback_cpu(
        lb_runner,
        batch_np,
        2,
        0,
        rejected_by_req=np.array([0.0, 0.0], dtype=np.float32),
        sampled_overlay=ov_partial,
    )
    assert out_part[0, 0].item() == 91
    # All-invalid lookback positions (starts_1d = [0]) + overlay miss
    ov_miss = he_mod._SampledOverlay(
        tokens=np.array([[91, 92]], dtype=np.int32),
        starts=np.array([5], dtype=np.int32),
        lengths=np.array([1], dtype=np.int32),
    )
    out_all_inv = he_mod._resolve_decode_lookback_cpu(
        lb_runner,
        batch_tensor,
        1,
        3,
        rejected_by_req=np.array([1], dtype=np.int32),
        sampled_overlay=ov_miss,
    )
    assert out_all_inv[0].tolist() == [-1, -1]
    # Overlay full hit uniform delta vs non-uniform delta
    ov_full_unif = he_mod._SampledOverlay(
        tokens=np.array([[70, 71, 72], [80, 81, 82]], dtype=np.int32),
        starts=np.array([1, 1], dtype=np.int32),
        lengths=np.array([3, 3], dtype=np.int32),
    )
    out_ov_u = he_mod._resolve_decode_lookback_cpu(
        lb_runner,
        batch_tensor,
        2,
        0,
        sampled_overlay=ov_full_unif,
        raw_starts=np.array([3, 3], dtype=np.int32),
    )
    assert out_ov_u[:2].tolist() == [[71, 70], [81, 80]]
    ov_full_nonunif = he_mod._SampledOverlay(
        tokens=np.array([[70, 71, 72], [80, 81, 82]], dtype=np.int32),
        starts=np.array([0, 1], dtype=np.int32),
        lengths=np.array([3, 3], dtype=np.int32),
    )
    out_ov_nu = he_mod._resolve_decode_lookback_cpu(
        lb_runner,
        batch_tensor,
        2,
        0,
        sampled_overlay=ov_full_nonunif,
        raw_starts=np.array([3, 3], dtype=np.int32),
    )
    assert out_ov_nu[:2].tolist() == [[72, 71], [81, 80]]

    # gather_host_engram_hashes branches: dp_shared_memory, gather_fn, xla offload,
    # cpu_group slot errors/padding/exact, fallback
    hashes = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    assert (
        he_mod.gather_host_engram_hashes(
            hashes, dp_shared_memory=True, gather_fn=lambda x, **kw: x
        )
        is hashes
    )
    assert he_mod.gather_host_engram_hashes(hashes, dp_shared_memory=True) is hashes

    def _fake_ag_into(out, inp, group=None):
        out[: inp.shape[0]].copy_(inp)
        out[inp.shape[0] :].copy_(inp)

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", _fake_ag_into)
    monkeypatch.setattr(
        he_mod,
        "get_engram_dp_group",
        lambda: SimpleNamespace(
            world_size=2, cpu_group="cpu_grp", all_gather=lambda x, dim=0: x
        ),
    )
    monkeypatch.setattr(he_mod, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        he_mod,
        "get_forward_context",
        lambda: SimpleNamespace(dp_metadata=SimpleNamespace(), additional_kwargs={}),
    )
    monkeypatch.setattr(he_mod, "engram_gathered_num_tokens", lambda: 1)
    with pytest.raises(ValueError, match="exceeds"):
        he_mod.gather_host_engram_hashes(hashes)

    monkeypatch.setattr(he_mod, "engram_gathered_num_tokens", lambda: 4)
    gathered_pad = he_mod.gather_host_engram_hashes(hashes)
    assert gathered_pad.shape == (8, 2)
    assert gathered_pad[2:4].tolist() == [[-1, -1], [-1, -1]]

    monkeypatch.setattr(he_mod, "engram_gathered_num_tokens", lambda: 2)
    gathered_exact = he_mod.gather_host_engram_hashes(hashes)
    assert gathered_exact.shape == (4, 2)

    # XLA device -> .to("cpu") offload path vs non-cpu all_gather path
    class _FakeXlaHashes(torch.Tensor):
        @property
        def device(self):  # type: ignore[override]
            return SimpleNamespace(type="xla")

        def to(self, *args, **kwargs):  # type: ignore[override]
            return torch.tensor(self.detach().tolist(), dtype=torch.int32)

    xla_hashes = hashes.as_subclass(_FakeXlaHashes)
    monkeypatch.setattr(he_mod.envs, "TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD", True)
    monkeypatch.setattr(he_mod.envs, "TPU_DEEPSEEK_V41_ENGRAM_HOST_DMA", False)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    monkeypatch.setattr(he_mod, "is_forward_context_available", lambda: False)
    gathered_off = he_mod.gather_host_engram_hashes(xla_hashes)
    assert gathered_off.shape == (4, 2)

    monkeypatch.setattr(he_mod.envs, "TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD", False)
    assert he_mod.gather_host_engram_hashes(xla_hashes) is xla_hashes
    assert (
        he_mod.gather_host_engram_hashes(xla_hashes, gather_fn=lambda x, **kw: x)
        is xla_hashes
    )


def test_capture_and_install_host_engram_inputs_all_branches(monkeypatch) -> None:
    base_runner = SimpleNamespace(
        input_ids_cpu=torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int32),
        positions_cpu=torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 3, 6], dtype=torch.int32),
        lookback_token_ids=SimpleNamespace(cpu=torch.zeros((2, 2), dtype=torch.int32)),
        seq_lens_np=np.array([4, 4], dtype=np.int32),
        max_num_tokens=6,
        max_num_reqs=2,
        input_batch=SimpleNamespace(
            num_prompt_tokens=np.array([2, 2], dtype=np.int32),
            num_computed_tokens_cpu=np.array([3, 3], dtype=np.int32),
            token_ids_cpu=torch.tensor(
                [[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.int32
            ),
            req_ids=["r0", "r1"],
            num_tokens_no_spec=np.array([4, 4], dtype=np.int32),
        ),
        _last_sequence_layout_plan=SimpleNamespace(
            kind=SequenceLayoutKind.ALL, start_index=0, num_reqs=2
        ),
        _attn_metadata_builder_ctx=None,
    )

    # 1. position_ids_override with non-ALL descriptor -> None, then ALL -> dummy
    base_runner._attn_metadata_builder_ctx = SimpleNamespace(
        position_ids_override=torch.zeros((1, 4), dtype=torch.int32),
        sequence_layout_descriptor=SimpleNamespace(kind="non_all"),
    )
    assert he_mod.capture_host_engram_inputs(base_runner, 2) is None
    base_runner._attn_metadata_builder_ctx.sequence_layout_descriptor.kind = (
        SequenceLayoutKind.ALL
    )
    assert he_mod.capture_host_engram_inputs(base_runner, 2).all_prompt is True

    # 2. plan=None -> None
    base_runner._attn_metadata_builder_ctx = None
    base_runner._last_sequence_layout_plan = None
    assert he_mod.capture_host_engram_inputs(base_runner, 2) is None

    # 3. all_prompt=True fast path
    base_runner._last_sequence_layout_plan = SimpleNamespace(
        kind=SequenceLayoutKind.ALL, start_index=0, num_reqs=2
    )
    base_runner.seq_lens_np = np.array([2, 2], dtype=np.int32)
    prompt_res = he_mod.capture_host_engram_inputs(base_runner, 2)
    assert prompt_res is not None and prompt_res.all_prompt is True

    # 4. unresolved when num_computed_tokens_cpu is None
    base_runner.seq_lens_np = np.array([4, 4], dtype=np.int32)
    saved_nct = base_runner.input_batch.num_computed_tokens_cpu
    base_runner.input_batch.num_computed_tokens_cpu = None
    assert he_mod.capture_host_engram_inputs(base_runner, 2).host_ready is False
    base_runner.input_batch.num_computed_tokens_cpu = saved_nct

    # 5. unresolved when is_async_spec=True but spec_state=None
    base_runner.speculative_config = SimpleNamespace(num_speculative_tokens=2)
    base_runner._pre_async_results = SimpleNamespace(
        req_id_to_index_copy=None, spec_decode_num_rejected_tokens=torch.tensor([1])
    )
    assert he_mod.capture_host_engram_inputs(base_runner, 2).host_ready is False

    # 6. Full identity uniform-stride speculative path (fast_qsl=True + rejected > 0 +
    # num_tokens_no_spec ndarray)
    base_runner._pre_async_results = SimpleNamespace(
        req_id_to_index_copy={"r0": 0, "r1": 1},
        wait_for_copy=lambda: torch.tensor(
            [[101, 102, -1], [201, 202, -1]], dtype=torch.int32
        ),
        spec_decode_num_rejected_tokens=torch.tensor([1, 1], dtype=torch.int32),
        num_draft_per_req={0: 2, 1: 2},
        discard_sampled_tokens_req_indices=[],
    )
    id_res = he_mod.capture_host_engram_inputs(base_runner, 2)
    assert id_res is not None and id_res.host_ready is True

    # 7. Second identity step with different dict instance (hits == ref_req[1] branch) +
    # non-fast_qsl sequential pre_pos
    base_runner._pre_async_results.req_id_to_index_copy = {"r0": 0, "r1": 1}
    base_runner.query_start_loc_cpu = torch.tensor([0, 2, 5], dtype=torch.int32)
    base_runner.input_batch.num_tokens_no_spec = [4, 4]
    id_res2 = he_mod.capture_host_engram_inputs(base_runner, 2)
    assert id_res2 is not None and id_res2.host_ready is True

    # 8. Non-identity partial match + num_tokens_no_spec=None + flat_src too short
    # fallback & valid ragged path
    base_runner.input_batch.num_tokens_no_spec = None
    base_runner.input_batch.num_computed_tokens_cpu = [3, 3]
    base_runner.input_batch.req_ids = ["r1", "missing", "r0"]
    base_runner.query_start_loc_cpu = torch.tensor([0, 2, 4], dtype=torch.int32)
    base_runner._pre_async_results = SimpleNamespace(
        req_id_to_index_copy={"r0": 0, "r1": 1},
        wait_for_copy=lambda: torch.tensor([[101], [201]], dtype=torch.int32),
        spec_decode_num_rejected_tokens=torch.tensor([1, 1], dtype=torch.int32),
        num_draft_per_req={0: 1, 1: 1},
        discard_sampled_tokens_req_indices=[],
    )
    # Short flat_src -> unresolved
    assert he_mod.capture_host_engram_inputs(base_runner, 2).host_ready is False

    # Valid flat_src length -> ragged scatter + non-uniform positions subtraction
    base_runner._pre_async_results.wait_for_copy = lambda: torch.tensor(
        [[101, 102, 103], [201, 202, 203]], dtype=torch.int32
    )
    nonid_res = he_mod.capture_host_engram_inputs(base_runner, 2)
    assert nonid_res is not None and nonid_res.host_ready is True

    # 9. No pre_async / chunk_req_ids=None path + no match (matched.any() == False) path
    base_runner._pre_async_results = SimpleNamespace(
        req_id_to_index_copy={"other": 0},
        wait_for_copy=lambda: torch.tensor([[101, 102, 103]], dtype=torch.int32),
        spec_decode_num_rejected_tokens=torch.tensor([0], dtype=torch.int32),
        num_draft_per_req={0: 0},
        discard_sampled_tokens_req_indices=[],
    )
    nomatch_res = he_mod.capture_host_engram_inputs(base_runner, 2)
    assert nomatch_res is not None and nomatch_res.host_ready is True

    base_runner.speculative_config = None
    base_runner.input_batch.req_ids = ["r0", "r1"]
    base_runner._pre_async_results = SimpleNamespace(
        req_id_to_index_copy={"r0": 0, "r1": 1},
        next_tokens_tpu=torch.tensor([111, 222, 333, 444], dtype=torch.int32),
        spec_decode_num_rejected_tokens=None,
    )
    nonspec_async = he_mod.capture_host_engram_inputs(base_runner, 2)
    assert nonspec_async is not None and nonspec_async.host_ready is True

    base_runner._pre_async_results = None
    plain_res = he_mod.capture_host_engram_inputs(base_runner, 2)
    assert plain_res is not None and plain_res.host_ready is True

    monkeypatch.setattr(he_mod, "get_engram_dp_group", lambda: None)
    h_small = torch.tensor([[1, 2]], dtype=torch.int32)
    assert (
        he_mod.gather_host_engram_hashes(
            h_small, dp_shared_memory=False, gather_fn=lambda x, **kw: x
        )
        is h_small
    )

    # 10. install_host_engram_inputs hook installation + idempotency + execution
    fake_runner_mod = types.ModuleType("vllm_torchtpu.runner.tpu_runner")

    class _FakeTPUModelRunner:
        def _prepare_lookback_token_ids(self, num_reqs: int) -> torch.Tensor:
            return torch.zeros((num_reqs, 2), dtype=torch.int32)

        def sample_tokens(self, grammar_output):
            return "sampled_ok"

    fake_runner_mod.TPUModelRunner = _FakeTPUModelRunner  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm_torchtpu.runner.tpu_runner", fake_runner_mod)
    he_mod.install_host_engram_inputs()
    # Second call exercises idempotency early return
    he_mod.install_host_engram_inputs()

    inst = _FakeTPUModelRunner()
    inst._attn_metadata_builder_ctx = None
    inst._last_sequence_layout_plan = None
    ctx_kwargs: dict = {}
    monkeypatch.setattr(he_mod, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        he_mod,
        "get_forward_context",
        lambda: SimpleNamespace(additional_kwargs=ctx_kwargs),
    )
    out_ids = inst._prepare_lookback_token_ids(2)
    assert out_ids.shape == (2, 2)
    assert he_mod.HOST_ENGRAM_KEY in ctx_kwargs

    monkeypatch.setattr(he_mod, "is_forward_context_available", lambda: False)
    assert inst._prepare_lookback_token_ids(2).shape == (2, 2)
    assert inst.sample_tokens(None) == "sampled_ok"
