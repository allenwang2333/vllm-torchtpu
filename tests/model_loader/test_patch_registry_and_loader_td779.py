# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Track TD7.79: patch_registry, model_loader_patches, and MTP loop."""

import os
import sys
from types import ModuleType, SimpleNamespace

import torch

from vllm_torchtpu import envs, patch_registry
from vllm_torchtpu import model_loader_patches as mlp
from vllm_torchtpu.spec_decode.mtp_loop_model import (
    MTP_DYNAMIC_ARG_DIMS,
    MtpLoopStepModel,
)


class _FakeTPUTensor(torch.Tensor):
    """CPU-backed tensor reporting device.type == 'tpu' for unit testing."""

    _fake_device = SimpleNamespace(type="tpu")

    @property
    def device(self):
        return self._fake_device

    def to(self, *args, **kwargs):
        new_args = tuple(
            torch.device("cpu")
            if getattr(a, "type", None) == "tpu" and not isinstance(a, torch.device)
            else a
            for a in args
        )
        if getattr(kwargs.get("device"), "type", None) == "tpu" and not isinstance(
            kwargs.get("device"), torch.device
        ):
            kwargs = {**kwargs, "device": torch.device("cpu")}
        out = super().to(*new_args, **kwargs)
        return out.as_subclass(_FakeTPUTensor)


def _make_tpu_param(*shape):
    t = torch.zeros(*shape).as_subclass(_FakeTPUTensor)
    return torch.nn.Parameter(t, requires_grad=False)


def test_patch_registry_direct_full_coverage(monkeypatch):
    # Direct test of _reset_after_fork on the imported patch_registry module
    monkeypatch.setattr(patch_registry, "_applying", {"dummy"})
    monkeypatch.setattr(patch_registry, "_active_stages", {"import"})
    patch_registry._reset_after_fork()
    assert not patch_registry._applying
    assert not patch_registry._active_stages

    # _invoke with module-only target (no ':name') and uncached module in sys.modules
    sys.modules.pop("antigravity_nonexistent", None)
    dummy_mod = ModuleType("_td779_dummy_mod")
    called = []
    dummy_mod.fn = lambda **kw: called.append(kw)

    orig_import = patch_registry.importlib.import_module

    def _fake_import(name):
        if name == "_td779_uncached_mod":
            sys.modules[name] = dummy_mod
            return dummy_mod
        return orig_import(name)

    monkeypatch.setattr(patch_registry.importlib, "import_module", _fake_import)
    try:
        assert patch_registry._invoke("_td779_uncached_mod") is None
        patch_registry._invoke("_td779_uncached_mod:fn", x=1)
        assert called == [{"x": 1}]
    finally:
        sys.modules.pop("_td779_uncached_mod", None)


def test_row_slice_and_should_skip_fast_paths(monkeypatch):
    monkeypatch.setattr(mlp, "_ROW_SLICES", {})
    assert mlp._row_slice_for("any.weight") is None

    mlp.register_row_slice("embed.weight", 2, 4)
    assert mlp._row_slice_for("embed.weight") == (2, 4)
    assert mlp._row_slice_for("model.embed.weight") == (2, 4)
    assert mlp._row_slice_for("my_embed.weight") is None
    assert mlp._row_slice_for("other.bias") is None

    # Force lazy resolution of _parse_expert_id on first call, then cached on second
    monkeypatch.setattr(mlp, "_parse_expert_id", None)
    assert not mlp._should_skip("layers.0.experts.1.w1.input_scale", {0})
    assert mlp._parse_expert_id is None  # suffix check short-circuited before import!
    assert mlp._should_skip("layers.0.experts.1.w1.weight", {0})
    assert mlp._parse_expert_id is not None
    assert not mlp._should_skip("layers.0.experts.0.w1.weight", {0})
    assert not mlp._should_skip("layers.0.dense.weight", {0})


def test_compute_local_expert_ids_all_branches(monkeypatch):
    from vllm import config as vllm_cfg
    from vllm import distributed as vllm_dist
    from vllm.model_executor.model_loader import default_loader

    def _make_cfg(
        *, is_moe=True, ep=True, filt=True, eplb=False, experts=8, dp=1, tp=1, pcp=1
    ):
        return SimpleNamespace(
            model_config=SimpleNamespace(
                is_moe=is_moe, get_num_experts=lambda: experts
            ),
            parallel_config=SimpleNamespace(
                enable_expert_parallel=ep,
                enable_ep_weight_filter=filt,
                enable_eplb=eplb,
                data_parallel_size=dp,
                tensor_parallel_size=tp,
                prefill_context_parallel_size=pcp,
                expert_placement_strategy="linear",
            ),
        )

    # 1. Not MoE / EP / filter -> None
    monkeypatch.setattr(
        vllm_cfg, "get_current_vllm_config", lambda: _make_cfg(is_moe=False)
    )
    assert mlp._compute_local_expert_ids() is None

    # 2. EPLB enabled -> None
    monkeypatch.setattr(
        vllm_cfg, "get_current_vllm_config", lambda: _make_cfg(eplb=True)
    )
    assert mlp._compute_local_expert_ids() is None

    # 3. num_experts <= 0 -> None
    monkeypatch.setattr(
        vllm_cfg, "get_current_vllm_config", lambda: _make_cfg(experts=0)
    )
    assert mlp._compute_local_expert_ids() is None

    # 4. dp > 1, tp > 1, pcp > 1 with local_ids = None
    monkeypatch.setattr(
        vllm_cfg,
        "get_current_vllm_config",
        lambda: _make_cfg(dp=2, tp=2, pcp=2),
    )
    monkeypatch.setattr(
        vllm_dist, "get_dp_group", lambda: SimpleNamespace(rank_in_group=1)
    )
    monkeypatch.setattr(
        vllm_dist, "get_pcp_group", lambda: SimpleNamespace(rank_in_group=1)
    )
    monkeypatch.setattr(vllm_dist, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(
        default_loader, "compute_local_expert_ids", lambda *a, **kw: None
    )
    assert mlp._compute_local_expert_ids() is None

    # 5. dp == 1, tp == 1, pcp == 1 with non-None local_ids
    monkeypatch.setattr(
        vllm_cfg,
        "get_current_vllm_config",
        lambda: _make_cfg(dp=1, tp=1, pcp=1),
    )
    monkeypatch.setattr(
        default_loader, "compute_local_expert_ids", lambda *a, **kw: {0, 1}
    )
    assert mlp._compute_local_expert_ids() == {0, 1}


def test_sharded_runai_weights_iterator_and_patch_full_coverage(monkeypatch):
    # Build fake runai_model_streamer package hierarchy in sys.modules
    runai_pkg = ModuleType("runai_model_streamer")
    fs_mod = ModuleType("runai_model_streamer.file_streamer")
    ss_pkg = ModuleType("runai_model_streamer.safetensors_streamer")
    ss_mod = ModuleType(
        "runai_model_streamer.safetensors_streamer.safetensors_streamer"
    )
    sp_mod = ModuleType("runai_model_streamer.safetensors_streamer.safetensors_pytorch")

    class _FileChunks:
        def __init__(self, req_id, path, offset, chunks):
            self.id = req_id
            self.path = path
            self.offset = offset
            self.chunks = chunks

    fs_mod.FileChunks = _FileChunks

    e = "m.layers.0.experts"
    file_meta = SimpleNamespace(
        offset=100,
        tensors_metadata=[
            SimpleNamespace(
                name=f"{e}.0.w1.weight", shape=[2, 2]
            ),  # kept @100 (size 8)
            SimpleNamespace(
                name=f"{e}.0.w2.weight", shape=[2, 2]
            ),  # kept @108 (size 8, same run)
            SimpleNamespace(
                name=f"{e}.5.w1.weight", shape=[0]
            ),  # skipped size=0 (doesn't flush)
            SimpleNamespace(
                name=f"{e}.5.w2.weight", shape=[2, 2]
            ),  # skipped size=8 (flushes run)
            SimpleNamespace(
                name=f"{e}.6.w2.weight", shape=[2, 2]
            ),  # skipped size=8 (empty flush)
            SimpleNamespace(name="m.empty.weight", shape=[0]),  # kept size=0
            SimpleNamespace(
                name="m.sliced.weight", shape=[10, 2]
            ),  # row-sliced (size 20)
            SimpleNamespace(name="m.tail.weight", shape=[2]),  # kept @144 (size 4)
        ],
        read_sizes=[8, 8, 0, 8, 8, 0, 20, 4],
    )

    captured_requests = []

    class _FakeSafetensorsStreamer:
        def __init__(self):
            self.file_streamer = self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def stream_files(self, requests, credentials, device, is_distributed):
            captured_requests.clear()
            captured_requests.extend(requests)

        def get_chunks(self):
            for req in captured_requests:
                for idx, sz in enumerate(req.chunks):
                    yield req.id, idx, torch.ones(sz, dtype=torch.uint8)

    ss_mod.SafetensorsStreamer = _FakeSafetensorsStreamer
    sp_mod.SafetensorsMetadata = SimpleNamespace(
        from_files=lambda fs, files, _: [file_meta]
    )
    sp_mod.create_torch_tensor = lambda buf, meta: torch.zeros(meta.shape)
    runai_pkg.file_streamer = fs_mod
    runai_pkg.safetensors_streamer = ss_pkg
    ss_pkg.safetensors_streamer = ss_mod
    ss_pkg.safetensors_pytorch = sp_mod

    for mod_name, mod in (
        ("runai_model_streamer", runai_pkg),
        ("runai_model_streamer.file_streamer", fs_mod),
        ("runai_model_streamer.safetensors_streamer", ss_pkg),
        ("runai_model_streamer.safetensors_streamer.safetensors_streamer", ss_mod),
        ("runai_model_streamer.safetensors_streamer.safetensors_pytorch", sp_mod),
    ):
        monkeypatch.setitem(sys.modules, mod_name, mod)

    monkeypatch.setattr(mlp, "_ROW_SLICES", {"m.sliced.weight": (2, 3)})

    # 1. sync_every = 0
    monkeypatch.setattr(envs, "TPU_SHARDED_LOAD_SYNC_EVERY", 0)
    out0 = list(mlp._sharded_runai_weights_iterator(["f0.safetensors"], {0}, False))
    assert len(out0) == 5

    # 2. sync_every = 2 with torch.accelerator.synchronize succeeding then raising
    monkeypatch.setattr(envs, "TPU_SHARDED_LOAD_SYNC_EVERY", 2)
    sync_calls = [0]

    def _fake_sync():
        sync_calls[0] += 1
        if sync_calls[0] > 1:
            raise RuntimeError("sync unavailable")

    monkeypatch.setattr(torch.accelerator, "synchronize", _fake_sync)
    out1 = list(mlp._sharded_runai_weights_iterator(["f0.safetensors"], {0}, False))
    assert len(out1) == 5
    assert sync_calls[0] >= 2

    # 3. patch_runai_sharded_expert_streaming branches
    from vllm.model_executor.model_loader import runai_streamer_loader as rsl

    orig_calls = []

    class _FakeRunaiLoader:
        _is_distributed = True
        load_config = SimpleNamespace(use_tqdm_on_load=False)

        def _get_weights_iterator(self, model_or_path, revision):
            orig_calls.append((model_or_path, revision))
            return iter([("orig", torch.zeros(1))])

        def _prepare_weights(self, model_or_path, revision):
            return ["f0.safetensors"]

    monkeypatch.setattr(rsl, "RunaiModelStreamerLoader", _FakeRunaiLoader)
    mlp.patch_runai_sharded_expert_streaming()
    loader = _FakeRunaiLoader()

    # AttributeError branch
    def _raise_attr():
        raise AttributeError("no ep field")

    monkeypatch.setattr(mlp, "_compute_local_expert_ids", _raise_attr)
    assert list(loader._get_weights_iterator("m", "rev"))[0][0] == "orig"

    # Generic Exception branch
    def _raise_err():
        raise RuntimeError("boom")

    monkeypatch.setattr(mlp, "_compute_local_expert_ids", _raise_err)
    assert list(loader._get_weights_iterator("m", "rev"))[0][0] == "orig"

    # Active filter with _is_distributed = True and _is_distributed = False
    monkeypatch.setattr(mlp, "_compute_local_expert_ids", lambda: {0})
    assert len(list(loader._get_weights_iterator("m", None))) == 5
    loader._is_distributed = False
    assert len(list(loader._get_weights_iterator("m", None))) == 5

    # Duplicate patch call hits already-patched guard (line 376)
    mlp.patch_runai_sharded_expert_streaming()


def test_prefetch_and_page_cache_eviction_full_coverage(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from vllm.model_executor.model_loader import default_loader as dl

    shard = tmp_path / "shard.safetensors"
    save_file({"dense.weight": torch.ones(4, dtype=torch.uint8)}, str(shard))

    # Run _start_ep_range_prefetch synchronously on the calling thread for coverage
    class _SyncThread:
        def __init__(self, target, *args, **kwargs):
            self._target = target

        def start(self):
            self._target()

    orig_thread = mlp.threading.Thread
    mlp.threading.Thread = _SyncThread
    try:
        mlp._start_ep_range_prefetch([str(shard)], {0}, num_threads=1, block_size=2)
        mlp._start_ep_range_prefetch([], {0}, num_threads=1, block_size=2)

        # Error branch inside _start_ep_range_prefetch (os.pread raises)
        def _bad_pread(fd, n, off):
            raise OSError("pread failed")

        monkeypatch.setattr(mlp.os, "pread", _bad_pread)
        mlp._start_ep_range_prefetch([str(shard)], {0}, num_threads=1, block_size=2)
    finally:
        mlp.threading.Thread = orig_thread

    # _evict_one_checkpoint_file branches (missing file, success, exception)
    assert mlp._evict_one_checkpoint_file(str(tmp_path / "missing.safetensors")) is None
    assert mlp._evict_one_checkpoint_file(str(shard)) is not None

    orig_fadvise = os.posix_fadvise

    def _bad_fadvise(fd, off, length, advice):
        raise OSError("fadvise error")

    monkeypatch.setattr(os, "posix_fadvise", _bad_fadvise)
    assert mlp._evict_one_checkpoint_file(str(shard)) is None

    # _evict_checkpoint_page_cache empty list and execution with files
    mlp._evict_checkpoint_page_cache([])
    mlp._evict_checkpoint_page_cache([str(shard)])
    monkeypatch.setattr(os, "posix_fadvise", orig_fadvise)
    monkeypatch.setattr(envs, "TPU_EVICT_WEIGHTS_PAGE_CACHE", True, raising=False)
    mlp._evict_checkpoint_page_cache([str(shard)])

    # patch_default_model_loader_page_cache disabled & enabled branches
    monkeypatch.setattr(envs, "TPU_EVICT_WEIGHTS_PAGE_CACHE", False, raising=False)
    mlp.patch_default_model_loader_page_cache()

    monkeypatch.setattr(envs, "TPU_EVICT_WEIGHTS_PAGE_CACHE", True, raising=False)
    monkeypatch.setattr(mlp, "_loaded_weight_files", {})

    class _FakeDefaultLoader:
        def _prepare_weights(self, *args, **kwargs):
            return args[0] if args else ("folder", ["w1.safetensors"])

        def load_weights(self, model, model_config):
            pass

    monkeypatch.setattr(dl, "DefaultModelLoader", _FakeDefaultLoader)
    mlp.patch_default_model_loader_page_cache()
    # Second call hits already-patched guard
    mlp.patch_default_model_loader_page_cache()

    dl_inst = _FakeDefaultLoader()
    # 1. load_weights before _prepare_weights (no _loaded_weight_files attr)
    dl_inst.load_weights(None, None)
    assert not mlp._loaded_weight_files
    # 2. _prepare_weights returning non-tuple
    assert dl_inst._prepare_weights("not_a_tuple") == "not_a_tuple"
    # 3. _prepare_weights returning 2-tuple -> sets _loaded_weight_files
    assert dl_inst._prepare_weights() == ("folder", ["w1.safetensors"])
    dl_inst.load_weights(None, None)
    assert list(mlp._loaded_weight_files) == ["w1.safetensors"]

    # evict_loaded_weights_page_cache: non-zero local rank branch
    monkeypatch.setenv("LOCAL_RANK", "1")
    mlp.evict_loaded_weights_page_cache()
    assert not mlp._loaded_weight_files

    # evict_loaded_weights_page_cache: empty _loaded_weight_files guard
    mlp.evict_loaded_weights_page_cache()

    # evict_loaded_weights_page_cache: exception branch on rank 0
    monkeypatch.setattr(mlp, "_loaded_weight_files", {"w1.safetensors": None})
    monkeypatch.setenv("LOCAL_RANK", "0")

    def _boom_evict(files):
        raise RuntimeError("evict boom")

    monkeypatch.setattr(mlp, "_evict_checkpoint_page_cache", _boom_evict)
    mlp.evict_loaded_weights_page_cache()

    # evict_loaded_weights_page_cache: normal execution on rank 0
    monkeypatch.undo()
    monkeypatch.setattr(envs, "TPU_EVICT_WEIGHTS_PAGE_CACHE", True, raising=False)
    monkeypatch.setattr(mlp, "_loaded_weight_files", {str(shard): None})
    monkeypatch.setenv("LOCAL_RANK", "0")
    mlp.evict_loaded_weights_page_cache()
    assert not mlp._loaded_weight_files


def test_expert_stager_and_patch_moe_expert_write_staging_full_coverage(monkeypatch):
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.model_loader import base_loader

    import vllm_torchtpu.utils as tpu_utils

    stager = mlp.ExpertParamStager()
    p_unused = _make_tpu_param(2, 4, 4)
    # Flush on unstaged parameter returns early
    stager.flush(p_unused)
    assert stager.flushed == 0

    synced = []
    monkeypatch.setattr(
        tpu_utils,
        "synchronize_tensors",
        lambda tensors, wait=True: synced.append((len(tensors), wait)),
    )

    # Flush on TPU parameter with DEBUG logging enabled and disabled, and pool reuse
    monkeypatch.setattr(mlp.logger, "isEnabledFor", lambda lvl: True)
    p1 = _make_tpu_param(1, 4, 4)
    stager.host_param(p1)[0] = 3.0
    assert stager.written(p1, 0, "w2")
    assert synced == [(1, True)]

    monkeypatch.setattr(mlp.logger, "isEnabledFor", lambda lvl: False)
    p2 = _make_tpu_param(1, 4, 4)
    stager.host_param(p2)[0] = 5.0
    assert stager.written(p2, 0, "w2")
    # Flush a third parameter of the same shape so free.append(buffer) runs
    p3 = _make_tpu_param(1, 4, 4)
    stager.host_param(p3)[0] = 7.0
    stager.flush(p2)  # already flushed -> no-op
    # Put buffer back in pool, then flush p3 while p2's buffer is already in pool
    stager._pool[(tuple(p3.shape), p3.dtype)] = [torch.zeros(1, 4, 4)]
    assert stager.written(p3, 0, "w2")
    assert len(stager._pool[(tuple(p3.shape), p3.dtype)]) == 2

    # Flush CPU parameter (device.type != 'tpu')
    cpu_p_flush = torch.nn.Parameter(torch.zeros(1, 4, 4), requires_grad=False)
    stager.host_param(cpu_p_flush)
    stager.flush(cpu_p_flush)

    # Multi-chunk flush (len(chunks) > 1)
    monkeypatch.setattr(mlp, "_UPLOAD_CHUNK_BYTES", 4)
    p_chunked = _make_tpu_param(4, 4, 4)
    stager.host_param(p_chunked)
    stager.flush(p_chunked)

    # ExpertWriteTracker unknown shard id
    tracker = mlp.ExpertWriteTracker()
    assert tracker.record(p1, 0, "unknown_shard") is False

    # Exceed _MAX_STAGED to trigger early write eviction
    stager_cap = mlp.ExpertParamStager()
    cap_params = [_make_tpu_param(1, 4, 4) for _ in range(mlp._MAX_STAGED + 1)]
    for p in cap_params:
        stager_cap.host_param(p)
    assert len(stager_cap.written_early) == 1
    assert id(cap_params[0]) in stager_cap.written_early
    assert not stager_cap.staging(cap_params[0])

    # patch_moe_expert_write_staging full coverage
    loader_calls = []

    class _FakeRoutedExperts:
        def _map_global_expert_id_to_local_expert_id(self, expert_id):
            return -1 if expert_id == 99 else expert_id

        def weight_loader(
            self,
            param,
            loaded_weight,
            weight_name,
            shard_id,
            expert_id,
            return_success=False,
        ):
            loader_calls.append((type(param), shard_id, expert_id))
            if expert_id == 88:
                return False
            if expert_id < param.shape[0]:
                param.data[expert_id] = loaded_weight
            return True

    process_calls = []
    monkeypatch.setattr(
        RoutedExperts, "weight_loader", _FakeRoutedExperts.weight_loader
    )
    monkeypatch.setattr(
        RoutedExperts, "_tpu_expert_staging_patch", False, raising=False
    )
    monkeypatch.setattr(
        base_loader,
        "process_weights_after_loading",
        lambda model, cfg, dev: process_calls.append((model, cfg, dev)) or "processed",
    )

    mlp.patch_moe_expert_write_staging()
    # Second call hits already-patched guard
    mlp.patch_moe_expert_write_staging()

    re_inst = _FakeRoutedExperts()
    re_inst.weight_loader = RoutedExperts.weight_loader.__get__(
        re_inst, _FakeRoutedExperts
    )

    # 1. CPU parameter -> direct path
    cpu_p = torch.nn.Parameter(torch.zeros(2, 4, 4), requires_grad=False)
    assert re_inst.weight_loader(
        cpu_p, torch.ones(4, 4), "w", "w2", 0, return_success=True
    )

    # 2. Remote expert (local_id == -1) -> direct path
    tpu_p = _make_tpu_param(2, 4, 4)
    assert re_inst.weight_loader(
        tpu_p, torch.ones(4, 4), "w", "w2", 99, return_success=True
    )

    # 3. Staged TPU parameter where orig_weight_loader returns False
    assert (
        re_inst.weight_loader(
            tpu_p, torch.ones(4, 4), "w", "w2", 88, return_success=True
        )
        is False
    )

    # 4. Staged TPU parameter where orig_weight_loader succeeds at calls % 1000 == 0
    freevars = dict(
        zip(
            RoutedExperts.weight_loader.__code__.co_freevars,
            (c.cell_contents for c in RoutedExperts.weight_loader.__closure__),
        )
    )
    freevars["calls"][0] = 999
    monkeypatch.setattr(mlp.logger, "isEnabledFor", lambda lvl: True)
    assert re_inst.weight_loader(
        tpu_p, torch.ones(4, 4), "w", "w2", 0, return_success=True
    )
    freevars["calls"][0] = 999
    monkeypatch.setattr(mlp.logger, "isEnabledFor", lambda lvl: False)
    assert re_inst.weight_loader(
        tpu_p, torch.ones(4, 4), "w", "w2", 1, return_success=True
    )

    # 5. Stage one incomplete parameter and call process_weights_after_loading
    tpu_incomplete = _make_tpu_param(2, 4, 4)
    assert re_inst.weight_loader(tpu_incomplete, torch.ones(4, 4), "w", "w2", 0)
    assert base_loader.process_weights_after_loading("m", "cfg", "dev") == "processed"
    assert process_calls == [("m", "cfg", "dev")]


def test_mtp_loop_step_model_forward():
    from vllm.config import CompilationMode

    assert set(MTP_DYNAMIC_ARG_DIMS) == {
        "input_ids",
        "positions",
        "hidden_states",
        "intermediate_tensors",
        "inputs_embeds",
    }

    class _DummyMTP(torch.nn.Module):
        def forward(
            self,
            input_ids,
            positions,
            hidden_states,
            intermediate_tensors=None,
            inputs_embeds=None,
            spec_step_idx=0,
        ):
            return hidden_states + float(spec_step_idx)

    vllm_cfg = SimpleNamespace(
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE)
    )
    mtp = _DummyMTP()
    model = MtpLoopStepModel(vllm_config=vllm_cfg, prefix="mtp_loop", mtp=mtp)
    out = model(
        torch.tensor([1, 2]),
        torch.tensor([0, 1]),
        torch.ones(2, 4),
        spec_step_idx=3,
    )
    assert torch.equal(out, torch.full((2, 4), 4.0))
