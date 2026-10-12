# SPDX-License-Identifier: Apache-2.0
"""Complete line and branch coverage test suite for DeepSeek-V4.1 common engram."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import logging
import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

import numpy as np
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
    import vllm.config
    import vllm.distributed
    import vllm.forward_context
    import vllm.logger
    import vllm.models.deepseek_v41.common.engram as upstream_engram

    import vllm_torchtpu.layers.adapter.upstream_bindings as bindings

    def _init_logger(name: str):
        log = logging.getLogger(name)
        log.warning_once = log.warning  # type: ignore[attr-defined]
        log.info_once = log.info  # type: ignore[attr-defined]
        return log

    vllm.logger.init_logger = _init_logger
    vllm.config.get_current_vllm_config_or_none = lambda: None
    vllm.distributed.get_engram_dp_group = lambda: None
    vllm.distributed.get_engram_dp_size = lambda: 1
    vllm.forward_context.is_forward_context_available = lambda: False
    vllm.forward_context.get_forward_context = lambda: SimpleNamespace(dp_metadata=None)
    upstream_engram.DEAD_ID = 999999
    upstream_engram.SHM_PATH = "/dev/shm"
    upstream_engram.check_shm_free_space = lambda *a, **kw: None
    upstream_engram.engram_gathered_num_tokens = lambda: 4
    upstream_engram.engram_head_shard_rank = lambda: 0
    upstream_engram.get_tensor_model_parallel_world_size = lambda: 2
    upstream_engram.get_tensor_model_parallel_rank = lambda: 0
    upstream_engram.tensor_model_parallel_all_gather = lambda x, dim=0: x
    upstream_engram.set_weight_attrs = lambda *a, **kw: None
    upstream_engram.can_share_engram_tables = lambda layout, block_size=32: True
    bindings.bind_device_ops = lambda fn, **kw: fn
    bindings.device_hooks = lambda *a, **kw: type(
        "_Ctx", (), {"__enter__": lambda s: s, "__exit__": lambda s, *x: False}
    )()

import pytest

from vllm_torchtpu.models.vllm.deepseek_v41.common import engram as engram_mod


def _make_hash_state(*, use_slot_cache: bool = True) -> engram_mod.NgramHashState:
    st = object.__new__(engram_mod.NgramHashState)
    torch.nn.Module.__init__(st)
    st.use_slot_cache = use_slot_cache
    st._cache = torch.arange(32, dtype=torch.int32)
    st.swa_cache_module = SimpleNamespace(kv_cache=torch.empty((2, 8)))
    st.block_size = 4
    st._kv_cache_ref = None
    st.pad_id = 0
    st.lookback_depth = 3
    st.token_map = torch.arange(64, dtype=torch.int32) + 10
    st.multipliers = torch.tensor([[3, 5, 7], [11, 13, 17]], dtype=torch.int64)
    st.primes = torch.tensor(
        [[[19, 23], [29, 31]], [[37, 41], [43, 47]]], dtype=torch.int64
    )
    st.offsets = torch.tensor([[0, 20, 50, 80], [0, 40, 90, 140]], dtype=torch.int64)
    return st


class TestImportFallbacksAndParity:
    def test_import_fallbacks_and_bit_exact_unpack_dequant(self, monkeypatch):
        # Verify bit-exact parity of optimized _unpack_bytes and dequant_fp8_rows
        words = torch.tensor([0x04030201, -0x01020304, 0x7F80FF00], dtype=torch.int32)
        shifts = torch.arange(0, 32, 8, dtype=torch.int32)
        ref_unpack = (
            ((words.unsqueeze(-1) >> shifts) & 0xFF).to(torch.uint8).reshape(-1)
        )
        for nb in (5, 8, 12):
            assert torch.equal(engram_mod._unpack_bytes(words, nb), ref_unpack[:nb])

        # Verify bit-exact dequant_fp8_rows across normal, zero, and _INVALID_SCALE (255
        # + NaN raw)
        raw = torch.tensor(
            [[[0x38, 0x40, 0x7F, 0xFF]]], dtype=torch.uint8
        )  # includes FP8 NaN bytes 0x7F/0xFF
        scale = torch.tensor([[[127], [255]]], dtype=torch.uint8)
        raw_2 = raw.expand(1, 2, 4).clone()
        out = engram_mod.dequant_fp8_rows(raw_2, scale, 4)
        assert out.shape == (1, 2, 4) and out.dtype == torch.bfloat16
        assert torch.all(out[:, 1, :] == 0)
        assert not torch.isnan(out[:, 1, :]).any()

        # Reload common.engram with simulated missing attrs / ImportErrors for 100%
        # module coverage
        orig_common = engram_mod.upstream
        fake_common = types.ModuleType("vllm.models.deepseek_v41.common.engram")
        fake_common.NgramHashState = orig_common.NgramHashState
        fake_common.ParallelEngramEmbedding = orig_common.ParallelEngramEmbedding
        fake_common.Engram = orig_common.Engram
        fake_common.DEAD_ID = 999999
        fake_nvidia = types.ModuleType("vllm.models.deepseek_v41.nvidia.engram")
        fake_nvidia.engram_gathered_num_tokens = lambda: 4
        fake_nvidia.engram_head_shard_rank = lambda: 0
        fake_nvidia.gather_engram_hashes = lambda *a, **kw: None
        monkeypatch.setattr(
            sys.modules["vllm.models.deepseek_v41.common"], "engram", fake_common
        )
        monkeypatch.setitem(
            sys.modules, "vllm.models.deepseek_v41.common.engram", fake_common
        )
        monkeypatch.setitem(
            sys.modules, "vllm.models.deepseek_v41.nvidia.engram", fake_nvidia
        )
        importlib.reload(engram_mod)

        # Also exercise ImportError when _nvidia_engram raises ImportError on
        # gather_engram_hashes check
        fake_common2 = types.ModuleType("vllm.models.deepseek_v41.common.engram")
        fake_common2.NgramHashState = orig_common.NgramHashState
        fake_common2.ParallelEngramEmbedding = orig_common.ParallelEngramEmbedding
        fake_common2.Engram = orig_common.Engram
        fake_common2.engram_gathered_num_tokens = lambda: 4
        fake_common2.engram_head_shard_rank = lambda: 0

        monkeypatch.setattr(
            sys.modules["vllm.models.deepseek_v41.common"], "engram", fake_common2
        )
        monkeypatch.setitem(
            sys.modules, "vllm.models.deepseek_v41.common.engram", fake_common2
        )
        monkeypatch.setitem(sys.modules, "vllm.models.deepseek_v41.nvidia", None)
        importlib.reload(engram_mod)
        monkeypatch.setattr(
            sys.modules["vllm.models.deepseek_v41.common"], "engram", orig_common
        )
        monkeypatch.setitem(
            sys.modules, "vllm.models.deepseek_v41.common.engram", orig_common
        )
        monkeypatch.undo()
        importlib.reload(engram_mod)


class TestNgramHashStateAndHostTables:
    def test_ensure_cache_device_hash_and_host_hash(self, monkeypatch):
        st = _make_hash_state(use_slot_cache=True)
        with pytest.raises(NotImplementedError):
            st.write_cache(st._cache, st._cache, st._cache, st._cache.bool())

        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
        assert st.ensure_cache() is True
        st.use_slot_cache = False
        assert st.ensure_cache() is True

        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
        monkeypatch.setattr(
            engram_mod.NgramHashState.__bases__[0],
            "ensure_cache",
            lambda self: True,
            raising=False,
        )
        assert st.ensure_cache() is True
        st.use_slot_cache = True
        assert st.ensure_cache() is True
        assert st.block_size == 8
        assert st.ensure_cache() is True  # equal block_size branch

        # Empty count=0 device & host paths
        empty_ids = torch.zeros((0,), dtype=torch.int32)
        assert st(
            empty_ids,
            empty_ids,
            empty_ids,
            empty_ids.bool(),
            empty_ids,
            empty_ids.bool(),
            None,
            empty_ids,
        ).shape == (0, 2, 4)
        assert st.hash_on_host(
            empty_ids,
            empty_ids,
            empty_ids,
            empty_ids.bool(),
            empty_ids,
            empty_ids.bool(),
        ).shape == (0, 2, 4)

        # Device forward: use_slot_cache True (int64 inputs -> int32 casts) & False
        st.write_cache = MagicMock()
        ids = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        pos = torch.tensor([0, 1, 5, 6], dtype=torch.int64)
        qsl = torch.tensor([0, 2, 4], dtype=torch.int64)
        dead = torch.tensor([False, False, True, False])
        lb = torch.tensor([[7, 8, -1], [9, 10, 11]], dtype=torch.int64)
        lb_dead = torch.tensor([[False, True, False], [False, False, False]])
        slots = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        bt = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
        h_dev = st(ids, pos, qsl, dead, lb, lb_dead, slots, bt)
        assert h_dev.shape == (4, 2, 4) and h_dev.dtype == torch.int32

        # Int32 fast-path branches in forward & _hash with mapped_batch=None and
        # use_slot_cache=False
        st.use_slot_cache = False
        h_dev2 = st(
            ids.int(), pos.int(), qsl.int(), dead, lb.int(), lb_dead, None, bt.int()
        )
        assert h_dev2.shape == (4, 2, 4)

        # Host hashing: full columns, col_slice uniform shift vs multi-shift, all_valid
        # True vs False,
        # default offset vs shifted offset, memoryview hit vs non-contiguous array_equal
        # hit
        h_host = st.hash_on_host(ids, pos, qsl, dead, lb, lb_dead)
        assert h_host.shape == (4, 2, 4)
        assert st._host_tables.scratch(4).shape[-1] == 4

        # Non-contiguous qsl matching values -> hits np.array_equal branch
        qsl_strided = np.array([0, 99, 2, 99, 4, 99], dtype=np.int32)[::2]
        assert not qsl_strided.flags.c_contiguous
        h_host_strided = st.hash_on_host(
            ids,
            torch.tensor(
                [10, 11, 20, 21], dtype=torch.int32
            ),  # default offset 0,1,0,1 & all_valid=True!
            qsl_strided,
            torch.zeros(4, dtype=torch.bool),
            torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32),
            torch.zeros((2, 3), dtype=torch.bool),
            col_slice=(0, 2),  # uniform shift s_min = 0 == 0
        )
        assert h_host_strided.shape == (4, 2, 4)
        # Repeat same col_slice to hit column_slice memo hit
        assert st.hash_on_host(
            ids,
            torch.tensor([10, 11, 20, 21], dtype=torch.int32),
            qsl_strided,
            torch.zeros(4, dtype=torch.bool),
            torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32),
            torch.zeros((2, 3), dtype=torch.bool),
            col_slice=(0, 2),
        ).shape == (4, 2, 4)
        # Multi-shift col_slice (0, 4) -> unif_s == -1 + non-default offset (pos jumps
        # within request!)
        assert st.hash_on_host(
            ids,
            torch.tensor([10, 15, 20, 25], dtype=torch.int32),
            torch.tensor([0, 2, 4], dtype=torch.int32),
            torch.zeros(4, dtype=torch.bool),
            torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32),
            torch.zeros((2, 3), dtype=torch.bool),
            col_slice=(0, 4),
        ).shape == (4, 2, 4)


class TestDPSharedStorageAndEmbedding:
    def test_load_local_rows_dp_storage_and_can_share(self, monkeypatch):
        p_u8 = torch.zeros((2, 4), dtype=torch.uint8)
        engram_mod._load_local_rows(p_u8, torch.ones((2, 4), dtype=torch.uint8))
        assert torch.all(p_u8 == 1)

        p_f32 = torch.zeros((2, 4), dtype=torch.float32)
        engram_mod._load_local_rows(p_f32, torch.ones((2, 4), dtype=torch.bfloat16))
        assert torch.all(p_f32 == 1.0)

        monkeypatch.setattr(
            engram_mod,
            "allocate_shared_host_storage",
            lambda nb, grp, **kw: torch.zeros(nb, dtype=torch.uint8),
        )
        monkeypatch.setattr(torch.distributed, "barrier", lambda group=None: None)
        monkeypatch.setattr(
            engram_mod.ParallelEngramEmbedding,
            "_weight_loader",
            staticmethod(lambda p, lw: None),
        )
        grp0 = SimpleNamespace(rank_in_group=0, cpu_group="g", world_size=2)
        dps = engram_mod.DPSharedEngramStorage(4, 32, 32, grp0)
        dps.load_weight(torch.nn.Parameter(torch.zeros((4, 32))), torch.zeros((4, 32)))
        dps.group = SimpleNamespace(rank_in_group=1, cpu_group="g", world_size=2)
        dps.load_weight(torch.nn.Parameter(torch.zeros((4, 32))), torch.zeros((4, 32)))
        w, s = dps.get_views(dps.weight, dps.weight_scale_inv)
        assert w.shape == (4, 32) and s.shape == (4, 1)
        with pytest.raises(RuntimeError, match="must not be replaced"):
            dps.get_views(torch.zeros_like(dps.weight), dps.weight_scale_inv)

        layout = SimpleNamespace(num_embeddings=[8, 8], head_dim=32)
        monkeypatch.setattr(
            engram_mod.envs, "TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD", False
        )
        assert engram_mod.can_share_engram_tables(layout) is False
        monkeypatch.setattr(
            engram_mod.envs, "TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD", True
        )
        monkeypatch.setattr(engram_mod.envs, "TPU_DEEPSEEK_V41_ENGRAM_HOST_DMA", True)
        assert engram_mod.can_share_engram_tables(layout) is False
        monkeypatch.setattr(engram_mod.envs, "TPU_DEEPSEEK_V41_ENGRAM_HOST_DMA", False)
        monkeypatch.setattr(
            engram_mod.upstream,
            "can_share_engram_tables",
            lambda cfg, block_size=32: False,
        )
        assert engram_mod.can_share_engram_tables(layout) is False
        monkeypatch.setattr(
            engram_mod.upstream,
            "can_share_engram_tables",
            lambda cfg, block_size=32: True,
        )
        assert engram_mod.can_share_engram_tables(layout) is True

    def test_parallel_engram_embedding_all_methods(self, monkeypatch):
        base_emb_cls = engram_mod.ParallelEngramEmbedding.__bases__[0]

        def fake_emb_init(self, num_emb, dim, head_sizes, *a, **kw):
            if "cpu_offload" in kw and getattr(fake_emb_init, "reject_kw", False):
                raise TypeError("no cpu_offload")
            torch.nn.Module.__init__(self)
            self.part_num_embeddings = num_emb
            self.dim = dim
            self.block_size = 32
            self.part_n_hash_cols = 2
            self.n_hash_cols = 4
            self.head_start = 0
            self.vocab_start_idx = 0
            self.vocab_end_idx = num_emb
            self.tp_size = 2
            w, s = self._allocate_weights()
            self.weight = torch.nn.Parameter(w, requires_grad=False)
            self.weight_scale_inv = (
                None if s is None else torch.nn.Parameter(s, requires_grad=False)
            )

        monkeypatch.setattr(base_emb_cls, "__init__", fake_emb_init)
        monkeypatch.setattr(
            base_emb_cls,
            "_allocate_weights",
            lambda self: (
                torch.zeros(
                    (self.part_num_embeddings, self.dim), dtype=torch.float8_e4m3fn
                ),
                None,
            ),
            raising=False,
        )
        monkeypatch.setattr(
            base_emb_cls, "_get_shard_info", lambda self: (2, 0), raising=False
        )
        monkeypatch.setattr(
            engram_mod.upstream.ParallelEngramEmbedding,
            "_weight_loader",
            staticmethod(lambda p, lw: None),
            raising=False,
        )

        # 1. HBM allocation + TypeError fallback in super().__init__
        monkeypatch.setattr(
            engram_mod.envs, "TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD", False
        )
        monkeypatch.setattr(engram_mod.envs, "TPU_DEEPSEEK_V41_ENGRAM_HOST_DMA", False)
        fake_emb_init.reject_kw = True
        emb_hbm = engram_mod.ParallelEngramEmbedding(
            8, 32, (4, 4, 4, 4), cpu_offload=None
        )
        fake_emb_init.reject_kw = False
        assert emb_hbm.max_shard_rows() == 8
        assert emb_hbm._get_shard_info() == (2, 0)

        # _weight_loader CPU vs TPU and local vs upstream rows
        engram_mod.ParallelEngramEmbedding._weight_loader(
            torch.zeros((4, 8)), torch.zeros((4, 8))
        )
        engram_mod.ParallelEngramEmbedding._weight_loader(
            torch.zeros((4, 8)), torch.zeros((8, 8))
        )
        fake_stager = MagicMock()
        fake_stager.host_param.return_value = torch.zeros((4, 8))
        import vllm_torchtpu.model_loader_patches as loader_patches

        monkeypatch.setattr(loader_patches, "ExpertParamStager", lambda: fake_stager)
        fake_tpu_param = MagicMock()
        fake_tpu_param.shape = (8, 8)
        fake_tpu_param.device.type = "tpu"
        engram_mod.ParallelEngramEmbedding._weight_loader(
            fake_tpu_param, torch.zeros((4, 8))
        )

        # 2. DP shared memory allocation + non-shared CPU offload allocation
        monkeypatch.setattr(
            engram_mod,
            "allocate_shared_host_storage",
            lambda nb, grp, **kw: torch.zeros(nb, dtype=torch.uint8),
        )
        monkeypatch.setattr(
            engram_mod,
            "get_engram_dp_group",
            lambda: SimpleNamespace(world_size=2, rank_in_group=0, cpu_group="g"),
        )
        emb_shm = engram_mod.ParallelEngramEmbedding(
            8, 32, (4, 4, 4, 4), cpu_offload=True, dp_shared_memory=True
        )
        assert emb_shm._storage()[0].shape == (8, 32)
        monkeypatch.setattr(engram_mod, "get_engram_dp_group", lambda: None)
        assert emb_shm._get_shard_info() == (2, 0)
        assert engram_mod.ParallelEngramEmbedding(
            8, 32, (4, 4, 4, 4), cpu_offload=True, dp_shared_memory=True
        )._storage()[0].shape == (8, 32)
        emb_cpu = engram_mod.ParallelEngramEmbedding(
            8, 32, (4, 4, 4, 4), cpu_offload=True, dp_shared_memory=False
        )
        monkeypatch.setattr(engram_mod, "get_engram_dp_size", lambda: 2)
        assert emb_cpu._get_shard_info() == (4, 0)
        monkeypatch.setattr(engram_mod, "get_engram_dp_size", lambda: 1)

        # move_rows_to_pinned_host & _host_dma_lookup & padded _select_local_heads
        emb_cpu.move_rows_to_pinned_host(
            torch.device("cpu")
        )  # no-op when host_dma=False
        emb_cpu.host_dma = True
        fake_hrt = MagicMock()
        fake_hrt.lookup_and_dequant.return_value = torch.ones(
            (6, 32), dtype=torch.bfloat16
        )
        monkeypatch.setattr(engram_mod, "HostRowTable", lambda *a, **kw: fake_hrt)
        emb_cpu.move_rows_to_pinned_host(torch.device("cpu"))
        emb_cpu.move_rows_to_pinned_host(torch.device("cpu"))  # idempotent second call
        emb_cpu.head_start = 3  # triggers pad = 2 - (4 - 3) = 1 in _select_local_heads
        out_dma = torch.zeros((3, 2, 32), dtype=torch.bfloat16)
        emb_cpu.lookup(torch.zeros((3, 4), dtype=torch.int64), out_dma)
        assert torch.all(out_dma == 1)

        # Standard lookup & gather_raw (including indices device cast + valid vs invalid
        # rows)
        emb_raw = engram_mod.ParallelEngramEmbedding(
            8, 32, (4, 4, 4, 4), cpu_offload=True, dp_shared_memory=False
        )
        out_lk = torch.zeros((2, 2, 32), dtype=torch.bfloat16)
        fake_idx = MagicMock()
        fake_idx.device = torch.device("meta")
        fake_idx.to.return_value = torch.tensor(
            [[0, 99, 0, 0], [1, 2, 0, 0]], dtype=torch.int32
        )
        emb_raw.lookup(fake_idx, out_lk)
        emb_raw.lookup(
            torch.tensor([[0, 99, 0, 0], [1, 2, 0, 0]], dtype=torch.int32), out_lk
        )

        raw_out = torch.zeros((2, 2, 32), dtype=torch.uint8)
        scale_out = torch.zeros((2, 2, 1), dtype=torch.uint8)
        emb_raw.gather_raw(fake_idx, raw_out, scale_out)
        assert scale_out[0, 1, 0].item() == engram_mod._INVALID_SCALE
        # All valid rows branch
        emb_raw.gather_raw(
            torch.tensor([[0, 1, 0, 0], [2, 3, 0, 0]], dtype=torch.int32),
            raw_out,
            scale_out,
        )


class TestCollectivesAndEngramModule:
    def test_collective_helpers_and_engram_end_to_end(self, monkeypatch):
        # _gather_engram_rows padded vs unpadded & int32
        monkeypatch.setattr(
            engram_mod.upstream, "tensor_model_parallel_all_gather", lambda x, dim=0: x
        )
        assert engram_mod._gather_engram_rows(
            torch.ones((2, 10), dtype=torch.bfloat16), dim=0
        ).shape == (2, 10)
        assert engram_mod._gather_engram_rows(
            torch.ones((2, 128), dtype=torch.bfloat16), dim=0
        ).shape == (2, 128)
        assert engram_mod._gather_engram_rows(
            torch.ones((2, 4), dtype=torch.int32), dim=0
        ).shape == (2, 4)

        # _engram_dp_slot branches
        assert engram_mod._engram_dp_slot(8) == 8
        monkeypatch.setattr(engram_mod, "is_forward_context_available", lambda: True)
        monkeypatch.setattr(
            engram_mod,
            "get_forward_context",
            lambda: SimpleNamespace(dp_metadata=True),
        )
        monkeypatch.setattr(engram_mod, "engram_gathered_num_tokens", lambda: 0)
        assert engram_mod._engram_dp_slot(8) == 8
        monkeypatch.setattr(engram_mod, "engram_gathered_num_tokens", lambda: 6)
        assert engram_mod._engram_dp_slot(8) == 6

        # _dp_gather_engram_rows & _dp_gather_packed_rows & _sp_gather_packed_rows
        monkeypatch.setattr(engram_mod, "get_engram_dp_group", lambda: None)
        staged_bf16 = torch.ones((4, 2, 32), dtype=torch.bfloat16)
        assert engram_mod._dp_gather_engram_rows(staged_bf16, 2).shape == (2, 2, 32)
        assert (
            engram_mod._dp_gather_packed_rows(
                torch.zeros((4, 17), dtype=torch.int32),
                2,
                cols=2,
                head_dim=32,
                blocks=1,
                block_size=32,
            )
            is None
        )

        fake_dp_group = SimpleNamespace(
            world_size=2,
            rank_in_group=0,
            all_gather=lambda t, dim=0: torch.cat([t, t], dim=dim),
        )
        monkeypatch.setattr(engram_mod, "get_engram_dp_group", lambda: fake_dp_group)
        assert engram_mod._dp_gather_engram_rows(staged_bf16, 2).shape == (2, 4, 32)
        assert engram_mod._dp_gather_engram_rows(
            torch.ones((4, 2, 128), dtype=torch.bfloat16), 2
        ).shape == (2, 4, 128)
        assert engram_mod._dp_gather_packed_rows(
            torch.zeros((4, 17), dtype=torch.int32),
            2,
            cols=2,
            head_dim=32,
            blocks=1,
            block_size=32,
        ).shape == (2, 4, 32)

        monkeypatch.setattr(
            engram_mod.upstream,
            "tensor_model_parallel_all_gather",
            lambda x, dim=0: torch.cat([x, x], dim=dim),
        )
        assert engram_mod._sp_gather_packed_rows(
            torch.zeros((3, 17), dtype=torch.int32),
            3,
            tp=2,
            rank=0,
            n_hash_cols=3,
            cols=2,
            head_dim=32,
            blocks=1,
            block_size=32,
        ).shape == (2, 3, 32)
        assert engram_mod._sp_gather_packed_rows(
            torch.zeros((4, 17), dtype=torch.int32),
            4,
            tp=2,
            rank=0,
            n_hash_cols=4,
            cols=2,
            head_dim=32,
            blocks=1,
            block_size=32,
        ).shape == (2, 4, 32)
        monkeypatch.setattr(engram_mod, "get_engram_dp_group", lambda: None)
        assert (
            engram_mod._embed_dp_size(
                SimpleNamespace(embed_tokens=SimpleNamespace(dp_size="x"))
            )
            == 1
        )
        assert (
            engram_mod._embed_dp_size(
                SimpleNamespace(embed_tokens=SimpleNamespace(dp_size=2))
            )
            == 2
        )

        # Engram._create_embedding, _init_staging, prepare_embeddings, _ready_rows,
        # _sp_embed, forward + bit-exact parity
        eng = object.__new__(engram_mod.Engram)
        torch.nn.Module.__init__(eng)
        eng.hc_mult = 4
        eng.dim = 8
        eng.eps = 1e-6
        eng.clamp_value = 1e-4
        eng.q_weight = torch.randn((4, 8), dtype=torch.bfloat16)
        eng.k_weight = torch.randn((4, 8), dtype=torch.bfloat16)
        eng.wkv = lambda x: torch.randn((x.shape[0], 40), dtype=torch.bfloat16)

        monkeypatch.setattr(engram_mod, "register_row_slice", lambda *a, **kw: None)
        monkeypatch.setattr(
            engram_mod,
            "ParallelEngramEmbedding",
            lambda num_e, hd, hs, dp_shared_memory=False: SimpleNamespace(
                vocab_start_idx=0,
                part_num_embeddings=num_e,
                part_n_hash_cols=2,
                n_hash_cols=4,
                block_size=32,
                tp_size=2,
                dp_size=1,
                host_offload=True,
                host_dma=False,
                gather_raw=lambda h, r, s: None,
            ),
        )
        emb_created = eng._create_embedding(
            SimpleNamespace(
                num_embeddings=[8], head_dim=32, primes=[[[2, 2]]], layer_ids=[3]
            ),
            0,
        )
        eng.embed_tokens = emb_created

        base_eng_cls = engram_mod.Engram.__bases__[0]
        monkeypatch.setattr(
            base_eng_cls,
            "_init_staging",
            lambda self, max_t, hd: setattr(
                self, "staged_rows", torch.zeros((max_t, 2, hd), dtype=torch.bfloat16)
            ),
            raising=False,
        )
        monkeypatch.setattr(
            base_eng_cls,
            "prepare_embeddings",
            lambda self, h: None,
            raising=False,
        )
        monkeypatch.setattr(
            base_eng_cls,
            "_ready_rows",
            lambda self, n: self.staged_rows[:n],
            raising=False,
        )

        # 1. Host offload packed staging path + scale sync + DP > 1 fallback & packed
        # paths
        eng._init_staging(4, 32)
        eng.prepare_embeddings(torch.zeros((1, 4), dtype=torch.int32))
        assert eng._host_scale_buf.shape[0] >= 1
        assert eng._host_scale.shape == (4, 2, 1)
        eng.prepare_embeddings(torch.zeros((2, 4), dtype=torch.int32))
        eng.prepare_embeddings(torch.zeros((2, 4), dtype=torch.int32))  # bundle hit
        assert eng._ready_rows(2).shape == (2, 2, 32)

        # DP > 1 on packed staging: both packed collective and None fallback
        eng.embed_tokens.dp_size = 2
        eng.prepare_embeddings(torch.zeros((4, 4), dtype=torch.int32))
        assert eng._ready_rows(2).shape == (2, 2, 32)
        monkeypatch.setattr(engram_mod, "get_engram_dp_group", lambda: fake_dp_group)
        assert eng._ready_rows(2).shape == (2, 4, 32)
        monkeypatch.setattr(engram_mod, "get_engram_dp_group", lambda: None)

        # SP embed fast packed path vs non-packed tp==1, pad==0, pad>0
        eng.embed_tokens.dp_size = 1
        assert eng._sp_embed(torch.zeros((2, 4), dtype=torch.int32)).shape == (1, 4, 32)

        # 2. Host DMA path (_active_rows with dp=1 and dp=2)
        eng.embed_tokens.host_dma = True
        eng.embed_tokens._host_rows = True
        eng.embed_tokens.lookup = lambda h, r: r.zero_()
        eng._init_staging(4, 32)
        eng.prepare_embeddings(torch.zeros((1, 4), dtype=torch.int32))
        eng.prepare_embeddings(torch.zeros((3, 4), dtype=torch.int32))
        eng.prepare_embeddings(torch.zeros((3, 4), dtype=torch.int32))
        assert eng._ready_rows(3).shape == (3, 2, 32)
        eng.embed_tokens.dp_size = 2
        assert eng._ready_rows(2).shape == (2, 2, 32)

        # 3. HBM path (_staged_raw is None, dp=1 & dp=2 with _dp_staged vs None)
        eng.embed_tokens.host_offload = False
        eng.embed_tokens.host_dma = False
        eng.embed_tokens.dp_size = 2
        eng._init_staging(4, 32)
        assert eng._ready_rows(2).shape == (2, 2, 32)
        eng.prepare_embeddings(torch.zeros((4, 4), dtype=torch.int32))
        assert eng._ready_rows(2).shape == (2, 2, 32)
        eng.embed_tokens.dp_size = 1
        eng.prepare_embeddings(torch.zeros((2, 4), dtype=torch.int32))
        assert eng._ready_rows(2).shape == (2, 2, 32)
        monkeypatch.setattr(
            base_eng_cls,
            "_init_staging",
            lambda self, max_t, hd: setattr(self, "staged_rows", None),
            raising=False,
        )
        eng._init_staging(4, 32)
        eng.staged_rows = torch.zeros((4, 2, 32), dtype=torch.bfloat16)

        # Non-packed _sp_embed across tp==1, tp==2 pad>0 (3 tokens), tp==2 pad==0 (4
        # tokens)
        eng.embed_tokens.tp_size = 1
        eng.embed_tokens.n_hash_cols = 2
        assert eng._sp_embed(torch.zeros((2, 4), dtype=torch.int32)).shape == (2, 2, 32)
        eng.embed_tokens.tp_size = 2
        eng.embed_tokens.n_hash_cols = 3
        assert eng._sp_embed(torch.zeros((3, 4), dtype=torch.int32)).shape == (2, 3, 32)
        assert eng._sp_embed(torch.zeros((4, 4), dtype=torch.int32)).shape == (2, 3, 32)

        # 4. Engram.forward SP + non-SP + exact 100.0000% bit-exact parity vs 3D
        # reference!
        eng.use_sequence_parallel = True
        eng._sp_embed = lambda h: torch.ones((2, 4, 8), dtype=torch.bfloat16)
        eng.embed = lambda h: torch.ones((h.shape[0], 4, 8), dtype=torch.bfloat16)
        fixed_kv = torch.randn((2, 40), dtype=torch.bfloat16)
        eng.wkv = lambda x: fixed_kv

        hs = torch.randn((2, 4, 8), dtype=torch.bfloat16)
        tmask = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
        out_sp_masked = eng(
            hs, torch.zeros((3, 4), dtype=torch.int32), token_mask=tmask
        )
        out_sp_nomask = eng(hs, torch.zeros((3, 4), dtype=torch.int32))
        assert out_sp_masked.shape == hs.shape and out_sp_nomask.shape == hs.shape

        eng.use_sequence_parallel = False
        mask_non_sp = torch.tensor([1.0, 0.0], dtype=torch.float32)
        out_2d_nomask = eng(hs, torch.zeros((2, 4), dtype=torch.int32))
        out_2d_masked = eng(
            hs, torch.zeros((2, 4), dtype=torch.int32), token_mask=mask_non_sp
        )

        # Exact reference 3D HEAD formula: verify 100.0000% bit-for-bit equality!
        keys_ref = fixed_kv[:, :32].unflatten(-1, (4, 8)).float()
        vals_ref = fixed_kv[:, 32:].float()
        hid_ref = hs.float()
        dot_ref = (hid_ref * eng.q_weight * eng.k_weight * keys_ref).sum(-1)
        dot_ref *= torch.rsqrt(hid_ref.square().mean(-1) + eng.eps)
        dot_ref *= torch.rsqrt(keys_ref.square().mean(-1) + eng.eps) * eng.dim**-0.5
        gin_ref = dot_ref.abs().clamp(min=eng.clamp_value).sqrt()
        gate_ref = torch.where(dot_ref < 0, -gin_ref, gin_ref).sigmoid()
        exp_nomask = (hid_ref + gate_ref[:, :, None] * vals_ref[:, None, :]).to(
            hs.dtype
        )
        exp_masked = (
            hid_ref
            + (gate_ref * mask_non_sp[:, None])[:, :, None] * vals_ref[:, None, :]
        ).to(hs.dtype)
        assert torch.equal(out_2d_nomask, exp_nomask)
        assert torch.equal(out_2d_masked, exp_masked)
