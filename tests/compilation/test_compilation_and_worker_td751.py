# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TD7.51 compilation and worker modules."""

import contextlib
import sys
import types
from types import SimpleNamespace

import pytest
import sympy
import torch
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

from vllm_torchtpu.compilation import kernel_reload as kr
from vllm_torchtpu.compilation.shape_variants import (
    ShapeSpecializationError,
    _finite,
    _holds_at,
    _placeholder_values,
    _signature,
    aot_cache_tag,
    graph_signature,
    remember,
    retrace,
    runnable_for,
    trace_shape_env,
    unsupported_reason,
    warmup,
)
from vllm_torchtpu.worker import runai_repull as rr
from vllm_torchtpu.worker.runai_repull import (
    COMPLETE_SENTINEL,
    PULLING_SENTINEL,
    _is_complete,
    _report_tokenizer_gap,
    _touch,
    ensure_runai_aux_files,
)
from vllm_torchtpu.worker.tpu_rank_binding import (
    SLICE_LOCAL_RANK_ENV,
    SLICE_LOCAL_WORLD_SIZE_ENV,
    SLICE_RANK_ENV,
    SLICE_WORLD_SIZE_ENV,
    TpuWorkerBinding,
    _get_int_env,
    _get_spawned_pcp_local_rank,
    _slice_binding_from_env,
    get_tpu_worker_binding,
    slice_binding_env,
)

# ---------------------------------------------------------------------------
# 1. shape_variants.py Tests
# ---------------------------------------------------------------------------


def _make_fake_graph(nodes):
    return SimpleNamespace(graph=SimpleNamespace(nodes=nodes))


class _FakeSymTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, sym_dim):
        t = torch.Tensor._make_subclass(cls, torch.empty((2, 4), dtype=torch.bfloat16))
        t._sym_dim = sym_dim
        return t

    @property
    def shape(self):  # type: ignore[override]
        return (self._sym_dim, 4)


def test_placeholder_values_trace_shape_env_and_signatures():
    # Node without example_value/val, node with value, followed by non-placeholder
    n1 = SimpleNamespace(op="placeholder", meta={})
    n2 = SimpleNamespace(
        op="placeholder",
        meta={"example_value": SimpleNamespace(fake_mode=None)},
    )
    shape_env_obj = SimpleNamespace(var_to_range={}, guards=[])
    n3 = SimpleNamespace(
        op="placeholder",
        meta={
            "val": SimpleNamespace(fake_mode=SimpleNamespace(shape_env=shape_env_obj))
        },
    )
    n4 = SimpleNamespace(op="call_function", meta={"val": torch.zeros(2)})

    g_with_env = _make_fake_graph([n1, n2, n3, n4])
    vals = _placeholder_values(g_with_env)
    assert len(vals) == 2
    assert trace_shape_env(g_with_env) is shape_env_obj
    # Second call hits cached branch
    assert trace_shape_env(g_with_env) is shape_env_obj

    # Graph with only placeholders and no shape_env
    g_no_env = _make_fake_graph(
        [
            SimpleNamespace(
                op="placeholder",
                meta={
                    "val": SimpleNamespace(fake_mode=SimpleNamespace(shape_env=None))
                },
            )
        ]
    )
    assert trace_shape_env(g_no_env) is None
    assert trace_shape_env(g_no_env) is None

    # Signature with non-tensor, concrete tensor, and symbolic dim tensor
    sym_int = torch.SymInt(8)
    sym_tensor = _FakeSymTensor(sym_int)
    g_sig = _make_fake_graph(
        [
            SimpleNamespace(op="placeholder", meta={"val": 42}),
            SimpleNamespace(
                op="placeholder", meta={"val": torch.zeros((2, 3), dtype=torch.float32)}
            ),
            SimpleNamespace(op="placeholder", meta={"val": sym_tensor}),
        ]
    )
    sig1 = graph_signature(g_sig)
    # Second call hits _cached_signature fast path
    sig2 = _signature(g_sig)
    assert sig1 is sig2
    assert sig1 == (
        "int",
        ("torch.float32", (2, 3)),
        ("torch.bfloat16", ("?", 4)),
    )


def test_finite_holds_at_and_unsupported_reason():
    assert _finite(16) == 16
    assert _finite(sympy.oo) is None
    assert _finite(None) is None

    s0 = sympy.Symbol("s0")
    s1 = sympy.Symbol("s1")
    assert _holds_at(s0 <= 8, {s0: 4}) is True
    assert _holds_at(s0 <= 8, {s0: 16}) is False
    # Expression remaining symbolic after partial substitution -> simplify -> None
    assert _holds_at(s0 + s1 > 10, {s0: 4}) is None

    # Custom expr whose .subs returns a non-BooleanAtom simplified to BooleanAtom
    class _DeferBooleanExpr:
        def subs(self, _sub):
            return sympy.Eq(s0 - s0, 0, evaluate=False)

    assert _holds_at(_DeferBooleanExpr(), {s0: 4}) is True

    assert unsupported_reason(None, 4) is None
    assert unsupported_reason(SimpleNamespace(var_to_range={}, guards=[]), 4) is None

    # Lower bound failure (lower > 2 and size < lower)
    env_lower = SimpleNamespace(
        var_to_range={s0: SimpleNamespace(lower=4, upper=sympy.oo)},
        guards=[],
    )
    assert unsupported_reason(env_lower, 2) == "s0 >= 4"

    # Upper bound failure (size > upper)
    env_upper = SimpleNamespace(
        var_to_range={s0: SimpleNamespace(lower=2, upper=16)},
        guards=[],
    )
    assert unsupported_reason(env_upper, 32) == "s0 <= 16"

    # Guard failure and guard pass
    env_guards = SimpleNamespace(
        var_to_range={s0: SimpleNamespace(lower=2, upper=sympy.oo)},
        guards=[
            SimpleNamespace(expr=s0 + s1 > 0),  # undecidable -> None
            SimpleNamespace(expr=s0 <= 8),
        ],
    )
    assert unsupported_reason(env_guards, 4) is None
    assert unsupported_reason(env_guards, 16) == "s0 <= 8"


def test_bucket_executables_warmup_runnable_for_remember_and_retrace(monkeypatch):
    s0 = sympy.Symbol("s0")
    shape_env = SimpleNamespace(
        var_to_range={s0: SimpleNamespace(lower=2, upper=8)},
        guards=[],
    )
    g1 = _make_fake_graph(
        [
            SimpleNamespace(
                op="placeholder",
                meta={
                    "val": SimpleNamespace(
                        fake_mode=SimpleNamespace(shape_env=shape_env)
                    )
                },
            ),
            SimpleNamespace(
                op="placeholder",
                meta={"val": torch.zeros((2, 4), dtype=torch.float32)},
            ),
        ]
    )
    g_diff = _make_fake_graph(
        [
            SimpleNamespace(
                op="placeholder",
                meta={"val": torch.zeros((2, 8), dtype=torch.float32)},
            )
        ]
    )

    # Outside warmup session
    assert aot_cache_tag() is None
    remember("dir", "k0", g1, lambda: "ok")
    assert runnable_for("dir", "k0", g1, size=4) is None
    assert runnable_for("dir", "k0", g1, size=None) is None
    with pytest.raises(ShapeSpecializationError, match="cannot be served"):
        runnable_for("dir", "k0", g1, size=16)

    # Inside warmup session
    with warmup() as session:
        assert aot_cache_tag() == "trace0"
        assert session.needs_retrace(16) is False

        # Refused bucket records size and returns a refuse stub
        refuse_fn = runnable_for("dir", "k_refused", g1, size=16)
        assert callable(refuse_fn)
        assert session.needs_retrace(16) is True
        with pytest.raises(ShapeSpecializationError, match="never compiled"):
            refuse_fn()

        # Supported bucket returns None, then remember stores it
        assert runnable_for("dir", "k_ok", g1, size=4) is None
        fn = lambda: "ran"  # noqa: E731
        remember("dir", "k_ok", g1, fn)
        assert runnable_for("dir", "k_ok", g1, size=4) is fn

        # Signature mismatch on hand-over raises ShapeSpecializationError
        with pytest.raises(ShapeSpecializationError, match="inputs differ"):
            session.get("dir", "k_ok", g_diff)

        # Retrace inside warmup increments trace counter and clears refused
        cleaned = []

        class _FakeWrapper(TorchCompileWithNoGuardsWrapper):
            def __init__(self):
                pass

            def cleanup(self):
                cleaned.append(self)

        reset_calls = []
        import vllm.compilation.wrapper as vcw
        import vllm.config as vcfg

        monkeypatch.setattr(
            vcw, "reset_compile_wrapper", lambda m: reset_calls.append(m)
        )
        monkeypatch.setattr(
            vcfg, "set_current_vllm_config", lambda _c: contextlib.nullcontext()
        )

        vllm_cfg = SimpleNamespace(
            compilation_config=SimpleNamespace(
                cache_dir="/tmp/c1", local_cache_dir="/tmp/lc1"
            )
        )
        direct_wrapper = _FakeWrapper()
        retrace(direct_wrapper, vllm_cfg)
        assert aot_cache_tag() == "trace1"
        assert session.needs_retrace(16) is False
        assert cleaned == [direct_wrapper]
        assert reset_calls == [direct_wrapper]
        assert vllm_cfg.compilation_config.cache_dir == "/tmp/c1"

    # Retrace outside warmup with a container holding child wrappers
    child_wrapper = _FakeWrapper()
    container = SimpleNamespace(modules=lambda: [child_wrapper, object()])
    retrace(container, vllm_cfg)
    assert cleaned == [direct_wrapper, child_wrapper]


# ---------------------------------------------------------------------------
# 2. kernel_reload.py Tests
# ---------------------------------------------------------------------------


def test_kernel_reload_all_branches(monkeypatch):
    monkeypatch.setattr(kr.envs, "TPU_KERNEL_ITER_MODE", True)
    assert kr.is_enabled() is True

    monkeypatch.setattr(kr.envs, "TPU_KERNEL_RELOAD_MODULES", "")
    assert kr.default_reload_modules() == kr._DEFAULT_RELOAD_MODULES

    monkeypatch.setattr(kr.envs, "TPU_KERNEL_RELOAD_MODULES", " mod_a , , mod_b ")
    assert kr.default_reload_modules() == ("mod_a", "mod_b")

    # _KernelDispatch with a function and a callable lacking __globals__
    def sample_fn(x: int = 1, *, y: int = 2) -> int:
        """Sample doc."""
        return x + y

    class _MinimalCallable:
        def __call__(self, x):
            return x * 2

    saved_live = dict(kr._LIVE)
    saved_builders = dict(kr._BUILDERS)
    try:
        kr._LIVE.clear()
        kr._BUILDERS.clear()

        disp1 = kr.make_dispatcher("op1", sample_fn)
        disp2 = kr.make_dispatcher("op2", _MinimalCallable())
        assert disp1.__wrapped__ is sample_fn
        assert disp2.__wrapped__ is not None

        kr.set_live("op1", lambda x, y=2: x + y + 10)
        assert kr.get_live("op1") is not None
        assert kr.get_live("nonexistent") is None
        assert disp1(3, y=4) == 17

        # Register a dummy module in sys.modules to test reload_kernels
        dummy_mod = types.ModuleType("_dummy_reload_mod_td751")
        dummy_mod.version = 1
        sys.modules["_dummy_reload_mod_td751"] = dummy_mod
        reloaded = []
        monkeypatch.setattr(
            kr.importlib, "reload", lambda mod: reloaded.append(mod.__name__)
        )

        kr.register_builder("op1", lambda: (lambda x: x + 100))

        # Explicit modules list with one loaded and one missing module
        res1 = kr.reload_kernels(
            modules=["_dummy_reload_mod_td751", "_missing_mod_td751"]
        )
        assert res1["reloaded_modules"] == [
            "_dummy_reload_mod_td751",
            "_missing_mod_td751",
        ]
        assert res1["rebuilt_ops"] == ["op1"]
        assert reloaded == ["_dummy_reload_mod_td751"]
        assert disp1(5) == 105

        # Default modules list when modules=None
        monkeypatch.setattr(
            kr.envs, "TPU_KERNEL_RELOAD_MODULES", "_dummy_reload_mod_td751"
        )
        res2 = kr.reload_kernels(modules=None)
        assert res2["reloaded_modules"] == ["_dummy_reload_mod_td751"]
    finally:
        sys.modules.pop("_dummy_reload_mod_td751", None)
        kr._LIVE.clear()
        kr._LIVE.update(saved_live)
        kr._BUILDERS.clear()
        kr._BUILDERS.update(saved_builders)


# ---------------------------------------------------------------------------
# 3. tpu_rank_binding.py Tests
# ---------------------------------------------------------------------------


def _make_parallel_cfg(
    *,
    world_size: int = 4,
    local_world_size: int = 4,
    pcp_size: int = 1,
    dp_size: int = 1,
    dp_index: int = 0,
    dp_rank_local: int | None = None,
    dp_size_local: int | None = None,
    enable_ep: bool = False,
):
    return SimpleNamespace(
        world_size=world_size,
        local_world_size=local_world_size,
        prefill_context_parallel_size=pcp_size,
        data_parallel_size=dp_size,
        data_parallel_index=dp_index,
        data_parallel_rank_local=dp_rank_local,
        data_parallel_size_local=dp_size_local,
        enable_expert_parallel=enable_ep,
    )


def test_tpu_rank_binding_helpers_and_validation():
    assert _get_int_env({"A": "7"}, "A", 1) == 7
    assert _get_int_env({"A": ""}, "A", 3) == 3
    assert _get_int_env({"A": "not_an_int"}, "A", 5) == 5

    assert (
        _get_spawned_pcp_local_rank(
            env={"LOCAL_RANK": "3"}, local_rank_offset=1, local_world=4
        )
        == 2
    )
    with pytest.raises(ValueError, match="out of range for PCP local world"):
        _get_spawned_pcp_local_rank(
            env={"LOCAL_RANK": "6"}, local_rank_offset=1, local_world=4
        )

    s_env = slice_binding_env(rank=2, local_rank=1, world_size=8, local_world_size=4)
    assert s_env == {
        SLICE_RANK_ENV: "2",
        SLICE_LOCAL_RANK_ENV: "1",
        SLICE_WORLD_SIZE_ENV: "8",
        SLICE_LOCAL_WORLD_SIZE_ENV: "4",
    }

    pcfg = _make_parallel_cfg(world_size=4, dp_size=2, dp_index=1, enable_ep=True)
    # None present -> None
    assert (
        _slice_binding_from_env(
            {},
            parallel_config=pcfg,
            rank=0,
            dp_rank=1,
            dp_size=2,
            pcp_size=1,
            local_rank_offset=0,
        )
        is None
    )

    # Partial present -> ValueError
    with pytest.raises(ValueError, match="incomplete: missing"):
        _slice_binding_from_env(
            {SLICE_RANK_ENV: "0"},
            parallel_config=pcfg,
            rank=0,
            dp_rank=1,
            dp_size=2,
            pcp_size=1,
            local_rank_offset=0,
        )

    # pcp_size > 1 -> NotImplementedError
    with pytest.raises(NotImplementedError, match="Prefill context parallelism"):
        _slice_binding_from_env(
            s_env,
            parallel_config=pcfg,
            rank=0,
            dp_rank=1,
            dp_size=2,
            pcp_size=2,
            local_rank_offset=0,
        )

    # Wrong slice_world -> ValueError
    with pytest.raises(ValueError, match="does not match world_size"):
        _slice_binding_from_env(
            slice_binding_env(rank=0, local_rank=0, world_size=16, local_world_size=4),
            parallel_config=pcfg,
            rank=0,
            dp_rank=1,
            dp_size=2,
            pcp_size=1,
            local_rank_offset=0,
        )

    # Out-of-range slice_rank -> ValueError
    with pytest.raises(ValueError, match="SLICE_RANK.*out of range"):
        _slice_binding_from_env(
            slice_binding_env(rank=8, local_rank=0, world_size=8, local_world_size=4),
            parallel_config=pcfg,
            rank=0,
            dp_rank=1,
            dp_size=2,
            pcp_size=1,
            local_rank_offset=0,
        )

    # Out-of-range slice_local_world -> ValueError
    with pytest.raises(ValueError, match="SLICE_LOCAL_WORLD_SIZE.*out of range"):
        _slice_binding_from_env(
            slice_binding_env(rank=0, local_rank=0, world_size=8, local_world_size=0),
            parallel_config=pcfg,
            rank=0,
            dp_rank=1,
            dp_size=2,
            pcp_size=1,
            local_rank_offset=0,
        )

    # Out-of-range slice_local_rank -> ValueError
    with pytest.raises(ValueError, match="SLICE_LOCAL_RANK.*out of range"):
        _slice_binding_from_env(
            slice_binding_env(rank=0, local_rank=4, world_size=8, local_world_size=4),
            parallel_config=pcfg,
            rank=0,
            dp_rank=1,
            dp_size=2,
            pcp_size=1,
            local_rank_offset=0,
        )


def test_get_tpu_worker_binding_all_modes():
    # 1. Executor-assigned slice binding under DP+EP
    pcfg_ep = _make_parallel_cfg(
        world_size=4, dp_size=2, dp_index=1, dp_rank_local=1, enable_ep=True
    )
    s_env = slice_binding_env(rank=5, local_rank=1, world_size=8, local_world_size=4)
    b_exec = get_tpu_worker_binding(pcfg_ep, rank=1, local_rank=1, env=s_env)
    assert isinstance(b_exec, TpuWorkerBinding)
    assert b_exec.as_env() == {
        "RANK": "5",
        "LOCAL_RANK": "1",
        "WORLD_SIZE": "8",
        "LOCAL_WORLD_SIZE": "4",
    }

    # 2. DP+EP without slice env: dp_rank_local set, pcp=1
    b_ep1 = get_tpu_worker_binding(pcfg_ep, rank=1, local_rank=1, env={})
    assert b_ep1.rank == 5
    assert b_ep1.local_rank == 5

    # 3. DP+EP without slice env: dp_rank_local=None fallback + spawned PCP local rank
    pcfg_ep_pcp = _make_parallel_cfg(
        world_size=4,
        pcp_size=2,
        dp_size=2,
        dp_index=1,
        dp_rank_local=None,
        dp_size_local=2,
        enable_ep=True,
    )
    b_ep_pcp = get_tpu_worker_binding(
        pcfg_ep_pcp,
        rank=1,
        local_rank=1,
        env={"LOCAL_RANK": "3"},
        use_spawned_pcp_local_rank=True,
    )
    assert b_ep_pcp.local_rank == 3
    assert b_ep_pcp.native_local_rank == 5

    # 4. Non-EP mode: pcp=1 and pcp=2 with use_spawned_pcp_local_rank=True
    pcfg_dense = _make_parallel_cfg(world_size=4, local_world_size=4, pcp_size=2)
    b_dense_no_spawn = get_tpu_worker_binding(
        pcfg_dense, rank=1, local_rank=1, env={}, use_spawned_pcp_local_rank=False
    )
    assert b_dense_no_spawn.local_rank == 1

    b_dense_spawn = get_tpu_worker_binding(
        pcfg_dense,
        rank=1,
        local_rank=1,
        env={"LOCAL_RANK": "2"},
        use_spawned_pcp_local_rank=True,
    )
    assert b_dense_spawn.local_rank == 2


# ---------------------------------------------------------------------------
# 4. runai_repull.py Tests
# ---------------------------------------------------------------------------


def test_runai_repull_all_branches(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "get_lock", lambda _uri: contextlib.nullcontext())

    # 1. _touch and _is_complete branches
    dir_a = tmp_path / "dir_a"
    dir_a.mkdir()
    assert _is_complete(str(dir_a)) is False

    # API server pull: config.json present, no sentinels -> complete
    _touch(str(dir_a / "config.json"), "{}")
    assert _is_complete(str(dir_a)) is True

    # Worker pull in progress: config.json + PULLING_SENTINEL -> incomplete
    _touch(str(dir_a / PULLING_SENTINEL), "gs://b/m")
    assert _is_complete(str(dir_a)) is False

    # Worker pull finished: COMPLETE_SENTINEL present -> complete
    _touch(str(dir_a / COMPLETE_SENTINEL), "gs://b/m")
    assert _is_complete(str(dir_a)) is True

    # 2. ensure_runai_aux_files: non-object-storage config is a no-op
    noop_cfg = SimpleNamespace(
        model="/local/path", model_weights=None, tokenizer="/tok"
    )
    ensure_runai_aux_files(noop_cfg)

    # 3. Ray V2 (model_weights is gs://, rewritten=True) when already complete
    errors = []
    monkeypatch.setattr(
        rr.logger, "error", lambda msg, *args: errors.append(msg % args)
    )
    v2_complete_cfg = SimpleNamespace(
        model=str(dir_a),
        model_weights="gs://bucket/model",
        tokenizer=str(tmp_path / "missing_abs_tokenizer_dir"),
        maybe_pull_model_tokenizer_for_runai=lambda *_a: pytest.fail("should not pull"),
    )
    ensure_runai_aux_files(v2_complete_cfg)
    assert len(errors) == 1
    assert "missing_abs_tokenizer_dir" in errors[0]

    # 4. Ray V2 (rewritten=True) when local dir is missing and tokenizer == model
    dir_v2 = tmp_path / "dir_v2"
    pulled_v2 = []
    v2_pull_cfg = SimpleNamespace(
        model=str(dir_v2),
        model_weights="gs://bucket/v2_model",
        tokenizer=str(dir_v2),
        maybe_pull_model_tokenizer_for_runai=lambda u, t: pulled_v2.append((u, t)),
    )
    ensure_runai_aux_files(v2_pull_cfg)
    assert pulled_v2 == [("gs://bucket/v2_model", "gs://bucket/v2_model")]
    assert _is_complete(str(dir_v2)) is True

    # 5. Ray V1 (model_weights is None, model is gs://, rewritten=False)
    dir_v1 = tmp_path / "dir_v1"
    dir_v1.mkdir()

    def fake_v1_pull(uri, tok):
        v1_cfg.model = str(dir_v1)

    v1_cfg = SimpleNamespace(
        model="gs://bucket/v1_model",
        model_weights=None,
        tokenizer="hf-org/tokenizer",
        maybe_pull_model_tokenizer_for_runai=fake_v1_pull,
    )
    ensure_runai_aux_files(v1_cfg)
    assert _is_complete(str(dir_v1)) is True
    assert (dir_v1 / PULLING_SENTINEL).exists()
    assert (dir_v1 / COMPLETE_SENTINEL).exists()

    # _report_tokenizer_gap when tokenizer is None
    _report_tokenizer_gap(SimpleNamespace(model="m", tokenizer=None))
