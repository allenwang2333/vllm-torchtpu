"""TPU_AOT_SKIP_GUARD_CHECK: _skip_aot_guard_checks disables the guard check of
every AOT-compiled forward under the target model and the drafter, only when
vLLM compiled them without guards."""

from types import SimpleNamespace

import torch

from vllm.config import CompilationMode
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


class _Fn:
    def __init__(self):
        self.enabled = True

    def disable_guard_check(self):
        self.enabled = False


class _Compiled(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aot_compiled_fn = _Fn()
        self.lin = torch.nn.Linear(2, 2)


def _runner(evaluate_guards=False, mode=CompilationMode.VLLM_COMPILE, drafter=True):
    target = torch.nn.Sequential(torch.nn.Linear(2, 2), _Compiled())
    shared = target[1]
    d = None
    if drafter:
        d = SimpleNamespace(model=torch.nn.Sequential(_Compiled()), loop=_Compiled(), same=shared, n=3)
    cc = SimpleNamespace(mode=mode, dynamic_shapes_config=SimpleNamespace(evaluate_guards=evaluate_guards))
    return SimpleNamespace(model=target, drafter=d, vllm_config=SimpleNamespace(compilation_config=cc))


def _fns(r):
    out = [r.model[1].aot_compiled_fn]
    if r.drafter is not None:
        out += [r.drafter.model[0].aot_compiled_fn, r.drafter.loop.aot_compiled_fn]
    return out


def test_disables_every_compiled_forward_once():
    r = _runner()
    assert TPUModelRunner._skip_aot_guard_checks(r) == 3  # the shared module counts once
    assert not any(f.enabled for f in _fns(r))


def test_no_drafter():
    r = _runner(drafter=False)
    assert TPUModelRunner._skip_aot_guard_checks(r) == 1
    assert not r.model[1].aot_compiled_fn.enabled


def test_kept_when_guards_are_evaluated_or_stock_compile():
    for r in (_runner(evaluate_guards=True), _runner(mode=CompilationMode.STOCK_TORCH_COMPILE)):
        assert TPUModelRunner._skip_aot_guard_checks(r) == 0
        assert all(f.enabled for f in _fns(r))


def test_real_aot_compiled_function_skips_check():
    """The torch object honours disable_guard_check: __call__ stops calling guard_check."""
    from torch._dynamo.aot_compile import AOTCompiledFunction

    called = []
    fn = AOTCompiledFunction.__new__(AOTCompiledFunction)
    fn._guard_check_enabled = True
    fn._artifacts = SimpleNamespace(guard_manager=object())
    fn.fn = lambda *a, **k: "out"
    fn.guard_check = lambda *a, **k: called.append(1) or True
    assert fn("x") == "out" and called == [1]
    fn.disable_guard_check()
    assert fn("x") == "out" and called == [1]
