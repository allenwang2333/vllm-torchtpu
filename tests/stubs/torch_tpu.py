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
"""A CPU stand-in for ``torch_tpu``, so model code can be exercised off-device.

``vllm_torchtpu``'s model and layer modules import ``torch_tpu._internal.pallas``
at module scope. The real package needs a bazel-built extension
(``tpu_torch_pallas``) and ``libtpu``, so on a CPU box the whole model tree is
unimportable -- which would push every structural question about a new model to
the back of a TPU queue.

This installs a fake ``torch_tpu`` into ``sys.modules`` whose ``jax_op`` runs
the wrapped JAX function eagerly on the host: torch tensors in, ``jnp`` arrays
through the function, torch tensors out. JAX itself is CPU-capable, so the
arithmetic is real; only the device, the Mosaic lowering, and the torch-level
op registration are not.

What this is good for:
  * constructing a model and checking shapes, wiring, and weight loading
  * numerically diffing a non-Pallas JAX op against a reference
  * catching layer-dispatch and config-plumbing bugs in seconds

What it is NOT good for, and will not tell you about:
  * Pallas kernels -- they need Mosaic and a TPU device kind; a kernel that
    reaches ``pl.pallas_call`` will fail here rather than silently lie
  * anything about layout, tiling, donation, sharding or performance
  * ``torch.compile`` behaviour: these ops are plain Python callables, not
    registered library ops, so a full-graph compile test is meaningless here

``tests/conftest.py`` installs it when ``VLLM_TORCHTPU_CPU_STUB=1``, which is
how this repo's model and layer tests are run on a CPU box. To use it from a
standalone script::

    from tests.stubs import torch_tpu as stub
    stub.install()
    import vllm_torchtpu.models.vllm.glm5_next  # now imports
"""

from __future__ import annotations

import sys
import types
from typing import Any, Callable, Sequence

import numpy as np
import torch


def _to_jax(value: Any):
    import jax.numpy as jnp

    if isinstance(value, torch.Tensor):
        t = value.detach().cpu()
        if t.dtype is torch.bfloat16:
            # Round-trip through float32; jnp will cast back on use.
            return jnp.asarray(t.to(torch.float32).numpy(), jnp.bfloat16)
        return jnp.asarray(t.numpy())
    if isinstance(value, (list, tuple)):
        return type(value)(_to_jax(v) for v in value)
    return value


def _to_torch(value: Any):
    import jax.numpy as jnp

    if isinstance(value, (list, tuple)):
        return type(value)(_to_torch(v) for v in value)
    if value is None or isinstance(value, (int, float, bool, str)):
        return value
    if hasattr(value, "dtype") and hasattr(value, "shape"):
        if value.dtype == jnp.bfloat16:
            arr = np.asarray(value.astype(jnp.float32))
            return torch.from_numpy(arr).to(torch.bfloat16)
        return torch.from_numpy(np.asarray(value))
    return value


class _EagerJaxOp:
    """Callable stand-in for ``torch.library.CustomOpDef``.

    The real thing registers a torch library op so the graph can be traced and
    lowered. Here we only need it to *run*, so it is a plain callable with the
    same attribute surface the call sites use.
    """

    def __init__(self, name: str, fn: Callable[..., Any], **_meta: Any):
        self.name = name
        self._fn = fn
        self._fake = None

    def register_fake(self, fake_fn: Callable[..., Any]):
        self._fake = fake_fn
        return fake_fn

    def register_kernel(self, _device, fn=None):  # pragma: no cover

        def dec(f):
            return f

        return dec if fn is None else fn

    def __call__(self, *args, **kwargs):
        jax_args = [_to_jax(a) for a in args]
        jax_kwargs = {k: _to_jax(v) for k, v in kwargs.items()}
        return _to_torch(self._fn(*jax_args, **jax_kwargs))

    def __repr__(self) -> str:
        return f"<eager-cpu jax_op {self.name}>"


def _assert_every_parameter_is_annotated(name: str, fn) -> None:
    """Reproduce the real ``pallas.jax_op``'s signature check.

    The real one refuses any parameter without a type annotation --
    ``ValueError: Missing argument type annotation for JAX function: (...)``
    -- because it derives the torch schema from the annotations.

    A permissive stub hides that class of bug completely. An op built from
    ``functools.partial(stages, config=config)`` leaves every parameter bound
    by keyword -- and every parameter after it -- unannotated in the inspected
    signature. Direct calls to the underlying function still work, so unit
    tests and kernel-level probes all pass and the first symptom is an engine
    that refuses to start.

    So the stub is *stricter* here than "run the function", on purpose: the
    op boundary's contract is part of what the CPU rung is supposed to be able
    to test.
    """
    import inspect

    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):  # builtins, C callables
        return
    missing = [
        p.name for p in signature.parameters.values()
        if p.annotation is inspect.Parameter.empty and p.kind not in (
            inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    if missing:
        raise ValueError(
            f"Missing argument type annotation for JAX function: {signature}. "
            f"Unannotated: {missing}. (Stub check, mirroring the real "
            "torch_tpu pallas.jax_op. A functools.partial that binds an "
            "argument by KEYWORD leaves it -- and every parameter after it --"
            " in this signature; bind positionally or wrap in an explicit "
            "annotated function.)")

    # The RETURN annotation, which the real one checks separately and rejects
    # with a different message: "Invalid signature for JAX function: (...).
    # The return annotation is invalid. Only jax.Arrays and POD types are
    # supported."
    ret = signature.return_annotation
    if ret is inspect.Signature.empty:
        raise ValueError(
            f"Invalid signature for JAX function: {signature}. The return "
            "annotation is invalid. Only jax.Arrays and POD types are "
            "supported. (Stub check: annotate the return, e.g. "
            "`-> tuple[jax.Array, jax.Array]`.)")


def _jax_op(name: str,
            fn: Callable[..., Any] | None = None,
            /,
            *,
            donate_argnums: Sequence[int] | None = None,
            mesh: Any = None,
            input_partition_specs: Any = None):
    if "::" not in name:
        raise ValueError(f"Op name {name} does not contain a namespace.")

    def dec(f):
        _assert_every_parameter_is_annotated(name, f)
        return _EagerJaxOp(name,
                           f,
                           donate_argnums=donate_argnums,
                           mesh=mesh,
                           input_partition_specs=input_partition_specs)

    return dec if fn is None else dec(fn)


def _custom_jax_kernel(fn=None, name=None, **_kw):
    return fn if fn is not None else (lambda f: f)


def _set_buffer_donor_(tensor, _donate=True):
    return tensor


#: Attribute names the stub was asked for but does not model. Most are only
#: used as type annotations; inspect this after a run to see whether anything
#: load-bearing slipped through.
UNMODELLED: set[str] = set()


class _PlaceholderMeta(type):
    """Resolve attribute access on the class itself, not just instances.

    Call sites reach these symbols both ways -- ``pallas.JaxCallable`` as an
    annotation and ``pallas.Foo.bar`` as a lookup -- so the fallback has to
    survive being touched at class level.
    """

    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        UNMODELLED.add(f"<placeholder>.{name}")
        return _Placeholder


class _Placeholder(metaclass=_PlaceholderMeta):
    """Stands in for an unmodelled ``torch_tpu`` symbol.

    Usable as a type annotation, a base class, and a no-op callable, which
    covers every way the repo touches the symbols this stub does not model.
    """

    def __init__(self, *_a, **_kw):
        pass

    def __call__(self, *_a, **_kw):
        return self

    def __getattr__(self, _name):
        return _Placeholder()


def _module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)

    def __getattr__(attr: str):  # PEP 562
        if attr.startswith("__"):
            raise AttributeError(attr)
        UNMODELLED.add(f"{name}.{attr}")
        return _Placeholder

    mod.__getattr__ = __getattr__
    sys.modules[name] = mod
    return mod


def install() -> None:
    """Install the fake ``torch_tpu`` package. Idempotent."""
    if getattr(sys.modules.get("torch_tpu"), "_is_cpu_stub", False):
        return

    _custom_kernels: dict[str, Any] = {}

    pallas_attrs = dict(
        jax_op=_jax_op,
        custom_jax_kernel=_custom_jax_kernel,
        set_buffer_donor_=_set_buffer_donor_,
        register_custom_kernel=lambda k, v=None: _custom_kernels.__setitem__(
            k, v),
        lookup_custom_kernel=_custom_kernels.get,
        call_custom_kernel=lambda k, *a, **kw: _custom_kernels[k](*a, **kw),
        tpu=types.SimpleNamespace(),
        ops=types.SimpleNamespace(),
        mosaic=types.SimpleNamespace(),
    )

    torch_tpu = _module("torch_tpu", _is_cpu_stub=True)
    internal = _module("torch_tpu._internal")
    pallas_pkg = _module("torch_tpu._internal.pallas", **pallas_attrs)
    # `from torch_tpu._internal.pallas import pallas as pallas_impl` -- bind the
    # submodule on the parent explicitly, or the parent's __getattr__ shadows it
    # with a placeholder.
    pallas_mod = _module("torch_tpu._internal.pallas.pallas", **pallas_attrs)
    pallas_pkg.pallas = pallas_mod

    _module("torch_tpu._internal.utils")
    _module("torch_tpu._internal.utils.hardware",
            is_tpu_available=lambda: False,
            get_tpu_version=lambda: None)
    _module("torch_tpu._internal.sync", wait_for_all=lambda *a, **k: None)
    _module("torch_tpu._internal.compile")
    _module("torch_tpu._internal.compile.tpu_torch_compile")
    _module("torch_tpu._internal.compile.compiler")
    _module("torch_tpu._internal.compile._backend",
            TpuBackend=types.SimpleNamespace)
    _module("torch_tpu._internal.distributed")
    _module("torch_tpu._internal.distributed.tpu_distributed")
    _module("torch_tpu._internal.device_utils")
    _module("torch_tpu._internal.device_utils.annotations",
            LayoutContext=types.SimpleNamespace,
            layout_context=types.SimpleNamespace)
    _module("torch_tpu._internal.batch_transfer",
            batch_transfer_d2h=lambda *a, **k: None,
            batch_transfer_h2d=lambda *a, **k: None)

    internal.pallas = pallas_pkg
    torch_tpu._internal = internal

    # The real package registers a "tpu" Dynamo backend at import time and
    # vllm_torchtpu asks for it by name. Map it to eager: on CPU we only care
    # that compilation is a no-op, not that it lowers.
    try:
        import torch._dynamo as dynamo

        if "tpu" not in dynamo.list_backends(exclude_tags=()):
            dynamo.register_backend(compiler_fn=lambda gm, _ex: gm.forward,
                                    name="tpu")
    except Exception:  # pragma: no cover - best effort
        pass


def materialize(dest_dir: str) -> str:
    """Write a real, importable ``torch_tpu`` package into ``dest_dir``.

    ``install()`` only affects the calling process, and vLLM inspects model
    architectures by re-importing them in a **subprocess**
    (``model_executor/models/registry.py:_run_in_subprocess``). That child
    inherits ``PYTHONPATH`` but not ``sys.modules``, so an in-process stub is
    invisible to it and every registered TPU model "fails to be inspected".

    Materialising a one-line package onto the path fixes it for every process
    at once.

    Returns the path to the created package directory.
    """
    import pathlib

    impl = pathlib.Path(__file__).resolve()
    pkg = pathlib.Path(dest_dir).resolve() / "torch_tpu"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text(
        '"""Auto-generated CPU stand-in for torch_tpu. Do not edit."""\n'
        "import importlib.util as _ilu\n"
        "\n"
        f"_spec = _ilu.spec_from_file_location('_torch_tpu_cpu_stub_impl',\n"
        f"                                     {str(impl)!r})\n"
        "_impl = _ilu.module_from_spec(_spec)\n"
        "_spec.loader.exec_module(_impl)\n"
        "# Replaces sys.modules['torch_tpu'] with the fully populated stub;\n"
        "# `import torch_tpu` rebinds from sys.modules after exec, so callers\n"
        "# see the stub rather than this shim.\n"
        "_impl.install()\n")
    return str(pkg)


if __name__ == "__main__":  # pragma: no cover
    import sys as _sys

    if len(_sys.argv) > 1:
        print(f"materialized {materialize(_sys.argv[1])}")
    else:
        install()
        print("installed torch_tpu CPU stub in-process")
