"""Process-wide defaults required before torch_tpu autoloads."""

import os

# torch_tpu reads this during import. If unset, it marks c10d functional
# collectives as Dynamo graph breaks, which prevents vLLM TP graphs from
# compiling. This file is imported by Python before vLLM imports torch.
os.environ.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS",
                      "false")
os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")

# Bisect-only diagnostics: log every rewrite performed by torch_tpu's
# reassociate_norm_weights pass (pass stays ENABLED) to /perf_eval_results.
import sys as _sys
import importlib.abc as _abc
import importlib.machinery as _mach
import importlib.util as _util

_RNW = "torch_tpu._internal.compile.fx_passes.reassociate_norm_weights"


def _install_rnw_logging(mod):
    import torch

    orig_apply = mod.apply
    role = os.environ.get("ROLE", "na")
    path = f"/perf_eval_results/rnw_rewrites_{role}_{os.getpid()}.log"

    def _fmt(n):
        if not isinstance(n, torch.fx.Node):
            return repr(n)
        v = n.meta.get("val", n.meta.get("example_value"))
        vs = f"{tuple(v.shape)}:{v.dtype}" if isinstance(v, torch.Tensor) else type(v).__name__
        return f"{n.name}<{getattr(n.target, '__name__', n.target)}>[{vs}]"

    def apply(gm_or_graph):
        gm = gm_or_graph if hasattr(gm_or_graph, "graph") else None
        graph = gm.graph if gm is not None else gm_or_graph
        matches = []
        for n in graph.nodes:
            if n.op != "call_function" or getattr(n.target, "overloadpacket", n.target) != torch.ops.aten.mul:
                continue
            for i, a in enumerate(n.args[:2]):
                m = mod._match_add_one_weight(a) if isinstance(a, torch.fx.Node) else None
                if m is not None:
                    other = n.args[1 - i]
                    add_node, w, outer = m
                    matches.append(
                        f"mul={_fmt(n)} other={_fmt(other)} add={_fmt(add_node)} add.args={[_fmt(x) for x in add_node.args]} "
                        f"weight={_fmt(w)} outer_to_copy={_fmt(outer) if outer else None} "
                        f"mul.users={[_fmt(u) for u in n.users]} other.users={len(getattr(other, 'users', {})) if isinstance(other, torch.fx.Node) else 'na'}"
                    )
        n_before = len(list(graph.nodes))
        orig_apply(gm_or_graph)
        try:
            with open(path, "a") as f:
                f.write(f"=== apply gm={type(gm).__name__ if gm else 'graph'} nodes_before={n_before} nodes_after={len(list(graph.nodes))} matches={len(matches)}\n")
                for m in matches:
                    f.write("  " + m + "\n")
        except Exception:  # noqa: BLE001
            pass

    mod.apply = apply


class _RnwFinder(_abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _RNW:
            return None
        for finder in _sys.meta_path:
            if finder is self:
                continue
            spec = finder.find_spec(fullname, path, target) if hasattr(finder, "find_spec") else None
            if spec is not None:
                break
        else:
            return None
        loader = spec.loader
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            try:
                _install_rnw_logging(module)
            except Exception:  # noqa: BLE001
                pass

        loader.exec_module = exec_module
        return spec


_sys.meta_path.insert(0, _RnwFinder())
