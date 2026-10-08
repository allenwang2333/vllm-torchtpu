"""shard_map wrapper for the TP x DCP megakernel: every operand carries a leading rank axis [R, ...]
(per-rank arrays differ along it; replicated ones are repeated), the kernel runs once per rank.
Used by the tests and the probes; the vLLM adapter builds its own global arrays the same way."""

import functools

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from . import kernel as K


def make(cfg, mesh, axis="d", **build_kw):
    """Returns (fn, info). fn(*operands) with every operand [R, ...] -> (out [R, M, H], caches [R, ...],
    debug outputs [R, ...])."""
    call, info = K.build(cfg, axis_name=axis, **build_kw)

    def body(*ops):
        outs = call(*[o[0] for o in ops])
        return tuple(o[None] for o in outs)

    n_in = 8 + len(info["w_names"]) + len(info["c_names"])
    n_out = 1 + len(info["c_names"]) + (2 if build_kw.get("debug") else 0)
    fn = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P(axis),) * n_in,
                               out_specs=(P(axis),) * n_out, check_vma=False))
    return fn, info
