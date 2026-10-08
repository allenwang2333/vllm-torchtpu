"""shard_map wrapper for the TP x DCP megakernel: every operand carries a leading rank axis [R, ...]
(per-rank arrays differ along it; replicated ones are repeated), and one step = the megakernel (caches
read-only) followed by the barrier-free cache writer (caches aliased + donated). Used by the tests and
the probes; the vLLM adapter builds its own global arrays the same way."""

import jax
from jax.sharding import PartitionSpec as P

from . import kernel as K


def make(cfg, mesh, axis="d", donate=True, **build_kw):
    """Returns (fn, info). fn(*operands), every operand [R, ...]: fixed (8), weights (info['w_names']),
    caches (info['c_names']) -> (out [R, M, H], *caches [R, ...], *debug [R, ...])."""
    call, info = K.build(cfg, axis_name=axis, **build_kw)
    wkw = {k: build_kw[k] for k in ("nb", "max_pages", "bmax", "interpret", "stack_c") if k in build_kw}
    writer, _ = K.build_writer(cfg, axis_name=axis, **wkw)
    n_w, n_c = len(info["w_names"]), len(info["c_names"])
    debug = build_kw.get("debug", False)

    def body(*ops):
        ops = [o[0] for o in ops]
        scal, rmeta, qmeta, bt = ops[:4]
        caches = ops[8 + n_w:]
        outs = call(*ops)
        out, nrow, rrow, irec = outs[:4]
        new = writer(scal, rmeta, bt, nrow, rrow, irec, *caches)
        return tuple(o[None] for o in (out, *new, *outs[4:]))

    n_in = 8 + n_w + n_c
    n_out = 1 + n_c + (4 if debug else 0)
    fn = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P(axis),) * n_in,
                               out_specs=(P(axis),) * n_out, check_vma=False),
                 donate_argnums=tuple(range(8 + n_w, n_in)) if donate else ())
    return fn, info
