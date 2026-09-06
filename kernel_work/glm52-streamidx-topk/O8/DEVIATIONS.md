# O8 deviations from `design/code-style.md`

Declared here and repeated in-file at each site, as the reference requires.

| # | Checklist item | Deviation | Why |
| --- | --- | --- | --- |
| 1 | "jaxtyping on the public wrapper using the same letters" | `api.py` uses trailing `# [B, N] f32` comments instead of `Float[Array, "B N"]`. | `jaxtyping` is not installed in the benchmark image. Importing it makes every arm fail to *import* rather than fail to be *typed*, which would cost a job. |
| 2 | "Every ref parameter has a trailing `# [shape] dtype` comment" in the signature | `_topk_kernel` takes `*refs` and the shape comments are on the tuple-unpack lines immediately below. | `emit_keys` adds one output ref and one scratch ref. A fixed signature would need two kernel bodies, i.e. a second copy of 200 lines of radix select, and the point of the copy is that `inc_copy` be transcription-identical to `inc`. The glossary comments are present, one line lower. |
| 3 | "Config is a frozen, kw_only, slots **pydantic** dataclass" | stdlib `dataclasses.dataclass(frozen=True, kw_only=True, slots=True)` with `__post_init__` cross-field checks. | `pydantic` is not installed in the image. Every other property the item asks for is kept. |
| 4 | "Four files ... no `pallas_call` outside the last" | Four files are present. `tiling.py` is added (§2's ints-only module) and three drivers — `run_selfcheck.py`, `run_correctness.py`, `run_bench.py`. | §2 explicitly sanctions `tiling.py`. The drivers are measurement scaffolding for step 3, not part of the operation; they contain no `pallas_call`. |
| 5 | "Tested against the XLA `_fwd` in `base.py`" | `base._fwd` exists and is the XLA reference, but the gate that decides this variant is the **shared** `instruments/correctness.py`, which carries its own reference and its own thresholds. | Moving the gate into this directory would be moving the gate. `_fwd` is kept because the checklist asks for it and because it is the only device-free statement of the operation here. |
| 6 | "Hardware constants from `pltpu.get_tpu_info()`" | `LANES = 16`, `NUM_BUCKETS = 256`, `MAX_SLICE_WORDS = 64*1024`, `L0_SHIFT`, `REFINEMENT_LEVELS` are literals. | Inherited verbatim from the shipping kernel. Deriving them here would make `inc_copy` a different kernel from `inc` and destroy the only arm that prices the transcription. `sc.num_cores` / `sc.num_subcores` **are** read from `get_tpu_info()`, as in the incumbent. |

## Not a deviation, but worth stating

`api.flat_topk` is built on the **shipping** `_sc_topk_direct` and `_pick_partition`, not on this
variant's copy and not imported from `O3/`. It is the arm that would otherwise ship, so it has to
be the shipping kernel with the gather line changed and nothing else.
