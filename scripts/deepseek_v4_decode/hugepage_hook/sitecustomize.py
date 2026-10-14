# SPDX-License-Identifier: Apache-2.0
"""Experimental process-local CPU staging allocator; model sources stay frozen.

Install after the normal runner import. Only host block-table scratch allocation
changes. Tensor contents, shapes, metadata programs and model programs are intact.
"""

import importlib.abc
import importlib.machinery
import sys

TARGET = "vllm_torchtpu.runner.tpu_runner"


def install(module):
    import ctypes
    import functools
    import json
    import mmap
    import os
    import re
    from pathlib import Path

    import torch

    original = module.stage_block_table_uploads
    libc = ctypes.CDLL(None, use_errno=True)
    page = 2 * 1024 * 1024

    @functools.wraps(original)
    def stage(runner, ctx, num_reqs_padded, gids=None):
        seen, total = set(), 0
        for groups in runner.attn_groups:
            for group in groups:
                builder = group.metadata_builders[0]
                gid = builder.kv_cache_group_id
                if gid in seen or (gids is not None and gid not in gids):
                    continue
                seen.add(gid)
                _, blocks = builder._block_table_geometry(ctx)
                total += num_reqs_padded * blocks
        if total:
            backing = getattr(runner, "_huge_stage_backing", None)
            scratch = runner._block_table_stage_cpu
            if backing is None or scratch is None or scratch.numel() < total:
                capacity = ((total * 4 + page - 1) // page) * page
                mm = mmap.mmap(
                    -1, capacity + page, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
                )
                base = ctypes.addressof(ctypes.c_char.from_buffer(mm))
                offset = (-base) % page
                address = base + offset
                rc = libc.madvise(
                    ctypes.c_void_p(address),
                    ctypes.c_size_t(capacity),
                    mmap.MADV_HUGEPAGE,
                )
                if rc:
                    raise OSError(ctypes.get_errno(), "MADV_HUGEPAGE failed")
                full = torch.frombuffer(
                    mm, dtype=torch.int32, count=capacity // 4, offset=offset
                )
                full.zero_()
                runner._block_table_stage_cpu = full
                runner._huge_stage_backing = (mm, full)
                sections = re.split(
                    r"(?=^[0-9a-f]+-[0-9a-f]+ )",
                    Path("/proc/self/smaps").read_text(),
                    flags=re.M,
                )
                section = next(
                    s
                    for s in sections
                    if re.match(r"[0-9a-f]+-[0-9a-f]+ ", s)
                    and int(s.split("-", 1)[0], 16)
                    <= address
                    < int(s.split("-", 1)[1].split()[0], 16)
                )
                huge_kb = int(re.search(r"^AnonHugePages:\s*(\d+)", section, re.M)[1])
                assert huge_kb * 1024 >= capacity, section
                dest = Path(os.environ["DSV4_STAGE_AUDIT_DIR"])
                dest.mkdir(exist_ok=True)
                (dest / f"pid-{os.getpid()}-{total}.json").write_text(
                    json.dumps(
                        dict(
                            pid=os.getpid(),
                            rank=os.environ.get("RANK"),
                            total=total,
                            address=address,
                            allocation_bytes=capacity,
                            huge_kb=huge_kb,
                            dtype=str(full.dtype),
                            smaps=section,
                        ),
                        indent=2,
                    )
                )
                print(
                    f"[huge-staging] pid={os.getpid()} elements={total} "
                    f"capacity={capacity} huge_kb={huge_kb}",
                    flush=True,
                )
        result = original(runner, ctx, num_reqs_padded, gids)
        if total:
            assert (
                runner._block_table_stage_cpu.data_ptr()
                == runner._huge_stage_backing[1].data_ptr()
            )
        return result

    module.stage_block_table_uploads = stage


class Loader:
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        install(module)

    def __getattr__(self, name):
        return getattr(self.original, name)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None:
            spec.loader = Loader(spec.loader)
        return spec


sys.meta_path.insert(0, Finder())
