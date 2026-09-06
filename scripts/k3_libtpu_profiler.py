"""Isolated diagnostic adapter for the documented PJRT profiler C API.

This does not patch library memory or change the JAX/PJRT execution client.
It bypasses the old JAX profiler's strict API-size equality check. Validate
device events independently before using a capture as performance evidence.

ABI references:
https://github.com/openxla/xla/blob/main/xla/backends/profiler/plugin/profiler_c_api.h
https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api_profiler_extension.h
"""
import contextlib
import ctypes as ct
import datetime
import importlib.metadata
import json
from pathlib import Path


class Extension(ct.Structure):
    pass


Extension._fields_ = [("struct_size", ct.c_size_t), ("type", ct.c_int),
                      ("next", ct.POINTER(Extension))]


class PjrtPrefix(ct.Structure):
    _fields_ = [("struct_size", ct.c_size_t),
                ("extension_start", ct.POINTER(Extension))]


class ProfilerExtension(ct.Structure):
    _fields_ = [("base", Extension), ("profiler_api", ct.c_void_p),
                ("traceme_context_id", ct.c_int64)]


class ApiPrefix(ct.Structure):
    _fields_ = [("struct_size", ct.c_size_t), ("priv", ct.c_void_p)] + [
        (name, ct.c_void_p)
        for name in ("error_destroy", "error_message", "error_get_code",
                     "create", "destroy", "start", "stop", "collect_data")
    ]


class ErrorMessage(ct.Structure):
    _fields_ = [("struct_size", ct.c_size_t), ("priv", ct.c_void_p),
                ("error", ct.c_void_p), ("message", ct.c_void_p),
                ("message_size", ct.c_size_t)]


class ErrorDestroy(ct.Structure):
    _fields_ = [("struct_size", ct.c_size_t), ("priv", ct.c_void_p),
                ("error", ct.c_void_p)]


class Create(ct.Structure):
    _fields_ = [("struct_size", ct.c_size_t), ("options", ct.c_char_p),
                ("options_size", ct.c_size_t), ("profiler", ct.c_void_p)]


class Handle(ct.Structure):
    _fields_ = [("struct_size", ct.c_size_t), ("profiler", ct.c_void_p)]


class Collect(ct.Structure):
    _fields_ = [("struct_size", ct.c_size_t), ("profiler", ct.c_void_p),
                ("buffer", ct.c_void_p), ("buffer_size_in_bytes", ct.c_size_t)]


def find_api(library):
    library.GetPjrtApi.argtypes = []
    library.GetPjrtApi.restype = ct.POINTER(PjrtPrefix)
    pjrt = library.GetPjrtApi()
    if not pjrt or pjrt.contents.struct_size < ct.sizeof(PjrtPrefix):
        raise RuntimeError("Invalid PJRT API prefix")
    extension = pjrt.contents.extension_start
    visited = set()
    while extension:
        address = ct.addressof(extension.contents)
        if address in visited or len(visited) >= 128:
            raise RuntimeError("Invalid PJRT extension chain")
        visited.add(address)
        if extension.contents.struct_size < ct.sizeof(Extension):
            raise RuntimeError("Truncated PJRT extension")
        # PJRT_Extension_Type_Profiler = 1 in the published C API enum.
        if extension.contents.type == 1:
            if extension.contents.struct_size < ct.sizeof(ProfilerExtension):
                raise RuntimeError("Truncated profiler extension")
            pointer = ct.cast(extension, ct.POINTER(ProfilerExtension))
            pointer = pointer.contents.profiler_api
            if not pointer:
                raise RuntimeError("Missing profiler API")
            api = ct.cast(pointer, ct.POINTER(ApiPrefix)).contents
            if api.struct_size < ct.sizeof(ApiPrefix):
                raise RuntimeError("Truncated profiler API")
            for name, _ in ApiPrefix._fields_[2:]:
                if not getattr(api, name):
                    raise RuntimeError(f"Missing profiler entrypoint: {name}")
            return api
        extension = extension.contents.next
    raise RuntimeError("PJRT has no profiler extension")


def call(api, name, args):
    args.struct_size = ct.sizeof(args)
    function = ct.CFUNCTYPE(ct.c_void_p,
                            ct.POINTER(type(args)))(getattr(api, name))
    error = function(ct.byref(args))
    if error:
        message = ErrorMessage(ct.sizeof(ErrorMessage), None, error, None, 0)
        destroy = ErrorDestroy(ct.sizeof(ErrorDestroy), None, error)
        try:
            ct.CFUNCTYPE(None, ct.POINTER(ErrorMessage))(api.error_message)(
                ct.byref(message))
            text = ct.string_at(message.message,
                                message.message_size).decode(errors="replace")
        finally:
            ct.CFUNCTYPE(None, ct.POINTER(ErrorDestroy))(api.error_destroy)(
                ct.byref(destroy))
        raise RuntimeError(f"Profiler {name}: {text}")


@contextlib.contextmanager
def trace(directory):
    distribution = importlib.metadata.distribution("libtpu")
    libraries = [
        distribution.locate_file(p) for p in distribution.files
        if str(p).endswith("libtpu.so")
    ]
    if len(libraries) != 1:
        raise RuntimeError(f"Expected one libtpu library: {libraries}")
    library = ct.CDLL(str(libraries[0]))
    api = find_api(library)
    print("LIBTPU_PROFILER_API",
          json.dumps(dict(size=api.struct_size, library=str(libraries[0]))),
          flush=True)
    # ProfileOptions protobuf: host=2, device=1, version=1, type=TPU, HLO=true.
    # Field definitions: tsl/profiler/protobuf/profiler_options.proto.
    options = bytes([16, 2, 24, 1, 40, 1, 48, 3, 56, 1])
    create = Create(0, options, len(options), None)
    call(api, "create", create)
    if not create.profiler:
        raise RuntimeError("Profiler create returned a null handle")
    handle = Handle(0, create.profiler)
    try:
        call(api, "start", handle)
        try:
            yield
        finally:
            call(api, "stop", handle)
        collect = Collect(0, create.profiler, None, 0)
        call(api, "collect_data", collect)
        size = collect.buffer_size_in_bytes
        if not 0 < size <= 2**31:
            raise RuntimeError(f"Unexpected profile buffer size: {size}")
        if not collect.buffer:
            buffer = ct.create_string_buffer(size)
            collect.buffer = ct.cast(buffer, ct.c_void_p)
            call(api, "collect_data", collect)
            if collect.buffer_size_in_bytes > size:
                raise RuntimeError("Profiler buffer grew during collection")
        payload = ct.string_at(collect.buffer, collect.buffer_size_in_bytes)
        timestamp = datetime.datetime.now(
            datetime.timezone.utc).strftime("%Y_%m_%d_%H_%M_%S")
        folder = Path(directory) / "plugins" / "profile" / timestamp
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "rank0-libtpu.xplane.pb").write_bytes(payload)
        print("LIBTPU_PROFILE_BYTES", len(payload), flush=True)
        if len(payload) < 1024:
            raise RuntimeError("Profile is too small; device evidence missing")
    finally:
        call(api, "destroy", handle)
