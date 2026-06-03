import builtins
import sys

_orig_import = builtins.__import__
_torch_import_depth = 0


def custom_import(name, globals=None, locals=None, fromlist=(), level=0):
    global _torch_import_depth

    is_torch_import = (name == 'torch' or name.startswith('torch.'))
    if is_torch_import:
        _torch_import_depth += 1

    try:
        mod = _orig_import(name, globals, locals, fromlist, level)
        return mod
    finally:
        if is_torch_import:
            _torch_import_depth -= 1
            if _torch_import_depth == 0:
                if not getattr(builtins, '_torch_cpu_sim_patched', False):
                    builtins._torch_cpu_sim_patched = True
                    _apply_torch_patches()


builtins.__import__ = custom_import


def _apply_torch_patches():
    import torch

    # 1. Register 'tpu' device name using the PrivateUse1 C++ loader
    try:
        import torch_tpu._loader as _loader
        _loader._init_device("tpu")
        print(
            "SUCCESSFULLY INITIALIZED native 'tpu' device name via PrivateUse1!",
            file=sys.stderr)
    except Exception as e:
        print(f"Failed to initialize native 'tpu' device: {e}",
              file=sys.stderr)

    # 2. Mock torch.tpu Stream and Event
    class MockStream:

        def __init__(self, device=None, priority=0, **kwargs):
            self.stream_id = 0
            self.device = device if device is not None else torch.device("tpu")
            self.priority = priority

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def synchronize(self):
            pass

        def wait_stream(self, stream):
            pass

        def wait_event(self, event):
            pass

    class MockEvent:

        def __init__(self,
                     enable_timing=False,
                     blocking=False,
                     interprocess=False,
                     external=False):
            pass

        def record(self, stream=None):
            pass

        def wait(self, stream=None):
            pass

        def query(self):
            return True

        def elapsed_time(self, end_event):
            return 0.0

    # Attach to torch.tpu
    if not hasattr(torch, 'tpu'):

        class TpuModule:
            pass

        torch.tpu = TpuModule()

    torch.tpu.Stream = MockStream
    torch.tpu.Event = MockEvent
    torch.tpu.current_stream = lambda device=None: MockStream(device=device)
    torch.tpu.default_stream = lambda device=None: MockStream(device=device)
    torch.tpu.stream = lambda stream: MockStream()
    torch.tpu.synchronize = lambda device=None: None
    torch.tpu.set_stream = lambda stream: None

    # 3. Wrap tensor allocation functions (keeping torch.device untouched)
    _orig_tensor = torch.tensor
    _orig_empty = torch.empty
    _orig_zeros = torch.zeros
    _orig_ones = torch.ones
    _orig_arange = torch.arange
    _orig_as_tensor = torch.as_tensor
    _orig_to = torch.Tensor.to

    def _redirect_device(device):
        if device is None:
            return None
        if isinstance(device, str) and device == 'tpu':
            return 'cpu'
        if isinstance(device, torch.device) and device.type == 'tpu':
            return torch.device('cpu')
        return device

    torch.tensor = lambda *args, **kwargs: _orig_tensor(
        *args, **{
            **kwargs, 'device': _redirect_device(kwargs.get('device'))
        }) if 'device' in kwargs else _orig_tensor(*args, **kwargs)
    torch.empty = lambda *args, **kwargs: _orig_empty(
        *args, **{
            **kwargs, 'device': _redirect_device(kwargs.get('device'))
        }) if 'device' in kwargs else _orig_empty(*args, **kwargs)
    torch.zeros = lambda *args, **kwargs: _orig_zeros(
        *args, **{
            **kwargs, 'device': _redirect_device(kwargs.get('device'))
        }) if 'device' in kwargs else _orig_zeros(*args, **kwargs)
    torch.ones = lambda *args, **kwargs: _orig_ones(
        *args, **{
            **kwargs, 'device': _redirect_device(kwargs.get('device'))
        }) if 'device' in kwargs else _orig_ones(*args, **kwargs)
    torch.arange = lambda *args, **kwargs: _orig_arange(
        *args, **{
            **kwargs, 'device': _redirect_device(kwargs.get('device'))
        }) if 'device' in kwargs else _orig_arange(*args, **kwargs)
    torch.as_tensor = lambda *args, **kwargs: _orig_as_tensor(
        *args, **{
            **kwargs, 'device': _redirect_device(kwargs.get('device'))
        }) if 'device' in kwargs else _orig_as_tensor(*args, **kwargs)

    def _mock_to(self, *args, **kwargs):
        new_args = list(args)
        if args:
            new_args[0] = _redirect_device(args[0])
        if 'device' in kwargs:
            kwargs['device'] = _redirect_device(kwargs['device'])
        return _orig_to(self, *new_args, **kwargs)

    torch.Tensor.to = _mock_to

    # 4. Mock torch_tpu hardware discovery
    try:
        import torch_tpu._internal.utils.hardware as hardware
        hardware.get_tpu_device_count = lambda: 1
        hardware.get_tpu_topology = lambda world_size=None: "1,1,1"
    except Exception:
        pass

    print(
        "SUCCESSFULLY ENABLED HYBRID NATIVE/MOCK CPU SIMULATION FOR PYTORCH TPU!",
        file=sys.stderr)
