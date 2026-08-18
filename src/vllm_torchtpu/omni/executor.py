import multiprocessing as mp
import os
import signal
import sys
import threading
import traceback

import torch
import torch.distributed
from vllm.logger import init_logger
from vllm_omni.diffusion.executor.multiproc_executor import \
    MultiprocDiffusionExecutor  # type: ignore[missing-import]
from vllm_omni.diffusion.worker.diffusion_worker import \
    WorkerProc  # type: ignore[missing-import]

logger = init_logger(__name__)


class MockWorkerThreadProcess:
    """Mock process interface wrapping a worker thread for the executor monitor."""

    def __init__(self,
                 thread: threading.Thread,
                 name: str = "DiffusionWorker-0"):
        self.thread = thread
        self.name = name
        r, w = os.pipe()
        self._sentinel_r = r
        self._sentinel_w = w
        self.exitcode = None

    def is_alive(self) -> bool:
        return self.thread.is_alive()

    @property
    def sentinel(self) -> int | None:
        return self._sentinel_r

    def join(self, timeout: float | None = None) -> None:
        self.thread.join(timeout)
        self._cleanup()

    def terminate(self) -> None:
        self._cleanup()

    def kill(self) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        if self._sentinel_r is not None:
            try:
                os.close(self._sentinel_r)
            except OSError:
                pass
            self._sentinel_r = None


class TpuDiffusionExecutor(MultiprocDiffusionExecutor):
    """TPU In-Process Diffusion Executor.

    Executes diffusion worker(s) inside background daemon threads rather than
    isolated multiprocessing subprocesses. This allows single-chip PJRT contexts to be
    shared seamlessly within the main host process without multiprocessing monkey-patches.
    """

    def _launch_workers(self, broadcast_handle, wake_events):
        od_config = self.od_config
        logger.info("Starting TPU in-process diffusion worker(s)...")

        num_gpus = od_config.num_gpus
        processes = []

        worker_extension_cls = od_config.worker_extension_cls
        custom_pipeline_args = getattr(od_config, "custom_pipeline_args", None)

        scheduler_pipe_readers = []
        scheduler_pipe_writers = []

        for i in range(num_gpus):
            reader, writer = mp.Pipe(duplex=False)
            scheduler_pipe_writers.append(writer)

            mock_proc = MockWorkerThreadProcess(None,
                                                name=f"DiffusionWorker-{i}")

            def _thread_worker_target(
                rank=i,
                cfg=od_config,
                w_pipe=writer,
                b_handle=broadcast_handle,
                w_event=wake_events[i],
                ext_cls=worker_extension_cls,
                pipe_args=custom_pipeline_args,
                proc=mock_proc,
            ):
                import portpicker

                if "TORCH_TPU_SLICEBUILDER_ADDRESSES" not in os.environ:
                    sb_port = portpicker.pick_unused_port()
                    os.environ[
                        "TORCH_TPU_SLICEBUILDER_ADDRESSES"] = f"localhost:{sb_port}"
                if "TORCH_TPU_TOPOLOGY" not in os.environ:
                    os.environ["TORCH_TPU_TOPOLOGY"] = "1,1,1"
                os.environ.setdefault("RANK", "0")
                os.environ.setdefault("LOCAL_RANK", "0")
                os.environ.setdefault("WORLD_SIZE", "1")

                # Mock signal handling (threads cannot register OS signals)
                orig_signal = signal.signal
                signal.signal = lambda *a, **kw: None

                # Mock distributed initialization (use gloo for in-process threads)
                orig_init = torch.distributed.init_process_group
                orig_new_group = torch.distributed.new_group

                def mock_init(*args_init, **kwargs_init):
                    new_args = list(args_init)
                    if len(new_args) > 0 and new_args[0] == "tpu_dist":
                        new_args[0] = "gloo"
                    if kwargs_init.get("backend") == "tpu_dist":
                        kwargs_init["backend"] = "gloo"
                    return orig_init(*new_args, **kwargs_init)

                def mock_new_group(*args_new, **kwargs_new):
                    new_args = list(args_new)
                    if len(new_args) > 2 and new_args[2] == "tpu_dist":
                        new_args[2] = "gloo"
                    if kwargs_new.get("backend") == "tpu_dist":
                        kwargs_new["backend"] = "gloo"
                    return orig_new_group(*new_args, **kwargs_new)

                torch.distributed.init_process_group = mock_init
                torch.distributed.new_group = mock_new_group

                try:
                    WorkerProc.worker_main(
                        rank,
                        cfg,
                        w_pipe,
                        b_handle,
                        w_event,
                        ext_cls,
                        pipe_args,
                    )
                except BaseException as exc:
                    sys.stderr.write(
                        f"\n[TPU-Worker-Thread-ERROR]: {type(exc).__name__}: {exc}\n"
                    )
                    traceback.print_exc(file=sys.stderr)
                    sys.stderr.flush()
                    raise
                finally:
                    signal.signal = orig_signal
                    torch.distributed.init_process_group = orig_init
                    torch.distributed.new_group = orig_new_group
                    if proc._sentinel_w is not None:
                        try:
                            os.write(proc._sentinel_w, b"x")
                            os.close(proc._sentinel_w)
                        except OSError:
                            pass
                        proc._sentinel_w = None

            t = threading.Thread(
                target=_thread_worker_target,
                name=f"DiffusionWorker-{i}",
                daemon=True,
            )
            mock_proc.thread = t
            t.start()

            processes.append(mock_proc)
            scheduler_pipe_readers.append(reader)

        # Wait for all workers to report ready
        scheduler_infos = []
        result_handle = None

        for i, reader in enumerate(scheduler_pipe_readers):
            try:
                data = reader.recv()
            except EOFError:
                logger.error("Rank %d worker thread died during startup.", i)
                processes[i].join()
                raise

            if data["status"] != "ready":
                raise RuntimeError(
                    "Initialization failed: Worker reported status != 'ready'")

            if i == 0:
                result_handle = data.get("result_handle")

            scheduler_infos.append(data)
            reader.close()
            scheduler_pipe_writers[i].close()

        logger.info("TPU in-process diffusion worker(s) ready.")
        return processes, [result_handle]
