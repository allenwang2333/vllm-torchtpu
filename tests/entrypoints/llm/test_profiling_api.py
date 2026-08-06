import glob
import logging
import os
import shutil
import subprocess
import tempfile
import time

import requests

# Configure basic logging to ensure we get clear output in CI
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_api_profiler_generation():
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Using temporary directory for profiles: {temp_dir}")

        # 1. Start server as a subprocess with a lightweight model
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server", "--model",
            "facebook/opt-125m", "--port", "8000",
            "--profiler-config.profiler", "torch",
            "--profiler-config.torch_profiler_dir", temp_dir,
            "--profiler-config.ignore_frontend", "true", "--max-model-len",
            "128", "--gpu-memory-utilization", "0.2"
        ]

        logger.info(f"Starting vLLM server: {' '.join(cmd)}")
        # We purposely do not capture stdout/stderr so it streams naturally into the CI logs
        proc = subprocess.Popen(cmd)

        server_url = "http://localhost:8000"

        try:
            # Wait for server to pass healthcheck (timeout 360s)
            # AOT TPU compilation of buckets takes significant time when cold-started
            logger.info("Waiting for server to become healthy...")
            server_ready = False
            for step in range(360):
                try:
                    res = requests.get(f"{server_url}/health", timeout=2)
                    if res.status_code == 200:
                        server_ready = True
                        break
                except requests.RequestException:
                    pass

                if step > 0 and step % 10 == 0:
                    logger.info(
                        f"Still waiting for server... ({step}s elapsed)")
                time.sleep(1)

            if not server_ready:
                raise RuntimeError(
                    "Server failed to start and pass health check in expected time (360s). Check CI logs above for server crash details."
                )

            # 2. Arm the profiler with advanced parameters
            # Testing the options parsing feature for TpuProfilerConfig
            # Note: We include complex nested strings like tpu_sctc_perf_counter_sampling_options
            profile_opts = (
                "host_tracer_level:2;"
                "device_tracer_level:1;"
                "tpu_trace_mode:TRACE_COMPUTE;"
                "tpu_perf_counters:true;"
                "tpu_enable_periodic_counter_sampling:true;"
                "e2e_enable_fw_throttle_event:true;"
                "e2e_enable_fw_power_level_event:true;"
                "e2e_enable_fw_thermal_event:true;"
                "tpu_tc_perf_counter_sampling_options:interval_us:10000 scaling:0 counter_size_bits:3 indices:0 indices:1 indices:3 indices:6 indices:8 indices:10 indices:12 indices:14 indices:18 indices:31 indices:40 indices:50 indices:56 indices:57 indices:73 indices:203;"
                # "tpu_sctc_perf_counter_sampling_options:interval_us:10000 scaling:0 counter_size_bits:2 indices:0 indices:1 indices:2;"
                # "tpu_cmn_perf_counter_sampling_options:interval_us:10000 scaling:0 counter_size_bits:3 indices:0 indices:2"
            )
            logger.info(
                f"Server is healthy. Arming the profiler via /start_profile JSON payload"
            )
            res = requests.post(f"{server_url}/start_profile",
                                json={"profile_prefix": profile_opts},
                                timeout=180)
            assert res.status_code == 200, f"/start_profile failed with status {res.status_code}: {res.text}"
            logger.info(
                "Profiler armed successfully with advanced prefix options.")

            # 3. Fire an inference request to capture trace data
            logger.info("Sending inference request to generate traffic...")
            gen_req = {
                "model": "facebook/opt-125m",
                "prompt": "The future of AI is",
                "max_tokens": 10
            }
            for _ in range(6):
                res = requests.post(f"{server_url}/v1/completions",
                                    json=gen_req,
                                    timeout=30)
                assert res.status_code == 200, f"Inference failed with status {res.status_code}: {res.text}"
            logger.info("Inference completed successfully.")

            # 4. Stop the profiler
            logger.info("Stopping the profiler via /stop_profile...")
            res = requests.post(f"{server_url}/stop_profile", timeout=180)
            assert res.status_code == 200, f"/stop_profile failed with status {res.status_code}: {res.text}"
            logger.info("Profiler stopped successfully.")

            # 5. Verify the profile was generated
            logger.info("Verifying generated traces...")
            pb_files = glob.glob(f"{temp_dir}/**/*.pb", recursive=True)
            assert len(
                pb_files
            ) > 0, f"No xplane.pb files found in the trace directory {temp_dir}!"
            logger.info(f"Found {len(pb_files)} trace artifact(s): {pb_files}")

            # 6. Copy the trace to an artifact directory for download from CI
            artifact_dir = "/tmp/vllm_profile_artifacts"
            os.makedirs(artifact_dir, exist_ok=True)
            for f in pb_files:
                basename = os.path.basename(f)
                dest = os.path.join(artifact_dir, basename)
                shutil.copy(f, dest)
                os.chmod(dest,
                         0o666)  # Ensure host Buildkite agent can read it
                logger.info(f"Copied trace artifact to: {dest}")

        finally:
            logger.info("Tearing down the server subprocess...")
            proc.terminate()
            try:
                # Add a timeout to wait, so a hung server doesn't freeze the CI suite forever
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Server did not terminate gracefully, sending SIGKILL...")
                proc.kill()
                proc.wait(timeout=5)
            logger.info("Server shutdown complete.")

            # CRITICAL: Clean up TPU multi-process lockfile.
            # If the server is SIGKILLed, libtpu won't release its lock, which will cause
            # subsequent CI jobs routed to this agent to fail with ABORTED: libtpu_lockfile errors.
            lockfile_path = "/tmp/libtpu_lockfile"
            if os.path.exists(lockfile_path):
                try:
                    os.remove(lockfile_path)
                    logger.info(f"Cleaned up orphaned {lockfile_path}")
                except OSError as e:
                    logger.warning(f"Could not remove {lockfile_path}: {e}")
