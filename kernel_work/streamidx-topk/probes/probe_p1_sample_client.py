#!/usr/bin/env python3
"""P1 -- the periodic-GFC-counter capture client.

Runs in its OWN venv (`uv venv /tmp/xpcap && uv pip install --python
/tmp/xpcap/bin/python xprof`), never over the image's jax.  Periodic counter
sampling cannot be turned on from inside the workload: `ProfileOptions()
.advanced_configuration` accepts the keys and discards them, and the
`--enable_periodic_counter_sampling` CLI is internal.  The only route that works
is an options map sent with a remote capture over the profiler gRPC service,
which is what this does.

This is the one part of `/xprof-llo` that produces **measured hardware** rather
than static compile-time analysis, and `count_sync_wait` (GFC index 105) is the
counter P1 is actually about.

    usage: probe_p1_sample_client.py <interval_us> <logdir> [host:port] [ms]

The GFC counters are 16 bit and wrap in 65535/1.9GHz ~= 34 us at 100% of a
unit, so a single capture cannot tell a real rate from a wrap residue.  Capture
at two intervals and diff: a counter whose percentage moves with the interval
wrapped.  interval_us=1 wedges the profiler -- do not use it.
"""

import os
import sys

GFC = [10, 11, 31, 32, 33, 34, 35, 37, 38, 56, 57, 58, 105]
GFC_NAMES = {
    10: "count_scalar_alu_instruction_0",
    11: "count_scalar_alu_instruction_1",
    31: "count_vector_alu_instruction_0",
    32: "count_vector_alu_instruction_1",
    33: "count_vector_alu_instruction_2",
    34: "count_vector_alu_instruction_3",
    35: "count_vst_instruction",
    37: "count_vld_instruction_0",
    38: "count_vld_instruction_1",
    56: "count_mxu_busy_0",
    57: "count_mxu_busy_1",
    58: "count_mxu_busy_2",
    105: "count_sync_wait",
}


def main():
    interval = int(sys.argv[1])
    logdir = sys.argv[2]
    target = sys.argv[3] if len(sys.argv) > 3 else "localhost:9012"
    duration_ms = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
    os.makedirs(logdir, exist_ok=True)

    from xprof.convert import _pywrap_profiler_plugin as pw

    opts = {
        "tpu_enable_periodic_counter_sampling":
        True,
        "tpu_tc_perf_counter_sampling_options":
        (f"interval_us: {interval} scaling: 0 counter_size_bits: 1 " +
         " ".join(f"indices: {i}" for i in GFC)),
    }
    print(f"[p1-client] target={target} logdir={logdir} "
          f"interval_us={interval} duration_ms={duration_ms}", flush=True)
    print(f"[p1-client] counters: "
          f"{ {i: GFC_NAMES[i] for i in GFC} }", flush=True)
    # num_tracing_attempts MUST be 1: all attempts pin their deadline to the
    # first session's expiry, so 2 and 3 fail in the same microsecond.
    pw.trace(target, logdir, "", True, duration_ms, 1, opts)
    n = sum(len(fs) for _, _, fs in os.walk(logdir))
    print(f"[p1-client] capture returned; {n} files under {logdir}",
          flush=True)


if __name__ == "__main__":
    main()
