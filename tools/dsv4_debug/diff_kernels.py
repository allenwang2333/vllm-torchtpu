"""Run tpu-inference's mla_swa correctness test against OUR kernel.

Both trees ship the same Pallas kernel with the same signature, so the
reference's own test suite can be pointed at our copy. That reuses its
validated input construction and its numpy reference implementation, instead of
hand-building inputs -- which is fiddly (the SWA cache is a uint8 byte-packed
record with a physical page size of `page_size + 4`, distinct from the bf16
main cache) and gets shape assertions wrong.

Costs seconds. A full model run costs ~20 minutes of weight loading and
requantization.

Why this kernel: our copy carries defensive logic the reference has none of --
zeroing masked KV rows "to prevent unwritten page NaNs from propagating", and a
guard against `exp(-inf - -inf)` in the online-softmax rescale. Both use the
same DEFAULT_MASK_VALUE, so the masking math should agree. The question is
whether the extra handling changes results.

    PYTHONPATH=src:$TPU_INFERENCE_ROOT:$TPU_INFERENCE_ROOT/tests \\
        python tools/dsv4_debug/diff_mla_swa_kernel.py [--ours|--ref]

`--ref` runs the reference kernel (a control: it should pass, and proves the
harness itself is sound). `--ours` swaps in ours. Default runs both.
"""
import importlib
import sys
import unittest


# (reference test module, attribute it calls the kernel through, our module)
SUITES = {
    "mla_swa": ("mla_swa_test", "mla_swa",
                "vllm_torchtpu.kernels.deepseek_v4.mla_swa"),
    "mla": ("mla_test", "mla", "vllm_torchtpu.kernels.deepseek_v4.mla"),
    "compressor": ("compressor_test", "compressor",
                   "vllm_torchtpu.kernels.deepseek_v4.compressor"),
    "streamindex_topk": ("test_streamindex_topk", "streamindex_topk",
                         "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk"),
}


def _load_test_module(base):
    for name in (f"kernels.deepseek_v4.{base}",
                 f"tests.kernels.deepseek_v4.{base}", base):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    return None
    sys.exit(
        "Could not import tpu-inference's mla_swa_test. Add its tests dir to "
        "PYTHONPATH, e.g.\n"
        "  PYTHONPATH=src:$TPU_INFERENCE_ROOT:$TPU_INFERENCE_ROOT/tests")


def run(kernel, which):
    test_mod_name, attr, our_path = SUITES[kernel]
    mod = _load_test_module(test_mod_name)
    if mod is None:
        return None, None
    if which == "ours":
        our = importlib.import_module(our_path)
        setattr(mod, attr, our)  # tests call <attr>.<fn> through the module
    suite = unittest.defaultTestLoader.loadTestsFromModule(mod)
    res = unittest.TextTestRunner(verbosity=1).run(suite)
    return res.wasSuccessful(), res


def main():
    args = [a.lstrip("-") for a in sys.argv[1:]]
    kernels = [a for a in args if a in SUITES] or list(SUITES)
    rows = []
    for k in kernels:
        out = {}
        for t in ("ref", "ours"):
            print("=" * 70)
            print(f"{k}: running reference suite against {t.upper()}")
            print("=" * 70)
            ok, res = run(k, t)
            out[t] = None if ok is None else (ok, len(res.failures),
                                              len(res.errors))
        rows.append((k, out))

    print("\n" + "=" * 70)
    print(f"{'kernel':<20}{'ref':>10}{'ours':>10}   verdict")
    print("-" * 62)
    for k, out in rows:
        if out["ref"] is None:
            print(f"{k:<20}{'no suite':>10}{'':>10}")
            continue
        r_ok, o_ok = out["ref"][0], out["ours"][0]
        if not r_ok:
            v = "HARNESS BROKEN (ref fails its own suite)"
        elif r_ok and not o_ok:
            v = "*** DIVERGENCE IS IN THIS KERNEL ***"
        else:
            v = "equivalent -- not the bug"
        print(f"{k:<20}{str(r_ok):>10}{str(o_ok):>10}   {v}")


if __name__ == "__main__":
    main()
