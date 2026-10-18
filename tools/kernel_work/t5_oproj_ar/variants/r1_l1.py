"""R1-L1 candidate in the instruments' contract (test_accuracy.py / bench.py --cand)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fused_oproj_ar as _F  # noqa: E402


def fused_oproj_all_reduce(x, w_stack, idx, w_s, rank, n_valid, *, num_cores, axis):
    return _F.fused_oproj_all_reduce(x, w_stack, idx, w_s, rank, n_valid, num_cores=num_cores, axis=axis)
