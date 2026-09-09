# GDN single-sequence benchmark evidence

This branch retains review artifacts for the HiERevo GDN metadata optimization.
It does not change the main branch or the implementation PR.

The captured 2,048-token pooled-prefill kernel improved from 155.297461 to
138.769539 microseconds, a 10.64% device-latency reduction. These measurements
belong to the frozen captured source revisions, not an unmeasured upstream port.

`gdn-single-sequence.tar.gz` contains all 20 independent confirmation traces,
execution records, decoded program events, compiled HLO, full LLO dumps,
representative executables, source provenance and audit receipts.
Separate compressed LLO and HLO files permit inspection without the full bundle.

After extraction, run `python verify_evidence.py` with NumPy installed.
The verifier checks hashes and independently recomputes timing samples and the
bootstrap confidence gate. No full-model improvement is claimed.
