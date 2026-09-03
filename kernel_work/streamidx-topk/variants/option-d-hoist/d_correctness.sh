#!/bin/bash
# Option D correctness shard.
#
#   $1 = which half (incumbent | optd), so the two can be fanned out.
#
# `incumbent`: `tests/kernels/test_dsv4_streamindex_topk.py`, UNMODIFIED.
#
#              Which copy matters. The image was built at an older commit and
#              carries a 16-test version of this file; the branch at `c3a18c95`
#              has 20, the extra four being the `early_exit` tests -- i.e.
#              exactly the path Option D deletes. Running the image's copy
#              would silently skip them. So the c3a18c95 copy is mapped in at
#              `$D_TESTS` and preferred when present, and its md5 is printed:
#              becf9f23ace9424a7d67cd3b34de9afd -> 20 collected.
#
#              Option D does not touch this file or the modules it imports, so
#              this is also the guard that the mapped-in kernels are the
#              incumbent's.
# `optd`:      `test_option_d_equivalence.py` -- the metadata-equality assertion
#              §3.3 specifies, plus D-vs-incumbent output equality.
#
# No `set -e`: a failed step is a result. The last command decides the shard rc.
WHICH=${1:?usage: d_correctness.sh <incumbent|optd>}
export D_VARIANT=${D_VARIANT:-/root/optd}
export D_TESTS=${D_TESTS:-/root/optd_tests}
cd /root/cloud-devkit/vllm-torchtpu || exit 90

bash "$D_VARIANT/d_readback.sh" python

echo "===== OPTD CORRECTNESS $WHICH BEGIN ====="
if [ "$WHICH" = "incumbent" ]; then
  T=tests/kernels/test_dsv4_streamindex_topk.py
  [ -f "$D_TESTS/test_dsv4_streamindex_topk.py" ] &&
    T=$D_TESTS/test_dsv4_streamindex_topk.py
  echo "INCUMBENT_TEST_FILE $T"
  md5sum "$T" tests/kernels/test_dsv4_streamindex_topk.py
  python -m pytest -q --timeout=1200 -p no:cacheprovider --collect-only "$T" |
    tail -2
  python -m pytest -q --timeout=1200 -p no:cacheprovider "$T"
else
  python -m pytest -q --timeout=1800 -p no:cacheprovider \
      "$D_VARIANT/test_option_d_equivalence.py"
fi
RC=$?
echo "===== OPTD CORRECTNESS $WHICH END rc=$RC ====="
exit $RC
