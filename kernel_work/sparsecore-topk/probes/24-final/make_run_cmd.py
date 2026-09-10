"""Emit probe 24's RUN_CMD.  usage: python3 make_run_cmd.py <final_md5> [suffix]

Why a generator rather than a hand-typed string: the cdk recipe renders RUN_CMD
inside single quotes in the YAML, so a single quote anywhere truncates it, and
the YAML folds the scalar to one line, so a `#` comments out everything after
it -- including any `exit 1`.  Both are asserted here rather than remembered.
"""
import hashlib
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

MD5 = {
    "rank.py": "e0bde4c103d277aa933fa585c1c49603",
    "p18_probe.py": "ceafc734d10632672d3db48f2cea23f6",
    "glm_dsa_block_bench.py": "08375d7068365e1a776c58fef09cd6d0",
    "subjects/base/sparsecore_topk.py": "44353b347c96e0eedbd742c710d4540a",
}
# final_rank.py is the one staged file still under edit, so its md5 is computed
# at generation time rather than pinned here.  It is still gated IN THE JOB --
# the value is baked into RUN_CMD at submit time and is visible in the recipe
# read-back, so the harness that runs is provably the harness that was staged.
MD5["final_rank.py"] = hashlib.md5(
    open(os.path.join(HERE, "stage", "final_rank.py"), "rb").read()).hexdigest()

if len(sys.argv) < 2:
    raise SystemExit("need the composed md5")
final_md5 = sys.argv[1].strip().lower()
suffix = sys.argv[2] if len(sys.argv) > 2 else "_final"
if len(final_md5) != 32 or any(c not in "0123456789abcdef" for c in final_md5):
    raise SystemExit(f"not a 32-hex md5: {final_md5}")

p = []
p.append("export P=/root/p24")
p.append("md5sum $P/final_rank.py $P/rank.py $P/p18_probe.py "
         "$P/glm_dsa_block_bench.py $P/subjects/*/*.py")
# The MAP gate: every staged file, asserted in-job, because a docker image tag
# is mutable and a --map-dir is not the thing the recipe read-back verifies.
for rel, m in MD5.items():
    p.append(f'test "$(md5sum $P/{rel} | cut -d" " -f1)" = "{m}" || exit 1')
p.append(f'test "$(md5sum $P/subjects/final/sparsecore_topk.py | '
         f'cut -d" " -f1)" = "{final_md5}" || exit 1')
p.append("echo MAP_GATE_OK")
p.append(f"export SCTOPK_FINAL_MD5={final_md5}")
p.append(f"export SCTOPK_FINAL_SUFFIX={suffix}")
p.append("python $P/final_rank.py --outdir $CDK_OUTPUT_DIR --dry-run || exit 1")
p.append("rc=1")
p.append("for i in 1 2 3; do python $P/final_rank.py --outdir $CDK_OUTPUT_DIR "
         "--rounds 12 --ctx 9216 --k 2048 --b-prefill 1024 --b-decode 64 2>&1 "
         "| tee $CDK_OUTPUT_DIR/final_$i.log; rc=${PIPESTATUS[0]}; "
         "if [ $rc -eq 0 ]; then break; fi; "
         "grep -q ABORTED $CDK_OUTPUT_DIR/final_$i.log || break; "
         "echo LOCKFILE_RETRY $i; sleep 90; done")
p.append("exit $rc")

cmd = "; ".join(p)
assert "'" not in cmd, "a single quote would truncate the YAML scalar"
assert "#" not in cmd, "a hash would comment out the rest of the folded line"
with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
    fh.write(cmd + "\n")
    tmp = fh.name
subprocess.run(["bash", "-n", tmp], check=True)
sys.stdout.write(cmd)
