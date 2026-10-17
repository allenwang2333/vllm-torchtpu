#!/usr/bin/env bash
# t5-oproj-ar probe 02b (2x2x2, both pods run it)
set -u
export PYTHONUNBUFFERED=1 INST=/root/inst VAR=/root/var
cd /root/p02b
echo "[host $TPU_WORKER_ID] $(md5sum README.md probe.py acc_cand.py /root/var/*.py /root/inst/*.py | tr '\n' ' ') $(tail -1 /root/BUILD_INFO 2>/dev/null)"
F="grep --line-buffered -v -e ^[IWE][0-9]"
H="${COORD%:*}"
COORD="$H:8479" timeout 1500 python acc_cand.py 2>&1 | $F | grep --line-buffered -E "^\{|Traceback|Error|error" ; echo "== acc_cand rc=${PIPESTATUS[0]}"
cd /root/inst
COORD="$H:8480" timeout 1500 python test_accuracy.py --cand /root/var/r1_l1.py 2>&1 | $F | grep --line-buffered -E "SUMMARY|DONE|Traceback|Error|error|\"arm\": \"cand\"" ; echo "== acc_l1 rc=${PIPESTATUS[0]}"
COORD="$H:8481" timeout 900 python test_accuracy.py --cand /root/var/r1_l2.py --quick 2>&1 | $F | grep --line-buffered -E "SUMMARY|DONE|Traceback|Error|error|\"arm\": \"cand\"" ; echo "== acc_l2 rc=${PIPESTATUS[0]}"
cd /root/p02b
COORD="$H:8482" timeout 1800 python probe.py 2>&1 | $F | grep --line-buffered -E "^\{|Traceback|Error|error" ; echo "== timing rc=${PIPESTATUS[0]}"
exit 0
