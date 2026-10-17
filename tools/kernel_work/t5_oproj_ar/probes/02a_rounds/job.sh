#!/usr/bin/env bash
# t5-oproj-ar probe 02a (2x2x2, both pods run it)
set -u
export PYTHONUNBUFFERED=1 INST=/root/inst
cd /root/p02a
echo "[host $TPU_WORKER_ID] $(md5sum README.md probe.py /root/inst/common.py | tr '\n' ' ') $(tail -1 /root/BUILD_INFO 2>/dev/null)"
F="grep --line-buffered -v -e ^[IWE][0-9]"
timeout 2400 python probe.py 2>&1 | $F | grep --line-buffered -E "^\{|Traceback|Error|error" ; echo "== rc=${PIPESTATUS[0]}"
exit 0
