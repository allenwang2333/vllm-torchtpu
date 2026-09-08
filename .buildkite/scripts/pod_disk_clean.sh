#!/bin/bash
# Report and free disk space on every host of the TPU pod this agent runs on.
# Same host discovery and ssh recipe as run_multihost.sh. Only removes things a
# CI job can re-create: stopped containers, unused images, build cache,
# dangling volumes, /tmp entries older than a day, old journal logs.
set -uo pipefail

SSH_USER="${SSH_USER:-$(whoami)}"
if [ ! -f ~/.ssh/id_rsa ]; then mkdir -p ~/.ssh; ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa -q; fi
SSH_OPTS=(-o StrictHostKeyChecking=no -o BatchMode=yes -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -i ~/.ssh/id_rsa)
ZONE="${ZONE:-$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/zone" | awk -F/ '{print $NF}')}"
TPU_NAME="${TPU_NAME:-$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/description" 2>/dev/null || echo "")}"
WORKER_IPS=""
if [ -n "$TPU_NAME" ] && [ -n "$ZONE" ]; then
  ALL_IPS=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone "$ZONE" --format="value(networkEndpoints[].ipAddress)" 2>/dev/null || true)
  ALL_IPS="${ALL_IPS//;/ }"; ALL_IPS="${ALL_IPS//,/ }"
  # shellcheck disable=SC2206
  ALL_IPS_ARRAY=($ALL_IPS)
  WORKER_IPS="${ALL_IPS_ARRAY[*]:1}"
fi
echo "agent host: $(hostname) TPU_NAME=${TPU_NAME:-?} ZONE=${ZONE:-?} workers: ${WORKER_IPS:-none}"

CLEAN='
set -uo pipefail
H=$(hostname)
echo "===== $H: BEFORE"; df -h / /tmp 2>/dev/null
echo "--- biggest directories (top level, this filesystem)"
sudo du -xsh /var/lib/docker /var/lib/docker/* /tmp /home/* /root /root/.cache /perf_eval_results* /var/log /opt 2>/dev/null | sort -h | tail -15
echo "--- docker system df"; docker system df 2>/dev/null
echo "--- containers"; docker ps -a --format "{{.ID}} {{.Image}} {{.Status}} {{.Names}}" 2>/dev/null | head -20
echo "--- clean: stopped containers, unused images, build cache, dangling volumes"
docker container prune -f 2>/dev/null | tail -1
docker image prune -af 2>/dev/null | tail -1
docker builder prune -af 2>/dev/null | tail -1
docker volume prune -f 2>/dev/null | tail -1
echo "--- clean: /tmp entries older than 1 day, journal over 500M"
sudo find /tmp -xdev -mindepth 1 -maxdepth 1 -mtime +1 ! -name "buildkite*" -exec rm -rf {} + 2>/dev/null
sudo journalctl --vacuum-size=500M 2>&1 | tail -1
echo "===== $H: AFTER"; df -h / /tmp 2>/dev/null
echo "--- still biggest"; sudo du -xsh /var/lib/docker /tmp /home/* /root/.cache /perf_eval_results* 2>/dev/null | sort -h | tail -8
'
echo "########## head host"
bash -c "$CLEAN"
for ip in $WORKER_IPS; do
  echo "########## worker $ip"
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${ip}" "bash -s" <<< "$CLEAN" || echo "ssh to $ip failed"
done
echo "########## summary: free space on / per host"
{ echo "$(hostname): $(df -h --output=avail / | tail -1)"; for ip in $WORKER_IPS; do echo "$ip: $(ssh "${SSH_OPTS[@]}" "${SSH_USER}@${ip}" 'df -h --output=avail / | tail -1' 2>/dev/null)"; done; }
