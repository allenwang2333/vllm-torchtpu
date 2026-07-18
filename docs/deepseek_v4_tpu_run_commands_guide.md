# DeepSeek-V4 TPU Run & Development Commands Guide

This document lists all the commands used to synchronize code, submit jobs, monitor the cluster queue, track logs, and manage job lifecycles in this workspace.

---

## 1. Code Synchronization

We edit files locally on our development host, but jobs execute on the GKE TPU cluster which mounts the NFS volume `/workspace` from the CDK VM. We must sync changes before launching or testing.

### Sync Local Workspace to CDK NFS Mount
Run this command from your local development host (`/usr/local/google/home/patemotter/torchtpu-vllm`):
```bash
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' --exclude '*.safetensors' --exclude '*.safetensors_*' --exclude '*gstmp' -e "ssh" /usr/local/google/home/patemotter/torchtpu-vllm/ patemotter-cdk:/mnt/workspace/torchtpu-vllm/
```

### Sync Local Workspace to Local debug TPU VM (Manual Eager Run)
```bash
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' --exclude '*.safetensors' --exclude '*.safetensors_*' --exclude '*gstmp' -e "ssh" /usr/local/google/home/patemotter/torchtpu-vllm/ patemotter-tpu-v6e-8-3:/home/patemotter_google_com/torchtpu-vllm/
```

---

## 2. CDK Job Lifecycle Management

All job submissions and log retrievals should be executed via the `cdk` CLI tool on `patemotter-cdk`.

### Copy & Register Recipe Configuration
If you modify `dsv4-verify-v7x-8-patemotter.yml` locally, you must sync it to NFS first and then register it into the CDK recipes folder:
```bash
ssh patemotter-cdk "cp /mnt/workspace/torchtpu-vllm/dsv4-verify-v7x-8-patemotter.yml /home/patemotter_google_com/cloud-devkit/recipes/experimental/torchtpu-vllm/dsv4-verify-v7x-8-patemotter.yml"
```

### Submit a New CDK Job
Submit a job using the registered recipe name:
```bash
ssh patemotter-cdk "cdk job create dsv4-verify-v7x-8-patemotter"
```

### List All CDK Jobs
```bash
ssh patemotter-cdk "cdk job list"
```

### Describe a CDK Job (Check State/Events)
```bash
ssh patemotter-cdk "cdk job desc <job_id>"
```

### Delete/Cancel an Active CDK Job
```bash
ssh patemotter-cdk "cdk job delete <job_id>"
```

---

## 3. Queue Monitoring & Logs

When jobs are submitted, they enter the GKE Kueue admission queue.

### Check Queue Length (Pending Workloads)
```bash
ssh patemotter-cdk "kubectl get clusterqueues tpu7x-cq"
```

### Check Admission Status of a Specific Workload
```bash
ssh patemotter-cdk "kubectl get workloads -n cloud-devkit-jobs | grep <job_id>"
```

### Check Pod Container Phase (Pending/Running/Failed)
```bash
ssh patemotter-cdk "kubectl get pods -n cloud-devkit-jobs | grep <job_id>"
```

### Tail Run Logs from Active Pod
```bash
ssh patemotter-cdk "cdk job log <job_id> -c vllm-server | tail -n 150"
```

---

## 4. Local TPU VM Running Configuration (Eager mode)

If the queue is heavily congested, you can bypass GKE Kueue entirely by executing directly on the local TPU VM (`patemotter-tpu-v6e-8-3`). Note that FP4 GMM compilation fails on TPU v6e and must use FP8 weights instead:

```bash
ssh patemotter-tpu-v6e-8-3 \
  "sudo docker run --privileged --net=host --shm-size=16g \
  -v /home/patemotter_google_com/torchtpu-vllm:/root/cloud-devkit/torchtpu-vllm \
  -v /mnt/workspace:/mnt/workspace \
  -w /root/cloud-devkit/torchtpu-vllm \
  -e PYTHONPATH=/root/cloud-devkit/torchtpu-vllm/src \
  -e DSV4_MODEL_PATH=/mnt/workspace/dsv4-flash-4layer-fp8 \
  -e TPU_LIBRARY_PATH=/mnt/workspace/libtpu.so \
  -e ENABLE_QUANTIZED_MATMUL_KERNEL=1 \
  -e REQUANTIZE_BLOCK_SIZE=128 \
  -e MOE_REQUANTIZE_BLOCK_SIZE=512 \
  -e MOE_REQUANTIZE_WEIGHT_DTYPE=fp8 \
  -e NEW_MODEL_DESIGN=1 \
  -e MODEL_IMPL_TYPE=vllm \
  us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-patemotter_google_com:latest \
  python3 -u examples/verify_dsv4.py"
```
