# TPU v6e-8 Capacity and Quota Report

This report outlines how to programmatically determine the quota limits, current usage, and physical capacity for creating a new **v6e-8** TPU VM in the project `cloud-tpu-inference-test`.

---

## 1. Current Quota vs. Usage Status (v6e)

Using our automated check, here is the current status of TPU v6e cores in the `cloud-tpu-inference-test` project across key zones:

| Zone | Quota Limit (Cores) | Current Usage (Cores) | Remaining Cores | v6e-8 Equivalent Available |
| :--- | :--- | :--- | :--- | :--- |
| **`us-east5-a`** | 32 | 18 | 14 | **1** |
| **`us-east5-b`** | 512 | 116 | 396 | **49** |
| **`us-east5-c`** | 0 | 0 | 0 | **0** |
| **`us-east7-ai1b`** | 512 | 8 | 504 | **63** |
| **`europe-west4-a`** | 384 | 52 | 332 | **41** |
| **`southamerica-west1-a`** | 512 | 256 | 256 | **32** |

> [!NOTE]
> Each `v6e-8` machine consumes **8 TPU cores**. Therefore, to create a new `v6e-8`, you need at least 8 remaining cores in your quota for that zone.

---

## 2. Programmatic Determination Methods

### Method A: Automated Python Script (Recommended)
We have created a helper script that queries your GCP project's TPU quotas, active nodes, and pending queued resources, then outputs the remaining capacity per zone.

- **Script Path:** `/Users/patemotter/.gemini/jetski/brain/29f8aac7-dd3f-4817-aaf7-e562f981c471/scratch/check_tpu_capacity.py`
- **Execution:**
  ```bash
  python3 /Users/patemotter/.gemini/jetski/brain/29f8aac7-dd3f-4817-aaf7-e562f981c471/scratch/check_tpu_capacity.py
  ```

### Method B: Raw CLI Commands

1. **Check Quota Limits per Zone:**
   ```bash
   gcloud alpha quotas info list \
     --project=cloud-tpu-inference-test \
     --service=tpu.googleapis.com \
     --filter="metric:tpu-v6e" \
     --format="json"
   ```

2. **List Active TPU Nodes (to count usage):**
   ```bash
   gcloud compute tpus tpu-vm list \
     --project=cloud-tpu-inference-test \
     --zone=- \
     --filter="acceleratorType:v6e-8" \
     --format="json"
   ```

3. **List Active Queued Resources (to count pending usage):**
   ```bash
   gcloud compute tpus queued-resources list \
     --project=cloud-tpu-inference-test \
     --zone=- \
     --format="json"
   ```

---

## 3. Quota Limit vs. Physical Capacity (Stockouts)

There is a critical distinction in GCP between **Quota** and **Physical Capacity**:
- **Quota Limit:** The maximum number of cores GCP *permits* your project to allocate.
- **Physical Capacity:** Whether GCP has the actual physical `v6e-8` hardware available in the datacenter at the moment you request it.

### How to Check Physical Capacity Programmatically
GCP does not expose a read-only "free physical capacity" API. The only way to programmatically determine if physical capacity exists is to attempt a creation and handle the response:

1. **Using Queued Resources (Recommended for v6e):**
   When creating a Queued Resource, you can check its state. If it remains in the `PROVISIONING` state for a long time or if the operations log shows `ZONE_RESOURCE_POOL_EXHAUSTED`, there is a physical stockout.
   
2. **Immediate Create Attempt:**
   Attempting to create a TPU VM directly (not via a queue) will fail immediately with a `429` or `403` error containing `ZONE_RESOURCE_POOL_EXHAUSTED` if there is no physical capacity.
