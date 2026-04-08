# Setup Data Volumes

> **Step 0** in the [deployment guide](README.md).

Before deploying any workloads, set up the storage.

## Cluster storage layout

| Path | Type | Access | Size | Purpose |
|------|------|--------|------|---------|
| `/models/` | Shared models PVC | RO (reuse from WattBot if available) | varies | Qwen2.5-VL-7B weights |
| `/data/documents/` | Input documents PVC | RO for batch jobs | depends on corpus | Source PDFs and TIFFs |
| `/data/extracted/` | Output PVC | RW for batch jobs | ~1-5% of input | Extracted JSON files |

---

## Input documents — `ocr-documents`

This is where your PDFs and TIFFs live. The batch script reads from a
mounted directory — how that directory gets populated depends on your
infrastructure.

### PoC (5 sample docs)

Skip the PVC entirely. Upload files directly to the setup workspace
(Step 2) via Jupyter's file upload button. Use the workspace's local
storage at `/home/jovyan/sample_docs/`.

### Production

The ideal setup is a **direct mount** — the cluster admin creates a
PersistentVolume backed by the source storage (NFS, CIFS/SMB). No copy
needed. The batch workspace reads files over the network from their
original location.

Ask your cluster admin:

> "Can you create a PV pointing to the imaging data share
> (e.g. `nfs-server:/imaging_archive`)? We need read-only access from
> our RunAI project."

Once the PV exists, create a Data Volume in the RunAI UI:

1. Go to **Data & Storage** > **Data Volumes** > **New Data Volume**
2. **Scope:** Your project
3. **PVC name:** Use the existing PVC backed by the NFS mount
4. **Data volume name:** `ocr-documents`
5. **Mount path:** `/data/documents`

If a direct mount isn't possible (data lives on a completely disconnected
system), you'll need to copy the data onto a cluster PVC. Options include
pulling from inside a workspace (`curl`, `wget`, `rclone` for S3) or
having the data team push to a staging location the cluster can access.

### Ongoing ingestion (~10K docs/month)

With a direct NFS mount, new documents appear automatically as the source
system writes them. The batch script's `--resume` flag means you can
re-run against the same input directory — it skips already-processed files
and only extracts the new ones.

For non-mounted setups, set up a periodic sync (cron job, scheduled
rsync, or rclone) to keep the cluster PVC in sync with the source.

---

## Output storage — `ocr-extracted`

1. Go to **Data & Storage** > **Data Volumes** > **New Data Volume**
2. Configure:
   - **Scope:** Your project
   - **PVC name:** `ocr-extracted` (creates a new PVC)
   - **Data volume name:** `ocr-extracted`
   - **Size:** 100Gi to start (JSON is much smaller than the source docs)
3. Mount path: `/data/extracted`

---

## Shared models PVC

The vLLM server needs `Qwen/Qwen2.5-VL-7B-Instruct` (~15 GB) on a
shared PVC at `/models/.cache/huggingface`.

**If you already have the WattBot shared models PVC,** reuse it — just
make sure the VL model is downloaded. From any workspace with write
access to the PVC:

```python
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct",
                  cache_dir="/models/.cache/huggingface")
```

**If starting fresh,** see
[rag_app/docs/runai/setup-shared-models.md](../../../rag_app/docs/runai/setup-shared-models.md)
for full PVC setup instructions — the process is the same, just download
the Qwen VL model instead of (or in addition to) the text-only models.
