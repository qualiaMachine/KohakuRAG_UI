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

This is where your PDFs and TIFFs live.

### Option A: Data already on cluster storage (NFS, Ceph, etc.)

If the imaging data is already on a shared filesystem accessible from the
cluster, create a Data Volume pointing directly to it:

1. Go to **Data & Storage** > **Data Volumes** > **New Data Volume**
2. **Scope:** Your project
3. **PVC name:** Use an existing PVC if the data is already on one, or
   create a new one backed by the existing storage class
4. **Data volume name:** `ocr-documents`
5. **Mount path:** `/data/documents`

### Option B: Upload data via a workspace

If you need to copy data onto the cluster:

1. Create a Data Volume called `ocr-documents` (new empty PVC — size it
   for your data)
2. Create a temporary **Workspace**:

| Field | Value |
|-------|-------|
| **Name** | `ocr-data-upload` |
| **Image** | `nvcr.io/nvidia/pytorch:25.02-py3` |
| **GPU** | `0` |
| **CPU** | `2` |
| **Memory** | `4Gi` |
| **Data volume** | `ocr-documents` → `/data/documents` (read-write) |
| **Tool** | Jupyter → port 8888 |

3. Open a terminal in the workspace and transfer files:

```bash
# From your local machine / data server:
rsync -avP /path/to/imaging_data/ user@<workspace-host>:/data/documents/

# Or from inside the workspace, pull from a file server:
rsync -avP user@fileserver:/imaging_archive/ /data/documents/

# Or use rclone for S3/cloud sources:
pip install rclone
rclone copy s3://bucket/path /data/documents/ --progress
```

> **Tip for large transfers:** `rsync` with `--progress` and `--partial`
> handles interruptions gracefully. For faster transfers, run multiple
> rsync jobs for different subdirectories in parallel.

4. Verify:

```bash
find /data/documents -type f | wc -l
du -sh /data/documents

# Check file types
find /data/documents -type f | sed 's/.*\.//' | sort | uniq -c | sort -rn | head
```

5. **Stop the upload workspace** once done.

### Option C: PoC with a few sample docs

For testing with 5 sample documents, skip the PVC entirely. Just upload
files directly to the setup workspace (Step 2) via Jupyter's file upload
button or `scp`. Use the workspace's local storage at
`/home/jovyan/sample_docs/`.

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
