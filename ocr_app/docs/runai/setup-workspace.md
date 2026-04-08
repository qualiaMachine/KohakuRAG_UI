# Setup & Test Workspace (`ocr-setup`)

> **Step 2** in the [deployment guide](README.md). Comes after
> [Deploy vLLM Server](deploy-vllm.md) (Step 1).

## What this workspace does

`ocr-setup` is a **one-time workspace** to verify the extraction pipeline
works end-to-end before committing to batch runs:

1. Confirms vLLM is reachable from the workspace
2. Tests extraction on a sample document
3. Validates the JSON output schema and content

Once verified, **stop the workspace** — you don't need it at runtime.
The batch workspace (Step 3) handles production processing.

## What this workspace does NOT do

- **Does not run production workloads.** Use the batch workspace (Step 3)
  for that.
- **Does not need GPU.** All GPU work happens in the vLLM server.

---

In the RunAI UI: **Workloads** > **New Workload** > **Workspace**

## Basic settings

| Field | Value |
|-------|-------|
| **Cluster** | `doit-ai-cluster` |
| **Project** | Your project |
| **Workspace name** | `ocr-setup` |

## Environment image

| Field | Value |
|-------|-------|
| **Image** | Custom image |
| **Image URL** | `nvcr.io/nvidia/pytorch:25.02-py3` |
| **Image pull** | Pull the image only if it's not already present on the host |

## Tools

Add Jupyter for browser access:

| Field | Value |
|-------|-------|
| **Tool type** | Jupyter |
| **Port** | `8888` |

## Runtime settings

| Field | Value |
|-------|-------|
| **Command** | `bash` |
| **Arguments** | See below |
| **Working directory** | *(leave empty)* |

### Arguments (copy-paste)

```
-c "pip install uv && rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED && curl -sL https://github.com/qualiaMachine/KohakuRAG_UI/archive/refs/heads/claude/ocr-vlm-application-hqgf2.tar.gz | tar xz -C /tmp && mv /tmp/KohakuRAG_UI-claude-ocr-vlm-application-hqgf2 /tmp/KohakuRAG_UI && cd /tmp/KohakuRAG_UI && uv pip install --system httpx pymupdf Pillow && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"
```

> **What this does:** Downloads the repo tarball, installs the lightweight
> extraction dependencies (no torch/GPU libs), and starts Jupyter Lab.
> The repo ends up at `/tmp/KohakuRAG_UI`.

**Environment variables:**

| Name | Value |
|------|-------|
| `LLM_BASE_URL` | `http://ocr-vllm.runai-<project>.svc.cluster.local/v1` |
| `VLM_MODEL` | `Qwen/Qwen2.5-VL-7B-Instruct` |

> **Replace `<project>`** with your actual RunAI project name
> (e.g. `jupyter-endemann01`). The full URL would look like:
> `http://ocr-vllm.runai-jupyter-endemann01.svc.cluster.local/v1`

## Compute resources

| Field | Value |
|-------|-------|
| **GPU devices** | `0` (none — CPU only) |
| **CPU request** | `2` |
| **CPU memory request** | `4Gi` |

## Data & storage

**For PoC (5 sample docs):** Skip data volumes. Upload docs directly via
Jupyter's file upload button after the workspace starts.

**For production testing:** Attach the document and output volumes:

| Data volume name | Container path |
|------------------|----------------|
| `ocr-documents` | `/data/documents` |
| `ocr-extracted` | `/data/extracted` |

---

## Access the workspace

Once the job reaches `Running` status, click the workspace name in the
RunAI UI → click the **Jupyter** tool link. This opens Jupyter Lab in
your browser.

Open a **Terminal** from Jupyter Lab's launcher.

---

## Verification checklist

### 1. Check vLLM is reachable

```bash
curl $LLM_BASE_URL/models
```

Expected output:
```json
{"data": [{"id": "Qwen/Qwen2.5-VL-7B-Instruct", ...}]}
```

If this fails, the vLLM server isn't ready yet or the URL is wrong.
See [Troubleshooting](troubleshooting.md).

### 2. Upload sample docs (PoC only)

Use Jupyter's file upload button (up arrow icon in the file browser) to
upload your 5 sample PDFs. They'll land in `/home/jovyan/` or wherever
Jupyter's file browser is pointed.

Create a directory for them:

```bash
mkdir -p /home/jovyan/sample_docs
# Move uploaded files there
mv /home/jovyan/*.pdf /home/jovyan/sample_docs/
ls /home/jovyan/sample_docs/
```

### 3. Test extraction on a single document

```bash
cd /tmp/KohakuRAG_UI

# Run on your sample docs
python ocr_app/scripts/batch_extract.py \
    --input-dir /home/jovyan/sample_docs \
    --output-dir /home/jovyan/extracted \
    --format award \
    --concurrency 1
```

Expected output:
```
[batch] Found 5 files, 0 already completed, 5 to process
[batch] LLM: Qwen/Qwen2.5-VL-7B-Instruct at http://ocr-vllm.../v1
[1/5] OK doc1.pdf (3p, 0d/3s, 14.2s) -> doc1.json
[2/5] OK doc2.pdf (2p, 0d/2s, 9.8s) -> doc2.json
...
```

The `0d/3s` means 0 digital pages, 3 scanned — confirming VLM OCR is
being used for your scanned PDFs.

### 4. Inspect the output

```bash
# Pretty-print the first result
cat /home/jovyan/extracted/*.json | python -m json.tool | head -80
```

Or in a Jupyter notebook cell:

```python
import json
from pathlib import Path

for out in sorted(Path("/home/jovyan/extracted").glob("*.json")):
    data = json.loads(out.read_text())
    print(f"\n{'='*60}")
    print(f"File: {data['source_file']}")
    print(f"Pages: {data['total_pages']} ({data['digital_pages']}d/{data['scanned_pages']}s)")
    for page in data['pages']:
        print(f"  Page {page['page']}: {page['method']} ({page['elapsed_ms']:.0f}ms)")
        # Show first 300 chars of extracted text
        print(f"    {page['text'][:300]}...")
```

### 5. Check if the right fields are being extracted

If the output JSON doesn't have the fields you need, try a different
format:

```bash
# More flexible — extracts any key-value pairs it finds
python ocr_app/scripts/batch_extract.py \
    --input-dir /home/jovyan/sample_docs \
    --output-dir /home/jovyan/extracted_kv \
    --format key_values \
    --concurrency 1

# Raw text — see exactly what the VLM reads from the image
python ocr_app/scripts/batch_extract.py \
    --input-dir /home/jovyan/sample_docs \
    --output-dir /home/jovyan/extracted_text \
    --format text \
    --concurrency 1
```

### 6. Done — stop the workspace

Once you're satisfied with the output, **stop the workspace** from the
RunAI UI. You can always restart it later to test new formats or
different document types.

For production batch runs, proceed to
[Batch Processing](batch-processing.md) (Step 3).
