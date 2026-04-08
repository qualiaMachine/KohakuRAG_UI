# Setup & Test Workspace (`ocr-setup`)

> **Step 2** in the [deployment guide](README.md). Comes after
> [Deploy vLLM Server](deploy-vllm.md) (Step 1).

## What this workspace does

`ocr-setup` is a **one-time workspace** to verify the extraction pipeline
works before committing to batch runs:

1. Confirms data volumes are mounted correctly
2. Tests vLLM connectivity
3. Runs extraction on a sample document
4. Validates the JSON output

Once verified, **stop the workspace**. The batch workspace (Step 3)
handles production processing.

---

## RunAI UI Settings

| Field | Value |
|-------|-------|
| **Workload type** | Workspace |
| **Name** | `ocr-setup` |
| **Image** | `nvcr.io/nvidia/pytorch:25.02-py3` |
| **Tool** | Jupyter → port `8888` |
| **Command** | `bash` |
| **Arguments** | See below |
| **GPU** | `0` (none) |
| **CPU** | `2` |
| **Memory** | `4Gi` |
| **Data volumes** | `ocr-documents` → `/data/documents` (read-only) |
| | `ocr-extracted` → `/data/extracted` (read-write) |

> **PoC with sample docs?** Skip the `ocr-documents` data volume. Upload
> your 5 sample files directly via Jupyter's upload button after the
> workspace starts.

## Arguments (copy-paste)

```
-c "pip install uv && rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED && curl -sL https://github.com/qualiaMachine/KohakuRAG_UI/archive/refs/heads/claude/ocr-vlm-application-hqgf2.tar.gz | tar xz -C /tmp && mv /tmp/KohakuRAG_UI-claude-ocr-vlm-application-hqgf2 /tmp/KohakuRAG_UI && cd /tmp/KohakuRAG_UI && uv pip install --system httpx pymupdf Pillow && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"
```

## Environment Variables

| Variable | Value |
|----------|-------|
| `LLM_BASE_URL` | `http://ocr-vllm.runai-<project>.svc.cluster.local/v1` |
| `VLM_MODEL` | `Qwen/Qwen2.5-VL-7B-Instruct` |

---

## Verification checklist

Open a terminal in the Jupyter workspace and run through these:

### 1. Check data volumes

```bash
# Input documents (or your uploaded sample docs)
ls /data/documents/ | head -20
find /data/documents -type f | wc -l
du -sh /data/documents

# Output directory (should be empty)
ls /data/extracted/
```

### 2. Check vLLM is reachable

```bash
curl http://ocr-vllm.runai-<project>.svc.cluster.local/v1/models
# Expected: {"data": [{"id": "Qwen/Qwen2.5-VL-7B-Instruct", ...}]}
```

If this fails, see [Troubleshooting](troubleshooting.md).

### 3. Test extraction on a single document

```bash
cd /tmp/KohakuRAG_UI

# See what's available
ls /data/documents/ | head -5

# Run on one file
python ocr_app/scripts/batch_extract.py \
    --input-dir /data/documents \
    --output-dir /data/extracted/test \
    --format award \
    --concurrency 1

# Check the output
cat /data/extracted/test/*.json | python -m json.tool | head -50
```

### 4. Validate the JSON output

```python
import json
from pathlib import Path

out = next(Path("/data/extracted/test").glob("*.json"))
data = json.loads(out.read_text())

print(f"File: {data['source_file']}")
print(f"Pages: {data['total_pages']} ({data['digital_pages']}d/{data['scanned_pages']}s)")
for page in data['pages']:
    print(f"  Page {page['page']}: {page['method']} ({page['elapsed_ms']}ms)")
    print(f"    {page['text'][:200]}...")
```

### 5. Clean up and stop

```bash
rm -rf /data/extracted/test
```

**Stop the workspace** — you don't need it for production runs.
