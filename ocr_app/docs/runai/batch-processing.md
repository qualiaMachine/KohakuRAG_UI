# Batch Processing Workspace (`ocr-batch`)

> **Step 3** in the [deployment guide](README.md). Comes after
> [Setup & Test Workspace](setup-workspace.md) (Step 2).

Production workspace for processing large document collections. Mounts
the document PVC and runs the batch script against vLLM.

---

## RunAI UI Settings

| Field | Value |
|-------|-------|
| **Workload type** | Workspace |
| **Name** | `ocr-batch` |
| **Image** | `nvcr.io/nvidia/pytorch:25.02-py3` |
| **Tool** | Jupyter → port `8888` |
| **Command** | `bash` |
| **Arguments** | See below |
| **GPU** | `0` (none — all GPU work is in ocr-vllm) |
| **CPU** | `4` |
| **Memory** | `8Gi` |
| **Data volumes** | `ocr-documents` → `/data/documents` (read-only) |
| | `ocr-extracted` → `/data/extracted` (read-write) |

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

## Running the batch script

Open a terminal in the Jupyter workspace:

```bash
cd /tmp/KohakuRAG_UI

# Process all PDFs — award notices
python ocr_app/scripts/batch_extract.py \
    --input-dir /data/documents \
    --output-dir /data/extracted \
    --format award \
    --concurrency 4

# Process only TIFFs
python ocr_app/scripts/batch_extract.py \
    --input-dir /data/documents \
    --output-dir /data/extracted \
    --format key_values \
    --extensions .tiff .tif \
    --concurrency 8

# Resume after failure
python ocr_app/scripts/batch_extract.py \
    --input-dir /data/documents \
    --output-dir /data/extracted \
    --format award \
    --resume
```

## Monitoring progress

The batch script prints per-file progress:

```
[batch] Found 45000 files, 0 already completed, 45000 to process
[batch] LLM: Qwen/Qwen2.5-VL-7B-Instruct at http://ocr-vllm.../v1
[1/45000] OK award_notice_2019.pdf (3p, 3d/0s, 2.1s) -> award_notice_2019.json
[2/45000] OK budget_fy2020.pdf (5p, 5d/0s, 3.4s) -> budget_fy2020.json
[3/45000] OK scanned_agreement.tiff (1p, 0d/1s, 8.2s) -> scanned_agreement.json
```

`3d/0s` = 3 digital pages, 0 scanned. Digital pages are much faster since
they skip VLM entirely.

## Tuning concurrency

| `--concurrency` | Best for |
|-----------------|----------|
| `2-4` | Default. Safe for single vLLM instance. |
| `8-16` | If vLLM is on a large GPU (A100 80GB) with headroom. |
| `1` | Debugging. Sequential, easy to read logs. |

vLLM handles batching internally, so higher concurrency doesn't always
mean more throughput — it depends on GPU memory and model size. Start at
4 and increase if vLLM's GPU utilization is below 80%.

## Output format

One JSON file per input document, preserving subdirectory structure:

```
/data/extracted/
├── subdir_a/
│   ├── doc1.json
│   └── doc2.json
└── subdir_b/
    └── doc3.json
```

Each JSON contains:

```json
{
  "source_file": "/data/documents/subdir_a/doc1.pdf",
  "format": "award",
  "total_pages": 3,
  "digital_pages": 3,
  "scanned_pages": 0,
  "pages": [
    {
      "page": 1,
      "text": "{\"document_type\": \"Notice of Award\", ...}",
      "method": "text_extraction",
      "elapsed_ms": 1250.5
    }
  ]
}
```
