# Deploying OCR Document Extraction on RunAI

Step-by-step guide for deploying the document extraction pipeline on RunAI
with a PowerEdge GPU cluster. Uses the hybrid pipeline: direct text
extraction for digital PDFs (no GPU), VLM OCR fallback for scans/TIFFs.

## Architecture

```
┌─────────────────────┐     ┌─────────────────────┐
│   Extraction Server │────▶│   vLLM Server       │
│   FastAPI (CPU)     │     │   Qwen2.5-VL-7B     │
│   Port 8090         │     │   Port 8000 (GPU)    │
└──────────┬──────────┘     └─────────────────────┘
           │
           │  (optional — for interactive use)
           ▼
┌─────────────────────┐
│   Streamlit UI      │
│   Port 8501 (CPU)   │
└─────────────────────┘
```

For batch processing, the extraction server is replaced by the batch
script running directly in a workspace — it talks to vLLM the same way.

| Workload | Type | What it does | GPU | Port |
|----------|------|-------------|-----|------|
| **`ocr-vllm`** | Inference | Serves Qwen2.5-VL-7B for text parsing + VLM OCR | 0.80 | 8000 |
| **`ocr-extract`** | Inference | Extraction server — text extraction + routes to vLLM | 0 | 8090 |
| **`ocr-app`** | Workspace | Streamlit UI (optional, for PoC demos) | 0 | 8501 |
| **`ocr-batch`** | Workspace | Batch processing workspace (for production runs) | 0 | — |

Only `ocr-vllm` uses GPU. Everything else is CPU-only.

---

## Prerequisites

- **Shared models PVC** with `Qwen/Qwen2.5-VL-7B-Instruct` downloaded.
  If you've already set this up for WattBot, the same PVC works — just
  add the Qwen VL model. See [setup-shared-models.md](../../docs/runai/setup-shared-models.md).

### Download Qwen2.5-VL-7B to shared PVC

If the model isn't already on your PVC, run this from any workspace with
write access to the PVC:

```python
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct",
                  cache_dir="/models/.cache/huggingface")
```

This downloads ~15 GB. Only needed once — all jobs mount the PVC read-only.

---

## Step 1: Deploy vLLM Server

The vLLM server handles both text parsing (digital PDFs) and VLM OCR
(scans/TIFFs). Qwen2.5-VL is a vision-language model, so it can do both.

### RunAI UI Settings

| Field | Value |
|-------|-------|
| **Workload type** | Inference |
| **Inference type** | Custom |
| **Name** | `ocr-vllm` |
| **Image** | `vllm/vllm-openai:latest` |
| **Container port** | `8000` |
| **Command** | *(leave empty)* |
| **Arguments** | `--model Qwen/Qwen2.5-VL-7B-Instruct --dtype bfloat16 --max-model-len 8192 --limit-mm-per-prompt image=1` |
| **GPU** | `0.80` (fractional) |
| **CPU** | `4` |
| **Memory** | `24Gi` |
| **Data volume** | `shared-models` → `/models` |

### Environment Variables

| Variable | Value |
|----------|-------|
| `HF_HOME` | `/models/.cache/huggingface` |
| `HF_HUB_CACHE` | `/models/.cache/huggingface` |
| `HF_HUB_OFFLINE` | `1` |

### GPU sizing

| GPU | Arguments |
|-----|-----------|
| A100 80GB | `--dtype bfloat16` (default above) |
| A100 40GB | `--dtype bfloat16 --max-model-len 4096` |
| A6000 48GB | `--dtype bfloat16` |
| L4/RTX 4090 24GB | `--quantization awq --max-model-len 4096` |

### Verify

Wait for the pod to reach `Running` state (2-5 min for model load), then
test from any workspace on the cluster:

```bash
curl http://ocr-vllm.runai-<project>.svc.cluster.local/v1/models
# Expected: {"data": [{"id": "Qwen/Qwen2.5-VL-7B-Instruct", ...}]}
```

> **Note:** Use the FQDN (`workload.runai-project.svc.cluster.local`) on
> port 80 (no port number). Knative envoy requires this — short names
> like `ocr-vllm:8000` return 404.

---

## Step 2: Deploy Extraction Server (optional — for API/UI use)

This is the FastAPI server that handles file uploads, text extraction,
and routes requests to vLLM. Skip this step if you only need batch
processing.

### RunAI UI Settings

| Field | Value |
|-------|-------|
| **Workload type** | Inference |
| **Inference type** | Custom |
| **Name** | `ocr-extract` |
| **Image** | `vllm/vllm-openai:latest` |
| **Container port** | `8090` |
| **Command** | `bash` |
| **Arguments** | See below |
| **GPU** | `0` (none — CPU only) |
| **CPU** | `2` |
| **Memory** | `4Gi` |

### Arguments (copy-paste)

```
-c "pip install uv && curl -sL https://github.com/qualiaMachine/KohakuRAG_UI/archive/refs/heads/claude/ocr-vlm-application-hqgf2.tar.gz | tar xz -C /tmp && mv /tmp/KohakuRAG_UI-claude-ocr-vlm-application-hqgf2 /tmp/KohakuRAG_UI && cd /tmp/KohakuRAG_UI && uv pip install --system fastapi uvicorn python-multipart httpx pymupdf Pillow && python3 ocr_app/scripts/ocr_server.py"
```

### Environment Variables

| Variable | Value |
|----------|-------|
| `LLM_BASE_URL` | `http://ocr-vllm.runai-<project>.svc.cluster.local/v1` |
| `VLM_MODEL` | `Qwen/Qwen2.5-VL-7B-Instruct` |
| `OCR_PORT` | `8090` |

---

## Step 3: Deploy Streamlit UI (optional — for PoC demos)

Browser-based UI for uploading individual documents and previewing results.
Good for demos, not for bulk processing.

### RunAI UI Settings

| Field | Value |
|-------|-------|
| **Workload type** | Workspace |
| **Name** | `ocr-app` |
| **Image** | `nvcr.io/nvidia/pytorch:25.02-py3` |
| **Tool** | Custom URL → `streamlit` → port `8501` |
| **Command** | `bash` |
| **Arguments** | See below |
| **GPU** | `0` (none) |
| **CPU** | `1` |
| **Memory** | `2Gi` |

### Arguments (copy-paste)

```
-c "pip install uv && rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED && curl -sL https://github.com/qualiaMachine/KohakuRAG_UI/archive/refs/heads/claude/ocr-vlm-application-hqgf2.tar.gz | tar xz -C /tmp && mv /tmp/KohakuRAG_UI-claude-ocr-vlm-application-hqgf2 /tmp/KohakuRAG_UI && cd /tmp/KohakuRAG_UI && uv pip install --system streamlit httpx Pillow python-dotenv && python -m streamlit run ocr_app/app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false --server.baseUrlPath=$STREAMLIT_BASE_PATH"
```

### Environment Variables

| Variable | Value |
|----------|-------|
| `OCR_SERVICE_URL` | `http://ocr-extract.runai-<project>.svc.cluster.local` |
| `STREAMLIT_BASE_PATH` | `/<project>/<workspace-name>/proxy/8501` |

### Access URL

```
https://<cluster-host>/<project>/ocr-app/proxy/8501/
```

---

## Step 4: Batch Processing Workspace

For production runs against 14TB / 20M documents. This workspace mounts
the document PVC directly and runs the batch script.

### RunAI UI Settings

| Field | Value |
|-------|-------|
| **Workload type** | Workspace |
| **Name** | `ocr-batch` |
| **Image** | `nvcr.io/nvidia/pytorch:25.02-py3` |
| **Tool** | Jupyter → port `8888` (for monitoring/debugging) |
| **Command** | `bash` |
| **Arguments** | See below |
| **GPU** | `0` (none — all GPU work is in ocr-vllm) |
| **CPU** | `4` |
| **Memory** | `8Gi` |
| **Data volumes** | `document-store` → `/data/documents` (input docs) |
|  | `extraction-output` → `/data/extracted` (output JSONs) |

### Arguments (copy-paste)

This starts Jupyter and installs dependencies. You then run the batch
script from a terminal inside the workspace.

```
-c "pip install uv && rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED && curl -sL https://github.com/qualiaMachine/KohakuRAG_UI/archive/refs/heads/claude/ocr-vlm-application-hqgf2.tar.gz | tar xz -C /tmp && mv /tmp/KohakuRAG_UI-claude-ocr-vlm-application-hqgf2 /tmp/KohakuRAG_UI && cd /tmp/KohakuRAG_UI && uv pip install --system httpx pymupdf Pillow && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"
```

### Environment Variables

| Variable | Value |
|----------|-------|
| `LLM_BASE_URL` | `http://ocr-vllm.runai-<project>.svc.cluster.local/v1` |
| `VLM_MODEL` | `Qwen/Qwen2.5-VL-7B-Instruct` |

### Running the batch script

Open a terminal in the Jupyter workspace, then:

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

### Monitoring progress

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

### Tuning concurrency

| `--concurrency` | Best for |
|-----------------|----------|
| `2-4` | Default. Safe for single vLLM instance. |
| `8-16` | If vLLM is on a large GPU (A100 80GB) with headroom. |
| `1` | Debugging. Sequential, easy to read logs. |

vLLM handles batching internally, so higher concurrency doesn't always
mean more throughput — it depends on GPU memory and model size. Start at
4 and increase if vLLM's GPU utilization is below 80%.

---

## Data Volume Setup

You need two PVCs for batch processing:

### Input documents PVC

Mount wherever DoIT's imaging data lives. If the documents are already on
a shared filesystem, create a Data Volume pointing to it.

| Field | Value |
|-------|-------|
| **Name** | `document-store` |
| **Mount path** | `/data/documents` |
| **Access** | Read-only is fine |

### Output PVC

For the extracted JSON files. Size depends on your corpus — JSON output is
typically 1-5% of input size (text is small compared to images/PDFs).

| Field | Value |
|-------|-------|
| **Name** | `extraction-output` |
| **Mount path** | `/data/extracted` |
| **Access** | Read-write |
| **Size** | Start with 100Gi, expand as needed |

---

## Deployment Order

1. **Shared models PVC** — download Qwen2.5-VL-7B (one-time, ~15GB)
2. **`ocr-vllm`** — start the GPU inference server (~2-5 min to load)
3. Then either:
   - **`ocr-extract`** + **`ocr-app`** for interactive PoC demos
   - **`ocr-batch`** for production batch runs

The vLLM server is shared — both the extraction server and batch script
talk to the same endpoint.

---

## Troubleshooting

### vLLM won't start / OOM

- Check GPU memory: `--max-model-len 4096` reduces KV cache memory
- For 24GB GPUs: add `--quantization awq`
- Check logs: `runai logs ocr-vllm`

### Extraction server can't reach vLLM

- Use FQDN: `http://ocr-vllm.runai-<project>.svc.cluster.local/v1`
- Do NOT include port number (Knative routes on port 80)
- Test from the extraction pod: `curl http://ocr-vllm.runai-<project>.svc.cluster.local/v1/models`

### Batch script hangs

- Check vLLM logs for errors: `runai logs ocr-vllm`
- Try `--concurrency 1` to isolate the issue
- Check if vLLM is OOM — reduce concurrency or `--max-model-len`

### Bad JSON output

- Try a different format: `--format key_values` is more flexible than `--format award`
- Check if the document type matches the format — e.g., don't use `award` for general correspondence
- Use `--format text` to see raw extraction, then use a more specific format

### Resume not working

- The state file is at `<output-dir>/.batch_state`
- It tracks completed files by full input path
- If you moved input files, the paths won't match — delete `.batch_state` to restart
