# Deploy Streamlit App (`ocr-app`) — Optional

> **Step 4** in the [deployment guide](README.md). This step is optional —
> only needed for interactive PoC demos.

Browser-based UI for uploading individual documents and previewing
extracted results. Requires the extraction server (`ocr-extract`)
running as well.

---

## Deploy the extraction server first

The Streamlit UI talks to the FastAPI extraction server, not directly to
vLLM.

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

### Extraction server arguments (copy-paste)

```
-c "pip install uv && curl -sL https://github.com/qualiaMachine/KohakuRAG_UI/archive/refs/heads/claude/ocr-vlm-application-hqgf2.tar.gz | tar xz -C /tmp && mv /tmp/KohakuRAG_UI-claude-ocr-vlm-application-hqgf2 /tmp/KohakuRAG_UI && cd /tmp/KohakuRAG_UI && uv pip install --system fastapi uvicorn python-multipart httpx pymupdf Pillow && python3 ocr_app/scripts/ocr_server.py"
```

### Extraction server environment variables

| Variable | Value |
|----------|-------|
| `LLM_BASE_URL` | `http://ocr-vllm.runai-<project>.svc.cluster.local/v1` |
| `VLM_MODEL` | `Qwen/Qwen2.5-VL-7B-Instruct` |
| `OCR_PORT` | `8090` |

---

## Deploy the Streamlit UI

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

### Streamlit arguments (copy-paste)

```
-c "pip install uv && rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED && curl -sL https://github.com/qualiaMachine/KohakuRAG_UI/archive/refs/heads/claude/ocr-vlm-application-hqgf2.tar.gz | tar xz -C /tmp && mv /tmp/KohakuRAG_UI-claude-ocr-vlm-application-hqgf2 /tmp/KohakuRAG_UI && cd /tmp/KohakuRAG_UI && uv pip install --system streamlit httpx Pillow python-dotenv && python -m streamlit run ocr_app/app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false --server.baseUrlPath=$STREAMLIT_BASE_PATH"
```

### Streamlit environment variables

| Variable | Value |
|----------|-------|
| `OCR_SERVICE_URL` | `http://ocr-extract.runai-<project>.svc.cluster.local` |
| `STREAMLIT_BASE_PATH` | `/<project>/<workspace-name>/proxy/8501` |

## Access URL

```
https://<cluster-host>/<project>/ocr-app/proxy/8501/
```
