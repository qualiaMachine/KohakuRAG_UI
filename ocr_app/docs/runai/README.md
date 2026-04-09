# Deploying OCR Document Extraction on RunAI

Production deployment for extracting structured JSON from research admin
documents (grant award notices, budgets, terms & conditions, scanned
TIFFs). Uses a hybrid pipeline: instant text extraction for digital PDFs,
VLM OCR (Qwen2.5-VL-7B) fallback for scans.

## Why this architecture?

Traditional OCR pipelines (Tesseract + regex) are brittle — they break on
layout changes and need per-document-type rules. This pipeline uses an LLM
to understand document structure semantically:

- **Digital PDFs:** PyMuPDF extracts text (instant, no GPU), then the LLM
  parses it into structured JSON
- **Scanned PDFs / TIFFs:** Qwen2.5-VL renders the page and does OCR +
  structuring in one shot

Both paths produce the same JSON output. The pipeline auto-detects which
path to use per page.

| Workload | Type | What it does | GPU | Port |
|----------|------|-------------|-----|------|
| **`ocr-vllm`** | Inference | Serves Qwen2.5-VL-7B for text parsing + VLM OCR | 0.80 | 8000 |
| **`ocr-setup`** | Workspace | One-time setup — test pipeline on sample docs | 0 | 8888 |
| **`ocr-batch`** | Workspace | Production batch processing | 0 | 8888 |
| **`ocr-extract`** | Inference | *(optional)* FastAPI extraction server for API/UI use | 0 | 8090 |
| **`ocr-app`** | Workspace | *(optional)* Streamlit UI for PoC demos | 0 | 8501 |

Only `ocr-vllm` uses GPU. Everything else is CPU-only.

### Service layout

```
                    +---------------------+
                    |   vLLM Server       |
                    |   Qwen2.5-VL-7B    |
                    |   Port 8000 (GPU)   |
                    +---------^-----------+
                              | HTTP (cluster DNS)
              +---------------+----------------+
              |               |                |
   +----------+---+  +-------+--------+  +----+-----------+
   | Batch Script |  | Extract Server |  | Streamlit UI   |
   | (workspace)  |  | (optional API) |  | (optional PoC) |
   | CPU only     |  | CPU only       |  | CPU only       |
   +--------------+  +----------------+  +----------------+
```

All paths talk to the same vLLM server. The batch script is the primary
tool for processing large collections. The extraction server and Streamlit
UI are optional — useful for interactive demos.

---

## Deployment Guide

Follow these docs in order:

0. **[Setup Data Volumes](setup-data-volumes.md)** — Create PVCs for input documents, output JSONs, and models *(skip for PoC — just drag-and-drop files in the Streamlit app)*
1. **[Deploy vLLM Server](deploy-vllm.md)** — Qwen2.5-VL-7B for text parsing + VLM OCR
2. **[Deploy Streamlit App](deploy-streamlit.md)** — Interactive UI for uploading docs and previewing results *(recommended starting point for PoC)*
3. **[Setup & Test Workspace](setup-workspace.md)** — Jupyter workspace for CLI-based testing *(alternative to Streamlit)*
4. **[Batch Processing](batch-processing.md)** — Production workspace for large-scale runs

All steps use the **RunAI web UI only** — no CLI tools required.

### PoC path (5 sample docs)

1. Deploy `ocr-vllm` (Step 1) — GPU inference
2. Deploy `ocr-extract` + `ocr-app` (Step 2) — CPU, Streamlit UI
3. Drag-and-drop your 5 PDFs into the app, pick a format, see results

### Production path (10K+ docs/month)

0. Setup data volumes (Step 0) — PVCs for input/output
1. Deploy `ocr-vllm` (Step 1) — GPU inference
4. Deploy `ocr-batch` (Step 4) — batch workspace with `--resume`

---

## Additional Docs

- **[Troubleshooting](troubleshooting.md)** — Common errors and fixes
