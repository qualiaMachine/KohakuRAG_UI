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
| **`qwen2-5--vl--7b--instruct`** | Inference | Serves Qwen2.5-VL-7B for text parsing + VLM OCR | 0.80 | 8000 |
| **`ocr-setup`** | Workspace | One-time setup — test pipeline on sample docs | 0 | 8888 |
| **`ocr-batch`** | Workspace | Production batch processing | 0 | 8888 |
| **`ocr-extract`** | Inference | *(optional)* FastAPI extraction server for API/UI use | 0 | 8090 |
| **`ocr-app`** | Workspace | *(optional)* Streamlit UI for PoC demos | 0 | 8501 |

Only `qwen2-5--vl--7b--instruct` uses GPU. Everything else is CPU-only.

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

0. **[Setup Data Volumes](setup-data-volumes.md)** — Download model to shared PVC, create output volume
1. **[Deploy vLLM Server](deploy-vllm.md)** — Qwen2.5-VL-7B for text parsing + VLM OCR
2. **[Setup & Test Workspace](setup-workspace.md)** — Experiment with pipeline in notebook, iterate on prompts/formats
3. **[Deploy Streamlit App](deploy-streamlit.md)** *(optional)* — Polished demo UI
4. **[Batch Processing](batch-processing.md)** — Production workspace for large-scale runs

All steps use the **RunAI web UI only** — no CLI tools required.

### PoC path (5 sample docs)

0. Download model to shared PVC (Step 0)
1. Deploy `qwen2-5--vl--7b--instruct` (Step 1) — GPU inference
2. Setup & test workspace (Step 2) — upload docs, run test notebook
3. Optionally deploy Streamlit app (Step 3) for a nicer demo UI

### Production path (10K+ docs/month)

0. Setup data volumes (Step 0) — PVCs for input/output
1. Deploy `qwen2-5--vl--7b--instruct` (Step 1) — GPU inference
2. Setup & test workspace (Step 2) — verify with notebook
4. Deploy `ocr-batch` (Step 4) — batch workspace with `--resume`

---

## Additional Docs

- **[Troubleshooting](troubleshooting.md)** — Common errors and fixes
