# VLM OCR Application

Document OCR powered by **Qwen2.5-VL-7B**, a state-of-the-art Vision Language Model. Replaces traditional OCR (Tesseract) with a VLM that understands document structure, tables, handwriting, and 90+ languages.

## Why VLMs over Tesseract?

| Feature | Tesseract | Qwen2.5-VL-7B |
|---------|-----------|----------------|
| Tables | Poor | Excellent — preserves structure |
| Handwriting | Very poor | Good (77%+ across languages) |
| Languages | Per-language models | 90+ languages, zero-shot |
| Layout understanding | Rule-based | Semantic understanding |
| Structured output | Text only | JSON, Markdown, LaTeX |
| Context awareness | None | Understands document meaning |

## Architecture

```
┌─────────────────────┐
│   Streamlit OCR UI  │  (CPU only, port 8501)
│   Upload & preview  │
└──────────┬──────────┘
           │ HTTP
           ▼
┌─────────────────────┐
│   OCR Server        │  (GPU, port 8090)
│   Qwen2.5-VL-7B    │
│   FastAPI           │
└─────────────────────┘
```

**Two deployment options for the OCR server:**
- **Option A: vLLM** (recommended) — Higher throughput, continuous batching, PagedAttention
- **Option B: Transformers** — Simpler setup, good for development

## Quick Start (Local)

### 1. Install dependencies

```bash
# GPU server (needs CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r ocr_app/requirements_server.txt

# Optional: flash attention for 2x speedup on A100/H100
pip install flash-attn --no-build-isolation
```

### 2. Start the OCR server

```bash
python ocr_app/scripts/ocr_server.py
# Server starts at http://localhost:8090
# First run downloads Qwen2.5-VL-7B (~15GB)
```

### 3. Start the Streamlit UI

```bash
# In a separate terminal
pip install -r ocr_app/requirements_ui.txt
streamlit run ocr_app/app.py
# UI available at http://localhost:8501
```

### 4. Use the API directly

```bash
# Health check
curl http://localhost:8090/health

# OCR an image (multipart upload)
curl -X POST http://localhost:8090/ocr/upload \
  -F "file=@document.png" \
  -F "format=markdown"

# OCR a PDF
curl -X POST http://localhost:8090/ocr/pdf \
  -F "file=@paper.pdf" \
  -F "format=text" \
  -F "pages=1-5"

# OCR with base64
curl -X POST http://localhost:8090/ocr \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "'$(base64 -w0 document.png)'", "format": "json"}'
```

## RunAI / PowerEdge Deployment

See `ocr_app/deploy/runai_jobs.yaml` for complete RunAI job definitions.

### Pre-requisite: Download model to shared PVC

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct',
                  cache_dir='/models/.cache/huggingface')
"
```

### Option A: vLLM backend (recommended)

```bash
runai submit ocr-vllm \
  --type inference \
  --image vllm/vllm-openai:latest \
  --gpu 0.80 --cpu 4 --memory 24Gi \
  --pvc shared-models:/models \
  --env HF_HOME=/models/.cache/huggingface \
  --env HF_HUB_OFFLINE=1 \
  --port 8090 \
  -- --model Qwen/Qwen2.5-VL-7B-Instruct \
    --dtype bfloat16 --port 8090 \
    --max-model-len 8192 \
    --limit-mm-per-prompt image=1
```

### Option B: FastAPI + Transformers

See the full command in `ocr_app/deploy/runai_jobs.yaml` (Option B section).

### Deploy the UI

```bash
# Workspace (not Inference) for browser-accessible proxy URL
# See runai_jobs.yaml for full UI deployment config
```

## GPU Requirements

| GPU | Config | Notes |
|-----|--------|-------|
| A100 80GB | `--dtype bfloat16` | Best experience, no quantization |
| A100 40GB | `--dtype bfloat16 --max-model-len 4096` | Tight fit |
| A6000 48GB | `--dtype bfloat16` | Works well |
| L4/RTX 4090 24GB | `--quantization awq --max-model-len 4096` | Needs quantization |

## Output Formats

| Format | Use case | Example |
|--------|----------|---------|
| `text` | Plain text extraction | Letters, articles, receipts |
| `markdown` | Structured documents | Papers, reports with headings |
| `json` | Forms and structured data | Invoices, applications |
| `table` | Tabular data | Spreadsheets, data tables |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/info` | GET | Model info and capabilities |
| `/ocr` | POST | OCR from base64 image |
| `/ocr/upload` | POST | OCR from uploaded image file |
| `/ocr/batch` | POST | OCR multiple base64 images |
| `/ocr/pdf` | POST | OCR a PDF (renders pages as images) |

## Configuration

All settings via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OCR_MODEL` | `Qwen/Qwen2.5-VL-7B-Instruct` | HuggingFace model ID |
| `OCR_PORT` | `8090` | Server port |
| `OCR_DEVICE` | `auto` | `cuda`, `cpu`, or `auto` |
| `OCR_MAX_PIXELS` | `1003520` | Max image resolution |
| `OCR_SERVICE_URL` | `http://localhost:8090` | UI -> server URL |
| `VLLM_BASE_URL` | (empty) | Set to use vLLM backend |

## Alternative Models

The server supports any Qwen2.5-VL model. Swap via `OCR_MODEL`:

| Model | VRAM | Best for |
|-------|------|----------|
| `Qwen/Qwen2.5-VL-7B-Instruct` | ~17GB | Default, great balance |
| `Qwen/Qwen2.5-VL-3B-Instruct` | ~8GB | Lower VRAM, still good |
| `Qwen/Qwen2.5-VL-72B-Instruct` | ~144GB | Maximum quality (multi-GPU) |

Other VLMs worth considering (may require server code changes):
- **GOT-OCR2.0** (580M) — Ultra-light, great for equations/LaTeX
- **PaddleOCR-VL** (0.9B) — Fastest, Apache 2.0 licensed
- **Chandra OCR 2** (9B) — Best for handwriting across 90 languages
