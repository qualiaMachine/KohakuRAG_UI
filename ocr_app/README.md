# Document Extraction — Research & Sponsored Programs

Structured data extraction from grant award notices, budgets, terms & conditions, and other research admin documents. Produces JSON for downstream systematic analysis.

**Hybrid pipeline** — digital PDFs are processed instantly via text extraction; scanned pages and TIFFs fall back to VLM OCR automatically.

## Why this over Tesseract + regex?

| | Tesseract + rules | This pipeline |
|---|---|---|
| Digital PDFs | Unnecessary OCR | Direct text extraction (instant) |
| Scanned docs | OCR only, no structure | VLM: OCR + structuring in one shot |
| Table extraction | Fragile heuristics | LLM understands layout semantically |
| New doc formats | New rules each time | Zero-shot — just describe the format |
| Structured output | Regex/rules per doc type | Structured JSON from any document |
| Maintenance | High — rules break on layout changes | Low — prompts generalize |

## Architecture

```
┌─────────────────────┐
│   Streamlit UI      │  (CPU, port 8501)
│   Upload & preview  │
└──────────┬──────────┘
           │ HTTP
           ▼
┌─────────────────────┐     ┌─────────────────────┐
│   Extraction Server │────▶│   vLLM / Ollama      │
│   FastAPI (CPU)     │     │   Qwen2.5-VL-7B     │
│   Port 8090         │     │   Port 8000 (GPU)    │
│                     │     │                      │
│   PDF → text extract│     │   Text → JSON parse  │
│   TIFF → send image │     │   Image → VLM OCR    │
└─────────────────────┘     └─────────────────────┘
```

**Digital PDF path** (fast, most docs): PyMuPDF text extract → LLM parses text → JSON
**Scan / TIFF path** (fallback): render page → VLM OCR + structuring → JSON

The extraction server is CPU-only. All GPU work happens in vLLM/Ollama.

## Quick Start (Local)

### 1. Start an LLM server

Using vLLM:
```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --dtype bfloat16 --max-model-len 8192
```

Or using Ollama (if already approved/available):
```bash
ollama serve
ollama pull qwen2.5-vl:7b
```

### 2. Start the extraction server

```bash
pip install -r ocr_app/requirements_server.txt

# For vLLM:
LLM_BASE_URL=http://localhost:8000/v1 python ocr_app/scripts/ocr_server.py

# For Ollama:
LLM_BASE_URL=http://localhost:11434/v1 python ocr_app/scripts/ocr_server.py
```

### 3. Start the Streamlit UI

```bash
pip install -r ocr_app/requirements_ui.txt
streamlit run ocr_app/app.py
# UI available at http://localhost:8501
```

### 4. Use the API directly

```bash
# Health check
curl http://localhost:8090/health

# Extract from PDF (auto-detects digital vs scanned)
curl -X POST http://localhost:8090/extract/pdf \
  -F "file=@award_notice.pdf" \
  -F "format=award"

# Extract from TIFF/image (always uses VLM)
curl -X POST http://localhost:8090/extract/image \
  -F "file=@scanned_doc.tiff" \
  -F "format=award"

# Specific pages only
curl -X POST http://localhost:8090/extract/pdf \
  -F "file=@big_document.pdf" \
  -F "format=budget" \
  -F "pages=1-5"
```

## Output Formats

| Format | Use case | Output |
|--------|----------|--------|
| `award` | Grant award notices, NOAs, subaward agreements | JSON: PI, award #, amounts, dates, F&A rate, conditions |
| `budget` | Budget pages, financial summaries, cost proposals | JSON: categories, line items, direct/indirect costs |
| `terms` | Award terms, RSP policies, compliance docs | JSON: sections, regulatory citations, definitions |
| `table` | Any tabular data | Markdown tables with exact numbers |
| `key_values` | Forms, labeled fields, summary pages | Flat JSON key-value pairs |
| `markdown` | General documents | Formatted Markdown |
| `json` | Generic structured extraction | JSON |
| `text` | Plain text | Raw text |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health + model info |
| `/info` | GET | Pipeline details and available formats |
| `/extract/pdf` | POST | Extract from PDF (hybrid: text + VLM fallback) |
| `/extract/image` | POST | Extract from image/TIFF (VLM OCR) |

## RunAI / PowerEdge Deployment

See `ocr_app/deploy/runai_jobs.yaml` for complete RunAI job definitions.

Three jobs:
1. **ocr-vllm** — Qwen2.5-VL-7B via vLLM (GPU 0.80)
2. **ocr-extract** — Extraction server (CPU only, calls vLLM over HTTP)
3. **ocr-app** — Streamlit UI (CPU only)

### Pre-requisite: Download model to shared PVC

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct',
                  cache_dir='/models/.cache/huggingface')
"
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BASE_URL` | `http://localhost:8000/v1` | vLLM / Ollama endpoint |
| `LLM_MODEL` | (auto-detected) | Model name for text parsing |
| `VLM_BASE_URL` | (same as LLM) | Separate VLM endpoint (optional) |
| `VLM_MODEL` | `Qwen/Qwen2.5-VL-7B-Instruct` | Vision model for scans |
| `OCR_PORT` | `8090` | Extraction server port |
| `MIN_TEXT_LENGTH` | `50` | Min chars to consider a page "digital" |
| `OCR_SERVICE_URL` | `http://localhost:8090` | UI → extraction server |

## GPU Requirements (vLLM server only)

| GPU | Config | Notes |
|-----|--------|-------|
| A100 80GB | `--dtype bfloat16` | Best experience |
| A100 40GB | `--dtype bfloat16 --max-model-len 4096` | Tight fit |
| A6000 48GB | `--dtype bfloat16` | Works well |
| L4/RTX 4090 24GB | `--quantization awq --max-model-len 4096` | Needs quantization |

## Scaling for 20M+ documents

For high-volume batch processing:
- The extraction server is stateless — run multiple replicas behind a load balancer
- vLLM handles concurrent requests with continuous batching
- Digital PDFs skip VLM entirely — throughput limited only by LLM text parsing speed
- Consider a text-only LLM (e.g. Qwen2.5-7B-Instruct, smaller/faster) for the
  text parsing path, with a separate VLM endpoint only for scans
