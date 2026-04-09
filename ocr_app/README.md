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
  +---------------------+
  |   Streamlit UI      |  (CPU, port 8501)
  |   Upload & preview  |
  +---------+-----------+
            | HTTP
            v
  +---------------------+     +----------------------+
  |  Extraction Server  |---->|   vLLM / Ollama      |
  |  FastAPI (CPU)      |     |   Qwen2.5-VL-7B     |
  |  Port 8090          |     |   Port 8000 (GPU)    |
  |                     |     |                      |
  |  PDF: text extract  |     |   Text: JSON parse   |
  |  TIFF: send image   |     |   Image: VLM OCR     |
  +---------------------+     +----------------------+
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

Full deployment guide: **[docs/runai/README.md](docs/runai/README.md)**

Follow these docs in order:

0. [Setup Data Volumes](docs/runai/setup-data-volumes.md) — PVCs for input docs, output JSONs, models
1. [Deploy vLLM Server](docs/runai/deploy-vllm.md) — Qwen2.5-VL-7B for text parsing + VLM OCR
2. [Setup & Test Workspace](docs/runai/setup-workspace.md) — Verify pipeline on sample docs
3. [Batch Processing](docs/runai/batch-processing.md) — Production batch workspace
4. [Deploy Streamlit App](docs/runai/deploy-streamlit.md) *(optional)* — Interactive UI for PoC demos

Additional: [Troubleshooting](docs/runai/troubleshooting.md)

## Batch Processing

For processing large document collections (14TB+, millions of docs):

```bash
python ocr_app/scripts/batch_extract.py \
    --input-dir /data/documents \
    --output-dir /data/extracted \
    --format award \
    --concurrency 4 \
    --resume
```

Features:
- Walks directories of PDFs/TIFFs, writes one JSON per document
- Concurrent async requests to vLLM (configurable concurrency)
- Resumable — tracks completed files, re-run with `--resume` after failures
- Preserves subdirectory structure in output
- Per-file progress logging with throughput stats

See [docs/runai/batch-processing.md](docs/runai/batch-processing.md) for the batch workspace setup.

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
- Use `batch_extract.py` with `--concurrency 4-16` depending on GPU headroom
- Digital PDFs skip VLM entirely — throughput limited only by LLM text parsing speed
- vLLM handles concurrent requests with continuous batching internally
- Resumable — `--resume` skips already-completed files after failures
- Consider a text-only LLM (e.g. Qwen2.5-7B-Instruct, smaller/faster) for the
  text parsing path, with a separate VLM endpoint only for scans

## Key Files

```
ocr_app/
├── app.py                          # Streamlit UI (interactive PoC)
├── scripts/
│   ├── ocr_server.py               # FastAPI extraction server
│   └── batch_extract.py            # Batch processing script
├── deploy/
│   └── runai_jobs.yaml             # RunAI job configs
├── docs/runai/                     # RunAI deployment guides
│   ├── README.md                   #   Overview + deployment order
│   ├── setup-data-volumes.md       #   PVC setup + data upload
│   ├── deploy-vllm.md             #   vLLM server (GPU)
│   ├── setup-workspace.md          #   Setup & test workspace
│   ├── batch-processing.md         #   Production batch runs
│   ├── deploy-streamlit.md         #   Streamlit UI (optional)
│   └── troubleshooting.md          #   Common issues
├── requirements_server.txt         # Server deps (lightweight, no GPU)
├── requirements_ui.txt             # Streamlit UI deps
├── .env.example                    # Environment variable template
└── README.md                       # This file
```
