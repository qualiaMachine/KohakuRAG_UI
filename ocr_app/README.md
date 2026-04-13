# Document Extraction

Structured data extraction from grant award notices, budgets, terms &
conditions, archival scans, and other institutional documents. Produces
JSON for downstream systematic analysis.

Hybrid pipeline — digital PDFs are processed instantly via text
extraction; scanned pages and TIFFs fall back to VLM OCR (Qwen3-VL-32B-Instruct)
automatically.

## Architecture

```
  +---------------------+     +----------------------+
  |  Extraction Server  |---->|   vLLM               |
  |  FastAPI (CPU)      |     |   Qwen3-VL-32B      |
  |  Port 8090          |     |   Port 8000 (GPU)    |
  |                     |     |                      |
  |  PDF: text extract  |     |   Text: JSON parse   |
  |  TIFF: send image   |     |   Image: VLM OCR     |
  +---------------------+     +----------------------+
```

Digital PDF path (fast): PyMuPDF text extract -> LLM parses text -> JSON

Scan / TIFF path (fallback): render page -> VLM OCR + structuring -> JSON

The extraction server is CPU-only. All GPU work happens in vLLM.

## RunAI Deployment

Full deployment guide: **[docs/runai/README.md](docs/runai/README.md)**

Follow these docs in order:

0. [Setup Data Volumes](docs/runai/setup-data-volumes.md) — download model to shared PVC, create output volume
1. [Setup & Test Workspace](docs/runai/setup-workspace.md) — experiment with pipeline in notebook, iterate on prompts/formats
2. [Deploy Streamlit App](docs/runai/deploy-streamlit.md) *(optional)* — polished demo UI, test from workspace first
3. [Deploy vLLM Server](docs/runai/deploy-vllm.md) — persistent Qwen3-VL-32B-Instruct inference endpoint
4. [Batch Processing](docs/runai/batch-processing.md) — production workspace for large-scale runs

Additional: [Troubleshooting](docs/runai/troubleshooting.md)

### PoC (5 sample docs)

0. Download model to shared PVC (Step 0)
1. Setup workspace (Step 1) — upload docs, run test notebook, launch Streamlit from workspace
2. Optionally deploy Streamlit as its own workload (Step 2)

### Production (10K+ docs/month)

0. Setup data volumes (Step 0)
1. Setup workspace (Step 1) — verify pipeline with notebook
3. Deploy vLLM as persistent endpoint (Step 3)
4. Batch processing workspace (Step 4) — `--resume` for incremental runs

## Output Formats

| Format | Use case | Output |
|--------|----------|--------|
| `award` | Grant award notices, NOAs, subaward agreements | JSON: PI, award #, amounts, dates, F&A rate |
| `budget` | Budget pages, financial summaries | JSON: categories, line items, costs |
| `terms` | Award terms, policies, compliance docs | JSON: sections, regulatory citations |
| `table` | Any tabular data | Markdown tables |
| `key_values` | Forms, labeled fields | Flat JSON key-value pairs |
| `text` | Plain text | Raw text |

## Key Files

```
ocr_app/
├── app.py                          # Streamlit UI (interactive PoC)
├── scripts/
│   ├── ocr_server.py               # FastAPI extraction server
│   └── batch_extract.py            # Batch processing script
├── notebooks/
│   └── test_extraction_pipeline.ipynb  # Step-by-step test notebook
├── deploy/
│   └── runai_jobs.yaml             # RunAI job configs
├── docs/runai/                     # RunAI deployment guides
│   ├── README.md                   #   Overview + deployment order
│   ├── setup-data-volumes.md       #   PVC + model download
│   ├── deploy-vllm.md             #   vLLM server (GPU)
│   ├── deploy-streamlit.md         #   Streamlit UI + extraction server
│   ├── setup-workspace.md          #   Setup & test workspace
│   ├── batch-processing.md         #   Production batch runs
│   └── troubleshooting.md          #   Common issues
├── requirements_server.txt         # Server deps (no GPU)
├── requirements_ui.txt             # Streamlit UI deps
└── .env.example                    # Environment variable template
```
