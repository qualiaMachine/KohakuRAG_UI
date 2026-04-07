# RunAI / PowerEdge Applications

A collection of GPU-accelerated applications deployed on RunAI (Kubernetes) with Dell PowerEdge infrastructure. Each app follows the same deployment pattern: lightweight CPU containers for UI/orchestration, with GPU work offloaded to vLLM or Ollama.

## Applications

### [Document Extraction (`ocr_app/`)](ocr_app/README.md)

Structured data extraction from grant award notices, budgets, terms & conditions, and other research admin documents. Hybrid pipeline: digital PDFs are processed via text extraction + LLM parsing (instant, no GPU); scanned pages and TIFFs fall back to VLM OCR (Qwen2.5-VL).

### [RAG Chat (`rag_app/`)](rag_app/README.md)

WattBot RAG — retrieval-augmented generation over research paper corpora. Streamlit chat UI backed by vLLM, Jina V4 embeddings, and optional cross-encoder reranking. Supports multiple knowledge bases and fractional GPU allocation across 4 services on a single GPU.

## Shared Infrastructure

### `docs/runai/`
RunAI deployment guides — architecture overview, vLLM deployment, embedding/reranker servers, Streamlit workspaces, shared model PVC setup, troubleshooting.

### `scripts/`
Shared infrastructure utilities:
- `hardware_metrics.py` — GPU/energy profiling (VRAM, power, energy per request)
- `provision_shared_models.py` — Download models to shared PVC
- `setup_poweredge_pod.sh` — PowerEdge pod initialization

## Deployment Pattern

All apps follow the same RunAI deployment pattern:

```
┌──────────────────┐
│   Streamlit UI   │  Workspace (CPU, port 8501)
│   or FastAPI     │  App code pulled at runtime via curl|tar
└────────┬─────────┘
         │ HTTP (internal cluster DNS)
         ▼
┌──────────────────┐
│  vLLM / Ollama   │  Inference workload (GPU, fractional)
│  Model serving   │  Models loaded from shared PVC
└──────────────────┘
```

- **No Docker builds required** for app code — base images provide the runtime environment (Python, CUDA, deps), app code is pulled from GitHub at container start
- **Fractional GPU allocation** — multiple services share a single GPU
- **Shared model PVC** — models downloaded once, mounted read-only across jobs
- **FQDN for service URLs** — `workload.runai-project.svc.cluster.local` (Knative requirement)

## Quick Start

1. Set up a shared model PVC: see `docs/runai/setup-shared-models.md`
2. Pick an app and follow its README
3. Deploy via RunAI UI or CLI using the configs in each app's `deploy/` directory
