# WattBot RAG

Retrieval-augmented generation over research paper corpora. Streamlit chat UI backed by vLLM, Jina V4 embeddings, and optional cross-encoder reranking. 2025 WattBot Challenge winner.

## Architecture

```
┌─────────────────────┐
│   Streamlit App     │  (Workspace, CPU only)
│   Port 8501         │
└──┬──────┬───────┬───┘
   │      │       │ HTTP (internal cluster DNS)
   ▼      ▼       ▼
┌──────┐ ┌──────┐ ┌──────────┐
│ vLLM │ │Embed │ │ Reranker │
│ 8000 │ │ 8080 │ │   8082   │
│GPU80%│ │GPU10%│ │ GPU 10%  │
└──────┘ └──────┘ └──────────┘
```

All 4 services fit on ~1 GPU via fractional allocation. Reranker is optional.

## Quick Start

```bash
# 1. Start vLLM (GPU)
vllm serve Qwen/Qwen2.5-7B-Instruct --dtype bfloat16

# 2. Start embedding server (GPU)
python rag_app/scripts/embedding_server.py

# 3. Start Streamlit UI (CPU)
pip install -r rag_app/requirements_remote.txt
RAG_MODE=remote streamlit run rag_app/app.py
```

## RunAI Deployment

See `rag_app/deploy/runai_jobs.yaml` for complete RunAI job definitions, or follow the step-by-step guides below.

## Documentation Index

### Setup & Deployment

| Doc | Description |
|-----|-------------|
| [RunAI Overview](docs/runai/README.md) | Architecture overview for RunAI deployment |
| [Setup Shared Models](docs/runai/setup-shared-models.md) | Download models to shared PVC |
| [Setup Workspace](docs/runai/setup-workspace.md) | Initialize a RunAI workspace |
| [Deploy vLLM](docs/runai/deploy-vllm.md) | Deploy the LLM inference server |
| [Deploy Embedding](docs/runai/deploy-embedding.md) | Deploy the Jina V4 embedding server |
| [Deploy Reranker](docs/runai/deploy-reranker.md) | Deploy the cross-encoder reranker (optional) |
| [Deploy Streamlit](docs/runai/deploy-streamlit.md) | Deploy the Streamlit UI workspace |
| [Setup PowerEdge](docs/Setup_PowerEdge.md) | On-premises PowerEdge setup (non-RunAI) |

### Architecture & Usage

| Doc | Description |
|-----|-------------|
| [Pipeline Architecture](docs/Pipeline_Architecture.md) | RAG pipeline technical details |
| [Streamlit App Guide](docs/Streamlit_App_Guide.md) | UI features, sidebar controls, modes |
| [OpenScholar Integration](docs/Explore_OpenScholar_Integration.md) | Science-tuned LLM integration |

### Benchmarks

| Doc | Description |
|-----|-------------|
| [Benchmark Report](docs/Benchmark_Report.md) | Results across models and hardware |
| [Benchmarking Guide](docs/Benchmarking_Guide.md) | How to run your own benchmarks |

### Troubleshooting

| Doc | Description |
|-----|-------------|
| [RunAI Troubleshooting](docs/runai/troubleshooting.md) | Common RunAI deployment issues |
| [RunAI Reference](docs/runai/reference.md) | Architecture rationale, data sharing |
| [Dependency Fixes](docs/Dep_fixes.md) | Known dependency issues and fixes |

## Key Files

```
rag_app/
├── app.py                          # Streamlit chat UI (3000+ lines)
├── pages/1_Corpus.py               # Corpus exploration page
├── vendor/
│   ├── KohakuRAG/                  # RAG engine (pipeline, embeddings, LLM, vision)
│   └── KohakuVault/                # Rust+PyO3 SQLite KV store with vectors
├── scripts/
│   ├── embedding_server.py         # FastAPI Jina V4 server
│   ├── reranker_server.py          # FastAPI cross-encoder server
│   ├── add_papers.py               # Corpus management
│   ├── run_app.sh / start_app.sh   # Launch helpers
├── deploy/
│   └── runai_jobs.yaml             # RunAI deployment configs
├── data/
│   ├── corpus/                     # Document chunks
│   ├── embeddings/                 # Vector databases
│   └── metadata.csv                # Paper metadata
├── artifacts/plots/                # PowerEdge benchmark plots
├── requirements_local.txt          # GPU/local inference deps
├── requirements_remote.txt         # Remote client deps (minimal)
└── docs/                           # App-specific documentation
```
