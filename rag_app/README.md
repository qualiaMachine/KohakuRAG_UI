# WattBot RAG

Retrieval-augmented generation over research paper corpora. Streamlit chat UI backed by vLLM, Jina V4 embeddings, and optional cross-encoder reranking. 2025 WattBot Challenge winner.

## Architecture

```
  +---------------------+
  |   Streamlit App     |  (Workspace, CPU only)
  |   Port 8501         |
  +--+------+-------+---+
     |      |       | HTTP (internal cluster DNS)
     v      v       v
  +------+ +------+ +----------+
  | vLLM | |Embed | | Reranker |
  | 8000 | | 8080 | |   8082   |
  |GPU80%| |GPU10%| | GPU 10%  |
  +------+ +------+ +----------+
```

All 4 services fit on ~1 GPU via fractional allocation. Reranker is optional.

## RunAI Deployment

Full deployment guide: **[docs/runai/README.md](docs/runai/README.md)**

Follow these docs in order:

0. [Setup Shared Models PVC](docs/runai/setup-shared-models.md) *(admin, one-time)*
1. [Setup Workspace](docs/runai/setup-workspace.md) — clone repo, build vector index
2. [Deploy vLLM Server](docs/runai/deploy-vllm.md) — LLM inference with Qwen 7B
3. [Deploy Embedding Server](docs/runai/deploy-embedding.md) — Jina V4 query encoding
4. [Deploy Reranker Server](docs/runai/deploy-reranker.md) *(optional)*
5. [Deploy Streamlit App](docs/runai/deploy-streamlit.md) — browser UI

Additional: [Troubleshooting](docs/runai/troubleshooting.md) | [Managing Models](docs/runai/managing-models.md) | [Reference](docs/runai/reference.md)

## Key Files

```
rag_app/
├── app.py                          # Streamlit chat UI
├── pages/1_Corpus.py               # Corpus exploration page
├── vendor/
│   ├── KohakuRAG/                  # RAG engine
│   └── KohakuVault/                # Rust+PyO3 SQLite vector store
├── scripts/
│   ├── embedding_server.py         # FastAPI Jina V4 server
│   ├── reranker_server.py          # FastAPI cross-encoder server
│   └── add_papers.py               # Corpus management
├── deploy/
│   └── runai_jobs.yaml             # RunAI job configs
├── data/                           # Corpus, embeddings, metadata
├── docs/runai/                     # Deployment guides (10 docs)
├── requirements_local.txt          # GPU/local inference deps
└── requirements_remote.txt         # Remote client deps (minimal)
```
