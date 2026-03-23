# WattBot RAG User Interface

A Streamlit-based chat interface for the KohakuRAG pipeline, enabling interactive Q&A over the WattBot research corpus on sustainable AI.

## Project Context

This repository supports the Research Cyberinfrastructure Exploration initiative at UW-Madison. The goal is to build a long-running chatbot that answers questions about the environmental impacts of AI using a curated corpus of energy and sustainability research papers.

The project uses [KohakuRAG](https://github.com/KohakuBlueleaf/KohakuRAG), the top-ranked solution from the 2025 WattBot Challenge, as the core retrieval engine. This repository focuses on:

1. Building a user-facing Streamlit interface
2. Deploying the system on AWS using Bedrock for LLM inference
3. Comparing managed cloud deployment against self-hosted alternatives

## Quick Start

### Bedrock mode (no GPU required)

```bash
pip install -r bedrock_requirements.txt
pip install -e vendor/KohakuVault -e vendor/KohakuRAG
streamlit run app.py -- --mode bedrock
```

### Local mode (requires CUDA GPU)

```bash
pip install -r local_requirements.txt
pip install -e vendor/KohakuVault -e vendor/KohakuRAG
streamlit run app.py -- --mode local
```

### Remote mode (RunAI / distributed)

```bash
pip install -r remote_requirements.txt
pip install -e vendor/KohakuVault -e vendor/KohakuRAG
RAG_MODE=remote VLLM_BASE_URL=http://... EMBEDDING_SERVICE_URL=http://... \
  streamlit run app.py -- --mode remote
```

When both local and Bedrock backends are installed and a GPU is detected, the
sidebar shows a toggle to switch between local and Bedrock models at runtime.

See [Setup Bedrock](docs/Setup_Bedrock.md) for full AWS configuration instructions
or [RunAI Deployment](docs/runai/README.md) for cluster deployment.

## Architecture

The system follows a standard RAG (Retrieval-Augmented Generation) architecture:

```mermaid
flowchart TB
    U[User Query] --> S[Streamlit UI]
    S --> P[RAG Pipeline]
    P --> E[Embeddings]
    P --> V[(Vector Store<br/>SQLite)]
    P --> L[LLM Backend]
    L --> B[AWS Bedrock]
    L --> Local[Local HF Models]
    L --> VLLM[vLLM Server<br/>Remote]
    E --> JE[Jina V4<br/>Local GPU]
    E --> TE[Titan V2<br/>Bedrock API]
    E --> RE[Embedding Server<br/>Remote]
```

### Deployment Options

| Approach | LLM Backend | Embeddings | Launch command |
|----------|-------------|------------|----------------|
| AWS Bedrock | Managed foundation models via API | Titan V2 (API) | `streamlit run app.py -- --mode bedrock` |
| Local GPU | HuggingFace models (Qwen, Llama, etc.) | Jina V4 (local) | `streamlit run app.py -- --mode local` |
| Remote (RunAI) | vLLM server (Qwen 7B+) | FastAPI embedding server | `streamlit run app.py -- --mode remote` |

If `--mode` is omitted, the app defaults to **bedrock**.

## Documentation

- [Streamlit App Guide](docs/Streamlit_App_Guide.md) - UI features, sidebar controls, ensemble modes
- [Pipeline Architecture](docs/Pipeline_Architecture.md) - RAG pipeline technical details
- [Bedrock Setup Guide](docs/Setup_Bedrock.md) - Full AWS Bedrock setup and usage instructions
- [RunAI Deployment Guide](docs/runai/README.md) - Multi-service cluster deployment (vLLM + embedding + Streamlit)
- [PowerEdge Setup](docs/Setup_PowerEdge.md) - On-prem GPU server setup
- [Benchmarking Guide](docs/Benchmarking_Guide.md) - How to run model benchmarks
- [Meeting Notes](docs/meeting-notes.md) - Team discussions and decisions

## Repository Structure

```
.
├── app.py                                # Streamlit app (supports --mode bedrock|local|remote)
├── GETTING_STARTED.ipynb                 # Interactive setup guide (for RunAI workspaces)
├── bedrock_requirements.txt              # Torch-free Bedrock dependencies
├── local_requirements.txt                # GPU/local model dependencies
├── remote_requirements.txt               # Minimal deps for remote mode (vLLM client)
├── scripts/
│   ├── llm_bedrock.py                    # BedrockChatModel & BedrockEmbeddingModel
│   ├── embedding_server.py               # FastAPI embedding server (Jina V4)
│   ├── run_experiment.py                 # Batch experiment runner
│   ├── run_full_benchmark.py             # Multi-model benchmark orchestrator
│   └── demo_bedrock_rag.py              # Bedrock RAG demo
├── deploy/
│   ├── runai_jobs.yaml                   # RunAI job definitions (K8s manifests)
│   ├── Dockerfile.streamlit              # Streamlit container (CPU only)
│   └── Dockerfile.embedding              # Embedding server container (GPU)
├── vendor/KohakuRAG/configs/             # Pipeline & experiment configs
│   ├── hf_*.py                           # Local HuggingFace model configs
│   └── bedrock_*.py                      # AWS Bedrock model configs
├── data/embeddings/                      # Vector databases
├── docs/                                 # Documentation
│   └── runai/                            # Modular RunAI deployment guides
├── .env.example                          # Environment template
└── README.md
```

## Development Branches

- **master**: Stable releases and documentation
- **bedrock**: AWS Bedrock integration (Nils)
- **local**: Local/on-prem LLM support (Blaise)

## Related Resources

- [KohakuRAG](https://github.com/KohakuBlueleaf/KohakuRAG) - Core RAG engine
- [WattBot 2025 Competition](https://www.kaggle.com/competitions/WattBot2025/overview) - Original challenge
- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/) - Managed LLM service
- [Generative AI with Amazon Bedrock](https://www.coursera.org/learn/generative-ai-applications-amazon-bedrock) - Coursera course

## Team

| Name | Role | GitHub |
|------|------|--------|
| Chris Endemann | Research Supervisor | [@qualiaMachine](https://github.com/qualiaMachine) |
| Blaise Enuh | Local deployment | [@EnuhBlaise](https://github.com/EnuhBlaise) |
| Nils Matteson | AWS Bedrock integration | [@matteso1](https://github.com/matteso1) |

## License

Research project under UW-Madison Research Cyberinfrastructure.
