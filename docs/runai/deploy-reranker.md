# Deploy the Reranker Server

The reranker server is a custom FastAPI service that wraps a
cross-encoder model (e.g. `BAAI/bge-reranker-v2-m3` or
`OpenSciLM/OpenScholar_Reranker`). It follows the same pattern as the
embedding server — same base image, same PVC overlay, same startup
approach.

> **Why a separate inference job?** Cross-encoder models need GPU for
> inference. Running one inside the Streamlit app (CPU-only workspace)
> would be too slow. A separate inference job lets the reranker run on
> its own GPU fraction while the Streamlit app calls it over HTTP.

In the RunAI UI: **Workloads** > **New Workload** > **Inference**

## Basic settings

| Field | Value |
|-------|-------|
| **Cluster** | `doit-ai-cluster` |
| **Project** | Your project (e.g. `jupyter-endemann01`) |
| **Inference type** | **Custom** (not "Model: from Hugging Face") |
| **Inference name** | `wattbot-reranker` |

## Environment image

| Field | Value |
|-------|-------|
| **Image** | Custom image |
| **Image URL** | `vllm/vllm-openai:latest` |
| **Image pull** | Pull only if not present (recommended) |

## Serving endpoint

| Field | Value |
|-------|-------|
| **Protocol** | HTTP |
| **Container port** | `8082` |

## Runtime settings

| Field | Value |
|-------|-------|
| **Command** | `bash` |
| **Arguments** | `-c "pip install uv && curl -sL https://github.com/qualiaMachine/KohakuRAG_UI/archive/refs/heads/master.tar.gz | tar xz -C /tmp && mv /tmp/KohakuRAG_UI-master /tmp/KohakuRAG_UI && cd /tmp/KohakuRAG_UI && uv pip install --system fastapi uvicorn httpx numpy sentence-transformers 'transformers>=4.42,<5' accelerate && python3 scripts/reranker_server.py"` |
| **Working directory** | *(leave empty)* |

> **Using a different branch?** Replace `master` in the URL and the `mv`
> target with your branch name. If the branch has slashes (e.g.
> `claude/my-feature`), use the full name in the URL but replace slashes
> with dashes in the `mv` target (GitHub converts `/` to `-` in tarball
> directory names).

**Environment variables:**

| Name | Value |
|------|-------|
| `HF_HOME` | `/models/.cache/huggingface` |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` |

> **Model options:**
> - `BAAI/bge-reranker-v2-m3` — small, fast, multilingual (~0.5 GB VRAM). Good default.
> - `BAAI/bge-reranker-large` — better quality (~1.3 GB VRAM)
> - `OpenSciLM/OpenScholar_Reranker` — science-tuned, best for research papers (~1.2 GB VRAM)
>
> You must pre-download the model to the shared PVC before deploying.
> See [Pre-downloading the model](#pre-downloading-the-model) below.

## Compute resources

| Field | Value |
|-------|-------|
| **GPU devices** | `1` |
| **GPU fractioning** | Enabled — set to `25%` of device (reranker models are small, ~0.5–1.3 GB VRAM) |
| **CPU request** | `2` cores |
| **CPU memory request** | `4 GB` |
| **Replica autoscaling** | Min `1`, Max `1` (no autoscaling) |

## Data & storage

| Data volume name | Container path |
|------------------|----------------|
| `shared-models` | `/models` |

> The reranker does not need `wattbot-data` — it only scores
> query–passage pairs, it doesn't read the vector index.

## General

| Field | Value |
|-------|-------|
| **Priority** | `very-high` (or as appropriate) |

## Expected startup time

**2-3 minutes** on first deploy:
- **Image pull**: Instant if vLLM image already cached on node
- **Dependency install** (~30s): Same packages as embedding server
- **Model loading** (~10-30s): BGE reranker is small (~0.5 GB)

---

## Pre-downloading the model

The reranker model must be on the shared PVC before deployment (the
server runs in offline mode). From your workspace terminal:

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('BAAI/bge-reranker-v2-m3', cache_dir='/models/.cache/huggingface')
"
```

For the science-tuned model:
```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('OpenSciLM/OpenScholar_Reranker', cache_dir='/models/.cache/huggingface')
"
```

---

## Pre-deploy testing

### Step 1: Test the startup command locally

```bash
rm -rf /tmp/KohakuRAG_UI && \
curl -sL https://github.com/qualiaMachine/KohakuRAG_UI/archive/refs/heads/<your-branch>.tar.gz \
  | tar xz -C /tmp && \
mv /tmp/KohakuRAG_UI-<your-branch> /tmp/KohakuRAG_UI && \
cd /tmp/KohakuRAG_UI && \
uv pip install --system fastapi uvicorn httpx numpy \
  sentence-transformers 'transformers>=4.42,<5' accelerate && \
python3 scripts/reranker_server.py
```

Expected: `[reranker_server] Model loaded in Xs. Serving on 0.0.0.0:8082`

### Step 2: Smoke-test the /health endpoint

```bash
curl http://localhost:8082/health
# Expected: {"status":"ok"}
```

### Step 3: Test an actual rerank call

```bash
curl -X POST http://localhost:8082/rerank \
  -H "Content-Type: application/json" \
  -d '{"query": "energy consumption of LLM training", "texts": ["Large language models require significant compute.", "The weather is nice today."]}'
# Expected: {"scores":[0.98, 0.01], "count":2, "elapsed_ms":...}
```

---

## Connecting the Streamlit app

Add one more environment variable to your Streamlit workspace:

| Name | Value |
|------|-------|
| `RERANKER_SERVICE_URL` | `http://wattbot-reranker.runai-<project>.svc.cluster.local` |

> **Important:** Use the full FQDN (no port number). Inference workloads
> use Knative/envoy which listens on port 80. Short names like
> `wattbot-reranker:8082` will return envoy 404.

Once set, the "Cross-encoder reranker" toggle will appear in the
Streamlit sidebar under "Retrieval enhancements". It works with or
without Semantic Scholar search enabled:
- **Reranker only**: Reranks local retrieval matches for better ordering
- **Reranker + S2**: Jointly reranks local + Semantic Scholar results so
  they compete fairly on relevance

---

## CLI equivalent

```bash
runai submit wattbot-reranker \
  --type inference \
  --image vllm/vllm-openai:latest \
  --gpu 0.25 \
  --cpu 2 \
  --memory 4Gi \
  --pvc shared-models:/models \
  --env HF_HOME=/models/.cache/huggingface \
  --env RERANKER_MODEL=BAAI/bge-reranker-v2-m3 \
  --port 8082 \
  --command -- /bin/bash -c \
    "pip install uv && \
     curl -sL https://github.com/qualiaMachine/KohakuRAG_UI/archive/refs/heads/master.tar.gz \
       | tar xz -C /tmp && \
     mv /tmp/KohakuRAG_UI-master /tmp/KohakuRAG_UI && \
     cd /tmp/KohakuRAG_UI && \
     uv pip install --system fastapi uvicorn httpx numpy \
       sentence-transformers 'transformers>=4.42,<5' accelerate && \
     python3 scripts/reranker_server.py"
```

## Updated architecture

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
  │GPU 1 │ │GPU½  │ │  GPU¼    │
  └──────┘ └──────┘ └──────────┘
     Total: 1.75 GPUs
```
