# Deploying WattBot RAG on RunAI

Production deployment using 2 RunAI **Inference** workloads + 1
**Workspace** across 1.5 GPUs (~90 GB), with vLLM for high-throughput
LLM serving.

| Workload | Type | What it does | GPU | Port |
|----------|------|-------------|-----|------|
| **`wattbot-vllm`** | Inference | Serves the LLM (Qwen 7B) via vLLM's OpenAI-compatible API | 1.0 | 8000 |
| **`wattbot-embedding`** | Inference | Encodes user questions into vectors (Jina V4) for DB lookup | 0.5 | 8080 |
| **`wattbot-app`** | Workspace | Streamlit UI — connects to the other two via HTTP | 0 | 8501 |

The GPU services use Inference workloads (always-on, autoscalable). The
Streamlit UI uses a Workspace because Workspaces provide browser-accessible
proxy URLs, while Inference workloads on most clusters only expose internal
Knative routes.

All three mount the shared model repository at `/models/` (read-only)
and share one physical GPU via RunAI's fractional allocation. A one-time
setup Workspace (Step 0) uses your personal workspace at
`/home/jovyan/work/` (writable) to clone the repo, install
dependencies, and build the vector index. Model weights (Qwen, Jina V4)
are already pre-cached on the shared PVC — no downloads needed.

```
  Users (browser)
       │
       ▼
┌─────────────────────┐
│   Streamlit App     │  CPU only, no GPU
│   Port 8501         │
└────────┬────────────┘
         │ HTTP (internal cluster DNS)
   ┌─────┴──────┐
   ▼            ▼
┌──────────┐  ┌──────────────────────────────┐
│  vLLM    │  │  Embedding Server            │
│  Server  │  │  Encodes user questions into │
│  Port    │  │  vectors for DB lookup       │
│  8000    │  │  Port 8080, GPU ~0.5         │
│  GPU     │  └──────────────────────────────┘
│  ~1.0    │
└──────────┘
```

**Query flow:** User asks a question → Streamlit [`wattbot-app`] sends
it to the Embedding Server [`wattbot-embedding`] → gets a vector back →
searches the pre-built vector DB → sends question + retrieved context to
vLLM [`wattbot-vllm`] → Streamlit [`wattbot-app`] displays the answer
with citations.

---

## Deployment Guide

Follow these docs in order:

1. **[Setup & Prerequisites](setup-workspace.md)** — Create the shared data volume, clone the repo, build the vector index (one-time)
2. **[Deploy vLLM Server](deploy-vllm.md)** — LLM inference with Qwen 7B
3. **[Deploy Embedding Server](deploy-embedding.md)** — Jina V4 query encoding
4. **[Deploy Streamlit App](deploy-streamlit.md)** — Browser UI connecting to both services

All steps use the **RunAI web UI only** — no CLI tools required.

### Deployment Order

1. **Setup** — Workspace: clone repo, install deps, build index, then stop
2. **vLLM** — loads Qwen from shared cache (~30s)
3. **Embedding server** — loads Jina V4 from shared cache (~30s)
4. **Streamlit app** — last, needs both services running

GPU budget: **1.5 GPUs** (~90 GB) total — 1.0 for vLLM, 0.5 for
embeddings, 0 for Streamlit. All model weights are pre-cached on
`/models/`. Restarts are fast.

---

## Additional Docs

- **[Troubleshooting](troubleshooting.md)** — Common errors and fixes
- **[Managing Models](managing-models.md)** — Adding, swapping, and provisioning models on the shared PVC
- **[Reference](reference.md)** — Architecture rationale, data sharing, access control, local development
