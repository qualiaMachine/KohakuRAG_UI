# Streamlit App Guide

How to run the WattBot RAG interactive app, what it does under the hood,
and ideas for deploying it on the PowerEdge via Run:ai.

---

## 1) Quick start

```bash
# From the repo root, with venv active
streamlit run app.py
```

The app opens at `http://localhost:8501` (or the next free port).
On a remote server, forward the port:

```bash
# From your laptop
ssh -L 8501:localhost:8501 user@poweredge
```

### Prerequisites

Everything in `local_requirements.txt` must be installed — the same deps
used for benchmarking. The key additions for the app are `streamlit` and
`python-dotenv` (both already listed).

```bash
uv pip install -r local_requirements.txt
```

You also need the vector database built (same as for experiments):

```bash
cd vendor/KohakuRAG
kogine run scripts/wattbot_build_index.py --config configs/jinav4/index.py
cd ../..
ls -lh data/embeddings/wattbot_jinav4.db   # verify
```

---

## 2) App overview

The app provides a chat interface backed by the full RAG pipeline
(retrieval + LLM generation). It supports two modes:

### Single model

Pick one model config from the sidebar, ask questions, get answers.
The model, embedder, and vector store are loaded once and cached across
Streamlit reruns via `st.cache_resource`.

### Ensemble

Pick 2+ models. The app queries all of them and aggregates answers using
a voting strategy. Two execution modes are selected automatically based
on available GPU memory:

| Execution mode | When chosen | How it works |
|----------------|-------------|--------------|
| **Parallel**   | All models fit in total free VRAM | All LLMs loaded into memory at once; each query runs through all models |
| **Sequential** | Not enough VRAM for all simultaneously | Loads one model at a time, queries it, unloads (gc + `cuda.empty_cache()`), loads next |

The sidebar shows detected GPU info (count, free VRAM) and which
execution mode was selected.

**Aggregation strategies:**

| Strategy           | Behavior |
|--------------------|----------|
| `majority`         | Most common answer wins (ties go to first model in list) |
| `first_non_blank`  | First model that returns a non-`is_blank` answer wins |

References are aggregated as the union across all models.

---

## 3) Architecture and key files

```
app.py                                   # Streamlit app entry point
vendor/KohakuRAG/src/kohakurag/
├── pipeline.py                          # RAGPipeline — retrieval + structured QA
├── llm.py                               # HuggingFaceLocalChatModel (4-bit, bf16, etc.)
├── embeddings.py                        # JinaV4EmbeddingModel (1024-dim)
├── datastore.py                         # KVaultNodeStore (SQLite vector DB)
└── types.py                             # ContextSnippet, RetrievalMatch, etc.
vendor/KohakuRAG/configs/hf_*.py         # Model configs (model ID, retrieval params)
data/embeddings/wattbot_jinav4.db        # Vector index (built locally, gitignored)
local_requirements.txt                   # Python dependencies
```

### Data flow per question

```
User question
  → JinaV4EmbeddingModel.embed()         # embed the query
  → KVaultNodeStore.search()              # vector search in jinav4.db
  → RAGPipeline.run_qa()                  # build prompt with retrieved context
  → HuggingFaceLocalChatModel.complete()  # local LLM inference
  → StructuredAnswer                      # parsed JSON answer
```

### What gets cached (stays in memory across questions)

| Resource | Cache key | Approx VRAM |
|----------|-----------|-------------|
| LLM (single) | `(config_name, precision)` | 2–40 GB depending on model |
| Embedder (Jina V4) | shared | ~3 GB |
| Vector store | shared | ~0 (mmap'd SQLite) |
| LLM (ensemble parallel) | `(tuple(config_names), precision)` | sum of all models |

Changing the model selection or precision in the sidebar triggers a full
reload. Changing only `top_k` or asking a new question does **not** reload.

### VRAM estimates (4-bit NF4)

These are hardcoded in `app.py:VRAM_4BIT_GB` for planning purposes:

| Config | VRAM |
|--------|------|
| hf_qwen1_5b | ~2 GB |
| hf_qwen3b | ~3 GB |
| hf_qwen7b | ~6 GB |
| hf_qwen14b | ~10 GB |
| hf_qwen32b | ~20 GB |
| hf_qwen72b | ~40 GB |
| hf_llama3_8b | ~6 GB |
| hf_gemma2_9b | ~7 GB |
| hf_gemma2_27b | ~17 GB |
| hf_mixtral_8x7b | ~26 GB |
| hf_mistral7b | ~6 GB |
| hf_phi3_mini | ~3 GB |

For bf16, multiply by ~4x. The app accounts for this when planning
parallel vs sequential ensemble execution.

---

## 4) Configuration

The app reads the same `hf_*.py` config files used by the benchmark
scripts. Config files define model identity only — precision is selected
in the sidebar at runtime.

Key config fields consumed by the app:

| Field | Purpose |
|-------|---------|
| `hf_model_id` | HuggingFace model identifier |
| `hf_max_new_tokens` | Max generation length (default 512) |
| `hf_temperature` | Sampling temperature (default 0.2) |
| `max_concurrent` | Max concurrent inference requests |
| `db` | Path to vector database |
| `table_prefix` | Vector table prefix |
| `embedding_dim` | Embedding dimension (1024 for jinav4) |
| `embedding_task` | Embedding task type ("retrieval") |

To add a new model to the app, just create a new `hf_*.py` config — the
app auto-discovers all `hf_*.py` files in `vendor/KohakuRAG/configs/`.

---

## 5) Gated models

Some models (Llama 3.1, Gemma 2) are gated on HuggingFace and require:

1. Accept the model license on the model's HuggingFace page
2. Set your HuggingFace token:

```bash
export HF_TOKEN="hf_your_token_here"
streamlit run app.py
```

If you see a 401 error when selecting a gated model, this is why.

---

## 6) Future: deploying on PowerEdge with Run:ai

Right now the app runs interactively — you SSH in, launch `streamlit run
app.py`, and it holds GPU(s) for the entire session. This wastes
resources when nobody is actively asking questions.

Below are notes on how we could deploy this more efficiently using
Run:ai on the PowerEdge, so that GPU resources are only consumed when
someone is actually using the app.

### Option A: Persistent interactive workspace (simplest)

Run:ai supports **interactive workspaces** that can be started/stopped
on demand. This is the lowest-effort approach:

1. Create a Run:ai interactive workspace with GPU allocation
2. Install deps and clone the repo (or mount a shared PVC)
3. Run `streamlit run app.py` inside the workspace
4. Stop the workspace when not in use (frees GPU)

**Pros:** Simple, no Docker needed, matches current workflow.
**Cons:** Manual start/stop, GPU allocated even when idle within a session.

### Option B: Docker container + Run:ai job (recommended)

Package the app as a Docker container. Submit it as a Run:ai inference
workload that can be stopped/started or scheduled.

**Dockerfile sketch:**

```dockerfile
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Python deps
COPY local_requirements.txt .
RUN pip install --no-cache-dir -r local_requirements.txt

# App code
COPY . .

# Pre-build the vector index (or mount from PVC)
# RUN cd vendor/KohakuRAG && kogine run scripts/wattbot_build_index.py --config configs/jinav4/index.py

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Run:ai submission:**

```bash
# Submit as an interactive workload with 1 GPU
runai submit wattbot-rag \
    --image registry.example.com/wattbot-rag:latest \
    --gpu 1 \
    --interactive \
    --service-type nodeport \
    --port 8501:8501 \
    --pvc data-pvc:/app/data \
    --pvc models-pvc:/root/.cache/huggingface
```

Key considerations:
- **Model cache:** Mount the HuggingFace cache as a PVC
  (`/root/.cache/huggingface`) so models aren't re-downloaded on every
  container start. This is the biggest time savings — model downloads
  are 5–80 GB per model.
- **Vector DB:** Either bake the index into the image or mount `data/`
  as a PVC. A PVC is more flexible (can rebuild index without
  rebuilding the image).
- **GPU sizing:** For single model (7B 4-bit), 1 GPU with 24+ GB is
  enough. For ensemble, request more based on the VRAM table above.

**Pros:** Reproducible, version-controlled environment, easy to
start/stop via Run:ai CLI or dashboard.
**Cons:** Need to maintain a container image and registry.

### Option C: Scale-to-zero with a proxy (most efficient)

The holy grail: GPU pods spin up only when someone sends a request, and
spin back down after a timeout. This requires a lightweight proxy that:

1. Receives incoming HTTP requests on port 8501
2. If no GPU pod is running, submits a Run:ai job and waits for it
3. Forwards the request to the running pod
4. After N minutes of inactivity, deletes the pod (frees GPU)

This is essentially a serverless GPU pattern. Some approaches:

- **KNative + Run:ai:** KNative Serving supports scale-to-zero. If
  Run:ai is on a K8s cluster, you could wrap the Streamlit container
  as a KNative service. Requires KNative to be installed on the cluster.
- **Custom proxy script:** A simple Python/Go proxy that calls
  `runai submit` on first request and `runai delete` after idle timeout.
  Simpler than KNative but more manual.
- **Run:ai scheduler policies:** Run:ai supports idle GPU reclamation
  and workload preemption. You can set a policy that reclaims GPU from
  idle interactive workloads after a configurable timeout.

**Pros:** Zero GPU waste when nobody is using the app.
**Cons:** Cold start latency (model loading takes 5–90s depending on
model size). Users see a loading screen on first query after idle.

### Reducing cold start time

Regardless of which option you pick, cold start (model loading) is the
main UX bottleneck. Ways to reduce it:

| Technique | Impact | Effort |
|-----------|--------|--------|
| PVC-mounted HF cache | Skips download, load from local disk | Low |
| Smaller default model (qwen3b) | ~3s load vs ~90s for 72B | None |
| Pre-warmed standby pod (always 1 running) | Zero cold start | Wastes 1 GPU |
| Model sharding across NVMe + GPU | Faster initial load | Medium |
| `safetensors` format (already default) | ~2x faster than .bin | Already done |

### Recommended starting point

Start with **Option B** (Docker + Run:ai job). It gives you:
- Reproducible environment
- Easy start/stop from Run:ai dashboard
- Shared model cache via PVC (fast restarts)
- Path to Option C later if utilization matters

The Streamlit app already handles all the GPU detection and
parallel/sequential logic, so no code changes are needed — just package
and deploy.
