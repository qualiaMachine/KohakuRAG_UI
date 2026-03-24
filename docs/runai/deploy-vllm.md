# Deploy the vLLM Server

Uses the official `vllm/vllm-openai` image — no pip-installing vLLM at
runtime.

> **Why not "Model: from Hugging Face"?** That inference type is a black
> box — crashes produce no logs and it's unclear how arguments are
> passed. It also has a **Model store** field (separate from Data &
> Storage) that expects a specially registered RunAI data source — the
> `shared-models` PVC doesn't qualify and can't be selected. While you
> can still attach `shared-models` as a regular data volume via Advanced
> setup's Data & Storage section, the empty model store likely causes
> the HF type to download the model to ephemeral storage, leading to
> crashes or silent failures. The **Custom** inference type avoids all
> of this: full logs, explicit command/arguments, and straightforward
> PVC mounting — no "model store" abstraction needed.

In the RunAI UI: **Workloads** > **New Workload** > **Inference**

## Basic settings

| Field | Value |
|-------|-------|
| **Cluster** | `doit-ai-cluster` |
| **Project** | Your project (e.g. `jupyter-endemann01`) |
| **Inference type** | **Custom** |
| **Inference name** | `wattbot-vllm` |

## Environment image

| Field | Value |
|-------|-------|
| **Image** | Custom image |
| **Image URL** | `vllm/vllm-openai:latest` |
| **Image pull** | Pull the image only if it's not already present on the host (recommended) |
| **Image pull secret** | *(leave empty — public Docker Hub image)* |

## Serving endpoint

| Field | Value |
|-------|-------|
| **Protocol** | HTTP |
| **Container port** | `8000` |

## Runtime settings

The `vllm/vllm-openai` image has a built-in entrypoint that launches the
API server — you only need to pass `--model` as an argument. No command
is required.

| Field | Value |
|-------|-------|
| **Command** | *(leave empty — image default launches the API server)* |
| **Arguments** | `--model Qwen/Qwen2.5-7B-Instruct` |
| **Environment variable** | Name: `HF_HOME`, Value: `/models/.cache/huggingface` |
| **Working directory** | *(leave empty)* |

> **Note:** The image defaults to `--host 0.0.0.0` and uses the
> container port from the serving endpoint config. If you need to
> tune memory usage, add `--max-model-len 8192` or `--dtype float16`
> to the Arguments field. For a first deploy, just `--model` is enough.

## Compute resources

| Field | Value |
|-------|-------|
| **GPU devices** | `1` (full GPU) |
| **GPU fractioning** | *(leave disabled — using full device)* |
| **CPU request** | `4` cores |
| **CPU memory request** | `16 GB` |
| **Replica autoscaling** | Min `1`, Max `1` (no autoscaling) |

## Data & storage

Under **Data & storage**, select the `shared-models` data volume and
set the container path. (In Custom inference type, data volumes appear
directly in the initial setup form — no need for Advanced setup.)

| Data volume name | Container path |
|------------------|----------------|
| `shared-models` | `/models` |

## General

| Field | Value |
|-------|-------|
| **Priority** | `very-high` (or as appropriate) |

## Expected startup time

First deploy takes **5-10 minutes**:
- **Image pull** (~2-5 min): The `vllm/vllm-openai` image is ~15 GB.
  Subsequent deploys skip this if the image is cached on the node.
- **Model loading** (~1-2 min): vLLM loads Qwen 7B weights (~14 GB)
  from the shared PVC into GPU memory.
- **Engine warmup** (~30s): vLLM compiles CUDA kernels and initializes
  the KV cache.

You'll see `Initializing` in the RunAI UI during this time — this is
normal. The job transitions to `Running` once the HTTP health check
passes. Subsequent restarts (same node, cached image) take ~2-3 minutes.

## How it works

Qwen 7B weights are already pre-cached on the shared PVC at
`/models/.cache/huggingface/` — vLLM loads them directly on startup
(no download needed). The Data Volume is read-only, so vLLM can't
accidentally modify or delete weights.

> **Note:** vLLM exposes an **OpenAI-compatible** API (`/v1/chat/completions`),
> but it runs **entirely on your local GPU** — no OpenAI account or API
> charges. The `openai` Python package is just used as a client library
> to talk to your local vLLM server.

**Verify (from any other pod's terminal):**
```bash
curl http://wattbot-vllm:8000/v1/models
```

## Switching to OpenScholar 8B

OpenScholar (`OpenSciLM/Llama-3.1_OpenScholar-8B`) is a Llama 3.1 8B
fine-tune trained for scientific literature synthesis. It's a drop-in
replacement for Qwen 7B — same VRAM footprint (~16 GB bf16, ~6 GB 4-bit).

1. Download the model to the shared PVC (see [Managing Models](managing-models.md)):
   ```bash
   python /models/provision_shared_models.py download OpenSciLM/Llama-3.1_OpenScholar-8B
   ```
2. Change the vLLM job's **Arguments** to:
   `--model OpenSciLM/Llama-3.1_OpenScholar-8B`
3. Change the Streamlit job's `VLLM_MODEL` env var to:
   `OpenSciLM/Llama-3.1_OpenScholar-8B`
4. Restart both jobs

No other changes needed — the embedding model, vector DB, and retrieval
pipeline are unchanged.

## GPU VRAM and quantization

Choose quantization based on your GPU VRAM:

| GPU VRAM | Flag | Notes |
|----------|------|-------|
| 80 GB (A100) | `--dtype bfloat16` | No quantization needed for 7B |
| 40 GB (A6000) | `--dtype bfloat16` | |
| 24 GB (L4/4090) | `--quantization awq` or `--quantization gptq` | |
| 16 GB | `--quantization awq --max-model-len 4096` | |

## CLI equivalent

If you prefer the CLI over the UI:

```bash
runai submit wattbot-vllm \
  --type inference \
  --image vllm/vllm-openai:latest \
  --gpu 1.0 \
  --cpu 4 \
  --memory 16Gi \
  --pvc shared-models:/models \
  --env HF_HOME=/models/.cache/huggingface \
  --port 8000 \
  --command -- python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 \
    --dtype auto
```
