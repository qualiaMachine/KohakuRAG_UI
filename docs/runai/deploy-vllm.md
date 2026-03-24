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

## GPU VRAM and quantization

Choose quantization based on your GPU VRAM:

| GPU VRAM | Flag | Notes |
|----------|------|-------|
| 80 GB (A100) | `--dtype bfloat16` | No quantization needed for 7B |
| 40 GB (A6000) | `--dtype bfloat16` | |
| 24 GB (L4/4090) | `--quantization awq` or `--quantization gptq` | |
| 16 GB | `--quantization awq --max-model-len 4096` | |

## Deploying larger models (14B, 72B)

A model's bf16 weight size is roughly **2 × parameters (in billions) GB**.
Qwen 72B at bf16 needs ~144 GB — it will **not** fit on a single A100 (80 GB).

### Option A: AWQ 4-bit quantization (recommended, 1 GPU)

Use a pre-quantized AWQ model. At 4-bit, Qwen 72B needs ~40 GB — fits on
one A100 with room for KV cache.

1. Download the quantized model to the PVC:
   ```bash
   python provision_shared_models.py download Qwen/Qwen2.5-72B-Instruct-AWQ
   ```
2. Update the vLLM job arguments:
   ```
   --model Qwen/Qwen2.5-72B-Instruct-AWQ --quantization awq --max-model-len 8192
   ```
3. Update `wattbot-app` env var: `VLLM_MODEL=Qwen/Qwen2.5-72B-Instruct-AWQ`
4. Increase memory request to `32 GB` (quantization still needs CPU RAM for loading)

### Option B: Tensor parallelism (2 GPUs, full precision)

Split the model across multiple GPUs. Preserves full bf16 precision.

1. Change GPU request to `2`
2. Update the vLLM job arguments:
   ```
   --model Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size 2 --dtype bfloat16
   ```
3. Update `wattbot-app` env var: `VLLM_MODEL=Qwen/Qwen2.5-72B-Instruct`
4. Increase CPU to `8` cores and memory to `32 GB`

> **Note:** The full-precision `Qwen/Qwen2.5-72B-Instruct` model is
> already on the shared PVC (~135 GB). The AWQ variant is **not** — you
> must download it first (see Option A step 1).

### Quick reference: Qwen model resource requirements

| Model | bf16 size | 4-bit AWQ size | Min GPUs (bf16) | Min GPUs (AWQ) |
|-------|-----------|----------------|-----------------|----------------|
| Qwen2.5-7B | ~14 GB | ~6 GB | 1× A100 | 1× any 24GB+ |
| Qwen2.5-14B | ~28 GB | ~10 GB | 1× A100 | 1× any 24GB+ |
| Qwen2.5-72B | ~144 GB | ~40 GB | 2× A100 | 1× A100 |

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
