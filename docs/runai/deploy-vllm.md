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
| **Inference name** | `wattbot-chat` |

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
API server — you only need to pass the model as a positional argument. No
command is required.

| Field | Value |
|-------|-------|
| **Command** | *(leave empty — image default launches the API server)* |
| **Arguments** | `Qwen/Qwen3-30B-A3B-GPTQ-Int4 --quantization gptq_marlin --dtype half` |
| **Working directory** | *(leave empty)* |

**Environment variables:**

| Name | Value |
|------|-------|
| `HF_HOME` | `/models/.cache/huggingface` |
| `HF_HUB_CACHE` | `/models/.cache/huggingface` |
| `HF_HUB_OFFLINE` | `1` |

> **Note:** The image defaults to `--host 0.0.0.0` and uses the
> container port from the serving endpoint config. `HF_HUB_OFFLINE=1`
> prevents downloads at runtime — the model must be pre-cached on the
> shared PVC.

## Compute resources

| Field | Value |
|-------|-------|
| **GPU devices** | `1` |
| **GPU fractioning** | Enabled — set to `80%` of device |
| **CPU request** | *(leave default)* |
| **CPU memory request** | *(leave default)* |
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
- **Model loading** (~1-2 min): vLLM loads OpenScholar 8B weights
  from the shared PVC into GPU memory (bitsandbytes quantized).
- **Engine warmup** (~30s): vLLM compiles CUDA kernels and initializes
  the KV cache.

You'll see `Initializing` in the RunAI UI during this time — this is
normal. The job transitions to `Running` once the HTTP health check
passes. Subsequent restarts (same node, cached image) take ~2-3 minutes.

## How it works

OpenScholar 8B weights are already pre-cached on the shared PVC at
`/models/.cache/huggingface/` — vLLM loads them directly on startup
(no download needed, `HF_HUB_OFFLINE=1`). The Data Volume is read-only,
so vLLM can't accidentally modify or delete weights.

> **Note:** vLLM exposes an **OpenAI-compatible** API (`/v1/chat/completions`),
> but it runs **entirely on your local GPU** — no OpenAI account or API
> charges. The `openai` Python package is just used as a client library
> to talk to your local vLLM server.

**Verify (from any other pod's terminal):**
```bash
curl http://wattbot-chat:8000/v1/models
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

**Rule of thumb:** if the model fits in GPU memory unquantized (BF16/FP16),
run it unquantized — you get better quality AND faster inference (no
dequantization overhead on every forward pass). Only quantize when the
model doesn't fit.

| Model size (BF16) | 24 GB GPU | 40 GB GPU | 80 GB GPU | 96 GB GPU |
|--------------------|-----------|-----------|-----------|-----------|
| 7–8B (~16 GB)      | Quantize  | ✅ No quant | ✅ No quant | ✅ No quant |
| 14B (~28 GB)       | Quantize  | Quantize  | ✅ No quant | ✅ No quant |
| 32B (~64 GB)       | Quantize  | Quantize  | ✅ No quant | ✅ No quant |
| 72B (~144 GB)      | 2 GPUs    | 2 GPUs    | Quantize  | Quantize  |
| 72B (2 GPUs)       | —         | ✅ No quant | ✅ No quant | ✅ No quant |

### Quantization methods (when you need to quantize)

| Method | Flag | Speed | Quality | Notes |
|--------|------|-------|---------|-------|
| **AWQ** | `--quantization awq` | ⚡ Fast | Good | Pre-quantized weights, best speed/quality tradeoff. Use `awq_marlin` kernel for extra speed. Requires a `-AWQ` model variant (e.g., `Qwen2.5-72B-Instruct-AWQ`). |
| **GPTQ** | `--quantization gptq_marlin` | ⚡ Fast | Good | Similar to AWQ. Use `gptq_marlin` kernel (plain `gptq` is buggy in vLLM). Requires a `-GPTQ` model variant. |
| **BitsAndBytes** | `--quantization bitsandbytes --load-format bitsandbytes` | 🐢 Slow | OK | Quantizes on-the-fly from any model (no special variant needed). Much slower inference than AWQ/GPTQ due to runtime dequantization. Only use when no AWQ/GPTQ variant exists. |
| **FP8** | `--quantization fp8` | ⚡⚡ Fastest | Best | 8-bit, minimal quality loss. Requires Hopper GPUs (H100) or Ada (L4/4090). |

**Decision tree:**
1. Model fits unquantized? → `--dtype auto` (no quantization flags)
2. AWQ variant available? → `--quantization awq` (or `awq_marlin`)
3. GPTQ variant available? → `--quantization gptq_marlin`
4. Neither? → `--quantization bitsandbytes --load-format bitsandbytes` (last resort)

### Multi-GPU (tensor parallelism)

If a model doesn't fit on one GPU, split it across multiple:
```
Qwen/Qwen2.5-72B-Instruct --dtype auto --tensor-parallel-size 2
```
Request 2 GPUs in the RunAI workload config. Each GPU handles half the
computation, giving ~2x inference speed vs single-GPU quantized.

## Model arguments reference

Copy-paste the **Arguments** field for each model. Remember to also update
the Streamlit job's `VLLM_MODEL` env var to match, and download the model
to the shared PVC first (see [Managing Models](managing-models.md)).

### Qwen 2.5 32B (unquantized — recommended default)

```
Qwen/Qwen2.5-32B-Instruct --dtype auto
```

> Dense 32B model in BF16 (~64 GB). Fits on a single 80/96 GB GPU
> without quantization. Best speed/quality balance for RAG with
> inline citations. No quantization overhead = faster than quantized 72B.

### Qwen 2.5 72B (unquantized, 2 GPUs)

```
Qwen/Qwen2.5-72B-Instruct --dtype auto --tensor-parallel-size 2
```

> Best quality. Requires 2 GPUs (~72 GB per GPU). Fastest 72B option
> since no quantization overhead.

### Qwen 2.5 72B (AWQ 4-bit, 1 GPU)

```
Qwen/Qwen2.5-72B-Instruct-AWQ --quantization awq --dtype auto
```

> Fits on a single GPU (~40 GB VRAM at 4-bit). Slower than unquantized
> 2-GPU but only needs 1 GPU. Use `awq_marlin` for extra speed if
> supported by your vLLM version.

### Qwen 2.5 72B (BitsAndBytes 4-bit, 1 GPU — slow)

```
Qwen/Qwen2.5-72B-Instruct --quantization bitsandbytes --load-format bitsandbytes --dtype auto
```

> Last resort for 72B on 1 GPU. Significantly slower inference than AWQ.
> Only use if you can't get the AWQ variant.

### OpenScholar 8B (unquantized — recommended)

```
OpenSciLM/Llama-3.1_OpenScholar-8B --dtype auto
```

> Only ~16 GB in BF16. No reason to quantize on any modern GPU.
> Faster and better quality than bitsandbytes-quantized.

### OpenScholar 8B (BitsAndBytes — legacy, not recommended)

```
OpenSciLM/Llama-3.1_OpenScholar-8B --quantization bitsandbytes --load-format bitsandbytes --dtype auto
```

> Only needed if running on a very small GPU (<24 GB) or sharing with
> other workloads at fractional allocation.

### Qwen 2.5 7B (unquantized)

```
Qwen/Qwen2.5-7B-Instruct --dtype auto
```

> Smallest model (~14 GB). Fast inference, fits anywhere.

### Qwen3-30B-A3B (GPTQ 4-bit, MoE)

```
Qwen/Qwen3-30B-A3B-GPTQ-Int4 --quantization gptq_marlin --dtype half
```

> Mixture-of-Experts: 30B total params, ~3B active per token. Faster
> inference than dense 32B at similar VRAM cost (~18 GB at 4-bit).
> Uses `gptq_marlin` for faster inference.

---

## CLI equivalent

If you prefer the CLI over the UI:

```bash
runai submit wattbot-chat \
  --type inference \
  --image vllm/vllm-openai:latest \
  --gpu 0.80 \
  --cpu 4 \
  --memory 16Gi \
  --pvc shared-models:/models \
  --env HF_HOME=/models/.cache/huggingface \
  --env HF_HUB_CACHE=/models/.cache/huggingface \
  --env HF_HUB_OFFLINE=1 \
  --port 8000 \
  -- Qwen/Qwen3-30B-A3B-GPTQ-Int4 \
    --quantization gptq_marlin \
    --dtype half
```
