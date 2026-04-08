# Deploy the vLLM Server (`ocr-vllm`)

> **Step 1** in the [deployment guide](README.md). Comes after
> [Setup Data Volumes](setup-data-volumes.md) (Step 0).

Uses the official `vllm/vllm-openai` image. Serves Qwen2.5-VL-7B for
both text parsing (digital PDFs) and VLM OCR (scans/TIFFs) — one model
handles both paths.

> **Why Qwen2.5-VL and not a text-only model?** A text-only model
> (e.g. Qwen2.5-7B) would be faster for digital PDFs, but can't handle
> scanned pages. Qwen2.5-VL is a vision-language model — it accepts both
> text and images. One server, one model, both paths. If your corpus turns
> out to be 100% digital, you can swap to a text-only model later for
> better throughput.

In the RunAI UI: **Workloads** > **New Workload** > **Inference**

## Basic settings

| Field | Value |
|-------|-------|
| **Cluster** | `doit-ai-cluster` |
| **Project** | Your project |
| **Inference type** | **Custom** |
| **Inference name** | `ocr-vllm` |

> **Why Custom and not "Model: from Hugging Face"?** The HF inference
> type is a black box — crashes produce no logs and it's unclear how
> arguments are passed. Custom gives full control over command,
> arguments, and logs.

## Environment image

| Field | Value |
|-------|-------|
| **Image** | Custom image |
| **Image URL** | `vllm/vllm-openai:latest` |
| **Image pull** | Pull the image only if it's not already present on the host |
| **Image pull secret** | *(leave empty — public Docker Hub image)* |

## Serving endpoint

| Field | Value |
|-------|-------|
| **Protocol** | HTTP |
| **Container port** | `8000` |

## Runtime settings

The `vllm/vllm-openai` image has a built-in entrypoint that launches the
API server — you only need to pass the model and flags as arguments.

| Field | Value |
|-------|-------|
| **Command** | *(leave empty — image default launches the API server)* |
| **Arguments** | `Qwen/Qwen2.5-VL-7B-Instruct --dtype bfloat16 --max-model-len 8192 --limit-mm-per-prompt image=1` |
| **Working directory** | *(leave empty)* |

**Environment variables:**

| Name | Value |
|------|-------|
| `HF_HOME` | `/models/.cache/huggingface` |
| `HF_HUB_CACHE` | `/models/.cache/huggingface` |
| `HF_HUB_OFFLINE` | `1` |

> **`HF_HUB_OFFLINE=1`** prevents downloads at runtime — the model must
> be pre-cached on the shared PVC. If the model is missing, you get a
> clear error instead of a surprise 15 GB download.

> **`--limit-mm-per-prompt image=1`** tells vLLM to expect at most 1
> image per prompt. This optimizes memory allocation for the VLM OCR path.

## Compute resources

| Field | Value |
|-------|-------|
| **GPU devices** | `1` |
| **GPU fractioning** | Enabled — set to `80%` of device |
| **CPU request** | *(leave default)* |
| **CPU memory request** | *(leave default)* |
| **Replica autoscaling** | Min `1`, Max `1` (no autoscaling) |

## Data & storage

Under **Data & storage**, select the `shared-models` data volume.

| Data volume name | Container path |
|------------------|----------------|
| `shared-models` | `/models` |

## General

| Field | Value |
|-------|-------|
| **Priority** | `very-high` (or as appropriate) |

---

## Expected startup time

First deploy takes **5-10 minutes**:
- **Image pull** (~2-5 min): The `vllm/vllm-openai` image is ~15 GB.
  Subsequent deploys skip this if the image is cached on the node.
- **Model loading** (~1-2 min): vLLM loads Qwen2.5-VL-7B weights from
  the shared PVC into GPU memory.
- **Engine warmup** (~30s): vLLM compiles CUDA kernels and initializes
  the KV cache.

You'll see `Initializing` in the RunAI UI during this time — this is
normal. The job transitions to `Running` once the HTTP health check
passes. Subsequent restarts (same node, cached image) take ~2-3 minutes.

---

## Verify

From any other workspace terminal on the cluster:

```bash
curl http://ocr-vllm.runai-<project>.svc.cluster.local/v1/models
# Expected: {"data": [{"id": "Qwen/Qwen2.5-VL-7B-Instruct", ...}]}
```

> **FQDN required.** Use `workload.runai-project.svc.cluster.local` on
> port 80 (no port number). Knative envoy requires this — short names
> like `ocr-vllm:8000` return 404.

If it doesn't respond:
- Check the job status in RunAI UI — is it still `Initializing`?
- Check logs: click the job → **Logs** tab
- See [Troubleshooting](troubleshooting.md)

---

## GPU sizing

Adjust the **Arguments** based on your GPU:

| GPU | Arguments |
|-----|-----------|
| A100 80GB | `Qwen/Qwen2.5-VL-7B-Instruct --dtype bfloat16 --max-model-len 8192 --limit-mm-per-prompt image=1` |
| A100 40GB | `Qwen/Qwen2.5-VL-7B-Instruct --dtype bfloat16 --max-model-len 4096 --limit-mm-per-prompt image=1` |
| A6000 48GB | `Qwen/Qwen2.5-VL-7B-Instruct --dtype bfloat16 --max-model-len 8192 --limit-mm-per-prompt image=1` |
| L4/RTX 4090 24GB | `Qwen/Qwen2.5-VL-7B-Instruct --quantization awq --max-model-len 4096 --limit-mm-per-prompt image=1` |

Qwen2.5-VL-7B needs ~17 GB in bfloat16. If your GPU has less than 40 GB,
use `--quantization awq` to reduce memory at slight quality cost.

---

## CLI equivalent

```bash
runai submit ocr-vllm \
  --type inference \
  --image vllm/vllm-openai:latest \
  --gpu 0.80 \
  --cpu 4 \
  --memory 24Gi \
  --pvc shared-models:/models \
  --env HF_HOME=/models/.cache/huggingface \
  --env HF_HUB_CACHE=/models/.cache/huggingface \
  --env HF_HUB_OFFLINE=1 \
  --port 8000 \
  -- Qwen/Qwen2.5-VL-7B-Instruct \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --limit-mm-per-prompt image=1
```
