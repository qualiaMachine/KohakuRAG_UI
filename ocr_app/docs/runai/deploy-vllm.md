# Deploy vLLM Server (`ocr-vllm`)

> **Step 1** in the [deployment guide](README.md). Comes after
> [Setup Data Volumes](setup-data-volumes.md) (Step 0).

The vLLM server handles both:
- **Text parsing** (digital PDFs) — extracted text sent as a chat message
- **VLM OCR** (scans/TIFFs) — page image sent for vision-language extraction

Qwen2.5-VL is a vision-language model, so one server handles both paths.

---

## RunAI UI Settings

| Field | Value |
|-------|-------|
| **Workload type** | Inference |
| **Inference type** | Custom |
| **Name** | `ocr-vllm` |
| **Image** | `vllm/vllm-openai:latest` |
| **Container port** | `8000` |
| **Command** | *(leave empty)* |
| **Arguments** | `--model Qwen/Qwen2.5-VL-7B-Instruct --dtype bfloat16 --max-model-len 8192 --limit-mm-per-prompt image=1` |
| **GPU** | `0.80` (fractional) |
| **CPU** | `4` |
| **Memory** | `24Gi` |
| **Data volume** | `shared-models` → `/models` |

## Environment Variables

| Variable | Value |
|----------|-------|
| `HF_HOME` | `/models/.cache/huggingface` |
| `HF_HUB_CACHE` | `/models/.cache/huggingface` |
| `HF_HUB_OFFLINE` | `1` |

---

## GPU sizing

| GPU | Arguments | Notes |
|-----|-----------|-------|
| A100 80GB | `--dtype bfloat16` | Best experience, no quantization |
| A100 40GB | `--dtype bfloat16 --max-model-len 4096` | Tight fit |
| A6000 48GB | `--dtype bfloat16` | Works well |
| L4/RTX 4090 24GB | `--quantization awq --max-model-len 4096` | Needs quantization |

---

## Verify

Wait for the pod to reach `Running` state (2-5 min for model load), then
test from any workspace on the cluster:

```bash
curl http://ocr-vllm.runai-<project>.svc.cluster.local/v1/models
# Expected: {"data": [{"id": "Qwen/Qwen2.5-VL-7B-Instruct", ...}]}
```

> **Note:** Use the FQDN (`workload.runai-project.svc.cluster.local`) on
> port 80 (no port number). Knative envoy requires this — short names
> like `ocr-vllm:8000` return 404.

If it doesn't respond, check logs: `runai logs ocr-vllm`
