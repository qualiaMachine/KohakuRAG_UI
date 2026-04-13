# Getting Started — For New Users

Quick guide to start extracting data from documents. Assumes an admin
(Mike or equivalent) has already set up the RunAI cluster and shared
models.

## What you need before starting

| Item | Where to get it |
|------|----------------|
| **Sample PDFs** | Grant award docs, archival scans, or other institutional records |
| **Gemini reference JSONs** (optional) | From colleague, for comparing VLM output against Gemini |

> **Note for admins:** Cluster prerequisites (RunAI account, project access,
> shared-models PVC, GPU allocation, model downloads) are documented in
> [setup-data-volumes.md](runai/setup-data-volumes.md) and
> [setup-workspace.md](runai/setup-workspace.md). These must be set up
> before users can follow this guide.

## Step 1: Create the workspace

In the RunAI UI: **Workloads** > **New Workload** > **Workspace**

Follow the detailed setup in
[setup-workspace.md](runai/setup-workspace.md). The key settings:

- **Image:** `nvcr.io/nvidia/pytorch:25.02-py3`
- **GPU:** 1 device, 85% fraction (needed to load the VLM)
- **Data Volume:** attach `shared-models` at `/models`
- **Persistent Volume:** create a 1 GB volume at `/ocr` for your documents
- **Arguments:** the startup command that installs packages and launches Jupyter

Copy the arguments block exactly from the setup doc — don't modify it
unless you know what you're doing.

## Step 2: Open Jupyter and upload documents

1. Wait for the workspace to reach **Running** status
2. Click the workspace name > click the **Jupyter** tool link
3. Upload your PDFs to the `/ocr/` directory using the file upload button
4. (Optional) Create a `gemini/` folder inside `/ocr/` and upload Gemini
   reference JSONs there

## Step 3: Open and run the notebook

1. Navigate to `repo/ocr_app/notebooks/test_extraction_pipeline.ipynb`
2. Run cells **top to bottom** — each cell has a markdown explanation above it

### What the notebook does

| Section | What happens | Time |
|---------|-------------|------|
| 1. Check GPU | Verifies GPU is available and model is on the PVC | ~1s |
| 2. Load model | Loads Qwen3-VL-32B onto GPU | ~2 min |
| 3. Install deps | Installs `qwen-vl-utils` and other packages | ~30s |
| 4. Load sample doc | Lists files in `/ocr/`, picks the last one | instant |
| 5. Render pages | Converts each PDF page to an image | instant |
| 6. Test one page | Extracts structured JSON from a single page | ~30-60s |
| 7. Inspect output | Pretty-prints the JSON so you can review quality | instant |
| 8. Batch process | Processes ALL PDFs in `/ocr/` | ~1 min/page |
| 9. Compare vs Gemini | Side-by-side comparison with plots | ~5s |
| 10. Streamlit (optional) | Launches a web UI for interactive extraction (requires Custom URL tool on port 8501 + `STREAMLIT_BASE_PATH` env var — see notebook for details) | ~3 min startup |

### Switching models

In cell 2, uncomment the model you want:

```python
MODEL_NAME = "Qwen/Qwen3-VL-32B-Instruct"          # ~64 GB bf16 — default
# MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct"      # ~40 GB 4-bit via bitsandbytes
```

For the 72B model, also set `LOAD_IN_4BIT = True` (it won't fit on one
GPU otherwise). **Restart the kernel** after changing the model.

## Step 4: Review results

- **VLM output** is saved to `/ocr/vlm_output/` as `.jsonl` files
- **Comparison plots** appear inline in the notebook
- Download results via Jupyter's file browser (right-click > Download)

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "No module named 'transformers'" | The startup pip install may have failed. Run the install command manually in a terminal. See [troubleshooting.md](runai/troubleshooting.md) |
| Model loading hangs or is very slow | Check GPU memory: `!nvidia-smi`. If VRAM is mostly used, restart the kernel to free it. |
| "XSRF cookie does not match" | Clear your browser cookies for the cluster URL, or open in incognito |
| Can't find the notebook | Navigate to `repo/ocr_app/notebooks/` in Jupyter's file browser |
| GPU not available | Ask admin to check GPU allocation for your project |
| Model not found on PVC | Ask admin to run `python provision_shared_models.py download Qwen/Qwen3-VL-32B-Instruct` on the shared-models workspace |

## What to ask your admin for

- **"Add me to the RunAI project"** — you need project access to create workspaces
- **"Is `shared-models` available as a data volume in my project?"** — you need the model weights
- **"Can I get 85% of a GPU?"** — the 32B model needs ~64 GB VRAM

> **Want to try other models?** The default (Qwen3-VL-32B) is already
> downloaded. If you want to experiment with other models (larger, smaller,
> or different architectures), let Chris know — he can download additional
> models to the shared storage.
