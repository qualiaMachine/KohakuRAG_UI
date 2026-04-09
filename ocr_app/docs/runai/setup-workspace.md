# Setup & Test Workspace (`ocr-setup`)

> **Step 1** in the [deployment guide](README.md). Comes after
> [Setup Data Volumes](setup-data-volumes.md) (Step 0).

## What this workspace does

`ocr-setup` is your **experimentation workspace** — this is where you
iterate on the extraction pipeline before deploying anything else:

1. Start vLLM locally (loads model from shared PVC)
2. Upload sample documents
3. Walk through the test notebook cell by cell — connect to vLLM, extract
   text, render scanned pages, send to LLM, inspect JSON output
4. Experiment with different output formats and prompts
5. Test the Streamlit app from this workspace (optional)

The notebook calls vLLM running locally in this workspace — no separate
inference deployment needed at this stage. You're working directly with
the pipeline code so you can see and tweak everything.

A **test notebook** is included at
`/tmp/KohakuRAG_UI/ocr_app/notebooks/test_extraction_pipeline.ipynb` —
this is the recommended starting point.

Once you're satisfied with the output, move on to:
- **Step 2** (Streamlit app) if you want a polished demo UI
- **Step 3** (deploy vLLM) for a persistent inference endpoint
- **Step 4** (batch processing) for production runs

---

In the RunAI UI: **Workloads** > **New Workload** > **Workspace**

## Basic settings

| Field | Value |
|-------|-------|
| **Cluster** | `doit-ai-cluster` |
| **Project** | Your project |
| **Workspace name** | `ocr-setup` |

## Environment image

| Field | Value |
|-------|-------|
| **Image** | Custom image |
| **Image URL** | `nvcr.io/nvidia/pytorch:25.02-py3` |
| **Image pull** | Pull the image only if it's not already present on the host |

## Tools

Add Jupyter for browser access:

| Field | Value |
|-------|-------|
| **Tool type** | Jupyter |
| **Port** | `8888` |

## Runtime settings

| Field | Value |
|-------|-------|
| **Command** | *(leave empty)* |
| **Arguments** | See below |
| **Working directory** | *(leave empty)* |

### Arguments (copy-paste)

```
-c "pip install --no-cache-dir transformers huggingface_hub accelerate httpx pymupdf Pillow fastapi uvicorn python-multipart streamlit python-dotenv qwen-vl-utils; curl -sL https://github.com/qualiaMachine/KohakuRAG_UI/archive/refs/heads/claude/ocr-vlm-application-hqgf2.tar.gz | tar xz -C /tmp; mv /tmp/KohakuRAG_UI-claude-ocr-vlm-application-hqgf2 /tmp/KohakuRAG_UI 2>/dev/null; ln -sf /tmp/KohakuRAG_UI /ocr/repo; jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.base_url=/${RUNAI_PROJECT}/${RUNAI_JOB_NAME} --ServerApp.token='' --ServerApp.allow_origin='*' --notebook-dir=/ocr"
```

> **`--ServerApp.base_url=/${RUNAI_PROJECT}/${RUNAI_JOB_NAME}`** is
> required so Jupyter's URL matches RunAI's proxy path. Without this
> you get 404 errors. `--notebook-dir=/ocr` opens Jupyter in the
> persistent volume where your docs and notebooks live.

**Environment variables:**

| Name | Value |
|------|-------|
| `HF_HOME` | `/models/.cache/huggingface` |
| `HF_HUB_CACHE` | `/models/.cache/huggingface` |
| `HF_HUB_OFFLINE` | `1` |
| `LLM_BASE_URL` | `http://localhost:8000/v1` |
| `VLM_MODEL` | `Qwen/Qwen2.5-VL-7B-Instruct` |

> vLLM runs locally in this workspace, so `LLM_BASE_URL` points to
> `localhost`. Model weights are loaded from the shared PVC at `/models/`.

## Compute resources

| Field | Value |
|-------|-------|
| **GPU devices** | `1` |
| **GPU fractioning** | Enabled — set to `25%` of device (or more if needed) |

> **Why GPU?** The setup workspace runs the full pipeline locally —
> including vLLM for model inference. You need GPU to load and run
> Qwen2.5-VL-7B.

## Data & storage

Two storage items:

**1. Data Volume** — shared models PVC so vLLM can load model weights:

Click **+ Data Volume**:

| Data volume name | Container path |
|------------------|----------------|
| `shared-models` | `/models` |

**2. Volume** — persistent local storage for your docs, notebooks, and output:

Click **+ Volume**:

| Field | Value |
|-------|-------|
| **Storage class** | `local-path` |
| **Access mode** | *(leave default)* |
| **Claim size** | `1` GB (increase if processing many docs) |
| **Volume mode** | Filesystem |
| **Container path** | `/ocr` |
| **Volume persistency** | Persistent |

This gives you `/ocr` as a persistent directory — upload sample docs
here, save notebooks here, store extraction output here. Survives
workspace restarts.

---

## Access the workspace

Once the job reaches `Running` status, click the workspace name in the
RunAI UI → click the **Jupyter** tool link. This opens Jupyter Lab in
your browser.

## Using the workspace

1. **Upload sample docs** to `/ocr/` using Jupyter's file upload button
2. **Open the test notebook** at `repo/ocr_app/notebooks/test_extraction_pipeline.ipynb`
3. **Work through it cell by cell** — everything runs from the notebook:

| Step | What it does |
|------|-------------|
| 1 | Checks GPU and shared models PVC |
| 2 | Loads Qwen2.5-VL-7B directly with transformers |
| 3 | Lists uploaded docs, you pick one |
| 4 | Checks which pages are digital vs scanned |
| 5 | Runs extraction on a single page (auto-picks text or VLM path) |
| 6 | Displays the JSON output |
| 7 | Alternative prompts to try (award, budget, terms, key_values, text) |
| 8 | Processes all pages and saves results to `/ocr/` |
| 9 | Launches extraction server + Streamlit app for interactive testing |
| 10 | Cleanup — stops all processes |

> **Streamlit test (step 9)** requires a Custom URL tool on port 8501
> configured in the workspace (in addition to Jupyter on 8888).
