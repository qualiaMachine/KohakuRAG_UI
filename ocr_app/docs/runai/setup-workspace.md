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
| **Command** | `bash` |
| **Arguments** | See below |
| **Working directory** | *(leave empty)* |

### Arguments (copy-paste)

```
-c "pip install uv && rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED && curl -sL https://github.com/qualiaMachine/KohakuRAG_UI/archive/refs/heads/claude/ocr-vlm-application-hqgf2.tar.gz | tar xz -C /tmp && mv /tmp/KohakuRAG_UI-claude-ocr-vlm-application-hqgf2 /tmp/KohakuRAG_UI && cd /tmp/KohakuRAG_UI && uv pip install --system httpx pymupdf Pillow fastapi uvicorn python-multipart streamlit python-dotenv && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"
```

> **What this does:** Downloads the repo tarball, installs all
> dependencies (extraction server, Streamlit, PDF handling), and starts
> Jupyter Lab. The repo ends up at `/tmp/KohakuRAG_UI`. vLLM is already
> installed in the NGC PyTorch image.

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

Attach the shared models PVC so vLLM can load model weights:

| Data volume name | Container path |
|------------------|----------------|
| `shared-models` | `/models` |

**For PoC (5 sample docs):** That's it — upload docs directly via
Jupyter's file upload button after the workspace starts.

**For production testing:** Also attach the document and output volumes:

| Data volume name | Container path |
|------------------|----------------|
| `ocr-documents` | `/data/documents` |
| `ocr-extracted` | `/data/extracted` |

---

## Access the workspace

Once the job reaches `Running` status, click the workspace name in the
RunAI UI → click the **Jupyter** tool link. This opens Jupyter Lab in
your browser.

Open a **Terminal** from Jupyter Lab's launcher.

---

## Step 1: Start vLLM

Open a terminal in Jupyter and start the vLLM server. It loads the model
from the shared PVC:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --dtype auto \
    --max-model-len 8192 \
    --limit-mm-per-prompt image=1
```

Wait for it to print `Uvicorn running on http://0.0.0.0:8000`. This
takes 1-2 minutes (model loading + CUDA kernel compilation).

Leave this terminal running.

---

## Step 2: Upload sample docs

Use Jupyter's file upload button (up arrow icon in the file browser) to
upload your sample PDFs/TIFFs.

---

## Step 3: Open the test notebook

Navigate to `/tmp/KohakuRAG_UI/ocr_app/notebooks/test_extraction_pipeline.ipynb`
in Jupyter's file browser and open it.

The notebook walks through the full pipeline cell by cell:

1. **Connect to vLLM** — verify the local server is responding
2. **Load a sample document** — set the path to your uploaded file
3. **Check digital vs scanned** — see which pages have extractable text
4. **Digital path** — extract text with PyMuPDF, send to LLM for parsing
5. **Scanned path** — render page as image, send to VLM for OCR
6. **Inspect the JSON** — parse and validate the structured output
7. **Try different formats** — swap prompts (award, budget, key_values, etc.)

Work through the notebook iteratively — tweak prompts, try different
formats, until the JSON output has the fields you need.

---

## Step 4: Test the Streamlit app (optional)

Once the pipeline is working in the notebook, you can test the full
Streamlit experience from this same workspace.

Open a **second terminal** in Jupyter (keep vLLM running in the first):

```bash
cd /tmp/KohakuRAG_UI
OCR_SERVICE_URL=http://localhost:8090 \
  python ocr_app/scripts/ocr_server.py &
streamlit run ocr_app/app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true
```

Access the app at the workspace proxy URL:
`https://<cluster-host>/<project>/ocr-setup/proxy/8501/`

> **Note:** For this to work, your workspace needs a **Custom URL** tool
> configured for port 8501 (in addition to Jupyter on 8888). Add it when
> creating the workspace, or edit the workspace config in the RunAI UI.

---

## Next steps

Once you're satisfied with the output:
- **Deploy the Streamlit app** as its own workload (Step 2) for a persistent demo
- **Deploy vLLM** as a persistent inference endpoint (Step 3)
- **Move to batch processing** (Step 4) for production runs
- Or just keep iterating in this workspace
