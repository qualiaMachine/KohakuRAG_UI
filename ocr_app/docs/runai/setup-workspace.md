# Setup & Test Workspace (`ocr-setup`)

> **Step 1** in the [deployment guide](README.md). Comes after
> [Setup Data Volumes](setup-data-volumes.md) (Step 0).

## What this workspace does

`ocr-setup` is your **experimentation workspace** — this is where you
iterate on the extraction pipeline before deploying anything else:

1. Upload sample documents and inspect them
2. Walk through the pipeline step-by-step in a notebook (text extraction,
   page rendering, LLM calls, JSON output)
3. Experiment with different output formats and prompts
4. Validate that the JSON output has the fields you need
5. Run the batch script on your sample docs once you're happy

The notebook calls vLLM directly — no extraction server or Streamlit app
needed at this stage. You're working directly with the pipeline code so
you can see and tweak everything.

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
| **CPU request** | `4` |
| **CPU memory request** | `8Gi` |

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

## Verification checklist

### 1. Start vLLM locally

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

### 2. Verify vLLM is responding

Open a **second terminal** and test:

```bash
curl http://localhost:8000/v1/models
```

Expected output:
```json
{"data": [{"id": "Qwen/Qwen2.5-VL-7B-Instruct", ...}]}
```

### 3. Upload sample docs (PoC only)

Use Jupyter's file upload button (up arrow icon in the file browser) to
upload your sample PDFs. They'll land in `/home/jovyan/` or wherever
Jupyter's file browser is pointed.

Create a directory for them:

```bash
mkdir -p /home/jovyan/sample_docs
# Move uploaded files there
mv /home/jovyan/*.pdf /home/jovyan/sample_docs/
ls /home/jovyan/sample_docs/
```

### 4. Test extraction on a single document

```bash
cd /tmp/KohakuRAG_UI

# Run on your sample docs
python ocr_app/scripts/batch_extract.py \
    --input-dir /home/jovyan/sample_docs \
    --output-dir /home/jovyan/extracted \
    --format award \
    --concurrency 1
```

Expected output:
```
[batch] Found 5 files, 0 already completed, 5 to process
[batch] LLM: Qwen/Qwen2.5-VL-7B-Instruct at http://qwen2-5--vl--7b--instruct.../v1
[1/5] OK doc1.pdf (3p, 0d/3s, 14.2s) -> doc1.json
[2/5] OK doc2.pdf (2p, 0d/2s, 9.8s) -> doc2.json
...
```

The `0d/3s` means 0 digital pages, 3 scanned — confirming VLM OCR is
being used for your scanned PDFs.

### 5. Inspect the output

```bash
# Pretty-print the first result
cat /home/jovyan/extracted/*.json | python -m json.tool | head -80
```

Or in a Jupyter notebook cell:

```python
import json
from pathlib import Path

for out in sorted(Path("/home/jovyan/extracted").glob("*.json")):
    data = json.loads(out.read_text())
    print(f"\n{'='*60}")
    print(f"File: {data['source_file']}")
    print(f"Pages: {data['total_pages']} ({data['digital_pages']}d/{data['scanned_pages']}s)")
    for page in data['pages']:
        print(f"  Page {page['page']}: {page['method']} ({page['elapsed_ms']:.0f}ms)")
        # Show first 300 chars of extracted text
        print(f"    {page['text'][:300]}...")
```

### 6. Check if the right fields are being extracted

If the output JSON doesn't have the fields you need, try a different
format:

```bash
# More flexible — extracts any key-value pairs it finds
python ocr_app/scripts/batch_extract.py \
    --input-dir /home/jovyan/sample_docs \
    --output-dir /home/jovyan/extracted_kv \
    --format key_values \
    --concurrency 1

# Raw text — see exactly what the VLM reads from the image
python ocr_app/scripts/batch_extract.py \
    --input-dir /home/jovyan/sample_docs \
    --output-dir /home/jovyan/extracted_text \
    --format text \
    --concurrency 1
```

### 7. Test the Streamlit app (optional)

You can run both the extraction server and Streamlit UI directly from
this workspace to test the full interactive experience before deploying
them as separate workloads.

Open **two terminals** in Jupyter:

**Terminal 1 — start the extraction server:**

```bash
cd /tmp/KohakuRAG_UI
pip install fastapi uvicorn python-multipart
python ocr_app/scripts/ocr_server.py
```

**Terminal 2 — start Streamlit:**

```bash
cd /tmp/KohakuRAG_UI
pip install streamlit python-dotenv
OCR_SERVICE_URL=http://localhost:8090 \
  streamlit run ocr_app/app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true
```

Access the app at the workspace proxy URL:
`https://<cluster-host>/<project>/ocr-setup/proxy/8501/`

> **Note:** For this to work, your workspace needs a **Custom URL** tool
> configured for port 8501 (in addition to Jupyter on 8888). If you didn't
> set that up when creating the workspace, you can add it by editing the
> workspace config in the RunAI UI, or just use the notebook approach
> instead.

### 8. Done — stop or keep iterating

Once you're satisfied with the output, either:
- **Stop the workspace** — restart later to test new formats or doc types
- **Deploy the Streamlit app** as a proper workload (Step 3) for a
  persistent demo
- **Move to batch processing** (Step 4) for production runs
