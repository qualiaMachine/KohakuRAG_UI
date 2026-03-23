# Managing Models

Model weights are pre-cached on the shared PVC at
`/models/.cache/huggingface/`. The PVC is scoped to the whole cluster
(`runai/doit-ai-cluster`), so any project can access the cached models.

---

## vLLM compatibility

Not every HuggingFace model works with vLLM. Before choosing a new LLM,
check the [vLLM supported models list](https://docs.vllm.ai/en/latest/models/supported_models/).
Well-supported families include:

- **Qwen** (Qwen2, Qwen2.5, Qwen3, Qwen3.5) — first-class support,
  [official deployment guide](https://qwen.readthedocs.io/en/latest/deployment/vllm.html)
- **Llama** (Llama 2, Llama 3, Llama 3.1, Llama 4)
- **Mistral / Mixtral**
- **Gemma** (Gemma 2)
- **Phi** (Phi-3, Phi-4)

Models that use non-standard architectures or custom generation code
(e.g., some multimodal or retrieval-augmented models) may not be
supported. When in doubt, search the
[vLLM GitHub issues](https://github.com/vllm-project/vllm/issues) for
the model name.

**Quantization:** vLLM supports AWQ and GPTQ quantized models out of
the box (pass `--quantization awq` or `--quantization gptq`). FP8
quantized models (e.g., `Qwen3-8B-FP8`) work on Ada Lovelace / Hopper
GPUs natively, and on Ampere GPUs via FP8 Marlin (vLLM v0.9.0+).

---

## Adding a new model to the shared PVC

1. Contact your cluster admin — the shared PVC at `/models/` is
   **read-only** for regular workloads. New models must be added by
   someone with write access to the underlying PVC.
2. Once added, verify it's cached from any workspace:
   ```bash
   ls /models/.cache/huggingface/models--<org>--<model-name>/
   ```

## Swapping the LLM (e.g., Qwen 7B → Llama 3 8B)

1. Make sure the new model is on the PVC (see above)
2. In the RunAI UI, edit the `wattbot-vllm` job's command: change `--model`
3. Edit the `wattbot-app` job's env var: change `VLLM_MODEL` to match
4. Restart both jobs

No code changes needed. The embedding model and vector DB are unchanged.

## Swapping the embedding model

Changing the embedding model requires rebuilding the vector index:

1. Download the new model to the PVC (see above)
2. Update the index build config and re-run [Step 0e](setup-workspace.md#0e-build-the-vector-index-writes-to-ppvc)
3. Update `wattbot-embedding` env vars (`EMBEDDING_MODEL`, `EMBEDDING_DIM`)
4. Restart the embedding server

---

## Models currently on the admin's shared PVC

Based on `ls /models/.cache/huggingface/`:

| Model | Est. Size | Notes |
|-------|-----------|-------|
| `Qwen/Qwen1.5-110B-Chat` | ~220 GB | Legacy model |
| `Qwen/Qwen2.5-14B-Instruct` | ~28 GB | |
| `Qwen/Qwen2.5-72B-Instruct` | ~145 GB | |
| `Qwen/Qwen2.5-7B-Instruct` | ~14 GB | Used by vLLM server |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | ~30 GB | MoE |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | ~17 GB | MoE (active ~3B) |
| `Qwen/Qwen3-Next-80B-A3B-Thinking-FP8` | ~17 GB | MoE FP8 |
| `Qwen/Qwen3.5-35B-A3B` | ~17 GB | MoE (active ~3B) |
| `jinaai/jina-embeddings-v4` | ~3 GB | Used by embedding server. Missing `adapters/` — auto-downloaded to `/tmp` on cold start |

---

## Why the shared PVC is read-only

RunAI **Data Volumes** are read-only by design when shared across
projects. This is not a bug or misconfiguration — it's how RunAI
ensures data integrity:

> "Shared data volumes are mounted with read-only permissions. Any
> modifications must be made by writing to the **original PVC** used
> to create the data volume."
> — [RunAI Data Volumes docs](https://run-ai-docs.nvidia.com/saas/workloads-in-nvidia-run-ai/assets/data-volumes)

The lifecycle is:

1. A **Data admin** creates a **PVC data source** (writable)
2. They populate it with model weights from a Workspace
3. They wrap it in a **Data Volume** and share it across projects
4. All consumers (including your workspaces) get **read-only** access

To write to the existing `shared-models` PVC, you'd need to be the
admin who created it (or ask them to run commands on your behalf).
Alternatively, create your own PVC — see below.

### How the embedding server handles read-only PVCs

The `embedding_server.py` script creates a writable overlay
automatically — no manual workaround needed:

1. Creates a writable cache at `/tmp/hf_home`
2. Symlinks model weight directories from the PVC (read-only is fine
   for reading weights)
3. Creates writable `refs/`, `.no_exist/` directories for HF metadata
4. Redirects xet logging and pip cache to `/tmp`
5. Auto-downloads missing Jina V4 adapters to `/tmp` on cold start

**Result:** Zero "Read-only file system" errors in logs. The only cost
is ~few hundred MB adapter re-download on each cold start (until
adapters are added to the PVC permanently).

### Cache directory structure

```
/models/                              ← shared-models PVC mount (read-only)
└── .cache/
    └── huggingface/                  ← HF_HOME points here
        ├── models--jinaai--jina-embeddings-v4/
        │   ├── snapshots/
        │   │   └── <commit-hash>/    ← model weights + config
        │   │       ├── model-00001-of-00002.safetensors
        │   │       ├── config.json
        │   │       ├── tokenizer.json
        │   │       ├── adapters/     ← LoRA adapters (if present)
        │   │       └── ...
        │   └── refs/
        │       └── main              ← commit hash pointer
        ├── models--Qwen--Qwen2.5-7B-Instruct/
        │   └── snapshots/...
        └── ...
```

---

## Creating your own shared models PVC

If you want full control — to add/remove models, include adapters,
choose exactly which models to cache — create your own PVC data source
and populate it from a Workspace.

### Step A: Create a PVC data source

In the RunAI UI:

1. Go to **Data & Storage** > **Data Sources** > **New Data Source**
2. Select **PVC** as the type
3. Configure:
   - **Scope:** Your project (e.g. `runai/doit-ai-cluster/default/<your-project>`)
   - **Data source name:** `my-shared-models`
   - **PVC name:** `my-shared-models` *(creates a new PVC)*
   - **Storage class:** Use your cluster's default (ask admin if unsure)
   - **Access mode:** **Read-write by many nodes** (ReadWriteMany / RWX)
     — this is critical for sharing across workloads
   - **Storage size:** See planning notes below
   - **Container path:** `/models`
4. Create the Data Source

> **Storage size planning:** Each 7B-parameter BF16 model is ~14 GB.
> Jina V4 with adapters is ~3 GB. Budget ~20 GB per model you plan to
> cache. For example:
>
> | Models you want | Total size | Recommended PVC |
> |-----------------|------------|-----------------|
> | Jina V4 + Qwen 7B only | ~17 GB | `50Gi` |
> | + Qwen 14B | ~45 GB | `100Gi` |
> | + Qwen 72B | ~190 GB | `250Gi` |
> | All models from admin PVC | ~490 GB | `600Gi` |
>
> The existing admin PVC has ~744 GB allocated for all models listed
> above, but you likely only need a subset.
>
> **Access mode matters:** If your storage class doesn't support
> `ReadWriteMany` (RWX), use `ReadWriteOnce` (RWO) — just ensure only
> one workload mounts the PVC at a time during provisioning. After
> populating, you can share it as a read-only Data Volume (which
> supports multi-mount regardless of access mode).

### Step B: Create a provisioning Workspace

1. Go to **Workloads** > **New Workload** > **Workspace**
2. Configure:
   - **Name:** `model-provisioner`
   - **Image:** `nvcr.io/nvidia/pytorch:25.02-py3`
   - **GPU:** `0` (no GPU needed — just downloading files)
   - **CPU:** `2`, **Memory:** `8 GB`
   - **Data Sources:**
     - `my-shared-models` → mount at `/models` (**read-write**)
   - **Environment variables:**

     | Key | Value |
     |-----|-------|
     | `HF_HOME` | `/models/.cache/huggingface` |

3. Create and connect to the terminal

### Step C: Download models

```bash
# Verify /models is writable
touch /models/.write_test && rm /models/.write_test \
    && echo "Writable!" || echo "READ-ONLY — wrong mount?"

# Install tools
pip install uv
git clone -b claude/fix-shared-pvc-permissions-Mo0Qr --depth 1 \
    https://github.com/qualiaMachine/KohakuRAG_UI.git /tmp/KohakuRAG_UI

# Create cache directory structure
mkdir -p /models/.cache/huggingface

# --- Download the models you need ---

# Jina V4 embeddings — FULL model with adapters (~3 GB)
python /tmp/KohakuRAG_UI/scripts/provision_shared_models.py \
    download jinaai/jina-embeddings-v4

# Qwen 2.5 7B for vLLM (~14 GB)
python /tmp/KohakuRAG_UI/scripts/provision_shared_models.py \
    download Qwen/Qwen2.5-7B-Instruct

# Optional: larger models
# python /tmp/KohakuRAG_UI/scripts/provision_shared_models.py \
#     download Qwen/Qwen2.5-14B-Instruct
# python /tmp/KohakuRAG_UI/scripts/provision_shared_models.py \
#     download Qwen/Qwen3.5-35B-A3B

# Or download manually with huggingface-cli:
# pip install huggingface_hub
# huggingface-cli download jinaai/jina-embeddings-v4
# huggingface-cli download Qwen/Qwen2.5-7B-Instruct
```

> **Download times:** Depend on your cluster's internet bandwidth.
> Expect ~5 min for Jina V4 (~3 GB) and ~15 min for Qwen 7B (~14 GB).
> Larger models (72B) can take 1+ hours.
>
> **Gated models (Llama, Gemma):** You'll need an HF token:
> ```bash
> export HF_TOKEN="hf_your_token_here"
> python /tmp/KohakuRAG_UI/scripts/provision_shared_models.py \
>     download meta-llama/Llama-3.1-8B-Instruct --token $HF_TOKEN
> ```

### Step D: Verify downloads

```bash
# List all downloaded models with sizes
python /tmp/KohakuRAG_UI/scripts/provision_shared_models.py list

# Verify specific models have all required files
python /tmp/KohakuRAG_UI/scripts/provision_shared_models.py \
    verify jinaai/jina-embeddings-v4

python /tmp/KohakuRAG_UI/scripts/provision_shared_models.py \
    verify Qwen/Qwen2.5-7B-Instruct
```

The `verify` command checks for:
- `config.json` (model configuration)
- Weight files (`.safetensors` or `.bin`)
- Tokenizer files
- `adapters/` directory (for Jina V4 specifically)

### Step E: Share as a Data Volume (optional)

If you want other projects to access your models (not just your own
project), wrap the PVC in a Data Volume:

1. Go to **Data & Storage** > **Data Volumes** > **New Data Volume**
2. Select your `my-shared-models` PVC as the data origin
3. Name it `my-shared-models` and share with the desired scope

> **Remember:** Once shared as a Data Volume, consumers get read-only
> access. To add more models later, re-use the `model-provisioner`
> Workspace (it mounts the original PVC with write access).

### Step F: Update your inference jobs

Update your vLLM and embedding server jobs to use the new PVC:

- **Data volume:** `my-shared-models` (instead of `shared-models`)
- **Mount path:** `/models` (same as before)
- Everything else stays the same — `HF_HOME=/models/.cache/huggingface`
  and the writable overlay handle the rest

### Step G: Stop the provisioning Workspace

Once downloads are complete, **stop the Workspace** from the RunAI UI
to free resources. Re-start it whenever you need to add or update
models.

---

## Adding models to the admin's existing PVC

If you need something added to the existing `shared-models` PVC (e.g.
the missing Jina V4 adapters), ask the admin who created it to run
from their workspace:

```bash
# From the admin's workspace (with write access to /models)
export HF_HOME=/models/.cache/huggingface

# Add Jina V4 adapters specifically
huggingface-cli download jinaai/jina-embeddings-v4 \
    --include "adapters/*"

# Or use the provisioning script
pip install uv
git clone --depth 1 https://github.com/qualiaMachine/KohakuRAG_UI.git /tmp/KohakuRAG_UI
python /tmp/KohakuRAG_UI/scripts/provision_shared_models.py \
    download jinaai/jina-embeddings-v4 --include "adapters/*"
```
