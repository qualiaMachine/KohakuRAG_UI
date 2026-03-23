# Managing Models

Model weights live on a shared PVC mounted at `/models/.cache/huggingface/`.
If you haven't created your own PVC yet, see
**[Setup Shared Models PVC](setup-shared-models.md)** first.

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

## Adding a new model to your PVC

If you own the PVC (you created it via [Setup Shared Models](setup-shared-models.md)),
re-start your provisioning Workspace and download:

```bash
huggingface-cli download <org>/<model-name>
# e.g. huggingface-cli download Qwen/Qwen2.5-14B-Instruct
```

If you're using someone else's PVC (e.g. `shared-model-repository`), you
**cannot** add models — only the original creator has write access. Either
ask them to add it, or create your own PVC.

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

See **[Setup Shared Models PVC](setup-shared-models.md)** for the
complete guide. In short:

1. Create a PVC data source in your project (you own it, you can write to it)
2. Spin up a provisioning Workspace, download models with `huggingface-cli`
3. Optionally wrap as a Data Volume for cross-project sharing
4. Stop the Workspace — re-start anytime to add/update models
