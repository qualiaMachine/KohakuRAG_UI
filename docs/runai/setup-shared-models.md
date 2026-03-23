# Setup Shared Models PVC (Admin / Owner)

> **Who needs this:** The person who will own and maintain the shared
> model weights. This is typically done once, and the resulting Data
> Volume is shared read-only with all consumers. **Only the PVC creator
> can write to it** — even other admins on the cluster get read-only
> access if they mount it from a different project.

This step creates a PVC you own, downloads model weights into it, and
wraps it as a cluster-wide Data Volume so everyone can read from it.
All subsequent deployment steps ([Setup Workspace](setup-workspace.md),
[Deploy vLLM](deploy-vllm.md), etc.) mount this at `/models/`.

---

## Why create your own PVC?

If your cluster already has a shared models PVC (e.g. `shared-model-repository`),
you might wonder why not just use it. The problem:

**RunAI enforces that only the original PVC creator's project can write
to it.** When a PVC is shared across projects (via a Data Volume or
cluster-wide label), RunAI creates read-only replica PVCs in each
consumer's namespace — all pointing to the same underlying storage, but
with write access stripped. This is by design:

> "Shared data volumes are mounted with read-only permissions. Any
> modifications must be made by writing to the **original PVC** used
> to create the data volume."
> — [RunAI Data Volumes docs](https://run-ai-docs.nvidia.com/saas/workloads-in-nvidia-run-ai/assets/data-volumes)

In practice this means:
- You **cannot** add new models or adapters to someone else's PVC
- You **cannot** fix missing files (e.g. Jina V4 `adapters/`)
- You **must** ask the original creator to make changes on your behalf

Creating your own PVC gives you full control: add models, include
adapters, remove unused weights, and update versions — all without
depending on someone else.

### Can I still update it after sharing?

**Yes.** Wrapping a PVC in a Data Volume does NOT make the original
PVC read-only. The Data Volume creates read-only replicas for
consumers, but the original PVC data source in your project stays
writable. To add models later:

1. Re-start your provisioning Workspace (mounts the **data source**)
2. Download new models
3. Stop the Workspace — consumers see the new data immediately

The key distinction:
- **Data Source** (PVC) → writable by creator, mount this to add/update models
- **Data Volume** (wrapper) → read-only replicas for consumers

---

## Step 1: Create a PVC data source

In the RunAI UI:

1. Go to **Data & Storage** > **Data Sources** > **New Data Source**
2. Select **PVC** as the type
3. Configure:

| Field | Value |
|-------|-------|
| **Scope** | **Cluster** (e.g. `runai/doit-ai-cluster`) — so all projects can see this data source. If you don't have cluster-level permissions, use your department or project scope and share via a Data Volume in Step 5. |
| **Data source name** | `wattbot-models` |
| **PVC name** | `wattbot-models` *(creates a new PVC)* |
| **Storage class** | Use your cluster's default (ask admin if unsure) |
| **Access mode** | **Read-write by many nodes** (ReadWriteMany / RWX) — allows multiple workloads to mount simultaneously. If your storage class only supports ReadWriteOnce (RWO), that works too — just ensure only one workload mounts during provisioning. |
| **Storage size** | See planning table below |
| **Container path** | `/models` |

4. Create the Data Source

> **"Pending" is normal.** If your storage class uses
> `WaitForFirstConsumer` volume binding (most do), the PVC will show
> as **Pending** in the UI until a workload actually mounts it. This
> is expected Kubernetes behavior — the disk isn't provisioned until
> the first pod claims it. It will become **Bound** when you start
> the provisioning Workspace in Step 2.

### Storage size planning

Each 7B-parameter BF16 model is ~14 GB. Jina V4 with adapters is ~3 GB.
Budget ~20 GB per model you plan to cache:

| Models you want | Total size | Recommended PVC |
|-----------------|------------|-----------------|
| Jina V4 + Qwen 7B only | ~17 GB | `50Gi` |
| + Qwen 14B | ~45 GB | `100Gi` |
| + Qwen 72B | ~190 GB | `250Gi` |
| All models from admin PVC | ~490 GB | `600Gi` |

> **Tip:** Start small. You can always create a larger PVC later and
> re-download. It's faster to over-provision than to migrate.

---

## Step 2: Create a provisioning Workspace

This Workspace mounts the PVC with write access so you can download
model weights. No GPU needed — it's just downloading files.

In the RunAI UI:

1. Go to **Workloads** > **New Workload** > **Workspace**
2. Configure:

| Field | Value |
|-------|-------|
| **Name** | `model-provisioner` |
| **Image** | `nvcr.io/nvidia/pytorch:25.02-py3` |
| **GPU** | `0` (no GPU needed) |
| **CPU** | `2` cores |
| **Memory** | `8 GB` |

3. Under **Data & Storage**, attach:

| Data source | Container path | Access |
|-------------|---------------|--------|
| `wattbot-models` | `/models` | **Read-write** |

4. Set environment variables:

| Key | Value |
|-----|-------|
| `HF_HOME` | `/models/.cache/huggingface` |
| `HF_HUB_CACHE` | `/models/.cache/huggingface/hub` |
| `TRANSFORMERS_CACHE` | `/models/.cache/huggingface/hub` |

5. Create the Workspace, wait for it to start, then **Connect** > open terminal

---

## Step 3: Download models

```bash
# ── Verify the PVC is writable ──
touch /models/.write_test && rm /models/.write_test \
    && echo "Writable!" || echo "READ-ONLY — wrong mount? See troubleshooting."

# ── Create the HuggingFace cache directory structure ──
mkdir -p /models/.cache/huggingface/hub

# ── Install tools ──
pip install huggingface_hub

# ── Download Jina V4 embeddings (FULL model with adapters, ~3 GB) ──
huggingface-cli download jinaai/jina-embeddings-v4
# Verify adapters were included:
ls /models/.cache/huggingface/hub/models--jinaai--jina-embeddings-v4/snapshots/*/adapters/
# Should list: retrieval.query/, retrieval.passage/, text-matching.query/, etc.

# ── Download Qwen 2.5 7B for vLLM (~14 GB) ──
huggingface-cli download Qwen/Qwen2.5-7B-Instruct

# ── Optional: download additional models ──
# huggingface-cli download Qwen/Qwen2.5-14B-Instruct   # ~28 GB
# huggingface-cli download Qwen/Qwen3.5-35B-A3B        # ~17 GB (MoE)
```

> **Download times** depend on your cluster's internet bandwidth.
> Rough estimates: ~5 min for Jina V4 (~3 GB), ~15 min for Qwen 7B
> (~14 GB). Larger models (72B) can take 1+ hours.
>
> **Gated models** (Llama, Gemma) require an HF token. Accept the
> license on huggingface.co first, then:
> ```bash
> export HF_TOKEN="hf_your_token_here"
> huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --token $HF_TOKEN
> ```

---

## Step 4: Verify downloads

```bash
# List all downloaded models with sizes
du -sh /models/.cache/huggingface/hub/models--*/

# Spot-check Jina V4 has config, weights, tokenizer, AND adapters
JINA_SNAP=$(ls -d /models/.cache/huggingface/hub/models--jinaai--jina-embeddings-v4/snapshots/*/  | head -1)
echo "Config:    $(test -f $JINA_SNAP/config.json     && echo OK || echo MISSING)"
echo "Weights:   $(ls $JINA_SNAP/*.safetensors 2>/dev/null | wc -l) files"
echo "Tokenizer: $(test -f $JINA_SNAP/tokenizer.json  && echo OK || echo MISSING)"
echo "Adapters:  $(test -d $JINA_SNAP/adapters         && echo OK || echo MISSING)"

# Spot-check Qwen 7B
QWEN_SNAP=$(ls -d /models/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*/ | head -1)
echo "Config:    $(test -f $QWEN_SNAP/config.json     && echo OK || echo MISSING)"
echo "Weights:   $(ls $QWEN_SNAP/*.safetensors 2>/dev/null | wc -l) files"
echo "Tokenizer: $(test -f $QWEN_SNAP/tokenizer.json  && echo OK || echo MISSING)"
```

All checks should show `OK`. If adapters are `MISSING` for Jina V4,
the embedding server will auto-download them to `/tmp` on each cold
start — but it's better to have them on the PVC permanently.

---

## Step 5: Share as a Data Volume (cluster-wide)

Wrap the PVC in a Data Volume so **everyone on the cluster** can mount
it read-only. This is how other projects/teams will access the models.

1. Go to **Data & Storage** > **Data Volumes** > **New Data Volume**
2. Configure:

| Field | Value |
|-------|-------|
| **Data origin** | Select your `wattbot-models` PVC |
| **Data volume name** | `wattbot-models` |
| **Description** | "Shared model weights for WattBot RAG (Qwen, Jina V4)" |
| **Sharing scope** | **Cluster** — so all projects can mount it |

3. Create the Data Volume

> **If cluster scope isn't available:** Your RunAI role may not allow
> cluster-wide Data Volumes. In that case, share at the **department**
> level, or ask a Data Volumes Administrator to set the scope for you.

### What happens to write access?

- **Your project:** You can still mount the original **data source**
  (`wattbot-models` PVC) with read-write access from any Workspace in
  your project. The Data Volume does not affect the original PVC.
- **Other projects:** They mount the **Data Volume** (read-only
  replicas). They can read all your models but cannot modify them.

To update models later, just re-start the `model-provisioner` Workspace
— it mounts the data source, not the data volume. Everyone else sees
the updates immediately.

---

## Step 6: Stop the provisioning Workspace

Once downloads are verified, **stop the Workspace** from the RunAI UI
to free resources. Re-start it whenever you need to add, update, or
remove models.

---

## Step 7: Update references in other workloads

When you deploy the inference jobs and setup workspace (next steps),
use your PVC name instead of the admin's:

| If the other docs say... | Replace with... |
|--------------------------|-----------------|
| `shared-model-repository` (data source) | `wattbot-models` |
| `shared-models` (data volume) | `wattbot-models` |

The mount path stays the same: `/models`

Everything else (env vars, `HF_HOME`, etc.) is unchanged.

---

## Updating models later

1. Go to **Workloads** in the RunAI UI
2. Find `model-provisioner` and **Start** it (or create a new Workspace
   mounting the `wattbot-models` **data source**)
3. Connect to the terminal
4. Download, remove, or update models as needed
5. Stop the Workspace

Consumers see updated data immediately — no need to recreate the Data
Volume or restart inference jobs (unless the model they're using was
changed, in which case restart that specific job).
