# Troubleshooting

## Quick fixes

- **vLLM OOM:** Reduce `--max-model-len` (e.g., 4096) or use `--quantization awq`
- **Embedding server 503:** Model still loading (~30s on first request). Check logs.
- **Streamlit can't connect:** Verify service DNS names match your job names in the RunAI UI
- **Vector DB not found:** Run [setup](setup-workspace.md) first — `wattbot_jinav4.db` must exist on the PPVC at `/wattbot-data/embeddings/`. Also verify the PPVC is mounted in your workload.
- **Mismatch errors:** Ensure `EMBEDDING_DIM=1024` matches what was used during index build
- **Job keeps crashing:** Check logs in RunAI UI (click job > Logs tab). Common causes: OOM, missing files, image pull failure

## PVC won't bind / "OriginalPvcNotBound" error

If you create a Data Volume in RunAI and see `OriginalPvcNotBound`, the
underlying PVC hasn't been claimed by any pod yet. Most clusters use
`WaitForFirstConsumer` binding mode, meaning the PVC stays `Pending`
until a workload actually mounts it.

**The fix — create the PVC with your first job:**

1. **Job 1 (e.g., `wattbot-test`):** Create a new workload and
   configure the PVC as part of that job (under **Data & Storage** >
   **New PVC**). When the job starts, the pod claims the PVC and it
   binds automatically.
2. **Next job:** Now go to **Data & Storage** > **Data Volumes** and
   create a Data Volume referencing the already-bound PVC. Attach
   that Data Volume to your next workload — it will mount successfully
   because the PVC is already bound.

**Why this happens:** RunAI's Data Volume wizard creates the PVC
object, but with `WaitForFirstConsumer`, Kubernetes won't actually
bind it to a storage backend until a pod schedules that references
it. Creating the Data Volume *before* any pod uses it leaves the PVC
in a `Pending` state, which RunAI reports as `OriginalPvcNotBound`.
The workaround is to let a job create and claim the PVC first, then
wrap it in a Data Volume afterward.

## Read-only file system errors in embedding server logs

The writable overlay handles this automatically. If you still see
these errors, ensure you're running the latest `embedding_server.py`
which includes `_setup_hf_cache_overlay()`.

## Missing adapters (Jina V4)

The embedding server auto-downloads adapters to `/tmp` on each cold
start if they're missing from the PVC. To fix permanently, either ask
the admin to add adapters to their PVC, or create your own PVC with
the full model (see [Managing Models](managing-models.md)).

## Can't write to PVC from any workspace

You're mounting the **Data Volume** (always read-only for consumers),
not the original **PVC data source** (writable by creator). Only the
workspace/project that created the PVC has write access. Create your
own PVC if you need write access (see [Managing Models](managing-models.md)).

## Storage class doesn't support ReadWriteMany

If your cluster only offers ReadWriteOnce (RWO) storage, you can still
use this approach — just ensure only one workload mounts the PVC at a
time during provisioning. After populating, share it as a Data Volume
(read-only, which supports multi-mount regardless of access mode).

## PVC shows 0 bytes used after download

Some network storage classes don't report usage accurately. Check with
`du -sh /models/.cache/huggingface/` from the provisioning workspace
instead of relying on RunAI's storage metrics.
