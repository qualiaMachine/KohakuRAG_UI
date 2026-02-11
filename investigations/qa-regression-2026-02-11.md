# QA Regression Investigation — 2026-02-11

## Summary

**overall_score dropped from 0.593 to 0.272 (-54%)** when comparing:
- Archive: `archive_pre-fixed-pipeline/experiments/PowerEdge/train_QA/qwen3b-bench/summary.json`
- Current: `artifacts/experiments/PowerEdge/train_QA/qwen3b-bench/summary.json`

## Metrics Comparison

| Metric | Before (archive) | After (current) | Delta |
|---|---|---|---|
| **overall_score** | **0.593** | **0.272** | **-54%** |
| value_accuracy | 0.561 | 0.220 | -61% |
| ref_overlap | 0.817 | 0.049 | **-94%** |
| na_accuracy | 0.50 | 1.00 | +100% (answering N/A for everything) |
| questions_correct | 23/41 | 9/41 | -61% |
| avg_retrieval (s) | 0.615 | 13.527 | **22x slower** |
| avg_generation (s) | 22.1 | 11.6 | -48% (shorter N/A responses) |
| num_snippets (all Qs) | avg 53.7 | **0** | **-100%** |

## Root Cause: Empty/Corrupted Vector Store

### Critical finding

The current run retrieved **ZERO snippets for ALL 41 questions**. The model answered every question blind, from parametric knowledge only.

### Different machines

- Archive ran on `endemann-pytorch21-0-0` (Feb 10)
- Current ran on `endemann-pytorch22-0-0` (Feb 11)

The vector database on pytorch22 is either empty or corrupted.

### How the vec table repair chain caused this

Between the archive and re-run, 5 commits fixed a "dim=1 corruption" bug (afb75c4, 6e4cb0c, df19009, 09d8ef2, d6bc198). The repair logic has a dangerous failure mode:

1. **Old bug**: `KVaultNodeStore.__init__()` used `VectorKVault(dimensions=1)` as a dimension probe, creating vec tables with `float[1]` on fresh databases.

2. **Rust-level repair** (d6bc198): Queries `SELECT COUNT(*) FROM [vec_table]` with `.unwrap_or(0)`. If COUNT(*) fails on a malformed virtual table (dim=1 vs actual data), it defaults to 0 and **drops both the vec table and _values companion table**, destroying the index data.

3. **Python-level repair** (09d8ef2): Renames the entire .db file when it detects a dimension mismatch on an "empty" table. Again checks the _values table count, but uses a different table name format that may not find the right table.

### Likely sequence on pytorch22

1. Old buggy code created vec table with dim=1 (before indexing or during app.py probe)
2. Data was indexed (vectors stored in _values table)
3. New code with repair logic opened the DB
4. Repair detected dim=1 vs expected dim=1024
5. DROP TABLE or DB rename destroyed the index
6. Fresh empty DB created; experiment ran against it

## Config Changes (NOT the primary cause)

| Change | Value Before | Value After | Impact |
|---|---|---|---|
| `planner_max_queries` | 3 | 4 | +1 extra LLM planning call (~3s), explains 22x retrieval slowdown |
| `use_reordered_prompt` | (absent) | True | C->Q prompt ordering, affects generation only |

These affect answer quality at the margins but **cannot** cause zero retrieval results.

## Why 9/41 still "correct"

The model answers from parametric knowledge. na_accuracy=1.0 confirms it correctly says "is_blank" for N/A questions (trivially correct with no context). The 9 correct answers are lucky guesses.

## Fix Required

1. **Rebuild the vector database on pytorch22** from scratch using the current indexing pipeline
2. Verify the Rust-level `.unwrap_or(0)` in core.rs — it should not default to "empty" when COUNT(*) fails
3. Re-run the experiment after DB rebuild
4. Evaluate config changes (planner_max_queries, use_reordered_prompt) separately on a working retrieval pipeline

## Commits Between Archive and Current Run

| Commit | Description |
|---|---|
| b052868 | Archive results before re-running |
| afb75c4 | Fix KVaultNodeStore init: pass embedding_dim instead of None |
| 6e4cb0c | Update docs + fix vector store dim inference |
| df19009 | Fix dimension inference: read ground truth from sqlite-vec |
| 09d8ef2 | Auto-repair corrupted empty vec table with wrong dimensions |
| 4119e3d | Align batch experiment configs (planner_max_queries=4, use_reordered_prompt=True) |
| d6bc198 | Fix corrupted dim=1 vec table (two-layer repair) + debug logging |
| a4779cd | New (broken) results |
