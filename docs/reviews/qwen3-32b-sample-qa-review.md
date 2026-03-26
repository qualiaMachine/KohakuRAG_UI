# Review: Qwen3-32B Sample Q&A (WattBot RAG Pipeline)

**Date**: 2026-03-26
**Model**: Qwen3-32B (vLLM remote)
**Settings**: top_k=5, cross-encoder reranker ON, query expansion OFF, best-guess ON, research mode OFF, max retries=2

## Settings Observed

- **Mode**: Remote (vLLM + embedding server)
- **LLM**: Qwen3-32B via vLLM
- **Retrieved chunks (top_k)**: 5
- **Cross-encoder reranker**: ON
- **Query expansion**: OFF
- **Best-guess answers**: ON
- **Research mode**: OFF
- **Max retries**: 2

## Q&A Analysis

### Q1: "How much energy to train an LLM?"

- **Answer**: "0.8 MWh to 3,500 MWh" with Llama-3.1 example (39.3M GPU hours on H100-80GB, 700W TDP)
- **Sources**: Han et al., 2024; Luccioni et al., 2025
- **Retrieved**: 5 chunks from 5 sources (43 total), 5 figures
- **Timing**: 0.8s retrieval, 2.4s generation, 3.2s total, 61.1 mWh
- **Verdict**: **Good**. Provides a concrete range with a specific example. The 700W TDP for H100-80GB is correct. Strong retrieval with supporting figures.

### Q2: "How much water to train an LLM?"

- **Answer**: "millions of liters of water" + spatial/temporal diversity affects footprint
- **Sources**: Li et al., 2025; Hyunwodesign et al., 2025
- **Retrieved**: 5 chunks from 5 sources (78 total), 1 figure
- **Timing**: 0.4s retrieval, 2.4s generation, 2.8s total, 52.3 mWh
- **Verdict**: **Weak**. The answer is vague — "millions of liters" is not a useful range. Compare with Q1 which gave concrete numbers. The model should have extracted specific figures (e.g., GPT-3 training ~700K liters per Li et al.). The "Hyunwodesign" source name looks unusual — worth verifying that ref_id is correct.

### Q3: "How much energy per ChatGPT call?"

- **Answer**: "approximately 0.42 Wh"
- **Sources**: Jegham et al., 2025; Luccioni et al., 2025
- **Retrieved**: 5 chunks from 3 sources (34 total), 2 figures
- **Supporting**: Figure 5 (Per-query and daily energy consumption of GPT-4o)
- **Timing**: 1.0s retrieval, 2.8s generation, 3.9s total, 73.3 mWh
- **Verdict**: **Good**. Specific numeric answer with proper units. The 0.42 Wh figure aligns with published estimates. Retrieved relevant supporting figures.

### Q4: "Which LLM provider is most energy efficient?"

- **Answer**: "varies widely depending on model architecture, hardware, and service level objectives" + best performing model can also be energy efficient, larger LLMs tend to consume more energy with lower accuracy
- **Sources**: Shumba et al., 2024; Moore et al., 2025
- **Retrieved**: 5 chunks from 4 sources (52 total)
- **Supporting**: Table 6
- **Timing**: 0.3s retrieval, 2.0s generation, 2.4s total, 45.1 mWh
- **Verdict**: **Evasive**. The question asks "which" and the model hedges without naming any provider. With best-guess mode ON, it should have at least named the top-performing providers from the retrieved context. The "Value: n/a" indicator confirms the model couldn't extract a concrete answer.

## Summary Metrics

| Question | Specificity | Accuracy | Speed | Energy |
|----------|-------------|----------|-------|--------|
| Q1 (train energy) | Concrete range + example | Correct | 3.2s | 61.1 mWh |
| Q2 (train water) | Vague ("millions") | Plausible | 2.8s | 52.3 mWh |
| Q3 (ChatGPT energy) | Specific (0.42 Wh) | Correct | 3.9s | 73.3 mWh |
| Q4 (efficient provider) | Evasive (no names) | Non-committal | 2.4s | 45.1 mWh |

**Session totals**: 231.8 mWh, 4 queries, 58.0 mWh avg

## Comparison with Formal Benchmarks

| Model | Overall Score | Value Accuracy | Ref Overlap |
|-------|--------------|----------------|-------------|
| Qwen2.5-32B (PowerEdge) | 81.4% | 78.7% | 90.4% |
| Qwen2.5-32B (GB10) | 81.9% | 79.8% | 89.0% |
| Qwen3-30B-A3B (MoE) | 79.7% | 76.6% | 90.1% |
| Qwen3-Next-80B-A3B | 83.5% | 80.9% | 90.0% |
| Claude 3.5 Sonnet | 83.0% | 80.9% | 90.8% |

Note: Formal benchmarks use top_k=8 and top_k_final=10 (vs. top_k=5 in this demo).

## Recommendations

1. **Increase top_k to 8**: The demo uses top_k=5 while benchmarks use top_k=8 with top_k_final=10 after reranking. This would likely improve answer specificity for broad questions like Q2 and Q4.
2. **Enable query expansion**: Currently OFF. For open-ended demo questions (vs. the precise competition questions), query expansion would help surface better chunks. The ~0.5s overhead is negligible.
3. **Improve synthesis for comparative questions**: Q4 shows the model defaults to hedging for ranking/comparison questions. Consider adjusting the system prompt to encourage naming specific entities when evidence exists, especially with best-guess mode ON.
4. **Verify "Hyunwodesign" reference**: The source name in Q2 is unusual — confirm this ref_id maps correctly in metadata.
