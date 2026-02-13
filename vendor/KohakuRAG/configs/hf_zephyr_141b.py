"""
WattBot Evaluation Config - Zephyr 141B-A35B (Local HF, MoE)

Zephyr 141B-A35B is an instruction-tuned Mixtral 8x22B fine-tuned with
ORPO (Odds Ratio Preference Optimization) by HuggingFace H4, Argilla,
and KAIST. 141B total parameters with ~35B active per token (MoE).
With 4-bit NF4 quantization it needs ~80GB VRAM.
device_map="auto" handles multi-GPU sharding automatically.

Apache 2.0 license, ungated on HuggingFace (uses the non-gated
mistral-community base, fine-tuned by HuggingFace H4).

Usage:
    python scripts/run_experiment.py --config vendor/KohakuRAG/configs/hf_zephyr_141b.py
"""

# Database settings
db = "../../data/embeddings/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"

# Input/output
questions = "../../data/train_QA.csv"
output = "../../artifacts/submission_zephyr_141b.csv"
metadata = "../../data/metadata.csv"

# LLM settings - Zephyr 141B-A35B (local MoE, Mixtral 8x22B fine-tune)
llm_provider = "hf_local"
hf_model_id = "HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1"
hf_max_new_tokens = 512
hf_temperature = 0.2

# Embedding settings (Jina v4 - must match index)
embedding_model = "jinav4"
embedding_dim = 1024
embedding_task = "retrieval"

# Retrieval settings
top_k = 8
planner_max_queries = 4
deduplicate_retrieval = True
rerank_strategy = "combined"
top_k_final = 10
use_reordered_prompt = True  # C->Q: place context before question

# Unanswerable detection
retrieval_threshold = 0.25

# Other settings
max_retries = 2
max_concurrent = 1  # Very large MoE model - conservative concurrency
