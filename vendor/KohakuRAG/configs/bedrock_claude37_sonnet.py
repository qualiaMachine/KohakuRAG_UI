"""
WattBot Evaluation Config - Claude 3.7 Sonnet via AWS Bedrock

Newest Sonnet model with improved reasoning capabilities.

Usage:
    python scripts/run_experiment.py --config vendor/KohakuRAG/configs/bedrock_claude37_sonnet.py
"""

# Database settings
db = "../../data/embeddings/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"

# Input/output
questions = "../../data/train_QA.csv"
output = "../../artifacts/submission_bedrock_claude37_sonnet.csv"
metadata = "../../data/metadata.csv"

# LLM settings
llm_provider = "bedrock"
bedrock_model = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
bedrock_region = "us-east-2"
bedrock_profile = "bedrock_nils"

# Embedding settings (Jina v4 - must match index)
embedding_model = "jinav4"
embedding_dim = 1024
embedding_task = "retrieval"

# Retrieval settings (matched to local configs for fair comparison)
top_k = 8
planner_max_queries = 4
deduplicate_retrieval = True
rerank_strategy = "combined"
top_k_final = 10
use_reordered_prompt = True  # C->Q: place context before question

# Unanswerable detection
retrieval_threshold = 0.25

# Other settings
max_retries = 3
max_concurrent = 1
