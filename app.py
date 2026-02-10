"""
WattBot RAG — Streamlit App

Interactive UI for querying the WattBot RAG pipeline with local HF models.

Launch:
    streamlit run app.py
    streamlit run app.py -- --config vendor/KohakuRAG/configs/hf_qwen7b.py
"""

import argparse
import asyncio
import importlib.util
import sys
import time
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Path setup (same as run_experiment.py)
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_repo_root / "vendor" / "KohakuRAG" / "src"))
sys.path.insert(0, str(_repo_root / "scripts"))

from kohakurag import RAGPipeline
from kohakurag.datastore import KVaultNodeStore
from kohakurag.embeddings import JinaV4EmbeddingModel
from kohakurag.llm import HuggingFaceLocalChatModel

# ---------------------------------------------------------------------------
# Prompts (shared with run_experiment.py)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You must answer strictly based on the provided context snippets.
Do NOT use external knowledge or assumptions.
If the context does not clearly support an answer, you must output the literal string "is_blank" for both answer_value and ref_id.
For True/False questions, you MUST output "1" for True and "0" for False in answer_value. Do NOT output the words "True" or "False".
""".strip()

USER_TEMPLATE = """
You will be given a question and context snippets taken from documents.
You must follow these rules:
- Use only the provided context; do not rely on external knowledge.
- If the context does not clearly support an answer, use "is_blank" for all fields except explanation.
- For unanswerable questions, set answer to "Unable to answer with confidence based on the provided documents."
- For True/False questions: answer_value must be "1" for True or "0" for False (not the words "True" or "False").

Question: {question}

Context:
{context}

Return STRICT JSON with the following keys, in this order:
- explanation          (1-3 sentences explaining how the context supports the answer; or "is_blank")
- answer               (short natural-language response, e.g. "1438 lbs", "Water consumption", "TRUE")
- answer_value         (ONLY the numeric or categorical value, e.g. "1438", "Water consumption", "1"; or "is_blank")
- ref_id               (list of document ids from the context used as evidence; or "is_blank")
- ref_url              (list of URLs for the cited documents; or "is_blank")
- supporting_materials (verbatim quote, table reference, or figure reference from the cited document; or "is_blank")

JSON Answer:
""".strip()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
CONFIGS_DIR = _repo_root / "vendor" / "KohakuRAG" / "configs"


def discover_configs() -> dict[str, Path]:
    """Find all hf_*.py config files and return {display_name: path}."""
    configs = {}
    for p in sorted(CONFIGS_DIR.glob("hf_*.py")):
        configs[p.stem] = p
    return configs


def load_config(config_path: Path) -> dict:
    """Load a Python config file into a dict."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {}
    for key in [
        "db", "table_prefix", "questions", "output", "metadata",
        "llm_provider", "top_k", "planner_max_queries", "deduplicate_retrieval",
        "rerank_strategy", "top_k_final", "retrieval_threshold",
        "max_retries", "max_concurrent",
        "embedding_model", "embedding_dim", "embedding_task", "embedding_model_id",
        "hf_model_id", "hf_dtype", "hf_max_new_tokens", "hf_temperature",
    ]:
        if hasattr(module, key):
            config[key] = getattr(module, key)
    return config


# ---------------------------------------------------------------------------
# Pipeline init (cached so model loads only once)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model and vector store...")
def init_pipeline(config_name: str, precision: str) -> RAGPipeline:
    """Load LLM, embedder, and vector store. Cached across reruns."""
    config_path = CONFIGS_DIR / f"{config_name}.py"
    config = load_config(config_path)
    config["hf_dtype"] = precision

    # LLM
    chat_model = HuggingFaceLocalChatModel(
        model=config.get("hf_model_id", "Qwen/Qwen2.5-7B-Instruct"),
        system_prompt=SYSTEM_PROMPT,
        dtype=config.get("hf_dtype", "4bit"),
        max_new_tokens=config.get("hf_max_new_tokens", 512),
        temperature=config.get("hf_temperature", 0.2),
        max_concurrent=config.get("max_concurrent", 2),
    )

    # Embedder
    embedder = JinaV4EmbeddingModel(
        task=config.get("embedding_task", "retrieval"),
        truncate_dim=config.get("embedding_dim", 1024),
    )

    # Vector store
    db_raw = config.get("db", "data/embeddings/wattbot_jinav4.db")
    db_path = _repo_root / db_raw.removeprefix("../").removeprefix("../")
    table_prefix = config.get("table_prefix", "wattbot_jv4")
    store = KVaultNodeStore(
        db_path,
        table_prefix=table_prefix,
        dimensions=None,
        paragraph_search_mode="averaged",
    )

    return RAGPipeline(
        store=store,
        embedder=embedder,
        chat_model=chat_model,
        planner=None,
    )


def run_query(pipeline: RAGPipeline, question: str, top_k: int):
    """Run pipeline.run_qa synchronously (wraps the async call)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            pipeline.run_qa(
                question,
                system_prompt=SYSTEM_PROMPT,
                user_template=USER_TEMPLATE,
                top_k=top_k,
            )
        )
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="WattBot RAG", page_icon="lightning", layout="wide")
    st.title("WattBot RAG Pipeline")

    # ---- Sidebar: model / precision selection ----
    configs = discover_configs()
    if not configs:
        st.error("No HF config files found in vendor/KohakuRAG/configs/")
        return

    with st.sidebar:
        st.header("Settings")
        config_name = st.selectbox("Model config", list(configs.keys()), index=list(configs.keys()).index("hf_qwen7b") if "hf_qwen7b" in configs else 0)
        precision = st.selectbox("Precision", ["4bit", "bf16", "fp16", "auto"], index=0)
        top_k = st.slider("Retrieved chunks (top_k)", min_value=1, max_value=20, value=8)

        st.divider()
        st.caption("Model is loaded once and cached. Changing model or precision triggers a reload.")

    # ---- Load pipeline ----
    try:
        pipeline = init_pipeline(config_name, precision)
    except FileNotFoundError as e:
        st.error(str(e))
        return

    # ---- Chat interface ----
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "details" in msg:
                _render_details(msg["details"])

    # User input
    if question := st.chat_input("Ask a question about the WattBot documents..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating..."):
                t0 = time.time()
                try:
                    result = run_query(pipeline, question, top_k)
                    elapsed = time.time() - t0
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    return

            answer = result.answer
            timing = result.timing

            # Main answer
            st.markdown(f"**{answer.answer}**")

            if answer.answer_value and answer.answer_value != "is_blank":
                st.markdown(f"Value: `{answer.answer_value}`")

            if answer.explanation and answer.explanation != "is_blank":
                st.markdown(answer.explanation)

            details = {
                "timing": timing,
                "elapsed": elapsed,
                "ref_id": answer.ref_id,
                "ref_url": answer.ref_url,
                "supporting_materials": answer.supporting_materials,
                "snippets": [
                    {"rank": s.rank, "score": s.score, "title": s.document_title, "text": s.text}
                    for s in result.retrieval.snippets
                ],
                "raw_response": result.raw_response,
            }
            _render_details(details)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer.answer,
                "details": details,
            })


def _render_details(details: dict):
    """Render expandable sections for references, context, and timing."""
    timing = details.get("timing", {})
    elapsed = details.get("elapsed", 0)

    cols = st.columns(3)
    cols[0].metric("Retrieval", f"{timing.get('retrieval_s', 0):.1f}s")
    cols[1].metric("Generation", f"{timing.get('generation_s', 0):.1f}s")
    cols[2].metric("Total", f"{elapsed:.1f}s")

    # References
    ref_ids = details.get("ref_id", [])
    ref_urls = details.get("ref_url", [])
    if ref_ids and ref_ids != "is_blank":
        with st.expander("References"):
            for i, rid in enumerate(ref_ids if isinstance(ref_ids, list) else [ref_ids]):
                url = ref_urls[i] if isinstance(ref_urls, list) and i < len(ref_urls) else None
                if url and url != "is_blank":
                    st.markdown(f"- [{rid}]({url})")
                else:
                    st.markdown(f"- {rid}")
            sm = details.get("supporting_materials", "")
            if sm and sm != "is_blank":
                st.caption(f"Supporting: {sm}")

    # Retrieved context
    snippets = details.get("snippets", [])
    if snippets:
        with st.expander(f"Retrieved context ({len(snippets)} chunks)"):
            for s in snippets:
                st.markdown(f"**#{s['rank']}** _{s['title']}_ (score: {s['score']:.3f})")
                st.text(s["text"][:500] + ("..." if len(s["text"]) > 500 else ""))
                st.divider()

    # Raw LLM output
    raw = details.get("raw_response", "")
    if raw:
        with st.expander("Raw LLM response"):
            st.code(raw, language="json")


if __name__ == "__main__":
    main()
