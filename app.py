"""
KohakuRAG WattBot - Streamlit Dashboard

Interactive UI for:
  1. Experiment Results Dashboard -- compare models, view scores
  2. Model Size Analysis -- interactive size vs. performance charts
  3. RAG Q&A -- ask questions through the pipeline

Usage:
    streamlit run app.py
    streamlit run app.py -- --profile bedrock_nils
"""

import asyncio
import json
import sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page Config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="KohakuRAG WattBot",
    page_icon="🍁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
EXPERIMENTS_DIR = PROJECT_ROOT / "artifacts" / "experiments"
DATA_DIR = PROJECT_ROOT / "data"

# Add source paths for imports
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "KohakuRAG" / "src"))

# ---------------------------------------------------------------------------
# Model Size Registry (shared with plot_model_size.py)
# ---------------------------------------------------------------------------
MODEL_SIZES = {
    "claude-3-haiku": ("Claude 3 Haiku", 8, True),
    "claude-3-5-haiku": ("Claude 3.5 Haiku", 8, True),
    "claude-3-5-sonnet": ("Claude 3.5 Sonnet", 70, True),
    "claude-3-7-sonnet": ("Claude 3.7 Sonnet", 70, True),
    "nova-pro": ("Nova Pro", 40, True),
    "llama3-3-70b": ("Llama 3.3 70B", 70, False),
    "llama3-1-70b": ("Llama 3.1 70B", 70, False),
    "llama4-scout-17b": ("Llama 4 Scout", 17, False),
    "llama4-maverick-17b": ("Llama 4 Maverick", 17, False),
    "mistral-small": ("Mistral Small", 24, False),
    "deepseek": ("DeepSeek R1 (distill)", 70, False),
}


def match_model_size(model_id: str):
    model_lower = model_id.lower()
    for key, (name, size, estimated) in MODEL_SIZES.items():
        if key in model_lower:
            return name, size, estimated
    return None, None, None


# ---------------------------------------------------------------------------
# Data Loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60)
def load_all_experiments():
    """Load all experiment summaries from disk."""
    experiments = []
    if not EXPERIMENTS_DIR.exists():
        return experiments

    for summary_path in sorted(EXPERIMENTS_DIR.glob("*/summary.json")):
        try:
            with open(summary_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        name = data.get("name", summary_path.parent.name)
        model_id = data.get("model_id", "")

        if not model_id or "ensemble" in name.lower():
            continue

        display_name, size_b, estimated = match_model_size(model_id)

        experiments.append({
            "Experiment": name,
            "Model": display_name or model_id.split(".")[-1],
            "Model ID": model_id,
            "Size (B)": size_b,
            "Size Est.": "~" if estimated else "",
            "Overall": data.get("overall_score", 0),
            "Value Acc.": data.get("value_accuracy", 0),
            "Ref Overlap": data.get("ref_overlap", 0),
            "NA Acc.": data.get("na_accuracy", 0),
            "Latency (s)": data.get("avg_latency_seconds", 0),
            "Cost ($)": data.get("estimated_cost_usd", 0),
            "Input Tokens": data.get("input_tokens", 0),
            "Output Tokens": data.get("output_tokens", 0),
            "Questions": data.get("num_questions", 0),
            "Errors": data.get("error_count", 0),
        })

    return experiments


# ============================================================================
# PAGE 1: Experiment Dashboard
# ============================================================================
def page_dashboard():
    st.title("Experiment Results Dashboard")
    st.caption("Compare model performance across all evaluation runs")

    experiments = load_all_experiments()
    if not experiments:
        st.warning("No experiments found. Run some experiments first!")
        st.code("python scripts/run_experiment.py --config configs/bedrock_haiku.py")
        return

    import pandas as pd

    df = pd.DataFrame(experiments)

    # --------------- Filters ---------------
    col1, col2 = st.columns(2)
    with col1:
        min_score = st.slider("Minimum overall score", 0.0, 1.0, 0.15, 0.05)
    with col2:
        models_available = sorted(df["Model"].dropna().unique())
        selected_models = st.multiselect("Filter models", models_available, default=models_available)

    filtered = df[(df["Overall"] >= min_score) & (df["Model"].isin(selected_models))]

    # --------------- Summary Cards ---------------
    if not filtered.empty:
        best = filtered.loc[filtered["Overall"].idxmax()]
        cheapest = filtered[filtered["Cost ($)"] > 0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best Score", f"{best['Overall']:.3f}", best["Model"])
        c2.metric("Models Tested", len(filtered["Model"].unique()))
        c3.metric("Total Experiments", len(filtered))
        if not cheapest.empty:
            cheapest_row = cheapest.loc[cheapest["Cost ($)"].idxmin()]
            c4.metric("Cheapest Run", f"${cheapest_row['Cost ($)']:.4f}", cheapest_row["Model"])

    # --------------- Results Table ---------------
    st.subheader("Results Table")

    display_cols = [
        "Experiment", "Model", "Size (B)", "Overall", "Value Acc.",
        "Ref Overlap", "NA Acc.", "Latency (s)", "Cost ($)", "Errors",
    ]

    st.dataframe(
        filtered[display_cols].sort_values("Overall", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Overall": st.column_config.ProgressColumn("Overall", min_value=0, max_value=1, format="%.3f"),
            "Value Acc.": st.column_config.ProgressColumn("Value Acc.", min_value=0, max_value=1, format="%.3f"),
            "Ref Overlap": st.column_config.ProgressColumn("Ref Overlap", min_value=0, max_value=1, format="%.3f"),
            "NA Acc.": st.column_config.ProgressColumn("NA Acc.", min_value=0, max_value=1, format="%.3f"),
            "Latency (s)": st.column_config.NumberColumn(format="%.2f"),
            "Cost ($)": st.column_config.NumberColumn(format="$%.4f"),
        },
    )

    # --------------- Bar Chart ---------------
    st.subheader("Score Comparison")

    # Deduplicate: keep best run per model
    best_per_model = filtered.sort_values("Overall", ascending=False).drop_duplicates("Model")

    chart_data = best_per_model.set_index("Model")[["Value Acc.", "Ref Overlap", "NA Acc."]].sort_values(
        "Value Acc.", ascending=False
    )
    st.bar_chart(chart_data, height=400)


# ============================================================================
# PAGE 2: Model Size Analysis
# ============================================================================
def page_model_size():
    st.title("Model Size Analysis")
    st.caption("How does model size relate to performance, latency, and cost?")

    experiments = load_all_experiments()
    if not experiments:
        st.warning("No experiments found.")
        return

    import pandas as pd

    df = pd.DataFrame(experiments)

    # Filter out broken runs and keep best per model
    df = df[df["Overall"] >= 0.15]
    df = df.sort_values("Overall", ascending=False).drop_duplicates("Model")
    df = df.dropna(subset=["Size (B)"])

    if df.empty:
        st.warning("No experiments with model size information.")
        return

    # --------------- Metric Selector ---------------
    metric = st.selectbox(
        "Y-axis metric",
        ["Overall", "Value Acc.", "Ref Overlap", "Latency (s)", "Cost ($)"],
        index=0,
    )

    # --------------- Scatter Plot ---------------
    st.subheader(f"Model Size vs. {metric}")

    chart_df = df[["Model", "Size (B)", metric, "Size Est."]].copy()
    chart_df["Label"] = chart_df.apply(
        lambda r: f"{r['Model']}{'*' if r['Size Est.'] == '~' else ''} ({r['Size (B)']:.0f}B)", axis=1
    )

    st.scatter_chart(
        chart_df,
        x="Size (B)",
        y=metric,
        color="Model",
        size=80,
        height=500,
    )

    st.caption("(*) = estimated model size (proprietary model)")

    # --------------- Data Table ---------------
    st.subheader("Data")
    display = df[["Model", "Size (B)", "Size Est.", "Overall", "Value Acc.", "Ref Overlap", "Latency (s)", "Cost ($)"]].copy()
    display = display.sort_values("Size (B)")
    st.dataframe(display, use_container_width=True, hide_index=True)

    # --------------- Key Insights ---------------
    st.subheader("Key Observations")

    # Auto-generate some insights
    best_overall = df.loc[df["Overall"].idxmax()]
    fastest = df.loc[df["Latency (s)"].idxmin()]
    best_small = df[df["Size (B)"] <= 20]

    insights = []
    insights.append(f"**Best overall score:** {best_overall['Model']} ({best_overall['Overall']:.3f}) at {best_overall['Size (B)']:.0f}B")
    insights.append(f"**Fastest model:** {fastest['Model']} ({fastest['Latency (s)']:.2f}s avg) at {fastest['Size (B)']:.0f}B")

    if not best_small.empty:
        best_small_row = best_small.loc[best_small["Overall"].idxmax()]
        insights.append(
            f"**Best small model (<=20B):** {best_small_row['Model']} ({best_small_row['Overall']:.3f}) at {best_small_row['Size (B)']:.0f}B"
        )

    for insight in insights:
        st.markdown(f"- {insight}")


# ============================================================================
# PAGE 3: RAG Q&A
# ============================================================================
def page_qa():
    st.title("Ask a Question")
    st.caption("Query the WattBot RAG pipeline with any energy/sustainability question")

    # --------------- Config Sidebar ---------------
    with st.sidebar:
        st.subheader("RAG Settings")
        aws_profile = st.text_input("AWS Profile", value="bedrock_nils")
        model_options = {
            "Claude 3 Haiku (fast, cheap)": "us.anthropic.claude-3-haiku-20240307-v1:0",
            "Claude 3.5 Haiku": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
            "Claude 3.5 Sonnet": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "Claude 3.7 Sonnet": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "Llama 4 Maverick 17B": "us.meta.llama4-maverick-17b-instruct-v1:0",
            "DeepSeek R1": "us.deepseek.r1-v1:0",
        }
        selected_model_name = st.selectbox("Model", list(model_options.keys()))
        selected_model_id = model_options[selected_model_name]
        top_k = st.slider("Retrieval top-k", 3, 15, 8)

    # --------------- DB Check ---------------
    db_path = PROJECT_ROOT / "artifacts" / "wattbot_jinav4.db"
    if not db_path.exists():
        st.error(f"Vector database not found at `{db_path}`")
        st.info("Download it first:")
        st.code(
            "aws s3 cp s3://wattbot-nils-kohakurag/indexes/wattbot_jinav4.db artifacts/wattbot_jinav4.db --profile bedrock_nils",
            language="bash",
        )
        return

    # --------------- Question Input ---------------
    question = st.text_area(
        "Your question",
        placeholder="e.g., What is the carbon footprint of training GPT-3?",
        height=100,
    )

    if st.button("Ask", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner(f"Thinking with {selected_model_name}..."):
            try:
                answer_data = _run_rag_query(
                    question=question,
                    model_id=selected_model_id,
                    profile_name=aws_profile,
                    db_path=str(db_path),
                    top_k=top_k,
                )
            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Make sure you're logged in: `aws sso login --profile bedrock_nils`")
                return

        # --------------- Display Answer ---------------
        st.divider()
        st.subheader("Answer")
        st.markdown(answer_data.get("answer", "No answer generated."))

        # Show parsed fields if available
        parsed = answer_data.get("parsed", {})
        if parsed:
            with st.expander("Structured Response", expanded=False):
                if parsed.get("explanation"):
                    st.markdown(f"**Explanation:** {parsed['explanation']}")
                if parsed.get("answer_value"):
                    st.markdown(f"**Value:** `{parsed['answer_value']}`")
                if parsed.get("ref_id"):
                    st.markdown(f"**References:** {parsed['ref_id']}")
                if parsed.get("supporting_materials"):
                    st.markdown(f"**Supporting:** {parsed['supporting_materials']}")

        # Show retrieved context
        if answer_data.get("sources"):
            with st.expander(f"Retrieved Context ({len(answer_data['sources'])} passages)", expanded=False):
                for i, source in enumerate(answer_data["sources"], 1):
                    st.markdown(f"**[{i}]** {source.get('doc_id', 'unknown')} (score: {source.get('score', 0):.3f})")
                    st.text(source.get("content", "")[:500])
                    st.divider()

        # Token usage
        usage = answer_data.get("usage", {})
        if usage:
            c1, c2, c3 = st.columns(3)
            c1.metric("Input Tokens", f"{usage.get('input', 0):,}")
            c2.metric("Output Tokens", f"{usage.get('output', 0):,}")
            c3.metric("Latency", f"{usage.get('latency', 0):.2f}s")


def _run_rag_query(question: str, model_id: str, profile_name: str, db_path: str, top_k: int) -> dict:
    """Run a RAG query and return structured results."""
    import time

    from llm_bedrock import BedrockChatModel
    from kohakurag.datastore import KVaultNodeStore
    from kohakurag.embeddings import JinaV4EmbeddingModel
    from kohakurag import RAGPipeline

    start = time.time()

    # Initialize components
    chat = BedrockChatModel(
        model_id=model_id,
        profile_name=profile_name,
        region_name="us-east-2",
        max_concurrent=3,
    )

    store = KVaultNodeStore(
        db_path,
        table_prefix="wattbot_jv4",
        dimensions=None,
        paragraph_search_mode="averaged",
    )

    embedder = JinaV4EmbeddingModel(task="retrieval", truncate_dim=512)

    pipeline = RAGPipeline(store=store, embedder=embedder, chat_model=chat, planner=None)

    # Run the query
    system_prompt = (
        "You must answer strictly based on the provided context snippets. "
        "Do NOT use external knowledge or assumptions. "
        "If the context does not clearly support an answer, say so."
    )

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(
            pipeline.run_qa(
                question=question,
                top_k=top_k,
                system_prompt=system_prompt,
            )
        )
    finally:
        loop.close()

    latency = time.time() - start

    # Extract answer
    answer_text = getattr(result.answer, "explanation", str(result.answer))
    parsed = {}
    for field in ["explanation", "answer_value", "ref_id", "supporting_materials"]:
        val = getattr(result.answer, field, None)
        if val:
            parsed[field] = val

    # Extract sources
    sources = []
    if hasattr(result, "context_nodes"):
        for node, score in result.context_nodes:
            sources.append({
                "doc_id": node.metadata.get("doc_id", "unknown"),
                "content": node.content[:500],
                "score": score,
            })

    return {
        "answer": answer_text,
        "parsed": parsed,
        "sources": sources,
        "usage": {
            "input": chat.token_usage.input_tokens,
            "output": chat.token_usage.output_tokens,
            "latency": latency,
        },
    }


# ============================================================================
# Navigation
# ============================================================================
PAGES = {
    "Dashboard": page_dashboard,
    "Model Size Analysis": page_model_size,
    "Ask a Question": page_qa,
}

with st.sidebar:
    st.title("KohakuRAG WattBot")
    st.caption("Energy & Sustainability RAG")
    st.divider()
    page = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")

# Run selected page
PAGES[page]()
