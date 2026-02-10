"""
KohakuRAG WattBot — Streamlit Dashboard

Interactive demo for the WattBot energy & sustainability RAG pipeline.

Pages:
  1. Dashboard — headline metrics and model comparison
  2. Benchmark Gallery — presentation-grade plots
  3. Ask a Question — ChatGPT-style RAG Q&A via AWS Bedrock
  4. Model Analysis — interactive size/cost/latency analysis

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
PLOTS_DIR = PROJECT_ROOT / "artifacts" / "plots"
DATA_DIR = PROJECT_ROOT / "data"

# Add source paths for imports
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "KohakuRAG" / "src"))

# ---------------------------------------------------------------------------
# Custom CSS — polished dark theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ---- Header banner ---- */
    .hero-banner {
        background: linear-gradient(135deg, #c5050c 0%, #9b0000 50%, #282728 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(197, 5, 12, 0.25);
    }
    .hero-banner h1 {
        color: white;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .hero-banner p {
        color: rgba(255,255,255,0.85);
        margin: 0.3rem 0 0 0;
        font-size: 1rem;
    }

    /* ---- Metric cards ---- */
    div[data-testid="stMetric"] {
        background: #1a1d24;
        border: 1px solid #2d3139;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"] label {
        color: #9ca3af !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background: #12151a;
        border-right: 1px solid #2d3139;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 1rem;
    }

    /* ---- Chat messages ---- */
    .stChatMessage {
        border-radius: 12px !important;
        border: 1px solid #2d3139 !important;
    }

    /* ---- Plot gallery card ---- */
    .plot-card {
        background: #1a1d24;
        border: 1px solid #2d3139;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .plot-card img {
        border-radius: 8px;
    }
    .plot-caption {
        color: #9ca3af;
        font-size: 0.85rem;
        margin-top: 0.5rem;
        font-style: italic;
    }

    /* ---- General polish ---- */
    .stDataFrame {
        border-radius: 8px;
    }
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model Registry
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
# PAGE 1: Dashboard
# ============================================================================
def page_dashboard():
    st.markdown("""
    <div class="hero-banner">
        <h1>🍁 KohakuRAG WattBot</h1>
        <p>Energy & Sustainability RAG — Model Benchmark Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

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
        c1.metric("🏆 Best Score", f"{best['Overall']:.3f}", best["Model"])
        c2.metric("🤖 Models Tested", len(filtered["Model"].unique()))
        c3.metric("🧪 Total Experiments", len(filtered))
        if not cheapest.empty:
            cheapest_row = cheapest.loc[cheapest["Cost ($)"].idxmin()]
            c4.metric("💰 Cheapest Run", f"${cheapest_row['Cost ($)']:.2f}", cheapest_row["Model"])

    # --------------- Results Table ---------------
    st.subheader("📊 Results Table")

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
    st.subheader("📈 Score Comparison")
    best_per_model = filtered.sort_values("Overall", ascending=False).drop_duplicates("Model")
    chart_data = best_per_model.set_index("Model")[["Value Acc.", "Ref Overlap", "NA Acc."]].sort_values(
        "Value Acc.", ascending=False
    )
    st.bar_chart(chart_data, height=400)


# ============================================================================
# PAGE 2: Benchmark Gallery
# ============================================================================
PLOT_INFO = {
    "01_overall_ranking.png": {
        "title": "Overall Model Ranking",
        "caption": "Horizontal bar chart with 95% CI error bars. Models ranked by WattBot composite score (0.75·Value + 0.15·Ref + 0.10·NA)."
    },
    "02_score_breakdown.png": {
        "title": "Score Component Breakdown",
        "caption": "Grouped bar chart showing Value Accuracy, Reference Overlap, and NA Recall for each model."
    },
    "03_refusal_rates.png": {
        "title": "Refusal Rate Comparison",
        "caption": "Percentage of questions where each model answered \"Unable\". Error bars show Wilson 95% CI."
    },
    "04_unique_wins.png": {
        "title": "Unique Wins",
        "caption": "Questions that only one model answered correctly — shows complementary strengths."
    },
    "05_cost_vs_score.png": {
        "title": "Cost vs. Performance (Pareto Frontier)",
        "caption": "Scatter plot with Pareto frontier. DeepSeek R1 is free and ranked #3. Claude Sonnets are premium ($10+)."
    },
    "06_cost_efficiency.png": {
        "title": "Cost Efficiency Ranking",
        "caption": "Score per dollar spent. DeepSeek R1 is free (infinite efficiency). Shows best value models for production deployment."
    },
    "07_latency_vs_score.png": {
        "title": "Latency vs. Performance (Pareto Frontier)",
        "caption": "Speed-quality tradeoff. GPT-OSS models are fastest (2-3s), Claude Sonnets are slower but highest scoring."
    },
    "08_agreement_heatmap.png": {
        "title": "Model Agreement Heatmap",
        "caption": "Pairwise agreement on question correctness. Higher = more similar behavior."
    },
    "09_accuracy_by_type.png": {
        "title": "Accuracy by Question Type",
        "caption": "Performance breakdown by question category: Table, Figure, Math, Quote, and N/A questions."
    },
}


def page_benchmark():
    st.markdown("""
    <div class="hero-banner">
        <h1>📊 Benchmark Gallery</h1>
        <p>Presentation-grade analysis of 10 models across 282 test questions</p>
    </div>
    """, unsafe_allow_html=True)

    if not PLOTS_DIR.exists():
        st.error("No plots found. Run `python scripts/make_presentation_plots.py` first.")
        return

    plot_files = sorted(PLOTS_DIR.glob("*.png"))
    if not plot_files:
        st.error("No PNG plots found in artifacts/plots/")
        return

    st.markdown(f"**{len(plot_files)} plots** generated from benchmark results")
    st.divider()

    # Display in 2-column layout
    for i in range(0, len(plot_files), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(plot_files):
                break
            pf = plot_files[idx]
            info = PLOT_INFO.get(pf.name, {"title": pf.stem.replace("_", " ").title(), "caption": ""})
            with col:
                st.markdown(f"#### {info['title']}")
                st.image(str(pf), width="stretch")
                if info["caption"]:
                    st.caption(info["caption"])
                st.markdown("")  # spacer


# ============================================================================
# PAGE 3: RAG Q&A (Chat-style)
# ============================================================================
EXAMPLE_QUESTIONS = [
    "What is the carbon footprint of training GPT-3?",
    "How much energy does a data center consume annually?",
    "What renewable energy sources are used to power AI systems?",
    "What is the water usage of large language model training?",
    "How do transformer models compare to CNNs in energy efficiency?",
]


def page_qa():
    st.markdown("""
    <div class="hero-banner">
        <h1>💬 Ask a Question</h1>
        <p>Query the WattBot RAG pipeline — powered by AWS Bedrock</p>
    </div>
    """, unsafe_allow_html=True)

    # --------------- Config Sidebar ---------------
    with st.sidebar:
        st.subheader("⚙️ RAG Settings")
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
    alt_db = PROJECT_ROOT / "artifacts" / "wattbot.db"
    if not db_path.exists() and alt_db.exists():
        db_path = alt_db
    if not db_path.exists():
        st.error(f"Vector database not found at `{db_path}`")
        st.info("Download it first:")
        st.code(
            "aws s3 cp s3://wattbot-nils-kohakurag/indexes/wattbot_jinav4.db artifacts/wattbot_jinav4.db --profile bedrock_nils",
            language="bash",
        )
        return

    # --------------- Initialize Chat History ---------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --------------- Example Questions ---------------
    st.markdown("**Try an example:**")
    example_cols = st.columns(len(EXAMPLE_QUESTIONS))
    clicked_example = None
    for i, (col, q) in enumerate(zip(example_cols, EXAMPLE_QUESTIONS)):
        with col:
            # Truncate for button label
            label = q[:35] + "..." if len(q) > 35 else q
            if st.button(label, key=f"example_{i}", use_container_width=True):
                clicked_example = q

    st.divider()

    # --------------- Display Chat History ---------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍🔬" if msg["role"] == "user" else "🍁"):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📚 Sources ({len(msg['sources'])} passages)"):
                    for s in msg["sources"]:
                        st.markdown(f"**{s['doc_id']}** (score: {s['score']:.3f})")
                        st.text(s["content"][:300])
                        st.divider()
            if msg.get("usage"):
                u = msg["usage"]
                c1, c2, c3 = st.columns(3)
                c1.metric("Input Tokens", f"{u.get('input', 0):,}")
                c2.metric("Output Tokens", f"{u.get('output', 0):,}")
                c3.metric("Latency", f"{u.get('latency', 0):.2f}s")

    # --------------- Chat Input ---------------
    question = st.chat_input("Ask an energy & sustainability question...")

    # Handle example button click
    if clicked_example:
        question = clicked_example

    if question:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑‍🔬"):
            st.markdown(question)

        # Generate response
        with st.chat_message("assistant", avatar="🍁"):
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

            answer_text = answer_data.get("answer", "No answer generated.")
            st.markdown(answer_text)

            # Parsed fields
            parsed = answer_data.get("parsed", {})
            if parsed:
                with st.expander("🔍 Structured Response", expanded=False):
                    if parsed.get("explanation"):
                        st.markdown(f"**Explanation:** {parsed['explanation']}")
                    if parsed.get("answer_value"):
                        st.markdown(f"**Value:** `{parsed['answer_value']}`")
                    if parsed.get("ref_id"):
                        st.markdown(f"**References:** {parsed['ref_id']}")

            # Sources
            sources = answer_data.get("sources", [])
            if sources:
                with st.expander(f"📚 Sources ({len(sources)} passages)"):
                    for s in sources:
                        st.markdown(f"**{s['doc_id']}** (score: {s['score']:.3f})")
                        st.text(s["content"][:300])
                        st.divider()

            # Usage metrics
            usage = answer_data.get("usage", {})
            if usage:
                c1, c2, c3 = st.columns(3)
                c1.metric("Input Tokens", f"{usage.get('input', 0):,}")
                c2.metric("Output Tokens", f"{usage.get('output', 0):,}")
                c3.metric("Latency", f"{usage.get('latency', 0):.2f}s")

        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "sources": sources,
            "usage": usage,
        })


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
# PAGE 4: Model Analysis
# ============================================================================
def page_model_size():
    st.markdown("""
    <div class="hero-banner">
        <h1>🔬 Model Analysis</h1>
        <p>Interactive analysis — how does size, cost, and speed relate to performance?</p>
    </div>
    """, unsafe_allow_html=True)

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
    st.subheader(f"📊 Model Size vs. {metric}")

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
    st.subheader("📋 Data")
    display = df[["Model", "Size (B)", "Size Est.", "Overall", "Value Acc.", "Ref Overlap", "Latency (s)", "Cost ($)"]].copy()
    display = display.sort_values("Size (B)")
    st.dataframe(display, use_container_width=True, hide_index=True)

    # --------------- Key Insights ---------------
    st.subheader("💡 Key Observations")

    best_overall = df.loc[df["Overall"].idxmax()]
    fastest = df.loc[df["Latency (s)"].idxmin()]
    best_small = df[df["Size (B)"] <= 20]

    insights = []
    insights.append(f"**Best overall score:** {best_overall['Model']} ({best_overall['Overall']:.3f}) at {best_overall['Size (B)']:.0f}B")
    insights.append(f"**Fastest model:** {fastest['Model']} ({fastest['Latency (s)']:.2f}s avg) at {fastest['Size (B)']:.0f}B")

    if not best_small.empty:
        best_small_row = best_small.loc[best_small["Overall"].idxmax()]
        insights.append(
            f"**Best small model (≤20B):** {best_small_row['Model']} ({best_small_row['Overall']:.3f}) at {best_small_row['Size (B)']:.0f}B"
        )

    for insight in insights:
        st.markdown(f"- {insight}")


# ============================================================================
# Navigation
# ============================================================================
PAGES = {
    "🏠 Dashboard": page_dashboard,
    "📊 Benchmarks": page_benchmark,
    "💬 Ask a Question": page_qa,
    "🔬 Model Analysis": page_model_size,
}

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <span style="font-size: 2.5rem;">🍁</span>
        <h2 style="margin: 0.3rem 0 0 0; font-weight: 700; letter-spacing: -0.5px;">KohakuRAG</h2>
        <p style="color: #9ca3af; margin: 0; font-size: 0.85rem;">WattBot · Energy & Sustainability RAG</p>
        <p style="color: #666; margin: 0.2rem 0 0 0; font-size: 0.75rem;">UW-Madison · ML+X Group</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    page = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")

    # Footer
    st.divider()
    st.caption("Built with Streamlit + AWS Bedrock")
    st.caption("Data: 282 test questions · 10 models")

# Run selected page
PAGES[page]()
