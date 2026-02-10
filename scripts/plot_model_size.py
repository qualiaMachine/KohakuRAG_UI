#!/usr/bin/env python3
"""
Model Size vs. Performance/Latency/Cost Plots

Generates publication-quality plots for ML+X meeting (2/17/2026):
  - Split individual plots for maximum clarity (no 2x2 grids).
  - Aggressive label repulsion with arrows to prevent overlap.
  - Presentation-grade font sizes (Poster/Slide ready).

Usage:
    python scripts/plot_model_size.py
    python scripts/plot_model_size.py --experiments artifacts/experiments --output artifacts/plots
"""

import argparse
import json
import math
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
except ImportError:
    print("matplotlib and numpy required: pip install matplotlib numpy")
    sys.exit(1)

try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False


# ============================================================================
# Model Size Registry
# ============================================================================

MODEL_SIZES = {
    # model_id substring -> (display_name, active_params_B, estimated, notes)
    "claude-3-haiku": ("Claude 3 Haiku", 8, True, "~8B"),
    "claude-3-5-haiku": ("Claude 3.5 Haiku", 8, True, "~8B"),
    "claude-3-5-sonnet": ("Claude 3.5 Sonnet", 70, True, "~70B"),
    "claude-3-7-sonnet": ("Claude 3.7 Sonnet", 70, True, "~70B"),
    "nova-pro": ("Nova Pro", 40, True, "~40B"),
    "llama3-3-70b": ("Llama 3.3 70B", 70, False, "70B"),
    "llama3-1-70b": ("Llama 3.1 70B", 70, False, "70B"),
    "llama4-scout-17b": ("Llama 4 Scout", 17, False, "17B"),
    "llama4-maverick-17b": ("Llama 4 Maverick", 17, False, "17B"),
    "mistral-small": ("Mistral Small", 24, False, "24B"),
    "deepseek": ("DeepSeek R1 (distill)", 70, False, "70B"),
    "gpt-oss-20b": ("GPT-OSS 20B", 20, False, "20B"),
    "gpt-oss-120b": ("GPT-OSS 120B", 120, False, "120B"),
}

FAMILY_COLORS = {
    "Claude": "#6366f1",      # Indigo
    "Llama": "#f59e0b",       # Amber
    "GPT": "#22c55e",         # Green
    "Mistral": "#10b981",     # Emerald
    "DeepSeek": "#ef4444",    # Red
    "Nova": "#ec4899",        # Pink
}

def match_model_size(model_id: str) -> tuple[str, int, bool]:
    model_lower = model_id.lower()
    for key, (name, size, estimated, _) in MODEL_SIZES.items():
        if key in model_lower:
            return name, size, estimated
    return None

def get_color(name: str) -> str:
    for family, color in FAMILY_COLORS.items():
        if family.lower() in name.lower():
            return color
    return "#6b7280"

def calculate_standard_error(values: list[float]) -> float:
    if not values or len(values) < 2:
        return 0.0
    return np.std(values, ddof=1) / math.sqrt(len(values))


# ============================================================================
# Loading Logic
# ============================================================================

def load_experiments(experiments_dir: Path, name_filter: str | None = None) -> list[dict]:
    experiments = []
    seen_models = {}

    for summary_path in sorted(experiments_dir.glob("*/summary.json")):
        try:
            with open(summary_path) as f:
                data = json.load(f)
        except Exception:
            continue

        name = data.get("name", summary_path.parent.name)
        model_id = data.get("model_id", "")

        if name_filter and name_filter not in name:
            continue
        if not model_id or "ensemble" in name.lower():
            continue
        if data.get("overall_score", 0) < 0.15:
            continue

        size_info = match_model_size(model_id)
        if not size_info:
            continue
        display_name, size_b, estimated = size_info

        # Load results for SE
        results_path = summary_path.parent / "results.json"
        se_overall = se_value = se_ref = se_na = 0.0
        if results_path.exists():
            try:
                with open(results_path) as f:
                    results = json.load(f)
                weighted = [r.get("weighted_score", 0.0) for r in results]
                value = [1.0 if r.get("value_correct") else 0.0 for r in results]
                ref = [r.get("ref_score", 0.0) for r in results]
                na = [1.0 if r.get("na_correct") else 0.0 for r in results]
                
                se_overall = calculate_standard_error(weighted)
                se_value = calculate_standard_error(value)
                se_ref = calculate_standard_error(ref)
                se_na = calculate_standard_error(na)
            except Exception:
                pass

        entry = {
            "experiment": name,
            "display_name": display_name,
            "size_b": size_b,
            "size_estimated": estimated,
            "num_questions": data.get("num_questions", 0),
            "overall_score": data.get("overall_score", 0),
            "se_overall": se_overall,
            "value_accuracy": data.get("value_accuracy", 0),
            "se_value": se_value,
            "ref_overlap": data.get("ref_overlap", 0),
            "se_ref": se_ref,
            "na_accuracy": data.get("na_accuracy", 0),
            "se_na": se_na,
            "avg_latency": data.get("avg_latency_seconds", 0),
            "total_cost": data.get("estimated_cost_usd", 0),
        }

        # Dedup: preserve best score
        if model_id in seen_models:
            if entry["overall_score"] > seen_models[model_id]["overall_score"]:
                seen_models[model_id] = entry
        else:
            seen_models[model_id] = entry

    return sorted(seen_models.values(), key=lambda x: x["size_b"])


# ============================================================================
# Plotting Core
# ============================================================================

def setup_plot_style():
    """Set rigorous presentation style: big fonts, clean lines."""
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 24,
        'axes.labelsize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 28,
        'figure.figsize': (12, 8),
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })

def smart_annotate(ax, xs, ys, labels, estimated_flags):
    """
    Annotate with aggressive repulsion.
    Using `adjust_text` with arrow props to allow labels to fly far away.
    """
    texts = []
    for x, y, label, est in zip(xs, ys, labels, estimated_flags):
        display = f"{label}*" if est else label
        
        # White background box for readability over grid lines
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8)
        
        t = ax.text(
            x, y, display,
            ha='center', va='center',
            fontsize=12,
            fontweight='bold',
            bbox=bbox_props
        )
        texts.append(t)

    if HAS_ADJUST_TEXT:
        try:
            adjust_text(
                texts,
                ax=ax,
                expand_points=(2.0, 2.0),  # Push away from points strongly
                expand_text=(1.5, 1.5),    # Push away from other text strongly
                force_text=1.0,            # Max repulsion
                force_points=1.0,          # Max repulsion
                arrowprops=dict(arrowstyle='-', color='gray', lw=1.0, alpha=0.6)
            )
        except Exception as e:
            print(f"Warning: adjust_text failed: {e}")

def plot_single_metric(experiments, x_key, y_key, y_err_key, title, xlabel, ylabel, filename, log_x=True, log_y=False):
    """Generic function to plot one metric cleanly."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    xs = [e[x_key] for e in experiments]
    ys = [e[y_key] for e in experiments]
    yerrs = [e[y_err_key] for e in experiments] if y_err_key else None
    names = [e["display_name"] for e in experiments]
    est = [e["size_estimated"] for e in experiments]
    colors = [get_color(n) for n in names]

    # Error bars (lighter, behind)
    if yerrs:
        ax.errorbar(xs, ys, yerr=yerrs, fmt='none', ecolor='#9ca3af', elinewidth=2, capsize=5, zorder=2)

    # Points
    ax.scatter(xs, ys, c=colors, s=250, edgecolors='white', linewidth=2, zorder=5)

    # Labels
    smart_annotate(ax, xs, ys, names, est)

    # Formatting
    ax.set_title(title, pad=20, fontweight='bold')
    ax.set_xlabel(xlabel, labelpad=15)
    ax.set_ylabel(ylabel, labelpad=15)

    if log_x:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}B"))
    if log_y:
        ax.set_yscale('log')
        
    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add N count annotation in corner
    n_qs = experiments[0]["num_questions"] if experiments else "?"
    ax.text(0.02, 0.98, f"N={n_qs}", transform=ax.transAxes, fontsize=14, 
            verticalalignment='top', bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.5))

    # Add Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=12, label=f)
        for f, c in FAMILY_COLORS.items()
    ]
    ax.legend(handles=legend_elements, loc='lower right', title="Model Family")

    plt.tight_layout()
    output_path = Path(f"artifacts/plots/{filename}")
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", default="artifacts/experiments")
    parser.add_argument("--output", default="artifacts/plots")
    parser.add_argument("--filter", default="test")
    args = parser.parse_args()

    setup_plot_style()
    
    exp_dir = Path(args.experiments)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = load_experiments(exp_dir, args.filter)
    if not experiments:
        print("No experiments found.")
        return

    print(f"Loaded {len(experiments)} experiments. Generating plots...")

    # 1. Overall Score
    plot_single_metric(
        experiments, "size_b", "overall_score", "se_overall",
        "Overall WattBot Score vs Model Size",
        "Model Size (Proportional to Active Params)", "Overall Score (0.0 - 1.0)",
        "score_overall.png"
    )

    # 2. Value Accuracy
    plot_single_metric(
        experiments, "size_b", "value_accuracy", "se_value",
        "Value Accuracy vs Model Size",
        "Model Size (Active Params)", "Value Accuracy (Exact Match/Range)",
        "score_value.png"
    )

    # 3. Reference Overlap
    plot_single_metric(
        experiments, "size_b", "ref_overlap", "se_ref",
        "Reference Overlap Score vs Model Size",
        "Model Size (Active Params)", "Reference Overlap Score",
        "score_ref.png"
    )

    # 4. Latency
    plot_single_metric(
        experiments, "size_b", "avg_latency", None,
        "Inference Latency vs Model Size",
        "Model Size (Active Params)", "Average Latency (seconds)",
        "latency.png",
        log_y=False 
    )

    # 5. Cost
    # Filter out 0 cost for log plot mostly, but linear scale might be better for cost if 0 exists
    # If using linear scale for cost, it often bunches up. Let's try Linear X, Log Y? 
    # Or just Log-Log. 0 cost models (OSS) need handling. 
    # We will replace 0 with a small epsilon for log plot or just use linear.
    # Let's use Linear for Cost to show the stark difference between $0 and $10.
    plot_single_metric(
        experiments, "size_b", "total_cost", None,
        "Total Benchmark Cost vs Model Size",
        "Model Size (Active Params)", "Total Cost (USD)",
        "cost.png",
        log_y=False # Linear scale shows the massive gap better
    )

if __name__ == "__main__":
    main()
