"""
Cascade Evaluation Visualization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Generates bar charts comparing summarization model performance (ROUGE-L scores)
across three evaluation tiers (raw, normalized, gold standard) and three models
(mBART, AraBART, Qwen).

This visualization demonstrates how ASR quality (baseline vs. fine-tuned Whisper)
cascades through the NLP pipeline into downstream summarization performance.

Output:
    cascade_evaluation_chart_with_ref.png — High-resolution comparison chart
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Evaluation data files for each tier
FILES = {
    "Tier 1: Model-Internal (Raw)": "eval_internal_raw.json",
    "Tier 2: Model-Internal (Normalized)": "eval_internal_norm.json",
    "Tier 3: Universal Gold Standard": "eval_system_gold.json",
}


def load_json(filepath):
    """
    Load and parse a JSON file with error handling.

    Args:
        filepath (str): Path to JSON file.

    Returns:
        dict: Parsed JSON object, or None if file not found.
    """
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_scores(data, metric="rouge_l"):
    """
    Extract ROUGE-L scores for baseline and fine-tuned models across all models.

    Args:
        data (dict): Evaluation results containing "results" list.
        metric (str): Metric name to extract (default: "rouge_l").

    Returns:
        tuple: (model_names_list, baseline_scores_list, finetuned_scores_list)
    """
    models = ["mBART", "AraBART", "Qwen"]
    baseline_scores = []
    finetuned_scores = []

    # Extract scores from evaluation results
    results = data.get("results", [])

    for model in models:
        # Find baseline score for this model
        b_res = next(
            (
                r
                for r in results
                if r["run"] == "Baseline Whisper" and r["model"] == model
            ),
            None,
        )
        baseline_scores.append(b_res["vs_reference"][metric] if b_res else 0.0)

        # Find fine-tuned score for this model
        f_res = next(
            (
                r
                for r in results
                if r["run"] == "Fine-tuned Whisper" and r["model"] == model
            ),
            None,
        )
        finetuned_scores.append(f_res["vs_reference"][metric] if f_res else 0.0)

    return models, baseline_scores, finetuned_scores


# Load and prepare data for all evaluation tiers
eval_data = {title: load_json(path) for title, path in FILES.items()}

# Create figure with three subplots (one per evaluation tier)
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
fig.suptitle(
    "ASR-NLP Cascade Effect: Summarization Performance (ROUGE-L)",
    fontsize=18,
    fontweight="bold",
    y=1.05,
)

# Color scheme: light blue for baseline, dark blue for fine-tuned
colors = ["#8b9dc3", "#3b5998"]

# Plot data for each evaluation tier
for ax, (title, data) in zip(axes, eval_data.items()):
    if not data:
        ax.set_title(f"{title}\n(Data Missing)")
        continue

    models, baseline, finetuned = extract_scores(data, metric="rouge_l")

    # Setup bar positions and width
    x = np.arange(len(models))
    width = 0.35

    # Plot baseline and fine-tuned bars
    rects1 = ax.bar(
        x - width / 2,
        baseline,
        width,
        label="Baseline Whisper",
        color=colors[0],
        edgecolor="black",
    )
    rects2 = ax.bar(
        x + width / 2,
        finetuned,
        width,
        label="Fine-tuned Whisper",
        color=colors[1],
        edgecolor="black",
    )

    # Add reference line at perfect score
    ax.axhline(
        y=1.0, color="red", linestyle="--", linewidth=1.5, label="Upper Bound (1.0)"
    )

    # Configure axes and labels
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_xlabel("Summarization Model", fontsize=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Add value labels on bars
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

# Configure shared y-axis label and legend
axes[0].set_ylabel("ROUGE-L Score", fontsize=14)
axes[0].legend(loc="upper left", fontsize=10)
plt.tight_layout()

# Save figure at high resolution
plt.savefig("cascade_evaluation_chart_with_ref.png", dpi=300, bbox_inches="tight")
print("Chart saved successfully to cascade_evaluation_chart_with_ref.png")
