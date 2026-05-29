"""
Fine-tuning Training Visualization and Results Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Generates comprehensive visualization plots of model training and evaluation results.
Creates four-panel figure showing:
    1. Training vs. validation loss across epochs
    2. WER and CER error rates as learning curves
    3. WER distribution comparison (box plots)
    4. CER distribution comparison (box plots)

Input files:
    - ./whisper-lora-ar-ps/trainer_state.json (training logs)
    - transcripts.json (transcription results with error metrics)

Output:
    - whisper_comprehensive_results.png (high-resolution comparison figure)
    - Console summary with average error rates
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load training logs from fine-tuning run
with open("./whisper-lora-ar-ps/trainer_state.json", "r") as f:
    trainer_state = json.load(f)

# Load transcription results and error metrics
with open("transcripts.json", "r") as f:
    transcripts_data = json.load(f)

# Convert training logs to DataFrame for easier analysis
logs = pd.DataFrame(trainer_state["log_history"])

# Extract validation loss records
eval_logs = logs[logs["eval_loss"].notna()].copy()

# Process segment-level error metrics for distribution analysis
all_segments = []
for lecture in transcripts_data:
    for seg in lecture["segments"]:
        all_segments.append(
            {
                "wer_base": seg["wer_base"],
                "wer_finetuned": seg["wer_finetuned"],
                "cer_base": seg["cer_base"],
                "cer_finetuned": seg["cer_finetuned"],
            }
        )
df_segments = pd.DataFrame(all_segments)

# Configure plotting style
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Subplot 1: Training and Validation Loss Curves
train_logs = logs[logs["loss"].notna()].copy()
axes[0, 0].plot(
    train_logs["epoch"], train_logs["loss"], label="Train Loss", marker="o", linewidth=2
)
axes[0, 0].plot(
    eval_logs["epoch"],
    eval_logs["eval_loss"],
    label="Validation Loss",
    marker="s",
    linestyle="--",
    linewidth=2,
)
axes[0, 0].set_title(
    "Training vs Validation Loss Across Epochs", fontsize=12, fontweight="bold"
)
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Subplot 2: Learning Curves (WER/CER by Epoch)
axes[0, 1].plot(
    eval_logs["epoch"],
    eval_logs["eval_wer"],
    label="Validation WER",
    color="blue",
    marker="o",
    linewidth=2,
)
# Check if CER is available in logs (not always present)
if "eval_cer" in eval_logs.columns:
    axes[0, 1].plot(
        eval_logs["epoch"],
        eval_logs["eval_cer"],
        label="Validation CER",
        color="red",
        marker="x",
        linewidth=2,
    )
axes[0, 1].set_title(
    "Error Rates (WER/CER) Learning Curves", fontsize=12, fontweight="bold"
)
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Error Rate")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Subplot 3: WER Distribution Comparison (Baseline vs Fine-tuned)
df_wer = df_segments.melt(
    value_vars=["wer_base", "wer_finetuned"], var_name="Model", value_name="WER"
)
df_wer["Model"] = df_wer["Model"].map(
    {"wer_base": "Baseline Whisper", "wer_finetuned": "Fine-tuned Whisper"}
)
sns.boxplot(x="Model", y="WER", data=df_wer, ax=axes[1, 0], palette="Set2")
axes[1, 0].set_title(
    "WER Distribution: Baseline vs Fine-tuned", fontsize=12, fontweight="bold"
)
axes[1, 0].set_ylabel("Word Error Rate (WER)")

# Subplot 4: CER Distribution Comparison (Baseline vs Fine-tuned)
df_cer = df_segments.melt(
    value_vars=["cer_base", "cer_finetuned"], var_name="Model", value_name="CER"
)
df_cer["Model"] = df_cer["Model"].map(
    {"cer_base": "Baseline Whisper", "cer_finetuned": "Fine-tuned Whisper"}
)
sns.boxplot(x="Model", y="CER", data=df_cer, ax=axes[1, 1], palette="Paired")
axes[1, 1].set_title(
    "CER Distribution: Baseline vs Fine-tuned", fontsize=12, fontweight="bold"
)
axes[1, 1].set_ylabel("Character Error Rate (CER)")

# Adjust layout and save
plt.tight_layout()
plt.savefig("whisper_comprehensive_results.png", dpi=300, bbox_inches="tight")
print("✅ Visualization saved to whisper_comprehensive_results.png")

# Print summary statistics
stats = {
    "Avg Baseline WER": df_segments["wer_base"].mean(),
    "Avg Fine-tuned WER": df_segments["wer_finetuned"].mean(),
    "Avg Baseline CER": df_segments["cer_base"].mean(),
    "Avg Fine-tuned CER": df_segments["cer_finetuned"].mean(),
    "WER Improvement": df_segments["wer_base"].mean()
    - df_segments["wer_finetuned"].mean(),
    "CER Improvement": df_segments["cer_base"].mean()
    - df_segments["cer_finetuned"].mean(),
}

print("\n📊 Summary Statistics:")
print("=" * 50)
for key, value in stats.items():
    if "Improvement" in key:
        print(f"{key:.<30} {value:>8.4f} (↓ lower is better)")
    else:
        print(f"{key:.<30} {value:>8.4f}")
print("=" * 50)
