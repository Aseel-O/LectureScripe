#!/usr/bin/env python3
"""
Research Visualization Suite
Arabic Code-Switched Lecture ASR–NLP Cascade Pipeline
An-Najah National University

Merged from:
  • visualize.py          (Whisper training diagnostics)
  • cascade evaluation    (ROUGE / BERTScore / Code-Switching)

Usage:
    python visualize_research.py

Output — figures/ directory, one PNG per figure at 300 DPI:
  fig01_dataset_composition.png
  fig02_training_scale.png
  fig03_whisper_loss_curves.png
  fig04_whisper_error_epochs.png
  fig05_wer_distribution.png
  fig06_cer_distribution.png
  fig07_asr_summary.png
  fig08_cascade_rouge_l.png
  fig09_rouge_full_breakdown.png
  fig10_bertscore_comparison.png
  fig11_finetune_deltas.png
  fig12_cs_english_ratio.png
  fig13_cs_cmi.png
  fig14_cs_mlu_switch.png
  fig15_eval_heatmap.png
  fig16_tier_comparison.png
"""

import json
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  —  edit paths here if needed
# ═══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = Path("figures")
DPI = 300

PATHS = {
    "trainer_state": "whisper-lora-ar-ps/trainer_state.json",
    "transcripts": "transcripts.json",
    "eval_main": "evaluation_results.json",
    "eval_raw": "eval_internal_raw.json",
    "eval_norm": "eval_internal_norm.json",
    "eval_gold": "eval_system_gold.json",
}

# ─── Dataset statistics (from dataset_raw_statistics.rtf) ───────────────────
# Update this list when new lectures are added
DATASET_STATS = [
    {
        "lecture": "CV_20260222",
        "course": "CV",
        "segments": 294,
        "duration_min": 100.60,
        "used_eval": False,
        "used_train": False,
    },
    {
        "lecture": "CV_20260316",
        "course": "CV",
        "segments": 329,
        "duration_min": 113.16,
        "used_eval": True,
        "used_train": True,
    },
    {
        "lecture": "NLP_20260214",
        "course": "NLP",
        "segments": 390,
        "duration_min": 142.31,
        "used_eval": False,
        "used_train": False,
    },
    {
        "lecture": "NLP_20260221",
        "course": "NLP",
        "segments": 319,
        "duration_min": 115.72,
        "used_eval": True,
        "used_train": True,
    },
    {
        "lecture": "NLP_20260228",
        "course": "NLP",
        "segments": 358,
        "duration_min": 131.07,
        "used_eval": False,
        "used_train": False,
    },
    {
        "lecture": "NLP_20260328",
        "course": "NLP",
        "segments": 391,
        "duration_min": 141.20,
        "used_eval": False,
        "used_train": False,
    },
    {
        "lecture": "NLP_20260404",
        "course": "NLP",
        "segments": 208,
        "duration_min": 74.77,
        "used_eval": False,
        "used_train": False,
    },
    {
        "lecture": "NLP_20260404_2",
        "course": "NLP",
        "segments": 114,
        "duration_min": 41.68,
        "used_eval": False,
        "used_train": False,
    },
]

# Fine-tuning training set scaling experiment
# Update NLP/CV segment counts here if you re-run with different splits
TRAIN_SETS = {
    "Run 1\n100 segs\n(NLP only)": {"NLP_20260221": 100, "CV_20260316": 0},
    "Run 2\n200 segs\n(NLP+CV)": {"NLP_20260221": 121, "CV_20260316": 79},
    "Run 3\n300 segs\n(NLP+CV)": {"NLP_20260221": 221, "CV_20260316": 79},
}

# ─── Colour palette ──────────────────────────────────────────────────────────
PAL = {
    "baseline": "#5B9BD5",
    "finetuned": "#1F3F6E",
    "reference": "#C00000",
    "mBART": "#2E75B6",
    "AraBART": "#ED7D31",
    "Qwen": "#7030A0",
    "nlp_used": "#4472C4",
    "nlp_other": "#A9C4E9",
    "cv_used": "#ED7D31",
    "cv_other": "#FAD7B0",
    "green": "#548235",
    "red": "#C00000",
    "gray": "#808080",
    "tier1": "#4472C4",
    "tier2": "#ED7D31",
    "tier3": "#548235",
}

RUN_STYLE = {
    "Baseline Whisper": ("Baseline", PAL["baseline"]),
    "Fine-tuned Whisper": ("Fine-tuned", PAL["finetuned"]),
    "Reference (upper bound)": ("Reference ↑", PAL["reference"]),
}

MODEL_ORDER = ["mBART", "AraBART", "Qwen"]

# ─── Global rcParams ─────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "legend.framealpha": 0.88,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.12,
    }
)

OUTPUT_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _save(fig, stem):
    p = OUTPUT_DIR / f"{stem}.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"  ✓  {stem}.png")


def _load(key):
    p = PATHS.get(key, key)
    if not Path(p).exists():
        print(f"  [skip] {p} not found — skipping related figure(s)")
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _annotate_bars(ax, rects, fmt="{:.3f}", fs=7.5, pad=0.005):
    for rect in rects:
        h = rect.get_height()
        if np.isnan(h) or h >= 0.999:
            continue
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            h + pad,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fs,
        )


def _get_model_run_val(results, model, run, metric_key="rouge_l"):
    for r in results:
        if r.get("model") == model and r.get("run") == run:
            return r.get("vs_reference", {}).get(metric_key, np.nan)
    return np.nan


def _cs_df(eval_data):
    """Flatten code_switching block to tidy DataFrame."""
    cs = eval_data.get("code_switching", {})
    rows = []
    key_label = {
        "baseline_data": "Baseline",
        "finetuned_data": "Fine-tuned",
        "reference_data": "Reference",
    }
    model_label = {"mbart": "mBART", "arabart": "AraBART", "qwen": "Qwen"}
    for dk, label in key_label.items():
        for mk, ml in model_label.items():
            mdata = cs.get(dk, {}).get("summaries", {}).get(mk, {})
            if not mdata:
                continue
            rows.append(
                {
                    "run": label,
                    "model": ml,
                    "english_ratio": mdata.get("english_ratio", np.nan),
                    "cmi": mdata.get("cmi", np.nan),
                    "mlu_switch": mdata.get("mlu_switch", np.nan),
                }
            )
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 01 — Dataset Composition (8 lectures)
# ═══════════════════════════════════════════════════════════════════════════════


def fig01_dataset_composition():
    print("→ fig01_dataset_composition")
    df = pd.DataFrame(DATASET_STATS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Dataset Composition — All 8 Recorded Lectures\n"
        "An-Najah National University · NLP Course · Palestinian Arabic + English",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )

    labels = [r["lecture"].replace("_", "\n") for r in DATASET_STATS]
    x = np.arange(len(df))
    w = 0.65

    bar_colors = []
    for row in DATASET_STATS:
        if row["course"] == "NLP":
            bar_colors.append(PAL["nlp_used"] if row["used_eval"] else PAL["nlp_other"])
        else:
            bar_colors.append(PAL["cv_used"] if row["used_eval"] else PAL["cv_other"])

    # --- Segment counts ---
    b1 = ax1.bar(
        x, df["segments"], color=bar_colors, edgecolor="white", linewidth=0.5, width=w
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("Number of Segments")
    ax1.set_title("Segments per Lecture")
    ax1.set_ylim(0, df["segments"].max() * 1.2)
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(True)
    for bar, val in zip(b1, df["segments"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 4,
            str(val),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    # --- Duration ---
    b2 = ax2.bar(
        x,
        df["duration_min"],
        color=bar_colors,
        edgecolor="white",
        linewidth=0.5,
        width=w,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Duration (minutes)")
    ax2.set_title("Duration per Lecture")
    ax2.set_ylim(0, df["duration_min"].max() * 1.2)
    ax2.set_axisbelow(True)
    ax2.yaxis.grid(True)
    for bar, val in zip(b2, df["duration_min"]):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.0f}m",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    # Legend
    legend_handles = [
        mpatches.Patch(color=PAL["nlp_used"], label="NLP — used for eval + training"),
        mpatches.Patch(color=PAL["nlp_other"], label="NLP — collected, not yet used"),
        mpatches.Patch(color=PAL["cv_used"], label="CV  — used for eval + training"),
        mpatches.Patch(color=PAL["cv_other"], label="CV  — collected, not yet used"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.10),
        fontsize=9,
    )

    total_segs = df["segments"].sum()
    total_dur = df["duration_min"].sum()
    fig.text(
        0.5,
        -0.02,
        f"Total: {total_segs:,} segments  ·  {total_dur:.0f} min ({total_dur/60:.1f} h)  ·  "
        f"6 NLP lectures  +  2 CV lectures",
        ha="center",
        fontsize=9,
        color="#555",
    )

    plt.tight_layout()
    _save(fig, "fig01_dataset_composition")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 02 — Training Data Scaling Experiment
# ═══════════════════════════════════════════════════════════════════════════════


def fig02_training_scale():
    print("→ fig02_training_scale")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Fine-Tuning Training Data Scaling Experiment", fontsize=12, fontweight="bold"
    )

    labels = list(TRAIN_SETS.keys())
    nlp_vals = [d["NLP_20260221"] for d in TRAIN_SETS.values()]
    cv_vals = [d["CV_20260316"] for d in TRAIN_SETS.values()]
    totals = [n + c for n, c in zip(nlp_vals, cv_vals)]
    x = np.arange(len(labels))

    # --- Stacked bar ---
    b1 = ax1.bar(
        x,
        nlp_vals,
        label="NLP_20260221",
        color=PAL["nlp_used"],
        edgecolor="white",
        width=0.5,
    )
    b2 = ax1.bar(
        x,
        cv_vals,
        label="CV_20260316",
        color=PAL["cv_used"],
        edgecolor="white",
        width=0.5,
        bottom=nlp_vals,
    )

    for i, (n, c, t) in enumerate(zip(nlp_vals, cv_vals, totals)):
        ax1.text(
            i,
            n / 2,
            f"{n}",
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            fontweight="bold",
        )
        if c > 0:
            ax1.text(
                i,
                n + c / 2,
                f"{c}",
                ha="center",
                va="center",
                color="white",
                fontsize=12,
                fontweight="bold",
            )
        ax1.text(
            i, t + 5, f"n={t}", ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Training Segments")
    ax1.set_title("Segment Composition per Run")
    ax1.set_ylim(0, max(totals) * 1.2)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(True)

    # --- Proportion bar ---
    nlp_pct = [n / t * 100 for n, t in zip(nlp_vals, totals)]
    cv_pct = [c / t * 100 if t > 0 else 0 for c, t in zip(cv_vals, totals)]
    ax2.bar(
        x,
        nlp_pct,
        label="NLP_20260221",
        color=PAL["nlp_used"],
        edgecolor="white",
        width=0.5,
    )
    ax2.bar(
        x,
        cv_pct,
        label="CV_20260316",
        color=PAL["cv_used"],
        edgecolor="white",
        width=0.5,
        bottom=nlp_pct,
    )

    for i, (n, c) in enumerate(zip(nlp_pct, cv_pct)):
        ax2.text(
            i,
            n / 2,
            f"{n:.0f}%",
            ha="center",
            va="center",
            color="white",
            fontsize=11,
            fontweight="bold",
        )
        if c > 0:
            ax2.text(
                i,
                n + c / 2,
                f"{c:.0f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=11,
                fontweight="bold",
            )

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Proportion (%)")
    ax2.set_title("Lecture Mix Proportions")
    ax2.set_ylim(0, 115)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.set_axisbelow(True)
    ax2.yaxis.grid(True)

    plt.tight_layout()
    _save(fig, "fig02_training_scale")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 03 — Whisper Training vs Validation Loss
# ═══════════════════════════════════════════════════════════════════════════════


def fig03_whisper_loss():
    print("→ fig03_whisper_loss_curves")
    data = _load("trainer_state")
    if data is None:
        return

    logs = pd.DataFrame(data["log_history"])
    train_logs = logs[logs["loss"].notna()].copy()
    eval_logs = logs[logs["eval_loss"].notna()].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        train_logs["epoch"],
        train_logs["loss"],
        marker="o",
        markersize=3,
        color=PAL["finetuned"],
        label="Train Loss",
        linewidth=1.8,
    )
    ax.plot(
        eval_logs["epoch"],
        eval_logs["eval_loss"],
        marker="s",
        markersize=4,
        color=PAL["reference"],
        label="Validation Loss",
        linestyle="--",
        linewidth=1.8,
    )

    if not eval_logs.empty:
        best_idx = eval_logs["eval_loss"].idxmin()
        best_epoch = eval_logs.loc[best_idx, "epoch"]
        best_val = eval_logs.loc[best_idx, "eval_loss"]
        ax.annotate(
            f"Best val loss\n{best_val:.4f} @ epoch {best_epoch:.0f}",
            xy=(best_epoch, best_val),
            xytext=(best_epoch + 1.2, best_val + 0.04),
            arrowprops=dict(arrowstyle="->", color="#333", lw=1.2),
            fontsize=8.5,
            color="#333",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Whisper LoRA Fine-Tuning — Training vs Validation Loss")
    ax.legend()
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    plt.tight_layout()
    _save(fig, "fig03_whisper_loss_curves")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 04 — WER/CER over Epochs
# ═══════════════════════════════════════════════════════════════════════════════


def fig04_whisper_error_epochs():
    print("→ fig04_whisper_error_epochs")
    data = _load("trainer_state")
    if data is None:
        return

    logs = pd.DataFrame(data["log_history"])
    eval_logs = logs[logs["eval_loss"].notna()].copy()
    has_wer = "eval_wer" in eval_logs.columns
    has_cer = "eval_cer" in eval_logs.columns

    if not has_wer and not has_cer:
        print("  [skip] eval_wer / eval_cer not in trainer_state.json")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    if has_wer:
        ax.plot(
            eval_logs["epoch"],
            eval_logs["eval_wer"],
            marker="o",
            color=PAL["finetuned"],
            label="Eval WER",
            linewidth=1.8,
        )
    if has_cer:
        ax.plot(
            eval_logs["epoch"],
            eval_logs["eval_cer"],
            marker="x",
            color=PAL["reference"],
            label="Eval CER",
            linewidth=1.8,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error Rate")
    ax.set_title("Whisper Fine-Tuning — WER and CER over Training Epochs")
    ax.legend()
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    plt.tight_layout()
    _save(fig, "fig04_whisper_error_epochs")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 05 & 06 — WER / CER Distribution (Base vs Fine-tuned)
# ═══════════════════════════════════════════════════════════════════════════════


def fig05_06_distributions():
    data = _load("transcripts")
    if data is None:
        return

    segs = []
    for lecture in data:
        for seg in lecture.get("segments", []):
            segs.append(
                {
                    "wer_base": seg.get("wer_base", np.nan),
                    "wer_finetuned": seg.get("wer_finetuned", np.nan),
                    "cer_base": seg.get("cer_base", np.nan),
                    "cer_finetuned": seg.get("cer_finetuned", np.nan),
                }
            )
    if not segs:
        return
    df = pd.DataFrame(segs)

    for metric_key, stem, title in [
        (
            "wer",
            "fig05_wer_distribution",
            "WER Distribution — Baseline vs Fine-tuned Whisper",
        ),
        (
            "cer",
            "fig06_cer_distribution",
            "CER Distribution — Baseline vs Fine-tuned Whisper",
        ),
    ]:
        print(f"→ {stem}")
        base_col = f"{metric_key}_base"
        ft_col = f"{metric_key}_finetuned"
        if base_col not in df.columns:
            continue

        df_m = df[[base_col, ft_col]].rename(
            columns={base_col: "Baseline Whisper", ft_col: "Fine-tuned Whisper"}
        )
        df_melt = df_m.melt(var_name="Model", value_name=metric_key.upper())

        fig, ax = plt.subplots(figsize=(6.5, 5))
        palette = {
            "Baseline Whisper": PAL["baseline"],
            "Fine-tuned Whisper": PAL["finetuned"],
        }
        sns.boxplot(
            x="Model",
            y=metric_key.upper(),
            data=df_melt,
            ax=ax,
            palette=palette,
            fliersize=2,
            linewidth=1.2,
            order=["Baseline Whisper", "Fine-tuned Whisper"],
        )

        for i, col in enumerate(["Baseline Whisper", "Fine-tuned Whisper"]):
            med = df_m[col].median()
            mean = df_m[col].mean()
            ax.text(
                i,
                df_m[col].quantile(0.75) + 0.02,
                f"med={med:.3f}\nμ={mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#222",
            )

        ax.set_title(title)
        ax.set_ylabel(metric_key.upper())
        ax.set_axisbelow(True)
        ax.yaxis.grid(True)
        plt.tight_layout()
        _save(fig, stem)


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 07 — ASR Performance Summary (avg WER + CER with improvement annotation)
# ═══════════════════════════════════════════════════════════════════════════════


def fig07_asr_summary():
    print("→ fig07_asr_summary")
    data = _load("transcripts")
    if data is None:
        return

    segs = []
    for lecture in data:
        for seg in lecture.get("segments", []):
            segs.append(
                {
                    "wer_base": seg.get("wer_base", np.nan),
                    "wer_finetuned": seg.get("wer_finetuned", np.nan),
                    "cer_base": seg.get("cer_base", np.nan),
                    "cer_finetuned": seg.get("cer_finetuned", np.nan),
                }
            )
    if not segs:
        return
    df = pd.DataFrame(segs)

    metrics = ["WER", "CER"]
    base_vals = [df["wer_base"].mean(), df["cer_base"].mean()]
    ft_vals = [df["wer_finetuned"].mean(), df["cer_finetuned"].mean()]
    x = np.arange(len(metrics))
    w = 0.3

    fig, ax = plt.subplots(figsize=(6.5, 5))
    b1 = ax.bar(
        x - w / 2,
        base_vals,
        w,
        label="Baseline Whisper",
        color=PAL["baseline"],
        edgecolor="white",
    )
    b2 = ax.bar(
        x + w / 2,
        ft_vals,
        w,
        label="Fine-tuned Whisper",
        color=PAL["finetuned"],
        edgecolor="white",
    )

    for bar, val in zip(list(b1) + list(b2), base_vals + ft_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
        )

    for i, (bv, fv) in enumerate(zip(base_vals, ft_vals)):
        pct_imp = (bv - fv) / bv * 100
        sign = "↓" if pct_imp > 0 else "↑"
        color = PAL["green"] if pct_imp > 0 else PAL["red"]
        ax.annotate(
            "",
            xy=(i + w / 2, fv),
            xytext=(i - w / 2, bv),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
        )
        ax.text(
            i + 0.03,
            (bv + fv) / 2,
            f"{sign}{abs(pct_imp):.1f}%",
            va="center",
            fontsize=9,
            color=color,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Average Error Rate")
    ax.set_title("ASR Performance Summary\nAverage WER and CER: Baseline vs Fine-tuned")
    ax.legend()
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    ax.set_ylim(0, max(base_vals + ft_vals) * 1.3)
    plt.tight_layout()
    _save(fig, "fig07_asr_summary")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 08 — 3-Tier Cascade ROUGE-L (one panel per evaluation tier)
# ═══════════════════════════════════════════════════════════════════════════════


def fig08_cascade_rouge_l():
    print("→ fig08_cascade_rouge_l")
    raw = _load("eval_raw")
    norm = _load("eval_norm")
    gold = _load("eval_gold")
    if None in (raw, norm, gold):
        return

    tier_info = [
        ("Tier 1: Model-Internal (Raw)", raw),
        ("Tier 2: Model-Internal (Normalized)", norm),
        ("Tier 3: Universal Gold Standard", gold),
    ]
    runs_shown = ["Baseline Whisper", "Fine-tuned Whisper"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=False)
    fig.suptitle(
        "ASR–NLP Cascade Effect: Summarization Performance (ROUGE-L)",
        fontsize=13,
        fontweight="bold",
    )

    for ax, (tier_title, data) in zip(axes, tier_info):
        results = data.get("results", [])
        d = {}
        for r in results:
            m = r["model"]
            run = r["run"]
            if m in ("Self", "Qwen") and run == "Reference (upper bound)":
                continue
            if run not in runs_shown:
                continue
            val = r.get("vs_reference", {}).get("rouge_l", np.nan)
            d.setdefault(m, {})[run] = val

        models = [m for m in MODEL_ORDER if m in d]
        x = np.arange(len(models))
        w = 0.35

        for i, run in enumerate(runs_shown):
            label, color = RUN_STYLE[run]
            vals = [d.get(m, {}).get(run, np.nan) for m in models]
            offset = (i - 0.5) * w
            rects = ax.bar(
                x + offset, vals, w, label=label, color=color, edgecolor="white"
            )
            for rect, v in zip(rects, vals):
                if not np.isnan(v):
                    ax.text(
                        rect.get_x() + rect.get_width() / 2,
                        rect.get_height() + 0.012,
                        f"{v:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8.5,
                        fontweight="bold",
                    )

        ax.axhline(
            1.0,
            color=PAL["reference"],
            linestyle="--",
            linewidth=1.2,
            alpha=0.6,
            label="Upper Bound (1.0)",
        )
        ax.set_title(tier_title, fontsize=9.5)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("ROUGE-L Score" if ax is axes[0] else "")
        ax.set_axisbelow(True)
        ax.yaxis.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.06),
        fontsize=9,
    )
    plt.tight_layout()
    _save(fig, "fig08_cascade_rouge_l")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 09 — Full ROUGE Breakdown (R-1, R-2, R-L side by side)
# ═══════════════════════════════════════════════════════════════════════════════


def fig09_rouge_breakdown():
    print("→ fig09_rouge_full_breakdown")
    data = _load("eval_main")
    if data is None:
        return

    results = data.get("results", [])
    runs_shown = ["Baseline Whisper", "Fine-tuned Whisper"]
    metrics = [("rouge_1", "ROUGE-1"), ("rouge_2", "ROUGE-2"), ("rouge_l", "ROUGE-L")]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=False)
    fig.suptitle(
        "Summarization Quality — Full ROUGE Breakdown by Model and ASR Condition",
        fontsize=12,
        fontweight="bold",
    )

    for ax, (mk, ml) in zip(axes, metrics):
        d = {}
        for r in results:
            m = r["model"]
            run = r["run"]
            if m == "Self" or run not in runs_shown:
                continue
            val = r.get("vs_reference", {}).get(mk, np.nan)
            d.setdefault(m, {})[run] = val

        models = [m for m in MODEL_ORDER if m in d]
        x = np.arange(len(models))
        w = 0.35

        for i, run in enumerate(runs_shown):
            label, color = RUN_STYLE[run]
            vals = [d.get(m, {}).get(run, np.nan) for m in models]
            offset = (i - 0.5) * w
            rects = ax.bar(
                x + offset, vals, w, label=label, color=color, edgecolor="white"
            )
            _annotate_bars(ax, rects, fmt="{:.3f}", fs=8)

        ax.set_title(ml)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylim(0, 0.75)
        ax.set_ylabel(ml if ax is axes[0] else "")
        ax.set_axisbelow(True)
        ax.yaxis.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.06),
        fontsize=9,
    )
    plt.tight_layout()
    _save(fig, "fig09_rouge_full_breakdown")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 10 — BERTScore Comparison
# ═══════════════════════════════════════════════════════════════════════════════


def fig10_bertscore():
    print("→ fig10_bertscore_comparison")
    data = _load("eval_main")
    if data is None:
        return

    results = data.get("results", [])
    runs_shown = ["Baseline Whisper", "Fine-tuned Whisper"]
    d = {}
    for r in results:
        m = r["model"]
        run = r["run"]
        if m == "Self" or run not in runs_shown:
            continue
        val = r.get("vs_reference", {}).get("bertscore_f1", np.nan)
        d.setdefault(m, {})[run] = val

    models = [m for m in MODEL_ORDER if m in d]
    x = np.arange(len(models))
    w = 0.3

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, run in enumerate(runs_shown):
        label, color = RUN_STYLE[run]
        vals = [d.get(m, {}).get(run, np.nan) for m in models]
        offset = (i - 0.5) * w
        rects = ax.bar(x + offset, vals, w, label=label, color=color, edgecolor="white")
        _annotate_bars(ax, rects, fmt="{:.4f}", fs=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("BERTScore F1")
    ax.set_title(
        "BERTScore Comparison: All Models × ASR Conditions\n"
        "(AraBERT: aubmindlab/bert-base-arabertv02)"
    )
    ax.set_ylim(0.50, 0.85)
    ax.legend()
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    plt.tight_layout()
    _save(fig, "fig10_bertscore_comparison")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 11 — Fine-tuned vs Baseline Deltas (bar + heatmap)
# ═══════════════════════════════════════════════════════════════════════════════


def fig11_finetune_deltas():
    print("→ fig11_finetune_deltas")
    data = _load("eval_main")
    if data is None:
        return

    deltas = data.get("deltas", {})
    model_map = {
        "mbart_finetuned_vs_baseline": "mBART",
        "arabart_finetuned_vs_baseline": "AraBART",
        "qwen_finetuned_vs_baseline": "Qwen",
    }
    metric_map = {
        "rouge_1_delta": "ROUGE-1",
        "rouge_2_delta": "ROUGE-2",
        "rouge_l_delta": "ROUGE-L",
        "bertscore_delta": "BERTScore",
    }
    metric_order = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore"]

    rows = []
    for key, model in model_map.items():
        d_entry = deltas.get(key, {})
        for mk, ml in metric_map.items():
            rows.append(
                {"model": model, "metric": ml, "delta": d_entry.get(mk, np.nan)}
            )
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="model", columns="metric", values="delta").reindex(
        index=MODEL_ORDER
    )[metric_order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Score Deltas: Fine-tuned vs Baseline Whisper (Δ = FT − Base)",
        fontsize=12,
        fontweight="bold",
    )

    # --- Grouped bar ---
    x = np.arange(len(MODEL_ORDER))
    w = 0.19
    mc = [PAL["tier1"], PAL["tier2"], PAL["tier3"], PAL["Qwen"]]
    for i, (metric, color) in enumerate(zip(metric_order, mc)):
        vals = [
            pivot.loc[m, metric] if m in pivot.index else np.nan for m in MODEL_ORDER
        ]
        offset = (i - 1.5) * w
        rects = ax1.bar(
            x + offset, vals, w, label=metric, color=color, edgecolor="white"
        )
        for rect, v in zip(rects, vals):
            if not np.isnan(v):
                va = "bottom" if v >= 0 else "top"
                pad = 0.003 if v >= 0 else -0.003
                ax1.text(
                    rect.get_x() + rect.get_width() / 2,
                    v + pad,
                    f"{v:+.3f}",
                    ha="center",
                    va=va,
                    fontsize=7,
                )

    ax1.axhline(0, color="black", linewidth=0.9, linestyle="-")
    ax1.set_xticks(x)
    ax1.set_xticklabels(MODEL_ORDER)
    ax1.set_ylabel("Δ Score (positive = improved)")
    ax1.set_title("Score Deltas by Model and Metric")
    ax1.legend(fontsize=8.5)
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(True)

    # --- Heatmap ---
    vmax = max(
        abs(pivot.values[~np.isnan(pivot.values)].max()),
        abs(pivot.values[~np.isnan(pivot.values)].min()),
    )
    sns.heatmap(
        pivot,
        ax=ax2,
        annot=True,
        fmt="+.3f",
        cmap="RdYlGn",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.6,
        annot_kws={"size": 10, "fontweight": "bold"},
        cbar_kws={"label": "Δ Score"},
    )
    ax2.set_title("Delta Heatmap\n(green = improvement, red = degradation)")
    ax2.set_xlabel("")
    ax2.set_ylabel("")

    plt.tight_layout()
    _save(fig, "fig11_finetune_deltas")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 12 — Code-Switching: English Ratio
# ═══════════════════════════════════════════════════════════════════════════════


def fig12_cs_english_ratio():
    print("→ fig12_cs_english_ratio")
    data = _load("eval_main")
    if data is None:
        return
    df = _cs_df(data)
    if df.empty:
        return

    runs = ["Baseline", "Fine-tuned", "Reference"]
    run_colors = [PAL["baseline"], PAL["finetuned"], PAL["reference"]]
    x = np.arange(len(MODEL_ORDER))
    w = 0.25

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, (run, color) in enumerate(zip(runs, run_colors)):
        vals = []
        for m in MODEL_ORDER:
            sub = df[(df["run"] == run) & (df["model"] == m)]
            vals.append(float(sub["english_ratio"].iloc[0]) if len(sub) else np.nan)
        rects = ax.bar(
            x + (i - 1) * w, vals, w, label=run, color=color, edgecolor="white"
        )
        _annotate_bars(ax, rects, fmt="{:.3f}", fs=8)

    # Expected CS range shading
    ax.axhspan(0.10, 0.35, alpha=0.07, color=PAL["green"])
    ax.text(
        len(MODEL_ORDER) - 0.35,
        0.225,
        "Typical CS\nrange (10–35%)",
        fontsize=7.5,
        color=PAL["green"],
        va="center",
        style="italic",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_ylabel("English Token Ratio")
    ax.set_title(
        "Code-Switching: English Ratio in Generated Summaries\n"
        "(fraction of language tokens that are English)"
    )
    ax.set_ylim(0, 0.42)
    ax.legend()
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    plt.tight_layout()
    _save(fig, "fig12_cs_english_ratio")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 13 — Code-Switching: CMI
# ═══════════════════════════════════════════════════════════════════════════════


def fig13_cs_cmi():
    print("→ fig13_cs_cmi")
    data = _load("eval_main")
    if data is None:
        return
    df = _cs_df(data)
    if df.empty:
        return

    runs = ["Baseline", "Fine-tuned", "Reference"]
    run_colors = [PAL["baseline"], PAL["finetuned"], PAL["reference"]]
    x = np.arange(len(MODEL_ORDER))
    w = 0.25

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, (run, color) in enumerate(zip(runs, run_colors)):
        vals = []
        for m in MODEL_ORDER:
            sub = df[(df["run"] == run) & (df["model"] == m)]
            vals.append(float(sub["cmi"].iloc[0]) if len(sub) else np.nan)
        rects = ax.bar(
            x + (i - 1) * w, vals, w, label=run, color=color, edgecolor="white"
        )
        _annotate_bars(ax, rects, fmt="{:.3f}", fs=8)

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_ylabel("Code-Mixing Index (CMI)")
    ax.set_title(
        "Code-Switching: Code-Mixing Index per Model × ASR Condition\n"
        "(Das & Gambäck 2014  ·  0 = monolingual  ·  0.5 = perfectly interleaved)"
    )
    ax.set_ylim(0, 0.40)
    ax.legend()
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    plt.tight_layout()
    _save(fig, "fig13_cs_cmi")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 14 — Code-Switching: MLU-Switch
# ═══════════════════════════════════════════════════════════════════════════════


def fig14_cs_mlu_switch():
    print("→ fig14_cs_mlu_switch")
    data = _load("eval_main")
    if data is None:
        return
    df = _cs_df(data)
    if df.empty:
        return

    runs = ["Baseline", "Fine-tuned", "Reference"]
    run_colors = [PAL["baseline"], PAL["finetuned"], PAL["reference"]]
    x = np.arange(len(MODEL_ORDER))
    w = 0.25

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, (run, color) in enumerate(zip(runs, run_colors)):
        vals = []
        for m in MODEL_ORDER:
            sub = df[(df["run"] == run) & (df["model"] == m)]
            vals.append(float(sub["mlu_switch"].iloc[0]) if len(sub) else np.nan)
        rects = ax.bar(
            x + (i - 1) * w, vals, w, label=run, color=color, edgecolor="white"
        )
        _annotate_bars(ax, rects, fmt="{:.1f}", fs=8)

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_ylabel("MLU-Switch (avg. tokens between language switches)")
    ax.set_title(
        "Code-Switching: Mean Utterance Length Between Language Switches\n"
        "(lower = more frequent switching = more natural code-switching)"
    )
    ax.legend()
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    plt.tight_layout()
    _save(fig, "fig14_cs_mlu_switch")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 15 — Full Evaluation Heatmap (all metrics × all models × both runs)
# ═══════════════════════════════════════════════════════════════════════════════


def fig15_eval_heatmap():
    print("→ fig15_eval_heatmap")
    data = _load("eval_main")
    if data is None:
        return

    results = data.get("results", [])
    runs_shown = ["Baseline Whisper", "Fine-tuned Whisper"]

    rows = []
    for r in results:
        m = r["model"]
        run = r["run"]
        if m == "Self" or run not in runs_shown:
            continue
        short = "Base" if "Baseline" in run else "FT"
        vs = r.get("vs_reference", {})
        rows.append(
            {
                "label": f"{m} ({short})",
                "ROUGE-1": vs.get("rouge_1", np.nan),
                "ROUGE-2": vs.get("rouge_2", np.nan),
                "ROUGE-L": vs.get("rouge_l", np.nan),
                "BERTScore": vs.get("bertscore_f1", np.nan),
            }
        )

    df = pd.DataFrame(rows).set_index("label")
    order = [f"{m} ({r})" for m in MODEL_ORDER for r in ["Base", "FT"]]
    df = df.reindex([l for l in order if l in df.index])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.heatmap(
        df,
        ax=ax,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        vmin=0.0,
        vmax=0.65,
        linewidths=0.6,
        annot_kws={"size": 10, "fontweight": "bold"},
        cbar_kws={"label": "Score"},
    )
    ax.set_title(
        "Evaluation Heatmap — All Models × ASR Conditions\n"
        "(vs. model-specific pseudo-reference on hand-corrected transcripts)"
    )
    ax.set_xlabel("Metric")
    ax.set_ylabel("")
    plt.tight_layout()
    _save(fig, "fig15_eval_heatmap")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 16 — Tier Comparison per Model (how eval framework choice affects scores)
# ═══════════════════════════════════════════════════════════════════════════════


def fig16_tier_comparison():
    print("→ fig16_tier_comparison")
    raw = _load("eval_raw")
    norm = _load("eval_norm")
    gold = _load("eval_gold")
    if None in (raw, norm, gold):
        return

    tier_files = [
        ("Tier 1\n(Raw)", raw, PAL["tier1"]),
        ("Tier 2\n(Norm)", norm, PAL["tier2"]),
        ("Tier 3\n(Gold)", gold, PAL["tier3"]),
    ]
    run_target = "Fine-tuned Whisper"

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)
    fig.suptitle(
        f"Evaluation Framework Comparison — ROUGE-L by Model ({run_target})\n"
        "Tier 1: vs own summary (noisy ASR)  ·  "
        "Tier 2: vs own summary (clean ref)  ·  "
        "Tier 3: vs Qwen summary (clean ref)",
        fontsize=10.5,
        fontweight="bold",
    )

    for ax, model in zip(axes, MODEL_ORDER):
        ax.set_title(model, fontsize=11.5, fontweight="bold")
        x = np.arange(len(tier_files))
        vals = []
        cols = []
        for tier_label, data, color in tier_files:
            res = next(
                (
                    r
                    for r in data.get("results", [])
                    if r.get("run") == run_target and r.get("model") == model
                ),
                None,
            )
            vals.append(
                res.get("vs_reference", {}).get("rouge_l", np.nan) if res else np.nan
            )
            cols.append(color)

        bars = ax.bar(x, vals, color=cols, edgecolor="white", width=0.5)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

        ax.set_xticks(x)
        ax.set_xticklabels([t for t, _, _ in tier_files], fontsize=8.5)
        ax.set_ylim(0, 0.7)
        ax.set_ylabel("ROUGE-L Score" if ax is axes[0] else "")
        ax.set_axisbelow(True)
        ax.yaxis.grid(True)

    # Tier legend patches
    legend_handles = [
        mpatches.Patch(color=c, label=t.replace("\n", " ")) for t, _, c in tier_files
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.07),
        fontsize=9,
    )
    plt.tight_layout()
    _save(fig, "fig16_tier_comparison")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 17 — Transcript-Level Code-Switching: Base vs Fine-tuned vs Reference
#           (exact metrics from transcripts.json)
# ═══════════════════════════════════════════════════════════════════════════════

# Regex helpers for CS tokenisation (mirrors run_evaluate_upgraded.py)
import re as _re

_ARABIC_RE_VIZ = _re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+")
_ENGLISH_RE_VIZ = _re.compile(r"[A-Za-z]{2,}")


def _classify_tokens_viz(text: str):
    return _ARABIC_RE_VIZ.findall(text), _ENGLISH_RE_VIZ.findall(text)


def _compute_cmi_viz(text: str) -> float:
    ar, en = _classify_tokens_viz(text)
    n_ar, n_en = len(ar), len(en)
    total = n_ar + n_en
    return round(1.0 - max(n_ar, n_en) / total, 4) if total else 0.0


def _compute_mlu_switch_viz(text: str) -> float:
    tokens = text.split()
    labels = []
    for tok in tokens:
        if _ARABIC_RE_VIZ.search(tok):
            labels.append("ar")
        elif _ENGLISH_RE_VIZ.search(tok):
            labels.append("en")
    if not labels:
        return 0.0
    runs, current, run_len = [], labels[0], 1
    for lang in labels[1:]:
        if lang == current:
            run_len += 1
        else:
            runs.append(run_len)
            current, run_len = lang, 1
    runs.append(run_len)
    return round(sum(runs) / len(runs), 4)


def _compute_en_ratio_viz(text: str) -> float:
    ar, en = _classify_tokens_viz(text)
    total = len(ar) + len(en)
    return round(len(en) / total, 4) if total else 0.0


def _extract_transcript_text(lecture: dict, variant: str) -> str:
    """Assemble full transcript text for a given lecture and variant.

    variant: 'base' | 'finetuned' | 'reference'
    """
    segments = lecture.get("segments", [])
    field_map = {
        "finetuned": "transcript_finetuned",
        "base": "transcript_base",
        "reference": "reference",
    }
    field = field_map.get(variant, "transcript_finetuned")

    texts = [str(seg.get(field) or "").strip() for seg in segments if seg.get(field)]
    if not texts and variant == "finetuned":
        full = (lecture.get("full_transcript") or "").strip()
        return full
    return " ".join(texts)


def _cs_transcript_df(transcripts: list[dict]) -> pd.DataFrame:
    """Compute per-lecture exact CS metrics for all three transcript variants.

    Returns a tidy DataFrame with columns:
        lecture_id, variant, cmi, mlu_switch, english_ratio
    """
    rows = []
    variant_labels = {
        "base": "Baseline",
        "finetuned": "Fine-tuned",
        "reference": "Reference",
    }
    for lecture in transcripts:
        lid = lecture.get("lecture_id", "?")
        for variant, label in variant_labels.items():
            text = _extract_transcript_text(lecture, variant)
            if not text:
                continue
            rows.append(
                {
                    "lecture_id": lid,
                    "variant": label,
                    "cmi": _compute_cmi_viz(text),
                    "mlu_switch": _compute_mlu_switch_viz(text),
                    "english_ratio": _compute_en_ratio_viz(text),
                }
            )
    return pd.DataFrame(rows)


def fig17_transcript_cs_comparison():
    """Three-panel figure comparing exact transcript CS metrics across ASR variants."""
    print("→ fig17_transcript_cs_comparison")
    transcripts = _load("transcripts")
    if transcripts is None:
        return

    df = _cs_transcript_df(transcripts)
    if df.empty:
        print("  [skip] No transcript CS data available")
        return

    # Aggregate to per-variant means
    summary = (
        df.groupby("variant")[["cmi", "mlu_switch", "english_ratio"]]
        .mean()
        .reset_index()
    )

    variant_order = ["Baseline", "Fine-tuned", "Reference"]
    variant_colors = [PAL["baseline"], PAL["finetuned"], PAL["reference"]]
    summary["variant"] = pd.Categorical(
        summary["variant"], categories=variant_order, ordered=True
    )
    summary = summary.sort_values("variant")

    metrics = [
        (
            "english_ratio",
            "English Token Ratio (EN-ratio)",
            "Fraction of language tokens that are English\n(higher = more English preserved in transcript)",
        ),
        (
            "cmi",
            "Code-Mixing Index (CMI)",
            "Das & Gambäck 2014  ·  0 = monolingual  ·  0.5 = perfectly interleaved",
        ),
        (
            "mlu_switch",
            "MLU-Switch (tokens between language switches)",
            "Mean run length between language switches\n(lower = more frequent switching = more natural code-switching)",
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle(
        "Transcript-Level Code-Switching Metrics — Exact Computation\n"
        "Baseline vs Fine-tuned vs Reference (Hand-corrected) Transcripts",
        fontsize=12,
        fontweight="bold",
    )

    x = np.arange(len(variant_order))
    w = 0.55

    for ax, (col, ylabel, subtitle) in zip(axes, metrics):
        vals = [
            (
                float(summary.loc[summary["variant"] == v, col].iloc[0])
                if v in summary["variant"].values
                else np.nan
            )
            for v in variant_order
        ]
        bars = ax.bar(
            x, vals, w, color=variant_colors, edgecolor="white", linewidth=0.6
        )

        # Value labels
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.015,
                    f"{v:.4f}" if col != "mlu_switch" else f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

        # Delta annotation: Fine-tuned vs Baseline
        base_v = vals[variant_order.index("Baseline")]
        ft_v = vals[variant_order.index("Fine-tuned")]
        if not np.isnan(base_v) and not np.isnan(ft_v) and base_v > 0:
            delta = ft_v - base_v
            pct_delta = delta / base_v * 100
            color_d = (
                PAL["green"]
                if (
                    (col == "english_ratio" and delta >= 0)
                    or (col == "cmi" and delta >= 0)
                    or (col == "mlu_switch" and delta <= 0)
                )
                else PAL["red"]
            )
            ax.annotate(
                (
                    f"FT vs Base\n{delta:+.4f}\n({pct_delta:+.1f}%)"
                    if col != "mlu_switch"
                    else f"FT vs Base\n{delta:+.2f}\n({pct_delta:+.1f}%)"
                ),
                xy=(x[variant_order.index("Fine-tuned")], ft_v),
                xytext=(
                    x[variant_order.index("Fine-tuned")] + 0.45,
                    ft_v + max(vals) * 0.06,
                ),
                fontsize=7.5,
                color=color_d,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=color_d, lw=1.1),
            )

        # Per-lecture scatter (shows spread across lectures)
        for i, v in enumerate(variant_order):
            lec_vals = df.loc[df["variant"] == v, col].values
            if len(lec_vals) > 1:
                ax.scatter(
                    [x[i]] * len(lec_vals),
                    lec_vals,
                    color="#333333",
                    s=28,
                    zorder=5,
                    alpha=0.75,
                    label="per-lecture" if i == 0 else "",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(variant_order, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(subtitle, fontsize=8.5, style="italic")
        ax.set_ylim(0, max(v for v in vals if not np.isnan(v)) * 1.35)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True)

    # Legend for per-lecture dots
    if len(df["lecture_id"].unique()) > 1:
        dot = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#333333",
            markersize=6,
            label="per-lecture value",
        )
        fig.legend(
            handles=[dot], loc="lower right", fontsize=8.5, bbox_to_anchor=(0.99, -0.04)
        )

    # Colour legend for variants
    patch_handles = [
        mpatches.Patch(color=c, label=v) for v, c in zip(variant_order, variant_colors)
    ]
    fig.legend(
        handles=patch_handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.07),
        fontsize=9,
    )

    plt.tight_layout()
    _save(fig, "fig17_transcript_cs_comparison")


def main():
    print("=" * 62)
    print("  Research Visualization Suite")
    print("  An-Najah National University")
    print(f"  Output → {OUTPUT_DIR.resolve()}")
    print("=" * 62)

    # ── Always generated (static dataset stats, no external files needed) ──
    fig01_dataset_composition()
    fig02_training_scale()

    # ── ASR figures (require trainer_state.json + transcripts.json) ────────
    fig03_whisper_loss()
    fig04_whisper_error_epochs()
    fig05_06_distributions()
    fig07_asr_summary()

    # ── Summarization evaluation figures (require eval_*.json) ─────────────
    fig08_cascade_rouge_l()
    fig09_rouge_breakdown()
    fig10_bertscore()
    fig11_finetune_deltas()

    # ── Code-switching figures (require evaluation_results.json) ───────────
    fig12_cs_english_ratio()
    fig13_cs_cmi()
    fig14_cs_mlu_switch()
    fig17_transcript_cs_comparison()  # exact transcript CS from transcripts.json

    # ── Overview / summary figures ──────────────────────────────────────────
    fig15_eval_heatmap()
    fig16_tier_comparison()

    print("=" * 62)
    print(f"  Done — all figures in {OUTPUT_DIR.resolve()}")
    print("=" * 62)


if __name__ == "__main__":
    main()
