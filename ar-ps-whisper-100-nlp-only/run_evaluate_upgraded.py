"""
run_evaluate.py — Cascade Evaluation
========================================
Measures how transcript quality (baseline vs. fine-tuned Whisper) propagates
into summary quality (mBART, AraBART, and Qwen).

Metrics computed per (run × model) pair:
  - ROUGE-1      — unigram overlap F-measure
  - ROUGE-2      — bigram overlap F-measure
  - ROUGE-L      — longest common subsequence F-measure
  - BERTScore-F1 — contextual embedding similarity (AraBERT)

Supports a dual-reference evaluation methodology:
  1. Internal/Self Reference (--reference-mode self): Each model is compared
     against its OWN summary of the hand-corrected transcripts.
  2. Universal Gold Standard (--reference-mode gold): All models are compared
     against Qwen's summary of the hand-corrected transcripts.

NEW: Supports per-run style selection. The reference run can use a different
     Qwen style (e.g., code-switch) than the baseline/finetuned runs (e.g., notes).
     Use --ref-style to specify the style used for the reference summaries.

Output files:
  evaluation_results.txt  — aligned plain-text table with all four metrics
  evaluation_results.json — structured results including per-model deltas
                            (rouge_1_delta, rouge_2_delta, rouge_l_delta,
                             bertscore_delta) for fine-tuned vs. baseline

Usage:
    python run_evaluate.py [--baseline summaries_baseline.json]
                       [--finetuned summaries_finetuned.json]
                       [--reference summaries_reference.json]
                       [--output-txt evaluation_results.txt]
                       [--output-json evaluation_results.json]
                       [--use-normalized]
                       [--reference-mode self|gold]
                       [--ref-style code-switch]    # NEW: style of reference Qwen summaries

Models evaluated per run:
    mBART   — facebook/mbart-large-cc25   (multilingual seq2seq)
    AraBART — moussaKam/AraBART           (Arabic-only seq2seq)
    Qwen    — Qwen2.5-7B-Instruct         (instruction-tuned LLM, primary lens)

Author: Palestinian Arabic NLP Pipeline
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# ─── Third-party ──────────────────────────────────────────────────────────────
try:
    from rouge_score import rouge_scorer
except ImportError:
    sys.exit("rouge-score not found. Run: pip install rouge-score>=0.1.3")

try:
    from bert_score import score as bertscore_score
except ImportError:
    sys.exit("bert-score not found. Run: pip install bert-score>=0.3.13")

import torch

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_BASELINE = "summaries_baseline.json"
DEFAULT_FINETUNED = "summaries_finetuned.json"
DEFAULT_REFERENCE = "summaries_reference.json"
DEFAULT_TRANSCRIPTS = "transcripts.json"
DEFAULT_OUT_TXT = "evaluation_results.txt"
DEFAULT_OUT_JSON = "evaluation_results.json"
DEFAULT_BERT_MODEL = "aubmindlab/bert-base-arabertv02"
DEFAULT_BERT_BATCH = 16

ROUGE_METRICS = ["rouge1", "rouge2", "rougeL"]
ROUGE_METRIC = "rougeL"  # kept for backwards compat / JSON field name
BERTSCORE_LANG = "ar"

# Terminal / file formatting
COL_W_RUN = 26
COL_W_MODEL = 8
COL_W_SCORE = 9
SEP_THICK = "=" * 82
SEP_THIN = "-" * 82


# ══════════════════════════════════════════════════════════════════════════════
# CODE-SWITCHING METRICS
# ══════════════════════════════════════════════════════════════════════════════
#
# We measure code-switching using two complementary metrics:
#
# 1. CMI — Code-Mixing Index (Das & Gambäck 2014)
#    Captures how interleaved the two languages are at the token level.
#    CMI = 1 - (max(n_ar, n_en) / (n_ar + n_en))   [ignoring punctuation/numbers]
#    Range: 0 (monolingual) → 0.5 (perfectly alternating).
#    A higher CMI means more genuine interleaving rather than one language
#    appearing in a single block.
#
# 2. MLU-switch — Mean Length of Utterance between switches
#    Average number of same-language tokens between each language switch.
#    Low MLU-switch = frequent switches (natural code-switching).
#    High MLU-switch = long runs in one language (code-switching in name only).
#
# Token classification:
#   Arabic  — any token containing Arabic Unicode block characters (U+0600–U+06FF)
#   English — any token of 2+ ASCII letters with no Arabic characters
#   Other   — numbers, punctuation, symbols — excluded from both counts
#
# These metrics are computed over:
#   - Transcripts (stored in key_terms_transcript + transcript_chars from summaries JSON)
#   - Summaries   (the summary_ar field of each model)
#
# For transcripts we only have the char count and key_terms list (the actual
# transcript text is not stored in summaries_*.json), so CMI is approximated
# from the ratio of English key terms to total words estimated from char count.
# For summaries we have the full text and compute exact CMI + MLU-switch.

_ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+")
_ENGLISH_RE = re.compile(r"[A-Za-z]{2,}")


def _classify_tokens(text: str):
    """Return (arabic_tokens, english_tokens) lists from a mixed text string."""
    arabic = _ARABIC_RE.findall(text)
    english = _ENGLISH_RE.findall(text)
    return arabic, english


def compute_cmi(text: str) -> float:
    """Compute Code-Mixing Index for a text string.

    CMI = 1 - max(n_ar, n_en) / (n_ar + n_en)
    Returns 0.0 for monolingual or empty text.
    Range: 0.0 (monolingual) to 0.5 (perfectly interleaved).
    """
    ar, en = _classify_tokens(text)
    n_ar, n_en = len(ar), len(en)
    total = n_ar + n_en
    if total == 0:
        return 0.0
    return round(1.0 - max(n_ar, n_en) / total, 4)


def compute_mlu_switch(text: str) -> float:
    """Compute Mean Length of Utterance between language switches.

    Tokenizes text by whitespace, labels each token as Arabic/English/Other,
    then measures the average run length of consecutive same-language tokens.
    Other tokens (numbers, punctuation) do not start a new run but also do
    not extend the current one — they are skipped.

    Returns 0.0 if no language tokens are found.
    A lower value means more frequent switching (more natural code-switching).
    """
    tokens = text.split()
    labels = []
    for tok in tokens:
        if _ARABIC_RE.search(tok):
            labels.append("ar")
        elif _ENGLISH_RE.search(tok):
            labels.append("en")
        # else: skip (numbers, punctuation)

    if not labels:
        return 0.0

    runs = []
    current_lang = labels[0]
    run_len = 1
    for lang in labels[1:]:
        if lang == current_lang:
            run_len += 1
        else:
            runs.append(run_len)
            current_lang = lang
            run_len = 1
    runs.append(run_len)

    return round(sum(runs) / len(runs), 4)


def compute_english_ratio(text: str) -> float:
    """Fraction of language tokens that are English.

    English ratio = n_en / (n_ar + n_en).
    0.0 = pure Arabic, 1.0 = pure English.
    Expected range for code-switched Arabic AI lectures: 0.10–0.35.
    """
    ar, en = _classify_tokens(text)
    total = len(ar) + len(en)
    if total == 0:
        return 0.0
    return round(len(en) / total, 4)


def cs_metrics_for_text(text: str) -> dict:
    """Return all three code-switching metrics for a text string."""
    return {
        "cmi": compute_cmi(text),
        "mlu_switch": compute_mlu_switch(text),
        "english_ratio": compute_english_ratio(text),
    }


def compute_cs_metrics_for_entries(
    entries: list[dict],
    model_field: str,
    label: str,
) -> dict:
    """Aggregate code-switching metrics across all entries for one model field.

    Args:
        entries:     List of summary JSON entries (one per lecture).
        model_field: Dot-path to the summary text, e.g. \"qwen.summary_ar\".
        label:       Identifier for log messages.

    Returns:
        Dict with mean cmi, mlu_switch, english_ratio, and n_docs.
    """
    cmis, mlus, ratios = [], [], []
    for entry in entries:
        text = str(_nested_get(entry, model_field) or "").strip()
        if not text:
            continue
        cmis.append(compute_cmi(text))
        mlus.append(compute_mlu_switch(text))
        ratios.append(compute_english_ratio(text))

    n = len(cmis)
    if n == 0:
        log.warning("[%s] No text found for CS metrics -- returning zeros", label)
        return {"cmi": 0.0, "mlu_switch": 0.0, "english_ratio": 0.0, "n_docs": 0}

    result = {
        "cmi": round(sum(cmis) / n, 4),
        "mlu_switch": round(sum(mlus) / n, 4),
        "english_ratio": round(sum(ratios) / n, 4),
        "n_docs": n,
    }
    log.info(
        "[%s] CS metrics -> CMI=%.4f  MLU-switch=%.2f  EN-ratio=%.4f  (n=%d)",
        label,
        result["cmi"],
        result["mlu_switch"],
        result["english_ratio"],
        n,
    )
    return result


def compute_transcript_cs_metrics(entries: list[dict], label: str) -> dict:
    """Approximate CS metrics for transcripts using key_terms_transcript.

    The actual transcript text is not stored in summaries_*.json — only
    key_terms_transcript (list of detected English terms) and transcript_chars
    (total character count) are available. We use these to approximate:

      english_ratio ≈ (sum of English term char lengths) / transcript_chars
      cmi           ≈ estimated from english_ratio using CMI formula
      mlu_switch    ≈ not computable without full text — reported as None

    This is clearly approximate but gives a comparable signal for the
    cascade analysis (does better ASR produce more English-preserving transcripts?)
    """
    ratios, cmis = [], []
    for entry in entries:
        terms = entry.get("key_terms_transcript", [])
        chars = entry.get("transcript_chars", 0)
        if not chars:
            continue
        en_chars = sum(len(t) for t in terms if _ENGLISH_RE.search(t))
        ratio = round(en_chars / chars, 4)
        # Approximate CMI: assume Arabic chars ≈ (1 - ratio) fraction of tokens
        # This is a rough estimate; real CMI needs token-level labels.
        cmi_approx = round(1.0 - max(ratio, 1 - ratio), 4) if ratio > 0 else 0.0
        ratios.append(ratio)
        cmis.append(cmi_approx)

    n = len(ratios)
    if n == 0:
        return {
            "cmi_approx": 0.0,
            "english_ratio_approx": 0.0,
            "n_docs": 0,
            "note": "no data",
        }

    result = {
        "cmi_approx": round(sum(cmis) / n, 4),
        "english_ratio_approx": round(sum(ratios) / n, 4),
        "mlu_switch": None,  # not computable without full transcript text
        "n_docs": n,
        "note": "approximated from key_terms_transcript and transcript_chars",
    }
    log.info(
        "[%s] Transcript CS (approx) -> CMI≈%.4f  EN-ratio≈%.4f  (n=%d)",
        label,
        result["cmi_approx"],
        result["english_ratio_approx"],
        n,
    )
    return result


def load_transcripts(path: str) -> Optional[list[dict]]:
    """Load transcripts.json — a list of lecture objects each containing 'segments'."""
    p = Path(path)
    if not p.exists():
        log.warning(
            "Transcripts file not found -- skipping exact transcript CS: %s", path
        )
        return None
    log.info("Loading transcripts: %s", path)
    with p.open(encoding="utf-8") as fh:
        data = json.load(fh)
    log.info("  └─ %d lectures loaded from %s", len(data), p.name)
    return data


def compute_exact_transcript_cs_metrics(
    transcripts: list[dict], variant: str, label: str
) -> dict:
    """Compute exact CS metrics (CMI, MLU-switch, English ratio) from actual transcript text.

    Args:
        transcripts: List of lecture objects loaded from transcripts.json.
                     Each must contain a 'segments' list with per-segment
                     transcript text fields.
        variant:     Which transcript field to use. Either:
                       'finetuned' — uses 'transcript_finetuned' per segment
                                     (or 'full_transcript' at lecture level as fallback)
                       'base'      — assembles text from 'transcript_base' per segment
                       'reference' — uses 'reference' per segment (hand-corrected)
        label:       Identifier for log messages.

    Returns:
        Dict with exact cmi, mlu_switch, english_ratio, n_docs, and method='exact'.
        Falls back to zeroes if no text is found.

    Notes:
        The baseline (base) transcript is assembled by concatenating
        segment['transcript_base'] across all segments in a lecture.
        The fine-tuned transcript uses segment['transcript_finetuned'] (or
        lecture['full_transcript'] when the per-segment field is absent).
        The reference transcript uses segment['reference'] (human-corrected).
        All three variants preserve the original code-switched content faithfully.
    """
    field_map = {
        "finetuned": "transcript_finetuned",
        "base": "transcript_base",
        "reference": "reference",
    }
    if variant not in field_map:
        raise ValueError(f"variant must be one of {list(field_map)}, got '{variant}'")

    seg_field = field_map[variant]
    cmis, mlus, ratios = [], [], []

    for lecture in transcripts:
        lecture_id = lecture.get("lecture_id", "?")
        segments = lecture.get("segments", [])

        if variant == "finetuned" and not any(seg.get(seg_field) for seg in segments):
            # Fall back to lecture-level full_transcript for fine-tuned variant
            text = (lecture.get("full_transcript") or "").strip()
            if text:
                log.debug(
                    "[%s / %s] Using lecture-level full_transcript for finetuned variant",
                    label,
                    lecture_id,
                )
                cmis.append(compute_cmi(text))
                mlus.append(compute_mlu_switch(text))
                ratios.append(compute_english_ratio(text))
            continue

        # Concatenate per-segment texts for this lecture
        texts = [str(seg.get(seg_field) or "").strip() for seg in segments]
        full_text = " ".join(t for t in texts if t)

        if not full_text:
            log.warning(
                "[%s / %s] No '%s' text found for lecture -- skipping",
                label,
                lecture_id,
                seg_field,
            )
            continue

        cmis.append(compute_cmi(full_text))
        mlus.append(compute_mlu_switch(full_text))
        ratios.append(compute_english_ratio(full_text))

    n = len(cmis)
    if n == 0:
        log.warning(
            "[%s] No transcript text found for variant '%s' -- returning zeros",
            label,
            variant,
        )
        return {
            "cmi": 0.0,
            "mlu_switch": 0.0,
            "english_ratio": 0.0,
            "n_docs": 0,
            "method": "exact",
            "variant": variant,
        }

    result = {
        "cmi": round(sum(cmis) / n, 4),
        "mlu_switch": round(sum(mlus) / n, 4),
        "english_ratio": round(sum(ratios) / n, 4),
        "n_docs": n,
        "method": "exact",
        "variant": variant,
    }
    log.info(
        "[%s / %s] Exact transcript CS -> CMI=%.4f  MLU-switch=%.2f  EN-ratio=%.4f  (n=%d)",
        label,
        variant,
        result["cmi"],
        result["mlu_switch"],
        result["english_ratio"],
        n,
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ScorePair:
    """All automatic metric scores for one (hypothesis, reference) comparison.

    Attributes:
        rouge_1:      ROUGE-1 F-measure (unigram overlap).
        rouge_2:      ROUGE-2 F-measure (bigram overlap).
        rouge_l:      ROUGE-L F-measure (longest common subsequence).
        bertscore_f1: BERTScore F1 using AraBERT contextual embeddings.
    """

    rouge_1: float
    rouge_2: float
    rouge_l: float
    bertscore_f1: float


@dataclass
class RunResult:
    """All scores for one (run x model) combination.

    Attributes:
        run_label:       Human-readable run name, e.g. "Baseline Whisper".
        model_label:     Summariser name, e.g. "mBART", "AraBART", "Qwen".
        vs_reference:    Scores when this model's summaries are compared
                         against the reference (self or gold, per --reference-mode).
        vs_counterpart:  Within-run scores when compared against Qwen's output
                         from the same run (None if Qwen is absent).
    """

    run_label: str
    model_label: str
    vs_reference: ScorePair
    vs_counterpart: Optional[ScorePair] = None


@dataclass
class EvaluationOutput:
    baseline_mbart: Optional[RunResult]
    baseline_arabart: Optional[RunResult]
    baseline_qwen: Optional[RunResult]
    finetuned_mbart: Optional[RunResult]
    finetuned_arabart: Optional[RunResult]
    finetuned_qwen: Optional[RunResult]
    reference_qwen: Optional[RunResult]
    reference_mode: str = "self"


# ══════════════════════════════════════════════════════════════════════════════
# DEVICE DETECTION
# ══════════════════════════════════════════════════════════════════════════════


def get_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' based on availability."""
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    log.info("Device selected: %s", device)
    return device


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════


def load_summaries(path: str) -> Optional[list[dict]]:
    p = Path(path)
    if not p.exists():
        log.warning("File not found -- skipping: %s", path)
        return None
    log.info("Loading: %s", path)
    with p.open(encoding="utf-8") as fh:
        data = json.load(fh)
    log.info("  \u2514\u2500 %d entries loaded from %s", len(data), p.name)
    return data


def extract_texts(entries: list[dict], field: str, label: str) -> list[str]:
    missing = 0
    texts: list[str] = []
    for entry in entries:
        val = _nested_get(entry, field)
        if val is None:
            missing += 1
            texts.append("")
        else:
            texts.append(str(val).strip())

    if missing:
        log.warning(
            "[%s] %d / %d entries missing field '%s' -- replaced with ''",
            label,
            missing,
            len(entries),
            field,
        )
    return texts


def _nested_get(d: dict, dotted_key: str):
    """
    Retrieve a value using 'parent.child' dot notation.
    Fallback logic: If the requested field ends in '_normalized' and is missing,
    automatically retry with the raw 'summary_ar' sibling field to ensure
    mBART and AraBART aren't silently dropped during normalized runs.
    """
    keys = dotted_key.split(".")
    obj = d
    for k in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(k)

    if obj is not None and str(obj).strip():
        return obj

    leaf = keys[-1]
    if leaf.endswith("_normalized"):
        fallback_key = ".".join(keys[:-1] + ["summary_ar"])
        fallback_keys = fallback_key.split(".")
        fallback_obj = d
        for k in fallback_keys:
            if not isinstance(fallback_obj, dict):
                return obj
            fallback_obj = fallback_obj.get(k)
        if fallback_obj is not None and str(fallback_obj).strip():
            return fallback_obj

    return obj


def align_by_key(
    source: list[dict], target: list[dict], key: str = "lecture_id"
) -> tuple[list[dict], list[dict]]:
    src_ids = [e.get(key) for e in source]
    tgt_ids = [e.get(key) for e in target]

    if any(i is None for i in src_ids) or any(i is None for i in tgt_ids):
        log.debug("Key '%s' absent in some entries -- using positional alignment.", key)
        n = min(len(source), len(target))
        return source[:n], target[:n]

    tgt_map = {e[key]: e for e in target}
    aligned_src, aligned_tgt = [], []
    for entry in source:
        eid = entry[key]
        if eid in tgt_map:
            aligned_src.append(entry)
            aligned_tgt.append(tgt_map[eid])
        else:
            log.debug("Entry '%s' not found in target -- skipping.", eid)

    log.info("Aligned %d / %d entries by '%s'.", len(aligned_src), len(source), key)
    return aligned_src, aligned_tgt


# ══════════════════════════════════════════════════════════════════════════════
# METRIC COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════


def compute_rouge(
    hypotheses: list[str], references: list[str], label: str
) -> tuple[float, float, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L F-measure averages in one pass.

    Pairs where either the hypothesis or reference is empty/whitespace-only
    are silently skipped. Stemming is disabled so Arabic tokens are compared
    as-is (stemmer does not support Arabic).

    Args:
        hypotheses: List of model-generated summary strings.
        references: List of reference summary strings (same length).
        label:      Identifier used in log messages (e.g. "Baseline/mBART").

    Returns:
        Tuple of (rouge_1, rouge_2, rouge_l), each a float rounded to 4 d.p.
        Returns (0.0, 0.0, 0.0) if no valid pairs exist.
    """
    scorer = rouge_scorer.RougeScorer(ROUGE_METRICS, use_stemmer=False)
    r1_scores, r2_scores, rl_scores = [], [], []
    for hyp, ref in zip(hypotheses, references):
        if not hyp.strip() or not ref.strip():
            continue
        result = scorer.score(ref, hyp)
        r1_scores.append(result["rouge1"].fmeasure)
        r2_scores.append(result["rouge2"].fmeasure)
        rl_scores.append(result["rougeL"].fmeasure)

    if not r1_scores:
        log.warning(
            "[%s] No valid pairs for ROUGE -- returning 0.0 for all variants", label
        )
        return 0.0, 0.0, 0.0

    n = len(r1_scores)
    avg_r1 = round(sum(r1_scores) / n, 4)
    avg_r2 = round(sum(r2_scores) / n, 4)
    avg_rl = round(sum(rl_scores) / n, 4)
    log.info(
        "[%s] ROUGE computed over %d pairs -> R-1: %.4f  R-2: %.4f  R-L: %.4f",
        label,
        n,
        avg_r1,
        avg_r2,
        avg_rl,
    )
    return avg_r1, avg_r2, avg_rl


def compute_bertscore(
    hypotheses: list[str],
    references: list[str],
    model_type: str,
    device: str,
    batch_size: int,
    label: str,
) -> float:
    """Compute average BERTScore F1 over hypothesis/reference pairs.

    Uses ``aubmindlab/bert-base-arabertv02`` by default (layer 9), which is
    well-calibrated for Modern Standard Arabic and code-switched Arabic-English
    text. Empty pairs are filtered before inference.

    Args:
        hypotheses:  List of model-generated summary strings.
        references:  List of reference summary strings (same length).
        model_type:  HuggingFace model ID for BERTScore embeddings.
        device:      Torch device string ("cuda", "mps", or "cpu").
        batch_size:  Number of pairs processed per forward pass.
        label:       Identifier used in log messages.

    Returns:
        Mean BERTScore F1 as a float rounded to 4 d.p.
        Returns 0.0 if no valid pairs exist.
    """
    valid_pairs = [
        (h, r) for h, r in zip(hypotheses, references) if h.strip() and r.strip()
    ]
    if not valid_pairs:
        log.warning("[%s] No valid pairs for BERTScore -- returning 0.0", label)
        return 0.0

    hyps, refs = zip(*valid_pairs)

    log.info(
        "[%s] Running BERTScore over %d pairs (model=%s, device=%s) ...",
        label,
        len(hyps),
        model_type,
        device,
    )
    _, _, F1 = bertscore_score(
        list(hyps),
        list(refs),
        model_type=model_type,
        num_layers=9,
        lang=BERTSCORE_LANG,
        device=device,
        batch_size=batch_size,
        verbose=False,
    )
    avg_f1 = float(F1.mean())
    log.info("[%s] BERTScore-F1 -> %.4f", label, avg_f1)
    return round(avg_f1, 4)


def score_pair(
    hypotheses: list[str],
    references: list[str],
    label: str,
    bert_model: str,
    device: str,
    batch_size: int,
) -> ScorePair:
    """Run all four metrics (ROUGE-1/2/L + BERTScore-F1) for one hypothesis/reference set.

    Args:
        hypotheses:  List of model-generated summary strings.
        references:  Corresponding reference summary strings.
        label:       Identifier passed through to metric log messages.
        bert_model:  HuggingFace model ID for BERTScore.
        device:      Torch device string.
        batch_size:  BERTScore inference batch size.

    Returns:
        A :class:`ScorePair` populated with rouge_1, rouge_2, rouge_l,
        and bertscore_f1.
    """
    r1, r2, rl = compute_rouge(hypotheses, references, label)
    b = compute_bertscore(hypotheses, references, bert_model, device, batch_size, label)
    return ScorePair(rouge_1=r1, rouge_2=r2, rouge_l=rl, bertscore_f1=b)


def _has_data(entries: list[dict], field: str) -> bool:
    return any(str(_nested_get(e, field) or "").strip() for e in entries)


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════


def evaluate_run(
    *,
    run_entries: list[dict],
    ref_entries: list[dict],
    run_label: str,
    bert_model: str,
    device: str,
    batch_size: int,
    field_key: str,
    reference_mode: str,
    ref_style: str = "code-switch",
) -> tuple[Optional[RunResult], Optional[RunResult], Optional[RunResult]]:
    """Evaluate one transcription run (baseline or fine-tuned) across all summarisers.

    For each summariser model present in the data (mBART, AraBART, Qwen),
    computes ROUGE-1, ROUGE-2, ROUGE-L, and BERTScore-F1 against the
    appropriate reference (self or gold, per ``reference_mode``).  Also
    computes within-run cross-scores of mBART and AraBART against Qwen.

    NEW: ref_style parameter controls which Qwen style was used for the
    reference summaries. This is logged for transparency but does not affect
    scoring (the summaries are already generated; we just load them).

    Args:
        run_entries:    Loaded JSON list for the run being evaluated
                        (e.g. summaries_baseline.json).
        ref_entries:    Loaded JSON list for the reference summaries
                        (e.g. summaries_reference.json).
        run_label:      Human-readable label, e.g. "Baseline Whisper".
        bert_model:     HuggingFace model ID for BERTScore.
        device:         Torch device string ("cuda", "mps", or "cpu").
        batch_size:     BERTScore inference batch size.
        field_key:      Summary field to extract, e.g. "summary_ar" or
                        "summary_ar_normalized".
        reference_mode: "self" -- each model scored against its own reference
                        output; "gold" -- all models scored against Qwen's
                        reference output.
        ref_style:      Style used for reference Qwen summaries (e.g. "code-switch",
                        "notes", "msa"). Logged for transparency only.

    Returns:
        Tuple of (mbart_result, arabart_result, qwen_result), any of which
        may be None if that model's data is absent in the run.
    """

    aligned_run, aligned_ref = align_by_key(run_entries, ref_entries)

    has_mbart = _has_data(aligned_run, f"mbart.{field_key}")
    has_arabart = _has_data(aligned_run, f"arabart.{field_key}")
    has_qwen = _has_data(aligned_run, f"qwen.{field_key}")

    log.info(
        "[%s] Models present -> mBART=%s  AraBART=%s  Qwen=%s  (ref_style=%s)",
        run_label,
        "Y" if has_mbart else "N",
        "Y" if has_arabart else "N",
        "Y" if has_qwen else "N",
        ref_style,
    )

    run_mbart = (
        extract_texts(aligned_run, f"mbart.{field_key}", f"{run_label}/mBART")
        if has_mbart
        else []
    )
    run_arabart = (
        extract_texts(aligned_run, f"arabart.{field_key}", f"{run_label}/AraBART")
        if has_arabart
        else []
    )
    run_qwen = (
        extract_texts(aligned_run, f"qwen.{field_key}", f"{run_label}/Qwen")
        if has_qwen
        else []
    )

    # -- Resolve Reference Targets --------------------------------------------
    if reference_mode == "gold":
        log.info(
            "[%s] Universal Gold Standard -> comparing all models to qwen.%s from reference (style=%s).",
            run_label,
            field_key,
            ref_style,
        )
        ref_gold = extract_texts(
            aligned_ref, f"qwen.{field_key}", "Reference/Qwen_Gold"
        )
        ref_mbart = ref_gold
        ref_arabart = ref_gold
        ref_qwen = ref_gold
        target_str = "GoldRef(Qwen)"
    else:
        log.info(
            "[%s] Model-Internal Reference -> comparing models to their OWN output on reference transcripts.",
            run_label,
        )
        ref_mbart = (
            extract_texts(aligned_ref, f"mbart.{field_key}", "Reference/mBART")
            if has_mbart
            else []
        )
        ref_arabart = (
            extract_texts(aligned_ref, f"arabart.{field_key}", "Reference/AraBART")
            if has_arabart
            else []
        )
        ref_qwen = (
            extract_texts(aligned_ref, f"qwen.{field_key}", "Reference/Qwen")
            if has_qwen
            else []
        )
        target_str = "SelfRef"

    # -- Score each present model ---------------------------------------------
    mbart_vs_ref, arabart_vs_ref, qwen_vs_ref = None, None, None

    if has_mbart:
        log.info("-- Scoring %s / mBART vs. %s --", run_label, target_str)
        mbart_vs_ref = score_pair(
            run_mbart,
            ref_mbart,
            f"{run_label}/mBART->{target_str}",
            bert_model,
            device,
            batch_size,
        )

    if has_arabart:
        log.info("-- Scoring %s / AraBART vs. %s --", run_label, target_str)
        arabart_vs_ref = score_pair(
            run_arabart,
            ref_arabart,
            f"{run_label}/AraBART->{target_str}",
            bert_model,
            device,
            batch_size,
        )

    if has_qwen:
        log.info("-- Scoring %s / Qwen vs. %s --", run_label, target_str)
        qwen_vs_ref = score_pair(
            run_qwen,
            ref_qwen,
            f"{run_label}/Qwen->{target_str}",
            bert_model,
            device,
            batch_size,
        )

    # -- Within-run cross-scores ----------------------------------------------
    mbart_vs_qwen, arabart_vs_qwen = None, None

    if has_mbart and has_qwen:
        mbart_vs_qwen = score_pair(
            run_mbart,
            run_qwen,
            f"{run_label}/mBART->Qwen",
            bert_model,
            device,
            batch_size,
        )

    if has_arabart and has_qwen:
        arabart_vs_qwen = score_pair(
            run_arabart,
            run_qwen,
            f"{run_label}/AraBART->Qwen",
            bert_model,
            device,
            batch_size,
        )

    # -- Build Results --------------------------------------------------------
    mbart_result = (
        RunResult(
            run_label=run_label,
            model_label="mBART",
            vs_reference=mbart_vs_ref,
            vs_counterpart=mbart_vs_qwen,
        )
        if has_mbart
        else None
    )

    arabart_result = (
        RunResult(
            run_label=run_label,
            model_label="AraBART",
            vs_reference=arabart_vs_ref,
            vs_counterpart=arabart_vs_qwen,
        )
        if has_arabart
        else None
    )

    qwen_result = (
        RunResult(
            run_label=run_label,
            model_label="Qwen",
            vs_reference=qwen_vs_ref,
            vs_counterpart=(
                ScorePair(
                    rouge_1=mbart_vs_qwen.rouge_1,
                    rouge_2=mbart_vs_qwen.rouge_2,
                    rouge_l=mbart_vs_qwen.rouge_l,
                    bertscore_f1=mbart_vs_qwen.bertscore_f1,
                )
                if mbart_vs_qwen
                else None
            ),
        )
        if has_qwen
        else None
    )

    return mbart_result, arabart_result, qwen_result


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT FORMATTING
# ══════════════════════════════════════════════════════════════════════════════


def _row(run: str, model: str, r1: float, r2: float, rouge: float, bert: float) -> str:
    fmt = "  {run:<{run_w}}  {model:<{model_w}}  {r1:>{score_w}.4f}  {r2:>{score_w}.4f}  {rouge:>{score_w}.4f}    {bert:>{score_w}.4f}"
    return fmt.format(
        run=run,
        run_w=COL_W_RUN,
        model=model,
        model_w=COL_W_MODEL,
        r1=r1,
        r2=r2,
        rouge=rouge,
        bert=bert,
        score_w=COL_W_SCORE,
    )


def build_table(output: EvaluationOutput) -> str:
    ref_desc = (
        "Gold Reference (Qwen on hand-corrected transcripts) -- Universal Gold Standard"
        if output.reference_mode == "gold"
        else "Model-Specific Reference (Each model on hand-corrected transcripts)"
    )

    header_lines = [
        "",
        "Evaluation vs. " + ref_desc,
        SEP_THICK,
        "  {run:<{run_w}}  {model:<{model_w}}  {rouge1:>{score_w}}  {rouge2:>{score_w}}  {rougeL:>{score_w}}    {bert:>{score_w}}".format(
            run="Run",
            run_w=COL_W_RUN,
            model="Model",
            model_w=COL_W_MODEL,
            rouge1="ROUGE-1",
            rouge2="ROUGE-2",
            rougeL="ROUGE-L",
            bert="BERTScore-F1",
            score_w=COL_W_SCORE,
        ),
        SEP_THIN,
    ]
    header = "\n".join(header_lines)

    rows = []
    for result in [
        output.baseline_mbart,
        output.baseline_arabart,
        output.baseline_qwen,
        output.finetuned_mbart,
        output.finetuned_arabart,
        output.finetuned_qwen,
    ]:
        if result is not None:
            rows.append(
                _row(
                    result.run_label,
                    result.model_label,
                    result.vs_reference.rouge_1,
                    result.vs_reference.rouge_2,
                    result.vs_reference.rouge_l,
                    result.vs_reference.bertscore_f1,
                )
            )

    rows.append(SEP_THIN)
    rows.append(
        _row(
            "Reference (upper bound)",
            "Qwen" if output.reference_mode == "gold" else "Self",
            1.0,
            1.0,
            1.0,
            1.0,
        )
    )
    rows.append(SEP_THICK)

    rows.append("")
    rows.append(
        "Within-run model agreement vs Qwen (ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F1):"
    )
    for result in [
        output.baseline_mbart,
        output.baseline_arabart,
        output.finetuned_mbart,
        output.finetuned_arabart,
    ]:
        if result is not None and result.vs_counterpart is not None:
            cp = result.vs_counterpart
            rows.append(
                "  {run:<{run_w}}  {model:<{model_w}}  {r1:.4f} | {r2:.4f} | {rl:.4f} | {bf:.4f}".format(
                    run=result.run_label,
                    run_w=COL_W_RUN,
                    model=result.model_label,
                    model_w=COL_W_MODEL,
                    r1=cp.rouge_1,
                    r2=cp.rouge_2,
                    rl=cp.rouge_l,
                    bf=cp.bertscore_f1,
                )
            )

    return header + "\n" + "\n".join(rows)


def build_interpretation(output: EvaluationOutput) -> str:
    lines = ["", "Interpretation:"]

    for run_data, label_name in [
        (output.finetuned_qwen, "Qwen"),
        (output.finetuned_mbart, "mBART"),
        (output.finetuned_arabart, "AraBART"),
    ]:
        base_data = getattr(output, f"baseline_{label_name.lower()}")
        if run_data is not None and base_data is not None:
            d_r1 = run_data.vs_reference.rouge_1 - base_data.vs_reference.rouge_1
            d_r2 = run_data.vs_reference.rouge_2 - base_data.vs_reference.rouge_2
            d_rl = run_data.vs_reference.rouge_l - base_data.vs_reference.rouge_l
            d_b = (
                run_data.vs_reference.bertscore_f1 - base_data.vs_reference.bertscore_f1
            )
            lines.append(
                "  Fine-tuned vs Baseline (" + label_name + "): "
                "ROUGE-1 "
                + format(d_r1, "+.4f")
                + " | ROUGE-2 "
                + format(d_r2, "+.4f")
                + " | ROUGE-L "
                + format(d_rl, "+.4f")
                + " | BERTScore "
                + format(d_b, "+.4f")
            )
            if label_name == "Qwen":
                direction = "closer to" if d_rl >= 0 else "further from"
                pct = (
                    abs(d_rl / base_data.vs_reference.rouge_l * 100)
                    if base_data.vs_reference.rouge_l > 0
                    else 0.0
                )
                lines.append(
                    "  -> Fine-tuned transcripts produce summaries "
                    + str(round(pct, 1))
                    + "% "
                    + direction
                    + " the reference than baseline transcripts."
                )

    for run_mbart, run_arabart, run_qwen, label in [
        (
            output.baseline_mbart,
            output.baseline_arabart,
            output.baseline_qwen,
            "Baseline",
        ),
        (
            output.finetuned_mbart,
            output.finetuned_arabart,
            output.finetuned_qwen,
            "Fine-tuned",
        ),
    ]:
        present = {
            m.model_label: m.vs_reference.rouge_l
            for m in [run_mbart, run_arabart, run_qwen]
            if m is not None
        }
        if len(present) >= 2:
            best = max(present, key=present.get)
            scores_str = "  |  ".join(
                f"{k}: {v:.4f}" for k, v in sorted(present.items(), key=lambda x: -x[1])
            )
            lines.append(
                "  "
                + label
                + " -- "
                + best
                + " produces summaries closest to the reference. ("
                + scores_str
                + ")"
            )

    if output.reference_mode == "gold":
        lines.append(
            ""
            "  Overall cascade interpretation (Universal Gold Standard):"
            "  All models (mBART, AraBART, Qwen) are scored against the same Gold"
            "  Reference: the Qwen summary from hand-corrected transcripts."
            "  A positive ROUGE-L delta for fine-tuned vs. baseline confirms a cascade"
            "  effect -- cleaner ASR transcripts propagate into higher-quality summaries."
            "  Cross-model comparison is fair: no model benefits from a lenient reference."
            "  AraBART and mBART can be directly ranked against Qwen."
        )
    else:
        lines.append(
            ""
            "  Overall cascade interpretation (Model-Internal / Self-Reference):"
            "  Each model is scored against its OWN summary of the hand-corrected transcripts."
            "  This isolates the impact of ASR noise on the model's structural and stylistic"
            "  consistency. A positive ROUGE-L delta means the fine-tuned acoustic model"
            "  helps the summarizer maintain its native performance."
        )

    return "\n".join(lines)


def _result_to_dict(r: RunResult) -> dict:
    def _sp_dict(sp: ScorePair) -> dict:
        return {
            "rouge_1": sp.rouge_1,
            "rouge_2": sp.rouge_2,
            "rouge_l": sp.rouge_l,
            "bertscore_f1": sp.bertscore_f1,
        }

    return {
        "run": r.run_label,
        "model": r.model_label,
        "vs_reference": _sp_dict(r.vs_reference),
        "vs_counterpart_model": (
            _sp_dict(r.vs_counterpart) if r.vs_counterpart else None
        ),
    }


def build_json_output(output: EvaluationOutput) -> dict:
    results = []
    for attr in [
        "baseline_mbart",
        "baseline_arabart",
        "baseline_qwen",
        "finetuned_mbart",
        "finetuned_arabart",
        "finetuned_qwen",
        "reference_qwen",
    ]:
        val = getattr(output, attr)
        if val is not None:
            results.append(_result_to_dict(val))

    deltas = {}
    for model_key in ["qwen", "mbart", "arabart"]:
        run_data = getattr(output, f"finetuned_{model_key}")
        base_data = getattr(output, f"baseline_{model_key}")
        if run_data is not None and base_data is not None:
            deltas[f"{model_key}_finetuned_vs_baseline"] = {
                "rouge_1_delta": round(
                    run_data.vs_reference.rouge_1 - base_data.vs_reference.rouge_1, 4
                ),
                "rouge_2_delta": round(
                    run_data.vs_reference.rouge_2 - base_data.vs_reference.rouge_2, 4
                ),
                "rouge_l_delta": round(
                    run_data.vs_reference.rouge_l - base_data.vs_reference.rouge_l, 4
                ),
                "bertscore_delta": round(
                    run_data.vs_reference.bertscore_f1
                    - base_data.vs_reference.bertscore_f1,
                    4,
                ),
            }

    pseudo_ref = (
        "Universal Gold Standard -- Qwen summaries on hand-corrected transcripts"
        if output.reference_mode == "gold"
        else "Model-Specific Reference -- Each model evaluated against its own summary on hand-corrected transcripts"
    )

    return {
        "pseudo_reference": pseudo_ref,
        "bertscore_model": DEFAULT_BERT_MODEL,
        "rouge_metric": ROUGE_METRIC,
        "results": results,
        "deltas": deltas,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Day 8: Cascade evaluation -- transcript quality -> summary quality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--baseline", default=DEFAULT_BASELINE, help="Path to summaries_baseline.json"
    )
    p.add_argument(
        "--finetuned",
        default=DEFAULT_FINETUNED,
        help="Path to summaries_finetuned.json",
    )
    p.add_argument(
        "--reference",
        default=DEFAULT_REFERENCE,
        help="Path to summaries_reference.json (pseudo-reference)",
    )
    p.add_argument(
        "--transcripts",
        default=DEFAULT_TRANSCRIPTS,
        help="Path to transcripts.json (enables exact transcript-level CS metrics). "
        "When present, computes CMI/MLU-switch/EN-ratio from actual transcript "
        "text for baseline, fine-tuned, and reference variants. "
        "Falls back to key_terms approximation when file is absent.",
    )
    p.add_argument(
        "--output-txt",
        default=DEFAULT_OUT_TXT,
        help="Terminal table saved to this file",
    )
    p.add_argument(
        "--output-json",
        default=DEFAULT_OUT_JSON,
        help="Structured results saved to this file",
    )
    p.add_argument(
        "--bertscore-model",
        default=DEFAULT_BERT_MODEL,
        help="HuggingFace model ID for BERTScore",
    )
    p.add_argument(
        "--bertscore-batch",
        type=int,
        default=DEFAULT_BERT_BATCH,
        help="Batch size for BERTScore inference",
    )
    p.add_argument(
        "--use-normalized",
        action="store_true",
        help="Evaluate the '_normalized' fields instead of raw summaries",
    )
    p.add_argument(
        "--reference-mode",
        choices=["self", "gold"],
        default="self",
        help="'self' compares models to their own reference; 'gold' compares all to Qwen-reference.",
    )
    # NEW: ref-style parameter for transparency and documentation
    p.add_argument(
        "--ref-style",
        choices=["notes", "code-switch", "msa"],
        default="code-switch",
        help="Style used for reference Qwen summaries. Logged for transparency. "
        "Does not affect scoring logic -- the summaries are already generated. "
        "code-switch (default): best for universal gold standard evaluation.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()

    baseline_data = load_summaries(args.baseline)
    finetuned_data = load_summaries(args.finetuned)
    reference_data = load_summaries(args.reference)
    transcripts_data = load_transcripts(
        args.transcripts
    )  # optional; enables exact transcript CS

    if reference_data is None:
        log.error("Reference file '%s' is required. Aborting.", args.reference)
        sys.exit(1)

    output = EvaluationOutput(
        baseline_mbart=None,
        baseline_arabart=None,
        baseline_qwen=None,
        finetuned_mbart=None,
        finetuned_arabart=None,
        finetuned_qwen=None,
        reference_qwen=None,
        reference_mode=args.reference_mode,
    )

    field_key = "summary_ar_normalized" if args.use_normalized else "summary_ar"

    shared_kwargs = dict(
        ref_entries=reference_data,
        bert_model=args.bertscore_model,
        device=device,
        batch_size=args.bertscore_batch,
        field_key=field_key,
        reference_mode=args.reference_mode,
        ref_style=args.ref_style,
    )

    if baseline_data is not None:
        log.info("==== Evaluating: Baseline Whisper ====")
        output.baseline_mbart, output.baseline_arabart, output.baseline_qwen = (
            evaluate_run(
                run_entries=baseline_data, run_label="Baseline Whisper", **shared_kwargs
            )
        )

    if finetuned_data is not None:
        log.info("==== Evaluating: Fine-tuned Whisper ====")
        output.finetuned_mbart, output.finetuned_arabart, output.finetuned_qwen = (
            evaluate_run(
                run_entries=finetuned_data,
                run_label="Fine-tuned Whisper",
                **shared_kwargs,
            )
        )

    output.reference_qwen = RunResult(
        run_label="Reference (upper bound)",
        model_label="Qwen" if args.reference_mode == "gold" else "Self",
        vs_reference=ScorePair(rouge_1=1.0, rouge_2=1.0, rouge_l=1.0, bertscore_f1=1.0),
    )

    full_report = build_table(output) + "\n" + build_interpretation(output) + "\n"

    # ── Code-Switching Metrics ────────────────────────────────────────────────
    # Computed separately from ROUGE/BERTScore since they measure a different
    # dimension: how well English technical terminology is preserved across the
    # ASR → summarization cascade, not just lexical/semantic overlap.
    #
    # Transcript CS metrics are computed in two ways:
    #   1. Exact   — when --transcripts points to transcripts.json, the actual
    #                transcript text (per segment) is used for all three variants
    #                (baseline / fine-tuned / reference). Produces CMI, MLU-switch,
    #                and EN-ratio with full precision.
    #   2. Approx  — fallback when transcripts.json is absent; uses
    #                key_terms_transcript + transcript_chars from summaries_*.json
    #                to estimate EN-ratio and CMI. MLU-switch is unavailable.
    log.info("==== Computing Code-Switching Metrics ====")
    cs_section_lines = [
        "",
        SEP_THICK,
        "  CODE-SWITCHING METRICS",
        "  (CMI = Code-Mixing Index [0=monolingual, 0.5=perfectly interleaved])",
        "  (MLU-switch = Mean tokens between language switches [lower = more switching])",
        "  (EN-ratio = fraction of language tokens that are English)",
        SEP_THICK,
    ]

    cs_json = {}

    # ── Exact transcript-level CS metrics (from transcripts.json) ─────────────
    transcript_cs_json: dict = {}
    if transcripts_data is not None:
        log.info("==== Exact Transcript CS Metrics (from transcripts.json) ====")
        cs_section_lines.append("")
        cs_section_lines.append(
            "  TRANSCRIPT-LEVEL CS METRICS (exact, from transcripts.json)"
        )
        cs_section_lines.append("  " + "-" * 76)

        # variant → (label_for_display, data_key_for_json)
        transcript_variants = [
            ("base", "Baseline Whisper", "baseline_transcript"),
            ("finetuned", "Fine-tuned Whisper", "finetuned_transcript"),
            ("reference", "Reference (hand-corrected)", "reference_transcript"),
        ]

        for variant, run_label_tr, data_key_tr in transcript_variants:
            cs = compute_exact_transcript_cs_metrics(
                transcripts_data, variant, f"Transcript/{run_label_tr}"
            )
            cs_section_lines.append(
                f"  {run_label_tr:<28}  CMI={cs['cmi']:.4f}  "
                f"EN-ratio={cs['english_ratio']:.4f}  "
                f"MLU-switch={cs['mlu_switch']:.2f}  (n={cs['n_docs']})"
            )
            transcript_cs_json[data_key_tr] = {
                "run_label": run_label_tr,
                **cs,
            }

        # Delta: does fine-tuned ASR preserve English terminology better?
        base_tr = transcript_cs_json.get("baseline_transcript", {})
        ft_tr = transcript_cs_json.get("finetuned_transcript", {})
        if base_tr and ft_tr:
            delta_en = round(ft_tr["english_ratio"] - base_tr["english_ratio"], 4)
            delta_cmi = round(ft_tr["cmi"] - base_tr["cmi"], 4)
            delta_mlu = round(ft_tr["mlu_switch"] - base_tr["mlu_switch"], 4)
            cs_section_lines.append("")
            cs_section_lines.append(
                f"  Fine-tuned vs Baseline transcript delta:  "
                f"EN-ratio {delta_en:+.4f}  CMI {delta_cmi:+.4f}  MLU-switch {delta_mlu:+.2f}"
            )
            transcript_cs_json["transcript_ft_vs_base_delta"] = {
                "english_ratio_delta": delta_en,
                "cmi_delta": delta_cmi,
                "mlu_switch_delta": delta_mlu,
            }

        cs_section_lines.append("  " + "-" * 76)

    else:
        log.info(
            "No transcripts.json provided -- transcript CS will use key_terms approximation"
        )

    # ── Summary-level CS + approximate transcript CS (from summaries_*.json) ──
    all_cs_data = [
        ("baseline_data", baseline_data, "Baseline Whisper"),
        ("finetuned_data", finetuned_data, "Fine-tuned Whisper"),
        ("reference_data", reference_data, "Reference (hand-corrected)"),
    ]

    for data_key, data, run_label in all_cs_data:
        if data is None:
            continue
        cs_section_lines.append(f"\n  [{run_label}]")

        # Transcript-level (approximate — only key_terms and char counts available)
        tr_cs = compute_transcript_cs_metrics(data, f"{run_label}/transcript")
        cs_section_lines.append(
            f"  Transcript  (approx): CMI≈{tr_cs['cmi_approx']:.4f}  "
            f"EN-ratio≈{tr_cs['english_ratio_approx']:.4f}  "
            f"MLU-switch=N/A  (n={tr_cs['n_docs']})"
        )

        # Summary-level per model
        model_cs = {}
        for model_key, model_label, field in [
            ("mbart", "mBART", "mbart.summary_ar"),
            ("arabart", "AraBART", "arabart.summary_ar"),
            ("qwen", "Qwen", "qwen.summary_ar"),
        ]:
            if not _has_data(data, field):
                continue
            cs = compute_cs_metrics_for_entries(
                data, field, f"{run_label}/{model_label}"
            )
            cs_section_lines.append(
                f"  {model_label:<8}  summary:       CMI={cs['cmi']:.4f}  "
                f"EN-ratio={cs['english_ratio']:.4f}  "
                f"MLU-switch={cs['mlu_switch']:.2f}  (n={cs['n_docs']})"
            )
            model_cs[model_key] = cs

        cs_json[data_key] = {
            "run_label": run_label,
            "transcript": tr_cs,
            "summaries": model_cs,
        }

    # Delta: does fine-tuned ASR produce more English-preserving summaries?
    if "baseline_data" in cs_json and "finetuned_data" in cs_json:
        cs_section_lines.append("")
        cs_section_lines.append(
            "  Fine-tuned vs Baseline delta (EN-ratio in summaries):"
        )
        for model_key, model_label in [
            ("mbart", "mBART"),
            ("arabart", "AraBART"),
            ("qwen", "Qwen"),
        ]:
            base_cs = cs_json["baseline_data"]["summaries"].get(model_key)
            ft_cs = cs_json["finetuned_data"]["summaries"].get(model_key)
            if base_cs and ft_cs:
                delta_en = round(ft_cs["english_ratio"] - base_cs["english_ratio"], 4)
                delta_cmi = round(ft_cs["cmi"] - base_cs["cmi"], 4)
                cs_section_lines.append(
                    f"  {model_label:<8}  EN-ratio {delta_en:+.4f}  CMI {delta_cmi:+.4f}"
                )

    cs_section_lines.append(SEP_THICK)
    cs_report = "\n".join(cs_section_lines) + "\n"
    full_report += cs_report

    print(full_report)

    Path(args.output_txt).write_text(full_report, encoding="utf-8")
    log.info("Saved plain-text report -> %s", args.output_txt)

    json_data = build_json_output(output)
    json_data["code_switching"] = cs_json
    if transcript_cs_json:
        json_data["transcript_cs"] = transcript_cs_json
    Path(args.output_json).write_text(
        json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("Saved structured JSON    -> %s", args.output_json)
    log.info("Evaluation complete.")


if __name__ == "__main__":
    main()
