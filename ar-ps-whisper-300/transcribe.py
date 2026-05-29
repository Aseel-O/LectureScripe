#!/usr/bin/env python3
"""
Stage 1 — Transcription + Baseline Comparison
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Runs BOTH the vanilla Whisper large-v3 (baseline) AND your fine-tuned
model on all audio segments, then computes WER and CER for each against
the hand-corrected reference transcripts.

This gives you the comparison table your report needs:
  "fine-tuned WER/CER vs baseline WER/CER"

Output JSON per segment:
  {
    "segment_id":           "segment_001",
    "audio_path":           "data/audio/segment_001.wav",
    "duration_s":           18.4,
    "reference":            "hand-corrected transcript",    ← from dataset_final.jsonl
    "transcript_base":      "vanilla Whisper output",       ← baseline
    "transcript_finetuned": "fine-tuned Whisper output",    ← your model
    "wer_base":             0.61,
    "wer_finetuned":        0.29,
    "wer_improvement":      0.32,                           ← positive = improvement
    "wer_base_norm":        0.55,                           ← after Arabic normalisation
    "wer_finetuned_norm":   0.25,
    "wer_improvement_norm": 0.30,
    "cer_base":             0.52,                           ← Character Error Rate
    "cer_finetuned":        0.31,
    "cer_improvement":      0.21,                           ← positive = improvement
    "cer_base_norm":        0.45,
    "cer_finetuned_norm":   0.28,
    "cer_improvement_norm": 0.17,
    "flagged":              false,
    "flag_reasons":         []
  }

Per-lecture wer_summary also includes:
  "avg_cer_base", "avg_cer_finetuned", "avg_cer_improvement"

Usage:
    python transcribe.py                          # uses defaults
    python transcribe.py --no-baseline            # skip baseline (faster)
    python transcribe.py --model ./whisper-merged-ar-ps --audio data/audio
    python transcribe.py --out results.json --jsonl data/dataset_final.jsonl
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import re
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import torch
import librosa
import numpy as np
import jiwer
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL_DIR = "./whisper-merged-ar-ps"
BASELINE_MODEL_ID = "openai/whisper-large-v3"
DEFAULT_AUDIO_DIR = "data/audio"
DEFAULT_OUTPUT = "transcripts.json"
DEFAULT_JSONL = "data/dataset_final.jsonl"
LANGUAGE = "arabic"
TASK = "transcribe"
SAMPLING_RATE = 16_000
MAX_NEW_TOKENS = 225


# ── Device ────────────────────────────────────────────────────────────────────
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Audio loading ─────────────────────────────────────────────────────────────
def load_audio(path: str, sr: int = SAMPLING_RATE) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sr, mono=True, dtype=np.float32)
    return audio


# ── Load one Whisper model ────────────────────────────────────────────────────
def load_whisper(model_id: str, device: str, label: str) -> tuple:
    logger.info(f"Loading {label} ({model_id}) …")
    processor = WhisperProcessor.from_pretrained(model_id, language=LANGUAGE, task=TASK)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model = model.to(device)
    if device != "cuda":
        model = model.float()
    logger.info(f"{label} loaded.")
    return processor, model


# ── Transcribe one segment ────────────────────────────────────────────────────
def transcribe_segment(
    audio_arr: np.ndarray,
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    device: str,
) -> str:
    dtype = torch.float16 if device == "cuda" else torch.float32
    input_features = processor.feature_extractor(
        audio_arr,
        sampling_rate=SAMPLING_RATE,
        return_tensors="pt",
    ).input_features.to(device, dtype=dtype)

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            language=LANGUAGE,
            task=TASK,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    return processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[
        0
    ].strip()


# ── WER / CER computation ─────────────────────────────────────────────────────
def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate using jiwer.

    Args:
        reference:  Ground-truth transcript string.
        hypothesis: Model-generated transcript string.

    Returns:
        WER as a float (0.0–1.0+); values above 1.0 are possible when
        there are more edits than reference words.
        Returns 1.0 if reference is empty or on any computation error.
    """
    if not reference.strip():
        return 1.0
    try:
        return round(jiwer.wer(reference, hypothesis), 4)
    except Exception:
        return 1.0


# ── CER computation ───────────────────────────────────────────────────────────
def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate using jiwer.cer.

    Operates at the character level rather than the word level, making it
    more sensitive to partial-word transcription errors and especially
    useful for Arabic, where diacritics and character-level differences
    matter for code-switched speech.

    Args:
        reference:  Ground-truth transcript string.
        hypothesis: Model-generated transcript string.

    Returns:
        CER as a float (0.0–1.0+); values above 1.0 are possible when
        there are more character edits than reference characters.
        Returns 1.0 if reference is empty or on any computation error.
    """
    if not reference.strip():
        return 1.0
    try:
        return round(jiwer.cer(reference, hypothesis), 4)
    except Exception:
        return 1.0


# ── Arabic normalization for WER/CER (same rules as fine-tuning script) ───────
def normalize_arabic(text: str) -> str:
    """Normalise an Arabic/code-switched string before metric computation.

    Applies the same normalisation used during fine-tuning so that WER and
    CER scores are comparable across training and evaluation:
      - Strips diacritics (tashkeel) and tatweel.
      - Unifies alef variants (أ إ آ ٱ → ا).
      - Maps ئ → ي, ؤ → و, ة → ه, ى → ي.
      - Removes Arabic punctuation and non-Arabic/non-Latin characters.
      - Lowercases Latin tokens (detected by >50 % ASCII-alpha ratio).
    """
    tokens = text.split()
    normalized = []
    for token in tokens:
        latin = sum(1 for c in token if c.isascii() and c.isalpha())
        alpha = sum(1 for c in token if c.isalpha())
        if alpha > 0 and latin / alpha > 0.5:
            normalized.append(token.lower())
        else:
            t = token
            t = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670]", "", t)
            t = re.sub(r"[أإآٱ]", "ا", t)
            t = t.replace("ئ", "ي").replace("ؤ", "و")
            t = t.replace("ـ", "")
            t = t.replace("ة", "ه")
            t = t.replace("ى", "ي")
            t = re.sub(r'[،؛؟!,;?!.:\'"()\[\]{}\-/\\]', "", t)
            t = re.sub(r"[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FFa-zA-Z0-9]", "", t)
            if t:
                normalized.append(t)
    return " ".join(normalized)


def compute_wer_norm(reference: str, hypothesis: str) -> float:
    """Compute WER after applying Arabic normalisation to both strings.

    Use this alongside the raw ``compute_wer`` result to separate genuine
    transcription errors from superficial orthographic variation (diacritics,
    alef variants, ta-marbuta, etc.).
    """
    return compute_wer(normalize_arabic(reference), normalize_arabic(hypothesis))


# ── Auto-flagging ─────────────────────────────────────────────────────────────
def repetition_ratio(text: str, window: int = 5) -> float:
    words = text.split()
    if len(words) < window * 2:
        return 0.0
    ngrams = [tuple(words[i : i + window]) for i in range(len(words) - window)]
    if not ngrams:
        return 0.0
    repeated = sum(1 for i, ng in enumerate(ngrams[:-1]) if ng == ngrams[i + 1])
    return repeated / len(ngrams)


def arabic_ratio(text: str) -> float:
    arabic = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
    alpha = sum(1 for c in text if c.isalpha())
    return arabic / alpha if alpha > 0 else 0.0


def auto_flag(transcript: str, reference: str, duration_s: float) -> tuple:
    """Heuristically flag a transcript segment for manual review.

    Checks for common failure modes of Whisper on Arabic/code-switched audio:
      - Empty or near-empty output       ("empty_transcript")
      - Repetition loops                 ("repetition_loop")
      - Missing Arabic content           ("no_arabic")
      - Abnormally low words-per-second  ("too_sparse")
      - Abnormally high words-per-second ("too_dense")
      - High Character Error Rate > 0.60 ("high_cer")  ← requires a reference

    Returns:
        (flagged: bool, reasons: list[str])
    """
    reasons = []

    if len(transcript.strip()) < 5:
        reasons.append("empty_transcript")

    rep = repetition_ratio(transcript)
    if rep > 0.35:
        reasons.append(f"repetition_loop (ratio={rep:.2f})")

    ar = arabic_ratio(transcript)
    if ar < 0.10 and len(transcript.strip()) > 10:
        reasons.append(f"no_arabic (arabic_ratio={ar:.2f})")

    wps = len(transcript.split()) / duration_s if duration_s > 0 else 0
    if wps < 0.3:
        reasons.append(f"too_sparse (wps={wps:.2f})")
    elif wps > 7.0:
        reasons.append(f"too_dense (wps={wps:.2f})")

    if reference.strip():
        cer = jiwer.cer(reference, transcript)
        if cer > 0.60:
            reasons.append(f"high_cer (cer={cer:.2f})")

    return len(reasons) > 0, reasons


# ── Load metadata from JSONL ──────────────────────────────────────────────────
def load_metadata(jsonl_path: str) -> dict:
    meta = {}
    if not os.path.exists(jsonl_path):
        logger.warning(f"JSONL not found at {jsonl_path} — no references available.")
        return meta
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "audio" in rec and isinstance(rec["audio"], dict):
                audio_path = rec["audio"]["path"]
            elif "audio.path" in rec:
                audio_path = rec["audio.path"]
            else:
                continue
            meta[os.path.basename(audio_path)] = rec
    return meta


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_DIR, help="Fine-tuned model path"
    )
    parser.add_argument(
        "--baseline", default=BASELINE_MODEL_ID, help="Baseline model ID"
    )
    parser.add_argument(
        "--no-baseline", action="store_true", help="Skip baseline model (faster)"
    )
    parser.add_argument("--audio", default=DEFAULT_AUDIO_DIR, help="Audio directory")
    parser.add_argument("--out", default=DEFAULT_OUTPUT, help="Output JSON path")
    parser.add_argument(
        "--jsonl", default=DEFAULT_JSONL, help="Dataset JSONL for references"
    )
    parser.add_argument("--lecture", default=None, help="Lecture ID override")
    args = parser.parse_args()

    device = get_device()
    logger.info(f"Device : {device}")
    logger.info(f"Model  : {args.model}")
    if not args.no_baseline:
        logger.info(f"Baseline: {args.baseline}")

    # ── Load fine-tuned model ─────────────────────────────────────────────────
    proc_ft, model_ft = load_whisper(args.model, device, "Fine-tuned Whisper")

    # ── Load baseline model ───────────────────────────────────────────────────
    proc_base = model_base = None
    if not args.no_baseline:
        proc_base, model_base = load_whisper(args.baseline, device, "Baseline Whisper")

    # ── Find audio files ──────────────────────────────────────────────────────
    audio_dir = Path(args.audio)
    wav_files = sorted(audio_dir.glob("*.wav"))
    if not wav_files:
        logger.error(f"No .wav files found in {audio_dir}")
        return
    logger.info(f"Found {len(wav_files)} audio files.")

    meta = load_metadata(args.jsonl)

    # ── Transcribe all segments ───────────────────────────────────────────────
    segments = []
    for wav_path in tqdm(wav_files, desc="Transcribing"):
        fname = wav_path.name
        rec = meta.get(fname, {})

        audio_arr = load_audio(str(wav_path))
        duration_s = round(len(audio_arr) / SAMPLING_RATE, 2)
        reference = rec.get("sentence", "")

        # Fine-tuned transcript
        transcript_ft = transcribe_segment(audio_arr, model_ft, proc_ft, device)

        # Baseline transcript
        transcript_base = ""
        if model_base is not None:
            transcript_base = transcribe_segment(
                audio_arr, model_base, proc_base, device
            )

        # WER for both (raw and normalized)
        wer_ft = compute_wer(reference, transcript_ft) if reference else None
        wer_ft_norm = compute_wer_norm(reference, transcript_ft) if reference else None
        wer_base = (
            compute_wer(reference, transcript_base)
            if reference and transcript_base
            else None
        )
        wer_base_norm = (
            compute_wer_norm(reference, transcript_base)
            if reference and transcript_base
            else None
        )

        # Improvement = baseline WER - finetuned WER (positive = better)
        wer_improvement = (
            round(wer_base - wer_ft, 4)
            if (wer_base is not None and wer_ft is not None)
            else None
        )
        wer_improvement_norm = (
            round(wer_base_norm - wer_ft_norm, 4)
            if (wer_base_norm is not None and wer_ft_norm is not None)
            else None
        )

        # CER for both (raw) — character-level complement to WER
        cer_finetuned = compute_cer(reference, transcript_ft) if reference else None
        cer_finetuned_norm = (
            compute_cer(normalize_arabic(reference), normalize_arabic(transcript_ft))
            if reference
            else None
        )
        cer_base = (
            compute_cer(reference, transcript_base)
            if reference and transcript_base
            else None
        )
        cer_base_norm = (
            compute_cer(normalize_arabic(reference), normalize_arabic(transcript_base))
            if reference and transcript_base
            else None
        )
        cer_improvement = (
            round(cer_base - cer_finetuned, 4)
            if (cer_base is not None and cer_finetuned is not None)
            else None
        )
        cer_improvement_norm = (
            round(cer_base_norm - cer_finetuned_norm, 4)
            if (cer_base_norm is not None and cer_finetuned_norm is not None)
            else None
        )

        # Auto-flagging (based on fine-tuned transcript)
        flagged, flag_reasons = auto_flag(transcript_ft, reference, duration_s)

        segment_entry = {
            "segment_id": wav_path.stem,
            "audio_path": str(wav_path),
            "duration_s": duration_s,
            "source_file": rec.get("source_file", args.lecture or "unknown"),
            "segment_index": rec.get("segment_index", -1),
            "reference": reference,
            "transcript_finetuned": transcript_ft,
            "transcript_base": transcript_base,
            "wer_finetuned": wer_ft,
            "wer_finetuned_norm": wer_ft_norm,
            "wer_base": wer_base,
            "wer_base_norm": wer_base_norm,
            "wer_improvement": wer_improvement,
            "wer_improvement_norm": wer_improvement_norm,
            "cer_finetuned": cer_finetuned,
            "cer_finetuned_norm": cer_finetuned_norm,
            "cer_base": cer_base,
            "cer_base_norm": cer_base_norm,
            "cer_improvement": cer_improvement,
            "cer_improvement_norm": cer_improvement_norm,
            "flagged": flagged,
            "flag_reasons": flag_reasons,
        }
        segments.append(segment_entry)

    # ── Group by lecture ──────────────────────────────────────────────────────
    lectures: dict = defaultdict(list)
    for seg in segments:
        lectures[seg["source_file"]].append(seg)

    output = []
    for lecture_id, segs in lectures.items():
        segs_sorted = sorted(
            segs, key=lambda s: s["segment_index"] if s["segment_index"] >= 0 else 0
        )

        # Aggregate WER across all segments that have a reference
        segs_with_ref = [s for s in segs_sorted if s["reference"]]
        n = len(segs_with_ref)

        def avg(key):
            vals = [s[key] for s in segs_with_ref if s[key] is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        # WER summary for the report
        wer_summary = {
            "num_segments_with_reference": n,
            "avg_wer_finetuned": avg("wer_finetuned"),
            "avg_wer_finetuned_norm": avg("wer_finetuned_norm"),
            "avg_wer_base": avg("wer_base"),
            "avg_wer_base_norm": avg("wer_base_norm"),
            "avg_wer_improvement": avg("wer_improvement"),
            "avg_wer_improvement_norm": avg("wer_improvement_norm"),
            "avg_cer_finetuned": avg("cer_finetuned"),
            "avg_cer_base": avg("cer_base"),
            "avg_cer_improvement": avg("cer_improvement"),
        }

        n_flagged = sum(1 for s in segs_sorted if s["flagged"])

        full_transcript = " ".join(
            s["transcript_finetuned"]
            for s in segs_sorted
            if s["transcript_finetuned"] and not s["flagged"]
        )

        output.append(
            {
                "lecture_id": lecture_id,
                "num_segments": len(segs_sorted),
                "num_flagged": n_flagged,
                "total_duration_s": round(sum(s["duration_s"] for s in segs_sorted), 1),
                "wer_summary": wer_summary,
                "full_transcript": full_transcript,
                "segments": segs_sorted,
            }
        )

        # ── Print comparison table to terminal ────────────────────────────────
        logger.info(f"\n{'━'*72}")
        logger.info(f"Lecture: {lecture_id}")
        logger.info(f"{'━'*72}")
        logger.info(f"  Segments with reference : {n}")
        logger.info(f"  Flagged segments        : {n_flagged}")
        logger.info(f"")
        logger.info(
            f"  {'Model':<25} {'WER (raw)':>10} {'WER (norm)':>12} {'CER (raw)':>10}"
        )
        logger.info(f"  {'─'*25} {'─'*10} {'─'*12} {'─'*10}")
        if wer_summary["avg_wer_base"] is not None:
            logger.info(
                f"  {'Baseline (large-v3)':<25} "
                f"{wer_summary['avg_wer_base']:>10.4f} "
                f"{wer_summary['avg_wer_base_norm']:>12.4f} "
                f"{wer_summary['avg_cer_base']:>10.4f}"
            )
        if wer_summary["avg_wer_finetuned"] is not None:
            logger.info(
                f"  {'Fine-tuned':<25} "
                f"{wer_summary['avg_wer_finetuned']:>10.4f} "
                f"{wer_summary['avg_wer_finetuned_norm']:>12.4f} "
                f"{wer_summary['avg_cer_finetuned']:>10.4f}"
            )
        else:
            logger.info(f"  {'Fine-tuned':<25} {'N/A':>10} {'N/A':>12} {'N/A':>10}")
        if wer_summary["avg_wer_improvement"] is not None:
            logger.info(
                f"  {'Improvement':<25} "
                f"{wer_summary['avg_wer_improvement']:>+10.4f} "
                f"{wer_summary['avg_wer_improvement_norm']:>+12.4f} "
                f"{wer_summary['avg_cer_improvement']:>+10.4f}"
            )
        logger.info(f"{'━'*72}")

        if n_flagged > 0:
            logger.info(f"\n  Auto-flagged segments:")
            for s in segs_sorted:
                if s["flagged"]:
                    logger.info(f"    {s['segment_id']}: {s['flag_reasons']}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"\n✓ Transcripts saved → {args.out}")

    # ── Save TXT report ───────────────────────────────────────────────────────
    txt_out = str(Path(args.out).with_suffix(".txt"))
    lines = []
    lines.append("=" * 72)
    lines.append("TRANSCRIPTION REPORT")
    lines.append("=" * 72)
    lines.append(f"Fine-tuned model : {args.model}")
    if not args.no_baseline:
        lines.append(f"Baseline model   : {args.baseline}")
    lines.append(f"Audio directory  : {args.audio}")
    lines.append(f"Lectures         : {len(output)}")
    lines.append(f"Total segments   : {sum(l['num_segments'] for l in output)}")
    lines.append("")

    for lec in output:
        ws = lec["wer_summary"]
        lines.append("━" * 72)
        lines.append(f"Lecture : {lec['lecture_id']}")
        lines.append(f"  Segments total  : {lec['num_segments']}")
        lines.append(f"  Flagged         : {lec['num_flagged']}")
        lines.append(f"  Duration        : {lec['total_duration_s']} s")
        lines.append(f"  Segs w/ ref     : {ws['num_segments_with_reference']}")
        lines.append("")
        lines.append(
            f"  {'Model':<25} {'WER (raw)':>10} {'WER (norm)':>12} {'CER (raw)':>10}"
        )
        lines.append(f"  {'─'*25} {'─'*10} {'─'*12} {'─'*10}")
        if ws["avg_wer_base"] is not None:
            lines.append(
                f"  {'Baseline (large-v3)':<25} "
                f"{ws['avg_wer_base']:>10.4f} "
                f"{ws['avg_wer_base_norm']:>12.4f} "
                f"{ws['avg_cer_base']:>10.4f}"
            )
        if ws["avg_wer_finetuned"] is not None:
            lines.append(
                f"  {'Fine-tuned':<25} "
                f"{ws['avg_wer_finetuned']:>10.4f} "
                f"{ws['avg_wer_finetuned_norm']:>12.4f} "
                f"{ws['avg_cer_finetuned']:>10.4f}"
            )
        if ws["avg_wer_improvement"] is not None:
            lines.append(
                f"  {'Improvement':<25} "
                f"{ws['avg_wer_improvement']:>+10.4f} "
                f"{ws['avg_wer_improvement_norm']:>+12.4f} "
                f"{ws['avg_cer_improvement']:>+10.4f}"
            )
        lines.append("")

        # Flagged segments
        flagged_segs = [s for s in lec["segments"] if s["flagged"]]
        if flagged_segs:
            lines.append("  Auto-flagged segments:")
            for s in flagged_segs:
                lines.append(f"    {s['segment_id']}: {s['flag_reasons']}")
            lines.append("")

        # Per-segment detail
        lines.append("  Segment details:")
        lines.append(
            f"  {'ID':<20} {'Dur(s)':>6} {'WER-FT':>8} {'WER-BL':>8} "
            f"{'CER-FT':>8} {'Flag':>5}"
        )
        lines.append(f"  {'─'*20} {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*5}")
        for s in lec["segments"]:
            wer_ft = (
                f"{s['wer_finetuned']:.4f}"
                if s["wer_finetuned"] is not None
                else "  N/A "
            )
            wer_bl = f"{s['wer_base']:.4f}" if s["wer_base"] is not None else "  N/A "
            cer_ft = (
                f"{s['cer_finetuned']:.4f}"
                if s["cer_finetuned"] is not None
                else "  N/A "
            )
            flag = "YES" if s["flagged"] else "no"
            lines.append(
                f"  {s['segment_id']:<20} {s['duration_s']:>6.1f} "
                f"{wer_ft:>8} {wer_bl:>8} {cer_ft:>8} {flag:>5}"
            )
        lines.append("")

        # Full transcript
        lines.append("  Full transcript (fine-tuned, unflagged segments):")
        lines.append("  " + "-" * 68)
        # Wrap at ~70 chars
        words = lec["full_transcript"].split()
        current_line = "  "
        for word in words:
            if len(current_line) + len(word) + 1 > 72:
                lines.append(current_line)
                current_line = "  " + word
            else:
                current_line = (current_line + " " + word).strip()
                current_line = "  " + current_line.strip()
        if current_line.strip():
            lines.append(current_line)
        lines.append("")

    lines.append("=" * 72)
    lines.append("END OF REPORT")
    lines.append("=" * 72)

    with open(txt_out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info(f"✓ Text report saved  → {txt_out}")
    logger.info(f"  Lectures : {len(output)}")
    logger.info(f"  Segments : {sum(l['num_segments'] for l in output)}")
    logger.info("")
    logger.info("Next steps:")
    logger.info(
        "  1. Review transcripts.json / transcripts.txt (check flagged segments)"
    )
    logger.info("  2. Run: python run_summarization.py --skip-flagged")


if __name__ == "__main__":
    main()
