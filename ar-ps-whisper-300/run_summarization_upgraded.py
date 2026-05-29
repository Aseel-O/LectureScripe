#!/usr/bin/env python3
"""
Multi-run Summarization Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Runs three summariser models (mBART, AraBART, and Qwen) on three
different transcript sources and produces a side-by-side comparison.
This is the core cascade experiment: it shows how upstream ASR quality
(baseline vs. fine-tuned vs. hand-corrected) propagates into downstream
summary quality, which is then measured by evaluate.py.

Transcript sources (all read from transcripts.json):
  Run A — baseline Whisper large-v3 output    field: transcript_base
  Run B — fine-tuned Whisper output           field: transcript_finetuned
  Run C — hand-corrected reference (upper bound)  field: reference

Summariser models:
  mBART   — facebook/mbart-large-cc25   (multilingual seq2seq, fp16)
  AraBART — moussaKam/AraBART           (Arabic-only seq2seq, fp16)
  Qwen    — Qwen2.5-7B-Instruct         (instruction-tuned LLM, 4-bit on CUDA)

Output files:
  summaries_baseline.json              ← Run A results
  summaries_finetuned.json             ← Run B results
  summaries_reference.json             ← Run C results
  summaries_comparison_report.txt      ← human-readable side-by-side report

The JSON files are merge-aware: running --qwen-only followed by
--mbart-only does not erase previous model outputs — only the fields
for the models run in the current session are overwritten.

Usage:
    python run_summarization.py
    python run_summarization.py --qwen-only           # skip mBART and AraBART
    python run_summarization.py --mbart-only          # skip Qwen and AraBART
    python run_summarization.py --arabart-only        # skip mBART and Qwen
    python run_summarization.py --arabart             # add AraBART alongside mBART + Qwen
    python run_summarization.py --runs base ft ref    # choose which transcript sources
    python run_summarization.py --runs ft             # fine-tuned transcripts only
    python run_summarization.py --skip-flagged        # exclude auto-flagged segments
    python run_summarization.py --qwen-style notes        # student-style bullet notes (default)
    python run_summarization.py --qwen-style code-switch  # formal paragraph + English terms inline
    python run_summarization.py --qwen-style msa          # pure formal MSA, no English terms
    python run_summarization.py --ref-style code-switch   # reference summaries use code-switch (default)
    python run_summarization.py --ref-style msa           # reference summaries use MSA
    python run_summarization.py --ref-style notes          # reference summaries use notes
    python run_summarization.py --early-stop              # stop generation at JSON closing brace
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import gc
import os
import re
import json
import time
import argparse
import logging
import textwrap
from typing import List, Dict

import torch
from transformers import (
    MBartForConditionalGeneration,
    #    MBart50TokenizerFast,
    MBartTokenizerFast,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_INPUT = "transcripts.json"
MBART_MODEL_ID = "facebook/mbart-large-cc25"
QWEN_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MBART_SRC_LANG = "ar_AR"
MBART_TGT_LANG = "ar_AR"

ARABART_MODEL_ID = "moussaKam/AraBART"
# AraBART is Arabic-only (monolingual BART-base architecture).
# No src/tgt language codes are needed — the model has a single vocabulary.
# Max position embeddings = 1024 tokens; input is chunked to stay within this.
# English technical terms are handled as subword tokens: short terms (e.g.
# "BERT", "F1") pass through largely intact; longer terms may fragment.


MLX_QWEN_MODEL_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit"
# Pre-quantized MLX build of Qwen2.5-7B-Instruct for Apple Silicon.
# Uses ~3.8 GB unified memory and runs ~3-5× faster than the MPS/PyTorch path.
# On CUDA (A100 20 GB), bitsandbytes NF4 4-bit is used instead (see load_qwen).
#
# VRAM budget on A100 20 GB:
#   mBART fp16    ~2.4 GB
#   AraBART fp16  ~0.5 GB
#   Qwen 4-bit    ~4.0 GB   (NF4 + double quantisation)
#   Activations   ~4-6 GB   (KV cache at maximum sequence length)
#   ─────────────────────
#   Peak          ~13 GB    — comfortable within the 20 GB limit


MBART_MAX_INPUT_TOKENS = 1024
MBART_MAX_NEW_TOKENS = 300
MBART_MIN_NEW_TOKENS = 80
ARABART_MAX_INPUT_TOKENS = 1024  # same BART architecture limit
ARABART_MAX_NEW_TOKENS = 300
ARABART_MIN_NEW_TOKENS = 80
QWEN_MAX_INPUT_CHARS = 8000  # increased — notes need full lecture context
QWEN_MAX_NEW_TOKENS = 1500  # increased — structured notes are longer than a paragraph

# Maps --runs CLI values to the corresponding field name in transcripts.json
RUN_FIELD_MAP = {
    "base": "transcript_base",
    "ft": "transcript_finetuned",
    "ref": "reference",
}
RUN_LABEL_MAP = {
    "base": "Baseline Whisper (large-v3)",
    "ft": "Fine-tuned Whisper",
    "ref": "Hand-corrected Reference",
}
RUN_OUTPUT_MAP = {
    "base": "summaries_baseline.json",
    "ft": "summaries_finetuned.json",
    "ref": "summaries_reference.json",
}

# ── Device helpers ────────────────────────────────────────────────────────────


def _mlx_available() -> bool:
    """True when mlx-lm is installed and we are on Apple Silicon."""
    try:
        import mlx.core  # noqa: F401
        import mlx_lm  # noqa: F401

        return True
    except ImportError:
        return False


def get_device() -> str:
    """Return the best available PyTorch device string for seq2seq models.

    Checks CUDA → MPS → CPU in order and logs the selected device with
    available VRAM (CUDA only). This device is used for mBART and AraBART.
    Qwen uses a separate backend selection path in load_qwen, since it
    can also run via MLX on Apple Silicon.
    """

    if torch.cuda.is_available():
        device = "cuda"
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / 1024**3
        logger.info("CUDA device : %s  (%.1f GB VRAM)", props.name, total)
        return device
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("PyTorch device : mps (Apple Silicon)")
        return "mps"
    logger.info("PyTorch device : cpu")
    return "cpu"


def log_vram(label: str) -> None:
    """Log current CUDA VRAM usage at a named checkpoint.

    Prints both allocated memory (actively used by tensors) and reserved
    memory (held by the PyTorch caching allocator). Useful for diagnosing
    OOM risk between model loads and inference calls. No-op on MPS or CPU.
    """
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    logger.info(
        "  VRAM [%s] : %.1f GB allocated / %.1f GB reserved", label, alloc, reserved
    )


# ── Text chunking ─────────────────────────────────────────────────────────────
def chunk_text(text: str, max_chars: int, overlap_chars: int = 200) -> List[str]:
    """Split a long transcript into overlapping chunks for seq2seq models.

    Production-hardened implementation with the following key features:
    1. Endpoint capping: Ensures end index never exceeds text length.
    2. Termination guard: Breaks immediately after final chunk is processed.
    3. Boundary progress guard: Only snaps to word/sentence boundary if it
       guarantees forward progress after overlap subtraction.
    4. No-boundary fallback: Hard-cuts at max_chars for space-less blocks
       (e.g., hallucinated repetitions from ASR).
    5. Empty-chunk filter: Skips empty chunks as a safety measure.
    6. Iteration limiter: Prevents infinite loops through max-iteration counter.
    """
    if not text:
        return []

    length = len(text)
    if length <= max_chars:
        return [text]

    # CRITICAL: overlap must be strictly less than chunk size. If not,
    # start = end - overlap can never advance, causing an infinite loop.
    if overlap_chars >= max_chars:
        raise ValueError(
            f"chunk_text: overlap_chars ({overlap_chars}) must be strictly less "
            f"than max_chars ({max_chars}). text_len={length}."
        )

    chunks: List[str] = []
    start = 0
    # Safety fuse: worst-case iterations if we only advance 1 char at a time
    max_iterations = (length // max(1, max_chars - overlap_chars)) + 10
    iterations = 0

    while start < length and iterations < max_iterations:
        iterations += 1

        # Endpoint capping: ensure end never exceeds text length
        end = min(start + max_chars, length)

        # Try to find a semantic boundary, but only if we're not at the end
        if end < length:
            # Prefer sentence boundary, fall back to word boundary
            boundary = text.rfind(".", start, end)
            if boundary == -1:
                boundary = text.rfind(" ", start, end)

            # Boundary progress guard: only snap if guaranteed forward progress
            # After snapping: new_start = boundary + 1 - overlap_chars
            # We need new_start > start  →  boundary > start + overlap_chars - 1
            # Using > start + overlap_chars is a safe conservative threshold.
            if boundary > start + overlap_chars:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Termination guard: break immediately when entire text is consumed
        if end == length:
            break

        # Advance with overlap
        start = end - overlap_chars

        # Ultimate safety: if we somehow didn't advance (should be impossible
        # with the boundary guard above, but protects against future edits),
        # force a hard break with zero overlap.
        if start <= 0:
            start = end

    if iterations >= max_iterations:
        logger.warning(
            "chunk_text hit safety fuse after %d iterations (text_len=%d, "
            "max_chars=%d, overlap=%d). Returning %d chunks.",
            max_iterations,
            length,
            max_chars,
            overlap_chars,
            len(chunks),
        )

    return chunks


def _truncate_repetition_loop(text: str) -> str:
    """Truncate end-of-text repetition loops (e.g., 'word word word...').

    Production guard for mlx_lm when temperature sampling cannot be set
    via the Python API. Detects if the last 6 tokens are the exact same
    word repeated, and truncates back to before the loop started.
    """
    words = text.split()
    if len(words) < 8:
        return text

    last_word = words[-1]
    # If the last 6 words are identical, we are in a loop
    if all(w == last_word for w in words[-6:]):
        # Walk backwards to find where the repetition starts
        for i in range(len(words) - 6, -1, -1):
            if words[i] != last_word:
                return " ".join(words[: i + 1])
        return last_word  # entire string is one repeated word

    return text


# ── Early stopping for MLX generation ───────────────────────────────────────


def _mlx_generate_with_early_stop(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    verbose: bool = False,
    sampler=None,
    logits_processors=None,
) -> str:
    """Generate with MLX, stopping early when JSON is complete.

    This wraps mlx_lm.generate() with a post-generation check that truncates
    at the first balanced JSON closing brace. This prevents the model from
    generating extra commentary after the JSON object is complete.

    If the model outputs valid JSON followed by extra text, we truncate at
    the end of the JSON. If the JSON is truncated, we return the raw output
    and let the parser handle repair.
    """
    from mlx_lm import generate as mlx_generate

    kwargs = {
        "max_tokens": max_tokens,
        "verbose": verbose,
    }
    if sampler is not None:
        kwargs["sampler"] = sampler
    if logits_processors is not None:
        kwargs["logits_processors"] = logits_processors

    raw = mlx_generate(model, tokenizer, prompt=prompt, **kwargs)

    # Post-generation: find the first balanced JSON object and truncate
    return _truncate_at_json_end(raw)


def _truncate_at_json_end(text: str) -> str:
    """Truncate text at the end of the first balanced JSON object.

    Scans for '{' and tracks brace depth. When depth returns to 0,
    truncates at that position (inclusive). This prevents the model from
    generating commentary after the JSON response.

    Returns the original text if no balanced JSON is found.
    """
    first_brace = text.find("{")
    if first_brace == -1:
        return text

    depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(text[first_brace:], start=first_brace):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                # Found balanced JSON end — truncate here
                return text[: i + 1]

    # No balanced JSON found — return original for parser repair
    return text


# ── Key terms extraction ──────────────────────────────────────────────────────
# Curated vocabulary of AI/ML terms covering NLP, Computer Vision, and shared
# concepts expected in Arabic-English code-switched lectures. Used to build a
# key-terms list that is injected into the Qwen system prompt, helping the
# model retain terminology in its JSON output.
# Extend this list when covering new topics (e.g. add "RL", "reward", "policy"
# for a reinforcement-learning lecture).

AI_TERMS = [
    # ── Shared ML / general ───────────────────────────────────────────────
    "classifier",
    "classification",
    "model",
    "training",
    "inference",
    "prediction",
    "label",
    "labels",
    "feature",
    "features",
    "accuracy",
    "precision",
    "recall",
    "F1",
    "loss",
    "overfitting",
    "underfitting",
    "regularization",
    "dropout",
    "backpropagation",
    "gradient descent",
    "learning rate",
    "optimizer",
    "Adam",
    "SGD",
    "batch",
    "epoch",
    "neural network",
    "deep learning",
    "machine learning",
    "supervised",
    "unsupervised",
    "semi-supervised",
    "dataset",
    "baseline",
    "class",
    "classes",
    "evaluation",
    "train set",
    "test set",
    "validation set",
    "cross-validation",
    "hyperparameter",
    "fine-tuning",
    "transfer learning",
    "variable",
    "distribution",
    "prior",
    "posterior",
    "likelihood",
    "probability",
    "conditional probability",
    "joint probability",
    "log",
    "log probability",
    "underflow",
    "overflow",
    "benchmark",
    "inference",
    "pipeline",
    # ── NLP ───────────────────────────────────────────────────────────────
    "independence assumption",
    "Naive Bayes",
    "Bayes",
    "tokenization",
    "tokenizer",
    "token",
    "tokens",
    "embedding",
    "embeddings",
    "word2vec",
    "BERT",
    "GPT",
    "transformer",
    "attention",
    "self-attention",
    "encoder",
    "decoder",
    "seq2seq",
    "language model",
    "LM",
    "n-gram",
    "bigram",
    "unigram",
    "perplexity",
    "corpus",
    "vocabulary",
    "vocab",
    "smoothing",
    "Laplace smoothing",
    "backoff",
    "document",
    "documents",
    "TF-IDF",
    "bag of words",
    "named entity recognition",
    "NER",
    "part of speech",
    "POS",
    "sentiment analysis",
    "machine translation",
    "summarization",
    "WER",
    "BLEU",
    "ROUGE",
    # ── Computer Vision ───────────────────────────────────────────────────
    "convolutional neural network",
    "CNN",
    "convolution",
    "pooling",
    "max pooling",
    "average pooling",
    "filter",
    "kernel",
    "stride",
    "padding",
    "object detection",
    "image classification",
    "segmentation",
    "semantic segmentation",
    "instance segmentation",
    "bounding box",
    "anchor box",
    "region proposal",
    "YOLO",
    "R-CNN",
    "Faster R-CNN",
    "SSD",
    "ResNet",
    "VGG",
    "AlexNet",
    "MobileNet",
    "EfficientNet",
    "feature map",
    "activation map",
    "receptive field",
    "batch normalization",
    "layer normalization",
    "data augmentation",
    "image preprocessing",
    "HOG",
    "SIFT",
    "edge detection",
    "histogram",
    "pixel",
    "resolution",
    "grayscale",
    "RGB",
    "keypoint",
    "descriptor",
    "matching",
    "optical flow",
    "tracking",
    "pose estimation",
    "GAN",
    "generator",
    "discriminator",
    "IoU",
    "mAP",
    "precision-recall curve",
]


def extract_key_terms(transcript: str) -> List[str]:
    """Scan a transcript for known AI/ML terms and uppercase acronyms.

    Performs case-insensitive whole-word matching against AI_TERMS (which
    covers NLP, Computer Vision, and shared ML concepts), then appends any
    2-6 character uppercase acronyms found in the text (e.g. "WER", "CNN",
    "IoU") that are not already in the term list.

    The returned list is passed to Qwen's system prompt so the model knows
    which English terms to preserve inline rather than translating.

    Args:
        transcript: Raw transcript string (may be Arabic/English mixed).

    Returns:
        Sorted, deduplicated list of matched term strings.
    """

    found = []
    tl = transcript.lower()
    for term in AI_TERMS:
        if re.search(r"\b" + re.escape(term.lower()) + r"\b", tl):
            found.append(term)
    acronyms = re.findall(r"\b[A-Z]{2,6}\b", transcript)
    for a in set(acronyms):
        if a not in found:
            found.append(a)
    return sorted(set(found))


# ══════════════════════════════════════════════════════════════════════════════
# mBART
# ══════════════════════════════════════════════════════════════════════════════


def load_mbart(device: str):
    """Load the mBART-large-cc25 tokenizer and model onto the given device.

    Uses fp16 on CUDA to keep VRAM usage around 2.4 GB; falls back to fp32
    on MPS and CPU. The source language is set to Arabic (ar_AR) at load
    time; the target language is forced at generation time via
    ``forced_bos_token_id``.

    Args:
        device: PyTorch device string ("cuda", "mps", or "cpu").

    Returns:
        (tokenizer, model) tuple — both ready for inference.
    """

    logger.info(f"Loading mBART ({MBART_MODEL_ID}) …")
    tokenizer = MBartTokenizerFast.from_pretrained(
        MBART_MODEL_ID, src_lang=MBART_SRC_LANG
    )
    # Use fp16 on both CUDA and MPS (Apple Silicon MPS supports fp16 since
    # PyTorch 2.0). This halves model memory from ~4.8 GB to ~2.4 GB on MPS,
    # which is critical for staying within unified memory limits. CPU falls
    # back to fp32 since it has no fp16 acceleration benefit.
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = MBartForConditionalGeneration.from_pretrained(
        MBART_MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    logger.info("mBART loaded.")
    return tokenizer, model


def summarize_mbart(transcript: str, tokenizer, model, device: str) -> str:
    """Summarize a transcript with mBART using beam search.

    Long transcripts are split into chunks (see chunk_text) to stay within
    the 1024-token input limit. Each chunk is summarised independently and
    the partial summaries are concatenated. The target language token
    (ar_AR) is forced as the first generated token via ``forced_bos_token_id``
    to ensure Arabic output regardless of source language mix.

    Generation hyperparameters:
        num_beams=4, length_penalty=1.2 (encourages longer outputs),
        no_repeat_ngram_size=3 (suppresses repetition loops).

    Args:
        transcript: Full lecture transcript string.
        tokenizer:  MBart50TokenizerFast instance.
        model:      MBartForConditionalGeneration instance.
        device:     PyTorch device string.

    Returns:
        Concatenated Arabic summary string.
    """

    chunks = chunk_text(transcript, max_chars=MBART_MAX_INPUT_TOKENS * 3)
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        if len(chunks) > 1:
            logger.info(f"    mBART chunk {i+1}/{len(chunks)}")
        # Re-set src_lang per chunk — tokenizer state is mutable
        tokenizer.src_lang = MBART_SRC_LANG
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            max_length=MBART_MAX_INPUT_TOKENS,
            truncation=True,
        ).to(device)
        # Force Arabic as the first decoder token so output stays in Arabic
        tgt_id = tokenizer.lang_code_to_id[MBART_TGT_LANG]
        with torch.no_grad():
            out = model.generate(
                **inputs,
                forced_bos_token_id=tgt_id,
                max_new_tokens=MBART_MAX_NEW_TOKENS,
                min_new_tokens=MBART_MIN_NEW_TOKENS,
                max_length=None,  # suppress conflict warning with max_new_tokens
                num_beams=4,
                length_penalty=1.2,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        chunk_summaries.append(
            tokenizer.decode(out[0], skip_special_tokens=True).strip()
        )
        # Explicitly free input/output tensors after each chunk so MPS doesn't
        # accumulate activation memory across all 16 chunks of a long lecture.
        del inputs, out
        if device == "mps":
            torch.mps.empty_cache()
    return " ".join(chunk_summaries)


# ══════════════════════════════════════════════════════════════════════════════


def load_arabart(device: str):
    logger.info(f"Loading AraBART ({ARABART_MODEL_ID}) …")
    tokenizer = AutoTokenizer.from_pretrained(ARABART_MODEL_ID)
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ARABART_MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    logger.info("AraBART loaded.")
    return tokenizer, model


def summarize_arabart(transcript: str, tokenizer, model, device: str) -> str:
    """
    Summarize with AraBART (Arabic-only seq2seq).
    No language codes required — the model is monolingual.
    Chunking mirrors the mBART approach since the token budget is identical.
    """
    chunks = chunk_text(transcript, max_chars=ARABART_MAX_INPUT_TOKENS * 3)
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        if len(chunks) > 1:
            logger.info(f"    AraBART chunk {i+1}/{len(chunks)}")
        logger.debug(f"    chunk {i+1}: {len(chunk)} chars")
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            max_length=ARABART_MAX_INPUT_TOKENS,
            truncation=True,
            padding=False,
        )
        logger.debug(
            f"    chunk {i+1}: tokenized → {inputs['input_ids'].shape[1]} tokens, moving to {device}"
        )
        inputs = inputs.to(device)
        logger.debug(f"    chunk {i+1}: on device, running generate()")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=ARABART_MAX_NEW_TOKENS,
                min_new_tokens=ARABART_MIN_NEW_TOKENS,
                num_beams=4,
                length_penalty=1.2,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        chunk_summaries.append(
            tokenizer.decode(out[0], skip_special_tokens=True).strip()
        )
        del inputs, out
        if device == "mps":
            torch.mps.empty_cache()
    return " ".join(chunk_summaries)


# ══════════════════════════════════════════════════════════════════════════════
# Qwen  — three backends: cuda (bitsandbytes 4-bit) | mlx | torch (mps/cpu)
# ══════════════════════════════════════════════════════════════════════════════
#
# load_qwen returns (tokenizer, model, backend) where backend is one of:
#   "cuda"  — PyTorch + bitsandbytes 4-bit on NVIDIA GPU
#   "mlx"   — mlx-lm on Apple Silicon (fastest local option)
#   "torch" — PyTorch float32 on MPS or CPU (fallback)
#
# summarize_qwen dispatches on backend so callers don't need to know which
# path is active.


def load_qwen(device: str):
    """
    Load Qwen2.5-7B-Instruct on the best available backend.
    Priority: cuda (4-bit) → mlx (Apple Silicon) → torch (mps/cpu)
    """
    # ── CUDA path — bitsandbytes NF4 + double quantisation ───────────────────
    if device == "cuda":
        logger.info("Loading Qwen [cuda / bitsandbytes 4-bit] (%s) …", QWEN_MODEL_ID)
        from transformers import BitsAndBytesConfig

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,  # saves ~0.4 GB vs single quant
            bnb_4bit_quant_type="nf4",  # better quality than fp4
        )
        tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)
        # device_map={"": 0} passes a dict instead of a string, which prevents
        # accelerate's dispatch_model from calling .to() on the 4-bit model.
        # Passing a plain string like "cuda:0" or "auto" can still trigger the
        # dispatch path in some accelerate versions, raising the ValueError.
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_ID,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model.eval()
        log_vram("after Qwen load")
        logger.info("Qwen loaded [cuda].")
        return tokenizer, model, "cuda"

    # ── MLX path — Apple Silicon native ──────────────────────────────────────
    if _mlx_available():
        logger.info("Loading Qwen [mlx] (%s) …", MLX_QWEN_MODEL_ID)
        from mlx_lm import load as mlx_load

        model, tokenizer = mlx_load(MLX_QWEN_MODEL_ID)
        logger.info("Qwen loaded [mlx].")
        return tokenizer, model, "mlx"

    # ── PyTorch fallback — MPS or CPU ─────────────────────────────────────────
    logger.info("Loading Qwen [torch / %s] (%s) …", device, QWEN_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_ID,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    logger.info("Qwen loaded [torch].")
    return tokenizer, model, "torch"


def summarize_qwen(
    transcript: str,
    tokenizer,
    model,
    device: str,
    style: str,
    backend: str,
    early_stop: bool = False,
    key_terms: List[str] = None,
) -> Dict:
    """
    Summarize with Qwen, dispatching on backend.
    All three paths share the same prompt-building and JSON-parsing logic;
    only the generation call differs.

    Args:
        early_stop: If True, truncate MLX generation at the first balanced
                    JSON closing brace. Prevents post-JSON commentary.
        key_terms:  English technical terms extracted from the transcript.
                    Injected into the system prompt so Qwen knows which words
                    to keep in English even when the transcript is heavily
                    Arabicized (e.g. baseline Whisper output).
    """
    messages = build_qwen_messages(transcript, style, key_terms=key_terms)

    # ── shared: format prompt via chat template ───────────────────────────────
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # ── generate ──────────────────────────────────────────────────────────────
    if backend == "mlx":
        from mlx_lm import generate as mlx_generate

        # STRATEGY 1: Modern mlx_lm API (>=0.19) uses sample_utils helpers.
        # This gives us temperature + repetition_penalty properly.
        try:
            from mlx_lm.sample_utils import make_sampler, make_logits_processors

            sampler = make_sampler(temp=0.3, top_p=0.9)
            logits_processors = make_logits_processors(repetition_penalty=1.15)

            if early_stop:
                raw = _mlx_generate_with_early_stop(
                    model,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=QWEN_MAX_NEW_TOKENS,
                    verbose=False,
                    sampler=sampler,
                    logits_processors=logits_processors,
                )
            else:
                raw = mlx_generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=QWEN_MAX_NEW_TOKENS,
                    verbose=False,
                    sampler=sampler,
                    logits_processors=logits_processors,
                )
        except (ImportError, TypeError) as _e:
            # STRATEGY 2: Fallback for older mlx_lm that lacks sample_utils.
            # We call bare generate() with zero sampling args.
            logger.debug("mlx_lm sample_utils unavailable, using bare generate()")
            raw = mlx_generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=QWEN_MAX_NEW_TOKENS,
                verbose=False,
            )

        # POST-GENERATION GUARD: Cut off repetition loops that the model
        # may still fall into when running without temperature control.
        raw = _truncate_repetition_loop(raw)

    else:  # "cuda" or "torch"
        inputs = tokenizer([prompt], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=QWEN_MAX_NEW_TOKENS,
                temperature=0.3,
                do_sample=True,
                repetition_penalty=1.15,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if backend == "cuda":
            torch.cuda.empty_cache()

    # ── shared: parse JSON response ───────────────────────────────────────────
    def _try_parse(text: str):
        """Parse JSON from LLM output with aggressive repair.

        Handles:
        - Markdown fences and invisible prefixes
        - Literal newlines / tabs inside JSON string values (MLX artifact)
        - Unterminated strings and unbalanced braces/brackets
        - Trailing commas before closers
        - Double-encoded JSON (JSON string containing JSON)
        """
        import json
        import re

        # 1. Strip markdown fences, control chars, and BOM
        text = re.sub(r"```(?:json)?|```", "", text).strip()
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        text = text.lstrip("\ufeff")

        # 2. Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2b. Arabic comma fix: the model sometimes uses the Arabic comma (،
        #     U+060C) as a JSON structural delimiter instead of ASCII comma.
        #     This happens in code-switch style when the surrounding text is
        #     Arabic and the model's tokenizer conflates the two characters.
        #     We replace ، with , ONLY when it appears outside a JSON string
        #     (inside strings it is valid Arabic punctuation and must be kept).
        def _fix_arabic_commas(s: str) -> str:
            out = []
            in_str = False
            esc = False
            for ch in s:
                if esc:
                    out.append(ch)
                    esc = False
                    continue
                if ch == "\\":
                    out.append(ch)
                    esc = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    out.append(ch)
                    continue
                if not in_str and ch == "\u060c":  # ،
                    out.append(",")
                else:
                    out.append(ch)
            return "".join(out)

        text_fixed = _fix_arabic_commas(text)
        if text_fixed != text:
            try:
                return json.loads(text_fixed)
            except json.JSONDecodeError:
                pass
            text = text_fixed  # carry the fix forward into later repair steps

        # 3. Extract outermost {...} block
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            candidate = text[first_brace : last_brace + 1]
        else:
            candidate = text

        # 4. Escape literal newlines/tabs inside JSON string values
        #    Uses a tiny state machine instead of regex (much more reliable
        #    for multi-line Arabic text with mixed Unicode).
        def _escape_json_strings(s: str) -> str:
            out = []
            in_str = False
            escape = False
            for ch in s:
                if escape:
                    out.append(ch)
                    escape = False
                    continue
                if ch == "\\":
                    out.append(ch)
                    escape = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    out.append(ch)
                    continue
                if in_str:
                    if ch == "\n":
                        out.append("\\n")
                    elif ch == "\r":
                        out.append("\\r")
                    elif ch == "\t":
                        out.append("\\t")
                    else:
                        out.append(ch)
                else:
                    out.append(ch)
            return "".join(out)

        escaped = _escape_json_strings(candidate)
        try:
            return json.loads(escaped)
        except json.JSONDecodeError:
            pass

        # 5. Truncation repair: close unterminated strings, balance braces/brackets
        repaired = escaped

        # 5a. Close unterminated strings (odd number of unescaped quotes)
        unescaped_quotes = 0
        escape = False
        for ch in repaired:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                unescaped_quotes += 1
        if unescaped_quotes % 2 == 1:
            repaired += '"'

        # 5b. Strip trailing commas before } or ]
        repaired = re.sub(r",(\s*[\}\]])", r"\1", repaired)

        # 5c. Balance braces and brackets
        open_braces = repaired.count("{") - repaired.count("}")
        open_brackets = repaired.count("[") - repaired.count("]")
        repaired += "}" * max(0, open_braces)
        repaired += "]" * max(0, open_brackets)

        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

        # 6. Double-encoded JSON: sometimes the model returns a JSON string
        #    that itself contains JSON. Try unwrapping one level.
        try:
            if repaired.startswith('"') and repaired.endswith('"'):
                inner = json.loads(repaired)
                if isinstance(inner, str):
                    return _try_parse(inner)
        except (json.JSONDecodeError, ValueError, RecursionError):
            pass

        return None

    result = _try_parse(raw)
    if result is not None:
        if style == "notes":
            content = result.get("notes", result.get("summary_ar", ""))
        else:
            content = result.get("summary_ar", result.get("notes", ""))
        return {
            "summary_ar": content,
            "key_terms": result.get("key_terms", []),
            "style": style,
            "backend": backend,
            "parse_error": None,
        }
    else:
        logger.warning("  Qwen JSON parse failed — storing raw output")
        return {
            "summary_ar": raw,
            "key_terms": [],
            "style": style,
            "backend": backend,
            "parse_error": "all parse strategies failed",
        }


# ── Qwen system prompts ───────────────────────────────────────────────────────
QWEN_PROMPTS = {
    # ── Student notes style (recommended) ────────────────────────────────────
    # Arabic explanations + English technical terms inline, structured like
    # how a student would write notes for revision — not a formal summary.
    # Matches the natural code-switching style of the lectures themselves.
    "notes": textwrap.dedent("""
        أنت طالب ذكي تكتب ملاحظات دراسية من محاضرة جامعية في مجال الذكاء الاصطناعي.

        القواعد الأساسية:
        - اكتب بالعربية العامية الطبيعية (مثل المحاضرة نفسها) مع الـ technical terms بالإنجليزية inline
        - لا تُترجم المصطلحات التقنية أبداً — اتركها بالإنجليزية كما هي (مثل: features, classifier, training, inference)
        - الأسلوب: نقاط قصيرة وواضحة — مثل ما يكتبه طالب فاهم للمحاضرة
        - ركز على: الـ concepts الرئيسية، ليش مهمة، وأمثلة ذكرها المحاضر
        - اشمل أي analogies أو أمثلة توضيحية ذكرها المحاضر — هاي بتساعد على الفهم
        - رتّب بعناوين منطقية حسب المواضيع اللي اتغطت في المحاضرة
        - لا تكتب مقدمة ولا خاتمة رسمية — مباشرة للنقاط

        صيغة الإخراج — JSON فقط، لا شرح، لا backticks:
        {
          "notes": "## عنوان الموضوع الأول\\n- نقطة\\n- نقطة\\n\\n## عنوان الموضوع الثاني\\n- نقطة\\n...",
          "key_terms": ["term1", "term2", ...]
        }
    """).strip(),
    # ── Formal MSA summary (for report baseline comparison only) ─────────────
    # Pure MSA paragraph form — useful to show contrast with notes style
    # and as the mBART-comparable output for ROUGE evaluation.
    "msa": textwrap.dedent("""
        أنت مساعد أكاديمي. اكتب ملخصاً رسمياً بالعربية الفصحى (150–200 كلمة).
        استخدم المصطلحات العربية المعتمدة. لا تُدرج مصطلحات إنجليزية.
        أعِد JSON فقط — لا شرح، لا مقدمة:
        {"summary_ar": "...", "key_terms": ["term1", "term2", ...]}
    """).strip(),
    # ── Code-switch paragraph (middle ground) ────────────────────────────────
    # Formal paragraph structure but keeps English terms inline.
    # Useful as a third comparison point in your evaluation.
    "code-switch": textwrap.dedent("""
        أنت مساعد أكاديمي متخصص في الذكاء الاصطناعي.
        اكتب ملخصاً بالعربية الفصحى (150–200 كلمة) مع الاحتفاظ بالمصطلحات التقنية
        الإنجليزية كما هي inline (مثل: classifier, features, training, inference).
        لا تُترجم أو تُعرِّب المصطلحات التقنية — اتركها بالإنجليزية.
        أعِد JSON فقط — لا شرح، لا مقدمة:
        {"summary_ar": "...", "key_terms": ["term1", "term2", ...]}
    """).strip(),
}


def build_qwen_messages(
    transcript: str, style: str, key_terms: List[str] = None
) -> List[Dict]:
    if len(transcript) > QWEN_MAX_INPUT_CHARS:
        transcript = transcript[:QWEN_MAX_INPUT_CHARS] + "\n[... النص مقتطع ...]"

    system = QWEN_PROMPTS[style]

    # Inject extracted key terms so Qwen preserves English terminology even
    # when the transcript is heavily Arabicized (e.g. baseline Whisper output
    # that transcribed "Gaussian" as "غاوس" etc.). Without this, the model
    # has no English to code-switch with and falls back to pure Arabic.
    # The terms list is appended AFTER the style prompt so it acts as a
    # hard constraint on top of the style instructions, not a competing signal.
    if key_terms:
        terms_str = ", ".join(key_terms)
        system += (
            "\n\nالمصطلحات التقنية التالية موجودة في هذه المحاضرة — "
            "احتفظ بها بالإنجليزية كما هي في ملخصك ولا تُترجمها أو تُعرِّبها:\n"
            + terms_str
        )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"النص:\n\n{transcript}"},
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Report writer
# ══════════════════════════════════════════════════════════════════════════════


def write_comparison_report(all_results: Dict, report_path: str) -> None:
    sep = "═" * 70
    sep2 = "─" * 70

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"{sep}\n")
        f.write("  SUMMARIZATION COMPARISON REPORT\n")
        f.write("  Palestinian Arabic NLP Lectures\n")
        f.write(f"{sep}\n\n")
        f.write("  Runs compared:\n")
        f.write(
            "    A — Baseline Whisper transcript   (transcripts from vanilla large-v3)\n"
        )
        f.write("    B — Fine-tuned Whisper transcript (transcripts from your model)\n")
        f.write(
            "    C — Hand-corrected reference      (upper bound — best possible input)\n"
        )
        f.write(f"\n{sep}\n\n")

        # Get all lecture IDs (union across all runs)
        all_lecture_ids = set()
        for run_results in all_results.values():
            for entry in run_results:
                all_lecture_ids.add(entry["lecture_id"])

        for lecture_id in sorted(all_lecture_ids):
            f.write(f"LECTURE: {lecture_id}\n")
            f.write(f"{sep2}\n\n")

            for run_key, run_results in all_results.items():
                entry = next(
                    (e for e in run_results if e["lecture_id"] == lecture_id), None
                )
                if not entry:
                    continue

                label = RUN_LABEL_MAP.get(run_key, run_key)
                f.write(f"[ {label} ]\n\n")

                terms = entry.get("key_terms_transcript", [])
                if terms:
                    f.write(f"  Key terms in transcript: {', '.join(terms)}\n\n")

                # mBART
                f.write("  mBART summary:\n")
                mbart = entry.get("mbart", {})
                if mbart.get("error"):
                    f.write(f"  ERROR: {mbart['error']}\n")
                elif mbart.get("summary_ar"):
                    f.write("  " + mbart["summary_ar"] + "\n")
                else:
                    f.write("  (skipped)\n")

                f.write("\n")

                # AraBART
                f.write("  AraBART summary:\n")
                arabart = entry.get("arabart", {})
                if arabart.get("error"):
                    f.write(f"  ERROR: {arabart['error']}\n")
                elif arabart.get("summary_ar"):
                    f.write("  " + arabart["summary_ar"] + "\n")
                else:
                    f.write("  (skipped)\n")

                f.write("\n")

                # Qwen
                f.write("  Qwen summary:\n")
                qwen = entry.get("qwen", {})
                if qwen.get("error"):
                    f.write(f"  ERROR: {qwen['error']}\n")
                elif qwen.get("summary_ar"):
                    f.write("  " + qwen["summary_ar"] + "\n")
                    if qwen.get("key_terms"):
                        f.write(f"  Key terms (Qwen): {', '.join(qwen['key_terms'])}\n")
                    f.write(f"  Style: {qwen.get('style', '?')}\n")
                else:
                    f.write("  (skipped)\n")

                f.write(f"\n{sep2}\n\n")

            f.write(f"{sep}\n\n")

    logger.info(f"Comparison report written → {report_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Merge-aware save helper
# ══════════════════════════════════════════════════════════════════════════════


def _merge_and_save(
    out_path: str,
    new_results: list,
    run_mbart: bool,
    run_arabart: bool,
    run_qwen: bool,
) -> None:
    """
    Save run_results to out_path, merging with any existing file so that
    running --qwen-only followed by --mbart-only (or any combination) does
    NOT erase the previously saved model outputs.

    Only the model fields that were actually computed this session are
    overwritten; all other model fields in the existing file are preserved.
    """
    existing: dict = {}
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                for entry in json.load(f):
                    lid = entry.get("lecture_id")
                    if lid:
                        existing[lid] = entry
            logger.info(
                "  Merging into existing %s (%d entries)", out_path, len(existing)
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning(
                "Could not parse existing %s (%s) — starting fresh.", out_path, e
            )

    for entry in new_results:
        lid = entry["lecture_id"]
        if lid not in existing:
            existing[lid] = entry
        else:
            prev = existing[lid]
            # Only overwrite model fields for models that ran this session
            if run_mbart:
                prev["mbart"] = entry["mbart"]
            if run_arabart:
                prev["arabart"] = entry["arabart"]
            if run_qwen:
                prev["qwen"] = entry["qwen"]
            # Always refresh shared metadata
            for key in (
                "key_terms_transcript",
                "transcript_chars",
                "transcript_source",
                "transcript_field",
                "run",
            ):
                if key in entry:
                    prev[key] = entry[key]

    merged = list(existing.values())
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default=DEFAULT_INPUT, help="transcripts.json from transcribe.py"
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["base", "ft", "ref"],
        choices=["base", "ft", "ref"],
        help="Which transcript sources to summarize",
    )
    parser.add_argument(
        "--mbart-only",
        action="store_true",
        help="Run mBART only (skip Qwen and AraBART)",
    )
    parser.add_argument(
        "--qwen-only",
        action="store_true",
        help="Run Qwen only (skip mBART and AraBART)",
    )
    parser.add_argument(
        "--arabart-only",
        action="store_true",
        help="Run AraBART only (skip mBART and Qwen)",
    )
    parser.add_argument(
        "--arabart",
        action="store_true",
        help="Also run AraBART alongside mBART and Qwen. "
        "Adds moussaKam/AraBART as a third Arabic-only seq2seq model. "
        "Ignored when --arabart-only is set (AraBART already the only model).",
    )
    parser.add_argument(
        "--skip-flagged",
        action="store_true",
        help="Exclude flagged segments from transcript text",
    )
    parser.add_argument(
        "--lecture",
        default=None,
        help="Process only this lecture ID (e.g. NLP_20260221). "
        "Results are merged into the existing output file so "
        "you can run lectures separately without losing prior results.",
    )
    parser.add_argument(
        "--qwen-style",
        default="notes",
        choices=["notes", "code-switch", "msa"],
        help="notes: student-style bullet notes (recommended) | "
        "code-switch: paragraph + English terms | "
        "msa: pure formal Arabic",
    )
    # NEW: Reference style — controls Qwen style for hand-corrected (ref) run
    parser.add_argument(
        "--ref-style",
        default="code-switch",
        choices=["notes", "code-switch", "msa"],
        help="Style for reference (hand-corrected) run only. "
        "code-switch (default): formal paragraph + English terms — "
        "best for evaluation as a universal gold standard. | "
        "notes: student-style bullet notes | "
        "msa: pure formal Arabic",
    )
    # NEW: Early stopping for MLX generation
    parser.add_argument(
        "--early-stop",
        action="store_true",
        help="Stop MLX generation at the first balanced JSON closing brace. "
        "Prevents post-JSON commentary and improves parse reliability.",
    )
    parser.add_argument("--report", default="summaries_comparison_report.txt")
    args = parser.parse_args()

    device = get_device()

    # Resolve which models to run (--arabart-only overrides everything)
    run_mbart = not (args.qwen_only or args.arabart_only)
    run_arabart = args.arabart_only or (args.arabart and not args.qwen_only)
    run_qwen = not (args.mbart_only or args.arabart_only)

    logger.info("Runs       : %s", args.runs)
    logger.info("Qwen style : %s", args.qwen_style)
    logger.info("Ref style  : %s", args.ref_style)
    logger.info("Early stop : %s", args.early_stop)
    logger.info(
        "Models     : %s",
        " + ".join(
            filter(
                None,
                [
                    "mBART" if run_mbart else None,
                    "AraBART" if run_arabart else None,
                    "Qwen" if run_qwen else None,
                ],
            )
        ),
    )
    if device == "mps" and run_qwen and not _mlx_available():
        logger.warning(
            "mlx-lm not found — Qwen will run on MPS/PyTorch (float32, slow). "
            "Run `pip install mlx-lm` to enable the MLX backend."
        )

    # ── Load transcripts.json ─────────────────────────────────────────────────
    if not os.path.exists(args.input):
        logger.error("Input not found: %s  — run transcribe.py first", args.input)
        return
    with open(args.input, "r", encoding="utf-8") as f:
        lectures = json.load(f)
    logger.info("Loaded %d lecture(s)", len(lectures))

    # ── Load models (once, shared across all runs) ────────────────────────────
    mbart_tok = mbart_model = None
    arabart_tok = arabart_model = None
    qwen_tok = qwen_model = None
    qwen_backend = "torch"  # updated by load_qwen

    if run_mbart:
        mbart_tok, mbart_model = load_mbart(device)
        log_vram("after mBART load")
    if run_arabart:
        arabart_tok, arabart_model = load_arabart(device)
        log_vram("after AraBART load")
    if run_qwen:
        qwen_tok, qwen_model, qwen_backend = load_qwen(device)
        log_vram("after Qwen load")

    # ── Helper: unload all models and flush MPS allocator ────────────────────
    def _mps_unload():
        nonlocal mbart_tok, mbart_model, arabart_tok, arabart_model
        nonlocal qwen_tok, qwen_model
        logger.info("  ↺ MPS: unloading models to free memory …")
        mbart_tok = mbart_model = None
        arabart_tok = arabart_model = None
        qwen_tok = qwen_model = None
        # Two GC passes: first clears Python reference cycles, second frees
        # any objects whose __del__ was triggered by the first pass.
        gc.collect()
        gc.collect()
        torch.mps.empty_cache()
        # Brief sleep to let macOS reclaim unified-memory pages before the next
        # model load. Without this, the allocator may hand back the same
        # fragmented pages immediately, causing MPS to OOM on the next .to(device).
        time.sleep(1.0)
        # Log process RSS so we can confirm memory was actually returned.
        try:
            import resource

            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
            logger.info("  ↺ MPS: process RSS after unload: %.0f MB", rss_mb)
        except Exception:
            pass

    def _mps_reload():
        nonlocal mbart_tok, mbart_model, arabart_tok, arabart_model
        nonlocal qwen_tok, qwen_model, qwen_backend
        logger.info("  ↺ MPS: reloading models …")
        if run_mbart:
            mbart_tok, mbart_model = load_mbart(device)
        if run_arabart:
            arabart_tok, arabart_model = load_arabart(device)
        if run_qwen:
            qwen_tok, qwen_model, qwen_backend = load_qwen(device)

    # ── Run summarization for each source ─────────────────────────────────────
    all_results: Dict[str, list] = {}

    # Build the filtered lecture list once so we can detect the last item.
    # Using a pre-filtered list avoids the bug where `lecture is not lectures[-1]`
    # checked against the raw list and skipped the unload when a --lecture filter
    # was active, causing MPS memory to accumulate across runs.
    def _filtered_lectures():
        return [
            lec
            for lec in lectures
            if not args.lecture or lec["lecture_id"] == args.lecture
        ]

    for run_idx, run_key in enumerate(args.runs):
        field = RUN_FIELD_MAP[run_key]
        label = RUN_LABEL_MAP[run_key]
        out_path = RUN_OUTPUT_MAP[run_key]

        logger.info(f"\n{'━'*60}")
        logger.info(f"RUN: {label}")
        logger.info(f"     Using field: '{field}'")
        logger.info(f"{'━'*60}")

        run_results = []
        active_lectures = _filtered_lectures()
        is_last_run = run_idx == len(args.runs) - 1

        for lec_idx, lecture in enumerate(active_lectures):
            lecture_id = lecture["lecture_id"]
            is_last_lecture = lec_idx == len(active_lectures) - 1

            segments = lecture["segments"]

            # Skip flagged if requested
            if args.skip_flagged:
                segments = [s for s in segments if not s.get("flagged", False)]

            # Build transcript from the chosen field
            transcript = " ".join(
                s.get(field, "") for s in segments if s.get(field, "").strip()
            )

            if not transcript.strip():
                logger.warning(
                    f"  No text in field '{field}' for {lecture_id} — skipping"
                )
                continue

            logger.info(f"\n  Lecture : {lecture_id}")
            logger.info(f"  Source  : {label}")
            logger.info(
                f"  Length  : {len(transcript)} chars, "
                f"~{len(transcript.split())} words"
            )

            key_terms = extract_key_terms(transcript)
            logger.info(f"  Terms   : {key_terms}")

            entry = {
                "lecture_id": lecture_id,
                "run": run_key,
                "transcript_source": label,
                "transcript_field": field,
                "transcript_chars": len(transcript),
                "key_terms_transcript": key_terms,
                "mbart": {},
                "arabart": {},
                "qwen": {},
            }

            # mBART
            if mbart_model is not None:
                logger.info("  → mBART …")
                try:
                    summary = summarize_mbart(
                        transcript, mbart_tok, mbart_model, device
                    )
                    entry["mbart"] = {"summary_ar": summary, "error": None}
                    logger.info(f"  ✓ mBART ({len(summary)} chars)")
                except Exception as e:
                    logger.error(f"  mBART error: {e}")
                    entry["mbart"] = {"summary_ar": "", "error": str(e)}

            # AraBART
            if arabart_model is not None:
                logger.info("  → AraBART …")
                try:
                    summary = summarize_arabart(
                        transcript, arabart_tok, arabart_model, device
                    )
                    entry["arabart"] = {"summary_ar": summary, "error": None}
                    logger.info(f"  ✓ AraBART ({len(summary)} chars)")
                except Exception as e:
                    logger.error(f"  AraBART error: {e}")
                    entry["arabart"] = {"summary_ar": "", "error": str(e)}

            # Qwen — use ref-style for reference run, qwen-style for others
            if qwen_model is not None:
                # NEW: Use ref-style for hand-corrected reference run
                if run_key == "ref":
                    effective_style = args.ref_style
                else:
                    effective_style = args.qwen_style

                logger.info(f"  → Qwen [{effective_style} / {qwen_backend}] …")
                try:
                    result = summarize_qwen(
                        transcript,
                        qwen_tok,
                        qwen_model,
                        device,
                        effective_style,
                        qwen_backend,
                        early_stop=args.early_stop,
                        key_terms=key_terms,
                    )
                    entry["qwen"] = result
                    logger.info(f"  ✓ Qwen ({len(result.get('summary_ar',''))} chars)")
                except Exception as e:
                    logger.error(f"  Qwen error: {e}")
                    entry["qwen"] = {"summary_ar": "", "key_terms": [], "error": str(e)}

            # Free memory after each lecture.
            # On MPS, empty_cache() alone doesn't release model weights back to
            # the OS — the kernel sees them as still in use and starts swapping.
            # Strategy:
            #   - Between lectures within a run: always unload + reload so each
            #     lecture starts with a clean MPS memory slate.
            #   - Between runs: always unload; reload only if more runs follow.
            #   - After the very last lecture of the very last run: just flush.
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
                log_vram(f"after {lecture_id}")
            elif device == "mps":
                if not is_last_lecture:
                    # More lectures remain in this run — cycle models
                    _mps_unload()
                    _mps_reload()
                elif not is_last_run:
                    # Last lecture of this run, but more runs follow — unload now;
                    # reload will happen at the top of the next run iteration
                    _mps_unload()
                else:
                    # Truly last lecture of last run — just flush
                    torch.mps.empty_cache()

            run_results.append(entry)

        # If we unloaded at end of the previous run's last lecture, reload now
        # before the next run starts (only needed when models are None on MPS).
        if device == "mps" and not is_last_run:
            if (
                (run_mbart and mbart_model is None)
                or (run_arabart and arabart_model is None)
                or (run_qwen and qwen_model is None)
            ):
                _mps_reload()

        # Merge-aware save: preserve outputs from other models in existing file
        _merge_and_save(out_path, run_results, run_mbart, run_arabart, run_qwen)
        logger.info(f"\n  Saved → {out_path}")
        all_results[run_key] = run_results

    # ── Write comparison report ───────────────────────────────────────────────
    write_comparison_report(all_results, args.report)

    logger.info(f"\n{'━'*60}")
    logger.info("✓ All runs complete.")
    for run_key in args.runs:
        logger.info(f"   {RUN_LABEL_MAP[run_key]:<35} → {RUN_OUTPUT_MAP[run_key]}")
    logger.info(f"   {'Comparison report':<35} → {args.report}")
    logger.info(f"{'━'*60}")
    logger.info("Models run this session:")
    if run_mbart:
        logger.info(
            "   mBART   (facebook/mbart-large-cc25)         — multilingual seq2seq"
        )
    if run_arabart:
        logger.info(
            "   AraBART (moussaKam/AraBART)                  — Arabic-only seq2seq"
        )
    if run_qwen:
        logger.info(
            "   Qwen    (Qwen2.5-7B-Instruct / %-4s)        — instruction-tuned LLM",
            qwen_backend,
        )
    logger.info("\nNext: Normalize Summaries → run normalize_summaries.py")


if __name__ == "__main__":
    main()
