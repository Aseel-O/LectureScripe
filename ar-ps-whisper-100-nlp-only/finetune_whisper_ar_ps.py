#!/usr/bin/env python3
"""
Fine-tune OpenAI Whisper large-v3 on Palestinian Arabic (ar-PS)
with Arabic-English code-switching using LoRA (PEFT).

Target hardware : A100-SXM4-80GB (shared, 20 GB VRAM limit)
CUDA            : 12.2
Expected VRAM   : ~14–17 GB with gradient_checkpointing=True
Expected time   : ~2–3 h for 20 epochs on 570 segments

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Required pip installs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
pip install torch==2.2.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.40.0
pip install peft>=0.10.0
pip install datasets>=2.18.0
pip install accelerate>=0.29.0
pip install evaluate>=0.4.1
pip install jiwer>=3.0.3
pip install soundfile>=0.12.1
pip install librosa>=0.10.1
pip install tensorboard>=2.16.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import re
import json
import random
import logging
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
import librosa
import jiwer
import evaluate

from datasets import Dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
)

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  ← change PROFILE to switch between environments
# ══════════════════════════════════════════════════════════════════════════════

import platform, torch

PROFILE = "local_test"  # ← "local_test" for Mac, "cluster" for A100

# ── Paths (same for both profiles) ───────────────────────────────────────────
JSONL_PATH = "data/dataset_final.jsonl"
AUDIO_ROOT = "data"  # resolves to: data/audio/segment_001.wav
OUTPUT_DIR = "./whisper-lora-ar-ps"
MERGED_DIR = "./whisper-merged-ar-ps"

# ── Model (same for both profiles) ───────────────────────────────────────────
MODEL_ID = "openai/whisper-large-v3"
LANGUAGE = "arabic"
TASK = "transcribe"
SAMPLING_RATE = 16_000

# ── Dataset split (same for both profiles) ───────────────────────────────────
TRAIN_FRAC = 0.80
DEV_FRAC = 0.10
SEED = 42

# ── LoRA (same for both profiles) ────────────────────────────────────────────
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# ── Optimizer (same for both profiles) ───────────────────────────────────────  ← ADD THIS BLOCK
LEARNING_RATE = 1e-4  # standard LoRA fine-tuning LR
LR_SCHEDULER = "linear"
MAX_GRAD_NORM = 1.0

# ── Profile-specific settings ─────────────────────────────────────────────────
if PROFILE == "local_test":
    # Mac M5 Pro — 48 GB unified memory, MPS backend
    # 100 segments: ~50 train → batch=2, accum=8 → ~6 optimizer steps/epoch
    TORCH_DTYPE = torch.float16  # bf16 unreliable on MPS
    # For local Mac testing:
    # Replace the two constants at the top with:
    IS_CUDA = torch.cuda.is_available()
    BF16 = IS_CUDA  # True on A100, False on Mac
    FP16 = False  # Never use fp16 (bfloat16 is strictly better on A100)
    # Restore BF16 = True / FP16 = False when you move to the A100
    PER_DEVICE_TRAIN_BS = 2
    PER_DEVICE_EVAL_BS = 2
    GRADIENT_ACCUMULATION_STEPS = 8  # effective batch = 2×8 = 16
    NUM_EPOCHS = 10  # enough to validate the pipeline
    WARMUP_STEPS = 5  # must be < total optimizer steps
    EVAL_STEPS = 3  # eval every 3 optimizer steps
    SAVE_STEPS = 3
    LOGGING_STEPS = 1
    EARLY_STOP_PATIENCE = 2
    DATALOADER_WORKERS = 0  # spawn-based macOS: must be 0
    DATALOADER_PIN = False  # CUDA-only feature

elif PROFILE == "cluster":
    # A100 SXM4 — 20 GB VRAM limit, CUDA backend
    # 709 segments: ~570 train → batch=4, accum=4 → ~36 optimizer steps/epoch
    TORCH_DTYPE = torch.bfloat16
    # Replace the two constants at the top with:
    IS_CUDA = torch.cuda.is_available()
    BF16 = IS_CUDA  # True on A100, False on Mac
    FP16 = False  # Never use fp16 (bfloat16 is strictly better on A100)
    PER_DEVICE_TRAIN_BS = 4
    PER_DEVICE_EVAL_BS = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    NUM_EPOCHS = 20
    WARMUP_STEPS = 50
    # EVAL_STEPS               = 36
    # SAVE_STEPS               = 36
    LOGGING_STEPS = 10
    EARLY_STOP_PATIENCE = 3
    DATALOADER_WORKERS = 2
    DATALOADER_PIN = True

MAX_NEW_TOKENS = 225
# ── Dataset field names ───────────────────────────────────────────────────────
TEXT_FIELD = "sentence"  # your JSONL uses "sentence", not "text"

# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATASET  — load & lecture-based split
# ══════════════════════════════════════════════════════════════════════════════


def load_jsonl(path: str) -> List[Dict]:
    """Load all records from a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records):,} segments from '{path}'")
    return records


def resolve_audio_path(record: Dict, audio_root: str) -> str:
    """
    Extract the audio file path from a record.
    Handles both nested  {"audio": {"path": "..."}}
    and flat             {"audio.path": "..."}  field naming conventions.
    """
    if "audio" in record and isinstance(record["audio"], dict):
        rel = record["audio"]["path"]
    elif "audio.path" in record:
        rel = record["audio.path"]
    else:
        raise KeyError(
            f"Cannot find audio path key in record. "
            f"Available keys: {list(record.keys())}"
        )
    if audio_root and not os.path.isabs(rel):
        return os.path.join(audio_root, rel)
    return rel


def split_by_lecture(
    records: List[Dict],
    train_frac: float = 0.80,
    dev_frac: float = 0.10,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """
    Assign complete lectures (source_file values) to train / dev / test.
    No segments from the same lecture appear in more than one split —
    this is essential to prevent acoustic leakage between splits.

    Special-case: if only 2 lectures are present (pilot run), lecture 0
    goes to train and lecture 1 is divided 50/50 between dev and test.
    """
    # Group segments by lecture
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for rec in records:
        groups[rec["source_file"]].append(rec)

    lectures = sorted(groups.keys())  # sorted → reproducible
    n_lec = len(lectures)
    logger.info(f"Found {n_lec} lecture(s): {lectures}")

    rng = random.Random(seed)
    rng.shuffle(lectures)

    # ── Special-case: only 1 lecture ─────────────────────────────────────────
    if n_lec == 1:
        logger.warning(
            "Only 1 lecture found — splitting segments in order: "
            "first 80% → train, next 10% → dev, last 10% → test."
        )
        segs = sorted(groups[lectures[0]], key=lambda r: r["segment_index"])
        n = len(segs)
        t1 = int(n * 0.80)
        t2 = int(n * 0.90)
        splits = {
            "train": segs[:t1],
            "dev": segs[t1:t2],
            "test": segs[t2:],
        }
        _log_splits(splits)
        return splits

    # ── Special-case: only 2 lectures ────────────────────────────────────────
    if n_lec == 2:
        logger.warning(
            "Only 2 lectures available (pilot run). "
            "Lecture 0 → train; lecture 1 split 50/50 → dev / test."
        )
        other = list(groups[lectures[1]])
        rng.shuffle(other)
        mid = len(other) // 2
        splits = {
            "train": list(groups[lectures[0]]),
            "dev": other[:mid],
            "test": other[mid:],
        }
        _log_splits(splits)
        return splits

    # ── General case: distribute lectures by cumulative segment count ─────────
    total = len(records)
    train_lectures, dev_lectures, test_lectures = [], [], []
    n_train = n_dev = 0

    for lec in lectures:
        n = len(groups[lec])
        if n_train / max(total, 1) < train_frac:
            train_lectures.append(lec)
            n_train += n
        elif n_dev / max(total, 1) < dev_frac:
            dev_lectures.append(lec)
            n_dev += n
        else:
            test_lectures.append(lec)

    # Safety: guarantee at least one lecture per split
    if not dev_lectures and train_lectures:
        dev_lectures = [train_lectures.pop()]
    if not test_lectures and dev_lectures:
        test_lectures = [dev_lectures.pop()]

    splits = {
        "train": [s for lec in train_lectures for s in groups[lec]],
        "dev": [s for lec in dev_lectures for s in groups[lec]],
        "test": [s for lec in test_lectures for s in groups[lec]],
    }
    _log_splits(splits, train_lectures, dev_lectures, test_lectures)
    return splits


def _log_splits(splits, train_lecs=None, dev_lecs=None, test_lecs=None):
    for name, segs in splits.items():
        lecs = {"train": train_lecs, "dev": dev_lecs, "test": test_lecs}.get(name)
        lec_str = f"← {lecs}" if lecs else ""
        logger.info(f"  {name:5s}: {len(segs):4d} segments {lec_str}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  AUDIO PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════


def load_audio(path: str, sr: int = 16_000) -> np.ndarray:
    """Load any audio file, resample to sr, convert to mono float32."""
    audio, _ = librosa.load(path, sr=sr, mono=True, dtype=np.float32)
    return audio


def build_hf_dataset(
    records: List[Dict],
    processor: WhisperProcessor,
    audio_root: str = "",
    sr: int = 16_000,
    split_name: str = "",
) -> Dataset:
    """
    Convert JSONL records → HuggingFace Dataset with:
      input_features : log-mel spectrogram [80, 3000] (fixed for ≤30 s audio)
      labels         : token IDs of the transcription
    """
    logger.info(f"  Preprocessing {len(records)} segments ({split_name})…")

    input_features_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []

    for i, rec in enumerate(records):
        if i % 100 == 0 and i > 0:
            logger.info(f"    {i}/{len(records)}")

        audio_path = resolve_audio_path(rec, audio_root)
        audio_arr = load_audio(audio_path, sr)

        # Log-mel spectrogram → always [80, 3000] for audio ≤ 30 s
        feats = processor.feature_extractor(
            audio_arr,
            sampling_rate=sr,
            return_tensors="pt",
        ).input_features[
            0
        ]  # shape: [80, 3000]

        # Tokenise transcription (max 448 tokens; Whisper decoder limit)
        tok = processor.tokenizer(
            rec[TEXT_FIELD],
            return_tensors="pt",
            truncation=True,
            max_length=448,
        ).input_ids[0]

        input_features_list.append(feats)
        labels_list.append(tok)

    return Dataset.from_dict(
        {
            "input_features": input_features_list,
            "labels": labels_list,
        }
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3.  DATA COLLATOR
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Batch collation for Whisper fine-tuning:
    • input_features – stack as-is (already [80, 3000] for every ≤30 s segment)
    • labels         – right-pad to batch max-length; fill padding with -100
                       so cross-entropy ignores those positions.
    The leading BOS token that the tokenizer prepends is stripped here because
    Seq2SeqTrainer re-adds it via decoder_start_token_id at the start of
    autoregressive decoding.
    """

    processor: Any
    decoder_start_token_id: int

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:

        # ── Input spectrograms ────────────────────────────────────────────────
        input_features = torch.stack(
            [
                (
                    f["input_features"]
                    if isinstance(f["input_features"], torch.Tensor)
                    else torch.tensor(f["input_features"])
                )
                for f in features
            ]
        )

        # ── Labels ───────────────────────────────────────────────────────────
        label_features = [
            {
                "input_ids": (
                    f["labels"].tolist()
                    if isinstance(f["labels"], torch.Tensor)
                    else f["labels"]
                )
            }
            for f in features
        ]

        padded = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
            padding=True,
        )

        # Replace PAD id with -100 → ignored by cross-entropy loss
        labels = padded["input_ids"].masked_fill(padded.attention_mask.ne(1), -100)

        # Strip the BOS token prepended by the tokenizer
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        return {"input_features": input_features, "labels": labels}


# ══════════════════════════════════════════════════════════════════════════════
# normalization function for Arabic text, preserving English words for WER scoring
# ═════════════════════════════════════════════════════════════════════════════
def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text before WER scoring, while preserving English words
    so that transliterating English terms (inference → الإنفرانس) is still
    counted as an error.

    Arabic normalization applied:
    1. Remove tashkeel (diacritics)
    2. Normalize alef variants → bare alef
    3. Normalize hamza letters (ئ→ي, ؤ→و)
    4. Remove tatweel (ـ)
    5. Normalize teh marbuta (ة→ه)
    6. Normalize alef maqsura (ى→ي)
    7. Strip punctuation
    8. Lowercase English portions only
    9. Collapse whitespace

    English words are left as-is (after lowercasing) so Latin-script
    technical terms are compared exactly against the reference.
    """
    # Split into Arabic and English tokens and process separately
    tokens = text.split()
    normalized_tokens = []

    for token in tokens:
        # Detect if token is primarily Latin (English word)
        latin_chars = sum(1 for c in token if c.isascii() and c.isalpha())
        total_alpha = sum(1 for c in token if c.isalpha())

        if total_alpha > 0 and latin_chars / total_alpha > 0.5:
            # English token — just lowercase, keep as-is
            normalized_tokens.append(token.lower())
        else:
            # Arabic token — apply full normalization
            t = token

            # 1. Remove tashkeel
            t = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670]", "", t)

            # 2. Normalize alef variants → bare alef
            t = re.sub(r"[أإآٱ]", "ا", t)

            # 3. Normalize hamza
            t = t.replace("ئ", "ي").replace("ؤ", "و")

            # 4. Remove tatweel
            t = t.replace("ـ", "")

            # 5. Teh marbuta → ha
            t = t.replace("ة", "ه")

            # 6. Alef maqsura → ya
            t = t.replace("ى", "ي")

            # 7. Strip punctuation
            t = re.sub(r'[،؛؟!,;?!.:\'"()\[\]{}\-/\\]', "", t)

            # 8. Remove any remaining non-Arabic/non-Latin/non-digit chars
            t = re.sub(r"[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FFa-zA-Z0-9]", "", t)

            if t:
                normalized_tokens.append(t)

    return " ".join(normalized_tokens)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  METRICS  — WER with Arabic normalisation
# ══════════════════════════════════════════════════════════════════════════════


def make_compute_metrics(processor: WhisperProcessor):
    """
    Return a compute_metrics closure that:
    • decodes predicted token IDs (with skip_special_tokens=True)
    • applies normalize_arabic() to both hypothesis and reference
    • computes normalised WER via the 'evaluate' library
    • logs 3 example pairs per eval for sanity checking
    """
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")  # إدراج هذه الأداة

    def compute_metrics(pred) -> Dict[str, float]:
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 padding so the tokenizer can decode labels
        label_ids = np.where(
            label_ids != -100,
            label_ids,
            processor.tokenizer.pad_token_id,
        )

        # Clip pred_ids to valid token range before decoding
        vocab_size = processor.tokenizer.vocab_size
        pred_ids = np.clip(pred_ids, 0, vocab_size - 1)

        hyp_strs = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        ref_strs = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Log examples
        for i in range(min(3, len(hyp_strs))):
            logger.info(f"  [ex {i}] REF: {ref_strs[i][:120]}")
            logger.info(f"  [ex {i}] HYP: {hyp_strs[i][:120]}")

        # Raw WER (no normalization)
        wer_raw = wer_metric.compute(predictions=hyp_strs, references=ref_strs)
        cer_raw = cer_metric.compute(predictions=hyp_strs, references=ref_strs)

        # Normalized WER (Arabic normalization applied)
        hyp_norm = [normalize_arabic(s) for s in hyp_strs]
        ref_norm = [normalize_arabic(s) for s in ref_strs]
        wer_norm = wer_metric.compute(predictions=hyp_norm, references=ref_norm)
        cer_norm = cer_metric.compute(predictions=hyp_norm, references=ref_norm)

        return {
            "wer": round(float(wer_raw), 4),  # raw
            "cer": round(float(cer_raw), 4),  # raw
            "wer_norm": round(float(wer_norm), 4),  # normalized
            "cer_norm": round(float(cer_norm), 4),  # normalized
        }

    return compute_metrics


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MODEL + LoRA SETUP
# ══════════════════════════════════════════════════════════════════════════════


def load_model_with_lora(
    model_id: str,
    processor: WhisperProcessor,
) -> WhisperForConditionalGeneration:
    """
    1. Load Whisper large-v3 in bfloat16 (A100-native precision).
    2. Configure forced_decoder_ids → language=Arabic, task=transcribe.
    3. Wrap with LoRA via PEFT.
    4. Enable gradient checkpointing (saves ~35 % activation memory).
    5. Call enable_input_require_grads() — required when combining
       gradient checkpointing with PEFT frozen base weights.
    """
    # bf16 only on CUDA (A100); MPS and CPU must use float32
    is_cuda = torch.cuda.is_available()
    load_dtype = torch.bfloat16 if is_cuda else torch.float32
    logger.info(f"Loading base model: {model_id} …")
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=load_dtype,
        low_cpu_mem_usage=True,
    )

    # Force Arabic transcription prefix tokens on every generation call
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=LANGUAGE, task=TASK
    )
    model.generation_config.suppress_tokens = []
    model.config.use_cache = False  # Mandatory during training with grad checkpointing

    # ── LoRA config ───────────────────────────────────────────────────────────
    # task_type=SEQ_2_SEQ_LM makes PEFT handle Whisper's encoder-decoder arch.
    # PEFT matches target_modules by layer-name suffix, so "q_proj" and "v_proj"
    # are found in encoder self-attn, decoder self-attn AND decoder cross-attn —
    # giving good dialect adaptation coverage without the VRAM cost of k_proj /
    # out_proj.
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",  # Bias terms stay frozen
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model = get_peft_model(model, lora_cfg)

    # Required when GC is used with a PEFT-wrapped frozen model
    model.enable_input_require_grads()

    # Log trainable parameter statistics
    model.print_trainable_parameters()
    # Expected output ≈:
    #   trainable params: 7,864,320 || all params: 1,551,685,632 || trainable%: 0.51%

    return model


# ══════════════════════════════════════════════════════════════════════════════
# 7.  TRAINING
# ══════════════════════════════════════════════════════════════════════════════


def build_training_args(eval_steps: int, save_steps: int) -> Seq2SeqTrainingArguments:
    return Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        # ── Batch & gradient ──────────────────────────────────────────────────
        # Effective batch size = 4 × 4 = 16 utterances per optimizer step
        per_device_train_batch_size=PER_DEVICE_TRAIN_BS,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BS,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        # gradient_checkpointing trades recomputation for ~35 % less VRAM
        gradient_checkpointing=True,
        # ── Optimizer / LR ────────────────────────────────────────────────────
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_steps=WARMUP_STEPS,
        max_grad_norm=MAX_GRAD_NORM,
        optim="adamw_torch",  # standard AdamW (LoRA params only)
        # ── Epochs ────────────────────────────────────────────────────────────
        num_train_epochs=NUM_EPOCHS,
        # ── Precision ─────────────────────────────────────────────────────────
        bf16=BF16,
        fp16=FP16,
        # ── Evaluation & generation ───────────────────────────────────────────
        eval_strategy="steps",
        eval_steps=eval_steps,
        predict_with_generate=True,  # Autoregressive eval
        generation_max_length=MAX_NEW_TOKENS,
        # ── Checkpointing ─────────────────────────────────────────────────────
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,  # Keeps disk usage manageable
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # Use normalized CER for model selection
        greater_is_better=False,  # Lower WER = better model
        # ── Logging ───────────────────────────────────────────────────────────
        logging_steps=LOGGING_STEPS,
        logging_dir=os.path.join(OUTPUT_DIR, "tensorboard"),
        report_to=["tensorboard"],
        # ── DataLoader ────────────────────────────────────────────────────────
        dataloader_num_workers=DATALOADER_WORKERS,
        dataloader_pin_memory=DATALOADER_PIN,
        # ── Misc ──────────────────────────────────────────────────────────────
        seed=SEED,
        remove_unused_columns=False,  # Must be False: keeps input_features
        label_names=["labels"],
        push_to_hub=False,
    )


def find_last_checkpoint(output_dir: str) -> Optional[str]:
    """Return the path to the latest checkpoint-N folder, or None."""
    if not os.path.isdir(output_dir):
        return None
    ckpts = [
        d
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: int(x.split("-")[1]))
    path = os.path.join(output_dir, ckpts[-1])
    logger.info(f"Resuming from checkpoint: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 8.  MERGE LoRA → BASE MODEL  (for inference-ready deployment)
# ══════════════════════════════════════════════════════════════════════════════


def merge_and_save(
    adapter_dir: str,
    merged_dir: str,
    processor: WhisperProcessor,
) -> None:
    """
    Load the base model, apply the saved LoRA adapter weights, call
    merge_and_unload() to bake LoRA deltas into the base weights, and
    save a standard HuggingFace model that can be loaded with
    WhisperForConditionalGeneration.from_pretrained() — no PEFT needed.
    """
    logger.info("Merging LoRA weights into base model for deployment …")

    save_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=save_dtype,
        low_cpu_mem_usage=True,
    )

    peft_model = PeftModel.from_pretrained(base, adapter_dir)
    merged = peft_model.merge_and_unload()  # inlines ΔW = BA into W

    os.makedirs(merged_dir, exist_ok=True)
    merged.save_pretrained(merged_dir, safe_serialization=True)
    processor.save_pretrained(merged_dir)

    logger.info(f"Merged model saved → {merged_dir}")
    logger.info("Load for inference with:")
    logger.info(
        f"  model = WhisperForConditionalGeneration.from_pretrained('{merged_dir}')"
    )
    logger.info(f"  proc  = WhisperProcessor.from_pretrained('{merged_dir}')")


# ══════════════════════════════════════════════════════════════════════════════
# 9.  QUICK INFERENCE TEST
# ══════════════════════════════════════════════════════════════════════════════


def inference_test(
    record: Dict,
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    device: str = get_device(),
) -> None:
    """
    Run greedy decoding on a single sample from the test split and print
    the transcription vs. ground truth with a single-sample WER.
    """
    model.eval()
    model = model.to(device)

    # Cast model to float32 on MPS/CPU; float16 only works reliably on CUDA
    if device != "cuda":
        model = model.float()

    audio_path = resolve_audio_path(record, AUDIO_ROOT)
    audio_arr = load_audio(audio_path, SAMPLING_RATE)

    dtype = torch.float16 if device == "cuda" else torch.float32
    inputs = processor.feature_extractor(
        audio_arr,
        sampling_rate=SAMPLING_RATE,
        return_tensors="pt",
    ).input_features.to(device, dtype=dtype)

    with torch.no_grad():
        pred_ids = model.generate(
            inputs,
            language=LANGUAGE,
            task=TASK,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    hypothesis = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0]
    ground_truth = record[TEXT_FIELD]

    sample_wer = jiwer.wer(
        normalize_arabic(ground_truth),
        normalize_arabic(hypothesis),
    )

    sep = "─" * 64
    logger.info(f"\n{sep}")
    logger.info(f"  File     : {audio_path}")
    logger.info(f"  Duration : {record.get('duration_seconds', '?')} s")
    logger.info(f"  Locale   : {record.get('locale', 'ar-PS')}")
    logger.info(f"  Reference: {ground_truth}")
    logger.info(f"  Hypothesis: {hypothesis}")
    logger.info(
        f"  Sample WER (normalised): {sample_wer:.4f}  ({sample_wer * 100:.1f} %)"
    )
    logger.info(sep)


class WhisperPEFTTrainer(Seq2SeqTrainer):
    def __init__(self, *args, processor, **kwargs):
        super().__init__(*args, **kwargs)
        self._processor = processor

    def compute_loss(self, model, inputs, **kwargs):
        inputs["input_features"] = inputs["input_features"].to(
            dtype=next(model.parameters()).dtype
        )
        outputs = model.base_model.model(
            input_features=inputs["input_features"],
            labels=inputs["labels"],
        )
        return outputs.loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            inputs["input_features"] = inputs["input_features"].to(
                dtype=next(model.parameters()).dtype
            )
            loss_out = model.base_model.model(
                input_features=inputs["input_features"],
                labels=inputs["labels"],
            )
            loss = loss_out.loss.detach()

            if prediction_loss_only:
                return (loss, None, None)

            gen_kwargs = {
                "max_new_tokens": self.args.generation_max_length or MAX_NEW_TOKENS,
            }
            if hasattr(model, "generation_config"):
                cfg = model.generation_config
                if getattr(cfg, "forced_decoder_ids", None):
                    gen_kwargs["forced_decoder_ids"] = cfg.forced_decoder_ids

            generated = model.generate(
                input_features=inputs["input_features"],
                **gen_kwargs,
            )

            labels = inputs["labels"]
            max_gen_len = generated.shape[-1]
            max_lab_len = labels.shape[-1]

            if max_gen_len < max_lab_len:
                pad = torch.full(
                    (generated.shape[0], max_lab_len - max_gen_len),
                    self._processor.tokenizer.pad_token_id,
                    dtype=generated.dtype,
                    device=generated.device,
                )
                generated = torch.cat([generated, pad], dim=-1)

        return (loss, generated, labels)


# ══════════════════════════════════════════════════════════════════════════════
# 10.  MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    set_seed(SEED)
    logger.info("━" * 64)
    logger.info(" Whisper large-v3  |  LoRA  |  Palestinian Arabic (ar-PS)")
    logger.info("━" * 64)

    # ── 1. Load records & split ───────────────────────────────────────────────
    records = load_jsonl(JSONL_PATH)
    splits = split_by_lecture(records, TRAIN_FRAC, DEV_FRAC, SEED)

    # ── 2. Load processor (feature extractor + tokeniser) ─────────────────────
    logger.info("Loading WhisperProcessor …")
    processor = WhisperProcessor.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK)
    # Embed language / task tokens into every decoding sequence
    processor.tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK)

    # ── 3. Preprocess audio → mel features + token labels ────────────────────
    train_ds = build_hf_dataset(
        splits["train"], processor, AUDIO_ROOT, SAMPLING_RATE, "train"
    )
    dev_ds = build_hf_dataset(
        splits["dev"], processor, AUDIO_ROOT, SAMPLING_RATE, "dev"
    )
    test_ds = build_hf_dataset(
        splits["test"], processor, AUDIO_ROOT, SAMPLING_RATE, "test"
    )

    # --- ADD THIS LOGIC TO DYNAMICALLY CALIBRATE STEPS ---
    effective_batch_size = PER_DEVICE_TRAIN_BS * GRADIENT_ACCUMULATION_STEPS
    steps_per_epoch = int(np.ceil(len(train_ds) / effective_batch_size))

    # We want to evaluate and save exactly once per epoch
    dynamic_eval_steps = steps_per_epoch
    dynamic_save_steps = steps_per_epoch

    logger.info(f"→ Steps per epoch calculated: {steps_per_epoch}")
    logger.info(f"→ Evaluation & saving will occur every {dynamic_eval_steps} steps.")

    logger.info(
        f"Dataset sizes → train: {len(train_ds)}, "
        f"dev: {len(dev_ds)}, test: {len(test_ds)}"
    )

    # ── 4. Load model + LoRA ──────────────────────────────────────────────────
    model = load_model_with_lora(MODEL_ID, processor)

    # ── 5. Data collator ──────────────────────────────────────────────────────
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # ── 6. Build trainer ──────────────────────────────────────────────────────
    training_args = build_training_args(
        eval_steps=dynamic_eval_steps, save_steps=dynamic_save_steps
    )

    trainer = WhisperPEFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        compute_metrics=make_compute_metrics(processor),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)],
        processor=processor,  # ← add this
        # NOTE: do not pass tokenizer= here; we save processor separately
    )

    # ── 7. Train (with auto-resume) ───────────────────────────────────────────
    last_ckpt = find_last_checkpoint(OUTPUT_DIR)
    logger.info("Starting training …")
    train_result = trainer.train(resume_from_checkpoint=last_ckpt)

    # ── 8. Save LoRA adapter + processor ─────────────────────────────────────
    adapter_dir = os.path.join(OUTPUT_DIR, "final_adapter")
    trainer.save_model(adapter_dir)
    processor.save_pretrained(adapter_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    logger.info(f"LoRA adapter saved → {adapter_dir}")

    # ── 9. Evaluate on held-out test set ─────────────────────────────────────
    logger.info("Evaluating on test set …")
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    logger.info(f"Test WER: {test_metrics.get('test_wer', 'N/A'):.4f}")
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)

    # ── 10. Merge LoRA → base model for deployment ────────────────────────────
    # Free the PEFT model from GPU before loading fresh weights
    del model
    torch.cuda.empty_cache()

    merge_and_save(adapter_dir, MERGED_DIR, processor)

    # ── 11. Quick inference demo ──────────────────────────────────────────────
    logger.info("Running inference test on one sample from test split …")
    merged_model = WhisperForConditionalGeneration.from_pretrained(
        MERGED_DIR,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    inference_test(splits["test"][0], merged_model, processor)

    logger.info("✓  All done.")
    logger.info(f"   LoRA adapter  →  {adapter_dir}")
    logger.info(f"   Merged model  →  {MERGED_DIR}")

    # ── Save results to JSON ──────────────────────────────────────────────────
    results_json_path = os.path.join(OUTPUT_DIR, "training_results.json")
    results = {
        "profile": PROFILE,
        "model_id": MODEL_ID,
        "language": LANGUAGE,
        "task": TASK,
        "lora": {
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "target_modules": LORA_TARGET_MODULES,
        },
        "training": {
            "num_epochs": NUM_EPOCHS,
            "per_device_train_batch_size": PER_DEVICE_TRAIN_BS,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": PER_DEVICE_TRAIN_BS * GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": LEARNING_RATE,
            "lr_scheduler": LR_SCHEDULER,
            "warmup_steps": WARMUP_STEPS,
        },
        "dataset": {
            "jsonl_path": JSONL_PATH,
            "total_records": len(records),
            "train_segments": len(splits["train"]),
            "dev_segments": len(splits["dev"]),
            "test_segments": len(splits["test"]),
            "seed": SEED,
        },
        "train_metrics": train_result.metrics,
        "test_metrics": test_metrics,
        "output_dirs": {
            "lora_adapter": adapter_dir,
            "merged_model": MERGED_DIR,
        },
    }
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"✓ Results JSON saved → {results_json_path}")

    # ── Save results to TXT ───────────────────────────────────────────────────
    results_txt_path = os.path.join(OUTPUT_DIR, "training_results.txt")
    lines = []
    lines.append("=" * 64)
    lines.append("WHISPER FINE-TUNING RESULTS")
    lines.append("=" * 64)
    lines.append(f"Profile         : {PROFILE}")
    lines.append(f"Base model      : {MODEL_ID}")
    lines.append(f"Language / task : {LANGUAGE} / {TASK}")
    lines.append("")
    lines.append("── LoRA Configuration ──────────────────────────────────────")
    lines.append(f"  r              : {LORA_R}")
    lines.append(f"  alpha          : {LORA_ALPHA}")
    lines.append(f"  dropout        : {LORA_DROPOUT}")
    lines.append(f"  target modules : {LORA_TARGET_MODULES}")
    lines.append("")
    lines.append("── Training Configuration ──────────────────────────────────")
    lines.append(f"  Epochs                     : {NUM_EPOCHS}")
    lines.append(f"  Batch size (per device)    : {PER_DEVICE_TRAIN_BS}")
    lines.append(f"  Gradient accumulation      : {GRADIENT_ACCUMULATION_STEPS}")
    lines.append(
        f"  Effective batch size       : {PER_DEVICE_TRAIN_BS * GRADIENT_ACCUMULATION_STEPS}"
    )
    lines.append(f"  Learning rate              : {LEARNING_RATE}")
    lines.append(f"  LR scheduler               : {LR_SCHEDULER}")
    lines.append(f"  Warmup steps               : {WARMUP_STEPS}")
    lines.append("")
    lines.append("── Dataset Split ───────────────────────────────────────────")
    lines.append(f"  Total records  : {len(records)}")
    lines.append(f"  Train segments : {len(splits['train'])}")
    lines.append(f"  Dev segments   : {len(splits['dev'])}")
    lines.append(f"  Test segments  : {len(splits['test'])}")
    lines.append(f"  Seed           : {SEED}")
    lines.append("")
    lines.append("── Training Metrics ────────────────────────────────────────")
    for k, v in sorted(train_result.metrics.items()):
        lines.append(f"  {k:<35}: {v}")
    lines.append("")
    lines.append("── Test Metrics ────────────────────────────────────────────")
    for k, v in sorted(test_metrics.items()):
        val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        lines.append(f"  {k:<35}: {val_str}")
    lines.append("")
    lines.append("── Output Paths ────────────────────────────────────────────")
    lines.append(f"  LoRA adapter : {adapter_dir}")
    lines.append(f"  Merged model : {MERGED_DIR}")
    lines.append("=" * 64)

    with open(results_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"✓ Results TXT saved  → {results_txt_path}")


if __name__ == "__main__":
    main()
