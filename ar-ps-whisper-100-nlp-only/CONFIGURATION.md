# Configuration Guide

Detailed reference for configuring each script to match your hardware and needs.

## 📋 Table of Contents

1. [Fine-tuning Configuration](#fine-tuning-configuration)
2. [Model Selection](#model-selection)
3. [Hardware Optimization](#hardware-optimization)
4. [Data Processing Options](#data-processing-options)
5. [Evaluation Settings](#evaluation-settings)
6. [Environment Variables](#environment-variables)

---

## Fine-tuning Configuration

Edit these settings in **`finetune_whisper_ar_ps.py`** (lines 84-160):

### Hardware Profile Selection (Line 84)

```python
PROFILE = "local_test"    # For macOS / local development
# OR
PROFILE = "cluster"       # For A100 CUDA cluster
```

### Profile: `local_test` (macOS)

```python
PROFILE = "local_test"

# Memory & Computation
TORCH_DTYPE = torch.float32        # FP32 required on MPS
PER_DEVICE_TRAIN_BS = 2            # Batch size per GPU
GRADIENT_ACCUMULATION_STEPS = 8    # Effective batch = 2×8=16
NUM_EPOCHS = 10                    # Usually enough for test
WARMUP_STEPS = 5                   # Few steps
EVAL_STEPS = 3                     # Eval frequently for quick feedback
SAVE_STEPS = 3                     # Save frequently
LOGGING_STEPS = 1                  # Verbose logging
EARLY_STOP_PATIENCE = 2            # Stop early if no improvement

# Data Loading
DATALOADER_WORKERS = 0             # MUST BE 0 on macOS (spawn issues)
DATALOADER_PIN = False             # Pinned memory not available on MPS

# Learning
LEARNING_RATE = 1e-4               # Standard LoRA LR
MAX_GRAD_NORM = 1.0                # Gradient clipping
```

**Expected time:** 6-8 hours for 10 epochs on 100 segments

### Profile: `cluster` (A100)

```python
PROFILE = "cluster"

# Memory & Computation
TORCH_DTYPE = torch.bfloat16       # BF16 better on A100
PER_DEVICE_TRAIN_BS = 4            # Can use larger batch
GRADIENT_ACCUMULATION_STEPS = 4    # Effective batch = 4×4=16
NUM_EPOCHS = 20                    # Full training
WARMUP_STEPS = 50                  # More gradual warmup
EVAL_STEPS = 36                    # Eval less frequently to save time
SAVE_STEPS = 36                    # Save less frequently
LOGGING_STEPS = 10                 # Less verbose to reduce overhead
EARLY_STOP_PATIENCE = 3            # Higher patience

# Data Loading
DATALOADER_WORKERS = 2             # Parallel data loading
DATALOADER_PIN = True              # Pin memory for speed

# Learning
LEARNING_RATE = 1e-4               # Same as local
MAX_GRAD_NORM = 1.0                # Same as local
```

**Expected time:** 2-3 hours for 20 epochs on 570 segments

### LoRA Configuration (Line 104-107)

```python
LORA_R = 16                           # Rank (16-32 typical)
LORA_ALPHA = 32                       # Alpha (2x RANK is good)
LORA_DROPOUT = 0.05                   # Dropout in LoRA layers
LORA_TARGET_MODULES = ["q_proj", "v_proj"]  # Which layers to adapt
```

**Tuning options:**
```python
# For smaller memory footprint:
LORA_R = 8
LORA_ALPHA = 16

# For more expressiveness (slower):
LORA_R = 32
LORA_ALPHA = 64

# For different architectures:
LORA_TARGET_MODULES = ["q_proj", "v_proj", "fc1", "fc2"]  # Adapt more layers
```

### Dataset Configuration (Line 87-101)

```python
JSONL_PATH = "data/dataset_final.jsonl"  # Training data (JSONL format)
AUDIO_ROOT = "data"                      # Where audio files live
OUTPUT_DIR = "./whisper-lora-ar-ps"      # Where to save adapters
MERGED_DIR = "./whisper-merged-ar-ps"    # Where to save merged model

MODEL_ID = "openai/whisper-large-v3"     # Base model
LANGUAGE = "arabic"                      # Language for generation
TASK = "transcribe"                      # Task (transcribe/translate)
SAMPLING_RATE = 16_000                   # Audio sample rate

TRAIN_FRAC = 0.80                        # 80% for training
DEV_FRAC = 0.10                          # 10% for validation
TEST_FRAC = 0.10                         # 10% for testing (remaining)
SEED = 42                                # Random seed
```

---

## Model Selection

### Change Base Whisper Model

In `finetune_whisper_ar_ps.py` (line 93):

```python
# Use smaller model (faster, less accurate)
MODEL_ID = "openai/whisper-base"      # 74M parameters
MODEL_ID = "openai/whisper-small"     # 244M parameters
MODEL_ID = "openai/whisper-medium"    # 769M parameters

# Use larger model (slower, more accurate)
MODEL_ID = "openai/whisper-large"     # 1.5B parameters
MODEL_ID = "openai/whisper-large-v3"  # Latest large (1.5B, RECOMMENDED)
```

**Model comparison:**

| Model | Size | VRAM | Speed | Accuracy |
|-------|------|------|-------|----------|
| base | 74M | 1 GB | ⚡⚡⚡ | ⭐⭐ |
| small | 244M | 2 GB | ⚡⚡ | ⭐⭐⭐ |
| medium | 769M | 4 GB | ⚡ | ⭐⭐⭐⭐ |
| large | 1.5B | 8 GB | 🐢 | ⭐⭐⭐⭐⭐ |
| large-v3 | 1.5B | 8 GB | 🐢 | ⭐⭐⭐⭐⭐ |

---

## Hardware Optimization

### For Low-Memory Devices (< 16 GB)

```python
# finetune_whisper_ar_ps.py
MODEL_ID = "openai/whisper-small"  # Not large
LORA_R = 8
PER_DEVICE_TRAIN_BS = 1
GRADIENT_ACCUMULATION_STEPS = 16

# Enable gradient checkpointing
model.config.use_cache = False
model.gradient_checkpointing_enable()
```

### For High-Memory Devices (> 48 GB)

```python
# finetune_whisper_ar_ps.py
MODEL_ID = "openai/whisper-large-v3"  # Large model
LORA_R = 32
PER_DEVICE_TRAIN_BS = 4
GRADIENT_ACCUMULATION_STEPS = 2

# Use full precision
TORCH_DTYPE = torch.float32  # macOS
# OR
TORCH_DTYPE = torch.bfloat16  # CUDA
```

### For Multi-GPU Setup

In `finetune_whisper_ar_ps.py`:

```python
# Enable distributed training
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    # ... existing config ...
    ddp_find_unused_parameters=False,
    ddp_backend="nccl",  # NCCL for CUDA, Gloo for CPU
    local_rank=-1,  # Will be set by distributed launcher
)
```

Then run with:
```bash
python3 -m torch.distributed.launch --nproc_per_node=4 finetune_whisper_ar_ps.py
```

---

## Data Processing Options

### Clean Data Pipeline

In **`process_and_clean.py`** (no configuration needed, but understand the steps):

```bash
python3 process_and_clean.py -i INPUT.jsonl -o OUTPUT.jsonl
```

**Input format:** JSONL with fields:
```json
{
  "sentence": "النص مع [speaker] وتاغات",
  "source_file": "lecture_001_seg001.wav",
  "segment_index": 0,
  "audio": {"path": "data/audio/segment_001.wav"}
}
```

**Output format:** Same fields, but cleaned:
```json
{
  "sentence": "النص مع تاغات",
  "source_file": "lecture_001",
  "segment_index": 0,
  "audio": {"path": "data/audio/segment_001.wav"}
}
```

### Validation Options

In **`validate_jsonl.py`** (no configuration, just review):

```bash
python3 validate_jsonl.py -v data/dataset_final.jsonl
```

Checks for:
- Unremoved speaker tags
- Raw newline characters
- Improper 'اللي' spelling
- Unnecessary Shadda
- Unremoved segment suffixes

---

## Evaluation Settings

### Transcription Configuration

In **`transcribe.py`** (lines 70-79):

```python
DEFAULT_MODEL_DIR = "./whisper-merged-ar-ps"  # Your fine-tuned model
BASELINE_MODEL_ID = "openai/whisper-large-v3"  # Baseline to compare
DEFAULT_AUDIO_DIR = "data/audio"              # Audio files
DEFAULT_OUTPUT = "transcripts.json"           # Output file
DEFAULT_JSONL = "data/dataset_final.jsonl"    # Reference transcripts
LANGUAGE = "arabic"                           # Language
TASK = "transcribe"                           # Task
SAMPLING_RATE = 16_000                        # Audio sample rate
MAX_NEW_TOKENS = 225                          # Max output length
```

**Command options:**

```bash
# Skip baseline (faster)
python3 transcribe.py --no-baseline

# Custom model path
python3 transcribe.py --model ./my-custom-model

# Custom output
python3 transcribe.py --out my-results.json

# Specific lecture
python3 transcribe.py --lecture lecture_005
```

### Summarization Configuration

In **`run_summarization_upgraded.py`** (lines 80-95):

```python
MBART_MODEL_ID = "facebook/mbart-large-cc25"    # Multilingual seq2seq
ARABART_MODEL_ID = "moussaKam/AraBART"          # Arabic-only seq2seq
QWEN_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"      # Instruction-tuned LLM
```

**Command options:**

```bash
# Run all three models
python3 run_summarization_upgraded.py

# Qwen only (fastest)
python3 run_summarization_upgraded.py --qwen-only

# mBART only
python3 run_summarization_upgraded.py --mbart-only

# Select which transcripts to summarize
python3 run_summarization_upgraded.py --runs base ft ref

# Change Qwen summary style
python3 run_summarization_upgraded.py --qwen-style code-switch
# Options: "notes" (default), "code-switch", "msa"

# Set reference style
python3 run_summarization_upgraded.py --ref-style msa
```

### Evaluation Configuration

In **`run_evaluate_upgraded.py`** (lines 75-80):

```python
DEFAULT_BASELINE = "summaries_baseline.json"
DEFAULT_FINETUNED = "summaries_finetuned.json"
DEFAULT_REFERENCE = "summaries_reference.json"
DEFAULT_OUT_TXT = "evaluation_results.txt"
DEFAULT_OUT_JSON = "evaluation_results.json"
```

**Command options:**

```bash
# Use normalized summaries
python3 run_evaluate_upgraded.py --use-normalized

# Self-reference mode (compare each model to its own reference)
python3 run_evaluate_upgraded.py --reference-mode self

# Gold standard mode (compare all to Qwen's reference)
python3 run_evaluate_upgraded.py --reference-mode gold

# Custom paths
python3 run_evaluate_upgraded.py \
  --baseline my_baseline.json \
  --finetuned my_finetuned.json \
  --reference my_reference.json \
  --output-json results.json
```

---

## Environment Variables

### Set Before Running Scripts

```bash
# Hugging Face Hub
export HF_HOME="/path/to/models"           # Where to cache models
export HUGGINGFACE_HUB_CACHE="/path/models"

# Google Generative AI (for text normalization)
export GEMINI_API_KEY="your-api-key"

# CUDA Settings
export CUDA_VISIBLE_DEVICES=0,1,2,3        # Which GPUs to use
export CUDA_LAUNCH_BLOCKING=1              # For debugging

# Torch Settings
export OMP_NUM_THREADS=16                  # CPU parallelization
export TOKENIZERS_PARALLELISM=false        # Tokenizer parallelization

# Weights & Biases (optional logging)
export WANDB_DISABLED=true                 # Disable if not using
```

### Load in Script

```bash
# Create .env file
cat > .env << EOF
HF_HOME=/path/to/models
GEMINI_API_KEY=your-api-key
CUDA_VISIBLE_DEVICES=0,1
EOF

# Load before running
source .env
python3 finetune_whisper_ar_ps.py
```

---

## Profile Comparison

Quick reference for choosing between `local_test` and `cluster`:

| Aspect | local_test | cluster |
|--------|-----------|---------|
| **Hardware** | macOS M1/M2/M3 | A100-SXM4 |
| **Batch size** | 2 | 4 |
| **Dtype** | float32 | bfloat16 |
| **Accumulation** | 8 | 4 |
| **Epochs** | 10 | 20 |
| **Time (100 segs)** | 6-8 hrs | 30 min |
| **Time (570 segs)** | 30-40 hrs | 2-3 hrs |
| **Cost** | Free | $$$ |
| **Best for** | Development | Production |

---

## Troubleshooting Configuration Issues

### Model Downloads Slow

```bash
# Use different mirror
export HF_ENDPOINT=https://huggingface.co
# OR use local cache
export HF_HOME=/local/fast/disk
```

### CUDA Memory Issues

```python
# In finetune_whisper_ar_ps.py
PER_DEVICE_TRAIN_BS = 1  # Reduce batch
GRADIENT_ACCUMULATION_STEPS = 32  # Increase accumulation
model.gradient_checkpointing_enable()  # Enable checkpointing
```

### Slow Training

```python
# Check configuration matches your hardware
PROFILE = "cluster"  # Use if you have A100
DATALOADER_WORKERS = 4  # Increase data loading
# Verify dtype matches hardware (float32 for MPS, bfloat16 for CUDA)
```

---

## Quick Config Presets

### Preset 1: Fast Development (macOS)
```python
PROFILE = "local_test"
PER_DEVICE_TRAIN_BS = 1
NUM_EPOCHS = 3
LORA_R = 8
MODEL_ID = "openai/whisper-base"
```

### Preset 2: Balanced (macOS)
```python
PROFILE = "local_test"
PER_DEVICE_TRAIN_BS = 2
NUM_EPOCHS = 10
LORA_R = 16
MODEL_ID = "openai/whisper-large-v3"
```

### Preset 3: Production (A100)
```python
PROFILE = "cluster"
PER_DEVICE_TRAIN_BS = 4
NUM_EPOCHS = 20
LORA_R = 32
MODEL_ID = "openai/whisper-large-v3"
```

---

**Last Updated:** 2026  
**Maintained By:** Aseel Omar
