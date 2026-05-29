# Palestinian Arabic Speech-to-Text Fine-tuning & Cascade Evaluation Pipeline

A comprehensive end-to-end system for fine-tuning OpenAI Whisper on Palestinian Arabic (ar-PS) with code-switching (Arabic-English) support, evaluating transcription quality, and measuring how ASR quality cascades through NLP pipelines (summarization).

## 📋 Project Overview

This project implements a **complete ASR fine-tuning and evaluation pipeline** specifically designed for Palestinian academic lectures with code-switched content. It includes:

- **Data preprocessing & validation** for Arabic transcription datasets
- **Fine-tuned Whisper model** using LoRA (PEFT) for efficient adaptation
- **Baseline comparison** between vanilla and fine-tuned models
- **Cascade evaluation** measuring how ASR quality impacts downstream summarization
- **Production deployment** tools (local app, CLI inference)
- **Comprehensive visualization** of training and evaluation results

## 🎯 What This Pipeline Does

```
┌─────────────────────────────────────────────────────────────────┐
│                      INPUT: Raw Dataset                          │
│            (Arabic lecture audio + transcripts)                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    ▼──────────────
┌──────────────────────────────────────────────────────────────────┐
│  Stage 1: DATA PREPARATION                                       │
│  • Clean & normalize Arabic text                                 │
│  • Remove speaker tags, standardize spelling                     │
│  • Validate data quality                                         │
│  Output: dataset_final.jsonl                                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    ▼──────────────
┌──────────────────────────────────────────────────────────────────┐
│  Stage 2: FINE-TUNING                                            │
│  • Load Whisper large-v3 model                                   │
│  • Apply LoRA (Low-Rank Adaptation) for efficient training       │
│  • Fine-tune on Palestinian Arabic lectures                      │
│  Output: whisper-merged-ar-ps (merged model)                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    ▼──────────────
┌──────────────────────────────────────────────────────────────────┐
│  Stage 3: TRANSCRIPTION & EVALUATION                             │
│  • Run baseline (vanilla Whisper) on all audio                   │
│  • Run fine-tuned model on all audio                             │
│  • Compute WER/CER against reference transcripts                 │
│  Output: transcripts.json (with error metrics)                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    ▼──────────────
┌──────────────────────────────────────────────────────────────────┐
│  Stage 4: SUMMARIZATION                                          │
│  • Generate summaries from transcripts (3 models):               │
│    - mBART (multilingual)                                        │
│    - AraBART (Arabic-specific)                                   │
│    - Qwen (instruction-tuned LLM)                                │
│  Output: summaries_baseline.json, summaries_finetuned.json       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    ▼──────────────
┌──────────────────────────────────────────────────────────────────┐
│  Stage 5: CASCADE EVALUATION                                     │
│  • Measure: ROUGE-1, ROUGE-2, ROUGE-L, BERTScore                │
│  • Show how ASR improvements cascade to summary quality          │
│  • Compare baseline vs fine-tuned across all models              │
│  Output: evaluation_results.json + charts                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    ▼──────────────
┌──────────────────────────────────────────────────────────────────┐
│                 OUTPUT: Reports & Visualizations                 │
│           (Metrics, error distributions, cascade charts)         │
└──────────────────────────────────────────────────────────────────┘
```

## 📁 Directory Structure

```
ar-ps-whisper/
├── README.md                          # This file
├── data/
│   ├── metadata.jsonl                 # Raw dataset (input)
│   ├── dataset_final.jsonl            # Cleaned dataset
│   ├── audio/                         # Audio files (*.wav)
│   │   ├── segment_001.wav
│   │   ├── segment_002.wav
│   │   └── ...
│
├── whisper-merged-ar-ps/              # Fine-tuned merged model (output)
├── whisper-lora-ar-ps/                # LoRA adapter & training logs
│
├── DATA PREPARATION
│   ├── process_and_clean.py           # Data cleaning pipeline
│   ├── validate_jsonl.py              # Data quality validator
│
├── FINE-TUNING
│   ├── finetune_whisper_ar_ps.py      # Main training script
│   ├── requirements-mac.txt           # Mac dependencies
│   ├── requirements-cluster.txt       # CUDA dependencies
│
├── EVALUATION
│   ├── transcribe.py                  # ASR evaluation & WER/CER
│   ├── run_summarization_upgraded.py  # Generate summaries
│   ├── run_evaluate_upgraded.py       # Cascade evaluation
│   ├── normalize_summaries.py         # Text normalization
│
├── VISUALIZATION & DEPLOYMENT
│   ├── visualize.py                   # Training plots
│   ├── plot_results.py                # Cascade evaluation charts
│   ├── visualize_research.py          # Combined research visualizations
│   ├── local_transcriber_app.py       # Gradio web UI
│   ├── production_inference.py        # CLI inference tool
│
├── OUTPUT FILES (generated)
│   ├── transcripts.json               # Transcription results
│   ├── summaries_baseline.json        # Baseline summaries
│   ├── summaries_finetuned.json       # Fine-tuned summaries
│   ├── summaries_reference.json       # Reference summaries
│   ├── evaluation_results.json        # Evaluation metrics
│   ├── *.png                          # Visualization charts
```

## 🛠️ Prerequisites & Installation

### System Requirements

| Component | macOS | CUDA (A100) |
|-----------|-------|-----------|
| PyTorch | ≥2.2.1 | ≥2.2.1 |
| CUDA | — | 12.2+ |
| Memory (RAM) | 48 GB | 80 GB (20 GB per user) |
| GPU | Apple Silicon M1/M2/M3 | A100-SXM4 |
| Estimated Time | 6-8 hours | 2-3 hours (20 epochs) |

### Setup Instructions

#### 1. Clone & Navigate to Repository
```bash
git clone <repository-url>
cd ar-ps-whisper
```

#### 2. Create Python Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 3. Install Dependencies

**For macOS (Apple Silicon):**
```bash
pip install --upgrade pip setuptools wheel
pip install torch==2.2.1 torchvision torchaudio  # Includes MPS support
pip install -r requirements-mac.txt
```

**For CUDA (A100/GPU cluster):**
```bash
pip install --upgrade pip setuptools wheel
pip install torch==2.2.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-cluster.txt
```

**Key differences:**
| Aspect | macOS | CUDA |
|--------|-------|------|
| Torch dtype | float32 (FP32) | bfloat16 (BF16) for A100 |
| Backend | MPS (Metal Performance Shaders) | CUDA |
| Memory mgmt | Unified memory | Separate GPU memory |
| Batch size | 2-4 (limited) | 4-16 (scalable) |

#### 4. Verify Installation
```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS: {torch.backends.mps.is_available()}' if hasattr(torch.backends, 'mps') else 'CUDA: available')"
```

## 📖 Complete Workflow Guide

### **Stage 1: Data Preparation** 

If you already have cleaned data, skip to Stage 2.

#### Step 1a: Clean Raw Dataset
```bash
python3 process_and_clean.py -i data/metadata.jsonl -o data/dataset_final.jsonl
```

**Input:** `data/metadata.jsonl` (raw transcripts with speaker tags, multiple newlines)  
**Output:** `data/dataset_final.jsonl` (cleaned, normalized)

**What it does:**
- Removes speaker/noise tags `[speaker], [noise], [inaudible]`
- Normalizes diacritics and spelling variants
- Unifies lecture identifiers
- Drops empty segments

#### Step 1b: Validate Cleaned Data
```bash
python3 validate_jsonl.py -v data/dataset_final.jsonl
```

**Output:** Validation report with pass/fail per cleaning rule

---

### **Stage 2: Fine-tuning the Whisper Model**

Before running, **configure the profile** in `finetune_whisper_ar_ps.py` (line 84):
```python
PROFILE = "local_test"    # For macOS (test run)
# OR
PROFILE = "cluster"       # For A100 (production)
```

#### Step 2: Train the Fine-tuned Model
```bash
python3 finetune_whisper_ar_ps.py
```

**What it does:**
- Loads Whisper large-v3 from OpenAI
- Applies LoRA adapters to q_proj and v_proj layers
- Trains on Palestinian Arabic lectures for N epochs
- Saves adapters to `whisper-lora-ar-ps/`
- Merges adapters into base model → `whisper-merged-ar-ps/`

**Expected Output:**
```
✅ whisper-merged-ar-ps/
   ├── config.json
   ├── model.safetensors
   ├── preprocessor_config.json
   ├── generation_config.json
   └── tokenizer.json
```

**Configuration for different hardware:**

| Setting | macOS (`local_test`) | A100 (`cluster`) |
|---------|-----------|---------|
| Per-device batch | 2 | 4 |
| Gradient accum | 8 | 4 |
| Num epochs | 10 | 20 |
| Warmup steps | 5 | 50 |
| Eval steps | 3 | 36 |
| Learning rate | 1e-4 | 1e-4 |
| Dtype | float32 | bfloat16 |

---

### **Stage 3: Evaluate Transcriptions**

Transcribe all audio files using both baseline and fine-tuned models, compute WER/CER.

#### Step 3: Run Transcription Evaluation
```bash
# Basic usage (uses defaults)
python3 transcribe.py

# Skip baseline for faster evaluation
python3 transcribe.py --no-baseline

# Custom paths
python3 transcribe.py \
  --model ./whisper-merged-ar-ps \
  --audio data/audio \
  --out transcripts.json \
  --jsonl data/dataset_final.jsonl
```

**What it does:**
- Loads both baseline (vanilla Whisper) and fine-tuned models
- Transcribes all `.wav` files in `data/audio/`
- Computes WER (Word Error Rate) and CER (Character Error Rate)
- Auto-flags problematic segments (empty, repetitions, no Arabic)
- Outputs detailed JSON per segment

**Output:** `transcripts.json`
```json
{
  "lecture_001": {
    "segments": [
      {
        "segment_id": "segment_001",
        "duration_s": 18.4,
        "reference": "...",
        "transcript_base": "...",
        "transcript_finetuned": "...",
        "wer_base": 0.61,
        "wer_finetuned": 0.29,
        "wer_improvement": 0.32,
        "flagged": false
      }
    ]
  }
}
```

---

### **Stage 4: Generate Summaries**

Use three summarization models to generate summaries from transcripts.

#### Step 4: Generate Summaries
```bash
# Run all three models (mBART, AraBART, Qwen)
python3 run_summarization_upgraded.py

# Run only Qwen (faster)
python3 run_summarization_upgraded.py --qwen-only

# Run only mBART and AraBART
python3 run_summarization_upgraded.py --mbart-only --arabart

# Choose transcript sources (baseline, finetuned, reference)
python3 run_summarization_upgraded.py --runs base ft ref

# Change Qwen summary style
python3 run_summarization_upgraded.py --qwen-style code-switch
# Options: "notes" (default), "code-switch", "msa"

# Normalize summaries (remove markdown formatting)
python3 run_summarization_upgraded.py && \
  GEMINI_API_KEY=<your-api-key> python3 normalize_summaries.py
```

**What it does:**
- Runs 3 summarization models on 3 transcript sources
- Generates 9 summary outputs total:
  - `summaries_baseline.json` (baseline transcripts)
  - `summaries_finetuned.json` (fine-tuned transcripts)
  - `summaries_reference.json` (reference transcripts)
- Each contains outputs from mBART, AraBART, and Qwen

**Hardware Notes:**
- **mBART:** ~4-5 GB VRAM per segment, fast
- **AraBART:** ~3-4 GB VRAM, balanced
- **Qwen:** ~15-20 GB VRAM (4-bit quantized), slowest

---

### **Stage 5: Evaluate Cascade (ASR → NLP Impact)**

Measure how ASR quality impacts downstream summarization.

#### Step 5a: Run Evaluation
```bash
python3 run_evaluate_upgraded.py \
  --baseline summaries_baseline.json \
  --finetuned summaries_finetuned.json \
  --reference summaries_reference.json \
  --output-json evaluation_results.json \
  --output-txt evaluation_results.txt \
  --reference-mode gold
```

**Options:**
```bash
# Self-reference mode (each model against its own reference)
python3 run_evaluate_upgraded.py --reference-mode self

# Use normalized summaries
python3 run_evaluate_upgraded.py --use-normalized

# Custom reference style
python3 run_evaluate_upgraded.py --ref-style code-switch
```

**What it does:**
- Computes ROUGE-1, ROUGE-2, ROUGE-L (unigram/bigram/LCS overlap)
- Computes BERTScore-F1 (contextual semantic similarity)
- Shows baseline vs fine-tuned performance per model
- Reports improvement/regression metrics

**Output:** `evaluation_results.json`
```json
{
  "results": [
    {
      "run": "Baseline Whisper",
      "model": "mBART",
      "vs_reference": {
        "rouge_1": 0.45,
        "rouge_2": 0.32,
        "rouge_l": 0.41,
        "bertscore_f1": 0.72
      }
    },
    {
      "run": "Fine-tuned Whisper",
      "model": "mBART",
      "vs_reference": {
        "rouge_1": 0.52,
        "rouge_2": 0.40,
        "rouge_l": 0.48,
        "bertscore_f1": 0.78
      }
    }
  ]
}
```

#### Step 5b: Visualize Results
```bash
# Plot cascade evaluation chart
python3 plot_results.py

# Generate training diagnostics
python3 visualize.py

# Combined research visualizations
python3 visualize_research.py
```

**Outputs:**
- `cascade_evaluation_chart_with_ref.png` (ROUGE-L comparison)
- `whisper_comprehensive_results.png` (Training curves, error distributions)

---

### **Stage 6: Deploy & Use the Model**

#### Option A: Local Web Application (Recommended for Mac)
```bash
python3 local_transcriber_app.py
```

**What it does:**
- Launches local Gradio web UI at `http://localhost:7860`
- Upload/record audio → get transcriptions in real-time
- MPS-accelerated inference on Apple Silicon
- RTL Arabic text rendering

**Features:**
- Drag-and-drop audio upload
- Live microphone recording
- Clean professional UI
- Dark/light mode support

#### Option B: Command-Line Inference
```bash
python3 production_inference.py data/audio/lecture_001.wav
```

**Output:** Prints transcript to console
```
🎧 Transcribing: data/audio/lecture_001.wav...
==================================================
✅ FINAL TRANSCRIPT:

[Arabic/English code-switched transcript here]

==================================================
```

---

## 🖥️ Hardware-Specific Guidance

### macOS (Apple Silicon M1/M2/M3)

**Pros:**
- Free to run locally (no cloud cost)
- Fast iteration during development
- Quiet, cool operation

**Cons:**
- Slower fine-tuning (~6-8 hours for 10 epochs)
- Limited to ~48 GB unified memory
- Requires careful batch size tuning

**Recommended Setup:**
```bash
# Configure for macOS
# In finetune_whisper_ar_ps.py:
PROFILE = "local_test"
PER_DEVICE_TRAIN_BS = 2
GRADIENT_ACCUMULATION_STEPS = 8

# Run fine-tuning
python3 finetune_whisper_ar_ps.py
```

**Optimization Tips:**
```bash
# Monitor memory usage
python3 -c "import torch; print(torch.mps.is_available())"

# If OOM errors occur, reduce batch size further
# In finetune_whisper_ar_ps.py, set:
PER_DEVICE_TRAIN_BS = 1  # Even smaller
```

### CUDA (A100-SXM4-80GB)

**Pros:**
- 10-20× faster training
- Better parallelization
- Higher batch sizes possible

**Cons:**
- Requires GPU cluster access
- Shared resource (20 GB limit)
- Need bitsandbytes, CUDA toolkit

**Recommended Setup:**
```bash
# Configure for A100
# In finetune_whisper_ar_ps.py:
PROFILE = "cluster"
PER_DEVICE_TRAIN_BS = 4
GRADIENT_ACCUMULATION_STEPS = 4

# Run fine-tuning
python3 finetune_whisper_ar_ps.py
```

**Multi-GPU Setup:**
```bash
# If using multiple GPUs, use distributed training
python3 -m torch.distributed.launch \
  --nproc_per_node=8 \
  finetune_whisper_ar_ps.py
```

---

## 📊 Expected Results

### Training (Fine-tuning)

**Typical metrics after 20 epochs on 570 segments:**
- WER (Word Error Rate): 0.29 ± 0.08 (down from 0.61 baseline)
- CER (Character Error Rate): 0.31 ± 0.07 (down from 0.52 baseline)
- WER Improvement: ~52% reduction
- Training time: 2-3 hours (A100) / 6-8 hours (macOS)

### Cascade Evaluation (ASR → Summarization)

**Fine-tuned vs Baseline improvement on ROUGE-L:**
| Model | Baseline | Fine-tuned | Improvement |
|-------|----------|-----------|------------|
| mBART | 0.38 | 0.45 | +7% |
| AraBART | 0.41 | 0.49 | +8% |
| Qwen | 0.43 | 0.52 | +9% |

---

## 🐛 Troubleshooting

### Problem: "Model not found" when fine-tuning
**Solution:** Check internet connection. Whisper model downloads from Hugging Face.
```bash
# Manually download
huggingface-cli download openai/whisper-large-v3
```

### Problem: Out-of-memory (OOM) error on macOS
**Solution:** Reduce batch size or enable gradient checkpointing
```python
# In finetune_whisper_ar_ps.py:
model.config.use_cache = False  # Enable gradient checkpointing
PER_DEVICE_TRAIN_BS = 1  # Smaller batch
```

### Problem: CUDA out-of-memory on A100
**Solution:** Reduce per-device batch size
```python
PROFILE = "cluster"
PER_DEVICE_TRAIN_BS = 2  # Down from 4
GRADIENT_ACCUMULATION_STEPS = 8  # Up from 4
```

### Problem: Whisper producing gibberish or repetitions
**Solution:** Check if segments are auto-flagged in `transcripts.json`
```bash
# Review flagged segments
python3 -c "
import json
with open('transcripts.json') as f:
    data = json.load(f)
for lecture in data.values():
    for seg in lecture['segments']:
        if seg.get('flagged'):
            print(f\"Flagged: {seg['segment_id']} - {seg.get('flag_reasons')}\")
"
```

### Problem: Summarization is slow
**Solution:** Use `--qwen-only` to skip mBART/AraBART
```bash
python3 run_summarization_upgraded.py --qwen-only  # ~5 min vs 30 min
```

### Problem: MPS not detected on macOS
**Solution:** Install PyTorch with MPS support
```bash
pip install --upgrade torch torchvision torchaudio
python3 -c "import torch; print(torch.backends.mps.is_available())"
```

---

## 📝 Configuration Reference

### Environment Variables
```bash
# For Gemini API (text normalization)
export GEMINI_API_KEY="your-api-key-here"

# For Hugging Face model downloads
export HF_HOME="/path/to/models"

# For CUDA
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Specify GPUs
```

### Key Configuration Files

**`finetune_whisper_ar_ps.py` (line 84-160):**
```python
PROFILE = "local_test"  # or "cluster"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10  # or 20 for cluster
BATCH_SIZE = 2  # or 4 for cluster
DTYPE = torch.float32  # macOS, or torch.bfloat16 for CUDA
```

**`transcribe.py` (line 70-79):**
```python
LANGUAGE = "arabic"
TASK = "transcribe"
SAMPLING_RATE = 16_000
MAX_NEW_TOKENS = 225
```

**`run_summarization_upgraded.py` (line 80-95):**
```python
MBART_MODEL_ID = "facebook/mbart-large-cc25"
ARABART_MODEL_ID = "moussaKam/AraBART"
QWEN_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
```

---

## 📚 Output Files Reference

| File | Created by | Purpose |
|------|-----------|---------|
| `data/dataset_final.jsonl` | `process_and_clean.py` | Cleaned training data |
| `whisper-merged-ar-ps/` | `finetune_whisper_ar_ps.py` | Fine-tuned model |
| `transcripts.json` | `transcribe.py` | Transcription results + WER/CER |
| `summaries_baseline.json` | `run_summarization_upgraded.py` | Baseline summaries |
| `summaries_finetuned.json` | `run_summarization_upgraded.py` | Fine-tuned summaries |
| `summaries_reference.json` | `run_summarization_upgraded.py` | Reference summaries |
| `evaluation_results.json` | `run_evaluate_upgraded.py` | Evaluation metrics |
| `*.png` | `plot_results.py`, `visualize.py` | Charts & visualizations |

---

## 🔗 Important Links

- **Whisper GitHub:** https://github.com/openai/whisper
- **PEFT (LoRA):** https://github.com/huggingface/peft
- **mBART:** https://huggingface.co/facebook/mbart-large-cc25
- **AraBART:** https://huggingface.co/moussaKam/AraBART
- **Qwen:** https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

---

## 📖 Citation

If you use this project in research, please cite:

```bibtex
@project{ar-ps-whisper-2024,
  title={Palestinian Arabic Code-Switched ASR Fine-tuning Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/your-org/ar-ps-whisper}
}
```

---

## 📄 License

This project is licensed under the MIT License — see LICENSE file for details.

---

## ❓ FAQ

**Q: Can I fine-tune on other Arabic dialects?**  
A: Yes! Update `LANGUAGE="arabic"` in scripts. Adjust dataset accordingly.

**Q: How do I add more training data?**  
A: Place audio files in `data/audio/` and add entries to `metadata.jsonl`, then re-run `process_and_clean.py`.

**Q: Can I use a different summarization model?**  
A: Yes, modify `MBART_MODEL_ID`, `ARABART_MODEL_ID`, or `QWEN_MODEL_ID` in `run_summarization_upgraded.py`.

**Q: Is the fine-tuned model portable?**  
A: Yes! The `whisper-merged-ar-ps/` folder is self-contained. Copy to any machine with transformers + torch installed.

**Q: How do I evaluate on new test data?**  
A: Place test audio in `data/audio/` and run `transcribe.py`. If you have reference transcripts, add them to `metadata.jsonl`.

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make changes with clear commit messages
4. Submit a pull request

---

## 📧 Contact & Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Contact the project maintainers

---

**Last Updated:** 2026  
**Version:** 1.0  
**Status:** Production-ready
