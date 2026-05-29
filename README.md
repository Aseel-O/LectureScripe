# LectureScribe: Palestinian Arabic Speech-to-Text & Summarization Pipeline

A comprehensive end-to-end NLP system for transcribing, segmenting, and summarizing Palestinian Arabic academic lectures with automatic code-switching detection and cascade quality evaluation.

## 📋 Project Overview

LectureScribe is a complete pipeline that:

1. **Segments** long-form lecture recordings into natural speech-bounded chunks
2. **Transcribes** Arabic audio using fine-tuned Whisper models
3. **Evaluates** transcription quality with multiple baseline comparisons
4. **Summarizes** transcripts using multilingual and Arabic-specific models
5. **Measures cascading effects** of ASR quality on downstream summarization
6. **Annotates** and validates data for human review

This project supports both **local development (macOS)** and **production deployment (CUDA clusters)**.

## 🗂️ Project Structure

```
LectureScribe/
├── Segmentation/                          # Stage 0: Audio segmentation
│   ├── segment.py                         # Main segmentation script
│   ├── verify_segments.py                 # Quality validation
│   ├── requirements.txt
│   ├── raw_audio/                         # Input audio files
│   ├── raw_video/                         # Input video files
│   └── sliced_segments/                   # Output segments
│
├── ar-ps-whisper-100-nlp-only/            # Stage 1-5: Core pipeline (NLP focus)
│   ├── finetune_whisper_ar_ps.py          # Fine-tune Whisper
│   ├── process_and_clean.py               # Data preparation
│   ├── validate_jsonl.py                  # Data validation
│   ├── transcribe.py                      # ASR transcription
│   ├── run_summarization_upgraded.py      # Summary generation
│   ├── run_evaluate_upgraded.py           # Cascade evaluation
│   ├── plot_results.py                    # Visualization
│   ├── local_transcriber_app.py           # Web UI (Gradio)
│   ├── production_inference.py            # CLI inference
│   ├── requirements-mac.txt               # macOS dependencies
│   ├── requirements-cluster.txt           # CUDA dependencies
│   ├── data/                              # Dataset directory
│   ├── whisper-merged-ar-ps/              # Fine-tuned model
│   └── whisper-lora-ar-ps/                # LoRA adapter weights
│
├── ar-ps-whisper-100-nlp-cv/              # Stage 1-5: Full pipeline (with CV)
│   ├── [Same structure as NLP-only]
│   └── [Includes computer vision components]
│
├── ar-ps-whisper-200/ & ar-ps-whisper-300/  # Variant configurations
│   └── [Alternative model sizes and datasets]
│
├── lecture-annotation-tool/               # Stage 6: Annotation & Review
│   ├── server.ts                          # Node.js backend
│   ├── src/                               # Frontend components
│   ├── package.json
│   ├── .env.local                         # API configuration
│   ├── data/                              # Annotation results
│   └── README.md                          # Tool-specific docs
│
├── annotation_guidelines_v1_7.docx        # Human annotation guidelines
└── README.md                              # This file
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (for Whisper and NLP models)
- **Node.js 14+** (for annotation tool only)
- **ffmpeg** (for audio processing)

For macOS:
```bash
brew install ffmpeg
```

For Ubuntu/Debian:
```bash
apt-get install ffmpeg
```

---

## 📖 Complete Pipeline Guide

### **Stage 0: Audio Segmentation** (Optional)

If you have long-form audio files (lectures, podcasts), segment them into ~60-second chunks:

```bash
cd Segmentation
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Edit segment.py with your input file path, then:
python segment.py

# Verify the output:
python verify_segments.py --segments_dir sliced_segments
```

**Output:** `Segmentation/sliced_segments/seg_001.mp3`, `seg_002.mp3`, etc.

**When to use:** Start here if you have raw lecture recordings. Skip if you already have pre-segmented audio.

---

### **Stage 1-5: Core ASR & Summarization Pipeline**

Choose your variant based on your needs:

- **`ar-ps-whisper-100-nlp-only/`** ← **Recommended for most users**
  - Fast, lightweight pipeline
  - Focuses on transcription and summarization
  - ~4-8 hours on macOS, ~2-3 hours on A100 GPU

- **`ar-ps-whisper-100-nlp-cv/`** (Advanced)
  - Includes speaker identification and visual processing
  - Longer runtime, more hardware required

- **`ar-ps-whisper-200/` & `ar-ps-whisper-300/`** (Experimental)
  - Larger datasets, different configurations
  - Contact maintainers for guidance

#### **Setup**

```bash
cd ar-ps-whisper-100-nlp-only
python3 -m venv venv
source venv/bin/activate

# For macOS:
pip install -r requirements-mac.txt

# For CUDA (A100, V100, etc.):
pip install torch==2.2.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-cluster.txt
```

#### **Run the Complete Pipeline**

**Option A: Full Pipeline (Recommended)**

```bash
# Stage 1: Clean & validate data
python3 process_and_clean.py -i data/metadata.jsonl -o data/dataset_final.jsonl
python3 validate_jsonl.py -v data/dataset_final.jsonl

# Stage 2: Fine-tune Whisper model
python3 finetune_whisper_ar_ps.py

# Stage 3: Transcribe audio
python3 transcribe.py

# Stage 4: Generate summaries
python3 run_summarization_upgraded.py

# Stage 5: Evaluate & compare
python3 run_evaluate_upgraded.py

# Stage 6: Visualize results
python3 plot_results.py
python3 visualize.py
```

**Estimated time:**
- macOS: 6-8 hours
- CUDA (A100): 2-3 hours

**Option B: Quick Test (Skip Fine-tuning)**

```bash
# Use baseline Whisper only (faster for testing):
python3 transcribe.py --no-baseline
python3 run_summarization_upgraded.py --qwen-only
python3 run_evaluate_upgraded.py
```

**Estimated time:** 30 minutes - 1 hour

**Option C: Just Transcribe**

```bash
# Single file transcription:
python3 production_inference.py path/to/audio.wav

# Or launch web UI:
python3 local_transcriber_app.py
# Open http://localhost:7860
```

---

#### **Hardware-Specific Configuration**

The pipeline auto-detects your hardware, but you can force a specific profile:

**For macOS:**
1. Edit `finetune_whisper_ar_ps.py` line 84:
   ```python
   PROFILE = "local_test"  # or "local_full"
   ```
2. Run: `python3 finetune_whisper_ar_ps.py`

**For CUDA:**
1. Edit `finetune_whisper_ar_ps.py` line 84:
   ```python
   PROFILE = "cluster"
   ```
2. Run: `python3 finetune_whisper_ar_ps.py`

**For Mac with MPS (Apple Silicon M1/M2/M3):**
- The pipeline auto-detects MPS support
- If you get "MPS not available", upgrade torch:
  ```bash
  pip install --upgrade torch
  ```

---

### **Stage 6: Annotation & Review Tool** (Optional)

After transcription and evaluation, use the web-based annotation tool to review and correct results:

```bash
cd lecture-annotation-tool
npm install
# Set your Gemini API key in .env.local
npm run dev
# Open http://localhost:5173
```

**For first-time setup:**
1. Copy `.env.example` to `.env.local`
2. Add your Gemini API key
3. Run `npm run dev`

---

## 📊 Input/Output Files

| Stage | Input | Output | Duration |
|-------|-------|--------|----------|
| **0. Segment** | `raw_audio/*.mp4` or `*.mp3` | `sliced_segments/*.mp3` | 5-15 min |
| **1. Clean** | `data/metadata.jsonl` | `data/dataset_final.jsonl` | 5 min |
| **2. Fine-tune** | `data/dataset_final.jsonl` | `whisper-merged-ar-ps/` | 2-8 hrs |
| **3. Transcribe** | `data/audio/*.wav` | `transcripts.json` | 30 min - 2 hrs |
| **4. Summarize** | `transcripts.json` | `summaries_baseline.json`, `summaries_finetuned.json` | 5-30 min |
| **5. Evaluate** | Summary JSON files | `evaluation_results.json` | 5 min |
| **6. Visualize** | `evaluation_results.json` | `*.png` charts | 2 min |

---

## 🖥️ macOS vs CUDA: Key Differences

### **macOS (Local Development)**

✅ **Pros:**
- Easy local development and testing
- Fast iteration
- No GPU required (uses CPU or Metal Performance Shaders)

⚠️ **Cons:**
- Slow fine-tuning (6-8 hours)
- Limited parallelization
- Batch sizes must be small (1-2)

**Use:** Development, testing, small datasets

**Recommended setup:**
```bash
pip install -r requirements-mac.txt
# Use PROFILE = "local_test" for quick experiments
```

### **CUDA (Production)**

✅ **Pros:**
- Fast fine-tuning (2-3 hours on A100)
- Large batch sizes (4-16)
- Parallel processing

⚠️ **Cons:**
- Requires NVIDIA GPU (A100, V100, A10, etc.)
- Higher setup complexity
- GPU memory management needed

**Use:** Production inference, large-scale training

**Recommended setup:**
```bash
pip install torch==2.2.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-cluster.txt
# Use PROFILE = "cluster"
```

**Handling CUDA out of memory:**
```python
# In finetune_whisper_ar_ps.py, adjust:
PER_DEVICE_TRAIN_BS = 1  # Reduce batch size
GRADIENT_ACCUMULATION_STEPS = 16  # Increase accumulation
```

---

## 🎯 Common Commands

### Quick Reference

```bash
# Just transcribe a file:
python3 production_inference.py audio.wav

# Launch interactive web app:
python3 local_transcriber_app.py

# Check transcription quality:
python3 -c "
import json
with open('transcripts.json') as f:
    data = json.load(f)
for lecture in data:
    wer = sum(s['wer'] for s in data[lecture]['segments']) / len(data[lecture]['segments'])
    print(f'{lecture}: WER={wer:.3f}')
"

# Check evaluation results:
python3 -c "
import json
with open('evaluation_results.json') as f:
    results = json.load(f)
for r in results['results']:
    print(f\"{r['run']} + {r['model']}: ROUGE-L={r['vs_reference']['rouge_l']:.3f}\")
"
```

---

## 📝 Configuration & Customization

### Data Format

The pipeline expects JSONL files with this structure:

```json
{
  "audio_path": "data/audio/segment_001.wav",
  "transcription": "النص العربي للتسجيل",
  "speaker": "المحاضر",
  "duration": 45.5,
  "language": "ar-PS"
}
```

### Fine-tuning Parameters

Edit `finetune_whisper_ar_ps.py` to customize:

```python
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
PER_DEVICE_TRAIN_BS = 4  # Batch size
WARMUP_STEPS = 500
```

### Model Selection

For summarization, choose models in `run_summarization_upgraded.py`:
- `mBART` (multilingual, balanced)
- `AraBART` (Arabic-specific, best for Arabic)
- `Qwen` (instruction-tuned LLM, flexible)

---

## 🐛 Troubleshooting

### "CUDA out of memory"
```python
# Reduce batch size in finetune_whisper_ar_ps.py
PER_DEVICE_TRAIN_BS = 1
GRADIENT_ACCUMULATION_STEPS = 16
```

### "Module not found" errors
```bash
# Reinstall dependencies:
pip install --upgrade -r requirements-mac.txt
# Or for CUDA:
pip install torch==2.2.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade -r requirements-cluster.txt
```

### "Model not found" from HuggingFace
```bash
# Download models manually:
huggingface-cli download openai/whisper-large-v3
huggingface-cli download facebook/mbart-large-cc25
```

### macOS: "MPS not available"
```bash
python3 -c "import torch; print(torch.backends.mps.is_available())"
pip install --upgrade torch
```

### Audio not being segmented properly
Edit `Segmentation/segment.py`:
```python
silence_thresh = -50  # Lower (more negative) to detect quieter speech
min_silence_len = 700  # Increase to avoid cutting mid-word
target_length_ms = 60000  # Adjust for your audio
```

---

## 📚 Documentation

For detailed information, see:

- **Segmentation:** `Segmentation/README.md`
- **ASR & Summarization:** `ar-ps-whisper-100-nlp-only/README.md`
- **Quick Start:** `ar-ps-whisper-100-nlp-only/QUICKSTART.md`
- **Annotation Tool:** `lecture-annotation-tool/README.md`
- **Guidelines:** `annotation_guidelines_v1_7.docx`

---

## 🔄 Workflow Summary

```
Raw Lectures (MP4/MP3)
    ↓
[Segmentation] → sliced_segments (30-60s chunks)
    ↓
[Data Prep] → dataset_final.jsonl (cleaned metadata)
    ↓
[Fine-tuning] → whisper-merged-ar-ps (custom model)
    ↓
[Transcription] → transcripts.json (baseline + finetuned)
    ↓
[Summarization] → summaries_*.json (mBART, AraBART, Qwen)
    ↓
[Evaluation] → evaluation_results.json (ROUGE, WER, CER)
    ↓
[Visualization] → charts.png (quality metrics)
    ↓
[Annotation] → annotations.json (human review + corrections)
```

---

## 💾 Resource Requirements

| Task | CPU | Memory | GPU | Duration |
|------|-----|--------|-----|----------|
| Segmentation | 1-2 cores | 2GB | None | 5-15 min |
| Data prep | 1-2 cores | 4GB | None | 5 min |
| Fine-tuning (macOS) | All cores | 16GB | CPU/MPS | 6-8 hrs |
| Fine-tuning (CUDA) | 4-8 cores | 40GB+ | 1x A100 | 2-3 hrs |
| Transcription | 4 cores | 8GB | Optional | 30 min - 2 hrs |
| Summarization | 4 cores | 12GB | Optional | 5-30 min |

---

## 🤝 Contributing

1. **Report issues** using GitHub Issues
2. **Test locally** before submitting PRs
3. **Document changes** in your commit messages
4. **Follow the existing code style** (Python Black formatting)

---

## 📄 License

[Add your license here]

---

## 👥 Contact & Support

For questions or support:
- **Issues:** GitHub Issues
- **Email:** [Add contact email]
- **Documentation:** See the detailed README files in each subdirectory

---

## 🎓 Citation

If you use LectureScribe in your research, please cite:

```bibtex
@software{lecturescript2024,
  title={LectureScribe: Palestinian Arabic Speech-to-Text & Summarization Pipeline},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-username]/lecturescript}
}
```

---

**Last Updated:** May 2026  
**Status:** Production Ready ✅
