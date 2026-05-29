# Quick Start Guide

Fast reference for running the complete pipeline.

## ⚡ 5-Minute Setup

```bash
# 1. Clone & navigate
git clone <repo-url>
cd ar-ps-whisper

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
# For macOS:
pip install -r requirements-mac.txt

# For CUDA:
pip install torch==2.2.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-cluster.txt
```

## 🚀 Running the Complete Pipeline

### Full Pipeline (Data → Fine-tune → Evaluate)

```bash
# Stage 1: Prepare data (if starting from raw data)
python3 process_and_clean.py -i data/metadata.jsonl -o data/dataset_final.jsonl
python3 validate_jsonl.py -v data/dataset_final.jsonl

# Stage 2: Fine-tune model
python3 finetune_whisper_ar_ps.py

# Stage 3: Evaluate transcriptions
python3 transcribe.py --no-baseline  # Skip baseline for speed

# Stage 4: Generate summaries
python3 run_summarization_upgraded.py --qwen-only  # Qwen only (faster)

# Stage 5: Evaluate cascade
python3 run_evaluate_upgraded.py

# Stage 6: Visualize
python3 plot_results.py
python3 visualize.py
```

**Total time:** ~4-8 hours (depending on data size and hardware)

---

## 🎯 Common Quick Commands

### Just Transcribe Audio
```bash
python3 production_inference.py audio.wav
```

### Launch Local Web App
```bash
python3 local_transcriber_app.py
# Opens http://localhost:7860
```

### Skip Fine-tuning, Use Baseline Only
```bash
python3 transcribe.py --no-baseline
```

### Regenerate Only Summaries
```bash
python3 run_summarization_upgraded.py --qwen-only
```

### Re-evaluate Without Re-running Everything
```bash
python3 run_evaluate_upgraded.py --baseline summaries_baseline.json --finetuned summaries_finetuned.json
```

---

## 🖥️ Hardware-Specific Commands

### macOS
```bash
# Edit finetune_whisper_ar_ps.py line 84:
PROFILE = "local_test"

# Then run:
python3 finetune_whisper_ar_ps.py
```

### A100 CUDA
```bash
# Edit finetune_whisper_ar_ps.py line 84:
PROFILE = "cluster"

# Then run:
python3 finetune_whisper_ar_ps.py
```

---

## 📊 Check Results

### View Transcription Quality
```bash
python3 -c "
import json
with open('transcripts.json') as f:
    data = json.load(f)
for lecture, content in data.items():
    avg_wer = sum(s['wer_finetuned'] for s in content['segments']) / len(content['segments'])
    print(f'{lecture}: WER = {avg_wer:.3f}')
"
```

### View Evaluation Results
```bash
python3 -c "
import json
with open('evaluation_results.json') as f:
    data = json.load(f)
for result in data['results']:
    print(f\"{result['run']} + {result['model']}: ROUGE-L = {result['vs_reference']['rouge_l']:.3f}\")
"
```

### View Training Logs
```bash
python3 -c "
import json
with open('whisper-lora-ar-ps/trainer_state.json') as f:
    logs = json.load(f)
for entry in logs['log_history'][-5:]:
    if 'eval_loss' in entry:
        print(f\"Epoch {entry['epoch']}: eval_loss={entry['eval_loss']:.4f}\")
"
```

---

## 🐛 Quick Fixes

### "CUDA out of memory"
```python
# In finetune_whisper_ar_ps.py, change:
PER_DEVICE_TRAIN_BS = 1  # Down from 2/4
GRADIENT_ACCUMULATION_STEPS = 16  # Up
```

### "Model not found"
```bash
# Download manually
huggingface-cli download openai/whisper-large-v3
```

### "MPS not available"
```bash
python3 -c "import torch; print(torch.backends.mps.is_available())"
pip install --upgrade torch
```

---

## 📝 Input/Output Files

| Stage | Input | Output |
|-------|-------|--------|
| **1. Clean** | `data/metadata.jsonl` | `data/dataset_final.jsonl` |
| **2. Fine-tune** | `data/dataset_final.jsonl` | `whisper-merged-ar-ps/` |
| **3. Transcribe** | `data/audio/*.wav` | `transcripts.json` |
| **4. Summarize** | `transcripts.json` | `summaries_*.json` |
| **5. Evaluate** | `summaries_*.json` | `evaluation_results.json` |
| **6. Visualize** | Results JSON | `*.png` charts |

---

## 🔗 Next Steps

- **Full details:** See [README.md](README.md)
- **Troubleshooting:** See [README.md#🐛-troubleshooting](README.md#-troubleshooting)
- **Configuration:** See [README.md#📝-configuration-reference](README.md#-configuration-reference)

---

**Estimated Times:**
- Data cleaning: 5 min
- Fine-tuning (macOS): 6-8 hrs
- Fine-tuning (A100): 2-3 hrs
- Transcription: 30 min - 2 hrs
- Summarization: 5-30 min (depending on models)
- Evaluation: 5 min
- Visualization: 2 min
