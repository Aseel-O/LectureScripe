# Whisper Fine-Tuning Dataset

Exported by Lecture Annotation Tool — NLP Dialect Lab.

## Contents
- `metadata.jsonl` — one record per segment (HuggingFace AudioFolder format)
- `audio/` — WAV audio files (16-bit PCM)

## Load with HuggingFace datasets
```python
from datasets import load_dataset

ds = load_dataset("audiofolder", data_dir="./whisper_dataset")
```

## Segments: 300
## Language: ar-PS
