# Audio Segmentation Pipeline

Converts raw lecture recordings (MP4/MP3) into natural speech-bounded segments suitable for transcription and annotation. Uses silence detection to preserve natural speech boundaries while maintaining consistent segment lengths (~60 seconds).

## Overview

This pipeline automatically segments long-form audio (lectures, podcasts, webinars) by:
1. Detecting speech intervals using silence thresholds
2. Identifying natural pause boundaries (500ms+ silence)
3. Greedily merging small speech segments until reaching ~60 seconds
4. Exporting clean segments for downstream processing

The output segments are ideally formatted for:
- Speech-to-text transcription (e.g., OpenAI Whisper)
- Manual annotation and labeling
- Machine learning training datasets
- Content analysis and information extraction

## Setup

### Prerequisites
- Python 3.7+
- ffmpeg (for audio format conversion)

### Installation

```bash
# 1. Install ffmpeg (required for MP4/MP3 conversion)
brew install ffmpeg  # On macOS
# or: apt-get install ffmpeg  # On Ubuntu/Debian
# or: choco install ffmpeg  # On Windows (with Chocolatey)

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt
```

## Project Structure

```
Segmentation/
├── segment.py              # Main segmentation pipeline
├── verify_segments.py      # Quality validation tool
├── requirements.txt        # Python dependencies
├── raw_audio/             # Input audio files (user-provided)
├── raw_video/             # Input video files (user-provided)
└── sliced_segments/       # Output directory (created automatically)
    ├── seg_001.mp3
    ├── seg_002.mp3
    └── ...
```

## Usage

### Basic Usage: Local File

Segment a single local audio file:
```bash
python segment.py
```
This will read from the hardcoded `lecture_path` variable. Modify the file or use the advanced options below.

### Advanced Options: Command Line Arguments

(Note: For current version, edit `segment.py` directly to change input/output paths)

**Recommended workflow:**
1. Place your input files in `raw_audio/` or `raw_video/` directory
2. Modify the configuration variables in `segment.py`:
   - `lecture_path`: Path to input file
   - `output_directory`: Output directory name
   - `target_segment_duration_ms`: Target duration per segment (default: 30,000ms)
3. Run: `python segment.py`
4. Verify results: `python verify_segments.py --segments_dir sliced_segments`

## Output Format

Each segment is saved as:
```
sliced_segments/seg_001.mp3
sliced_segments/seg_002.mp3
...
sliced_segments/seg_NNN.mp3
```

**File format specifications:**
- **Format**: MP3 (compressed) or WAV (uncompressed, recommended for Whisper)
- **Sample Rate**: Original sample rate preserved (typically 44.1kHz or 48kHz)
- **Channels**: Mono or stereo (original preserved)
- **Naming**: Zero-padded segment numbers (001, 002, ..., 999)

## Configuration

Edit these variables in `segment.py` to adjust segmentation behavior:

| Parameter | Default | Purpose | When to Adjust |
|---|---|---|---|
| `lecture_path` | "lecture_1.mp3" | Input audio file | Change to your source file |
| `target_length_ms` | 30000 | Target segment duration | Increase for lectures with long sentences; decrease for rapid speech |
| `min_silence_len` | 500ms | Minimum pause to consider as boundary | Lower for more granular segments; increase for coarser boundaries |
| `silence_thresh` | -40 dBFS | Silence threshold | Lower (more negative) if VAD picks up background noise; raise if it misses quiet speech |
| `keep_silence` | 250ms | Silence to preserve at boundaries | Increase for better context continuity |

## Quality Verification

After running the segmentation pipeline, always verify the output:

```bash
python verify_segments.py --segments_dir sliced_segments
```

This generates a detailed report showing:
- Duration statistics (min, max, mean)
- Duration distribution histogram
- Per-lecture breakdown
- Issues detected (bad sample rates, mono/stereo mismatch, etc.)

**Typical output:**
```
Found 47 segments in sliced_segments

── Duration stats ──────────────────────────
  Total segments : 47
  Total audio    : 45.3 min
  Mean duration  : 57.9s
  Min duration   : 12.4s
  Max duration   : 72.1s

── Duration distribution ───────────────────
  <20s     3
  20–40s   5
  40–60s   24
  60–70s   15
  >70s     0

── Per lecture ─────────────────────────────
  lecture_1: 47 segments, 45.3 min total

✓ No issues found. All segments look good!
```

## Algorithm Details

### Silence Detection Phase
1. Load the entire audio file into memory
2. Scan for silence using a threshold of -40 dBFS with a minimum duration of 500ms
3. Identify all speech intervals (periods between silence gaps)
4. Preserve 250ms of silence at segment boundaries for audio context

### Segment Merging Phase
1. Start with an empty accumulator
2. For each detected speech interval:
   - If adding it won't exceed the target duration (30 seconds), append it
   - Otherwise, save the accumulator and start a new segment
3. Save any remaining audio as the final segment

### Output Phase
1. Export each merged segment as MP3 (or WAV for higher quality)
2. Apply zero-padded numbering (001, 002, etc.)
3. Maintain the original audio properties (sample rate, channels)

## Troubleshooting

### No segments created or very few segments
- **Issue**: Silence threshold too strict
- **Solution**: Lower `silence_thresh` (make more negative, e.g., -50 instead of -40)

### Segments cut off mid-word or mid-sentence
- **Issue**: Silence boundary detection too sensitive, or speech has many pauses
- **Solution**: Increase `min_silence_len` (e.g., 700ms instead of 500ms)

### Segments too long or too short
- **Issue**: Target duration not appropriate for lecture content
- **Solution**: Adjust `target_length_ms` (e.g., 45000 for 45-second segments or 60000 for 60 seconds)

### "Could not read file" errors
- **Issue**: Unsupported audio format
- **Solution**: Convert to MP3 or WAV using ffmpeg: `ffmpeg -i input.m4a output.mp3`

## Dependencies

See `requirements.txt` for complete list:
- **pydub**: Audio processing and MP3/WAV format handling
- **torchaudio**: Audio format detection and metadata reading (for verify_segments.py)
- **torch**: Dependency for torchaudio
- **tqdm**: Progress bars (optional, for future enhancements)

## Limitations and Future Work

- Current implementation requires sufficient RAM to load entire lecture in memory
- Best results for clear speech with natural pauses (lectures, podcasts)
- May struggle with:
  - Heavy background noise
  - Music or non-speech audio
  - Very fast or very slow speech
  - Languages with different silence patterns

