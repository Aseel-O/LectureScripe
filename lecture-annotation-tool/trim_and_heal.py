"""
Audio Trimming and Segment Merging Utility

This utility processes a directory of WAV files to:
1. Trim silence from the beginning and end of each audio segment
2. Merge short segments (< 3 seconds) with adjacent longer segments to improve dataset balance
3. Enforce maximum duration limits (30 seconds per segment) for transcription API compatibility

Configuration:
- THRESHOLD: RMS-based silence detection threshold (0.01 is standard)
- MIN_DURATION_S: Minimum segment duration; shorter segments are marked for merging
- MAX_DURATION_S: Maximum segment duration; merged segments cannot exceed this
- SILENCE_GAP_S: Duration of artificial silence inserted between merged segments
- PAD_MS: Padding (ms) preserved around detected speech to avoid clipping

Usage:
    python trim_and_heal.py --dir /path/to/wav/files

Output:
    Modified WAV files in-place with silence removed and short segments merged.
"""

import argparse
import re
import wave
from pathlib import Path
import numpy as np

# Configuration parameters
THRESHOLD = 0.01  # RMS-based silence detection threshold (lower = more sensitive)
MIN_DURATION_S = 3.0  # Minimum duration; shorter segments will be merged with neighbors
MAX_DURATION_S = 30.0  # Maximum duration per segment (API constraint)
SILENCE_GAP_S = (
    0.75  # Duration of silence between merged segments (realistic speech pauses)
)
PAD_MS = 150  # Padding around detected speech to prevent clipping


def natural_sort_key(s):
    """
    Generate a sorting key that handles numeric portions naturally.

    Example: ["segment_001.wav", "segment_010.wav", "segment_002.wav"]
    sorts naturally instead of lexicographically.

    @param s: String to generate sort key for
    @return: List suitable for sorting (mixes int and str)
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", str(s))
    ]


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """
    Read a WAV file and return samples and sample rate.

    Converts integer PCM samples to normalized float32 [-1.0, 1.0] range.

    @param path: Path to WAV file
    @return: Tuple of (audio_samples, sample_rate_hz)
    """
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    # Convert 16-bit PCM to float32 normalized to [-1.0, 1.0]
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sr


def write_wav(path: Path, samples: np.ndarray, sr: int) -> None:
    """
    Write audio samples to a WAV file in 16-bit PCM format.

    Converts normalized float32 samples back to 16-bit integer PCM range.

    @param path: Output file path
    @param samples: Audio samples in float32 format [-1.0, 1.0]
    @param sr: Sample rate in Hz
    """
    # Clip and convert float32 to 16-bit signed integer
    pcm = np.clip(samples * 32768.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def trim_samples(samples, sr, threshold=THRESHOLD):
    """
    Remove silence from the beginning and end of audio.

    Uses RMS-based activity detection on 50ms windows. Clips audio to the
    first and last windows containing speech, with padding to prevent clipping.

    @param samples: Audio samples as float32 array
    @param sr: Sample rate in Hz
    @param threshold: RMS threshold for silence detection
    @return: Tuple of (trimmed_samples, duration_seconds)
    """
    if len(samples) == 0:
        return samples, 0

    # RMS-based activity detection using 50ms windows
    win = int(0.05 * sr)  # 50ms window
    active = []
    for i in range(0, len(samples), win):
        chunk = samples[i : i + win]
        if len(chunk) == 0:
            continue
        # RMS = sqrt(mean(sample^2))
        rms = np.sqrt(np.mean(chunk**2))
        active.append(rms > threshold)

    if not any(active):
        return np.array([], dtype=np.float32), 0

    # Find the first and last active windows
    first_win = next(i for i, a in enumerate(active) if a)
    last_win = next(i for i, a in enumerate(reversed(active)) if a)
    last_win = len(active) - 1 - last_win

    # Add padding to avoid clipping the beginning and end
    start_idx = max(0, (first_win * win) - int(PAD_MS / 1000 * sr))
    end_idx = min(len(samples), ((last_win + 1) * win) + int(PAD_MS / 1000 * sr))

    trimmed = samples[start_idx:end_idx]
    return trimmed, len(trimmed) / sr


def main():
    parser = argparse.ArgumentParser(
        description="Trim silence and merge short audio segments."
    )
    parser.add_argument(
        "--dir", type=str, required=True, help="Directory containing .wav files"
    )
    args = parser.parse_args()

    audio_dir = Path(args.dir)
    # Sort files naturally so segment_001 comes before segment_010
    wav_files = sorted(audio_dir.glob("*.wav"), key=natural_sort_key)

    # Phase 1: Trim all files and collect metadata
    print(f"Phase 1: Trimming silence from {len(wav_files)} files...")
    segments = []
    for f in wav_files:
        samples, sr = read_wav(f)
        trimmed_samples, duration = trim_samples(samples, sr)

        if duration == 0:
            # Delete fully silent files
            print(f"  🗑  Deleting {f.name} (fully silent)")
            f.unlink()
            continue

        # Overwrite file with trimmed version immediately
        write_wav(f, trimmed_samples, sr)
        segments.append({"path": f, "dur": duration, "sr": sr})

    # Phase 2: Plan merge operations
    print(
        f"\nPhase 2: Planning merges (Min: {MIN_DURATION_S}s, Max: {MAX_DURATION_S}s)..."
    )
    i = 0
    while i < len(segments):
        seg = segments[i]
        if seg["dur"] < MIN_DURATION_S:
            # Try merging with the previous segment
            if i > 0:
                combined = segments[i - 1]["dur"] + seg["dur"] + SILENCE_GAP_S
                if combined <= MAX_DURATION_S:
                    print(
                        f"  🔗 [BACK]    {seg['path'].name} -> {segments[i - 1]['path'].name} ({combined:.1f}s)"
                    )
                    segments[i - 1]["dur"] = combined
                    segments[i - 1].setdefault("append", []).append(seg["path"])
                    segments.pop(i)
                    continue
            # Try merging with the next segment
            if i < len(segments) - 1:
                combined = segments[i + 1]["dur"] + seg["dur"] + SILENCE_GAP_S
                if combined <= MAX_DURATION_S:
                    print(
                        f"  🔗 [FORWARD] {seg['path'].name} -> {segments[i + 1]['path'].name} ({combined:.1f}s)"
                    )
                    segments[i + 1]["dur"] = combined
                    segments[i + 1].setdefault("prepend", []).insert(0, seg["path"])
                    segments.pop(i)
                    continue
        i += 1

    # Phase 3: Execute merges
    print("\nPhase 3: Finalizing audio files...")
    for seg in segments:
        if "prepend" in seg or "append" in seg:
            samples, sr = read_wav(seg["path"])
            # Insert artificial silence gap between merged segments
            silence = np.zeros(int(SILENCE_GAP_S * sr), dtype=np.float32)

            # Prepend files
            for p_path in seg.get("prepend", []):
                p_samples, _ = read_wav(p_path)
                samples = np.concatenate([p_samples, silence, samples])
                p_path.unlink()

            # Append files
            for a_path in seg.get("append", []):
                a_samples, _ = read_wav(a_path)
                samples = np.concatenate([samples, silence, a_samples])
                a_path.unlink()

            write_wav(seg["path"], samples, sr)

    print("\n" + "═" * 60)
    print("Done! Dead air removed and short segments merged successfully.")


if __name__ == "__main__":
    main()
