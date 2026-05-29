import argparse
import re
import wave
from pathlib import Path
import numpy as np

"""verify_dataset

Utility script to scan a directory of WAV files and report basic issues
that commonly affect automatic transcription pipelines. The script checks
duration bounds and detects leading/trailing silence using a short RMS
window. It prints a concise table and a summary of issues found.

Usage example:
    python verify_dataset.py --dir /path/to/wav/folder

Notes:
    - The script only reads WAV files that are readable by the stdlib
        `wave` module and assumes 16-bit PCM samples.
    - Configuration constants below control the acceptable duration range
        and the RMS threshold used for silence detection.
"""

# Configuration: acceptable duration (seconds) and RMS silence threshold
MIN_DUR = 3.0
MAX_DUR = 30.0
SILENCE_THRESH = 0.002  # RMS threshold for silence detection


def natural_sort_key(s):
    """Return a key for natural/alphanumeric sorting.

    Splits the input string into runs of digits and non-digits so that
    file names like "file2" and "file10" sort in the expected human
    order (2 before 10).

    Args:
        s: Any object coercible to string (typically a Path or filename).

    Returns:
        A list of alternating integers and lowercased strings suitable for
        passing as the ``key`` argument to :func:`sorted`.
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", str(s))
    ]


def get_audio_info(path):
    """Read a WAV file and return duration and silence flags.

    The function opens the file using the stdlib ``wave`` module and
    converts 16-bit PCM frames into a floating point numpy array in the
    range [-1.0, 1.0]. It then inspects the first and last 200ms of
    audio to estimate whether there is leading or trailing silence.

    Args:
        path: Path-like object referring to a WAV file.

    Returns:
        A tuple ``(duration_seconds, lead_silent, trail_silent)`` where
        ``lead_silent`` and ``trail_silent`` are booleans indicating if
        the respective ends contain RMS energy below ``SILENCE_THRESH``.
    """
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        frames = wf.getnframes()
        duration = frames / sr
        raw = wf.readframes(frames)
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    # Inspect the first and last ~200ms of audio for low RMS energy.
    check_samples = int(0.2 * sr)

    def is_silent(chunk):
        if len(chunk) == 0:
            return True
        return np.sqrt(np.mean(chunk**2)) < SILENCE_THRESH

    lead_silent = is_silent(samples[:check_samples])
    trail_silent = is_silent(samples[-check_samples:])

    return duration, lead_silent, trail_silent


def main():
    """Command-line entry point.

    Parses a single required argument ``--dir`` pointing to a directory
    containing WAV files to verify. The function prints a small table of
    per-file status and a short summary including total/average
    durations and the number of files requiring attention.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Folder to verify")
    args = parser.parse_args()

    audio_dir = Path(args.dir)
    files = sorted(audio_dir.glob("*.wav"), key=natural_sort_key)

    print(f"Checking {len(files)} files in '{audio_dir.name}'...\n")
    print(f"{'Filename':<25} | {'Dur':<7} | {'Status'}")
    print("-" * 50)

    issues = 0
    total_dur = 0

    for f in files:
        dur, lead_sil, trail_sil = get_audio_info(f)
        total_dur += dur

        status_msgs = []
        if dur < MIN_DUR:
            status_msgs.append(f"TOO SHORT ({dur:.2f}s)")
        if dur > MAX_DUR:
            status_msgs.append(f"TOO LONG ({dur:.2f}s)")
        if lead_sil:
            status_msgs.append("LEAD SILENCE")
        if trail_sil:
            status_msgs.append("TRAIL SILENCE")

        if status_msgs:
            print(f"{f.name:<25} | {dur:>5.2f}s | ❌ {', '.join(status_msgs)}")
            issues += 1
        else:
            # Optional: enable the line below to print files that pass checks
            # print(f"{f.name:<25} | {dur:>5.2f}s | ✅ OK")
            pass

    print("-" * 50)
    print(f"TOTAL FILES      : {len(files)}")
    print(f"TOTAL DURATION   : {total_dur / 60:.2f} minutes")
    print(f"AVERAGE DURATION : {total_dur / len(files) if files else 0:.2f}s")
    print(f"FILES WITH ISSUES: {issues}")

    if issues == 0:
        print("\n✨ ALL GOOD! Your dataset is ready for AI transcription.")
    else:
        print(f"\n⚠️ Found {issues} items to double-check.")


if __name__ == "__main__":
    main()
