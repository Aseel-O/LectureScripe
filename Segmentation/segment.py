"""
Audio Segmentation Module

This module segments long-form audio files (lectures, podcasts) into manageable chunks
by detecting natural silence boundaries. It uses silence detection to identify natural
speech pauses and groups speech segments into roughly equal-sized chunks suitable for
transcription or annotation.

The segmentation algorithm:
1. Detects all speech intervals separated by at least 500ms of silence
2. Greedily merges adjacent speech intervals until reaching ~30 seconds
3. Exports each grouped segment as a separate MP3 file

This approach preserves natural speech boundaries while maintaining consistent segment
lengths, making the output ideal for downstream processing pipelines.
"""

from pydub import AudioSegment
from pydub.silence import split_on_silence
import os


def segment_audio(lecture_path, output_dir="sliced_segments", target_length_ms=30000):
    """
    Segment a long-form audio file into chunks at natural silence boundaries.

    This function splits audio at natural pauses (silence periods) and combines small
    segments to reach a target duration. This preserves natural speech boundaries while
    maintaining consistent segment lengths.

    Args:
        lecture_path (str): Path to the input audio file (MP3, WAV, etc.).
        output_dir (str): Directory to save output segments. Created if it doesn't exist.
                         Default is 'sliced_segments'.
        target_length_ms (int): Target duration for each output segment in milliseconds.
                               Default is 30000ms (30 seconds).

    Returns:
        int: Total number of segments created.

    Note:
        - Requires ffmpeg for certain audio formats
        - Each output segment is saved as "{output_dir}/seg_{NNN}.mp3" where NNN is
          a zero-padded segment number (001, 002, etc.)
        - The last segment may be shorter than the target duration
    """
    # Load the audio file
    audio = AudioSegment.from_file(lecture_path)

    print("Analyzing audio and splitting on natural silence...")

    # Split audio at silence points to identify natural speech boundaries
    # min_silence_len: Minimum duration of silence to be considered a boundary (500ms)
    # silence_thresh: Silence threshold in dBFS (-40 dBFS is the cutoff for "silent")
    # keep_silence: Preserve a small amount of silence at boundaries for context (250ms)
    chunks = split_on_silence(
        audio, min_silence_len=500, silence_thresh=-40, keep_silence=250
    )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize variables for chunk processing
    current_chunk = AudioSegment.empty()
    chunk_count = 1

    # Process speech segments: combine small chunks until reaching target duration
    for sub_chunk in chunks:
        # If adding this segment doesn't exceed target length, accumulate it
        if len(current_chunk) + len(sub_chunk) < target_length_ms:
            current_chunk += sub_chunk
        else:
            # Current accumulation is full; save it and start a new segment
            if len(current_chunk) > 0:
                current_chunk.export(
                    f"{output_dir}/seg_{chunk_count:03d}.mp3", format="mp3"
                )
                chunk_count += 1
            current_chunk = sub_chunk

    # Export any remaining audio that didn't fill a complete segment
    if len(current_chunk) > 0:
        current_chunk.export(f"{output_dir}/seg_{chunk_count:03d}.mp3", format="mp3")

    return chunk_count


if __name__ == "__main__":
    # Configuration for segmentation
    lecture_path = "lecture_1.mp3"
    output_directory = "sliced_segments"
    target_segment_duration_ms = 30000  # 30 seconds

    # Run segmentation
    num_segments = segment_audio(
        lecture_path=lecture_path,
        output_dir=output_directory,
        target_length_ms=target_segment_duration_ms,
    )

    print(
        f"Done! Created {num_segments} naturally sliced segments in '{output_directory}'"
    )
