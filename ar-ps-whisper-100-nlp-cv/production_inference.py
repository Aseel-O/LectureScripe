"""
Production Inference Script for Palestinian Arabic Whisper
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Command-line utility for transcribing audio files using a fine-tuned Whisper model.
Supports code-switched Arabic-English transcription with automatic device detection
(GPU acceleration on Apple Silicon or CPU fallback).

Usage:
    python production_inference.py <audio_file.wav>
    python production_inference.py <audio_file.mp3>

Requirements:
    - torch with hardware acceleration support
    - transformers
    - Pre-trained merged model at ./whisper-merged-ar-ps
"""

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sys
import os


def load_local_model(model_path):
    """
    Load a fine-tuned Whisper model and initialize the transcription pipeline.

    Automatically detects available hardware acceleration (Apple Silicon GPU or CPU)
    and selects appropriate precision settings for optimal performance.

    Args:
        model_path (str): Path to the directory containing the merged Whisper model.

    Returns:
        transformers.pipeline: Initialized speech recognition pipeline ready for inference.

    Raises:
        FileNotFoundError: If the model directory does not exist.
        RuntimeError: If model loading fails due to missing dependencies or memory issues.
    """
    print("Checking hardware acceleration availability...")

    # Detect Apple Silicon GPU (M1/M2/M3) or fallback to CPU
    if torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float32  # Float32 for stability on MPS
        print("🚀 Apple Silicon GPU (MPS) detected! Using GPU acceleration.")
    else:
        device = "cpu"
        torch_dtype = torch.float32
        print("⚠️ GPU acceleration unavailable. Using CPU mode.")

    print(f"Loading fine-tuned Whisper model from: {model_path}...")

    # Load the fine-tuned model with memory-efficient settings
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)

    # Load the audio processor and tokenizer
    processor = AutoProcessor.from_pretrained(model_path)

    # Initialize the automatic speech recognition pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe


if __name__ == "__main__":
    # Validate command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python production_inference.py <path_to_audio_file.wav>")
        sys.exit(1)

    audio_file = sys.argv[1]

    # Check that the audio file exists
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found.")
        sys.exit(1)

    # Path to the pre-trained merged Whisper model
    MY_MODEL_PATH = "./whisper-merged-ar-ps"

    # Load the model and initialize the transcription pipeline
    pipeline_stt = load_local_model(MY_MODEL_PATH)

    print(f"\n🎧 Transcribing: {audio_file}...\n" + "=" * 50)

    # Run inference on the audio file
    result = pipeline_stt(audio_file, generate_kwargs={"task": "transcribe"})

    print("\n✅ FINAL TRANSCRIPT:\n")
    print(result["text"])
    print("\n" + "=" * 50)
