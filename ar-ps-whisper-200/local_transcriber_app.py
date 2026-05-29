#!/usr/bin/env python3
"""
Palestinian Academic Speech-to-Text Local Production App
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A local, lightweight desktop application for running fine-tuned Palestinian Arabic
Whisper models directly on macOS. Features full Apple Silicon (MPS) GPU acceleration,
drag-and-drop audio upload, and a clean Gradio UI with light and dark mode support.

Requirements:
    - torch (with MPS support for Apple Silicon)
    - transformers
    - gradio
    - A pre-trained merged Whisper model at ./whisper-merged-ar-ps

Usage:
    python local_transcriber_app.py

    The app will launch a local web server with a Gradio interface. Upload an audio
    file (WAV, MP3) or record directly from your microphone to generate transcriptions.
"""

import os
import sys
import torch
import gradio as gr
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def detect_device():
    """
    Detect available hardware acceleration and select appropriate device and precision.

    Returns:
        tuple: (device_string, torch_dtype, status_message)
            - device_string: "mps" for Apple Silicon GPU or "cpu" for CPU
            - torch_dtype: torch.float32 (preferred for stability on MPS)
            - status_message: Human-readable status for UI display

    Note:
        Float32 is used on MPS instead of Float16 due to numerical stability issues
        with Whisper's attention mechanisms on Apple Silicon GPU. Float32 ensures
        accurate transcriptions on Mac hardware.
    """
    if torch.backends.mps.is_available():
        return (
            "mps",
            torch.float32,
            "🎉 Apple Silicon GPU (MPS) Active (Stable FP32 Precision)!",
        )
    else:
        return (
            "cpu",
            torch.float32,
            "⚠️ MPS not available. Running on CPU (Inference will be slower).",
        )


DEVICE, DTYPE, HARDWARE_STATUS = detect_device()
MODEL_PATH = "./whisper-merged-ar-ps"

# Global pipeline variable for lazy loading on first use
stt_pipeline = None


def load_model():
    """
    Load the fine-tuned Whisper model and initialize the speech recognition pipeline.

    This function performs lazy loading: the model is not loaded until the first
    transcription request. Subsequent calls return immediately without reloading.

    Returns:
        tuple: (status_message, gradio_update)
            - status_message: String indicating success or failure
            - gradio_update: Gradio update object for UI status display

    Side effects:
        Sets the global stt_pipeline variable on successful load.
    """
    global stt_pipeline
    if stt_pipeline is not None:
        return "Model already loaded.", gr.update(visible=False)

    if not os.path.exists(MODEL_PATH):
        error_msg = f"❌ Error: Model directory '{MODEL_PATH}' not found. Please ensure this script is in your project directory!"
        print(error_msg)
        return error_msg, gr.update(visible=True)

    try:
        print(
            f"Loading merged Whisper model from '{MODEL_PATH}' onto {DEVICE} with {DTYPE}..."
        )
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_PATH, torch_dtype=DTYPE, low_cpu_mem_usage=True, use_safetensors=True
        ).to(DEVICE)

        processor = AutoProcessor.from_pretrained(MODEL_PATH)

        stt_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=256,
            chunk_length_s=30,
            batch_size=8,
            torch_dtype=DTYPE,
            device=DEVICE,
        )
        print("🎉 Model loaded successfully!")
        return "✅ Model loaded successfully on Mac GPU!", gr.update(visible=False)
    except Exception as e:
        err_str = f"❌ Failed to load model: {str(e)}"
        print(err_str)
        return err_str, gr.update(visible=True)


def transcribe_audio(audio_path):
    """
    Transcribe an audio file using the fine-tuned Whisper model.

    The model supports code-switched Arabic-English transcription. Generate kwargs
    specify only "transcribe" task without language constraints, allowing the model
    to freely output both Arabic and English words as appropriate.

    Args:
        audio_path (str): File path to the audio file (WAV, MP3, etc.)

    Returns:
        str: Transcribed text from the audio file, or error message if transcription fails.

    Note:
        Language is not explicitly specified in generate_kwargs to avoid suppressing
        Latin characters, which would break code-switching capability.
    """
    global stt_pipeline
    if audio_path is None:
        return "⚠️ Please upload or record an audio file first."

    if stt_pipeline is None:
        status_msg, _ = load_model()
        if stt_pipeline is None:
            return f"Could not run transcription: {status_msg}"

    try:
        print(f"Processing local transcription for: {audio_path}")

        result = stt_pipeline(audio_path, generate_kwargs={"task": "transcribe"})
        return result["text"]
    except Exception as e:
        return f"❌ Inference Error: {str(e)}"


# Define custom theme with emerald primary color for professional appearance
custom_theme = gr.themes.Default(
    primary_hue="emerald", secondary_hue="slate", neutral_hue="slate"
)

# Custom CSS for right-to-left Arabic text rendering in output area
custom_css = """
footer {visibility: hidden}
.rtl-text textarea {
    direction: rtl !important;
    text-align: right !important;
    font-family: 'Tajawal', sans-serif !important;
    font-size: 1.1rem !important;
    line-height: 1.8 !important;
}
"""

with gr.Blocks(title="Palestinian Academic STT Engine") as demo:

    gr.Markdown("""
        # 🎙️ Palestinian Academic STT Engine
        ### Local Desktop Transcription Workstation (Optimized for macOS)
        """)

    # Display system hardware status
    with gr.Row():
        gr.HTML(f"""
            <div style="background-color: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); padding: 12px; border-radius: 8px; margin-bottom: 10px;">
                <p style="margin: 0; font-size: 13px; font-weight: bold; color: #10b981;">💻 SYSTEM STATUS:</p>
                <p style="margin: 4px 0 0 0; font-size: 12px; color: #6b7280;">{HARDWARE_STATUS} (Base Model: whisper-merged-ar-ps)</p>
            </div>
            """)

    # Status message for model loading state
    model_status_box = gr.Markdown(
        f"⚠️ **Model Status:** Ready to load. (Will automatically initialize on your first transcription attempt).",
        visible=True,
    )

    with gr.Row():
        # Audio input section: accepts uploaded files or microphone recording
        with gr.Column(scale=5):
            audio_input = gr.Audio(
                label="Upload CV Lecture Audio Segment (WAV / MP3)",
                type="filepath",
                sources=["upload", "microphone"],
            )
            transcribe_btn = gr.Button("🚀 Run Local ASR", variant="primary", size="lg")

        # Transcription output section: displays right-to-left Arabic text
        with gr.Column(scale=6):
            output_text = gr.Textbox(
                label="Fine-tuned ASR Transcript (Code-Switched Arabic/English)",
                placeholder="Your Palestinian academic transcript will appear here...",
                elem_classes=["rtl-text"],
                lines=10,
            )

    # Connect transcription button to transcribe_audio function
    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=audio_input,
        outputs=output_text,
        api_name="transcribe",
    )

    # Load model on app startup
    demo.load(fn=load_model, outputs=[model_status_box, model_status_box])

if __name__ == "__main__":
    # Launch Gradio web interface on local machine
    demo.launch(inbrowser=True, css=custom_css, theme=custom_theme)
