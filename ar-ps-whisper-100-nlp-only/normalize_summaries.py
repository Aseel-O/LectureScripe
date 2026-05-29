"""
Text Normalization Pipeline for Arabic Summaries
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Normalizes generated Arabic summaries to eliminate layout-based evaluation bias.
Converts structured formats (markdown lists, headers, bullet points) into clean
continuous prose using Google's Gemini API, with a regex fallback for robustness.

This preprocessing step ensures that summarization models are evaluated on content
quality rather than formatting choices, providing fair comparison across different
model outputs.

Usage:
    python normalize_summaries.py --input_json raw_summaries.json --output_json normalized.json

Requirements:
    - google.genai (Google Generative AI SDK)
    - GEMINI_API_KEY environment variable set
"""

import os
import re
import json
import argparse
import logging
from google import genai
from google.genai import types

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("FormatNormalizer")


def get_gemini_client():
    """
    Initialize and return a Google Gemini API client.

    Returns:
        genai.Client: Authenticated Gemini client instance.

    Raises:
        ValueError: If GEMINI_API_KEY environment variable is not set.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing GEMINI_API_KEY environment variable. Please export it."
        )
    return genai.Client(api_key=api_key)


def clean_markdown_fallback(text: str) -> str:
    """
    Remove markdown formatting artifacts using deterministic regex patterns.

    Cleans structural tokens including headers, bold formatting, bullet points,
    and numbered lists. Used as a fallback when API calls fail and as a final
    validation layer to ensure 100% markdown removal.

    Args:
        text (str): Raw text potentially containing markdown formatting.

    Returns:
        str: Cleaned text with markdown elements removed and whitespace normalized.
    """
    if not text:
        return ""

    # Remove markdown headers (e.g., #, ##, ###)
    text = re.sub(re.compile(r"^#+\s+", re.MULTILINE), "", text)
    # Remove markdown bold/italic formatting marks (*, **, _, __)
    text = re.sub(r"[\*_]{1,2}", "", text)
    # Remove bullet markers at the start of lines (*, -, •)
    text = re.sub(re.compile(r"^\s*[\-\*•]\s+", re.MULTILINE), "", text)
    # Remove numbered list markers at the start of lines (e.g., 1., 2.)
    text = re.sub(re.compile(r"^\s*\d+[\.\)]\s+", re.MULTILINE), "", text)
    # Collapse multiple newlines/tabs into clean continuous spacing
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_text_to_prose(client: genai.Client, text: str) -> str:
    """
    Convert structured text into fluent, continuous Arabic prose.

    Uses Google Gemini to intelligently reformat text from various structured
    formats (lists, headers, bullets) into a single cohesive paragraph while
    preserving all academic content and technical terms.

    Args:
        client (genai.Client): Authenticated Gemini API client.
        text (str): Raw structured text to normalize.

    Returns:
        str: Clean prose text without markdown or structural elements.

    Note:
        Falls back to regex-based cleaning if API call fails.
    """
    if not text:
        return ""

    prompt = f"""
    You are an expert academic text editor. Take the following Arabic lecture text and rewrite it into a single, continuous, fluent, formal Arabic paragraph prose style.
    
    STRICT FORMATTING RULES:
    1. Do NOT use any Markdown structural elements. Never output '##', '#', or '**'.
    2. Do NOT use any lists, bullet points, numbering, dashes, or line breaks. The output must be one solid paragraph block.
    3. Maintain 100% of the original academic, technical terms, and facts.
    4. Do NOT include introductory remarks or preambles (e.g., do not say "إليك النص المعدل:"). Start directly with the prose content.
    
    Original Text:
    {text}
    """
    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
            ),
        )
        llm_output = response.text.strip()
        # Apply regex fallback to ensure 100% compliance with markdown removal
        return clean_markdown_fallback(llm_output)
    except Exception as e:
        log.error(f"Gemini API call failed: {e}. Falling back to regex-based cleaning.")
        return clean_markdown_fallback(text)


def main():
    """
    Main entry point: normalize all summaries in the input JSON file.

    Processes each entry and applies text normalization to summary fields,
    saving the normalized versions to the output JSON file.
    """
    parser = argparse.ArgumentParser(
        description="Normalize model summary layouts to eliminate evaluation bias."
    )
    parser.add_argument(
        "--input_json",
        default="summaries_finetuned.json",
        help="Path to your raw generated summaries JSON.",
    )
    parser.add_argument(
        "--output_json",
        default="summaries_finetuned_normalized.json",
        help="Path to save normalized summaries.",
    )
    args = parser.parse_args()

    # Initialize Gemini client
    try:
        client = get_gemini_client()
    except ValueError as e:
        log.error(e)
        return

    # Validate input file exists
    if not os.path.exists(args.input_json):
        log.error(f"Input file {args.input_json} not found.")
        return

    # Load JSON data
    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    log.info(f"Normalizing text fields from {args.input_json}...")

    # Process each entry in the dataset
    for idx, entry in enumerate(data):
        log.info(f"Processing item {idx+1}/{len(data)}...")

        # Normalize Qwen's summary blocks if present
        if (
            "qwen" in entry
            and isinstance(entry["qwen"], dict)
            and "summary_ar" in entry["qwen"]
        ):
            raw_text = entry["qwen"]["summary_ar"]
            entry["qwen"]["summary_ar_normalized"] = normalize_text_to_prose(
                client, raw_text
            )

        elif "qwen_summary" in entry and entry["qwen_summary"]:
            entry["qwen_summary_normalized"] = normalize_text_to_prose(
                client, entry["qwen_summary"]
            )

        # Clean reference text block if it contains markdown headings/bullets
        if "reference_summary" in entry and entry["reference_summary"]:
            entry["reference_summary_normalized"] = normalize_text_to_prose(
                client, entry["reference_summary"]
            )

    # Save normalized data
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    log.info(f"Successfully saved cleanly normalized dataset to → {args.output_json}")


if __name__ == "__main__":
    main()
