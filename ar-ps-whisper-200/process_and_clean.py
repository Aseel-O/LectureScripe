"""
Data Cleaning and Preprocessing Pipeline for JSONL Datasets
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Comprehensive preprocessing pipeline for Arabic transcription datasets in JSONL format.
Applies linguistic and structural normalization rules to prepare data for model training.

Key Operations:
    1. Remove speaker/noise tags (e.g., [speaker], [noise])
    2. Normalize newline handling to prevent word merging
    3. Standardize the word 'اللي' (al-ladi) diacritical variants
    4. Remove unnecessary Shadda from solar letters
    5. Unify lecture identifiers by removing segment suffixes

Usage:
    python process_and_clean.py -i data/metadata.jsonl -o data/dataset_final.jsonl

Input Format:
    JSONL file with fields: "sentence" (transcript text), "source_file" (lecture identifier)

Output:
    Cleaned JSONL file with all annotation guidelines applied and normalized text.
"""

import json
import re
import argparse


def clean_text_annotations(text: str) -> str:
    """
    Apply comprehensive Arabic text normalization pipeline.

    Sequential processing ensures stability: each rule builds on previous results.

    Args:
        text (str): Raw Arabic text potentially containing annotations and diacritics.

    Returns:
        str: Fully cleaned text with all artifacts and inconsistencies resolved.

    Processing Steps:
        1. Remove bracketed speaker/noise tags [speaker], [noise], [inaudible]
        2. Replace newlines with spaces to preserve word boundaries
        3. Standardize spelling variant of 'اللي' (remove Shadda)
        4. Remove Shadda from solar letters (ت، ث، د، etc.) after definite article
        5. Collapse whitespace and trim leading/trailing characters
    """
    if not text:
        return ""

    # 1. Remove bracketed tags and annotations
    text = re.sub(r"\[.*?\]", "", text)

    # 2. Replace newlines and carriage returns with single space
    text = text.replace("\n", " ").replace("\r", " ")

    # 3. Standardize 'اللي' variants (remove Shadda to get single form)
    text = re.sub(r"الّلي|اللّي", "اللي", text)

    # 4. Remove unnecessary Shadda from solar letters after definite article 'ال'
    # Solar letters: ت ث د ذ ر ز س ش ص ض ط ظ ل ن
    solar_letters = "تثدذرزسشصضطظلن"
    solar_pattern = rf"(ال[{solar_letters}])ّ"
    text = re.sub(solar_pattern, r"\1", text)

    # 5. Collapse multiple spaces and strip edges
    text = re.sub(r" +", " ", text).strip()

    return text


def process_jsonl_pipeline(input_path, output_path):
    """
    Read, clean, and normalize a JSONL dataset file line by line.

    Processes records sequentially to minimize memory usage on large datasets.
    Maintains statistics on dropped records and lecture distribution.

    Args:
        input_path (str): Path to input JSONL file.
        output_path (str): Path to save cleaned JSONL file.

    Side Effects:
        Creates output_path with cleaned records.
        Prints summary statistics to console.
    """
    processed_count = 0
    dropped_count = 0

    with open(input_path, "r", encoding="utf-8") as infile, open(
        output_path, "w", encoding="utf-8"
    ) as outfile:

        for line in infile:
            # Skip empty lines
            if not line.strip():
                continue

            data = json.loads(line)

            # Clean the sentence text field
            if "sentence" in data:
                original_text = data["sentence"]
                cleaned_text = clean_text_annotations(original_text)

                # Drop segments that become empty after cleaning (only contained tags)
                if not cleaned_text:
                    dropped_count += 1
                    continue

                data["sentence"] = cleaned_text

            # Unify lecture identifiers by removing segment suffixes
            if "source_file" in data:
                # Remove _segXXX.wav or _segXXX suffix from filename
                lecture_id = re.sub(
                    r"_seg\d+\\.wav$|_seg\d+\.wav$", "", data["source_file"]
                )
                data["source_file"] = lecture_id

            # Write cleaned record to output
            json_line = json.dumps(data, ensure_ascii=False)
            outfile.write(json_line + "\n")
            processed_count += 1

    # Print processing summary
    print("\n🚀 Processing Pipeline Completed Successfully!")
    print(f"✅ Total clean records retained: {processed_count}")
    print(f"🗑️ Total empty records dropped: {dropped_count}")
    print(f"📁 Cleaned dataset saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean and normalize text in JSONL datasets for Arabic ASR training."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to the input JSONL dataset file"
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to save the cleaned output JSONL file",
    )

    args = parser.parse_args()
    process_jsonl_pipeline(args.input, args.output)
