"""
Dataset Quality Validator for Arabic JSONL Files
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Automated validation utility to verify that all annotation guidelines and
normalization rules have been successfully applied to cleaned datasets.

Validates:
    - Removal of bracketed speaker/noise tags
    - Elimination of raw newline characters
    - Proper normalization of 'اللي' spelling
    - Removal of unnecessary Shadda from solar letters
    - Unification of source file identifiers (no _segXXX.wav suffixes)
    - Distribution statistics per lecture

Usage:
    python validate_jsonl.py -v data/dataset_final.jsonl

Output:
    Comprehensive validation report with pass/fail status for each rule
    and segment count statistics per lecture.
"""

import json
import re
import argparse
from collections import Counter


def validate_dataset(file_path):
    """
    Validate a cleaned JSONL dataset against all annotation guidelines.

    Reads the dataset and checks each record for violations of the cleaning
    rules. Reports detailed statistics on data quality and completeness.

    Args:
        file_path (str): Path to the JSONL file to validate.

    Output:
        Prints a detailed validation report including:
        - Per-rule compliance status (pass/fail)
        - Violation counts for each rule
        - Segment count distribution per lecture
    """
    print(f"🔍 Validating dataset file: {file_path}\n" + "=" * 60)

    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    solar_letters = "تثدذرزسشصضطظلن"

    # Track violations of each cleaning rule
    issues = {
        "speaker_tags": 0,
        "newlines": 0,
        "shadda_elli": 0,
        "solar_shadda": 0,
        "seg_suffix": 0,
    }

    # Check each record for rule violations
    for r in records:
        text = r.get("sentence", "")
        source = r.get("source_file", "")

        # Check for unremoved bracketed tags
        if re.search(r"\[.*?\]", text):
            issues["speaker_tags"] += 1
        # Check for raw newline characters
        if "\n" in text or "\r" in text:
            issues["newlines"] += 1
        # Check for improper 'اللي' spelling with Shadda
        if "الّلي" in text or "اللّي" in text:
            issues["shadda_elli"] += 1
        # Check for Shadda on solar letters after definite article
        if re.search(rf"ال[{solar_letters}]ّ", text):
            issues["solar_shadda"] += 1
        # Check for unremoved segment suffixes in source file
        if re.search(r"_seg\d+\.wav", source):
            issues["seg_suffix"] += 1

    # Print quality control report
    print("📊 [Data Quality & Cleanliness Report]:")
    all_clean = True

    if issues["speaker_tags"] == 0:
        print(
            "  ✅ Success: All speaker and noise tags [ ] have been completely removed."
        )
    else:
        print(
            f"  ❌ Warning: Found {issues['speaker_tags']} records containing unremoved brackets/tags!"
        )
        all_clean = False

    if issues["newlines"] == 0:
        print("  ✅ Success: Text field is free from raw newline characters (\\n).")
    else:
        print(
            f"  ❌ Warning: Found {issues['newlines']} records containing raw newlines!"
        )
        all_clean = False

    if issues["shadda_elli"] == 0:
        print("  ✅ Success: The word 'اللي' is properly standardized without Shadda.")
    else:
        print(
            f"  ❌ Warning: Found {issues['shadda_elli']} instances of improperly formatted 'اللي'!"
        )
        all_clean = False

    if issues["solar_shadda"] == 0:
        print(
            "  ✅ Success: All redundant solar Shaddas have been removed from standard words."
        )
    else:
        print(
            f"  ❌ Warning: Found {issues['solar_shadda']} words still carrying improper solar Shaddas."
        )
        all_clean = False

    if issues["seg_suffix"] == 0:
        print(
            "  ✅ Success: All source file identifiers are unified (no _segXXX.wav suffixes)."
        )
    else:
        print(
            f"  ❌ Warning: Found {issues['seg_suffix']} records still carrying segment suffixes!"
        )
        all_clean = False

    # Print final verdict
    if all_clean:
        print(
            "\n🎉 VERDICT: Dataset is 100% CLEAN and fully compliant with all guidelines!"
        )
    else:
        print("\n⚠️ VERDICT: Validation FAILED. Please review the issues listed above.")

    # Print lecture statistics
    print("\n📈 [Segment Counts Per Lecture]:")
    counts = Counter(r["source_file"] for r in records)
    for lecture, n in sorted(counts.items()):
        print(f"  📁 {lecture}: {n} segments")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate cleaned JSONL datasets against annotation guidelines."
    )
    parser.add_argument(
        "-v",
        "--validate",
        required=True,
        help="Path to the cleaned JSONL file to validate",
    )

    args = parser.parse_args()
    validate_dataset(args.validate)
