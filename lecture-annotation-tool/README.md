# Lecture Annotation Tool - verify_dataset

Small utility to validate a directory of WAV files before running
automatic transcription. It checks per-file duration and detects
leading/trailing silence to help surface common issues.

Usage
-----

Run the verifier from the repository root:

```bash
python3 lecture-annotation-tool/verify_dataset.py --dir /path/to/wav/folder
```

Requirements
------------

- Python 3.8+
- numpy

Notes
-----
- The script expects standard 16-bit PCM WAV files readable by the
  Python `wave` stdlib module.
- Adjust `MIN_DUR`, `MAX_DUR`, and `SILENCE_THRESH` in the script if
  your dataset has different constraints.

Contact
-------
If you need help integrating this into your preprocessing pipeline,
open an issue or contact the maintainer.
<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://ai.google.dev/static/site-assets/images/share-ais-513315318.png" />
</div>

# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/a4f6a531-e03e-4115-aabf-d48c79c5b56b

## Run Locally

**Prerequisites:**  Node.js


1. Install dependencies:
   `npm install`
2. Set the `GEMINI_API_KEY` in [.env.local](.env.local) to your Gemini API key
3. Run the app:
   `npm run dev`
