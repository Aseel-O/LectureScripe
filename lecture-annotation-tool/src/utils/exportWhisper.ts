/**
 * exportWhisper.ts
 *
 * Exports annotated audio segments in the format required for
 * fine-tuning openai/whisper-large-v3.
 *
 * Output zip structure:
 *   whisper_dataset/
 *     metadata.jsonl          ← one JSON object per line
 *     audio/
 *       segment_001.wav
 *       segment_002.wav
 *       ...
 *
 * Each metadata.jsonl line:
 * {
 *   "audio": { "path": "audio/segment_001.wav" },
 *   "sentence": "...",                    ← the verified transcript
 *   "language": "ar",                     ← primary language tag (ISO 639-1)
 *   "locale": "ar-PS",                    ← Palestinian Arabic BCP-47 tag
 *   "duration_seconds": 12.4,
 *   "source_file": "lecture_03.mp3",
 *   "segment_index": 0,
 *   "start_ms": 0,
 *   "end_ms": 12400
 * }
 *
 * This matches the HuggingFace datasets AudioFolder convention so you can
 * load the exported folder directly with:
 *   datasets.load_dataset("audiofolder", data_dir="whisper_dataset/")
 */

// We use JSZip (v3) loaded via CDN in index.html, or imported when bundled.
// The function receives the JSZip constructor via dependency injection so it
// works in both environments.
export interface AnnotatedSegment {
  /** Unique segment id, e.g. "segment_001" */
  id: string;
  /** Verified / corrected transcript text */
  transcript: string;
  /** Raw audio blob (will be converted to WAV if needed) */
  audioBlob: Blob;
  /** MIME type of audioBlob, e.g. "audio/wav" or "audio/webm" */
  audioMimeType: string;
  /** Source lecture file name */
  sourceFile: string;
  /** Zero-based index within the source file */
  segmentIndex: number;
  /** Segment start in milliseconds */
  startMs: number;
  /** Segment end in milliseconds */
  endMs: number;
}

export interface ExportOptions {
  /** Folder name inside the zip (default: "whisper_dataset") */
  datasetName?: string;
  /** Primary ISO 639-1 language code (default: "ar") */
  language?: string;
  /** BCP-47 locale tag (default: "ar-PS") */
  locale?: string;
}

/**
 * Converts any audio Blob to WAV format using the Web Audio API.
 * If the blob is already a WAV, it is returned as-is.
 */
async function toWav(blob: Blob, mimeType: string): Promise<Blob> {
  if (mimeType === "audio/wav" || mimeType === "audio/wave") return blob;

  const arrayBuffer = await blob.arrayBuffer();
  const audioCtx = new AudioContext();
  const decoded = await audioCtx.decodeAudioData(arrayBuffer);

  const numChannels = decoded.numberOfChannels;
  const sampleRate = decoded.sampleRate;
  const numFrames = decoded.length;
  const bytesPerSample = 2; // 16-bit PCM

  // Interleave channels
  const pcmData = new Int16Array(numFrames * numChannels);
  for (let ch = 0; ch < numChannels; ch++) {
    const channelData = decoded.getChannelData(ch);
    for (let i = 0; i < numFrames; i++) {
      const sample = Math.max(-1, Math.min(1, channelData[i]));
      pcmData[i * numChannels + ch] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
    }
  }

  // Build WAV header
  const dataLength = pcmData.byteLength;
  const buffer = new ArrayBuffer(44 + dataLength);
  const view = new DataView(buffer);

  const writeStr = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataLength, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true);  // PCM format
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * bytesPerSample, true); // byte rate
  view.setUint16(32, numChannels * bytesPerSample, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeStr(36, "data");
  view.setUint32(40, dataLength, true);

  new Int16Array(buffer, 44).set(pcmData);

  await audioCtx.close();
  return new Blob([buffer], { type: "audio/wav" });
}

/**
 * Builds a zip file containing the metadata.jsonl and all audio segments.
 *
 * @param segments   - Array of annotated segments to export
 * @param JSZip      - The JSZip constructor (import JSZip from "jszip")
 * @param options    - Optional export configuration
 * @returns          - A Blob of the final zip file
 */
export async function buildWhisperExportZip(
  segments: AnnotatedSegment[],
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  JSZip: any,
  options: ExportOptions = {}
): Promise<Blob> {
  const {
    datasetName = "whisper_dataset",
    language = "ar",
    locale = "ar-PS",
  } = options;

  const zip = new JSZip();
  const root = zip.folder(datasetName)!;
  const audioFolder = root.folder("audio")!;

  const jsonlLines: string[] = [];

  for (const seg of segments) {
    // Pad index to 3 digits, e.g. "segment_007"
    const paddedIndex = String(seg.segmentIndex + 1).padStart(3, "0");
    const wavName = `${seg.id || `segment_${paddedIndex}`}.wav`;
    const audioPath = `audio/${wavName}`;

    // Convert to WAV
    const wavBlob = await toWav(seg.audioBlob, seg.audioMimeType);
    const wavArrayBuffer = await wavBlob.arrayBuffer();
    audioFolder.file(wavName, wavArrayBuffer);

    // Build metadata record
    const durationSeconds = parseFloat(
      ((seg.endMs - seg.startMs) / 1000).toFixed(3)
    );

    const record = {
      audio: { path: audioPath },
      sentence: seg.transcript,
      language,
      locale,
      duration_seconds: durationSeconds,
      source_file: seg.sourceFile,
      segment_index: seg.segmentIndex,
      start_ms: seg.startMs,
      end_ms: seg.endMs,
    };

    jsonlLines.push(JSON.stringify(record));
  }

  root.file("metadata.jsonl", jsonlLines.join("\n") + "\n");

  // Also add a README inside the dataset
  root.file(
    "README.md",
    `# Whisper Fine-Tuning Dataset

Exported by Lecture Annotation Tool — NLP Dialect Lab.

## Contents
- \`metadata.jsonl\` — one record per segment (HuggingFace AudioFolder format)
- \`audio/\` — WAV audio files (16-bit PCM)

## Load with HuggingFace datasets
\`\`\`python
from datasets import load_dataset

ds = load_dataset("audiofolder", data_dir="./whisper_dataset")
\`\`\`

## Segments: ${segments.length}
## Language: ${locale}
`
  );

  return zip.generateAsync({
    type: "blob",
    compression: "DEFLATE",
    compressionOptions: { level: 6 },
  });
}

/**
 * Triggers a browser download of the given Blob.
 */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}
