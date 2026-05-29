/**
 * TypeScript Type Definitions for Lecture Annotation Tool
 *
 * Defines core data structures for managing lecture segments, speaker turns,
 * and annotation project metadata.
 */

/** Segment annotation status in the annotation workflow */
export type SegmentStatus = "not_started" | "transcribing" | "draft" | "completed";

/**
 * Represents a speaker turn (dialogue segment) within a lecture segment.
 * Used for speaker diarization and dialogue tracking.
 */
export interface SpeakerTurn {
  /** Unique identifier for this turn */
  id: string;
  /** Speaker identifier (e.g., "Speaker 1", "Dr. Ahmed", "Student Q") */
  speaker: string;
  /** Text content of what the speaker said */
  text: string;
  /** Optional: start time in MM:SS format */
  timeStart?: string;
  /** Optional: end time in MM:SS format */
  timeEnd?: string;
}

/**
 * Represents a single voice segment (audio clip) from a lecture.
 * Contains metadata, transcription, speaker information, and annotation status.
 */
export interface VoiceSegment {
  /** Unique segment identifier (usually derived from filename) */
  id: string;
  /** Display name of the segment (usually filename) */
  name: string;
  /** File size in bytes */
  size: number;
  /** MIME type of the audio file (e.g., "audio/wav", "audio/mp3", "audio/webm") */
  mimeType: string;
  /** Duration in seconds */
  duration: number;
  /** Raw transcription text (either AI-generated or verified) */
  transcript: string;
  /** Current annotation status in the workflow */
  status: SegmentStatus;
  /** Text direction: "rtl" for Arabic, "ltr" for English */
  direction: "rtl" | "ltr";
  /** Temporary object URL for playback (created with URL.createObjectURL) */
  audioUrl?: string;
  /** Array of speaker turns in this segment (dialogue structure) */
  speakerTurns: SpeakerTurn[];
  /** Whether the actual audio file is loaded in the current session */
  hasAudioFile: boolean;
  /** Optional: User-defined tags for categorization */
  tags?: string[];
  /** Optional: Path to the audio file on disk */
  audio_path?: string;
  /** Optional: Primary language code (ISO 639-1, e.g., "ar", "en") */
  language?: string;
  /** Optional: Dataset split assignment ("train", "val", "test") */
  split?: "train" | "val" | "test";
}

/**
 * Aggregated statistics for the entire annotation project.
 * Useful for progress tracking and project overview.
 */
export interface AnnotationProjectStats {
  /** Total number of segments in the project */
  total: number;
  /** Number of completed (fully annotated) segments */
  completed: number;
  /** Number of segments in draft (AI-generated, awaiting review) status */
  drafts: number;
  /** Number of segments currently being transcribed */
  transcribing: number;
  /** Number of segments not yet started */
  notStarted: number;
}
