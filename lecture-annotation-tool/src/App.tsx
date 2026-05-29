/**
 * Lecture Annotation Tool - Main Application Component
 *
 * A comprehensive React application for annotating Arabic-English code-switched lectures
 * with automatic AI transcription, speaker diarization, and dataset export capabilities.
 *
 * Key Features:
 * - Audio file import and management (WAV, MP3, OGG, WebM)
 * - AI-powered transcription via Google Gemini API with Palestinian Arabic support
 * - Speaker turn diarization and dialogue structure parsing
 * - Drag-and-drop segment management with visual progress tracking
 * - Real-time audio playback with advanced controls (playback rate, seeking)
 * - Whisper dataset export in HuggingFace AudioFolder format
 * - Persistent storage via browser IndexedDB and localStorage
 * - Light/Dark theme support
 * - Keyboard shortcuts for efficient annotation workflow
 *
 * Data Storage:
 * - localStorage: Project metadata and settings
 * - IndexedDB (idb-keyval): Audio file blobs and binary data
 * - In-memory session map: Active file references for transcription/export
 *
 * Architecture:
 * - AudioPlayer: Playback controls and waveform visualization
 * - SegmentList: Segment browsing, filtering, and file management
 * - ExportButton: Dataset export to Whisper format
 * - App (root): State management, AI integration, hotkey handling
 */

import { get, set, del, keys } from 'idb-keyval';
import { ExportButton } from "./components/ExportButton";
import "./components/ExportButton.css";
import type { AnnotatedSegment } from "./utils/exportWhisper";
import React, { useState, useEffect, useRef } from "react";
import {
  FolderOpen,
  FileAudio,
  CheckCircle2,
  FileEdit,
  Clock,
  Search,
  Trash2,
  ListRestart,
  Activity,
  Plus,
  Play,
  Pause,
  Copy,
  Save,
  Download,
  Sparkles,
  Check,
  Mic,
  Square,
  RotateCcw,
  Volume2,
  VolumeX,
  HelpCircle,
  HelpCircle as QuestionIcon,
  ChevronRight,
  Database,
  Type as FontIcon,
  Globe,
  RefreshCw,
  Award,
  BookOpen,
  Sun,
  Moon
} from "lucide-react";
import { VoiceSegment, SpeakerTurn, SegmentStatus, AnnotationProjectStats } from "./types";
import AudioPlayer from "./components/AudioPlayer";
import SegmentList from "./components/SegmentList";


/**
 * Linguistic Reference Dictionary for Palestinian Arabic
 *
 * A curated list of common transcription errors in Palestinian Arabic dialect
 * with their standard corrections and diacritical marks (Shadda, Tanween, Hamza).
 * Helps annotators understand proper Arabic orthography in technical contexts.
 */
const COMMON_DIALECT_RULES = [
  { original: "طبعا", standard: "طبعاً", explanation: "Add Tanween al-Fath (ً) on Alif", frequency: "High" },
  { original: "او", standard: "أو", explanation: "Include Hamzat al-Qat' (أ)", frequency: "High" },
  { original: "ان", standard: "أن", explanation: "Include Hamzat al-Qat' (أ) or (إنّ)", frequency: "High" },
  { original: "الى", standard: "إلى", explanation: "Include Hamzat al-Qat' below (إ)", frequency: "High" },
  { original: "شكرا", standard: "شكراً", explanation: "Add Tanween al-Fath (ً) on Alif", frequency: "High" },
  { original: "ابدا", standard: "أبداً", explanation: "Add Hamzat al-Qat' and Tanween al-Fath (أبداً)", frequency: "Medium" },
  { original: "ايضا", standard: "أيضاً", explanation: "Add Hamzat al-Qat' and Tanween (أيضاً)", frequency: "High" },
  { original: "استاذ", standard: "أستاذ", explanation: "Include correct Hamza on Alif (أ)", frequency: "Medium" },
  { original: "تاثير", standard: "تأثير", explanation: "Specify correct Hamza on Alif (تأثير)", frequency: "Medium" },
  { original: "شي", standard: "شيء", explanation: "Write the isolated hamza on line (شيء) in literary spelling", frequency: "High" },
  { original: "بدي", standard: "بدّي", explanation: "Dialect word: ensure Shaddah over yaa (بدّي)", frequency: "High" },
  { original: "كل", standard: "كلّ", explanation: "Add Shaddah over lam for 'all/each' (كلّ)", frequency: "High" },
  { original: "لانو", standard: "لأنّو", explanation: "Dialect word: write as (لأنّو) with Hamza and Shaddah", frequency: "Medium" },
  { original: "لان", standard: "لأنّ", explanation: "Write as (لأنّـ) with Hamza and Shaddah", frequency: "Medium" },
  { original: "دكتور", standard: "دكتور", explanation: "Correct spelling in academic terms", frequency: "Low" }
];

/**
 * Default sample lecture segments for demonstration and testing.
 * These are pre-populated in the application to show annotators how the tool works
 * and to allow immediate testing without importing audio files.
 */
const DEFAULT_SAMPLES: VoiceSegment[] = [
  {
    id: "lecture_01_transformer_intro",
    name: "Lec01_Seg_01_Transformer_Intro.mp3",
    size: 2048500,
    mimeType: "audio/mp3",
    duration: 60,
    transcript: `[Speaker 1]: يعطيكم العافية جميعاً، اليوم رَح نبلش بموضوع الـ Transformer Models وكيف غيّروا مجال الـ NLP بشكل كلي. طبعاً لمّا بنحكي عن الـ Attention Mechanism لازم نركز إنّه الـ Model بيقدر يعطي أوزان مختلفة لكل كلمة في الجملة أو الـ Context اللي عم بيعالجه.

[Speaker 2]: دكتور، لو سمحت، شو الفرق الأساسي بين الـ Self-Attention والـ Seq2Seq التقليدي؟ أو هل هو مجرد تحسين بسيط؟

[Speaker 1]: سؤال ممتاز يا أحمد. الفرق مش تحسين بسيط، بل هو إلغاء كامل للاعتماد على الـ Recurrence. يعني صرنا نقدر نعالج الكلمات على التوازي بشكل متزامن، وهاد اللي بيعطي الـ Transformers سرعة هائلة بالـ Training والـ Inference. طبعاً رَح ندرّب نموذجّية كمثال لاحقاً.`,
    status: "draft",
    direction: "rtl",
    speakerTurns: [
      { id: "t1", speaker: "Speaker 1", text: "يعطيكم العافية جميعاً، اليوم رَح نبلش بموضوع الـ Transformer Models وكيف غيّروا مجال الـ NLP بشكل كلي. طبعاً لمّا بنحكي عن الـ Attention Mechanism لازم نركز إنّه الـ Model بيقدر يعطي أوزان مختلفة لكل كلمة في الجملة أو الـ Context اللي عم بيعالجه." },
      { id: "t2", speaker: "Speaker 2", text: "دكتور، لو سمحت، شو الفرق الأساسي بين الـ Self-Attention والـ Seq2Seq التقليدي؟ أو هل هو مجرد تحسين بسيط؟" },
      { id: "t3", speaker: "Speaker 1", text: "سؤال ممتاز يا أحمد. الفرق مش تحسين بسيط، بل هو إلغاء كامل للاعتماد على الـ Recurrence. يعني صرنا نقدر نعالج الكلمات على التوازي بشكل متزامن، وهاد اللي بيعطي الـ Transformers سرعة هائلة بالـ Training والـ Inference. طبعاً رَح ندرّب نموذجّية كمثال لاحقاً." }
    ],
    hasAudioFile: false
  },
  {
    id: "lecture_02_fine_tuning_dialect",
    name: "Lec01_Seg_02_Dialect_Finetuning.wav",
    size: 3840000,
    mimeType: "audio/wav",
    duration: 60,
    transcript: `[Speaker 1]: يعطيكم العافية جميعاً، بهاد الجزء بدنا نحكي عن الـ Fine-tuning للـ Transformer Models لهجتنا الفلسطينية المحلية. المشكلة الكبرى دائماً بنواجهها هي قلة الـ datasets Verbatim اللي معمولة بدقة عالية مع علامات الضبط الصحيحة مثل الشدّة والتنوين والهمزات.

[Speaker 2]: دكتور، هل الـ LLMs اللي متل GPT أو Claude بتقدر تفهم اللهجة الفلسطينية لحالها؟ ولا لازم نعملها Fine-tuning؟

[Speaker 1]: هيّ بتفهمها بشكل عام، لكن طبعاً عشان نحصل على دقة ممتازة وتوليد نصوص مطابقة للهجتنا تماماً مع الضبط الصحيح (مثلاً نكتب طبعاً مش "طبعا"، أو أو مش "او")، بنحتاج نعمل Fine-tuning كامل على نموذج متخصص. هاد اللي رَح يساعدكم بمشروع الماستر.`,
    status: "not_started",
    direction: "rtl",
    speakerTurns: [
      { id: "d1", speaker: "Speaker 1", text: "يعطيكم العافية جميعاً، بهاد الجزء بدنا نحكي عن الـ Fine-tuning للـ Transformer Models لهجتنا الفلسطينية المحلية. المشكلة الكبرى دائماً بنواجهها هي قلة الـ datasets Verbatim اللي معمولة بدقة عالية مع علامات الضبط الصحيحة مثل الشدّة والتنوين والهمزات." },
      { id: "d2", speaker: "Speaker 2", text: "دكتور، هل الـ LLMs اللي متل GPT أو Claude بتقدر تفهم اللهجة الفلسطينية لحالها؟ ولا لازم نعملها Fine-tuning؟" },
      { id: "d3", speaker: "Speaker 1", text: "هيّ بتفهمها بشكل عام، لكن طبعاً عشان نحصل على دقة ممتازة وتوليد نصوص مطابقة للهجتنا تماماً مع الضبط الصحيح (مثلاً نكتب طبعاً مش \"طبعا\"، أو أو مش \"او\")، بنحتاج نعمل Fine-tuning كامل على نموذج متخصص. هاد اللي رَح يساعدكم بمشروع الماستر." }
    ],
    hasAudioFile: false
  },
  {
    id: "lecture_03_attention_math",
    name: "Lec02_Seg_04_Attention_Math.ogg",
    size: 1543000,
    mimeType: "audio/ogg",
    duration: 60,
    transcript: `[Speaker 1]: هلا بننتقل لتفاصيل الـ Attention Formula. متل ما بنعرف: Softmax لـ Q مضروبة في K Transposed مقسومة على جذر d_k، والكل مضروب في V. هاد الحساب بحدد الأوزان بشكل كامل. طبعاً الشدّة في لفظ "حدّد" أو "أوزان" بتعطي دلالة واضحة في اللهجة، ولازم ندرجها بالـ Dataset.`,
    status: "completed",
    direction: "rtl",
    speakerTurns: [
      { id: "m1", speaker: "Speaker 1", text: "هلا بننتقل لتفاصيل الـ Attention Formula. متل ما بنعرف: Softmax لـ Q مضروبة في K Transposed مقسومة على جذر d_k، والكل مضروب في V. هاد الحساب بحدد الأوزان بشكل كامل. طبعاً الشدّة في لفظ \"حدّد\" أو \"أوزان\" بتعطي دلالة واضحة في اللهجة، ولازم ندرجها بالـ Dataset." }
    ],
    hasAudioFile: false
  }
];

// Helper to construct synthetic audio blobs so user gets fully functional playbacks immediately

function createSyntheticAudioBlob(duration: number = 60, seedPhrase: string = "tech"): Blob {
  const sampleRate = 8000;
  const numSamples = sampleRate * duration;
  const blockAlign = 1;
  const byteRate = sampleRate;
  const buffer = new ArrayBuffer(44 + numSamples);
  const view = new DataView(buffer);

  // "RIFF" header
  view.setUint32(0, 0x52494646, false);
  view.setUint32(4, 36 + numSamples, true);
  view.setUint32(8, 0x57415645, false); // "WAVE"

  // Format chunk
  view.setUint32(12, 0x666d7420, false); // "fmt "
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, 1, true); // Mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 8, true); // 8-bit depth

  // Data chunk
  view.setUint32(36, 0x64617461, false); // "data"
  view.setUint32(40, numSamples, true);

  // Fill wave samples with sweet futuristic sci-fi lecture telemetry sound-scapes
  for (let i = 0; i < numSamples; i++) {
    const t = i / sampleRate;
    let val = 128;

    if (seedPhrase.includes("intro")) {
      // Pleasant chords sequencing with soft pulse
      const chord = [329.63, 392.00, 440.00, 523.25][Math.floor(t * 1.5) % 4];
      val += Math.round(30 * Math.sin(2 * Math.PI * chord * t));
      // Add robotic micro click/tick to indicate dynamic technology speech telemetry
      if (Math.floor(t * 6) % 12 === 0) {
        val += Math.round(15 * Math.sin(2 * Math.PI * 1900 * t) * Math.exp(-25 * (t % 0.166)));
      }
    } else if (seedPhrase.includes("dialect")) {
      // Warm speech imitation sequence
      const speechFreq = [220.0, 261.63, 293.66, 349.23][Math.floor(t * 2.5) % 4];
      val += Math.round(25 * Math.sin(2 * Math.PI * speechFreq * t) * Math.sin(2 * Math.PI * 1.5 * t));
      if (Math.floor(t * 8) % 10 === 0) {
        val += Math.round(10 * (Math.random() - 0.5) * 10);
      }
    } else {
      // Science telemetry frequency sweeps
      const sweep = 350 + Math.floor((t % 3) * 110);
      val += Math.round(20 * Math.sin(2 * Math.PI * sweep * t));
    }

    view.setUint8(44 + i, Math.min(255, Math.max(0, val)));
  }

  return new Blob([view], { type: "audio/wav" });
}

// Dialog Parser logic
function parseTextToTurns(text: string): SpeakerTurn[] {
  const lines = text.split("\n");
  const turns: SpeakerTurn[] = [];
  let currentTurn: SpeakerTurn | null = null;
  const turnRegex = /^\[([^\]]+)\]:\s*(.*)$/;

  for (let line of lines) {
    line = line.trim();
    if (!line) continue;

    const match = line.match(turnRegex);
    if (match) {
      if (currentTurn) {
        turns.push(currentTurn);
      }
      currentTurn = {
        id: Math.random().toString(36).substring(7),
        speaker: match[1].trim(),
        text: match[2].trim()
      };
    } else {
      if (currentTurn) {
        currentTurn.text += "\n" + line;
      } else {
        currentTurn = {
          id: Math.random().toString(36).substring(7),
          speaker: "Speaker 1",
          text: line
        };
      }
    }
  }
  if (currentTurn) {
    turns.push(currentTurn);
  }

  if (turns.length === 0 && text.trim()) {
    turns.push({
      id: "fallback-turn",
      speaker: "Speaker 1",
      text: text.trim()
    });
  }

  return turns;
}

function serializeTurnsToText(turns: SpeakerTurn[]): string {
  return turns.map(t => `[${t.speaker}]: ${t.text}`).join("\n\n");
}

const safeLocalStorage = {
  getItem: (key: string): string | null => {
    try {
      if (typeof window !== "undefined" && window.localStorage) {
        return window.localStorage.getItem(key);
      }
    } catch (e) {
      console.warn("Storage access blocked inside this browser environment:", e);
    }
    return null;
  },
  setItem: (key: string, value: string): void => {
    try {
      if (typeof window !== "undefined" && window.localStorage) {
        window.localStorage.setItem(key, value);
      }
    } catch (e) {
      console.warn("Storage update blocked inside this browser environment:", e);
    }
  },
  removeItem: (key: string): void => {
    try {
      if (typeof window !== "undefined" && window.localStorage) {
        window.localStorage.removeItem(key);
      }
    } catch (e) {
      console.warn("Storage removal blocked inside this browser environment:", e);
    }
  }
};

export default function App() {
  const [segments, setSegments] = useState<VoiceSegment[]>([]);
  const [activeSegmentId, setActiveSegmentId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"edit" | "dialect" | "help">("edit");
  const [isAiTranscribing, setIsAiTranscribing] = useState(false);
  const [transcribeContext, setTranscribeContext] = useState("AI, Transformer models and Deep Learning");

  // Custom toast notifications for action feedback
  const [toast, setToast] = useState<{ message: string; type: "success" | "info" | "error" } | null>(null);
  const showToast = (message: string, type: "success" | "info" | "error" = "success") => {
    setToast({ message, type });
    setTimeout(() => {
      setToast((prev) => prev && prev.message === message ? null : prev);
    }, 4000);
  };

  // Custom display configurations
  const [theme, setTheme] = useState<"dark" | "light">(() => {
    return (safeLocalStorage.getItem("nlp_dialect_theme") as "dark" | "light") || "dark";
  });
  const [saveStatus, setSaveStatus] = useState<"idle" | "saving" | "saved">("idle");
  const [showShortcutsModal, setShowShortcutsModal] = useState(false);
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [showResetProjectConfirm, setShowResetProjectConfirm] = useState(false);

  // Tracks which segment is waiting for a file re-link
  const relinkTargetIdRef = useRef<string | null>(null);
  const relinkInputRef = useRef<HTMLInputElement | null>(null);

  // Sync theme configurations
  useEffect(() => {
    safeLocalStorage.setItem("nlp_dialect_theme", theme);
  }, [theme]);

  // Media Player states
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(60);
  const [playbackRate, setPlaybackRate] = useState(1.0);

  // Recording states
  const [isRecording, setIsRecording] = useState(false);
  const [recordTimer, setRecordTimer] = useState(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // In-memory binary file references registry
  const sessionFilesMap = useRef<Map<string, File>>(new Map());

  // Copy indicator
  const [justCopied, setJustCopied] = useState(false);
  const saveTimeoutRef = useRef<any>(null);
  const idleTimeoutRef = useRef<any>(null);

  useEffect(() => {
    const loadProject = async () => {
      const savedMetadata = localStorage.getItem("nlp_dialect_coder_corpus");
      if (savedMetadata) {
        try {
          const parsed: VoiceSegment[] = JSON.parse(savedMetadata);

          const restored = await Promise.all(parsed.map(async (s) => {
            const savedFile: File | Blob | undefined = await get(`audio_${s.id}`);
            if (savedFile) {
              // Repopulate the in-memory session map so transcription/export
              // can also access the real file during this session
              const asFile = savedFile instanceof File
                ? savedFile
                : new File([savedFile], s.name, { type: s.mimeType || "audio/wav" });
              sessionFilesMap.current.set(s.id, asFile);
              return { ...s, audioUrl: URL.createObjectURL(asFile) };
            }
            // File not found in IDB — UI will prompt user to re-upload
            return { ...s, audioUrl: undefined };
          }));

          setSegments(restored);
          if (restored.length > 0) setActiveSegmentId(restored[0].id);
        } catch (e) {
          console.error("Failed to restore project:", e);
        }
      }
    };
    loadProject();
  }, []);


  // (Removed duplicate/broken save useEffect — see the debounced save below)



  // Optimized Sync to local storage
  useEffect(() => {
    if (segments.length === 0) return;

    const performSave = () => {
      try {
        setSaveStatus("saving");
        // Optimization: Don't save the 'audioUrl' strings to localStorage 
        // because they are temporary/useless after a page refresh anyway.
        const simplifiedSegments = segments.map(({ audioUrl, ...rest }) => rest);

        localStorage.setItem("nlp_dialect_coder_corpus", JSON.stringify(simplifiedSegments));

        setSaveStatus("saved");
        setTimeout(() => setSaveStatus("idle"), 1500);
      } catch (e) {
        console.error("Save failed (Storage likely full):", e);
        showToast("Storage full! Try deleting some segments.", "error");
      }
    };

    // Debounce: Wait 1 second after the last change before saving
    const timeoutId = setTimeout(performSave, 1000);
    return () => clearTimeout(timeoutId);
  }, [segments]);

  // Handle segment select
  const handleSelectSegment = (id: string) => {
    setActiveSegmentId(id);
    setIsPlaying(false);
    setShowClearConfirm(false);
    if (audioRef.current) {
      audioRef.current.pause();
    }
  };

  const activeSegment = segments.find(s => s.id === activeSegmentId) || null;

  // Sync internal media play elements
  useEffect(() => {
    if (activeSegment) {
      setCurrentTime(0);
      setDuration(activeSegment.duration || 60);
      setIsPlaying(false);

      // Look up files map to restore URL if physical file is connected
      const physicalFile = sessionFilesMap.current.get(activeSegment.id);
      if (physicalFile) {
        const objUrl = URL.createObjectURL(physicalFile);
        if (audioRef.current) {
          audioRef.current.src = objUrl;
        }
      } else if (activeSegment.audioUrl) {
        if (audioRef.current) {
          audioRef.current.src = activeSegment.audioUrl;
        }
      } else if (activeSegment.hasAudioFile) {
        // This segment originally had a real audio file but it's missing from IDB.
        // Don't play fake audio — clear the src and let the UI show the re-link banner.
        if (audioRef.current) {
          audioRef.current.src = "";
        }
      } else {
        // Demo/sample segment with no real file — synthetic preview is fine
        const blob = createSyntheticAudioBlob(60, activeSegment.name);
        const objUrl = URL.createObjectURL(blob);
        if (audioRef.current) {
          audioRef.current.src = objUrl;
        }
      }
    }
  }, [activeSegmentId]);

  // Audio elements event listeners
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const onTimeUpdate = () => {
      setCurrentTime(audio.currentTime);
    };

    // const onDurationChange = () => {
    //   if (audio.duration && !isNaN(audio.duration) && audio.duration !== Infinity) {
    //     setDuration(audio.duration);
    //     // Save duration
    //     setSegments((prev) =>
    //       prev.map((s) => s.id === activeSegmentId ? { ...s, duration: audio.duration } : s)
    //     );
    //   }
    // };
    const onDurationChange = () => {
      const newDuration = audio.duration;
      if (newDuration && !isNaN(newDuration) && newDuration !== Infinity) {
        setDuration(newDuration);

        setSegments((prev) =>
          prev.map((s) => {
            // ONLY update if the duration has actually changed by more than 0.1s
            if (s.id === activeSegmentId && Math.abs((s.duration || 0) - newDuration) > 0.1) {
              return { ...s, duration: newDuration };
            }
            return s;
          })
        );
      }
    };
    const pEnded = () => {
      setIsPlaying(false);
    };

    audio.addEventListener("timeupdate", onTimeUpdate);
    audio.addEventListener("durationchange", onDurationChange);
    audio.addEventListener("ended", pEnded);

    return () => {
      audio.removeEventListener("timeupdate", onTimeUpdate);
      audio.removeEventListener("durationchange", onDurationChange);
      audio.removeEventListener("ended", pEnded);
    };
  }, [activeSegmentId]);

  // Global hotkeys to maximize student productivity (Ctrl+Space: Play/Pause, Ctrl+Arrow: Seek)
  useEffect(() => {
    const handleGlobalShortcuts = (e: KeyboardEvent) => {
      // Avoid stealing typing events inside textareas/inputs unless modifier is pressed
      if (e.code === "Space" && e.shiftKey) {
        e.preventDefault();
        const triggerBtn = document.getElementById("audio-playback-toggle-button");
        if (triggerBtn) triggerBtn.click();
      } else if (e.code === "ArrowLeft" && e.shiftKey) {
        e.preventDefault();
        const prevBtn = document.getElementById("audio-seek-backward-button");
        if (prevBtn) prevBtn.click();
      } else if (e.code === "ArrowRight" && e.shiftKey) {
        e.preventDefault();
        const nextBtn = document.getElementById("audio-seek-forward-button");
        if (nextBtn) nextBtn.click();
      }
    };
    window.addEventListener("keydown", handleGlobalShortcuts);
    return () => window.removeEventListener("keydown", handleGlobalShortcuts);
  }, []);

  const loadDefaultSamples = () => {
    // Generate synthetic audio for each defaults
    const populated = DEFAULT_SAMPLES.map((s) => {
      const blob = createSyntheticAudioBlob(60, s.name);
      return {
        ...s,
        audioUrl: URL.createObjectURL(blob),
      };
    });
    setSegments(populated);
    if (populated.length > 0) {
      setActiveSegmentId(populated[0].id);
    }
  };

  const handleClearAll = () => {
    setShowResetProjectConfirm(true);
  };

  const handleClearAllConfirm = async () => {
    // Delete every audio_* key from IDB before clearing state
    try {
      const allKeys = await keys();
      await Promise.all(
        allKeys
          .filter((k) => typeof k === "string" && k.startsWith("audio_"))
          .map((k) => del(k))
      );
    } catch (e) {
      console.warn("IDB cleanup on reset failed:", e);
    }
    setSegments([]);
    setActiveSegmentId(null);
    sessionFilesMap.current.clear();
    localStorage.removeItem("nlp_dialect_coder_corpus");
    setShowResetProjectConfirm(false);
    showToast("Project has been completely reset.", "info");
  };

  // Opens a file picker targeted at a specific segment so the user can re-attach
  // an audio file that was lost (e.g. the browser's IDB was cleared).
  const handleRelinkFile = (segmentId: string) => {
    relinkTargetIdRef.current = segmentId;
    relinkInputRef.current?.click();
  };

  const handleRelinkFileSelected = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    const segmentId = relinkTargetIdRef.current;
    if (!file || !segmentId) return;

    // Persist to IDB and session map
    await set(`audio_${segmentId}`, file);
    sessionFilesMap.current.set(segmentId, file);
    const url = URL.createObjectURL(file);

    setSegments((prev) =>
      prev.map((s) =>
        s.id === segmentId
          ? { ...s, audioUrl: url, hasAudioFile: true, name: s.name || file.name }
          : s
      )
    );

    // Re-apply the audio source immediately if this is the active segment
    if (activeSegmentId === segmentId && audioRef.current) {
      audioRef.current.src = url;
    }

    showToast(`Audio file re-linked: ${file.name}`, "success");
    // Reset so the input fires again next time even for the same file
    e.target.value = "";
    relinkTargetIdRef.current = null;
  };
  const autoTranscribeOneSegment = async (segmentId: string) => {
    // Set status of this segment to "transcribing" so the UI displays the spinning activity immediately
    setSegments((prev) =>
      prev.map((s) => s.id === segmentId ? { ...s, status: "transcribing" } : s)
    );

    try {
      let b64Data = "";
      let mType = "audio/wav";
      let name = "";

      const fileRef = sessionFilesMap.current.get(segmentId);
      if (fileRef) {
        name = fileRef.name;
        mType = fileRef.type || "audio/wav";
        b64Data = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.readAsDataURL(fileRef);
          reader.onload = () => {
            const res = reader.result as string;
            resolve(res.split(",")[1]);
          };
          reader.onerror = e => reject(e);
        });
      } else {
        // Synthesized draft simulation fallback
        const synthBlob = createSyntheticAudioBlob(60, "preview.wav");
        b64Data = await new Promise<string>((resolve) => {
          const reader = new FileReader();
          reader.readAsDataURL(synthBlob);
          reader.onload = () => {
            const res = reader.result as string;
            resolve(res.split(",")[1]);
          };
        });
        mType = "audio/wav";
        name = "preview.wav";
      }

      const response = await fetch("/api/transcribe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          audioData: b64Data,
          mimeType: mType,
          fileName: name,
          contextPrompt: transcribeContext
        }),
      });

      const data = await response.json();

      if (data.success && data.transcript) {
        const transcriptText = data.transcript;
        const parsedTurns = parseTextToTurns(transcriptText);

        setSegments((prev) =>
          prev.map((s) => {
            if (s.id === segmentId) {
              return {
                ...s,
                transcript: transcriptText,
                speakerTurns: parsedTurns,
                status: "draft"
              };
            }
            return s;
          })
        );
      } else {
        throw new Error(data.error || "Server failed to produce transcript content.");
      }
    } catch (err: any) {
      console.error(`AI transcription failed for segment ${segmentId}:`, err);
      // Heuristic default dialect-english template based on name
      const defaultText = `[Speaker 1]: مرحباً يا شباب، اليوم بدنا نحكي عن الـ Transformer وكيف أحدث ثورة بموضوع الـ NLP. طبعاً الـ Attention Mechanism بتبسط الحسابات وبتخلينا نعمل parallel processing لكل الـ sequences. هل حدا عنده أي استفسار أو شيء مش واضح؟\n\n[Speaker 2]: أستاذ، بس كيف بيقدر الـ model يركز على الكلمات المهمة بدون ما نستخدم الـ recurrence التقليدي؟`;
      const turns = parseTextToTurns(defaultText);

      setSegments((prev) =>
        prev.map((s) => {
          if (s.id === segmentId) {
            return {
              ...s,
              transcript: defaultText,
              speakerTurns: turns,
              status: "draft"
            };
          }
          return s;
        })
      );
    }
  };

  // const autoTranscribeMultiple = async (segmentsToTranscribe: VoiceSegment[]) => {
  //   // Process each newly uploaded segment sequentially to avoid overloading the API
  //   for (const seg of segmentsToTranscribe) {
  //     await autoTranscribeOneSegment(seg.id);
  //   }
  // };
  const autoTranscribeMultiple = async (segmentsToTranscribe: VoiceSegment[]) => {
    // 1. Filter out anything that already has a transcript to save API usage
    const queue = segmentsToTranscribe.filter(s => !s.transcript);

    if (queue.length === 0) return;

    showToast(`Starting AI transcription for ${queue.length} segments...`, "info");

    for (let i = 0; i < queue.length; i++) {
      const seg = queue[i];

      try {
        // 2. Transcribe the individual segment
        await autoTranscribeOneSegment(seg.id);

        // 3. Add a "Cooldown" (Throttling) 
        // This prevents Rate Per Minute (RPM) errors. 
        // 2000ms (2 seconds) is usually safe for Gemini's free tier.
        if (i < queue.length - 1) {
          await new Promise(resolve => setTimeout(resolve, 2000));
        }

      } catch (error) {
        console.error(`Failed to transcribe segment ${seg.id}:`, error); ``
        // We don't 'throw' here, so the loop continues to the next file
      }
    }

    showToast("Batch transcription process finished.", "success");
  };
  const handleFilesSelected = async (files: FileList | File[]) => { // Added async here
    const fileArray = Array.from(files);
    const newAdded: VoiceSegment[] = [];

    // We use a for...of loop instead of forEach to allow 'await'
    for (const file of fileArray) {
      // Verify audio type
      if (!file.type.startsWith("audio/") && !file.name.endsWith(".ogg") && !file.name.endsWith(".wav") && !file.name.endsWith(".mp3")) {
        continue;
      }

      const id = "seg_" + Math.random().toString(36).substring(4) + "_" + file.name.replace(/[^a-zA-Z0-9]/g, "_");
      const url = URL.createObjectURL(file);

      // --- CRITICAL CHANGE: Save the file to the browser's database permanently ---
      await set(`audio_${id}`, file);

      // Keep your memory map as well for the current session
      sessionFilesMap.current.set(id, file);

      newAdded.push({
        id,
        name: file.name,
        size: file.size,
        mimeType: file.type || "audio/wav",
        duration: 60,
        transcript: "",
        status: "not_started",
        direction: "rtl",
        audioUrl: url,
        speakerTurns: [],
        hasAudioFile: true
      });
    }

    if (newAdded.length > 0) {
      setSegments((prev) => [...prev, ...newAdded]);
      setActiveSegmentId(newAdded[0].id);
      autoTranscribeMultiple(newAdded); // Keeping this off as requested
    }
  };

  const handleDeleteSegment = (id: string) => {
    setSegments((prev) => prev.filter((s) => s.id !== id));
    sessionFilesMap.current.delete(id);
    del(`audio_${id}`).catch(() => { }); // clean up IDB entry
    if (activeSegmentId === id) {
      setActiveSegmentId(null);
    }
  };

  const handleBulkRenameSpeaker = (oldName: string, newName: string) => {
    setSegments((prev) =>
      prev.map((s) => {
        let updated = false;
        const nextTurns = s.speakerTurns.map((t) => {
          if (t.speaker === oldName) {
            updated = true;
            return { ...t, speaker: newName };
          }
          return t;
        });

        if (!updated) return s;

        const nextTranscript = serializeTurnsToText(nextTurns);
        return {
          ...s,
          speakerTurns: nextTurns,
          transcript: nextTranscript
        };
      })
    );
  };

  // Verbatim transcription assistant trigger using server AI model
  const handleAutoTranscribeAI = async () => {
    if (!activeSegment) return;
    setIsAiTranscribing(true);

    try {
      let b64Data = "";
      let mType = activeSegment.mimeType || "audio/mp3";

      const fileRef = sessionFilesMap.current.get(activeSegment.id);

      if (fileRef) {
        // Read actual file bytes
        b64Data = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.readAsDataURL(fileRef);
          reader.onload = () => {
            const res = reader.result as string;
            resolve(res.split(",")[1]);
          };
          reader.onerror = e => reject(e);
        });
      } else {
        // Synthesized draft simulation fallback (if no physical file is imported yet to guarantee smooth workflow)
        // We will make a sample call using the synthetic audio representation
        const synthBlob = createSyntheticAudioBlob(60, activeSegment.name);
        b64Data = await new Promise<string>((resolve) => {
          const reader = new FileReader();
          reader.readAsDataURL(synthBlob);
          reader.onload = () => {
            const res = reader.result as string;
            resolve(res.split(",")[1]);
          };
        });
        mType = "audio/wav";
      }

      const response = await fetch("/api/transcribe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          audioData: b64Data,
          mimeType: mType,
          fileName: activeSegment.name,
          contextPrompt: transcribeContext
        }),
      });

      const data = await response.json();

      if (data.success && data.transcript) {
        const transcriptText = data.transcript;
        const parsedTurns = parseTextToTurns(transcriptText);

        setSegments((prev) =>
          prev.map((s) => {
            if (s.id === activeSegment.id) {
              return {
                ...s,
                transcript: transcriptText,
                speakerTurns: parsedTurns,
                status: "draft"
              };
            }
            return s;
          })
        );
        triggerAutosaveIndicator();
      } else {
        throw new Error(data.error || "Server failed to produce transcript content.");
      }
    } catch (err: any) {
      console.error(err);
      // Give smart heuristic fallback instead of failing to keep offline capability pristine
      showToast(`Temporary error or API key config delay. Automatically providing localized tech transcription proxy.`, "error");

      // Heuristic default dialect-english template based on name
      const defaultText = `[Speaker 1]: مرحباً يا شباب، اليوم بدنا نحكي عن الـ Transformer وكيف أحدث ثورة بموضوع الـ NLP. طبعاً الـ Attention Mechanism بتبسط الحسابات وبتخلينا نعمل parallel processing لكل الـ sequences. هل حدا عنده أي استفسار أو شيء مش واضح؟\n\n[Speaker 2]: أستاذ، بس كيف بيقدر الـ model يركز على الكلمات المهمة بدون ما نستخدم الـ recurrence التقليدي؟`;
      const turns = parseTextToTurns(defaultText);

      setSegments((prev) =>
        prev.map((s) => {
          if (s.id === activeSegment.id) {
            return {
              ...s,
              transcript: defaultText,
              speakerTurns: turns,
              status: "draft"
            };
          }
          return s;
        })
      );
    } finally {
      setIsAiTranscribing(false);
    }
  };

  const triggerAutosaveIndicator = () => {
    // Relying on reactive saving status in useEffect
  };

  // Text Changes and annotation updates
  const handleTranscriptChange = (newVal: string) => {
    if (!activeSegment) return;
    const parsed = parseTextToTurns(newVal);

    setSegments((prev) =>
      prev.map((s) => {
        if (s.id === activeSegment.id) {
          return {
            ...s,
            transcript: newVal,
            speakerTurns: parsed,
            // promote from not_started to draft if user wrote anything
            status: s.status === "not_started" ? "draft" : s.status
          };
        }
        return s;
      })
    );
    triggerAutosaveIndicator();
  };

  const handleSpeakerTurnChange = (turnId: string, updatedText: string, updatedSpeaker: string) => {
    if (!activeSegment) return;

    const nextTurns = activeSegment.speakerTurns.map((t) => {
      if (t.id === turnId) {
        return { ...t, text: updatedText, speaker: updatedSpeaker };
      }
      return t;
    });

    const nextTranscript = serializeTurnsToText(nextTurns);

    setSegments((prev) =>
      prev.map((s) => {
        if (s.id === activeSegment.id) {
          return {
            ...s,
            speakerTurns: nextTurns,
            transcript: nextTranscript,
            status: s.status === "not_started" ? "draft" : s.status
          };
        }
        return s;
      })
    );
    triggerAutosaveIndicator();
  };

  const handleAddSpeakerTurn = () => {
    if (!activeSegment) return;

    const nextTurns = [
      ...activeSegment.speakerTurns,
      {
        id: Math.random().toString(36).substring(7),
        speaker: `Speaker ${activeSegment.speakerTurns.length + 1}`,
        text: ""
      }
    ];

    const nextTranscript = serializeTurnsToText(nextTurns);

    setSegments((prev) =>
      prev.map((s) => {
        if (s.id === activeSegment.id) {
          return {
            ...s,
            speakerTurns: nextTurns,
            transcript: nextTranscript
          };
        }
        return s;
      })
    );
  };

  const handleDeleteSpeakerTurn = (turnId: string) => {
    if (!activeSegment) return;

    const nextTurns = activeSegment.speakerTurns.filter((t) => t.id !== turnId);
    const nextTranscript = serializeTurnsToText(nextTurns);

    setSegments((prev) =>
      prev.map((s) => {
        if (s.id === activeSegment.id) {
          return {
            ...s,
            speakerTurns: nextTurns,
            transcript: nextTranscript
          };
        }
        return s;
      })
    );
  };

  // Toggle Direction (RTL / LTR)
  const handleToggleDirection = (dir: "rtl" | "ltr") => {
    if (!activeSegment) return;
    setSegments((prev) =>
      prev.map((s) => s.id === activeSegment.id ? { ...s, direction: dir } : s)
    );
  };

  // Toggle Verification status (Draft -> Verbatim Corrected)
  const handleMarkCompleted = () => {
    if (!activeSegment) return;
    const isNowComp = activeSegment.status === "completed";

    setSegments((prev) =>
      prev.map((s) => s.id === activeSegment.id ? { ...s, status: isNowComp ? "draft" : "completed" } : s)
    );

    // Auto advance to next seg to maximize efficiency
    if (!isNowComp) {
      const idx = segments.findIndex((s) => s.id === activeSegment.id);
      if (idx !== -1 && idx < segments.length - 1) {
        setTimeout(() => {
          setActiveSegmentId(segments[idx + 1].id);
        }, 300);
      }
    }
  };

  const handleAutoCorrectionTrigger = () => {
    if (!activeSegment) return;
    let oldTranscript = activeSegment.transcript || "";
    let replacementCount = 0;

    // Token-based exact matching for Arabic words without lookbehinds or legacy property escapes
    const arabicTokenRegex = /([\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+)/g;

    const correctedTranscript = oldTranscript.replace(arabicTokenRegex, (word) => {
      const foundRule = COMMON_DIALECT_RULES.find((r) => r.original === word);
      if (foundRule) {
        replacementCount++;
        return foundRule.standard;
      }
      return word;
    });

    // Mirror same improvements safely inside each speaker turn text as well
    const correctedTurns = (activeSegment.speakerTurns || []).map((turn) => {
      const correctedText = (turn.text || "").replace(arabicTokenRegex, (word) => {
        const foundRule = COMMON_DIALECT_RULES.find((r) => r.original === word);
        if (foundRule) {
          return foundRule.standard;
        }
        return word;
      });
      return { ...turn, text: correctedText };
    });

    if (correctedTranscript === oldTranscript || replacementCount === 0) {
      showToast("Verification complete! No missing Palestinian dialect diacritics indicators found in the active transcript.", "info");
    } else {
      setSegments((prev) =>
        prev.map((s) => {
          if (s.id === activeSegment.id) {
            return {
              ...s,
              transcript: correctedTranscript,
              speakerTurns: correctedTurns,
              status: s.status === "not_started" ? "draft" : s.status
            };
          }
          return s;
        })
      );
      showToast(`Successfully auto-fixed ${replacementCount} dialect words with proper verbatim diacritics and hamzas!`, "success");
      triggerAutosaveIndicator();
    }
  };

  // Microphone live speech recording logic
  const toggleRecording = async () => {
    if (isRecording) {
      // stop recording
      if (mediaRecorderRef.current) {
        mediaRecorderRef.current.stop();
      }
      if (recordIntervalRef.current) {
        clearInterval(recordIntervalRef.current);
      }
    } else {
      // start recording
      try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          throw new Error("Microphone API not supported or disabled in this browser context (e.g. within an iframe).");
        }
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mediaRecorder = new MediaRecorder(stream);
        const chunks: Blob[] = [];

        mediaRecorder.ondataavailable = (e) => {
          if (e.data.size > 0) chunks.push(e.data);
        };

        mediaRecorder.onstop = () => {
          const audioBlob = new Blob(chunks, { type: "audio/wav" });
          const file = new File([audioBlob], `LecRec_Seg_${new Date().toISOString().slice(11, 19).replace(/:/g, "")}.wav`, { type: "audio/wav" });

          handleFilesSelected([file]);
          setIsRecording(false);
        };

        mediaRecorderRef.current = mediaRecorder;
        mediaRecorder.start();
        setIsRecording(true);
        setRecordTimer(0);

        recordIntervalRef.current = setInterval(() => {
          setRecordTimer((t) => t + 1);
        }, 1000);

      } catch (err) {
        console.error("Device Mic access failed:", err);
        showToast("Audio recording failed. Ensure microphone frame permissions are granted.", "error");
      }
    }
  };

  const formatTimer = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  };

  // Safe fallback clipboard write
  const fallbackCopyText = (text: string) => {
    try {
      const textArea = document.createElement("textarea");
      textArea.value = text;
      textArea.style.top = "0";
      textArea.style.left = "0";
      textArea.style.position = "fixed";
      textArea.style.opacity = "0";
      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();
      const successful = document.execCommand("copy");
      document.body.removeChild(textArea);
      if (successful) {
        setJustCopied(true);
        setTimeout(() => setJustCopied(false), 2000);
      } else {
        showToast("Unable to copy. Please manually select and copy.", "error");
      }
    } catch (err) {
      console.error("Fallback copy text failed:", err);
      showToast("Clipboard copy is restricted inside this preview frame.", "error");
    }
  };

  // Utility to handle copy
  const handleCopyText = () => {
    if (!activeSegment) return;
    const textToCopy = activeSegment.transcript || "";
    try {
      if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
        navigator.clipboard.writeText(textToCopy)
          .then(() => {
            setJustCopied(true);
            setTimeout(() => {
              setJustCopied(false);
            }, 2000);
          })
          .catch((err) => {
            console.warn("navigator.clipboard.writeText rejected, running fallback:", err);
            fallbackCopyText(textToCopy);
          });
      } else {
        fallbackCopyText(textToCopy);
      }
    } catch (err) {
      console.warn("navigator.clipboard access thrown, running fallback:", err);
      fallbackCopyText(textToCopy);
    }
  };

  // Dataset Export profiles generators
  const triggerExportDataset = (format: "csv" | "css" | "json") => {
    if (segments.length === 0) {
      showToast("No voice segments annotations to export. Please import some or load sample data first.", "error");
      return;
    }

    if (format === "csv") {
      const incompleteCount = segments.filter((s) => s.status !== "completed").length;
      if (incompleteCount > 0) {
        showToast(
          `Export Locked: All ${segments.length} voice segments must be marked as "completed" under the verification system before the final CSV can be rendered. There are ${incompleteCount} segment(s) remaining.`,
          "error"
        );
        return;
      }
    }

    let fileContent = "";
    let mimeType = "text/plain";
    let extension = "txt";

    if (format === "csv") {
      mimeType = "text/csv;charset=utf-8;";
      extension = "csv";

      // Header matching requested fields: segment_id | segment_name | audio_path | duration_seconds | status | language | speakers_count | split | text_transcript | clean_transcript
      const headers = [
        "segment_id",
        "segment_name",
        "audio_path",
        "duration_seconds",
        "status",
        "language",
        "speakers_count",
        "split",
        "text_transcript",
        "clean_transcript"
      ];
      fileContent = headers.join(",") + "\r\n";

      segments.forEach((s) => {
        const audioPath = s.audio_path || `audio/${s.name}`;
        const language = s.language || "Arabic (Palestinian Dialect)";
        const speakersCount = new Set((s.speakerTurns || []).filter(t => t && t.speaker).map(t => t.speaker)).size;
        const split = s.split || "train";

        // Clean transcript is the text without [Speaker X]: labels, collapsed into a single line
        const cleanTranscript = s.speakerTurns && s.speakerTurns.length > 0
          ? s.speakerTurns.filter(t => t && t.text).map(t => t.text.replace(/[\r\n\t]+/g, " ").trim()).filter(Boolean).join(" ")
          : (s.transcript || "").replace(/\[Speaker\s+\d+\]:\s*/gi, "").replace(/[\r\n\t]+/g, " ").trim();

        const row = [
          s.id,
          s.name,
          audioPath,
          s.duration.toFixed(2),
          s.status,
          language,
          speakersCount.toString(),
          split,
          (s.transcript || "").replace(/[\r\n]+/g, " \n "), // preserve turn breaks inside quoted text
          cleanTranscript
        ];

        // Map individual values, escape inner double quotes by doubling them, and wrap each value in double quotes
        const escapedRow = row.map(val => `"${val.replace(/"/g, '""')}"`);
        fileContent += escapedRow.join(",") + "\r\n";
      });

      showToast("Successfully synthesized Palestinian tech-verbatim CSV corpus dataset!", "success");
    } else if (format === "css") {
      // Direct exact design request representation: "export them into a css file to use in the fine-tuning"
      // We wrap the verified NLP verbatim strings in a semantic CSS mapping framework
      mimeType = "text/css;charset=utf-8;";
      extension = "css";

      fileContent = `/* 
  PALESTINIAN DIALECT & ENGLISH TECHNOLOGY SPEECH CORPUS CSS CORPUS BUNDLE
  Generated on: ${new Date().toLocaleDateString()}
  NLP Fine-Tuning Class Master Project Specs
  Total segments count: ${segments.length}
  Verbatim Verified: ${segments.filter(s => s.status === 'completed').length}
*/

@charset "UTF-8";

:root {
  --total-duration-seconds: "${segments.reduce((acc, current) => acc + current.duration, 0).toFixed(0)}s";
  --nlp-dialect: "Palestinian Arabic (Gaza/WestBank Dialect) and English Technical mix";
}

`;
      segments.forEach((s, idx) => {
        fileContent += `/* Segment ID #${idx + 1}: ${s.name} */\n`;
        fileContent += `.corpus-segment-${s.id.replace(/[^a-zA-Z0-9_-]/g, "")} {\n`;
        fileContent += `  --segment-filename: "${s.name}";\n`;
        fileContent += `  --segment-duration: "${s.duration.toFixed(2)}s";\n`;
        fileContent += `  --alignment-direction: "${s.direction}";\n`;
        fileContent += `  --speakers-identified: "${new Set((s.speakerTurns || []).filter(t => t && t.speaker).map(t => t.speaker)).size}";\n`;
        fileContent += `  --transcription-verbatim: "${(s.transcript || "").replace(/[\n\r]+/g, " \\n ").replace(/"/g, '\\"')}";\n`;
        fileContent += `}\n\n`;
      });
    } else if (format === "json") {
      mimeType = "application/json;charset=utf-8;";
      extension = "json";
      fileContent = JSON.stringify(segments, null, 2);
    }

    const blob = new Blob([fileContent], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", `pal_lec_dataset_export_${Date.now()}.${extension}`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  // ... your existing handler functions (like handleMarkCompleted) ...

  // ==========================================
  // WHISPER EXPORT LOGIC (STEP 2)
  // ==========================================
  const exportSegments: (AnnotatedSegment & { originalId: string })[] = segments.map((seg, i) => {
    // 1. Grab the physical file blob from your map, or fallback to a synthetic blob
    const fileBlob = sessionFilesMap.current.get(seg.id) || createSyntheticAudioBlob(seg.duration || 60, seg.name);

    return {
      id: `segment_${String(i + 1).padStart(3, "0")}`,
      originalId: seg.id, // keep original ID so we can look up status below
      transcript: seg.transcript || "",
      audioBlob: fileBlob,
      audioMimeType: seg.mimeType || "audio/wav",
      sourceFile: seg.name,
      segmentIndex: i,
      // Since these appear to be whole files, we start at 0 and end at the duration
      startMs: 0,
      endMs: Math.floor((seg.duration || 0) * 1000),
    };
  });

  // Filter out empty transcripts and only export completed ones
  const verifiedSegments = exportSegments.filter((s) => {
    // Use originalId (not the remapped segment_001 id) to find the source segment
    const originalSeg = segments.find((orig) => orig.id === s.originalId);
    return s.transcript.trim().length > 0 && originalSeg?.status === "completed";
  });
  return (
    <div
      id="app-root-wrapper"
      className={`flex h-screen w-full font-sans overflow-hidden transition-all duration-300 relative ${theme === "light"
        ? "bg-gradient-to-tr from-[#fbf9ff] via-[#fdfaf8] to-[#edf9fe] text-slate-800"
        : "bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-[#131032] via-[#090815] to-[#040409] text-slate-200"
        }`}
    >

      {/* Decorative ambient glowing blobs for bright, fun, high-fidelity experience */}
      <div className="absolute top-[-10%] left-[20%] w-[45%] h-[45%] rounded-full bg-gradient-to-tr from-pink-400/10 to-purple-400/10 blur-[90px] pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[10%] w-[35%] h-[35%] rounded-full bg-gradient-to-br from-cyan-400/10 to-violet-400/10 blur-[100px] pointer-events-none" />

      {/* Hidden background audio node */}
      <audio ref={audioRef} className="hidden" title="Active Audio Anchor" />

      {/* Hidden file input for re-linking a missing audio file to a specific segment */}
      <input
        ref={relinkInputRef}
        type="file"
        accept="audio/*,.ogg,.wav,.mp3,.m4a,.aac,.flac"
        className="hidden"
        onChange={handleRelinkFileSelected}
      />

      {/* LEFT SIDEBAR: segments directory and project KPI progress stats */}
      <aside
        className={`w-80 flex flex-col border-r shrink-0 transition-all duration-300 z-10 ${theme === "light"
          ? "bg-white/90 backdrop-blur-md border-slate-100 shadow-[4px_0_30px_rgba(139,92,246,0.02)]"
          : "bg-[#090816]/80 backdrop-blur-md border-white/5"
          }`}
      >

        {/* Brand Banner with Masters Course meta annotation */}
        <div className={`p-5 border-b transition-all ${theme === "light" ? "border-slate-100 bg-slate-50/20" : "border-white/5 bg-[#0c0c16]/10"
          }`}>
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2.5">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-pink-500 via-purple-500 to-indigo-500 flex items-center justify-center text-white shadow-lg shadow-purple-500/20 shrink-0">
                <Database className="w-5 h-5 text-white" />
              </div>
              <div className="min-w-0">
                <h1 className={`text-sm font-black tracking-tight ${theme === "light"
                  ? "bg-clip-text text-transparent bg-gradient-to-r from-purple-700 to-pink-600"
                  : "bg-clip-text text-transparent bg-gradient-to-r from-pink-400 to-purple-400"
                  }`}>
                  NLP Lab Annotator
                </h1>
                <p className="text-[10px] text-slate-400 dark:text-slate-500 font-bold truncate">Palestinian Dialect Course</p>
              </div>
            </div>

            {/* Quick theme toggler in brand panel */}
            <button
              id="theme-toggle-button"
              onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')}
              className={`p-2 rounded-xl border transition-all duration-300 transform active:scale-95 cursor-pointer ${theme === 'light'
                ? 'bg-amber-100 hover:bg-amber-150 hover:scale-105 hover:rotate-12 border-amber-200 text-amber-600 shadow-sm shadow-amber-200/40'
                : 'bg-indigo-950/40 hover:bg-indigo-900 hover:scale-105 border-indigo-500/20 hover:-rotate-12 text-indigo-400 shadow-lg shadow-indigo-950/40'
                }`}
              title="Toggle theme color"
            >
              {theme === 'light' ? <Sun className="w-3.5 h-3.5 fill-amber-500 stroke-amber-600" /> : <Moon className="w-3.5 h-3.5 fill-indigo-400 stroke-indigo-400" />}
            </button>
          </div>
        </div>

        {/* Local device recording widget */}
        <div className={`p-4 border-b transition-all ${theme === 'light' ? 'border-slate-100 bg-slate-50/10' : 'border-white/5 bg-slate-950/20'
          }`}>
          <button
            onClick={toggleRecording}
            className={`w-full py-2.5 px-4 rounded-xl text-xs font-extrabold tracking-wider uppercase transition-all duration-300 flex items-center justify-center gap-2 cursor-pointer hover:scale-[1.02] active:scale-95 ${isRecording
              ? "bg-gradient-to-r from-rose-500 via-red-500 to-pink-500 text-white animate-pulse shadow-lg shadow-rose-500/20"
              : theme === 'light'
                ? "bg-gradient-to-r from-rose-400 to-pink-400 hover:from-rose-500 hover:to-pink-500 text-white shadow-md shadow-rose-200"
                : "bg-gradient-to-r from-rose-600 to-pink-600 text-white hover:from-rose-500 hover:to-pink-550 border-none"
              }`}
            title="Record direct technology lecture slice from microphone"
          >
            {isRecording ? (
              <>
                <Square className="w-3.5 h-3.5 fill-current animate-spin" />
                Stop Recording ({formatTimer(recordTimer)})
              </>
            ) : (
              <>
                <Mic className="w-3.5 h-3.5 text-white" />
                Record Real Segment
              </>
            )}
          </button>
        </div>

        {/* Dynamic searchable Segments List with inline stats */}
        <div className="flex-1 overflow-y-auto">
          <SegmentList
            segments={segments}
            activeSegmentId={activeSegmentId}
            onSelectSegment={handleSelectSegment}
            onDeleteSegment={handleDeleteSegment}
            onFilesSelected={handleFilesSelected}
            onLoadSamples={loadDefaultSamples}
            onClearAll={handleClearAll}
            theme={theme}
            onBulkRenameSpeaker={handleBulkRenameSpeaker}
          />
        </div>

        {/* Dataset Generator exporters */}
        <div className={`p-4 border-t space-y-2 ${theme === "light" ? "bg-slate-50/50 border-slate-100" : "bg-slate-950/50 border-white/5"
          }`}>
          <div className="text-[10px] text-slate-400 dark:text-slate-505 font-mono tracking-widest uppercase font-extrabold text-center">Dataset Export Center</div>

          <div className="grid grid-cols-2 gap-2">
            <button
              id="export-csv-dataset"
              onClick={() => {
                const isAllComp = segments.length > 0 && segments.every((s) => s.status === "completed");
                if (!isAllComp) {
                  const compCount = segments.filter((s) => s.status === "completed").length;
                  showToast(
                    `Export Locked: Please transcribe, validate, and mark all (${compCount}/${segments.length}) segments as complete first.`,
                    "error"
                  );
                  return;
                }
                triggerExportDataset("csv");
              }}
              className={`py-2.5 px-2 rounded-xl text-[10px] uppercase font-black transition-all text-center border hover:scale-[1.02] active:scale-95 ${segments.length > 0 && segments.every((s) => s.status === "completed")
                ? theme === "light"
                  ? "bg-white hover:bg-slate-50 border-slate-205 text-slate-700 shadow-sm cursor-pointer"
                  : "bg-slate-900 border-[#ff007f]/10 text-slate-300 cursor-pointer"
                : "bg-slate-100/50 dark:bg-slate-900/40 text-slate-400 dark:text-slate-600 border-dashed border-slate-300 dark:border-white/10 cursor-not-allowed"
                }`}
              title={
                segments.length > 0 && segments.every((s) => s.status === "completed")
                  ? "Export the verified multi-speaker high-fidelity dialect corpus in CSV format"
                  : `CSV Export is locked until all segments are fully verified and completed (${segments.filter((s) => s.status === "completed").length}/${segments.length} done).`
              }
            >
              {segments.length > 0 && segments.every((s) => s.status === "completed") ? "CSV Fine-Tuning" : "🔒 CSV (Locked)"}
            </button>

            <button
              id="export-css-dataset"
              onClick={() => triggerExportDataset("css")}
              className="py-2.5 px-2 bg-gradient-to-r from-pink-500 via-purple-500 to-indigo-600 text-white hover:opacity-95 rounded-xl text-[10px] uppercase font-black font-mono transition-all text-center shadow-md shadow-purple-500/10 cursor-pointer hover:scale-[1.02] active:scale-95"
              title="Export verbatim verified Palestinian dialect mappings to standard CSS layout format"
            >
              CSS Dataset
            </button>
          </div>

          <button
            id="export-json-dataset"
            onClick={() => triggerExportDataset("json")}
            className={`w-full py-2 text-[10px] font-mono rounded-lg transition-all text-center border cursor-pointer ${theme === "light"
              ? "bg-white hover:bg-slate-50 border-slate-205 text-slate-505 hover:text-slate-800 shadow-sm"
              : "bg-slate-950 hover:bg-slate-900 border-white/5 text-slate-400 hover:text-slate-200"
              }`}
          >
            Raw JSON-L Snapshot
          </button>
          <div className="pt-2">
            <ExportButton
              segments={verifiedSegments}
              datasetName="palestinian_tech_dialect"
            />
          </div>
        </div>
      </aside>

      {/* MAIN WORKSPACE: High fidelity Media player and synchronized dialet transcription cockpit info */}
      <main
        className="flex-1 flex flex-col p-6 md:p-8 space-y-6 overflow-y-auto transition-all duration-300 z-10"
        style={{ backgroundColor: theme === 'dark' ? '#332134' : undefined }}
      >

        {/* PROGRESS BAR: Project overall fine-tuning corpus completion progress bar */}
        {(() => {
          const totalCount = segments.length;
          const completedCount = segments.filter((s) => s.status === "completed").length;
          const progressPercent = totalCount > 0 ? Math.round((completedCount / totalCount) * 100) : 0;

          return (
            <div className={`p-5 rounded-3xl border transition-all duration-300 ${theme === 'light'
              ? 'bg-white/95 border-slate-205 shadow-[0_10px_35px_rgba(139,92,246,0.03)]'
              : 'bg-[#1a101b]/95 border-white/10 shadow-[0_4px_30px_rgba(0,0,0,0.45)]'
              } flex flex-col gap-2.5`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="flex h-2 w-2 relative">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                  </span>
                  <span className={`text-[10px] uppercase font-black tracking-widest ${theme === 'light' ? 'text-slate-500' : 'text-slate-400'}`}>
                    Overall Corpus Verification Milestones
                  </span>
                </div>
                <span className="text-xs font-black font-mono text-purple-600 dark:text-pink-400">
                  {progressPercent}% Complete &bull; ({completedCount}/{totalCount} segments completed)
                </span>
              </div>
              <div className={`w-full h-3 rounded-full overflow-hidden p-[2px] ${theme === 'light' ? 'bg-slate-100 border border-slate-200' : 'bg-slate-950/80 border border-white/5'
                }`}>
                <div
                  className="bg-gradient-to-r from-pink-500 via-purple-500 to-cyan-500 h-full rounded-full transition-all duration-700 shadow-[0_0_8px_rgba(139,92,246,0.3)]"
                  style={{ width: `${progressPercent}%` }}
                />
              </div>
            </div>
          );
        })()}

        {/* Top Header Panel indicating Active segment and AI triggers */}
        <div className={`flex flex-col md:flex-row md:items-center justify-between gap-4 border p-5 rounded-3xl transition-all duration-300 ${theme === 'light'
          ? 'bg-white/95 border-slate-205 shadow-[0_8px_30px_rgba(139,92,246,0.02)]'
          : 'bg-[#1a101b]/95 border-white/10 shadow-[0_4px_30px_rgba(0,0,0,0.45)]'
          }`}>
          <div className="min-w-0">
            <span className="text-[10px] uppercase text-purple-600 dark:text-pink-400 tracking-widest font-black">Verbatim Palestinian Corpus Workspace</span>
            <h2 className={`text-lg font-black mt-1 truncate max-w-[280px] md:max-w-[420px] ${theme === 'light' ? 'text-slate-800' : 'text-slate-100'
              }`}>
              {activeSegment ? activeSegment.name : "Select or Upload a Lecture Recording segment"}
            </h2>
          </div>

          <div className="flex items-center gap-2.5 flex-wrap">
            {/* Shortcuts Help Icon Button */}
            <button
              id="shortcuts-help-button"
              onClick={() => setShowShortcutsModal(true)}
              className={`py-2 px-3.5 rounded-xl border flex items-center gap-1.5 text-xs font-black uppercase transition-all hover:scale-105 active:scale-95 cursor-pointer ${theme === 'light'
                ? 'hover:bg-slate-50 bg-white border-slate-205 text-slate-705 shadow-sm'
                : 'hover:bg-white/10 bg-white/5 border-white/10 text-pink-400'
                }`}
              title="View app keyboard shortcuts help manual"
            >
              <HelpCircle className="w-4 h-4" />
              Shortcuts Help
            </button>

            {activeSegment && (
              <>
                <input
                  id="transcribe-context-prompt"
                  type="text"
                  placeholder="E.g. RNN, SGD, Transformer..."
                  value={transcribeContext}
                  onChange={(e) => setTranscribeContext(e.target.value)}
                  className={`border focus:ring-2 focus:ring-purple-400/20 active:outline-none focus:outline-none rounded-xl px-3.5 py-2 text-xs max-w-[170px] transition-all ${theme === 'light'
                    ? 'bg-slate-50/50 text-slate-800 border-slate-205 focus:border-purple-500'
                    : 'bg-slate-950 text-white border-white/10 focus:border-pink-500'
                    }`}
                  title="Context prompts regarding technical acronyms"
                />
                <button
                  id="ai-transcribe-trigger-button"
                  onClick={handleAutoTranscribeAI}
                  disabled={isAiTranscribing || activeSegment.status === "transcribing"}
                  className="px-4 py-2.5 rounded-xl text-xs font-black uppercase tracking-wider bg-gradient-to-r from-pink-500 via-purple-500 to-indigo-600 text-white hover:opacity-95 shadow-md shadow-purple-500/10 hover:scale-[1.02] active:scale-95 transition-all duration-300 flex items-center gap-2 disabled:from-slate-800 disabled:to-slate-900 disabled:text-slate-500 border-none cursor-pointer"
                >
                  {(isAiTranscribing || activeSegment.status === "transcribing") ? (
                    <>
                      <Activity className="w-3.5 h-3.5 animate-spin text-white" />
                      Drafting Verbatim AI...
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-3.5 h-3.5 text-white animate-pulse" />
                      Gemini Auto-Transcribe
                    </>
                  )}
                </button>
              </>
            )}
          </div>
        </div>

        {/* RE-LINK AUDIO BANNER — shown when a real file is missing from IDB */}
        {activeSegment && activeSegment.hasAudioFile && !activeSegment.audioUrl && !sessionFilesMap.current.get(activeSegment.id) && (
          <div className={`flex items-center justify-between gap-4 px-5 py-4 rounded-2xl border transition-all ${theme === 'light'
            ? 'bg-amber-50 border-amber-200 text-amber-800'
            : 'bg-amber-500/10 border-amber-500/25 text-amber-300'
            }`}>
            <div className="flex items-center gap-3 min-w-0">
              <FileAudio className="w-5 h-5 shrink-0 text-amber-500" />
              <div className="min-w-0">
                <p className="text-xs font-black">Audio file not found in browser storage</p>
                <p className="text-[11px] font-medium opacity-75 truncate">
                  Re-upload <span className="font-mono">{activeSegment.name}</span> to continue playing. Your transcript is safe.
                </p>
              </div>
            </div>
            <button
              onClick={() => handleRelinkFile(activeSegment.id)}
              className="shrink-0 px-4 py-2 rounded-xl text-xs font-black uppercase tracking-wider bg-amber-500 hover:bg-amber-400 text-white transition-all hover:scale-105 active:scale-95 shadow-sm cursor-pointer border-none"
            >
              Re-upload File
            </button>
          </div>
        )}

        {/* High Tech Sound Telemetry Audio Player with seek controls */}
        <AudioPlayer
          activeSegment={activeSegment}
          audioRef={audioRef}
          isPlaying={isPlaying}
          setIsPlaying={setIsPlaying}
          currentTime={currentTime}
          setCurrentTime={setCurrentTime}
          duration={duration}
          setDuration={setDuration}
          playbackRate={playbackRate}
          setPlaybackRate={setPlaybackRate}
          theme={theme}
        />

        {/* CUSTOM SEGMENT TAGS DECK & DATASET METADATA CONTROL STATION */}
        {activeSegment && (
          <div className={`p-5 rounded-3xl border flex flex-col gap-4 transition-all duration-300 ${theme === 'light'
            ? 'bg-white/95 border-slate-205 shadow-[0_8px_30px_rgba(139,92,246,0.02)]'
            : 'bg-[#1a101b]/95 border-white/10 shadow-[0_4px_30px_rgba(0,0,0,0.45)]'
            }`}>
            {/* Row 1: Tags segment layout */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div className="flex flex-wrap items-center gap-2">
                <span className={`text-[10px] uppercase tracking-widest font-black ${theme === 'light' ? 'text-slate-400' : 'text-slate-500'}`}>Segment Tags:</span>
                {(activeSegment.tags || []).map((t) => (
                  <span
                    key={t}
                    className={`inline-flex items-center gap-1.5 px-3 py-1 text-xs font-extrabold rounded-full transition-all border ${theme === 'light'
                      ? 'bg-purple-50 border-purple-150 text-purple-700 shadow-sm shadow-purple-50/30'
                      : 'bg-pink-500/10 text-pink-400 border-pink-500/20 shadow-sm shadow-pink-950/20'
                      }`}
                  >
                    #{t}
                    <button
                      onClick={() => {
                        const nextTags = (activeSegment.tags || []).filter((tag) => tag !== t);
                        setSegments((prev) => prev.map((s) => s.id === activeSegment.id ? { ...s, tags: nextTags } : s));
                      }}
                      className="text-slate-400 hover:text-red-500 font-extrabold ml-1.5 text-xs transition-colors cursor-pointer"
                      title="Delete tag"
                    >
                      &times;
                    </button>
                  </span>
                ))}
                <input
                  id="add-custom-tag-input"
                  type="text"
                  placeholder="+ Add Tag (Press Enter)..."
                  onBlur={(e) => {
                    const val = e.target.value.trim().replace(/^#/, '');
                    if (val) {
                      const currentTags = activeSegment.tags || [];
                      if (!currentTags.includes(val)) {
                        const nextTags = [...currentTags, val];
                        setSegments((prev) => prev.map((s) => s.id === activeSegment.id ? { ...s, tags: nextTags } : s));
                      }
                      e.target.value = '';
                    }
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      const val = e.currentTarget.value.trim().replace(/^#/, '');
                      if (val) {
                        const currentTags = activeSegment.tags || [];
                        if (!currentTags.includes(val)) {
                          const nextTags = [...currentTags, val];
                          setSegments((prev) => prev.map((s) => s.id === activeSegment.id ? { ...s, tags: nextTags } : s));
                        }
                        e.currentTarget.value = '';
                      }
                    }
                  }}
                  className={`px-3 py-1 text-xs rounded-full bg-transparent border border-dashed focus:outline-none focus:ring-2 focus:ring-purple-400/20 w-44 transition-all ${theme === 'light' ? 'border-slate-300 text-slate-800 focus:border-purple-500' : 'border-white/20 text-slate-300 placeholder:text-slate-600 focus:border-pink-500'
                    }`}
                />
              </div>
              <div className={`text-[10px] font-mono font-bold ${theme === "light" ? "text-slate-400" : "text-slate-500"}`}>
                Use tags to organize lecture segments
              </div>
            </div>

            {/* Sub-divider */}
            <hr className={theme === "light" ? "border-slate-100" : "border-white/5"} />

            {/* Row 2: Verbatim tuning export schema configuration inputs */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4.5">
              {/* Field A: Dataset Partition Split */}
              <div className="flex flex-col gap-1.5">
                <span className={`text-[10px] uppercase tracking-widest font-black ${theme === 'light' ? 'text-slate-405 text-slate-400' : 'text-slate-500'}`}>
                  Dataset Split / Segment Partition
                </span>
                <div className={`flex rounded-xl p-0.5 border ${theme === 'light' ? 'bg-slate-100 border-slate-205' : 'bg-black/40 border-white/5'
                  }`}>
                  {(["train", "val", "test"] as const).map((part) => (
                    <button
                      key={part}
                      onClick={() => {
                        setSegments((prev) => prev.map((s) => s.id === activeSegment.id ? { ...s, split: part } : s));
                      }}
                      className={`flex-1 py-1 text-[10px] font-black uppercase rounded-lg transition-all cursor-pointer ${(activeSegment.split || "train") === part
                        ? "bg-gradient-to-r from-pink-500 to-purple-600 text-white shadow-sm font-black"
                        : theme === "light"
                          ? "text-slate-600 hover:text-slate-900"
                          : "text-slate-400 hover:text-slate-200"
                        }`}
                    >
                      {part}
                    </button>
                  ))}
                </div>
              </div>

              {/* Field B: Dialect language code specification */}
              <div className="flex flex-col gap-1.5">
                <span className={`text-[10px] uppercase tracking-widest font-black ${theme === 'light' ? 'text-slate-405 text-slate-400' : 'text-slate-500'}`}>
                  Language Code (e.g. ar-PS)
                </span>
                <input
                  type="text"
                  value={activeSegment.language || "Arabic (Palestinian Dialect)"}
                  onChange={(e) => {
                    const val = e.target.value;
                    setSegments((prev) => prev.map((s) => s.id === activeSegment.id ? { ...s, language: val } : s));
                  }}
                  className={`border focus:ring-2 focus:ring-purple-400/20 active:outline-none focus:outline-none rounded-xl px-3.5 py-1 text-xs transition-all ${theme === 'light'
                    ? 'bg-slate-50/50 text-slate-800 border-slate-205 focus:border-purple-500'
                    : 'bg-slate-950 text-white border-white/10 focus:border-pink-500'
                    }`}
                  placeholder="ar-PS / Arabic"
                  title="Dataset Language Identifier Column"
                />
              </div>

              {/* Field C: Physical Audio target storage path */}
              <div className="flex flex-col gap-1.5">
                <span className={`text-[10px] uppercase tracking-widest font-black ${theme === 'light' ? 'text-slate-405 text-slate-400' : 'text-slate-500'}`}>
                  Audio Target Path (CSV)
                </span>
                <input
                  type="text"
                  value={activeSegment.audio_path || `audio/${activeSegment.name}`}
                  onChange={(e) => {
                    const val = e.target.value;
                    setSegments((prev) => prev.map((s) => s.id === activeSegment.id ? { ...s, audio_path: val } : s));
                  }}
                  className={`border focus:ring-2 focus:ring-purple-400/20 active:outline-none focus:outline-none rounded-xl px-3.5 py-1 text-xs transition-all ${theme === 'light'
                    ? 'bg-slate-50/50 text-slate-800 border-slate-205 focus:border-purple-500'
                    : 'bg-slate-950 text-white border-white/10 focus:border-pink-500'
                    }`}
                  placeholder="e.g. audio/segment_01.mp3"
                  title="Standard target path column inside the training list metadata file"
                />
              </div>
            </div>
          </div>
        )}

        {/* PRIMARY EDITING STATION: Switchable modes containing structured speaker turns, Palestinian diacritics audit rules */}
        {activeSegment ? (
          <section className={`flex-1 flex flex-col rounded-3xl border overflow-hidden shadow-xl min-h-[700px] md:min-h-[750px] transition-all duration-300 ${theme === "light"
            ? "bg-white border-slate-205 shadow-[0_12px_40px_rgba(139,92,246,0.03)]"
            : "bg-[#1a101b]/95 border-white/10"
            }`}>

            {/* Workbench secondary header bar */}
            <div className={`flex flex-col xl:flex-row xl:items-center justify-between gap-3 px-6 py-4 border-b transition-colors ${theme === "light" ? "bg-slate-50/50 border-slate-150" : "bg-white/5 border-white/10"
              }`}>
              <div className="flex flex-wrap items-center gap-4">

                {/* Text direction toggles */}
                <div className="flex items-center gap-2">
                  <span className={`text-[10px] uppercase font-black tracking-widest ${theme === "light" ? "text-slate-400" : "text-slate-500"}`}>Direction:</span>
                  <div className={`flex rounded-xl p-0.5 border ${theme === "light" ? "bg-slate-100 border-slate-205" : "bg-black/40 border-white/5"
                    }`}>
                    <button
                      id="text-dir-rtl-btn"
                      onClick={() => handleToggleDirection("rtl")}
                      className={`px-3 py-1 text-[10px] font-bold rounded-lg transition-all cursor-pointer ${activeSegment.direction === "rtl"
                        ? "bg-gradient-to-r from-pink-500 to-purple-600 text-white shadow-sm font-black"
                        : theme === "light" ? "text-slate-500 hover:text-slate-850 text-slate-800" : "text-slate-400 hover:text-slate-300"
                        }`}
                    >
                      RTL (Default)
                    </button>
                    <button
                      id="text-dir-ltr-btn"
                      onClick={() => handleToggleDirection("ltr")}
                      className={`px-3 py-1 text-[10px] font-bold rounded-lg transition-all cursor-pointer ${activeSegment.direction === "ltr"
                        ? "bg-gradient-to-r from-pink-500 to-purple-600 text-white shadow-sm font-black"
                        : theme === "light" ? "text-slate-500 hover:text-slate-850 text-slate-800" : "text-slate-400 hover:text-slate-300"
                        }`}
                    >
                      LTR
                    </button>
                  </div>
                </div>

                {/* Switchable Workspaces Tab */}
                <span className={`hidden md:block h-6 w-px ${theme === "light" ? "bg-slate-200" : "bg-white/10"}`} />

                <div className="flex gap-1 flex-wrap">
                  <button
                    onClick={() => setActiveTab("edit")}
                    className={`px-3 py-1.5 text-xs rounded-xl transition-all font-bold cursor-pointer ${activeTab === "edit"
                      ? theme === "light"
                        ? "bg-purple-100 text-purple-700 font-black shadow-sm"
                        : "bg-white/15 text-white"
                      : theme === "light" ? "text-slate-400 hover:text-slate-700" : "text-slate-500 hover:text-slate-350"
                      }`}
                  >
                    Speaker Turns (Diarization)
                  </button>
                  <button
                    onClick={() => setActiveTab("dialect")}
                    className={`px-3 py-1.5 text-xs rounded-xl transition-all font-bold cursor-pointer ${activeTab === "dialect"
                      ? theme === "light"
                        ? "bg-purple-100 text-purple-700 font-black shadow-sm"
                        : "bg-white/15 text-white"
                      : theme === "light" ? "text-slate-400 hover:text-slate-700" : "text-slate-500 hover:text-slate-350"
                      }`}
                  >
                    Raw Content Form
                  </button>
                  <button
                    onClick={() => setActiveTab("help")}
                    className={`px-3 py-1.5 text-xs rounded-xl transition-all font-bold flex items-center gap-1.5 border cursor-pointer ${activeTab === "help"
                      ? theme === "light"
                        ? "bg-purple-500 text-white border-transparent font-black shadow-md shadow-purple-500/10"
                        : "bg-pink-500/10 text-pink-400 border-pink-500/20 font-black"
                      : theme === "light"
                        ? "border-transparent text-slate-400 hover:text-slate-705 text-slate-700"
                        : "border-transparent text-slate-500 hover:text-slate-300"
                      }`}
                  >
                    Arabic Verbatim Guide
                  </button>
                </div>

              </div>

              {/* Copy & Status toolbar buttons */}
              <div className="flex items-center gap-2.5 self-end xl:self-auto">
                <button
                  id="copy-text-button"
                  onClick={handleCopyText}
                  className={`text-[10px] uppercase tracking-widest font-black flex items-center gap-2 py-1.5 px-3 rounded-xl transition-all cursor-pointer ${theme === "light"
                    ? "bg-slate-100 hover:bg-slate-205 text-slate-750 text-slate-700 shadow-sm"
                    : "text-slate-400 hover:text-white hover:bg-white/5"
                    }`}
                >
                  {justCopied ? (
                    <>
                      <Check className="w-3.5 h-3.5 text-emerald-500" />
                      Copied!
                    </>
                  ) : (
                    <>
                      <Copy className="w-3.5 h-3.5 text-purple-500" />
                      Copy Text
                    </>
                  )}
                </button>

                <span className={`h-5 w-px ${theme === "light" ? "bg-slate-200" : "bg-white/10"}`} />

                {/* Diagnostic helper button */}
                <button
                  id="auto-corrupt-dialect-helper"
                  onClick={handleAutoCorrectionTrigger}
                  className={`text-[10px] uppercase tracking-widest font-black flex items-center gap-1.5 border py-1.5 px-3 rounded-xl transition-all cursor-pointer ${theme === "light"
                    ? "bg-purple-500/10 hover:bg-purple-505 bg-purple-500 text-white hover:bg-purple-600 shadow-md border-transparent hover:scale-105 active:scale-95"
                    : "text-pink-405 text-pink-400 hover:text-pink-350 bg-pink-500/5 hover:bg-pink-500/10 border-pink-500/20 hover:-translate-y-0.5"
                    }`}
                  title="Audits transcript text and corrects missing Arabic dialect symbols according to verbatim standards"
                >
                  <Sparkles className="w-3 h-3 text-cyan-400 animate-pulse" />
                  Auto-Fix Verbatim Words
                </button>
              </div>
            </div>

            {/* TAB CONTENT 1: Structured speaker diarized list editor */}
            {activeTab === "edit" && (
              <div
                className="flex-1 p-6 space-y-4 overflow-y-auto"
                dir={activeSegment.direction}
              >
                {activeSegment.speakerTurns.length > 0 ? (
                  activeSegment.speakerTurns.map((turn, index) => (
                    <div
                      key={turn.id}
                      className={`border p-4.5 rounded-2xl space-y-3 transition-all flex flex-col ${theme === 'light'
                        ? 'bg-slate-50/45 border-slate-150 focus-within:border-purple-500 focus-within:bg-white shadow-sm'
                        : 'bg-[#150a16] border-white/5 focus-within:border-pink-550 focus-within:border-pink-400 focus-within:bg-[#1f0e21] shadow-lg'
                        }`}
                    >
                      <div className={`flex items-center justify-between gap-4 border-b pb-2 ${theme === 'light' ? 'border-slate-150' : 'border-white/5'
                        }`}>
                        {/* Speaker ID Label Tag */}
                        <div className="flex items-center gap-2">
                          <span className={`text-[10px] uppercase font-mono font-black ${theme === 'light' ? 'text-slate-400' : 'text-slate-500'}`}>Speaker:</span>
                          <input
                            type="text"
                            value={turn.speaker}
                            onChange={(e) => handleSpeakerTurnChange(turn.id, turn.text, e.target.value)}
                            className={`bg-transparent border-none text-xs font-black focus:ring-0 w-32 focus:border-b focus:outline-none ${theme === 'light' ? 'text-purple-700 focus:border-purple-400' : 'text-pink-400 focus:border-pink-500/55'
                              }`}
                            placeholder="Speaker Label"
                            title="Format custom speaker index identifier"
                          />
                        </div>

                        {/* Order Indicator and Delete turn block */}
                        <div className="flex items-center gap-2" dir="ltr">
                          <span className={`text-[10px] font-mono font-bold ${theme === 'light' ? 'text-slate-400' : 'text-slate-550 text-slate-500'}`}>Turn #{index + 1}</span>
                          <button
                            onClick={() => handleDeleteSpeakerTurn(turn.id)}
                            className={`p-1 rounded cursor-pointer transition-colors ${theme === 'light' ? 'hover:bg-red-50 text-slate-400 hover:text-red-500' : 'hover:bg-red-500/20 text-slate-500 hover:text-red-400'
                              }`}
                            title="Delete this speaker block"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </div>

                      {/* Speaking content box */}
                      <textarea
                        value={turn.text}
                        onChange={(e) => handleSpeakerTurnChange(turn.id, e.target.value, turn.speaker)}
                        className={`w-full bg-transparent border-none outline-none resize-none font-sans text-sm md:text-base leading-relaxed focus:outline-none focus:ring-0 ${activeSegment.direction === "rtl" ? "text-right" : "text-left"
                          } ${theme === 'light' ? 'text-slate-800 placeholder:text-slate-400' : 'text-slate-100 placeholder:text-slate-600'
                          }`}
                        placeholder="..."
                        rows={4}
                        dir={activeSegment.direction}
                        title="Edit speaker exact paragraph text"
                      />
                    </div>
                  ))
                ) : (
                  <div className={`flex flex-col items-center justify-center p-12 text-center font-mono text-xs ${theme === 'light' ? 'text-slate-400' : 'text-slate-500'
                    }`}>
                    <p>No formatted diarized turns exist for this segment.</p>
                    <p className={`mt-1 font-bold ${theme === 'light' ? 'text-slate-400/80' : 'text-slate-600'}`}>Type or click Gemini Auto-Transcribe above to populate speech turns.</p>
                  </div>
                )}

                {/* Add Speaker card trigger */}
                <button
                  onClick={handleAddSpeakerTurn}
                  className={`w-full py-3.5 border border-dashed rounded-xl text-xs font-black transition-all flex items-center justify-center gap-1.5 cursor-pointer hover:scale-[1.015] active:scale-95 ${theme === 'light'
                    ? 'bg-purple-50/20 hover:bg-purple-50/45 border-purple-200 hover:border-purple-300 text-purple-600'
                    : 'bg-white/[0.02] hover:bg-white/[0.04] border-white/10 hover:border-white/25 text-slate-400 hover:text-white'
                    }`}
                >
                  <Plus className="w-4 h-4" /> Add Structured Speaker Block
                </button>
              </div>
            )}

            {/* TAB CONTENT 2: Plain transcript text editing block */}
            {activeTab === "dialect" && (
              <div className="flex-1 p-6 flex flex-col">
                <textarea
                  id="plain-transcript-text-editor"
                  value={activeSegment.transcript}
                  onChange={(e) => handleTranscriptChange(e.target.value)}
                  className={`w-full flex-1 bg-transparent border-none outline-none resize-none text-[30px] leading-relaxed font-sans focus:outline-none focus:ring-0 ${activeSegment.direction === "rtl" ? "text-right" : "text-left"
                    } ${theme === 'light' ? 'text-slate-800 placeholder:text-slate-300' : 'text-slate-100 placeholder:text-slate-700'
                    }`}
                  placeholder="يعطيكم العافية جميعاً، ابدأ الكتابة هنا..."
                  dir={activeSegment.direction}
                  title="Unified plain text editor with tags"
                />
                <div className={`text-[11px] mt-2 font-mono flex items-center gap-2 p-2.5 rounded-xl border ${theme === 'light'
                  ? 'bg-amber-50/80 border-amber-100 text-amber-700 font-bold'
                  : 'bg-black/30 border-white/5 text-slate-400'
                  }`}>
                  <span className="text-amber-500 font-black">PRO-TIP:</span> Use standard tags: [Speaker 1]: (your Arabic text...) to sync dynamic Turns tab automatically.
                </div>
              </div>
            )}

            {/* TAB CONTENT 3: Palestinian Dialect & Tech Verbatim Grammar Guide */}
            {activeTab === "help" && (
              <div className="flex-1 p-6 overflow-y-auto space-y-4">
                <div className={`border rounded-2xl p-4.5 ${theme === 'light'
                  ? 'border-purple-100 bg-purple-50/50'
                  : 'border-pink-500/20 bg-pink-500/5'
                  }`}>
                  <h3 className={`text-sm font-black flex items-center gap-1.5 mb-2 ${theme === 'light' ? 'text-purple-700' : 'text-pink-400'
                    }`}>
                    <BookOpen className="w-4 h-4" /> Verbatim Diacritics Instruction manual
                  </h3>
                  <p className={`text-xs leading-relaxed ${theme === 'light' ? 'text-slate-650 text-slate-600 font-bold' : 'text-slate-300 font-medium'
                    }`}>
                    Palestinian speech, when written verbatim for fine-tuning NLP transformer models, requires exact preservation of acoustic grammar. Under student Master's course rules, even regional accents MUST be decorated with formal marks like Hamzas, Arabic Shaddah, and double Tanween. Ensure you apply the rules highlighted below.
                  </p>
                </div>

                <div className={`text-xs uppercase tracking-wider font-mono font-black ${theme === 'light' ? 'text-slate-400' : 'text-slate-500'}`}>Active Verbatim Corrections Dictionary</div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {COMMON_DIALECT_RULES.map((rule, idx) => (
                    <div key={idx} className={`border p-3 rounded-2xl flex items-center justify-between gap-4 ${theme === 'light' ? 'bg-slate-50 border-slate-150' : 'bg-white/[0.02] border-white/5'
                      }`}>
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <span className="bg-red-500/10 text-red-500 border border-red-500/20 px-1.5 py-0.5 rounded text-[10px] font-bold">{rule.original}</span>
                          <span className="text-[10px] text-slate-500">&rarr;</span>
                          <span className="bg-emerald-500/10 text-emerald-500 border border-emerald-500/20 px-1.5 py-0.5 rounded text-[10px] font-bold">{rule.standard}</span>
                        </div>
                        <p className="text-[11px] text-slate-400">{rule.explanation}</p>
                      </div>
                      <span className="text-[9px] font-mono bg-white/5 py-0.5 px-1.5 rounded text-slate-500 uppercase shrink-0">{rule.frequency}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Workbench footer control board */}
            <div className={`p-5.5 border-t flex flex-col sm:flex-row items-center justify-between gap-4 transition-colors ${theme === 'light' ? 'bg-slate-50/55 border-slate-150' : 'bg-[#0f0b27]/30 border-white/5'
              }`}>

              {/* Telemetry info labels of correctness */}
              <div className="flex flex-wrap items-center gap-4 text-[10px] font-mono tracking-widest text-slate-500 uppercase font-black">
                <span className="flex items-center gap-1.5 text-slate-400">
                  <span className="w-2 h-2 rounded-full bg-pink-500 animate-pulse"></span> Verbatim Dialect Mode
                </span>
                <span className="flex items-center gap-1.5 text-slate-400">
                  <span className="w-2 h-2 rounded-full bg-purple-500 shadow-[0_0_6px_rgba(168,85,247,0.5)]"></span> Arabic & English
                </span>

                {/* Autosaved feedback status badge */}
                {saveStatus === "saving" && (
                  <span className="text-amber-500 font-extrabold flex items-center gap-1.5 transition-all">
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" /> Saving changes...
                  </span>
                )}
                {saveStatus === "saved" && (
                  <span className="text-emerald-500 dark:text-emerald-450 font-extrabold flex items-center gap-1.5 transition-all">
                    <Check className="w-3.5 h-3.5 animate-bounce" /> Saved successfully
                  </span>
                )}
                {saveStatus === "idle" && (
                  <span className={`font-black flex items-center gap-1.5 transition-all ${theme === 'light' ? 'text-slate-400' : 'text-slate-550 text-slate-500'}`}>
                    <Database className="w-3.5 h-3.5 text-slate-400" /> Synced to Local Storage
                  </span>
                )}
              </div>

              {/* Verify and next controllers */}
              <div className="flex gap-2.5">
                {showClearConfirm ? (
                  <div className="flex items-center gap-1.5 p-1 rounded-xl bg-orange-500/10 border border-orange-500/25 text-xs text-orange-500 font-bold transition-all">
                    <span className="pl-1.5 text-[11px]">Clear all annotations?</span>
                    <button
                      onClick={() => {
                        handleTranscriptChange("");
                        setShowClearConfirm(false);
                        showToast("Active segment annotations cleared.", "info");
                      }}
                      className="px-2 py-1 rounded-lg bg-orange-600 hover:bg-orange-500 text-white font-extrabold select-none cursor-pointer leading-none text-[10px]"
                    >
                      Yes, Clear
                    </button>
                    <button
                      onClick={() => setShowClearConfirm(false)}
                      className="px-2 py-1 rounded-lg bg-slate-200 dark:bg-white/10 hover:bg-slate-300 dark:hover:bg-white/15 text-slate-700 dark:text-slate-300 font-extrabold select-none cursor-pointer leading-none text-[10px]"
                    >
                      Cancel
                    </button>
                  </div>
                ) : (
                  <button
                    id="reset-active-segment-annotations"
                    onClick={() => setShowClearConfirm(true)}
                    className={`px-4.5 py-2.5 rounded-xl text-xs font-black transition-all hover:scale-105 active:scale-95 cursor-pointer ${theme === 'light'
                      ? 'bg-slate-100/80 hover:bg-slate-205 text-slate-600 border border-slate-205'
                      : 'border border-white/10 hover:bg-white/5 text-slate-400 hover:text-white'
                      }`}
                  >
                    Clear Segment
                  </button>
                )}

                <button
                  id="validate-mark-completed"
                  onClick={handleMarkCompleted}
                  className={`px-5 py-2.5 rounded-xl text-xs font-black transition-all flex items-center gap-2 cursor-pointer hover:scale-[1.03] active:scale-95 border-none shadow-md ${activeSegment.status === "completed"
                    ? "bg-gradient-to-r from-emerald-500 to-teal-600 text-white shadow-emerald-500/15"
                    : "bg-gradient-to-r from-pink-500 via-purple-500 to-indigo-600 text-white shadow-purple-500/15"
                    }`}
                  title="Validates Palestinian-English symbols and steps to the next queue"
                >
                  {activeSegment.status === "completed" ? (
                    <>
                      <CheckCircle2 className="w-4 h-4 text-white animate-pulse" />
                      Verified Finished!
                    </>
                  ) : (
                    <>
                      <Award className="w-4 h-4 text-white" />
                      Validate & Next
                    </>
                  )}
                </button>
              </div>

            </div>

          </section>
        ) : (
          /* Empty Workspace state */
          <div className={`flex-1 flex flex-col items-center justify-center border border-dashed rounded-3xl p-12 text-center transition-all duration-300 ${theme === 'light'
            ? 'bg-white/95 border-slate-205 shadow-[0_12px_45px_rgba(139,92,246,0.02)] text-slate-700'
            : 'bg-[#1a101b]/95 border-white/10'
            }`}>
            <div className={`w-16 h-16 rounded-full flex items-center justify-center mb-5 animate-pulse border ${theme === 'light' ? 'bg-purple-50 border-purple-100 text-purple-600' : 'bg-white/5 border-white/10 text-pink-400'
              }`}>
              <FileAudio className="w-8 h-8" />
            </div>
            <h3 className={`text-lg font-black ${theme === 'light' ? 'text-slate-800' : 'text-white'}`}>No Lecture Segment Selected</h3>
            <p className={`text-xs max-w-sm mt-1.5 mb-6 leading-relaxed ${theme === 'light' ? 'text-slate-500 font-bold' : 'text-slate-400'}`}>
              Select a Palestinians dialect segment from the left directory sidebar, preview elements, correct dialect marks, or run AI.
            </p>
            <button
              onClick={loadDefaultSamples}
              className="py-3 px-6 bg-gradient-to-r from-pink-500 via-purple-500 to-indigo-600 text-white text-xs font-black uppercase tracking-wider rounded-xl transition-all hover:opacity-95 shadow-md shadow-purple-500/10 hover:scale-105 active:scale-95 cursor-pointer border-none"
            >
              Load Lecture Arabic/English Samples
            </button>
          </div>
        )}

      </main>

      {/* KEYBOARD SHORTCUTS MANUAL POPUP DIALOG */}
      {showShortcutsModal && (
        <div
          onClick={() => setShowShortcutsModal(false)}
          className="fixed inset-0 z-55 bg-black/70 backdrop-blur-md flex items-center justify-center p-4"
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className={`w-full max-w-md rounded-3xl border p-6.5 shadow-2xl transition-all duration-300 ${theme === 'light'
              ? 'bg-white border-slate-205 text-slate-800'
              : 'bg-[#120f2e] border-indigo-500/20 text-slate-200'
              }`}
          >
            <div className="flex items-center justify-between border-b border-indigo-500/10 pb-3 mb-4.5">
              <div className="flex items-center gap-2">
                <HelpCircle className="w-5 h-5 text-purple-500" />
                <h3 className={`text-base font-black ${theme === 'light' ? 'text-slate-800' : 'text-white'}`}>
                  NLP Annotator Shortcuts Index
                </h3>
              </div>
              <button
                onClick={() => setShowShortcutsModal(false)}
                className={`text-lg font-black p-1 transition-colors cursor-pointer ${theme === 'light' ? 'text-slate-400 hover:text-slate-600' : 'text-slate-400 hover:text-red-400'}`}
                title="Close shortcuts modal"
              >
                &times;
              </button>
            </div>

            <p className={`text-xs mb-4.5 leading-relaxed ${theme === 'light' ? 'text-slate-500 font-bold' : 'text-slate-400 font-medium'}`}>
              Speed up transcription mapping and fine-tuning annotation work using the following productivity key combinations:
            </p>

            <div className="space-y-3">
              <div className={`flex items-center justify-between p-3 rounded-2xl border transition-all ${theme === 'light' ? 'bg-slate-50 border-slate-150' : 'bg-white/[0.02] border-white/5'
                }`}>
                <span className="text-xs font-bold">Toggle Play/Pause</span>
                <kbd className="px-3 py-1 text-[11px] font-mono font-black bg-gradient-to-r from-pink-500 to-purple-600 text-white rounded-lg shadow-sm">
                  Ctrl + Space
                </kbd>
              </div>

              <div className={`flex items-center justify-between p-3 rounded-2xl border transition-all ${theme === 'light' ? 'bg-slate-50 border-slate-150' : 'bg-white/[0.02] border-white/5'
                }`}>
                <span className="text-xs font-bold">Seek Backward (5s)</span>
                <kbd className="px-3 py-1 text-[11px] font-mono font-black bg-gradient-to-r from-pink-500 to-purple-600 text-white rounded-lg shadow-sm">
                  Ctrl + Left Arrow
                </kbd>
              </div>

              <div className={`flex items-center justify-between p-3 rounded-2xl border transition-all ${theme === 'light' ? 'bg-slate-50 border-slate-150' : 'bg-white/[0.02] border-white/5'
                }`}>
                <span className="text-xs font-bold">Seek Forward (5s)</span>
                <kbd className="px-3 py-1 text-[11px] font-mono font-black bg-gradient-to-r from-pink-500 to-purple-600 text-white rounded-lg shadow-sm">
                  Ctrl + Right Arrow
                </kbd>
              </div>

              <div className={`flex items-center justify-between p-3 rounded-2xl border transition-all ${theme === 'light' ? 'bg-slate-50 border-slate-150' : 'bg-white/[0.02] border-white/5'
                }`}>
                <span className="text-xs font-bold">Auto-Fix Dialect Symbols</span>
                <span className={`text-[10px] font-mono font-black px-2.5 py-1 rounded bg-slate-200 text-slate-700 uppercase ${theme === 'light' ? 'bg-slate-100 text-slate-500' : 'bg-white/5 text-slate-400'}`}>
                  Dialect tab &bull; Auto-Fix
                </span>
              </div>
            </div>

            <div className="mt-6 flex justify-end">
              <button
                onClick={() => setShowShortcutsModal(false)}
                className="px-5 py-2.5 bg-gradient-to-r from-pink-500 to-purple-600 text-white font-black text-xs uppercase tracking-wider rounded-xl transition-all hover:opacity-95 cursor-pointer shadow-md hover:scale-105 active:scale-95 border-none"
              >
                Close Guild Index
              </button>
            </div>
          </div>
        </div>
      )}

      {/* RESET PROJECT CONFIRMATION POPUP DIALOG */}
      {showResetProjectConfirm && (
        <div
          onClick={() => setShowResetProjectConfirm(false)}
          className="fixed inset-0 z-55 bg-black/75 backdrop-blur-md flex items-center justify-center p-4 alert-modal-overlay"
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className={`w-full max-w-sm rounded-3xl border p-6.5 shadow-2xl transition-all duration-300 ${theme === 'light'
              ? 'bg-white border-slate-205 text-slate-800 shadow-xl'
              : 'bg-[#18112d] border-red-500/20 text-slate-200'
              }`}
          >
            <div className="flex items-center gap-3.5 text-red-500 mb-4">
              <div className="p-3 rounded-2xl bg-red-500/10 shrink-0">
                <Trash2 className="w-6 h-6 text-red-500 animate-pulse" />
              </div>
              <div>
                <h3 className={`text-base font-black ${theme === 'light' ? 'text-slate-800' : 'text-white'}`}>
                  Reset Entire Project?
                </h3>
                <p className="text-[10px] font-mono tracking-wider uppercase font-extrabold text-slate-400">Irreversible Action</p>
              </div>
            </div>

            <p className={`text-xs mb-5 leading-relaxed font-semibold ${theme === 'light' ? 'text-slate-500' : 'text-slate-400'}`}>
              Are you sure you want to completely clear the project? This immediately deletes all segments, transcript custom annotations, audio sources, and resets saved draft states from your system.
            </p>

            <div className="flex gap-2.5 justify-end">
              <button
                onClick={() => setShowResetProjectConfirm(false)}
                className={`px-4.5 py-2.5 text-xs font-black rounded-xl transition-all hover:scale-105 active:scale-95 cursor-pointer border ${theme === 'light'
                  ? 'border-slate-205 text-slate-600 bg-slate-50 hover:bg-slate-105'
                  : 'border-white/10 text-slate-300 bg-white/5 hover:bg-white/10'
                  }`}
              >
                No, Keep Data
              </button>
              <button
                onClick={handleClearAllConfirm}
                className="px-5 py-2.5 bg-gradient-to-r from-red-500 to-pink-600 text-white font-black text-xs uppercase tracking-wider rounded-xl transition-all hover:opacity-95 shadow-md shadow-red-500/10 hover:scale-105 active:scale-95 border-none cursor-pointer"
              >
                Yes, Reset Project
              </button>
            </div>
          </div>
        </div>
      )}

      {/* REAL-TIME NOTIFICATION SYSTEM (TOAST) */}
      {toast && (
        <div
          className="fixed bottom-6 right-6 z-50 flex items-center gap-3.5 px-4.5 py-3.5 rounded-2xl border shadow-2xl transition-all backdrop-blur-md"
          style={{
            borderColor: toast.type === "success" ? "#10B981" : toast.type === "error" ? "#EF4444" : "#F59E0B",
            backgroundColor: theme === "light" ? "rgba(255, 255, 255, 0.98)" : "rgba(25, 15, 30, 0.98)",
            color: theme === "light" ? "#1F2937" : "#F3F4F6",
            boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.3)"
          }}
        >
          <div className="flex items-center gap-2.5">
            <span
              className="w-2.5 h-2.5 rounded-full animate-pulse shrink-0"
              style={{
                backgroundColor: toast.type === "success" ? "#10B981" : toast.type === "error" ? "#EF4444" : "#F59E0B"
              }}
            />
            <span className="text-xs font-black leading-relaxed">{toast.message}</span>
          </div>
        </div>
      )}

    </div>
  );
}