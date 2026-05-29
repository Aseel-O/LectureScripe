/**
 * SegmentList Component
 *
 * Manages the display and filtering of audio segments with features:
 * - Segment browsing and selection with visual feedback
 * - Search by filename, transcript, or tags
 * - Filtering by status (not started, transcribing, draft, completed)
 * - Tag-based filtering
 * - Drag-and-drop file import
 * - File upload (single or multiple files)
 * - Bulk speaker name editing across all segments
 * - Segment deletion with progress tracking
 * - Statistics dashboard showing annotation progress
 *
 * The component supports both individual file selection and folder imports,
 * with file type validation and natural sorting for consistent ordering.
 */

import React, { useState, useRef } from "react";
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
  Users
} from "lucide-react";
import { VoiceSegment, SegmentStatus } from "../types";

interface SegmentListProps {
  segments: VoiceSegment[];
  activeSegmentId: string | null;
  onSelectSegment: (id: string) => void;
  onDeleteSegment: (id: string) => void;
  onFilesSelected: (files: FileList | File[]) => void;
  onLoadSamples: () => void;
  onClearAll: () => void;
  theme: "dark" | "light";
  onBulkRenameSpeaker: (oldName: string, newName: string) => void;
}

/**
 * Format file size in human-readable format (Bytes, KB, MB, GB).
 *
 * @param bytes - File size in bytes
 * @return Formatted string (e.g., "2.5 MB")
 */
function formatSize(bytes: number): string {
  if (!bytes) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
}

export default function SegmentList({
  segments,
  activeSegmentId,
  onSelectSegment,
  onDeleteSegment,
  onFilesSelected,
  onLoadSamples,
  onClearAll,
  theme,
  onBulkRenameSpeaker,
}: SegmentListProps) {
  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState<"all" | SegmentStatus>("all");
  const [selectedTagFilter, setSelectedTagFilter] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const folderInputRef = useRef<HTMLInputElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [renameSpeakerFrom, setRenameSpeakerFrom] = useState<string | null>(null);
  const [renameSpeakerTo, setRenameSpeakerTo] = useState<string>("");

  // Extract all unique tags from segments for filter display
  const allUniqueTags = Array.from(
    new Set((segments || []).flatMap((s) => (s && s.tags) || []))
  ).filter(Boolean);

  /**
   * Apply search and filter criteria to segments.
   * Filters by search term (filename, transcript, tags) and status/tag selection.
   */
  const filteredSegments = segments.filter((s) => {
    // Match against filename, transcript content, and tags
    const tagMatchesSearch = s.tags && s.tags.some(t => t.toLowerCase().includes(searchTerm.toLowerCase()));
    const matchesSearch =
      s.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      s.transcript.toLowerCase().includes(searchTerm.toLowerCase()) ||
      !!tagMatchesSearch;

    // Match against status filter
    const matchesFilter = statusFilter === "all" || s.status === statusFilter;

    // Match against selected tag filter
    const matchesTagFilter = !selectedTagFilter || (s.tags && s.tags.includes(selectedTagFilter));

    return matchesSearch && matchesFilter && matchesTagFilter;
  });

  // Calculate project statistics
  const totalCount = segments.length;
  const completedCount = segments.filter((s) => s.status === "completed").length;
  const draftCount = segments.filter((s) => s.status === "draft").length;
  const notStartedCount = segments.filter((s) => s.status === "not_started").length;
  const progressPercent = totalCount > 0 ? Math.round((completedCount / totalCount) * 100) : 0;

  /**
   * Handle file selection from input element.
   */
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFilesSelected(e.target.files);
    }
  };

  /**
   * Handle drag-over event to show visual feedback.
   */
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  /**
   * Handle drag-leave to reset visual feedback.
   */
  const handleDragLeave = () => {
    setIsDragging(false);
  };

  /**
   * Handle drop event to import files via drag-and-drop.
   */
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFilesSelected(e.dataTransfer.files);
    }
  };

  /**
   * Return the appropriate status badge component for a segment.
   * Displays color-coded status with descriptive text.
   */
  const getStatusBadge = (status: SegmentStatus) => {
    switch (status) {
      case "completed":
        return (
          <span className="inline-flex items-center gap-1 text-[10px] font-bold bg-emerald-500/10 text-emerald-500 dark:text-emerald-400 border border-emerald-500/20 px-2 py-0.5 rounded-full">
            <CheckCircle2 className="w-3 h-3 text-emerald-500" /> Verbatim Corrected
          </span>
        );
      case "draft":
        return (
          <span className="inline-flex items-center gap-1 text-[10px] font-bold bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20 px-2 py-0.5 rounded-full">
            <FileEdit className="w-3 h-3 text-amber-500" /> Transcribed AI Draft
          </span>
        );
      case "transcribing":
        return (
          <span className="inline-flex items-center gap-1 text-[10px] font-bold bg-sky-500/10 text-sky-600 dark:text-sky-400 border border-sky-500/30 px-2 py-0.5 rounded-full animate-pulse">
            <Activity className="w-3 h-3 animate-spin text-sky-500" /> Transcribing...
          </span>
        );
      case "not_started":
      default:
        return (
          <span className="inline-flex items-center gap-1 text-[10px] font-bold bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 border border-slate-205 dark:border-white/5 px-2 py-0.5 rounded-full">
            <Clock className="w-3 h-3" /> No Annotation
          </span>
        );
    }
  };

  // Extract all unique speaker names across segments for bulk rename feature
  const allUniqueSpeakers = Array.from(
    new Set((segments || []).flatMap((s) => ((s && s.speakerTurns) || []).map((t) => t.speaker)))
  ).filter(Boolean);

  return (
    <div
      id="segment-list-container"
      className={`flex flex-col h-full border rounded-3xl overflow-hidden shadow-2xl transition-all duration-300 ${theme === "light"
        ? "bg-white border-slate-200/85 text-slate-800"
        : "bg-[#09071a]/95 backdrop-blur-md border-white/10 text-slate-200"
        }`}
    >
      {/* progress panel */}
      <div className={`p-4.5 border-b transition-colors ${theme === "light"
        ? "border-slate-100 bg-gradient-to-b from-slate-50 to-white"
        : "border-white/5 bg-slate-950/40"
        }`}>
        <div className="flex justify-between items-center mb-1.5">
          <span className={`text-[10px] font-mono tracking-widest font-black uppercase ${theme === "light" ? "text-slate-400" : "text-slate-500"
            }`}>
            Corpus Progress
          </span>
          <span className="text-xs font-black text-purple-600 dark:text-pink-400 font-mono">{progressPercent}% DONE</span>
        </div>
        <div className={`w-full h-2 rounded-full overflow-hidden p-[1px] ${theme === "light" ? "bg-slate-100 border border-slate-200" : "bg-slate-950 border border-white/5"
          }`}>
          <div
            className="bg-gradient-to-r from-pink-500 to-purple-500 h-full rounded-full transition-all duration-500"
            style={{ width: `${progressPercent}%` }}
          />
        </div>

        {/* Colorful dashboard cells */}
        <div className="grid grid-cols-4 gap-1.5 mt-4 text-center">
          <div className={`p-2 rounded-xl border transition-all hover:scale-105 ${theme === "light"
            ? "bg-sky-50/50 hover:bg-sky-50 border-sky-100 text-sky-800 shadow-sm"
            : "bg-sky-500/5 hover:bg-sky-500/10 border-sky-500/15 text-sky-400"
            }`}>
            <div className={`text-base font-black`}>{totalCount}</div>
            <div className="text-[8px] text-slate-400 font-extrabold uppercase tracking-wide">Total</div>
          </div>
          <div className={`p-2 rounded-xl border transition-all hover:scale-105 ${theme === "light"
            ? "bg-emerald-50/50 hover:bg-emerald-50 border-emerald-100 text-emerald-800 shadow-sm"
            : "bg-emerald-500/5 hover:bg-emerald-500/10 border-emerald-500/15 text-emerald-400"
            }`}>
            <div className={`text-base font-black`}>{completedCount}</div>
            <div className="text-[8px] text-slate-400 font-extrabold uppercase tracking-wide">Done</div>
          </div>
          <div className={`p-2 rounded-xl border transition-all hover:scale-105 ${theme === "light"
            ? "bg-amber-50/50 hover:bg-amber-50 border-amber-100 text-amber-800 shadow-sm"
            : "bg-amber-500/5 hover:bg-amber-500/10 border-amber-500/15 text-amber-400"
            }`}>
            <div className={`text-base font-black`}>{draftCount}</div>
            <div className="text-[8px] text-slate-400 font-extrabold uppercase tracking-wide">Drafts</div>
          </div>
          <div className={`p-2 rounded-xl border transition-all hover:scale-105 ${theme === "light"
            ? "bg-purple-50/50 hover:bg-purple-50 border-purple-100 text-purple-800 shadow-sm"
            : "bg-purple-500/5 hover:bg-purple-500/10 border-purple-500/15 text-purple-400"
            }`}>
            <div className={`text-base font-black`}>{notStartedCount}</div>
            <div className="text-[8px] text-slate-400 font-extrabold uppercase tracking-wide">Empty</div>
          </div>
        </div>
      </div>

      {/* Directory Selector and Import Controls */}
      <div className={`p-4 border-b flex flex-col gap-2 transition-colors ${theme === "light" ? "border-slate-100 bg-slate-50/30" : "border-white/5 bg-slate-900/15"
        }`}>
        <div className="grid grid-cols-2 gap-2">
          {/* Load files */}
          <button
            id="select-files-btn"
            onClick={() => fileInputRef.current?.click()}
            className={`flex items-center justify-center gap-1.5 py-2 px-3 text-xs font-black rounded-xl transition-all border cursor-pointer hover:scale-[1.02] active:scale-95 ${theme === "light"
              ? "bg-white hover:bg-slate-50 border-slate-200 text-slate-700 shadow-sm"
              : "bg-slate-800/80 hover:bg-slate-700/80 border-slate-700 text-slate-200"
              }`}
            title="Import selected files"
          >
            <FileAudio className="w-3.5 h-3.5 text-pink-500" />
            Select Files
          </button>

          {/* Load Folder */}
          <button
            id="select-folder-btn"
            onClick={() => folderInputRef.current?.click()}
            className="flex items-center justify-center gap-1.5 py-2 px-3 bg-purple-500/10 hover:bg-purple-500/20 border border-purple-500/30 text-xs text-purple-600 dark:text-purple-400 font-black rounded-xl transition-all cursor-pointer hover:scale-[1.02] active:scale-95"
            title="Import whole folder structure"
          >
            <FolderOpen className="w-3.5 h-3.5 text-purple-500" />
            Select Folder
          </button>
        </div>

        {/* Hidden File inputs */}
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          multiple
          accept="audio/*"
          className="hidden"
          title="Upload Audio Files"
        />
        <input
          type="file"
          ref={folderInputRef}
          onChange={handleFileChange}
          multiple
          className="hidden"
          title="Upload Directory of Audio Files"
          {...({ webkitdirectory: true, directory: true } as any)}
        />

        <div className="flex justify-between items-center text-[10px] text-slate-400 font-bold mt-1">
          <span>Accepts WAV, MP3, OGG</span>
          {segments.length > 0 && (
            <button
              onClick={onClearAll}
              className="text-red-500 hover:text-red-400 font-black flex items-center gap-0.5 transition-colors"
              title="Clear all segments from the list"
            >
              <Trash2 className="w-3 h-3" /> Reset Project
            </button>
          )}
        </div>
      </div>

      {/* Search and Filters */}
      <div className={`p-4 border-b flex flex-col gap-2.5 transition-colors ${theme === "light" ? "border-slate-100 bg-white" : "border-white/5 bg-slate-950/20"
        }`}>
        <div className="relative">
          <Search className="w-4 h-4 absolute left-3 top-2.5 text-slate-400 dark:text-slate-500" />
          <input
            id="search-segments-input"
            type="text"
            placeholder="Search filenames, tags..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className={`w-full pl-9 pr-3 py-2 text-xs focus:ring-2 focus:ring-purple-400/20 focus:outline-none rounded-xl transition-all border ${theme === "light"
              ? "bg-slate-50/50 border-slate-200 text-slate-800 placeholder:text-slate-400"
              : "bg-slate-950 border-slate-800 text-slate-300 placeholder:text-slate-600 focus:border-pink-500/50"
              }`}
          />
        </div>

        {/* Status filter pills */}
        <div className="flex items-center gap-1 overflow-x-auto pb-1 scrollbar-thin">
          <button
            onClick={() => setStatusFilter("all")}
            className={`px-2.5 py-1 text-[9px] font-black rounded-lg transition-all shrink-0 uppercase tracking-widest ${statusFilter === "all"
              ? "bg-[#6d28d9] text-white shadow-sm"
              : theme === "light"
                ? "bg-slate-50 text-slate-500 hover:bg-slate-100"
                : "bg-white/5 text-slate-400 hover:text-slate-250 hover:bg-white/10"
              }`}
          >
            All ({totalCount})
          </button>
          <button
            onClick={() => setStatusFilter("completed")}
            className={`px-2.5 py-1 text-[9px] font-black rounded-lg transition-all shrink-0 uppercase tracking-widest border ${statusFilter === "completed"
              ? "bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 border-emerald-500/30 font-extrabold"
              : "text-slate-400 dark:text-slate-500 hover:text-emerald-400 border-transparent"
              }`}
          >
            Done ({completedCount})
          </button>
          <button
            onClick={() => setStatusFilter("draft")}
            className={`px-2.5 py-1 text-[9px] font-black rounded-lg transition-all shrink-0 uppercase tracking-widest border ${statusFilter === "draft"
              ? "bg-amber-500/20 text-amber-700 dark:text-amber-400 border-amber-500/30 font-extrabold"
              : "text-slate-400 dark:text-slate-500 hover:text-amber-504 hover:text-amber-500 border-transparent"
              }`}
          >
            Draft ({draftCount})
          </button>
          <button
            onClick={() => setStatusFilter("not_started")}
            className={`px-2.5 py-1 text-[9px] font-black rounded-lg transition-all shrink-0 uppercase tracking-widest border ${statusFilter === "not_started"
              ? theme === "light"
                ? "bg-slate-100 text-slate-705 border-slate-200"
                : "bg-slate-800/40 text-slate-300 border-slate-700/50"
              : "text-slate-400 dark:text-slate-500 hover:text-slate-300 border-transparent"
              }`}
          >
            Empty ({notStartedCount})
          </button>
        </div>

        {/* Custom Unique List tag filter row if available */}
        {allUniqueTags.length > 0 && (
          <div className="flex items-center gap-1.5 overflow-x-auto pb-1 pt-1.5 border-t border-slate-150 dark:border-slate-850 dark:border-slate-800/40">
            <span className="text-[8px] uppercase font-mono font-black text-slate-400 shrink-0">Tags:</span>
            <button
              onClick={() => setSelectedTagFilter(null)}
              className={`px-2 py-0.5 text-[10px] rounded-md shrink-0 font-bold transition-all ${!selectedTagFilter
                ? "bg-purple-100 text-purple-705 dark:bg-pink-950/30 dark:text-pink-400 border border-purple-200 dark:border-pink-500/20 font-black"
                : "text-slate-400 hover:text-slate-700"
                }`}
            >
              Clear
            </button>
            {allUniqueTags.map((tag) => (
              <button
                key={tag}
                onClick={() => setSelectedTagFilter(tag)}
                className={`px-2 py-0.5 text-[10px] rounded-md shrink-0 font-bold border transition-all ${selectedTagFilter === tag
                  ? "bg-gradient-to-r from-pink-500 to-purple-500 text-white border-transparent font-black shadow-sm"
                  : theme === 'light'
                    ? "bg-slate-50 hover:bg-slate-100 text-slate-600 border-slate-200"
                    : "bg-slate-900 hover:bg-slate-800 text-slate-400 border-white/5"
                  }`}
              >
                #{tag}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Segment Cards List */}
      <div
        id="segments-dropzone"
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`flex-1 overflow-y-auto p-4.5 flex flex-col gap-3 min-h-0 transition-all ${theme === "light" ? "bg-slate-50/20" : "bg-transparent"
          } ${isDragging ? "bg-purple-500/5 border-2 border-dashed border-purple-500 rounded-2xl" : ""}`}
      >
        {filteredSegments.length > 0 ? (
          filteredSegments.map((s) => {
            const isActive = s.id === activeSegmentId;
            return (
              <div
                key={s.id}
                onClick={() => onSelectSegment(s.id)}
                className={`p-4 rounded-2xl border transition-all duration-300 cursor-pointer relative group flex flex-col gap-2.5 hover:scale-[1.015] ${isActive
                  ? theme === "light"
                    ? "bg-gradient-to-r from-violet-50/60 to-purple-50/60 border-purple-500 shadow-[0_4px_20px_rgba(139,92,246,0.08)]"
                    : "bg-gradient-to-r from-[#1c183b]/90 to-[#120f2c]/90 border-pink-500/80 shadow-[0_4px_25px_rgba(236,72,153,0.12)]"
                  : theme === "light"
                    ? "bg-white border-slate-200/80 hover:border-purple-300 hover:bg-slate-50/40"
                    : "bg-slate-900/35 border-slate-800/60 hover:border-purple-500/40 hover:bg-[#13112a]/30"
                  }`}
              >
                <div className="flex justify-between items-start gap-2">
                  <div className="min-w-0 flex-1">
                    <p className={`text-xs font-bold truncate pr-6 ${isActive
                      ? theme === "light" ? "text-purple-950 font-black" : "text-white font-black"
                      : theme === "light" ? "text-slate-850 text-slate-800" : "text-slate-205 text-slate-200"
                      }`} title={s.name}>
                      {s.name}
                    </p>
                    <div className="flex items-center gap-2 mt-1 text-[10px] text-slate-400 font-mono font-bold">
                      <span>{formatSize(s.size)}</span>
                      <span>•</span>
                      <span>{Math.round(s.duration)}s</span>
                    </div>
                  </div>

                  {/* Delete button */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteSegment(s.id);
                    }}
                    className="p-1 text-slate-500 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity absolute right-3.5 top-3.5"
                    title="Delete segment"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>

                {/* Tags associated shown inline on card */}
                {s.tags && s.tags.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {s.tags.map((t) => (
                      <span
                        key={t}
                        className="text-[9px] font-extrabold bg-[#6d28d9]/5 dark:bg-pink-500/10 text-purple-600 dark:text-pink-400 border border-purple-100 dark:border-pink-500/20 px-2 py-0.5 rounded-md"
                      >
                        #{t}
                      </span>
                    ))}
                  </div>
                )}

                {/* Status and Diarization info */}
                <div className={`flex justify-between items-center p-2 rounded-xl border ${theme === "light" ? "bg-slate-50 border-slate-100" : "bg-slate-950/40 border-slate-805 border-slate-800/40"
                  }`}>
                  {getStatusBadge(s.status)}
                  <span className={`text-[9px] font-mono font-bold ${theme === "light" ? "text-slate-400 font-medium" : "text-slate-500"}`}>
                    {s.speakerTurns && s.speakerTurns.length > 0
                      ? `${new Set((s.speakerTurns || []).filter(t => t && t.speaker).map(t => t.speaker)).size} Speakers`
                      : "No speaker info"}
                  </span>
                </div>

                {/* Horizontal snippet lines */}
                {s.transcript && (
                  <p className={`text-[10px] line-clamp-1 italic text-right font-sans ${theme === "light" ? "text-slate-400" : "text-slate-500"}`} dir="rtl">
                    {s.transcript}
                  </p>
                )}
              </div>
            );
          })
        ) : (
          <div className={`flex-1 flex flex-col items-center justify-center text-center p-6 border border-dashed rounded-2xl m-2 ${theme === "light" ? "border-slate-300 bg-slate-50/50" : "border-slate-800 bg-slate-950/20"
            }`}>
            {segments.length === 0 ? (
              <>
                <FileAudio className="w-10 h-10 text-purple-500 mb-3 animate-pulse" />
                <h4 className={`text-xs font-bold ${theme === "light" ? "text-slate-705 text-slate-700" : "text-slate-300"}`}>No segments imported</h4>
                <p className="text-[11px] text-slate-400 max-w-[200px] mt-1.5 leading-relaxed font-semibold">
                  Drag and drop a folder of audio files, select them, or populate class lecture samples!
                </p>
                <button
                  onClick={onLoadSamples}
                  className="mt-4 flex items-center gap-1.5 py-2 px-4 bg-gradient-to-r from-pink-500 to-purple-600 hover:from-pink-400 hover:to-purple-550 text-white text-xs font-black rounded-xl cursor-pointer shadow-md shadow-purple-500/10 transition-all hover:scale-[1.02] active:scale-95"
                >
                  <ListRestart className="w-3.5 h-3.5" />
                  Load Lecture Samples
                </button>
              </>
            ) : (
              <>
                <Search className="w-8 h-8 text-slate-450 text-slate-500 mb-2 animate-bounce" />
                <h4 className={`text-xs font-bold ${theme === "light" ? "text-slate-700" : "text-slate-300"}`}>No matches found</h4>
                <p className="text-[11px] text-slate-400 font-bold mt-1">
                  Try clearing tag filters or writing a different search term.
                </p>
              </>
            )}
          </div>
        )}
      </div>

      {/* Bulk Speaker Editor Dropdown */}
      {allUniqueSpeakers.length > 0 && (
        <div className={`p-4 border-t transition-colors ${theme === "light" ? "border-slate-100 bg-slate-50/50" : "border-white/5 bg-slate-950/60"
          }`}>
          <div className="flex items-center gap-1.5 mb-2 text-[10px] uppercase font-mono font-black text-slate-400 shrink-0">
            <Users className="w-3.5 h-3.5 text-purple-500" />
            <span>Bulk Edit Speaker Names</span>
          </div>
          <select
            id="bulk-speaker-rename-dropdown"
            onChange={(e) => {
              const oldSpeaker = e.target.value;
              if (!oldSpeaker) return;
              setRenameSpeakerFrom(oldSpeaker);
              setRenameSpeakerTo(oldSpeaker);
              e.target.value = ""; // reset
            }}
            className={`w-full text-xs px-3 py-2.5 rounded-xl focus:outline-none focus:border-purple-500 transition-colors border ${theme === "light"
              ? "bg-white border-slate-200 text-slate-800"
              : "bg-slate-950 border-slate-800 text-slate-300"
              }`}
          >
            <option value="">-- Select Speaker to Rename --</option>
            {allUniqueSpeakers.map((sp) => (
              <option key={sp} value={sp}>
                {sp}
              </option>
            ))}
          </select>

          {renameSpeakerFrom && (
            <div className="mt-3.5 p-3 rounded-2xl border border-dashed text-xs animate-fadeIn space-y-2 bg-purple-500/5 border-purple-500/20">
              <div className="text-[10px] uppercase font-mono font-black text-purple-500">
                Rename "{renameSpeakerFrom}" globally:
              </div>
              <div className="flex gap-1.5">
                <input
                  type="text"
                  value={renameSpeakerTo}
                  onChange={(e) => setRenameSpeakerTo(e.target.value)}
                  placeholder="New Speaker Name"
                  className={`flex-1 text-xs px-2.5 py-1.5 rounded-lg border focus:outline-none focus:border-purple-500 ${theme === "light"
                    ? "bg-white border-slate-200 text-slate-800"
                    : "bg-slate-900 border-slate-800 text-slate-200"
                    }`}
                />
                <button
                  onClick={() => {
                    if (renameSpeakerTo.trim() && renameSpeakerTo.trim() !== renameSpeakerFrom) {
                      onBulkRenameSpeaker(renameSpeakerFrom, renameSpeakerTo.trim());
                    }
                    setRenameSpeakerFrom(null);
                  }}
                  className="px-3 py-1.5 rounded-lg bg-purple-600 hover:bg-purple-500 text-white font-black text-xs cursor-pointer select-none transition-colors border-none"
                >
                  Rename
                </button>
                <button
                  onClick={() => setRenameSpeakerFrom(null)}
                  className={`px-2.5 py-1.5 rounded-lg border text-xs font-bold cursor-pointer select-none transition-all ${theme === "light"
                    ? "border-slate-200 text-slate-500 bg-slate-50 hover:bg-slate-100"
                    : "border-white/5 text-slate-400 bg-white/5 hover:bg-white/10"
                    }`}
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
