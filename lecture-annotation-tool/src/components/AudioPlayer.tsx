/**
 * AudioPlayer Component
 *
 * A feature-rich audio player with:
 * - Play/Pause controls with visual feedback
 * - Seek buttons (±5 seconds)
 * - Playback rate adjustment (0.5x to 2.0x)
 * - Volume control with mute toggle
 * - Real-time progress slider
 * - Animated waveform visualizer synchronized to playback
 * - Time display (current/total)
 * - Responsive design with light/dark theme support
 *
 * The visualizer uses canvas rendering for smooth animation and displays
 * a dynamic waveform that responds to playback progress.
 */

import React, { useRef, useEffect, useState } from "react";
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Volume2,
  VolumeX,
  Gauge,
  Music,
  FileAudio
} from "lucide-react";
import { VoiceSegment } from "../types";

interface AudioPlayerProps {
  activeSegment: VoiceSegment | null;
  audioRef: React.RefObject<HTMLAudioElement | null>;
  isPlaying: boolean;
  setIsPlaying: (playing: boolean) => void;
  currentTime: number;
  setCurrentTime: (time: number) => void;
  duration: number;
  setDuration: (duration: number) => void;
  playbackRate: number;
  setPlaybackRate: (rate: number) => void;
  theme: "dark" | "light";
}

export default function AudioPlayer({
  activeSegment,
  audioRef,
  isPlaying,
  setIsPlaying,
  currentTime,
  setCurrentTime,
  duration,
  setDuration,
  playbackRate,
  setPlaybackRate,
  theme,
}: AudioPlayerProps) {
  const [isMuted, setIsMuted] = useState(false);
  const [volume, setVolume] = useState(1.0);
  const visualizerRef = useRef<HTMLCanvasElement | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Available playback speed options
  const speeds = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0];

  // Sync volume with element
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = isMuted ? 0 : volume;
    }
  }, [volume, isMuted, audioRef]);

  // Sync playback rate with element
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.playbackRate = playbackRate;
    }
  }, [playbackRate, activeSegment, audioRef]);

  // Synchronize playing states
  const togglePlay = () => {
    if (!activeSegment) return;
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
      setIsPlaying(false);
    } else {
      audio.play().then(() => {
        setIsPlaying(true);
      }).catch((err) => {
        console.warn("Audio play failed, maybe user interaction required:", err);
      });
    }
  };

  // Seek back/forward 5s
  const seek = (seconds: number) => {
    if (!activeSegment) return;
    const audio = audioRef.current;
    if (!audio) return;

    let targetTime = audio.currentTime + seconds;
    if (targetTime < 0) targetTime = 0;
    if (targetTime > duration) targetTime = duration;

    audio.currentTime = targetTime;
    setCurrentTime(targetTime);
  };

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!activeSegment) return;
    const audio = audioRef.current;
    if (!audio) return;

    const targetTime = parseFloat(e.target.value);
    audio.currentTime = targetTime;
    setCurrentTime(targetTime);
  };

  const formatTime = (timeInSecs: number) => {
    if (isNaN(timeInSecs)) return "00:00";
    const mins = Math.floor(timeInSecs / 60);
    const secs = Math.floor(timeInSecs % 60);
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  // Waveform canvas visualizer logic
  useEffect(() => {
    const canvas = visualizerRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let width = canvas.width;
    let height = canvas.height;

    // Resize handler inside visualizer
    const resizeObserver = new ResizeObserver((entries) => {
      for (let entry of entries) {
        if (canvas) {
          canvas.width = entry.contentRect.width;
          canvas.height = entry.contentRect.height;
          width = canvas.width;
          height = canvas.height;
        }
      }
    });

    if (canvas.parentElement) {
      resizeObserver.observe(canvas.parentElement);
    }

    const draw = () => {
      ctx.clearRect(0, 0, width, height);

      const barCount = 42;
      const barWidth = Math.max(2, (width / barCount) - 3);
      const gap = 3;

      // Active glowing color linear gradient across the canvas length
      const gradient = ctx.createLinearGradient(0, height, width, 0);
      gradient.addColorStop(0, '#ec4899'); // Pink
      gradient.addColorStop(0.5, '#8b5cf6'); // Purple / Violet
      gradient.addColorStop(1, '#06b6d4'); // Cyan

      for (let i = 0; i < barCount; i++) {
        // Calculate dynamic height multipliers based on index and play state
        let scale = 0.15 + 0.1 * Math.sin(i * 0.4);
        if (isPlaying) {
          scale += 0.5 * Math.sin(currentTime * 12 + i * 0.9) * Math.cos(currentTime * 3 + i * 0.2);
          scale = Math.abs(scale);
        }

        // Ensure within sensible boundaries
        const barHeight = Math.max(4, scale * height * 0.7);
        const x = i * (barWidth + gap);
        const y = (height - barHeight) / 2;

        const basePct = i / barCount;
        const currentProgress = duration > 0 ? (currentTime / duration) : 0;

        // Highlight bars up to current playback percentage
        if (basePct <= currentProgress && activeSegment) {
          ctx.fillStyle = gradient;
        } else {
          ctx.fillStyle = theme === 'light' ? "rgba(139, 92, 246, 0.12)" : "rgba(255, 255, 255, 0.12)";
        }

        // Draw rounded rectangle bars with safe cross-browser fallback
        ctx.beginPath();
        if (typeof ctx.roundRect === "function") {
          ctx.roundRect(x, y, barWidth, barHeight, 2.5);
        } else {
          ctx.rect(x, y, barWidth, barHeight);
        }
        ctx.fill();
      }

      animationFrameRef.current = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      resizeObserver.disconnect();
    };
  }, [isPlaying, currentTime, duration, activeSegment, theme]);

  return (
    <div
      id="audio-player-container"
      className={`border rounded-3xl p-6 shadow-xl flex flex-col gap-4.5 transition-all duration-300 ${theme === "light"
          ? "bg-white border-slate-200/80 shadow-[0_8px_30px_rgba(139,92,246,0.03)] text-slate-800"
          : "bg-gradient-to-b from-[#120f30]/40 to-[#0e0c1f]/40 border-white/5 shadow-2xl text-slate-200"
        }`}
    >
      {/* Top Banner indicating Active Segment */}
      <div className={`flex items-center justify-between border-b pb-3.5 ${theme === 'light' ? 'border-slate-100' : 'border-white/5'
        }`}>
        <div className="flex items-center gap-3">
          <div className="bg-gradient-to-br from-pink-500 to-purple-500 p-2 text-white rounded-xl shadow-md shadow-purple-500/10 shrink-0">
            {activeSegment?.hasAudioFile ? (
              <FileAudio className="w-5 h-5 animate-pulse" />
            ) : (
              <Music className="w-5 h-5" />
            )}
          </div>
          <div className="min-w-0">
            <h3 className={`font-bold text-sm truncate max-w-[210px] md:max-w-[360px] ${theme === 'light' ? 'text-slate-800' : 'text-slate-100'
              }`}>
              {activeSegment ? activeSegment.name : "No Segment Selected"}
            </h3>
            <p className={`text-[11px] font-medium mt-0.5 ${theme === 'light' ? 'text-slate-400' : 'text-slate-550 text-slate-500'
              }`}>
              {activeSegment
                ? `${activeSegment.hasAudioFile ? "Physical Audio" : "Synthesized Preview"} • ${formatTime(duration)}`
                : "Select or upload an audio segment"
              }
            </p>
          </div>
        </div>

        {/* Speed adjust */}
        {activeSegment && (
          <div className={`flex items-center gap-1.5 py-1 px-3 rounded-xl text-xs font-mono font-bold border transition-colors ${theme === 'light'
              ? 'bg-slate-100 hover:bg-slate-150 border-slate-200 text-slate-600'
              : 'bg-black/35 border-white/5 text-slate-300 hover:border-white/10'
            }`}>
            <Gauge className="w-3.5 h-3.5" />
            <select
              title="Playback Velocity"
              id="playback-rate-select"
              value={playbackRate}
              onChange={(e) => setPlaybackRate(parseFloat(e.target.value))}
              className="bg-transparent border-none focus:outline-none cursor-pointer text-xs font-bold leading-none text-current"
            >
              {speeds.map((rate) => (
                <option key={rate} value={rate} className={theme === 'light' ? 'bg-white text-slate-800' : 'bg-[#0a0a0f] text-slate-250'}>
                  {rate.toFixed(2)}x
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {/* Visual Waveform Canvas */}
      <div className={`w-full h-18 rounded-2xl relative overflow-hidden flex items-center justify-center p-1 border transition-colors ${theme === 'light'
          ? 'bg-slate-50/75 border-slate-100'
          : 'bg-black/45 border-white/5'
        }`}>
        {activeSegment ? (
          <canvas ref={visualizerRef} className="w-full h-full block" />
        ) : (
          <p className="text-xs text-slate-400 font-mono tracking-wide">Select a lecture segment card below to play</p>
        )}
      </div>

      {/* Range Seeker bar */}
      <div className="flex flex-col gap-1.5">
        <div className={`flex justify-between text-[11px] font-mono font-bold ${theme === 'light' ? 'text-slate-400' : 'text-slate-500'
          }`}>
          <span>{formatTime(currentTime)}</span>
          <span>{formatTime(duration)}</span>
        </div>
        <input
          id="audio-progress-slider"
          type="range"
          min={0}
          max={duration || 60}
          step={0.05}
          value={currentTime}
          onChange={handleSliderChange}
          disabled={!activeSegment}
          className={`w-full h-1.5 rounded-lg appearance-none cursor-pointer focus:outline-none focus:ring-2 focus:ring-purple-450/30 transition-all ${theme === 'light' ? 'bg-slate-200 accent-purple-600' : 'bg-slate-800 accent-pink-500'
            }`}
          title="Playhead Position"
        />
      </div>

      {/* Media Player Control Deck */}
      <div className="flex items-center justify-between mt-1">
        {/* Left Side: Volume Controls */}
        <div className="flex items-center gap-2.5 w-28 md:w-36 shrink-0">
          <button
            id="audio-mute-button"
            onClick={() => setIsMuted(!isMuted)}
            disabled={!activeSegment}
            className={`p-2 rounded-xl disabled:opacity-40 transition-all active:scale-90 ${theme === 'light'
                ? 'bg-slate-100 hover:bg-slate-200 text-slate-600'
                : 'bg-white/5 hover:bg-white/10 text-slate-300'
              }`}
            title={isMuted ? "Unmute" : "Mute"}
          >
            {isMuted || volume === 0 ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
          </button>
          <input
            id="audio-volume-slider"
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={volume}
            onChange={(e) => {
              const v = parseFloat(e.target.value);
              setVolume(v);
              if (v > 0) setIsMuted(false);
            }}
            disabled={!activeSegment}
            className={`w-full h-1 rounded-lg appearance-none cursor-pointer ${theme === 'light' ? 'bg-slate-200 accent-purple-600' : 'bg-slate-800 accent-slate-300'
              }`}
            title="Volume Slider"
          />
        </div>

        {/* Center: Core Control Buttons */}
        <div className="flex items-center gap-3 md:gap-4 shrink-0">
          <button
            id="audio-seek-backward-button"
            onClick={() => seek(-5)}
            disabled={!activeSegment}
            className={`p-2.5 sm:p-3 rounded-xl transition-all disabled:opacity-40 active:scale-90 ${theme === 'light'
                ? 'bg-slate-100 hover:bg-slate-200 text-slate-600'
                : 'bg-white/5 hover:bg-white/10 text-slate-200'
              }`}
            title="Seek back 5 seconds (Ctrl+Left)"
          >
            <SkipBack className="w-4 h-4 fill-current" />
          </button>

          <button
            id="audio-playback-toggle-button"
            onClick={togglePlay}
            disabled={!activeSegment}
            className="p-3.5 sm:p-4 bg-gradient-to-r from-pink-500 to-purple-600 hover:from-pink-400 hover:to-purple-500 text-white rounded-full shadow-lg shadow-purple-500/20 active:scale-95 transform hover:scale-105 transition-all disabled:opacity-40 border-none"
            title="Play / Pause Spacebar"
          >
            {isPlaying ? <Pause className="w-5 h-5 fill-white text-white" /> : <Play className="w-5 h-5 fill-white text-white ml-0.5" />}
          </button>

          <button
            id="audio-seek-forward-button"
            onClick={() => seek(5)}
            disabled={!activeSegment}
            className={`p-2.5 sm:p-3 rounded-xl transition-all disabled:opacity-40 active:scale-90 ${theme === 'light'
                ? 'bg-slate-100 hover:bg-slate-200 text-slate-600'
                : 'bg-white/5 hover:bg-white/10 text-slate-200'
              }`}
            title="Seek forward 5 seconds (Ctrl+Right)"
          >
            <SkipForward className="w-4 h-4 fill-current" />
          </button>
        </div>

        {/* Right Side: Indicators */}
        <div className={`text-right text-[11px] font-mono font-bold hidden md:block shrink-0 ${theme === 'light' ? 'text-slate-400' : 'text-slate-500'
          }`}>
          {activeSegment && (
            <span>
              {activeSegment.mimeType.split("/")[1]?.toUpperCase() || "MP3"}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
