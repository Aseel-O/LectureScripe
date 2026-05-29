/**
 * Lecture Annotation Tool - Express Server
 *
 * A production-ready backend server that handles:
 * - Audio transcription via the Google Gemini API
 * - API key rotation to manage rate limits across multiple keys
 * - Static file serving for the React frontend
 * - Live development server support via Vite middleware
 *
 * Key Features:
 * - Supports multiple Google Gemini API keys for load balancing
 * - Automatic daily request counter reset
 * - Graceful fallback when a key exceeds quota
 * - Specialized transcription prompts for Palestinian Arabic + English code-switching
 * - Real-time API key status monitoring
 *
 * Environment Variables Required:
 * - GEMINI_API_KEY (primary)
 * - GEMINI_API_KEY_2 through GEMINI_API_KEY_10 (optional, for load balancing)
 * - NODE_ENV (development or production)
 */

import express from "express";
import path from "path";
import { fileURLToPath } from "url";
import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";
import { p } from "motion/react-client";
dotenv.config({ path: ".env.local" });

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3000;

// Enable JSON and URL-encoded body parsing with 50MB limit for large audio data
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ limit: "50mb", extended: true }));

/**
 * API Key Management System
 *
 * Manages a pool of Google Gemini API keys to distribute requests and handle quota limits.
 * Each key has a daily request limit of 250 requests. Keys are automatically reset at UTC midnight.
 */
const DAILY_LIMIT = 250;

/**
 * KeyEntry represents a single API key with its usage metadata.
 *
 * @interface KeyEntry
 * @property {string} key - The actual API key string
 * @property {GoogleGenAI} client - Initialized Gemini API client
 * @property {number} requestsToday - Current request count for today
 * @property {boolean} exhausted - True if key has hit quota (recovers at midnight)
 * @property {string} label - Display name for logging (e.g., "Key 1")
 */
interface KeyEntry {
  key: string;
  client: GoogleGenAI;
  requestsToday: number;
  exhausted: boolean;
  label: string;
}

/**
 * Initializes the API key pool from environment variables.
 *
 * Loads up to 10 API keys from GEMINI_API_KEY and GEMINI_API_KEY_2 through GEMINI_API_KEY_10.
 * Each key is validated and wrapped in a GoogleGenAI client instance.
 *
 * @returns {KeyEntry[]} Array of validated API key entries
 * @throws {Error} If no valid API keys are found in environment
 */
function buildKeyPool(): KeyEntry[] {
  const raw = [
    process.env.GEMINI_API_KEY,
    process.env.GEMINI_API_KEY_2,
    process.env.GEMINI_API_KEY_3,
    process.env.GEMINI_API_KEY_4,
    process.env.GEMINI_API_KEY_5,
    process.env.GEMINI_API_KEY_6,
    process.env.GEMINI_API_KEY_7,
    process.env.GEMINI_API_KEY_8,
    process.env.GEMINI_API_KEY_9,
    process.env.GEMINI_API_KEY_10,
  ];

  const pool: KeyEntry[] = [];
  for (let i = 0; i < raw.length; i++) {
    const key = raw[i];
    // Only include non-empty keys
    if (key && key.trim() !== "") {
      pool.push({
        key,
        client: new GoogleGenAI({
          apiKey: key,
          httpOptions: { headers: { "User-Agent": "aistudio-build" } },
        }),
        requestsToday: 0,
        exhausted: false,
        label: `Key ${i + 1}`,
      });
    }
  }

  if (pool.length === 0) {
    throw new Error(
      "No Gemini API keys found. Set GEMINI_API_KEY (and optionally GEMINI_API_KEY_2 through GEMINI_API_KEY_10) in .env.local"
    );
  }

  console.log(`Loaded ${pool.length} API key(s).`);
  return pool;
}

const keyPool = buildKeyPool();

// Reset all key counters at UTC midnight to distribute requests across multiple days
setInterval(() => {
  console.log("Resetting daily usage counters for all keys.");
  for (const entry of keyPool) {
    entry.requestsToday = 0;
    entry.exhausted = false;
  }
}, 24 * 60 * 60 * 1000);

/**
 * Selects the next available API key from the pool.
 *
 * Prioritizes keys with the fewest requests today to load-balance across the pool.
 * Keys marked as exhausted are skipped.
 *
 * @returns {KeyEntry} The selected API key entry
 * @throws {Error} If all keys have reached the daily limit
 */
function getNextKey(): KeyEntry {
  const available = keyPool.filter(
    (e) => !e.exhausted && e.requestsToday < DAILY_LIMIT
  );
  if (available.length === 0) {
    throw new Error(
      "All API keys have reached their daily limit (250 requests each). " +
      "Please wait until tomorrow or add more keys."
    );
  }
  // Select the key with the fewest requests today for even distribution
  available.sort((a, b) => a.requestsToday - b.requestsToday);
  return available[0];
}

/**
 * POST /api/transcribe
 *
 * Transcribes audio segments to text using the Google Gemini API with specialized
 * handling for Palestinian Arabic dialect mixed with English technical terms.
 *
 * Request Body:
 * @param {string} audioData - Base64-encoded audio data (WebM, MP3, WAV, etc.)
 * @param {string} mimeType - MIME type of the audio (e.g., "audio/webm", "audio/mp3")
 * @param {string} fileName - Original file name for reference
 * @param {string} [contextPrompt] - Optional context hint for better transcription accuracy
 *
 * Response:
 * @returns {Object} {
 *   success: true,
 *   fileName: string,
 *   transcript: string (plain text, no markdown),
 *   _debug: { keyUsed: string, requestsUsed: number, requestsRemaining: number }
 * }
 */
app.post("/api/transcribe", async (req, res) => {
  try {
    const { audioData, mimeType, fileName, contextPrompt } = req.body;

    if (!audioData) {
      return res.status(400).json({ error: "Missing audioData in request body." });
    }
    if (!mimeType) {
      return res.status(400).json({ error: "Missing mimeType (e.g., audio/mp3) in request." });
    }

    const entry = getNextKey();
    console.log(`Using ${entry.label} (${entry.requestsToday + 1} requests today)`);

    const audioPart = {
      inlineData: { mimeType, data: audioData },
    };

    // Specialized transcription prompt optimized for Palestinian Arabic + English code-switching
    // with strict formatting requirements for linguistic accuracy
    const promptText = `You are a highly precise, verbatim audio transcriber specialized in processing university machine learning lectures.
The input audio is a short segment (under 30 seconds) bounded by natural speech pauses. It contains code-switched speech alternating between Palestinian Arabic dialect and English technical AI terminology.
Topic / Context: ${contextPrompt || "AI, Machine Learning, and Technology Lecture"}.

Execute the transcription according to these strict rules:
1. NO META-LABELS OR TAGS: Do not include speaker labels (e.g., "[Speaker 1]:", "Doctor:"), timestamps, or introduction markers. Output only the exact words spoken in the audio file as a single continuous line of text.
2. ARABIC DIACRITICS & PHONETICS:
   - Apply accurate grammatical Shadda (ّ), Tanween (ً ٌ ٍ), and Hamzas (أ إ آ ء) where linguistically appropriate.
   - Maintain the raw phonetics of the Palestinian Arabic dialect (e.g., preserve spoken phrases like "رَح نبلش", "هاد", "بدي"). Do NOT translate, modify, or normalize these dialect words into formal Modern Standard Arabic (MSA).
3. ENGLISH TECHNICAL TERMS:
   - Keep specialized computer science and machine learning terms in their native English orthography (e.g., "transformer models", "attention mechanism", "backpropagation").
   - Write all English words in lowercase letters to maintain dataset consistency, unless it is a standard capitalized acronym (e.g., use "NLP", "RNN", "CNN", "VRAM").
4. MATHEMATICAL EXPRESSIONS (NO LATEX):
   - Do NOT use any LaTeX syntax (avoid \\, $, _, ^, or curly braces).
   - Write out mathematical equations using basic keyboard symbols (+, -, =, x, /) or write the terms out in plain text exactly as spoken by the lecturer (e.g., write "W transpose x + b" or "المجموع من i يساوي 1").
Provide ONLY the raw transcript complying with all the above guidelines. Do not write any explanations, notifications, greetings, or markdown fences around the text.`;

    try {
      // Call the Gemini API with the audio and specialized transcription prompt
      const response = await entry.client.models.generateContent({
        model: "gemini-3.5-flash",
        contents: [audioPart, { text: promptText }],
      });

      entry.requestsToday += 1;

      const transcript = response.text || "";
      return res.json({
        success: true,
        fileName,
        transcript,
        _debug: {
          keyUsed: entry.label,
          requestsUsed: entry.requestsToday,
          requestsRemaining: DAILY_LIMIT - entry.requestsToday,
        },
      });

    } catch (apiError: any) {
      // Detect quota/rate limit errors and attempt fallback with another key
      const isQuotaError =
        apiError?.status === 429 ||
        apiError?.message?.toLowerCase().includes("quota") ||
        apiError?.message?.toLowerCase().includes("rate limit");

      if (isQuotaError) {
        console.warn(`${entry.label} hit quota/rate limit — marking exhausted.`);
        entry.exhausted = true;

        // Recursively get the next available key and retry
        const fallback = getNextKey();
        console.log(`Retrying with ${fallback.label}`);

        const retryResponse = await fallback.client.models.generateContent({
          model: "gemini-3.5-flash",
          contents: [audioPart, { text: promptText }],
        });

        fallback.requestsToday += 1;

        const transcript = retryResponse.text || "";
        return res.json({
          success: true,
          fileName,
          transcript,
          _debug: {
            keyUsed: fallback.label,
            requestsUsed: fallback.requestsToday,
            requestsRemaining: DAILY_LIMIT - fallback.requestsToday,
          },
        });
      }

      throw apiError;
    }

  } catch (error: any) {
    console.error("Transcription error:", error);
    res.status(500).json({
      success: false,
      error: error.message || "An unknown error occurred during transcription.",
    });
  }
});

/**
 * GET /api/key-status
 *
 * Returns the current usage status for all API keys in the pool.
 * Useful for monitoring and debugging quota management.
 *
 * Example: http://localhost:3000/api/key-status
 *
 * Response:
 * @returns {Object} {
 *   keys: Array of { label, requestsUsed, requestsRemaining, exhausted },
 *   totals: { requestsUsed, requestsRemaining }
 * }
 */
app.get("/api/key-status", (_req, res) => {
  res.json({
    keys: keyPool.map((e) => ({
      label: e.label,
      requestsUsed: e.requestsToday,
      requestsRemaining: DAILY_LIMIT - e.requestsToday,
      exhausted: e.exhausted,
    })),
    totals: {
      requestsUsed: keyPool.reduce((s, e) => s + e.requestsToday, 0),
      requestsRemaining: keyPool
        .filter((e) => !e.exhausted)
        .reduce((s, e) => s + (DAILY_LIMIT - e.requestsToday), 0),
    },
  });
});

/**
 * Static File & Development Server Configuration
 *
 * In development: Uses Vite middleware for hot module reloading
 * In production: Serves pre-built assets from the dist/ directory
 */
async function configureServer() {
  if (process.env.NODE_ENV !== "production") {
    // Development mode: Use Vite middleware for HMR and fast refresh
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    // Production mode: Serve pre-built static files
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    // Fallback to index.html for client-side routing
    app.get("*", (_req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Lecture Annotation Tool Server running on http://localhost:${PORT}`);
    console.log(`Key status: http://localhost:${PORT}/api/key-status`);
  });
}

configureServer();