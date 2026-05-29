/**
 * Simple Test File for Google Generative AI Client
 *
 * This file demonstrates basic usage of the Google Generative AI SDK.
 * It can be used for quick testing of API connectivity and model functionality.
 *
 * Note: For production usage, API keys should be stored in environment variables,
 * not hardcoded in source files. See server.ts for proper key management.
 */

import { GoogleGenerativeAI } from "@google/generative-ai";

const genAI = new GoogleGenerativeAI("AIzaSyDLJaea_d-dt11QUfXrMx3_T39LtAjkcK8");
const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

/**
 * Test the API connection with a simple prompt.
 */
async function run() {
  try {
    const result = await model.generateContent("Hello");
    console.log(result.response.text());
  } catch (e) {
    console.error("API call failed:", e);
  }
}
run();