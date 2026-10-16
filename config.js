// VERIFY BEFORE DEPLOY: pin the newest @huggingface/transformers release that supports Gemma 4
// (check https://www.npmjs.com/package/@huggingface/transformers and the model card).
export const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0"; // 4.0.1+ adds Gemma 4 (4.0.0 does not); 4.3.0 also fixes the Gemma 4 WebGPU KV cache
export const KOKORO_URL = "https://cdn.jsdelivr.net/npm/kokoro-js@1.2.1/+esm"; // check this too
export const GEMMA_ID = "onnx-community/gemma-4-E2B-it-ONNX";
export const KOKORO_ID = "onnx-community/Kokoro-82M-v1.0-ONNX";
