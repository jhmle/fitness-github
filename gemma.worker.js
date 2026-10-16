import { TRANSFORMERS_URL, GEMMA_ID } from "./config.js";
// API below follows the official model card for onnx-community/gemma-4-E2B-it-ONNX
// (Gemma4ForConditionalGeneration, q4f16, webgpu, processor(prompt, image, audio)).
let T, processor, model;
const post = (m) => self.postMessage(m);

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "load") {
      T = await import(TRANSFORMERS_URL);
      const progress_callback = (p) => p.status === "progress_total" && post({ type: "progress", pct: p.progress });
      processor = await T.AutoProcessor.from_pretrained(GEMMA_ID);
      model = await T.Gemma4ForConditionalGeneration.from_pretrained(GEMMA_ID, {
        dtype: data.dtype || "q4f16", device: "webgpu", progress_callback,
      });
      post({ type: "ready" });
    } else if (data.type === "run") {
      const { frames, text } = data;
      const audio = data.audio ? new Float32Array(data.audio) : null; // audio: Float32Array@16k | null, frames: Blob[]
      // UNVERIFIED: the model card shows ONE image only. We send the latest frame; multi-image is untested.
      const image = frames.length ? await T.RawImage.fromBlob(frames[frames.length - 1]) : null;
      const content = [
        ...(image ? [{ type: "image" }] : []),
        ...(audio ? [{ type: "audio" }] : []),
        { type: "text", text: text || (audio ? "Listen to the audio and respond to what the speaker says." : "") },
      ];
      const prompt = processor.apply_chat_template([{ role: "user", content }], {
        enable_thinking: false, add_generation_prompt: true,
      });
      const inputs = await processor(prompt, image, audio, { add_special_tokens: false });
      console.log("processor outputs:", Object.keys(inputs), "prompt:", prompt);
      if (audio && !Object.keys(inputs).some(k => /audio|feature/i.test(k)))
        throw new Error("Audio was sent but the processor produced no audio features (keys: " + Object.keys(inputs).join(", ") + ")");
      const streamer = new T.TextStreamer(processor.tokenizer, {
        skip_prompt: true, skip_special_tokens: true,
        callback_function: (t) => post({ type: "token", text: t }),
      });
      await model.generate({ ...inputs, max_new_tokens: 256, do_sample: false, streamer });
      post({ type: "done" });
    }
  } catch (e) { post({ type: "error", message: String(e.message || e) }); }
};
