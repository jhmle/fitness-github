import { TRANSFORMERS_URL, GEMMA_ID } from "./config.js";
let T, processor, model;
const post = (m) => self.postMessage(m);

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "load") {
      T = await import(TRANSFORMERS_URL);
      const progress_callback = (p) => p.status === "progress" && post({ type: "progress", file: p.file, pct: p.progress });
      processor = await T.AutoProcessor.from_pretrained(GEMMA_ID, { progress_callback });
      // UNVERIFIED: class name and per-module dtype keys. Gemma 3n's ONNX card used per-module q4;
      // compare with the Gemma 4 card's Transformers.js example and adjust.
      model = await T.AutoModelForImageTextToText.from_pretrained(GEMMA_ID, {
        device: "webgpu", dtype: "q4f16", progress_callback,
      });
      post({ type: "ready" });
    } else if (data.type === "run") {
      const { audio, frames, text } = data; // audio: Float32Array@16k | null, frames: Blob[]
      const images = await Promise.all(frames.map((b) => T.RawImage.fromBlob(b)));
      // Media goes BEFORE text, per the model card.
      const content = [
        ...images.map(() => ({ type: "image" })),
        ...(audio ? [{ type: "audio" }] : []),
        { type: "text", text },
      ];
      const prompt = processor.apply_chat_template([{ role: "user", content }], { add_generation_prompt: true });
      // UNVERIFIED: multi-image array and argument order (copied from the Gemma 3n example).
      const inputs = await processor(prompt, images.length ? images : null, audio, { add_special_tokens: false });
      const streamer = new T.TextStreamer(processor.tokenizer, {
        skip_prompt: true, skip_special_tokens: true,
        callback_function: (t) => post({ type: "token", text: t }),
      });
      await model.generate({ ...inputs, max_new_tokens: 256, do_sample: false, streamer });
      post({ type: "done" });
    }
  } catch (e) { post({ type: "error", message: String(e.message || e) }); }
};
