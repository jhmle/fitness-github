import { TRANSFORMERS_URL, GEMMA_ID } from "./config.js";
// API below follows the official model card for onnx-community/gemma-4-E2B-it-ONNX
// (Gemma4ForConditionalGeneration, webgpu, processor(prompt, image, audio)).
let T, processor, model;
const post = (m) => self.postMessage(m);

const SYSTEM_PROMPT =
  "You are a helpful assistant. You hear the user through a microphone and reply with short, friendly, " +
  "conversational text. Every user turn may include a recording of what you just heard and an image from " +
  "the camera; treat both as part of the user's input.";

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "load") {
      T = await import(TRANSFORMERS_URL);
      const progress_callback = (p) => p.status === "progress_total" && post({ type: "progress", pct: p.progress });
      processor = await T.AutoProcessor.from_pretrained(GEMMA_ID);
      // FIX: the audio encoder must NOT be loaded at q4f16 (it produces unusable embeddings, so the
      // model "can't hear"). Google's reference demo loads each component separately like this.
      const q = data.dtype || "q4f16"; // applies to embeddings + decoder only
      model = await T.Gemma4ForConditionalGeneration.from_pretrained(GEMMA_ID, {
        dtype: {
          audio_encoder: "q4",
          vision_encoder: "q4",
          embed_tokens: q,
          decoder_model_merged: q,
        },
        device: "webgpu",
        progress_callback,
      });
      post({ type: "ready" });
    } else if (data.type === "run") {
      const { frames, text } = data;
      const audio = data.audio ? new Float32Array(data.audio) : null; // audio: Float32Array@16k | null, frames: Blob[]
      // We send the latest frame only (the model card shows a single image).
      const image = frames.length ? await T.RawImage.fromBlob(frames[frames.length - 1]) : null;
      const content = [
        ...(image ? [{ type: "image" }] : []),
        ...(audio ? [{ type: "audio" }] : []),
        { type: "text", text: text || (audio ? "Listen to the audio and respond to what the speaker says." : "") },
      ];
      const prompt = processor.apply_chat_template(
        [
          { role: "system", content: SYSTEM_PROMPT },
          { role: "user", content },
        ],
        { enable_thinking: false, add_generation_prompt: true },
      );
      const inputs = await processor(prompt, image, audio, { add_special_tokens: false });
      console.log("processor outputs:", Object.keys(inputs), "prompt:", prompt);
      if (audio) {
        // Verify the model will really "see" the audio: placeholder tokens + sane features.
        const aid = processor.tokenizer.convert_tokens_to_ids("<|audio|>");
        const nTok = Array.from(inputs.input_ids.data, Number).filter((x) => x === aid).length;
        const fk = Object.keys(inputs).find((k) => /feature/i.test(k) && !/mask/i.test(k));
        if (!fk) throw new Error("Processor produced no audio features. Keys: " + Object.keys(inputs).join(", "));
        let bad = 0, mx = 0;
        for (const v of inputs[fk].data) { const n = Number(v); if (!Number.isFinite(n)) bad++; else mx = Math.max(mx, Math.abs(n)); }
        const info = `audio: ${(audio.length / 16000).toFixed(1)}s | placeholder tokens in prompt: ${nTok} | ${fk} dims [${inputs[fk].dims}] | NaN/Inf: ${bad} | max|x|: ${mx.toFixed(2)}`;
        console.log(info); post({ type: "debug", text: info });
        if (nTok === 0) throw new Error("The prompt contains no audio tokens, so the model cannot hear anything. (" + info + ")");
        if (bad) throw new Error("Audio features contain NaN/Inf. (" + info + ")");
      }
      const streamer = new T.TextStreamer(processor.tokenizer, {
        skip_prompt: true, skip_special_tokens: true,
        callback_function: (t) => post({ type: "token", text: t }),
      });
      await model.generate({ ...inputs, max_new_tokens: 256, do_sample: false, streamer });
      post({ type: "done" });
    }
  } catch (e) { post({ type: "error", message: String(e.message || e) }); }
};
