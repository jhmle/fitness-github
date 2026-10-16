import { TRANSFORMERS_URL, GEMMA_ID } from "./config.js";
// Follows the official model card for onnx-community/gemma-4-E2B-it-ONNX
// (Gemma4ForConditionalGeneration, webgpu, processor(prompt, image, audio)).
let T, processor, model;
const post = (m) => self.postMessage(m);

const SYSTEM_PROMPT =
  "You are a helpful voice assistant. You hear the user through a microphone and reply with short, friendly, " +
  "conversational text (one to three sentences). Whatever you write is read aloud, so write plain spoken words " +
  "with no markdown, lists or stage directions. Each user turn may include a recording of what you just heard " +
  "and an image from the camera; treat both as the user's input and respond to what they said.";

// Official ASR prompt from the model card.
const ASR_PROMPT =
  "Transcribe the following speech segment in English into English text.\n\n" +
  "Follow these specific instructions for formatting the answer:\n" +
  "* Only output the transcription, with no newlines.\n" +
  "* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three.";

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "load") {
      T = await import(TRANSFORMERS_URL);
      const progress_callback = (p) => p.status === "progress_total" && post({ type: "progress", pct: p.progress });
      processor = await T.AutoProcessor.from_pretrained(GEMMA_ID);
      // The audio encoder must NOT be loaded at q4f16 (unusable embeddings). Each component is set separately.
      const q = data.dtype || "q4f16"; // applies to embeddings + decoder only
      model = await T.Gemma4ForConditionalGeneration.from_pretrained(GEMMA_ID, {
        dtype: { audio_encoder: "q4", vision_encoder: "q4", embed_tokens: q, decoder_model_merged: q },
        device: "webgpu",
        progress_callback,
      });
      post({ type: "ready" });
    } else if (data.type === "run") {
      const { frames = [], text = "", task = "chat" } = data;
      const audio = data.audio ? new Float32Array(data.audio) : null; // Float32Array @ 16 kHz | null
      // Only the latest frame is sent (the model card shows a single image).
      const image = task !== "transcribe" && frames.length ? await T.RawImage.fromBlob(frames[frames.length - 1]) : null;

      const userText = task === "transcribe" && audio ? ASR_PROMPT : text;
      const content = [
        ...(image ? [{ type: "image" }] : []),   // media goes BEFORE text, per the model card
        ...(audio ? [{ type: "audio" }] : []),
        ...(userText ? [{ type: "text", text: userText }] : []),
      ];
      if (!content.length) throw new Error("Nothing to send to the model.");

      const messages = [];
      if (task === "chat") messages.push({ role: "system", content: SYSTEM_PROMPT });
      messages.push({ role: "user", content });

      const prompt = processor.apply_chat_template(messages, { enable_thinking: false, add_generation_prompt: true });
      const inputs = await processor(prompt, image, audio, { add_special_tokens: false });

      if (audio) {
        // Sanity-check that the model will really "hear" the audio.
        const aid = processor.tokenizer.convert_tokens_to_ids("<|audio|>");
        const nTok = Array.from(inputs.input_ids.data, Number).filter((x) => x === aid).length;
        const fk = Object.keys(inputs).find((k) => /feature/i.test(k) && !/mask/i.test(k));
        if (!fk) throw new Error("Processor produced no audio features. Keys: " + Object.keys(inputs).join(", "));
        let bad = 0, mx = 0;
        for (const v of inputs[fk].data) { const n = Number(v); if (!Number.isFinite(n)) bad++; else mx = Math.max(mx, Math.abs(n)); }
        const info = `model input: ${(audio.length / 16000).toFixed(1)}s audio | audio tokens: ${nTok} | ${fk} dims [${inputs[fk].dims}] | NaN/Inf: ${bad} | max|x|: ${mx.toFixed(2)}`;
        post({ type: "debug", text: info });
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
