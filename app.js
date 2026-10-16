const $ = (id) => document.getElementById(id);
const setStatus = (text, kind = "idle") => { const s = $("status"); s.textContent = text; s.className = "pill pill-" + kind; };

if (!navigator.gpu) {
  setStatus("This needs a WebGPU browser (recent desktop Chrome or Edge).", "error");
  $("load").disabled = true;
  throw new Error("no WebGPU");
}

const gemma = new Worker("gemma.worker.js", { type: "module" });
const kokoro = new Worker("kokoro.worker.js", { type: "module" });

const TARGET = 16000;          // Gemma wants 16 kHz mono
const MAX_SECONDS = 30;        // Gemma's audio limit
let ready = 0, busy = false, starting = false, rec = null;
let speakNow = true, captureInfo = "", heardUrl = null;

function refresh() {
  const ok = ready === 2;
  $("mic").disabled = !ok || busy;
  for (const id of ["up", "txt", "sample"]) $(id).disabled = !ok || busy || !!rec || starting;
}
const setBusy = (b) => { busy = b; refresh(); };

// ---------------------------------------------------------------- loading
$("load").onclick = () => {
  $("load").disabled = true;
  setStatus("Downloading models…", "loading");
  gemma.postMessage({ type: "load", dtype: new URLSearchParams(location.search).get("dtype") || "q4f16" });
  kokoro.postMessage({ type: "load" });
};
const onReady = () => {
  if (++ready < 2) return;
  $("setup").classList.add("done");
  setStatus("Ready. Press the mic and speak.", "ready");
  refresh();
};
const fail = (e) => setStatus("Error: " + e.message, "error");

// ---------------------------------------------------------------- speech playback (24 kHz chunks, back to back)
let actx, nextT = 0, sources = [];
function play(samples, rate) {
  actx ??= new AudioContext();
  const b = actx.createBuffer(1, samples.length, rate); b.copyToChannel(samples, 0);
  const s = actx.createBufferSource(); s.buffer = b; s.connect(actx.destination);
  nextT = Math.max(nextT, actx.currentTime + 0.05); s.start(nextT); nextT += b.duration;
  sources.push(s); s.onended = () => (sources = sources.filter((x) => x !== s));
}
function stopSpeech() {
  sources.forEach((s) => { s.onended = null; try { s.stop(); } catch {} });
  sources = []; nextT = 0;
}
$("stopSpeak").onclick = stopSpeech;

// ---------------------------------------------------------------- sentence chunker -> TTS
let buf = "";
const cleanForSpeech = (t) => t.replace(/[*_#`>~]+/g, "").replace(/\s+/g, " ").trim();
function speak(text) {
  const c = cleanForSpeech(text);
  if (c.length > 1 && /[a-z0-9]/i.test(c)) kokoro.postMessage({ type: "speak", text: c, voice: $("voice").value });
}
function feed(t, final) {
  buf += t; $("out").textContent += t;
  if (speakNow) {
    let m;
    while ((m = buf.match(/^([\s\S]*?[.!?\n])\s/))) { speak(m[1]); buf = buf.slice(m[0].length); }
    if (final && buf.trim()) speak(buf);
  }
  if (final) buf = "";
}

gemma.onmessage = ({ data: d }) => {
  if (d.type === "progress") $("pg").value = d.pct;
  else if (d.type === "ready") onReady();
  else if (d.type === "debug") { $("dbg").textContent = (captureInfo ? captureInfo + "\n" : "") + d.text; console.log(d.text); }
  else if (d.type === "token") feed(d.text, false);
  else if (d.type === "done") { feed("", true); setBusy(false); setStatus("Done. Press the mic to talk again.", "ready"); }
  else if (d.type === "error") { setBusy(false); fail(d); }
};
kokoro.onmessage = ({ data: d }) => {
  if (d.type === "progress") $("pk").value = d.pct;
  else if (d.type === "ready") {
    $("voice").innerHTML = d.voices.map((v) => `<option>${v}</option>`).join("");
    $("voice").value = d.voices.includes("af_heart") ? "af_heart" : d.voices[0];
    onReady();
  }
  else if (d.type === "audio") play(d.samples, d.rate);
  else if (d.type === "error") fail(d);
};

// ---------------------------------------------------------------- audio helpers
function resample(input, from, to = TARGET) {
  if (from === to) return input;
  const ratio = from / to, n = Math.floor(input.length / ratio), out = new Float32Array(n);
  for (let i = 0; i < n; i++) {           // box filter: average the source samples under each output sample
    const a = Math.floor(i * ratio), b = Math.min(Math.ceil((i + 1) * ratio), input.length);
    let s = 0; for (let j = a; j < b; j++) s += input[j];
    out[i] = b > a ? s / (b - a) : 0;
  }
  return out;
}
function concat(chunks, total) {
  const out = new Float32Array(total); let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}
function prepare(x) {                      // cap length, reject silence, lift quiet mics (gain capped)
  if (x.length > TARGET * MAX_SECONDS) x = x.slice(0, TARGET * MAX_SECONDS);
  let peak = 0; for (let i = 0; i < x.length; i++) { const a = Math.abs(x[i]); if (a > peak) peak = a; }
  if (peak < 0.01) return { samples: null, peak, gain: 1 };
  let gain = 1;
  if (peak < 0.3) { gain = Math.min(0.7 / peak, 20); x = x.map((v) => v * gain); }
  return { samples: x, peak, gain };
}
function toWav(samples, rate = TARGET) {
  const buf = new ArrayBuffer(44 + samples.length * 2), v = new DataView(buf);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); w(8, "WAVE"); w(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  w(36, "data"); v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}
function showHeard(samples) {
  if (heardUrl) URL.revokeObjectURL(heardUrl);
  heardUrl = URL.createObjectURL(toWav(samples));
  $("heard").src = heardUrl; $("heardWrap").hidden = false;
}
async function decodeBlob(blob) {          // for uploaded files / sample clip
  const ctx = new AudioContext({ sampleRate: TARGET });
  const ab = await ctx.decodeAudioData(await blob.arrayBuffer()); await ctx.close();
  const out = new Float32Array(ab.length);
  for (let c = 0; c < ab.numberOfChannels; c++) {
    const ch = ab.getChannelData(c);
    for (let i = 0; i < out.length; i++) out[i] += ch[i] / ab.numberOfChannels;
  }
  return out;
}
function grab(video) {
  const c = document.createElement("canvas"), k = 448 / Math.max(video.videoWidth, video.videoHeight);
  c.width = Math.round(video.videoWidth * k); c.height = Math.round(video.videoHeight * k);
  c.getContext("2d").drawImage(video, 0, 0, c.width, c.height);
  return new Promise((r) => c.toBlob(r, "image/jpeg", 0.85));
}

// ---------------------------------------------------------------- send to Gemma
async function run(audio, frames) {
  if (busy) return;
  const mode = $("mode").value, task = $("task").value;
  if (mode === "audio" || task === "transcribe") frames = [];
  if (mode === "image") audio = null;
  const text = $("prompt").value.trim();
  if (!audio && !frames.length && !text) { setStatus("Nothing to send. Record, upload, or type something.", "error"); return; }
  setBusy(true); stopSpeech();
  $("out").textContent = ""; $("dbg").textContent = captureInfo; buf = "";
  speakNow = $("speakOn").checked && task === "chat";
  setStatus("Thinking…", "busy");
  gemma.postMessage({ type: "run", audio, frames, text, task });
}
async function sendClip(mono, label, frames = []) {
  const p = prepare(mono);
  showHeard(p.samples || mono);
  captureInfo = `${label}: ${(Math.min(mono.length, TARGET * MAX_SECONDS) / TARGET).toFixed(1)}s | peak ${p.peak.toFixed(3)} | gain x${p.gain.toFixed(1)}`;
  $("dbg").textContent = captureInfo;
  if (!p.samples) {
    setStatus(`The recording is silent (peak ${p.peak.toFixed(4)}). Check your input device and mic permission.`, "error");
    return;
  }
  await run(p.samples, frames);
}

// ---------------------------------------------------------------- waveform
const cv = $("wave"), g = cv.getContext("2d");
function fit() {
  const r = window.devicePixelRatio || 1;
  cv.width = Math.max(1, Math.round(cv.clientWidth * r)); cv.height = Math.max(1, Math.round(cv.clientHeight * r));
  return r;
}
function idleWave() {
  const r = fit(); g.clearRect(0, 0, cv.width, cv.height);
  g.strokeStyle = "#cfd6e4"; g.lineWidth = 2 * r; g.lineCap = "round";
  g.beginPath(); g.moveTo(10 * r, cv.height / 2); g.lineTo(cv.width - 10 * r, cv.height / 2); g.stroke();
}
window.addEventListener("resize", () => { if (!rec) idleWave(); });
idleWave();

// ---------------------------------------------------------------- microphone capture (raw PCM, no MediaRecorder)
function setMic(recording) {
  $("mic").classList.toggle("recording", recording);
  $("micLabel").textContent = recording ? "Stop & send · 0:00" : "Start talking";
}
function loop() {
  if (!rec) return;
  rec.raf = requestAnimationFrame(loop);
  const r = fit(); rec.an.getByteTimeDomainData(rec.data);
  let lvl = 0; for (const v of rec.data) lvl = Math.max(lvl, Math.abs(v - 128) / 128);
  rec.peak = Math.max(rec.peak, lvl);
  g.clearRect(0, 0, cv.width, cv.height);
  g.strokeStyle = lvl > 0.02 ? "#4f46e5" : "#cfd6e4"; g.lineWidth = 2 * r; g.lineJoin = "round"; g.beginPath();
  for (let i = 0; i < rec.data.length; i++) {
    const x = (i / (rec.data.length - 1)) * cv.width, y = (rec.data[i] / 255) * cv.height;
    i ? g.lineTo(x, y) : g.moveTo(x, y);
  }
  g.stroke();
  const secs = (performance.now() - rec.t0) / 1000;
  $("micLabel").textContent = `Stop & send · ${Math.floor(secs / 60)}:${String(Math.floor(secs % 60)).padStart(2, "0")}`;
  if (secs > 2 && rec.peak < 0.02) setStatus("Recording, but no sound detected. Check your mic.", "error");
  else setStatus("Recording. Press Stop when you're done.", "rec");
}
async function startRec() {
  if (busy || rec || starting) return;
  starting = true; refresh();
  stopSpeech();                                   // don't let our own TTS bleed into the mic
  const useCam = $("useCam").checked;
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      video: useCam ? { facingMode: "user" } : false,
    });
  } catch (e) {
    starting = false; refresh();
    setStatus("Mic/camera access failed: " + e.message, "error");
    return;
  }
  const ctx = new AudioContext(); await ctx.resume();
  const src = ctx.createMediaStreamSource(stream);
  const an = ctx.createAnalyser(); an.fftSize = 1024;
  const sp = ctx.createScriptProcessor(4096, 1, 1);
  const mute = ctx.createGain(); mute.gain.value = 0;   // keeps the processor running without playing the mic back
  const chunks = []; let total = 0;
  sp.onaudioprocess = (e) => {
    if (!rec) return;
    const d = new Float32Array(e.inputBuffer.getChannelData(0));
    chunks.push(d); total += d.length; rec.total = total;
  };
  src.connect(an); src.connect(sp); sp.connect(mute); mute.connect(ctx.destination);

  if (useCam) {
    $("cam").srcObject = stream;
    try { await $("cam").play(); } catch {}
    $("viewport").classList.add("live");
  }
  rec = { stream, ctx, src, an, sp, mute, chunks, total: 0, useCam, data: new Uint8Array(an.fftSize), peak: 0, t0: performance.now(), raf: 0 };
  rec.timeout = setTimeout(() => rec && stopRec(), MAX_SECONDS * 1000);
  starting = false; setMic(true); refresh(); loop();
}
async function stopRec() {
  const r = rec; if (!r) return; rec = null;
  clearTimeout(r.timeout); cancelAnimationFrame(r.raf);
  let frames = [];
  if (r.useCam && $("cam").videoWidth) { try { frames = [await grab($("cam"))]; } catch {} }
  r.sp.onaudioprocess = null;
  try { r.src.disconnect(); r.sp.disconnect(); r.mute.disconnect(); } catch {}
  r.stream.getTracks().forEach((t) => t.stop());
  $("cam").srcObject = null; $("viewport").classList.remove("live");
  const rate = r.ctx.sampleRate; await r.ctx.close();
  setMic(false); idleWave(); refresh();

  const mono = resample(concat(r.chunks, r.total), rate, TARGET);
  if (mono.length < TARGET * 0.4) { setStatus("That was too short. Hold on a bit longer and try again.", "error"); return; }
  await sendClip(mono, `mic (${rate} Hz, resampled to 16 kHz)`, frames);
}
$("mic").onclick = () => (rec ? stopRec() : startRec());

// ---------------------------------------------------------------- upload / sample / text
async function frameFromFile(f) {
  const v = $("cam"), url = URL.createObjectURL(f);
  v.src = url;
  await new Promise((res, rej) => { v.onloadeddata = res; v.onerror = rej; });
  v.currentTime = isFinite(v.duration) ? v.duration / 2 : 0;
  await new Promise((res) => (v.onseeked = res));
  const frame = await grab(v);
  URL.revokeObjectURL(url); v.removeAttribute("src"); v.load();
  return frame;
}
$("up").onclick = () => $("file").click();
$("file").onchange = async (e) => {
  const f = e.target.files[0]; e.target.value = ""; if (!f) return;
  setStatus("Reading file…", "busy");
  let mono = null, frames = [];
  try { mono = await decodeBlob(f); } catch { /* no audio track */ }
  if (f.type.startsWith("video/")) { try { frames = [await frameFromFile(f)]; } catch {} }
  if (mono) await sendClip(mono, f.name, frames);
  else if (frames.length) await run(null, frames);
  else setStatus("Couldn't read audio or video from that file.", "error");
};
$("txt").onclick = () => run(null, []);

// Known-good clip: if this works but your mic does not, the problem is the recording, not the model.
$("sample").onclick = async () => {
  try {
    setStatus("Fetching sample audio…", "busy");
    const r = await fetch("https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav");
    await sendClip(await decodeBlob(await r.blob()), "sample clip");
  } catch (e) { setStatus("Sample failed: " + e.message, "error"); }
};

refresh();
