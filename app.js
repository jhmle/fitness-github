const $ = (id) => document.getElementById(id);
const status = (s) => ($("status").textContent = s);
if (!navigator.gpu) {
  status("This needs a WebGPU browser (recent desktop Chrome or Edge).");
  $("load").disabled = true; throw new Error("no WebGPU");
}
const gemma = new Worker("gemma.worker.js", { type: "module" });
const kokoro = new Worker("kokoro.worker.js", { type: "module" });
let ready = 0, busy = false, rec = null; // rec = {stream, mr, chunks, frames, timer}

$("load").onclick = () => {
  $("load").disabled = true; status("Downloading models…");
  gemma.postMessage({ type: "load", dtype: new URLSearchParams(location.search).get("dtype") || "q4f16" }); kokoro.postMessage({ type: "load" });
};
const onReady = () => { if (++ready === 2) { status("Ready."); ["mic","camBtn","up","txt"].forEach(i => $(i).disabled = false); } };
const fail = (e) => status("Error: " + e.message);

// ---- Audio playback queue (24 kHz chunks, scheduled back to back)
let actx, nextT = 0, sources = [];
function play(samples, rate) {
  actx ??= new AudioContext();
  const b = actx.createBuffer(1, samples.length, rate); b.copyToChannel(samples, 0);
  const s = actx.createBufferSource(); s.buffer = b; s.connect(actx.destination);
  nextT = Math.max(nextT, actx.currentTime + 0.05); s.start(nextT); nextT += b.duration;
  sources.push(s); s.onended = () => (sources = sources.filter(x => x !== s));
}
// ---- Sentence chunker
let buf = "";
function feed(t, final) {
  buf += t; $("out").textContent += t;
  let m; while ((m = buf.match(/^([\s\S]*?[.!?\n])\s/)) ) { speak(m[1]); buf = buf.slice(m[0].length); }
  if (final && buf.trim()) { speak(buf); buf = ""; }
}
const speak = (text) => text.trim() && kokoro.postMessage({ type: "speak", text, voice: $("voice").value });

gemma.onmessage = ({ data: d }) => {
  if (d.type === "progress") $("pg").value = d.pct;
  else if (d.type === "ready") onReady();
  else if (d.type === "token") feed(d.text, false);
  else if (d.type === "done") { feed("", true); busy = false; status("Done."); }
  else if (d.type === "error") { busy = false; fail(d); }
};
kokoro.onmessage = ({ data: d }) => {
  if (d.type === "progress") $("pk").value = d.pct;
  else if (d.type === "ready") { $("voice").innerHTML = d.voices.map(v => `<option>${v}</option>`).join(""); $("voice").value = d.voices.includes("af_heart") ? "af_heart" : d.voices[0]; onReady(); }
  else if (d.type === "audio") play(d.samples, d.rate);
  else if (d.type === "error") fail(d);
};

// ---- Capture helpers
async function decodeAudio(blob) {
  const ctx = new AudioContext({ sampleRate: 16000 }); // Chrome resamples on decode
  const ab = await ctx.decodeAudioData(await blob.arrayBuffer()); await ctx.close();
  // mix all channels down to mono (channel 0 alone can be silent on some devices/files)
  const n = Math.min(ab.length, 16000 * 30); // cap ~30 s
  const out = new Float32Array(n);
  for (let c = 0; c < ab.numberOfChannels; c++) {
    const ch = ab.getChannelData(c);
    for (let i = 0; i < n; i++) out[i] += ch[i] / ab.numberOfChannels;
  }
  let peak = 0; for (let i = 0; i < n; i++) peak = Math.max(peak, Math.abs(out[i]));
  console.log("decoded audio", { seconds: (n / 16000).toFixed(1), peak: peak.toFixed(4), channels: ab.numberOfChannels, rate: ab.sampleRate });
  if (peak < 0.005) return { samples: null, peak, seconds: n / 16000 }; // effectively silence
  if (peak < 0.3) { const g = 0.7 / peak; for (let i = 0; i < n; i++) out[i] *= g; } // normalise quiet mics
  return { samples: out, peak, seconds: n / 16000 };
}
function grab(video) {
  const c = document.createElement("canvas"), k = 448 / Math.max(video.videoWidth, video.videoHeight);
  c.width = video.videoWidth * k; c.height = video.videoHeight * k;
  c.getContext("2d").drawImage(video, 0, 0, c.width, c.height);
  return new Promise(r => c.toBlob(r, "image/jpeg", 0.8));
}
async function run(audio, frames) {
  const m = $("mode").value; if (m === "audio") frames = []; if (m === "image") audio = null;
  console.log("sending", { audioSamples: audio?.length ?? 0, frames: frames.length, mode: m });
  if (busy) return; busy = true; $("out").textContent = ""; buf = ""; nextT = 0;
  status("Thinking…");
  gemma.postMessage({ type: "run", audio, frames, text: $("prompt").value });
}
async function start(useVideo, useAudio) {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: useAudio && { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true }, video: useVideo });
  rec = { stream, chunks: [], frames: [] };
  if (useVideo) { $("cam").srcObject = stream; await $("cam").play();
    rec.timer = setInterval(async () => rec.frames.length < 6 && rec.frames.push(await grab($("cam"))), 1000); }
  rec.mr = new MediaRecorder(new MediaStream(stream.getAudioTracks()));
  rec.mr.ondataavailable = e => rec.chunks.push(e.data); rec.mr.start();
  status("Recording… press Stop."); const me = rec; setTimeout(() => rec === me && stopRec(), 30000);
}
async function stopRec() {
  const r = rec; rec = null; clearInterval(r.timer);
  await new Promise(res => { r.mr.onstop = res; r.mr.stop(); });
  r.stream.getTracks().forEach(t => t.stop()); $("cam").srcObject = null;
  let audio = null;
  if (r.chunks.length) {
    try {
      const d = await decodeAudio(new Blob(r.chunks, { type: r.mr.mimeType }));
      if (!d.samples) { status(`Mic recorded only silence (peak ${d.peak.toFixed(4)}). Check your input device / browser mic permission.`); r.frames = []; return; }
      audio = d.samples; status(`Captured ${d.seconds.toFixed(1)} s of audio.`);
    } catch (e) { status("Could not decode recording: " + e.message); r.frames = []; return; }
  }
  await run(audio, r.frames); r.frames = []; // release references
}
$("mic").onclick = () => start(false, true);
$("camBtn").onclick = () => start(true, true);
$("stop").onclick = () => { if (rec) stopRec(); sources.forEach(s => { s.onended = null; try { s.stop(); } catch {} }); sources = []; nextT = 0; };
$("up").onclick = () => $("file").click();
$("file").onchange = async (e) => {
  const f = e.target.files[0]; if (!f) return; status("Reading file…");
  let audio = null, frames = [];
  try { audio = (await decodeAudio(f)).samples; } catch { status("Could not decode audio track."); }
  if (f.type.startsWith("video/")) {
    const v = $("cam"); v.src = URL.createObjectURL(f); await v.play().catch(()=>{}); v.pause();
    const n = Math.min(6, Math.max(1, Math.floor(v.duration)));
    for (let i = 0; i < n; i++) { v.currentTime = (v.duration * i) / n; await new Promise(r => (v.onseeked = r)); frames.push(await grab(v)); }
    URL.revokeObjectURL(v.src);
  }
  await run(audio, frames); e.target.value = "";
};

$("txt").onclick = () => run(null, []);
