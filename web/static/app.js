/**
 * app.js — simplified web biofeedback app.
 *
 * One page, three screens (landing / join / room). No recording, no
 * transcript, no dyadic coupling, no meditation — live per-partner metrics
 * and a paced-breathing recovery mode, fed by the browser's own Web
 * Bluetooth connection to a heart rate strap (or a dev-only simulated
 * device) rather than a server-side BLE connection.
 */

"use strict";

const HR_SERVICE_UUID = "0000180d-0000-1000-8000-00805f9b34fb";
const HR_MEASUREMENT_UUID = "00002a37-0000-1000-8000-00805f9b34fb";

// ── state ─────────────────────────────────────────────────────────────────

const state = {
  me: null,          // "A" | "B"
  roomCode: null,
  myName: null,
  mode: "dashboard",
  A: emptyPartner("Partner A"),
  B: emptyPartner("Partner B"),
};

function emptyPartner(name) {
  return {
    name, sensorOnline: false, baselineSet: false,
    mean_hr: null, rmssd: null, hf: null, coherence: null, resp_rate: null,
    activation: null, direction: null, flooded: false, calm_zone_s: 0,
    trace_activation: [],
  };
}

let ws = null;
let btDevice = null;
let simTimer = null;
let breathTimer = null;

// ── screen routing ───────────────────────────────────────────────────────

function showScreen(id) {
  for (const el of document.querySelectorAll(".screen")) el.hidden = el.id !== id;
}

function boot() {
  const params = new URLSearchParams(location.search);
  const room = params.get("room");
  if (room) {
    showScreen("screen-join");
    initJoinScreen(room);
  } else {
    showScreen("screen-landing");
  }
  wireStaticHandlers();
}

// ── landing: create a room ───────────────────────────────────────────────

function wireStaticHandlers() {
  document.getElementById("btn-start").addEventListener("click", onStart);
  document.getElementById("btn-join").addEventListener("click", onJoin);
  document.getElementById("btn-copy-link").addEventListener("click", copyInviteLink);
  document.getElementById("btn-pair-bt").addEventListener("click", pairBluetooth);
  document.getElementById("btn-pair-sim").addEventListener("click", pairSimulated);
  document.getElementById("btn-baseline-A").addEventListener("click", () => resetBaseline("A"));
  document.getElementById("btn-baseline-B").addEventListener("click", () => resetBaseline("B"));
  for (const btn of document.querySelectorAll(".tab-btn")) {
    btn.addEventListener("click", () => switchMode(btn.dataset.mode));
  }
  document.getElementById("breath-pattern").addEventListener("change", restartBreathing);
}

async function onStart() {
  const name = document.getElementById("landing-name").value.trim() || "Partner A";
  const btn = document.getElementById("btn-start");
  btn.disabled = true;
  try {
    const resp = await fetch("/api/rooms", { method: "POST" });
    const data = await resp.json();
    history.replaceState(null, "", `?room=${data.code}`);
    connectRoom(data.code, "A", name);
  } catch (err) {
    btn.disabled = false;
    alert("Could not start a session — check your connection and try again.");
  }
}

// ── join: room code already in URL ───────────────────────────────────────

async function initJoinScreen(code) {
  document.getElementById("join-code-display").textContent = code;
  const statusEl = document.getElementById("join-status");
  const picker = document.getElementById("join-partner-picker");
  let occupied = { A: false, B: false };
  try {
    const resp = await fetch(`/api/rooms/${encodeURIComponent(code)}`);
    if (!resp.ok) throw new Error("not found");
    const data = await resp.json();
    occupied = data.occupied;
    statusEl.textContent = "enter your name and join below.";
  } catch {
    statusEl.textContent = "this session doesn't exist or has ended.";
    document.getElementById("btn-join").disabled = true;
    return;
  }

  let selected = occupied.A ? "B" : "A";
  for (const btn of picker.querySelectorAll(".seg-btn")) {
    const p = btn.dataset.partner;
    if (occupied[p]) btn.disabled = true;
    btn.classList.toggle("selected", p === selected);
    btn.addEventListener("click", () => {
      if (btn.disabled) return;
      selected = p;
      for (const b of picker.querySelectorAll(".seg-btn")) b.classList.toggle("selected", b === btn);
    });
  }
  document.getElementById("btn-join").dataset.code = code;
  document.getElementById("btn-join")._getSelected = () => selected;
}

function onJoin() {
  const btn = document.getElementById("btn-join");
  const code = btn.dataset.code;
  const partner = btn._getSelected ? btn._getSelected() : "B";
  const name = document.getElementById("join-name").value.trim() || `Partner ${partner}`;
  connectRoom(code, partner, name);
}

// ── invite link ───────────────────────────────────────────────────────────

function copyInviteLink() {
  const url = `${location.origin}/?room=${state.roomCode}`;
  navigator.clipboard?.writeText(url).then(() => {
    const btn = document.getElementById("btn-copy-link");
    const prev = btn.textContent;
    btn.textContent = "copied";
    setTimeout(() => (btn.textContent = prev), 1500);
  });
}

// ── WebSocket ─────────────────────────────────────────────────────────────

function connectRoom(code, partner, name) {
  state.roomCode = code;
  state.me = partner;
  state.myName = name;

  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const url = `${proto}//${location.host}/ws?room=${encodeURIComponent(code)}&partner=${partner}&name=${encodeURIComponent(name)}`;
  ws = new WebSocket(url);

  ws.onopen = () => {
    showScreen("screen-room");
    document.getElementById("room-badge").hidden = false;
    document.getElementById("room-code-display").textContent = code;
    document.getElementById("pairing-who").textContent = name;
    updateBaselinePill("A");
    updateBaselinePill("B");
  };

  ws.onmessage = (evt) => {
    let data;
    try { data = JSON.parse(evt.data); } catch { return; }
    handleMessage(data);
  };

  ws.onclose = (evt) => {
    if (evt.code === 4404) {
      alert("That session doesn't exist or has ended.");
      location.href = "/";
    } else if (evt.code === 4409) {
      alert("That partner slot is already connected from another device.");
      location.href = `/?room=${code}`;
    }
  };
}

function send(msg) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
}

function handleMessage(data) {
  const p = data.partner;
  switch (data.type) {
    case "session_init":
      applyNames(data.names);
      if (data.sensor_online) {
        for (const partner of ["A", "B"]) state[partner].sensorOnline = !!data.sensor_online[partner];
        renderPeerStatus(null);
      }
      break;
    case "peer_update":
      applyNames(data.names);
      renderPeerStatus(data.occupied);
      break;
    case "sensor_status":
      if (p) { state[p].sensorOnline = data.online; renderPeerStatus(null); }
      break;
    case "baseline_status":
      if (p) {
        state[p].baselineSet = data.ok;
        updateBaselinePill(p);
      }
      break;
    case "fast":
      if (p) { state[p].mean_hr = data.mean_hr; renderTiles(p); }
      break;
    case "mid":
      if (p) {
        state[p].mean_hr = data.mean_hr;
        state[p].rmssd = data.rmssd;
        state[p].flooded = data.flooded;
        state[p].calm_zone_s = data.calm_zone_s ?? 0;
        renderTiles(p);
        renderFloodBanner();
        trackAutoBaseline(p, data);
      }
      break;
    case "slow":
      if (p) {
        state[p].hf = data.hf;
        state[p].coherence = data.coherence;
        state[p].resp_rate = data.resp_rate;
        state[p].activation = data.activation;
        state[p].direction = data.direction;
        if (data.trace_activation) state[p].trace_activation = data.trace_activation;
        renderTiles(p);
        renderActivation(p);
        renderRecoveryCoherence(p);
        drawActivationTrace();
      }
      break;
  }
}

function applyNames(names) {
  if (!names) return;
  if (names.A) { state.A.name = names.A; }
  if (names.B) { state.B.name = names.B; }
  for (const p of ["A", "B"]) {
    const el = document.getElementById(`name-${p}`);
    if (el) el.textContent = state[p].name;
    const legend = document.getElementById(`legend-${p}`);
    if (legend) legend.textContent = state[p].name;
    const rec = document.getElementById(`rec-name-${p}`);
    if (rec) rec.textContent = state[p].name;
  }
}

function renderPeerStatus(occupied) {
  const el = document.getElementById("peer-status");
  const other = state.me === "A" ? "B" : "A";
  const parts = [];
  if (occupied) state._occupied = occupied;
  const occ = state._occupied || { A: true, B: true };
  if (!occ[other]) {
    parts.push(`waiting for ${state[other].name || "your partner"} to join…`);
  } else if (!state[other].sensorOnline) {
    parts.push(`${state[other].name} has joined — connecting their monitor…`);
  } else {
    parts.push(`${state[other].name}'s monitor is streaming.`);
  }
  el.textContent = parts.join(" ");
  document.getElementById("mode-tabs").hidden = false;
  document.getElementById("view-dashboard").hidden = state.mode !== "dashboard";
}

// ── mode tabs ─────────────────────────────────────────────────────────────

function switchMode(mode) {
  state.mode = mode;
  for (const btn of document.querySelectorAll(".tab-btn")) {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  }
  document.getElementById("view-dashboard").hidden = mode !== "dashboard";
  document.getElementById("view-recovery").hidden = mode !== "recovery";
  send({ type: "set_mode", mode: mode === "recovery" ? "recovery" : "conversation" });
  if (mode === "recovery") startBreathing(); else stopBreathing();
}

// ── tile rendering ────────────────────────────────────────────────────────

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function renderTiles(p) {
  const s = state[p];
  setText(`hr-${p}`, s.mean_hr != null ? Math.round(s.mean_hr) : "—");
  setText(`rmssd-${p}`, s.rmssd != null ? Math.round(s.rmssd) : "—");
  setText(`coh-${p}`, s.coherence != null ? s.coherence.toFixed(2) : "—");
  setText(`resp-${p}`, s.resp_rate != null ? Math.round(s.resp_rate) : "—");
}

function renderActivation(p) {
  const s = state[p];
  const bar = document.getElementById(`act-bar-${p}`);
  const num = document.getElementById(`act-num-${p}`);
  const dir = document.getElementById(`act-dir-${p}`);
  if (s.activation == null) {
    num.textContent = "—"; bar.style.width = "0%";
    dir.textContent = ""; return;
  }
  const v = Math.round(s.activation);
  num.textContent = v;
  bar.style.width = `${v}%`;
  bar.style.background = v < 35 ? "var(--calm)" : v < 65 ? "var(--moderate)" : "var(--high)";
  dir.textContent = s.direction === "rising" ? "↑" : s.direction === "falling" ? "↓" : "";
}

// Resolves which socket (if any) this tab can send set_baseline/clear_baseline
// for a given partner on — the server resolves the target from the socket
// itself, so a message can only ever affect the partner slot that socket
// claimed. That's your own primary `ws` for state.me, or `ws2` when this tab
// is also dev-simulating the other partner. Neither exists for a real
// partner's own separate device, so their controls correctly stay disabled.
function baselineSendFor(p) {
  if (p === state.me) return send;
  if (ws2 && ws2.readyState === WebSocket.OPEN) return (msg) => ws2.send(JSON.stringify(msg));
  return null;
}

function updateBaselinePill(p) {
  const pill = document.getElementById(`baseline-${p}`);
  const btn = document.getElementById(`btn-baseline-${p}`);
  pill.textContent = state[p].baselineSet ? "baseline set" : "no baseline";
  pill.classList.toggle("set", state[p].baselineSet);
  btn.textContent = state[p].baselineSet ? "reset baseline" : "detecting automatically…";
  btn.disabled = !state[p].baselineSet || !baselineSendFor(p);
  btn.classList.toggle("set", state[p].baselineSet);
}

function renderFloodBanner() {
  const anyFlooded = state.A.flooded || state.B.flooded;
  const banner = document.getElementById("flood-banner");
  banner.hidden = !anyFlooded;
  if (anyFlooded) {
    const who = [state.A.flooded && state.A.name, state.B.flooded && state.B.name].filter(Boolean).join(" and ");
    document.getElementById("flood-text").textContent = `${who} past flooding threshold — consider a break.`;
  }
}

function resetBaseline(p) {
  const sendFn = baselineSendFor(p);
  if (!sendFn) return;
  sendFn({ type: "clear_baseline" });
}

// ── automatic baseline capture ────────────────────────────────────────────
// A meaningful baseline needs to be measured while the person is in a settled
// resting state — capturing it during a spike or a motion artifact would
// anchor "normal" on an elevated reading and throw off every activation
// score after it. So instead of a fixed timer, this waits for a run of
// consecutive good-quality mid-tier samples (each ~5s apart) whose HR stays
// within a tight band, then fires the same set_baseline the manual button
// used to. Any quality dropout or HR swing resets the run so it only fires
// once things have actually settled.
const AUTO_BASELINE_MIN_SAMPLES = 8;      // ~40s of settled mid-tier readings
const AUTO_BASELINE_HR_SPREAD_MAX = 8;    // bpm range allowed across that window
const AUTO_BASELINE_MIN_QUALITY = 0.7;    // matches processor.py's motion "warn" threshold

function trackAutoBaseline(p, mid) {
  if (state[p].baselineSet) return;
  const sendFn = baselineSendFor(p);
  if (!sendFn) return;

  const s = state[p];
  s.hrHistory = s.hrHistory || [];

  const qualityOk = mid.signal_quality == null || mid.signal_quality >= AUTO_BASELINE_MIN_QUALITY;
  if (mid.mean_hr != null && qualityOk) {
    s.hrHistory.push(mid.mean_hr);
    if (s.hrHistory.length > AUTO_BASELINE_MIN_SAMPLES) s.hrHistory.shift();
  } else {
    s.hrHistory = []; // motion or a dropout breaks the settled run
    return;
  }

  if (s.hrHistory.length >= AUTO_BASELINE_MIN_SAMPLES) {
    const spread = Math.max(...s.hrHistory) - Math.min(...s.hrHistory);
    if (spread <= AUTO_BASELINE_HR_SPREAD_MAX) {
      sendFn({ type: "set_baseline" });
      s.hrHistory = [];
    }
  }
}

// ── activation trace canvas ──────────────────────────────────────────────

function drawActivationTrace() {
  const canvas = document.getElementById("activation-trace");
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight || 140;
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr; canvas.height = h * dpr;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const styles = getComputedStyle(document.documentElement);
  ctx.strokeStyle = styles.getPropertyValue("--border").trim();
  ctx.lineWidth = 1;
  for (const frac of [0, 0.35, 0.65, 1]) {
    const y = h - frac * h;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }

  drawTraceLine(ctx, state.A.trace_activation, w, h, styles.getPropertyValue("--partner-a").trim());
  drawTraceLine(ctx, state.B.trace_activation, w, h, styles.getPropertyValue("--partner-b").trim());
}

// processor.py emits one activation trace point per slow tick (~10s) and
// keeps the last 600s, so ~60 points span a full 10-minute window.
const TRACE_WINDOW_POINTS = 60;

function drawTraceLine(ctx, trace, w, h, color) {
  if (!trace || trace.length < 2) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  const n = trace.length;
  trace.forEach((v, i) => {
    // newest point anchored at the right edge; older points step left at a
    // fixed spacing, so the trace fills in from the right and only reaches
    // the left edge once a full ~10 minutes of data has accumulated.
    const slotsFromRight = (n - 1) - i;
    const x = w - (slotsFromRight / (TRACE_WINDOW_POINTS - 1)) * w;
    const y = h - (Math.max(0, Math.min(100, v)) / 100) * h;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
}

window.addEventListener("resize", drawActivationTrace);

// ── recovery / breathing ─────────────────────────────────────────────────

const BREATH_PATTERNS = {
  coherence: [{ phase: "in", s: 5.5 }, { phase: "out", s: 5.5 }],
  box: [{ phase: "in", s: 4 }, { phase: "hold", s: 4 }, { phase: "out", s: 4 }, { phase: "hold", s: 4 }],
  "4-7-8": [{ phase: "in", s: 4 }, { phase: "hold", s: 7 }, { phase: "out", s: 8 }],
};

function startBreathing() {
  stopBreathing();
  runBreathStep(0);
}

function restartBreathing() {
  if (state.mode === "recovery") startBreathing();
}

function runBreathStep(stepIdx) {
  const patternName = document.getElementById("breath-pattern").value;
  const pattern = BREATH_PATTERNS[patternName] || BREATH_PATTERNS.coherence;
  const step = pattern[stepIdx % pattern.length];
  const circle = document.getElementById("breath-circle");
  const label = document.getElementById("breath-phase");

  circle.style.transitionDuration = `${step.s}s`;
  if (step.phase === "in") { circle.classList.add("expand"); label.textContent = "breathe in"; }
  else if (step.phase === "out") { circle.classList.remove("expand"); label.textContent = "breathe out"; }
  else { label.textContent = "hold"; }

  breathTimer = setTimeout(() => runBreathStep(stepIdx + 1), step.s * 1000);
}

function stopBreathing() {
  if (breathTimer) { clearTimeout(breathTimer); breathTimer = null; }
}

function renderRecoveryCoherence(p) {
  const bar = document.getElementById(`coh-bar-${p}`);
  const val = document.getElementById(`coh-val-${p}`);
  const c = state[p].coherence;
  if (c == null) { val.textContent = "—"; bar.style.width = "0%"; return; }
  val.textContent = c.toFixed(2);
  bar.style.width = `${Math.min(100, c * 100)}%`;
}

// ── Bluetooth pairing ─────────────────────────────────────────────────────

function parseHrMeasurement(dataView) {
  const flags = dataView.getUint8(0);
  let idx = 1;
  if (flags & 0x01) { idx += 2; } else { idx += 1; } // bpm field, unused server-side
  if (flags & 0x08) { idx += 2; } // energy expended, skip
  const rr = [];
  if (flags & 0x10) {
    while (idx + 1 < dataView.byteLength) {
      const raw = dataView.getUint16(idx, true);
      rr.push((raw * 1000.0) / 1024.0);
      idx += 2;
    }
  }
  return rr;
}

async function pairBluetooth() {
  const statusEl = document.getElementById("pairing-status");
  if (!navigator.bluetooth) {
    statusEl.textContent = "Web Bluetooth isn't available in this browser — try Chrome or Edge over HTTPS.";
    return;
  }
  try {
    statusEl.textContent = "select your strap in the browser dialog…";
    const device = await navigator.bluetooth.requestDevice({
      filters: [{ services: [HR_SERVICE_UUID] }],
    });
    btDevice = device;
    const server = await device.gatt.connect();
    const service = await server.getPrimaryService(HR_SERVICE_UUID);
    const char = await service.getCharacteristic(HR_MEASUREMENT_UUID);
    await char.startNotifications();
    char.addEventListener("characteristicvaluechanged", (evt) => {
      const rr = parseHrMeasurement(evt.target.value);
      if (rr.length) send({ type: "rr_sample", rr_ms: rr });
    });
    device.addEventListener("gattserverdisconnected", onSensorDisconnected);
    onSensorConnected(device.name || "heart rate strap");
  } catch (err) {
    statusEl.textContent = `pairing cancelled or failed (${err.message || err}).`;
  }
}

// Generates a plausible, slowly-drifting RR-interval stream and hands each
// sample to `sendFn` — shared by "simulate my own strap" and "also simulate
// my partner's strap" so both drive the exact same rr_sample message path
// a real Web Bluetooth device would.
function startFakeRrStream(sendFn) {
  let bpm = 66 + Math.random() * 14;
  return setInterval(() => {
    bpm += (Math.random() - 0.5) * 2;
    bpm = Math.max(50, Math.min(100, bpm));
    const rrMs = (60000 / bpm) * (0.95 + Math.random() * 0.1);
    sendFn({ type: "rr_sample", rr_ms: [rrMs] });
  }, 800);
}

function pairSimulated() {
  simTimer = startFakeRrStream(send);
  onSensorConnected("simulated strap (dev)");
  simulateOtherPartnerIfFree();
}

// Dev convenience: if the other partner hasn't joined from a real device yet,
// open a second, silent WebSocket connection as that partner and simulate
// their strap too, so a single tab can preview the whole two-partner
// dashboard. The main connection already receives broadcast metrics for
// both partners, so this second socket only ever sends — nothing needs to
// read from it.
let ws2 = null;
let sim2Timer = null;

async function simulateOtherPartnerIfFree() {
  const other = state.me === "A" ? "B" : "A";
  try {
    const resp = await fetch(`/api/rooms/${encodeURIComponent(state.roomCode)}`);
    const data = await resp.json();
    if (data.occupied[other]) return; // a real partner is already connected — don't steal their slot
  } catch {
    return;
  }

  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const label = other === "A" ? "Partner A" : "Partner B";
  const url = `${proto}//${location.host}/ws?room=${encodeURIComponent(state.roomCode)}&partner=${other}&name=${encodeURIComponent(label)}`;
  ws2 = new WebSocket(url);
  ws2.onopen = () => {
    ws2.send(JSON.stringify({ type: "sensor_connect" }));
    sim2Timer = startFakeRrStream((msg) => ws2.send(JSON.stringify(msg)));
    updateBaselinePill(other); // reset button for the simulated partner is now controllable from this tab
    const statusEl = document.getElementById("pairing-status");
    statusEl.textContent = `${statusEl.textContent} also simulating ${label} (dev).`;
    // baseline capture for this simulated partner is driven by the same
    // trackAutoBaseline() stability check as a real device, via ws2 — see
    // the "mid" case in handleMessage.
  };
}

function onSensorConnected(label) {
  document.getElementById("pairing-status").textContent = `connected: ${label}`;
  document.getElementById("pairing-panel").querySelector("h2").textContent = "Heart rate monitor connected";
  document.getElementById("btn-pair-bt").disabled = true;
  document.getElementById("btn-pair-sim").disabled = true;
  send({ type: "sensor_connect" });
}

function onSensorDisconnected() {
  document.getElementById("pairing-status").textContent = "monitor disconnected.";
  send({ type: "sensor_disconnect" });
  btDevice = null;
}

window.addEventListener("beforeunload", () => {
  if (simTimer) clearInterval(simTimer);
  if (sim2Timer) clearInterval(sim2Timer);
  if (ws2) ws2.close();
});

boot();
