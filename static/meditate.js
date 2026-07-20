// ── meditate: activation-driven ambient soundscape ─────────────────────────
// Waves are the constant base layer. Seagulls and a foghorn crossfade
// in as the live activation score (from the same metrics pipeline as the
// main dashboard / recovery mode) rises through its zones.

const ZONES = [
  { max: 35,  label: "calm" },
  { max: 65,  label: "slightly activated" },
  { max: 101, label: "activated" },
];

const RAMP_S = 3.5; // crossfade duration per activation update (~every 4s)

let audioCtx    = null;
let masterGain  = null;
let layers      = {};   // name -> { el, gain }
let soundOn     = false;
let ws          = null;
let missingAudioWarned = false;
let baselineOk  = false;
let wsOpen      = false;
let sessionActive = false;

const lastTargets = { waves: 1, seagulls: 0, foghorn: 0 };

function smoothstep(x, lo, hi) {
  const t = Math.min(1, Math.max(0, (x - lo) / (hi - lo)));
  return t * t * (3 - 2 * t);
}

function showMissingAudioHint() {
  if (missingAudioWarned) return;
  missingAudioWarned = true;
  const hint = document.getElementById("meditate-hint");
  if (hint) hint.hidden = false;
}

function initAudio() {
  if (audioCtx) return;
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();

  masterGain = audioCtx.createGain();
  masterGain.gain.value = parseFloat(document.getElementById("meditate-volume").value);
  masterGain.connect(audioCtx.destination);

  ["waves", "seagulls", "foghorn"].forEach(name => {
    const el = new Audio(`/static/${name}.mp3`);
    el.loop = true;
    el.addEventListener("error", showMissingAudioHint);

    const source = audioCtx.createMediaElementSource(el);
    const gain = audioCtx.createGain();
    gain.gain.value = 0;
    source.connect(gain);
    gain.connect(masterGain);

    layers[name] = { el, gain };
  });
}

function updateMeter(name, value) {
  const el = document.getElementById(`layer-fill-${name}`);
  if (el) el.style.width = `${Math.round(value * 100)}%`;
}

function rampLayer(name, value) {
  const layer = layers[name];
  if (!layer || !audioCtx) return;
  const now = audioCtx.currentTime;
  layer.gain.gain.cancelScheduledValues(now);
  layer.gain.gain.setValueAtTime(layer.gain.gain.value, now);
  layer.gain.gain.linearRampToValueAtTime(value, now + RAMP_S);
}

function applyActivation(score) {
  lastTargets.seagulls = smoothstep(score, 25, 45);
  lastTargets.foghorn = smoothstep(score, 55, 75);

  updateMeter("waves", lastTargets.waves);
  updateMeter("seagulls", lastTargets.seagulls);
  updateMeter("foghorn", lastTargets.foghorn);

  if (soundOn) {
    rampLayer("seagulls", lastTargets.seagulls);
    rampLayer("foghorn", lastTargets.foghorn);
  }

  const scoreEl = document.getElementById("meditate-score");
  const zoneEl  = document.getElementById("meditate-zone");
  if (scoreEl) scoreEl.textContent = Math.round(score);
  const zone = ZONES.find(z => score < z.max);
  if (zoneEl && zone) zoneEl.textContent = zone.label;

  const orb = document.getElementById("meditate-orb");
  if (orb) {
    orb.classList.toggle("zone-calm", score < 35);
    orb.classList.toggle("zone-mid", score >= 35 && score < 65);
    orb.classList.toggle("zone-high", score >= 65);
  }
}

function toggleMeditateSound() {
  if (!audioCtx) initAudio();
  if (audioCtx.state === "suspended") audioCtx.resume();

  soundOn = !soundOn;
  const btn = document.getElementById("meditate-toggle");

  if (soundOn) {
    Object.values(layers).forEach(l => l.el.play().catch(showMissingAudioHint));
    rampLayer("waves", lastTargets.waves);
    rampLayer("seagulls", lastTargets.seagulls);
    rampLayer("foghorn", lastTargets.foghorn);
  } else {
    const now = audioCtx.currentTime;
    Object.values(layers).forEach(l => {
      l.gain.gain.cancelScheduledValues(now);
      l.gain.gain.setValueAtTime(l.gain.gain.value, now);
      l.gain.gain.linearRampToValueAtTime(0, now + 1.0);
    });
    setTimeout(() => {
      if (!soundOn) Object.values(layers).forEach(l => l.el.pause());
    }, 1100);
  }

  if (btn) btn.textContent = soundOn ? "pause" : "start";
}

function onMeditateVolumeChange(val) {
  if (masterGain && audioCtx) {
    masterGain.gain.setTargetAtTime(parseFloat(val), audioCtx.currentTime, 0.1);
  }
}

function toggleDarkMode() {
  const html   = document.documentElement;
  const isDark = html.getAttribute("data-theme") === "dark";
  const theme  = isDark ? "light" : "dark";
  html.setAttribute("data-theme", theme);
  localStorage.setItem("theme", theme);
  document.querySelectorAll(".btn-theme").forEach(btn => {
    btn.textContent = theme === "dark" ? "◑" : "◐";
  });
}

function setStatus(text) {
  const zoneEl = document.getElementById("meditate-zone");
  if (zoneEl) zoneEl.textContent = text;
}

function showSessionControls(active) {
  sessionActive = active;
  const beginBtn = document.getElementById("meditate-begin");
  const finishBtn = document.getElementById("meditate-finish");
  if (beginBtn) beginBtn.hidden = active;
  if (finishBtn) finishBtn.hidden = !active;
}

function beginSession() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  // set_standard_baseline always succeeds instantly (no calibration wait), so
  // the UI updates optimistically rather than waiting on baseline_status —
  // which also broadcasts on every reconnect once a baseline has ever been
  // set, and must NOT be used to drive button visibility (see session_init).
  showSessionControls(true);
  resetLiveDisplay();
  ws.send(JSON.stringify({ type: "set_standard_baseline", partner: "A" }));
}

function connectWS() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.onopen = () => {
    wsOpen = true;
  };
  ws.onmessage = (evt) => {
    let msg;
    try { msg = JSON.parse(evt.data); } catch { return; }
    if (msg.type === "session_init") {
      // Authoritative on connect/reconnect: whether a meditation session log
      // is actually open right now. The baseline itself is fixed and stays
      // set indefinitely once first applied, so baseline_status alone can't
      // tell us this.
      showSessionControls(!!msg.meditation_session_active);
      if (!msg.meditation_session_active) setStatus("tap begin to start");
      return;
    }
    if (msg.type === "baseline_status" && msg.partner === "A") {
      baselineOk = !!msg.ok;
      return;
    }
    if (msg.type === "slow" && msg.partner === "A" && typeof msg.activation === "number") {
      applyActivation(msg.activation);
    }
    if (msg.type === "meditation_ended") {
      showSessionControls(false);
      showMeditationResults(msg.stats);
      renderProgress(msg.progress);
      return;
    }
  };
  ws.onclose = () => {
    wsOpen = false;
    setTimeout(connectWS, 1500);
  };
}

// ── end of session: deterministic dashboard + AI narrative ─────────────────

function endMeditation() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  if (soundOn) toggleMeditateSound();
  setStatus("ending session…");
  ws.send(JSON.stringify({ type: "end_meditation" }));
}

function showMeditationResults(stats) {
  document.getElementById("meditate-session-view").hidden = true;
  document.getElementById("meditate-results").hidden = false;

  const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;

  document.getElementById("mr-duration").textContent = mmss(stats.duration_s);
  document.getElementById("mr-act-mean").textContent =
    stats.activation.mean != null ? Math.round(stats.activation.mean) : "—";
  document.getElementById("mr-act-range").textContent =
    stats.activation.min != null
      ? `${Math.round(stats.activation.min)}–${Math.round(stats.activation.max)}` : "—";
  document.getElementById("mr-zone-calm").textContent =
    stats.zone_pct.calm != null ? `${stats.zone_pct.calm}%` : "—";
  document.getElementById("mr-zone-moderate").textContent =
    stats.zone_pct.moderate != null ? `${stats.zone_pct.moderate}%` : "—";
  document.getElementById("mr-zone-activated").textContent =
    stats.zone_pct.activated != null ? `${stats.zone_pct.activated}%` : "—";
  document.getElementById("mr-coherence").textContent =
    stats.coherence.avg != null ? `${stats.coherence.avg} / ${stats.coherence.peak}` : "—";
  document.getElementById("mr-resp-trend").textContent =
    stats.resp_rate.first_third_avg != null
      ? `${stats.resp_rate.first_third_avg} → ${stats.resp_rate.last_third_avg} br/min` : "—";
  document.getElementById("mr-episodes").textContent = stats.activated_episodes;

  drawSessionActivationTrace(document.getElementById("meditate-chart"), stats.activation.series);
  fetchMeditationReport();
}

function renderProgress(progress) {
  const firstNote = document.getElementById("mp-first-note");
  const tiles     = document.getElementById("mp-tiles");
  const chartWrap = document.getElementById("mp-chart-wrap");

  if (!progress || progress.is_first_session) {
    firstNote.hidden = false;
    tiles.hidden = true;
    chartWrap.hidden = true;
    return;
  }

  firstNote.hidden = true;
  tiles.hidden = false;
  chartWrap.hidden = false;

  document.getElementById("mp-count").textContent = progress.session_count;
  const avg = progress.all_time_avg || {};
  document.getElementById("mp-avg-activation").textContent =
    avg.activation_mean != null ? Math.round(avg.activation_mean) : "—";
  document.getElementById("mp-avg-calm").textContent =
    avg.zone_pct_calm != null ? `${avg.zone_pct_calm}%` : "—";

  drawProgressTrace(document.getElementById("meditate-progress-chart"), progress.recent_series);
}

function drawProgressTrace(canvas, series) {
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  if (W === 0 || H === 0) return;
  if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
    canvas.width = W * dpr; canvas.height = H * dpr;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  const BOTTOM_PAD = 16, chartH = H - BOTTOM_PAD;
  const toY = (v) => chartH - (Math.min(Math.max(v, 0), 100) / 100) * chartH;

  ctx.fillStyle = "#fdecea"; ctx.fillRect(0, 0, W, toY(65));
  ctx.fillStyle = "#fef5e7"; ctx.fillRect(0, toY(65), W, toY(35) - toY(65));
  ctx.fillStyle = "#d5f5f0"; ctx.fillRect(0, toY(35), W, chartH - toY(35));
  ctx.strokeStyle = "rgba(0,0,0,0.08)"; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
  [35, 65].forEach(v => {
    ctx.beginPath(); ctx.moveTo(0, toY(v)); ctx.lineTo(W, toY(v)); ctx.stroke();
  });
  ctx.setLineDash([]);

  const vals = (series || []).filter(p => p.activation_mean != null);
  if (vals.length < 2) return;
  const toX = (i) => (i / (vals.length - 1)) * W;

  ctx.beginPath();
  ctx.moveTo(toX(0), toY(vals[0].activation_mean));
  for (let i = 1; i < vals.length; i++) ctx.lineTo(toX(i), toY(vals[i].activation_mean));
  ctx.strokeStyle = "rgba(44,62,80,0.85)"; ctx.lineWidth = 2; ctx.lineJoin = "round";
  ctx.stroke();

  vals.forEach((p, i) => {
    const color = p.activation_mean >= 65 ? "#c0392b" : p.activation_mean >= 35 ? "#e67e22" : "#16a085";
    ctx.beginPath();
    ctx.arc(toX(i), toY(p.activation_mean), 3.5, 0, Math.PI * 2);
    ctx.fillStyle = color; ctx.fill();
  });

  ctx.font = "10px sans-serif"; ctx.fillStyle = "rgba(0,0,0,0.4)"; ctx.textBaseline = "top";
  ctx.textAlign = "left";
  ctx.fillText("earliest of last " + vals.length, toX(0), chartH + 3);
  ctx.textAlign = "right";
  ctx.fillText("most recent", toX(vals.length - 1), chartH + 3);
}

async function fetchMeditationReport() {
  const el  = document.getElementById("meditation-narrative-text");
  const btn = document.getElementById("meditate-new-session");
  el.textContent = "";
  if (btn) btn.disabled = true; // re-enabled only once streaming completes — avoids
                                 // clearing the baseline mid-report (see plan notes)
  try {
    const res = await fetch("/api/meditation_report");
    if (!res.ok) { el.textContent = "Error: " + (await res.text()); return; }
    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      el.textContent += decoder.decode(value, { stream: true });
    }
  } catch (err) {
    el.textContent = "Error: " + err.message;
  } finally {
    if (btn) btn.disabled = false;
  }
}

function newMeditationSession() {
  // The baseline is fixed, so a new session doesn't need re-calibration —
  // just reopen a fresh session log and go straight back to biofeedback.
  document.getElementById("meditate-results").hidden = true;
  document.getElementById("meditate-session-view").hidden = false;
  showSessionControls(true);
  resetLiveDisplay();
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "set_standard_baseline", partner: "A" }));
  }
}

function resetLiveDisplay() {
  // Clears the previous session's leftover score/orb color so "warming up…"
  // doesn't sit next to a stale number that looks like a fresh reading.
  const scoreEl = document.getElementById("meditate-score");
  if (scoreEl) scoreEl.textContent = "—";
  const orb = document.getElementById("meditate-orb");
  if (orb) orb.classList.remove("zone-mid", "zone-high");
  updateMeter("waves", 1);
  updateMeter("seagulls", 0);
  updateMeter("foghorn", 0);
  setStatus("warming up…");
}

function drawSessionActivationTrace(canvas, series) {
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  if (W === 0 || H === 0) return;
  if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
    canvas.width = W * dpr; canvas.height = H * dpr;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  const BOTTOM_PAD = 16, chartH = H - BOTTOM_PAD;
  const toY = (v) => chartH - (Math.min(Math.max(v, 0), 100) / 100) * chartH;

  ctx.fillStyle = "#fdecea"; ctx.fillRect(0, 0, W, toY(65));
  ctx.fillStyle = "#fef5e7"; ctx.fillRect(0, toY(65), W, toY(35) - toY(65));
  ctx.fillStyle = "#d5f5f0"; ctx.fillRect(0, toY(35), W, chartH - toY(35));
  ctx.strokeStyle = "rgba(0,0,0,0.08)"; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
  [35, 65].forEach(v => {
    ctx.beginPath(); ctx.moveTo(0, toY(v)); ctx.lineTo(W, toY(v)); ctx.stroke();
  });
  ctx.setLineDash([]);

  const vals = (series || []).filter(p => p.activation != null);
  if (vals.length < 2) return;
  const total = vals[vals.length - 1].t || 1;
  const toX = (t) => (t / total) * W;

  ctx.beginPath();
  ctx.moveTo(toX(vals[0].t), toY(vals[0].activation));
  for (let i = 1; i < vals.length; i++) ctx.lineTo(toX(vals[i].t), toY(vals[i].activation));
  ctx.strokeStyle = "rgba(44,62,80,0.85)"; ctx.lineWidth = 2; ctx.lineJoin = "round";
  ctx.stroke();

  const last = vals[vals.length - 1];
  const dotColor = last.activation >= 65 ? "#c0392b" : last.activation >= 35 ? "#e67e22" : "#16a085";
  ctx.beginPath();
  ctx.arc(toX(last.t), toY(last.activation), 4, 0, Math.PI * 2);
  ctx.fillStyle = dotColor; ctx.fill();

  const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;
  ctx.font = "10px sans-serif"; ctx.fillStyle = "rgba(0,0,0,0.4)"; ctx.textBaseline = "top";
  [{ label: mmss(0), x: toX(0), align: "left" },
   { label: mmss(total / 2), x: toX(total / 2), align: "center" },
   { label: mmss(total), x: toX(total), align: "right" }]
    .forEach(({ label, x, align }) => { ctx.textAlign = align; ctx.fillText(label, x, chartH + 3); });
}

// ── setup overlay: pick or create a user, then pair a device ───────────────

let _msKnownUsers     = [];
let _msScannedDevices = [];

async function checkMeditateSetupNeeded() {
  const overlay = document.getElementById("meditate-setup-overlay");
  if (!overlay) return;
  try {
    const res  = await fetch("/api/state");
    const data = await res.json();
    if (!data.configured) {
      overlay.hidden = false;
      loadMeditateUsers();
    }
  } catch (e) { /* server not ready yet — overlay stays hidden */ }
}

async function loadMeditateUsers() {
  try {
    const res  = await fetch("/api/users");
    const data = await res.json();
    _msKnownUsers = data.users || [];
  } catch (e) {
    _msKnownUsers = [];
  }
  populateMeditateUserSelect();
}

function populateMeditateUserSelect() {
  const sel = document.getElementById("ms-user-select");
  if (!sel) return;
  const prevValue = sel.value;
  sel.innerHTML = "";

  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "— select or add user —";
  sel.appendChild(placeholder);

  _msKnownUsers.forEach(u => {
    const opt = document.createElement("option");
    opt.value = u.id;
    opt.textContent = u.name;
    sel.appendChild(opt);
  });

  const newOpt = document.createElement("option");
  newOpt.value = "__new__";
  newOpt.textContent = "+ new user…";
  sel.appendChild(newOpt);

  if (prevValue) sel.value = prevValue;

  const deleteBtn = document.getElementById("ms-user-delete");
  if (deleteBtn) deleteBtn.hidden = !sel.value || sel.value === "__new__";
}

function onMeditateUserSelectChange() {
  const sel       = document.getElementById("ms-user-select");
  const nameInput = document.getElementById("ms-name-input");
  const deleteBtn = document.getElementById("ms-user-delete");
  if (sel.value === "__new__") {
    nameInput.hidden = false;
    nameInput.value = "";
    nameInput.focus();
  } else {
    nameInput.hidden = true;
  }
  if (deleteBtn) deleteBtn.hidden = !sel.value || sel.value === "__new__";
}

function resolveMeditateUserName() {
  const sel       = document.getElementById("ms-user-select");
  const nameInput = document.getElementById("ms-name-input");
  if (sel.value === "__new__") return nameInput.value.trim();
  if (sel.value) {
    const user = _msKnownUsers.find(u => u.id === sel.value);
    return user ? user.name : "";
  }
  return "";
}

async function deleteSelectedMeditateUser() {
  const sel = document.getElementById("ms-user-select");
  const id  = sel.value;
  if (!id || id === "__new__") return;
  const user = _msKnownUsers.find(u => u.id === id);
  const label = user ? user.name : id;
  if (!confirm(`Delete the account "${label}"? This also deletes their saved meditation history. This can't be undone.`)) {
    return;
  }
  try {
    const res = await fetch(`/api/users/${encodeURIComponent(id)}`, { method: "DELETE" });
    if (!res.ok) throw new Error(`delete failed (${res.status})`);
  } catch (e) {
    alert("Couldn't delete that user — please try again.");
    return;
  }
  sel.value = "";
  await loadMeditateUsers();
  onMeditateUserSelectChange();
}

async function scanMeditateDevices() {
  const statusEl = document.getElementById("ms-scan-status");
  const btn      = document.getElementById("ms-btn-scan");
  btn.disabled    = true;
  btn.textContent = "scanning… (10s)";
  statusEl.textContent = "";

  try {
    const res     = await fetch("/api/scan");
    const devices = await res.json();
    if (devices.error) throw new Error(devices.error);
    _msScannedDevices = devices;

    const sel = document.getElementById("ms-device-select");
    sel.innerHTML = '<option value="">— select device —</option>';
    devices.forEach(d => {
      const opt = document.createElement("option");
      opt.value = d.address;
      opt.textContent = d.name ? `${d.name}  (${d.address})` : d.address;
      sel.appendChild(opt);
    });
    if (devices.length === 1) sel.value = devices[0].address;

    statusEl.textContent = `${devices.length} device${devices.length !== 1 ? "s" : ""} found`;
    document.getElementById("ms-btn-start").disabled = devices.length === 0;
  } catch (e) {
    statusEl.textContent = "scan failed — check that the sensor is powered on";
  } finally {
    btn.disabled = false;
    btn.textContent = "scan for devices";
  }
}

async function startMeditateSession(simulate) {
  const name    = resolveMeditateUserName() || "You";
  const errorEl = document.getElementById("ms-setup-error");
  errorEl.hidden = true;

  if (simulate) {
    try {
      const res  = await fetch("/api/configure", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ simulate: true, names: [name], bpm: [68] }),
      });
      const data = await res.json();
      if (data.ok) document.getElementById("meditate-setup-overlay").hidden = true;
    } catch (e) {
      errorEl.textContent = "Connection error — is the server running?";
      errorEl.hidden = false;
    }
    return;
  }

  const addr = document.getElementById("ms-device-select").value;
  if (!addr) {
    errorEl.textContent = "Select a device, or use simulate.";
    errorEl.hidden = false;
    return;
  }

  const btn = document.getElementById("ms-btn-start");
  btn.disabled    = true;
  btn.textContent = "connecting…";

  try {
    const res  = await fetch("/api/configure", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ simulate: false, partners: [{ idx: 0, name, address: addr }] }),
    });
    const data = await res.json();
    if (data.ok) {
      document.getElementById("meditate-setup-overlay").hidden = true;
    } else {
      errorEl.textContent = data.error || "Configuration failed.";
      errorEl.hidden = false;
      btn.disabled = false;
      btn.textContent = "begin";
    }
  } catch (e) {
    errorEl.textContent = "Connection error — is the server running?";
    errorEl.hidden = false;
    btn.disabled = false;
    btn.textContent = "begin";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const savedTheme = localStorage.getItem("theme") ||
    (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  if (savedTheme === "dark") {
    document.documentElement.setAttribute("data-theme", "dark");
    document.querySelectorAll(".btn-theme").forEach(btn => btn.textContent = "◑");
  }
  checkMeditateSetupNeeded();
  connectWS();
});
