/**
 * app.js — shared JS for Mode A (index.html) and Mode B (mode_b.html)
 *
 * WebSocket auto-reconnect, state management, DOM updates, canvas drawing.
 */

"use strict";

// ── state ─────────────────────────────────────────────────────────────────────

const state = {
  A: {
    name: "Partner A",
    mean_hr: null,
    rmssd: null,
    hf: null,
    coherence: null,
    resp_rate: null,
    activation: null,
    direction: null,
    confidence: null,
    state_description: null,
    flooded: false,
    hr_baseline_pct: null,
    trace_hr: [],
    trace_times: [],
    baseline_set: false,
  },
  B: {
    name: "Partner B",
    mean_hr: null,
    rmssd: null,
    hf: null,
    coherence: null,
    resp_rate: null,
    activation: null,
    direction: null,
    confidence: null,
    state_description: null,
    flooded: false,
    hr_baseline_pct: null,
    trace_hr: [],
    trace_times: [],
    baseline_set: false,
  },
  dyadic: {
    peak_r: null,
    lag_s: null,
    phase: null,
    leader: null,
  },
};

// ── WebSocket connection ──────────────────────────────────────────────────────

let ws = null;

function connectWS() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const url = `${proto}//${location.host}/ws`;
  ws = new WebSocket(url);

  ws.onopen = () => {
    console.log("[ws] connected");
  };

  ws.onmessage = (evt) => {
    let data;
    try { data = JSON.parse(evt.data); } catch { return; }
    handleMessage(data);
  };

  ws.onclose = () => {
    console.log("[ws] closed, reconnecting in 1s");
    setTimeout(connectWS, 1000);
  };

  ws.onerror = () => ws.close();
}

// ── message dispatcher ────────────────────────────────────────────────────────

function handleMessage(data) {
  const p = data.partner; // "A" or "B"

  switch (data.type) {

    case "fast":
      if (p) {
        state[p].mean_hr = data.mean_hr;
        updateHeroHR(p);
      }
      break;

    case "mid":
      if (p) {
        state[p].mean_hr       = data.mean_hr;
        state[p].rmssd         = data.rmssd;
        state[p].flooded       = data.flooded;
        state[p].hr_baseline_pct = data.hr_baseline_pct;
        if (data.trace_hr) state[p].trace_hr = data.trace_hr;
        if (data.trace_times) state[p].trace_times = data.trace_times;
        updateHeroHR(p);
        updateMidTiles(p);
        updateFloodState(p);
        redrawSparkline(p);
      }
      break;

    case "slow":
      if (p) {
        state[p].hf                = data.hf;
        state[p].coherence         = data.coherence;
        state[p].resp_rate         = data.resp_rate;
        state[p].activation        = data.activation;
        state[p].direction         = data.direction;
        state[p].confidence        = data.confidence;
        state[p].state_description = data.state_description;
        updateSlowTiles(p);
        updateCoherenceBarModeB(p);
      }
      break;

    case "dyadic":
      state.dyadic.peak_r = data.peak_r;
      state.dyadic.lag_s  = data.lag_s;
      state.dyadic.phase  = data.phase;
      state.dyadic.leader = data.leader;
      updateDyadicPanel();
      redrawDualTrace();
      break;

    case "session_init":
      state.A.name = (data.names && data.names.A) || state.A.name;
      state.B.name = (data.names && data.names.B) || state.B.name;
      applyPartnerNames();
      break;

    case "baseline_status":
      if (p) {
        state[p].baseline_set = data.ok;
        state[p].name = data.name || state[p].name;
        applyPartnerNames();
        updateBaselineIndicator();
      }
      break;

    case "pong":
      break;
  }
}

// ── canvas drawing ────────────────────────────────────────────────────────────

/**
 * Draw a simple line sparkline on a canvas.
 * @param {HTMLCanvasElement} canvas
 * @param {number[]} hrArray
 * @param {string} color  CSS color string
 */
function drawSparkline(canvas, hrArray, color) {
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth;
  const H = canvas.clientHeight;
  if (W === 0 || H === 0) return;

  // resize backing store if needed
  if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
    canvas.width  = W * dpr;
    canvas.height = H * dpr;
  }

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  if (!hrArray || hrArray.length < 2) return;

  const Y_MIN = 45, Y_MAX = 185;
  const toX = (i) => (i / (hrArray.length - 1)) * W;
  const toY = (v) => H - ((Math.min(Math.max(v, Y_MIN), Y_MAX) - Y_MIN) / (Y_MAX - Y_MIN)) * H;

  ctx.beginPath();
  ctx.moveTo(toX(0), toY(hrArray[0]));
  for (let i = 1; i < hrArray.length; i++) {
    ctx.lineTo(toX(i), toY(hrArray[i]));
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.lineJoin = "round";
  ctx.stroke();
}

/**
 * Draw two overlaid HR traces on the same canvas.
 */
function drawDualTrace(canvas, hrA, hrB, colorA, colorB) {
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth;
  const H = canvas.clientHeight;
  if (W === 0 || H === 0) return;

  if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
    canvas.width  = W * dpr;
    canvas.height = H * dpr;
  }

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  const Y_MIN = 45, Y_MAX = 185;
  const drawLine = (arr, color) => {
    if (!arr || arr.length < 2) return;
    const toX = (i) => (i / (arr.length - 1)) * W;
    const toY = (v) => H - ((Math.min(Math.max(v, Y_MIN), Y_MAX) - Y_MIN) / (Y_MAX - Y_MIN)) * H;
    ctx.beginPath();
    ctx.moveTo(toX(0), toY(arr[0]));
    for (let i = 1; i < arr.length; i++) ctx.lineTo(toX(i), toY(arr[i]));
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = "round";
    ctx.globalAlpha = 0.8;
    ctx.stroke();
    ctx.globalAlpha = 1.0;
  };

  drawLine(hrA, colorA);
  drawLine(hrB, colorB);
}

// ── Mode A DOM updates ────────────────────────────────────────────────────────

function updateHeroHR(p) {
  const s = state[p];
  const el = document.getElementById(`hr-value-${p}`);
  if (!el) return;
  el.textContent = s.mean_hr !== null ? Math.round(s.mean_hr) : "—";
  el.className = "hr-value" + (s.flooded ? " flooded" : "");

  const deltaEl = document.getElementById(`hr-delta-${p}`);
  if (deltaEl) {
    if (s.hr_baseline_pct !== null && s.baseline_set) {
      const sign = s.hr_baseline_pct >= 0 ? "+" : "";
      deltaEl.textContent = `${sign}${s.hr_baseline_pct}% vs baseline`;
      deltaEl.className = "hr-delta " + (s.hr_baseline_pct > 0 ? "over" : "under");
    } else {
      deltaEl.textContent = "baseline not set";
      deltaEl.className = "hr-delta";
    }
  }
}

function updateMidTiles(p) {
  const s = state[p];
  const rmssdEl = document.getElementById(`rmssd-val-${p}`);
  if (rmssdEl) rmssdEl.textContent = s.rmssd !== null ? s.rmssd.toFixed(1) : "—";
}

function updateSlowTiles(p) {
  const s = state[p];

  const actEl = document.getElementById(`act-score-${p}`);
  if (actEl) actEl.textContent = s.activation !== null ? Math.round(s.activation) : "—";

  const dirEl = document.getElementById(`act-dir-${p}`);
  if (dirEl) dirEl.textContent = s.direction || "";

  const descEl = document.getElementById(`state-desc-${p}`);
  if (descEl) descEl.textContent = s.state_description || "";

  const hfEl = document.getElementById(`hf-val-${p}`);
  if (hfEl) hfEl.textContent = s.hf !== null ? s.hf.toExponential(2) : "—";

  const rrEl = document.getElementById(`resp-val-${p}`);
  if (rrEl) rrEl.textContent = s.resp_rate !== null ? s.resp_rate.toFixed(1) : "—";

  const cohEl = document.getElementById(`coh-val-${p}`);
  const cohBarEl = document.getElementById(`coh-bar-${p}`);
  if (s.coherence !== null) {
    const pct = Math.min(s.coherence / 3.0, 1.0) * 100;
    if (cohEl)    cohEl.textContent = s.coherence.toFixed(2);
    if (cohBarEl) cohBarEl.style.width = pct.toFixed(1) + "%";
  } else {
    if (cohEl)    cohEl.textContent = "—";
    if (cohBarEl) cohBarEl.style.width = "0%";
  }
}

function updateFloodState(p) {
  const s = state[p];
  const panel = document.getElementById(`panel-${p}`);
  if (panel) {
    if (s.flooded) {
      panel.classList.add("flooded");
    } else {
      panel.classList.remove("flooded");
    }
  }

  const badge = document.getElementById(`state-badge-${p}`);
  if (badge) {
    badge.textContent = s.flooded ? "flooded" : "regulated";
    badge.className = "state-badge " + (s.flooded ? "flooded" : "regulated");
  }

  // flood banner
  const banner = document.getElementById("flood-banner");
  if (banner) {
    const anyFlooded = state.A.flooded || state.B.flooded;
    banner.classList.toggle("visible", anyFlooded);
    const textEl = document.getElementById("flood-banner-text");
    if (textEl) {
      const floodedNames = [];
      if (state.A.flooded) floodedNames.push(state.A.name || "Partner A");
      if (state.B.flooded) floodedNames.push(state.B.name || "Partner B");
      if (floodedNames.length) {
        textEl.textContent =
          `⚠ ${floodedNames.join(" and ")} ${floodedNames.length > 1 ? "are" : "is"} ` +
          `flooded. Suggest a ~20-minute break with active distraction.`;
      }
    }
  }
}

function redrawSparkline(p) {
  const canvas = document.getElementById(`sparkline-${p}`);
  const color = p === "A" ? "#c0392b" : "#1a6baa";
  drawSparkline(canvas, state[p].trace_hr, color);
}

function updateDyadicPanel() {
  const d = state.dyadic;

  const rEl = document.getElementById("dyadic-r");
  if (rEl) rEl.textContent = d.peak_r !== null ? d.peak_r.toFixed(3) : "—";

  const lagEl = document.getElementById("dyadic-lag");
  if (lagEl) {
    if (d.lag_s !== null && d.leader) {
      const absLag = Math.abs(d.lag_s).toFixed(1);
      lagEl.textContent = `${absLag}s · ${d.leader} leads`;
    } else {
      lagEl.textContent = "—";
    }
  }

  const phaseEl = document.getElementById("dyadic-phase");
  if (phaseEl) phaseEl.textContent = d.phase || "—";
}

function redrawDualTrace() {
  const canvas = document.getElementById("dual-trace");
  drawDualTrace(canvas, state.A.trace_hr, state.B.trace_hr, "#c0392b", "#1a6baa");
}

function applyPartnerNames() {
  const nameA  = document.getElementById("name-A");
  const nameB  = document.getElementById("name-B");
  const labelA = document.getElementById("label-a");
  const labelB = document.getElementById("label-b");
  if (nameA)  nameA.textContent  = state.A.name;
  if (nameB)  nameB.textContent  = state.B.name;
  if (labelA) labelA.textContent = state.A.name;
  if (labelB) labelB.textContent = state.B.name;
}

function updateBaselineIndicator() {
  const ind = document.getElementById("baseline-indicator");
  if (!ind) return;
  const bothSet = state.A.baseline_set && state.B.baseline_set;
  const eitherSet = state.A.baseline_set || state.B.baseline_set;
  if (bothSet) {
    ind.textContent = "baseline set (both)";
    ind.className = "baseline-indicator set";
  } else if (eitherSet) {
    const who = state.A.baseline_set ? (state.A.name || "Partner A") : (state.B.name || "Partner B");
    ind.textContent = `baseline set (${who})`;
    ind.className = "baseline-indicator set";
  } else {
    ind.textContent = "no baseline";
    ind.className = "baseline-indicator";
  }
}

// ── Mode B DOM updates ────────────────────────────────────────────────────────

function updateCoherenceBarModeB(p) {
  const fillEl = document.getElementById(`coh-fill-b-${p}`);
  const valEl  = document.getElementById(`coh-val-b-${p}`);
  const s = state[p];
  if (s.coherence !== null) {
    const pct = Math.min(s.coherence / 3.0, 1.0) * 100;
    if (fillEl) fillEl.style.width = pct.toFixed(1) + "%";
    if (valEl)  valEl.textContent  = s.coherence.toFixed(2);
  } else {
    if (fillEl) fillEl.style.width = "0%";
    if (valEl)  valEl.textContent  = "—";
  }
}

// ── baseline button ───────────────────────────────────────────────────────────

function setBaseline(partner) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "set_baseline", partner }));
  }
}

// ── session clock (Mode A) ────────────────────────────────────────────────────

function startSessionClock() {
  const el = document.getElementById("session-clock");
  if (!el) return;
  const start = Date.now();
  function tick() {
    const elapsed = Math.floor((Date.now() - start) / 1000);
    const h = String(Math.floor(elapsed / 3600)).padStart(2, "0");
    const m = String(Math.floor((elapsed % 3600) / 60)).padStart(2, "0");
    const s = String(elapsed % 60).padStart(2, "0");
    el.textContent = `${h}:${m}:${s}`;
  }
  tick();
  setInterval(tick, 1000);
}

// ── break clock (Mode B) ─────────────────────────────────────────────────────

function startBreakClock() {
  const el = document.getElementById("break-clock");
  if (!el) return;
  const start = Date.now();
  function tick() {
    const elapsed = Math.floor((Date.now() - start) / 1000);
    const m = String(Math.floor(elapsed / 60)).padStart(2, "0");
    const s = String(elapsed % 60).padStart(2, "0");
    el.textContent = `${m}:${s}`;
  }
  tick();
  setInterval(tick, 1000);
}

// ── reduced motion guard ─────────────────────────────────────────────────────

function applyReducedMotion() {
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduced) {
    // CSS @media handles animation; nothing extra needed from JS
    document.documentElement.setAttribute("data-reduced-motion", "true");
  }
}

// ── init ──────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  applyReducedMotion();
  connectWS();
  startSessionClock();   // no-op if element absent (Mode B)
  startBreakClock();     // no-op if element absent (Mode A)

  // resize canvases on window resize
  window.addEventListener("resize", () => {
    redrawSparkline("A");
    redrawSparkline("B");
    redrawDualTrace();
  });
});
