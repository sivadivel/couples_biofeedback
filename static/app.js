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
    connectedAt: null,
    sensorOnline: false,
    signalQuality: null,
    batteryLevel: null,
    mean_hr: null,
    rmssd: null,
    hf: null,
    coherence: null,
    prevCoherence: null,
    resp_rate: null,
    activation: null,
    direction: null,
    confidence: null,
    state_description: null,
    trace_activation: [],
    flooded: false,
    hr_baseline_pct: null,
    trace_hr: [],
    trace_times: [],
    baseline_set: false,
    calm_zone_s: 0,
  },
  B: {
    name: "Partner B",
    connectedAt: null,
    sensorOnline: false,
    signalQuality: null,
    batteryLevel: null,
    mean_hr: null,
    rmssd: null,
    hf: null,
    coherence: null,
    prevCoherence: null,
    resp_rate: null,
    activation: null,
    direction: null,
    confidence: null,
    state_description: null,
    trace_activation: [],
    flooded: false,
    hr_baseline_pct: null,
    trace_hr: [],
    trace_times: [],
    baseline_set: false,
    calm_zone_s: 0,
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
let singlePartnerMode = false;

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
        if (!state[p].connectedAt) state[p].connectedAt = Date.now();
        state[p].mean_hr = data.mean_hr;
        updateHeroHR(p);
      }
      break;

    case "mid":
      if (p) {
        state[p].mean_hr         = data.mean_hr;
        state[p].rmssd           = data.rmssd;
        state[p].flooded         = data.flooded;
        state[p].hr_baseline_pct = data.hr_baseline_pct;
        state[p].signalQuality   = data.signal_quality ?? null;
        if (data.trace_hr) state[p].trace_hr = data.trace_hr;
        if (data.trace_times) state[p].trace_times = data.trace_times;
        updateHeroHR(p);
        updateMidTiles(p);
        updateFloodState(p);
        updateSignalQuality(p);
        redrawDualTrace();
      }
      break;

    case "sensor_status":
      if (p) {
        state[p].sensorOnline = data.online;
        updateSensorStatus(p);
      }
      break;

    case "battery":
      if (p) {
        state[p].batteryLevel = data.level;
        updateBatteryDisplay(p);
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
        state[p].calm_zone_s       = data.calm_zone_s ?? 0;
        if (data.trace_activation) state[p].trace_activation = data.trace_activation;
        updateSlowTiles(p);
        updateCoherenceBarModeB(p);
        redrawActivationTrace(p);
        updateActivationModeB(p);
        updateCalmZone(p);
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
      if (data.single_partner) {
        singlePartnerMode = true;
        const panelB = document.getElementById("panel-B");
        if (panelB) panelB.style.display = "none";
        const dyadic = document.querySelector(".dyadic-panel");
        if (dyadic) dyadic.style.display = "none";
        // mode A: hide partner B coherence row
        const labelB = document.getElementById("label-b");
        if (labelB) labelB.closest(".coherence-row") && (labelB.closest(".coherence-row").style.display = "none");
        // mode B: hide partner B section
        const recSectionB = document.getElementById("rec-section-B");
        if (recSectionB) recSectionB.style.display = "none";
        // mode B: hide partner B coherence row in breath section
        const bCohRow = document.getElementById("breath-coh-row-B");
        if (bCohRow) bCohRow.style.display = "none";
      }
      break;

    case "baseline_status":
      if (p) {
        state[p].baseline_set = data.ok;
        state[p].name = data.name || state[p].name;
        applyPartnerNames();
        updateBaselineIndicator();
      }
      break;

    case "transcript":
      appendTranscriptEvent(data);
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
  if (hfEl) hfEl.textContent = s.hf !== null ? Math.round(s.hf * 1e4) : "—";

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

  const interpEl = document.getElementById("dyadic-interp");
  if (interpEl) {
    const r = d.peak_r;
    const bothFlooded = state.A.flooded && state.B.flooded;
    const anyFlooded  = state.A.flooded || state.B.flooded;
    let text = "";
    if (r !== null) {
      if (r < 0.2) {
        text = "Physiologies moving independently.";
      } else if (r >= 0.4 && bothFlooded) {
        text = "High coupling while both flooded — locked in conflict, not co-regulation.";
      } else if (r >= 0.4 && d.phase === "in-phase" && !anyFlooded) {
        text = "Synchronized and both settled — co-regulation signal.";
      } else if (r >= 0.4 && d.phase === "anti-phase") {
        text = "Synchronized but moving in opposite directions.";
      } else if (r >= 0.4) {
        text = "Physiologies closely coupled.";
      } else {
        text = "Weak coupling — physiologies loosely linked.";
      }
    }
    interpEl.textContent = text;
  }
}

function redrawDualTrace() {
  const canvas = document.getElementById("dual-trace");
  drawDualTrace(canvas, state.A.trace_hr, state.B.trace_hr, "#c0392b", "#1a6baa");
}

function drawActivationTrace(canvas, values) {
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

  const toY = (v) => H - (Math.min(Math.max(v, 0), 100) / 100) * H;

  // zone bands (low / moderate / high)
  ctx.fillStyle = "#fdecea"; ctx.fillRect(0, 0,        W, toY(65));           // high — red tint
  ctx.fillStyle = "#fef5e7"; ctx.fillRect(0, toY(65),  W, toY(35) - toY(65)); // moderate — amber tint
  ctx.fillStyle = "#d5f5f0"; ctx.fillRect(0, toY(35),  W, H - toY(35));       // low — calm tint

  // zone dividers
  ctx.strokeStyle = "rgba(0,0,0,0.08)";
  ctx.lineWidth = 1;
  [[35, "dashed"], [65, "dashed"]].forEach(([v]) => {
    ctx.beginPath();
    ctx.setLineDash([4, 4]);
    ctx.moveTo(0, toY(v)); ctx.lineTo(W, toY(v));
    ctx.stroke();
  });
  ctx.setLineDash([]);

  if (!values || values.length < 2) return;

  // Fixed window: 120 points ≈ 10 min at ~5 s/slow cycle.
  // Points anchor to the left so the trace grows rightward as data accumulates.
  const MAX_POINTS = 120;
  const toX = (i) => (i / (MAX_POINTS - 1)) * W;

  ctx.beginPath();
  ctx.moveTo(toX(0), toY(values[0]));
  for (let i = 1; i < values.length; i++) ctx.lineTo(toX(i), toY(values[i]));
  ctx.strokeStyle = "rgba(44,62,80,0.85)";
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.stroke();

  // dot at current value, colored by zone
  const last = values[values.length - 1];
  const dotColor = last >= 65 ? "#c0392b" : last >= 35 ? "#e67e22" : "#16a085";
  ctx.beginPath();
  ctx.arc(toX(values.length - 1), toY(last), 4, 0, Math.PI * 2);
  ctx.fillStyle = dotColor;
  ctx.fill();
}

function redrawActivationTrace(p) {
  const canvas = document.getElementById(`act-trace-${p}`);
  drawActivationTrace(canvas, state[p].trace_activation);
}

function applyPartnerNames() {
  const nameA     = document.getElementById("name-A");
  const nameB     = document.getElementById("name-B");
  const labelA    = document.getElementById("label-a");
  const labelB    = document.getElementById("label-b");
  const labelACoh = document.getElementById("label-a-coh");
  const labelBCoh = document.getElementById("label-b-coh");
  if (nameA)     nameA.textContent     = state.A.name;
  if (nameB)     nameB.textContent     = state.B.name;
  if (labelA)    labelA.textContent    = state.A.name;
  if (labelB)    labelB.textContent    = state.B.name;
  if (labelACoh) labelACoh.textContent = state.A.name;
  if (labelBCoh) labelBCoh.textContent = state.B.name;
  // sync enrollment button labels and _speakerNames lookup
  _speakerNames.A = state.A.name;
  _speakerNames.B = state.B.name;
  const enrollA = document.getElementById("enroll-name-A");
  const enrollB = document.getElementById("enroll-name-B");
  if (enrollA) enrollA.textContent = state.A.name;
  if (enrollB) enrollB.textContent = state.B.name;
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

  // show clear button only when baseline is set for that partner
  for (const p of ["A", "B"]) {
    const clearBtn = document.getElementById(`btn-clear-${p}`);
    if (clearBtn) clearBtn.hidden = !state[p].baseline_set;
  }
}

// ── Mode B DOM updates ────────────────────────────────────────────────────────

function updateSensorStatus(p) {
  const panel = document.getElementById(`panel-${p}`);
  const badge = document.getElementById(`state-badge-${p}`);
  const online = state[p].sensorOnline;

  if (panel) panel.classList.toggle("offline", !online);

  if (badge) {
    if (!online) {
      badge.textContent = "offline";
      badge.className = "state-badge offline";
    } else {
      // restore normal badge from flooded state
      badge.textContent = state[p].flooded ? "flooded" : "regulated";
      badge.className   = "state-badge " + (state[p].flooded ? "flooded" : "regulated");
    }
  }
}

function updateSignalQuality(p) {
  const el = document.getElementById(`sig-dot-${p}`);
  if (!el) return;
  const q = state[p].signalQuality;
  if (q === null) {
    el.className = "signal-dot";
    el.title = "signal quality: waiting";
  } else if (q >= 0.90) {
    el.className = "signal-dot good";
    el.title = `signal quality: ${Math.round(q * 100)}% good`;
  } else if (q >= 0.75) {
    el.className = "signal-dot fair";
    el.title = `signal quality: ${Math.round(q * 100)}% — check strap`;
  } else {
    el.className = "signal-dot poor";
    el.title = `signal quality: ${Math.round(q * 100)}% — poor contact`;
  }
}

function updateBatteryDisplay(p) {
  const el = document.getElementById(`battery-${p}`);
  if (!el) return;
  const lvl = state[p].batteryLevel;
  if (lvl === null) { el.textContent = ""; el.className = "battery-level"; return; }
  el.textContent = `${lvl}%`;
  el.className = "battery-level" + (lvl <= 20 ? " low" : "");
  el.title = `battery: ${lvl}%`;
}

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

  // Prominent breath-coherence-section (Feature 1)
  const bFillEl  = document.getElementById(`breath-coh-fill-${p}`);
  const bValEl   = document.getElementById(`breath-coh-val-${p}`);
  const bTrendEl = document.getElementById(`breath-coh-trend-${p}`);
  if (s.coherence !== null) {
    const pct = Math.min(s.coherence / 3.0, 1.0) * 100;
    if (bFillEl) bFillEl.style.width = pct.toFixed(1) + "%";
    if (bValEl)  bValEl.textContent  = s.coherence.toFixed(2);
    if (bTrendEl) {
      const prev = s.prevCoherence;
      let arrow = "→", cls = "trend-flat";
      if (prev !== null) {
        if (s.coherence > prev + 0.05)      { arrow = "↑"; cls = "trend-up"; }
        else if (s.coherence < prev - 0.05) { arrow = "↓"; cls = "trend-down"; }
      }
      bTrendEl.textContent = arrow;
      bTrendEl.className   = `breath-coh-trend ${cls}`;
    }
  } else {
    if (bFillEl)  bFillEl.style.width = "0%";
    if (bValEl)   bValEl.textContent  = "—";
    if (bTrendEl) { bTrendEl.textContent = ""; bTrendEl.className = "breath-coh-trend"; }
  }
  s.prevCoherence = s.coherence;
}

function updateCalmZone(p) {
  const row = document.getElementById(`calm-row-${p}`);
  const el  = document.getElementById(`calm-zone-${p}`);
  const s   = state[p].calm_zone_s;
  if (s > 0) {
    if (row) row.hidden = false;
    if (el) {
      const m  = Math.floor(s / 60);
      const ss = String(s % 60).padStart(2, "0");
      el.textContent = `${m}:${ss}`;
    }
  } else {
    if (row) row.hidden = true;
  }
}

function updateActivationModeB(p) {
  const scoreEl = document.getElementById(`rec-act-${p}`);
  const zoneEl  = document.getElementById(`rec-zone-${p}`);
  if (!scoreEl && !zoneEl) return;

  const v = state[p].activation;
  const zoneClass = v === null ? "" : v < 35 ? " zone-low" : v < 65 ? " zone-mid" : " zone-high";
  const zoneLabel = v === null ? "" : v < 35 ? "low" : v < 65 ? "moderate" : "high";

  if (scoreEl) {
    scoreEl.textContent = v !== null ? Math.round(v) : "—";
    scoreEl.className   = "rec-act-score" + zoneClass;
  }
  if (zoneEl) {
    zoneEl.textContent = zoneLabel;
    zoneEl.className   = "rec-act-zone" + (zoneLabel ? " " + zoneClass.trim() : "");
  }

  redrawRecActivationTrace(p);
  checkReadiness();
}

function redrawRecActivationTrace(p) {
  const canvas = document.getElementById(`rec-trace-${p}`);
  drawActivationTrace(canvas, state[p].trace_activation);
}

function checkReadiness() {
  const block = document.getElementById("readiness-block");
  if (!block) return;
  const aReady = state.A.activation !== null && state.A.activation < 35;
  const bReady = singlePartnerMode || (state.B.activation !== null && state.B.activation < 35);
  const ready  = aReady && bReady;
  block.hidden = !ready;

  if (ready) {
    const timeEl = document.getElementById("readiness-time");
    if (timeEl) {
      const maxS = Math.max(state.A.calm_zone_s || 0, singlePartnerMode ? 0 : (state.B.calm_zone_s || 0));
      const m  = Math.floor(maxS / 60);
      const ss = String(maxS % 60).padStart(2, "0");
      timeEl.textContent = `${m}:${ss}`;
    }
  }
}

// ── Mode B: breathing animation ──────────────────────────────────────────────

const BREATH_PATTERNS = {
  coherence: {
    caption: "5.5 breaths / min · resonance",
    phases: [
      { label: "breathe in",  s: 5.5, toScale: 1.0,  color: "#16a085" },
      { label: "breathe out", s: 5.5, toScale: 0.56, color: "#2980b9" },
    ],
  },
  box: {
    caption: "box breathing · 4 breaths / min",
    phases: [
      { label: "breathe in",  s: 4, toScale: 1.0,  color: "#16a085" },
      { label: "hold",        s: 4, toScale: 1.0,  color: "#b7950b" },
      { label: "breathe out", s: 4, toScale: 0.56, color: "#2980b9" },
      { label: "hold",        s: 4, toScale: 0.56, color: "#b7950b" },
    ],
  },
  "478": {
    caption: "4-7-8 · ~3 breaths / min",
    phases: [
      { label: "breathe in",  s: 4, toScale: 1.0,  color: "#16a085" },
      { label: "hold",        s: 7, toScale: 1.0,  color: "#b7950b" },
      { label: "breathe out", s: 8, toScale: 0.56, color: "#2980b9" },
    ],
  },
};

const ARC_CIRC = 2 * Math.PI * 122; // circumference for r=122

const breathAnim = {
  patternKey: "coherence",
  phaseIdx: 0,
  phaseStartTs: null,
  fromScale: 0.56,
  rafId: null,
};

function easeInOutSine(t) {
  return -(Math.cos(Math.PI * t) - 1) / 2;
}

function breathTick(ts) {
  const phases = BREATH_PATTERNS[breathAnim.patternKey].phases;
  if (breathAnim.phaseStartTs === null) breathAnim.phaseStartTs = ts;

  const phase = phases[breathAnim.phaseIdx];
  const elapsed = (ts - breathAnim.phaseStartTs) / 1000;
  const progress = Math.min(elapsed / phase.s, 1.0);
  const eased = easeInOutSine(progress);
  const scale = breathAnim.fromScale + (phase.toScale - breathAnim.fromScale) * eased;

  const circleEl  = document.getElementById("breath-circle");
  const arcEl     = document.getElementById("arc-fill");
  const countdownEl = document.getElementById("breath-countdown");

  if (circleEl) {
    circleEl.style.transform = `scale(${scale.toFixed(4)})`;
    circleEl.style.opacity   = (0.75 + 0.25 * eased).toFixed(3);
  }
  if (arcEl) {
    arcEl.style.strokeDashoffset = (ARC_CIRC * (1 - progress)).toFixed(1);
    arcEl.style.stroke = phase.color;
  }
  if (countdownEl) {
    const rem = Math.ceil(phase.s - elapsed);
    countdownEl.textContent = rem > 0 ? rem : "";
  }

  if (progress >= 1.0) {
    breathAnim.fromScale   = phase.toScale;
    breathAnim.phaseIdx    = (breathAnim.phaseIdx + 1) % phases.length;
    breathAnim.phaseStartTs = ts;
    const next = phases[breathAnim.phaseIdx];
    const labelEl = document.getElementById("breath-phase-label");
    if (labelEl) labelEl.textContent = next.label;
    if (arcEl) arcEl.style.strokeDashoffset = ARC_CIRC.toFixed(1); // reset arc
  }

  breathAnim.rafId = requestAnimationFrame(breathTick);
}

function startBreathAnim() {
  if (breathAnim.rafId) cancelAnimationFrame(breathAnim.rafId);
  breathAnim.phaseStartTs = null;

  const phases = BREATH_PATTERNS[breathAnim.patternKey].phases;
  const labelEl = document.getElementById("breath-phase-label");
  if (labelEl) labelEl.textContent = phases[breathAnim.phaseIdx].label;

  if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    breathAnim.rafId = requestAnimationFrame(breathTick);
  }
}

function switchPattern(key) {
  // capture current scale before switching
  const circleEl = document.getElementById("breath-circle");
  if (circleEl) {
    const m = circleEl.style.transform.match(/scale\(([^)]+)\)/);
    if (m) breathAnim.fromScale = parseFloat(m[1]);
  }
  breathAnim.patternKey   = key;
  breathAnim.phaseIdx     = 0;
  breathAnim.phaseStartTs = null;

  const captionEl = document.getElementById("breath-rate-caption");
  if (captionEl) captionEl.textContent = BREATH_PATTERNS[key].caption;

  document.querySelectorAll(".pattern-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.pattern === key);
  });

  startBreathAnim();
}

// ── Mode B: ambient sound (pink noise) ───────────────────────────────────────

let audioCtx  = null;
let soundGain = null;
let soundOn   = false;

function initAudio() {
  if (audioCtx) return;
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();

  // Generate 20s stereo pink noise buffer
  const sr = audioCtx.sampleRate;
  const len = sr * 20;
  const buf = audioCtx.createBuffer(2, len, sr);

  for (let ch = 0; ch < 2; ch++) {
    const d = buf.getChannelData(ch);
    let b0=0,b1=0,b2=0,b3=0,b4=0,b5=0,b6=0;
    for (let i = 0; i < len; i++) {
      const w = Math.random() * 2 - 1;
      b0 = 0.99886*b0 + w*0.0555179;
      b1 = 0.99332*b1 + w*0.0750759;
      b2 = 0.96900*b2 + w*0.1538520;
      b3 = 0.86650*b3 + w*0.3104856;
      b4 = 0.55000*b4 + w*0.5329522;
      b5 = -0.7616*b5  - w*0.0168980;
      d[i] = (b0+b1+b2+b3+b4+b5+b6+w*0.5362) / 9;
      b6 = w*0.115926;
    }
    // crossfade loop ends (100ms) to eliminate click
    const fade = Math.floor(sr * 0.1);
    for (let i = 0; i < fade; i++) {
      const t = i / fade;
      d[len - fade + i] = d[len - fade + i] * (1 - t) + d[i] * t;
    }
  }

  const src = audioCtx.createBufferSource();
  src.buffer = buf;
  src.loop   = true;

  const lpf = audioCtx.createBiquadFilter();
  lpf.type            = "lowpass";
  lpf.frequency.value = 700;
  lpf.Q.value         = 0.6;

  soundGain = audioCtx.createGain();
  soundGain.gain.value = 0;

  src.connect(lpf);
  lpf.connect(soundGain);
  soundGain.connect(audioCtx.destination);
  src.start();
}

function toggleSound() {
  if (!audioCtx) initAudio();
  if (audioCtx.state === "suspended") audioCtx.resume();

  soundOn = !soundOn;
  const vol = parseFloat(document.getElementById("sound-volume").value);
  soundGain.gain.setTargetAtTime(soundOn ? vol * 0.35 : 0, audioCtx.currentTime, 0.6);

  const btn = document.getElementById("sound-toggle");
  if (btn) btn.textContent = soundOn ? "sounds on" : "sounds off";
}

function onVolumeChange(val) {
  if (soundGain && soundOn) {
    soundGain.gain.setTargetAtTime(parseFloat(val) * 0.35, audioCtx.currentTime, 0.1);
  }
}

// ── transcript recording ──────────────────────────────────────────────────────

let mediaRecorder    = null;
let activeStream     = null;
let recordingStartMs = null;
let recClockInterval = null;
let sendQueue        = Promise.resolve();
let isRecording      = false;

function toggleRecording() {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
}

function startRecording() {
  if (isRecording) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setMicError("mic not supported in this browser");
    return;
  }
  navigator.mediaDevices.getUserMedia({ audio: true })
    .then((stream) => {
      activeStream     = stream;
      isRecording      = true;
      recordingStartMs = Date.now();
      recClockInterval = setInterval(updateMicUI, 1000);
      updateMicUI();
      startChunkCycle();
    })
    .catch((err) => {
      console.error("[mic] getUserMedia failed:", err);
      const msg = err.name === "NotAllowedError" ? "mic access denied"
                : err.name === "NotFoundError"   ? "no mic found"
                : "mic error";
      setMicError(msg);
    });
}

// ── voice enrollment ──────────────────────────────────────────────────────────

const _speakerNames = { A: "Partner A", B: "Partner B", T: "Therapist" };

function enrollVoice(partner) {
  const btn = document.getElementById(`enroll-btn-${partner}`);
  if (isRecording) {
    // Flash the button with an explanation
    if (btn) {
      const prev = btn.textContent;
      btn.textContent = "⚠ stop recording first";
      setTimeout(() => { if (btn) btn.textContent = prev; }, 2500);
    }
    return;
  }
  if (btn) { btn.textContent = "⏺ recording…"; btn.disabled = true; }

  navigator.mediaDevices.getUserMedia({ audio: true })
    .then((stream) => {
      const mr = new MediaRecorder(stream);
      const chunks = [];
      mr.ondataavailable = (e) => { if (e.data && e.data.size > 0) chunks.push(e.data); };
      mr.onstop = async () => {
        stream.getTracks().forEach(t => t.stop());
        const blob = new Blob(chunks, { type: mr.mimeType || "audio/webm" });
        try {
          const res = await fetch(`/api/enroll/${partner}`, {
            method: "POST",
            headers: { "Content-Type": blob.type },
            body: blob,
          });
          const data = await res.json();
          if (data.ok) {
            _speakerNames[partner] = data.name;
            if (btn) {
              btn.textContent = `✓ ${data.name}`;
              btn.classList.add("enrolled");
              btn.disabled = true;
            }
          } else {
            if (btn) { btn.textContent = `enroll ${_speakerNames[partner]}`; btn.disabled = false; }
          }
        } catch (err) {
          console.error("[enroll]", err);
          if (btn) { btn.textContent = `enroll ${_speakerNames[partner]}`; btn.disabled = false; }
        }
      };
      mr.start();
      setTimeout(() => { if (mr.state !== "inactive") mr.stop(); }, 4000);
    })
    .catch((err) => {
      console.error("[enroll] getUserMedia failed:", err);
      if (btn) { btn.textContent = `enroll ${_speakerNames[partner]}`; btn.disabled = false; }
    });
}

function showTranscribingBadge(on) {
  const el = document.getElementById("transcript-mic-status");
  if (!el) return;
  el.textContent = on ? "transcribing…" : (isRecording ? "recording" : "mic off");
  el.className   = "transcript-status" + (on || isRecording ? " recording" : "");
}

function setMicError(msg) {
  const btn = document.getElementById("btn-mic");
  if (btn) { btn.textContent = `⚠ ${msg}`; btn.classList.remove("recording"); }
  setTimeout(() => {
    if (!isRecording && btn) btn.textContent = "● record";
  }, 4000);
}

function stopRecording() {
  isRecording = false;
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
  if (activeStream) {
    activeStream.getTracks().forEach(t => t.stop());
    activeStream = null;
  }
  if (recClockInterval) {
    clearInterval(recClockInterval);
    recClockInterval = null;
  }
  updateMicUI();
}

function startChunkCycle() {
  if (!isRecording || !activeStream) return;
  // New MediaRecorder per cycle — each instance produces its own WebM EBML
  // header, making every blob independently decodable by ffmpeg.
  const mr = new MediaRecorder(activeStream);
  mediaRecorder = mr;
  let chunks = [];

  mr.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) chunks.push(e.data);
  };

  mr.onstop = () => {
    if (chunks.length > 0) {
      const blob = new Blob(chunks, { type: mr.mimeType || "audio/webm" });
      console.log(`[mic] chunk: ${blob.size} bytes  type=${blob.type}`);
      showTranscribingBadge(true);
      sendQueue = sendQueue
        .then(() => sendAudioChunk(blob))
        .then(() => showTranscribingBadge(false))
        .catch((err) => { console.error(err); showTranscribingBadge(false); });
    }
    if (isRecording) startChunkCycle();
  };

  mr.start();
  setTimeout(() => {
    if (mr.state !== "inactive") mr.stop();
  }, 5_000);
}

async function sendAudioChunk(blob) {
  try {
    const res = await fetch("/api/transcribe", {
      method: "POST",
      headers: { "Content-Type": blob.type || "audio/webm" },
      body: blob,
    });
    if (!res.ok) throw new Error(`transcribe ${res.status}`);
    const data = await res.json();
    // Update UI directly from the HTTP response — don't wait for the WS
    // broadcast (which may arrive later or be deduplicated by _seenSeqs).
    if (data.text) appendTranscriptEvent(data);
    return data;
  } catch (err) {
    console.error("[transcribe]", err);
  }
}

function updateMicUI() {
  const btn       = document.getElementById("btn-mic");
  const statusEl  = document.getElementById("rec-status");
  const badgeEl   = document.getElementById("transcript-mic-status");

  if (isRecording) {
    if (btn) { btn.textContent = "◼ stop"; btn.classList.add("recording"); }
    if (statusEl) {
      statusEl.hidden = false;
      const elapsed = recordingStartMs ? Math.floor((Date.now() - recordingStartMs) / 1000) : 0;
      const m  = String(Math.floor(elapsed / 60)).padStart(2, "0");
      const s  = String(elapsed % 60).padStart(2, "0");
      statusEl.textContent = `rec ${m}:${s}`;
    }
    if (badgeEl) { badgeEl.textContent = "recording"; badgeEl.classList.add("recording"); }
  } else {
    if (btn) { btn.textContent = "● record"; btn.classList.remove("recording"); }
    if (statusEl) { statusEl.hidden = true; statusEl.textContent = ""; }
    if (badgeEl) { badgeEl.textContent = "mic off"; badgeEl.classList.remove("recording"); }
  }
}

function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

const _seenSeqs = new Set();

function appendTranscriptEvent(data) {
  // Deduplicate: HTTP response and WS broadcast carry the same seq number
  if (data.seq !== undefined) {
    if (_seenSeqs.has(data.seq)) return;
    _seenSeqs.add(data.seq);
  }

  const emptyEl    = document.getElementById("transcript-empty");
  const scrollEl   = document.getElementById("transcript-scroll");
  const downloadBtn = document.getElementById("btn-download-session");
  if (!scrollEl) return;

  if (emptyEl)    emptyEl.hidden = true;
  if (downloadBtn) downloadBtn.hidden = false;

  const elapsed = data.session_elapsed_s ?? 0;
  const m  = String(Math.floor(elapsed / 60)).padStart(2, "0");
  const s  = String(Math.floor(elapsed % 60)).padStart(2, "0");
  const timestamp = `${m}:${s}`;

  const metrics = data.metrics || {};
  const mA = metrics.A || {};
  const mB = metrics.B || {};

  const describePartner = (m, name) => {
    if (m.activation === null || m.activation === undefined) return null;
    const act = Math.round(m.activation);
    const zone = act >= 65 ? "high activation" : act >= 35 ? "moderate" : "calm";
    const flooded = m.flooded ? ", flooded" : "";
    return `${name}: ${act} activation, ${zone}${flooded}`;
  };

  const parts = [];
  const pA = describePartner(mA, state.A.name || "A");
  const pB = describePartner(mB, state.B.name || "B");
  if (pA) parts.push(pA);
  if (pB && !singlePartnerMode) parts.push(pB);

  const speaker = data.speaker || null;   // "A", "B", "T", or null
  const speakerName = speaker ? (_speakerNames[speaker] || speaker) : null;
  const isTherapist = speaker === "T";

  // Physio line: shown for A/B speakers (or unknown), suppressed for Therapist
  const showPhysio = !isTherapist && parts.length > 0;

  const entry = document.createElement("div");
  entry.className = "transcript-entry";
  entry.innerHTML =
    (speakerName
      ? `<div class="transcript-speaker speaker-${speaker}">${escapeHtml(speakerName)}</div>`
      : "") +
    `<div class="transcript-time">${timestamp}</div>` +
    `<div class="transcript-text">${escapeHtml(data.text || "")}</div>` +
    (showPhysio
      ? `<div class="transcript-physio">${escapeHtml(parts.join(" · "))}</div>`
      : "");

  scrollEl.appendChild(entry);
  scrollEl.scrollTop = scrollEl.scrollHeight;
}

async function downloadSession() {
  try {
    const res = await fetch("/api/session_download");
    if (!res.ok) { console.error("download failed", res.status); return; }
    const disposition = res.headers.get("Content-Disposition") || "";
    const match = disposition.match(/filename="([^"]+)"/);
    const filename = match ? match[1] : "session.json";
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (err) {
    console.error("[download]", err);
  }
}

// ── dark mode ─────────────────────────────────────────────────────────────────

function toggleDarkMode() {
  const html    = document.documentElement;
  const isDark  = html.getAttribute("data-theme") === "dark";
  const theme   = isDark ? "light" : "dark";
  html.setAttribute("data-theme", theme);
  localStorage.setItem("theme", theme);
  document.querySelectorAll(".btn-theme").forEach(btn => {
    btn.textContent = theme === "dark" ? "◑" : "◐";
  });
}

// ── baseline button ───────────────────────────────────────────────────────────

function setBaseline(partner) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "set_baseline", partner }));
  }
}

function clearBaseline(partner) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "clear_baseline", partner }));
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
  // apply saved/preferred theme
  const savedTheme = localStorage.getItem("theme") ||
    (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  if (savedTheme === "dark") {
    document.documentElement.setAttribute("data-theme", "dark");
    document.querySelectorAll(".btn-theme").forEach(btn => { btn.textContent = "◑"; });
  }

  applyReducedMotion();
  connectWS();
  startSessionClock();   // no-op if element absent (Mode B)
  startBreakClock();     // no-op if element absent (Mode A)

  // connect-time ticker (Mode A)
  setInterval(() => {
    for (const p of ["A", "B"]) {
      const el = document.getElementById(`connect-time-${p}`);
      if (!el) continue;
      if (!state[p].connectedAt) { el.textContent = ""; continue; }
      const s = Math.floor((Date.now() - state[p].connectedAt) / 1000);
      const mm = String(Math.floor(s / 60)).padStart(2, "0");
      const ss = String(s % 60).padStart(2, "0");
      el.textContent = `${mm}:${ss}`;
    }
  }, 1000);

  // Mode B: breathing animation + pattern buttons
  if (document.getElementById("breath-circle")) {
    document.querySelectorAll(".pattern-btn").forEach(btn => {
      btn.addEventListener("click", () => switchPattern(btn.dataset.pattern));
    });
    switchPattern("coherence");
  }

  // keyboard shortcuts
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.isContentEditable) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    switch (e.key) {
      case "b":
        setBaseline("A");
        if (!singlePartnerMode) setBaseline("B");
        break;
      case "r":
        if (location.pathname !== "/mode_b") location.href = "/mode_b";
        break;
      case "Escape":
        if (location.pathname !== "/") location.href = "/";
        break;
      case "d":
        toggleDarkMode();
        break;
      case "m":
        toggleRecording();
        break;
    }
  });

  // resize canvases on window resize
  window.addEventListener("resize", () => {
    redrawActivationTrace("A");
    redrawActivationTrace("B");
    redrawDualTrace();
    redrawRecActivationTrace("A");
    redrawRecActivationTrace("B");
  });
});
