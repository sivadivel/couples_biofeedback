"""
server.py
aiohttp HTTP + WebSocket server for the couples biofeedback dashboard.
"""

from __future__ import annotations
import asyncio
import copy
import json
import os
import random
import statistics
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from aiohttp import web, WSMsgType

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # shared processor.py lives at repo root
from processor import PartnerProcessor, DyadicProcessor
from transcription import transcribe_audio
from voice_id import get_embedding, identify
from emotion import classify_emotion
import users as user_registry

STATIC_DIR = Path(__file__).parent / "static"

_REPORT_SYSTEM = """\
You are a clinical supervisor writing a post-session summary for a couples therapist.
You have access to a session transcript annotated with real-time physiological data
from both partners (activation index, HRV, vagal tone, coherence, respiration) and
emotion tone labels (aggressive / neutral / kind) for each utterance.

Write a concise, plain-English clinical summary in four sections:

SESSION ARC
One or two sentences describing the overall trajectory — how the session opened,
whether it escalated or de-escalated, and how it ended physiologically.

KEY MOMENTS
Three to five specific moments worth examining. For each, include the approximate
timestamp, paraphrase or quote the utterance, and explain what the physiological
data shows (e.g. "Alex's activation spiked from 38 to 71 within the response window,
while Jordan's HRV dropped — both partners were dysregulated simultaneously").
Prioritize moments where: activation crossed 65 (flooding), one partner's physiology
responded measurably to the other's speech, or a repair attempt correlated with
a drop in activation.

DYADIC PATTERN
Describe the co-regulation dynamic this session: who tends to escalate first,
who follows, whether there were periods of mutual synchrony, and whether any
repair attempts landed physiologically.

FOR NEXT SESSION
Two or three concrete clinical observations or questions to carry forward, grounded
in what the data showed (not generic advice).

Rules:
- Be specific. Cite timestamps and numbers from the data.
- Do not speculate beyond what the data shows.
- Write in clear language a therapist can read in under two minutes.
- Do not use markdown formatting symbols (no ##, **, *, -). Use plain text.
- Use blank lines between sections.
"""

_METRIC_GUIDE = """
COUPLES BIOFEEDBACK SESSION — METRIC GUIDE FOR AI ANALYSIS
===========================================================

This session log captures continuous physiology from two partners (A and B) during
a therapy session, with speaker-attributed transcript. Use this guide to interpret
the data and identify meaningful patterns.

TEMPORAL STRUCTURE OF EACH TRANSCRIPT EVENT
--------------------------------------------
Each event has three metric snapshots:

  pre_metrics      10-second average BEFORE the utterance (baseline for that moment).
                   Covers the window from ~15s to ~5s before the utterance ended.

  metrics          Instantaneous snapshot AT the moment the utterance ended.

  response_metrics 10-15 second average AFTER the utterance ended.
                   Captures the autonomic nervous system response to what was said.

To infer causal effects: compare pre_metrics → response_metrics around key utterances.
A rise in activation or fall in rmssd/hf from pre to response suggests a stress response
triggered by that speech. The autonomic system responds on a 5–30s timescale, so the
response window captures the beginning of the response, not necessarily its peak.

If response_metrics is absent, the session ended before the 15s window closed.

METRICS EXPLAINED
-----------------
activation (0–100)
  Two-signal robust-z index (HR elevation + RMSSD suppression vs. this
  person's own median/MAD baseline) during conversation; a third vagal/
  coherence signal folds in only during controlled, stationary breathing.
  Anchored near the bottom of the moderate zone at rest (~37 when current
  metrics match the personal baseline) — it is not centered at mid-scale
  (50), so "at baseline" reads as calm-to-moderate, not neutral. Rises with
  genuine departure from that baseline.
  < 35  = regulated, calm zone (parasympathetic dominant)
  35–65 = moderate engagement (sympathetic rising)
  ≥ 65  = high arousal territory (Gottman flooding threshold zone)
  KEY: Look for spikes in response_metrics.A.activation or response_metrics.B.activation
  after emotionally charged utterances.

confidence (0.0–1.0) / signals_mixed
  min(data quality, signal agreement, speech penalty). signals_mixed=true means
  HR and RMSSD point opposite directions with both |z| > 0.5 ("signals mixed") —
  treat the activation number that tick with more caution.

*_status fields (rmssd_status, hf_status, coherence_status, resp_rate_status)
  "ok" = the value is real and window-length requirements were met.
  "warming_up" = not enough clean data yet for this metric's minimum window
    (RMSSD needs ~20s of clean beats, HF/resp 60s, coherence 120s stationary).
  "paused_speech" = gated off because this partner was talking during the
    window (HF/coherence/resp_rate are not interpretable over irregular
    breathing; RMSSD is instead just widened + lower-confidence, not gated).
  "motion" = gated off due to a high RR-interval artifact rate in the window
    (this app's proxy for motion, since no accelerometer is available) —
    the value may reflect movement/gesture, not a real physiological change.
  When status != "ok", the corresponding numeric field is null — treat that
  as "no reading this tick," not zero or a dropped signal.

speech_fraction (0.0–1.0)
  Fraction of the metric window during which this partner was talking
  (approximated from completed transcript timing, extended a couple of
  seconds past each utterance for the respiratory tail).

mean_hr (bpm)
  Heart rate. Elevated HR = sympathetic arousal. Sustained elevated baseline before
  an utterance may indicate anticipatory stress or ongoing dysregulation.

rmssd (ms)
  Heart rate variability — root mean square of successive R-R interval differences.
  HIGHER = more parasympathetically regulated, more emotionally flexible.
  Drops within seconds of acute stress. Recovery of rmssd after a spike indicates
  the nervous system returning toward baseline. Null when rmssd_status != "ok".

hf (arbitrary units)
  Vagal activity — high-frequency (0.15–0.4 Hz) power of the R-R interval spectrum,
  normalized. HIGHER = more vagal tone and self-regulation capacity. Only computed
  outside speech (see hf_status) — reflects sustained state, not acute spikes.

coherence (ratio)
  McCraty heart coherence: peak HRV spectral power / (total power − peak power).
  Higher coherence = more rhythmic, resonant heart rate pattern. Requires a 120s
  stationary, non-speech window (see coherence_status) — meaningful mainly during
  paced-breathing/recovery contexts, not general conversation.

resp_rate (breaths/min)
  Respiration rate derived from the heart rate variability pattern. Only computed
  outside speech (see resp_rate_status).
  Elevated values (> 20 br/min) often accompany anxiety or hyperventilation.
  Slow, regular breathing (< 12 br/min) supports coherence and co-regulation.

flooded (boolean)
  Requires BOTH HR elevation and RMSSD suppression vs. this person's own baseline,
  sustained ≥10s (or an absolute HR ceiling, ~95bpm, independently) — talking alone
  raises HR but rarely collapses RMSSD the way sympathetic flooding does, so this
  is a stricter signal than activation ≥ 65 alone. Clears only after ≥5s continuously
  false (hysteresis, prevents flicker at the boundary).

calm_zone_s (seconds)
  Continuous seconds spent below activation 35. Resets when activation rises above 35.
  Longer calm zone = sustained co-regulation. Look for what ended calm zones.

DYADIC COUPLING (metrics.dyadic)
---------------------------------
peak_r
  Pearson correlation of both partners' HR time series (windowed cross-correlation),
  after removing shared slow drift (moving-median detrend). Range −1 to +1.

above_chance / p_value / percentile
  A circular-shift surrogate null test: peak_r is compared against ~200 shuffled
  copies of partner B's series. above_chance=true (p_value<0.05) means the coupling
  is unlikely to be coincidental; at-chance coupling should be described as
  "parallel, not linked," not as synchrony. ONLY discuss peak_r as meaningful
  relational synchrony when above_chance is true.
  CRITICAL CAVEAT: even when above chance, high coupling does NOT equal
  co-regulation. If both partners are flooded simultaneously, peak_r may be high —
  that indicates "locked in conflict," not safety. Always read peak_r jointly
  against both partners' activation levels (above-chance + both calm = co-regulation;
  above-chance + both activated = conflict amplification; at-chance = parallel).

lag_s
  Seconds by which one partner's HR changes precede the other's.

leader
  Which partner's HR changes tend to precede the other's (physiological leadership).
  The leader may be driving the dyadic emotional state — or responding to it first.

phase
  "in-phase" = both HRs rise and fall together.
  "anti-phase" = one rises as the other falls (possible protective regulation).

SUGGESTED ANALYSIS QUESTIONS
-----------------------------
- Which utterances (by which speaker) preceded the largest activation spikes in response_metrics?
- Were there moments where one partner was flooded while the other stayed regulated?
  What was being said? Did the regulated partner's physiology subsequently change?
- How did calm_zone_s durations correlate with conversation topics?
- Who physiologically led the recovery after each flooding episode?
- Did the therapist's speech tend to precede regulation or dysregulation in either partner?
- Where did rmssd drop sharply? What was the topic?
- Were there utterances with no physiological response (pre ≈ response)? What made those different?
- Was the dyadic coupling above chance, and if so, did it coincide with both partners
  calm (co-regulation) or both activated (conflict amplification)?

IMPORTANT CAVEATS
-----------------
- These are exploratory physiological proxies, not emotion readouts or relationship verdicts.
- Individual baseline physiology varies. Treat within-session changes as more meaningful
  than absolute values.
- Simultaneous speech blends voice embeddings and may yield speaker = null (unknown).
- The autonomic nervous system has individual variability in response latency.
  Some people show responses in 5s; others take 20–30s.
- A metric reading null with a non-"ok" *_status is a gated/unavailable reading, not zero.
""".strip()

_MEDITATION_REPORT_SYSTEM = """\
You are a meditation and mindfulness guide writing a post-session summary for the
person who just practiced. You have access to physiological data collected during
a solo meditation session (activation index, heart rate variability, vagal tone,
heart coherence, respiration), scored against a fixed reference baseline shared by
every session (not this person's own resting state — see the metric guide's
"baseline" entry). There is no transcript — the session is silent, guided only by
an ambient soundscape that layers in as activation rises.

Write a plain-text report (no markdown formatting symbols — no ##, **, *, -) with
exactly these five sections, each starting with its heading in capitals on its own
line, followed by a blank line, then prose. Use blank lines between sections.

SESSION ARC
One or two paragraphs describing how activation and coherence evolved across the
session, start to finish.

NOTABLE MOMENTS
Call out specific moments using the zone_transitions timestamps and activation
series — when activation rose or fell, how long recovery took, anything unusual.

PHYSIOLOGICAL NARRATIVE
Interpret the numbers: activation range and time-in-zone, coherence, and the
breathing trend across the session, relative to the fixed reference baseline.

PROGRESS OVER TIME
Use the progress field to compare this session to this person's own history.
If progress.is_first_session is true, say plainly that this is their first
recorded session and there is nothing to compare yet — do not invent a trend.
Otherwise compare this session's numbers to progress.all_time_avg and describe
the direction and size of any real difference, and note anything the
progress.recent_series shows about the trajectory across recent sessions.

FOR NEXT PRACTICE
One or two concrete, specific observations or suggestions grounded in this
session's data and, if available, its trend over time — not generic meditation
advice.

Cite timestamps and numbers from the data. Be warm but precise; avoid therapy-speak
and avoid overstating what physiological proxies can tell you.
""".strip()

_MEDITATION_METRIC_GUIDE = """
SOLO MEDITATION SESSION — METRIC GUIDE FOR AI ANALYSIS
=========================================================

This is a computed summary of one person's physiology during a silent, solo
meditation session, plus a light ambient soundscape that layers in as activation
rises. There is no transcript. Use this guide to interpret the fields below.

duration_s
  Total session length in seconds, from "begin session" to "finish session".

activation.min / max / mean / median (0–100)
  Robust-z activation index, same scale/thresholds used throughout this app.
  Meditation runs in the app's "recovery" (controlled-breathing) mode, so once
  a session has a long enough stationary window it uses three signals (HR +
  RMSSD + vagal/coherence) rather than the two-signal conversation formula.
  It is anchored near the bottom of the moderate zone: matching the fixed
  reference baseline reads as ~37, not a neutral 50 or deep calm — activation
  rises toward high with a real departure from that reference, and falls into
  the calm zone (<35) during genuine down-regulation below the reference.
  < 35  = calm zone (parasympathetic dominant)
  35–65 = moderate engagement (sympathetic rising)
  ≥ 65  = activated (fight/flight/freeze territory)

activation.series
  The full-session time series, one point roughly every 30 seconds:
  [{"t": seconds_since_start, "activation": 0-100}, ...].
  Use this to describe the overall arc and locate specific moments.

zone_pct.calm / moderate / activated
  Percent of total session time spent in each zone (see activation thresholds above).

coherence.avg / peak
  McCraty heart coherence (peak HRV spectral power / (total power − peak power)).
  Higher = more rhythmic, resonant heart rate pattern, often seen during regulated
  breathing. Values > 1.0 indicate strong coherence.

resp_rate.first_third_avg / last_third_avg (breaths/min)
  Average respiration rate in the first third vs. last third of the session — a
  meaningful "did breathing slow down" signal across the practice.

activated_episodes
  Count of distinct times activation crossed up into the activated (≥65) zone.

zone_transitions
  Timestamped entered_<zone>/left_<zone> events (session_elapsed_s, activation) —
  the closest thing to "notable moments" in a session with no transcript.

baseline
  A FIXED reference (mean, std) for mean_hr, rmssd, hf, coherence, resp_rate — the
  same values for every meditation session, not fit from this particular person.
  It represents a population-typical resting adult, chosen so that an activation
  score of 50 means "matches that typical reference," not "this person's own calm
  state." Do not describe it as this person's own calibrated or personal baseline.

progress
  This person's history across all their past meditation sessions (matched by
  name), independent of the fixed physiological baseline above.
  is_first_session: true if this is the only session on record — see instructions.
  session_count: total completed sessions on record for this person, including this one.
  current: this session's headline numbers (activation_mean, zone_pct_calm, duration_s).
  all_time_avg: the same fields averaged over every PRIOR session (excludes this
    one); null if is_first_session is true.
  recent_series: up to the last 10 sessions, oldest to newest, each with
    session_id, ended_at, activation_mean, zone_pct_calm — use this to describe
    a trajectory, not just a single before/after comparison.

IMPORTANT CAVEATS
------------------
- These are exploratory physiological proxies, not a measure of "how well" someone
  meditated. Activated periods are not failures.
- Because the baseline is fixed rather than personal, someone whose true resting
  physiology differs a lot from the reference (age, fitness, medication, time of
  day) may read as persistently more or less "activated" than they subjectively
  feel — that is a property of the fixed reference, not necessarily their state.
  Read within-session change and trend as more informative than a single absolute
  number.
""".strip()


def _zone_of(score: float) -> str:
    if score < 35:
        return "calm"
    if score < 65:
        return "moderate"
    return "activated"


def _history_key(name: str) -> str:
    """Filesystem-safe key for a meditator's history file — same slug as the user registry."""
    return user_registry.slug(name)


class BiofeedbackServer:
    def __init__(
        self,
        proc_a: PartnerProcessor,
        proc_b: PartnerProcessor | None = None,
        port: int = 8765,
        setup_mode: bool = False,
    ):
        self.proc_a = proc_a
        self.proc_b = proc_b
        self.dyadic = DyadicProcessor(proc_a.name, proc_b.name) if proc_b else None
        self.port = port
        self._configured: bool = not setup_mode
        self.clients: set[web.WebSocketResponse] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._sensor_online: dict[int, bool] = {0: False, 1: False}

        _partner_fields = {
            k: None for k in (
                "mean_hr", "rmssd", "hf", "coherence", "resp_rate",
                "activation", "direction", "flooded", "calm_zone_s", "hr_baseline_pct",
                "signal_quality", "confidence", "state_description",
                "rmssd_status", "hf_status", "coherence_status", "resp_rate_status",
                "speech_fraction", "signals_mixed", "used_vagal",
            )
        }
        self._metrics_snapshot: dict = {
            "A":      {**_partner_fields, "flooded": False, "calm_zone_s": 0},
            "B":      {**_partner_fields, "flooded": False, "calm_zone_s": 0},
            "dyadic": {"peak_r": None, "lag_s": None, "phase": None, "leader": None,
                       "p_value": None, "percentile": None, "above_chance": None,
                       "trace": []},
        }
        self._voice_enrollments:  dict = {}   # "A"/"B"/"T" → np.ndarray
        self._session_id:         str | None = None
        self._session_file        = None   # line-buffered NDJSON handle
        self._session_seq:        int = 0
        self._session_start_mono: float | None = None
        self._whisper_model:      str = "base"
        self._metric_buffer:      deque = deque()  # (monotonic_time, snapshot_copy)
        self._last_log_snapshot:  float = 0.0
        self._prev_flooded:       dict[str, bool] = {"A": False, "B": False}
        self._last_session_id:    str | None = None
        self._prev_zone:          dict[str, str | None] = {"A": None, "B": None}
        # Meditate mode's own "begin"/"finish" state — distinct from whether a
        # session LOG file happens to be open, since that already opens as soon
        # as a sensor connects (couples flow), before a meditation round begins.
        self._meditation_active:  bool = False
        # simulator-only: mutable scenario state shared with both simulated
        # partner tasks (None for real BLE sessions — nothing to switch)
        self._scenario_state = None
        self._sim_session_start: float | None = None

    # ── callbacks for BLE / simulator ────────────────────────────────────────

    def on_rr(self, idx: int, rr_ms: float) -> None:
        if idx == 0:
            proc = self.proc_a
        elif idx == 1 and self.proc_b:
            proc = self.proc_b
        else:
            return
        if self._loop is not None:
            self._loop.call_soon_threadsafe(proc.push_rr, rr_ms)
        else:
            proc.push_rr(rr_ms)

    def on_bpm(self, idx: int, bpm: float) -> None:
        # bpm notifications not used directly; RR intervals are the source of truth
        pass

    def on_sensor_connect(self, idx: int) -> None:
        self._sensor_online[idx] = True
        partner = "A" if idx == 0 else "B"
        if self._loop:
            msg = {"type": "sensor_status", "partner": partner, "online": True}
            self._loop.call_soon_threadsafe(asyncio.create_task, self.broadcast(msg))
            self._loop.call_soon_threadsafe(self._open_session_log)
            self._loop.call_soon_threadsafe(self._log_lifecycle, "sensor_connect", partner)

    def on_sensor_disconnect(self, idx: int) -> None:
        self._sensor_online[idx] = False
        partner = "A" if idx == 0 else "B"
        if self._loop:
            msg = {"type": "sensor_status", "partner": partner, "online": False}
            self._loop.call_soon_threadsafe(asyncio.create_task, self.broadcast(msg))
            self._loop.call_soon_threadsafe(self._log_lifecycle, "sensor_disconnect", partner)

    def on_battery(self, idx: int, level: int) -> None:
        partner = "A" if idx == 0 else "B"
        if self._loop:
            msg = {"type": "battery", "partner": partner, "level": level}
            self._loop.call_soon_threadsafe(asyncio.create_task, self.broadcast(msg))
        else:
            # called before loop starts — queue via broadcast at first opportunity
            self._pending_battery = getattr(self, "_pending_battery", {})
            self._pending_battery[partner] = level

    # ── WebSocket handler ─────────────────────────────────────────────────────

    async def ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)
        self.clients.add(ws)
        names = {"A": self.proc_a.name}
        if self.proc_b:
            names["B"] = self.proc_b.name
        await ws.send_str(json.dumps({
            "type": "session_init",
            "names": names,
            "single_partner": self.proc_b is None,
            "meditation_session_active": self._meditation_active,
            "simulated": self._scenario_state is not None,
            "scenario": self._scenario_state.scenario if self._scenario_state else None,
        }))
        if self.proc_a.baseline is not None:
            await ws.send_str(json.dumps({
                "type": "baseline_status",
                "partner": "A",
                "ok": True,
                "name": self.proc_a.name,
            }))
        if self.proc_b and self.proc_b.baseline is not None:
            await ws.send_str(json.dumps({
                "type": "baseline_status",
                "partner": "B",
                "ok": True,
                "name": self.proc_b.name,
            }))
        # re-send sensor online state and any pending battery levels
        for idx, partner in ((0, "A"), (1, "B")):
            if idx == 1 and not self.proc_b:
                continue
            await ws.send_str(json.dumps({
                "type": "sensor_status",
                "partner": partner,
                "online": self._sensor_online.get(idx, False),
            }))
        for partner, level in getattr(self, "_pending_battery", {}).items():
            await ws.send_str(json.dumps({
                "type": "battery", "partner": partner, "level": level
            }))
        # Replay last known mid/slow snapshots so the page populates immediately
        # without waiting up to 10 s for the next metrics cycle.
        for p_id, proc in (("A", self.proc_a), ("B", self.proc_b)):
            if proc is None:
                continue
            snap = self._metrics_snapshot[p_id]
            traces = proc.reconnect_snapshot()
            if snap.get("mean_hr") is not None:
                await ws.send_str(json.dumps({
                    "type":           "mid",
                    "partner":        p_id,
                    "mean_hr":        snap.get("mean_hr"),
                    "rmssd":          snap.get("rmssd"),
                    "hr_baseline_pct": snap.get("hr_baseline_pct"),
                    "flooded":        snap.get("flooded", False),
                    "calm_zone_s":    snap.get("calm_zone_s", 0),
                    "signal_quality": snap.get("signal_quality"),
                    "trace_hr":       traces["trace_hr"],
                }))
            if snap.get("activation") is not None or snap.get("hf") is not None:
                await ws.send_str(json.dumps({
                    "type":              "slow",
                    "partner":           p_id,
                    "hf":                snap.get("hf"),
                    "coherence":         snap.get("coherence"),
                    "resp_rate":         snap.get("resp_rate"),
                    "activation":        snap.get("activation"),
                    "direction":         snap.get("direction"),
                    "confidence":        snap.get("confidence"),
                    "state_description": snap.get("state_description"),
                    "calm_zone_s":       snap.get("calm_zone_s", 0),
                    "trace_activation":  traces["trace_activation"],
                }))
        if self.dyadic:
            dsnap = self._metrics_snapshot["dyadic"]
            if dsnap.get("trace"):
                await ws.send_str(json.dumps({
                    "type":         "dyadic",
                    "peak_r":       dsnap.get("peak_r"),
                    "lag_s":        dsnap.get("lag_s"),
                    "phase":        dsnap.get("phase"),
                    "leader":       dsnap.get("leader"),
                    "p_value":      dsnap.get("p_value"),
                    "percentile":   dsnap.get("percentile"),
                    "above_chance": dsnap.get("above_chance"),
                    "trace":        dsnap.get("trace", []),
                }))
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue
                    mtype = data.get("type")
                    if mtype == "ping":
                        await ws.send_str(json.dumps({"type": "pong"}))
                    elif mtype == "set_baseline":
                        partner = data.get("partner", "A")
                        if partner == "B" and not self.proc_b:
                            continue
                        proc = self.proc_a if partner == "A" else self.proc_b
                        ok = proc.set_baseline()
                        await self.broadcast({
                            "type": "baseline_status",
                            "partner": partner,
                            "ok": ok,
                            "name": proc.name,
                        })
                        if ok:
                            self._open_session_log()
                            self._log_lifecycle("baseline_set", partner)
                    elif mtype == "clear_baseline":
                        partner = data.get("partner", "A")
                        if partner == "B" and not self.proc_b:
                            continue
                        proc = self.proc_a if partner == "A" else self.proc_b
                        proc.clear_baseline()
                        await self.broadcast({
                            "type": "baseline_status",
                            "partner": partner,
                            "ok": False,
                            "name": proc.name,
                        })
                        self._log_lifecycle("baseline_clear", partner)
                    elif mtype == "set_mode":
                        partner = data.get("partner", "A")
                        if partner == "B" and not self.proc_b:
                            continue
                        proc = self.proc_a if partner == "A" else self.proc_b
                        proc.set_mode(data.get("mode", "conversation"))
                    elif mtype == "set_scenario":
                        if self._scenario_state is None or self._sim_session_start is None:
                            continue  # not a simulated session — nothing to switch
                        new_scenario = data.get("scenario", "")
                        elapsed_now = time.monotonic() - self._sim_session_start
                        if self._scenario_state.set(new_scenario, elapsed_now):
                            await self.broadcast({
                                "type": "scenario_state",
                                "scenario": self._scenario_state.scenario,
                            })
                    elif mtype == "set_standard_baseline":
                        partner = data.get("partner", "A")
                        if partner == "B" and not self.proc_b:
                            continue
                        proc = self.proc_a if partner == "A" else self.proc_b
                        proc.set_standard_baseline()
                        proc.set_mode("recovery")  # meditate is silent/sustained, like paced breathing
                        await self.broadcast({
                            "type": "baseline_status",
                            "partner": partner,
                            "ok": True,
                            "name": proc.name,
                        })
                        self._open_session_log()
                        self._log_lifecycle("baseline_set", partner, standard=True)
                        self._meditation_active = True
                    elif mtype == "end_meditation":
                        ended_id = self._session_id
                        if ended_id is None:
                            continue
                        self._close_session_log()
                        stats = self._compute_meditation_stats(ended_id)
                        self._append_history(self.proc_a.name, stats)
                        progress = self._compute_progress(self.proc_a.name)
                        await self.broadcast({
                            "type": "meditation_ended",
                            "session_id": ended_id,
                            "stats": stats,
                            "progress": progress,
                        })
                elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break
        finally:
            self.clients.discard(ws)
        return ws

    # ── static file handlers ─────────────────────────────────────────────────

    _NO_CACHE = {"Cache-Control": "no-store"}

    async def index_handler(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(STATIC_DIR / "index.html", headers=self._NO_CACHE)

    async def mode_b_handler(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(STATIC_DIR / "mode_b.html", headers=self._NO_CACHE)

    async def meditate_handler(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(STATIC_DIR / "meditate.html", headers=self._NO_CACHE)

    async def whitepaper_handler(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(Path(__file__).parent / "whitepaper.html", headers=self._NO_CACHE)

    async def static_handler(self, request: web.Request) -> web.FileResponse:
        filename = request.match_info["filename"]
        path = STATIC_DIR / filename
        if not path.exists():
            raise web.HTTPNotFound()
        return web.FileResponse(path, headers=self._NO_CACHE)

    # ── metrics snapshot (feeds transcript annotation) ────────────────────────

    def _update_snapshot(self, msg: dict) -> None:
        mtype = msg.get("type")
        p = msg.get("partner")
        if mtype == "fast" and p in ("A", "B"):
            self._metrics_snapshot[p]["mean_hr"] = msg.get("mean_hr")
        elif mtype == "mid" and p in ("A", "B"):
            snap = self._metrics_snapshot[p]
            snap["mean_hr"]         = msg.get("mean_hr")
            snap["rmssd"]           = msg.get("rmssd")
            snap["rmssd_status"]    = msg.get("rmssd_status")
            snap["hr_baseline_pct"] = msg.get("hr_baseline_pct")
            snap["signal_quality"]  = msg.get("signal_quality")
            snap["speech_fraction"] = msg.get("speech_fraction")
            new_flooded = msg.get("flooded", False)
            snap["flooded"] = new_flooded
            if new_flooded != self._prev_flooded[p] and self._session_file is not None:
                evt = "flood_start" if new_flooded else "flood_end"
                dpa = msg.get("dpa") or {}
                self._log_lifecycle(evt, p,
                    mean_hr=msg.get("mean_hr"),
                    hr_baseline_pct=msg.get("hr_baseline_pct"),
                    by_primary=dpa.get("by_primary"),
                    by_ceiling=dpa.get("by_ceiling"))
            self._prev_flooded[p] = new_flooded
        elif mtype == "slow" and p in ("A", "B"):
            snap = self._metrics_snapshot[p]
            snap["hf"]               = msg.get("hf")
            snap["hf_status"]        = msg.get("hf_status")
            snap["coherence"]        = msg.get("coherence")
            snap["coherence_status"] = msg.get("coherence_status")
            snap["resp_rate"]        = msg.get("resp_rate")
            snap["resp_rate_status"] = msg.get("resp_rate_status")
            snap["activation"]       = msg.get("activation")
            snap["direction"]        = msg.get("direction")
            snap["calm_zone_s"]      = msg.get("calm_zone_s", 0)
            snap["confidence"]       = msg.get("confidence")
            snap["signals_mixed"]    = msg.get("signals_mixed")
            snap["used_vagal"]       = msg.get("used_vagal")
            snap["speech_fraction"]  = msg.get("speech_fraction")
            snap["state_description"] = msg.get("state_description")
            if p == "A" and snap["activation"] is not None and self._session_file is not None:
                new_zone = _zone_of(snap["activation"])
                old_zone = self._prev_zone["A"]
                if old_zone is not None and new_zone != old_zone:
                    elapsed = (round(time.monotonic() - self._session_start_mono, 1)
                               if self._session_start_mono else None)
                    self._log_lifecycle(f"left_{old_zone}", "A",
                        activation=snap["activation"], session_elapsed_s=elapsed)
                    self._log_lifecycle(f"entered_{new_zone}", "A",
                        activation=snap["activation"], session_elapsed_s=elapsed)
                self._prev_zone["A"] = new_zone
        elif mtype == "dyadic":
            d = self._metrics_snapshot["dyadic"]
            d["peak_r"]       = msg.get("peak_r")
            d["lag_s"]        = msg.get("lag_s")
            d["phase"]        = msg.get("phase")
            d["leader"]       = msg.get("leader")
            d["p_value"]      = msg.get("p_value")
            d["percentile"]   = msg.get("percentile")
            d["above_chance"] = msg.get("above_chance")
            d["trace"]        = msg.get("trace", d.get("trace", []))
        now = time.monotonic()
        self._metric_buffer.append((now, copy.deepcopy(self._metrics_snapshot)))
        cutoff = now - 30.0
        while self._metric_buffer and self._metric_buffer[0][0] < cutoff:
            self._metric_buffer.popleft()

    # ── metric time-window helpers ────────────────────────────────────────────

    def _window_metrics(self, t_start: float, t_end: float) -> dict:
        """Average numeric metrics from buffer over [t_start, t_end] (monotonic)."""
        entries = [m for t, m in self._metric_buffer if t_start <= t <= t_end]
        if not entries:
            return {}
        result = {}
        for partner in ("A", "B"):
            result[partner] = {}
            for key in entries[0].get(partner, {}):
                vals = [e.get(partner, {}).get(key) for e in entries]
                nums = [v for v in vals if isinstance(v, (int, float))]
                if nums:
                    result[partner][key] = round(sum(nums) / len(nums), 2)
        result["dyadic"] = entries[-1].get("dyadic", {})
        return result

    async def _schedule_response_patch(self, seq: int, t_start: float) -> None:
        """Wait 15s after utterance end, then write the autonomic response window."""
        await asyncio.sleep(15)
        response_metrics = self._window_metrics(t_start, t_start + 15.0)
        if self._session_file and response_metrics:
            self._session_file.write(json.dumps({
                "type": "response_patch",
                "seq": seq,
                "response_metrics": response_metrics,
            }) + "\n")

    # ── session log ───────────────────────────────────────────────────────────

    def _open_session_log(self) -> None:
        if self._session_file is not None:
            return
        sessions_dir = Path(__file__).parent / "sessions"
        sessions_dir.mkdir(exist_ok=True)
        now_utc = datetime.now(timezone.utc)
        self._session_id = now_utc.strftime("%Y-%m-%dT%H-%M-%S")
        self._session_start_mono = time.monotonic()
        self._last_log_snapshot  = time.monotonic()
        path = sessions_dir / f"{self._session_id}.ndjson"
        self._session_file = open(path, "w", buffering=1)  # line-buffered
        names = {
            "A": self.proc_a.name,
            "B": self.proc_b.name if self.proc_b else None,
        }
        header = {
            "type": "header",
            "session_id": self._session_id,
            "started_at": now_utc.isoformat(),
            "names": names,
        }
        self._session_file.write(json.dumps(header) + "\n")

    def _close_session_log(self) -> None:
        if self._session_file is None:
            return
        self._last_session_id = self._session_id
        try:
            self._session_file.flush()
            self._session_file.close()
        except Exception:
            pass
        self._session_file        = None
        self._session_id          = None
        self._session_seq         = 0
        self._session_start_mono  = None
        self._last_log_snapshot   = 0.0
        self._prev_flooded        = {"A": False, "B": False}
        self._prev_zone           = {"A": None, "B": None}
        self._meditation_active   = False
        self._metric_buffer.clear()

    def _log_lifecycle(self, event: str, partner: str | None = None, **extra) -> None:
        if self._session_file is None:
            return
        rec: dict = {
            "type":      "lifecycle",
            "event":     event,
            "wall_time": datetime.now(timezone.utc).isoformat(),
        }
        if partner is not None:
            rec["partner"] = partner
        rec.update(extra)
        self._session_file.write(json.dumps(rec) + "\n")

    def _append_transcript_event(self, text: str, speaker: str | None = None,
                                 emotion: dict | None = None) -> dict:
        self._open_session_log()
        now_mono = time.monotonic()
        utterance_start = now_mono - 5.0  # chunk is 5s; approximate utterance start
        pre_metrics = self._window_metrics(utterance_start - 10.0, utterance_start)
        snapshot = copy.deepcopy(self._metrics_snapshot)
        elapsed = round(now_mono - self._session_start_mono, 1)
        event = {
            "type": "event",
            "seq": self._session_seq,
            "wall_time": datetime.now(timezone.utc).isoformat(),
            "session_elapsed_s": elapsed,
            "text": text,
            "speaker": speaker,
            "emotion": emotion or {"tone": "neutral", "confidence": 1.0, "note": ""},
            "pre_metrics": pre_metrics,
            "metrics": snapshot,
        }
        self._session_file.write(json.dumps(event) + "\n")
        self._session_seq += 1
        return event

    # ── transcription handlers ────────────────────────────────────────────────

    async def transcribe_handler(self, request: web.Request) -> web.Response:
        audio_bytes = await request.read()
        if not audio_bytes:
            print("[transcribe] received empty body")
            return web.Response(content_type="application/json", text='{"text":""}')
        content_type = request.content_type or "audio/webm"
        suffix = ".mp4" if ("mp4" in content_type or "aac" in content_type) else ".webm"
        print(f"[transcribe] {len(audio_bytes):,} bytes  type={content_type}")
        try:
            # Run Whisper and speaker embedding concurrently to reduce latency
            text_task = asyncio.create_task(
                transcribe_audio(audio_bytes, self._whisper_model, content_type)
            )
            emb_task = asyncio.create_task(
                get_embedding(audio_bytes, suffix)
            )
            text_result, embedding = await asyncio.gather(text_task, emb_task)
        except Exception as exc:
            print(f"[transcribe] ERROR: {exc}")
            return web.Response(status=500, content_type="application/json",
                                text=json.dumps({"error": str(exc)}))
        text = text_result.get("text", "").strip()
        print(f"[transcribe] whisper → {repr(text[:120]) if text else '(empty)'}")
        if not text:
            return web.Response(content_type="application/json", text='{"text":""}')
        emotion_task = asyncio.create_task(classify_emotion(text))
        speaker, sims = identify(embedding, self._voice_enrollments)
        print(f"[transcribe] speaker → {speaker}  sims={sims}")
        # Approximate speech-fraction gating input (spec v2 §4.1): no real-time
        # VAD exists in this app, so treat the ~5s recording chunk that just
        # completed as the speech span for whichever partner it resolved to.
        end_mono = time.monotonic()
        speech_proc = self.proc_a if speaker == "A" else self.proc_b if speaker == "B" else None
        if speech_proc is not None:
            speech_proc.record_speech_segment(end_mono - 5.0, end_mono)
        emotion = await emotion_task
        print(f"[transcribe] emotion → {emotion}")
        event = self._append_transcript_event(text, speaker, emotion=emotion)
        asyncio.create_task(self._schedule_response_patch(event["seq"], time.monotonic()))
        await self.broadcast({"type": "transcript", **event})
        return web.Response(
            content_type="application/json",
            text=json.dumps({"type": "transcript", **event}),
        )

    async def enroll_handler(self, request: web.Request) -> web.Response:
        partner = request.match_info["partner"]
        if partner not in ("A", "B", "T"):
            return web.Response(status=400, text="partner must be A, B, or T")
        audio_bytes = await request.read()
        if not audio_bytes:
            return web.Response(status=400, text="empty audio")
        content_type = request.content_type or "audio/webm"
        suffix = ".mp4" if ("mp4" in content_type or "aac" in content_type) else ".webm"
        try:
            embedding = await get_embedding(audio_bytes, suffix)
        except Exception as exc:
            print(f"[enroll] ERROR: {exc}")
            return web.Response(status=500, content_type="application/json",
                                text=json.dumps({"error": str(exc)}))
        self._voice_enrollments[partner] = embedding
        self._open_session_log()
        self._log_lifecycle("voice_enrolled", partner)
        if partner == "A":
            name = self.proc_a.name
        elif partner == "B":
            name = self.proc_b.name if self.proc_b else "B"
        else:
            name = "Therapist"
        print(f"[enroll] {name} voice enrolled")
        return web.Response(
            content_type="application/json",
            text=json.dumps({"ok": True, "partner": partner, "name": name}),
        )

    async def session_download_handler(self, request: web.Request) -> web.Response:
        if self._session_file is None or self._session_id is None:
            return web.Response(status=404, text="No session log yet.")
        path = Path(__file__).parent / "sessions" / f"{self._session_id}.ndjson"
        if not path.exists():
            return web.Response(status=404, text="Session file not found.")
        header_obj: dict | None = None
        events = []
        patches: dict[int, dict] = {}
        snapshots = []
        lifecycle_events = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") == "header":
                    header_obj = obj
                elif obj.get("type") == "event":
                    events.append(obj)
                elif obj.get("type") == "response_patch":
                    patches[obj["seq"]] = obj.get("response_metrics", {})
                elif obj.get("type") == "metrics_snapshot":
                    snapshots.append(obj)
                elif obj.get("type") == "lifecycle":
                    lifecycle_events.append(obj)
        for ev in events:
            if ev["seq"] in patches:
                ev["response_metrics"] = patches[ev["seq"]]
        doc = {
            "session_id":     self._session_id,
            "started_at":     header_obj.get("started_at") if header_obj else None,
            "names":          header_obj.get("names") if header_obj else {},
            "metric_guide":   _METRIC_GUIDE,
            "events":         events,
            "metric_snapshots": snapshots,
            "lifecycle_events": lifecycle_events,
        }
        filename = f"session_{self._session_id}.json"
        return web.Response(
            content_type="application/json",
            text=json.dumps(doc, indent=2),
            headers={
                **self._NO_CACHE,
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )

    def _compute_meditation_stats(self, session_id: str) -> dict:
        """Deterministic post-session stats for partner A, read from the closed ndjson log."""
        path = Path(__file__).parent / "sessions" / f"{session_id}.ndjson"
        header_obj: dict | None = None
        snapshots: list[dict] = []
        lifecycle_events: list[dict] = []
        if path.exists():
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    t = obj.get("type")
                    if t == "header":
                        header_obj = obj
                    elif t == "metrics_snapshot":
                        snapshots.append(obj)
                    elif t == "lifecycle":
                        lifecycle_events.append(obj)

        def _a(s, key):
            return s.get("metrics", {}).get("A", {}).get(key)

        activations = [_a(s, "activation") for s in snapshots if _a(s, "activation") is not None]
        coherences  = [_a(s, "coherence")  for s in snapshots if _a(s, "coherence")  is not None]
        resp_rates  = [_a(s, "resp_rate")  for s in snapshots if _a(s, "resp_rate")  is not None]

        if snapshots:
            duration_s = snapshots[-1]["session_elapsed_s"]
        elif header_obj and header_obj.get("started_at"):
            started = datetime.fromisoformat(header_obj["started_at"])
            duration_s = (datetime.now(timezone.utc) - started).total_seconds()
        else:
            duration_s = 0.0

        # Zone time: each snapshot's activation is assumed to hold since the
        # previous snapshot; the tail from the last snapshot to session end is
        # attributed to the last known zone.
        zone_time_s = {"calm": 0.0, "moderate": 0.0, "activated": 0.0}
        prev_t, last_zone = 0.0, None
        for s in snapshots:
            t = s["session_elapsed_s"]
            act = _a(s, "activation")
            if act is not None:
                last_zone = _zone_of(act)
                zone_time_s[last_zone] += max(0.0, t - prev_t)
            prev_t = t
        if last_zone and duration_s > prev_t:
            zone_time_s[last_zone] += duration_s - prev_t

        def _pct(zone):
            return round(100.0 * zone_time_s[zone] / duration_s, 1) if duration_s else None

        n = len(resp_rates)
        third = max(1, n // 3) if n else 0
        resp_first = round(sum(resp_rates[:third]) / third, 1) if n else None
        resp_last  = round(sum(resp_rates[-third:]) / third, 1) if n else None

        activated_episodes = sum(
            1 for ev in lifecycle_events
            if ev.get("event") == "entered_activated" and ev.get("partner") == "A"
        )

        baseline = self.proc_a.baseline
        baseline_stats = (
            {k: {"mean": v[0], "std": v[1]} for k, v in baseline.stats.items()}
            if baseline else {}
        )

        return {
            "session_id": session_id,
            "names": header_obj.get("names") if header_obj else {"A": self.proc_a.name},
            "duration_s": round(duration_s, 1),
            "activation": {
                "min":    round(min(activations), 1) if activations else None,
                "max":    round(max(activations), 1) if activations else None,
                "mean":   round(statistics.mean(activations), 1) if activations else None,
                "median": round(statistics.median(activations), 1) if activations else None,
                "series": [
                    {"t": s["session_elapsed_s"], "activation": _a(s, "activation")}
                    for s in snapshots
                ],
            },
            "zone_pct": {
                "calm":      _pct("calm"),
                "moderate":  _pct("moderate"),
                "activated": _pct("activated"),
            },
            "coherence": {
                "avg":  round(statistics.mean(coherences), 3) if coherences else None,
                "peak": round(max(coherences), 3) if coherences else None,
            },
            "resp_rate": {"first_third_avg": resp_first, "last_third_avg": resp_last},
            "activated_episodes": activated_episodes,
            "zone_transitions": [
                ev for ev in lifecycle_events
                if ev.get("partner") == "A" and ev.get("event", "").startswith(("entered_", "left_"))
            ],
            "baseline": baseline_stats,
        }

    # ── cross-session history / progress ────────────────────────────────────────

    def _history_path(self, name: str) -> Path:
        history_dir = Path(__file__).parent / "history"
        history_dir.mkdir(exist_ok=True)
        return history_dir / f"{_history_key(name)}.ndjson"

    def _append_history(self, name: str, stats: dict) -> None:
        """Append a compact summary of a completed session to this person's history file."""
        record = {
            "session_id":         stats["session_id"],
            "ended_at":           datetime.now(timezone.utc).isoformat(),
            "duration_s":         stats["duration_s"],
            "activation_mean":    stats["activation"]["mean"],
            "activation_min":     stats["activation"]["min"],
            "activation_max":     stats["activation"]["max"],
            "zone_pct_calm":      stats["zone_pct"]["calm"],
            "zone_pct_moderate":  stats["zone_pct"]["moderate"],
            "zone_pct_activated": stats["zone_pct"]["activated"],
            "coherence_avg":      stats["coherence"]["avg"],
            "coherence_peak":     stats["coherence"]["peak"],
            "resp_first":         stats["resp_rate"]["first_third_avg"],
            "resp_last":          stats["resp_rate"]["last_third_avg"],
            "activated_episodes": stats["activated_episodes"],
        }
        with open(self._history_path(name), "a") as f:
            f.write(json.dumps(record) + "\n")

    def _load_history(self, name: str) -> list[dict]:
        path = self._history_path(name)
        if not path.exists():
            return []
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def _compute_progress(self, name: str) -> dict:
        """This person's history vs. their all-time average, for the AI report and dashboard."""
        history = self._load_history(name)
        session_count = len(history)
        is_first = session_count <= 1
        current = history[-1] if history else None
        prior = history[:-1] if len(history) > 1 else []

        def _avg(records, key):
            vals = [r[key] for r in records if r.get(key) is not None]
            return round(statistics.mean(vals), 1) if vals else None

        return {
            "is_first_session": is_first,
            "session_count": session_count,
            "current": {
                "activation_mean": current.get("activation_mean") if current else None,
                "zone_pct_calm":   current.get("zone_pct_calm") if current else None,
                "duration_s":      current.get("duration_s") if current else None,
            },
            "all_time_avg": {
                "activation_mean": _avg(prior, "activation_mean"),
                "zone_pct_calm":   _avg(prior, "zone_pct_calm"),
                "duration_s":      _avg(prior, "duration_s"),
            } if prior else None,
            "recent_series": [
                {
                    "session_id":      r["session_id"],
                    "ended_at":        r["ended_at"],
                    "activation_mean": r.get("activation_mean"),
                    "zone_pct_calm":   r.get("zone_pct_calm"),
                }
                for r in history[-10:]
            ],
        }

    # ── broadcast ─────────────────────────────────────────────────────────────

    async def broadcast(self, msg: dict) -> None:
        self._update_snapshot(msg)
        if not self.clients:
            return
        text = json.dumps(msg)
        dead = set()
        for ws in list(self.clients):
            try:
                await ws.send_str(text)
            except Exception:
                dead.add(ws)
        self.clients -= dead

    # ── metrics loop ─────────────────────────────────────────────────────────

    async def metrics_loop(self) -> None:
        while True:
            await asyncio.sleep(0.5)
            # A bug in any single tick (e.g. a bad get_updates() message) must
            # not permanently kill this loop for the rest of the process --
            # log it and keep going, rather than silently stopping all live
            # metrics until a manual restart.
            try:
                now = time.monotonic()
                for msg in self.proc_a.get_updates(now):
                    await self.broadcast(msg)
                if self.proc_b:
                    for msg in self.proc_b.get_updates(now):
                        await self.broadcast(msg)
                if self.dyadic:
                    for msg in self.dyadic.get_updates(self.proc_a, self.proc_b, now):
                        await self.broadcast(msg)
                if (self._session_file is not None
                        and now - self._last_log_snapshot >= 30.0):
                    self._last_log_snapshot = now
                    elapsed = round(now - self._session_start_mono, 1)
                    # exclude the growing dyadic trace array from the periodic
                    # log write -- it's redundant with the per-tick peak_r/
                    # lag_s/etc. already logged and would otherwise bloat
                    # every 30s snapshot line with an up-to-40-entry copy
                    snapshot_for_log = copy.deepcopy(self._metrics_snapshot)
                    snapshot_for_log.get("dyadic", {}).pop("trace", None)
                    self._session_file.write(json.dumps({
                        "type":              "metrics_snapshot",
                        "wall_time":         datetime.now(timezone.utc).isoformat(),
                        "session_elapsed_s": elapsed,
                        "metrics":           snapshot_for_log,
                    }) + "\n")
            except Exception as exc:
                print(f"[metrics_loop] ERROR (tick skipped): {exc}")

    # ── setup endpoints ───────────────────────────────────────────────────────

    async def state_handler(self, request: web.Request) -> web.Response:
        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "configured": self._configured,
                "names": {
                    "A": self.proc_a.name,
                    "B": self.proc_b.name if self.proc_b else None,
                },
            }),
            headers=self._NO_CACHE,
        )

    async def users_list_handler(self, request: web.Request) -> web.Response:
        return web.Response(
            content_type="application/json",
            text=json.dumps({"users": user_registry.list_users()}),
            headers=self._NO_CACHE,
        )

    async def users_create_handler(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
        except Exception:
            return web.Response(status=400, content_type="application/json",
                                text=json.dumps({"error": "invalid JSON"}))
        name = (data.get("name") or "").strip()
        if not name:
            return web.Response(status=400, content_type="application/json",
                                text=json.dumps({"error": "name is required"}))
        user = user_registry.create_user(name)
        return web.Response(content_type="application/json", text=json.dumps(user))

    async def users_delete_handler(self, request: web.Request) -> web.Response:
        user_id = request.match_info["id"]
        found = user_registry.delete_user(user_id)
        if not found:
            return web.Response(status=404, content_type="application/json",
                                text=json.dumps({"error": "user not found"}))
        # Also remove this person's meditation history — deleting the account
        # should delete their personal data, not just the registry entry.
        # Shared couple session logs under sessions/ are left alone since they
        # aren't owned by a single user and may still belong to a partner.
        history_path = self._history_path_for_id(user_id)
        history_path.unlink(missing_ok=True)
        return web.Response(content_type="application/json", text=json.dumps({"deleted": user_id}))

    def _history_path_for_id(self, user_id: str) -> Path:
        history_dir = Path(__file__).parent / "history"
        return history_dir / f"{user_id}.ndjson"

    async def scan_handler(self, request: web.Request) -> web.Response:
        from ble import scan_for_hr_monitors
        try:
            devices = await scan_for_hr_monitors(timeout=10.0)
            result = [{"name": d.name or "", "address": d.address} for d in devices]
        except Exception as exc:
            return web.Response(status=500, content_type="application/json",
                                text=json.dumps({"error": str(exc)}))
        return web.Response(content_type="application/json", text=json.dumps(result))

    async def configure_handler(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
        except Exception:
            return web.Response(status=400, content_type="application/json",
                                text=json.dumps({"error": "invalid JSON"}))

        simulate = data.get("simulate", False)

        if simulate:
            names = data.get("names", ["Partner A", "Partner B"])
            bpms  = data.get("bpm",   [68, 75])
            scenario = data.get("scenario", "low_low")
            self.proc_a.name = names[0] if names else "Partner A"
            if self.proc_b and len(names) > 1:
                self.proc_b.name = names[1]
            user_registry.create_user(self.proc_a.name)
            if self.proc_b:
                user_registry.create_user(self.proc_b.name)
            if self.dyadic:
                self.dyadic.name_a = self.proc_a.name
                self.dyadic.name_b = self.proc_b.name if self.proc_b else "Partner B"

            from simulator import simulate_stream, ScenarioState

            # session_start and shared_resp_freq must be identical for both
            # partners' tasks — that shared reference is what makes "high
            # coupling" scenarios actually produce correlated signals. The
            # ScenarioState object is likewise shared by reference so the
            # live selector (set_scenario WS message) can switch both
            # streams' behavior mid-session without restarting them.
            session_start = time.monotonic()
            shared_resp_freq = random.uniform(0.20, 0.28)
            self._scenario_state = ScenarioState(scenario)
            self._sim_session_start = session_start

            async def _sim(idx: int, name: str, base_bpm: int, role: str) -> None:
                self.on_sensor_connect(idx)
                await simulate_stream(
                    name=name, base_bpm=base_bpm,
                    on_bpm=lambda bpm: self.on_bpm(idx, bpm),
                    on_rr=lambda rr: self.on_rr(idx, rr),
                    on_connect=lambda label: print(f"[sim] {label} connected"),
                    scenario=self._scenario_state, role=role,
                    session_start=session_start, shared_resp_freq=shared_resp_freq,
                )

            asyncio.create_task(_sim(0, self.proc_a.name, bpms[0] if bpms else 68, "a"))
            if self.proc_b and len(names) > 1:
                asyncio.create_task(_sim(1, self.proc_b.name,
                                         bpms[1] if len(bpms) > 1 else 75, "b"))
        else:
            partners = data.get("partners", [])
            if not partners:
                return web.Response(status=400, content_type="application/json",
                                    text=json.dumps({"error": "no partners specified"}))
            for p in partners:
                idx  = p.get("idx", 0)
                name = p.get("name", "").strip() or ("Partner A" if idx == 0 else "Partner B")
                if idx == 0:
                    self.proc_a.name = name
                elif idx == 1 and self.proc_b:
                    self.proc_b.name = name
                user_registry.create_user(name)
            if self.dyadic:
                self.dyadic.name_a = self.proc_a.name
                if self.proc_b:
                    self.dyadic.name_b = self.proc_b.name

            from ble import stream as ble_stream
            from battery import read_battery_once

            async def _ble(idx: int, address: str) -> None:
                proc = self.proc_a if idx == 0 else self.proc_b
                batt = await read_battery_once(address)
                if batt is not None:
                    self.on_battery(idx, batt)
                await ble_stream(
                    address=address,
                    name=proc.name,
                    on_bpm=lambda bpm: self.on_bpm(idx, bpm),
                    on_rr=lambda rr: self.on_rr(idx, rr),
                    on_connect=lambda label: (
                        print(f"[ble] {label} connected"),
                        self.on_sensor_connect(idx),
                    ),
                    on_disconnect=lambda label, reason: (
                        print(f"[ble] {label} disconnected: {reason}"),
                        self.on_sensor_disconnect(idx),
                    ),
                )

            for p in partners:
                asyncio.create_task(_ble(p["idx"], p["address"]))

        self._configured = True
        single_partner = (
            not simulate and len(data.get("partners", [])) == 1
        ) or (
            simulate and len(data.get("names", [])) < 2
        )
        await self.broadcast({
            "type": "session_init",
            "names": {
                "A": self.proc_a.name,
                "B": self.proc_b.name if self.proc_b else None,
            },
            "single_partner": single_partner,
            "meditation_session_active": self._meditation_active,
            "simulated": self._scenario_state is not None,
            "scenario": self._scenario_state.scenario if self._scenario_state else None,
        })
        return web.Response(content_type="application/json", text=json.dumps({"ok": True}))

    # ── AI report ─────────────────────────────────────────────────────────────

    async def report_handler(self, request: web.Request) -> web.StreamResponse | web.Response:
        if self._session_file is None or self._session_id is None:
            return web.Response(status=404, text="No session log yet.")
        path = Path(__file__).parent / "sessions" / f"{self._session_id}.ndjson"
        if not path.exists():
            return web.Response(status=404, text="Session file not found.")

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return web.Response(status=503, text="ANTHROPIC_API_KEY not set.")

        header_obj: dict | None = None
        events: list[dict] = []
        patches: dict[int, dict] = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = obj.get("type")
                if t == "header":
                    header_obj = obj
                elif t == "event":
                    events.append(obj)
                elif t == "response_patch":
                    patches[obj["seq"]] = obj.get("response_metrics", {})

        if not events:
            return web.Response(status=400, text="No transcript events recorded yet.")

        for ev in events:
            if ev["seq"] in patches:
                ev["response_metrics"] = patches[ev["seq"]]

        session_doc = {
            "names": header_obj.get("names", {}) if header_obj else {},
            "metric_guide": _METRIC_GUIDE,
            "events": events,
        }

        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=api_key)

        stream_resp = web.StreamResponse(headers={
            "Content-Type": "text/plain; charset=utf-8",
            **self._NO_CACHE,
        })
        await stream_resp.prepare(request)

        try:
            async with client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=1500,
                system=_REPORT_SYSTEM,
                messages=[{
                    "role": "user",
                    "content": f"Session data:\n\n{json.dumps(session_doc, indent=2)}",
                }],
            ) as stream:
                async for chunk in stream.text_stream:
                    await stream_resp.write(chunk.encode())
        except Exception as exc:
            print(f"[report] ERROR: {exc}")
            await stream_resp.write(f"\n\n[Error generating report: {exc}]".encode())

        await stream_resp.write_eof()
        return stream_resp

    async def meditation_report_handler(self, request: web.Request) -> web.StreamResponse | web.Response:
        session_id = self._last_session_id
        if session_id is None:
            return web.Response(status=404, text="No completed meditation session yet.")
        path = Path(__file__).parent / "sessions" / f"{session_id}.ndjson"
        if not path.exists():
            return web.Response(status=404, text="Session file not found.")

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return web.Response(status=503, text="ANTHROPIC_API_KEY not set.")

        stats = self._compute_meditation_stats(session_id)
        progress = self._compute_progress(self.proc_a.name)
        session_doc = {"metric_guide": _MEDITATION_METRIC_GUIDE, **stats, "progress": progress}

        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=api_key)

        stream_resp = web.StreamResponse(headers={
            "Content-Type": "text/plain; charset=utf-8",
            **self._NO_CACHE,
        })
        await stream_resp.prepare(request)

        try:
            async with client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=1500,
                system=_MEDITATION_REPORT_SYSTEM,
                messages=[{
                    "role": "user",
                    "content": f"Session data:\n\n{json.dumps(session_doc, indent=2)}",
                }],
            ) as stream:
                async for chunk in stream.text_stream:
                    await stream_resp.write(chunk.encode())
        except Exception as exc:
            print(f"[meditation_report] ERROR: {exc}")
            await stream_resp.write(f"\n\n[Error generating report: {exc}]".encode())

        await stream_resp.write_eof()
        return stream_resp

    # ── app factory ──────────────────────────────────────────────────────────

    def build_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/ws", self.ws_handler)
        app.router.add_get("/", self.index_handler)
        app.router.add_get("/mode_b", self.mode_b_handler)
        app.router.add_get("/meditate", self.meditate_handler)
        app.router.add_get("/whitepaper", self.whitepaper_handler)
        app.router.add_get("/static/{filename}", self.static_handler)
        app.router.add_post("/api/transcribe", self.transcribe_handler)
        app.router.add_post("/api/enroll/{partner}", self.enroll_handler)
        app.router.add_get("/api/session_download", self.session_download_handler)
        app.router.add_get("/api/report", self.report_handler)
        app.router.add_get("/api/meditation_report", self.meditation_report_handler)
        app.router.add_get("/api/state", self.state_handler)
        app.router.add_get("/api/scan", self.scan_handler)
        app.router.add_post("/api/configure", self.configure_handler)
        app.router.add_get("/api/users", self.users_list_handler)
        app.router.add_post("/api/users", self.users_create_handler)
        app.router.add_delete("/api/users/{id}", self.users_delete_handler)
        return app

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        app = self.build_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "localhost", self.port)
        await site.start()
        print(f"Server running at http://localhost:{self.port}/")
        asyncio.create_task(self.metrics_loop())
        # run forever
        await asyncio.Event().wait()
