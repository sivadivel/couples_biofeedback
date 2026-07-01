"""
server.py
aiohttp HTTP + WebSocket server for the couples biofeedback dashboard.
"""

from __future__ import annotations
import asyncio
import copy
import json
import os
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from aiohttp import web, WSMsgType

from processor import PartnerProcessor, DyadicProcessor
from transcription import transcribe_audio
from voice_id import get_embedding, identify
from emotion import classify_emotion

STATIC_DIR = Path(__file__).parent / "static"

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
  Composite arousal/dysregulation score.
  < 35  = regulated, calm zone (parasympathetic dominant)
  35–65 = moderate engagement (sympathetic rising)
  ≥ 65  = flooded (fight/flight/freeze territory — Gottman flooding threshold)
  KEY: Look for spikes in response_metrics.A.activation or response_metrics.B.activation
  after emotionally charged utterances.

mean_hr (bpm)
  Heart rate. Elevated HR = sympathetic arousal. Sustained elevated baseline before
  an utterance may indicate anticipatory stress or ongoing dysregulation.

rmssd (ms)
  Heart rate variability — root mean square of successive R-R interval differences.
  HIGHER = more parasympathetically regulated, more emotionally flexible.
  Drops within seconds of acute stress. Recovery of rmssd after a spike indicates
  the nervous system returning toward baseline.

hf (arbitrary units)
  Vagal activity — high-frequency (0.15–0.4 Hz) power of the R-R interval spectrum,
  normalized. HIGHER = more vagal tone and self-regulation capacity.
  NOTE: hf has a slower response time (~30–60s window) — don't expect large changes
  within a single 15s response window. It reflects sustained state, not acute spikes.

coherence (ratio)
  McCraty heart coherence: peak HRV spectral power / (total power − peak power).
  Higher coherence = more rhythmic, resonant heart rate pattern, often seen during
  regulated breathing and emotional composure. Values > 1.0 indicate strong coherence.

resp_rate (breaths/min)
  Respiration rate derived from the heart rate variability pattern.
  Elevated values (> 20 br/min) often accompany anxiety or hyperventilation.
  Slow, regular breathing (< 12 br/min) supports coherence and co-regulation.

flooded (proportion 0.0–1.0 in window averages)
  Boolean flag set when activation ≥ 65. In pre/response window averages, this
  represents the fraction of samples where the person was in flooded state.
  A value of 0.5 means flooded half the time in that window.

calm_zone_s (seconds)
  Continuous seconds spent below activation 35. Resets when activation rises above 35.
  Longer calm zone = sustained co-regulation. Look for what ended calm zones.

DYADIC COUPLING (metrics.dyadic)
---------------------------------
peak_r
  Pearson correlation of both partners' HR time series (windowed cross-correlation).
  Range −1 to +1. Higher = more heart rate synchrony.
  CRITICAL CAVEAT: High coupling does NOT equal co-regulation. If both partners are
  flooded simultaneously, peak_r may be high — this indicates "locked in conflict,"
  not safety. Always read peak_r against activation levels.

lag_s
  Seconds by which one partner's HR changes precede the other's.

leader
  Which partner's HR changes tend to precede the other's (physiological leadership).
  The leader may be driving the dyadic emotional state — or responding to it first.

phase
  "in-phase" = both HRs rise and fall together.
  "anti-phase" = one rises as the other falls (possible protective regulation).
  "uncorrelated" = no meaningful synchrony detected.

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

IMPORTANT CAVEATS
-----------------
- These are exploratory physiological proxies, not emotion readouts or relationship verdicts.
- Individual baseline physiology varies. Treat within-session changes as more meaningful
  than absolute values.
- Simultaneous speech blends voice embeddings and may yield speaker = null (unknown).
- The autonomic nervous system has individual variability in response latency.
  Some people show responses in 5s; others take 20–30s.
""".strip()


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
            )
        }
        self._metrics_snapshot: dict = {
            "A":      {**_partner_fields, "flooded": False, "calm_zone_s": 0},
            "B":      {**_partner_fields, "flooded": False, "calm_zone_s": 0},
            "dyadic": {"peak_r": None, "lag_s": None, "phase": None, "leader": None},
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
            snap["hr_baseline_pct"] = msg.get("hr_baseline_pct")
            snap["signal_quality"]  = msg.get("signal_quality")
            new_flooded = msg.get("flooded", False)
            snap["flooded"] = new_flooded
            if new_flooded != self._prev_flooded[p] and self._session_file is not None:
                evt = "flood_start" if new_flooded else "flood_end"
                self._log_lifecycle(evt, p,
                    mean_hr=msg.get("mean_hr"),
                    hr_baseline_pct=msg.get("hr_baseline_pct"))
            self._prev_flooded[p] = new_flooded
        elif mtype == "slow" and p in ("A", "B"):
            snap = self._metrics_snapshot[p]
            snap["hf"]               = msg.get("hf")
            snap["coherence"]        = msg.get("coherence")
            snap["resp_rate"]        = msg.get("resp_rate")
            snap["activation"]       = msg.get("activation")
            snap["direction"]        = msg.get("direction")
            snap["calm_zone_s"]      = msg.get("calm_zone_s", 0)
            snap["confidence"]       = msg.get("confidence")
            snap["state_description"] = msg.get("state_description")
        elif mtype == "dyadic":
            d = self._metrics_snapshot["dyadic"]
            d["peak_r"] = msg.get("peak_r")
            d["lag_s"]  = msg.get("lag_s")
            d["phase"]  = msg.get("phase")
            d["leader"] = msg.get("leader")
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
                self._session_file.write(json.dumps({
                    "type":              "metrics_snapshot",
                    "wall_time":         datetime.now(timezone.utc).isoformat(),
                    "session_elapsed_s": elapsed,
                    "metrics":           copy.deepcopy(self._metrics_snapshot),
                }) + "\n")

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
            self.proc_a.name = names[0] if names else "Partner A"
            if self.proc_b and len(names) > 1:
                self.proc_b.name = names[1]
            if self.dyadic:
                self.dyadic.name_a = self.proc_a.name
                self.dyadic.name_b = self.proc_b.name if self.proc_b else "Partner B"

            from simulator import simulate_stream

            async def _sim(idx: int, name: str, base_bpm: int) -> None:
                self.on_sensor_connect(idx)
                await simulate_stream(
                    name=name, base_bpm=base_bpm,
                    on_bpm=lambda bpm: self.on_bpm(idx, bpm),
                    on_rr=lambda rr: self.on_rr(idx, rr),
                    on_connect=lambda label: print(f"[sim] {label} connected"),
                )

            asyncio.create_task(_sim(0, self.proc_a.name, bpms[0] if bpms else 68))
            if self.proc_b and len(names) > 1:
                asyncio.create_task(_sim(1, self.proc_b.name,
                                         bpms[1] if len(bpms) > 1 else 75))
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
        })
        return web.Response(content_type="application/json", text=json.dumps({"ok": True}))

    # ── app factory ──────────────────────────────────────────────────────────

    def build_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/ws", self.ws_handler)
        app.router.add_get("/", self.index_handler)
        app.router.add_get("/mode_b", self.mode_b_handler)
        app.router.add_get("/whitepaper", self.whitepaper_handler)
        app.router.add_get("/static/{filename}", self.static_handler)
        app.router.add_post("/api/transcribe", self.transcribe_handler)
        app.router.add_post("/api/enroll/{partner}", self.enroll_handler)
        app.router.add_get("/api/session_download", self.session_download_handler)
        app.router.add_get("/api/state", self.state_handler)
        app.router.add_get("/api/scan", self.scan_handler)
        app.router.add_post("/api/configure", self.configure_handler)
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
