# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A real-time biofeedback web app for couples therapy sessions. It streams live physiological
metrics from two wearable BLE heart rate monitors, annotates a session transcript with
speaker-attributed speech and physio data, and produces a downloadable NDJSON session log for
AI-assisted post-session analysis by the therapist.

The repo contains **two separate apps** that share `processor.py`:

- **Root app** (`launch.py`, `server.py`, `ble.py`, ...): the full single-tenant app. One process
  per couple, server-side BLE connection via `bleak`, recording/transcription/voice-ID/dyadic
  coupling/meditation. Meant to run on a machine with direct Bluetooth adapter access, physically
  near the wearables (not a cloud box).
- **`web/` app** (`web/server.py`, `web/room.py`): a simplified multi-tenant variant. Many
  concurrent "rooms," each browser pairs its own strap via the **Web Bluetooth API** and relays
  RR intervals over WebSocket. No recording/transcription/dyadic coupling/meditation — just live
  per-partner metrics and a recovery/breathing mode. Fully in-memory/ephemeral, no session logs.
  This is what's Dockerized and deployed to Fly.io (`fly.toml` + `web/Dockerfile`).

`fly.toml` deliberately lives at the repo root, not in `web/`, because Fly uses the directory
containing it as the Docker build context — the web app's image needs both `web/` and the
shared root-level `processor.py` / `specs/`.

## HARD CONSTRAINT — do not modify

`ble.py`, `simulator.py`, `main.py`, `plot.py`, `specs/dyadic_biofeedback_metrics.py`

These are the data acquisition and processing core. Breaking them breaks the sensor pipeline.

## Commands

Root app (from repo root, using `.venv`):

```bash
# simulated data, no hardware needed, two partners
.venv/bin/python launch.py --simulate --names "Alex" "Jordan" --bpm 68 75

# simulated data, one partner
.venv/bin/python launch.py --simulate --names "Alex" --bpm 68

# real BLE hardware, addresses known
.venv/bin/python launch.py --addresses UUID1 UUID2 --names "Alex" "Jordan"

# real BLE hardware, interactive scan/setup in browser
.venv/bin/python launch.py --names "Alex" "Jordan"

# solo meditation mode
.venv/bin/python launch.py --simulate --names "Solo" --meditate

# simulate a specific dyadic scenario (low_low/low_high/high_low/high_high/mixed_low/mixed_high)
.venv/bin/python launch.py --simulate --names "Alex" "Jordan" --scenario high_high
```

Serves at `http://localhost:8765` by default (`--port` to override).

`web/` app:

```bash
cd web && python server.py --port 8080   # or PORT env var
```

There is no test suite, linter, or build step in this repo — verify changes by running the
app (prefer `--simulate` first) and checking the browser/terminal output.

## Environment

- Python 3.13, deps in `requirements.txt` (root) / `web/requirements.txt` (web app), installed
  into `.venv/`.
- `ffmpeg` required at the system level (WebM → WAV conversion for transcription/voice-ID).
- `ANTHROPIC_API_KEY` env var enables emotion tone analysis (`emotion.py`); silently skipped if
  unset.
- Whisper model (`base`) is lazy-loaded on first transcription call; on networks with SSL
  interception, pre-download to `~/.cache/whisper/base.pt` to avoid a cert error at runtime.
- Root app is meant to run on a machine with a real Bluetooth adapter reachable by `bleak`
  (BlueZ + `bluetoothd` on Linux, user in the `bluetooth` group or `setcap` on the python binary
  to avoid needing root).

## Architecture (root app)

### Key files

| File | Role |
|---|---|
| `server.py` | aiohttp server, WebSocket broadcast, metric snapshot + buffer, transcript/enroll/report handlers, session log |
| `processor.py` | `PartnerProcessor`, `DyadicProcessor` — R-R interval → metrics pipeline (shared with `web/` via `room.py`) |
| `transcription.py` | Whisper wrapper, single `ThreadPoolExecutor`, ffmpeg suffix detection |
| `voice_id.py` | resemblyzer GE2E embeddings, ffmpeg WebM→WAV pre-conversion, cosine similarity, `identify()` |
| `emotion.py` | Per-utterance tone classification (aggressive/neutral/kind) via Claude Haiku |
| `users.py` | Persistent user registry (`users.json`), shared slug scheme with meditation history files |
| `static/app.js` | All couples-dashboard UI logic: recording cycle, enrollment, transcript rendering, WS handling |
| `static/meditate.js` | Solo meditation mode UI |

### Metric pipeline

```
BLE R-R interval → PartnerProcessor.push_rr()
  → metrics_loop() every 0.5s → proc.get_updates(now)
  → fast (250ms): mean_hr
  → mid (1s): mean_hr, rmssd, flooded, hr_baseline_pct
  → slow (4s): hf, coherence, resp_rate, activation, direction, calm_zone_s
  → broadcast(msg) → _update_snapshot(msg) → _metric_buffer.append(...)
                   → WebSocket clients
```

Metric definitions: activation zones are <35 teal, 35–65 amber, ≥65 red. McCraty coherence is
`peak_power / (total_power - peak_power)`. HF power uses Lomb-Scargle over 0.15–0.4Hz,
normalized, ×10k for display.

### Session log (NDJSON in `sessions/`, gitignored)

```
{"type":"header","session_id":"...","started_at":"...","names":{...}}
{"type":"event","seq":0,"wall_time":"...","session_elapsed_s":12.3,
 "text":"I feel unheard","speaker":"A",
 "pre_metrics":{...},   ← snapshot from 10s before the utterance started
 "metrics":{...}}       ← snapshot at utterance end
{"type":"response_patch","seq":0,"response_metrics":{...}}  ← written 15s later
```

The session download handler merges `response_patch` lines into their matching `event` by `seq`
and adds a `metric_guide` field.

### Audio chunk flow (recording → transcript)

```
Browser MediaRecorder (5s stop/restart cycle, NOT timeslice)
  → blob → sendAudioChunk(blob) → serialized via Promise chain (sendQueue)
  → POST /api/transcribe
  → transcribe_audio() + get_embedding() run concurrently via asyncio.gather
  → identify() cosine sim → speaker
  → _append_transcript_event() writes event with pre_metrics
  → _schedule_response_patch() schedules a 15s-delayed write
  → HTTP response body → UI renders immediately (not via WS)
  → WS broadcast also sent (deduped by _seenSeqs Set in JS)
```

### Voice enrollment

`enrollVoice(partner)` in JS is blocked while recording is active; it records 4s and POSTs to
`/api/enroll/{A|B|T}`. Server calls `get_embedding()` and stores it in `_voice_enrollments[partner]`.
`_embed()` in `voice_id.py` writes the audio to a temp file, runs `ffmpeg -ar 16000 -ac 1` to WAV,
then `preprocess_wav()` + `enc.embed_utterance()`, and cleans up both temp files.

### Concurrency notes

- All BLE callbacks go through `asyncio.create_task` via `call_soon_threadsafe`.
- `_enc_executor` (resemblyzer) and `_executor` (whisper) are separate single-worker
  `ThreadPoolExecutor`s — neither library is thread-safe, do not share or parallelize them.

### Untracked/gitignored data

`sessions/`, `history/`, `audio_raw_sources/`, `users.json` are all gitignored — they hold
per-run/per-user state, not code, and won't exist on a fresh clone.

## Architecture (`web/` app)

`web/server.py` hosts a `RoomRegistry` of `Room`s (`web/room.py`). Each room is identified by an
unguessable code, holds up to two `PartnerProcessor`s (imported unmodified from the root
`processor.py`), and is pruned once empty and idle (`prune_loop`, 60s interval). Browsers pair
their own strap via Web Bluetooth and push `rr_sample` messages over `/ws`; the server's
`metrics_loop` pulls `proc.get_updates()` the same way the root app does and broadcasts to both
clients in the room. No recording, transcription, or session persistence.
