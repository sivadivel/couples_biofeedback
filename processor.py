"""
processor.py
Tiered metrics adapter for the couples biofeedback web server.
Imports from specs/dyadic_biofeedback_metrics.py.

Implements biofeedback-spec-v2.md: robust (median/MAD) baselines, a two-signal
(HR+RMSSD) activation index that adds vagal/coherence only in controlled-
breathing contexts, per-metric window gating ("warming up" instead of a
stale number), speech/motion gating, reformalized confidence, two-signal
flooding with hysteresis, and surrogate-tested dyadic coupling.
"""

from __future__ import annotations
import sys
import os
import time
import numpy as np

import activation_model

# ── add specs/ to path so we can import the reference module ──────────────────
_SPECS = os.path.join(os.path.dirname(__file__), "specs")
if _SPECS not in sys.path:
    sys.path.insert(0, _SPECS)

from dyadic_biofeedback_metrics import (
    clean_rr,
    rr_time_axis,
    time_domain,
    frequency_domain,
    mccraty_coherence,
    windowed_lagged_xcorr,
    surrogate_test_coupling,
    moving_median_detrend,
    common_grid_hr,
    Baseline,
    dpa_flag,
)

# ── constants ─────────────────────────────────────────────────────────────────
# Metric windowing (spec v2 §1)
MID_WIN_S        = 30.0   # RMSSD target window
RMSSD_FLOOR_S    = 20.0   # RMSSD min reliable window (floor)
RMSSD_SPEECH_WIN_S = 55.0 # widen toward 45-60s when speech_fraction > 0.5
SLOW_WIN_S       = 60.0   # HF / respiration window
COHERENCE_WIN_S  = 120.0  # coherence needs its own longer stationary window

BASELINE_WIN_S  = 300.0  # max window for baseline capture (5 min)
BASELINE_SUB_S  = 60.0   # sub-window size for each baseline sample
BASELINE_STEP_S = 30.0   # step between sub-windows

FAST_CADENCE_S  = 0.0    # every beat (queued immediately)
MID_CADENCE_S   = 5.0    # emit every 5 s
SLOW_CADENCE_S  = 10.0   # emit every 10 s
DYADIC_CADENCE_S = 15.0  # dyadic every 15 s

HF_LO, HF_HI = 0.15, 0.40  # HF band (Hz) == 9-24 breaths/min

# Speech gating (spec v2 §4.1)
SPEECH_TAIL_S            = 2.0   # extend each speech segment by +2s (respiratory tail)
SPEECH_GATE_HF_COH_RESP  = 0.2   # above this speech_fraction, gate HF/coherence/resp off
SPEECH_WIDEN_RMSSD       = 0.5   # above this speech_fraction, widen RMSSD window + lower confidence

# Motion gating (spec v2 §4.2) -- no accelerometer available from ble.py's
# standard Heart Rate Service, so this app uses the spec's own documented
# fallback: RR-interval artifact rate, already computed by clean_rr's local-
# median ectopic rejection (functionally the same kind of check as the
# spec's suggested ">25% RR gap from local median" heuristic).
MOTION_QUALITY_GATE  = 0.5   # signal_quality below this -> gate off ("motion")
MOTION_QUALITY_WARN  = 0.7   # below this (but above gate) -> lower confidence only

# Activation index (spec v2 §3)
# A0 retuned from the spec's own suggested default (12, deep calm) after live
# testing: at-baseline (d=0) reading in the middle of the calm zone made the
# score feel like it was "starting pretty low." A0=37 anchors baseline near
# the bottom of the moderate (yellow) zone instead -- resting physiology's
# normal moment-to-moment fluctuation now sits close to the calm/moderate
# boundary rather than requiring a full swing up from deep calm.
#
# G also retuned down from the spec's default (26) after live testing showed
# activation pinned at 0 or 100 far too often -- d only needed to reach
# about -1.4/+2.4 to saturate, well within normal short-window HR/RMSSD
# noise. G=18 pushes the saturation points out to roughly d=-2.1/+3.5, so
# ordinary fluctuation (d roughly -1..+2) shows up as graduated movement
# across the scale instead of slamming into the bounds:
#   d=-2 -> ~1   d=-1 -> ~19   d=0 -> 37   d=+1 -> ~55   d=+2 -> ~73   d=+3 -> ~91
# [CAL] -- still provisional, see spec v2 §10 for real validation against self-report.
A0_ACTIVATION = 37.0   # resting anchor (bottom of moderate/yellow zone)
G_ACTIVATION  = 18.0   # gain
W_HR_TALK, W_RMSSD_TALK = 0.5, 0.5
W_HR_RECOVERY, W_RMSSD_RECOVERY, W_VAGAL_RECOVERY = 0.3, 0.3, 0.4
S_THRESH = 0.15  # activation-units/second, rising/falling slope threshold

# activation_state_model(): EMA smoothing applied to the WESAD model's raw
# P(activated) before scaling to 0-100. Logistic-regression probabilities
# saturate near 0/1 much more readily than the z-score formula above, which
# would otherwise make the index feel binary and destabilize direction/slope
# (tuned against the old scale's noise characteristics).
# [CAL] -- provisional, retune once live behavior is observed.
ACTIVATION_MODEL_EMA_ALPHA = 0.3

# Flooding (spec v2 §6)
T_HR_PCT      = 0.11   # midpoint of the 10-12% band, converted to z-units per partner
T_RMSSD_Z     = 1.0
HR_CEILING    = 95.0
FLOOD_SUSTAIN_S = 10.0  # must hold to raise
FLOOD_CLEAR_S   = 5.0   # must hold false to clear

# Fixed, population-representative baseline for standalone meditation mode.
# Retuned for spec v2: robust (median, MAD-scale) center/scale, derived by
# running this module's own set_baseline() against three representative
# "typical resting" synthetic profiles (65/70/75 bpm), then pooling
# (median of centers, mean of scales) across those profiles.
#
# IMPORTANT SEMANTIC CHANGE vs. the pre-spec-v2 version of this constant:
# activation is no longer designed to center on 50 for "matches the
# reference." It's anchored near the bottom of the moderate/yellow zone
# (A0=37, retuned from spec v2's original suggested default of 12 after live
# testing -- see A0_ACTIVATION) and rises toward high/red under real
# activation (§3.2) -- this applies uniformly to meditate mode too now. A
# person whose physiology matches this baseline reads near the calm/moderate
# boundary, not straddling 50/50; excursions into high reflect genuine
# departures from their resting state.
STANDARD_BASELINE_STATS: dict[str, tuple[float, float]] = {
    "mean_hr":       (70.9, 6.9),
    "rmssd":         (62.4, 7.3),
    "hf":            (0.0055, 0.0026),
    "coherence":     (0.052, 0.018),
    "resp_rate":     (18.4, 1.7),
    "rsa_corrected": (0.0055, 0.0026),
    # hr_std derived differently from the rest: median/MAD of the WESAD
    # dataset's own "baseline" condition windows (n=557), since that's the
    # activation model's training population rather than this module's
    # synthetic profiles.
    "hr_std":        (5.3, 1.7),
}


def standard_baseline() -> Baseline:
    """A fixed Baseline shared by every meditation session — see STANDARD_BASELINE_STATS.
    Sets (center, scale) directly rather than going through fit() -- feeding
    fit() two symmetric synthetic points no longer recovers the scale
    verbatim now that it computes 1.4826*MAD instead of std."""
    bl = Baseline()
    for key, (center, scale) in STANDARD_BASELINE_STATS.items():
        bl.stats[key] = (center, scale)
        bl.meta[key] = {"n": None, "clean_fraction": None}
    return bl


def _resp_rate_from_rr(rr_ms: np.ndarray) -> float:
    """
    Derive respiration rate (br/min) as the peak frequency in the HF band
    of the Lomb-Scargle periodogram of the RR series.
    """
    if len(rr_ms) < 6:
        return float("nan")
    t = rr_time_axis(rr_ms)
    rr_c = rr_ms - rr_ms.mean()
    freqs = np.linspace(HF_LO, HF_HI, 256)
    from scipy import signal as scipy_signal
    pxx = scipy_signal.lombscargle(t, rr_c, 2 * np.pi * freqs, normalize=True)
    return float(freqs[np.argmax(pxx)] * 60.0)


def _hr_std(rr_ms: np.ndarray) -> float:
    """Std-dev of instantaneous HR (bpm) over a window -- distinct from
    time_domain()'s sdnn, which is std-dev of the RR intervals themselves
    (ms). This is the feature the WESAD activation model was trained on."""
    return float(np.std(60000.0 / rr_ms))


def _theil_sen_slope(t: list[float], y: list[float]) -> float:
    """Robust slope (units of y per second) over the last 60-90s of a trace (spec §3.4)."""
    if len(t) < 3:
        return 0.0
    try:
        from scipy.stats import theilslopes
        slope, *_ = theilslopes(y, t)
        return float(slope)
    except Exception:
        return 0.0


def _direction_from_slope(slope: float, s_thresh: float = S_THRESH) -> str:
    if slope > s_thresh:
        return "rising"
    if slope < -s_thresh:
        return "falling"
    return "stable"


def activation_state(metrics: dict, baseline: Baseline, mode: str = "conversation",
                      coherence_window_met: bool = False) -> dict:
    """
    0-100 activation index, spec v2 §3. Two-signal (HR+RMSSD) robust-z index
    during conversation; folds in vagal/coherence only in recovery/meditate
    mode once a long-enough stationary window is available (§3.3) -- three
    signals otherwise double-counts vagal-respiratory variance while adding
    the two most window-fragile, respiration-confounded channels.

    Returns activation plus the intermediate per-signal contributions, which
    the caller uses for confidence (§5) and description text.
    """
    z_hr    = float(np.clip(baseline.z("mean_hr", metrics["mean_hr"]), -3.0, 3.0))
    z_rmssd = float(np.clip(baseline.z("rmssd", metrics["rmssd"]), -3.0, 3.0))
    c_hr, c_rmssd = z_hr, -z_rmssd

    resp_rate = metrics.get("resp_rate")
    respiration_regular = resp_rate is not None and 4.0 <= resp_rate <= 12.0

    use_vagal = bool(mode == "recovery" and coherence_window_met and respiration_regular
                     and "coherence" in baseline.stats and metrics.get("coherence") is not None)

    if use_vagal:
        z_vagal = float(np.clip(baseline.z("coherence", metrics["coherence"]), -3.0, 3.0))
        c_vagal = -z_vagal
        d = W_HR_RECOVERY * c_hr + W_RMSSD_RECOVERY * c_rmssd + W_VAGAL_RECOVERY * c_vagal
        contribs = {"hr": c_hr, "rmssd": c_rmssd, "vagal": c_vagal}
    else:
        d = W_HR_TALK * c_hr + W_RMSSD_TALK * c_rmssd
        contribs = {"hr": c_hr, "rmssd": c_rmssd}

    activation = float(np.clip(A0_ACTIVATION + G_ACTIVATION * d, 0.0, 100.0))

    return {"activation": activation, "d": d, "contribs": contribs,
            "z_hr": z_hr, "z_rmssd": z_rmssd, "used_vagal": use_vagal}


def activation_state_model(metrics: dict, baseline: Baseline, mode: str, hr_std: float,
                            coherence_window_met: bool, ema_prev: float | None) -> dict:
    """
    Activation index driven by the WESAD-trained classifier instead of the
    hand-tuned z-score formula above -- calls activation_state() first purely
    to get contribs/z_hr/z_rmssd/used_vagal (still needed by
    compute_confidence() and describe_state()), then overrides `activation`
    with the model's smoothed P(activated) x 100. Falls back silently to the
    z-score activation on any failure (missing model file, a baseline fitted
    before hr_std existed, etc).
    """
    s = activation_state(metrics, baseline, mode, coherence_window_met)
    try:
        variant = "recovery" if s["used_vagal"] else "conversation"

        d_hr_mean   = metrics["mean_hr"] - baseline.stats["mean_hr"][0]
        d_hr_std    = hr_std - baseline.stats["hr_std"][0]
        d_log_rmssd = np.log1p(metrics["rmssd"]) - np.log1p(baseline.stats["rmssd"][0])
        d_log_coherence = None
        if variant == "recovery":
            d_log_coherence = np.log1p(metrics["coherence"]) - np.log1p(baseline.stats["coherence"][0])

        p = activation_model.predict_proba(variant, d_hr_mean, d_hr_std, d_log_rmssd, d_log_coherence)

        new_ema = p if ema_prev is None else (
            ACTIVATION_MODEL_EMA_ALPHA * p + (1 - ACTIVATION_MODEL_EMA_ALPHA) * ema_prev)

        s["activation"] = float(np.clip(100.0 * new_ema, 0.0, 100.0))
        s["ema_activation"] = new_ema
        s["model_variant"] = variant
    except Exception:
        pass
    return s


def compute_confidence(c_hr: float, c_rmssd: float, signal_quality: float | None,
                        speech_fraction: float) -> tuple[float, bool]:
    """
    Spec v2 §5: confidence = min(conf_quality, conf_agreement, conf_speech).
    conf_agreement is the formalized "signals mixed" check: HR and RMSSD
    point opposite directions with both |z| > 0.5. Returns (confidence, mixed).
    """
    conf_quality = float(np.clip(signal_quality, 0.0, 1.0)) if signal_quality is not None else 1.0
    mixed = bool(np.sign(c_hr) != np.sign(c_rmssd)) and (min(abs(c_hr), abs(c_rmssd)) > 0.5)
    conf_agreement = 0.4 if mixed else 1.0
    conf_speech = float(np.clip(1.0 - speech_fraction, 0.0, 1.0))
    confidence = min(conf_quality, conf_agreement, conf_speech)
    return confidence, mixed


def describe_state(name: str, s: dict, metrics: dict, baseline: Baseline,
                    flooded: bool, mixed: bool) -> str:
    """
    Multi-sentence body-state description. Describes only body facts — no
    emotion words, no causal claims.
    """
    level = ("low" if s["activation"] < 35 else
             "moderate" if s["activation"] < 65 else "high")
    resp = metrics.get("resp_rate")
    if resp is not None and np.isfinite(resp):
        breath = (f"slow ({resp:.0f}/min)" if resp < 10
                  else f"rapid ({resp:.0f}/min)" if resp > 20
                  else f"steady ({resp:.0f}/min)")
    else:
        breath = "unavailable this window"
    hedge = " (signals mixed)" if mixed else ""

    base_hr = baseline.stats["mean_hr"][0]
    hr = metrics.get("mean_hr", base_hr)
    hr_pct = round(baseline.pct_change("mean_hr", hr))

    parts = [f"{level} activation, {s['direction']}{hedge}."]

    if flooded:
        parts.append(
            f"Heart rate {hr:.0f} bpm — past flooding threshold. A break is indicated.")
    elif abs(hr_pct) <= 5:
        parts.append(f"Heart rate {hr:.0f} bpm — near baseline.")
    else:
        word = "above" if hr_pct > 0 else "below"
        parts.append(f"Heart rate {hr:.0f} bpm — {abs(hr_pct)}% {word} baseline.")

    parts.append(f"Breathing {breath}.")

    if "rmssd" in baseline.stats and metrics.get("rmssd") is not None:
        rz = baseline.z("rmssd", metrics["rmssd"])
        if rz < -1.0:
            parts.append("Heart rhythm variability reduced — vagal tone withdrawn.")
        elif rz > 1.0:
            parts.append("Heart rhythm variability elevated — parasympathetic active.")

    coh = metrics.get("coherence")
    if coh is not None:
        if coh > 0.6:
            parts.append("Heart rhythm organized — physiologically settled.")
        elif coh < 0.15 and s["activation"] > 50:
            parts.append("Heart rhythm disorganized.")

    return " ".join(parts)


class PartnerProcessor:
    """
    Per-partner tiered processor.

    Push RR intervals as they arrive via push_rr(); call get_updates() on the
    500 ms server tick to collect any messages that are ready to broadcast.
    """

    def __init__(self, name: str, partner_id: str, mode: str = "conversation"):
        self.name = name
        self.partner_id = partner_id  # "A" or "B"
        self.mode = mode  # "conversation" (couples) or "recovery" (paced-breathing / meditate)

        # rolling buffer: list of (wall_clock_s, rr_ms)
        self._buf: list[tuple[float, float]] = []

        # per-tier last-emit timestamps
        self._last_mid  = 0.0
        self._last_slow = 0.0

        # fast-tier queue: pending mean_hr values computed per beat
        self._fast_queue: list[float] = []

        # HR trace: last 60 s of mean-HR samples (one per beat) — used by dyadic panel
        self._trace_hr: list[float] = []
        self._trace_t:  list[float] = []  # relative seconds since session start

        # activation trace: last 10 min of slow-tier activation scores
        self._trace_act:   list[float] = []
        self._trace_act_t: list[float] = []

        # speech segments (start_mono, end_mono), approximated from completed
        # transcript events (no real-time VAD exists in this app -- see §4.1)
        self._speech_segments: list[tuple[float, float]] = []

        self.baseline: Baseline | None = None
        self._prev_activation: float | None = None
        self._ema_activation: float | None = None  # activation_state_model()'s smoothed P(activated)
        self._session_start = time.monotonic()
        self._calm_start_ts: float | None = None

        # flooding hysteresis state (spec §6): asymmetric sustain/clear timers
        self._flood_state: bool = False
        self._flood_true_since: float | None = None
        self._flood_false_since: float | None = None

    # ── ingest ────────────────────────────────────────────────────────────────

    def push_rr(self, rr_ms: float) -> None:
        now = time.monotonic()
        self._buf.append((now, rr_ms))

        # prune buffer — keep enough for baseline capture (up to 5 min)
        cutoff = now - (BASELINE_WIN_S + 5.0)
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.pop(0)

        # fast tier: compute mean_hr from recent clean beats (last 10 s)
        rr_arr = self._rr_within(now, 10.0)
        if len(rr_arr) >= 2:
            rr_clean, _ = clean_rr(rr_arr)
            if len(rr_clean) >= 2:
                td = time_domain(rr_clean)
                hr = td["mean_hr"]
                self._fast_queue.append(hr)
                # maintain sparkline (60 s)
                rel_t = now - self._session_start
                self._trace_hr.append(hr)
                self._trace_t.append(rel_t)
                # prune sparkline older than 60 s
                cutoff_t = rel_t - 60.0
                while self._trace_t and self._trace_t[0] < cutoff_t:
                    self._trace_t.pop(0)
                    self._trace_hr.pop(0)

    def record_speech_segment(self, start_mono: float, end_mono: float) -> None:
        """
        Called by server.py once a transcript chunk resolves to this partner.
        There is no real-time VAD in this app, so the segment is an
        approximation (the ~5s recording chunk), extended by the spec's +2s
        post-speech respiratory tail (§4.1).
        """
        self._speech_segments.append((start_mono, end_mono + SPEECH_TAIL_S))
        cutoff = start_mono - BASELINE_WIN_S
        while self._speech_segments and self._speech_segments[0][1] < cutoff:
            self._speech_segments.pop(0)

    def _speech_fraction(self, win_start: float, win_end: float) -> float:
        win_dur = win_end - win_start
        if win_dur <= 0 or not self._speech_segments:
            return 0.0
        covered = 0.0
        for (s, e) in self._speech_segments:
            s = max(s, win_start)
            e = min(e, win_end)
            if e > s:
                covered += (e - s)
        return float(np.clip(covered / win_dur, 0.0, 1.0))

    # ── helpers ───────────────────────────────────────────────────────────────

    def _rr_within(self, now: float, win_s: float) -> np.ndarray:
        cutoff = now - win_s
        return np.array([r for (t, r) in self._buf if t >= cutoff], dtype=float)

    def _buf_within(self, now: float, win_s: float):
        """Return (wall_times_array, rr_array) for the given window."""
        cutoff = now - win_s
        pairs = [(t, r) for (t, r) in self._buf if t >= cutoff]
        if not pairs:
            return np.array([]), np.array([], dtype=float)
        times, rrs = zip(*pairs)
        return np.array(times), np.array(rrs, dtype=float)

    def _update_flood_state(self, now: float, instant_flooded: bool) -> bool:
        """Asymmetric hysteresis: >=10s sustained to raise, >=5s sustained false to clear."""
        if instant_flooded:
            if self._flood_true_since is None:
                self._flood_true_since = now
            self._flood_false_since = None
        else:
            if self._flood_false_since is None:
                self._flood_false_since = now
            self._flood_true_since = None

        if not self._flood_state:
            if self._flood_true_since is not None and (now - self._flood_true_since) >= FLOOD_SUSTAIN_S:
                self._flood_state = True
        else:
            if self._flood_false_since is not None and (now - self._flood_false_since) >= FLOOD_CLEAR_S:
                self._flood_state = False
        return self._flood_state

    # ── tiered update ─────────────────────────────────────────────────────────

    def get_updates(self, now: float) -> list[dict]:
        msgs: list[dict] = []

        # ── fast ──────────────────────────────────────────────────────────────
        if self._fast_queue:
            hr = float(np.mean(self._fast_queue))
            self._fast_queue.clear()
            msgs.append({
                "type": "fast",
                "partner": self.partner_id,
                "mean_hr": round(hr, 1),
            })

        # ── mid (every 5 s, RMSSD window) ────────────────────────────────────
        if now - self._last_mid >= MID_CADENCE_S:
            self._last_mid = now

            speech_fraction = self._speech_fraction(now - MID_WIN_S, now)
            win_s = RMSSD_SPEECH_WIN_S if speech_fraction > SPEECH_WIDEN_RMSSD else MID_WIN_S
            rr = self._rr_within(now, win_s)

            rmssd_window_met = False
            mean_hr = rmssd = signal_quality = None
            # Default to the persisted hysteresis state, not False -- a window
            # with too little clean RR data (BLE dropout, motion) must not be
            # reported as an implicit flood-clear; that bypasses the >=5s
            # clear timer in _update_flood_state and desyncs the flood banner
            # from the (unaffected, still-elevated) activation trace.
            flooded = self._flood_state
            dpa = None
            hr_baseline_pct = None

            if len(rr) >= 5:
                rr_clean, mask = clean_rr(rr)
                signal_quality = round(float(mask.mean()), 3) if len(mask) else None
                if len(rr_clean) >= 5:
                    td = time_domain(rr_clean)
                    mean_hr = td["mean_hr"]
                    rmssd = td["rmssd"]
                    rmssd_window_met = len(rr_clean) >= 15  # ~20s floor of clean beats

                    if self.baseline is not None and "mean_hr" in self.baseline.stats:
                        bl_hr_center, bl_hr_scale = self.baseline.stats["mean_hr"]
                        z_hr = (mean_hr - bl_hr_center) / bl_hr_scale
                        z_rmssd = self.baseline.z("rmssd", rmssd) if "rmssd" in self.baseline.stats else 0.0
                        t_hr_z = (T_HR_PCT * bl_hr_center) / bl_hr_scale
                        dpa = dpa_flag(z_hr, z_rmssd, mean_hr,
                                       t_hr_z=t_hr_z, t_rmssd_z=T_RMSSD_Z, hr_ceiling=HR_CEILING)
                        flooded = self._update_flood_state(now, dpa["flooded"])
                        hr_baseline_pct = round(self.baseline.pct_change("mean_hr", mean_hr), 1)
                    else:
                        flooded = self._update_flood_state(now, False)

            motion_gated = signal_quality is not None and signal_quality < MOTION_QUALITY_GATE
            rmssd_ok = rmssd_window_met and not motion_gated
            rmssd_status = ("ok" if rmssd_ok else
                            "motion" if motion_gated else
                            "warming_up")

            msgs.append({
                "type": "mid",
                "partner": self.partner_id,
                "mean_hr": round(mean_hr, 1) if mean_hr is not None else None,
                "rmssd": round(rmssd, 1) if (rmssd is not None and rmssd_ok) else None,
                "rmssd_status": rmssd_status,
                "flooded": flooded,
                "dpa": dpa,
                "hr_baseline_pct": hr_baseline_pct,
                "signal_quality": signal_quality,
                "speech_fraction": round(speech_fraction, 2),
                "trace_hr": [round(v, 1) for v in self._trace_hr],
                "trace_times": [round(v, 2) for v in self._trace_t],
            })

        # ── slow (every 10 s) ─────────────────────────────────────────────────
        if now - self._last_slow >= SLOW_CADENCE_S:
            self._last_slow = now

            speech_fraction_slow = self._speech_fraction(now - SLOW_WIN_S, now)
            speech_gated = speech_fraction_slow > SPEECH_GATE_HF_COH_RESP

            rr = self._rr_within(now, SLOW_WIN_S)
            hf_window_met = len(rr) >= 20
            hf = coherence = resp_rate = None
            signal_quality_slow = None

            if hf_window_met and not speech_gated:
                rr_clean, mask = clean_rr(rr)
                signal_quality_slow = round(float(mask.mean()), 3) if len(mask) else None
                if len(rr_clean) >= 20:
                    td_slow = time_domain(rr_clean)
                    fd = frequency_domain(rr_clean, method="lombscargle")
                    hf = fd["hf"]
                    resp = _resp_rate_from_rr(rr_clean)
                    resp_rate = resp if np.isfinite(resp) and resp >= HF_LO * 60.0 else None
                else:
                    hf_window_met = False

            # coherence needs its own longer, independently-pulled window
            rr_coh = self._rr_within(now, COHERENCE_WIN_S)
            coherence_window_met = len(rr_coh) >= 40
            if coherence_window_met and not speech_gated:
                rr_coh_clean, _ = clean_rr(rr_coh)
                if len(rr_coh_clean) >= 40:
                    coh = mccraty_coherence(rr_coh_clean)
                    coherence = coh if np.isfinite(coh) else None
                else:
                    coherence_window_met = False

            hf_motion = signal_quality_slow is not None and signal_quality_slow < MOTION_QUALITY_GATE

            act_result = None
            state_desc = None
            confidence = None
            mixed = False
            if self.baseline is not None and "mean_hr" in self.baseline.stats and "hr_std" in self.baseline.stats:
                # activation itself only needs HR+RMSSD (always available at
                # mid cadence); pull the latest mid-tier values fresh here
                rr_mid = self._rr_within(now, MID_WIN_S)
                if len(rr_mid) >= 5:
                    rr_mid_clean, mid_mask = clean_rr(rr_mid)
                    if len(rr_mid_clean) >= 5:
                        td_mid = time_domain(rr_mid_clean)
                        m = {
                            "mean_hr": td_mid["mean_hr"],
                            "rmssd": td_mid["rmssd"],
                            "resp_rate": resp_rate,
                            "coherence": coherence,
                        }
                        try:
                            s = activation_state_model(
                                m, self.baseline, mode=self.mode, hr_std=_hr_std(rr_mid_clean),
                                coherence_window_met=(coherence_window_met and not speech_gated and not hf_motion),
                                ema_prev=self._ema_activation)
                            self._ema_activation = s.get("ema_activation", self._ema_activation)
                            conf, mixed = compute_confidence(
                                s["contribs"]["hr"], s["contribs"]["rmssd"],
                                float(mid_mask.mean()) if len(mid_mask) else None,
                                speech_fraction_slow,
                            )
                            confidence = conf

                            rel_t = now - self._session_start
                            self._trace_act.append(s["activation"])
                            self._trace_act_t.append(rel_t)
                            cutoff_act = rel_t - 600.0
                            while self._trace_act_t and self._trace_act_t[0] < cutoff_act:
                                self._trace_act_t.pop(0)
                                self._trace_act.pop(0)

                            recent_mask = [t >= rel_t - 90.0 for t in self._trace_act_t]
                            recent_t = [t for t, keep in zip(self._trace_act_t, recent_mask) if keep]
                            recent_y = [v for v, keep in zip(self._trace_act, recent_mask) if keep]
                            slope = _theil_sen_slope(recent_t, recent_y)
                            direction = _direction_from_slope(slope)
                            s["direction"] = direction

                            self._prev_activation = s["activation"]
                            state_desc = describe_state(self.name, s, {**m, "mean_hr": m["mean_hr"]},
                                                        self.baseline, flooded=self._flood_state, mixed=mixed)
                            act_result = s
                        except Exception:
                            pass

            # calm-zone tracking
            calm_zone_s = 0
            if act_result and act_result["activation"] < 35:
                if self._calm_start_ts is None:
                    self._calm_start_ts = now
                calm_zone_s = int(now - self._calm_start_ts)
            else:
                self._calm_start_ts = None

            def _gated_status(window_met: bool) -> str:
                if speech_gated:
                    return "paused_speech"
                if hf_motion:
                    return "motion"
                if not window_met:
                    return "warming_up"
                return "ok"

            hf_status = _gated_status(hf_window_met)
            coh_status = _gated_status(coherence_window_met)
            resp_status = _gated_status(hf_window_met) if resp_rate is not None else _gated_status(False)

            msgs.append({
                "type": "slow",
                "partner": self.partner_id,
                "hf": round(hf, 5) if (hf is not None and hf_status == "ok") else None,
                "hf_status": hf_status,
                "coherence": round(coherence, 3) if (coherence is not None and coh_status == "ok") else None,
                "coherence_status": coh_status,
                "resp_rate": round(resp_rate, 1) if (resp_rate is not None and resp_status == "ok") else None,
                "resp_rate_status": resp_status,
                "activation": round(act_result["activation"], 1) if act_result else None,
                "direction":  act_result["direction"] if act_result else None,
                "confidence": round(confidence, 2) if confidence is not None else None,
                "signals_mixed": mixed,
                "used_vagal": act_result["used_vagal"] if act_result else False,
                "model_variant": act_result.get("model_variant") if act_result else None,
                "state_description": state_desc,
                "calm_zone_s": calm_zone_s,
                "speech_fraction": round(speech_fraction_slow, 2),
                "trace_activation": [round(v, 1) for v in self._trace_act],
            })

        return msgs

    # ── baseline ──────────────────────────────────────────────────────────────

    def set_baseline(self) -> bool:
        """
        Fit calm baseline using up to 5 minutes of buffered data.
        Slides a 60 s sub-window every 30 s to collect multiple samples,
        giving Baseline.fit() a real distribution so the robust scale is
        meaningful (spec v2 §2 -- median/MAD, not mean/SD).
        Returns True on success, False if not enough data.
        """
        now = time.monotonic()
        times, rr_full = self._buf_within(now, BASELINE_WIN_S)
        if len(rr_full) < 20:
            return False

        samples: dict[str, list[float]] = {
            "mean_hr": [], "rmssd": [], "hf": [],
            "coherence": [], "resp_rate": [], "hr_std": [],
        }

        # slide a 60 s window backward from now in 30 s steps
        t_end = times[-1]
        t_first = times[0]
        win_end = t_end
        while win_end - BASELINE_SUB_S >= t_first:
            win_start = win_end - BASELINE_SUB_S
            mask = (times >= win_start) & (times <= win_end)
            rr_w = rr_full[mask]
            if len(rr_w) >= 15:
                rr_c, _ = clean_rr(rr_w)
                if len(rr_c) >= 15:
                    td = time_domain(rr_c)
                    fd = frequency_domain(rr_c, method="lombscargle")
                    coh = mccraty_coherence(rr_c)
                    rsp = _resp_rate_from_rr(rr_c)
                    samples["mean_hr"].append(td["mean_hr"])
                    samples["rmssd"].append(td["rmssd"])
                    samples["hf"].append(fd["hf"])
                    samples["coherence"].append(coh if np.isfinite(coh) else 0.0)
                    samples["resp_rate"].append(rsp if np.isfinite(rsp) else 15.0)
                    samples["hr_std"].append(_hr_std(rr_c))
            win_end -= BASELINE_STEP_S

        if not samples["mean_hr"]:
            # less than one full sub-window available — use what we have
            rr_c, _ = clean_rr(rr_full)
            if len(rr_c) < 15:
                return False
            td = time_domain(rr_c)
            fd = frequency_domain(rr_c, method="lombscargle")
            coh = mccraty_coherence(rr_c)
            rsp = _resp_rate_from_rr(rr_c)
            samples["mean_hr"]   = [td["mean_hr"]]
            samples["rmssd"]     = [td["rmssd"]]
            samples["hf"]        = [fd["hf"]]
            samples["coherence"] = [coh if np.isfinite(coh) else 0.0]
            samples["resp_rate"] = [rsp if np.isfinite(rsp) else 15.0]
            samples["hr_std"]    = [_hr_std(rr_c)]

        bl = Baseline()
        clean_fraction = len(rr_full) and None
        bl.fit("mean_hr",       samples["mean_hr"])
        bl.fit("rmssd",         samples["rmssd"])
        bl.fit("hf",            samples["hf"])
        bl.fit("coherence",     samples["coherence"])
        bl.fit("resp_rate",     samples["resp_rate"])
        bl.fit("rsa_corrected", samples["hf"])  # alias, kept for compatibility
        bl.fit("hr_std",        samples["hr_std"])

        self.baseline = bl
        return True

    def set_standard_baseline(self) -> bool:
        """
        Apply the fixed population baseline (see STANDARD_BASELINE_STATS) instead
        of fitting one from this person's own buffered data. Used by standalone
        meditation mode, where every session should score against the same
        physiological reference point rather than each person's own resting state.
        Always succeeds — no buffered data is required.
        """
        self.baseline = standard_baseline()
        self._prev_activation = None
        self._ema_activation = None
        self._calm_start_ts = None
        return True

    def set_mode(self, mode: str) -> None:
        """
        "conversation" (default, two-signal HR+RMSSD) or "recovery" (paced
        breathing / meditate, three-signal once a 120s stationary window is
        available -- spec v2 §3.3). Does not affect the baseline itself.
        """
        if mode in ("conversation", "recovery"):
            self.mode = mode

    def clear_baseline(self) -> None:
        """Remove the fitted baseline and reset activation state."""
        self.baseline = None
        self._prev_activation = None
        self._ema_activation = None
        self._calm_start_ts = None
        self._trace_act.clear()
        self._trace_act_t.clear()
        self._flood_state = False
        self._flood_true_since = None
        self._flood_false_since = None

    def reconnect_snapshot(self) -> dict:
        """Return current trace data for immediate WebSocket reconnect replay."""
        return {
            "trace_hr":         list(self._trace_hr),
            "trace_activation": [round(v, 1) for v in self._trace_act],
        }

    # ── for dyadic use ────────────────────────────────────────────────────────

    def hr_series(self, win_s: float = 60.0):
        """Return (timestamps_s, hr_values) arrays for the last win_s seconds."""
        now = time.monotonic()
        rr = self._rr_within(now, win_s)
        if len(rr) < 4:
            return np.array([]), np.array([])
        rr_clean, _ = clean_rr(rr)
        if len(rr_clean) < 4:
            return np.array([]), np.array([])
        t = rr_time_axis(rr_clean)
        hr = 60000.0 / rr_clean
        return t, hr


class DyadicProcessor:
    """Computes inter-partner coupling metrics every DYADIC_CADENCE_S seconds."""

    DETREND_WIN_S = 150.0  # spec v2 §7.1 default window W

    def __init__(self, name_a: str, name_b: str):
        self.name_a = name_a
        self.name_b = name_b
        self._last_dyadic = 0.0
        self._session_start = time.monotonic()
        # last 10 min of dyadic ticks, for the coupling time-series chart
        self._trace: list[dict] = []

    def get_updates(
        self,
        proc_a: PartnerProcessor,
        proc_b: PartnerProcessor,
        now: float,
    ) -> list[dict]:
        if now - self._last_dyadic < DYADIC_CADENCE_S:
            return []
        self._last_dyadic = now

        t_a, hr_a = proc_a.hr_series(win_s=self.DETREND_WIN_S)
        t_b, hr_b = proc_b.hr_series(win_s=self.DETREND_WIN_S)

        if len(hr_a) < 8 or len(hr_b) < 8:
            return []

        # interpolate both onto a common 4 Hz grid
        t_start = max(t_a[0], t_b[0])
        t_end   = min(t_a[-1], t_b[-1])
        if t_end - t_start < 5.0:
            return []

        FS = 4.0
        from scipy import interpolate as sci_interp
        grid = np.arange(t_start, t_end, 1.0 / FS)
        hr_a_even = sci_interp.interp1d(t_a, hr_a, bounds_error=False,
                                        fill_value="extrapolate")(grid)
        hr_b_even = sci_interp.interp1d(t_b, hr_b, bounds_error=False,
                                        fill_value="extrapolate")(grid)

        # spec v2 §7.1: subtract a moving median (not just linear detrend) to
        # reduce spurious coupling from shared slow drift before correlating
        hr_a_even = moving_median_detrend(hr_a_even, window_s=10.0, fs=FS)
        hr_b_even = moving_median_detrend(hr_b_even, window_s=10.0, fs=FS)

        n_surrogates = 200 if len(grid) >= 40 else 50  # fewer for very short windows
        result = surrogate_test_coupling(hr_a_even, hr_b_even, FS, max_lag_s=4.0,
                                         n_surrogates=n_surrogates)
        peak_r = result["peak_r"]
        # windowed_lagged_xcorr's raw lag_s is the OPPOSITE sign of "who leads":
        # for lag>0 it correlates a[lag:] against b[:-lag], which peaks when
        # b's pattern happens BEFORE a's (b leads a). Negate so positive here
        # consistently means "a leads b", matching the leader logic below and
        # the dyadic chart's up=A/down=B convention.
        lag_s  = -result["lag_s"]
        r_values = [round(v, 3) for v in result.get("r_values", [])]

        phase = "in-phase" if peak_r >= 0 else "anti-phase"
        # positive lag_s = A leads B; negative = B leads A
        if abs(lag_s) < 1.0 / FS:
            leader = "tied"
        elif lag_s > 0:
            leader = proc_a.name
        else:
            leader = proc_b.name

        rel_t = now - self._session_start
        self._trace.append({
            "t":            round(rel_t, 2),
            "peak_r":       round(float(peak_r), 3),
            "lag_s":        round(float(lag_s), 2),
            "leader":       leader,
            "phase":        phase,
            "above_chance": result["above_chance"],
        })
        cutoff_t = rel_t - 600.0
        while self._trace and self._trace[0]["t"] < cutoff_t:
            self._trace.pop(0)

        return [{
            "type": "dyadic",
            "peak_r":       round(float(peak_r), 3),
            "lag_s":        round(float(lag_s), 2),
            "phase":        phase,
            "leader":       leader,
            "r_values":     r_values,
            "lag_step_s":   result.get("lag_step_s", 0.25),
            "p_value":      round(result["p_value"], 3),
            "percentile":   round(result["percentile"], 1),
            "above_chance": result["above_chance"],
            "trace":        list(self._trace),
        }]

    def reconnect_snapshot(self) -> dict:
        """Return current trace data for immediate WebSocket reconnect replay."""
        return {"trace": list(self._trace)}
