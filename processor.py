"""
processor.py
Tiered metrics adapter for the couples biofeedback web server.
Imports from specs/dyadic_biofeedback_metrics.py (read-only reference).
"""

from __future__ import annotations
import sys
import os
import time
import numpy as np

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
    common_grid_hr,
    Baseline,
    dpa_flag,
)

# ── constants ─────────────────────────────────────────────────────────────────
MID_WIN_S  = 30.0   # RMSSD window
SLOW_WIN_S = 60.0   # HF / coherence window
BASELINE_WIN_S  = 300.0  # max window for baseline capture (5 min)
BASELINE_SUB_S  = 60.0   # sub-window size for each baseline sample
BASELINE_STEP_S = 30.0   # step between sub-windows

FAST_CADENCE_S  = 0.0    # every beat (queued immediately)
MID_CADENCE_S   = 5.0    # emit every 5 s
SLOW_CADENCE_S  = 10.0   # emit every 10 s
DYADIC_CADENCE_S = 15.0  # dyadic every 15 s

HF_LO, HF_HI = 0.15, 0.40  # HF band (Hz)


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


def activation_state(
    metrics: dict, baseline: Baseline, prev: float | None = None, alpha: float = 0.5
) -> dict:
    """
    0–100 activation score per spec §4.5.
    Returns dict with activation, direction, confidence, flooded.
    """
    z = {k: float(np.clip(baseline.z(k, metrics[k]), -2.5, 2.5))
         for k in metrics if k in baseline.stats}

    slow_breathing = metrics.get("resp_rate", 99) < 9
    vagal = -z.get("rmssd", 0) if slow_breathing else -(z.get("rmssd", 0) + z.get("hf", 0)) / 2

    contribs = {
        "hr":    (z.get("mean_hr", 0),          1.0),
        "vagal": (vagal,                          1.0),
        "resp":  (z.get("resp_rate", 0),          0.5),
        "coher": (-z.get("coherence", 0), 0.0 if slow_breathing else 0.5),
    }
    total_w = sum(w for _, w in contribs.values())
    raw = sum(v * w for v, w in contribs.values()) / total_w if total_w else 0.0
    score = 100.0 / (1.0 + np.exp(np.clip(-raw / 1.5, -20.0, 20.0)))

    if prev is not None:
        score = alpha * score + (1.0 - alpha) * prev

    signs = [np.sign(v) for v, w in contribs.values() if w > 0]
    confidence = abs(sum(signs)) / len(signs) if signs else 1.0

    direction = ("rising"  if prev is not None and score > prev + 2 else
                 "falling" if prev is not None and score < prev - 2 else "stable")

    flooded = (metrics.get("mean_hr", 0) >= 1.10 * baseline.stats["mean_hr"][0])

    return {"activation": score, "direction": direction,
            "confidence": confidence, "flooded": flooded}


def describe_state(name: str, s: dict, metrics: dict, baseline: Baseline) -> str:
    """
    Multi-sentence body-state description per spec §4.7.
    Describes only body facts — no emotion words, no causal claims.
    """
    level = ("low" if s["activation"] < 35 else
             "moderate" if s["activation"] < 65 else "high")
    resp = metrics.get("resp_rate", 15.0)
    if np.isfinite(resp):
        breath = (f"slow ({resp:.0f}/min)" if resp < 10
                  else f"rapid ({resp:.0f}/min)" if resp > 20
                  else f"steady ({resp:.0f}/min)")
    else:
        breath = "steady"
    hedge = " (signals mixed)" if s["confidence"] <= 0.7 else ""

    base_hr = baseline.stats["mean_hr"][0]
    hr = metrics.get("mean_hr", base_hr)
    hr_pct = round(100.0 * (hr - base_hr) / base_hr)

    parts = [f"{level} activation, {s['direction']}{hedge}."]

    if s["flooded"]:
        parts.append(
            f"Heart rate {hr:.0f} bpm — past flooding threshold. A break is indicated.")
    elif abs(hr_pct) <= 5:
        parts.append(f"Heart rate {hr:.0f} bpm — near baseline.")
    else:
        word = "above" if hr_pct > 0 else "below"
        parts.append(f"Heart rate {hr:.0f} bpm — {abs(hr_pct)}% {word} baseline.")

    parts.append(f"Breathing {breath}.")

    if "rmssd" in baseline.stats:
        rmssd = metrics.get("rmssd", baseline.stats["rmssd"][0])
        rz = baseline.z("rmssd", rmssd)
        if rz < -1.0:
            parts.append("Heart rhythm variability reduced — vagal tone withdrawn.")
        elif rz > 1.0:
            parts.append("Heart rhythm variability elevated — parasympathetic active.")

    coh = metrics.get("coherence", 0.0)
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

    def __init__(self, name: str, partner_id: str):
        self.name = name
        self.partner_id = partner_id  # "A" or "B"

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

        self.baseline: Baseline | None = None
        self._prev_activation: float | None = None
        self._session_start = time.monotonic()
        self._calm_start_ts: float | None = None
        self._flood_count: int = 0  # consecutive mid-cycles above threshold

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

        # ── mid (every 5 s, 30 s window) ─────────────────────────────────────
        if now - self._last_mid >= MID_CADENCE_S:
            self._last_mid = now
            rr = self._rr_within(now, MID_WIN_S)
            if len(rr) >= 5:
                rr_clean, mask = clean_rr(rr)
                signal_quality = round(float(mask.mean()), 3) if len(mask) else None
                if len(rr_clean) >= 5:
                    td = time_domain(rr_clean)
                    mean_hr = td["mean_hr"]
                    rmssd = td["rmssd"]

                    flooded = False
                    dpa = None
                    hr_baseline_pct = None
                    if self.baseline is not None and "mean_hr" in self.baseline.stats:
                        bl_hr = self.baseline.stats["mean_hr"][0]
                        dpa_res = dpa_flag(mean_hr, bl_hr)
                        if dpa_res["flooded"]:
                            self._flood_count += 1
                        else:
                            self._flood_count = 0
                        flooded = self._flood_count >= 2
                        dpa = dpa_res
                        hr_baseline_pct = round(
                            100.0 * (mean_hr - bl_hr) / bl_hr, 1
                        )

                    msgs.append({
                        "type": "mid",
                        "partner": self.partner_id,
                        "mean_hr": round(mean_hr, 1),
                        "rmssd": round(rmssd, 1),
                        "flooded": flooded,
                        "dpa": dpa,
                        "hr_baseline_pct": hr_baseline_pct,
                        "signal_quality": signal_quality,
                        "trace_hr": [round(v, 1) for v in self._trace_hr],
                        "trace_times": [round(v, 2) for v in self._trace_t],
                    })

        # ── slow (every 10 s, 60 s window) ───────────────────────────────────
        if now - self._last_slow >= SLOW_CADENCE_S:
            self._last_slow = now
            rr = self._rr_within(now, SLOW_WIN_S)
            if len(rr) >= 20:
                rr_clean, _ = clean_rr(rr)
                if len(rr_clean) >= 20:
                    td_slow = time_domain(rr_clean)
                    fd = frequency_domain(rr_clean, method="lombscargle")
                    hf = fd["hf"]
                    coherence = mccraty_coherence(rr_clean)
                    resp_rate = _resp_rate_from_rr(rr_clean)

                    act_result = None
                    state_desc = None
                    if self.baseline is not None and "mean_hr" in self.baseline.stats:
                        m = {
                            "mean_hr":  td_slow["mean_hr"],
                            "rmssd":    td_slow["rmssd"],
                            "hf":       hf,
                            "resp_rate": resp_rate if np.isfinite(resp_rate) else 15.0,
                            "coherence": coherence if np.isfinite(coherence) else 0.0,
                        }
                        try:
                            s = activation_state(m, self.baseline, self._prev_activation)
                            self._prev_activation = s["activation"]
                            state_desc = describe_state(self.name, s, m, self.baseline)
                            act_result = s
                            # append to 10-min rolling activation trace
                            rel_t = now - self._session_start
                            self._trace_act.append(s["activation"])
                            self._trace_act_t.append(rel_t)
                            cutoff_act = rel_t - 180.0
                            while self._trace_act_t and self._trace_act_t[0] < cutoff_act:
                                self._trace_act_t.pop(0)
                                self._trace_act.pop(0)
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

                    msgs.append({
                        "type": "slow",
                        "partner": self.partner_id,
                        "hf": hf,
                        "coherence": coherence if np.isfinite(coherence) else None,
                        "resp_rate": round(resp_rate, 1) if np.isfinite(resp_rate) else None,
                        "activation": round(act_result["activation"], 1) if act_result else None,
                        "direction":  act_result["direction"] if act_result else None,
                        "confidence": round(act_result["confidence"], 2) if act_result else None,
                        "state_description": state_desc,
                        "calm_zone_s": calm_zone_s,
                        "trace_activation": [round(v, 1) for v in self._trace_act],
                    })

        return msgs

    # ── baseline ──────────────────────────────────────────────────────────────

    def set_baseline(self) -> bool:
        """
        Fit calm baseline using up to 5 minutes of buffered data.
        Slides a 60 s sub-window every 30 s to collect multiple samples,
        giving Baseline.fit() a real distribution so std is meaningful.
        Returns True on success, False if not enough data.
        """
        now = time.monotonic()
        times, rr_full = self._buf_within(now, BASELINE_WIN_S)
        if len(rr_full) < 20:
            return False

        samples: dict[str, list[float]] = {
            "mean_hr": [], "rmssd": [], "hf": [],
            "coherence": [], "resp_rate": [],
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

        bl = Baseline()
        bl.fit("mean_hr",       samples["mean_hr"])
        bl.fit("rmssd",         samples["rmssd"])
        bl.fit("hf",            samples["hf"])
        bl.fit("coherence",     samples["coherence"])
        bl.fit("resp_rate",     samples["resp_rate"])
        bl.fit("rsa_corrected", samples["hf"])  # alias for activation_index

        self.baseline = bl
        return True

    def clear_baseline(self) -> None:
        """Remove the fitted baseline and reset activation state."""
        self.baseline = None
        self._prev_activation = None
        self._calm_start_ts = None
        self._trace_act.clear()
        self._trace_act_t.clear()

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

    def __init__(self, name_a: str, name_b: str):
        self.name_a = name_a
        self.name_b = name_b
        self._last_dyadic = 0.0

    def get_updates(
        self,
        proc_a: PartnerProcessor,
        proc_b: PartnerProcessor,
        now: float,
    ) -> list[dict]:
        if now - self._last_dyadic < DYADIC_CADENCE_S:
            return []
        self._last_dyadic = now

        t_a, hr_a = proc_a.hr_series(win_s=60.0)
        t_b, hr_b = proc_b.hr_series(win_s=60.0)

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

        result = windowed_lagged_xcorr(hr_a_even, hr_b_even, FS, max_lag_s=4.0)
        peak_r = result["peak_r"]
        lag_s  = result["lag_s"]
        r_values = [round(v, 3) for v in result.get("r_values", [])]

        phase = "in-phase" if peak_r >= 0 else "anti-phase"
        # positive lag_s = A leads B; negative = B leads A
        if abs(lag_s) < 1.0 / FS:
            leader = "tied"
        elif lag_s > 0:
            leader = proc_a.name
        else:
            leader = proc_b.name

        return [{
            "type": "dyadic",
            "peak_r":    round(float(peak_r), 3),
            "lag_s":     round(float(lag_s), 2),
            "phase":     phase,
            "leader":    leader,
            "r_values":  r_values,
            "lag_step_s": result.get("lag_step_s", 0.25),
        }]
