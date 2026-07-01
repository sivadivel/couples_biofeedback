"""
dyadic_biofeedback_metrics.py
=================================================================
Reference implementations of the physiological metrics proposed for
real-time couples-therapy biofeedback, organized in three layers:

    1. Individual physiological state (per partner)
    2. Respiration / RSA / cardiac coherence (uses both channels)
    3. Dyadic coupling (the relational layer)

plus personalized baselining and a composite arousal / flooding detector.

PRIMARY INPUTS
--------------
    rr_ms   : 1-D array of successive RR (interbeat) intervals in MILLISECONDS.
              This is NOT averaged HR-in-bpm. You need beat-to-beat timing
              from ECG R-peaks or PPG systolic peaks.
    resp    : 1-D array of a raw respiration waveform (e.g. RIP chest belt),
              evenly sampled at fs_resp Hz.

Dependencies: numpy, scipy  (no other requirements)

These are physiological proxies, not direct readouts of emotion. Validate
against self-report / therapist coding before trusting any threshold in session.
"""

from __future__ import annotations
import numpy as np
from scipy import signal, interpolate, stats

MS_PER_S = 1000.0

# numpy >=2.0 renamed trapz -> trapezoid; support both
_trapz = getattr(np, "trapezoid", None) or np.trapz


# =====================================================================
# 0. PREPROCESSING  -- artifact rejection on the RR series
# =====================================================================
def clean_rr(rr_ms, low=300.0, high=2000.0, ectopic_thresh=0.20):
    """
    Remove physiologically implausible and ectopic RR intervals.

    - Absolute range filter: keep 300-2000 ms (i.e. 30-200 bpm).
    - Ectopic filter: drop any interval differing from the running median
      of its neighbours by more than `ectopic_thresh` (20% is a common
      default; tighten for resting data, loosen for active/talking subjects).

    Returns (rr_clean, keep_mask). Downstream functions assume cleaned input.
    Artifact handling matters far more in a moving, conversing, aroused
    couple than in a resting lab subject -- every metric below inherits
    these errors if you skip this step.
    """
    rr = np.asarray(rr_ms, dtype=float)
    mask = (rr >= low) & (rr <= high)
    # local median (window of 5) ectopic rejection
    med = signal.medfilt(np.where(mask, rr, np.nan_to_num(np.nanmedian(rr))), 5)
    with np.errstate(invalid="ignore"):
        rel = np.abs(rr - med) / med
    mask &= rel <= ectopic_thresh
    return rr[mask], mask


def rr_time_axis(rr_ms):
    """Cumulative time (seconds) of each beat, t[0] = first interval end."""
    return np.cumsum(np.asarray(rr_ms, dtype=float)) / MS_PER_S


def resample_tachogram(rr_ms, fs=4.0, kind="cubic"):
    """
    Interpolate the (unevenly sampled) RR tachogram onto an even grid at
    `fs` Hz so it can be fed to FFT-based spectral / coherence estimators.
    4 Hz is the conventional resampling rate for HRV. Returns (t, rr_even).

    NOTE: for SHORT windows prefer the Lomb-Scargle path (welch_psd's
    `method="lombscargle"`), which operates on the raw event-timed series
    and avoids interpolation artifacts.
    """
    rr = np.asarray(rr_ms, dtype=float)
    t = rr_time_axis(rr)
    f = interpolate.interp1d(t, rr, kind=kind, fill_value="extrapolate")
    t_even = np.arange(t[0], t[-1], 1.0 / fs)
    return t_even, f(t_even)


# =====================================================================
# 1a. TIME-DOMAIN HRV
# =====================================================================
def time_domain(rr_ms):
    """
    Core time-domain HRV. RMSSD is the workhorse for short windows:
    vagally mediated, drops reliably under stress, validates best at
    ultra-short lengths.

        mean_hr  = 60000 / mean(RR)                       [bpm]
        SDNN     = std(RR)                                [ms]
        RMSSD    = sqrt( mean( diff(RR)^2 ) )             [ms]
        SDSD     = std( diff(RR) )                        [ms]
        pNN50    = 100 * #{|diff(RR)| > 50ms} / (N-1)     [%]
    """
    rr = np.asarray(rr_ms, dtype=float)
    d = np.diff(rr)
    return {
        "mean_rr": rr.mean(),
        "mean_hr": MS_PER_S * 60.0 / rr.mean(),
        "sdnn": rr.std(ddof=1),
        "rmssd": np.sqrt(np.mean(d ** 2)),
        "sdsd": d.std(ddof=1),
        "pnn50": 100.0 * np.sum(np.abs(d) > 50.0) / len(d),
    }


# =====================================================================
# 1b. NONLINEAR -- Poincare descriptors
# =====================================================================
def poincare(rr_ms):
    """
    Poincare (return-map) descriptors.
        SD1 = sqrt(0.5) * SDSD        (short-term variability ~ RMSSD/sqrt2)
        SD2 = sqrt(2*SDNN^2 - 0.5*SDSD^2)   (long-term variability)
        ratio = SD1/SD2

    SD1 carries essentially the same fast-vagal information as RMSSD;
    SD1/SD2 is a cheap shape index sometimes used in arousal classifiers.
    """
    rr = np.asarray(rr_ms, dtype=float)
    sdsd = np.diff(rr).std(ddof=1)
    sdnn = rr.std(ddof=1)
    sd1 = np.sqrt(0.5) * sdsd
    sd2 = np.sqrt(max(2.0 * sdnn ** 2 - 0.5 * sdsd ** 2, 0.0))
    return {"sd1": sd1, "sd2": sd2, "sd1_sd2": sd1 / sd2 if sd2 else np.nan}


# =====================================================================
# 1c. FREQUENCY-DOMAIN HRV
# =====================================================================
LF_BAND = (0.04, 0.15)
HF_BAND = (0.15, 0.40)


def frequency_domain(rr_ms, fs=4.0, method="lombscargle"):
    """
    LF / HF / LF:HF from the RR series.

    method="lombscargle"  -> operates on raw event-timed series, best for
                             SHORT/ultra-short windows (no interpolation).
    method="welch"        -> resample to `fs` Hz then Welch PSD.

    Caveats baked into the band logic:
      * HF (0.15-0.40 Hz) ~ respiratory / parasympathetic, BUT shifts with
        breathing rate -- if a partner breathes slower than 9/min (0.15 Hz)
        the respiratory peak moves INTO the LF band and HF becomes
        misleading. The `resp_rate_hz` argument lets you flag this.
      * LF/HF as "sympathovagal balance" is contested -- report it, don't
        over-interpret it.
      * Spectral measures need >= ~30-60 s of data to resolve these bands.
    """
    rr = np.asarray(rr_ms, dtype=float)
    rr = rr - rr.mean()

    if method == "lombscargle":
        t = rr_time_axis(rr_ms)
        freqs = np.linspace(LF_BAND[0], HF_BAND[1], 256)
        ang = 2 * np.pi * freqs
        pxx = signal.lombscargle(t, rr, ang, normalize=True)
    else:
        _, rr_even = resample_tachogram(rr_ms, fs=fs)
        rr_even = rr_even - rr_even.mean()
        nperseg = min(len(rr_even), int(fs * 60))
        freqs, pxx = signal.welch(rr_even, fs=fs, nperseg=nperseg)

    def band_power(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return _trapz(pxx[m], freqs[m]) if m.any() else 0.0

    lf, hf = band_power(*LF_BAND), band_power(*HF_BAND)
    return {
        "lf": lf, "hf": hf,
        "lf_hf": lf / hf if hf else np.nan,
        "hf_norm": hf / (lf + hf) if (lf + hf) else np.nan,
    }


# =====================================================================
# 2a. RESPIRATION METRICS
# =====================================================================
def respiration_metrics(resp, fs_resp):
    """
    Respiration rate, a tidal-volume proxy, and breath-to-breath
    variability from a raw respiration waveform.

        resp_rate  : dominant spectral peak in 0.1-0.5 Hz -> breaths/min
        tidal_proxy: mean peak-to-trough amplitude (uncalibrated, AU)
        rr_resp_cv : coefficient of variation of breath periods

    Resting adults breathe ~9-20/min; rate climbs with sympathetic
    activation. tidal_proxy is needed to CORRECT RSA (see rsa_correction).
    """
    x = np.asarray(resp, dtype=float)
    x = x - x.mean()

    # spectral respiration rate
    f, p = signal.welch(x, fs=fs_resp, nperseg=min(len(x), int(fs_resp * 30)))
    band = (f >= 0.1) & (f <= 0.5)
    f_peak = f[band][np.argmax(p[band])] if band.any() else np.nan
    resp_rate = f_peak * 60.0

    # peak detection for amplitude + period variability
    min_dist = int(fs_resp / 0.5)  # >= 0.5 Hz max
    peaks, _ = signal.find_peaks(x, distance=min_dist)
    troughs, _ = signal.find_peaks(-x, distance=min_dist)
    n = min(len(peaks), len(troughs))
    tidal = float(np.mean(x[peaks[:n]] - x[troughs[:n]])) if n else np.nan
    periods = np.diff(peaks) / fs_resp if len(peaks) > 2 else np.array([])
    cv = float(periods.std() / periods.mean()) if periods.size else np.nan

    return {"resp_rate": resp_rate, "resp_rate_hz": f_peak,
            "tidal_proxy": tidal, "rr_resp_cv": cv,
            "resp_peaks": peaks, "resp_troughs": troughs}


# =====================================================================
# 2b. RSA -- peak-to-trough (Grossman) + respiration correction
# =====================================================================
def rsa_peak_to_trough(rr_ms, resp, fs_resp):
    """
    Peak-to-trough (P2T) RSA: within each respiration cycle, the spread of
    instantaneous HR. Uses respiration to define cycle boundaries, so it
    needs both channels -- which is exactly the advantage of your setup.

        For each breath cycle (trough->trough of respiration):
            RSA_cycle = max(HR_within) - min(HR_within)   [bpm]
        RSA = mean(RSA_cycle)

    Returns RSA amplitude (bpm). High RSA ~ vagal/regulated; withdrawal
    (a drop) ~ activation. MUST be respiration-corrected before comparing
    across moments (see rsa_correction): rate & tidal volume drive RSA
    amplitude independent of vagal tone and can explain ~60% of its variance.
    """
    # instantaneous HR (bpm) on the respiration time grid
    t_beat = rr_time_axis(rr_ms)
    hr_inst = MS_PER_S * 60.0 / np.asarray(rr_ms, dtype=float)
    t_resp = np.arange(len(resp)) / fs_resp
    f = interpolate.interp1d(t_beat, hr_inst, bounds_error=False,
                             fill_value="extrapolate")
    hr_on_resp = f(t_resp)

    troughs, _ = signal.find_peaks(-(np.asarray(resp) - np.mean(resp)),
                                   distance=int(fs_resp / 0.5))
    amps = []
    for a, b in zip(troughs[:-1], troughs[1:]):
        seg = hr_on_resp[a:b]
        if seg.size > 1:
            amps.append(seg.max() - seg.min())
    return float(np.mean(amps)) if amps else np.nan


def fit_rsa_correction(rsa_vals, resp_rates, tidal_vals):
    """
    Fit RSA ~ b0 + b1*resp_rate + b2*tidal on BASELINE data spanning a range
    of natural breathing. Returns coefficients + the centering means.
    """
    X = np.column_stack([np.ones_like(rsa_vals), resp_rates, tidal_vals])
    beta, *_ = np.linalg.lstsq(X, np.asarray(rsa_vals, float), rcond=None)
    return {"beta": beta,
            "resp_mean": float(np.mean(resp_rates)),
            "tidal_mean": float(np.mean(tidal_vals))}


def apply_rsa_correction(rsa, resp_rate, tidal, fit):
    """
    Remove the breathing-parameter component so the residual reflects
    vagal tone rather than how fast/shallow the person happens to breathe.

        rsa_corrected = rsa - [ b1*(resp_rate - resp_mean)
                              + b2*(tidal - tidal_mean) ]
    """
    b = fit["beta"]
    return rsa - (b[1] * (resp_rate - fit["resp_mean"])
                  + b[2] * (tidal - fit["tidal_mean"]))


# =====================================================================
# 2c. CARDIAC COHERENCE / RESONANCE  (the biofeedback target)
# =====================================================================
def mccraty_coherence(rr_ms, fs=4.0):
    """
    HeartMath-style coherence score on the HRV spectrum:
      1. find the highest peak in 0.04-0.26 Hz
      2. integrate power in a 0.030 Hz window centered on that peak
      3. coherence = peak_power / (total_power - peak_power)

    High coherence = a single dominant ~0.1 Hz oscillation = the regulated /
    self-soothing state. This is the natural thing to display to a partner
    during a time-out, and is roughly the inverse of the flooding state.
    """
    _, rr_even = resample_tachogram(rr_ms, fs=fs)
    rr_even = rr_even - rr_even.mean()
    f, p = signal.welch(rr_even, fs=fs, nperseg=min(len(rr_even), int(fs * 60)))
    band = (f >= 0.04) & (f <= 0.26)
    if not band.any():
        return np.nan
    fb, pb = f[band], p[band]
    f0 = fb[np.argmax(pb)]
    df = f[1] - f[0]
    win = (f >= f0 - 0.015) & (f <= f0 + 0.015)
    # sum-based integration: robust when the window spans only 1-2 bins
    # (trapz over a single bin would collapse to 0 at short-window resolution)
    peak_power = np.sum(p[win]) * df
    total_power = np.sum(p) * df
    denom = total_power - peak_power
    return float(peak_power / denom) if denom > 0 else np.nan


def resp_cardiac_coherence(rr_ms, resp, fs_resp, fs=4.0):
    """
    Magnitude-squared coherence between RESPIRATION and the HR tachogram at
    the breathing frequency:

        C_xy(f) = |P_xy(f)|^2 / (P_xx(f) * P_yy(f))        in [0, 1]

    This is the cleanest use of having both channels: it measures how
    tightly heart-rate oscillation is locked to breathing (resonance),
    rather than inferring it from the tachogram alone. Returns coherence
    evaluated at the dominant respiration frequency.
    """
    # put both signals on a common even grid at `fs`
    t_beat = rr_time_axis(rr_ms)
    hr_inst = MS_PER_S * 60.0 / np.asarray(rr_ms, float)
    t_even = np.arange(t_beat[0], t_beat[-1], 1.0 / fs)
    hr_even = interpolate.interp1d(t_beat, hr_inst, bounds_error=False,
                                   fill_value="extrapolate")(t_even)
    t_resp = np.arange(len(resp)) / fs_resp
    resp_even = interpolate.interp1d(t_resp, resp, bounds_error=False,
                                     fill_value="extrapolate")(t_even)
    nper = min(len(t_even), int(fs * 30))
    f, cxy = signal.coherence(hr_even - hr_even.mean(),
                              resp_even - resp_even.mean(),
                              fs=fs, nperseg=nper)
    rm = respiration_metrics(resp, fs_resp)["resp_rate_hz"]
    idx = np.argmin(np.abs(f - rm)) if np.isfinite(rm) else np.argmax(cxy)
    return float(cxy[idx])


# =====================================================================
# 3. DYADIC COUPLING  (the relational layer)
# =====================================================================
def windowed_lagged_xcorr(a, b, fs, max_lag_s=4.0, detrend=True):
    """
    Time-lagged cross-correlation between two partners' evenly-sampled
    series (e.g. instantaneous HR resampled to a common grid, or RSA).

    Returns (peak_r, lag_s):
        peak_r : max |Pearson r| over lags in [-max_lag_s, +max_lag_s]
        lag_s  : the lag (seconds) at that peak; sign tells you who leads.

    IMPORTANT methodological notes:
      * DETREND first (default on) -- shared slow drift inflates synchrony
        spuriously. Differencing or linear detrend per window is standard.
      * SIGN matters: positive peak = in-phase coupling, negative = anti-
        phase. They mean different things (Reed/Randall/Butler).
      * Always compare against a SURROGATE/pseudo-couple baseline (shuffle
        or pair non-partners) before calling a value "synchrony" -- some
        coupling arises just from two humans in a room.
      * INTERPRET JOINTLY with each partner's arousal: high synchrony while
        both are flooded ("locked into conflict") is the opposite of high
        synchrony while both are coherent. Synchrony is the exploratory
        layer; per-partner arousal/coherence is the reliable core.
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    if detrend:
        a = signal.detrend(a); b = signal.detrend(b)
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    max_lag = int(max_lag_s * fs)
    best_r, best_lag = 0.0, 0
    r_list = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            r = np.corrcoef(a[:lag], b[-lag:])[0, 1]
        elif lag > 0:
            r = np.corrcoef(a[lag:], b[:-lag])[0, 1]
        else:
            r = np.corrcoef(a, b)[0, 1]
        r_list.append(float(r) if np.isfinite(r) else 0.0)
        if np.isfinite(r) and abs(r) > abs(best_r):
            best_r, best_lag = r, lag
    return {"peak_r": float(best_r), "lag_s": best_lag / fs,
            "r_values": r_list, "lag_step_s": 1.0 / fs}


def common_grid_hr(rr_ms, t_start, t_end, fs=4.0):
    """Instantaneous HR resampled to a shared [t_start, t_end] grid so two
    partners' series are directly comparable for synchrony."""
    t_beat = rr_time_axis(rr_ms)
    hr = MS_PER_S * 60.0 / np.asarray(rr_ms, float)
    grid = np.arange(t_start, t_end, 1.0 / fs)
    return grid, interpolate.interp1d(t_beat, hr, bounds_error=False,
                                      fill_value="extrapolate")(grid)


# =====================================================================
# 4. PERSONALIZED BASELINING + COMPOSITE ACTIVATION / FLOODING
# =====================================================================
class Baseline:
    """
    Per-partner resting baseline. EVERY threshold here is individual --
    population constants don't work. Collect a few minutes of calm
    baseline, store mean/SD per metric, express live values as z-scores
    or % change against it.
    """
    def __init__(self):
        self.stats = {}

    def fit(self, metric_name, values):
        v = np.asarray(values, float)
        self.stats[metric_name] = (float(v.mean()), float(v.std() + 1e-9))
        return self

    def z(self, metric_name, value):
        m, s = self.stats[metric_name]
        return (value - m) / s

    def pct_change(self, metric_name, value):
        m, _ = self.stats[metric_name]
        return 100.0 * (value - m) / m


def dpa_flag(mean_hr, baseline_hr, abs_thresh=95.0, rel_thresh=0.10):
    """
    Gottman Diffuse Physiological Arousal / flooding detector.

    Flags when HR exceeds EITHER an absolute ceiling (~95-100 bpm) OR the
    partner's personal baseline by `rel_thresh` (10% default; some use
    15-20%). In a real stream, require this to be SUSTAINED for N seconds
    before raising it -- and pair it with the intervention (a ~20-minute
    break with active distraction, since norepinephrine clears slowly).
    """
    rel_hit = mean_hr >= baseline_hr * (1.0 + rel_thresh)
    abs_hit = mean_hr >= abs_thresh
    return {"flooded": bool(rel_hit or abs_hit),
            "by_relative": bool(rel_hit), "by_absolute": bool(abs_hit)}


def activation_index(metrics, baseline, weights=None):
    """
    Convergent per-partner arousal score in baseline z-units. Combines
    signals that should move TOGETHER under genuine activation, which is
    more robust than any single channel:

        + HR up,  - RMSSD down,  + resp_rate up,
        - RSA_corrected down,  - coherence down

    `metrics` keys expected: mean_hr, rmssd, resp_rate, rsa_corrected, coherence.
    Returns a single z-like scalar; higher = more activated. Tune weights
    and validate against self-report / observer coding.
    """
    w = weights or {"mean_hr": 1.0, "rmssd": 1.0, "resp_rate": 0.7,
                    "rsa_corrected": 1.0, "coherence": 0.7}
    sign = {"mean_hr": +1, "rmssd": -1, "resp_rate": +1,
            "rsa_corrected": -1, "coherence": -1}
    total, wsum = 0.0, 0.0
    for k, s in sign.items():
        if k in metrics and k in baseline.stats:
            total += w[k] * s * baseline.z(k, metrics[k])
            wsum += w[k]
    return total / wsum if wsum else np.nan


# =====================================================================
# 5. STREAMING SCAFFOLD -- tiered update cadence
# =====================================================================
class RealTimeProcessor:
    """
    Illustrative tiered-cadence processor for ONE partner. The window-size
    tradeoff is the central design decision:

        per beat / breath : mean HR, respiration rate      (fast display)
        ~30 s rolling      : RMSSD, SD1, time-domain        (RMSSD survives
                                                             short windows)
        ~60 s rolling      : LF/HF, RSA, coherence          (need the cycles)
        ~longer + lag      : dyadic synchrony (handled at the dyad level)

    Feed it cleaned RR intervals as they arrive; it keeps a rolling buffer
    and recomputes each tier at its own cadence. Respiration is passed
    alongside for the 60 s tier.
    """
    def __init__(self, fast_win_s=10, mid_win_s=30, slow_win_s=60):
        self.fast, self.mid, self.slow = fast_win_s, mid_win_s, slow_win_s
        self.rr_buffer = []          # (timestamp_s, rr_ms)

    def push(self, t_s, rr_interval_ms):
        self.rr_buffer.append((t_s, rr_interval_ms))
        # drop anything older than the slow window
        cutoff = t_s - self.slow
        self.rr_buffer = [(t, r) for (t, r) in self.rr_buffer if t >= cutoff]

    def _rr_within(self, now_s, win_s):
        return np.array([r for (t, r) in self.rr_buffer if t >= now_s - win_s])

    def compute(self, now_s, resp=None, fs_resp=None, baseline=None):
        out = {}
        mid_rr = self._rr_within(now_s, self.mid)
        slow_rr = self._rr_within(now_s, self.slow)

        if mid_rr.size > 5:
            mid_rr, _ = clean_rr(mid_rr)
            out.update(time_domain(mid_rr))
            out.update(poincare(mid_rr))
        if slow_rr.size > 20:
            slow_rr, _ = clean_rr(slow_rr)
            out.update(frequency_domain(slow_rr))
            out["coherence"] = mccraty_coherence(slow_rr)
            if resp is not None and fs_resp is not None:
                rm = respiration_metrics(resp, fs_resp)
                out["resp_rate"] = rm["resp_rate"]
                out["rsa"] = rsa_peak_to_trough(slow_rr, resp, fs_resp)
        if baseline is not None and "mean_hr" in out:
            out["dpa"] = dpa_flag(out["mean_hr"],
                                  baseline.stats.get("mean_hr", (out["mean_hr"],))[0])
        return out


# =====================================================================
# Quick self-test on synthetic data
# =====================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # synthetic RR around 800 ms (75 bpm) with a 0.1 Hz (resonance) oscillation
    t = np.arange(0, 120, 0.8)
    rr = 800 + 40 * np.sin(2 * np.pi * 0.1 * t) + rng.normal(0, 8, t.size)
    rr_clean, _ = clean_rr(rr)
    fs_resp = 25.0
    resp = np.sin(2 * np.pi * 0.1 * np.arange(0, 120, 1 / fs_resp))

    print("time-domain :", {k: round(v, 1) for k, v in time_domain(rr_clean).items()})
    print("poincare    :", {k: round(v, 2) for k, v in poincare(rr_clean).items()})
    print("frequency   :", {k: round(v, 3) for k, v in frequency_domain(rr_clean).items()})
    print("respiration :", {k: round(v, 2) for k, v in respiration_metrics(resp, fs_resp).items()
                            if isinstance(v, float)})
    print("RSA (P2T)   :", round(rsa_peak_to_trough(rr_clean, resp, fs_resp), 2), "bpm")
    print("coherence   :", round(mccraty_coherence(rr_clean), 2))
    print("resp-cardiac:", round(resp_cardiac_coherence(rr_clean, resp, fs_resp), 2))
