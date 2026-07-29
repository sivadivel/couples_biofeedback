"""Simulated heart rate + R-R interval streams for developing without hardware."""

import asyncio
import math
import random
import time

# "activation" scenario: a sustained BPM offset + RMSSD suppression layered on
# top of the natural resting variation, so the activation index reads clearly
# low/high (or cycles between them for "mixed") independent of coupling.
_ACTIVATION_BPM_OFFSET  = 28.0
_ACTIVATION_RMSSD_SCALE = 0.45   # RMSSD multiplier while activated
_MIXED_PERIOD_S         = 150.0  # low->high->low cycle length for "mixed" (4 cycles / 10 min)

# "set baseline" fits against whatever's already buffered (up to the last 5 min,
# minimum ~60 s) — if "high"/"mixed" activation were elevated from t=0, clicking
# it would just re-center the baseline on the elevated state and activation
# would read as "normal" all session. A calm lead-in guarantees a resting
# reference is always available in the first _LEAD_IN_S seconds of ANY
# session, no matter which scenario is selected or how soon it's switched.
_LEAD_IN_S = 90.0
_RAMP_S    = 15.0   # smoothstep ramp duration when a level takes effect

# "coupling" scenario: the dyadic coupling metric is computed on RR-derived
# instantaneous HR after a 10 s moving-median detrend, so only faster (~3-5 s
# period) shared fluctuations survive into the cross-correlation — i.e. the
# respiratory/RSA modulation, not the slow BPM drift. Locking both partners to
# the same respiratory frequency (plus a fixed lag) makes them strongly
# coupled; independent frequencies (the default) wash the correlation out.
_COUPLED_RESP_FREQ_RANGE = (0.20, 0.28)
_LEAD_LAG_S              = 1.5   # partner "a" leads partner "b" by this many seconds

# a PURE single-tone RSA is a problem for the surrogate test specifically: its
# period (~3.6-5s) is close to the +/-4s lag search window, so a randomly
# circular-shifted copy can often "realign" at some other lag almost as well
# as the true one, making genuinely-coupled signals read as chance more often
# than they should. Adding a second, non-harmonic-ratio frequency breaks that
# near-periodicity (the composite only realigns with itself near true zero
# shift) while still being fully shared/lagged the same way as the base tone.
_RSA_F2_RATIO  = 1.65
_RSA_F2_WEIGHT = 0.35

SCENARIOS = {
    "low_low":    {"activation": "low",   "coupling": "low"},
    "low_high":   {"activation": "low",   "coupling": "high"},
    "high_low":   {"activation": "high",  "coupling": "low"},
    "high_high":  {"activation": "high",  "coupling": "high"},
    "mixed_low":  {"activation": "mixed", "coupling": "low"},
    "mixed_high": {"activation": "mixed", "coupling": "high"},
}


class ScenarioState:
    """Mutable, shared-by-reference between both partners' simulate_stream
    tasks, so switching the scenario mid-session (the live selector) takes
    effect in both streams immediately without restarting them."""

    def __init__(self, scenario: str = "low_low"):
        self.scenario   = scenario if scenario in SCENARIOS else "low_low"
        self.changed_at = 0.0   # elapsed session-seconds when last changed

    def set(self, scenario: str, elapsed_now: float) -> bool:
        if scenario not in SCENARIOS or scenario == self.scenario:
            return False
        self.scenario   = scenario
        self.changed_at = elapsed_now
        return True


def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def _activation_offset(level: str, t: float, level_changed_at: float) -> tuple[float, float]:
    """Return (bpm_offset, rmssd_scale) for the given activation level at time t.

    Ramps start at `level_changed_at` (when this level took effect) but never
    before `_LEAD_IN_S`, so the first _LEAD_IN_S seconds of any session stay
    calm regardless of the initial scenario or how quickly it's switched.
    """
    if level == "low":
        return 0.0, 1.0

    anchor = max(level_changed_at, _LEAD_IN_S)
    if t < anchor:
        return 0.0, 1.0
    t_eff = t - anchor

    if level == "high":
        frac = _smoothstep(t_eff / _RAMP_S)
        return _ACTIVATION_BPM_OFFSET * frac, 1.0 - (1.0 - _ACTIVATION_RMSSD_SCALE) * frac
    if level == "mixed":
        phase = (t_eff % _MIXED_PERIOD_S) / _MIXED_PERIOD_S
        frac = _smoothstep(phase / 0.5) if phase < 0.5 else 1.0 - _smoothstep((phase - 0.5) / 0.5)
        return _ACTIVATION_BPM_OFFSET * frac, 1.0 - (1.0 - _ACTIVATION_RMSSD_SCALE) * frac
    return 0.0, 1.0


async def simulate_stream(name: str, base_bpm: int, on_bpm, on_rr=None,
                          on_connect=None, scenario: str | ScenarioState = "low_low",
                          role: str = "a", session_start: float | None = None,
                          shared_resp_freq: float | None = None, **_):
    """
    Emit realistic sinusoidal HR and R-R intervals at ~1 Hz.

    R-R intervals include:
      - RSA modulation at a respiratory frequency (0.20-0.28 Hz = 12-17 br/min)
        so the FFT-based respiratory estimator has a real signal to find.
      - HRV noise scaled to RMSSD ~35 ms at rest, falling as BPM rises.

    `scenario` is either a plain SCENARIOS key (fixed for the whole call) or a
    `ScenarioState` instance shared with the other partner's call — pass the
    *same* ScenarioState object to both to allow live scenario switching.
    `role` is "a" or "b" — under a "high coupling" scenario, "a" is the leader
    and "b" follows `_LEAD_LAG_S` seconds behind, matching the dyadic
    processor's "positive lag = A leads B" convention. `session_start` and
    `shared_resp_freq` must be the same values passed to both partners' calls
    so their signals actually line up.
    """
    if on_connect:
        on_connect(f"{name} (simulated)")

    state = scenario if isinstance(scenario, ScenarioState) else ScenarioState(scenario)

    # both frequencies are generated once up front and held for the whole
    # session, so switching the coupling level live just changes which one
    # the RSA term reads — no discontinuity beyond an ordinary phase change.
    own_resp_freq = random.uniform(*_COUPLED_RESP_FREQ_RANGE)
    if shared_resp_freq is None:
        shared_resp_freq = random.uniform(*_COUPLED_RESP_FREQ_RANGE)

    # Amplitude tuned so "high coupling" reliably clears the surrogate test:
    # the per-beat noise (from rmssd_target, ~30-55ms SD) otherwise swamps a
    # smaller RSA signal, since correlation strength depends on the RSA
    # variance relative to that noise floor, not on RSA amplitude alone.
    RSA_AMP = 65.0
    t0 = session_start if session_start is not None else time.monotonic()

    # Emit one beat at a time, sleeping for that beat's own (synthetic) RR
    # duration rather than batching N beats into a fixed 1s tick. The dyadic
    # coupling analysis reconstructs each partner's time axis by summing RR
    # intervals (rr_time_axis), not from wall-clock timestamps — so if the
    # emitted RR values don't sum to real elapsed time, that reconstructed
    # axis drifts away from real time. At bpm != 60 that drift is large
    # (~17 s over 150 s at 68 bpm) and, since it drifts at a different rate
    # per partner, it smears out any injected cross-partner correlation.
    # Sleeping the true RR duration keeps cumulative RR time (and therefore
    # both partners' reconstructed axes) locked to real elapsed time.
    while True:
        t_sec = time.monotonic() - t0
        cfg = SCENARIOS.get(state.scenario, SCENARIOS["low_low"])
        act_offset, rmssd_scale = _activation_offset(cfg["activation"], t_sec, state.changed_at)

        if cfg["coupling"] == "high":
            resp_freq  = shared_resp_freq
            resp_lag_s = _LEAD_LAG_S if role == "b" else 0.0
        else:
            resp_freq  = own_resp_freq
            resp_lag_s = 0.0

        variation = 8 * math.sin(t_sec / 25.0) + 3 * math.sin(t_sec / 7.0)
        bpm = max(45.0, min(195.0, base_bpm + act_offset + variation + random.gauss(0, 1.5)))
        on_bpm(round(bpm))

        mean_rr = 60_000.0 / bpm
        rr = mean_rr
        if on_rr:
            rmssd_target = max(8.0, (55.0 - (bpm - base_bpm) * 1.2) * rmssd_scale)
            sigma   = rmssd_target / 2 ** 0.5
            phase_t = t_sec - resp_lag_s
            rsa     = RSA_AMP * (
                (1.0 - _RSA_F2_WEIGHT) * math.sin(2 * math.pi * resp_freq * phase_t) +
                _RSA_F2_WEIGHT * math.sin(2 * math.pi * resp_freq * _RSA_F2_RATIO * phase_t)
            )
            rr = max(300.0, min(2000.0, mean_rr + rsa + random.gauss(0, sigma)))
            on_rr(rr)

        await asyncio.sleep(rr / 1000.0)
