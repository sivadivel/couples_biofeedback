# Couples Biofeedback — Signal Processing Spec v2

**Purpose.** This document updates the real-time signal-processing layer to align it with what short-window HRV can actually support. It is written for a downstream coding agent: each section states the change, the algorithm, and default parameters. Parameters marked **[CAL]** are provisional and must be tuned against ground truth (see §10); do not treat them as validated constants.

**Design principle running through all of it.** This is a *talking* application, and speech is breathing. Almost every HRV-derived signal (RMSSD, HF power, coherence) is shaped by respiration as much as by autonomic state. The core strategy is therefore: (1) rely on the two signals that survive irregular breathing — HR relative to baseline and RMSSD trend — during conversation; (2) scope respiration-dependent metrics (HF/"vagal activity", coherence) to controlled-breathing contexts; and (3) gate every metric by whether its required window length and signal quality were actually met.

---

## 1. Metric windowing rules

Compute each metric only on a window long enough for it to be reliable, and tag every emitted value with the window used and whether it met minimum length. Sources: Task Force 1996; Shaffer & Ginsberg 2017; Munoz et al.; Castaldo et al.; Pecchia et al.

| Metric | Min reliable window | Update cadence | Valid during speech? |
|---|---|---|---|
| Mean HR | 5–10 s | 1 s | Yes |
| RMSSD | 30 s (floor 20 s of clean beats) | 5–10 s | Degraded — see §4 speech fraction |
| HF power ("vagal activity") | 60 s | 30 s | **No** — controlled breathing only |
| Coherence / 0.1 Hz resonance | 120 s stationary | 30 s | **No** — recovery screen only |
| Respiration (EDR) | 60 s | 30 s | **No** — unreliable during speech |

Any tile whose window minimum is not met renders in a "warming up" state, not a number. Any tile gated off by speech/motion renders as "paused" with the reason, not a stale value.

---

## 2. Baseline computation (robust)

Replace mean/SD baselines with robust statistics. A 2–5 min pre-session baseline almost always contains a few movement artifacts; mean/SD lets those inflate the scale and compress every downstream comparison (this is a likely cause of the "moderate at rest" problem).

For each partner and each metric, over the baseline window (sliding 60 s window, 30 s step, per current design):

```
center = median(samples)
scale  = max(1.4826 * MAD(samples), scale_floor[metric])
```

- `MAD(x) = median(|x - median(x)|)`; the `1.4826` makes it a consistent estimator of SD for normal data.
- `scale_floor[metric]` prevents divide-by-zero on unusually stable baselines. Defaults **[CAL]**: HR 1.5 bpm, RMSSD 3 ms.
- Store `center`, `scale`, and the raw sample count / clean-beat fraction so confidence (§5) can reflect baseline quality.
- Baselines remain resettable mid-session, as today.

---

## 3. Activation index (redesign)

The current 0–100 index blends four signals (HR, RMSSD, HF/"vagal", coherence). Three of those measure substantially the same vagal-respiratory variance, so the composite partly double-counts them while folding in the two most window-fragile and respiration-confounded ones. Replace with a two-signal robust index during conversation; add the respiration-dependent signals only in controlled-breathing mode.

### 3.1 Per-update z-scores

Compute against the robust baseline from §2:

```
z_HR    = (HR_now    - HR_center)    / HR_scale       # positive = elevated
z_RMSSD = (RMSSD_now - RMSSD_center) / RMSSD_scale    # negative = suppressed
```

Activation pushes HR up and RMSSD down, so the RMSSD contribution is negated:

```
c_HR    = clamp(z_HR,     -3, +3)
c_RMSSD = clamp(-z_RMSSD, -3, +3)
d = w_HR * c_HR + w_RMSSD * c_RMSSD          # weights sum to 1
```

Defaults **[CAL]**: `w_HR = 0.5`, `w_RMSSD = 0.5`. Clamping at ±3 caps single-signal blowups from residual artifact.

### 3.2 Map to 0–100

`d = 0` means "at baseline" (calm); `d > 0` means activated relative to self. Anchor baseline low, not at mid-scale:

```
activation = clamp(A0 + G * d, 0, 100)
```

Defaults **[CAL]**: `A0 = 12` (resting anchor), `G = 26` (gain). With these, `d≈0 → ~12` (calm), `d≈1 → ~38`, `d≈2 → ~64`, `d≈3 → ~90`, aligning to the existing zone cutoffs (calm 0–35 / moderate 35–65 / high 65–100). Keep the cutoffs; tune `A0`/`G` to them.

### 3.3 Controlled-breathing mode (recovery screen)

Only here — where breathing is paced and the window can be held stationary for 2+ minutes — fold in the vagal/coherence signal, because that is where it is measuring the thing of interest (down-regulation):

```
if mode == RECOVERY and respiration_regular and window >= 120s:
    c_vagal = clamp(-z_coherence_or_HF, -3, +3)   # higher coherence => lower activation
    d = w_HR*c_HR + w_RMSSD*c_RMSSD + w_vagal*c_vagal   # renormalize weights to sum to 1
```

Default recovery weights **[CAL]**: `w_HR = 0.3`, `w_RMSSD = 0.3`, `w_vagal = 0.4`.

### 3.4 Trend / direction

Direction is often more informative than level (retain this from v1). Fit a robust slope (Theil–Sen preferred, ordinary least squares acceptable) of `activation` over the last 60–90 s:

```
rising  if slope >  s_thresh
falling if slope < -s_thresh
stable  otherwise
```

Default `s_thresh` **[CAL]**: 0.15 activation-units/second.

---

## 4. Speech and motion gating

### 4.1 Speech fraction

You already have speaker diarization from voice enrollment. For each metric window, compute `speech_fraction` = proportion of the window during which this partner was speaking (extend each speech segment by +2 s to cover the post-speech respiratory tail).

- **RMSSD:** if `speech_fraction > 0.5`, widen the window toward 45–60 s to average over the irregularity, and lower confidence (§5). Do not gate it off entirely — RMSSD trend still carries signal.
- **HF / coherence / EDR respiration:** if `speech_fraction > 0.2`, gate **off** (render "paused during speech"). These are not interpretable over irregular breathing.

### 4.2 Motion

If the strap exposes accelerometer data over BLE, compute a per-window `motion_index` (e.g., mean absolute jerk or windowed SD of acceleration magnitude). This is the single highest-value artifact defense: it distinguishes "RMSSD dropped because they leaned in and gestured" from "RMSSD dropped because vagal tone withdrew."

```
if motion_index > motion_high:   gate RMSSD, HF, coherence (render "motion")
elif motion_index > motion_low:  lower confidence, do not gate
```

Thresholds **[CAL]**. If no accelerometer is available, fall back to more aggressive RR-interval artifact detection: flag windows with any RR gap > 25% from the local median, or ectopic/missed-beat patterns, and treat those windows as motion-gated.

---

## 5. Confidence

Confidence is the minimum of three components; surface "(signals mixed)" when the agreement term dominates.

```
conf_quality   = f(SQI, clean_beat_fraction, window_met)   # 0..1
conf_agreement = agreement(c_HR, c_RMSSD)                   # 0..1
conf_speech    = 1 - clamp(speech_fraction_penalty, 0, 1)   # 0..1
confidence = min(conf_quality, conf_agreement, conf_speech)
```

**Agreement** captures the v1 "signals mixed" idea explicitly: if HR says activated but RMSSD says calm (opposite signs, both |z| > 0.5), the signals disagree.

```
if sign(c_HR) != sign(c_RMSSD) and min(|z_HR|, |z_RMSSD|) > 0.5:
    conf_agreement = low        # -> display "(signals mixed)"
else:
    conf_agreement = high
```

Confidence should visibly resolve within 1–2 update cycles, as documented.

---

## 6. Flooding detection (two-signal confirmation)

The current rule (HR ≥ 10% above baseline **or** HR ≥ 95 bpm) is HR-only, and 10% above a low resting rate is a level reachable just by speaking with emphasis. Require the vagal-withdrawal signature as confirmation for the primary criterion; keep the absolute ceiling as an independent escape hatch.

**Primary criterion — both must hold, sustained ≥ 10 s:**
```
z_HR    >= T_HR_z          # HR elevated vs personal baseline
-z_RMSSD >= T_RMSSD_z      # RMSSD suppressed vs personal baseline
```
Talking raises HR but does not collapse RMSSD the way sympathetic flooding does, so requiring both cuts speech-driven false positives.

**Absolute ceiling — independent, sustained ≥ 10 s:**
```
HR_now >= HR_ceiling       # floor for already-elevated baselines
```

**Defaults [CAL]:** `T_HR_z` ≈ the robust-z equivalent of ~10–12% above baseline (compute per-partner from their scale; also expose a raw-percent fallback), `T_RMSSD_z = 1.0` (RMSSD ≥ 1 MAD below baseline), `HR_ceiling = 95 bpm`.

**Hysteresis.** Clear only after the condition is false for ≥ 5 continuous seconds, to prevent flicker at the boundary. Log every flood start/end with HR, % above baseline, and z_RMSSD at the transition.

All thresholds must be per-session configurable; the calibration task in §10.2 is the intended way to set them.

---

## 7. Dyadic coupling (surrogate-tested)

Implement the null model the v1 white paper twice admits is missing, so the displayed correlation can be shown as above/below chance rather than as a bare r.

### 7.1 Preprocessing (reduces spurious coupling from shared drift)

```
resample each partner's series (HR or activation) to a uniform 1 Hz grid
within a sliding window W (default 150 s):
    detrend: subtract moving median (or difference the series)
```

### 7.2 Cross-correlation

```
for lag in [-L .. +L]:            # L default 10 s
    r[lag] = pearson(A_window, shift(B_window, lag))
peak_r, peak_lag = argmax_abs(r)
```

### 7.3 Surrogate null

```
for i in 1..N:                    # N default 200
    offset = random circular shift (or block-shuffle) of B_window
    null_i = max_abs_lag_correlation(A_window, offset)
p_value    = fraction of null_i >= |peak_r|
percentile = 100 * (1 - p_value)
```

Display `peak_r` together with "above chance" (p < 0.05) / "at chance". Only surface strong relational-dynamics claims (co-regulation vs conflict-amplification) when coupling is above chance **and** gate the *interpretation* by arousal state, exactly as the v1 table does:

| Coupling | Both calm | Both activated |
|---|---|---|
| Above chance | co-regulation (positive) | conflict amplification |
| At chance | parallel, not linked | parallel, not linked |

Keep the "exploratory layer" label.

---

## 8. Respiration handling

- EDR (respiration inferred from the tachogram) is gated off during speech and below ~9 br/min (out of detectable band), per §1/§4. Show it only in recovery mode as breathing-pacing feedback.
- **Recommended hardware upgrade:** a direct respiration channel (respiration belt or a strap reporting it). This lets you respiration-correct HF instead of discarding it, and makes recovery-screen feedback trustworthy. If added, HF can be reported during controlled breathing with a respiration-rate annotation.

---

## 9. Recovery screen

- Coherence / 0.1 Hz resonance is scoped **here only** (controlled breathing, ≥120 s window). Relabel it as "resonance / 0.1 Hz rhythm" rather than a general early-warning emotional signal — it is a signature of paced breathing at the cardiovascular resonance frequency, not a validated heart-brain state detector. The breathing intervention itself (≈5.5–6 br/min) is well supported (Lehrer/Vaschillo/Gevirtz) and unchanged.
- Calm-zone timer logic unchanged: start when both partners' activation < 35 sustained; the timer, not the wall clock, is the readiness signal.

---

## 10. Validation instrumentation

You are positioned to validate rather than just caveat. Add the hooks now so data accrues from day one.

### 10.1 Ground-truth self-report
Add an optional 2-second arousal/valence capture (SAM or Affect-Grid tap) that can be triggered periodically or on demand. Log it with a timestamp alongside the full metric snapshot. This enables within-person checks that the activation index tracks self-reported arousal, and lets you tune `A0`, `G`, and the flooding thresholds to a target sensitivity/specificity.

### 10.2 Per-user stressor calibration
Add an optional pre-session calibration block: a brief standardized mild stressor (e.g., serial subtraction or a recall task, ~2 min) bracketed by rest. Record how far *this user's* HR and RMSSD actually move under mild stress, and set `T_HR_z` / `T_RMSSD_z` from their observed reactivity instead of assuming a fixed 10%.

### 10.3 Logging additions
Extend the JSON log so every metric snapshot and transcript event also records: per-metric SQI, `window_met` flag, `speech_fraction`, `motion_index` (if available), which metrics were gated and why, dyadic `peak_r` + `peak_lag` + surrogate `p_value`, and any self-report entries. Keep the embedded metric guide in sync.

---

## 11. Consolidated parameters

All **[CAL]** — defaults are starting points, to be tuned per §10.

| Param | Default | Meaning |
|---|---|---|
| `w_HR`, `w_RMSSD` (talk) | 0.5, 0.5 | Activation weights, conversation mode |
| `w_HR`, `w_RMSSD`, `w_vagal` (recovery) | 0.3, 0.3, 0.4 | Activation weights, recovery mode |
| `A0`, `G` | 12, 26 | Activation resting anchor and gain |
| `s_thresh` | 0.15 /s | Rising/falling slope threshold |
| `scale_floor` HR / RMSSD | 1.5 bpm / 3 ms | Robust-scale floors |
| RMSSD window (base / speech) | 30 s / 45–60 s | Widen under high speech fraction |
| HF window | 60 s | Vagal tile minimum |
| Coherence window | 120 s | Recovery only |
| `speech_fraction` gate (HF/coh) | 0.2 | Above this, gate off |
| `speech_fraction` gate (RMSSD) | 0.5 | Above this, widen + lower confidence |
| `motion_low`, `motion_high` | — | Accelerometer gating (device-specific) |
| `T_HR_z`, `T_RMSSD_z` | ~10–12% equiv, 1.0 | Flooding thresholds (z units) |
| `HR_ceiling` | 95 bpm | Absolute flooding floor |
| Flood sustain / clear hysteresis | 10 s / 5 s | Debounce |
| Dyadic `W`, `L`, `N` | 150 s, ±10 s, 200 | Window, max lag, surrogate count |

---

## 12. Change summary vs v1

1. Metric-specific window gating; no tile reads before its window is met (§1).
2. Robust (median/MAD) baselines replace mean/SD (§2).
3. Two-signal activation index (HR + RMSSD), respiration-dependent signals scoped to recovery mode (§3).
4. Explicit speech-fraction and motion gating (§4).
5. Confidence formalized, "signals mixed" defined as sign-disagreement (§5).
6. Flooding requires HR elevation **and** RMSSD suppression, plus hysteresis; absolute ceiling retained (§6).
7. Dyadic coupling gets detrending + surrogate null testing with an above/below-chance display (§7).
8. Coherence relabeled and confined to controlled breathing (§9).
9. Validation hooks: self-report ground truth, per-user stressor calibration, expanded logging (§10).
