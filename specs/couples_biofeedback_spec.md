# Couples Therapy Biofeedback System — Build Spec

**Audience:** Claude Code (implementation agent) + developer.
**Status:** design handoff. Metrics and UI are specified; thresholds and a few
product decisions are flagged as open (Section 9).

---

## 1. What this is

A real-time biofeedback tool for couples therapy. It reads each partner's
heartbeat signal during a session and surfaces stress / emotional activation,
plus the coupling between partners, to support the therapist and the couple. It
has **two modes**:

- **Mode A — Live monitor.** A dyadic dashboard the therapist watches during conversation. Per-partner arousal, a flooding alert, and a dyadic-coupling panel.
- **Mode B — Recovery / breathe-together.** A calm screen the couple uses during a break to down-regulate via paced breathing, with coherence as the feedback signal.

It is grounded in three literatures: Gottman & Levenson's *Diffuse Physiological
Arousal* (flooding) and physiological linkage; HRV biofeedback / resonance-
frequency breathing (~0.1 Hz) and cardiac coherence; and interpersonal
physiological synchrony.

**This is not a medical device.** All outputs are physiological *proxies*, not
direct readings of emotion. See Section 10.

---

## 2. Current direction (read this before building)

- **HR / HRV only. No respiration belt.** Respiration is inferred from the heartbeat series (breathing shows up as the high-frequency oscillation in the RR intervals). This removes hardware and a calibration step; the one tradeoff is documented in Section 4.3.
- Implement the **six-metric set** in Section 4.2 — not a larger battery.
- Two screens only (Section 6 and Section 7). Earlier drafts had three break-mode screens; the break mode is now **one** shared screen.

---

## 3. Data inputs

### 3.1 Required signal

Per partner: a stream of **beat-to-beat RR intervals (interbeat intervals) in
milliseconds** — the time between successive heartbeats, from ECG R-peaks or PPG
systolic peaks.

> **Critical:** this is NOT an averaged heart-rate-in-bpm number. A smoothed bpm
> value has already discarded the beat-to-beat variation that every HRV metric
> below depends on. Before building, confirm the sensor/SDK exposes RR / IBI (or
> raw R-peak timestamps). If it only gives smoothed bpm, most of this cannot be
> computed.

### 3.2 Shape

- Two independent partner streams (A and B), each an append-only series of `(timestamp_s, rr_ms)`.
- Arrival is irregular (one sample per heartbeat, ~0.5–1.5 Hz).
- The system keeps rolling buffers and recomputes metrics on a schedule (Section 4.6).

---

## 4. Signal processing and metrics

### 4.1 Preprocessing — artifact rejection

Run on the RR series before any metric. A moving, talking, aroused couple
produces far more artifact than a resting lab subject; everything downstream
inherits these errors.

- **Range filter:** keep intervals in 300–2000 ms (30–200 bpm).
- **Ectopic / misdetection filter:** drop any interval differing from the running median of its neighbours by more than ~20% (tighten for resting baseline, loosen for active conversation).

### 4.2 The six metrics

Notation: `RR_i` = cleaned intervals (ms); `d_i = RR_{i+1} − RR_i`;
`S(f)` = power spectrum of the RR series.

**1. Mean HR + flooding flag (DPA)** — the load-bearing, most actionable signal.

```
mean_HR = 60000 / mean(RR)            # bpm

flooded = (mean_HR >= 1.10 * HR_baseline) OR (mean_HR >= 95)
          AND sustained for N seconds   # N ~ 5–10 s, tunable
```

Threshold is individual; prefer the relative (% over personal baseline) rule,
with the absolute ceiling as a backstop. When flagged, the UI should offer the
break (Mode B); the recommended break is ~20 minutes with active distraction,
because the stress hormones driving the state clear slowly.

**2. RMSSD** — short-window workhorse for vagal withdrawal; falls under stress.

```
RMSSD = sqrt( mean( d_i^2 ) )          # ms
```

Valid on short (10–30 s) windows, unlike SDNN / spectral measures.

**3. HF power — RSA / vagal-tone proxy** — emotion-regulation index; falls with
activation. Replaces belt-based respiratory sinus arrhythmia.

```
HF_power = integral of S(f) over 0.15–0.40 Hz
```

Use the **Lomb–Scargle periodogram** on the raw (unevenly sampled) RR series for
short windows; switch to Welch on a 4 Hz cubic-spline-resampled tachogram only
with >= 60 s of clean data.

**4. Derived respiration rate** — free from the tachogram (breathing *is* the HF
oscillation).

```
resp_rate = 60 * argmax_f S(f) over 0.15–0.40 Hz   # breaths/min
```

A useful arousal proxy (resting adults ~9–20/min; rises with activation) and a
caveat flag (see 4.3).

**5. Coherence — the biofeedback target** (McCraty method). The regulated state
to move *toward*; roughly the inverse of flooding. Drives Mode B.

```
1. find the highest peak f0 in 0.04–0.26 Hz of S(f)
2. peak_power  = power in a 0.030 Hz window centered on f0
3. coherence   = peak_power / (total_power − peak_power)
```

Implementation note: at short-window frequency resolution this peak window may
span only 1–2 FFT bins, so integrate by **bin-sum**, not trapezoidally (trapz
over a single bin collapses to 0).

**6. Dyadic coupling — lagged cross-correlation.** The relational layer. Compute
on both partners' instantaneous HR (or their RMSSD / HF time series) on a common,
evenly sampled grid.

```
peak_r = max over |tau| <= tau_max of | corr( A_t , B_{t+tau} ) |
# detrend each window first; tau in seconds; report the lag at the max
```

Cautions (encode in UI and interpretation):
- **Detrend each window** — shared slow drift fakes synchrony.
- **Sign matters** — positive = in-phase, negative = anti-phase; different meanings.
- **Surrogate baseline** — compare against shuffled / non-partner pairings before calling coupling meaningful.
- **Read jointly with arousal** — high synchrony while both are flooded = "locked in conflict"; high synchrony while both are calm = co-regulation. Same number, opposite meaning. Treat per-partner arousal/coherence as the reliable core and synchrony as an **exploratory** overlay.

### 4.3 The one tradeoff (HF vs coherence during slow breathing)

HF is a slightly noisier RSA stand-in than a belt because it cannot be corrected
for breathing rate / tidal volume. Two consequences to build in:

- **Pair HF with the derived respiration rate** so a drop in HF can be checked against a change in breathing rather than vagal tone.
- **During slow-breathing exercises (Mode B), use coherence, not HF.** Slow breathing (below ~9/min = 0.15 Hz) pushes the respiratory peak *out of the HF band into the LF band*, so HF under-reads exactly when the person is calming down. Rule of thumb: **HF for spontaneous activation (Mode A), coherence for the regulation exercise (Mode B).**

### 4.4 Personalized baselining

Every threshold is individual; population constants do not transfer.

- At session start, record a few minutes of calm baseline per partner.
- Store mean and SD of each metric.
- Express live values as z-scores or % change: `z = (x − mu_baseline) / sigma_baseline`.

### 4.5 Composite activation index

A single 0–100 score for **autonomic activation / arousal** per partner.
Convergence across channels is far more trustworthy than any single spike, so the
index combines the signals that move together under genuine activation. Note the
construct is *arousal/mobilization*, NOT a specific emotion — see 4.7.

Four design rules that separate this from a naive weighted sum:

1. **Z-score against the person's own baseline first.** Absolute values are meaningless across people; the signal is each channel's departure from that partner's calm baseline.
2. **Don't double-count the vagal channels.** RMSSD and HF are the time- and frequency-domain views of the *same* vagal withdrawal — they are largely redundant. Collapse them into one vagal sub-score so "parasympathetic pullback" isn't weighted twice.
3. **Gate HF out during slow breathing.** Below ~9/min, HF collapses (reads as activation) while coherence rises (reads as calm) and they fight each other. When `resp_rate < 9`, drop HF and coherence from the sum (same confound documented in 4.3).
4. **Carry a confidence term.** When channels disagree in sign (HR up but vagal tone also up), the number is less trustworthy. Surface that so the text layer (4.7) can hedge instead of asserting.

Keep the **flooding flag (4.2, metric 1) as a separate discrete overlay** on top
of the continuous score — it is a clinically meaningful threshold that triggers
the break, not merely "high activation."

```python
def activation_state(metrics, baseline, prev=None, alpha=0.3):
    """0-100 activation, direction, confidence in [0,1].
       metrics: dict w/ mean_hr, rmssd, hf, resp_rate, coherence
       baseline: fitted Baseline; prev: previous smoothed value."""
    z = {k: baseline.z(k, metrics[k]) for k in metrics if k in baseline.stats}

    slow_breathing = metrics.get("resp_rate", 99) < 9     # paced-breathing guard
    vagal = -z["rmssd"] if slow_breathing else -(z["rmssd"] + z["hf"]) / 2

    contribs = {                                          # higher = more activated
        "hr":    (+z["mean_hr"], 1.0),
        "vagal": (vagal,         1.0),
        "resp":  (+z.get("resp_rate", 0), 0.5),
        "coher": (-z.get("coherence", 0), 0.0 if slow_breathing else 0.5),
    }
    raw = sum(v * w for v, w in contribs.values()) / sum(w for _, w in contribs.values())
    activation = 100 / (1 + np.exp(-raw / 1.5))           # logistic squash to 0-100

    if prev is not None:                                  # temporal smoothing
        activation = alpha * activation + (1 - alpha) * prev

    signs = [np.sign(v) for v, w in contribs.values() if w > 0]
    agree = abs(sum(signs)) / len(signs)                  # 0 = split, 1 = unanimous

    return {"activation": activation,
            "direction": "rising"  if prev and activation > prev + 2 else
                         "falling" if prev and activation < prev - 2 else "stable",
            "confidence": agree,
            "flooded": metrics["mean_hr"] >= 1.10 * baseline.stats["mean_hr"][0]}
```

**Calibration upgrade path:** the weights above are an educated starting point.
If self-reported activation (or a therapist-coded "flooded / not" label) is ever
collected alongside the signals, fit the weights with a regression / logistic
regression instead of hand-tuning. That turns the index from a guess into
something calibrated to the actual population. Until then, treat weights as
tunable and validate against self-report / observer coding.

### 4.6 Real-time cadence

Different metrics need different amounts of data — run them at different rates
from a rolling buffer:

| Tier   | Window        | Metrics                                   | Why                                |
|--------|---------------|-------------------------------------------|------------------------------------|
| Fast   | per beat      | mean HR                                    | immediate display, minimal data    |
| Mid    | ~30 s rolling | RMSSD, flooding flag                        | RMSSD valid at short lengths       |
| Slow   | ~60 s rolling | HF, respiration rate, coherence            | need several oscillation cycles    |
| Dyadic | longer + lag  | synchrony                                   | computed at the dyad level         |

### 4.7 Per-partner state description (text layer)

A short natural-language description of each partner's state, generated from the
structured output of 4.5. **This is the highest-risk feature in the system and
must be built to the constraint below.**

**The hard constraint: describe the body, never name the emotion.** Autonomic
signals measure *arousal*, not *valence* and not *thoughts*. A racing heart with
vagal withdrawal looks nearly identical whether the person is angry, afraid,
excited, or attracted. The physiology supports statements about **how activated,
how regulated, and the trend** — it cannot identify *which* emotion, and any text
that prints "Partner B is angry" is fabricating the most important word. In a
couples-therapy room a screen that asserts an emotion can be used against a
partner or can override their own account of how they feel — the opposite of the
tool's purpose.

**Therefore:**
- Generate from a **rule-based template** off the structured state, not free-form LLM prose. The template physically cannot drift into mind-reading.
- Every sentence must be a claim about the body that the person could verify or dispute (heart rate, breathing, rhythm organization, trend).
- **No** emotion words, **no** causal claims ("because…"), **no** inferred thoughts ("seems to feel…").
- Hedge when `confidence` is low.

```python
def describe_state(name, s, metrics, baseline):
    level = ("low" if s["activation"] < 35 else
             "moderate" if s["activation"] < 65 else "high")
    breath = (f"slow ({metrics['resp_rate']:.0f}/min)" if metrics["resp_rate"] < 10
              else f"rapid ({metrics['resp_rate']:.0f}/min)" if metrics["resp_rate"] > 20
              else "steady")
    hedge = "" if s["confidence"] > 0.7 else " (signals mixed — read with care)"

    desc = f"{name}: {level} physiological activation, {s['direction']}."
    if s["flooded"]:
        desc += " Heart rate is past the flooding threshold; a break is indicated."
    else:
        base_hr = baseline.stats["mean_hr"][0]
        desc += f" Heart rate {'above' if metrics['mean_hr'] > base_hr else 'near'} baseline, breathing {breath}."
    if not s["flooded"] and metrics.get("coherence", 0) > 0.6:
        desc += " Heart rhythm is organized — physiologically settled."
    return desc + hedge
```

Example outputs:
- "Partner B: high physiological activation, rising. Heart rate is past the flooding threshold; a break is indicated."
- "Partner A: low physiological activation, stable. Heart rate near baseline, breathing steady. Heart rhythm is organized — physiologically settled."
- "Partner B: moderate physiological activation, rising. Heart rate above baseline, breathing rapid (24/min) (signals mixed — read with care)."

**If richer prose than templates is wanted later:** an LLM may be layered on top
ONLY if constrained to narrate the structured fields passed to it (activation,
direction, confidence, breathing, flooding) and *explicitly forbidden* from naming
emotions, inferring causes, or describing thoughts. The structured fields are the
guardrail; the LLM rewrites, it does not infer. Validate any generated text
against self-report before it is shown in session.

---

## 5. Reference implementation already written

A Python reference module exists: **`dyadic_biofeedback_metrics.py`** (numpy +
scipy only). It was written for the fuller, belt-based design, so for the current
HR/HRV-only direction:

**Keep / reuse as-is:**
`clean_rr`, `resample_tachogram`, `time_domain` (mean HR, RMSSD, SDNN, pNN50),
`frequency_domain` (LF/HF via Lomb–Scargle or Welch), `mccraty_coherence`,
`windowed_lagged_xcorr`, `common_grid_hr`, `Baseline`, `dpa_flag`,
`RealTimeProcessor`.

**Add (new, specified in 4.5 and 4.7):**
`activation_state` (the 0–100 index with vagal-collapse, slow-breathing guard,
confidence, and smoothing) and `describe_state` (the constrained text layer).
These supersede the old `activation_index`.

**Drop (belt-dependent, not used in HR/HRV-only build):**
`respiration_metrics` (belt version), `rsa_peak_to_trough`, `fit_rsa_correction`,
`apply_rsa_correction`, `resp_cardiac_coherence`.

**Substitutions for the dropped functions:**
- RSA / vagal tone → use `frequency_domain(...)["hf"]` (HF power).
- Respiration rate → take the argmax frequency in the HF band of the RR spectrum (small helper; the spectral machinery already exists in `frequency_domain`).
- Resp–cardiac coherence → use `mccraty_coherence` (RR-only).

The old `activation_index` (which expected an `rsa_corrected` key) is replaced by
`activation_state` per 4.5 — build the new function rather than repointing the old one.

Companion docs in the same folder: `biofeedback_metrics_simplified.md` / `.pdf`
(the HR/HRV-only reference these metrics come from) and
`biofeedback_metrics_explained.md` / `.pdf` (the fuller version, for background).

---

## 6. UI — Mode A: Live monitor dashboard

**Purpose:** therapist-facing, dense, glanceable. Shows where each partner's
nervous system is, raises the flooding alert, and shows the coupling.

**Layout (top to bottom):**

1. **Session header** — title, running clock, and a "baseline set" indicator (the dashboard should not trust thresholds until the calm baseline is captured).
2. **Flooding alert banner** — appears session-wide when either partner is flagged (Section 4.2, metric 1). Names the partner and states the recommended action (suggest a ~20-minute break). Danger styling.
3. **Two partner panels, side by side.** Each contains:
   - Name + state badge: `regulated` (success styling) or `flooded` (danger styling).
   - **Activation score (4.5)** as a 0–100 readout or small gauge, with the **one-line state description (4.7)** beneath it — the human-readable headline for the panel. Show the "signals mixed" hedge when confidence is low.
   - **Mean HR** as the hero number, with "vs baseline" delta. Turns danger-colored when over threshold.
   - **HR trace** — rolling ~60 s sparkline.
   - Four secondary tiles: **RMSSD**, **HF power** (labeled "RSA proxy"), **respiration rate** (labeled "derived"), **coherence** (shown as a small bar — it's the *target*, not just a readout). Values that breach baseline render in the danger color.
4. **Dyadic coupling panel** (full width), labeled "exploratory layer":
   - Peak correlation `r`, lag + who-leads, and phase (in-phase / anti-phase).
   - Overlaid dual HR trace.
   - Standing caption: read synchrony against arousal, and against a surrogate baseline.
5. **Footer note:** proxies, not emotion readouts; thresholds personalized.

**States:** the two partner panels switch between regulated (calm/success accent)
and flooded (danger accent on the HR number, border, and breached tiles). The
alert banner is conditional on any partner being flooded.

**Refresh:** wire panel values to the cadence in Section 4.6 (HR fast, RMSSD mid,
HF/resp/coherence slow, synchrony dyadic).

---

## 7. UI — Mode B: Recovery / breathe-together screen

**Purpose:** couple-facing, calm, single-focus. One shared breathing exercise
both partners follow during a break, with coherence as the only feedback.

**Layout:**

1. **Header** — "Recovery · breathe together" + break clock.
2. **Shared breathing pacer (center, the main element):**
   - A static dashed **guide ring** with an animated **filled circle** inside it that expands and contracts.
   - Animation: smooth scale cycle, **~11 s per breath** (= 5.5 breaths/min, near the 0.1 Hz resonance frequency; equal-ish inhale/exhale). Implemented in earlier wireframe as a CSS `@keyframes` scaling 0.56 → 1.0 → 0.56 over 11 s, `ease-in-out`, infinite.
   - Cross-fading **"breathe in" / "breathe out"** label synced to the half-cycle (CSS keyframes, no JS needed).
   - Caption: the resonance rate (e.g. "5.5 breaths/min").
3. **Per-partner coherence** — two compact readouts (Partner A / Partner B), each a small bar + value. This is the biofeedback: bars rise as each settles. Keep understated.
4. **Footer:** gentle guidance ("follow the circle together; nothing to fix").

**Entry:** launched from the flooding alert in Mode A (or manually).
**Exit:** advisory by default — see open question in Section 9 about whether to
gate re-entry on both partners settling.

**Aesthetic:** calm and minimal, deliberately different from Mode A's density —
generous whitespace, a single calm accent color (a teal/green family worked well
in the wireframes), no dense data, no HR numbers competing for attention.

---

## 8. Design system notes

The wireframes were built to a specific design language; reproduce the *intent*
in whatever stack you choose:

- **Flat and clean** — no gradients, drop shadows, glow, or neon. Solid fills.
- **Two visual registers** — Mode A is dense and informational; Mode B is calm and spacious.
- **Color encodes state, not decoration:** a calm accent (teal/green) for regulated / coherence / the breathing pacer; danger (red) for flooding and breached metrics; success (green) for regulated badges and "settled" states. Keep to ~2 color families per screen.
- **Sentence case everywhere.** No ALL CAPS, no Title Case except proper nouns.
- **Dark-mode safe** — drive colors from theme tokens / CSS variables, not hardcoded hex, so both light and dark render correctly.
- **Accessibility** — every screen needs a one-line text summary for screen readers; the breathing animation should respect `prefers-reduced-motion` (offer a non-animated fallback).

---

## 9. Open decisions (need product/clinical input)

1. **Resonance rate per person.** 5.5/min is a good default, but individual resonance frequency ranges ~4.5–6.5/min. Decide: fixed default, a one-time per-person assessment sweep, or user-adjustable.
2. **Flooding thresholds.** The relative margin (10% vs 15–20% over baseline), the absolute ceiling (~95–100 bpm), and the sustain duration before flagging.
3. **Re-entry after a break: gated vs advisory.** Locking "resume" until both partners settle is safer (encodes the Gottman principle that re-engaging while flooded restarts the cycle) but can feel paternalistic. Earlier wireframe gated it; current simplified screen does not. Pick one.
4. **Privacy in the dual view.** Showing each partner the other's live coherence can create pressure ("why isn't yours rising?"). Consider showing only a shared rhythm and keeping each person's score private.
5. **Shared vs individual breathing pace** in Mode B. One shared pace is more visibly bonding; individual rates give cleaner per-person coherence.
6. **Synchrony metric details.** Window length, max lag, and which signal to couple on (HR vs RMSSD vs HF). Tune against real recordings.

---

## 10. Validation, safety, ethics

- **Proxies, not emotion.** Validate every metric and threshold against self-report and/or therapist coding before clinical reliance. Do not present any number as "how the person feels."
- **Clinician in the loop.** Designed to assist a therapist, not to replace clinical judgment or to be used unsupervised for diagnosis.
- **Not a medical device.** Avoid diagnostic or treatment claims.
- **Sensitive data.** Heart-rate and relationship-conflict data are sensitive. Specify storage, retention, consent, and access up front; prefer on-device / ephemeral processing where possible; both partners should consent to what each can see.
- **The dyadic synchrony literature is heterogeneous** — keep synchrony framed as exploratory, never as a verdict on the relationship.

---

## 11. Suggested architecture (non-binding)

- **Signal layer:** Python (numpy/scipy), reusing `dyadic_biofeedback_metrics.py` per Section 5. Wrap the `RealTimeProcessor` per partner; add a dyad-level processor for synchrony.
- **Transport:** push computed metrics to the UI over WebSocket (or local IPC) at the Section 4.6 cadences.
- **UI:** web front-end (the wireframes are HTML/CSS); two routes/views for Mode A and Mode B. Charts can be lightweight (sparklines) — no heavy dependencies needed.
- **Baseline + session state:** per-session store of each partner's baseline stats and the RR rolling buffers.
- Build order suggestion: (1) RR ingest + `clean_rr` + per-partner metrics, (2) baselining + flooding flag, (3) Mode A dashboard, (4) Mode B breathing screen, (5) dyadic synchrony overlay, (6) composite index + tuning.

---

## 12. Key sources

- Levenson & Gottman (1983, 1985) — physiological linkage; flooding / DPA framework.
- Shaffer & Ginsberg; ultra-short-term HRV reviews — HRV metric definitions, HF power as an RSA/vagal proxy, short-window validity.
- Lehrer, Vaschillo & Gevirtz; McCraty et al. (2009) — resonance-frequency breathing (~0.1 Hz) and cardiac coherence.
- Butler (2015); Palumbo et al. (2017); Timmons et al. (2015) — interpersonal physiological synchrony and its context-dependence.
