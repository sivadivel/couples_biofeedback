"""Simulated heart rate + R-R interval streams for developing without hardware."""

import asyncio
import math
import random


async def simulate_stream(name: str, base_bpm: int, on_bpm, on_rr=None,
                          on_connect=None, **_):
    """
    Emit realistic sinusoidal HR and R-R intervals at ~1 Hz.

    R-R intervals include:
      - RSA modulation at a random respiratory frequency (0.20–0.28 Hz = 12–17 br/min)
        so the FFT-based respiratory estimator has a real signal to find.
      - HRV noise scaled to RMSSD ~35 ms at rest, falling as BPM rises.
    """
    if on_connect:
        on_connect(f"{name} (simulated)")

    resp_freq = random.uniform(0.20, 0.28)   # unique breathing rate per athlete
    RSA_AMP   = 15.0                          # ms amplitude of respiratory modulation

    t     = 0      # integer second counter for BPM variation
    t_sec = 0.0    # continuous seconds for RSA phase

    while True:
        variation = 8 * math.sin(t / 25.0) + 3 * math.sin(t / 7.0)
        bpm = max(45, min(195, round(base_bpm + variation + random.gauss(0, 1.5))))
        on_bpm(bpm)

        if on_rr:
            mean_rr = 60_000.0 / bpm
            rmssd_target = max(8.0, 55.0 - (bpm - base_bpm) * 1.2)
            sigma = rmssd_target / 2 ** 0.5
            n_beats = max(1, round(bpm / 60))
            beat_dur_sec = 1.0 / n_beats      # seconds between synthetic beats

            for j in range(n_beats):
                beat_t = t_sec + j * beat_dur_sec
                rsa    = RSA_AMP * math.sin(2 * math.pi * resp_freq * beat_t)
                rr     = mean_rr + rsa + random.gauss(0, sigma)
                on_rr(max(300.0, min(2000.0, rr)))

        await asyncio.sleep(1.0)
        t     += 1
        t_sec += 1.0
