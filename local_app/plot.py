"""Real-time scrolling heart rate, HRV, and respiration display."""

import threading
from collections import deque
from datetime import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates

matplotlib.rcParams["toolbar"] = "None"

WINDOW_SECS        = 60
WINDOW_DAYS        = WINDOW_SECS / 86400.0
HRV_WINDOW_SECS    = 30
HRV_WINDOW_DAYS    = HRV_WINDOW_SECS / 86400.0
UPDATE_MS          = 500
RESP_COMPUTE_EVERY = 5

TRACK_COLORS = ["#c0392b", "#1a6baa"]

FIG_BG   = "#1a1a2e"
AX_BG    = "#ffffff"
SPINE_C  = "#cccccc"
TICK_C   = "#555555"
LABEL_C  = "#444444"
GRID_C   = "#eeeeee"
ZONE_LBL = "#aaaaaa"

HR_ZONES = [
    (0,   60,  "#cfe5f7"),
    (60,  100, "#cff2de"),
    (100, 140, "#f7f2cf"),
    (140, 170, "#f7decf"),
    (170, 220, "#f7cfcf"),
]
HR_ZONE_LABELS = {60: "Rest", 100: "Light", 140: "Cardio", 170: "Hard"}

HRV_ZONES = [
    (0,  20,  "#f7cfcf"),
    (20, 40,  "#f7e8cf"),
    (40, 70,  "#cff2de"),
    (70, 160, "#cfe5f7"),
]
HRV_ZONE_LABELS = {20: "Poor", 40: "Fair", 70: "Good"}


# ------------------------------------------------------------------ #
# RSA respiratory rate estimator                                      #
# ------------------------------------------------------------------ #

def _compute_resp_rate(rr_times: list, rr_vals: list, fs: float = 4.0):
    """Dominant frequency in the 0.15–0.40 Hz HRV band → breaths/min."""
    if len(rr_vals) < 20:
        return None
    t  = np.array(rr_times) * 86400.0
    rr = np.array(rr_vals, dtype=float)
    if t[-1] - t[0] < 15.0:
        return None
    t_grid  = np.arange(t[0], t[-1], 1.0 / fs)
    if len(t_grid) < 8:
        return None
    rr_grid = np.interp(t_grid, t, rr)
    tau = t_grid - t_grid[0]
    rr_grid -= np.polyval(np.polyfit(tau, rr_grid, 1), tau)
    rr_grid *= np.hanning(len(rr_grid))
    freqs = np.fft.rfftfreq(len(rr_grid), d=1.0 / fs)
    power = np.abs(np.fft.rfft(rr_grid)) ** 2
    mask  = (freqs >= 0.15) & (freqs <= 0.40)
    if not mask.any():
        return None
    return round(freqs[mask][np.argmax(power[mask])] * 60.0, 1)


# ------------------------------------------------------------------ #
# Figure helpers                                                       #
# ------------------------------------------------------------------ #

def _style_primary(ax):
    """Style the main (left) axis of a panel."""
    ax.set_facecolor(AX_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_C)
    ax.tick_params(colors=TICK_C, labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.grid(True, color=GRID_C, linewidth=0.6, zorder=0)


def _style_twin(ax, color):
    """Style a twinx right-side axis — no background, no gridlines."""
    ax.tick_params(colors=color, labelsize=8)
    ax.spines["right"].set_edgecolor(color)
    ax.spines["right"].set_linewidth(1.2)
    # Hide the left/top/bottom spines that would duplicate the primary axis
    for side in ("left", "top", "bottom"):
        ax.spines[side].set_visible(False)


def _add_bands(ax, zones, labels):
    for lo, hi, fc in zones:
        ax.axhspan(lo, hi, facecolor=fc, alpha=0.85, zorder=1)
    for y, lbl in labels.items():
        ax.axhline(y, color=SPINE_C, lw=0.8, zorder=2)
        ax.text(0.002, y, lbl, transform=ax.get_yaxis_transform(),
                fontsize=7, color=ZONE_LBL, va="bottom")


# ------------------------------------------------------------------ #
# Dashboard                                                            #
# ------------------------------------------------------------------ #

class HeartRateDashboard:
    def __init__(self, names: list[str]):
        self.names = names
        self.n     = len(names)
        self._lock = threading.Lock()

        self._bpm_times  = [deque() for _ in range(self.n)]
        self._bpms       = [deque() for _ in range(self.n)]
        self._latest_bpm = [None] * self.n

        self._rr_times = [deque() for _ in range(self.n)]
        self._rr_vals  = [deque() for _ in range(self.n)]

        self._hrv_times    = [deque() for _ in range(self.n)]
        self._hrv_vals     = [deque() for _ in range(self.n)]
        self._latest_rmssd = [None] * self.n

        self._resp_times  = [deque() for _ in range(self.n)]
        self._resp_vals   = [deque() for _ in range(self.n)]
        self._latest_resp = [None] * self.n

        self._connected     = [False] * self.n
        self._frame_counter = 0

        self._build_figure()

    # ------------------------------------------------------------------ #
    # Public thread-safe API                                               #
    # ------------------------------------------------------------------ #

    def add_bpm(self, idx: int, bpm: int):
        t = mdates.date2num(datetime.now())
        with self._lock:
            self._bpm_times[idx].append(t)
            self._bpms[idx].append(bpm)
            self._latest_bpm[idx] = bpm
            cutoff = t - WINDOW_DAYS
            while self._bpm_times[idx] and self._bpm_times[idx][0] < cutoff:
                self._bpm_times[idx].popleft()
                self._bpms[idx].popleft()

    def add_rr(self, idx: int, rr_ms: float):
        t = mdates.date2num(datetime.now())
        with self._lock:
            self._rr_times[idx].append(t)
            self._rr_vals[idx].append(rr_ms)
            cutoff = t - HRV_WINDOW_DAYS
            while self._rr_times[idx] and self._rr_times[idx][0] < cutoff:
                self._rr_times[idx].popleft()
                self._rr_vals[idx].popleft()
            if len(self._rr_vals[idx]) >= 2:
                diffs = np.diff(np.array(self._rr_vals[idx]))
                rmssd = float(np.sqrt(np.mean(diffs ** 2)))
                self._latest_rmssd[idx] = rmssd
                self._hrv_times[idx].append(t)
                self._hrv_vals[idx].append(rmssd)
                cutoff_d = t - WINDOW_DAYS
                while self._hrv_times[idx] and self._hrv_times[idx][0] < cutoff_d:
                    self._hrv_times[idx].popleft()
                    self._hrv_vals[idx].popleft()

    def mark_connected(self, idx: int, label: str):
        with self._lock:
            self._connected[idx] = True
        print(f"  Connected: {label}")

    def mark_disconnected(self, idx: int, reason: str = ""):
        with self._lock:
            self._connected[idx] = False
        print(f"  Disconnected: {self.names[idx]}" + (f" — {reason}" if reason else ""))

    def run(self):
        self._ani = animation.FuncAnimation(
            self._fig, self._update, interval=UPDATE_MS, blit=False
        )
        plt.show()

    # ------------------------------------------------------------------ #
    # Private                                                              #
    # ------------------------------------------------------------------ #

    def _build_figure(self):
        fig, axes = plt.subplots(
            2, self.n,
            figsize=(8 * self.n, 9),
            gridspec_kw={"height_ratios": [2, 1]},
            squeeze=False,
        )
        fig.patch.set_facecolor(FIG_BG)
        fig.suptitle("Heart Rate Monitor", color="#dddddd", fontsize=13, y=0.99)

        self._axes_bpm   = []   # primary axis (BPM, left y)
        self._axes_resp  = []   # twin axis (Resp, right y)
        self._axes_hrv   = []
        self._bpm_lines  = []
        self._resp_lines = []
        self._hrv_lines  = []
        self._bpm_texts    = []
        self._bpm_statuses = []
        self._resp_texts   = []
        self._hrv_texts    = []

        for i in range(self.n):
            color  = TRACK_COLORS[i % len(TRACK_COLORS)]
            ax_bpm = axes[0][i]
            ax_hrv = axes[1][i]

            # ── Combined BPM + Resp panel ─────────────────────────────
            _style_primary(ax_bpm)
            _add_bands(ax_bpm, HR_ZONES, HR_ZONE_LABELS)

            bpm_line, = ax_bpm.plot([], [], color=color, lw=2.2,
                                    zorder=5, solid_capstyle="round",
                                    label="HR (bpm)")
            self._bpm_lines.append(bpm_line)

            ax_bpm.set_ylim(40, 205)
            ax_bpm.set_ylabel("Heart Rate (BPM)", color=LABEL_C, fontsize=9)
            ax_bpm.set_title(self.names[i], color=color,
                             fontsize=13, fontweight="bold", pad=8)

            # Right y-axis for respiration
            ax_resp = ax_bpm.twinx()
            _style_twin(ax_resp, color)
            resp_line, = ax_resp.plot([], [], color=color, lw=2.0,
                                      zorder=4, solid_capstyle="round",
                                      linestyle="--", alpha=0.75,
                                      label="Resp (br/min)")
            self._resp_lines.append(resp_line)
            ax_resp.set_ylim(5, 40)
            ax_resp.set_ylabel("Respiration (br/min)", color=color, fontsize=9,
                               labelpad=6)

            # Legend inside the panel
            lines  = [bpm_line, resp_line]
            labels = ["HR (bpm)", "Resp (br/min)"]
            ax_bpm.legend(lines, labels, loc="upper left",
                          fontsize=8, framealpha=0.6,
                          facecolor="white", edgecolor=SPINE_C,
                          labelcolor=LABEL_C)

            # Readout texts (top-right of BPM axis)
            self._bpm_texts.append(ax_bpm.text(
                0.99, 0.97, "—",
                transform=ax_bpm.transAxes,
                fontsize=30, fontweight="bold", color=color,
                ha="right", va="top", zorder=6,
            ))
            self._bpm_statuses.append(ax_bpm.text(
                0.99, 0.74, "connecting…",
                transform=ax_bpm.transAxes,
                fontsize=9, color="#999999",
                ha="right", va="top", zorder=6,
            ))
            self._resp_texts.append(ax_bpm.text(
                0.99, 0.65, "— br/min",
                transform=ax_bpm.transAxes,
                fontsize=11, color=color, alpha=0.75,
                ha="right", va="top", zorder=6,
            ))

            self._axes_bpm.append(ax_bpm)
            self._axes_resp.append(ax_resp)

            # ── HRV panel ─────────────────────────────────────────────
            _style_primary(ax_hrv)
            _add_bands(ax_hrv, HRV_ZONES, HRV_ZONE_LABELS)

            hrv_line, = ax_hrv.plot([], [], color=color, lw=2.0,
                                    zorder=5, solid_capstyle="round")
            self._hrv_lines.append(hrv_line)
            ax_hrv.set_ylim(0, 120)
            ax_hrv.set_ylabel("RMSSD (ms)", color=LABEL_C, fontsize=9)
            ax_hrv.set_xlabel("Time", color=LABEL_C, fontsize=9)

            self._hrv_texts.append(ax_hrv.text(
                0.99, 0.95, "— ms",
                transform=ax_hrv.transAxes,
                fontsize=13, fontweight="bold", color=color,
                ha="right", va="top", zorder=6,
            ))

            self._axes_hrv.append(ax_hrv)

        self._fig = fig
        fig.autofmt_xdate(rotation=30, ha="right")
        fig.tight_layout(rect=[0, 0, 1, 0.98], h_pad=1.5, w_pad=2.5)

    def _update(self, _frame):
        now  = mdates.date2num(datetime.now())
        xmin = now - WINDOW_DAYS
        self._frame_counter += 1
        compute_resp = (self._frame_counter % RESP_COMPUTE_EVERY == 0)

        with self._lock:
            for i in range(self.n):
                color = TRACK_COLORS[i % len(TRACK_COLORS)]

                # BPM line
                ts = list(self._bpm_times[i])
                if ts:
                    self._bpm_lines[i].set_data(ts, list(self._bpms[i]))
                bpm = self._latest_bpm[i]
                if bpm is not None:
                    self._bpm_texts[i].set_text(str(bpm))
                    zone = next(
                        (lbl for y, lbl in sorted(HR_ZONE_LABELS.items()) if bpm < y),
                        "Peak",
                    )
                    self._bpm_statuses[i].set_text(zone)
                    self._bpm_statuses[i].set_color(
                        color if self._connected[i] else "#aaaaaa"
                    )

                # Respiration line (right axis, twinx follows bpm xlim automatically)
                if compute_resp:
                    resp = _compute_resp_rate(
                        list(self._rr_times[i]), list(self._rr_vals[i])
                    )
                    if resp is not None:
                        self._latest_resp[i] = resp
                        self._resp_times[i].append(now)
                        self._resp_vals[i].append(resp)
                        cutoff = now - WINDOW_DAYS
                        while self._resp_times[i] and self._resp_times[i][0] < cutoff:
                            self._resp_times[i].popleft()
                            self._resp_vals[i].popleft()

                ts = list(self._resp_times[i])
                if ts:
                    self._resp_lines[i].set_data(ts, list(self._resp_vals[i]))
                resp = self._latest_resp[i]
                self._resp_texts[i].set_text(
                    f"{resp:.1f} br/min" if resp is not None else "— br/min"
                )

                # HRV line
                ts = list(self._hrv_times[i])
                if ts:
                    self._hrv_lines[i].set_data(ts, list(self._hrv_vals[i]))
                rmssd = self._latest_rmssd[i]
                self._hrv_texts[i].set_text(
                    f"{rmssd:.1f} ms" if rmssd is not None else "— ms"
                )

                # twinx shares x with ax_bpm, so only set xlim on primary axes
                self._axes_bpm[i].set_xlim(xmin, now)
                self._axes_hrv[i].set_xlim(xmin, now)

        return self._bpm_lines + self._resp_lines + self._hrv_lines
