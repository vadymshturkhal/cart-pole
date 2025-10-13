from collections import deque
from typing import Optional
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import config
import csv


class LossPlot(FigureCanvas):
    """
    Professional, flexible loss plot for training diagnostics.

    Features
    - Two series: raw loss and moving average.
    - Auto y-scale with margin or log option.
    - Incremental updates for performance.
    - Resettable between runs.
    - Export helpers (PNG, CSV).
    """

    def __init__(self, max_steps: Optional[int] = None, ma_window: int = 20):
        self.fig = Figure(figsize=(5, 2.6), tight_layout=True)
        super().__init__(self.fig)

        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Loss Curve", fontsize=10)
        self.ax.set_xlabel("Episode", fontsize=9)
        self.ax.set_ylabel("Loss", fontsize=9)
        self.ax.margins(x=0.02, y=0.1)

        # if use_log_scale:
            # self.ax.set_yscale("log")

        # Lines — purple tone for distinction
        (self.raw_line,) = self.ax.plot([], [], color=(0.6, 0.1, 0.8, 0.5), lw=1.2, label="Loss")
        (self.ma_line,) = self.ax.plot([], [], color=(0.6, 0.1, 0.8, 0.9), lw=2.0, label=f"MA({ma_window})")

        # Legend at top-left (to match RewardPlot)
        self.ax.legend(loc="upper left", frameon=False, fontsize=8)

        # Internal state
        self._losses: list[float] = []
        self._ma: list[float] = []
        self._ma_window = max(1, int(ma_window))
        self._window = deque(maxlen=self._ma_window)
        self._window_sum = 0.0
        self._max_steps = max_steps or getattr(config, "EPISODES", 1000)

        # Axes limits
        self.ax.set_xlim(0, self._max_steps)
        self.ax.set_ylim(0.0, 1.0)

    # ---------- Public API ----------
    def reset(self, max_steps: Optional[int] = None):
        """Reset plot to empty state."""
        self._losses.clear()
        self._ma.clear()
        self._window.clear()
        self._window_sum = 0.0

        if max_steps is not None:
            self._max_steps = int(max_steps)

        self.raw_line.set_data([], [])
        self.ma_line.set_data([], [])
        self.ax.set_xlim(0, self._max_steps)
        self.ax.set_ylim(0, 1.0)
        self.draw_idle()

    def add_point(self, loss: float):
        """Incrementally add a new loss value."""
        self._losses.append(float(loss))

        # Rolling mean
        self._window_sum += loss
        if len(self._window) == self._window.maxlen:
            self._window_sum -= self._window[0]
        self._window.append(loss)
        self._ma.append(self._window_sum / len(self._window))

        xs = np.arange(1, len(self._losses) + 1)
        ys = np.array(self._losses)
        ma_y = np.array(self._ma)

        self.raw_line.set_data(xs, ys)
        self.ma_line.set_data(xs, ma_y)

        self.ax.set_xlim(0, max(self._max_steps, len(self._losses) + 1))
        self._autoscale_y(ys, ma_y)
        self.draw_idle()

    def export_png(self, path: str):
        """Save current figure as PNG."""
        self.fig.savefig(path, dpi=160)

    def export_csv(self, path: str):
        """Export loss values and moving average to CSV."""
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "loss", f"ma_{self._ma_window}"])
            for i, (l, m) in enumerate(zip(self._losses, self._ma), start=1):
                w.writerow([i, l, m])

    # ---------- Internals ----------
    def _autoscale_y(self, *ys_arrays):
        data = np.concatenate([y for y in ys_arrays if y is not None and len(y) > 0]) \
            if ys_arrays else np.array(self._losses, dtype=float)
        if data.size == 0 or not np.isfinite(data).any():
            self.ax.set_ylim(0, 1.0)
            return
        y_min = float(np.nanmin(data))
        y_max = float(np.nanmax(data))
        if y_min == y_max:
            pad = max(1.0, abs(y_max) * 0.1)
            self.ax.set_ylim(y_min - pad, y_max + pad)
        else:
            span = y_max - y_min
            margin = span * 0.08 if span > 0 else 1.0
            self.ax.set_ylim(y_min - margin, y_max + margin)

    def _recompute_ma(self):
        """Recalculate moving average from all data."""
        self._ma = []
        if not self._losses:
            return
        w = self._ma_window
        cumsum = np.cumsum(np.insert(self._losses, 0, 0.0))
        for i in range(1, len(self._losses) + 1):
            start = max(0, i - w)
            window_sum = cumsum[i] - cumsum[start]
            window_len = i - start
            self._ma.append(window_sum / window_len)

    def _redraw(self):
        xs = np.arange(1, len(self._losses) + 1)
        self.raw_line.set_data(xs, np.array(self._losses))
        self.ma_line.set_data(xs, np.array(self._ma))
        self.ax.set_xlim(0, max(self._max_steps, len(self._losses) + 1))
        self._autoscale_y()
        self.draw_idle()
