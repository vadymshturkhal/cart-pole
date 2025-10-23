from collections import deque
from typing import Optional
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import config
import csv


class RewardPlot(FigureCanvas):
    """
    Professional, flexible training plot.

    Features
    - Two series: raw reward and moving average.
    - Auto y-scale with margin, or normalized [0,1].
    - Incremental point updates for speed.
    - Resettable between runs.
    - Configurable smoothing window.
    - Export helpers (PNG, CSV).
    """

    def __init__(self,
                 max_episodes: Optional[int] = None,
                 ma_window: int = 20,
                 normalize: bool = False):
        
        """
            MA means Moving Average.
            It smooths the reward curve by averaging the last N episode rewards, where N is the chosen window.
        """

        self.fig = Figure(figsize=(5, 2.6), tight_layout=True)
        super().__init__(self.fig)

        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Training Curve", fontsize=10)
        self.ax.set_xlabel("Episode", fontsize=9)
        self.ax.set_ylabel("Reward", fontsize=9)
        self.ax.margins(x=0.02, y=0.1)

        # Lines
        (self.raw_line,) = self.ax.plot([], [], lw=1.0, label="Reward")
        (self.ma_line,)  = self.ax.plot([], [], lw=2.0, alpha=0.85, label=f"MA({ma_window})")

        self.ax.legend(loc="upper left", frameon=False, fontsize=8)

        # State
        self._rewards: list[float] = []
        self._ma: list[float] = []
        self._ma_window = max(1, int(ma_window))
        self._normalize = bool(normalize)
        self._max_episodes = max_episodes or getattr(config, "EPISODES", 1000)

        # Prealloc buffers for fast rolling mean (deque for window)
        self._window = deque(maxlen=self._ma_window)
        self._window_sum = 0.0

        # Initial axes limits
        self.ax.set_xlim(0, self._max_episodes)
        self.ax.set_ylim(-1.0, 1.0)

    # ---------- Public API ----------
    def reset(self,
              max_episodes: Optional[int] = None,
              ma_window: Optional[int] = None,
              normalize: Optional[bool] = None):
        self._rewards.clear()
        self._ma.clear()
        self._window.clear()
        self._window_sum = 0.0

        if max_episodes is not None:
            self._max_episodes = int(max_episodes)
        if ma_window is not None:
            self._ma_window = max(1, int(ma_window))
            self._window = deque(maxlen=self._ma_window)
        if normalize is not None:
            self._normalize = bool(normalize)

        self.raw_line.set_data([], [])
        self.ma_line.set_data([], [])
        self.ax.set_xlim(0, self._max_episodes)
        self._autoscale_y()  # sets to a sane default
        self.draw_idle()

    def set_ma_window(self, ma_window: int):
        self._ma_window = max(1, int(ma_window))
        self._window = deque(maxlen=self._ma_window)
        # Recompute MA from history
        self._recompute_ma()
        self._redraw()

    def set_normalize(self, normalize: bool):
        self._normalize = bool(normalize)
        self._redraw()

    def set_title(self, text: str):
        self.ax.set_title(text)
        self.draw_idle()

    def add_point(self, rewards: list[float]):
        """Backward compatible batch update used by existing code."""
        if not rewards:
            return

        self._rewards = [float(r) for r in rewards]
        self._recompute_ma()
        xs = np.arange(1, len(self._rewards) + 1)

        if self._normalize and len(self._rewards) > 1:
            rmin = min(self._rewards)
            rmax = max(self._rewards)
            denom = max(1e-9, (rmax - rmin))
            raw_y = (np.array(self._rewards) - rmin) / denom
            ma_y  = (np.array(self._ma)      - rmin) / denom
            self.ax.set_ylabel("Reward (normalized)")
            self.ax.set_ylim(0.0, 1.0)
        else:
            raw_y = np.array(self._rewards)
            ma_y  = np.array(self._ma)
            self.ax.set_ylabel("Reward")
            self._autoscale_y(raw_y, ma_y)

        self.raw_line.set_data(xs, raw_y)
        self.ma_line.set_data(xs, ma_y)
        self.ax.set_xlim(0, max(self._max_episodes, len(self._rewards) + 1))
        self.draw_idle()

    def export_png(self, path: str):
        self.fig.savefig(path, dpi=160)

    def export_csv(self, path: str):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode", "reward", f"ma_{self._ma_window}"])
            for i, (r, m) in enumerate(zip(self._rewards, self._ma), start=1):
                w.writerow([i, r, m])

    # ---------- Internals ----------
    def _recompute_ma(self):
        self._ma = []
        if not self._rewards:
            return
        w = self._ma_window
        cumsum = np.cumsum(np.insert(self._rewards, 0, 0.0))
        for i in range(1, len(self._rewards) + 1):
            start = max(0, i - w)
            window_sum = cumsum[i] - cumsum[start]
            window_len = i - start
            self._ma.append(window_sum / window_len)

    def _autoscale_y(self, *ys_arrays):
        data = np.concatenate([y for y in ys_arrays if y is not None and len(y) > 0]) \
               if ys_arrays else np.array(self._rewards, dtype=float)
        if data.size == 0 or not np.isfinite(data).any():
            self.ax.set_ylim(-1.0, 1.0)
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

    def _redraw(self):
        xs = np.arange(1, len(self._rewards) + 1)
        self.raw_line.set_data(xs, np.array(self._rewards))
        self.ma_line.set_data(xs, np.array(self._ma))
        self.ax.set_xlim(0, max(self._max_episodes, len(self._rewards) + 1))
        if self._normalize:
            self.set_normalize(True)  # reuse transform path
        else:
            self._autoscale_y()
        self.draw_idle()
