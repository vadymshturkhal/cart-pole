from collections import deque
from typing import Optional
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import config


class LossPlot(FigureCanvas):
    """
    Smooth, semi-transparent loss plot for training diagnostics.

    Features
    - Red/purple line for aesthetic distinction from reward curve.
    - Optional moving average smoothing.
    - Auto y-scaling with safe margins.
    - Fast incremental updates.
    """

    def __init__(self,
                 max_episodes: Optional[int] = None,
                 ma_window: int = 20,
                 use_log_scale: bool = False):

        self.fig = Figure(figsize=(5, 2.4), tight_layout=True)
        super().__init__(self.fig)

        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Loss Curve")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Loss")
        if use_log_scale:
            self.ax.set_yscale("log")

        # Line styles: purple, semi-transparent
        (self.raw_line,) = self.ax.plot([], [], color=(0.6, 0.1, 0.8, 0.5), lw=1.5, label="Loss")
        (self.ma_line,) = self.ax.plot([], [], color=(0.6, 0.1, 0.8, 0.9), lw=2.0, label=f"MA({ma_window})")

        self.ax.legend(loc="upper left", frameon=False, fontsize=8)

        # Internal buffers
        self._losses: list[float] = []
        self._ma: list[float] = []
        self._ma_window = max(1, int(ma_window))
        self._window = deque(maxlen=self._ma_window)
        self._window_sum = 0.0
        self._max_episodes = max_episodes or getattr(config, "EPISODES", 1000)

        #  Initial axes limits
        self.ax.set_xlim(0, self._max_episodes)
        self.ax.set_ylim(0, 1.0)

    # ---------- Public API ----------
    def reset(self, max_steps: Optional[int] = None):
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
        """Incremental update from training loop."""
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

        # Adjust axes
        self.ax.set_xlim(0, max(self._max_steps, len(self._losses) + 1))
        self._autoscale_y(ys, ma_y)
        self.draw_idle()

    # ---------- Internals ----------
    def _autoscale_y(self, *ys_arrays):
        data = np.concatenate([y for y in ys_arrays if y is not None and len(y) > 0])
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
            margin = span * 0.08
            self.ax.set_ylim(y_min - margin, y_max + margin)
