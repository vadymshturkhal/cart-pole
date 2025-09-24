from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class RewardPlot(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(4, 2))
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Training Curve")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward (normalized)")
        self.line, = self.ax.plot([], [], color="blue")
        self.rewards = []

    def update_plot(self, rewards, episodes):
        if not rewards:
            return
        xs = list(range(len(rewards)))
        ys = [(r - min(rewards)) / max(1, (max(rewards) - min(rewards))) for r in rewards]
        self.line.set_data(xs, ys)
        self.ax.set_xlim(0, episodes)
        self.ax.set_ylim(0, 1.0)
        self.draw()
