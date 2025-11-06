from ui.reward_plot import RewardPlot


class LossPlot(RewardPlot):
    """
    LossPlot — subclass of RewardPlot for visualizing training loss.

    Inherits all behavior (moving average, autoscaling, reset, export)
    and only overrides labels, title, and styling for distinction.
    """

    def __init__(self, max_episodes=None, ma_window=20, normalize=False):
        super().__init__(max_episodes=max_episodes, ma_window=ma_window, normalize=normalize)

        self.ylabel = "Loss"
        self.ax.set_ylabel(self.ylabel, fontsize=9)

        # Labels
        self.raw_line.set_label(self.ylabel)
        self.ma_line.set_label(f"MA({ma_window})")

        # Line styles
        self.raw_line.set_color("#8888ff")  # soft blue
        self.ma_line.set_color("#0040ff")   # stronger blue

        # Update legend to match new label colors
        self.ax.legend(loc="upper left", frameon=False, fontsize=8)

    # Optional override (for semantic clarity)
    def add_point(self, losses: list[float]):
        """Identical to RewardPlot.add_point but with Loss semantics."""
        super().add_point(losses)

    def append_point(self, loss: float):
        super().append_point(loss)
        