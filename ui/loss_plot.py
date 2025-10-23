from ui.reward_plot import RewardPlot


class LossPlot(RewardPlot):
    """
    LossPlot — subclass of RewardPlot for visualizing training loss.

    Inherits all behavior (moving average, autoscaling, reset, export)
    and only overrides labels, title, and styling for distinction.
    """

    def __init__(self, max_episodes=None, ma_window=20, normalize=False):
        super().__init__(max_episodes=max_episodes, ma_window=ma_window, normalize=normalize)

        # Override titles and labels
        self.ax.set_title("Loss Curve", fontsize=10)
        self.ax.set_ylabel("Loss", fontsize=9)

        # Adjust line styles for clarity
        self.raw_line.set_color("#8888ff")  # soft blue
        self.ma_line.set_color("#0040ff")   # stronger blue
        self.ma_line.set_label(f"MA({ma_window})")

        # Update legend to match new label colors
        self.ax.legend(loc="upper right", frameon=False, fontsize=8)

    # Optional override (for semantic clarity)
    def add_point(self, losses: list[float]):
        """Identical to RewardPlot.add_point but with Loss semantics."""
        super().add_point(losses)
