import os
import json
import datetime


class RunLogger:
    """
    Handles experiment folder creation and artifact saving for training runs.

    Creates a timestamped directory under base_dir for each run, and provides
    helper methods to save model weights, plots, metrics, and configuration data.
    """

    def __init__(self, base_dir: str, env_name: str, agent: str):
        self.env_name = env_name
        self.agent = agent
        self.agent_name = agent.name

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.join(base_dir, f"{env_name}_{self.agent_name}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.timestamp = timestamp

    # Core artifact saving
    def save_model(self):
        """Save model checkpoint to run_dir/model.pth."""
        if not hasattr(self.agent, "save"):
            raise AttributeError("Agent must have a .save() method")
        model_path = os.path.join(self.run_dir, f"{self.env_name}_{self.agent_name}_{self.timestamp}.pth")
        self.agent.save(model_path, self.agent.get_checkpoint())
        return model_path

    def save_plots(self, reward_plot=None, loss_plot=None):
        """Export reward/loss plots as PNG/CSV."""
        if reward_plot:
            reward_plot.export_csv(os.path.join(self.run_dir, "rewards.csv"))
            reward_plot.export_png(os.path.join(self.run_dir, "rewards.png"))
        if loss_plot:
            loss_plot.export_png(os.path.join(self.run_dir, "loss_curve.png"))

    def save_config(self):
        """Dump environment, agent, and hyperparams to config.json."""
        cfg = self.agent.get_checkpoint()

        # Delete tensor from config
        del cfg["model_state"]

        cfg_path = os.path.join(self.run_dir, "config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4)

        return cfg_path

    def __repr__(self):
        return f"<RunLogger dir='{self.run_dir}'>"
