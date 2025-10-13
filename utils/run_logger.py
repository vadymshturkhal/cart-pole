import os
import json
import datetime
import config


class RunLogger:
    """
    Handles experiment folder creation and artifact saving for training runs.

    Creates a timestamped directory under base_dir for each run, and provides
    helper methods to save model weights, plots, metrics, and configuration data.
    """

    def __init__(self, env_name: str, agent: str, reward_plot=None, loss_plot=None):
        self.env_name = env_name
        self.agent = agent
        self.agent_name = agent.name
        self.reward_plot = reward_plot
        self.loss_plot = loss_plot

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.timestamp = timestamp

        # Ensure autosave folder exist
        self.autosave_dir = os.path.join(config.TRAINED_MODELS_FOLDER, "autosave")
        os.makedirs(self.autosave_dir, exist_ok=True)

    def save_model(self, userdir):
        """Save model checkpoint to run_dir/model.pth."""
        if not hasattr(self.agent, "save"):
            raise AttributeError("Agent must have a .save() method")

        os.makedirs(userdir, exist_ok=True)
        save_model_path = os.path.join(userdir, f"{self.env_name}_{self.agent_name}.pth")
        self.agent.save(save_model_path, self.agent.get_checkpoint())

        # Save plots and config
        self._save_plots(userdir)
        self._save_config(userdir)

    def autosave_model(self):
        """Save model checkpoint to run_dir/self.autosave_dir"""
        if not hasattr(self.agent, "save"):
            raise AttributeError("Agent must have a .save() method")

        autosave_model_path = os.path.join(self.autosave_dir, f"{self.env_name}_{self.agent_name}.pth")
        self.agent.save(autosave_model_path, self.agent.get_checkpoint())
        
        # Save plots and config
        self._save_plots(self.autosave_dir)
        self._save_config(self.autosave_dir)

    def _save_plots(self, dir):
        """Export reward/loss plots as PNG/CSV."""
        if self.reward_plot:
            self.reward_plot.export_csv(os.path.join(dir, "rewards.csv"))
            self.reward_plot.export_png(os.path.join(dir, "rewards.png"))
        if self.loss_plot:
            self.loss_plot.export_csv(os.path.join(dir, "loss.csv"))
            self.loss_plot.export_png(os.path.join(dir, "loss.png"))

    def _save_config(self, dir):
        """Dump environment, agent, and hyperparams to config.json."""
        cfg = self.agent.get_checkpoint()

        # Delete tensor from config
        del cfg["model_state"]

        cfg_path = os.path.join(dir, "config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4)

        return cfg_path

    def __repr__(self):
        return f"<RunLogger dir='{self.run_dir}'>"
