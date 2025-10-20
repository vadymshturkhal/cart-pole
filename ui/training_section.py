from __future__ import annotations
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QTextCursor
from PySide6.QtCore import Signal
from ui.training_controller import TrainingController
from ui.training_ui_builder import TrainingUIBuilder
from ui.training_actions import TrainingActions
from utils.agent_factory import AGENTS
import config
import torch
from datetime import datetime


class TrainingSection(QWidget):
    """High-level training controller and UI manager."""
    back_to_main = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # --- Build UI ---
        self.ui = TrainingUIBuilder(self)

        # --- Controller and state ---
        self.controller = TrainingController()
        self.controller.add_plots(self.ui.reward_plot, self.ui.loss_plot)
        self.selected_model_file = None
        self.training_active = False
        self.nn_locked = False

        # Delegate logic
        self.actions = TrainingActions(self, self.controller)
        self.ui.back_btn.clicked.connect(self.back_to_main.emit)
        self.ui.device_box.currentTextChanged.connect(self._on_device_changed)

        # --- Default agent setup ---
        self.agent_name = config.DEFAULT_AGENT
        AgentClass = AGENTS[self.agent_name]
        self.hyperparams = AgentClass.get_default_hyperparams()
        self.ui.agent_btn.setText(f"Agent: {self.agent_name}")
        self.ui.agent_config_btn.setText(f"{self.agent_name} Configuration")

        # --- Connect Controller Signals ---
        self.controller.progress.connect(self._on_progress)
        self.controller.finished.connect(self._on_finished)
        self.controller.status.connect(self._update_status)

    # Controller callbacks
    def _on_progress(self, ep: int, episodes: int, ep_reward: float, rewards: list, avg_loss: float) -> None:
        avg20 = sum(rewards[-20:]) / min(len(rewards), 20)
        global_avg = sum(rewards) / len(rewards)
        self._log(
            f"Ep {ep+1}/{episodes} — R {ep_reward:.1f}, Avg20 {avg20:.1f}, Global {global_avg:.1f}, AvgLoss {avg_loss:.2f}"
        )
        self.ui.reward_plot.update_plot(rewards, episodes)
        self.ui.loss_plot.add_point(avg_loss)

    def _on_finished(self) -> None:
        self._set_training_buttons(True)

    def _update_status(self, message: str) -> None:
        self._log(message)

    # Device and UI states
    def _on_device_changed(self, choice: str) -> None:
        config.DEVICE = torch.device(choice if choice == "cpu" or torch.cuda.is_available() else "cpu")
        color = "#3a7" if "cuda" in str(config.DEVICE) else "#666"
        self.ui.device_label.setStyleSheet(f"font-weight:bold; color:{color}; margin-left:10px;")
        self._log(f"🖥️ Device set to {config.DEVICE}")

    def _set_training_buttons(self, enable: bool) -> None:
        """Enable/disable UI controls during training."""
        self.training_active = not enable  # True while training
        toggled_widgets = [
            self.ui.device_label, self.ui.device_box, self.ui.agent_btn, 
            self.ui.train_btn, self.ui.save_btn, self.ui.load_btn,
        ]
        
        # self.agent_config_btn, , self.env_config_btn, self.nn_btn are always enable

        for w in toggled_widgets:
            w.setEnabled(enable)
        self.ui.stop_btn.setEnabled(not enable)

    def _log(self, message: str) -> None:
        """Append text to the console output."""

        ts = datetime.now().strftime("[%H:%M:%S]")
        self.ui.console_output.append(f"{ts} {message}")
        self.ui.console_output.moveCursor(QTextCursor.End)
