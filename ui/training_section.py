from __future__ import annotations
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit,
    QComboBox, QFileDialog, QTabWidget, QHBoxLayout,
)
from PySide6.QtGui import QTextCursor
from PySide6.QtCore import Signal
from ui.agent_dialog import AgentDialog
from ui.agent_config_dialog import AgentConfigDialog
from ui.nn_config_dialog import NNConfigDialog
from ui.environment_config_dialog import EnvironmentConfigDialog
from ui.reward_plot import RewardPlot
from ui.loss_plot import LossPlot
from ui.training_controller import TrainingController
from ui.training_ui_builder import TrainingUIBuilder
from utils.agent_factory import AGENTS
import config
import os
import torch
from ui.test_model_dialog import TestModelDialog


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

        # --- Connect UI buttons ---
        self._connect_ui()

    # ------------------------------------------------------------------
    # UI Connections
    # ------------------------------------------------------------------
    def _connect_ui(self):
        u = self.ui
        u.back_btn.clicked.connect(self.back_to_main.emit)
        u.agent_btn.clicked.connect(self._choose_agent)
        u.train_btn.clicked.connect(self._start_training)
        u.stop_btn.clicked.connect(self._stop_training)
        u.save_btn.clicked.connect(self._save_agent_as)
        u.load_btn.clicked.connect(self._load_model)
        u.agent_config_btn.clicked.connect(self._show_agent_config)
        u.env_config_btn.clicked.connect(self._show_environment_config)
        u.nn_btn.clicked.connect(self._show_nn_config)
        u.device_box.currentTextChanged.connect(self._on_device_changed)

    # ------------------------------------------------------------------
    # Core Actions
    # ------------------------------------------------------------------
    def _choose_agent(self) -> None:
        dlg = AgentDialog(self, current_agent=self.agent_name)
        if dlg.exec():
            self.nn_locked = False
            self.agent_name = dlg.get_selection()
            AgentClass = AGENTS[self.agent_name]
            self.hyperparams = AgentClass.get_default_hyperparams()
            self.ui.agent_btn.setText(f"Agent: {self.agent_name}")
            self.ui.agent_config_btn.setText(f"{self.agent_name} Configuration")
            self._log(f"✅ Selected {self.agent_name} agent")

    def _start_training(self) -> None:
        # Clear plots
        self.ui.reward_plot.reset()
        self.ui.loss_plot.reset()

        self._set_training_buttons(False)

        self.controller.start_training(
            config.ENV_NAME,
            self.agent_name,
            self.hyperparams,
            config.EPISODES,
            config.RENDER_MODE,
            config.MAX_STEPS,
            self.selected_model_file,
        )

        self._log(
            f"🚀 Training started on {config.ENV_NAME} — "
            f"{config.EPISODES} episodes, {config.MAX_STEPS} steps/episode | Render: {config.RENDER_MODE}"
        )

    def _stop_training(self) -> None:
        self.controller.stop_training()
        self._set_training_buttons(True)

    def _save_agent_as(self) -> None:
        default_dir = os.path.join(config.TRAINED_MODELS_FOLDER)
        user_dir, _ = QFileDialog.getSaveFileName(self, "Save Agent As", default_dir)
        if not user_dir:
            self._log("💡 Save canceled by user.")
            return
        self.controller.save_model(user_dir, self.ui.reward_plot, self.ui.loss_plot)

    def _load_model(self):
        """Load an existing trained model and apply its configuration."""
        dlg = TestModelDialog("trained_models")
        if dlg.exec():
            self.selected_model_file = dlg.get_selected()
            if not self.selected_model_file:
                self._log("⚠ Load canceled by user.")
                return

            try:
                checkpoint = torch.load(self.selected_model_file, map_location=config.DEVICE)
            except Exception as e:
                self._log(f"❌ Failed to load model: {e}")
                return

            # Agent setup
            self.agent_name = checkpoint.get("agent_name", self.agent_name)
            self.ui.agent_btn.setText(f"Agent: {self.agent_name}")
            AgentClass = AGENTS.get(self.agent_name)
            if AgentClass:
                self.hyperparams = checkpoint.get("hyperparams", AgentClass.get_default_hyperparams())
            self.ui.agent_config_btn.setText(f"{self.agent_name} Configuration")

            # Environment setup
            config.ENV_NAME = checkpoint.get("environment", config.ENV_NAME)
            config.MAX_STEPS = checkpoint.get("max_steps", config.MAX_STEPS)
            config.EPISODES = checkpoint.get("episodes_total", config.EPISODES)
            config.RENDER_MODE = checkpoint.get("render_mode", config.RENDER_MODE)

            # Neural Network setup
            nn_cfg = checkpoint.get("nn_config", {})
            config.HIDDEN_LAYERS = nn_cfg.get("hidden_layers", config.HIDDEN_LAYERS)
            config.LR = nn_cfg.get("lr", config.LR)
            config.ACTIVATION = nn_cfg.get("activation", config.ACTIVATION)
            config.DROPOUT = nn_cfg.get("dropout", config.DROPOUT)
            config.HIDDEN_ACTIVATION = nn_cfg.get("activation", config.ACTIVATION)
            config.OPTIMIZER = nn_cfg.get("optimizer", getattr(config, "OPTIMIZER", "adam"))
            device_str = nn_cfg.get("device", str(config.DEVICE))
            config.DEVICE = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")

            # Log
            self._log(f"📦 Model loaded: {self.agent_name} in {config.ENV_NAME}")
            self._log(f"✅ NN Config loaded: LR={config.LR}, Dropout={config.DROPOUT}, Act={config.ACTIVATION}")
            self._log(f"→ Hyperparameters: {len(self.hyperparams)} params")
            self._log(f"→ NN: layers={config.HIDDEN_LAYERS}, lr={config.LR}, dropout={config.DROPOUT}, device={config.DEVICE}")
            self._log(f"→ Episodes trained: {checkpoint.get('episodes_trained', 'N/A')}")

            # Lock NN architecture
            self.nn_locked = True

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------
    def _show_environment_config(self) -> None:
        dlg = EnvironmentConfigDialog(self, read_only=self.training_active)
        if dlg.exec():
            updates = dlg.get_updated_config()
            self._log(
                f"🌍 Environment configured: {updates['ENV_NAME']} | "
                f"{updates['MAX_STEPS']} steps/ep | {updates['EPISODES']} episodes | "
                f"Render: {updates['RENDER_MODE']}"
            )
    
    def _show_agent_config(self):
        dlg = AgentConfigDialog(self.agent_name, self.hyperparams.copy(), self, self.training_active)
        if dlg.exec() and not self.training_active:
            self.hyperparams = dlg.get_updated_params()
            self._log("⚙️ Agent hyperparameters updated.")

    def _show_nn_config(self):
        dlg = NNConfigDialog(self, read_only=self.training_active, lock_hidden_layers=self.nn_locked)
        if dlg.exec() and not self.training_active:
            updates = dlg.get_updated_config()
            config.HIDDEN_LAYERS = updates["HIDDEN_LAYERS"]
            config.LR = updates["LR"]
            config.ACTIVATION = updates["ACTIVATION"]
            config.DROPOUT = updates["DROPOUT"]
            config.DEVICE = config.torch.device(updates["DEVICE"])
            self._log(
                f"🧠 NN updated: layers={updates['HIDDEN_LAYERS']} activation={updates['ACTIVATION']} "
                f"dropout={updates['DROPOUT']:.2f} device={config.DEVICE}"
            )

    # ------------------------------------------------------------------
    # UI Feedback and States
    # ------------------------------------------------------------------
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

        # Dim NN button if model is locked
        if self.nn_locked:
            self.ui.nn_btn.setStyleSheet("color:#888;")
        else:
            self.ui.nn_btn.setStyleSheet("")

    def _log(self, message: str) -> None:
        """Append text to the console output."""
        self.ui.console_output.append(message)
        self.ui.console_output.moveCursor(QTextCursor.End)
