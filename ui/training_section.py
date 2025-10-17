from __future__ import annotations
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, 
    QComboBox, QFileDialog, QTabWidget, QHBoxLayout,
)
from PySide6.QtCore import Signal
from ui.agent_dialog import AgentDialog
from ui.agent_details_dialog import AgentDetailsDialog
from ui.nn_config_dialog import NNConfigDialog
from ui.environment_config_dialog import EnvironmentConfigDialog
from ui.reward_plot import RewardPlot
from ui.loss_plot import LossPlot
from utils.agent_factory import AGENTS
from ui.training_controller import TrainingController
import config
import os
import torch
import gymnasium as gym


class TrainingSection(QWidget):
    """UI layer for managing training interactions and visualization."""
    back_to_main = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.controller = TrainingController()

        # --- Default agent setup ---
        self.agent_name = config.DEFAULT_AGENT
        AgentClass = AGENTS[self.agent_name]
        self.hyperparams = AgentClass.get_default_hyperparams()

        # --- Connect Controller Signals ---
        self.controller.progress.connect(self._on_progress)
        self.controller.finished.connect(self._on_finished)
        self.controller.status.connect(self._update_status)

        self._build_training_section()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------
    def _build_training_section(self) -> None:
        layout = QVBoxLayout(self)

        # Tabs for plots
        self.tabs = QTabWidget()
        self.reward_plot, self.loss_plot = RewardPlot(), LossPlot()
        self.tabs.addTab(self.reward_plot, "Training Curve")
        self.tabs.addTab(self.loss_plot, "Loss Curve")
        self.tabs.setMinimumHeight(300)
        self.tabs.setMaximumHeight(310)
        layout.addWidget(self.tabs)

        # --- Agent row ---
        layout.addWidget(QLabel("Agent:"))
        self.agent_btn = QPushButton(f"{self.agent_name}")
        self.train_btn = QPushButton("Start Training")
        self.stop_btn = QPushButton("Stop Training")
        self.save_btn = QPushButton("Save Model")
        self._add_row(layout, [self.agent_btn, self.train_btn, self.stop_btn, self.save_btn])

        # --- Configuration row ---
        # Environment
        self.env_config_btn = QPushButton("Configure Environment")
        self.env_config_btn.setStyleSheet("font-weight:bold; padding:6px;")
        self.env_config_btn.clicked.connect(self._show_environment_config)

        # Agent
        self.agent_config_btn = QPushButton(f"{self.agent_name} configuration")
        self.agent_config_btn.clicked.connect(self._show_agent_config)

        # NN
        self.nn_btn = QPushButton("NN configuration")
        self.nn_btn.clicked.connect(self._show_nn_config)

        # Device
        self.device_label = QLabel("Device:")
        self.device_label.setStyleSheet("font-weight:bold; margin-left:10px;")
        self.device_box = QComboBox()
        self.device_box.addItems(["cpu", "cuda"])
        self._init_device_box()
        self.device_box.currentTextChanged.connect(self._on_device_changed)

        # Config row
        self._add_row(layout, [self.env_config_btn, self.agent_config_btn, self.nn_btn, self.device_label, self.device_box])

        # --- Status label ---
        self.status_label = QLabel("Idle")
        layout.addWidget(self.status_label)

        # --- Back button ---
        back_btn = QPushButton("⬅ Back to Main Menu")
        back_btn.setMinimumHeight(40)
        back_btn.setStyleSheet("font-size: 16px;")
        back_btn.clicked.connect(self.back_to_main.emit)
        layout.addWidget(back_btn)

        # --- Button connections ---
        self.agent_btn.clicked.connect(self._choose_agent)
        self.train_btn.clicked.connect(self._start_training)
        self.stop_btn.clicked.connect(self._stop_training)
        self.save_btn.clicked.connect(self._save_agent_as)

        # Initial button states
        self.save_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Core Actions
    # ------------------------------------------------------------------
    def _choose_agent(self) -> None:
        dlg = AgentDialog(self, current_agent=self.agent_name)
        if dlg.exec():
            agent_name = dlg.get_selection()
            self.agent_name = agent_name
            AgentClass = AGENTS[self.agent_name]
            self.hyperparams = AgentClass.get_default_hyperparams()
            self.agent_btn.setText(agent_name)
            self.details_btn.setText(f"Configure {self.agent_name}")
            self.status_label.setText(f"✅ Selected {self.agent_name} agent")

    def _start_training(self) -> None:
        self._set_training_buttons(False)
        self.controller.start_training(
            config.ENV_NAME,
            self.agent_name,
            self.hyperparams,
            config.EPISODES,
            config.RENDER_MODE,
            config.MAX_STEPS,
        )

        self.status_label.setText(
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
            self.status_label.setText("💡 Save canceled by user.")
            return
        self.controller.save_model(user_dir, self.reward_plot, self.loss_plot)

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------
    def _show_environment_config(self) -> None:
        dlg = EnvironmentConfigDialog(self)
        if dlg.exec():
            updates = dlg.get_updated_config()
            self.status_label.setText(
                f"🌍 Environment configured: {updates['ENV_NAME']} | "
                f"{updates['MAX_STEPS']} steps/ep | {updates['EPISODES']} episodes | "
                f"Render: {updates['RENDER_MODE']}"
            )
    
    def _show_agent_config(self):
        dlg = AgentDetailsDialog(self.agent_name, self.hyperparams.copy(), self)
        if dlg.exec():
            self.hyperparams = dlg.get_updated_params()
            self.status_label.setText("⚙️ Agent hyperparameters updated.")

    def _show_nn_config(self):
        dlg = NNConfigDialog(self)
        if dlg.exec():
            updates = dlg.get_updated_config()
            config.HIDDEN_LAYERS = updates["HIDDEN_LAYERS"]
            config.LR = updates["LR"]
            config.ACTIVATION = updates["ACTIVATION"]
            config.DROPOUT = updates["DROPOUT"]
            config.DEVICE = config.torch.device(updates["DEVICE"])
            self.status_label.setText(
                f"🧠 NN updated: layers={updates['HIDDEN_LAYERS']} activation={updates['ACTIVATION']} "
                f"dropout={updates['DROPOUT']:.2f} device={config.DEVICE}"
            )

    def _on_progress(self, ep: int, episodes: int, ep_reward: float, rewards: list, avg_loss: float) -> None:
        avg20 = sum(rewards[-20:]) / min(len(rewards), 20)
        global_avg = sum(rewards) / len(rewards)
        self.status_label.setText(
            f"Ep {ep+1}/{episodes} — R {ep_reward:.1f}, Avg20 {avg20:.1f}, Global {global_avg:.1f}, AvgLoss {avg_loss:.2f}"
        )
        self.reward_plot.update_plot(rewards, episodes)
        self.loss_plot.add_point(avg_loss)

    def _on_finished(self) -> None:
        self._set_training_buttons(True)

    def _update_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _add_row(self, parent_layout, widgets) -> None:
        row = QHBoxLayout()
        for w in widgets:
            row.addWidget(w)
        parent_layout.addLayout(row)

    def _init_device_box(self) -> None:
        gpu_available = torch.cuda.is_available()
        model = self.device_box.model()
        cuda_index = self.device_box.findText("cuda")
        if not gpu_available and cuda_index >= 0:
            model.item(cuda_index).setEnabled(False)
        default_device = "cuda" if gpu_available else "cpu"
        self.device_box.setCurrentText(default_device)
        config.DEVICE = torch.device(default_device)

    def _on_device_changed(self, choice: str) -> None:
        config.DEVICE = torch.device(choice if choice == "cpu" or torch.cuda.is_available() else "cpu")
        color = "#3a7" if "cuda" in str(config.DEVICE) else "#666"
        self.device_label.setStyleSheet(f"font-weight:bold; color:{color}; margin-left:10px;")
        self.status_label.setText(f"🖥️ Device set to {config.DEVICE}")

    def _set_training_buttons(self, enable: bool) -> None:
        """Toggle interactive buttons based on training state."""
        toggled_widgets = [
            self.env_config_btn, self.agent_config_btn, self.nn_btn, self.device_label, 
            self.device_box, self.agent_btn, self.train_btn, self.save_btn, 
        ]
        for w in toggled_widgets:
            w.setEnabled(enable)
        self.stop_btn.setEnabled(not enable)
