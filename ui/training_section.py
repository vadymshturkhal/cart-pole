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
from utils.agent_factory import AGENTS
from ui.training_controller import TrainingController
import config
import os
import torch
from ui.test_model_dialog import TestModelDialog


class TrainingSection(QWidget):
    """UI layer for managing training interactions and visualization."""
    back_to_main = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.controller = TrainingController()
        self.selected_model_file = None

        # --- Default agent setup ---
        self.agent_name = config.DEFAULT_AGENT
        AgentClass = AGENTS[self.agent_name]
        self.hyperparams = AgentClass.get_default_hyperparams()

        # --- Connect Controller Signals ---
        self.controller.progress.connect(self._on_progress)
        self.controller.finished.connect(self._on_finished)
        self.controller.status.connect(self._update_status)

        self.training_active = False
        self.nn_locked = False
        self._build_training_section()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------
    def _build_training_section(self) -> None:
        layout = QVBoxLayout(self)

        # Tabs for plots
        self.tabs = QTabWidget()
        self.reward_plot, self.loss_plot = RewardPlot(), LossPlot()
        self.controller.add_plots(self.reward_plot, self.loss_plot)
        self.tabs.addTab(self.reward_plot, "Training Curve")
        self.tabs.addTab(self.loss_plot, "Loss Curve")
        self.tabs.setMinimumHeight(300)
        self.tabs.setMaximumHeight(310)
        layout.addWidget(self.tabs)

        # --- Agent row ---
        self.agent_btn = QPushButton(f"Agent: {self.agent_name}")
        self.train_btn = QPushButton("Start Training")
        self.stop_btn = QPushButton("Stop Training")
        self.save_btn = QPushButton("Save Model")
        self.load_btn = QPushButton("Load Model")
        self._add_row(layout, [self.agent_btn, self.train_btn, self.stop_btn, self.save_btn, self.load_btn])

        # --- Configuration row ---
        # Agent
        self.agent_config_btn = QPushButton(f"{self.agent_name} Configuration")
        self.agent_config_btn.clicked.connect(self._show_agent_config)

        # Environment
        self.env_config_btn = QPushButton("Environment Configuration")
        self.env_config_btn.clicked.connect(self._show_environment_config)

        # NN
        self.nn_btn = QPushButton("NN Configuration")
        self.nn_btn.clicked.connect(self._show_nn_config)

        # Device
        self.device_label = QLabel("Device:")
        self.device_label.setStyleSheet("font-weight:bold; margin-left:10px;")
        self.device_box = QComboBox()
        self.device_box.addItems(["cpu", "cuda"])
        self._init_device_box()
        self.device_box.currentTextChanged.connect(self._on_device_changed)

        # Config row
        self._add_row(layout, [self.agent_config_btn, self.env_config_btn, self.nn_btn, self.device_label, self.device_box])

        # --- Console output ---
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setPlaceholderText("Console output will appear here...")
        self.console_output.setStyleSheet("""
            QTextEdit {
                background-color: #111;
                color: #ddd;
                font-family: Consolas, monospace;
                font-size: 13px;
                border: 1px solid #333;
                border-radius: 6px;
                padding: 4px;
            }
        """)
        self.console_output.setMinimumHeight(140)
        layout.addWidget(self.console_output)

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
        self.load_btn.clicked.connect(self._load_model)

        # Initial button states
        self.save_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Core Actions
    # ------------------------------------------------------------------
    def _choose_agent(self) -> None:
        dlg = AgentDialog(self, current_agent=self.agent_name)
        if dlg.exec():
            self.nn_locked = False
            agent_name = dlg.get_selection()
            self.agent_name = agent_name
            AgentClass = AGENTS[self.agent_name]
            self.hyperparams = AgentClass.get_default_hyperparams()
            self.agent_btn.setText(f"Agent:{agent_name}")
            self.agent_config_btn.setText(f"Configure {self.agent_name}")
            self._log(f"✅ Selected {self.agent_name} agent")

    def _start_training(self) -> None:
        # Clear plots
        self.reward_plot.reset()
        self.loss_plot.reset()

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
        self.controller.save_model(user_dir, self.reward_plot, self.loss_plot)

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
        dlg = NNConfigDialog(self, read_only=self.training_active, lock_hidden_layers=getattr(self, "nn_locked", False))
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

    def _on_progress(self, ep: int, episodes: int, ep_reward: float, rewards: list, avg_loss: float) -> None:
        avg20 = sum(rewards[-20:]) / min(len(rewards), 20)
        global_avg = sum(rewards) / len(rewards)
        self._log(
            f"Ep {ep+1}/{episodes} — R {ep_reward:.1f}, Avg20 {avg20:.1f}, Global {global_avg:.1f}, AvgLoss {avg_loss:.2f}"
        )
        self.reward_plot.update_plot(rewards, episodes)
        self.loss_plot.add_point(avg_loss)

    def _on_finished(self) -> None:
        self._set_training_buttons(True)

    def _update_status(self, message: str) -> None:
        self._log(message)

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
        self._log(f"🖥️ Device set to {config.DEVICE}")

    def _set_training_buttons(self, enable: bool) -> None:
        """Toggle interactive buttons based on training state."""
        self.training_active = not enable  # True while training
        toggled_widgets = [
            self.device_label, self.device_box, self.agent_btn, self.train_btn, self.save_btn, self.load_btn
        ]
        
        # self.agent_config_btn, , self.env_config_btn, self.nn_btn are always enable

        for w in toggled_widgets:
            w.setEnabled(enable)
        self.stop_btn.setEnabled(not enable)

    def _log(self, message: str) -> None:
        """Append text to the console output."""
        self.console_output.append(message)
        self.console_output.moveCursor(QTextCursor.End)
        
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

            # --- Update Agent ---
            self.agent_name = checkpoint.get("agent_name", self.agent_name)
            self.agent_btn.setText(f"Agent: {self.agent_name}")
            AgentClass = AGENTS.get(self.agent_name)
            if AgentClass:
                self.hyperparams = checkpoint.get("hyperparams", AgentClass.get_default_hyperparams())
            self.agent_config_btn.setText(f"{self.agent_name} Configuration")

            # --- Update Environment Config ---
            config.ENV_NAME = checkpoint.get("environment", config.ENV_NAME)
            config.MAX_STEPS = checkpoint.get("max_steps", config.MAX_STEPS)
            config.EPISODES = checkpoint.get("episodes_total", config.EPISODES)
            config.RENDER_MODE = checkpoint.get("render_mode", config.RENDER_MODE)

            # --- Update Neural Network Config ---
            nn_cfg = checkpoint.get("nn_config", {})
            config.HIDDEN_LAYERS = nn_cfg.get("hidden_layers", config.HIDDEN_LAYERS)
            config.LR = nn_cfg.get("lr", config.LR)
            config.ACTIVATION = nn_cfg.get("activation", config.ACTIVATION)
            config.DROPOUT = nn_cfg.get("dropout", config.DROPOUT)
            device_str = nn_cfg.get("device", str(config.DEVICE))
            config.DEVICE = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
            # Ensure NN config values persist in runtime
            config.HIDDEN_ACTIVATION = nn_cfg.get("activation", config.ACTIVATION)
            config.OPTIMIZER = nn_cfg.get("optimizer", getattr(config, "OPTIMIZER", "adam"))

            self.nn_locked = True

            # Log sanity check
            self._log(f"✅ NN Config loaded: LR={config.LR}, Dropout={config.DROPOUT}, Act={config.ACTIVATION}")

            # --- Logging and Confirmation ---
            env_name = config.ENV_NAME
            self._log(f"📦 Model loaded: {self.agent_name} in {env_name}")
            self._log(f"→ Hyperparameters: {len(self.hyperparams)} params")
            self._log(f"→ NN: layers={config.HIDDEN_LAYERS}, lr={config.LR}, dropout={config.DROPOUT}, device={config.DEVICE}")
            self._log(f"→ Episodes trained: {checkpoint.get('episodes_trained', 'N/A')}")
