from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout,
    QComboBox, QSpinBox, QFileDialog, QTabWidget
)
from PySide6.QtCore import Signal
from ui.agent_dialog import AgentDialog
from ui.agent_details_dialog import AgentDetailsDialog
from ui.reward_plot import RewardPlot
from ui.loss_plot import LossPlot
from ui.nn_config_dialog import NNConfigDialog
from utils.agent_factory import AGENTS
from ui.training_controller import TrainingController
import config
import os


class TrainingSection(QWidget):
    """UI layer for managing training interactions and visualization."""
    back_to_main = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # Core logic controller
        self.controller = TrainingController()
        self.agent_name = config.DEFAULT_AGENT
        AgentClass = AGENTS[self.agent_name]
        self.hyperparams = AgentClass.get_default_hyperparams()

        # Signal connections
        self.controller.progress.connect(self._on_progress)
        self.controller.finished.connect(self._on_finished)
        self.controller.status.connect(self._update_status)

        # Build the interface
        self._build_training_section()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------
    def _build_training_section(self):
        layout = QVBoxLayout(self)

        # --- Tabs for plots ---
        self.tabs = QTabWidget()
        self.reward_plot = RewardPlot()
        self.loss_plot = LossPlot()
        self.tabs.addTab(self.reward_plot, "Training Curve")
        self.tabs.addTab(self.loss_plot, "Loss Curve")
        self.tabs.setMinimumHeight(300)
        self.tabs.setMaximumHeight(310)
        layout.addWidget(self.tabs)

        # --- Environment selection ---
        layout.addWidget(QLabel("Environment:"))
        self.env_box = QComboBox()
        self.env_box.addItems(config.AVAILABLE_ENVIRONMENTS)
        self.env_box.setCurrentText(config.DEFAULT_ENVIRONMENT)
        layout.addWidget(self.env_box)

        # --- Agent row ---
        layout.addWidget(QLabel("Agent:"))
        self.agent_btn = QPushButton(f"{self.agent_name}")
        self.train_btn = QPushButton("Start Training")
        self.stop_btn = QPushButton("Stop Training")
        self.save_btn = QPushButton("Save Model")

        row = QHBoxLayout()
        for btn in [self.agent_btn, self.train_btn, self.stop_btn, self.save_btn]:
            row.addWidget(btn)
        layout.addLayout(row)

        # --- Configuration Row ---
        config_row = QHBoxLayout()
        self.details_btn = QPushButton(f"{self.agent_name} configuration")
        self.details_btn.clicked.connect(self._show_agent_details)
        self.nn_btn = QPushButton("NN configuration")
        self.nn_btn.clicked.connect(self._show_nn_config)

        for w in [self.details_btn, self.nn_btn]:
            w.setMinimumWidth(150)
            config_row.addWidget(w)

        # --- Device Selector ---
        self.device_label = QLabel("Device:")
        self.device_label.setStyleSheet("font-weight:bold; margin-left:10px;")
        self.device_box = QComboBox()
        self.device_box.addItems(["cpu", "cuda"])
        self._init_device_box()
        self.device_box.currentTextChanged.connect(self._on_device_changed)
        config_row.addWidget(self.device_label)
        config_row.addWidget(self.device_box)
        layout.addLayout(config_row)

        # --- Rendering mode & episodes ---
        layout.addWidget(QLabel("Rendering Mode:"))
        self.render_box = QComboBox()
        self.render_box.addItems(["off", "human"])
        layout.addWidget(self.render_box)

        layout.addWidget(QLabel("Training Episodes:"))
        self.episodes_box = QSpinBox()
        self.episodes_box.setRange(100, 10000)
        self.episodes_box.setValue(config.EPISODES)
        layout.addWidget(self.episodes_box)

        # --- Status label ---
        self.status_label = QLabel("Idle")
        layout.addWidget(self.status_label)

        # --- Back button ---
        back_btn = QPushButton("⬅ Back to Main Menu")
        back_btn.setMinimumHeight(40)
        back_btn.setStyleSheet("font-size: 16px;")
        back_btn.clicked.connect(self.back_to_main.emit)
        layout.addWidget(back_btn)

        # --- Button Connections ---
        self.agent_btn.clicked.connect(self._choose_agent)
        self.train_btn.clicked.connect(self._start_training)
        self.stop_btn.clicked.connect(self._stop_training)
        self.save_btn.clicked.connect(self._save_agent_as)

        # Initial button state
        self.save_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Button Handlers
    # ------------------------------------------------------------------
    def _choose_agent(self):
        dlg = AgentDialog(self, current_agent=self.agent_name)
        if dlg.exec():
            agent_name = dlg.get_selection()
            self.agent_name = agent_name
            AgentClass = AGENTS[self.agent_name]
            self.hyperparams = AgentClass.get_default_hyperparams()
            self.agent_btn.setText(agent_name)
            self.details_btn.setText(f"Configure {self.agent_name}")
            self.status_label.setText(f"✅ Selected {self.agent_name} agent")

    def _start_training(self):
        self._set_training_buttons_availability(False)
        env = self.env_box.currentText()
        episodes = self.episodes_box.value()
        render = self.render_box.currentText()
        self.controller.start_training(env, self.agent_name, self.hyperparams, episodes, render)

    def _stop_training(self):
        self.controller.stop_training()
        self._set_training_buttons_availability(True)

    def _save_agent_as(self):
        default_dir = os.path.join(config.TRAINED_MODELS_FOLDER)
        user_dir, _ = QFileDialog.getSaveFileName(self, "Save Agent As", default_dir)
        if not user_dir:
            self.status_label.setText("💡 Save canceled by user.")
            return
        self.controller.save_model(user_dir, self.reward_plot, self.loss_plot)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _on_progress(self, ep, episodes, ep_reward, rewards, avg_loss):
        avg20 = sum(rewards[-20:]) / min(len(rewards), 20)
        global_avg = sum(rewards) / len(rewards)
        self.status_label.setText(
            f"Ep {ep+1}/{episodes} — R {ep_reward:.1f}, Avg20 {avg20:.1f}, Global {global_avg:.1f}, AvgLoss {avg_loss:.2f}"
        )
        self.reward_plot.update_plot(rewards, episodes)
        self.loss_plot.add_point(avg_loss)

    def _on_finished(self):
        self._set_training_buttons_availability(True)

    def _update_status(self, message: str):
        self.status_label.setText(message)

    def _show_agent_details(self):
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

    def _on_device_changed(self, choice: str):
        import torch
        config.DEVICE = torch.device(choice if choice == "cpu" or torch.cuda.is_available() else "cpu")
        color = "#3a7" if "cuda" in str(config.DEVICE) else "#666"
        self.device_label.setStyleSheet(f"font-weight:bold; color:{color}; margin-left:10px;")
        self.status_label.setText(f"🖥️ Device set to {config.DEVICE}")

    def _init_device_box(self):
        gpu_available = config.torch.cuda.is_available()
        model = self.device_box.model()
        cuda_index = self.device_box.findText("cuda")
        if not gpu_available and cuda_index >= 0:
            item = model.item(cuda_index)
            item.setEnabled(False)
        default_device = "cuda" if gpu_available else "cpu"
        self.device_box.setCurrentText(default_device)
        config.DEVICE = config.torch.device(default_device)

    def _set_training_buttons_availability(self, enable=True):
        self.agent_btn.setEnabled(enable)
        self.train_btn.setEnabled(enable)
        self.env_box.setEnabled(enable)
        self.device_box.setEnabled(enable)
        self.render_box.setEnabled(enable)
        self.episodes_box.setEnabled(enable)
        self.save_btn.setEnabled(enable)
        self.stop_btn.setEnabled(not enable)
