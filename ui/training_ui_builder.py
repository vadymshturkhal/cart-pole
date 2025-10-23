from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTextEdit, QComboBox, QTabWidget, QSpinBox
)
from ui.reward_plot import RewardPlot
from ui.loss_plot import LossPlot
import torch
import config


class TrainingUIBuilder:
    """Builds and initializes all UI components for TrainingSection."""

    def __init__(self, parent):
        self.parent = parent
        self.layout = QVBoxLayout(parent)
        self._build_tabs()
        self._build_plot_interval_row()   # new row added
        self._build_agent_row()
        self._build_config_row()
        self._build_console()
        self._build_back_button()

    # ------------------------------------------------------------------
    # UI Components
    # ------------------------------------------------------------------
    def _build_tabs(self):
        self.tabs = QTabWidget()
        self.reward_plot = RewardPlot()
        self.loss_plot = LossPlot()
        self.tabs.addTab(self.reward_plot, "Training Curve")
        self.tabs.addTab(self.loss_plot, "Loss Curve")
        self.tabs.setMinimumHeight(300)
        self.tabs.setMaximumHeight(310)
        self.layout.addWidget(self.tabs)

    def _build_agent_row(self):
        row = QHBoxLayout()
        self.agent_btn = QPushButton()
        self.train_btn = QPushButton("Start Training")
        self.stop_btn = QPushButton("Stop Training")
        self.save_btn = QPushButton("Save Model")
        self.load_btn = QPushButton("Load Model")
        for w in [self.agent_btn, self.train_btn, self.stop_btn, self.save_btn, self.load_btn]:
            row.addWidget(w)
        self.layout.addLayout(row)

        # Default states
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(False)

    def _build_config_row(self):
        row = QHBoxLayout()

        self.agent_config_btn = QPushButton("Agent Configuration")
        self.env_config_btn = QPushButton("Environment Configuration")
        self.nn_btn = QPushButton("NN Configuration")

        self.device_label = QLabel("Device:")
        self.device_label.setStyleSheet("font-weight:bold; margin-left:10px;")
        self.device_box = QComboBox()
        self.device_box.addItems(["cpu", "cuda"])
        self._init_device_box()

        for w in [
            self.agent_config_btn, self.env_config_btn, self.nn_btn,
            self.device_label, self.device_box
        ]:
            row.addWidget(w)
        self.layout.addLayout(row)

    def _build_console(self):
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
        self.layout.addWidget(self.console_output)

    def _build_back_button(self):
        self.back_btn = QPushButton("⬅ Back to Main Menu")
        self.back_btn.setMinimumHeight(40)
        self.back_btn.setStyleSheet("font-size: 16px;")
        self.layout.addWidget(self.back_btn)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _init_device_box(self):
        gpu_available = torch.cuda.is_available()
        model = self.device_box.model()
        cuda_index = self.device_box.findText("cuda")
        if not gpu_available and cuda_index >= 0:
            model.item(cuda_index).setEnabled(False)
        default_device = "cuda" if gpu_available else "cpu"
        self.device_box.setCurrentText(default_device)
        config.DEVICE = torch.device(default_device)

    def _build_plot_interval_row(self):
        """Adds a GUI control for config.PLOT_UPDATE_INTERVAL."""
        row = QHBoxLayout()
        self.plot_interval_label = QLabel("Update plots every (episodes):")
        self.plot_interval_label.setStyleSheet("margin-left:10px; font-weight:bold;")
        self.plot_interval_box = QSpinBox()
        self.plot_interval_box.setRange(1, 500)
        self.plot_interval_box.setValue(getattr(config, "PLOT_UPDATE_INTERVAL", 5))
        self.plot_interval_box.setToolTip("Number of episodes between plot updates")
        row.addWidget(self.plot_interval_label)
        row.addWidget(self.plot_interval_box)
        self.layout.addLayout(row)
