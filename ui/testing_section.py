import torch
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from PySide6.QtCore import Signal
from ui.test_model_dialog import TestModelDialog
from ui.env_viewer import EnvViewer
from utils.agent_factory import build_agent
from environments.factory import create_environment
import config


class TestingSection(QWidget):
    back_to_main = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        self.root_layout = QVBoxLayout(self)

        # --- Top row (info + environment) ---
        self.top_row = QHBoxLayout()
        self.root_layout.addLayout(self.top_row)

        # Left panel: model info
        self.env_label = QLabel("Environment: (not loaded)")
        self.env_label.setWordWrap(True)
        self.env_label.setMinimumWidth(400)
        self.top_row.addWidget(self.env_label, 1)  # stretch 1

        # Right panel placeholder for environment viewer
        self.viewer_container = QWidget()
        self.viewer_layout = QVBoxLayout(self.viewer_container)
        self.viewer_layout.setContentsMargins(0, 0, 0, 0)
        self.top_row.addWidget(self.viewer_container, 2)  # stretch 2 for more space

        # --- Buttons row ---
        button_row = QHBoxLayout()
        self.choose_model_btn = QPushButton("Choose Model")
        self.start_testing_btn = QPushButton("Start Testing")
        self.stop_testing_btn = QPushButton("Stop Testing")

        button_row.addWidget(self.choose_model_btn)
        button_row.addWidget(self.start_testing_btn)
        button_row.addWidget(self.stop_testing_btn)
        self.root_layout.addLayout(button_row)

        # --- Back button ---
        back_btn = QPushButton("⬅ Back to Main Menu")
        back_btn.setMinimumHeight(40)
        back_btn.setStyleSheet("font-size: 16px;")
        back_btn.clicked.connect(self.back_to_main.emit)
        self.root_layout.addWidget(back_btn)

        # --- Signals ---
        self.choose_model_btn.clicked.connect(self._choose_model)
        self.start_testing_btn.clicked.connect(self._start_testing_model)
        self.stop_testing_btn.clicked.connect(self._stop_testing_model)

        # Visual border between info and the viewer
        self.env_label.setStyleSheet("""
            background-color: #f6f6f6;
            border: 1px solid #ccc;
            padding: 8px;
        """)

        self.viewer_container.setStyleSheet("""
            background-color: #ffffff;
            border: 1px solid #ddd;
        """)

        # Internal viewer ref
        self.viewer = None

    def _choose_model(self):
        dlg = TestModelDialog("trained_models")
        if dlg.exec():
            model_file = dlg.get_selected()
            if model_file:
                self.choose_model_btn.setText(model_file.split("/")[-1])

                self.selected_model_file = model_file
                checkpoint = torch.load(model_file, map_location=config.DEVICE)

                agent_name = checkpoint.get("agent_name", "Unknown")
                env_name = checkpoint.get("environment", "Unknown")
                episodes_trained = checkpoint.get("episodes_trained", "N/A")
                episodes_total = checkpoint.get("episodes_total", "N/A")
                hps = checkpoint.get("hyperparams", {})
                timestamp = checkpoint.get("timestamp", "Unknown")

                # --- Hyperparameters (existing) ---
                hps_html = "<ul>" + "".join(
                    f"<li>{k}: {v}</li>" for k, v in sorted(hps.items())
                ) + "</ul>"

                # --- NN section (new) ---
                nn_cfg = checkpoint.get("nn_config", None)
                if nn_cfg:
                    # normalize keys / fallbacks
                    layers   = nn_cfg.get("hidden_layers", "n/a")
                    act      = nn_cfg.get("activation", "n/a")
                    drop     = nn_cfg.get("dropout", "n/a")
                    lr       = nn_cfg.get("lr", "n/a")
                    opt      = nn_cfg.get("optimizer", "n/a")
                    device   = nn_cfg.get("device", "n/a")

                    nn_html = (
                        "<ul>"
                        f"<li>hidden_layers: {layers}</li>"
                        f"<li>activation: {act}</li>"
                        f"<li>dropout: {drop}</li>"
                        f"<li>optimizer: {opt}</li>"
                        f"<li>learning_rate: {lr}</li>"
                        f"<li>device: {device}</li>"
                        "</ul>"
                    )
                else:
                    nn_html = "<i>not saved in this checkpoint</i>"

                # --- Compose info block with two sections ---
                info_html = (
                    f"<b>Environment:</b> {env_name}<br>"
                    f"<b>Agent:</b> {agent_name}<br>"
                    f"<b>Episodes:</b> {episodes_trained}/{episodes_total}<br><br>"
                    f"<b>Hyperparameters:</b>{hps_html}"
                    f"<b>Neural Network:</b> {nn_html if isinstance(nn_cfg, dict) else nn_html}<br>"
                    f"<b>Created:</b> {timestamp}<br>"
                )

                self.env_label.setText(info_html)
                self.selected_checkpoint = checkpoint
            else:
                self.selected_model_file = None
                self.selected_checkpoint = None
                self.env_label.setText("⚠ No model selected")
                self.choose_model_btn.setText("Choose Model")


    def _start_testing_model(self):
        if not hasattr(self, "selected_checkpoint") or self.selected_checkpoint is None:
            self.env_label.setText("⚠ Please choose a model first")
            return

        checkpoint = self.selected_checkpoint
        env_name = checkpoint.get("environment", config.DEFAULT_ENVIRONMENT)
        agent_name = checkpoint.get("agent_name", "nstep_dqn")
        hyperparams = checkpoint.get("hyperparams", {})

        env, state_dim, action_dim = create_environment(env_name, render="rgb_array")

        agent = build_agent(agent_name, state_dim, action_dim, hyperparams)
        agent.load(self.selected_model_file)

        # Remove old viewer if exists
        if self.viewer:
            self.viewer_layout.removeWidget(self.viewer)
            self.viewer.deleteLater()
            self.viewer = None

        # Create new viewer
        self.viewer = EnvViewer(env, agent, episodes=5, fps=30)
        self.viewer_layout.addWidget(self.viewer)
        self.viewer.start()

    def _stop_testing_model(self):
        if hasattr(self, "viewer") and self.viewer:
            self.viewer.stop()
            self.env_label.setText("Environment: (stopped)")
