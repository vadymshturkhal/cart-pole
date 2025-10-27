import torch
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from PySide6.QtCore import Signal
from ui.test_model_dialog import TestModelDialog
from ui.env_viewer import EnvViewer
from utils.agent_factory import build_agent
from environments.factory import create_environment
import config
from enum import Enum


class TestState(Enum):
    FINISHED = ("Testing ended for", "✅")
    STOPPED = ("Stopped", "⏹")
    

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

        # LEFT: model info + status stacked
        left_col = QVBoxLayout()
        self.info_label = QLabel("No model loaded.")
        self.info_label.setWordWrap(True)
        self.info_label.setMinimumWidth(400)

        self.status_label = QLabel("")  # ← ephemeral run status
        self.status_label.setStyleSheet("color:#555; margin-top:6px;")

        left_col.addWidget(self.info_label)
        left_col.addWidget(self.status_label)
        left_wrap = QWidget(); 
        left_wrap.setLayout(left_col)
        left_wrap.setStyleSheet("background:#f6f6f6; border:1px solid #ccc; padding:8px;")
        self.top_row.addWidget(left_wrap, 1)

        # RIGHT: viewer container
        self.viewer_container = QWidget()
        self.viewer_layout = QVBoxLayout(self.viewer_container)
        self.viewer_layout.setContentsMargins(0, 0, 0, 0)
        self.viewer_container.setStyleSheet("background:#fff; border:1px solid #ddd;")
        self.top_row.addWidget(self.viewer_container, 2)

        # Keep references
        self.viewer = None
        self.selected_model_file = None
        self.selected_checkpoint = None
        self._meta = {"env": None, "agent": None}

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

        self.start_testing_btn.setEnabled(True)
        self.stop_testing_btn.setEnabled(False)

        # Internal viewer ref
        self.viewer = None

    def _choose_model(self):
        dlg = TestModelDialog()
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

                self.info_label.setText(info_html)
                self.status_label.setText(f"Model {agent_name} loaded in {env_name} environment")
                self.selected_checkpoint = checkpoint
                self._meta["env"] = env_name
                self._meta["agent"] = agent_name

    def _start_testing_model(self):
        if not self.selected_checkpoint:
            self.status_label.setText("⚠ Please choose a model first.")
            return

        ckpt = self.selected_checkpoint
        env_name = ckpt.get("environment", config.DEFAULT_ENVIRONMENT)
        agent_name = ckpt.get("agent_name", "nstep_dqn")
        hps = ckpt.get("hyperparams", {})

        env, state_dim, action_dim = create_environment(env_name, render="rgb_array")
        agent = build_agent(agent_name, state_dim, action_dim, hps)
        agent.load(self.selected_model_file)

        # remove old viewer
        if self.viewer:
            self.viewer_layout.removeWidget(self.viewer)
            self.viewer.deleteLater()
            self.viewer = None

        # add viewer
        self.viewer = EnvViewer(env, agent, episodes=5, fps=30)
        self.viewer_layout.addWidget(self.viewer)
        self.viewer.start()

        # UI state + status
        self._meta["env"] = env_name
        self._meta["agent"] = agent_name
        self.status_label.setText(f"▶ Running {agent_name} in {env_name} environment")
        self.start_testing_btn.setEnabled(False)
        self.choose_model_btn.setEnabled(False)
        self.stop_testing_btn.setEnabled(True)

        # Connect callback to viewer 
        self.viewer.finished.connect(self._on_testing_finished)

    def _stop_testing_model(self):
        """Handle manual user stop during testing."""
        self._update_testing_state(TestState.STOPPED)

    def _on_testing_finished(self):
        """Handle natural completion of all test episodes."""
        self._update_testing_state(TestState.FINISHED)

    def _update_testing_state(self, state: TestState):
        if self.viewer:
            self.viewer.stop()

        msg, icon = state.value
        agent_name = self._meta.get("agent") or "agent"
        env_name = self._meta.get("env") or "env"
        self.status_label.setText(f"{icon} {msg} {agent_name} on {env_name}.")
        self.start_testing_btn.setEnabled(True)
        self.choose_model_btn.setEnabled(True)
        self.stop_testing_btn.setEnabled(False)
