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
        layout = QVBoxLayout(self)

        # Show environment name
        self.env_label = QLabel("Environment: (not loaded)")
        layout.addWidget(self.env_label)
    
        # Buttons row
        row = QHBoxLayout()
        self.choose_model_btn = QPushButton("Choose Model")
        self.start_testing_btn = QPushButton("Start Testing")
        self.stop_testing_btn = QPushButton("Stop Testing")
        row.addWidget(self.choose_model_btn)
        row.addWidget(self.start_testing_btn)
        row.addWidget(self.stop_testing_btn)
        layout.addLayout(row)

        self.choose_model_btn.clicked.connect(self._choose_model)
        self.start_testing_btn.clicked.connect(self._start_testing_model)
        self.stop_testing_btn.clicked.connect(self._stop_testing_model)

        # Back button
        back_btn = QPushButton("⬅ Back to Main Menu")
        back_btn.setMinimumHeight(40)
        back_btn.setStyleSheet("font-size: 16px;")
        # Emit Signal
        back_btn.clicked.connect(self.back_to_main.emit)
        layout.addWidget(back_btn)

    def _choose_model(self):
        dlg = TestModelDialog("trained_models")
        if dlg.exec():
            model_file = dlg.get_selected()
            if model_file:
                # ✅ set button text to model name
                self.choose_model_btn.setText(model_file.split("/")[-1])  

                self.selected_model_file = model_file
                checkpoint = torch.load(model_file, map_location=config.DEVICE)

                agent_name = checkpoint.get("agent_name", "Unknown")
                env_name = checkpoint.get("environment", "Unknown")
                episodes_trained = checkpoint.get("episodes_trained", "N/A")
                episodes_total = checkpoint.get("episodes_total", "N/A")
                hps = checkpoint.get("hyperparams", {})
                hps_html = "<ul>" + "".join(f"<li>{k}: {v}</li>" for k, v in hps.items()) + "</ul>"

                info_html = (
                    f"<b>Environment:</b> {env_name}<br>"
                    f"<b>Agent:</b> {agent_name}<br>"
                    f"<b>Episodes:</b> {episodes_trained}/{episodes_total}<br>"
                    f"<b>Hyperparameters:</b>{hps_html}"
                )
                self.env_label.setText(info_html)

                self.selected_checkpoint = checkpoint
            else:
                self.selected_model_file = None
                self.selected_checkpoint = None
                self.env_label.setText("⚠ No model selected")
                # reset button text
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

        if "model_state" in checkpoint:
            agent.q_net.load_state_dict(checkpoint["model_state"])
        agent.q_net.eval()
        if hasattr(agent, "epsilon"):
            agent.epsilon = 0.0

        # Remove old viewer
        if hasattr(self, "viewer") and self.viewer:
            self.layout().removeWidget(self.viewer)
            self.viewer.deleteLater()
            self.viewer = None

        # Add new viewer
        self.viewer = EnvViewer(env, agent, episodes=5, fps=30)
        self.layout().insertWidget(1, self.viewer)  # right under env_label
        self.viewer.start()

    def _stop_testing_model(self):
        if hasattr(self, "viewer") and self.viewer:
            self.viewer.stop()
            self.env_label.setText("Environment: (stopped)")
