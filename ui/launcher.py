import torch
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QStackedWidget
from agents.nstep_dqn_agent import NStepDeepQLearningAgent
from agents.nstep_ddqn_agent import NStepDoubleDeepQLearningAgent
from ui.hyperparams_dialog import HyperparamsDialog
from ui.settings_dialog import SettingsDialog
from ui.test_model_dialog import TestModelDialog
from ui.env_viewer import EnvViewer
from ui.training_section import TrainingSection
from environments.factory import create_environment
import config
import datetime


class CartPoleLauncher(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CartPole RL Launcher")
        self.resize(*config.RESOLUTION)

        # === Stack of pages ===
        self.stack = QStackedWidget()

        # Training section
        self.training_section = TrainingSection()
        self.stack.addWidget(self.training_section)
        # Connect signal
        self.training_section.back_to_main.connect(
            lambda: self.stack.setCurrentWidget(self.main_section)
        )

        layout = QVBoxLayout(self)
        layout.addWidget(self.stack)

        # Create Main and Testing sections
        self.main_section = QWidget()
        self.testing_section = QWidget()

        self.stack.addWidget(self.main_section)
        self.stack.addWidget(self.testing_section)

        # Build content
        self._build_main_section()
        self._build_testing_section()

        # Default page
        self.stack.setCurrentWidget(self.main_section)

        # State
        self.training_thread = None
        self.training_worker = None
        self.agent_name = None
        self.hyperparams = {
            "gamma": config.GAMMA, 
            "lr": config.LR,
            "buffer_size": config.BUFFER_SIZE, 
            "batch_size": config.BATCH_SIZE,
            "n_step": config.N_STEP, "eps_start": 
            config.EPSILON_START,
            "eps_end": config.EPSILON_END, 
            "eps_decay": config.EPSILON_DECAY,
        }

    # === Pages ===
    def _build_main_section(self):
        layout = QVBoxLayout(self.main_section)

        # === Column of big menu buttons ===
        self.training_section_btn = QPushButton("▶ Train Agent")
        self.testing_section_btn = QPushButton("🎮 Test Agent")
        self.settings_btn = QPushButton("⚙ Settings")

        # Make buttons taller/wider like a game menu
        for btn in (self.training_section_btn, self.testing_section_btn, self.settings_btn):
            btn.setMinimumHeight(50)
            btn.setStyleSheet("font-size: 18px;")  # bigger font
            layout.addWidget(btn)

        self.training_section_btn.clicked.connect(lambda: self.stack.setCurrentWidget(self.training_section))
        self.testing_section_btn.clicked.connect(lambda: self.stack.setCurrentWidget(self.testing_section))
        self.settings_btn.clicked.connect(self.open_settings)

    def _build_testing_section(self):
        layout = QVBoxLayout(self.testing_section)

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

        self.choose_model_btn.clicked.connect(self.choose_model)
        self.start_testing_btn.clicked.connect(self.start_testing_model)
        self.stop_testing_btn.clicked.connect(self.stop_testing_model)

        # Back to main menu
        back_btn = QPushButton("⬅ Back to Main Menu")
        back_btn.setMinimumHeight(40)
        back_btn.setStyleSheet("font-size: 16px;")
        layout.addWidget(back_btn)
        back_btn.clicked.connect(lambda: self.stack.setCurrentWidget(self.main_section))

    def open_hyperparams(self):
        dlg = HyperparamsDialog(self, defaults=self.hyperparams)
        if dlg.exec():
            self.hyperparams = dlg.get_params()
        
    def test_model(self):
        dlg = TestModelDialog("trained_models")
        if dlg.exec():
            model_file = dlg.get_selected()

            if not model_file:
                return

            # Model's data
            checkpoint = torch.load(model_file, map_location=config.DEVICE)
            agent_name = checkpoint.get("agent_name", "nstep_dqn")
            env_name = checkpoint.get("environment")
            self.env_label.setText(f"Environment: {env_name}")

            # Create environment with rgb_array mode
            env, state_dim, action_dim = create_environment(env_name, render='rgb_array')

            # Build Agent
            if agent_name == "nstep_dqn":
                ag = NStepDeepQLearningAgent(state_dim, action_dim, **self.hyperparams)
            else:
                ag = NStepDoubleDeepQLearningAgent(state_dim, action_dim, **self.hyperparams)

            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                ag.q_net.load_state_dict(checkpoint["model_state"])
            else:  # legacy
                ag.q_net.load_state_dict(checkpoint)

            ag.q_net.eval()

            # Remove old environment if it exists
            if hasattr(self, "viewer") and self.viewer:
                self.testing_section.layout().removeWidget(self.viewer)
                self.viewer.deleteLater()
                self.viewer = None
                
            # Use embedded EnvViewer for rendering in GUI
            self.viewer = EnvViewer(env, ag, episodes=5, fps=30)
            self.testing_section.layout().insertWidget(1, self.viewer)  # put under env_label
            self.viewer.start()

    def closeEvent(self, event):
        if self.training_worker:
            self.training_worker.stop()
        event.accept()

    def open_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec():
            values = dlg.get_values()
            config.save_user_config(values)
            if "RESOLUTION" in values:
                self.resize(*values["RESOLUTION"])
            if "EPISODES" in values:
                self.episodes_box.setValue(values["EPISODES"])

    def stop_testing_model(self):
        if hasattr(self, "viewer") and self.viewer:
            self.viewer.stop()
            self.env_label.setText("Environment: (stopped)")

    def choose_model(self):
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
                
    def start_testing_model(self):
        if not hasattr(self, "selected_checkpoint") or self.selected_checkpoint is None:
            self.env_label.setText("⚠ Please choose a model first")
            return

        checkpoint = self.selected_checkpoint
        env_name = checkpoint.get("environment", config.DEFAULT_ENVIRONMENT)
        agent_name = checkpoint.get("agent_name", "nstep_dqn")
        hps = checkpoint.get("hyperparams", self.hyperparams)

        env, state_dim, action_dim = create_environment(env_name, render="rgb_array")

        if agent_name == "nstep_dqn":
            ag = NStepDeepQLearningAgent(state_dim, action_dim, **hps)
        else:
            ag = NStepDoubleDeepQLearningAgent(state_dim, action_dim, **hps)

        if "model_state" in checkpoint:
            ag.q_net.load_state_dict(checkpoint["model_state"])
        ag.q_net.eval()
        if hasattr(ag, "epsilon"):
            ag.epsilon = 0.0

        # Remove old viewer
        if hasattr(self, "viewer") and self.viewer:
            self.testing_section.layout().removeWidget(self.viewer)
            self.viewer.deleteLater()
            self.viewer = None

        # Add new viewer
        self.viewer = EnvViewer(env, ag, episodes=5, fps=30)
        self.testing_section.layout().insertWidget(1, self.viewer)  # right under env_label
        self.viewer.start()
        