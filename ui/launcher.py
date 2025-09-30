import torch
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QSpinBox, QFileDialog, QStackedWidget
)
import config
from agents.nstep_dqn_agent import NStepDeepQLearningAgent
from agents.nstep_ddqn_agent import NStepDoubleDeepQLearningAgent
from utils.rendering import render_agent
from PySide6.QtCore import QThread
from ui.agent_dialog import AgentDialog
from ui.hyperparams_dialog import HyperparamsDialog
from ui.reward_plot import RewardPlot
from ui.settings_dialog import SettingsDialog
from ui.training_worker import TrainingWorker
from ui.test_model_dialog import TestModelDialog
from ui.env_viewer import EnvViewer
from environments.factory import create_environment
import datetime


class CartPoleLauncher(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CartPole RL Launcher")
        self.resize(*config.RESOLUTION)

        # === Stack of pages ===
        self.stack = QStackedWidget()
        layout = QVBoxLayout(self)
        layout.addWidget(self.stack)

        # Create 3 pages
        self.main_page = QWidget()
        self.train_page = QWidget()
        self.test_page = QWidget()

        self.stack.addWidget(self.main_page)
        self.stack.addWidget(self.train_page)
        self.stack.addWidget(self.test_page)

        # Build content
        self._build_main_page()
        self._build_train_page()
        self._build_test_page()

        # Default page
        self.stack.setCurrentWidget(self.main_page)

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
    def _build_main_page(self):
        layout = QVBoxLayout(self.main_page)

        # === Column of big menu buttons ===
        self.train_page_btn = QPushButton("▶ Train Agent")
        self.test_page_btn = QPushButton("🎮 Test Agent")
        self.settings_btn = QPushButton("⚙ Settings")

        # Make buttons taller/wider like a game menu
        for btn in (self.train_page_btn, self.test_page_btn, self.settings_btn):
            btn.setMinimumHeight(50)
            btn.setStyleSheet("font-size: 18px;")  # bigger font
            layout.addWidget(btn)

        self.train_page_btn.clicked.connect(lambda: self.stack.setCurrentWidget(self.train_page))
        self.test_page_btn.clicked.connect(lambda: self.stack.setCurrentWidget(self.test_page))
        self.settings_btn.clicked.connect(self.open_settings)

    def _build_train_page(self):
        layout = QVBoxLayout(self.train_page)

        # Plot
        self.plot = RewardPlot()
        layout.addWidget(self.plot)

        layout.addWidget(QLabel("Environment:"))
        self.env_box = QComboBox()
        self.env_box.addItems(config.AVAILABLE_ENVIRONMENTS)
        self.env_box.setCurrentText(config.DEFAULT_ENVIRONMENT)
        layout.addWidget(self.env_box)

        layout.addWidget(QLabel("Agent:"))
        self.agent_btn = QPushButton("Choose Agent")
        self.train_btn = QPushButton("Start Training")
        self.stop_btn = QPushButton("Stop Training")
        self.save_btn = QPushButton("Save Model")
        row = QHBoxLayout()
        row.addWidget(self.agent_btn)
        row.addWidget(self.train_btn)
        row.addWidget(self.stop_btn)
        row.addWidget(self.save_btn)
        layout.addLayout(row)

        layout.addWidget(QLabel("Rendering Mode:"))
        self.render_box = QComboBox()
        self.render_box.addItems(["off", "human", "gif", "mp4"])
        layout.addWidget(self.render_box)

        layout.addWidget(QLabel("Training Episodes:"))
        self.episodes_box = QSpinBox()
        self.episodes_box.setRange(100, 10000)
        self.episodes_box.setValue(config.EPISODES)
        layout.addWidget(self.episodes_box)

        self.status_label = QLabel("Idle")
        layout.addWidget(self.status_label)

        # Back to main menu
        back_btn = QPushButton("⬅ Back to Main Menu")
        back_btn.setMinimumHeight(40)
        back_btn.setStyleSheet("font-size: 16px;")
        back_btn.clicked.connect(lambda: self.stack.setCurrentWidget(self.main_page))
        back_btn.clicked.connect(self.stop_viewer_and_back)
        layout.addWidget(back_btn)

        # Connects
        self.agent_btn.clicked.connect(self.choose_agent)
        self.train_btn.clicked.connect(self.start_training)
        self.stop_btn.clicked.connect(self.stop_training)
        self.save_btn.clicked.connect(self.save_agent_as)

    def _build_test_page(self):
        layout = QVBoxLayout(self.test_page)

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
        back_btn.clicked.connect(lambda: self.stack.setCurrentWidget(self.main_page))

    def open_hyperparams(self):
        dlg = HyperparamsDialog(self, defaults=self.hyperparams)
        if dlg.exec():
            self.hyperparams = dlg.get_params()

    def start_training(self):
        if self.training_thread and self.training_thread.isRunning():
            self.status_label.setText("⚠ Training is already running!")
            return

        if self.agent_name is None:
            self.status_label.setText("⚠ Please select an agent to train first by pressing the Choose Agent button")
            return

        agent_name = self.agent_name
        render = self.render_box.currentText()
        episodes = self.episodes_box.value()

        env_name = self.env_box.currentText()
        env, state_dim, action_dim = create_environment(env_name, render)

        params = self.hyperparams
        if agent_name == "nstep_dqn":
            agent = NStepDeepQLearningAgent(state_dim, action_dim, **params)
        else:
            agent = NStepDoubleDeepQLearningAgent(state_dim, action_dim, **params)

        # timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_path = f"{config.TRAINED_MODELS_FOLDER}/{env_name}_{agent_name}.pth"

        # === Create Worker & Thread ===
        self.training_thread = QThread()
        self.training_worker = TrainingWorker(env_name, env, agent_name, agent, episodes, 
                                              model_path, hyperparams=self.hyperparams, render=(render == "human"))
        self.training_worker.moveToThread(self.training_thread)

        # Connect signals
        self.training_thread.started.connect(self.training_worker.run)
        self.training_worker.progress.connect(self._on_progress)
        self.training_worker.finished.connect(self._on_finished)

        # Cleanup
        self.training_worker.finished.connect(self.training_thread.quit)
        self.training_worker.finished.connect(self.training_worker.deleteLater)
        self.training_thread.finished.connect(self.training_thread.deleteLater)
        self.training_thread.finished.connect(self._reset_training_refs)

        # Start training
        self.training_thread.start()
        self.status_label.setText("🚀 Training started...")

    def _on_progress(self, ep, episodes, ep_reward, rewards):
        avg20 = sum(rewards[-20:]) / min(len(rewards), 20)
        global_avg = sum(rewards) / len(rewards)
        self.status_label.setText(
            f"Ep {ep+1}/{episodes} — R {ep_reward:.1f}, Avg20 {avg20:.1f}, Global {global_avg:.1f}"
        )
        self.plot.update_plot(rewards, episodes)

    def _on_finished(self, rewards, checkpoint):
        self.status_label.setText("✅ Training finished!")
        self.last_checkpoint = checkpoint

        # FIXME
        self.last_checkpoint['hyperparams'] = self.hyperparams
        self.save_btn.setEnabled(True)
        
    def test_model(self):
        dlg = TestModelDialog("trained_models")
        if dlg.exec():
            model_file = dlg.get_selected()

            if not model_file:
                return

            # Model's data
            checkpoint = torch.load(model_file, map_location=config.DEVICE)
            agent_name = self.agent_name
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
                self.test_page.layout().removeWidget(self.viewer)
                self.viewer.deleteLater()
                self.viewer = None
                
            # Use embedded EnvViewer for rendering in GUI
            self.viewer = EnvViewer(env, ag, episodes=5, fps=30)
            self.test_page.layout().insertWidget(1, self.viewer)  # put under env_label
            self.viewer.start()
            

    def closeEvent(self, event):
        if self.training_worker:
            self.training_worker.stop()
        event.accept()

    def stop_training(self):
        if self.training_worker:
            self.training_worker.stop()
            self.status_label.setText("⏹ Training stopped by user")
        else:
            self.status_label.setText("⚠ No training is running")

    def _reset_training_refs(self):
        self.training_thread = None
        self.training_worker = None

    def choose_agent(self):
        dlg = AgentDialog(self, defaults=self.hyperparams)
        if dlg.exec():
            agent, hps = dlg.get_selection()
            self.agent_name = agent
            self.hyperparams = hps
            self.agent_btn.setText(agent)  # show chosen agent

    def save_agent_as(self):
        if not hasattr(self, "last_checkpoint"):
            self.status_label.setText("⚠ No trained agent to save")
            return

        # Suggest default filename with timestamp
        default_name = f"{self.last_checkpoint['environment']}_{self.last_checkpoint['agent_name']}.pth"
        path, _ = QFileDialog.getSaveFileName(self, "Save Agent", f"trained_models/{default_name}", "Model Files (*.pth)")

        if path:
            torch.save(self.last_checkpoint, path)
            self.status_label.setText(f"✅ Agent saved as {path}")

    def open_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec():
            values = dlg.get_values()
            config.save_user_config(values)
            # apply changes immediately
            if "RESOLUTION" in values:
                self.resize(*values["RESOLUTION"])
            if "EPISODES" in values:
                self.episodes_box.setValue(values["EPISODES"])
                
    def open_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec():
            values = dlg.get_values()
            config.save_user_config(values)
            if "RESOLUTION" in values:
                self.resize(*values["RESOLUTION"])
            if "EPISODES" in values:
                self.episodes_box.setValue(values["EPISODES"])

    def stop_viewer_and_back(self):
        if hasattr(self, "viewer") and self.viewer:
            self.viewer.stop()
        self.stack.setCurrentWidget(self.main_page)
        self.stop_testing_model()

    def stop_testing_model(self):
        if hasattr(self, "viewer") and self.viewer:
            self.viewer.stop()
            self.env_label.setText("Environment: (stopped)")

    def choose_model(self):
        dlg = TestModelDialog("trained_models")
        if dlg.exec():
            model_file = dlg.get_selected()
            if model_file:
                self.selected_model_file = model_file
                checkpoint = torch.load(model_file, map_location=config.DEVICE)
                env_name = checkpoint.get("environment", "N/A")
                self.env_label.setText(f"Environment: {env_name}")
                self.selected_checkpoint = checkpoint
            else:
                self.selected_model_file = None
                self.selected_checkpoint = None
                self.env_label.setText("Environment: (not loaded)")
                
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
            self.test_page.layout().removeWidget(self.viewer)
            self.viewer.deleteLater()
            self.viewer = None

        # Add new viewer
        self.viewer = EnvViewer(env, ag, episodes=5, fps=30)
        self.test_page.layout().insertWidget(1, self.viewer)  # right under env_label
        self.viewer.start()