import torch
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QSpinBox, QHBoxLayout, QFileDialog
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
from environments.factory import create_environment
import datetime


class CartPoleLauncher(QWidget):
    def __init__(self):
        super().__init__()
        self.plot = RewardPlot()

        self.setWindowTitle("CartPole RL Launcher")
        self.resize(*config.RESOLUTION)

        self.training_thread = None
        self.training_worker = None 

        layout = QVBoxLayout()
        layout.addWidget(self.plot)

        # Environment row
        env_row = QHBoxLayout()
        env_row.addWidget(QLabel("Environment:"))

        self.env_box = QComboBox()
        self.env_box.addItems(config.AVAILABLE_ENVIRONMENTS)
        self.env_box.setCurrentText(config.DEFAULT_ENVIRONMENT)  # default from config
        env_row.addWidget(self.env_box)

        layout.addLayout(env_row)

        # === Agent row with training controls ===
        self.status_label = QLabel("Agent:")
        layout.addWidget(self.status_label)
        agent_row = QHBoxLayout()

        self.agent_btn = QPushButton("Choose Agent")
        self.agent_btn.clicked.connect(self.choose_agent)
        agent_row.addWidget(self.agent_btn)

        self.train_btn = QPushButton("Start Training")
        agent_row.addWidget(self.train_btn)

        self.stop_btn = QPushButton("Stop Training")
        agent_row.addWidget(self.stop_btn)

        layout.addLayout(agent_row)

        # === Test button row ===
        test_row = QHBoxLayout()
        self.test_btn = QPushButton("Test Pre-trained Model")
        test_row.addWidget(self.test_btn)
        layout.addLayout(test_row)

        self.save_btn = QPushButton("Save Agent As...")
        self.save_btn.setEnabled(False)  # disabled until training/test agent exists
        layout.addWidget(self.save_btn)

        self.save_btn.clicked.connect(self.save_agent_as)

        # Render mode
        layout.addWidget(QLabel("Rendering Mode:"))
        self.render_box = QComboBox(); 
        self.render_box.addItems(["off","human","gif","mp4"])
        layout.addWidget(self.render_box)

        # Episodes
        layout.addWidget(QLabel("Training Episodes:"))
        self.episodes_box = QSpinBox(); self.episodes_box.setRange(100, 10000); self.episodes_box.setValue(config.EPISODES)
        layout.addWidget(self.episodes_box)

        # Settings button
        settings_row = QHBoxLayout()
        self.settings_btn = QPushButton("Settings")
        settings_row.addWidget(self.settings_btn)
        layout.addLayout(settings_row)

        self.settings_btn.clicked.connect(self.open_settings)

        # Agent's name
        self.agent_name = None

        # Hyperparams defaults
        self.hyperparams = {
            "gamma": config.GAMMA, 
            "lr": config.LR,
            "buffer_size": config.BUFFER_SIZE, 
            "batch_size": config.BATCH_SIZE,
            "n_step": config.N_STEP, 
            "eps_start": config.EPSILON_START,
            "eps_end": config.EPSILON_END, 
            "eps_decay": config.EPSILON_DECAY,
        }

        self.setLayout(layout)

        self.train_btn.clicked.connect(self.start_training)
        self.stop_btn.clicked.connect(self.stop_training)
        self.test_btn.clicked.connect(self.test_model)

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

            checkpoint = torch.load(model_file, map_location=config.DEVICE)

            agent_name = self.agent_name

            env_name = checkpoint.get("environment")
            env, state_dim, action_dim = create_environment(env_name, render='human')

            if agent_name == "nstep_dqn":
                ag = NStepDeepQLearningAgent(state_dim, action_dim, **self.hyperparams)
            else:
                ag = NStepDoubleDeepQLearningAgent(state_dim, action_dim, **self.hyperparams)

            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                ag.q_net.load_state_dict(checkpoint["model_state"])
            else:  # legacy
                ag.q_net.load_state_dict(checkpoint)

            ag.q_net.eval()
            render_agent(env, ag, episodes=5)
            env.close()

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
                