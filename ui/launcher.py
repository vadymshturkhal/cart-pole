import os, torch, gymnasium as gym
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QSpinBox, QHBoxLayout, QFileDialog, QCheckBox, QApplication
import config
from ui.hyperparams_dialog import HyperparamsDialog
from ui.reward_plot import RewardPlot
from agents.nstep_dqn_agent import NStepDeepQLearningAgent
from agents.nstep_ddqn_agent import NStepDoubleDeepQLearningAgent
from utils.training import train
from utils.plotting import plot_rewards
from utils.rendering import render_agent

class CartPoleLauncher(QWidget):
    def __init__(self):
        super().__init__()
        self.plot = RewardPlot()

        self.setWindowTitle("CartPole RL Launcher")
        self.resize(config.WIDTH, config.HEIGHT)

        layout = QVBoxLayout()
        layout.addWidget(self.plot)
        self.status_label = QLabel("Idle")
        layout.addWidget(self.status_label)

        # === Agent row ===
        agent_row = QHBoxLayout()
        self.agent_box = QComboBox(); self.agent_box.addItems(["nstep_dqn", "nstep_ddqn"]); self.agent_box.setMaximumWidth(200)
        agent_row.addWidget(QLabel("Choose Agent:"))
        agent_row.addWidget(self.agent_box)

        self.hyper_btn = QPushButton("Hyperparameters")
        self.hyper_btn.clicked.connect(self.open_hyperparams)
        agent_row.addWidget(self.hyper_btn)
        layout.addLayout(agent_row)

        # Render mode
        layout.addWidget(QLabel("Rendering Mode:"))
        self.render_box = QComboBox(); self.render_box.addItems(["off","human","gif","mp4"])
        layout.addWidget(self.render_box)

        # Episodes
        layout.addWidget(QLabel("Training Episodes:"))
        self.episodes_box = QSpinBox(); self.episodes_box.setRange(100, 10000); self.episodes_box.setValue(config.EPISODES)
        layout.addWidget(self.episodes_box)

        # Hyperparams defaults
        self.hyperparams = {
            "gamma": config.GAMMA, "lr": config.LR,
            "buffer_size": config.BUFFER_SIZE, "batch_size": config.BATCH_SIZE,
            "n_step": config.N_STEP, "eps_start": config.EPSILON_START,
            "eps_end": config.EPSILON_END, "eps_decay": config.EPSILON_DECAY,
        }

        # Sutton reward
        self.sutton_cb = QCheckBox("Use Sutton & Barto reward")
        self.sutton_cb.setChecked(config.SUTTON_BARTO_REWARD)
        layout.addWidget(self.sutton_cb)

        # Buttons
        btn_row = QHBoxLayout()
        self.train_btn = QPushButton("Start Training")
        self.test_btn = QPushButton("Test Pre-trained Model")
        btn_row.addWidget(self.train_btn); btn_row.addWidget(self.test_btn)
        layout.addLayout(btn_row)

        self.setLayout(layout)

        self.train_btn.clicked.connect(self.start_training)
        self.test_btn.clicked.connect(self.test_model)

    def open_hyperparams(self):
        dlg = HyperparamsDialog(self, defaults=self.hyperparams)
        if dlg.exec():
            self.hyperparams = dlg.get_params()

    def start_training(self):
        agent_name = self.agent_box.currentText()
        render = self.render_box.currentText()
        episodes = self.episodes_box.value()
        sutton = self.sutton_cb.isChecked()

        env = gym.make(config.ENV_NAME, sutton_barto_reward=sutton)
        state_dim, action_dim = env.observation_space.shape[0], env.action_space.n

        params = self.hyperparams
        if agent_name == "nstep_dqn":
            ag = NStepDeepQLearningAgent(state_dim, action_dim, **params)
        else:
            ag = NStepDoubleDeepQLearningAgent(state_dim, action_dim, **params)

        def progress_cb(ep, eps, ep_reward, all_rewards):
            avg20 = sum(all_rewards[-20:]) / min(len(all_rewards), 20)
            global_avg = sum(all_rewards) / len(all_rewards)
            self.status_label.setText(f"Ep {ep+1}/{eps} — R {ep_reward:.1f}, Avg20 {avg20:.1f}, Global {global_avg:.1f}")
            self.plot.update_plot(all_rewards, eps)
            QApplication.processEvents()

        rewards = train(env, ag, episodes=episodes, progress_cb=progress_cb)
        os.makedirs(config.TRAINED_MODELS_FOLDER, exist_ok=True)
        torch.save(ag.q_net.state_dict(), f"{config.TRAINED_MODELS_FOLDER}/{agent_name}_qnet.pth")
        plot_rewards(from_file=False, rewards=rewards)
        env.close()
        self.status_label.setText("✅ Training finished!")

    def test_model(self):
        model_file, _ = QFileDialog.getOpenFileName(self, "Select Pre-trained Model", "trained_models", "Model Files (*.pth)")
        if not model_file: return
        agent_name = self.agent_box.currentText()
        render = self.render_box.currentText()
        env = gym.make(config.ENV_NAME, render_mode="rgb_array" if render in ["gif","mp4"] else "human")
        state_dim, action_dim = env.observation_space.shape[0], env.action_space.n
        if agent_name == "nstep_dqn": ag = NStepDeepQLearningAgent(state_dim, action_dim, **self.hyperparams)
        else: ag = NStepDoubleDeepQLearningAgent(state_dim, action_dim, **self.hyperparams)
        ag.q_net.load_state_dict(torch.load(model_file, map_location=config.DEVICE))
        ag.q_net.eval()
        render_agent(env, ag, mode=render, episodes=5)
        env.close()
