import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QHBoxLayout, QFileDialog, QCheckBox
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import gymnasium as gym, torch, os
from agents.nstep_dqn_agent import NStepDeepQLearningAgent
from agents.nstep_ddqn_agent import NStepDoubleDeepQLearningAgent
from utils.training import train
from utils.plotting import plot_rewards
from utils.rendering import render_agent
import config


class RewardPlot(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(4, 2))
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Training Curve")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward (normalized)")
        self.line, = self.ax.plot([], [], color="blue")
        self.rewards = []

    def update_plot(self, rewards, episodes):
        if not rewards:
            return
        self.rewards = rewards
        xs = list(range(len(rewards)))
        ys = [(r - min(rewards)) / max(1, (max(rewards) - min(rewards))) for r in rewards]  # normalize
        self.line.set_data(xs, ys)
        self.ax.set_xlim(0, episodes)
        self.ax.set_ylim(0, 1.0)
        self.draw()


# === GUI Launcher ===
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

        # === Agent selection ===
        layout.addWidget(QLabel("Choose Agent:"))
        self.agent_box = QComboBox()
        self.agent_box.addItems(["nstep_dqn", "nstep_ddqn"])
        layout.addWidget(self.agent_box)

        # === Render mode selection ===
        layout.addWidget(QLabel("Rendering Mode:"))
        self.render_box = QComboBox()
        self.render_box.addItems(["off", "human", "gif", "mp4"])
        layout.addWidget(self.render_box)

        # === Episodes ===
        layout.addWidget(QLabel("Training Episodes:"))
        self.episodes_box = QSpinBox()
        self.episodes_box.setRange(100, 10000)
        self.episodes_box.setValue(config.EPISODES)
        layout.addWidget(self.episodes_box)

        # === Sutton & Barto reward ===
        self.sutton_cb = QCheckBox("Use Sutton & Barto reward")
        self.sutton_cb.setChecked(config.SUTTON_BARTO_REWARD)
        layout.addWidget(self.sutton_cb)

        # === Buttons ===
        btn_layout = QHBoxLayout()
        self.train_btn = QPushButton("Start Training")
        self.test_btn = QPushButton("Test Pre-trained Model")
        btn_layout.addWidget(self.train_btn)
        btn_layout.addWidget(self.test_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

        # === Connect actions ===
        self.train_btn.clicked.connect(self.start_training)
        self.test_btn.clicked.connect(self.test_model)

    def start_training(self):
        agent = self.agent_box.currentText()
        render = self.render_box.currentText()
        episodes = self.episodes_box.value()
        sutton = self.sutton_cb.isChecked()

        self.status_label.setText(f"Training {agent} for {episodes} episodes…")

        env = gym.make(config.ENV_NAME, sutton_barto_reward=sutton)
        state_dim, action_dim = env.observation_space.shape[0], env.action_space.n

        if agent == "nstep_dqn":
            from agents.nstep_dqn_agent import NStepDeepQLearningAgent
            ag = NStepDeepQLearningAgent(state_dim, action_dim)
        else:
            from agents.nstep_ddqn_agent import NStepDoubleDeepQLearningAgent
            ag = NStepDoubleDeepQLearningAgent(state_dim, action_dim)

        rewards = []

        def progress_cb(ep, eps, ep_reward, all_rewards):
            rewards[:] = all_rewards
            self.status_label.setText(f"Episode {ep+1}/{eps} — Reward {ep_reward:.1f}, Avg(20) {sum(all_rewards[-20:])/min(len(all_rewards),20):.1f}")
            self.plot.update_plot(all_rewards, eps)
            QApplication.processEvents()  # refresh GUI

        rewards = train(env, ag, episodes=episodes, progress_cb=progress_cb)

        os.makedirs(config.TRAINED_MODELS_FOLDER, exist_ok=True)
        torch.save(ag.q_net.state_dict(), f"{config.TRAINED_MODELS_FOLDER}/{agent}_qnet.pth")

        plot_rewards(from_file=False, rewards=rewards)
        env.close()
        self.status_label.setText("✅ Training finished!")

    def test_model(self):
        model_file, _ = QFileDialog.getOpenFileName(self, "Select Pre-trained Model", "trained_models", "Model Files (*.pth)")
        if not model_file:
            return

        agent = self.agent_box.currentText()
        render = self.render_box.currentText()
        env = gym.make(config.ENV_NAME, render_mode="rgb_array" if render in ["gif","mp4"] else "human")
        state_dim, action_dim = env.observation_space.shape[0], env.action_space.n

        if agent == "nstep_dqn":
            ag = NStepDeepQLearningAgent(state_dim, action_dim)
        else:
            ag = NStepDoubleDeepQLearningAgent(state_dim, action_dim)

        ag.q_net.load_state_dict(torch.load(model_file, map_location=config.DEVICE))
        ag.q_net.eval()

        render_agent(env, ag, mode=render, episodes=5)
        env.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CartPoleLauncher()
    window.show()
    sys.exit(app.exec())
