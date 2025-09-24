import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QHBoxLayout, QFileDialog, QCheckBox,
    QDoubleSpinBox, QSpinBox, QFormLayout
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


from PySide6.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QPushButton, QDoubleSpinBox, QSpinBox

class HyperparamsDialog(QDialog):
    def __init__(self, parent=None, defaults=None):
        super().__init__(parent)
        self.setWindowTitle("Adjust Hyperparameters")
        self.setMinimumWidth(300)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        # === Gamma ===
        self.gamma_box = QDoubleSpinBox()
        self.gamma_box.setDecimals(4)
        self.gamma_box.setValue(defaults.get("gamma", 0.99))
        form.addRow("Gamma:", self.gamma_box)

        # === Learning Rate ===
        self.lr_box = QDoubleSpinBox()
        self.lr_box.setDecimals(6)
        self.lr_box.setValue(defaults.get("lr", 1e-3))
        form.addRow("Learning Rate:", self.lr_box)

        # === Buffer Size ===
        self.buffer_box = QSpinBox()
        self.buffer_box.setMaximum(10**9)
        self.buffer_box.setValue(defaults.get("buffer_size", 10000))
        form.addRow("Buffer Size:", self.buffer_box)

        # === Batch Size ===
        self.batch_box = QSpinBox()
        self.batch_box.setMaximum(10**6)
        self.batch_box.setValue(defaults.get("batch_size", 64))
        form.addRow("Batch Size:", self.batch_box)

        # === N-step ===
        self.nstep_box = QSpinBox()
        self.nstep_box.setMaximum(100)
        self.nstep_box.setValue(defaults.get("n_step", 3))
        form.addRow("N-step:", self.nstep_box)

        # === Epsilon Start ===
        self.eps_start_box = QDoubleSpinBox()
        self.eps_start_box.setDecimals(3)
        self.eps_start_box.setValue(defaults.get("eps_start", 1.0))
        form.addRow("Epsilon Start:", self.eps_start_box)

        # === Epsilon End ===
        self.eps_end_box = QDoubleSpinBox()
        self.eps_end_box.setDecimals(3)
        self.eps_end_box.setValue(defaults.get("eps_end", 0.05))
        form.addRow("Epsilon End:", self.eps_end_box)

        # === Epsilon Decay ===
        self.eps_decay_box = QSpinBox()
        self.eps_decay_box.setMaximum(10**9)
        self.eps_decay_box.setValue(defaults.get("eps_decay", 10000))
        form.addRow("Epsilon Decay:", self.eps_decay_box)

        layout.addLayout(form)

        # === Save Button ===
        btn = QPushButton("Save")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)

    def get_params(self):
        """Return updated hyperparameters as dict"""
        return {
            "gamma": self.gamma_box.value(),
            "lr": self.lr_box.value(),
            "buffer_size": self.buffer_box.value(),
            "batch_size": self.batch_box.value(),
            "n_step": self.nstep_box.value(),
            "eps_start": self.eps_start_box.value(),
            "eps_end": self.eps_end_box.value(),
            "eps_decay": self.eps_decay_box.value(),
        }


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

        # === Agent selection + Hyperparameters button ===
        agent_row = QHBoxLayout()
        self.agent_box = QComboBox()
        self.agent_box.addItems(["nstep_dqn", "nstep_ddqn"])
        self.agent_box.setMaximumWidth(200)   # keep dropdown compact
        agent_row.addWidget(QLabel("Choose Agent:"))
        agent_row.addWidget(self.agent_box)

        self.hyper_btn = QPushButton("Hyperparameters")
        self.hyper_btn.clicked.connect(self.open_hyperparams)
        agent_row.addWidget(self.hyper_btn)

        layout.addLayout(agent_row)

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

        # Store defaults
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

    def open_hyperparams(self):
        dlg = HyperparamsDialog(self, defaults=self.hyperparams)
        if dlg.exec():
            self.hyperparams = dlg.get_params()
            print("✅ Updated hyperparameters:", self.hyperparams)

    def start_training(self):
        agent = self.agent_box.currentText()
        render = self.render_box.currentText()
        episodes = self.episodes_box.value()
        sutton = self.sutton_cb.isChecked()

        self.status_label.setText(f"Training {agent} for {episodes} episodes…")

        env = gym.make(config.ENV_NAME, sutton_barto_reward=sutton)
        state_dim, action_dim = env.observation_space.shape[0], env.action_space.n

        params = self.hyperparams
        gamma = params["gamma"]
        lr = params["lr"]
        buffer_size = params["buffer_size"]
        batch_size = params["batch_size"]
        n_step = params["n_step"]
        eps_start = params["eps_start"]
        eps_end = params["eps_end"]
        eps_decay = params["eps_decay"]

        if agent == "nstep_dqn":
            from agents.nstep_dqn_agent import NStepDeepQLearningAgent
            ag = NStepDeepQLearningAgent(state_dim, action_dim,
                                        gamma=gamma, lr=lr,
                                        buffer_size=buffer_size,
                                        batch_size=batch_size,
                                        n_step=n_step,
                                        eps_start=eps_start,
                                        eps_end=eps_end,
                                        eps_decay=eps_decay)
        else:
            from agents.nstep_ddqn_agent import NStepDoubleDeepQLearningAgent
            ag = NStepDoubleDeepQLearningAgent(state_dim, action_dim,
                                        gamma=gamma, lr=lr,
                                        buffer_size=buffer_size,
                                        batch_size=batch_size,
                                        n_step=n_step,
                                        eps_start=eps_start,
                                        eps_end=eps_end,
                                        eps_decay=eps_decay)

        rewards = []

        def progress_cb(ep, eps, ep_reward, all_rewards):
            rewards[:] = all_rewards
            avg20 = sum(all_rewards[-20:]) / min(len(all_rewards), 20)
            global_avg = sum(all_rewards) / len(all_rewards)

            self.status_label.setText(
                f"Episode {ep+1}/{eps} — Reward {ep_reward:.1f}, Avg(20): {avg20:.1f}, Global Avg: {global_avg:.1f}"
            )

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
