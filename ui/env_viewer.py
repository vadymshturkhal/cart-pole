from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer, Qt, Signal
import numpy as np


class EnvViewer(QLabel):
    """
    Simple environment viewer for Gymnasium environments.
    Displays rendered frames and steps through environment automatically.
    Emits `finished` signal after the last episode.
    """

    finished = Signal()

    def __init__(self, env, agent, episodes=1, fps=30, parent=None):
        super().__init__(parent)
        self.env = env
        self.agent = agent
        self.episodes = episodes
        self.fps = fps
        self.timer = QTimer()
        self.timer.timeout.connect(self.step_env)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: white;")

        self.obs, _ = env.reset()
        self.done = False
        self.current_episode = 0
        self.running = False
        self.total_rewards_in_episode = 0

    def start(self):
        """Begin stepping through the environment automatically."""
        if not self.running:
            self.running = True
            self.timer.start(int(1000 / self.fps))
        
    def stop(self):
        """Stop stepping and reset viewer state."""
        if self.timer.isActive():
            self.timer.stop()
        self.running = False

    def step_env(self):
        if self.done:
            self._on_done()

        # Agent selects an action
        action = self.agent.select_action(self.obs, greedy=True)

        # Step environment
        self.obs, reward, terminated, truncated, _ = self.env.step(action)
        self.total_rewards_in_episode += reward
        self.done = terminated or truncated

        # Render frame
        frame = self.env.render()
        if isinstance(frame, np.ndarray):
            h, w, _ = frame.shape
            qimg = QImage(frame.data, w, h, 3 * w, QImage.Format_RGB888)
            self.setPixmap(QPixmap.fromImage(qimg))

    def _on_done(self):
        self.current_episode += 1

        # Set total rewards to 0
        self.total_rewards_in_episode = 0

        # All episodes done → emit signal & stop
        if self.current_episode >= self.episodes:
            self.timer.stop()
            self.obs, _ = self.env.reset()
            self.finished.emit()
            return

        self.obs, _ = self.env.reset()
        self.done = False
