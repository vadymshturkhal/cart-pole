from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer, Qt
import numpy as np


class EnvViewer(QLabel):
    def __init__(self, env, agent, episodes=1, fps=30, parent=None):
        super().__init__(parent)
        self.env = env
        self.agent = agent
        self.episodes = episodes
        self.fps = fps
        self.timer = QTimer()
        self.timer.timeout.connect(self.step_env)
        self.setAlignment(Qt.AlignCenter)

        self.obs, _ = env.reset()
        self.done = False
        self.current_episode = 0

    def start(self):
        self.timer.start(int(1000 / self.fps))

    def step_env(self):
        if self.done:
            self.current_episode += 1
            if self.current_episode >= self.episodes:
                self.timer.stop()
                return
            self.obs, _ = self.env.reset()
            self.done = False

        action = self.agent.select_action(self.obs, greedy=True)  # use policy
        self.obs, _, terminated, truncated, _ = self.env.step(action)
        self.done = terminated or truncated

        frame = self.env.render()
        if isinstance(frame, np.ndarray):
            h, w, c = frame.shape
            qimg = QImage(frame.data, w, h, 3 * w, QImage.Format_RGB888)
            self.setPixmap(QPixmap.fromImage(qimg))

    def stop(self):
        self.timer.stop()
        