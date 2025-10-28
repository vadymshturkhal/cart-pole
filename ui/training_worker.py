from PySide6.QtCore import QObject, Signal, Slot
from utils.training import train


class TrainingWorker(QObject):
    progress = Signal(int, int, float, list, list, float)  # ep, episodes, ep_reward, rewards, losses, epsilon
    finished = Signal()

    def __init__(self, env_name, env, agent_name, agent, episodes, hyperparams, render=False):
        super().__init__()
        self.env_name = env_name
        self.env = env
        self.agent_name = agent_name
        self.agent = agent
        self.episodes = episodes
        self._stop_flag = False
        self.hyperparams = hyperparams
        self.render = render
        self.episodes_done = 0

    @Slot()
    def run(self):
        rewards = train(
            self.env, self.agent, 
            episodes=self.episodes,
            progress_cb=self._progress_cb, 
            stop_flag=lambda: self._stop_flag,
            render=self.render
        )

        self.episodes_done = len(rewards)
        # Save model and close environment
        self._update_model_checkpoint(self.episodes, self.episodes_done)
        self.finished.emit()
        self.env.close()

    def _update_model_checkpoint(self, episodes, episodes_done):
        self.extra = {
            "environment": self.env_name,
            "episodes_trained": episodes_done,
            "episodes_total": episodes,
        }

        self.agent.update_checkpoint(extra=self.extra)

    def _progress_cb(self, ep, episodes, ep_reward, rewards, average_loss, epsilon):
        self.progress.emit(ep, episodes, ep_reward, rewards, average_loss, epsilon)

    def _on_finished(self):
        self.status_label.setText("✅ Training finished!")

    def stop(self):
        self._stop_flag = True
