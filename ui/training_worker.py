from PySide6.QtCore import QObject, Signal, Slot
from utils.training import train


class TrainingWorker(QObject):
    progress = Signal(int, int, float, list, float)  # ep, episodes, ep_reward, rewards
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

    @Slot()
    def run(self):
        rewards = train(
            self.env, self.agent, 
            episodes=self.episodes,
            progress_cb=self._progress_cb, 
            stop_flag=lambda: self._stop_flag,
            render=self.render
        )

        # Save model and close environment
        self._update_model_checkpoint(self.episodes, rewards)
        self.finished.emit()
        self.env.close()

    def _update_model_checkpoint(self, episodes, rewards):
        self.extra = {
            "environment": self.env_name,
            "episodes_trained": len(rewards),
            "episodes_total": episodes,
        }

        self.agent.update_checkpoint(extra=self.extra)

    def _progress_cb(self, ep, episodes, ep_reward, rewards, average_loss):
        self.progress.emit(ep, episodes, ep_reward, rewards, average_loss)

    def _on_finished(self):
        self.status_label.setText("✅ Training finished!")

    def stop(self):
        self._stop_flag = True
