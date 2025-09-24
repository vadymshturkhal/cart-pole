from PySide6.QtCore import QObject, Signal, Slot
import torch, os
import config
from utils.training import train
from utils.plotting import plot_rewards

class TrainingWorker(QObject):
    progress = Signal(int, int, float, list)  # ep, episodes, ep_reward, rewards
    finished = Signal(list)                   # rewards when done

    def __init__(self, env, agent, episodes, model_path):
        super().__init__()
        self.env = env
        self.agent = agent
        self.episodes = episodes
        self.model_path = model_path
        self._stop_flag = False

    @Slot()
    def run(self):
        rewards = train(
            self.env, self.agent, episodes=self.episodes,
            progress_cb=self._progress_cb, stop_flag=lambda: self._stop_flag
        )
        # Save model & plot after training
        os.makedirs(config.TRAINED_MODELS_FOLDER, exist_ok=True)
        torch.save(self.agent.q_net.state_dict(), self.model_path)
        plot_rewards(from_file=False, rewards=rewards)
        self.finished.emit(rewards)

    def _progress_cb(self, ep, episodes, ep_reward, rewards):
        self.progress.emit(ep, episodes, ep_reward, rewards)

    def stop(self):
        self._stop_flag = True
