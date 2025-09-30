from PySide6.QtCore import QObject, Signal, Slot
import torch, os
import config
from utils.training import train
from utils.plotting import plot_rewards


class TrainingWorker(QObject):
    progress = Signal(int, int, float, list)  # ep, episodes, ep_reward, rewards
    finished = Signal(list, dict) # rewards and checkpoint

    def __init__(self, env_name, env, agent_name, agent, episodes, model_path, hyperparams, render=False):
        super().__init__()
        self.env_name = env_name
        self.env = env
        self.agent_name = agent_name
        self.agent = agent
        self.episodes = episodes
        self.model_path = model_path
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
        # Save model & plot after training
        os.makedirs(config.TRAINED_MODELS_FOLDER, exist_ok=True)

        # ensure q_net has the final weights
        if hasattr(self.agent, "target_net"):
            self.agent.q_net.load_state_dict(self.agent.target_net.state_dict())

        checkpoint = {
            "model_state": self.agent.q_net.state_dict(),
            "hyperparams": self.hyperparams,
            "agent_name": self.agent_name,
            "episodes_trained": len(rewards),
            "episodes_total": self.episodes,
            "environment": self.env_name,
        }

        torch.save(checkpoint, self.model_path)
        self.finished.emit(rewards, checkpoint)

        self.env.close()

    def _progress_cb(self, ep, episodes, ep_reward, rewards):
        self.progress.emit(ep, episodes, ep_reward, rewards)

    def stop(self):
        self._stop_flag = True
