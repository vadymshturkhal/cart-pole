from PySide6.QtCore import QObject, Signal, Slot
import torch, os
import config
from utils.training import train
from utils.plotting import plot_rewards


class TrainingWorker(QObject):
    progress = Signal(int, int, float, list)  # ep, episodes, ep_reward, rewards
    finished = Signal(list)                   # rewards when done

    def __init__(self, env, agent, episodes, model_path, render=False):
        super().__init__()
        self.env = env
        self.agent = agent
        self.episodes = episodes
        self.model_path = model_path
        self._stop_flag = False
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

        checkpoint = {
            "model_state": self.agent.q_net.state_dict(),
            "hyperparams": {
                "gamma": self.agent.gamma,
                "lr": self.agent.optimizer.param_groups[0]["lr"],
                "buffer_size": getattr(self.agent.memory.buffer, "maxlen", None),
                "batch_size": self.agent.batch_size,
                "n_step": getattr(self.agent.memory, "n_step", None),
                "eps_start": self.agent.eps_start,
                "eps_end": self.agent.eps_end,
                "eps_decay": self.agent.eps_decay,
            },
            "episodes_trained": len(rewards),
        }

        torch.save(checkpoint, self.model_path)
        self.finished.emit(rewards)

        # plot_rewards(from_file=False, rewards=rewards)
        self.env.close()

    def _progress_cb(self, ep, episodes, ep_reward, rewards):
        self.progress.emit(ep, episodes, ep_reward, rewards)

    def stop(self):
        self._stop_flag = True
