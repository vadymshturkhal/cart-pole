from PySide6.QtCore import QObject, QThread, Signal
from utils.agent_factory import build_agent
from utils.run_logger import RunLogger
import traceback
from ui.training_worker import TrainingWorker
import gymnasium as gym
from environments.factory import create_environment


class TrainingController(QObject):
    """
    Handles training lifecycle, threading, and model persistence.
    Acts as a logic/controller layer separated from the GUI.
    """

    progress: Signal = Signal(int, int, float, list, list, float)
    finished: Signal = Signal()
    status: Signal = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.training_thread: QThread | None = None
        self.training_worker = None
        self.agent = None
        self.env_name: str | None = None
        self.agent_name: str | None = None
        self.hyperparams: dict | None = None
        self.render_mode: str = "off"
        self.total_episodes: int = 0
        self.episodes_done: int = 0

    # ------------------------------------------------------------------
    # Core Lifecycle
    # ------------------------------------------------------------------
    def start_training(
        self,
        env_name: str,
        agent_name: str,
        hyperparams: dict,
        total_episodes: int,
        render_mode: str,
        max_steps: int | None = None,
        model_file = None,
    ) -> None:
        """
        Start background training in a separate thread.
        """

        self.env_name = env_name
        self.agent_name = agent_name
        self.hyperparams = hyperparams
        self.total_episodes = total_episodes
        self.render_mode = render_mode
        self.selected_model_file = model_file

        try:
            env, state_dim, action_dim = create_environment(env_name, render_mode)

            if max_steps:
                env = gym.wrappers.TimeLimit(env.unwrapped, max_episode_steps=max_steps)

            self.agent = build_agent(agent_name, state_dim, action_dim, self.hyperparams)
            self.agent.set_total_episodes(total_episodes)
            self.agent.add_max_episode_steps_to_checkpoint(max_steps)

            if self.selected_model_file is not None:
                self.agent.load(self.selected_model_file, self.hyperparams, apply_nn_config=False)

            # Setup QThread + Worker
            self.training_thread = QThread()
            self.training_worker = TrainingWorker(
                env_name,
                env,
                agent_name,
                self.agent,
                total_episodes,
                hyperparams=hyperparams,
                render=(render_mode == "human"),
            )
            self.training_worker.moveToThread(self.training_thread)

            # Thread-Signal wiring
            self.training_thread.started.connect(self.training_worker.run)
            self.training_worker.progress.connect(self.progress.emit)
            self.training_worker.finished.connect(self._on_finished)
            self.training_worker.finished.connect(self.training_thread.quit)
            self.training_worker.finished.connect(self.training_worker.deleteLater)
            self.training_thread.finished.connect(self.training_thread.deleteLater)

            self.status.emit("🚀 Training started...")
            self.training_thread.start()

        except Exception as e:
            self.status.emit(f"❌ Failed to start training: {e}")
            traceback.print_exc()

    def stop_training(self) -> None:
        """Gracefully stop current training session."""
        if self.training_worker:
            self.training_worker.stop()
            self.status.emit("⏹ Training stopped by user")
        else:
            self.status.emit("⚠ No training running")

        self._autosave_model()
        self.episodes_done = self.training_worker.episodes_done

    def _on_finished(self) -> None:
        """Handle training completion."""
        self.status.emit("✅ Training finished!")
        self.finished.emit()
        self._autosave_model()
        self.episodes_done = self.training_worker.episodes_done

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_model(self, user_dir: str, reward_plot, loss_plot) -> None:
        """Manually save model and logs to file."""
        if not self.agent:
            self.status.emit("⚠ No trained agent to save.")
            return

        try:
            run_logger = RunLogger(self.env_name, self.agent, reward_plot, loss_plot)
            run_logger.save_model(user_dir)
            self.status.emit(f"✅ Model saved to: {user_dir}")
        except Exception as e:
            self.status.emit(f"❌ Save failed: {e}")
            traceback.print_exc()

    def _autosave_model(self) -> None:
        """Autosave model data after training ends."""
        if not self.agent:
            self.status.emit("⚠ Nothing to autosave.")
            return

        try:
            run_logger = RunLogger(self.env_name, self.agent, self.reward_plot, self.loss_plot)
            run_logger.autosave_model()
            self.status.emit(f"💾 Autosaved to {run_logger.autosave_dir}")
        except Exception as e:
            self.status.emit(f"❌ Autosave failed: {e}")
            traceback.print_exc()

    def add_plots(self, reward_plot, loss_plot):
        self.reward_plot = reward_plot
        self.loss_plot = loss_plot
