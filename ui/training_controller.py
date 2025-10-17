from PySide6.QtCore import QObject, QThread, Signal
from environments.factory import create_environment
from utils.agent_factory import build_agent
from utils.run_logger import RunLogger
import traceback


class TrainingController(QObject):
    """Handles training lifecycle, threading, and model persistence."""
    progress = Signal(int, int, float, list, float)
    finished = Signal()
    status = Signal(str)

    def __init__(self):
        super().__init__()
        self.training_thread = None
        self.training_worker = None
        self.agent = None
        self.env_name = None
        self.agent_name = None
        self.hyperparams = None
        self.render_mode = "off"
        self.episodes = 0

    # ------------------------------------------------------------------
    # Core Lifecycle
    # ------------------------------------------------------------------
    def start_training(self, env_name, agent_name, hyperparams, episodes, render_mode):
        """Start the background training process."""
        from ui.training_worker import TrainingWorker

        # Setup environment and agent
        self.env_name = env_name
        self.agent_name = agent_name
        self.hyperparams = hyperparams
        self.episodes = episodes
        self.render_mode = render_mode

        try:
            env, state_dim, action_dim = create_environment(env_name, render_mode)
            self.agent = build_agent(agent_name, state_dim, action_dim, hyperparams)

            # Thread + Worker
            self.training_thread = QThread()
            self.training_worker = TrainingWorker(
                env_name, env, agent_name, self.agent, episodes,
                hyperparams=hyperparams, render=(render_mode == "human")
            )
            self.training_worker.moveToThread(self.training_thread)

            # Connections
            self.training_thread.started.connect(self.training_worker.run)
            self.training_worker.progress.connect(self.progress.emit)
            self.training_worker.finished.connect(self._on_finished)
            self.training_worker.finished.connect(self.training_thread.quit)
            self.training_worker.finished.connect(self.training_worker.deleteLater)
            self.training_thread.finished.connect(self.training_thread.deleteLater)

            # Start
            self.status.emit("🚀 Training started...")
            self.training_thread.start()

        except Exception as e:
            self.status.emit(f"❌ Failed to start training: {e}")
            traceback.print_exc()

    def stop_training(self):
        """Gracefully stop the current training session."""
        if self.training_worker:
            self.training_worker.stop()
            self.status.emit("⏹ Training stopped by user")
        else:
            self.status.emit("⚠ No training running")

        self._autosave_model()

    def _on_finished(self):
        """Handle training completion."""
        self.status.emit("✅ Training finished!")
        self.finished.emit()
        self._autosave_model()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_model(self, user_dir, reward_plot, loss_plot):
        """Save model manually."""
        if not self.agent:
            self.status.emit("⚠ No trained agent to save.")
            return

        try:
            run_logger = RunLogger(self.env_name, self.agent, reward_plot, loss_plot)
            run_logger.save_model(user_dir)
            self.status.emit(f"✅ Model saved to: {user_dir}")
        except Exception as e:
            self.status.emit(f"❌ Failed to save: {e}")
            traceback.print_exc()

    def _autosave_model(self):
        """Autosave after training."""
        if not self.agent:
            self.status.emit("⚠ Nothing to autosave.")
            return

        try:
            run_logger = RunLogger(self.env_name, self.agent, None, None)
            run_logger.autosave_model()
            self.status.emit(f"💾 Autosaved to {run_logger.autosave_dir}")
        except Exception as e:
            self.status.emit(f"❌ Autosave failed: {e}")
            traceback.print_exc()
