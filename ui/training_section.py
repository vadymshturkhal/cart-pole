from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, 
    QComboBox, QSpinBox, QFileDialog, QVBoxLayout, QTabWidget
)
from PySide6.QtCore import QThread, Signal
from ui.agent_dialog import AgentDialog
from ui.agent_details_dialog import AgentDetailsDialog
from ui.reward_plot import RewardPlot
from ui.loss_plot import LossPlot
from ui.training_worker import TrainingWorker
from ui.nn_config_dialog import NNConfigDialog
from utils.agent_factory import AGENTS, build_agent
from environments.factory import create_environment
import config
import os
from utils.run_logger import RunLogger


class TrainingSection(QWidget):
    back_to_main = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.training_thread = None
        self.training_worker = None
        self.training_done = False
        self.hyperparams = None

        # Default Agent
        self.agent_name = config.DEFAULT_AGENT
        AgentClass = AGENTS[self.agent_name]
        self.hyperparams = AgentClass.get_default_hyperparams()

        self._build_training_section()


    def _build_training_section(self):
        layout = QVBoxLayout(self)

        # Plots
        self.tabs = QTabWidget()
        self.reward_plot = RewardPlot()
        self.loss_plot = LossPlot()

        self.tabs.addTab(self.reward_plot, "Training Curve")
        self.tabs.addTab(self.loss_plot, "Loss Curve")

        layout.addWidget(self.tabs)
        self.tabs.setMinimumHeight(300)
        self.tabs.setMaximumHeight(310)

        layout.addWidget(QLabel("Environment:"))
        self.env_box = QComboBox()
        self.env_box.addItems(config.AVAILABLE_ENVIRONMENTS)
        self.env_box.setCurrentText(config.DEFAULT_ENVIRONMENT)
        layout.addWidget(self.env_box)

        # Agent row
        layout.addWidget(QLabel("Agent:"))
        self.agent_btn = QPushButton(f"{self.agent_name}")
        self.train_btn = QPushButton("Start Training")
        self.stop_btn = QPushButton("Stop Training")
        self.save_btn = QPushButton("Save Model")
        row = QHBoxLayout()
        row.addWidget(self.agent_btn)
        row.addWidget(self.train_btn)
        row.addWidget(self.stop_btn)
        row.addWidget(self.save_btn)
        layout.addLayout(row)

        # Configure row
        agent_row = QHBoxLayout()
        self.details_btn = QPushButton(f"Configure {self.agent_name}")
        self.details_btn.setVisible(True)
        self.details_btn.setMinimumWidth(150)
        self.details_btn.clicked.connect(self._show_agent_details)
        agent_row.addWidget(self.details_btn)

        self.nn_btn = QPushButton("Configure NN")
        self.nn_btn.setMinimumWidth(150)
        self.nn_btn.clicked.connect(self._show_nn_config)
        agent_row.addWidget(self.nn_btn)
        layout.addLayout(agent_row)

        layout.addWidget(QLabel("Rendering Mode:"))
        self.render_box = QComboBox()
        self.render_box.addItems(["off", "human", "gif", "mp4"])
        layout.addWidget(self.render_box)

        layout.addWidget(QLabel("Training Episodes:"))
        self.episodes_box = QSpinBox()
        self.episodes_box.setRange(100, 10000)
        self.episodes_box.setValue(config.EPISODES)
        layout.addWidget(self.episodes_box)

        self.status_label = QLabel("Idle")
        layout.addWidget(self.status_label)

        # Back button
        back_btn = QPushButton("⬅ Back to Main Menu")
        back_btn.setMinimumHeight(40)
        back_btn.setStyleSheet("font-size: 16px;")

        # Emit Signal
        back_btn.clicked.connect(self.back_to_main.emit)
        layout.addWidget(back_btn)

        # Connects
        self.agent_btn.clicked.connect(self._choose_agent)
        self.train_btn.clicked.connect(self._start_training)
        self.stop_btn.clicked.connect(self._stop_training)
        self.save_btn.clicked.connect(self._save_agent_as)
    
    def _choose_agent(self):
        dlg = AgentDialog(self, current_agent=self.agent_name)
        if dlg.exec():
            agent_name = dlg.get_selection()
            self.agent_name = agent_name
            AgentClass = AGENTS[self.agent_name]
            self.hyperparams = AgentClass.get_default_hyperparams()
            self.agent_btn.setText(agent_name)
            self.details_btn.setVisible(True)
            self.details_btn.setText(f"Configure {self.agent_name}")
            self.status_label.setText(f"✅ Selected {self.agent_name} agent")

    def _start_training(self):
        if self.training_thread and self.training_thread.isRunning():
            self.status_label.setText("⚠ Training is already running!")
            return

        if self.agent_name is None:
            self.status_label.setText("⚠ Please select an agent to train first by pressing the Choose Agent button")
            return

        render = self.render_box.currentText()
        episodes = self.episodes_box.value()

        # Reset plots
        self.reward_plot.reset(max_episodes=episodes)
        self.loss_plot.reset(max_steps=episodes)

        self.env_name = self.env_box.currentText()
        env, state_dim, action_dim = create_environment(self.env_name, render)

        self.agent = build_agent(self.agent_name, state_dim, action_dim, self.hyperparams)

        # Saving model's data
        self.run_logger = RunLogger(config.TRAINED_MODELS_FOLDER, self.env_name, self.agent)
        model_path = os.path.join(self.run_logger.run_dir, "model.pth")

        # Create Worker & Thread
        self.training_thread = QThread()
        self.training_worker = TrainingWorker(self.env_name, env, self.agent_name, self.agent, episodes, 
                                              model_path, hyperparams=self.hyperparams, render=(render == "human"))
        self.training_worker.moveToThread(self.training_thread)

        # Connect signals
        self.training_thread.started.connect(self.training_worker.run)
        self.training_worker.progress.connect(self._on_progress)
        self.training_worker.finished.connect(self._on_finished)

        # Cleanup
        self.training_worker.finished.connect(self.training_thread.quit)
        self.training_worker.finished.connect(self.training_worker.deleteLater)
        self.training_thread.finished.connect(self.training_thread.deleteLater)
        self.training_thread.finished.connect(self._reset_training_refs)

        # Start training
        self.training_thread.start()
        self.status_label.setText("🚀 Training started...")
    
    def _stop_training(self):
        if self.training_worker:
            self.training_worker.stop()
            self.status_label.setText("⏹ Training stopped by user")
        else:
            self.status_label.setText("⚠ No training is running")

        self._export_training_data()

    def _save_agent_as(self):
        """Manual save: lets the user export the trained agent and optional artifacts."""
        if not self.training_done or not hasattr(self, "agent"):
            self.status_label.setText("⚠ No trained agent available to save.")
            return

        if not hasattr(self, "run_logger"):
            # Fallback if no logger exists (e.g. manually loaded model)
            self.run_logger = RunLogger(config.TRAINED_MODELS_FOLDER, self.env_name, self.agent_name)

        # Suggest default filename based on latest run
        default_name = f"{self.env_name}_{self.agent_name}.pth"
        default_path = os.path.join(self.run_logger.run_dir, default_name)

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Agent As",
            default_path,
            "Model Files (*.pth)"
        )

        if not path:
            self.status_label.setText("💡 Save canceled by user.")
            return

        try:
            # Save the agent model itself
            self.agent.save(path, self.agent.get_checkpoint())

            # If user saved outside the default run folder, offer to export artifacts too
            user_dir = os.path.dirname(path)
            if os.path.abspath(user_dir) != os.path.abspath(self.run_logger.run_dir):
                self.run_logger.save_plots(self.reward_plot, self.loss_plot)
                self.run_logger.save_config(self.hyperparams)
                self.status_label.setText(
                    f"✅ Agent and related artifacts saved to:\n{user_dir}"
                )
            else:
                self.status_label.setText(f"✅ Agent saved successfully at {path}")

        except Exception as e:
            self.status_label.setText(f"❌ Failed to save agent: {e}")
            print(f"[TrainingSection] Save error: {e}")

    def _reset_training_refs(self):
        self.training_thread = None
        self.training_worker = None

    def _on_progress(self, ep, episodes, ep_reward, rewards, average_loss):
        avg20 = sum(rewards[-20:]) / min(len(rewards), 20)
        global_avg = sum(rewards) / len(rewards)
        self.status_label.setText(
            f"Ep {ep+1}/{episodes} — R {ep_reward:.1f}, Avg20 {avg20:.1f}, Global {global_avg:.1f}, Average episode loss {average_loss:.2f}"
        )

        # Update plots
        self.reward_plot.update_plot(rewards, episodes)
        self.loss_plot.add_point(average_loss)
    
    def _on_finished(self ):
        self.training_done = True
        self.status_label.setText("✅ Training finished!")
        self.save_btn.setEnabled(True)
        self._export_training_data()

    def _show_agent_details(self):
        if not hasattr(self, "hyperparams") or not self.hyperparams:
            return

        dlg = AgentDetailsDialog(self.agent_name, self.hyperparams.copy(), self)
        if dlg.exec():
            # update internal hyperparams
            self.hyperparams = dlg.get_updated_params()
            self.status_label.setText("⚙️ Agent hyperparameters updated.")

    def _show_nn_config(self):
        dlg = NNConfigDialog(self)
        if dlg.exec():
            updates = dlg.get_updated_config()

            # Update runtime config
            config.HIDDEN_LAYERS = updates["HIDDEN_LAYERS"]
            config.LR = updates["LR"]
            config.ACTIVATION = updates["ACTIVATION"]
            config.DROPOUT = updates["DROPOUT"]
            config.DEVICE = config.torch.device(updates["DEVICE"])
            self.status_label.setText(
                f"🧠 NN updated: \
                    layers={updates['HIDDEN_LAYERS']} \
                    activation={updates['ACTIVATION']} \
                    dropout={updates['DROPOUT']:.2f} \
                    device={config.DEVICE} \
                    "
            )
            
    def _export_training_data(self):
        """Export all run data using RunLogger."""
        if not hasattr(self, "run_logger") or not hasattr(self, "agent"):
            self.status_label.setText("⚠ Nothing to export — missing logger or agent.")
            return

        try:
            self.run_logger.save_model()
            self.run_logger.save_plots(self.reward_plot)
            self.run_logger.save_config()

            self.status_label.setText(f"💾 Training data exported to {self.run_logger.run_dir}")
        except Exception as e:
            self.status_label.setText(f"❌ Export failed: {e}")
            print(f"[TrainingSection] Export error: {e}")
        