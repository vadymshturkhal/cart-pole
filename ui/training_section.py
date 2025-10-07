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
from utils.agent_factory import build_agent
from environments.factory import create_environment
import config


class TrainingSection(QWidget):
    back_to_main = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.training_thread = None
        self.training_worker = None
        self.training_done = False
        self.agent_name = None
        self.hyperparams = None
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

        layout.addWidget(QLabel("Agent:"))
        self.agent_btn = QPushButton("Choose Agent")
        self.train_btn = QPushButton("Start Training")
        self.stop_btn = QPushButton("Stop Training")
        self.save_btn = QPushButton("Save Model")
        row = QHBoxLayout()
        row.addWidget(self.agent_btn)
        row.addWidget(self.train_btn)
        row.addWidget(self.stop_btn)
        row.addWidget(self.save_btn)
        layout.addLayout(row)

        agent_row = QHBoxLayout()
        self.details_btn = QPushButton("Selected Agent Details")
        self.details_btn.setVisible(False)
        self.details_btn.setMinimumWidth(150)
        self.details_btn.clicked.connect(self._show_agent_details)
        agent_row.addWidget(self.details_btn)
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
            agent_name, hyperparams = dlg.get_selection()
            self.agent_name = agent_name
            self.hyperparams = hyperparams
            self.agent_btn.setText(agent_name)

            # reveal the already-placed button
            self.details_btn.setVisible(True)

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

        # timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_path = f"{config.TRAINED_MODELS_FOLDER}/{self.env_name}_{self.agent_name}.pth"

        # === Create Worker & Thread ===
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

    def _save_agent_as(self):
        if not self.training_done:
            self.status_label.setText("⚠ No trained agent to save")
            return

        # Suggest default filename with timestamp
        default_name = f"{self.env_name}_{self.agent_name}.pth"
        path, _ = QFileDialog.getSaveFileName(self, "Save Agent", f"trained_models/{default_name}", "Model Files (*.pth)")

        if path:
            self.agent.save(path, self.agent.get_checkpoint())
            self.status_label.setText(f"✅ Agent saved as {path}")

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

    def _show_agent_details(self):
        if not hasattr(self, "hyperparams") or not self.hyperparams:
            return
        dlg = AgentDetailsDialog(self.agent_name, self.hyperparams, self)
        dlg.exec()
