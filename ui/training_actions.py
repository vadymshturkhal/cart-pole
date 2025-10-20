import os
import torch
import config
from PySide6.QtWidgets import QFileDialog
from ui.agent_dialog import AgentDialog
from ui.agent_config_dialog import AgentConfigDialog
from ui.nn_config_dialog import NNConfigDialog
from ui.environment_config_dialog import EnvironmentConfigDialog
from utils.agent_factory import AGENTS
from ui.load_model_panel import LoadModelPanel


class TrainingActions:
    """Encapsulates all button-driven logic for TrainingSection."""

    def __init__(self, section, controller):
        self.section = section  # TrainingSection
        self.controller = controller
        self.ui = section.ui
        self._connect()

    # --------------------------------------------------------------
    # Connect UI
    # --------------------------------------------------------------
    def _connect(self):
        u = self.ui
        u.agent_btn.clicked.connect(self.choose_agent)
        u.train_btn.clicked.connect(self.start_training)
        u.stop_btn.clicked.connect(self.stop_training)
        u.save_btn.clicked.connect(self.save_agent)
        u.load_btn.clicked.connect(self.load_model)
        u.agent_config_btn.clicked.connect(self.show_agent_config)
        u.env_config_btn.clicked.connect(self.show_environment_config)
        u.nn_btn.clicked.connect(self.show_nn_config)

    # --------------------------------------------------------------
    # Core actions
    # --------------------------------------------------------------
    def choose_agent(self):
        section = self.section
        dlg = AgentDialog(section, current_agent=section.agent_name)
        if not dlg.exec():
            return

        section.nn_locked = False
        section.agent_name = dlg.get_selection()
        AgentClass = AGENTS[section.agent_name]
        section.hyperparams = AgentClass.get_default_hyperparams()
        section.ui.agent_btn.setText(f"Agent: {section.agent_name}")
        section.ui.agent_config_btn.setText(f"{section.agent_name} Configuration")
        section._log(f"✅ Selected {section.agent_name} agent")

    def start_training(self):
        section = self.section

        section.ui.reward_plot.reset()
        section.ui.loss_plot.reset()
        section._set_training_buttons(False)

        self.controller.start_training(
            config.ENV_NAME,
            section.agent_name,
            section.hyperparams,
            config.EPISODES,
            config.RENDER_MODE,
            config.MAX_STEPS,
            section.selected_model_file,
        )

        section._log(
            f"🚀 Training started on {config.ENV_NAME} — "
            f"{config.EPISODES} episodes, {config.MAX_STEPS} steps/episode | Render: {config.RENDER_MODE}"
        )

    def stop_training(self):
        self.controller.stop_training()
        self.section._set_training_buttons(True)

    def save_agent(self):
        section = self.section

        default_dir = os.path.join(config.TRAINED_MODELS_FOLDER)
        user_dir, _ = QFileDialog.getSaveFileName(section, "Save Agent As", default_dir)
        if not user_dir:
            section._log("💡 Save canceled by user.")
            return
        self.controller.save_model(user_dir, section.ui.reward_plot, section.ui.loss_plot)

    def load_model(self):
        """Show model selection panel and handle model loading."""
        section = self.section
        ui = section.ui

        section._log("📂 Loading available models...")
        ui.tabs.setVisible(False)
        self.load_panel = self._show_load_panel(ui, section)

    def _show_load_panel(self, ui, section):
        """Temporarily replace plots with the model list panel."""
        panel = LoadModelPanel(on_select_callback=self._on_model_selected)
        ui.layout.insertWidget(0, panel)
        return panel


    def _on_model_selected(self, path: str | None):
        """Callback for when user selects a model or cancels."""
        section = self.section
        ui = section.ui

        # Restore layout
        ui.layout.removeWidget(self.load_panel)
        self.load_panel.deleteLater()
        ui.tabs.setVisible(True)

        if not path:
            section._log("💡 Model load canceled by user.")
            return

        section.selected_model_file = path
        section._log(f"📦 Selected model: {os.path.basename(path)}")

        try:
            checkpoint = torch.load(path, map_location=config.DEVICE)
        except Exception as e:
            section._log(f"❌ Failed to load model: {e}")
            return

        self._apply_checkpoint(checkpoint)

    def _apply_checkpoint(self, checkpoint: dict):
        """Apply checkpoint contents to configuration and UI."""
        section = self.section

        try:
            # Agent setup
            section.agent_name = checkpoint.get("agent_name", section.agent_name)
            section.ui.agent_btn.setText(f"Agent: {section.agent_name}")
            AgentClass = AGENTS.get(section.agent_name)
            if AgentClass:
                section.hyperparams = checkpoint.get(
                    "hyperparams", AgentClass.get_default_hyperparams()
                )
            section.ui.agent_config_btn.setText(f"{section.agent_name} Configuration")

            # Environment setup
            config.ENV_NAME = checkpoint.get("environment", config.ENV_NAME)
            config.MAX_STEPS = checkpoint.get("max_steps", config.MAX_STEPS)
            config.EPISODES = checkpoint.get("episodes_total", config.EPISODES)
            config.RENDER_MODE = checkpoint.get("render_mode", config.RENDER_MODE)

            # NN setup
            nn_cfg = checkpoint.get("nn_config", {})
            config.HIDDEN_LAYERS = nn_cfg.get("hidden_layers", config.HIDDEN_LAYERS)
            config.LR = nn_cfg.get("lr", config.LR)
            config.ACTIVATION = nn_cfg.get("activation", config.ACTIVATION)
            config.DROPOUT = nn_cfg.get("dropout", config.DROPOUT)
            config.HIDDEN_ACTIVATION = nn_cfg.get("activation", config.ACTIVATION)
            config.OPTIMIZER = nn_cfg.get("optimizer", getattr(config, "OPTIMIZER", "adam"))
            device_str = nn_cfg.get("device", str(config.DEVICE))
            config.DEVICE = torch.device(
                device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu"
            )

            section.nn_locked = True

            # Log results
            section._log(f"✅ Model loaded: {section.agent_name} on {config.ENV_NAME}")
            section._log(
                f"→ NN: layers={config.HIDDEN_LAYERS}, lr={config.LR}, "
                f"dropout={config.DROPOUT}, device={config.DEVICE}"
            )
            section._log(f"→ Episodes trained: {checkpoint.get('episodes_trained', 'N/A')}")

        except Exception as e:
            section._log(f"❌ Error applying model checkpoint: {e}")

        
    # --------------------------------------------------------------
    # Config dialogs
    # --------------------------------------------------------------
    def show_environment_config(self):
        section = self.section
        dlg = EnvironmentConfigDialog(section, read_only=section.training_active)
        if not dlg.exec():
            return

        updates = dlg.get_updated_config()
        section._log(
            f"🌍 Environment configured: {updates['ENV_NAME']} | "
            f"{updates['MAX_STEPS']} steps/ep | {updates['EPISODES']} episodes | Render: {updates['RENDER_MODE']}"
        )

    def show_agent_config(self):
        section = self.section
        dlg = AgentConfigDialog(section.agent_name, section.hyperparams.copy(), section, section.training_active)
        if dlg.exec() and not section.training_active:
            section.hyperparams = dlg.get_updated_params()
            section._log("⚙️ Agent hyperparameters updated.")

    def show_nn_config(self):
        section = self.section
        dlg = NNConfigDialog(section, read_only=section.training_active, lock_hidden_layers=section.nn_locked)
        if dlg.exec() and not section.training_active:
            updates = dlg.get_updated_config()
            config.HIDDEN_LAYERS = updates["HIDDEN_LAYERS"]
            config.LR = updates["LR"]
            config.ACTIVATION = updates["ACTIVATION"]
            config.DROPOUT = updates["DROPOUT"]
            config.DEVICE = torch.device(updates["DEVICE"])
            section._log(
                f"🧠 NN updated: layers={updates['HIDDEN_LAYERS']} activation={updates['ACTIVATION']} "
                f"dropout={updates['DROPOUT']:.2f} device={config.DEVICE}"
            )
