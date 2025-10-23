import os
import torch
import config
from PySide6.QtWidgets import QFileDialog
from utils.agent_factory import AGENTS
from ui.load_model_panel import LoadModelPanel
from ui.nn_config_panel import NNConfigPanel
from ui.environment_config_panel import EnvironmentConfigPanel
from ui.agent_config_panel import AgentConfigPanel
from ui.agent_select_panel import AgentSelectPanel


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
    def start_training(self):
        section = self.section

        section.ui.reward_plot.reset(max_episodes=config.EPISODES)
        section.ui.loss_plot.reset(max_episodes=config.EPISODES)
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
        section._log("📂 Loading available models...")
        panel = LoadModelPanel(on_select_callback=self._on_model_selected)
        section.panel_manager.show_panel(panel)
        self.load_panel = panel

    def _on_model_selected(self, path: str | None):
        """Callback for when user selects a model or cancels."""
        section = self.section
        section.panel_manager.close_panel()

        # Handle cancel
        if not path:
            section._log("💡 Model load canceled by user.")
            return

        # Load checkpoint file
        section.selected_model_file = path
        section._log(f"📦 Selected model: {os.path.basename(path)}")

        try:
            checkpoint = torch.load(path, map_location=config.DEVICE)
        except Exception as e:
            section._log(f"❌ Failed to load model: {e}")
            return

        # Apply checkpoint
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

            # Device setup
            device_str = nn_cfg.get("device", str(config.DEVICE))
            config.DEVICE = torch.device(
                device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu"
            )

            section.nn_locked = True

            # Log summary
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
    def choose_agent(self):
        """Inline agent selection using InlinePanelManager."""
        section = self.section
        read_only = section.training_active
        if read_only:
            section._log("⚠️ Agent cannot be changed during training.")
            return

        section._log("🤖 Opening Agent selection panel...")
        panel = AgentSelectPanel(
            on_close_callback=self._on_agent_selected,
            current_agent=section.agent_name,
        )
        section.panel_manager.show_panel(panel)
        self.agent_panel = panel

    def _on_agent_selected(self, applied: bool, agent_name: str | None):
        """Handle agent selection result."""
        section = self.section
        section.panel_manager.close_panel()

        if not applied or not agent_name:
            section._log("💡 Agent selection canceled.")
            return

        section.agent_name = agent_name
        from utils.agent_factory import AGENTS
        AgentClass = AGENTS[agent_name]
        section.hyperparams = AgentClass.get_default_hyperparams()
        section.ui.agent_btn.setText(f"Agent: {agent_name}")
        section.ui.agent_config_btn.setText(f"{agent_name} Configuration")
        section.nn_locked = False

        section._log(f"✅ Selected agent: {agent_name}")
    
    def show_agent_config(self):
        """Open inline Agent configuration panel using InlinePanelManager."""
        section = self.section
        read_only = section.training_active
        mode = "read-only" if read_only else "editable"
        section._log(f"🤖 Opening Agent configuration ({mode})...")

        panel = AgentConfigPanel(
            agent_name=section.agent_name,
            hyperparams=section.hyperparams.copy(),
            on_close_callback=self._on_agent_config_closed,
            read_only=read_only,
        )
        section.panel_manager.show_panel(panel)
        self.agent_panel = panel

    def _on_agent_config_closed(self, applied: bool, updates: dict | None):
        """Handle closing of Agent configuration panel."""
        section = self.section
        section.panel_manager.close_panel()

        if not applied:
            section._log("💡 Agent configuration canceled.")
            return

        section.hyperparams = updates
        section._log("⚙️ Agent hyperparameters updated.")

    def show_environment_config(self):
        """Open inline Environment configuration panel."""
        section = self.section
        read_only = section.training_active
        mode = "read-only" if read_only else "editable"
        section._log(f"🌍 Opening Environment configuration ({mode})...")

        panel = EnvironmentConfigPanel(
            on_close_callback=self._on_env_config_closed,
            read_only=read_only,
        )
        section.panel_manager.show_panel(panel)
        self.env_panel = panel

    def _on_env_config_closed(self, applied: bool, updates: dict | None):
        """Handle closing of Environment configuration panel."""
        section = self.section
        section.panel_manager.close_panel()

        if not applied:
            section._log("💡 Environment configuration canceled.")
            return

        section._log(
            f"🌍 Environment updated: {updates['ENV_NAME']} | "
            f"{updates['MAX_STEPS']} steps/ep | {updates['EPISODES']} episodes | "
            f"Render: {updates['RENDER_MODE']}"
        )

    def show_nn_config(self):
        """Open inline NN configuration panel using InlinePanelManager."""
        section = self.section

        read_only = section.training_active
        mode_text = "read-only" if read_only else "editable"
        section._log(f"🧩 Opening NN configuration editor ({mode_text})...")

        panel = NNConfigPanel(
            on_close_callback=self._on_nn_config_closed,
            lock_hidden_layers=section.nn_locked,
            read_only=read_only,
        )
        section.panel_manager.show_panel(panel)
        self.nn_panel = panel

    def _on_nn_config_closed(self, applied: bool, updates: dict | None):
        """Handle close of NN config panel."""
        section = self.section
        section.panel_manager.close_panel()

        if not applied:
            section._log("💡 NN configuration changes canceled.")
            return

        # Already applied by panel, just log summary
        section._log(
            f"🧠 NN updated: layers={updates['HIDDEN_LAYERS']} activation={updates['HIDDEN_ACTIVATION']} "
            f"dropout={updates['DROPOUT']:.2f}, lr={updates['LR']}, optimizer={updates['OPTIMIZER']}"
        )
