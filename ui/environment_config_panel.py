from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QComboBox, QSpinBox,
    QPushButton, QHBoxLayout
)
import config
import gymnasium as gym
import json, os
from config_manager import ConfigManager



class EnvironmentConfigPanel(QWidget):
    """Inline panel for configuring environment-related training settings."""

    PANEL_NAME = "environment"

    def __init__(self, section, on_close_callback, read_only=False):
        super().__init__()
        self.section = section
        self.on_close_callback = on_close_callback
        self.read_only = read_only
        self.config_mgr = ConfigManager.instance()

        fallback = {
            "ENV_NAME": "CartPole-v1",
            "MAX_STEPS": 500,
            "EPISODES": 1000,
            "RENDER_MODE": "off",
        }

        self.updated_env_config = self.config_mgr.get_section(self.PANEL_NAME, fallback)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        title = QLabel("<b>🌍 Environment Configuration</b>")
        title.setStyleSheet("font-size:16px; color:#ddd;")
        layout.addWidget(title)

        # Environment selection
        layout.addWidget(QLabel("Environment:"))
        self.env_box = QComboBox()
        self.env_box.addItems(config.AVAILABLE_ENVIRONMENTS)
        self.env_box.setCurrentText(self.updated_env_config["ENV_NAME"])
        self.env_box.currentTextChanged.connect(self._update_default_steps)
        layout.addWidget(self.env_box)

        # Default steps label
        self.default_label = QLabel()
        layout.addWidget(self.default_label)

        # Max steps
        layout.addWidget(QLabel("Max Steps per Episode:"))
        self.steps_box = QSpinBox()
        self.steps_box.setRange(50, 20000)
        self.steps_box.setValue(self.updated_env_config["MAX_STEPS"])
        layout.addWidget(self.steps_box)

        # Episodes
        layout.addWidget(QLabel("Training Episodes:"))
        self.episodes_box = QSpinBox()
        self.episodes_box.setRange(*config.EPISODE_RANGE)
        self.episodes_box.setValue(self.updated_env_config["EPISODES"])
        layout.addWidget(self.episodes_box)

        # Rendering mode
        layout.addWidget(QLabel("Rendering Mode:"))
        self.render_box = QComboBox()
        self.render_box.addItems(["off", "human"])
        self.render_box.setCurrentText(self.updated_env_config["RENDER_MODE"])
        layout.addWidget(self.render_box)

        # Buttons
        btn_row = QHBoxLayout()
        self.apply_btn = QPushButton("Apply")
        self.set_default_btn = QPushButton("Set as Default")
        self.cancel_btn = QPushButton("Close" if read_only else "Cancel")
        btn_row.addWidget(self.apply_btn)
        btn_row.addWidget(self.set_default_btn)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

        self.apply_btn.clicked.connect(self._on_apply)
        self.set_default_btn.clicked.connect(self._on_set_default)
        self.cancel_btn.clicked.connect(self._on_cancel)

        # Disable editing if read-only
        if self.read_only:
            for w in [self.env_box, self.steps_box, self.episodes_box, self.render_box]:
                w.setEnabled(False)
            self.apply_btn.setEnabled(False)
            self.set_default_btn.setEnabled(False)
            layout.addWidget(QLabel("<span style='color:#bbb;'>🔒 Read-only mode (Training in progress)</span>"))

        self._update_default_steps(self.updated_env_config["ENV_NAME"])

    def _update_default_steps(self, env_name: str):
        """Display default step length for selected environment."""
        try:
            env = gym.make(env_name)
            default_steps = getattr(env.spec, "max_episode_steps", config.MAX_STEPS)
            env.close()
        except Exception:
            default_steps = 500

        self.default_label.setText(f"📏 Default Steps per Episode: {default_steps}")

        # Always update spin boxes to reflect defaults
        self.steps_box.setValue(default_steps)

        # Reset episodes to environment defaults if available
        default_episodes = getattr(config, "DEFAULT_EPISODES", 1000)
        self.episodes_box.setValue(default_episodes)

        # Also keep ENV_NAME synced
        self.updated_env_config["ENV_NAME"] = env_name
        self.updated_env_config["MAX_STEPS"] = default_steps

    def _on_apply(self):
        self.updated_env_config = {
            "ENV_NAME": self.env_box.currentText(),
            "MAX_STEPS": self.steps_box.value(),
            "EPISODES": self.episodes_box.value(),
            "RENDER_MODE": self.render_box.currentText(),
        }
        self.config_mgr.set_section_runtime(self.PANEL_NAME, self.updated_env_config)
        self.section._log("✅ Environment runtime configuration applied.")
        self.on_close_callback(True, self.updated_env_config)

    def _on_set_default(self):
        defaults = {
            "ENV_NAME": self.env_box.currentText(),
            "MAX_STEPS": self.steps_box.value(),
            "EPISODES": self.episodes_box.value(),
            "RENDER_MODE": self.render_box.currentText(),
        }
        self.config_mgr.set_section_defaults(self.PANEL_NAME, defaults)
        self.section._log("✅ Environment defaults saved.")

    def _on_cancel(self):
        """Close without applying."""
        self.on_close_callback(False, None)
