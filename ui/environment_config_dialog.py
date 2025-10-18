from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QMessageBox
)
from PySide6.QtCore import Qt
import config
import gymnasium as gym


class EnvironmentConfigDialog(QDialog):
    """Dialog for configuring environment-related training settings."""

    def __init__(self, parent=None, read_only: bool = False):
        super().__init__(parent)
        self.setWindowTitle("Environment Configuration")
        self.resize(420, 320)
        self.read_only = read_only

        self.updated_env_config = {
            "ENV_NAME": getattr(config, "DEFAULT_ENVIRONMENT", "CartPole-v1"),
            "MAX_STEPS": config.MAX_STEPS,
            "EPISODES": getattr(config, "DEFAULT_EPISODES", 1000),
            "RENDER_MODE": "off",
        }

        layout = QVBoxLayout(self)

        # --- Environment selection ---
        layout.addWidget(QLabel("Environment:"))
        self.env_box = QComboBox()
        self.env_box.addItems(config.AVAILABLE_ENVIRONMENTS)
        self.env_box.setCurrentText(self.updated_env_config["ENV_NAME"])
        self.env_box.currentTextChanged.connect(self._update_default_steps)
        layout.addWidget(self.env_box)

        # --- Default steps label ---
        self.default_label = QLabel()
        layout.addWidget(self.default_label)

        # --- Max steps ---
        layout.addWidget(QLabel("Max Steps per Episode:"))
        self.steps_box = QSpinBox()
        self.steps_box.setRange(50, 20000)
        self.steps_box.setValue(self.updated_env_config["MAX_STEPS"])
        layout.addWidget(self.steps_box)

        # --- Episodes ---
        layout.addWidget(QLabel("Training Episodes:"))
        self.episodes_box = QSpinBox()
        self.episodes_box.setRange(*config.EPISODE_RANGE)
        self.episodes_box.setValue(self.updated_env_config["EPISODES"])
        layout.addWidget(self.episodes_box)

        # --- Rendering mode ---
        layout.addWidget(QLabel("Rendering Mode:"))
        self.render_box = QComboBox()
        self.render_box.addItems(["off", "human"])
        self.render_box.setCurrentText(self.updated_env_config["RENDER_MODE"])
        layout.addWidget(self.render_box)

        # --- Buttons ---
        btn_layout = QHBoxLayout()
        apply_btn = QPushButton("Apply")
        save_default_btn = QPushButton("Save as Default")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(apply_btn)
        btn_layout.addWidget(save_default_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        apply_btn.clicked.connect(self._on_apply)
        save_default_btn.clicked.connect(self._on_save_default)
        cancel_btn.clicked.connect(self.reject)

        # --- Disable editing if read-only mode ---
        if self.read_only:
            for widget in [self.env_box, self.steps_box, self.episodes_box, self.render_box]:
                widget.setEnabled(False)
            apply_btn.setEnabled(False)
            save_default_btn.setEnabled(False)
            layout.addWidget(QLabel("<span style='color:#bbb;'>🔒 Read-only mode (Training in progress)</span>"))


        self._update_default_steps(self.updated_env_config["ENV_NAME"])

    # ------------------------------------------------------------------
    def _update_default_steps(self, env_name: str):
        """Detect and show the default step length for the selected environment."""
        try:
            env = gym.make(env_name)
            default_steps = getattr(env.spec, "max_episode_steps", config.MAX_STEPS)
            env.close()
        except Exception:
            default_steps = 500

        self.default_label.setText(f"📏 Default Steps per Episode: {default_steps}")
        if not self.steps_box.value() or self.steps_box.value() == config.MAX_STEPS:
            self.steps_box.setValue(default_steps)

    def _collect_updates(self):
        """Collect current settings from the form."""
        self.updated_env_config = {
            "ENV_NAME": self.env_box.currentText(),
            "MAX_STEPS": self.steps_box.value(),
            "EPISODES": self.episodes_box.value(),
            "RENDER_MODE": self.render_box.currentText(),
        }

    def _on_apply(self):
        """Apply changes to runtime config."""
        self._collect_updates()
        self._apply_to_runtime_config()
        self.accept()

    def _on_save_default(self):
        """Save configuration permanently."""
        self._collect_updates()
        self._apply_to_runtime_config()
        config.save_user_config(self.updated_env_config)
        QMessageBox.information(self, "Saved", "✅ Environment configuration saved as default.")
        self.accept()

    def get_updated_config(self):
        """Return updated values."""
        return self.updated_env_config

    def _apply_to_runtime_config(self):
        """Apply environment configuration to the runtime config module."""
        for key, value in self.updated_env_config.items():
            setattr(config, key, value)