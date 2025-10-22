from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QDoubleSpinBox,
    QComboBox, QPushButton, QLabel, QHBoxLayout
)
import config


class NNConfigPanel(QWidget):
    """Inline panel for editing Neural Network configuration."""

    def __init__(self, on_close_callback, lock_hidden_layers=False, read_only=False):
        super().__init__()
        self.on_close_callback = on_close_callback
        self.lock_hidden_layers = lock_hidden_layers
        self.read_only = read_only
        self.updated_config = {}

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        title = QLabel("<b>🧠 Neural Network Configuration</b>")
        title.setStyleSheet("font-size:16px; color:#ddd;")
        layout.addWidget(title)

        form = QFormLayout()
        layout.addLayout(form)

        # Hidden Layers
        self.hidden_layers_input = QComboBox()
        presets = ["[64, 64]", "[128, 128]", "[256, 128]", "[256, 256]"]
        self.hidden_layers_input.addItems(presets)
        current = str(config.HIDDEN_LAYERS)
        if current in presets:
            self.hidden_layers_input.setCurrentText(current)
        form.addRow("Hidden Layers:", self.hidden_layers_input)

        if self.lock_hidden_layers:
            self.hidden_layers_input.setEnabled(False)
            layout.addWidget(QLabel("<span style='color:#bbb;'>🔒 Hidden layers locked (loaded model)</span>"))

        # Activation
        self.activation_box = QComboBox()
        self.activation_box.addItems(config.HIDDEN_ACTIVATIONS)
        self.activation_box.setCurrentText(config.HIDDEN_ACTIVATION)
        form.addRow("Activation:", self.activation_box)

        # Dropout
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.9)
        self.dropout_spin.setSingleStep(0.05)
        self.dropout_spin.setDecimals(2)
        self.dropout_spin.setValue(config.DROPOUT)
        form.addRow("Dropout Rate:", self.dropout_spin)

        # Learning Rate
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(6)
        self.lr_spin.setRange(1e-6, 1.0)
        self.lr_spin.setSingleStep(1e-4)
        self.lr_spin.setValue(config.LR)
        form.addRow("Learning Rate:", self.lr_spin)

        # Optimizer
        self.optimizer_box = QComboBox()
        self.optimizer_box.addItems(["adam", "rmsprop", "sgd"])
        self.optimizer_box.setCurrentText(getattr(config, "OPTIMIZER", "adam"))
        form.addRow("Optimizer:", self.optimizer_box)

        # Buttons
        btn_row = QHBoxLayout()
        self.apply_btn = QPushButton("Apply")
        self.cancel_btn = QPushButton("Close")
        btn_row.addWidget(self.apply_btn)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

        self.apply_btn.clicked.connect(self._on_apply)
        self.cancel_btn.clicked.connect(self._on_cancel)

        # Disable editing in read-only mode
        if self.read_only:
            for w in [
                self.hidden_layers_input,
                self.activation_box,
                self.dropout_spin,
                self.lr_spin,
                self.optimizer_box,
            ]:
                w.setEnabled(False)
            self.apply_btn.setEnabled(False)
            layout.addWidget(QLabel("<span style='color:#bbb;'>🔒 Read-only mode (Training in progress)</span>"))

    def _on_apply(self):
        """Collect and apply updates."""
        self.updated_config = {
            "HIDDEN_LAYERS": eval(self.hidden_layers_input.currentText()),
            "HIDDEN_ACTIVATION": self.activation_box.currentText(),
            "DROPOUT": self.dropout_spin.value(),
            "LR": self.lr_spin.value(),
            "OPTIMIZER": self.optimizer_box.currentText(),
        }

        # Apply to runtime config immediately
        for key, val in self.updated_config.items():
            setattr(config, key, val)

        self.on_close_callback(True, self.updated_config)

    def _on_cancel(self):
        """Cancel without applying changes."""
        self.on_close_callback(False, None)
